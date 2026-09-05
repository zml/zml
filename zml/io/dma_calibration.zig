const std = @import("std");

const pjrt = @import("pjrt");

const host_memory = @import("host_memory.zig");
const platform_mod = @import("../platform.zig");
const Platform = platform_mod.Platform;
const limits = @import("limits.zig");

const log = std.log.scoped(.@"zml/io");

/// Immutable result shared by every device participating in one load.
pub const Result = struct {
    block_size: usize,
    max_in_flight_per_device: usize,

    pub const default: Result = .{
        .block_size = 4 * 1024 * 1024,
        .max_in_flight_per_device = 8,
    };
};

pub const default_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

pub const Options = struct {
    block_sizes: []const usize = &default_block_sizes,
    /// Fixed per-device width used by the block screen and the loader.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until the representative device completes the transfer target.
    duration_ns: u64 = 2 * std.time.ns_per_ms,
    minimum_transfers: u64 = 32,
    /// Borderline block candidates receive longer alternating paired windows.
    confirmation_duration_ns: u64 = 25 * std.time.ns_per_ms,
    confirmation_minimum_transfers: u64 = 256,
    confirmation_margin: f64 = 0.02,
    /// Prefer the smallest block within this tolerance of the measured peak
    /// transfer throughput.
    block_selection_tolerance: f64 = 0.08,
};

/// Called only during direct-loader initialization. Measures one representative
/// device into the loader-owned workspace; CPU returns defaults without measuring.
pub fn calibrate(
    workspace: *host_memory.Workspace,
    platform: *const platform_mod.Platform,
    opts: Options,
) !Result {
    // Nothing to measure on CPU: the plugin's `transferData` is a memcpy on
    // the submitting thread and a load takes the same time at every block
    // size, so the defaults stand and the loader grows its own arenas.
    if (platform.target == .cpu) return .default;

    try validateOptions(opts, workspace.max_mapped_bytes);
    const result = try measureTransfer(workspace, platform, opts);
    logReport(platform, &result);
    return result.calibration;
}

const sample_count = 3;

/// What one benchmark reports internally for the summary log.
const Report = struct {
    calibration: Result,
    retained_mapped_bytes: usize,
    measured_bytes_per_second: f64,
    /// Whole measurement, including arena mapping.
    elapsed_ns: u64,
    /// End of the device allocator warm-up to the selected block size: the
    /// calibration ring, screening, confirmation and cohort teardown.
    calibration_ns: u64,
    device_allocator_warmup_ns: u64,
};

/// Measures synthetic PJRT transfers on one representative device.
/// Every addressable device allocator is still warmed; benchmark allocations
/// remain mapped in the supplied workspace for later use.
fn measureTransfer(
    workspace: *host_memory.Workspace,
    platform: *const platform_mod.Platform,
    opts: Options,
) !Report {
    const allocator = workspace.allocator;
    const io = workspace.io;
    const benchmark_started: std.Io.Timestamp = .now(io, .awake);
    const device_warmup_started: std.Io.Timestamp = .now(io, .awake);
    try platform.warmupDeviceAllocators(io);
    const calibration_started: std.Io.Timestamp = .now(io, .awake);
    const device_allocator_warmup_ns = elapsedNanoseconds(
        device_warmup_started,
        calibration_started,
    );
    const representative = selection: {
        var session: Session = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
        };
        // Release the cohorts' device buffers before measuring calibration
        // cost; the mapped host ring remains in the workspace for loading.
        defer session.deinit(workspace);
        break :selection try selectBlockSize(&session, opts, workspace);
    };
    const calibration_ns = elapsedNanoseconds(
        calibration_started,
        .now(io, .awake),
    );

    const calibration: Result = .{
        .block_size = representative.block_size,
        .max_in_flight_per_device = opts.block_parallelism,
    };
    return .{
        .calibration = calibration,
        .retained_mapped_bytes = workspace.mapped_bytes.load(.acquire),
        .measured_bytes_per_second = representative.metrics.bytesPerSecond(),
        .elapsed_ns = elapsedNanoseconds(
            benchmark_started,
            .now(io, .awake),
        ),
        .calibration_ns = calibration_ns,
        .device_allocator_warmup_ns = device_allocator_warmup_ns,
    };
}

fn selectBlockSize(
    session: *Session,
    opts: Options,
    workspace: *host_memory.Workspace,
) !Selection {
    var block_count: usize = 0;
    var block_source_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!fitsWorkspace(workspace.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_count += 1;
        block_source_bytes = @max(block_source_bytes, block_size * opts.block_parallelism);
    }
    // Map one ring for the largest candidate at the fixed transfer width;
    // every candidate cohort reuses it.
    const calibration_source = workspace.findArena(block_source_bytes) orelse
        try workspace.allocate(block_source_bytes);

    const block_candidates = try session.allocator.alloc(Candidate, block_count);
    defer session.allocator.free(block_candidates);
    var block_index: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!fitsWorkspace(workspace.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_candidates[block_index] = .{
            .block_size = block_size,
            .cohort = try session.createCohort(0, block_size),
        };
        block_index += 1;
    }
    try measureCandidates(session, opts, block_candidates, calibration_source);
    return selectCandidate(session, opts, block_candidates, calibration_source);
}

fn fitsWorkspace(max_mapped_bytes: usize, block_size: usize, parallelism: usize) bool {
    return parallelism != 0 and block_size <= max_mapped_bytes / parallelism;
}

fn measureCandidates(
    session: *Session,
    opts: Options,
    candidates: []Candidate,
    source: []const u8,
) !void {
    for (0..sample_count) |repeat| {
        for (0..candidates.len) |offset| {
            const index = (offset + repeat) % candidates.len;
            const candidate = &candidates[index];
            const metrics = try runWindow(
                session.io,
                candidate.cohort,
                source[0 .. candidate.block_size * opts.block_parallelism],
                opts.block_parallelism,
                opts.duration_ns,
                opts.minimum_transfers,
            );
            candidate.appendMetric(metrics);
        }
    }
}

fn selectCandidate(
    session: *Session,
    opts: Options,
    candidates: []const Candidate,
    source: []const u8,
) !Selection {
    const tolerance = opts.block_selection_tolerance;
    const medians = try session.allocator.alloc(Measurement, candidates.len);
    defer session.allocator.free(medians);
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);
    const confirmed_metrics = try session.allocator.alloc(?Measurement, candidates.len);
    defer session.allocator.free(confirmed_metrics);
    @memset(confirmed_metrics, null);

    for (candidates, medians) |candidate, *median| {
        median.* = candidate.median();
    }
    var peak_index: usize = 0;
    for (medians[1..], 1..) |median, index| {
        if (median.bytesPerSecond() > medians[peak_index].bytesPerSecond())
            peak_index = index;
    }
    const peak_rate = medians[peak_index].bytesPerSecond();
    for (medians, ratios) |median, *ratio| {
        ratio.* = if (peak_rate == 0) 0 else median.bytesPerSecond() / peak_rate;
    }

    for (candidates, 0..) |_, candidate_index| {
        if (!needsConfirmation(
            candidates,
            candidate_index,
            peak_index,
            tolerance,
            opts.confirmation_margin,
        )) continue;
        var candidate_runs: [sample_count]Measurement = undefined;
        var baseline_runs: [sample_count]Measurement = undefined;
        for (0..sample_count) |repeat| {
            const order = if (repeat % 2 == 0)
                [_]usize{ candidate_index, peak_index }
            else
                [_]usize{ peak_index, candidate_index };
            for (order) |measured_index| {
                const measured = candidates[measured_index];
                const metrics = try runWindow(
                    session.io,
                    measured.cohort,
                    source[0 .. measured.block_size * opts.block_parallelism],
                    opts.block_parallelism,
                    opts.confirmation_duration_ns,
                    opts.confirmation_minimum_transfers,
                );
                if (measured_index == candidate_index)
                    candidate_runs[repeat] = metrics
                else
                    baseline_runs[repeat] = metrics;
            }
        }
        const representative = medianRatioIndex(
            &candidate_runs,
            &baseline_runs,
        );
        const baseline_rate = baseline_runs[representative].bytesPerSecond();
        ratios[candidate_index] = if (baseline_rate == 0) 0 else candidate_runs[representative].bytesPerSecond() / baseline_rate;
        confirmed_metrics[candidate_index] = candidate_runs[representative];
    }

    var maximum_ratio: f64 = 1;
    for (ratios) |ratio| maximum_ratio = @max(maximum_ratio, ratio);
    const floor = maximum_ratio * (1.0 - tolerance);
    var selected_index = peak_index;
    for (candidates, ratios, 0..) |candidate, ratio, index| {
        if (ratio >= floor and candidate.block_size < candidates[selected_index].block_size)
            selected_index = index;
    }
    return .{
        .block_size = candidates[selected_index].block_size,
        .metrics = confirmed_metrics[selected_index] orelse medians[selected_index],
    };
}

fn needsConfirmation(
    candidates: []const Candidate,
    candidate_index: usize,
    peak_index: usize,
    tolerance: f64,
    margin: f64,
) bool {
    if (candidate_index == peak_index) return false;
    const candidate = candidates[candidate_index];
    const peak = candidates[peak_index];
    std.debug.assert(candidate.metrics_len == peak.metrics_len);
    var qualified_once = false;
    var rejected_once = false;
    for (candidate.metricSlice(), 0..) |metric, repeat| {
        var peak_rate: f64 = 0;
        for (candidates) |round_candidate| {
            std.debug.assert(round_candidate.metrics_len == candidate.metrics_len);
            peak_rate = @max(peak_rate, round_candidate.metrics[repeat].bytesPerSecond());
        }
        const ratio = if (peak_rate == 0) 0 else metric.bytesPerSecond() / peak_rate;
        if (ratio >= 1.0 - tolerance)
            qualified_once = true
        else
            rejected_once = true;
    }
    if (qualified_once and rejected_once) return true;
    const candidate_median = candidate.median();
    const peak_median = peak.median();
    const peak_rate = peak_median.bytesPerSecond();
    const ratio = if (peak_rate == 0) 0 else candidate_median.bytesPerSecond() / peak_rate;
    return @abs(ratio - (1.0 - tolerance)) <= margin;
}

fn medianRatioIndex(
    candidates: []const Measurement,
    baselines: []const Measurement,
) usize {
    std.debug.assert(candidates.len == baselines.len and candidates.len > 0);
    std.debug.assert(candidates.len <= sample_count);
    var order_storage: [sample_count]usize = undefined;
    const order = order_storage[0..candidates.len];
    for (order, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const Measurement,
        baselines: []const Measurement,
    };
    std.mem.sort(usize, order, Context{ .candidates = candidates, .baselines = baselines }, struct {
        fn lessThan(context: Context, lhs: usize, rhs: usize) bool {
            const lhs_baseline = context.baselines[lhs].bytesPerSecond();
            const rhs_baseline = context.baselines[rhs].bytesPerSecond();
            const lhs_ratio = if (lhs_baseline == 0) 0 else context.candidates[lhs].bytesPerSecond() / lhs_baseline;
            const rhs_ratio = if (rhs_baseline == 0) 0 else context.candidates[rhs].bytesPerSecond() / rhs_baseline;
            return lhs_ratio < rhs_ratio;
        }
    }.lessThan);
    return order[order.len / 2];
}

fn runWindow(
    io: std.Io,
    cohort: *Cohort,
    source: []const u8,
    parallelism: usize,
    duration_ns: u64,
    minimum_transfers: u64,
) !Measurement {
    var metrics: Counters = .{};
    try cohort.ensureReady(source, parallelism);

    const Worker = struct {
        cohort: *Cohort,
        source: []const u8,
        slot: usize,
        metrics: *Counters,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.cohort.io);
            while (!self.stop.load(.acquire)) {
                self.cohort.transfer(self.source, self.slot, self.metrics);
                if (self.cohort.first_error.load(.acquire) != 0) return;
            }
        }
    };

    var ready: std.atomic.Value(usize) = .init(0);
    var start: std.Io.Event = .unset;
    var stop: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    defer {
        // Workers borrow this window's stack. Every exit, including a canceled
        // sleep or partial spawn, must release the start gate and join them.
        stop.store(true, .release);
        start.set(io);
        group.await(io) catch {};
    }
    for (0..parallelism) |slot| {
        try group.concurrent(io, Worker.run, .{Worker{
            .cohort = cohort,
            .source = source,
            .slot = slot,
            .metrics = &metrics,
            .ready = &ready,
            .start = &start,
            .stop = &stop,
        }});
    }
    while (ready.load(.acquire) != parallelism) try io.sleep(.fromMilliseconds(1), .awake);
    const measured_at: std.Io.Timestamp = .now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed_ns = elapsedNanoseconds(measured_at, .now(io, .awake));
        const error_code = cohort.first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
        if (windowComplete(
            elapsed_ns,
            duration_ns,
            metrics.transfers.load(.acquire),
            minimum_transfers,
        )) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    stop.store(true, .release);
    try group.await(io);
    const elapsed_ns: u64 = @intCast(@max(measured_at.untilNow(io, .awake).nanoseconds, 1));
    const error_code = cohort.first_error.load(.acquire);
    if (error_code != 0) return @errorFromInt(error_code);
    return .{
        .bytes = metrics.bytes.load(.acquire),
        .transfers = metrics.transfers.load(.acquire),
        .elapsed_ns = elapsed_ns,
    };
}

fn windowComplete(
    elapsed_ns: u64,
    minimum_duration_ns: u64,
    completed_transfers: u64,
    minimum_transfers: u64,
) bool {
    return elapsed_ns >= minimum_duration_ns and completed_transfers >= minimum_transfers;
}

/// One line per calibration: what was selected, at what rate, and what it
/// cost. Per-arena mapping is logged where each arena is mapped.
fn logReport(platform: *const platform_mod.Platform, result: *const Report) void {
    log.info("dma_bench version=13 platform={s} devices={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} elapsed_ms={d:.3} calibration_ms={d:.3} allocator_warmup_ms={d:.3} retained_mapped_bytes={d}", .{
        @tagName(platform.target),
        platform.devices.len,
        platform.devices[0].kind(),
        result.calibration.block_size,
        result.calibration.max_in_flight_per_device,
        result.measured_bytes_per_second / (1024 * 1024 * 1024),
        @as(f64, @floatFromInt(result.elapsed_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.calibration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.device_allocator_warmup_ns)) / std.time.ns_per_ms,
        result.retained_mapped_bytes,
    });
}

const Candidate = struct {
    block_size: usize,
    cohort: *Cohort,
    metrics: [sample_count]Measurement = undefined,
    metrics_len: usize = 0,

    fn appendMetric(self: *Candidate, metric: Measurement) void {
        std.debug.assert(self.metrics_len < self.metrics.len);
        self.metrics[self.metrics_len] = metric;
        self.metrics_len += 1;
    }

    fn metricSlice(self: *const Candidate) []const Measurement {
        return self.metrics[0..self.metrics_len];
    }

    fn median(self: Candidate) Measurement {
        std.debug.assert(self.metrics_len > 0);
        var scratch = self.metrics;
        const populated = scratch[0..self.metrics_len];
        std.mem.sort(Measurement, populated, {}, struct {
            fn lessThan(_: void, lhs: Measurement, rhs: Measurement) bool {
                return lhs.bytesPerSecond() < rhs.bytesPerSecond();
            }
        }.lessThan);
        return populated[populated.len / 2];
    }
};

const Selection = struct {
    block_size: usize,
    metrics: Measurement,
};

const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    cohorts: std.ArrayListUnmanaged(*Cohort) = .empty,

    fn createCohort(
        self: *Session,
        device_index: usize,
        block_size: usize,
    ) !*Cohort {
        const cohort = try self.allocator.create(Cohort);
        errdefer self.allocator.destroy(cohort);
        cohort.* = .init(
            self.allocator,
            self.io,
            self.platform,
            device_index,
            block_size,
        );
        try self.cohorts.append(self.allocator, cohort);
        return cohort;
    }

    fn deinit(self: *Session, workspace: *const host_memory.Workspace) void {
        for (self.cohorts.items) |cohort| {
            cohort.deinit(workspace.findArena(cohort.block_size) orelse unreachable);
            self.allocator.destroy(cohort);
        }
        self.cohorts.deinit(self.allocator);
        self.* = undefined;
    }
};

const Cohort = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    device_index: usize,
    block_size: usize,
    managers: std.ArrayListUnmanaged(TransferBuffer) = .empty,
    warmed_managers: usize = 0,
    first_error: std.atomic.Value(u16) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const platform_mod.Platform,
        device_index: usize,
        block_size: usize,
    ) Cohort {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .device_index = device_index,
            .block_size = block_size,
        };
    }

    fn recordError(self: *Cohort, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn transfer(
        self: *Cohort,
        source: []const u8,
        slot: usize,
        metrics: ?*Counters,
    ) void {
        const len = self.block_size;
        const source_offset = slot * self.block_size;
        const event = self.managers.items[slot].manager.transferData(
            self.platform.pjrt_api,
            0,
            source[source_offset..][0..len],
            0,
            false,
        ) catch |err| {
            self.recordError(err);
            return;
        };
        event.await(self.platform.pjrt_api, self.io) catch |err| {
            event.deinit(self.platform.pjrt_api);
            self.recordError(err);
            return;
        };
        event.deinit(self.platform.pjrt_api);
        if (metrics) |output| {
            _ = output.bytes.fetchAdd(@intCast(len), .monotonic);
            _ = output.transfers.fetchAdd(1, .monotonic);
        }
    }

    fn ensureReady(self: *Cohort, source: []const u8, parallelism: usize) !void {
        const required_bytes = self.block_size * parallelism;
        if (required_bytes > source.len) return error.DmaBenchmarkPinnedBudgetExceeded;
        var dims = [_]i64{@intCast(self.block_size)};
        const shape_spec: pjrt.ShapeSpec = .init(&dims, .u8);
        const memory = self.platform.devices[self.device_index].memory(.default).?;
        while (self.managers.items.len < parallelism) {
            try self.managers.ensureUnusedCapacity(self.allocator, 1);
            const manager = try self.platform.pjrt_client.createBuffersForAsyncHostToDevice(self.platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(self.platform.pjrt_api);
            const buffer = try manager.retrieveBuffer(self.platform.pjrt_api, 0);
            self.managers.appendAssumeCapacity(.{ .manager = manager, .buffer = buffer });
        }
        while (self.warmed_managers < parallelism) : (self.warmed_managers += 1) {
            const slot = self.warmed_managers;
            self.transfer(source, slot, null);
            self.transfer(source, slot, null);
            const error_code = self.first_error.load(.acquire);
            if (error_code != 0) return @errorFromInt(error_code);
        }
    }

    fn deinit(self: *Cohort, source: []const u8) void {
        for (self.managers.items) |manager| {
            const event = manager.manager.transferData(
                self.platform.pjrt_api,
                0,
                source[0..self.block_size],
                0,
                true,
            ) catch null;
            if (event) |done| {
                done.await(self.platform.pjrt_api, self.io) catch {};
                done.deinit(self.platform.pjrt_api);
            }
            manager.manager.deinit(self.platform.pjrt_api);
            manager.buffer.deinit(self.platform.pjrt_api);
        }
        self.managers.deinit(self.allocator);
        self.* = undefined;
    }
};

const TransferBuffer = struct {
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffer: *pjrt.Buffer,
};

const Counters = struct {
    bytes: std.atomic.Value(u64) = .init(0),
    transfers: std.atomic.Value(u64) = .init(0),
};

const Measurement = struct {
    bytes: u64,
    transfers: u64,
    elapsed_ns: u64,

    fn bytesPerSecond(self: Measurement) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }
};

fn validateOptions(opts: Options, max_mapped_bytes: usize) !void {
    if (opts.block_sizes.len == 0) return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.duration_ns == 0 or opts.confirmation_duration_ns == 0)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.block_parallelism == 0 or opts.block_parallelism > limits.max_dma_parallelism)
        return error.InvalidDmaBenchmarkOptions;
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1) or
        !(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1))
        return error.InvalidDmaBenchmarkOptions;
    var has_feasible_block = false;
    for (opts.block_sizes) |block_size| {
        if (block_size == 0 or block_size > limits.max_read_request_size)
            return error.InvalidDmaBenchmarkOptions;
        if (fitsWorkspace(max_mapped_bytes, block_size, opts.block_parallelism))
            has_feasible_block = true;
    }
    if (!has_feasible_block) return error.NoFeasibleDmaBenchmarkTuple;
}

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

/// A platform with only the fields the calibration decisions read.
fn testPlatform(target: platform_mod.Target) platform_mod.Platform {
    return .{
        .arena = undefined,
        .target = target,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .state = .init(target),
        .devices = &.{},
        .memories = &.{},
        .physical_mesh = undefined,
        .replicated_sharding = undefined,
        .shardings = .empty,
    };
}

test "DMA benchmark on CPU returns the defaults without mapping" {
    const platform = testPlatform(.cpu);
    var workspace = try host_memory.Workspace.initForTesting(std.testing.allocator, std.testing.io, 64);
    workspace.backend = .{ .pageable = .{ .platform = &platform } };
    defer workspace.deinit();
    try std.testing.expectEqual(Result.default, try calibrate(&workspace, &platform, .{}));
    try std.testing.expectEqual(0, workspace.mapped_bytes.load(.acquire));
}

test "DMA benchmark validates options" {
    const max_mapped_bytes = 16 * 1024 * 1024 * 1024;
    try validateOptions(.{}, max_mapped_bytes);
    try std.testing.expectError(
        error.InvalidDmaBenchmarkOptions,
        validateOptions(.{ .block_parallelism = 0 }, max_mapped_bytes),
    );
}

test "DMA benchmark completion target has no time cap" {
    try std.testing.expect(!windowComplete(9, 10, 128, 128));
    try std.testing.expect(!windowComplete(10, 10, 127, 128));
    try std.testing.expect(windowComplete(10, 10, 128, 128));
    try std.testing.expect(windowComplete(1_000, 10, 128, 128));
    try std.testing.expect(windowComplete(10, 10, 0, 0));
}

test "DMA benchmark selection uses medians and prefers the smallest near-peak value" {
    // No candidate below is borderline, so the production selector never
    // schedules a confirmation window and the session supplies only its
    // allocator.
    var session: Session = .{
        .allocator = std.testing.allocator,
        .io = undefined,
        .platform = undefined,
    };
    const opts: Options = .{ .block_selection_tolerance = 0.05 };

    var candidates = [_]Candidate{
        .{ .block_size = 2, .cohort = undefined },
        .{ .block_size = 4, .cohort = undefined },
        .{ .block_size = 8, .cohort = undefined },
    };
    const rates = [_][3]u64{
        .{ 60, 10, 62 },
        .{ 98, 99, 97 },
        .{ 100, 101, 99 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| {
            candidate.appendMetric(.{
                .bytes = rate,
                .transfers = 1,
                .elapsed_ns = std.time.ns_per_s,
            });
        }
    }
    const decision = try selectCandidate(
        &session,
        opts,
        &candidates,
        &.{},
    );
    try std.testing.expectEqual(@as(usize, 4), decision.block_size);
    try std.testing.expectEqual(@as(f64, 98), decision.metrics.bytesPerSecond());

    // A dip between two near-peak values must not end the scan early.
    var bimodal = [_]Candidate{
        .{ .block_size = 2, .cohort = undefined },
        .{ .block_size = 4, .cohort = undefined },
        .{ .block_size = 8, .cohort = undefined },
        .{ .block_size = 16, .cohort = undefined },
    };
    const bimodal_rates = [_]u64{ 80, 100, 70, 99 };
    for (&bimodal, bimodal_rates) |*candidate, rate| {
        candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    const bimodal_decision = try selectCandidate(
        &session,
        opts,
        &bimodal,
        &.{},
    );
    try std.testing.expectEqual(@as(usize, 4), bimodal_decision.block_size);
    try std.testing.expectEqual(@as(f64, 100), bimodal_decision.metrics.bytesPerSecond());
}

test "DMA benchmark tuple feasibility rejects pinned budget overflow" {
    try std.testing.expect(fitsWorkspace(128, 16, 8));
    try std.testing.expect(!fitsWorkspace(127, 16, 8));
    try std.testing.expect(!fitsWorkspace(std.math.maxInt(usize), std.math.maxInt(usize), 2));
}

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    var candidates = [_]Candidate{
        .{ .block_size = 4, .cohort = undefined },
        .{ .block_size = 8, .cohort = undefined },
    };
    const rates = [_][3]u64{
        .{ 96, 80, 97 },
        .{ 100, 100, 100 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    try std.testing.expect(needsConfirmation(
        &candidates,
        0,
        1,
        0.05,
        0.02,
    ));
}

test "DMA benchmark cancellation drains transfer workers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = platform_mod.Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var source: [16]u8 = @splat(0);
    var cohort: Cohort = .init(allocator, io, platform, 0, source.len);
    defer cohort.deinit(&source);
    var vtable = io.vtable.*;
    vtable.sleep = struct {
        fn sleep(_: ?*anyopaque, _: std.Io.Timeout) std.Io.Cancelable!void {
            return error.Canceled;
        }
    }.sleep;
    const canceled_io: std.Io = .{ .userdata = io.userdata, .vtable = &vtable };

    // A cancellation can arrive while workers wait for the start gate or
    // while they transfer. In either case they must finish before teardown.
    try std.testing.expectError(error.Canceled, runWindow(
        canceled_io,
        &cohort,
        &source,
        1,
        std.math.maxInt(u64),
        0,
    ));
}
