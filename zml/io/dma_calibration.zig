const std = @import("std");

const pjrt = @import("pjrt");

const Platform = @import("../platform.zig").Platform;
const Target = @import("../platform.zig").Target;
const host_memory = @import("host_memory.zig");
const limits = @import("limits.zig");
const log = @import("log.zig").io;

/// Immutable result shared by every device participating in one load.
pub const Result = struct {
    block_size: usize,
    max_in_flight_per_device: usize,

    pub const default: Result = .{
        .block_size = 4 * 1024 * 1024,
        .max_in_flight_per_device = 8,
    };
};

/// Keep a block screen rather than adopting one accelerator's preferred size.
/// Historical loads on one MI300X measured 23.84/24.90/25.43 GiB/s at
/// 8/16/32 MiB; replicated Gemma (58.25 GiB logical) on eight MI300X took
/// 10.694 s at 8 MiB versus 7.829 s at 16 MiB. One B70 instead favored the
/// 8 MiB neighborhood; using the 16 MiB preference cost about 10.5% goodput.
/// These older-plugin results motivate screening, not fixed performance
/// targets.
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
    /// Eight repeatedly won or tied on MI300X; wider stages mostly increased
    /// callback latency and pinned memory. Adaptive DMA width added state with
    /// little load benefit, so source width is the only runtime search.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until the representative device completes the transfer target.
    /// Reducing 10 ms/128 transfers to 2 ms/32 shortened calibration with
    /// eight MI300X from 4.834 to 0.956 s while still selecting 16 MiB and
    /// width eight. Short screens sometimes selected the wrong block under
    /// noise, hence the longer borderline confirmation below.
    minimum_duration: std.Io.Duration = .fromMilliseconds(2),
    minimum_transfers: u64 = 32,
    /// Borderline block candidates receive longer paired windows, alternating which runs first.
    confirmation_duration: std.Io.Duration = .fromMilliseconds(25),
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
    platform: *const Platform,
    opts: Options,
) !Result {
    // Nothing to measure on CPU: the plugin's `transferData` is a memcpy on
    // the submitting thread. With four PJRT CPU devices, warm sharded Llama took
    // 1.55-1.60 s at 2, 8 and 16 MiB blocks, so the defaults stand and the
    // loader grows its own arenas. Calibration spent 607 ms measuring
    // 102 GiB/s into a reused ring; the load instead first-touched fresh
    // device buffers in 4 KiB pages (3.94M minor faults for 14.96 GiB).
    // The synthetic bandwidth was not the load's bottleneck; other
    // page-size/THP configurations were not measured.
    if (platform.target == .cpu) return .default;

    std.debug.assert(opts.block_sizes.len > 0);
    std.debug.assert(opts.minimum_duration.nanoseconds > 0);
    std.debug.assert(opts.confirmation_duration.nanoseconds > 0);
    std.debug.assert(opts.block_parallelism > 0 and opts.block_parallelism <= limits.max_dma_parallelism);
    std.debug.assert(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1);
    std.debug.assert(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1);
    for (opts.block_sizes) |block_size| {
        std.debug.assert(block_size > 0 and block_size <= limits.max_read_request_size);
        std.debug.assert(block_size <= workspace.max_mapped_bytes / opts.block_parallelism);
    }

    const result = try measureTransfer(workspace, platform, opts);

    log.debug("dma_bench version=13 platform={s} devices={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} duration={f} retained_mapped_bytes={d}", .{
        @tagName(platform.target),
        platform.devices.len,
        platform.devices[0].kind(),
        result.calibration.block_size,
        result.calibration.max_in_flight_per_device,
        result.measured_bytes_per_second / (1024 * 1024 * 1024),
        result.duration,
        result.retained_mapped_bytes,
    });

    return result.calibration;
}

const sample_count = 3;

/// What one benchmark reports internally for the summary log.
const Report = struct {
    calibration: Result,
    retained_mapped_bytes: usize,
    measured_bytes_per_second: f64,
    duration: std.Io.Duration,
};

/// Measures synthetic PJRT transfers on one representative device.
/// Every addressable device allocator is still warmed; benchmark allocations
/// remain mapped in the supplied workspace for later use.
/// The reported rate is that one device's synthetic H2D rate, not aggregate
/// platform throughput or a prediction of checkpoint load speed. All-device
/// warm-up and retained capacity do not turn this into an all-device sample.
fn measureTransfer(
    workspace: *host_memory.Workspace,
    platform: *const Platform,
    opts: Options,
) !Report {
    const allocator = workspace.allocator;
    const io = workspace.io;
    const start: std.Io.Timestamp = .now(io, .awake);

    var max_total_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        max_total_bytes = @max(max_total_bytes, block_size * opts.block_parallelism);
    }
    // Map one ring for the largest candidate at the fixed transfer width;
    // every candidate reuses it.
    const calibration_source = workspace.findArena(max_total_bytes) orelse
        try workspace.allocate(max_total_bytes);

    var session = try Session.init(allocator, io, platform, opts.block_sizes, opts.block_parallelism);
    defer session.deinit();

    const candidates = try allocator.alloc(Candidate, opts.block_sizes.len);
    defer allocator.free(candidates);
    for (candidates, opts.block_sizes, 0..) |*candidate, block_size, index| {
        candidate.* = .{
            .io = io,
            .platform = platform,
            .manager = session.manager,
            .buffer_offset = index * opts.block_parallelism,
            .buffer_count = opts.block_parallelism,
            .block_size = block_size,
        };
    }
    try measureAllCandidates(&session, opts, candidates, calibration_source);
    const winner = try selectCandidate(&session, opts, candidates, calibration_source);

    return .{
        .calibration = .{
            .block_size = winner.block_size,
            .max_in_flight_per_device = opts.block_parallelism,
        },
        .retained_mapped_bytes = workspace.mapped_bytes,
        .measured_bytes_per_second = winner.metrics.bytesPerSecond(),
        .duration = start.durationTo(.now(io, .awake)),
    };
}

fn measureAllCandidates(
    session: *Session,
    opts: Options,
    candidates: []Candidate,
    source: []const u8,
) !void {
    for (0..sample_count) |start_offset| {
        for (0..candidates.len) |i| {
            const index = (i + start_offset) % candidates.len;
            const candidate = &candidates[index];
            const metrics = try runWindow(
                session.io,
                candidate,
                source[0 .. candidate.block_size * opts.block_parallelism],
                opts.block_parallelism,
                opts.minimum_duration,
                opts.minimum_transfers,
            );
            candidate.appendMetric(metrics);
        }
    }
}

fn selectCandidate(
    session: *Session,
    opts: Options,
    candidates: []Candidate,
    source: []const u8,
) !struct {
    block_size: usize,
    metrics: Measurement,
} {
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);
    const confirmed_metrics = try session.allocator.alloc(?Measurement, candidates.len);
    defer session.allocator.free(confirmed_metrics);
    @memset(confirmed_metrics, null);

    // Compute peak_rate and ratios for all candidates.
    var peak_index: usize = 0;
    var peak_rate = candidates[0].median().bytesPerSecond();
    for (candidates[1..], 1..) |candidate, index| {
        const rate = candidate.median().bytesPerSecond();
        if (rate > peak_rate) {
            peak_index = index;
            peak_rate = rate;
        }
    }
    for (candidates, ratios) |candidate, *ratio| {
        ratio.* = if (peak_rate == 0) 0 else candidate.median().bytesPerSecond() / peak_rate;
    }

    // Confirm all candidates again against peak with a new run.
    for (candidates, 0..) |_, candidate_index| {
        if (isCloseToPeak(
            candidates,
            candidate_index,
            peak_index,
            opts.block_selection_tolerance,
            opts.confirmation_margin,
        )) {
            const confirmation = try confirmCandidateAgainstPeak(
                session.io,
                opts,
                &candidates[candidate_index],
                &candidates[peak_index],
                source,
            );
            ratios[candidate_index] = confirmation.ratio;
            confirmed_metrics[candidate_index] = confirmation.metrics;
        }
    }

    // Take the candidate with the highest throughput, in case of ties use the smallest block size.
    var maximum_ratio: f64 = 1;
    for (ratios) |ratio| maximum_ratio = @max(maximum_ratio, ratio);
    const floor = maximum_ratio * (1.0 - opts.block_selection_tolerance);
    var selected_index = peak_index;
    for (candidates, ratios, 0..) |candidate, ratio, index| {
        if (ratio >= floor and candidate.block_size < candidates[selected_index].block_size)
            selected_index = index;
    }
    return .{
        .block_size = candidates[selected_index].block_size,
        .metrics = confirmed_metrics[selected_index] orelse candidates[selected_index].median(),
    };
}

fn isCloseToPeak(
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

    // Whether the candidate is near-peak in one round but fails in others.
    var qualified_once = false;
    var rejected_once = false;
    for (candidate.metricSlice(), 0..) |metric, round| {
        var round_peak_rate: f64 = 0;
        for (candidates) |round_candidate| {
            std.debug.assert(round_candidate.metrics_len == candidate.metrics_len);
            round_peak_rate = @max(round_peak_rate, round_candidate.metrics[round].bytesPerSecond());
        }
        const ratio = if (round_peak_rate == 0) 0 else metric.bytesPerSecond() / round_peak_rate;
        if (ratio >= 1.0 - tolerance)
            qualified_once = true
        else
            rejected_once = true;
    }
    if (qualified_once and rejected_once) return true;

    // Whether the candiate's ratio is close enough overall.
    const candidate_rate = candidate.median().bytesPerSecond();
    const peak_rate = peak.median().bytesPerSecond();
    const ratio = if (peak_rate == 0) 0 else candidate_rate / peak_rate;
    return @abs(ratio - (1.0 - tolerance)) <= margin;
}

fn confirmCandidateAgainstPeak(
    io: std.Io,
    opts: Options,
    candidate: *Candidate,
    peak: *Candidate,
    source: []const u8,
) !struct { ratio: f64, metrics: Measurement } {
    var candidate_runs: [sample_count]Measurement = undefined;
    var baseline_runs: [sample_count]Measurement = undefined;
    for (0..sample_count) |repeat| {
        const order = if (repeat % 2 == 0)
            [_]*Candidate{ candidate, peak }
        else
            [_]*Candidate{ peak, candidate };
        for (order) |measured| {
            const metrics = try runWindow(
                io,
                measured,
                source[0 .. measured.block_size * opts.block_parallelism],
                opts.block_parallelism,
                opts.confirmation_duration,
                opts.confirmation_minimum_transfers,
            );
            if (measured == candidate)
                candidate_runs[repeat] = metrics
            else
                baseline_runs[repeat] = metrics;
        }
    }
    const index = medianThroughputRatioIndex(
        &candidate_runs,
        &baseline_runs,
    );
    const metrics = candidate_runs[index];
    const baseline_rate = baseline_runs[index].bytesPerSecond();
    return .{
        .ratio = if (baseline_rate == 0) 0 else metrics.bytesPerSecond() / baseline_rate,
        .metrics = metrics,
    };
}

fn medianThroughputRatioIndex(
    candidates: []const Measurement,
    baselines: []const Measurement,
) usize {
    var ordered_indices: [sample_count]usize = undefined;
    for (&ordered_indices, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const Measurement,
        baselines: []const Measurement,
    };
    std.mem.sort(usize, &ordered_indices, Context{ .candidates = candidates, .baselines = baselines }, struct {
        fn lessThan(context: Context, lhs: usize, rhs: usize) bool {
            const lhs_baseline = context.baselines[lhs].bytesPerSecond();
            const rhs_baseline = context.baselines[rhs].bytesPerSecond();
            const lhs_ratio = if (lhs_baseline == 0) 0 else context.candidates[lhs].bytesPerSecond() / lhs_baseline;
            const rhs_ratio = if (rhs_baseline == 0) 0 else context.candidates[rhs].bytesPerSecond() / rhs_baseline;
            return lhs_ratio < rhs_ratio;
        }
    }.lessThan);
    return ordered_indices[ordered_indices.len / 2];
}

fn runWindow(
    io: std.Io,
    candidate: *Candidate,
    source: []const u8,
    parallelism: usize,
    minimum_duration: std.Io.Duration,
    minimum_transfers: u64,
) !Measurement {
    const Worker = struct {
        candidate: *Candidate,
        source: []const u8,
        slot: usize,
        metrics: *Counters,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.candidate.io);
            while (!self.stop.load(.acquire)) {
                self.candidate.transfer(self.source, self.slot, self.metrics);
                if (self.candidate.first_error.load(.acquire) != 0) return;
            }
        }
    };

    var metrics: Counters = .{};
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
            .candidate = candidate,
            .source = source,
            .slot = slot,
            .metrics = &metrics,
            .ready = &ready,
            .start = &start,
            .stop = &stop,
        }});
    }

    try candidate.warmupBuffers(source, parallelism);

    while (ready.load(.acquire) != parallelism) try io.sleep(.fromMicroseconds(100), .awake);
    const start_at: std.Io.Timestamp = .now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed = start_at.durationTo(.now(io, .awake));
        if (candidate.firstError()) |err| return err;
        // is window complete
        if (elapsed.nanoseconds >= minimum_duration.nanoseconds and metrics.transfers.load(.acquire) >= minimum_transfers) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    stop.store(true, .release);
    try group.await(io);
    const elapsed = start_at.durationTo(.now(io, .awake));
    if (candidate.firstError()) |err| return err;
    return .{
        .bytes = metrics.bytes.load(.acquire),
        .transfers = metrics.transfers.load(.acquire),
        .elapsed = elapsed,
    };
}

const Candidate = struct {
    block_size: usize,
    io: std.Io,
    platform: *const Platform,
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffer_offset: usize,
    buffer_count: usize,
    warmed_buffers: usize = 0,
    first_error: std.atomic.Value(u16) = .init(0),

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
    fn recordError(self: *Candidate, err: pjrt.ApiError) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn firstError(self: *const Candidate) ?pjrt.ApiError {
        const code = self.first_error.load(.acquire);
        return if (code == 0) null else @errorCast(@errorFromInt(code));
    }

    fn transfer(
        self: *Candidate,
        source: []const u8,
        slot: usize,
        metrics: ?*Counters,
    ) void {
        const len = self.block_size;
        const source_offset = slot * self.block_size;
        const event = self.manager.transferData(
            self.platform.pjrt_api,
            self.buffer_offset + slot,
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

    fn warmupBuffers(self: *Candidate, source: []const u8, parallelism: usize) !void {
        while (self.warmed_buffers < parallelism) : (self.warmed_buffers += 1) {
            const slot = self.warmed_buffers;
            self.transfer(source, slot, null);
            self.transfer(source, slot, null);
            if (self.firstError()) |err| return err;
        }
    }
};

const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffers: []*pjrt.Buffer,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        block_sizes: []const usize,
        parallelism: usize,
    ) !Session {
        const buffer_count = std.math.mul(usize, block_sizes.len, parallelism) catch return error.OutOfMemory;
        const dims = try allocator.alloc([1]i64, block_sizes.len);
        defer allocator.free(dims);
        const specs = try allocator.alloc(pjrt.ShapeSpec, buffer_count);
        defer allocator.free(specs);
        for (block_sizes, dims, 0..) |block_size, *dim, index| {
            dim.* = .{@intCast(block_size)};
            @memset(specs[index * parallelism ..][0..parallelism], pjrt.ShapeSpec.init(dim, .u8));
        }
        const buffers = try allocator.alloc(*pjrt.Buffer, buffer_count);
        errdefer allocator.free(buffers);
        const manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = specs,
            .memory = platform.devices[0].memory(.default).?.pjrt_memory,
        });
        var retrieved: usize = 0;
        errdefer {
            manager.deinit(platform.pjrt_api);
            for (buffers[0..retrieved]) |buffer| buffer.deinit(platform.pjrt_api);
        }
        for (buffers, 0..) |*buffer, index| {
            buffer.* = try manager.retrieveBuffer(platform.pjrt_api, index);
            retrieved += 1;
        }
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .manager = manager,
            .buffers = buffers,
        };
    }

    fn deinit(self: *Session) void {
        // XLA commit 1b19ae012aa67426658f7ca3c1503bb781c863a9 switched GPU
        // transfers to CommonAsyncHostToDeviceTransferManager, whose destructor
        // waits for outstanding transfers and marks unfinished buffers as errored.
        // We discard these buffers, so no final is_last_transfer=true copy is needed.
        self.manager.deinit(self.platform.pjrt_api);
        for (self.buffers) |buffer| buffer.deinit(self.platform.pjrt_api);
        self.allocator.free(self.buffers);
        self.* = undefined;
    }
};

const Counters = struct {
    bytes: std.atomic.Value(u64) = .init(0),
    transfers: std.atomic.Value(u64) = .init(0),
};

const Measurement = struct {
    bytes: u64,
    transfers: u64,
    elapsed: std.Io.Duration,

    fn bytesPerSecond(self: Measurement) f64 {
        if (self.elapsed.nanoseconds <= 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed.nanoseconds));
    }
};

test "DMA benchmark selection uses medians and prefers the smallest near-peak value" {
    // No candidate below is borderline, so the production selector never
    // schedules a confirmation window and the session supplies only its
    // allocator.
    var session: Session = .{
        .allocator = std.testing.allocator,
        .io = undefined,
        .platform = undefined,
        .manager = undefined,
        .buffers = undefined,
    };
    const opts: Options = .{ .block_selection_tolerance = 0.05 };

    var candidates = [_]Candidate{
        .{ .block_size = 2, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 4, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 8, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
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
                .elapsed = .fromSeconds(1),
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
        .{ .block_size = 2, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 4, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 8, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 16, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
    };
    const bimodal_rates = [_]u64{ 80, 100, 70, 99 };
    for (&bimodal, bimodal_rates) |*candidate, rate| {
        candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .elapsed = .fromSeconds(1),
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

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    var candidates = [_]Candidate{
        .{ .block_size = 4, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
        .{ .block_size = 8, .io = undefined, .platform = undefined, .manager = undefined, .buffer_offset = undefined, .buffer_count = undefined },
    };
    const rates = [_][3]u64{
        .{ 96, 80, 97 },
        .{ 100, 100, 100 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .elapsed = .fromSeconds(1),
        });
    }
    try std.testing.expect(isCloseToPeak(
        &candidates,
        0,
        1,
        0.05,
        0.02,
    ));
}

test "DMA benchmark shares one manager across candidate sizes and slots" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var session = try Session.init(allocator, io, platform, &.{ 8, 16 }, 2);
    defer session.deinit();
    const source: [32]u8 = @splat(0);
    for ([_]usize{ 8, 16 }, 0..) |block_size, index| {
        var candidate: Candidate = .{
            .io = io,
            .platform = platform,
            .manager = session.manager,
            .buffer_offset = index * 2,
            .buffer_count = 2,
            .block_size = block_size,
        };
        try candidate.warmupBuffers(&source, 2);
        try std.testing.expectEqual(2, candidate.warmed_buffers);
        try std.testing.expectEqual(null, candidate.firstError());
    }
}

test "DMA benchmark cancellation drains transfer workers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var source: [16]u8 = @splat(0);
    var session = try Session.init(allocator, io, platform, &.{source.len}, 1);
    defer session.deinit();
    var candidate: Candidate = .{
        .io = io,
        .platform = platform,
        .manager = session.manager,
        .buffer_offset = 0,
        .buffer_count = 1,
        .block_size = source.len,
    };
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
        &candidate,
        &source,
        1,
        .max,
        0,
    ));
}
