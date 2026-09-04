const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const mem = @import("../mem.zig");
const platform_mod = @import("../platform.zig");
const limits = @import("limits.zig");

const log = std.log.scoped(.@"zml/io");

pub const default_benchmark_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

const benchmark_repeats = 3;
const max_devices = 64;

/// Immutable DMA calibration shared by every device participating in one load.
pub const Calibration = struct {
    block_size: usize,
    max_in_flight_per_device: usize,
};

pub const Options = struct {
    block_sizes: []const usize = &default_benchmark_block_sizes,
    /// Fixed per-device width used by the block screen and the loader.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until the representative device completes the transfer target.
    duration_ns: u64 = 2 * std.time.ns_per_ms,
    minimum_transfers_per_device: u64 = 32,
    /// Borderline local decisions receive longer alternating paired windows.
    confirmation_duration_ns: u64 = 25 * std.time.ns_per_ms,
    confirmation_minimum_transfers_per_device: u64 = 256,
    confirmation_margin: f64 = 0.02,
    /// Prefer a smaller transaction once it supplies enough headroom over the
    /// source pipeline instead of maximizing isolated copy-engine throughput.
    block_selection_tolerance: f64 = 0.08,
    max_mapped_bytes: usize = 2 * 1024 * 1024 * 1024,
    /// Optional device-index to NUMA-node override. When absent, complete PJRT
    /// `numa_node` attributes select local pools; incomplete or unsupported
    /// topology falls back to one shared DmaMapped pool.
    device_numa_nodes: []const usize = &.{},
};

/// What one benchmark reports: the workspace and calibration it produced, the
/// selected tuple, and the three phase times of the single summary line.
const BenchmarkReport = struct {
    resources: BenchmarkResult,
    measured_bytes_per_second: f64,
    /// Whole `benchmarkSyntheticTransfer` call, including arena mapping.
    elapsed_ns: u64,
    /// End of the device allocator warm-up to the selected tuple: the
    /// calibration ring, screening, confirmation and cohort teardown.
    calibration_ns: u64,
    device_allocator_warmup_ns: u64,
};

/// Returns a benchmark result for direct-transfer platforms and `null` for
/// platforms that use the buffered loader.
pub fn benchmarkIfSupported(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    opts: Options,
) !?BenchmarkResult {
    if (!isSupported(platform)) return null;
    return try benchmark(allocator, io, platform, opts);
}

/// Measures one representative device and returns an owned, reusable DMA
/// workspace configured for every addressable device.
pub fn benchmark(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    opts: Options,
) !BenchmarkResult {
    if (!isSupported(platform)) return error.DmaBenchmarkUnsupported;
    try validatePlatform(platform);
    try validateOptions(opts, platform.devices.len);
    var result = try benchmarkSyntheticTransfer(allocator, io, platform, opts);
    errdefer result.resources.deinit();
    logBenchmarkReport(platform, &result);
    return result.resources;
}

/// Benchmarks synthetic DmaMapped PJRT transfers on one representative device.
/// Every addressable device allocator is still warmed and retained workspace is
/// prepared for the complete platform.
fn benchmarkSyntheticTransfer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    opts: Options,
) !BenchmarkReport {
    const benchmark_started = std.Io.Timestamp.now(io, .awake);
    const resolved_numa_nodes = try resolveNumaNodes(
        allocator,
        platform,
        opts.device_numa_nodes,
    );
    defer allocator.free(resolved_numa_nodes);

    const device_warmup_started = std.Io.Timestamp.now(io, .awake);
    try platform.warmupDeviceAllocators(io);
    const calibration_started = std.Io.Timestamp.now(io, .awake);
    const device_allocator_warmup_ns = elapsedNanoseconds(
        device_warmup_started,
        calibration_started,
    );
    var source_pools: BenchmarkSourcePools = try .init(
        allocator,
        io,
        platform,
        resolved_numa_nodes,
        opts.max_mapped_bytes,
    );
    var source_pools_active = true;
    defer if (source_pools_active) source_pools.deinit();

    var session: BenchmarkSession = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
    };
    var session_active = true;
    defer if (session_active) session.deinit(&source_pools);

    const representative = try tuneDevice(
        &session,
        opts,
        &source_pools,
        0,
    );
    // The candidate cohorts are done; release their device buffers before the
    // retained working set is mapped.
    session.deinit(&source_pools);
    session_active = false;
    const calibration_ns = elapsedNanoseconds(
        calibration_started,
        std.Io.Timestamp.now(io, .awake),
    );

    const uniform_block_size = representative.value;
    const uniform_parallelism = opts.block_parallelism;
    const calibrated_node_reserves = try allocator.alloc(usize, source_pools.pools.len);
    defer allocator.free(calibrated_node_reserves);
    @memset(calibrated_node_reserves, 0);
    for (platform.devices, 0..) |_, device_index| {
        const pool_index = source_pools.device_pool_indices[device_index];
        calibrated_node_reserves[pool_index] = try std.math.add(
            usize,
            calibrated_node_reserves[pool_index],
            uniform_parallelism,
        );
    }

    try source_pools.ensureLoadBlockReserves(
        uniform_block_size,
        calibrated_node_reserves,
    );
    // Map the source working set here, before any load starts: pinned
    // allocation is slow (about 200 ms for 272 MiB of hipHostMalloc on
    // MI300X) and must not sit inside a measured load. A loader whose
    // request size exceeds `preallocated_request_size` grows the remainder
    // when it is created.
    try source_pools.ensureSourceWorkingSet(
        uniform_block_size,
        try limits.maximumCoalescedJobBlocks(
            @max(uniform_block_size, BenchmarkResult.preallocated_request_size),
            uniform_block_size,
        ),
        BenchmarkResult.preallocated_source_width,
        calibrated_node_reserves,
    );
    const resources = BenchmarkResult.adopt(
        .{
            .block_size = uniform_block_size,
            .max_in_flight_per_device = uniform_parallelism,
        },
        source_pools,
    );
    source_pools_active = false;
    return .{
        .resources = resources,
        .measured_bytes_per_second = representative.metrics.bytesPerSecond(),
        .elapsed_ns = elapsedNanoseconds(
            benchmark_started,
            std.Io.Timestamp.now(io, .awake),
        ),
        .calibration_ns = calibration_ns,
        .device_allocator_warmup_ns = device_allocator_warmup_ns,
    };
}

fn resolveNumaNodes(
    allocator: std.mem.Allocator,
    platform: *const platform_mod.Platform,
    override: []const usize,
) ![]?usize {
    const result = try allocator.alloc(?usize, platform.devices.len);
    @memset(result, null);
    if (override.len != 0) {
        for (override, result) |node, *stored| stored.* = node;
        return result;
    }

    for (platform.devices, 0..) |_, device_index| {
        const node = platform.devices[device_index].numaNode() orelse {
            @memset(result, null);
            return result;
        };
        if (node >= NumaAllocator.max_nodes) {
            @memset(result, null);
            return result;
        }
        result[device_index] = node;
    }
    if (comptime builtin.os.tag != .linux) @memset(result, null);
    return result;
}

fn tuneDevice(
    session: *BenchmarkSession,
    opts: Options,
    source_pools: *BenchmarkSourcePools,
    device_index: usize,
) !BenchmarkDecision {
    var block_count: usize = 0;
    var block_source_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!benchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_count += 1;
        block_source_bytes = @max(block_source_bytes, block_size * opts.block_parallelism);
    }
    // One calibration ring, sized for the largest candidate tuple, is mapped
    // once and reused by every candidate cohort.
    const pool_index = source_pools.device_pool_indices[device_index];
    if (source_pools.pools[pool_index].source.len < block_source_bytes)
        _ = try source_pools.allocate(pool_index, block_source_bytes);
    const calibration_source = source_pools.pools[pool_index].source;

    const block_candidates = try session.allocator.alloc(BenchmarkCandidate, block_count);
    defer session.allocator.free(block_candidates);
    var block_index: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!benchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_candidates[block_index] = .{
            .value = block_size,
            .cohort = try session.createCohort(device_index, block_size),
        };
        block_index += 1;
    }
    try measureBenchmarkCandidates(
        session,
        block_candidates,
        calibration_source,
        opts.block_parallelism,
        opts.duration_ns,
        opts.minimum_transfers_per_device,
        benchmark_repeats,
    );
    const block_decision = try confirmAndSelectBenchmarkCandidate(
        session,
        opts,
        block_candidates,
        calibration_source,
        opts.block_parallelism,
        opts.block_selection_tolerance,
    );
    return block_decision;
}

fn benchmarkTupleFeasible(source_len: usize, block_size: usize, parallelism: usize) bool {
    const bytes = std.math.mul(usize, block_size, parallelism) catch return false;
    return bytes <= source_len;
}

fn measureBenchmarkCandidates(
    session: *BenchmarkSession,
    candidates: []BenchmarkCandidate,
    source: []const u8,
    parallelism: usize,
    duration_ns: u64,
    minimum_transfers_per_device: u64,
    repeats: usize,
) !void {
    for (0..repeats) |repeat| {
        for (0..candidates.len) |offset| {
            const index = (offset + repeat) % candidates.len;
            const candidate = &candidates[index];
            const metrics = try runBenchmarkWindow(
                session.io,
                candidate.cohort,
                source[0 .. candidate.cohort.block_size * parallelism],
                parallelism,
                duration_ns,
                minimum_transfers_per_device,
            );
            candidate.appendMetric(metrics);
        }
    }
}

fn confirmAndSelectBenchmarkCandidate(
    session: *BenchmarkSession,
    opts: Options,
    candidates: []const BenchmarkCandidate,
    source: []const u8,
    parallelism: usize,
    tolerance: f64,
) !BenchmarkDecision {
    const medians = try session.allocator.alloc(BenchmarkRunMetrics, candidates.len);
    defer session.allocator.free(medians);
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);
    const confirmed_metrics = try session.allocator.alloc(?BenchmarkRunMetrics, candidates.len);
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
        if (!benchmarkCandidateNeedsConfirmation(
            candidates,
            candidate_index,
            peak_index,
            tolerance,
            opts.confirmation_margin,
        )) continue;
        var candidate_runs: [benchmark_repeats]BenchmarkRunMetrics = undefined;
        var baseline_runs: [benchmark_repeats]BenchmarkRunMetrics = undefined;
        for (0..benchmark_repeats) |repeat| {
            const order = if (repeat % 2 == 0)
                [_]usize{ candidate_index, peak_index }
            else
                [_]usize{ peak_index, candidate_index };
            for (order) |measured_index| {
                const measured = candidates[measured_index];
                const metrics = try runBenchmarkWindow(
                    session.io,
                    measured.cohort,
                    source[0 .. measured.cohort.block_size * parallelism],
                    parallelism,
                    opts.confirmation_duration_ns,
                    opts.confirmation_minimum_transfers_per_device,
                );
                if (measured_index == candidate_index)
                    candidate_runs[repeat] = metrics
                else
                    baseline_runs[repeat] = metrics;
            }
        }
        const representative = medianMetricRatioIndex(
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
        if (ratio >= floor and candidate.value < candidates[selected_index].value)
            selected_index = index;
    }
    return .{
        .value = candidates[selected_index].value,
        .metrics = confirmed_metrics[selected_index] orelse medians[selected_index],
    };
}

fn benchmarkCandidateNeedsConfirmation(
    candidates: []const BenchmarkCandidate,
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

fn medianMetricRatioIndex(
    candidates: []const BenchmarkRunMetrics,
    baselines: []const BenchmarkRunMetrics,
) usize {
    std.debug.assert(candidates.len == baselines.len and candidates.len > 0);
    std.debug.assert(candidates.len <= benchmark_repeats);
    var order_storage: [benchmark_repeats]usize = undefined;
    const order = order_storage[0..candidates.len];
    for (order, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const BenchmarkRunMetrics,
        baselines: []const BenchmarkRunMetrics,
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

fn runBenchmarkWindow(
    io: std.Io,
    cohort: *BenchmarkCohort,
    source: []const u8,
    parallelism: usize,
    duration_ns: u64,
    minimum_transfers: u64,
) !BenchmarkRunMetrics {
    var metrics: BenchmarkAtomicMetrics = .{};
    try cohort.ensureReady(source, parallelism);

    const Worker = struct {
        cohort: *BenchmarkCohort,
        source: []const u8,
        slot: usize,
        metrics: *BenchmarkAtomicMetrics,
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
    for (0..parallelism) |slot| {
        group.concurrent(io, Worker.run, .{Worker{
            .cohort = cohort,
            .source = source,
            .slot = slot,
            .metrics = &metrics,
            .ready = &ready,
            .start = &start,
            .stop = &stop,
        }}) catch |err| {
            stop.store(true, .release);
            start.set(io);
            group.await(io) catch {};
            return err;
        };
    }
    while (ready.load(.acquire) != parallelism) try io.sleep(.fromMilliseconds(1), .awake);
    const measured_at = std.Io.Timestamp.now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed_ns = elapsedNanoseconds(measured_at, std.Io.Timestamp.now(io, .awake));
        const error_code = cohort.first_error.load(.acquire);
        if (error_code != 0) {
            stop.store(true, .release);
            try group.await(io);
            return @errorFromInt(error_code);
        }
        if (benchmarkWindowComplete(
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

fn benchmarkWindowComplete(
    elapsed_ns: u64,
    minimum_duration_ns: u64,
    completed_transfers: u64,
    minimum_transfers: u64,
) bool {
    return elapsed_ns >= minimum_duration_ns and completed_transfers >= minimum_transfers;
}

/// One line per calibration: what was selected, at what rate, and what it
/// cost. Per-arena mapping is logged where each arena is mapped.
fn logBenchmarkReport(platform: *const platform_mod.Platform, result: *const BenchmarkReport) void {
    log.info("dma_bench version=13 platform={s} devices={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} elapsed_ms={d:.3} calibration_ms={d:.3} allocator_warmup_ms={d:.3} retained_mapped_bytes={d} numa_pools={d}", .{
        @tagName(platform.target),
        platform.devices.len,
        platform.devices[0].kind(),
        result.resources.calibration.block_size,
        result.resources.calibration.max_in_flight_per_device,
        result.measured_bytes_per_second / (1024 * 1024 * 1024),
        @as(f64, @floatFromInt(result.elapsed_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.calibration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.device_allocator_warmup_ns)) / std.time.ns_per_ms,
        result.resources.retainedMappedBytes(),
        result.resources.numaPoolCount(),
    });
}

const BenchmarkCandidate = struct {
    value: usize,
    cohort: *BenchmarkCohort,
    metrics: [benchmark_repeats]BenchmarkRunMetrics = undefined,
    metrics_len: usize = 0,

    fn appendMetric(self: *BenchmarkCandidate, metric: BenchmarkRunMetrics) void {
        std.debug.assert(self.metrics_len < self.metrics.len);
        self.metrics[self.metrics_len] = metric;
        self.metrics_len += 1;
    }

    fn metricSlice(self: *const BenchmarkCandidate) []const BenchmarkRunMetrics {
        return self.metrics[0..self.metrics_len];
    }

    fn median(self: BenchmarkCandidate) BenchmarkRunMetrics {
        std.debug.assert(self.metrics_len > 0);
        var scratch = self.metrics;
        const populated = scratch[0..self.metrics_len];
        std.mem.sort(BenchmarkRunMetrics, populated, {}, struct {
            fn lessThan(_: void, lhs: BenchmarkRunMetrics, rhs: BenchmarkRunMetrics) bool {
                return lhs.bytesPerSecond() < rhs.bytesPerSecond();
            }
        }.lessThan);
        return populated[populated.len / 2];
    }
};

const BenchmarkDecision = struct {
    value: usize,
    metrics: BenchmarkRunMetrics,
};

const BenchmarkSession = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    cohorts: std.ArrayListUnmanaged(*BenchmarkCohort) = .empty,

    fn createCohort(
        self: *BenchmarkSession,
        device_index: usize,
        block_size: usize,
    ) !*BenchmarkCohort {
        const cohort = try self.allocator.create(BenchmarkCohort);
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

    fn deinit(self: *BenchmarkSession, source_pools: *const BenchmarkSourcePools) void {
        for (self.cohorts.items) |cohort| {
            cohort.deinit(source_pools.cleanupSourceForDevice(
                cohort.device_index,
                cohort.block_size,
            ));
            self.allocator.destroy(cohort);
        }
        self.cohorts.deinit(self.allocator);
        self.* = undefined;
    }
};

const BenchmarkCohort = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const platform_mod.Platform,
    device_index: usize,
    block_size: usize,
    managers: std.ArrayListUnmanaged(BenchmarkManager) = .empty,
    warmed_managers: usize = 0,
    first_error: std.atomic.Value(u16) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const platform_mod.Platform,
        device_index: usize,
        block_size: usize,
    ) BenchmarkCohort {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .device_index = device_index,
            .block_size = block_size,
        };
    }

    fn recordError(self: *BenchmarkCohort, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn transfer(
        self: *BenchmarkCohort,
        source: []const u8,
        slot: usize,
        metrics: ?*BenchmarkAtomicMetrics,
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

    fn ensureReady(self: *BenchmarkCohort, source: []const u8, parallelism: usize) !void {
        const required_bytes = std.math.mul(usize, self.block_size, parallelism) catch return error.OutOfMemory;
        if (required_bytes > source.len) return error.DmaBenchmarkPinnedBudgetExceeded;
        var dims = [_]i64{@intCast(self.block_size)};
        const shape_spec: pjrt.ShapeSpec = .init(&dims, .u8);
        const memory = self.platform.devices[self.device_index].memory(.default).?;
        while (self.managers.items.len < parallelism) {
            const manager = try self.platform.pjrt_client.createBuffersForAsyncHostToDevice(self.platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(self.platform.pjrt_api);
            const buffer = try manager.retrieveBuffer(self.platform.pjrt_api, 0);
            try self.managers.append(self.allocator, .{ .manager = manager, .buffer = buffer });
        }
        while (self.warmed_managers < parallelism) : (self.warmed_managers += 1) {
            const slot = self.warmed_managers;
            self.transfer(source, slot, null);
            self.transfer(source, slot, null);
            const error_code = self.first_error.load(.acquire);
            if (error_code != 0) return @errorFromInt(error_code);
        }
    }

    fn deinit(self: *BenchmarkCohort, source: []const u8) void {
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

const BenchmarkManager = struct {
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffer: *pjrt.Buffer,
};

const BenchmarkAtomicMetrics = struct {
    bytes: std.atomic.Value(u64) = .init(0),
    transfers: std.atomic.Value(u64) = .init(0),
};

const BenchmarkRunMetrics = struct {
    bytes: u64,
    transfers: u64,
    elapsed_ns: u64,

    fn bytesPerSecond(self: BenchmarkRunMetrics) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }
};

pub fn isSupported(platform: *const platform_mod.Platform) bool {
    return platform.target == .cuda or platform.target == .rocm or
        platform.target == .oneapi;
}

/// Owned, reusable host-DMA workspace. A workspace may be borrowed by only one
/// load at a time, must be deinitialized before its platform, and keeps all
/// registered arenas mapped until `deinit`.
pub const BenchmarkResult = struct {
    const Status = enum(u8) {
        idle,
        loading,
        destroying,
    };

    calibration: Calibration,

    workspace: BenchmarkSourcePools,
    status: std.atomic.Value(Status) = .init(.idle),

    fn adopt(
        calibration: Calibration,
        workspace: BenchmarkSourcePools,
    ) BenchmarkResult {
        return .{
            .calibration = calibration,
            .workspace = workspace,
        };
    }

    pub fn acquire(self: *BenchmarkResult) !void {
        if (self.status.cmpxchgStrong(
            .idle,
            .loading,
            .acq_rel,
            .acquire,
        ) != null) return error.DmaWorkspaceBusy;
    }

    pub fn release(self: *BenchmarkResult) void {
        const previous = self.status.swap(.idle, .release);
        std.debug.assert(previous == .loading);
    }

    pub fn retainedMappedBytes(self: *const BenchmarkResult) usize {
        return self.workspace.allocatedBytes();
    }

    pub fn maxMappedBytes(self: *const BenchmarkResult) usize {
        return self.workspace.max_mapped_bytes;
    }

    /// The widest source rung whose working set is mapped before a load
    /// starts. The adaptive controller climbs from 12 through 16 and 24 to
    /// 32 on every local backend measured so far and holds at or below 24
    /// on all of them, so every NUMA pool that feeds a device is grown at
    /// loader creation to serve 32 concurrent requests plus the lifecycle
    /// spare: a pinned slab mapped mid-load (146 ms for the first 64 MiB
    /// hipHostMalloc slab on MI300X) would otherwise land inside a scored
    /// window. A high-latency bootstrap above 32 still grows on demand.
    pub const preallocated_source_width = 32;

    /// Request size assumed when calibration maps the source working set:
    /// the default profile chunk and the HTTP/S3/GCS minimum. Local 8 MiB
    /// profiles need less; a 32 MiB HF profile grows the rest at loader
    /// creation.
    pub const preallocated_request_size: usize = 16 * 1024 * 1024;

    /// Grows the retained arenas so that `width + 1` requests of
    /// `request_blocks` blocks plus each pool's `feed_reserves` (the DMA
    /// stage's blocks: the calibrated depth per device) can be leased from
    /// each pool without mapping a slab during the load, clipped to the
    /// largest width that fits the mapped ceiling. The arenas stay owned by
    /// this benchmark result and are reused by every later loader.
    pub fn ensureSourceWorkingSet(
        self: *BenchmarkResult,
        request_blocks: usize,
        width: usize,
        feed_reserves: []const usize,
    ) !void {
        return self.workspace.ensureSourceWorkingSet(
            self.calibration.block_size,
            request_blocks,
            width,
            feed_reserves,
        );
    }

    pub fn numaPoolCount(self: *const BenchmarkResult) usize {
        return self.workspace.pools.len;
    }

    pub fn hasStrictAffinity(self: *const BenchmarkResult) bool {
        return self.workspace.pools[0].numa_allocator.node != null;
    }

    pub fn deinit(self: *BenchmarkResult) void {
        if (self.status.cmpxchgStrong(
            .idle,
            .destroying,
            .acq_rel,
            .acquire,
        ) != null) @panic("BenchmarkResult.deinit called while borrowed");
        const io = self.workspace.io;
        const mapped_bytes = self.workspace.allocatedBytes();
        const started = std.Io.Timestamp.now(io, .awake);
        self.workspace.deinit();
        const elapsed_ns = elapsedNanoseconds(started, std.Io.Timestamp.now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }
};

fn validatePlatform(platform: *const platform_mod.Platform) !void {
    if (platform.devices.len == 0 or platform.devices.len > 64)
        return error.DmaDeviceMismatch;
    const device_kind = platform.devices[0].kind();
    for (platform.devices[1..]) |device| {
        if (!std.mem.eql(u8, device_kind, device.kind()))
            return error.HeterogeneousDmaUnsupported;
    }
}

fn validateOptions(opts: Options, device_count: usize) !void {
    if (opts.block_sizes.len == 0) return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.duration_ns == 0 or opts.confirmation_duration_ns == 0)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.block_parallelism == 0 or opts.block_parallelism > limits.max_dma_parallelism)
        return error.InvalidDmaBenchmarkOptions;
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1) or
        !(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1))
        return error.InvalidDmaBenchmarkOptions;
    if (opts.device_numa_nodes.len != 0) {
        if (opts.device_numa_nodes.len != device_count)
            return error.InvalidDmaBenchmarkOptions;
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        for (opts.device_numa_nodes) |node| {
            if (node >= NumaAllocator.max_nodes)
                return error.InvalidDmaBenchmarkOptions;
        }
    }

    var has_feasible_block = false;
    for (opts.block_sizes) |block_size| {
        if (block_size == 0 or block_size > limits.max_read_request_size)
            return error.InvalidDmaBenchmarkOptions;
        if (benchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            has_feasible_block = true;
    }
    if (!has_feasible_block) return error.NoFeasibleDmaBenchmarkTuple;
}

const KnownPoolTopology = struct {
    pool_count: usize = 0,
    pool_nodes: [max_devices]usize = undefined,
    first_device_indices: [max_devices]usize = undefined,
    device_pool_indices: [max_devices]usize = undefined,

    fn init(device_numa_nodes: []const ?usize) !KnownPoolTopology {
        var result: KnownPoolTopology = .{};
        devices: for (device_numa_nodes, 0..) |maybe_node, device_index| {
            const node = maybe_node orelse return error.InvalidDmaLoadConfig;
            for (result.pool_nodes[0..result.pool_count], 0..) |known, pool_index| {
                if (known == node) {
                    result.device_pool_indices[device_index] = pool_index;
                    continue :devices;
                }
            }
            const pool_index = result.pool_count;
            result.pool_nodes[pool_index] = node;
            result.first_device_indices[pool_index] = device_index;
            result.device_pool_indices[device_index] = pool_index;
            result.pool_count += 1;
        }
        return result;
    }
};

pub const BenchmarkSourcePools = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pools: []BenchmarkSourcePool,
    device_pool_indices: []usize,
    max_mapped_bytes: usize,
    allocated_bytes: std.atomic.Value(usize) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const platform_mod.Platform,
        device_numa_nodes: []const ?usize,
        max_mapped_bytes: usize,
    ) !BenchmarkSourcePools {
        if (device_numa_nodes.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        const topology: ?KnownPoolTopology = if (device_numa_nodes[0] == null)
            null
        else
            try .init(device_numa_nodes);
        const pool_count = if (topology) |known| known.pool_count else 1;
        const pools = try allocator.alloc(BenchmarkSourcePool, pool_count);
        errdefer allocator.free(pools);
        const device_pool_indices = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(device_pool_indices);
        if (topology == null) {
            const pool = &pools[0];
            pool.numa_allocator = .{ .parent = allocator, .node = null };
            pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[0].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
            @memset(device_pool_indices, 0);
            return .{
                .allocator = allocator,
                .io = io,
                .pools = pools,
                .device_pool_indices = device_pool_indices,
                .max_mapped_bytes = max_mapped_bytes,
            };
        }

        const known = topology.?;
        @memcpy(device_pool_indices, known.device_pool_indices[0..device_numa_nodes.len]);
        for (
            known.pool_nodes[0..known.pool_count],
            known.first_device_indices[0..known.pool_count],
            pools,
        ) |node, device_index, *pool| {
            pool.numa_allocator = .{ .parent = allocator, .node = node };
            pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[device_index].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
        }
        return .{
            .allocator = allocator,
            .io = io,
            .pools = pools,
            .device_pool_indices = device_pool_indices,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    fn deinit(self: *BenchmarkSourcePools) void {
        for (self.pools) |*pool| {
            for (pool.allocations.items) |allocation| {
                allocation.deinit(pool.dma_map_allocator.allocator());
            }
            pool.allocations.deinit(self.allocator);
        }
        self.allocator.free(self.device_pool_indices);
        self.allocator.free(self.pools);
        self.* = undefined;
    }

    fn cleanupSourceForDevice(
        self: *const BenchmarkSourcePools,
        device_index: usize,
        minimum_len: usize,
    ) []const u8 {
        const pool = &self.pools[self.device_pool_indices[device_index]];
        var index = pool.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = pool.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    /// The one arena growth path: the calibration ring, the post-selection
    /// reserves, the pre-grown source working set and load-time demand growth
    /// all map here. Refuses to cross the mapped ceiling, times the mapping,
    /// and returns the new arena.
    fn allocate(self: *BenchmarkSourcePools, pool_index: usize, bytes: usize) ![]u8 {
        if (try std.math.add(usize, self.allocatedBytes(), bytes) > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;
        const pool = &self.pools[pool_index];
        const started = std.Io.Timestamp.now(self.io, .awake);
        var allocation: BenchmarkSourceAllocation = if (pool.pjrt_host_memory) |host_memory|
            .{ .pjrt_host = try .init(host_memory, bytes) }
        else blk: {
            const dma_map_allocator = pool.dma_map_allocator.allocator();
            break :blk .{ .dma_map = try dma_map_allocator.alignedAlloc(
                u8,
                .fromByteUnits(std.heap.page_size_min),
                bytes,
            ) };
        };
        errdefer allocation.deinit(pool.dma_map_allocator.allocator());
        const mapped_at = std.Io.Timestamp.now(self.io, .awake);
        try pool.allocations.append(self.allocator, allocation);
        const replacement = allocation.data();
        _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
        pool.source = replacement;
        const finished_at = std.Io.Timestamp.now(self.io, .awake);
        const map_ns = if (pool.pjrt_host_memory != null)
            0
        else
            elapsedNanoseconds(started, mapped_at);
        const elapsed_ns = elapsedNanoseconds(started, finished_at);
        const allocation_kind = if (pool.pjrt_host_memory != null) "pjrt_host" else "dma_map";
        if (pool.numa_allocator.node) |node| {
            log.info("DMA mapped arena numa_node={d} allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                node,
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        } else {
            log.info("DMA mapped arena numa_node=single allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        }
        return replacement;
    }

    fn allocatedBytes(self: *const BenchmarkSourcePools) usize {
        return self.allocated_bytes.load(.acquire);
    }

    /// Ensures every NUMA pool can feed its calibrated devices and hold one
    /// complete fixed-size source request. Independent nodes register their
    /// missing slabs concurrently; existing retained arenas are reused first.
    fn ensureLoadBlockReserves(
        self: *BenchmarkSourcePools,
        block_size: usize,
        calibrated_reserves: []const usize,
    ) !void {
        if (block_size == 0 or calibrated_reserves.len != self.pools.len)
            return error.InvalidDmaLoadConfig;
        const request_blocks = try limits.maximumCoalescedJobBlocks(
            limits.max_read_request_size,
            block_size,
        );
        const targets = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(targets);
        for (calibrated_reserves, targets) |reserve, *target|
            target.* = @max(reserve, request_blocks);
        return self.growToBlockTargets(block_size, targets);
    }

    fn usableBlocks(pool: *const BenchmarkSourcePool, block_size: usize) !usize {
        var usable: usize = 0;
        for (pool.allocations.items) |arena| {
            usable = std.math.add(usize, usable, arena.data().len / block_size) catch
                return error.DmaMappedBudgetExceeded;
        }
        return usable;
    }

    /// See `BenchmarkResult.ensureSourceWorkingSet`. Every pool gets the
    /// same source target because a strict-affinity load draws a device's
    /// blocks from its own node, plus its own DMA reserve.
    fn ensureSourceWorkingSet(
        self: *BenchmarkSourcePools,
        block_size: usize,
        request_blocks: usize,
        width: usize,
        feed_reserves: []const usize,
    ) !void {
        if (block_size == 0 or request_blocks == 0 or feed_reserves.len != self.pools.len)
            return error.InvalidDmaLoadConfig;
        const targets = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(targets);
        // Reserves first; when not even one request fits beside them the
        // reserves stay non-materialized and the source set alone is fitted.
        var with_reserves = true;
        var fitted_width = width;
        while (true) : (fitted_width -= 1) {
            const source_blocks = std.math.mul(usize, fitted_width + 1, request_blocks) catch
                return error.DmaMappedBudgetExceeded;
            var growth_bytes: usize = 0;
            for (self.pools, feed_reserves, targets) |*pool, reserve, *target| {
                const usable = try usableBlocks(pool, block_size);
                target.* = std.math.add(usize, source_blocks, if (with_reserves) reserve else 0) catch
                    return error.DmaMappedBudgetExceeded;
                growth_bytes = std.math.add(
                    usize,
                    growth_bytes,
                    (target.* -| usable) * block_size,
                ) catch return error.DmaMappedBudgetExceeded;
            }
            if (self.allocatedBytes() +| growth_bytes <= self.max_mapped_bytes) break;
            if (fitted_width == 0) {
                if (!with_reserves) return; // Leave growth to the load.
                with_reserves = false;
                fitted_width = width + 1;
            }
        }
        if (fitted_width < width or !with_reserves) {
            log.debug("DMA source working set clipped by the mapped ceiling: width={d} of {d}, reserves_materialized={}", .{
                fitted_width,
                width,
                with_reserves,
            });
        }
        return self.growToBlockTargets(block_size, targets);
    }

    /// Grows every pool that holds fewer than `block_targets[pool]` blocks,
    /// mapping the missing slabs of independent nodes concurrently. The
    /// aggregate check keeps a partial growth from crossing the ceiling.
    fn growToBlockTargets(
        self: *BenchmarkSourcePools,
        block_size: usize,
        block_targets: []const usize,
    ) !void {
        const missing_bytes = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing_bytes);
        var missing_total: usize = 0;
        for (self.pools, block_targets, missing_bytes) |*pool, target, *missing| {
            const usable_blocks = try usableBlocks(pool, block_size);
            missing.* = std.math.mul(usize, target -| usable_blocks, block_size) catch
                return error.DmaMappedBudgetExceeded;
            missing_total = std.math.add(usize, missing_total, missing.*) catch
                return error.DmaMappedBudgetExceeded;
        }
        if (missing_total == 0) return;
        const mapped_after_growth = std.math.add(
            usize,
            self.allocatedBytes(),
            missing_total,
        ) catch return error.DmaMappedBudgetExceeded;
        if (mapped_after_growth > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            pools: *BenchmarkSourcePools,
            pool_index: usize,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                _ = worker.pools.allocate(worker.pool_index, worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(
                        0,
                        @intFromError(err),
                        .release,
                        .monotonic,
                    );
                };
            }
        };
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        var group_error: ?anyerror = null;
        for (missing_bytes, 0..) |bytes, pool_index| {
            if (bytes == 0) continue;
            group.concurrent(self.io, Worker.run, .{Worker{
                .pools = self,
                .pool_index = pool_index,
                .bytes = bytes,
                .first_error = &first_error,
            }}) catch |err| {
                group_error = err;
                break;
            };
        }
        group.await(self.io) catch |err| if (group_error == null) {
            group_error = err;
        };
        if (group_error) |err| return err;
        const error_code = first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
    }

    pub fn blockPoolArenaProvider(self: *BenchmarkSourcePools) mem.DmaBlockPool.ArenaProvider {
        return .{
            .context = self,
            .node_count = self.pools.len,
            .arenaCountFn = struct {
                fn call(context: *anyopaque, node_index: usize) usize {
                    const pools: *BenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items.len;
                }
            }.call,
            .arenaFn = struct {
                fn call(context: *anyopaque, node_index: usize, arena_index: usize) []u8 {
                    const pools: *BenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items[arena_index].data();
                }
            }.call,
            .allocateFn = struct {
                fn call(context: *anyopaque, node_index: usize, len: usize) ![]u8 {
                    const pools: *BenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.allocate(node_index, len);
                }
            }.call,
            .mappedBytesFn = struct {
                fn call(context: *anyopaque) usize {
                    const pools: *BenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.allocatedBytes();
                }
            }.call,
        };
    }
};

const BenchmarkSourcePool = struct {
    numa_allocator: NumaAllocator,
    dma_map_allocator: mem.DmaMapAllocator,
    pjrt_host_memory: ?*const platform_mod.Memory,
    allocations: std.ArrayListUnmanaged(BenchmarkSourceAllocation) = .empty,
    source: []u8 = &.{},
};

const BenchmarkSourceAllocation = union(enum) {
    dma_map: []u8,
    pjrt_host: PjrtPinnedHostAllocation,

    fn data(self: *const BenchmarkSourceAllocation) []u8 {
        return switch (self.*) {
            .dma_map => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }

    fn deinit(
        self: BenchmarkSourceAllocation,
        dma_map_allocator: std.mem.Allocator,
    ) void {
        switch (self) {
            .dma_map => |bytes| dma_map_allocator.free(bytes),
            .pjrt_host => |allocation| allocation.deinit(),
        }
    }
};

const PjrtPinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,

    fn init(memory: *const platform_mod.Memory, size: usize) !PjrtPinnedHostAllocation {
        const api = memory.platform.pjrt_api;
        const buffer = try memory.platform.pjrt_client.createUninitializedBuffer(api, .{
            .dims = &.{@intCast(size)},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = memory.pjrt_memory },
        });
        errdefer buffer.deinit(api);
        if (!buffer.isOnCpu(api)) return error.PinnedHostMemoryNotHostVisible;

        // The writable pointer is borrowed from PJRT. Keep both the external
        // reference and its owning buffer alive for the arena's whole lifetime.
        try buffer.increaseExternalReferenceCount(api);
        errdefer buffer.decreaseExternalReferenceCount(api) catch {};
        const ptr: [*]u8 = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
        return .{
            .buffer = buffer,
            .api = api,
            .data = ptr[0..size],
        };
    }

    fn deinit(self: PjrtPinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

const NumaAllocator = struct {
    const max_nodes = 1024;
    const mpol_bind = 2;

    parent: std.mem.Allocator,
    node: ?usize,

    fn allocator(self: *NumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        const allocation = self.parent.rawAlloc(len, alignment, ret_addr) orelse return null;
        const node = self.node orelse return allocation;
        if (comptime builtin.os.tag != .linux) {
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }

        const word_bits = @bitSizeOf(usize);
        var node_mask: [max_nodes / word_bits]usize = @splat(0);
        node_mask[node / word_bits] = @as(usize, 1) << @intCast(node % word_bits);
        const rc = std.os.linux.syscall6(
            .mbind,
            @intFromPtr(allocation),
            len,
            mpol_bind,
            @intFromPtr(&node_mask),
            // Linux get_nodes() decrements maxnode before copying the mask;
            // raw callers include the same extra sentinel bit as libnuma.
            node + 2,
            0,
        );
        if (std.os.linux.errno(rc) != .SUCCESS) {
            log.err("unable to bind DMA benchmark allocation ({Bi:.2}) to NUMA node {d}: {s}", .{
                len,
                node,
                @tagName(std.os.linux.errno(rc)),
            });
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }
        return allocation;
    }

    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

test "DMA benchmark is optional on buffered platforms" {
    var platform: platform_mod.Platform = .{
        .arena = undefined,
        .target = .cpu,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .state = .init(.cpu),
        .devices = &.{},
        .memories = &.{},
        .physical_mesh = undefined,
        .replicated_sharding = undefined,
        .shardings = .empty,
    };

    try std.testing.expect(
        try benchmarkIfSupported(std.testing.allocator, std.testing.io, &platform, .{}) == null,
    );
}

test "DMA benchmark validates options" {
    try validateOptions(.{}, 8);
    try std.testing.expectError(
        error.InvalidDmaBenchmarkOptions,
        validateOptions(.{ .block_parallelism = 0 }, 8),
    );
}

test "DMA benchmark completion target has no time cap" {
    try std.testing.expect(!benchmarkWindowComplete(9, 10, 128, 128));
    try std.testing.expect(!benchmarkWindowComplete(10, 10, 127, 128));
    try std.testing.expect(benchmarkWindowComplete(10, 10, 128, 128));
    try std.testing.expect(benchmarkWindowComplete(1_000, 10, 128, 128));
    try std.testing.expect(benchmarkWindowComplete(10, 10, 0, 0));
}

test "DMA benchmark selection uses medians and prefers the smallest near-peak value" {
    // No candidate below is borderline, so the production selector never
    // schedules a confirmation window and the session supplies only its
    // allocator.
    var session: BenchmarkSession = .{
        .allocator = std.testing.allocator,
        .io = undefined,
        .platform = undefined,
    };
    const opts: Options = .{};

    var candidates = [_]BenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
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
    const decision = try confirmAndSelectBenchmarkCandidate(
        &session,
        opts,
        &candidates,
        &.{},
        opts.block_parallelism,
        0.05,
    );
    try std.testing.expectEqual(@as(usize, 4), decision.value);
    try std.testing.expectEqual(@as(f64, 98), decision.metrics.bytesPerSecond());

    // A dip between two near-peak values must not end the scan early.
    var bimodal = [_]BenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
        .{ .value = 16, .cohort = undefined },
    };
    const bimodal_rates = [_]u64{ 80, 100, 70, 99 };
    for (&bimodal, bimodal_rates) |*candidate, rate| {
        candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    const bimodal_decision = try confirmAndSelectBenchmarkCandidate(
        &session,
        opts,
        &bimodal,
        &.{},
        opts.block_parallelism,
        0.05,
    );
    try std.testing.expectEqual(@as(usize, 4), bimodal_decision.value);
    try std.testing.expectEqual(@as(f64, 100), bimodal_decision.metrics.bytesPerSecond());
}

test "DMA benchmark tuple feasibility rejects pinned budget overflow" {
    try std.testing.expect(benchmarkTupleFeasible(128, 16, 8));
    try std.testing.expect(!benchmarkTupleFeasible(127, 16, 8));
    try std.testing.expect(!benchmarkTupleFeasible(std.math.maxInt(usize), std.math.maxInt(usize), 2));
}

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    var candidates = [_]BenchmarkCandidate{
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
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
    try std.testing.expect(benchmarkCandidateNeedsConfirmation(
        &candidates,
        0,
        1,
        0.05,
        0.02,
    ));
}
