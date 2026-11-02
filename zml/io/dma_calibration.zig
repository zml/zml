//! DMA sizing for the direct backend: the block size, decided once when the
//! backend is created. It screens `default_block_sizes` (2 to 32 MiB), or
//! the one block `zml.io.Loader.Options.dma_block_size` gives, on one
//! representative device, at `depth` transfers per device and over fixed
//! windows, and keeps the smallest block within 8% of the measured peak;
//! borderline candidates get longer, alternating confirmation windows. The
//! ring it measures in stays mapped as the load's first pinned capacity.
//! CPU returns the smallest candidate without measuring: its plugin's
//! transfer is a memcpy, and a load takes the same time at every block
//! size. The buffered backend has nothing to size. `measure` runs the same
//! warm-up and screen without a loader, with every knob, for
//! `examples/io`'s `dma-bench`.
const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Platform = @import("../platform.zig").Platform;
const backend = @import("backend.zig");
const host_memory = @import("host_memory.zig");
const log = @import("log.zig").io;

/// Blocks in flight per device: the screen runs `depth` transfers (the
/// default `Options.block_parallelism`), and the direct backend's DMA
/// budget is `depth` blocks per device, in bytes (`DevicePump.hasRoom` in
/// direct_loader.zig). Eight repeatedly won or tied on MI300X; wider stages
/// mostly increased callback latency and pinned memory, and an adaptive DMA
/// width added state with little load benefit. At 32, four GB300 loaded
/// DeepSeek 9% faster for 2.5x the pinned memory.
pub const depth: usize = 8;

/// Screened, not fixed: eight MI300X loaded replicated Gemma in 7.83 s at
/// 16 MiB against 10.69 s at 8, whereas one B70 lost 10.5% at 16 MiB.
pub const default_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

/// The screen's knobs. The loader sets only `block_sizes` and keeps the
/// other defaults; `measure` takes them all.
pub const Options = struct {
    block_sizes: []const usize = &default_block_sizes,
    /// Transfers in flight per device in the screen, at most
    /// `max_block_parallelism`. The loader keeps `depth`.
    block_parallelism: usize = depth,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until the representative device completes the transfer target.
    /// On eight MI300X, 2 ms/32-transfer windows calibrate in 0.96 s against
    /// 4.83 s at 10 ms/128, selecting the same block. Short screens sometimes
    /// selected the wrong block under noise, hence the longer borderline
    /// confirmation windows (`confirmation_duration`).
    minimum_duration: std.Io.Duration = .fromMilliseconds(2),
    minimum_transfers: u64 = 32,
    /// Prefer the smallest block within this tolerance of the measured peak
    /// transfer throughput.
    block_selection_tolerance: f64 = 0.08,
};

/// The widest screen `measure` takes.
pub const max_block_parallelism: usize = 32;

/// Borderline block candidates receive longer paired windows, alternating
/// which runs first. A candidate is borderline when its rounds disagree on
/// whether it is near the peak, or when its median ratio to the peak lies
/// within `confirmation_margin` of the tolerance's floor.
const confirmation_duration: std.Io.Duration = .fromMilliseconds(25);
const confirmation_minimum_transfers: u64 = 256;
const confirmation_margin: f64 = 0.02;

/// Measures as the direct loader's creation does, without a loader and with
/// every knob of `opts`, and returns the block it selects: warms the device
/// allocators, then screens the blocks in a workspace of its own, released
/// before it returns. CPU returns the smallest block without measuring; a
/// platform that loads through the buffered backend (`backend.loadsDirect`)
/// returns `UnsupportedPlatform`. A knob out of range is `InvalidOptions`,
/// logged. For `examples/io`'s `dma-bench`.
pub fn measure(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, opts: Options) !usize {
    try validate(opts);
    if (!backend.loadsDirect(platform)) {
        log.err("DMA calibration: {t} loads through the buffered backend, which has no DMA to measure: UnsupportedPlatform", .{platform.target});
        return error.UnsupportedPlatform;
    }
    try platform.warmupDeviceAllocators(io);
    var workspace = try host_memory.Workspace.init(allocator, io, platform);
    defer workspace.deinit();
    return calibrate(&workspace, platform, opts);
}

/// A block is at most the largest source request, the bound
/// `zml.io.Loader.init` puts on its own block sizes, so every candidate's
/// ring of `block_parallelism` blocks fits the workspace's mapped ceiling.
fn validate(opts: Options) error{InvalidOptions}!void {
    comptime std.debug.assert(backend.max_read_request_size * max_block_parallelism <= host_memory.Workspace.mapped_bytes_ceiling);
    if (opts.block_parallelism == 0 or opts.block_parallelism > max_block_parallelism) {
        log.err("invalid DMA calibration options: block_parallelism={d}, expected 1..{d}: InvalidOptions", .{ opts.block_parallelism, max_block_parallelism });
        return error.InvalidOptions;
    }
    if (opts.block_sizes.len == 0) {
        log.err("invalid DMA calibration options: block_sizes is empty: InvalidOptions", .{});
        return error.InvalidOptions;
    }
    for (opts.block_sizes) |block_size| {
        if (block_size == 0 or block_size > backend.max_read_request_size) {
            log.err("invalid DMA calibration options: block size {d}, expected 1..{d}: InvalidOptions", .{ block_size, backend.max_read_request_size });
            return error.InvalidOptions;
        }
    }
    if (opts.minimum_duration.nanoseconds <= 0) {
        log.err("invalid DMA calibration options: minimum_duration={f}, expected more than zero: InvalidOptions", .{opts.minimum_duration});
        return error.InvalidOptions;
    }
    // Written so that NaN fails too.
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1)) {
        log.err("invalid DMA calibration options: block_selection_tolerance={d}, expected [0, 1): InvalidOptions", .{opts.block_selection_tolerance});
        return error.InvalidOptions;
    }
}

/// Measures one representative device into the direct loader's workspace
/// and returns the block it selects; CPU returns the smallest without
/// measuring. On a platform mixing device kinds, device 0's block serves
/// every device: it is a tuning choice, valid for all of them. `opts` were
/// checked (`validate`, or `zml.io.Loader.init` for the loader's block
/// sizes). The rate the `dma_bench` line logs is that one device's
/// synthetic H2D rate, not the platform's nor a prediction of a
/// checkpoint's load speed.
pub fn calibrate(
    workspace: *host_memory.Workspace,
    platform: *const Platform,
    opts: Options,
) !usize {
    // Every candidate loads alike on CPU: four PJRT CPU devices loaded warm
    // sharded Llama in 1.55-1.60 s at 2, 8 and 16 MiB, so the smallest,
    // which maps the least, is taken unmeasured and the loader grows its
    // own arenas.
    if (platform.target == .cpu) return std.mem.min(usize, opts.block_sizes);

    const allocator = workspace.allocator;
    const io = workspace.io;
    const start: std.Io.Timestamp = .now(io, .awake);

    var max_total_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        max_total_bytes = @max(max_total_bytes, block_size * opts.block_parallelism);
    }
    // Map one ring for the largest candidate at the fixed transfer width;
    // every candidate reuses it, and the load keeps it.
    const calibration_source = try workspace.allocate(max_total_bytes);

    var session = try Session.init(allocator, io, platform, opts.block_sizes, opts.block_parallelism);
    defer session.deinit();

    const candidates = try allocator.alloc(Candidate, opts.block_sizes.len);
    defer allocator.free(candidates);
    for (candidates, opts.block_sizes, 0..) |*candidate, block_size, index| {
        candidate.* = .{ .block_size = block_size, .buffer_offset = index * opts.block_parallelism };
    }
    try measureAllCandidates(&session, opts, candidates, calibration_source);
    const winner = candidates[try selectCandidate(&session, opts, candidates, calibration_source)];

    log.debug("dma_bench platform={s} devices={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} duration={f} retained_mapped_bytes={d}", .{
        @tagName(platform.target),
        platform.devices.len,
        platform.devices[0].kind(),
        winner.block_size,
        opts.block_parallelism,
        winner.median().bytesPerSecond() / (1024 * 1024 * 1024),
        start.durationTo(.now(io, .awake)),
        workspace.mappedBytes(),
    });
    return winner.block_size;
}

const sample_count = 3;

fn measureAllCandidates(
    session: *Session,
    opts: Options,
    candidates: []Candidate,
    source: []const u8,
) !void {
    for (0..sample_count) |round| {
        for (0..candidates.len) |i| {
            const candidate = &candidates[(i + round) % candidates.len];
            candidate.metrics[round] = try runWindow(
                session.io,
                session,
                candidate,
                source,
                opts.minimum_duration,
                opts.minimum_transfers,
            );
        }
    }
}

/// Returns the index of the selected candidate (`pick`).
fn selectCandidate(
    session: *Session,
    opts: Options,
    candidates: []Candidate,
    source: []const u8,
) !usize {
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);

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

    // A borderline candidate's ratio is its confirmation's against the peak.
    for (candidates, 0..) |_, candidate_index| {
        if (isCloseToPeak(
            candidates,
            candidate_index,
            peak_index,
            opts.block_selection_tolerance,
        )) {
            ratios[candidate_index] = try confirmCandidateAgainstPeak(
                session,
                &candidates[candidate_index],
                &candidates[peak_index],
                source,
            );
        }
    }
    return pick(candidates, ratios, opts.block_selection_tolerance);
}

/// The smallest block whose ratio is within `tolerance` of the best ratio,
/// screened or confirmed. A confirmation can rate a block above the screen's
/// peak, so the best ratio may belong to a larger block.
fn pick(candidates: []const Candidate, ratios: []const f64, tolerance: f64) usize {
    var best_index: usize = 0;
    for (ratios, 0..) |ratio, index| {
        if (ratio > ratios[best_index]) best_index = index;
    }
    const floor = ratios[best_index] * (1.0 - tolerance);
    var selected_index = best_index;
    for (candidates, ratios, 0..) |candidate, ratio, index| {
        if (ratio >= floor and candidate.block_size < candidates[selected_index].block_size)
            selected_index = index;
    }
    return selected_index;
}

fn isCloseToPeak(
    candidates: []const Candidate,
    candidate_index: usize,
    peak_index: usize,
    tolerance: f64,
) bool {
    if (candidate_index == peak_index) return false;
    const candidate = candidates[candidate_index];
    const peak = candidates[peak_index];

    // Whether the candidate is near-peak in one round but fails in others.
    var qualified_once = false;
    var rejected_once = false;
    for (candidate.metrics, 0..) |metric, round| {
        var round_peak_rate: f64 = 0;
        for (candidates) |round_candidate| {
            round_peak_rate = @max(round_peak_rate, round_candidate.metrics[round].bytesPerSecond());
        }
        const ratio = if (round_peak_rate == 0) 0 else metric.bytesPerSecond() / round_peak_rate;
        if (ratio >= 1.0 - tolerance)
            qualified_once = true
        else
            rejected_once = true;
    }
    if (qualified_once and rejected_once) return true;

    // Whether the candidate's ratio is close enough overall.
    const candidate_rate = candidate.median().bytesPerSecond();
    const peak_rate = peak.median().bytesPerSecond();
    const ratio = if (peak_rate == 0) 0 else candidate_rate / peak_rate;
    return @abs(ratio - (1.0 - tolerance)) <= confirmation_margin;
}

/// The median, over `sample_count` paired windows, of `candidate`'s rate
/// over `peak`'s.
fn confirmCandidateAgainstPeak(
    session: *Session,
    candidate: *Candidate,
    peak: *Candidate,
    source: []const u8,
) !f64 {
    var ratios: [sample_count]f64 = undefined;
    for (&ratios, 0..) |*ratio, repeat| {
        const order = if (repeat % 2 == 0)
            [_]*Candidate{ candidate, peak }
        else
            [_]*Candidate{ peak, candidate };
        var candidate_rate: f64 = undefined;
        var peak_rate: f64 = undefined;
        for (order) |measured| {
            const rate = (try runWindow(
                session.io,
                session,
                measured,
                source,
                confirmation_duration,
                confirmation_minimum_transfers,
            )).bytesPerSecond();
            if (measured == candidate) candidate_rate = rate else peak_rate = rate;
        }
        ratio.* = if (peak_rate == 0) 0 else candidate_rate / peak_rate;
    }
    std.mem.sort(f64, &ratios, {}, std.sort.asc(f64));
    return ratios[ratios.len / 2];
}

/// Runs one window of `candidate`'s transfers, `session.parallelism` at a
/// time. `io` sleeps and joins the window; the workers transfer on the
/// session's.
fn runWindow(
    io: std.Io,
    session: *Session,
    candidate: *Candidate,
    source: []const u8,
    minimum_duration: std.Io.Duration,
    minimum_transfers: u64,
) !Measurement {
    const Worker = struct {
        session: *Session,
        buffer_index: usize,
        block: []const u8,
        metrics: *Counters,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.session.io);
            while (!self.stop.load(.acquire)) {
                self.session.transfer(self.buffer_index, self.block, self.metrics);
                if (self.session.first_error.get() != null) return;
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
    for (0..session.parallelism) |slot| {
        try group.concurrent(io, Worker.run, .{Worker{
            .session = session,
            .buffer_index = candidate.buffer_offset + slot,
            .block = candidate.block(source, slot),
            .metrics = &metrics,
            .ready = &ready,
            .start = &start,
            .stop = &stop,
        }});
    }

    try session.warmup(candidate, source);

    while (ready.load(.acquire) != session.parallelism) try io.sleep(.fromMicroseconds(100), .awake);
    const start_at: std.Io.Timestamp = .now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed = start_at.durationTo(.now(io, .awake));
        if (session.first_error.get()) |err| return err;
        // is window complete
        if (elapsed.nanoseconds >= minimum_duration.nanoseconds and metrics.transfers.load(.acquire) >= minimum_transfers) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    stop.store(true, .release);
    try group.await(io);
    const elapsed = start_at.durationTo(.now(io, .awake));
    if (session.first_error.get()) |err| return err;
    return .{
        .bytes = metrics.bytes.load(.acquire),
        .transfers = metrics.transfers.load(.acquire),
        .elapsed = elapsed,
    };
}

/// One block size and its screen measurements. Its slots are the session's
/// buffers from `buffer_offset` on, one per transfer in flight.
const Candidate = struct {
    block_size: usize,
    buffer_offset: usize,
    /// Slots warmed so far (`Session.warmup`).
    warmed_buffers: usize = 0,

    /// One window per screen round (`measureAllCandidates`).
    metrics: [sample_count]Measurement = undefined,

    fn median(self: Candidate) Measurement {
        var sorted = self.metrics;
        std.mem.sort(Measurement, &sorted, {}, struct {
            fn lessThan(_: void, lhs: Measurement, rhs: Measurement) bool {
                return lhs.bytesPerSecond() < rhs.bytesPerSecond();
            }
        }.lessThan);
        return sorted[sorted.len / 2];
    }

    /// The ring's bytes that `slot` transfers.
    fn block(self: Candidate, source: []const u8, slot: usize) []const u8 {
        return source[slot * self.block_size ..][0..self.block_size];
    }
};

/// One transfer manager for the whole screen, sliced per candidate, and the
/// screen's first transfer error, which ends it.
const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffers: []*pjrt.Buffer,
    /// Each candidate's slots, and the transfers a window runs at a time.
    parallelism: usize,
    first_error: stdx.FirstError(pjrt.ApiError) = .{},

    fn transfer(self: *Session, buffer_index: usize, block: []const u8, metrics: ?*Counters) void {
        const api = self.platform.pjrt_api;
        const event = self.manager.transferData(api, buffer_index, block, 0, false) catch |err| {
            _ = self.first_error.set(err);
            return;
        };
        defer event.deinit(api);
        event.await(api, self.io) catch |err| {
            _ = self.first_error.set(err);
            return;
        };
        if (metrics) |output| {
            _ = output.bytes.fetchAdd(@intCast(block.len), .monotonic);
            _ = output.transfers.fetchAdd(1, .monotonic);
        }
    }

    /// Transfers twice into each of `candidate`'s slots not yet warmed.
    fn warmup(self: *Session, candidate: *Candidate, source: []const u8) !void {
        while (candidate.warmed_buffers < self.parallelism) : (candidate.warmed_buffers += 1) {
            const slot = candidate.warmed_buffers;
            self.transfer(candidate.buffer_offset + slot, candidate.block(source, slot), null);
            self.transfer(candidate.buffer_offset + slot, candidate.block(source, slot), null);
            if (self.first_error.get()) |err| return err;
        }
    }

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
            .parallelism = parallelism,
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
        .parallelism = undefined,
    };
    const opts: Options = .{ .block_selection_tolerance = 0.05 };

    var candidates = [_]Candidate{
        .{ .block_size = 2, .buffer_offset = undefined },
        .{ .block_size = 4, .buffer_offset = undefined },
        .{ .block_size = 8, .buffer_offset = undefined },
    };
    const rates = [_][sample_count]u64{
        .{ 60, 10, 62 },
        .{ 98, 99, 97 },
        .{ 100, 101, 99 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (&candidate.metrics, candidate_rates) |*metric, rate| {
            metric.* = .{ .bytes = rate, .transfers = 1, .elapsed = .fromSeconds(1) };
        }
    }
    const winner = candidates[try selectCandidate(&session, opts, &candidates, &.{})];
    try std.testing.expectEqual(@as(usize, 4), winner.block_size);
    try std.testing.expectEqual(@as(f64, 98), winner.median().bytesPerSecond());

    // A dip between two near-peak values must not end the scan early.
    var bimodal = [_]Candidate{
        .{ .block_size = 2, .buffer_offset = undefined },
        .{ .block_size = 4, .buffer_offset = undefined },
        .{ .block_size = 8, .buffer_offset = undefined },
        .{ .block_size = 16, .buffer_offset = undefined },
    };
    const bimodal_rates = [_]u64{ 80, 100, 70, 99 };
    for (&bimodal, bimodal_rates) |*candidate, rate| {
        candidate.metrics = @splat(.{ .bytes = rate, .transfers = 1, .elapsed = .fromSeconds(1) });
    }
    try std.testing.expectEqual(@as(usize, 4), bimodal[try selectCandidate(&session, opts, &bimodal, &.{})].block_size);
}

test "DMA benchmark picks the smallest block within the tolerance of the best ratio" {
    const candidates = [_]Candidate{
        .{ .block_size = 16 * 1024 * 1024, .buffer_offset = undefined },
        .{ .block_size = 32 * 1024 * 1024, .buffer_offset = undefined },
    };
    // 16 MiB is the screen's peak; a confirmation rates 32 MiB more than the
    // tolerance above it, then within it.
    try std.testing.expectEqual(1, pick(&candidates, &.{ 1.0, 1.10 }, 0.08));
    try std.testing.expectEqual(0, pick(&candidates, &.{ 1.0, 1.05 }, 0.08));
}

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    var candidates = [_]Candidate{
        .{ .block_size = 4, .buffer_offset = undefined },
        .{ .block_size = 8, .buffer_offset = undefined },
    };
    const rates = [_][sample_count]u64{
        .{ 96, 80, 97 },
        .{ 100, 100, 100 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (&candidate.metrics, candidate_rates) |*metric, rate| {
            metric.* = .{ .bytes = rate, .transfers = 1, .elapsed = .fromSeconds(1) };
        }
    }
    try std.testing.expect(isCloseToPeak(
        &candidates,
        0,
        1,
        0.05,
    ));
}

test "measure refuses a knob out of range before it touches the platform" {
    const cases = [_]Options{
        .{ .block_sizes = &.{} },
        .{ .block_sizes = &.{0} },
        .{ .block_sizes = &.{backend.max_read_request_size + 1} },
        .{ .block_sizes = &.{host_memory.Workspace.mapped_bytes_ceiling / 8 + 1} },
        .{ .block_parallelism = 0 },
        .{ .block_parallelism = max_block_parallelism + 1 },
        .{ .minimum_duration = .zero },
        .{ .minimum_duration = .fromMilliseconds(-1) },
        .{ .block_selection_tolerance = 1 },
        .{ .block_selection_tolerance = -0.01 },
        .{ .block_selection_tolerance = std.math.nan(f64) },
    };
    for (cases) |opts| {
        try std.testing.expectError(error.InvalidOptions, measure(std.testing.allocator, std.testing.io, undefined, opts));
    }
}

test "measure needs no loader and selects the smallest block on CPU" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    try std.testing.expectEqual(4 << 20, try measure(allocator, io, platform, .{ .block_sizes = &.{ 8 << 20, 4 << 20 } }));
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
        var candidate: Candidate = .{ .block_size = block_size, .buffer_offset = index * 2 };
        try session.warmup(&candidate, &source);
        try std.testing.expectEqual(2, candidate.warmed_buffers);
    }
    try std.testing.expectEqual(null, session.first_error.get());
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
    var candidate: Candidate = .{ .block_size = source.len, .buffer_offset = 0 };
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
        &session,
        &candidate,
        &source,
        .max,
        0,
    ));
}
