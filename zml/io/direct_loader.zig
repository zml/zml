const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const dma = @import("dma_calibration.zig");
const DispatchSpans = @import("dispatch_spans.zig").DispatchSpans;
const load_limits = @import("limits.zig");
const loader_types = @import("loader_types.zig");
const platform_mod = @import("../platform.zig");
const CreateOptions = platform_mod.CreateOptions;
const mem = @import("../mem.zig");
const pjrtx = @import("../pjrtx.zig");
const Platform = platform_mod.Platform;
const tracer = @import("../profiling/tracer.zig");
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");

const load_log = std.log.scoped(.@"zml/io/load");

/// Destroy a DMA event as soon as its ready callback has run, from the next
/// pump (never inside the callback), instead of at its batch's retirement:
/// live PJRT events are then bounded by the DMA width plus one pump batch
/// rather than by a submission's transfer count. Checked against the oneAPI
/// plugin under sustained load with the playground's
/// `ZML_LOAD_EVENT_RETIRE_CHECK`; set to false to keep every event until its
/// batch retires.
const retire_events_early = true;

const Parallelism = loader_types.Parallelism;
const LoaderOptions = loader_types.LoaderOptions;
const max_load_read_parallelism = load_limits.max_read_parallelism;
const max_load_read_request_size = load_limits.max_read_request_size;
const max_load_positional_iovecs = load_limits.max_positional_iovecs;
const maximumCoalescedJobBlocks = load_limits.maximumCoalescedJobBlocks;
const effectiveSourceRequestSize = load_limits.effectiveSourceRequestSize;

pub const LoadSpec = struct {
    source: *safetensors.Tensor,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
};

const VectoredLoadMetrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    source_calls: std.atomic.Value(u64) = .init(0),
    transfer_pieces: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    pending_source_jobs: std.atomic.Value(usize) = .init(0),
    /// Time workers spend waiting for a lifecycle credit, for pinned
    /// blocks, and inside the source read, summed over requests: waits
    /// above the read time mean the load is DMA-completion bound.
    lifecycle_wait_ns: std.atomic.Value(u64) = .init(0),
    block_wait_ns: std.atomic.Value(u64) = .init(0),
    read_ns: std.atomic.Value(u64) = .init(0),
    /// From a request's enqueue to its last DMA callback: how long the DMA
    /// stage holds a lifecycle credit and pinned blocks per request.
    dma_stage_ns: std.atomic.Value(u64) = .init(0),
    /// Inside `ensureState` for a claimed job's items: the first worker to
    /// touch a tensor creates its PJRT buffers and transfer managers there,
    /// and the other workers of the same tensor wait for it.
    tensor_init_ns: std.atomic.Value(u64) = .init(0),
    config_epoch: std.atomic.Value(u64) = .init(0),
    probe_epoch: u64 = std.math.maxInt(u64),
    probe_admission_start: u64 = std.math.maxInt(u64),
    probe_window_start_ns: u64 = 0,
    probe_active_reads: usize = 0,
    probe_peak_reads: usize = 0,
    probe_read_operations: u64 = 0,
    probe_read_bytes: u64 = 0,
    probe_mutex: std.Io.Mutex = .init,

    const Snapshot = struct {
        probe_epoch: u64,
        probe_window_start_ns: u64,
        probe_active_reads: usize,
        probe_peak_reads: usize,
        probe_read_operations: u64,
        probe_read_bytes: u64,
    };

    fn snapshot(self: *VectoredLoadMetrics, io: std.Io) Snapshot {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        return .{
            .probe_epoch = self.probe_epoch,
            .probe_window_start_ns = self.probe_window_start_ns,
            .probe_active_reads = self.probe_active_reads,
            .probe_peak_reads = self.probe_peak_reads,
            .probe_read_operations = self.probe_read_operations,
            .probe_read_bytes = self.probe_read_bytes,
        };
    }

    fn beginRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, admission_id: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch or admission_id < self.probe_admission_start) return;
        self.probe_active_reads += 1;
        self.probe_peak_reads = @max(self.probe_peak_reads, self.probe_active_reads);
    }

    fn endRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, admission_id: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch or admission_id < self.probe_admission_start) return;
        std.debug.assert(self.probe_active_reads > 0);
        self.probe_active_reads -= 1;
    }

    fn recordProbeRead(
        self: *VectoredLoadMetrics,
        io: std.Io,
        epoch: u64,
        admission_id: u64,
        bytes: usize,
    ) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch or admission_id < self.probe_admission_start) return;
        // The window opens at the generation's first completion, which is
        // not counted: from then on completions arrive at the source's
        // steady rate, whereas a clock started at the first admission would
        // charge a high-latency source its whole round trip and make longer
        // windows at higher rungs look faster than they are.
        if (self.probe_window_start_ns == 0) {
            self.probe_window_start_ns = @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
            return;
        }
        self.probe_read_operations +|= 1;
        self.probe_read_bytes +|= @intCast(bytes);
    }

    fn prepareProbe(
        self: *VectoredLoadMetrics,
        io: std.Io,
        epoch: u64,
        admission_start: u64,
    ) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        self.probe_epoch = std.math.maxInt(u64);
        self.probe_window_start_ns = 0;
        self.probe_active_reads = 0;
        self.probe_peak_reads = 0;
        self.probe_read_operations = 0;
        self.probe_read_bytes = 0;
        self.probe_admission_start = admission_start;
        self.probe_epoch = epoch;
        self.config_epoch.store(epoch, .release);
    }

    fn clearProbe(self: *VectoredLoadMetrics, io: std.Io) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        self.probe_epoch = std.math.maxInt(u64);
        self.probe_admission_start = std.math.maxInt(u64);
        self.probe_window_start_ns = 0;
        self.probe_active_reads = 0;
        self.probe_peak_reads = 0;
        self.probe_read_operations = 0;
        self.probe_read_bytes = 0;
    }
};

const VectoredTensorTransfer = struct {
    /// One destination buffer of a tensor. PJRT makes the buffer ready once
    /// every transfer submitted to it has completed and one of them carried
    /// the last-transfer flag, whatever their completion order: the flag
    /// only closes the buffer to further calls. The pump, the only
    /// submitter, therefore flags the submission that completes the
    /// placement's bytes, and no piece ever waits for another one.
    const Target = struct {
        manager: *pjrt.AsyncHostToDeviceTransferManager,
        device_index: usize,
        /// Bytes of the placement on this device; the pieces partition them.
        total: usize,
        /// Owned by the pump; read by the awaiting task once the batch is
        /// done and nothing submits any more.
        submitted_bytes: usize = 0,
        /// The pump issued the last-transfer call, accepted or not: PJRT
        /// then decides the buffer's outcome and a `SetBufferError` would
        /// trip its checks. A transfer that fails asynchronously is not
        /// visible here: the transfer's done event always resolves without
        /// an error and the failure surfaces on the buffer's definition
        /// event when the buffer is first used.
        closed: bool = false,

        /// Whether the loader may still mark the buffer as failed.
        fn canFail(self: *const Target) bool {
            return !self.closed;
        }

        /// Whether a submission of `len` bytes closes the buffer.
        fn nextIsLast(self: *const Target, len: usize) bool {
            std.debug.assert(len != 0 and self.submitted_bytes + len <= self.total);
            return self.submitted_bytes + len == self.total;
        }

        fn noteSubmitted(self: *Target, len: usize) void {
            std.debug.assert(self.submitted_bytes + len <= self.total);
            self.submitted_bytes += len;
        }

        /// Every byte went out, so the last-transfer flag did too.
        fn fullySubmitted(self: *const Target) bool {
            return self.submitted_bytes == self.total;
        }
    };

    allocator: std.mem.Allocator,
    platform: *const Platform,
    targets: []Target,
    completed_read_bytes: std.atomic.Value(usize) = .init(0),
    progress: ?std.Progress.Node = null,

    fn initResolved(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        source: *const safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
        progress_parent: ?*std.Progress.Node,
    ) !VectoredTensorTransfer {
        const packed_shape = shape.packedShape();
        const packed_placement = try sharding.placement(packed_shape);
        const ordered_devices = sharding.devicesInCanonicalOrder();
        const targets = try allocator.alloc(Target, ordered_devices.len);
        errdefer allocator.free(targets);

        var pjrt_buffers: Buffer.Shards = .empty;
        var initialized: usize = 0;
        errdefer {
            for (targets[0..initialized]) |target| target.manager.deinit(platform.pjrt_api);
            for (pjrt_buffers.constSlice()) |buffer| buffer.deinit(platform.pjrt_api);
        }

        const shape_spec: pjrt.ShapeSpec = .init(
            packed_placement.shape.dims(),
            pjrtx.bufferTypeFromDtype(packed_placement.shape.dtype()),
        );
        for (ordered_devices, 0..) |device, i| {
            const memory = platform.devices[device.id].memory(.default).?;
            const manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(platform.pjrt_api);
            const pjrt_buffer = try manager.retrieveBuffer(platform.pjrt_api, 0);
            targets[i] = .{
                .manager = manager,
                .device_index = device.id,
                .total = packed_placement.shape.byteSize(),
            };
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(pjrt_buffer);
        }

        output.* = .fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());
        const progress = if (progress_parent) |parent|
            parent.start(source.name, std.math.divCeil(usize, shape.byteSize(), 1024) catch unreachable)
        else
            null;

        return .{
            .allocator = allocator,
            .platform = platform,
            .targets = targets,
            .progress = progress,
        };
    }

    fn deinit(self: *VectoredTensorTransfer) void {
        if (self.progress) |*progress| progress.end();
        for (self.targets) |target| target.manager.deinit(self.platform.pjrt_api);
        self.allocator.free(self.targets);
    }

    fn recordReadProgress(self: *VectoredTensorTransfer, bytes: usize) void {
        const completed = self.completed_read_bytes.fetchAdd(bytes, .acq_rel) + bytes;
        if (self.progress) |*progress| {
            progress.setCompletedItems(std.math.divCeil(usize, completed, 1024) catch unreachable);
        }
    }
};

/// A value initialized at most once by whichever task touches it first.
/// Concurrent callers wait on the event; a failed initialization keeps its
/// error code and re-materializes the same error for every later caller.
fn LazyOnce(comptime T: type, comptime Ctx: type, comptime initFn: fn (Ctx) anyerror!T) type {
    return struct {
        const Self = @This();
        const Status = enum(u8) {
            uninitialized,
            initializing,
            ready,
            failed,
        };

        value: T = undefined,
        status: std.atomic.Value(Status) = .init(.uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(self: *Self, io: std.Io, ctx: Ctx) !*T {
            while (true) switch (self.status.load(.acquire)) {
                .uninitialized => {
                    if (self.status.cmpxchgStrong(.uninitialized, .initializing, .acq_rel, .acquire) != null) continue;
                    self.value = initFn(ctx) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(.failed, .release);
                        self.initialized.set(io);
                        return err;
                    };
                    self.status.store(.ready, .release);
                    self.initialized.set(io);
                    return &self.value;
                },
                .initializing => self.initialized.waitUncancelable(io),
                .ready => return &self.value,
                .failed => return @errorFromInt(self.error_code.load(.acquire)),
            };
        }

        /// The value when initialization has completed successfully.
        fn readyValue(self: *Self) ?*T {
            return if (self.status.load(.acquire) == .ready) &self.value else null;
        }
    };
}

const LoaderSourceSlot = struct {
    const OpenContext = struct { io: std.Io, uri: []const u8 };

    fn openFile(ctx: OpenContext) !std.Io.File {
        return std.Io.Dir.openFile(.cwd(), ctx.io, ctx.uri, .{ .mode = .read_only });
    }

    uri: []const u8,
    file: LazyOnce(std.Io.File, OpenContext, openFile) = .{},

    fn ensure(self: *LoaderSourceSlot, io: std.Io) !std.Io.File {
        const file = try self.file.ensure(io, .{ .io = io, .uri = self.uri });
        return file.*;
    }

    fn deinit(self: *LoaderSourceSlot, io: std.Io) void {
        if (self.file.readyValue()) |file| file.close(io);
    }
};

const LoaderLoadItem = struct {
    const InitContext = struct { item: *LoaderLoadItem, direct: *DirectLoader };

    fn initTransfer(ctx: InitContext) !VectoredTensorTransfer {
        return VectoredTensorTransfer.initResolved(
            ctx.direct.allocator,
            ctx.direct.platform,
            ctx.item.source,
            ctx.item.shape,
            ctx.item.sharding,
            ctx.item.output,
            ctx.direct.progress,
        );
    }

    source: *const safetensors.Tensor,
    source_slot: *LoaderSourceSlot,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
    state: LazyOnce(VectoredTensorTransfer, InitContext, initTransfer) = .{},

    fn ensureState(self: *LoaderLoadItem, direct: *DirectLoader) !*VectoredTensorTransfer {
        return self.state.ensure(direct.io, .{ .item = self, .direct = direct });
    }

    fn deinit(self: *LoaderLoadItem, allocator: std.mem.Allocator) void {
        if (self.state.readyValue()) |state| state.deinit();
        allocator.destroy(self);
    }
};

const AdaptiveRequestGate = struct {
    limit: usize,
    in_use: usize = 0,
    closed: bool = false,
    mutex: std.Io.Mutex = .init,
    /// Admission waiters; one release wakes one of them.
    condition: std.Io.Condition = .init,
    /// `waitEmpty` waiters, woken when the gate drains. Kept apart from the
    /// admission waiters: workers spawned for a wide rung stay parked on
    /// the read gate after a backoff, and at width 1 every completion
    /// drains the gate.
    drained: std.Io.Condition = .init,

    fn init(limit: usize) AdaptiveRequestGate {
        return .{ .limit = limit };
    }

    fn acquire(self: *AdaptiveRequestGate, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed and self.in_use >= self.limit) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.closed) return false;
        self.in_use += 1;
        return true;
    }

    fn release(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.in_use > 0);
        self.in_use -= 1;
        if (self.in_use == 0) self.drained.broadcast(io);
        // One release creates one admission slot. Waking every worker here
        // turns a high adaptive cap into a thundering herd even when the
        // active limit is small.
        self.condition.signal(io);
    }

    fn waitEmpty(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.in_use != 0) self.drained.waitUncancelable(io, &self.mutex);
    }

    fn setLimit(self: *AdaptiveRequestGate, io: std.Io, new_limit: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.limit = new_limit;
        self.condition.broadcast(io);
    }

    fn inUse(self: *AdaptiveRequestGate, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.in_use;
    }

    fn currentLimit(self: *AdaptiveRequestGate, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.limit;
    }

    fn close(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }
};

/// Read permits and lifecycle credits at one width. A request holds a
/// lifecycle credit from its claim to its last DMA callback, so the credits
/// beyond the read width are the requests the DMA stage can hold and, with
/// the pinned blocks each holds, bound the pinned memory in use. They are
/// the pre-grown capacity (`retained`: the source working set plus the
/// calibrated DMA depth per device), so the DMA stage takes every block
/// the reads leave free and nothing grows during a load; above that width
/// the stage keeps `dma_stage` requests, the calibrated in-flight bytes.
/// One credit beyond the width fed the DMA engines one request at a time:
/// on a GB300 loading DeepSeek-V4 (pieces of 4 MiB and 256 KiB) that was
/// 24 GiB/s at width 16 against 44 GiB/s with eight requests queued, and
/// the workers spent 9 ms per request waiting for a credit against 3 ms
/// reading. Workers stay at `read + 1`: a worker hands its request to the
/// DMA stage and claims the next, so credits need no workers of their own.
const RequestGateLimits = struct {
    read: usize,
    lifecycle: usize,

    fn init(read: usize, feasible_width: usize, retained: usize, dma_stage: usize) RequestGateLimits {
        std.debug.assert(feasible_width > 0 and dma_stage > 0);
        const effective_read = @min(read, feasible_width);
        return .{
            .read = effective_read,
            .lifecycle = @min(feasible_width, @max(effective_read +| dma_stage, retained)),
        };
    }

    fn workers(self: RequestGateLimits) usize {
        return @min(self.lifecycle, self.read +| 1);
    }
};

/// Requests whose pieces fill the DMA stage: `per_device` blocks of
/// in-flight bytes on every device, counted in requests of `request_size`.
/// Calibration pre-grows the same blocks per device as the stage's floor;
/// the lifecycle credits bound what the stage holds beyond it.
fn dmaStageRequests(per_device: usize, devices: usize, block_size: usize, request_size: usize) usize {
    std.debug.assert(request_size > 0);
    const bytes = per_device *| devices *| block_size;
    return @max(@as(usize, 1), std.math.divCeil(usize, bytes, request_size) catch 1);
}

test "DMA stage requests cover the per-device in-flight bytes" {
    const mib = 1024 * 1024;
    try std.testing.expectEqual(@as(usize, 8), dmaStageRequests(8, 1, 16 * mib, 16 * mib));
    try std.testing.expectEqual(@as(usize, 32), dmaStageRequests(8, 4, 16 * mib, 16 * mib));
    // A 32 MiB HF request holds two blocks: half as many requests.
    try std.testing.expectEqual(@as(usize, 4), dmaStageRequests(8, 1, 16 * mib, 32 * mib));
    try std.testing.expectEqual(@as(usize, 1), dmaStageRequests(1, 1, 16 * mib, 64 * mib));
}

/// Round-robin over devices with a ready transfer whose in-flight bytes are
/// under the per-device budget. The budget is in bytes, not submissions:
/// the calibrated depth is `max_in_flight_per_device` blocks, and a piece
/// is one tensor's slice of a block, so a model of small tensors needs many
/// more pieces in flight to keep the same bytes moving (DeepSeek-V4 on a
/// GB300: 8 pieces of 2.2 MiB reached 24 GiB/s, 58 reached 44).
fn selectLoaderDmaDevice(
    active_bytes: []const usize,
    active_pieces: []const usize,
    budget_bytes: usize,
    ready_mask: u64,
    next_device: usize,
) ?usize {
    std.debug.assert(active_bytes.len > 0 and active_bytes.len <= 64);
    std.debug.assert(active_pieces.len == active_bytes.len);
    std.debug.assert(budget_bytes > 0 and next_device < active_bytes.len);
    for (0..active_bytes.len) |offset| {
        const device_index = (next_device + offset) % active_bytes.len;
        if (ready_mask & (@as(u64, 1) << @intCast(device_index)) == 0 or
            active_bytes[device_index] >= budget_bytes or
            active_pieces[device_index] >= max_dma_pieces_per_device)
        {
            continue;
        }
        return device_index;
    }
    return null;
}

/// Submissions in flight per device on top of the byte budget: a bound on
/// event and callback overhead when a block holds many tiny tensors, above
/// the depth that saturated a GB300 (64 pieces of DeepSeek-V4).
const max_dma_pieces_per_device: usize = 64;

/// One submission: its plans (one per source file, published as each file
/// is planned), the per-tensor items it writes, and every request, block and
/// event context created while loading it. `remaining` counts completion
/// units: one per published job plus a publish sentinel held until the
/// submission is sealed. A job's unit is released exactly once, by whichever
/// of these happens: its request's last reference drops (final DMA callback
/// or abandonment), a worker abandons the claimed job before a request
/// exists, or `FairVectoredReadScheduler.fail` retires it unclaimed. The
/// batch is done when `remaining` reaches zero; the awaiting task then
/// retires it, so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    /// One file's jobs in claim order with their transfer records and every
    /// context the jobs can need, allocated by the planner: one request per
    /// job, the job's blocks (`Job.blocks` slices `blocks`) and one event per
    /// planned DMA submission, handed out in submission order. Contexts hold
    /// the plan's address, so plans are heap objects freed with the batch.
    const Plan = struct {
        allocator: std.mem.Allocator,
        jobs: []FairVectoredReadScheduler.Job,
        transfers: []VectoredLoadPipeline.PlannedTransfer,
        requests: []VectoredLoadPipeline.RequestContext,
        blocks: []VectoredLoadPipeline.BlockContext,
        events: []VectoredLoadPipeline.EventContext,
        /// Event slots handed out so far; owned by `metadata_mutex`.
        events_used: usize = 0,
        source_bytes: u64,
        source_runs: usize,
        planning_ns: u64 = 0,
        /// Next job to claim; owned by the scheduler mutex.
        cursor: usize = 0,

        /// Frees the plan; the pipeline retired its contexts first.
        fn destroy(self: *Plan) void {
            const allocator = self.allocator;
            allocator.free(self.events);
            allocator.free(self.blocks);
            allocator.free(self.requests);
            allocator.free(self.transfers);
            allocator.free(self.jobs);
            allocator.destroy(self);
        }
    };

    const Diagnostics = struct {
        /// Submission number within the loader, for log correlation.
        sequence: usize = 0,
        logical_bytes: usize = 0,
        source_bytes: u64 = 0,
        source_jobs: usize = 0,
        source_runs: usize = 0,
        source_items: usize = 0,
        planned_transfers: usize = 0,
        planned_dma_submissions: usize = 0,
        /// Published plans and their planning time in total.
        plans: usize = 0,
        planning_ns: u64 = 0,
        /// The first publish; a submission without plans is stamped at its
        /// seal.
        published_at: ?std.Io.Timestamp = null,
        sealed_at: ?std.Io.Timestamp = null,
        /// Stamped by the scheduler when the first job is claimed.
        first_claim_at: ?std.Io.Timestamp = null,
        /// Awake-clock nanoseconds of the first read admitted for the batch,
        /// stamped by the worker that admitted it; 0 until then.
        first_read_ns: std.atomic.Value(u64) = .init(0),
        /// Aggregate source statistics at publish; the completion log reports
        /// the delta against them (loader-wide while batches overlap).
        source_stats: ?VFS.ReadStats = null,

        fn noteRead(self: *Diagnostics, io: std.Io) void {
            if (self.first_read_ns.load(.monotonic) != 0) return;
            _ = self.first_read_ns.cmpxchgStrong(0, awakeNs(io), .monotonic, .monotonic);
        }
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    /// Published plans in file order: appended by the submitting task and
    /// read by claims, both under the scheduler mutex; freed at `destroy`.
    plans: std.ArrayListUnmanaged(*Plan) = .empty,
    /// First plan that may hold unclaimed jobs; owned by the scheduler mutex.
    plan_cursor: usize = 0,
    /// No further plan will be published; owned by the scheduler mutex.
    sealed: bool = false,
    /// In the scheduler's queue; owned by the scheduler mutex.
    queued: bool = false,
    /// Owned by the batch and freed at `destroy`.
    items: []*LoaderLoadItem = &.{},
    remaining: std.atomic.Value(usize),
    done: std.Io.Event = .unset,
    diagnostics: Diagnostics,
    /// Set by retirement so a unit released after `done` trips `finishJobs`.
    freeing: if (builtin.mode == .Debug) bool else void = if (builtin.mode == .Debug) false else {},

    /// An open batch holding its publish sentinel and nothing else. The
    /// sentinel keeps the batch from completing before its last plan is
    /// visible.
    fn create(allocator: std.mem.Allocator, io: std.Io, diagnostics: Diagnostics) !*Batch {
        const self = try allocator.create(Batch);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .remaining = .init(1),
            .diagnostics = diagnostics,
        };
        return self;
    }

    /// Scheduler mutex. Takes ownership of a prepared plan and adds one
    /// completion unit per job; the caller reserved the list capacity, so
    /// the plan and its units appear together.
    fn appendPlanAssumeCapacity(self: *Batch, plan: *Plan, planning_ns: u64) void {
        plan.planning_ns = planning_ns;
        self.plans.appendAssumeCapacity(plan);
        _ = self.remaining.fetchAdd(plan.jobs.len, .acq_rel);
    }

    /// Scheduler mutex. The next unclaimed job in plan order, or null when
    /// every published plan is exhausted.
    fn claimJob(self: *Batch) ?FairVectoredReadScheduler.Claim {
        while (self.plan_cursor < self.plans.items.len) : (self.plan_cursor += 1) {
            const plan = self.plans.items[self.plan_cursor];
            if (plan.cursor == plan.jobs.len) continue;
            const job = plan.jobs[plan.cursor];
            plan.cursor += 1;
            return .{ .batch = self, .plan = plan, .job = job };
        }
        return null;
    }

    /// Scheduler mutex. Every published job is claimed or retired.
    fn exhausted(self: *const Batch) bool {
        for (self.plans.items[self.plan_cursor..]) |plan| {
            if (plan.cursor != plan.jobs.len) return false;
        }
        return true;
    }

    /// Scheduler mutex. Marks every unclaimed job claimed and returns their
    /// number; the caller releases their units.
    fn retireUnclaimed(self: *Batch) usize {
        var retired: usize = 0;
        for (self.plans.items[self.plan_cursor..]) |plan| {
            retired += plan.jobs.len - plan.cursor;
            plan.cursor = plan.jobs.len;
        }
        self.plan_cursor = self.plans.items.len;
        return retired;
    }

    /// Releases `count` completion units. MEMORY-ORDER RULE: this must be the
    /// caller's final access to the batch and to anything it owns (requests,
    /// blocks, events, items, plans). The last unit sets `done`, and the
    /// awaiting task frees all of it as soon as it observes the event.
    fn finishJobs(self: *Batch, count: usize) void {
        if (count == 0) return;
        if (builtin.mode == .Debug) std.debug.assert(!self.freeing);
        const previous = self.remaining.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.done.set(self.io);
    }

    /// Frees the items and the plans with their contexts, which
    /// `VectoredLoadPipeline.retireBatch` must already have retired.
    fn destroy(self: *Batch) void {
        for (self.items) |item| item.deinit(self.allocator);
        self.allocator.free(self.items);
        for (self.plans.items) |plan| plan.destroy();
        self.plans.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// After `done`: every item was touched and every target received its
    /// last transfer.
    fn fullySubmitted(self: *const Batch) bool {
        for (self.items) |item| {
            const state = item.state.readyValue() orelse return false;
            for (state.targets) |target| {
                if (!target.fullySubmitted()) return false;
            }
        }
        return true;
    }
};

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
        batch: *Batch,
        plan: *Batch.Plan,
        /// The job's block contexts; its worker registers them in order.
        blocks: []BlockContext,
        blocks_registered: usize = 0,
        pending: std.atomic.Value(usize) = .init(1), // scheduling sentinel
        completed: std.atomic.Value(bool) = .init(false),
        source_finished: std.atomic.Value(bool) = .init(false),
        read_epoch: u64,
        admission_id: u64 = 0,
        /// Awake-clock nanoseconds of the enqueue; 0 until then.
        enqueued_ns: u64 = 0,

        /// The slot of a job that was never claimed: nothing pending, so the
        /// retirement checks hold.
        const idle: RequestContext = .{
            .pipeline = undefined,
            .batch = undefined,
            .plan = undefined,
            .blocks = &.{},
            .pending = .init(0),
            .completed = .init(true),
            .source_finished = .init(true),
            .read_epoch = 0,
        };

        fn addBlock(self: *RequestContext) void {
            _ = self.pending.fetchAdd(1, .acq_rel);
        }

        fn markReadFinished(self: *RequestContext) void {
            self.finishSourceJob();
        }

        fn finishScheduling(self: *RequestContext) void {
            self.finishSourceJob();
            self.completeOne();
        }

        fn finishSourceJob(self: *RequestContext) void {
            if (!self.source_finished.swap(true, .acq_rel)) {
                const previous = self.pipeline.metrics.pending_source_jobs.fetchSub(1, .acq_rel);
                std.debug.assert(previous > 0);
            }
        }

        fn completeBlock(self: *RequestContext) void {
            self.completeOne();
        }

        fn completeOne(self: *RequestContext) void {
            const previous = self.pending.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous != 1) return;

            // Locals first: releasing the batch unit may complete the batch
            // that owns this request, so nothing is touched after it.
            const pipeline = self.pipeline;
            const batch = self.batch;
            if (self.enqueued_ns != 0) {
                _ = pipeline.metrics.dma_stage_ns.fetchAdd(awakeNs(pipeline.io) -| self.enqueued_ns, .monotonic);
            }
            self.completed.store(true, .release);
            pipeline.request_gate.release(pipeline.io);
            batch.finishJobs(1);
        }
    };

    const BlockContext = struct {
        pipeline: *VectoredLoadPipeline,
        request: *RequestContext,
        lease: mem.DmaBlockPool.Lease,

        fn complete(self: *BlockContext) void {
            if (self.lease.complete()) self.request.completeBlock();
        }
    };

    const ReadyTransfer = struct {
        target: *VectoredTensorTransfer.Target,
        block: *BlockContext,
        source_offset: usize,
        destination_offset: usize,
        len: usize,
    };

    const PlannedTransfer = struct {
        item: *LoaderLoadItem,
        block_index: usize,
        block_offset: usize,
        writer_mask: u64,
        destination_offset: usize,
        len: usize,
    };

    /// Transfers ready for one device, submitted in arrival order so the
    /// oldest requests and submissions complete first. Owned by
    /// `metadata_mutex`.
    const ReadyQueue = std.Deque(ReadyTransfer);

    const EventContext = struct {
        pipeline: *VectoredLoadPipeline,
        block: *BlockContext,
        /// Null once destroyed, by a pump or by the batch's retirement.
        pjrt_event: ?*pjrt.Event,
        err: ?*pjrt.Error = null,
        device_index: usize,
        /// Bytes this submission holds against the device's in-flight budget.
        len: usize,
        /// Link of `VectoredLoadPipeline.retired`; owned by `metadata_mutex`.
        next_retired: ?*EventContext = null,

        /// `metadata_mutex`. Destroys the event and its error, once.
        fn destroyEvent(self: *EventContext) void {
            if (self.pjrt_event) |event| event.deinit(self.pipeline.platform.pjrt_api);
            self.pjrt_event = null;
            if (self.err) |err| err.deinit(self.pipeline.platform.pjrt_api);
            self.err = null;
        }
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    pool: *mem.DmaBlockPool,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    block_size: usize,
    device_pool_indices: []const usize,
    numa_explicit: bool,
    metrics: *VectoredLoadMetrics,
    scheduler: *FairVectoredReadScheduler,
    next_read_admission: std.atomic.Value(u64) = .init(1),
    first_error: std.atomic.Value(u16) = .init(0),
    metadata_mutex: std.Io.Mutex = .init,
    ready_queues: []ReadyQueue,
    /// In-flight DMA bytes and submissions per device, owned by
    /// `metadata_mutex`.
    active_bytes_by_device: []usize,
    active_pieces_by_device: []usize,
    /// Per-device in-flight budget: the calibrated depth in blocks, in bytes.
    dma_budget_bytes: usize,
    next_device: usize = 0,
    pumping: bool = false,
    active_events: usize = 0,
    ready_entries: usize = 0,
    /// Contexts whose callback fired, for the next pump to destroy: an
    /// intrusive stack through `EventContext.next_retired`, owned by
    /// `metadata_mutex`. Empty unless `retire_events_early`.
    retired: ?*EventContext = null,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *mem.DmaBlockPool,
        read_gate: *AdaptiveRequestGate,
        request_gate: *AdaptiveRequestGate,
        block_size: usize,
        device_pool_indices: []const usize,
        numa_explicit: bool,
        metrics: *VectoredLoadMetrics,
        scheduler: *FairVectoredReadScheduler,
        dma_budget_bytes: usize,
    ) !VectoredLoadPipeline {
        std.debug.assert(platform.devices.len <= 64);
        std.debug.assert(dma_budget_bytes > 0);
        const ready_queues = try allocator.alloc(ReadyQueue, platform.devices.len);
        errdefer allocator.free(ready_queues);
        @memset(ready_queues, .empty);
        const active_bytes_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(active_bytes_by_device);
        @memset(active_bytes_by_device, 0);
        const active_pieces_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(active_pieces_by_device);
        @memset(active_pieces_by_device, 0);
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .pool = pool,
            .read_gate = read_gate,
            .request_gate = request_gate,
            .block_size = block_size,
            .device_pool_indices = device_pool_indices,
            .numa_explicit = numa_explicit,
            .metrics = metrics,
            .scheduler = scheduler,
            .ready_queues = ready_queues,
            .active_bytes_by_device = active_bytes_by_device,
            .active_pieces_by_device = active_pieces_by_device,
            .dma_budget_bytes = dma_budget_bytes,
        };
    }

    fn deinit(self: *VectoredLoadPipeline) void {
        // Every batch was retired: no DMA event, queued transfer or request
        // context may outlive its batch.
        std.debug.assert(self.active_events == 0);
        std.debug.assert(self.ready_entries == 0);
        std.debug.assert(self.retired == null);
        std.debug.assert(self.request_gate.inUse(self.io) == 0);
        for (self.ready_queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.ready_queues);
        self.allocator.free(self.active_bytes_by_device);
        self.allocator.free(self.active_pieces_by_device);
    }

    fn failed(self: *const VectoredLoadPipeline) bool {
        return self.first_error.load(.acquire) != 0;
    }

    fn errorValue(self: *const VectoredLoadPipeline) ?anyerror {
        const value = self.first_error.load(.acquire);
        return if (value == 0) null else @errorFromInt(value);
    }

    fn recordError(self: *VectoredLoadPipeline, err: anyerror) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            self.scheduler.fail(self.io);
            self.pool.close(self.io);
            self.read_gate.close(self.io);
            self.request_gate.close(self.io);
            self.abortReady();
        }
    }

    /// Takes the claimed job's request context. The claim holds the batch's
    /// completion unit; the request's final reference drop releases it.
    fn registerRequest(
        self: *VectoredLoadPipeline,
        claim: FairVectoredReadScheduler.Claim,
    ) *RequestContext {
        const request = claim.job.request;
        request.* = .{
            .pipeline = self,
            .batch = claim.batch,
            .plan = claim.plan,
            .blocks = claim.job.blocks,
            .read_epoch = 0,
        };
        return request;
    }

    /// Retires the contexts of a done batch: destroys the PJRT events a pump
    /// has not destroyed yet (from the awaiting task after their callbacks
    /// fired, exactly as `pjrt.Event.await` does), unlinks the batch's
    /// contexts from `retired` and checks that every request and block
    /// completed. Runs under `metadata_mutex` so an `abortReady` still
    /// iterating queued entries or a pump draining `retired` cannot race the
    /// free that follows.
    fn retireBatch(self: *VectoredLoadPipeline, batch: *Batch) void {
        std.debug.assert(batch.done.isSet());
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        if (builtin.mode == .Debug) batch.freeing = true;
        for (batch.plans.items) |plan| {
            for (plan.events[0..plan.events_used]) |*ctx| ctx.destroyEvent();
            for (plan.requests) |*request| {
                std.debug.assert(request.completed.load(.acquire));
                for (request.blocks[0..request.blocks_registered]) |*block| {
                    std.debug.assert(block.lease.isComplete());
                }
            }
        }
        // Only this batch's contexts have a destroyed event while still
        // linked: a pump unlinks what it destroys.
        var link = &self.retired;
        while (link.*) |ctx| {
            if (ctx.pjrt_event == null) link.* = ctx.next_retired else link = &ctx.next_retired;
        }
    }

    /// Hands a context whose callback fired to the next pump. Under
    /// `metadata_mutex`, so the batch's retirement sees it before the batch
    /// is freed.
    fn retireEvent(self: *VectoredLoadPipeline, ctx: *EventContext) void {
        self.metadata_mutex.lockUncancelable(self.io);
        ctx.next_retired = self.retired;
        self.retired = ctx;
        self.metadata_mutex.unlock(self.io);
    }

    /// `metadata_mutex`. Destroys every retired event: their callbacks have
    /// run, and only the pump or the batch's retirement destroys an event,
    /// never its own callback.
    fn destroyRetired(self: *VectoredLoadPipeline) void {
        while (self.retired) |ctx| {
            self.retired = ctx.next_retired;
            ctx.next_retired = null;
            ctx.destroyEvent();
        }
    }

    fn reserveSourceJob(self: *VectoredLoadPipeline) void {
        _ = self.metrics.pending_source_jobs.fetchAdd(1, .acq_rel);
    }

    /// Takes the request's next block context for a leased block. Only the
    /// request's worker touches its slots, in order.
    fn registerBlock(
        self: *VectoredLoadPipeline,
        request: *RequestContext,
        dma_block: mem.DmaBlockPool.Block,
        references: usize,
    ) *BlockContext {
        const block = &request.blocks[request.blocks_registered];
        block.* = .{
            .pipeline = self,
            .request = request,
            .lease = .init(self.pool, self.io, dma_block, references),
        };
        request.blocks_registered += 1;
        request.addBlock();
        return block;
    }

    fn enqueueBlocks(
        self: *VectoredLoadPipeline,
        transfers: []const PlannedTransfer,
        blocks: []BlockContext,
        queue_counts: []const usize,
    ) !void {
        std.debug.assert(queue_counts.len == self.ready_queues.len);
        self.metadata_mutex.lockUncancelable(self.io);
        errdefer self.metadata_mutex.unlock(self.io);
        // Reserve every destination before mutating any queue so the batch is
        // either fully visible or not visible at all.
        for (self.ready_queues, queue_counts) |*queue, count| {
            try queue.ensureUnusedCapacity(self.allocator, count);
        }
        for (transfers) |transfer| {
            const block = &blocks[transfer.block_index];
            const tensor = &transfer.item.state.value;
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const target = &tensor.targets[writer_index];
                self.ready_queues[target.device_index].pushBackAssumeCapacity(.{
                    .target = target,
                    .block = block,
                    .source_offset = transfer.block_offset,
                    .destination_offset = transfer.destination_offset,
                    .len = transfer.len,
                });
                self.ready_entries += 1;
            }
            _ = self.metrics.transfer_pieces.fetchAdd(1, .monotonic);
        }
        self.metadata_mutex.unlock(self.io);
        self.requestPump();
    }

    /// Drops `count` never-submitted references of a registered block. Only
    /// the worker that still holds the request's scheduling sentinel calls
    /// this, so the request cannot reach zero here and the batch stays alive
    /// until that worker's `finishScheduling`.
    fn abandonSubmissions(
        block: *BlockContext,
        count: usize,
    ) void {
        if (count == 0) return;
        for (0..count) |_| block.complete();
    }

    fn requestPump(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        if (self.pumping or self.failed()) {
            self.metadata_mutex.unlock(self.io);
            return;
        }
        self.pumping = true;
        self.metadata_mutex.unlock(self.io);
        self.pump();
    }

    fn pump(self: *VectoredLoadPipeline) void {
        while (true) {
            var selected: ?ReadyTransfer = null;
            self.metadata_mutex.lockUncancelable(self.io);
            if (retire_events_early) self.destroyRetired();
            if (!self.failed()) {
                // Any queued transfer can go: nothing waits for another
                // piece, so each queue is served in arrival order.
                var ready_mask: u64 = 0;
                for (self.ready_queues, 0..) |queue, device_index| {
                    if (queue.len != 0) ready_mask |= @as(u64, 1) << @intCast(device_index);
                }
                const device_index = selectLoaderDmaDevice(
                    self.active_bytes_by_device,
                    self.active_pieces_by_device,
                    self.dma_budget_bytes,
                    ready_mask,
                    self.next_device,
                );
                if (device_index) |index| {
                    selected = self.ready_queues[index].popFront().?;
                    self.next_device = (index + 1) % self.ready_queues.len;
                    // The piece that crosses the budget is admitted: a
                    // device with room always has a transfer in flight.
                    self.active_bytes_by_device[index] += selected.?.len;
                    self.active_pieces_by_device[index] += 1;
                    self.active_events += 1;
                    self.ready_entries -= 1;
                }
            }
            if (selected == null) {
                self.pumping = false;
                self.metadata_mutex.unlock(self.io);
                return;
            }
            self.metadata_mutex.unlock(self.io);
            self.submitOne(selected.?);
        }
    }

    /// MEMORY-ORDER RULE: `transfer.target` and `transfer.block` belong to a
    /// batch, and the block's final completion may complete that batch, after
    /// which the awaiting task frees it. Every path therefore copies what it
    /// needs into locals, retires the DMA slot with `eventCompleted`, and
    /// calls `block.complete()` last.
    fn submitOne(self: *VectoredLoadPipeline, transfer: ReadyTransfer) void {
        const device_index = transfer.target.device_index;
        const len = transfer.len;
        const block = transfer.block;
        self.submitTransfer(transfer) catch |err| {
            self.recordError(err);
            self.eventCompleted(device_index, len);
            block.complete();
        };
    }

    fn submitTransfer(self: *VectoredLoadPipeline, transfer: ReadyTransfer) !void {
        const api = self.platform.pjrt_api;
        const target = transfer.target;
        const is_last = target.nextIsLast(transfer.len);
        // Even a rejected last call closes the buffer on PJRT's side.
        if (is_last) target.closed = true;
        const event = try target.manager.transferData(
            api,
            0,
            transfer.block.lease.data[transfer.source_offset..][0..transfer.len],
            @intCast(transfer.destination_offset),
            is_last,
        );
        target.noteSubmitted(transfer.len);

        // The plan holds one event slot per planned submission; the batch
        // owns it. A pump destroys the event once its callback has run, or
        // the batch's retirement does.
        const plan = transfer.block.request.plan;
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(plan.events_used < plan.events.len);
        const ctx = &plan.events[plan.events_used];
        plan.events_used += 1;
        self.metadata_mutex.unlock(self.io);
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .device_index = target.device_index,
            .len = transfer.len,
        };

        _ = self.metrics.dma_submissions.fetchAdd(1, .monotonic);
        event.onReady(api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                // MEMORY-ORDER RULE: the batch owns `ctx_` and its block, and
                // `block.complete()` may complete that batch, after which the
                // awaiting task frees the context, the block and the batch.
                // Load every field first, store the error, retire the DMA
                // slot (which may pump on this thread), hand the context to
                // the next pump, and complete the block last.
                const pipeline = ctx_.pipeline;
                const device_index = ctx_.device_index;
                const len = ctx_.len;
                const block = ctx_.block;
                ctx_.err = err;
                // The shipped plugins resolve this event without an error
                // whatever happened to the copy (the C API wrapper sets the
                // promise with an OK status); kept for a plugin that reports.
                if (err) |pjrt_error| {
                    pipeline.recordError(pjrt_error.getCode(pipeline.platform.pjrt_api).toApiError());
                }
                pipeline.eventCompleted(device_index, len);
                // After the pump this callback may have run, so the event is
                // destroyed by a later pump or by the batch's retirement,
                // never inside its own callback.
                if (retire_events_early) pipeline.retireEvent(ctx_);
                block.complete();
            }
        }.call, ctx) catch |err| {
            // The batch owns `ctx` and destroys the event at retirement.
            event.awaitRaw(api) catch {};
            return err;
        };
    }

    fn eventCompleted(self: *VectoredLoadPipeline, device_index: usize, len: usize) void {
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(self.active_events > 0);
        std.debug.assert(self.active_bytes_by_device[device_index] >= len);
        std.debug.assert(self.active_pieces_by_device[device_index] > 0);
        self.active_events -= 1;
        self.active_bytes_by_device[device_index] -= len;
        self.active_pieces_by_device[device_index] -= 1;
        self.metadata_mutex.unlock(self.io);
        // A ready callback can be the first place an asynchronous PJRT error
        // becomes visible. Once outside the metadata lock, retire every
        // queued transfer so request lifecycles cannot wait forever on entries
        // that the failed pump will no longer submit.
        if (self.failed())
            self.abortReady()
        else
            self.requestPump();
    }

    /// Force-completes every queued transfer. A completion here may finish
    /// its batch, but the batch is retired under `metadata_mutex`, which this
    /// holds, so no queued entry is touched after its block completes.
    fn abortReady(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        for (self.ready_queues) |*queue| {
            while (queue.popFront()) |transfer| {
                transfer.block.complete();
                self.ready_entries -= 1;
            }
        }
        self.metadata_mutex.unlock(self.io);
    }
};

const VectoredReadRequest = struct {
    const Scratch = struct {
        allocator: std.mem.Allocator,
        leased: []mem.DmaBlockPool.Block,
        affinities: []mem.DmaBlockPool.Affinity,
        references: []usize,
        iovecs: [][]u8,
        queue_counts: []usize,
        pool: mem.DmaBlockPool.AcquireScratch,

        fn init(
            allocator: std.mem.Allocator,
            pool: *const mem.DmaBlockPool,
            maximum_blocks: usize,
            device_count: usize,
        ) !Scratch {
            const leased = try allocator.alloc(mem.DmaBlockPool.Block, maximum_blocks);
            errdefer allocator.free(leased);
            const affinities = try allocator.alloc(mem.DmaBlockPool.Affinity, maximum_blocks);
            errdefer allocator.free(affinities);
            const references = try allocator.alloc(usize, maximum_blocks);
            errdefer allocator.free(references);
            const iovecs = try allocator.alloc([]u8, maximum_blocks);
            errdefer allocator.free(iovecs);
            const queue_counts = try allocator.alloc(usize, device_count);
            errdefer allocator.free(queue_counts);
            const pool_scratch = try pool.acquireScratch(allocator, maximum_blocks);
            return .{
                .allocator = allocator,
                .leased = leased,
                .affinities = affinities,
                .references = references,
                .iovecs = iovecs,
                .queue_counts = queue_counts,
                .pool = pool_scratch,
            };
        }

        fn deinit(self: *Scratch) void {
            self.pool.deinit();
            self.allocator.free(self.queue_counts);
            self.allocator.free(self.iovecs);
            self.allocator.free(self.references);
            self.allocator.free(self.affinities);
            self.allocator.free(self.leased);
            self.* = undefined;
        }
    };

    fn readAbsoluteAllV(
        io: std.Io,
        file: std.Io.File,
        buffers: []const []u8,
        file_offset: u64,
        metrics: *VectoredLoadMetrics,
    ) !void {
        return safetensors.readFilePositionalAllV(
            io,
            file,
            buffers,
            file_offset,
            &metrics.source_calls,
        );
    }

    fn beginRead(
        request: *VectoredLoadPipeline.RequestContext,
        pipeline: *VectoredLoadPipeline,
    ) bool {
        if (!pipeline.read_gate.acquire(pipeline.io)) return false;
        // Generation and admission identity belong to the source-call permit,
        // not to earlier job claim or pinned-block waits.
        request.read_epoch = pipeline.metrics.config_epoch.load(.acquire);
        request.admission_id = pipeline.next_read_admission.fetchAdd(1, .monotonic);
        pipeline.metrics.beginRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        return true;
    }

    fn endRead(
        request: *VectoredLoadPipeline.RequestContext,
        pipeline: *VectoredLoadPipeline,
    ) void {
        pipeline.metrics.endRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        pipeline.read_gate.release(pipeline.io);
    }

    fn runCoalesced(
        request: *VectoredLoadPipeline.RequestContext,
        source_slot: *LoaderSourceSlot,
        pipeline: *VectoredLoadPipeline,
        file_offset: u64,
        request_len: usize,
        transfers: []const VectoredLoadPipeline.PlannedTransfer,
        direct: *DirectLoader,
        scratch: *Scratch,
    ) void {
        defer request.finishScheduling();
        if (pipeline.failed()) return;

        const file = source_slot.ensure(pipeline.io) catch |err| {
            pipeline.recordError(err);
            return;
        };
        const block_count = std.math.divCeil(usize, request_len, pipeline.block_size) catch unreachable;
        if (block_count == 0) {
            request.markReadFinished();
            return;
        }

        std.debug.assert(block_count <= scratch.leased.len);
        std.debug.assert(block_count == request.blocks.len);
        const leased = scratch.leased[0..block_count];
        @memset(leased, .{ .data = &.{}, .node_index = 0 });
        defer for (leased) |block| {
            if (block.data.len != 0) pipeline.pool.release(pipeline.io, block);
        };

        const affinities = scratch.affinities[0..block_count];
        @memset(affinities, .{});
        const references = scratch.references[0..block_count];
        @memset(references, 0);

        const queue_counts = scratch.queue_counts;
        @memset(queue_counts, 0);

        for (transfers) |transfer| {
            const init_started = awakeNs(pipeline.io);
            const tensor = transfer.item.ensureState(direct) catch |err| {
                pipeline.recordError(err);
                return;
            };
            _ = pipeline.metrics.tensor_init_ns.fetchAdd(awakeNs(pipeline.io) -| init_started, .monotonic);
            if (transfer.block_index >= block_count or
                transfer.block_offset >= pipeline.block_size or
                transfer.len > pipeline.block_size - transfer.block_offset)
            {
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            }
            references[transfer.block_index] += @popCount(transfer.writer_mask);
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                if (writer_index >= tensor.targets.len) {
                    pipeline.recordError(error.InvalidLoaderJob);
                    return;
                }
                const target = &tensor.targets[writer_index];
                queue_counts[target.device_index] += 1;
                if (pipeline.numa_explicit) {
                    const node_index = pipeline.device_pool_indices[target.device_index];
                    affinities[transfer.block_index].eligible_nodes |= @as(u64, 1) << @intCast(node_index);
                }
            }
        }

        // Every block of a job is covered by a transfer: a block without a
        // reference would never be released.
        for (references) |refs| {
            if (refs == 0) {
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            }
        }
        const block_wait_started = awakeNs(pipeline.io);
        pipeline.pool.acquireMany(pipeline.io, leased, affinities, &scratch.pool) catch |err| {
            pipeline.recordError(err);
            return;
        };
        _ = pipeline.metrics.block_wait_ns.fetchAdd(awakeNs(pipeline.io) -| block_wait_started, .monotonic);
        if (pipeline.failed()) return;

        const iovecs = scratch.iovecs[0..block_count];
        for (iovecs, leased, 0..) |*iovec, block, block_index| {
            const consumed = block_index * pipeline.block_size;
            const len = @min(pipeline.block_size, request_len - consumed);
            iovec.* = block.data[0..len];
        }

        if (!beginRead(request, pipeline)) return;
        request.batch.diagnostics.noteRead(pipeline.io);
        const read_started = awakeNs(pipeline.io);
        const read_result = readAbsoluteAllV(
            pipeline.io,
            file,
            iovecs,
            file_offset,
            pipeline.metrics,
        );
        _ = pipeline.metrics.read_ns.fetchAdd(awakeNs(pipeline.io) -| read_started, .monotonic);
        read_result catch |err| {
            endRead(request, pipeline);
            pipeline.recordError(err);
            return;
        };
        pipeline.metrics.recordProbeRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
            request_len,
        );
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        for (transfers) |transfer| transfer.item.state.value.recordReadProgress(transfer.len);
        request.markReadFinished();
        endRead(request, pipeline);
        if (pipeline.failed()) return;

        const blocks = request.blocks;
        for (leased, references) |*lease, refs| {
            _ = pipeline.registerBlock(request, lease.*, refs);
            lease.data = &.{};
        }
        request.enqueued_ns = awakeNs(pipeline.io);
        pipeline.enqueueBlocks(transfers, blocks, queue_counts) catch |err| {
            request.enqueued_ns = 0;
            for (transfers) |transfer| {
                VectoredLoadPipeline.abandonSubmissions(
                    &blocks[transfer.block_index],
                    @popCount(transfer.writer_mask),
                );
            }
            pipeline.recordError(err);
            return;
        };
    }
};

/// A strict FIFO of published batches. A batch holds one plan per source
/// file, published as soon as that file is planned. Within a plan, jobs are
/// handed out in the planned order (fair by destination-device bytes); plans
/// in file order; a later batch's first job only after every job of the
/// earlier ones. The queue holds only batches that
/// are open (their submission is still planning) or have unclaimed jobs: a
/// sealed batch is popped with its last claim, at its seal when already
/// exhausted, or by `fail`, and a batch can only be freed after `done`,
/// which needs the sentinel dropped after the seal and every job claimed or
/// retired, so a queued batch is never freed. An open head whose published
/// plans are exhausted keeps the head and the workers wait for its next
/// plan, at most one file's planning time.
const FairVectoredReadScheduler = struct {
    /// A claimable job: its source range, its transfer records and the
    /// contexts the plan preallocated for it.
    const Job = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfers: []const VectoredLoadPipeline.PlannedTransfer,
        request: *VectoredLoadPipeline.RequestContext,
        blocks: []VectoredLoadPipeline.BlockContext,
    };

    const PlanningJob = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfer_start: usize,
        transfer_len: usize,
        block_start: usize,
        block_len: usize,
    };

    const Snapshot = struct {
        remaining_jobs: usize,
    };

    /// A claimed job with the plan and batch that own it. The claim holds
    /// one of the batch's completion units until the request releases it.
    const Claim = struct {
        batch: *Batch,
        plan: *Batch.Plan,
        job: Job,
    };

    allocator: std.mem.Allocator,
    /// Open batches and batches with unclaimed jobs in publish order; `head`
    /// is the first.
    queue: std.ArrayListUnmanaged(*Batch) = .empty,
    head: usize = 0,
    unclaimed_total: usize = 0,
    stopping: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn init(allocator: std.mem.Allocator) FairVectoredReadScheduler {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *FairVectoredReadScheduler) void {
        // Every batch was awaited (and so claimed out or retired) first.
        std.debug.assert(self.head == self.queue.items.len);
        self.queue.deinit(self.allocator);
        self.* = undefined;
    }
    fn fairOrder(
        allocator: std.mem.Allocator,
        jobs: []const PlanningJob,
        physical_bytes: []const usize,
        queues: []const std.ArrayListUnmanaged(usize),
    ) ![]usize {
        const device_count = queues.len;
        if (device_count == 0 or device_count > 64) return error.DmaDeviceMismatch;
        if (physical_bytes.len != jobs.len * device_count) return error.InvalidLoaderJob;

        const order = try allocator.alloc(usize, jobs.len);
        errdefer allocator.free(order);
        const cursors = try allocator.alloc(usize, device_count);
        defer allocator.free(cursors);
        @memset(cursors, 0);
        const scheduled = try allocator.alloc(u64, device_count);
        defer allocator.free(scheduled);
        @memset(scheduled, 0);
        const claimed = try allocator.alloc(bool, jobs.len);
        defer allocator.free(claimed);
        @memset(claimed, false);

        var next_device: usize = 0;
        for (order) |*ordered_job| {
            var selected_device: ?usize = null;
            var selected_job: ?usize = null;
            for (0..device_count) |offset| {
                const device_index = (next_device + offset) % device_count;
                const queue = queues[device_index];
                while (cursors[device_index] < queue.items.len and
                    claimed[queue.items[cursors[device_index]]])
                {
                    cursors[device_index] += 1;
                }
                if (cursors[device_index] == queue.items.len) continue;
                const candidate = queue.items[cursors[device_index]];
                if (selected_device == null or
                    scheduled[device_index] < scheduled[selected_device.?])
                {
                    selected_device = device_index;
                    selected_job = candidate;
                }
            }
            const device_index = selected_device orelse return error.InvalidLoaderJob;
            const job_index = selected_job.?;
            claimed[job_index] = true;
            ordered_job.* = job_index;
            const row = physical_bytes[job_index * device_count ..][0..device_count];
            for (row, scheduled) |bytes, *total| total.* +|= @intCast(bytes);
            next_device = (device_index + 1) % device_count;
        }
        return order;
    }

    fn appendTransfers(
        allocator: std.mem.Allocator,
        output: *std.ArrayList(VectoredLoadPipeline.PlannedTransfer),
        transfer_start: usize,
        item: *LoaderLoadItem,
        tensor_offset: usize,
        len: usize,
        job_file_offset: u64,
        block_size: usize,
        dispatch: DispatchSpans,
        device_indices: []const usize,
        physical_bytes: []usize,
    ) !void {
        const piece_end = try std.math.add(usize, tensor_offset, len);
        var cursor = tensor_offset;
        var span_index = dispatch.spanIndexAt(cursor) orelse return error.InvalidLoaderJob;
        while (cursor < piece_end) {
            const span = dispatch.spans[span_index];
            const absolute = try std.math.add(u64, item.source.offset, @as(u64, @intCast(cursor)));
            if (absolute < job_file_offset) return error.InvalidLoaderJob;
            const source_relative = std.math.cast(usize, absolute - job_file_offset) orelse
                return error.InvalidLoaderJob;
            const block_index = source_relative / block_size;
            const block_offset = source_relative % block_size;
            const take = @min(
                @min(piece_end - cursor, span.end - cursor),
                block_size - block_offset,
            );
            const writer_mask = dispatch.writerMask(span);
            std.debug.assert(writer_mask != 0);
            const destination_offset = span.writer_offset + cursor - span.start;
            var merged = false;
            if (output.items.len > transfer_start) merge: {
                const previous = &output.items[output.items.len - 1];
                if (previous.item != item or previous.block_index != block_index or
                    previous.writer_mask != writer_mask or
                    previous.block_offset + previous.len != block_offset or
                    previous.destination_offset + previous.len != destination_offset)
                    break :merge;
                previous.len += take;
                merged = true;
            }
            if (!merged) try output.append(allocator, .{
                .item = item,
                .block_index = block_index,
                .block_offset = block_offset,
                .writer_mask = writer_mask,
                .destination_offset = destination_offset,
                .len = take,
            });
            var mask = writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                if (writer_index >= device_indices.len) return error.InvalidLoaderJob;
                const device_index = device_indices[writer_index];
                physical_bytes[device_index] = try std.math.add(
                    usize,
                    physical_bytes[device_index],
                    take,
                );
            }
            cursor += take;
            if (cursor == span.end) span_index += 1;
        }
    }

    /// Item indices sorted by file URI, offset, size and index: the planner's
    /// input, taken one file group at a time (`fileGroupEnd`).
    fn sortedItemOrder(allocator: std.mem.Allocator, items: []const *LoaderLoadItem) ![]usize {
        const order = try allocator.alloc(usize, items.len);
        for (order, 0..) |*index, i| index.* = i;
        const SortContext = struct {
            items: []const *LoaderLoadItem,

            fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                const left = ctx.items[lhs];
                const right = ctx.items[rhs];
                const uri_order = std.mem.order(u8, left.source.file_uri, right.source.file_uri);
                if (uri_order != .eq) return uri_order == .lt;
                if (left.source.offset != right.source.offset)
                    return left.source.offset < right.source.offset;
                const left_size = left.source.byteSize();
                const right_size = right.source.byteSize();
                if (left_size != right_size) return left_size < right_size;
                return lhs < rhs;
            }
        };
        std.mem.sort(usize, order, SortContext{ .items = items }, SortContext.lessThan);
        return order;
    }

    /// The end of the file group that begins at `order[start]`.
    fn fileGroupEnd(items: []const *LoaderLoadItem, order: []const usize, start: usize) usize {
        const uri = items[order[start]].source.file_uri;
        var end = start + 1;
        while (end < order.len and std.mem.eql(u8, uri, items[order[end]].source.file_uri)) : (end += 1) {}
        return end;
    }

    /// Plans one file: `order` indexes `items` sorted by offset then size,
    /// all on the same file. Runs of touching ranges are cut into the
    /// minimum number of jobs at tensor-safe boundaries, in a fair order
    /// across the destination devices (the planning order when there is one
    /// device); no job depends on another, since every DMA piece is
    /// submitted as soon as its block is read. The plan also carries the
    /// contexts its jobs need: one request per job, one block per job block,
    /// one event per DMA submission (a transfer's writer count).
    fn preparePlan(
        allocator: std.mem.Allocator,
        device_count: usize,
        items: []const *LoaderLoadItem,
        order: []const usize,
        block_size: usize,
        request_size: usize,
    ) !*Batch.Plan {
        const scatter_limit = std.math.mul(
            usize,
            block_size,
            max_load_positional_iovecs,
        ) catch std.math.maxInt(usize);
        const maximum_job_len = @min(request_size, scatter_limit);
        if (maximum_job_len == 0) return error.InvalidLoaderJob;
        const TensorPlan = struct {
            dispatch_spans: DispatchSpans,
            device_indices: []usize,
            total: usize,
        };
        const tensor_plans = try allocator.alloc(TensorPlan, order.len);
        var initialized_plans: usize = 0;
        defer {
            for (tensor_plans[0..initialized_plans]) |*plan| {
                plan.dispatch_spans.deinit(allocator);
                allocator.free(plan.device_indices);
            }
            allocator.free(tensor_plans);
        }
        for (order, tensor_plans) |item_index, *plan| {
            const item = items[item_index];
            std.debug.assert(std.mem.eql(u8, item.source.file_uri, items[order[0]].source.file_uri));
            const packed_shape = item.shape.packedShape();
            plan.* = .{
                .dispatch_spans = try .init(allocator, packed_shape, item.sharding),
                .device_indices = &.{},
                .total = packed_shape.byteSize(),
            };
            initialized_plans += 1;
            if (plan.total != item.source.byteSize()) return error.InvalidLoaderJob;
            const ordered_devices = item.sharding.devicesInCanonicalOrder();
            plan.device_indices = try allocator.alloc(usize, ordered_devices.len);
            for (ordered_devices, plan.device_indices) |device, *device_index| {
                device_index.* = @intCast(device.id);
                if (device_index.* >= device_count) return error.DmaDeviceMismatch;
            }
        }
        var jobs_list: std.ArrayList(PlanningJob) = .empty;
        defer jobs_list.deinit(allocator);
        var transfers_list: std.ArrayList(VectoredLoadPipeline.PlannedTransfer) = .empty;
        defer transfers_list.deinit(allocator);
        var physical_list: std.ArrayList(usize) = .empty;
        defer physical_list.deinit(allocator);
        var safe_boundaries: std.ArrayList(u64) = .empty;
        defer safe_boundaries.deinit(allocator);
        // One device is charged every job, so the fair order is the planning
        // order and the queues stay empty.
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        defer allocator.free(queues);
        @memset(queues, .empty);
        defer for (queues) |*queue| queue.deinit(allocator);
        var source_bytes: u64 = 0;
        var source_runs: usize = 0;
        var block_total: usize = 0;
        var run_cursor: usize = 0;
        while (run_cursor < order.len) {
            safe_boundaries.clearRetainingCapacity();
            const first_index = order[run_cursor];
            const first_offset = items[first_index].source.offset;
            var run_end = std.math.add(
                u64,
                first_offset,
                items[first_index].source.byteSize(),
            ) catch return error.InvalidLoaderJob;
            if (run_end == first_offset) {
                run_cursor += 1;
                continue;
            }
            var run_item_end = run_cursor + 1;
            while (run_item_end < order.len) : (run_item_end += 1) {
                const candidate = items[order[run_item_end]].source;
                if (candidate.offset > run_end) break;
                const candidate_end = std.math.add(u64, candidate.offset, candidate.byteSize()) catch
                    return error.InvalidLoaderJob;
                // A touching range starts where every preceding range has
                // ended. Unlike an arbitrary tensor end, this is safe even
                // when the batch contains overlapping or duplicate ranges.
                if (candidate.byteSize() != 0 and candidate.offset == run_end and
                    (safe_boundaries.items.len == 0 or
                        safe_boundaries.items[safe_boundaries.items.len - 1] != candidate.offset))
                {
                    try safe_boundaries.append(allocator, candidate.offset);
                }
                run_end = @max(run_end, candidate_end);
            }
            source_runs += 1;

            var job_start = first_offset;
            var candidate_start = run_cursor;
            var boundary_cursor: usize = 0;
            const run_len = run_end - first_offset;
            var jobs_remaining: usize = @intCast(std.math.divCeil(
                u64,
                run_len,
                @intCast(maximum_job_len),
            ) catch unreachable);
            while (jobs_remaining != 0) {
                const hard_end = @min(
                    run_end,
                    std.math.add(u64, job_start, maximum_job_len) catch run_end,
                );
                const job_end = if (jobs_remaining == 1)
                    run_end
                else boundary: {
                    const remaining_capacity = std.math.mul(
                        u64,
                        @intCast(jobs_remaining - 1),
                        @intCast(maximum_job_len),
                    ) catch return error.InvalidLoaderJob;
                    const minimum_end = @max(job_start + 1, run_end - remaining_capacity);
                    const maximum_end = @min(
                        hard_end,
                        run_end - @as(u64, @intCast(jobs_remaining - 1)),
                    );
                    while (boundary_cursor < safe_boundaries.items.len and
                        safe_boundaries.items[boundary_cursor] <= job_start)
                    {
                        boundary_cursor += 1;
                    }
                    var scan = boundary_cursor;
                    var selected: ?u64 = null;
                    while (scan < safe_boundaries.items.len and
                        safe_boundaries.items[scan] <= maximum_end) : (scan += 1)
                    {
                        if (safe_boundaries.items[scan] >= minimum_end)
                            selected = safe_boundaries.items[scan];
                    }
                    boundary_cursor = scan;
                    break :boundary selected orelse maximum_end;
                };
                const job_len: usize = @intCast(job_end - job_start);
                const job_index = jobs_list.items.len;
                const transfer_start = transfers_list.items.len;
                try physical_list.appendNTimes(allocator, 0, device_count);
                const row = physical_list.items[job_index * device_count ..][0..device_count];
                while (candidate_start < run_item_end) {
                    const candidate = items[order[candidate_start]].source;
                    const candidate_end = std.math.add(u64, candidate.offset, candidate.byteSize()) catch
                        return error.InvalidLoaderJob;
                    if (candidate_end > job_start) break;
                    candidate_start += 1;
                }
                for (order[candidate_start..run_item_end], candidate_start..) |item_index, position| {
                    const item = items[item_index];
                    if (item.source.offset >= job_end) break;
                    const item_end = std.math.add(u64, item.source.offset, item.source.byteSize()) catch
                        return error.InvalidLoaderJob;
                    const intersection_start = @max(job_start, item.source.offset);
                    const intersection_end = @min(job_end, item_end);
                    if (intersection_start >= intersection_end) continue;
                    try appendTransfers(
                        allocator,
                        &transfers_list,
                        transfer_start,
                        item,
                        @intCast(intersection_start - item.source.offset),
                        @intCast(intersection_end - intersection_start),
                        job_start,
                        block_size,
                        tensor_plans[position].dispatch_spans,
                        tensor_plans[position].device_indices,
                        row,
                    );
                }
                std.debug.assert(transfers_list.items.len > transfer_start);
                const block_len = std.math.divCeil(usize, job_len, block_size) catch unreachable;
                try jobs_list.append(allocator, .{
                    .source_slot = items[first_index].source_slot,
                    .file_offset = job_start,
                    .len = job_len,
                    .transfer_start = transfer_start,
                    .transfer_len = transfers_list.items.len - transfer_start,
                    .block_start = block_total,
                    .block_len = block_len,
                });
                block_total += block_len;
                source_bytes +|= @intCast(job_len);
                if (device_count > 1) {
                    for (row, queues) |bytes, *queue| {
                        if (bytes != 0) try queue.append(allocator, job_index);
                    }
                }
                job_start = job_end;
                jobs_remaining -= 1;
            }
            std.debug.assert(job_start == run_end);
            run_cursor = run_item_end;
        }

        const planning_jobs = jobs_list.items;
        var dma_submissions: usize = 0;
        for (transfers_list.items) |transfer| dma_submissions += @popCount(transfer.writer_mask);
        const transfers = try transfers_list.toOwnedSlice(allocator);
        errdefer allocator.free(transfers);
        const jobs = try allocator.alloc(Job, planning_jobs.len);
        errdefer allocator.free(jobs);
        const requests = try allocator.alloc(VectoredLoadPipeline.RequestContext, planning_jobs.len);
        errdefer allocator.free(requests);
        @memset(requests, VectoredLoadPipeline.RequestContext.idle);
        const blocks = try allocator.alloc(VectoredLoadPipeline.BlockContext, block_total);
        errdefer allocator.free(blocks);
        const events = try allocator.alloc(VectoredLoadPipeline.EventContext, dma_submissions);
        errdefer allocator.free(events);
        const plan = try allocator.create(Batch.Plan);
        errdefer allocator.destroy(plan);
        plan.* = .{
            .allocator = allocator,
            .jobs = jobs,
            .transfers = transfers,
            .requests = requests,
            .blocks = blocks,
            .events = events,
            .source_bytes = source_bytes,
            .source_runs = source_runs,
        };
        if (device_count == 1) {
            for (jobs, planning_jobs, 0..) |*job, planned, index| job.* = finalJob(plan, planned, index);
        } else {
            const fair_order = try fairOrder(allocator, planning_jobs, physical_list.items, queues);
            defer allocator.free(fair_order);
            for (jobs, fair_order, 0..) |*job, planning_index, index| {
                job.* = finalJob(plan, planning_jobs[planning_index], index);
            }
        }
        return plan;
    }

    /// The claimable job at `index` of `plan`: its transfers, its request
    /// slot and its block slots.
    fn finalJob(plan: *Batch.Plan, planned: PlanningJob, index: usize) Job {
        return .{
            .source_slot = planned.source_slot,
            .file_offset = planned.file_offset,
            .len = planned.len,
            .transfers = plan.transfers[planned.transfer_start..][0..planned.transfer_len],
            .request = &plan.requests[index],
            .blocks = plan.blocks[planned.block_start..][0..planned.block_len],
        };
    }

    /// Plans `items` one file at a time and publishes each plan as soon as
    /// it exists, so workers claim the first file while the rest is planned.
    /// Stops at the first error; the caller seals or fails the batch.
    fn publishFiles(
        self: *FairVectoredReadScheduler,
        io: std.Io,
        batch: *Batch,
        device_count: usize,
        items: []const *LoaderLoadItem,
        block_size: usize,
        request_size: usize,
    ) !void {
        const order = try sortedItemOrder(self.allocator, items);
        defer self.allocator.free(order);
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(items, order, file_start);
            const planning_started: std.Io.Timestamp = .now(io, .awake);
            const plan = try preparePlan(
                self.allocator,
                device_count,
                items,
                order[file_start..file_end],
                block_size,
                request_size,
            );
            const planning_ns: u64 = @intCast(@max(planning_started.untilNow(io, .awake).nanoseconds, 0));
            self.publish(io, batch, plan, planning_ns) catch |err| {
                plan.destroy();
                return err;
            };
            file_start = file_end;
        }
    }
    /// Publishes one plan of an open batch behind every earlier plan and
    /// batch. The batch joins the queue with its first plan that has jobs;
    /// a plan without jobs only counts in the diagnostics.
    fn publish(
        self: *FairVectoredReadScheduler,
        io: std.Io,
        batch: *Batch,
        plan: *Batch.Plan,
        planning_ns: u64,
    ) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.LoaderShuttingDown;
        std.debug.assert(!batch.sealed);
        const job_count = plan.jobs.len;
        const joins_queue = !batch.queued and job_count != 0;
        try batch.plans.ensureUnusedCapacity(batch.allocator, 1);
        if (joins_queue) try self.queue.ensureUnusedCapacity(self.allocator, 1);
        // Nothing below fails: the plan, its units and the queue entry
        // appear together.
        const diagnostics = &batch.diagnostics;
        if (diagnostics.published_at == null) diagnostics.published_at = .now(io, .awake);
        diagnostics.plans += 1;
        diagnostics.planning_ns += planning_ns;
        diagnostics.source_bytes += plan.source_bytes;
        diagnostics.source_jobs += job_count;
        diagnostics.source_runs += plan.source_runs;
        diagnostics.planned_transfers += plan.transfers.len;
        diagnostics.planned_dma_submissions += plan.events.len;
        batch.appendPlanAssumeCapacity(plan, planning_ns);
        if (joins_queue) {
            self.queue.appendAssumeCapacity(batch);
            batch.queued = true;
        }
        self.unclaimed_total += job_count;
        if (job_count != 0) self.condition.broadcast(io);
    }

    /// No further plan for the batch. A batch whose published plans are
    /// already exhausted leaves the queue here; otherwise its last claim
    /// pops it. The publisher drops the sentinel only after this, so the
    /// batch cannot complete while still queued.
    fn seal(self: *FairVectoredReadScheduler, io: std.Io, batch: *Batch) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(!batch.sealed);
        batch.sealed = true;
        const now: std.Io.Timestamp = .now(io, .awake);
        if (batch.diagnostics.published_at == null) batch.diagnostics.published_at = now;
        batch.diagnostics.sealed_at = now;
        if (batch.queued and batch.exhausted()) {
            // Only the head can have been claimed empty.
            std.debug.assert(self.queue.items[self.head] == batch);
            self.popHead();
        }
    }

    /// Scheduler mutex.
    fn popHead(self: *FairVectoredReadScheduler) void {
        self.queue.items[self.head].queued = false;
        self.head += 1;
        if (self.head == self.queue.items.len) {
            self.queue.clearRetainingCapacity();
            self.head = 0;
        }
    }

    fn stop(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
    }

    /// Stops claims and retires the unclaimed units of every plan of every
    /// queued batch so each still reaches `done` through its claimed
    /// requests (an open batch through its seal and sentinel as well).
    /// Claims and this pass both move cursors under the mutex, so they
    /// partition the jobs exactly: every claimed job keeps its unit with its
    /// worker.
    fn fail(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
        for (self.queue.items[self.head..]) |batch| {
            batch.queued = false;
            const unclaimed = batch.retireUnclaimed();
            // Last access: the retired units may complete the batch.
            batch.finishJobs(unclaimed);
        }
        self.queue.clearRetainingCapacity();
        self.head = 0;
        self.unclaimed_total = 0;
    }

    /// Hands out the head batch's next job; a sealed batch leaves the queue
    /// with its last one. An open head whose published plans are exhausted
    /// hands out nothing until its next plan is published.
    fn claim(self: *FairVectoredReadScheduler, io: std.Io) ?Claim {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.head == self.queue.items.len) return null;
        const batch = self.queue.items[self.head];
        const claimed = batch.claimJob() orelse {
            std.debug.assert(!batch.sealed);
            return null;
        };
        if (batch.diagnostics.first_claim_at == null) batch.diagnostics.first_claim_at = .now(io, .awake);
        self.unclaimed_total -= 1;
        if (batch.sealed and batch.exhausted()) self.popHead();
        return claimed;
    }
    fn waitForWork(self: *FairVectoredReadScheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and self.unclaimed_total == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }

    fn snapshot(self: *FairVectoredReadScheduler, io: std.Io) Snapshot {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .remaining_jobs = self.unclaimed_total };
    }
};

/// Planning input for the fair-order tests: one job per entry; `file_offset`
/// is the entry index.
const FairOrderJob = struct {
    physical_bytes: []const usize,
};

fn testFairOrder(
    allocator: std.mem.Allocator,
    device_count: usize,
    jobs: []const FairOrderJob,
) ![]usize {
    const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
    defer allocator.free(queues);
    @memset(queues, .empty);
    defer for (queues) |*queue| queue.deinit(allocator);
    const planning_jobs = try allocator.alloc(FairVectoredReadScheduler.PlanningJob, jobs.len);
    defer allocator.free(planning_jobs);
    const physical_bytes = try allocator.alloc(usize, jobs.len * device_count);
    defer allocator.free(physical_bytes);
    for (jobs, planning_jobs, 0..) |job, *planned, job_index| {
        if (job.physical_bytes.len != device_count) return error.InvalidTestJob;
        planned.* = .{
            .source_slot = undefined,
            .file_offset = job_index,
            .len = 1,
            .transfer_start = 0,
            .transfer_len = 0,
            .block_start = 0,
            .block_len = 0,
        };
        for (job.physical_bytes, queues, 0..) |bytes, *queue, device_index| {
            physical_bytes[job_index * device_count + device_index] = bytes;
            if (bytes != 0) try queue.append(allocator, job_index);
        }
    }
    return FairVectoredReadScheduler.fairOrder(allocator, planning_jobs, physical_bytes, queues);
}

fn expectFairOrder(
    device_count: usize,
    jobs: []const FairOrderJob,
    expected: []const usize,
) !void {
    const order = try testFairOrder(std.testing.allocator, device_count, jobs);
    defer std.testing.allocator.free(order);
    try std.testing.expectEqualSlices(usize, expected, order);
}

/// A plan of `job_count` unit jobs (`file_offset` = index), each with one
/// block slot and one event slot, and no transfers.
fn testPlan(allocator: std.mem.Allocator, job_count: usize) !*Batch.Plan {
    const jobs = try allocator.alloc(FairVectoredReadScheduler.Job, job_count);
    errdefer allocator.free(jobs);
    const requests = try allocator.alloc(VectoredLoadPipeline.RequestContext, job_count);
    errdefer allocator.free(requests);
    @memset(requests, VectoredLoadPipeline.RequestContext.idle);
    const blocks = try allocator.alloc(VectoredLoadPipeline.BlockContext, job_count);
    errdefer allocator.free(blocks);
    const events = try allocator.alloc(VectoredLoadPipeline.EventContext, job_count);
    errdefer allocator.free(events);
    for (jobs, requests, 0..) |*job, *request, index| job.* = .{
        .source_slot = undefined,
        .file_offset = index,
        .len = 1,
        .transfers = &.{},
        .request = request,
        .blocks = blocks[index..][0..1],
    };
    const plan = try allocator.create(Batch.Plan);
    plan.* = .{
        .allocator = allocator,
        .jobs = jobs,
        .transfers = &.{},
        .requests = requests,
        .blocks = blocks,
        .events = events,
        .source_bytes = job_count,
        .source_runs = job_count,
    };
    return plan;
}

/// Publishes a plan of `job_count` unit jobs into an open batch.
fn publishTestPlan(scheduler: *FairVectoredReadScheduler, batch: *Batch, job_count: usize) !void {
    const plan = try testPlan(std.testing.allocator, job_count);
    scheduler.publish(std.testing.io, batch, plan, 0) catch |err| {
        plan.destroy();
        return err;
    };
}

/// `DirectLoader.submit`'s sequence for a one-file submission: publish,
/// seal, drop the sentinel.
fn publishTestBatch(scheduler: *FairVectoredReadScheduler, job_count: usize) !*Batch {
    const io = std.testing.io;
    const batch = try Batch.create(std.testing.allocator, io, .{});
    errdefer batch.destroy();
    try publishTestPlan(scheduler, batch, job_count);
    scheduler.seal(io, batch);
    batch.finishJobs(1);
    return batch;
}

/// Claims out, seals and frees an open batch that a test only inspected;
/// it must be the only queued batch.
fn discardTestBatch(scheduler: *FairVectoredReadScheduler, batch: *Batch) void {
    const io = std.testing.io;
    var claimed: usize = 0;
    while (scheduler.claim(io)) |claim| {
        std.debug.assert(claim.batch == batch);
        claimed += 1;
    }
    scheduler.seal(io, batch);
    batch.finishJobs(1 + claimed);
    std.debug.assert(batch.done.isSet());
    batch.destroy();
}
test "fair order rotates sharded devices by scheduled bytes" {
    try expectFairOrder(2, &.{
        .{ .physical_bytes = &.{ 10, 0 } },
        .{ .physical_bytes = &.{ 10, 0 } },
        .{ .physical_bytes = &.{ 0, 10 } },
        .{ .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 2, 1, 3 });
}

test "source planner coalesces exact adjacent and overlapping tensor ranges per file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "a", .name = "a0", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a1", .shape = .init(.{4}, .u8), .offset = 14 },
        .{ .file_uri = "a", .name = "a0-copy", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a-gap", .shape = .init(.{4}, .u8), .offset = 20 },
        .{ .file_uri = "b", .name = "b0", .shape = .init(.{12}, .u8), .offset = 3 },
    };
    var slots = [_]LoaderSourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var items: [sources.len]LoaderLoadItem = undefined;
    var item_ptrs: [sources.len]*LoaderLoadItem = undefined;
    for (&items, &item_ptrs, 0..) |*item, *item_ptr, i| {
        item.* = .{
            .source = &sources[i],
            .source_slot = if (i == sources.len - 1) &slots[1] else &slots[0],
            .shape = sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &outputs[i],
        };
        item_ptr.* = item;
    }
    var device_count: usize = 0;
    for (platform.replicated_sharding.devicesInCanonicalOrder()) |device| {
        device_count = @max(device_count, @as(usize, @intCast(device.id)) + 1);
    }

    var scheduler: FairVectoredReadScheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try scheduler.publishFiles(io, batch, device_count, &item_ptrs, 4, 8);

    // One plan per file. a:[10,18) merges adjacency and the duplicate,
    // a:[20,24) remains exact, and b:[3,15) is split at the request-size
    // boundary.
    const plans = batch.plans.items;
    try std.testing.expectEqual(@as(usize, 2), plans.len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 4), plans[0].transfers.len);
    try std.testing.expectEqual(@as(u64, 10), plans[0].jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), plans[0].jobs[0].len);
    try std.testing.expectEqual(@as(usize, 3), plans[0].jobs[0].transfers.len);
    try std.testing.expectEqual(@as(u64, 20), plans[0].jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].transfers.len);
    try std.testing.expectEqual(@as(u64, 3), plans[1].jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), plans[1].jobs[0].len);
    try std.testing.expectEqual(@as(usize, 4), plans[1].jobs[1].len);
    // The totals equal the former single-plan numbers for the same inputs.
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 3), batch.diagnostics.source_runs);
    try std.testing.expectEqual(@as(usize, 4), batch.diagnostics.source_jobs);
    try std.testing.expectEqual(@as(u64, 24), batch.diagnostics.source_bytes);
    try std.testing.expectEqual(@as(usize, 7), batch.diagnostics.planned_transfers);
    // One writer per transfer on one device: one event per transfer. Each
    // job's contexts are slices of its plan's arrays, blocks by 4-byte block.
    try std.testing.expectEqual(@as(usize, 7), batch.diagnostics.planned_dma_submissions);
    try std.testing.expectEqual(@as(usize, 4), plans[0].events.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].events.len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].requests.len);
    try std.testing.expectEqual(@as(usize, 3), plans[0].blocks.len);
    try std.testing.expectEqual(&plans[0].requests[1], plans[0].jobs[1].request);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs[0].blocks.len);
    try std.testing.expectEqual(plans[0].blocks[2..3], plans[0].jobs[1].blocks);
    try std.testing.expectEqual(@as(usize, 3), plans[1].blocks.len);
    try std.testing.expectEqual(plans[1].blocks[0..2], plans[1].jobs[0].blocks);
    try std.testing.expectEqual(plans[1].blocks[2..3], plans[1].jobs[1].blocks);
    for (plans) |plan| {
        for (plan.requests) |*request| try std.testing.expect(request.completed.load(.acquire));
    }
    discardTestBatch(&scheduler, batch);

    var iov_source: safetensors.Tensor = .{
        .file_uri = "iov",
        .name = "iov0",
        .shape = .init(.{@as(i64, @intCast(max_load_positional_iovecs + 1))}, .u8),
        .offset = 0,
    };
    var iov_slot: LoaderSourceSlot = .{ .uri = "iov" };
    var iov_output: Buffer = undefined;
    var iov_item: LoaderLoadItem = .{
        .source = &iov_source,
        .source_slot = &iov_slot,
        .shape = iov_source.shape,
        .sharding = platform.replicated_sharding,
        .output = &iov_output,
    };
    const iov_plan = try FairVectoredReadScheduler.preparePlan(
        allocator,
        device_count,
        &.{&iov_item},
        &.{0},
        1,
        max_load_positional_iovecs + 1,
    );
    defer iov_plan.destroy();
    try std.testing.expectEqual(@as(usize, 2), iov_plan.jobs.len);
    try std.testing.expectEqual(max_load_positional_iovecs, iov_plan.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 1), iov_plan.jobs[1].len);

    var aligned_sources = [_]safetensors.Tensor{
        .{ .file_uri = "aligned", .name = "aligned0", .shape = .init(.{7}, .u8), .offset = 0 },
        .{ .file_uri = "aligned", .name = "aligned1", .shape = .init(.{8}, .u8), .offset = 7 },
        .{ .file_uri = "aligned", .name = "aligned2", .shape = .init(.{5}, .u8), .offset = 15 },
    };
    var aligned_slot: LoaderSourceSlot = .{ .uri = "aligned" };
    var aligned_outputs: [aligned_sources.len]Buffer = undefined;
    var aligned_items: [aligned_sources.len]LoaderLoadItem = undefined;
    var aligned_item_ptrs: [aligned_sources.len]*LoaderLoadItem = undefined;
    for (&aligned_items, &aligned_item_ptrs, 0..) |*item, *item_ptr, i| {
        item.* = .{
            .source = &aligned_sources[i],
            .source_slot = &aligned_slot,
            .shape = aligned_sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &aligned_outputs[i],
        };
        item_ptr.* = item;
    }
    const aligned_plan = try FairVectoredReadScheduler.preparePlan(
        allocator,
        device_count,
        &aligned_item_ptrs,
        &.{ 0, 1, 2 },
        4,
        8,
    );
    defer aligned_plan.destroy();
    try std.testing.expectEqual(@as(usize, 3), aligned_plan.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_plan.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_plan.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 8), aligned_plan.jobs[1].len);
    try std.testing.expectEqual(@as(usize, 5), aligned_plan.jobs[2].len);
}

test "scheduler publishes a submission one file at a time and claims the files in order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    // Submitted out of file order: the sort groups "a" before "b".
    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "b", .name = "b0", .shape = .init(.{4}, .u8), .offset = 0 },
        .{ .file_uri = "a", .name = "a1", .shape = .init(.{4}, .u8), .offset = 4 },
        .{ .file_uri = "a", .name = "a0", .shape = .init(.{4}, .u8), .offset = 0 },
    };
    var slots = [_]LoaderSourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var items: [sources.len]LoaderLoadItem = undefined;
    var item_ptrs: [sources.len]*LoaderLoadItem = undefined;
    for (&items, &item_ptrs, 0..) |*item, *item_ptr, i| {
        item.* = .{
            .source = &sources[i],
            .source_slot = if (i == 0) &slots[1] else &slots[0],
            .shape = sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &outputs[i],
        };
        item_ptr.* = item;
    }
    var device_count: usize = 0;
    for (platform.replicated_sharding.devicesInCanonicalOrder()) |device| {
        device_count = @max(device_count, @as(usize, @intCast(device.id)) + 1);
    }

    var scheduler: FairVectoredReadScheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try scheduler.publishFiles(io, batch, device_count, &item_ptrs, 4, 4);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 3), batch.diagnostics.source_jobs);
    try std.testing.expectEqual(@as(usize, 3), scheduler.snapshot(io).remaining_jobs);
    try std.testing.expect(batch.diagnostics.published_at != null);
    try std.testing.expect(batch.diagnostics.sealed_at == null);

    // File a's jobs in offset order, then file b's.
    var claim = scheduler.claim(io).?;
    try std.testing.expectEqual(&slots[0], claim.job.source_slot);
    try std.testing.expectEqual(@as(u64, 0), claim.job.file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(&slots[0], claim.job.source_slot);
    try std.testing.expectEqual(@as(u64, 4), claim.job.file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(&slots[1], claim.job.source_slot);
    try std.testing.expectEqual(@as(u64, 0), claim.job.file_offset);
    // Open and exhausted: the batch keeps the head until it is sealed.
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 1), scheduler.queue.items.len);
    scheduler.seal(io, batch);
    try std.testing.expect(batch.diagnostics.sealed_at != null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.items.len);
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    batch.finishJobs(3);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}
test "fair order places a replicated job once and credits every replica" {
    // The replicated entry is skipped in device 1's queue; tie rotation gives
    // that device the next scheduling turn.
    try expectFairOrder(2, &.{
        .{ .physical_bytes = &.{ 20, 20 } },
        .{ .physical_bytes = &.{ 10, 0 } },
        .{ .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 2, 1 });
}

test "fair order compares physical bytes rather than scheduling turns" {
    // Device 0 receives a third turn because it has 8 scheduled bytes while
    // device 1 has 10; a turn-count scheduler would alternate.
    try expectFairOrder(2, &.{
        .{ .physical_bytes = &.{ 4, 0 } },
        .{ .physical_bytes = &.{ 4, 0 } },
        .{ .physical_bytes = &.{ 4, 0 } },
        .{ .physical_bytes = &.{ 0, 10 } },
        .{ .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 3, 1, 2, 4 });
}

test "fair order validates jobs and cleans up allocation failures" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidTestJob, testFairOrder(allocator, 2, &.{
        .{ .physical_bytes = &.{1} },
    }));
    // A job that no device queue lists can never be selected.
    try std.testing.expectError(error.InvalidLoaderJob, testFairOrder(allocator, 2, &.{
        .{ .physical_bytes = &.{ 0, 0 } },
    }));
    try std.testing.expectError(error.DmaDeviceMismatch, testFairOrder(allocator, 0, &.{}));
    var planning = [_]FairVectoredReadScheduler.PlanningJob{.{
        .source_slot = undefined,
        .file_offset = 0,
        .len = 1,
        .transfer_start = 0,
        .transfer_len = 0,
        .block_start = 0,
        .block_len = 0,
    }};
    const queues = [_]std.ArrayListUnmanaged(usize){ .empty, .empty };
    try std.testing.expectError(
        error.InvalidLoaderJob,
        FairVectoredReadScheduler.fairOrder(allocator, &planning, &.{1}, &queues),
    );

    const AllocationTest = struct {
        fn run(allocator_: std.mem.Allocator) !void {
            const order = try testFairOrder(allocator_, 2, &.{
                .{ .physical_bytes = &.{ 1, 1 } },
                .{ .physical_bytes = &.{ 1, 0 } },
            });
            allocator_.free(order);
        }
    };
    try std.testing.checkAllAllocationFailures(allocator, AllocationTest.run, .{});
}

test "fifo scheduler claims batches in publish order" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 2);
    const second = try publishTestBatch(&scheduler, 1);
    try std.testing.expectEqual(@as(usize, 3), scheduler.snapshot(io).remaining_jobs);

    var claim = scheduler.claim(io).?;
    try std.testing.expectEqual(first, claim.batch);
    try std.testing.expectEqual(@as(u64, 0), claim.job.file_offset);
    try std.testing.expect(first.diagnostics.first_claim_at != null);
    try std.testing.expect(second.diagnostics.first_claim_at == null);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(first, claim.batch);
    try std.testing.expectEqual(@as(u64, 1), claim.job.file_offset);
    try std.testing.expectEqual(@as(usize, 1), scheduler.snapshot(io).remaining_jobs);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(second, claim.batch);
    try std.testing.expectEqual(@as(u64, 0), claim.job.file_offset);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);

    first.finishJobs(2);
    second.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    try std.testing.expect(second.done.isSet());
    first.destroy();
    second.destroy();
}

test "fifo scheduler completes a batch while a later batch has unclaimed jobs" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 1);
    const second = try publishTestBatch(&scheduler, 2);

    try std.testing.expectEqual(first, scheduler.claim(io).?.batch);
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    try std.testing.expect(!second.done.isSet());
    try std.testing.expectEqual(@as(usize, 2), scheduler.snapshot(io).remaining_jobs);
    // The completed batch left the queue with its last claim, so it can go
    // away while the other one is still being claimed.
    first.destroy();

    try std.testing.expectEqual(second, scheduler.claim(io).?.batch);
    try std.testing.expectEqual(second, scheduler.claim(io).?.batch);
    try std.testing.expect(scheduler.claim(io) == null);
    second.finishJobs(2);
    try std.testing.expect(second.done.isSet());
    second.destroy();
}

test "fifo scheduler failure retires the unclaimed units of every queued batch" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 2);
    const second = try publishTestBatch(&scheduler, 3);
    // A batch without jobs is never queued and completes at publish.
    const empty = try publishTestBatch(&scheduler, 0);
    try std.testing.expect(empty.done.isSet());
    empty.destroy();

    // One claim in flight: its unit stays with the worker.
    try std.testing.expectEqual(first, scheduler.claim(io).?.batch);
    scheduler.fail(io);
    try std.testing.expect(!first.done.isSet());
    try std.testing.expectEqual(@as(usize, 1), first.remaining.load(.acquire));
    try std.testing.expect(second.done.isSet());
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);
    try std.testing.expect(!scheduler.waitForWork(io));
    try std.testing.expectError(error.LoaderShuttingDown, publishTestBatch(&scheduler, 1));
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    first.destroy();
    second.destroy();

    // Everything claimed before the failure: nothing to retire.
    var exhausted: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer exhausted.deinit();
    const claimed = try publishTestBatch(&exhausted, 1);
    try std.testing.expectEqual(claimed, exhausted.claim(io).?.batch);
    exhausted.fail(io);
    try std.testing.expect(!claimed.done.isSet());
    try std.testing.expectEqual(@as(usize, 1), claimed.remaining.load(.acquire));
    claimed.finishJobs(1);
    try std.testing.expect(claimed.done.isSet());
    claimed.destroy();
}

test "fifo scheduler wakes waiting workers on publish and releases them on stop" {
    const WaitResult = enum(u8) { waiting, work, stopped };
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();

    var woke: std.atomic.Value(WaitResult) = .init(.waiting);
    var group: std.Io.Group = .init;
    const Waiter = struct {
        fn run(scheduler_: *FairVectoredReadScheduler, io_: std.Io, woke_: *std.atomic.Value(WaitResult)) void {
            woke_.store(if (scheduler_.waitForWork(io_)) .work else .stopped, .release);
        }
    };
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(WaitResult.waiting, woke.load(.acquire));
    const batch = try publishTestBatch(&scheduler, 1);
    try group.await(io);
    try std.testing.expectEqual(WaitResult.work, woke.load(.acquire));
    _ = scheduler.claim(io).?;
    batch.finishJobs(1);
    batch.destroy();

    woke.store(.waiting, .release);
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(WaitResult.waiting, woke.load(.acquire));
    scheduler.stop(io);
    try group.await(io);
    try std.testing.expectEqual(WaitResult.stopped, woke.load(.acquire));
}

test "fifo scheduler concurrent claims across two batches return each job once" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batches = [_]*Batch{
        try publishTestBatch(&scheduler, 16),
        try publishTestBatch(&scheduler, 16),
    };
    var seen: std.atomic.Value(u64) = .init(0);
    var claim_count: std.atomic.Value(usize) = .init(0);
    var duplicate: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    for (0..8) |_| try group.concurrent(io, struct {
        fn run(
            scheduler_: *FairVectoredReadScheduler,
            batches_: []const *Batch,
            seen_: *std.atomic.Value(u64),
            claim_count_: *std.atomic.Value(usize),
            duplicate_: *std.atomic.Value(bool),
        ) void {
            while (scheduler_.claim(std.testing.io)) |claim| {
                const base: u64 = if (claim.batch == batches_[0]) 0 else 16;
                const mask = @as(u64, 1) << @intCast(base + claim.job.file_offset);
                if (seen_.fetchOr(mask, .acq_rel) & mask != 0) duplicate_.store(true, .release);
                _ = claim_count_.fetchAdd(1, .monotonic);
            }
        }
    }.run, .{ &scheduler, &batches, &seen, &claim_count, &duplicate });
    try group.await(io);
    try std.testing.expectEqual(std.math.maxInt(u32), @as(u32, @truncate(seen.load(.acquire))));
    try std.testing.expectEqual(@as(usize, 32), claim_count.load(.acquire));
    try std.testing.expect(!duplicate.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);
    for (batches) |batch| {
        batch.finishJobs(16);
        try std.testing.expect(batch.done.isSet());
        batch.destroy();
    }
}

test "fifo scheduler keeps an open batch at the head until its next plan is published" {
    const WaitResult = enum(u8) { waiting, work, stopped };
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 1);
    var claim = scheduler.claim(io).?;
    try std.testing.expectEqual(batch, claim.batch);
    // The published plan is exhausted but the batch is open: nothing to
    // claim, and the head is kept.
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);
    try std.testing.expect(batch.queued);
    try std.testing.expect(!batch.done.isSet());

    // A worker sleeps until the next plan is published.
    var woke: std.atomic.Value(WaitResult) = .init(.waiting);
    var group: std.Io.Group = .init;
    const Waiter = struct {
        fn run(scheduler_: *FairVectoredReadScheduler, io_: std.Io, woke_: *std.atomic.Value(WaitResult)) void {
            woke_.store(if (scheduler_.waitForWork(io_)) .work else .stopped, .release);
        }
    };
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(WaitResult.waiting, woke.load(.acquire));
    try publishTestPlan(&scheduler, batch, 2);
    try group.await(io);
    try std.testing.expectEqual(WaitResult.work, woke.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 3), batch.diagnostics.source_jobs);

    // The new plan's jobs follow within the same batch; sealed with a job
    // left, the last claim pops it.
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(batch, claim.batch);
    try std.testing.expectEqual(@as(u64, 0), claim.job.file_offset);
    scheduler.seal(io, batch);
    try std.testing.expect(batch.queued);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(@as(u64, 1), claim.job.file_offset);
    try std.testing.expect(!batch.queued);
    try std.testing.expect(scheduler.claim(io) == null);
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 3), batch.remaining.load(.acquire));
    batch.finishJobs(3);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();

    // A batch sealed while already exhausted leaves the queue at its seal.
    const exhausted = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, exhausted, 1);
    try std.testing.expectEqual(exhausted, scheduler.claim(io).?.batch);
    scheduler.seal(io, exhausted);
    try std.testing.expect(!exhausted.queued);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.items.len);
    exhausted.finishJobs(2);
    try std.testing.expect(exhausted.done.isSet());
    exhausted.destroy();
}

test "fifo scheduler failure retires every published plan of an open batch" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 2);
    try publishTestPlan(&scheduler, batch, 3);
    try std.testing.expectEqual(@as(usize, 5), scheduler.snapshot(io).remaining_jobs);
    try std.testing.expectEqual(@as(usize, 6), batch.remaining.load(.acquire));
    try std.testing.expectEqual(batch, scheduler.claim(io).?.batch);

    // The claim keeps its unit; the four unclaimed jobs of both plans are
    // retired; the sentinel is still held.
    scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expect(!batch.queued);
    try std.testing.expectEqual(@as(usize, 2), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);
    try std.testing.expect(scheduler.claim(io) == null);
    // The submission goes on: its next plan is refused, and the seal with
    // the sentinel drop leaves only the claim to complete it.
    try std.testing.expectError(error.LoaderShuttingDown, publishTestPlan(&scheduler, batch, 1));
    scheduler.seal(io, batch);
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    batch.finishJobs(1);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}

test "fair order is the identity for one device" {
    // Every job charges the one device, so `preparePlan` skips the fair
    // order and keeps the planning order for `device_count == 1`.
    try expectFairOrder(1, &.{
        .{ .physical_bytes = &.{10} },
        .{ .physical_bytes = &.{5} },
        .{ .physical_bytes = &.{1} },
        .{ .physical_bytes = &.{20} },
        .{ .physical_bytes = &.{20} },
        .{ .physical_bytes = &.{2} },
    }, &.{ 0, 1, 2, 3, 4, 5 });
}
const read_width_ladder = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

/// Source-only adaptive state. DMA width and request size never enter its
/// evidence or decisions.
/// Climb-and-hold source width policy. Every scored window is attributed by
/// the admission fence to the rung in effect when its reads were admitted,
/// so a rung change never drains the read gate. The controller climbs the
/// ladder one rung per window while each rung beats the best rate seen by
/// 3%, tolerating one rung that does not, then holds at the lowest rung
/// within 3% of the best. One downward probe below the start rung bounds the
/// number of windows a load spends away from its final width; the probe is
/// adopted only when it beats the best rate, never on retention. The climb
/// stops at the widest rung the pre-grown pinned capacity already covers, so
/// no scored window pays for a slab the load has yet to map.
const SourceReadWidthController = struct {
    const State = enum { climbing, holding };

    /// A rung must beat the best rate by this factor to keep the climb going.
    const improvement_ratio = 1.03;
    /// Consecutive rungs that may fail `improvement_ratio` before the climb
    /// stops. A single window resolves a 3% difference only when the rungs
    /// differ by more than the window's noise, which a DMA-bound source does
    /// not: on a GB300 loading DeepSeek-V4 the rungs from 12 to 48 measure
    /// 44.8 to 48.8 GiB/s with a per-window spread of the same size, so one
    /// unlucky window at 16 ended the climb at 12 (3.3 s against 3.05 s at
    /// 32). Climbing past one such rung reaches the plateau: the rung above
    /// a stall is compared with the same best rate, so a real decline still
    /// stops the climb one window later, and the hold rule then picks the
    /// lowest rung within the band whatever the climb passed through.
    const stall_tolerance = 2;
    /// The hold rung is the lowest rung retaining this fraction of the best.
    const hold_ratio = 0.97;

    const Evidence = struct {
        completed_requests: usize,
        elapsed_ns: u64,
        bytes: u64,
        exercised_width: usize,

        fn scoreable(self: Evidence, expected_width: usize) bool {
            return self.exercised_width >= expected_width and
                self.completed_requests >= @max(@as(usize, 8), expected_width) and
                self.elapsed_ns >= 100 * std.time.ns_per_ms and self.bytes != 0;
        }

        fn bytesPerSecond(self: Evidence) f64 {
            if (self.elapsed_ns == 0) return 0;
            return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
                @as(f64, @floatFromInt(self.elapsed_ns));
        }
    };

    /// A width and the generation that measures it. Every decision opens a
    /// new generation, so the runtime re-fences its window on each one.
    const Decision = struct {
        width: usize,
        generation: u64,
    };

    fixed_width: ?usize = null,
    index: usize,
    /// Where measurement began: the configured initial rung, or the blind
    /// bootstrap's last rung for a high-latency source.
    start_index: usize,
    /// Pinned-feasibility clip, lowered by backpressure.
    max_index: usize,
    best_index: usize,
    /// Per rung: the mean of its scored windows.
    rates: [read_width_ladder.len]?f64 = @splat(null),
    samples: [read_width_ladder.len]u8 = @splat(0),
    state: State,
    /// Consecutive climb samples that failed `improvement_ratio`.
    stalls: u8 = 0,
    probed_down: bool = false,
    generation: u64 = 0,
    last_backoff_generation: u64 = std.math.maxInt(u64),

    /// `growth_free_width` is the widest read width whose lifecycle credits
    /// the pool already holds mapped (`retained - dma_stage`): above it a
    /// scored window maps a new pinned slab, which on a GB300 cost a whole
    /// window (20.8 GiB/s at 48 against 48.8 sustained at 32). A fixed width
    /// the caller asked for is not clipped by it, only by feasibility.
    fn init(
        configured: Parallelism,
        pinned_feasible_width: usize,
        growth_free_width: usize,
    ) SourceReadWidthController {
        const configured_max = @min(configured.maximum(), pinned_feasible_width, @max(@as(usize, 1), growth_free_width));
        const max_index = widthIndexAtMost(configured_max);
        if (!configured.isAdaptive()) {
            const fixed = @min(configured.initial(), pinned_feasible_width);
            const fixed_index = widthIndexAtMost(fixed);
            return .{
                .fixed_width = @max(@as(usize, 1), fixed),
                .index = fixed_index,
                .start_index = fixed_index,
                .max_index = fixed_index,
                .best_index = fixed_index,
                .state = .holding,
            };
        }
        const initial_index = @min(widthIndexAtMost(configured.initial()), max_index);
        return .{
            .index = initial_index,
            .start_index = initial_index,
            .max_index = max_index,
            .best_index = initial_index,
            .state = .climbing,
        };
    }

    fn widthIndexAtMost(maximum: usize) usize {
        var result: usize = 0;
        for (read_width_ladder, 0..) |candidate_width, index| {
            if (candidate_width > maximum) break;
            result = index;
        }
        return result;
    }

    fn width(self: *const SourceReadWidthController) usize {
        return self.fixed_width orelse read_width_ladder[self.index];
    }

    fn isAdaptive(self: *const SourceReadWidthController) bool {
        return self.fixed_width == null;
    }

    fn currentDecision(self: *const SourceReadWidthController) Decision {
        return .{ .width = self.width(), .generation = self.generation };
    }

    /// Opens a new generation at the current width: the first measured
    /// window after a blind bootstrap, whose admissions overlap generations.
    fn newGeneration(self: *SourceReadWidthController) Decision {
        self.generation +|= 1;
        return self.currentDecision();
    }

    /// Pre-response growth for a high-latency source: 24 then 32, before any
    /// window is scored. Measurement then starts from the reached rung.
    fn blindGrow(self: *SourceReadWidthController) ?Decision {
        if (!self.isAdaptive() or self.state != .climbing or
            self.rates[self.best_index] != null or self.index >= self.max_index)
            return null;
        const ceiling: usize = if (self.width() < 24) 24 else if (self.width() < 32) 32 else return null;
        const target = @min(widthIndexAtMost(ceiling), self.max_index);
        if (target <= self.index) return null;
        self.start_index = target;
        self.best_index = target;
        return self.moveTo(target);
    }

    fn observe(self: *SourceReadWidthController, evidence: Evidence) Decision {
        if (!self.isAdaptive() or self.state == .holding) return self.currentDecision();
        std.debug.assert(evidence.scoreable(self.width()));
        // While climbing the scored rung is the best rung or the one above
        // it; a re-measure or the downward probe scores a rung below it.
        const climb_sample = self.index >= self.best_index;
        const best_rate = self.rates[self.best_index];
        const rate = self.addSample(self.index, evidence.bytesPerSecond());
        const improved = if (best_rate) |best| rate > improvement_ratio * best else true;
        if (improved) self.best_index = self.index;
        if (climb_sample) {
            self.stalls = if (improved) 0 else self.stalls +| 1;
            if (self.index < self.max_index and (improved or self.stalls < stall_tolerance))
                return self.moveTo(self.index + 1);
            if (improved) return self.hold(self.index);
        }

        // A rung below the start rung is the downward probe. Its one window
        // reads high when it inherits the wider rung's queued transfers: on
        // a GB300 a probe at 8 measured 41 GiB/s right after 16 against 36.9
        // sustained, and retention then held the whole load at the narrowest
        // rung it ever tried (3.70 s against 3.05 s at 32). Only an
        // improvement adopts it.
        if (self.index < self.start_index and !improved) return self.hold(self.start_index);

        const hold_index = self.holdIndex();
        if (hold_index == self.start_index and self.start_index > 0 and !self.probed_down) {
            self.probed_down = true;
            return self.moveTo(self.start_index - 1);
        }
        return self.hold(hold_index);
    }

    fn addSample(self: *SourceReadWidthController, index: usize, rate: f64) f64 {
        const count: f64 = @floatFromInt(self.samples[index]);
        const mean = if (self.rates[index]) |previous|
            (previous * count + rate) / (count + 1)
        else
            rate;
        self.rates[index] = mean;
        self.samples[index] +|= 1;
        return mean;
    }

    /// The lowest measured rung at or below the best one that retains
    /// `hold_ratio` of the best rate.
    fn holdIndex(self: *const SourceReadWidthController) usize {
        const best_rate = self.rates[self.best_index].?;
        for (self.rates[0 .. self.best_index + 1], 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate >= hold_ratio * best_rate) return index;
        }
        return self.best_index;
    }

    fn moveTo(self: *SourceReadWidthController, index: usize) Decision {
        self.index = index;
        return self.newGeneration();
    }

    fn hold(self: *SourceReadWidthController, index: usize) Decision {
        self.index = index;
        self.state = .holding;
        return self.newGeneration();
    }

    /// Source backpressure: one rung down, clipped there, and holding. At
    /// most once per generation of fresh admissions: a further sample in the
    /// generation a backoff opened is delayed feedback from the old width
    /// unless a read admitted under the new generation has begun
    /// (`fresh_admissions`), so it cannot ratchet through several rungs.
    fn backoff(self: *SourceReadWidthController, fresh_admissions: bool) ?Decision {
        if (!self.isAdaptive()) return null;
        if (self.last_backoff_generation == self.generation and !fresh_admissions) return null;
        self.index -|= 1;
        self.max_index = self.index;
        self.state = .holding;
        return self.openBackoffGeneration();
    }

    /// Transient backpressure (retries, connection failures, 5xx without a
    /// throttle): one rung down, ceiling and state unchanged. A climbing
    /// controller restarts its climb at the lower rung: it becomes the best
    /// rung and its mean is forgotten, so the next window there is a fresh
    /// climb sample that can lead back above the step. A holding one keeps
    /// holding. Same once-per-generation rule as `backoff`.
    fn stepDownTransient(self: *SourceReadWidthController, fresh_admissions: bool) ?Decision {
        if (!self.isAdaptive()) return null;
        if (self.last_backoff_generation == self.generation and !fresh_admissions) return null;
        self.index -|= 1;
        if (self.state == .climbing) {
            self.best_index = self.index;
            self.rates[self.index] = null;
            self.samples[self.index] = 0;
            self.stalls = 0;
        }
        return self.openBackoffGeneration();
    }

    fn openBackoffGeneration(self: *SourceReadWidthController) Decision {
        const decision = self.newGeneration();
        self.last_backoff_generation = self.generation;
        return decision;
    }
};

fn sourceReadTestEvidence(
    controller: *const SourceReadWidthController,
    rate: f64,
) SourceReadWidthController.Evidence {
    return .{
        .completed_requests = @max(@as(usize, 8), controller.width()),
        .elapsed_ns = std.time.ns_per_s,
        .bytes = @intFromFloat(rate * 1024 * 1024),
        .exercised_width = controller.width(),
    };
}

const SourceReadCurvePoint = struct { width: usize, rate: f64 };

/// Replays a curve of per-width rates: each window scores the controller's
/// current width and returns the number of windows until it holds.
fn replaySourceReadCurve(
    controller: *SourceReadWidthController,
    curve: []const SourceReadCurvePoint,
) !usize {
    var windows: usize = 0;
    while (controller.state == .climbing) : (windows += 1) {
        const rate = for (curve) |point| {
            if (point.width == controller.width()) break point.rate;
        } else return error.UnmeasuredWidth;
        const generation = controller.generation;
        _ = controller.observe(sourceReadTestEvidence(controller, rate));
        try std.testing.expect(controller.generation == generation + 1);
    }
    return windows;
}

test "source read controller bounds blind growth at 32" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expectEqual(@as(usize, 24), controller.blindGrow().?.width);
    try std.testing.expectEqual(@as(usize, 32), controller.blindGrow().?.width);
    try std.testing.expect(controller.blindGrow() == null);
    try std.testing.expectEqual(@as(usize, 32), read_width_ladder[controller.start_index]);
    // Measurement starts at the reached rung; a scored window ends growth.
    _ = controller.observe(sourceReadTestEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 48), controller.width());
    try std.testing.expect(controller.blindGrow() == null);
}

test "source read controller clips infeasible adaptive and fixed widths" {
    var adaptive = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        10,
        10,
    );
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());
    try std.testing.expect(adaptive.blindGrow() == null);
    // At the clip the first window holds the only rung it can use.
    _ = adaptive.observe(sourceReadTestEvidence(&adaptive, 100));
    try std.testing.expectEqual(SourceReadWidthController.State.holding, adaptive.state);
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());

    const fixed = SourceReadWidthController.init(.{ .fixed = 20 }, 7, 7);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expectEqual(SourceReadWidthController.State.holding, fixed.state);

    const configured_initial = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 48, .maximum = 128 } },
        128,
        128,
    );
    try std.testing.expectEqual(@as(usize, 48), configured_initial.width());
}

test "source read evidence requires enough concurrency and duration" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    var short = sourceReadTestEvidence(&controller, 100);
    short.elapsed_ns = 99 * std.time.ns_per_ms;
    try std.testing.expect(!short.scoreable(controller.width()));
    var unexercised = sourceReadTestEvidence(&controller, 100);
    unexercised.exercised_width -= 1;
    try std.testing.expect(!unexercised.scoreable(controller.width()));
    var few = sourceReadTestEvidence(&controller, 100);
    few.completed_requests = 11;
    try std.testing.expect(!few.scoreable(controller.width()));
    var empty = sourceReadTestEvidence(&controller, 100);
    empty.bytes = 0;
    try std.testing.expect(!empty.scoreable(controller.width()));
    try std.testing.expectEqual(
        @as(usize, 16),
        controller.observe(sourceReadTestEvidence(&controller, 100)).width,
    );
}

test "source read controller replays the B70 32 MiB curve and holds 12" {
    // Recorded on one B70 at 32 MiB requests (CTX "Source request size is
    // backend-dependent"), GiB/s. Eight was not screened there; the probe
    // below the start rung gets a value below the best.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 8, .rate = 19.90 },
        .{ .width = 12, .rate = 21.33 },
        .{ .width = 16, .rate = 20.69 },
        .{ .width = 24, .rate = 18.90 },
        .{ .width = 32, .rate = 17.33 },
    });
    // 12, 16 (0.970 of 12: the first stall), 24 (0.886: the second stops the
    // climb), the downward probe of 8. A declining curve costs one window
    // more than it did on a single-strike rule and reaches the same rung.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expect(controller.probed_down);
    try std.testing.expectEqual(@as(u8, 1), controller.samples[SourceReadWidthController.widthIndexAtMost(24)]);
    // Holding: further evidence and blind growth change nothing.
    const held = controller.observe(sourceReadTestEvidence(&controller, 30));
    try std.testing.expectEqual(@as(usize, 12), held.width);
    try std.testing.expectEqual(controller.generation, held.generation);
    try std.testing.expect(controller.blindGrow() == null);
}

test "source read controller climbs past one rung inside the noise band" {
    // gb300-2 loading DeepSeek-V4 at 16 MiB requests: the rungs from 12 to 48
    // sustain 44.8 to 48.8 GiB/s over a whole load while one 120 ms window at
    // a single rung spreads 37.8 to 53.1, so a rung is regularly measured
    // below its neighbour by more than the 3% band. Run 6 of the baseline set
    // read 40.50 at 12 and 41.61 at 16 (1.027 of it) and ended its climb
    // there.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 40.50));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    // The stalled rung carries the climb on instead of ending it.
    _ = controller.observe(sourceReadTestEvidence(&controller, 41.61));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    try std.testing.expectEqual(@as(u8, 1), controller.stalls);
    // The rung above it is the sustained plateau, and clears the stall.
    _ = controller.observe(sourceReadTestEvidence(&controller, 48.20));
    try std.testing.expectEqual(@as(usize, 32), controller.width());
    try std.testing.expectEqual(@as(u8, 0), controller.stalls);
    // Two rungs in a row inside the band stop the climb; the hold rule then
    // picks the lowest rung within 3% of the best, whatever the climb passed
    // through.
    _ = controller.observe(sourceReadTestEvidence(&controller, 48.80));
    try std.testing.expectEqual(@as(usize, 48), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 47.30));
    try std.testing.expectEqual(SourceReadWidthController.State.holding, controller.state);
    try std.testing.expectEqual(@as(usize, 24), controller.width());
}

test "source read controller keeps the start rung when the probe only matches it" {
    // Same host, run 6: the climb stopped at 12 and the probe at 8 read
    // 41.07 GiB/s, above 12's 40.50 window but well under the 36.9 that 8
    // sustains -- a rung stepped down to inherits the wider rung's queued
    // transfers. Retention used to adopt it and hold the load at 8 (3.70 s
    // against 3.05 s at 32).
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 8, .rate = 41.07 },
        .{ .width = 12, .rate = 40.50 },
        .{ .width = 16, .rate = 39.00 },
        .{ .width = 24, .rate = 39.50 },
    });
    // 12, 16, 24, then the probe of 8: 1.014 of the best is not the 3% an
    // adoption needs, so the load holds the rung it started from.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expect(controller.probed_down);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
}

test "source read controller holds at the lowest rung within 3% on a flat curve" {
    // Real AWS shape: 16 MiB requests plateau near 950 MiB/s from 16 up.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 12, .rate = 900 },
        .{ .width = 16, .rate = 940 },
        .{ .width = 24, .rate = 948 },
        .{ .width = 32, .rate = 950 },
        .{ .width = 48, .rate = 950 },
    });
    // 12, 16 (better by 4.4%), 24 and 32 (not better by 3%): hold at 16, the
    // lowest rung within 3% of it; 12 at 0.957 is below the band.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    try std.testing.expectEqual(@as(usize, 16), read_width_ladder[controller.best_index]);
    try std.testing.expect(!controller.probed_down);
}

test "source read controller probes below the start rung once" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 8, .rate = 105 },
        .{ .width = 12, .rate = 100 },
        .{ .width = 16, .rate = 100 },
        .{ .width = 24, .rate = 100 },
    });
    // 12, 16 and 24 (flat), 8 (better by 5%): hold 8 without climbing further
    // down.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    try std.testing.expectEqual(@as(usize, 8), read_width_ladder[controller.best_index]);

    var from_one = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 1, .maximum = 128 } },
        128,
        128,
    );
    // Nothing below the lowest rung to probe.
    _ = try replaySourceReadCurve(&from_one, &.{
        .{ .width = 1, .rate = 100 },
        .{ .width = 2, .rate = 100 },
        .{ .width = 4, .rate = 100 },
    });
    try std.testing.expectEqual(@as(usize, 1), from_one.width());
    try std.testing.expect(!from_one.probed_down);
}

test "source read controller backs off once per generation of fresh admissions" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const first = controller.backoff(false).?;
    try std.testing.expectEqual(@as(usize, 12), first.width);
    try std.testing.expectEqual(controller.generation, first.generation);
    try std.testing.expectEqual(SourceReadWidthController.State.holding, controller.state);
    try std.testing.expectEqual(@as(usize, 12), read_width_ladder[controller.max_index]);
    // Delayed feedback from the old width in the same generation is ignored.
    try std.testing.expect(controller.backoff(false) == null);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    // Feedback after a fresh admission under the new generation counts.
    try std.testing.expectEqual(@as(usize, 8), controller.backoff(true).?.width);
    try std.testing.expectEqual(@as(usize, 8), read_width_ladder[controller.max_index]);
    // Holding: evidence no longer moves the width.
    _ = controller.observe(sourceReadTestEvidence(&controller, 1000));
    try std.testing.expectEqual(@as(usize, 8), controller.width());

    var floor = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 1, .maximum = 64 } },
        64,
        64,
    );
    try std.testing.expectEqual(@as(usize, 1), floor.backoff(false).?.width);
}

test "source read controller steps down on transient backpressure and climbs again" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const step = controller.stepDownTransient(false).?;
    try std.testing.expectEqual(@as(usize, 12), step.width);
    try std.testing.expectEqual(controller.generation, step.generation);
    // Still climbing, ceiling untouched, the climb restarts at 12.
    try std.testing.expectEqual(SourceReadWidthController.State.climbing, controller.state);
    try std.testing.expectEqual(@as(usize, 64), read_width_ladder[controller.max_index]);
    try std.testing.expectEqual(@as(usize, 12), read_width_ladder[controller.best_index]);
    try std.testing.expect(controller.rates[controller.index] == null);
    // Once per generation of fresh admissions, shared with the throttle rule.
    try std.testing.expect(controller.stepDownTransient(false) == null);
    try std.testing.expect(controller.backoff(false) == null);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    // A fresh window at 12 is a climb sample: back to 16, then above the step.
    _ = controller.observe(sourceReadTestEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 110));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    try std.testing.expectEqual(SourceReadWidthController.State.climbing, controller.state);
    // A fresh admission under the step's generation admits another step.
    try std.testing.expectEqual(@as(usize, 16), controller.stepDownTransient(false).?.width);
    try std.testing.expectEqual(@as(usize, 12), controller.stepDownTransient(true).?.width);
    try std.testing.expectEqual(@as(usize, 64), read_width_ladder[controller.max_index]);

    // Holding: one rung down, still holding, evidence changes nothing.
    var holding = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    // 12 -> 16 and 24 (not better) -> the downward probe of 8 (10% below the
    // best) -> hold 12.
    for ([_]f64{ 100, 100, 90, 90 }) |rate| _ = holding.observe(sourceReadTestEvidence(&holding, rate));
    try std.testing.expectEqual(@as(usize, 12), holding.width());
    try std.testing.expectEqual(SourceReadWidthController.State.holding, holding.state);
    try std.testing.expectEqual(@as(usize, 8), holding.stepDownTransient(false).?.width);
    try std.testing.expectEqual(SourceReadWidthController.State.holding, holding.state);
    try std.testing.expectEqual(@as(usize, 64), read_width_ladder[holding.max_index]);
    _ = holding.observe(sourceReadTestEvidence(&holding, 1000));
    try std.testing.expectEqual(@as(usize, 8), holding.width());
}

test "source read controller stops climbing at the growth-free width" {
    // gb300-2: 41 retained credits and a DMA stage of 8 leave 33 requests the
    // pool already holds mapped, so the climb stops at 32. Above it the
    // lifecycle limit maps a new pinned slab inside a scored window: one such
    // window measured 20.8 GiB/s at 48 against 48.8 sustained at 32.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        33,
    );
    try std.testing.expectEqual(@as(usize, 32), read_width_ladder[controller.max_index]);
    // Each rung beats the last: the climb runs up to the ceiling and holds.
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 12, .rate = 100 },
        .{ .width = 16, .rate = 110 },
        .{ .width = 24, .rate = 120 },
        .{ .width = 32, .rate = 130 },
    });
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 32), controller.width());
}

test "source read controller keeps a fixed width" {
    // A fixed width the caller asked for is not clipped by the growth-free
    // width, only by feasibility.
    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64, 4);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.backoff(true) == null);
    try std.testing.expect(fixed.stepDownTransient(true) == null);
    try std.testing.expect(fixed.blindGrow() == null);
    const observed = fixed.observe(sourceReadTestEvidence(&fixed, 100));
    try std.testing.expectEqual(@as(usize, 7), observed.width);
    try std.testing.expectEqual(@as(u64, 0), fixed.generation);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
}

const ReadStatsCursor = struct {
    provider: VFS.ReadStatsProvider,
    previous: VFS.ReadStats,

    /// The two classes of source backpressure, exclusive: a throttle in the
    /// same interval outranks a transient failure.
    const Backpressure = struct {
        /// Throttles and timeouts: the source rejects this width.
        throttle: bool = false,
        /// Retries, connection failures and other 5xx: an unhealthy request,
        /// not evidence about the width.
        transient: bool = false,

        fn any(self: Backpressure) bool {
            return self.throttle or self.transient;
        }
    };

    fn takeBackpressure(self: *ReadStatsCursor) Backpressure {
        const current = self.provider.snapshot();
        const delta = current.sub(self.previous);
        self.previous = current;
        const throttle = delta.throttles != 0 or delta.timeouts != 0;
        const transient = delta.retries != 0 or delta.transient_retries != 0 or delta.server_failures != 0;
        return .{ .throttle = throttle, .transient = transient and !throttle };
    }
};

test "one load-profile feedback cursor reports only new backpressure" {
    const FakeProvider = struct {
        stats: VFS.ReadStats = .{},

        fn snapshot(userdata: *anyopaque) VFS.ReadStats {
            const self: *@This() = @ptrCast(@alignCast(userdata));
            return self.stats;
        }
    };

    var fake: FakeProvider = .{};
    const provider: VFS.ReadStatsProvider = .{
        .userdata = &fake,
        .snapshotFn = FakeProvider.snapshot,
    };
    var cursor: ReadStatsCursor = .{
        .provider = provider,
        .previous = provider.snapshot(),
    };

    const Backpressure = ReadStatsCursor.Backpressure;
    try std.testing.expectEqual(Backpressure{}, cursor.takeBackpressure());
    fake.stats.retries = 2;
    fake.stats.server_failures = 1;
    try std.testing.expectEqual(Backpressure{ .transient = true }, cursor.takeBackpressure());
    try std.testing.expectEqual(Backpressure{}, cursor.takeBackpressure());
    fake.stats.transient_retries = 1;
    try std.testing.expectEqual(Backpressure{ .transient = true }, cursor.takeBackpressure());
    // A throttle outranks the retries it caused in the same interval.
    fake.stats.retries = 3;
    fake.stats.throttles = 1;
    try std.testing.expectEqual(Backpressure{ .throttle = true }, cursor.takeBackpressure());
    fake.stats.timeouts = 1;
    try std.testing.expectEqual(Backpressure{ .throttle = true }, cursor.takeBackpressure());
    try std.testing.expectEqual(Backpressure{}, cursor.takeBackpressure());
}

test "source warm-up window is discarded once when the DMA stage held requests as long as the reads" {
    var metrics: VectoredLoadMetrics = .{};
    var runtime: SourceReadRuntime = undefined;
    runtime.metrics = &metrics;
    runtime.window_dma_base_ns = 0;
    runtime.window_read_base_ns = 0;
    runtime.warmup_pending = true;
    // Reads dominated the window (a network load): the first window is
    // scored.
    metrics.read_ns.store(1000, .monotonic);
    metrics.dma_stage_ns.store(10, .monotonic);
    try std.testing.expect(!runtime.discardWarmup());
    try std.testing.expect(!runtime.warmup_pending);

    // Requests sat in the DMA stage longer than they read: discarded.
    runtime.warmup_pending = true;
    metrics.dma_stage_ns.store(3000, .monotonic);
    try std.testing.expect(runtime.discardWarmup());
    // Only once: the re-measured window is scored even under pressure.
    try std.testing.expect(!runtime.discardWarmup());

    // Residency is counted from the generation's start, not the load's.
    runtime.warmup_pending = true;
    runtime.window_dma_base_ns = 3000;
    runtime.window_read_base_ns = 1000;
    metrics.read_ns.store(2000, .monotonic);
    metrics.dma_stage_ns.store(3005, .monotonic);
    try std.testing.expect(!runtime.discardWarmup());
}

test "source measurement rejects another controller generation" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    var runtime: SourceReadRuntime = undefined;
    runtime.controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    runtime.metrics = &metrics;
    runtime.clock = .{};
    metrics.prepareProbe(io, runtime.controller.generation + 1, 1);
    try std.testing.expect(runtime.currentEvidence(io, 1_000) == null);
}

/// The window clock of the width controller counts busy time only. A
/// control tick that found nothing unclaimed, nothing pending and no read
/// permit held charges the interval since the previous tick to idle, so
/// many short submissions jointly complete one window and the controller
/// never learns that batches exist.
const BusyWindowClock = struct {
    idle_ns: u64 = 0,
    last_tick_ns: u64 = 0,

    fn tick(self: *BusyWindowClock, now_ns: u64, idle: bool) void {
        if (idle) self.idle_ns +|= now_ns -| self.last_tick_ns;
        self.last_tick_ns = now_ns;
    }

    /// A new window starts: idle time accrued before it does not count.
    fn reset(self: *BusyWindowClock) void {
        self.idle_ns = 0;
    }

    fn busyNs(self: *const BusyWindowClock, from_ns: u64, now_ns: u64) u64 {
        return (now_ns -| from_ns) -| self.idle_ns;
    }
};

test "busy window clock subtracts idle intervals from a window" {
    var clock: BusyWindowClock = .{};
    clock.tick(0, false);
    clock.reset();
    // Busy 0-40 ms, idle 40-240 ms, busy 240-280 ms, sampled every 20 ms.
    var now_ns: u64 = 0;
    while (now_ns < 280 * std.time.ns_per_ms) {
        now_ns += 20 * std.time.ns_per_ms;
        const idle = now_ns > 40 * std.time.ns_per_ms and now_ns <= 240 * std.time.ns_per_ms;
        clock.tick(now_ns, idle);
    }
    try std.testing.expectEqual(200 * std.time.ns_per_ms, clock.idle_ns);
    try std.testing.expectEqual(80 * std.time.ns_per_ms, clock.busyNs(0, now_ns));
    // The next window starts clean.
    clock.reset();
    clock.tick(now_ns + 25 * std.time.ns_per_ms, false);
    try std.testing.expectEqual(25 * std.time.ns_per_ms, clock.busyNs(now_ns, now_ns + 25 * std.time.ns_per_ms));
}

/// Worker tasks, spawned as the width first needs them (`width + 1`, never
/// more than the configured maximum; 128 persistent workers cost about 7%
/// on one MI300X while the controller held width 12) and never retired.
/// A worker whose index is beyond what the current width needs parks
/// between jobs instead of competing for lifecycle credits: after a rung
/// steps down, the workers spawned for the wider rung would otherwise queue
/// at the credit gate for the rest of the load and inflate the credit wait
/// the summary reports without moving a byte.
const WorkerPool = struct {
    loader: *DirectLoader,
    maximum: usize,
    mutex: std.Io.Mutex = .init,
    /// Parked workers; woken when `wanted` grows or the pool stops.
    condition: std.Io.Condition = .init,
    spawned: usize = 0,
    /// Workers the current width needs (`RequestGateLimits.workers`).
    wanted: usize = 0,
    stopping: bool = false,

    fn ensure(self: *WorkerPool, io: std.Io, wanted: usize) void {
        const target = @min(wanted, self.maximum);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // A controller tick racing `stopWorkers` must not spawn into a
        // group being awaited.
        if (self.stopping) return;
        self.wanted = target;
        while (self.spawned < target) : (self.spawned += 1) {
            self.loader.worker_group.concurrent(io, DirectLoader.workerMain, .{ self.loader, self.spawned }) catch |err| {
                load_log.err("cannot spawn source worker {d}: {}", .{ self.spawned + 1, err });
                break;
            };
        }
        self.condition.broadcast(io);
    }

    /// Parks worker `index` while the width does not need it. False once
    /// the pool stops.
    fn admit(self: *WorkerPool, io: std.Io, index: usize) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and index >= self.wanted) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }

    fn stop(self: *WorkerPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
    }
};

const SourceReadRuntime = struct {
    /// `measuring` while the controller climbs: the current generation's
    /// window is open. `blind` during the pre-response bootstrap of a
    /// high-latency source. `inactive` while holding.
    const Measurement = enum { inactive, measuring, blind };

    controller: SourceReadWidthController,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    metrics: *VectoredLoadMetrics,
    next_read_admission: *std.atomic.Value(u64),
    scheduler: *FairVectoredReadScheduler,
    pinned_feasible_width: usize,
    read_stats: ?ReadStatsCursor,
    source_bootstrap_enabled: bool,
    /// Lifecycle credits: see `RequestGateLimits`.
    retained_credits: usize = 1,
    dma_stage_requests: usize = 1,
    /// The load's first scoreable window is a warm-up when the load is
    /// DMA-bound, which the window itself shows: the requests completing in
    /// it spent at least as long in the DMA stage (enqueue to last
    /// callback) as reading. Such a window opened on an empty DMA stage and
    /// measured the burst that filled it (47 GiB/s on a GB300 whose steady
    /// rate was 42), which no later rung can beat, so it is discarded and
    /// the start rung measured again. A read-bound load keeps its first
    /// window: on a Hugging Face load a request spends a millisecond in the
    /// DMA stage per 1.3 s read, whatever a burst of completions does to
    /// the credit gate's occupancy. Credit waiting is not the signal: at a
    /// narrow start rung on a GB300 the workers wait 2 ms per 3 ms read
    /// while the stage holds each request for 10.
    warmup_pending: bool = true,
    /// `dma_stage_ns` and `read_ns` when the current generation opened.
    window_dma_base_ns: u64 = 0,
    window_read_base_ns: u64 = 0,
    /// Grown with the lifecycle limit; null in unit tests without workers.
    workers: ?*WorkerPool = null,
    source_response_observed: bool = false,
    measurement: Measurement = .inactive,
    last_blind_growth_ns: u64 = 0,
    clock: BusyWindowClock = .{},
    reported_width: usize = 1,
    /// Control ticks that found the read gate closed while jobs were still
    /// unclaimed. Nothing closes the gate any more; the counter stays as
    /// the invariant's witness in the loader summary.
    gate_closed_ticks: u64 = 0,
    control: std.Io.Event = .unset,
    done: std.Io.Event = .unset,

    fn takeRemoteBackpressure(self: *SourceReadRuntime) ReadStatsCursor.Backpressure {
        const cursor = if (self.read_stats) |*value| value else return .{};
        return cursor.takeBackpressure();
    }

    /// Applies a decision: both gates at its width and a window for its
    /// generation fenced at the next admission. The read gate is never
    /// closed: reads admitted under the previous generation are excluded by
    /// the fence and return at their own pace.
    fn applyDecision(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
    ) void {
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width, self.retained_credits, self.dma_stage_requests);
        std.debug.assert(limits.read > 0);
        self.reported_width = decision.width;
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        if (self.workers) |pool| pool.ensure(io, limits.workers());
        // Advance the diagnostic baseline at the generation boundary.
        _ = self.takeRemoteBackpressure();
        self.metrics.prepareProbe(io, decision.generation, self.next_read_admission.load(.acquire));
        self.clock.reset();
        self.window_dma_base_ns = self.metrics.dma_stage_ns.load(.monotonic);
        self.window_read_base_ns = self.metrics.read_ns.load(.monotonic);
        self.measurement = switch (self.controller.state) {
            .climbing => .measuring,
            .holding => .inactive,
        };
    }

    /// Born busy: the gates already carry the controller's width; the first
    /// window is fenced before any worker can admit a read.
    fn start(self: *SourceReadRuntime, io: std.Io) void {
        self.applyDecision(io, self.controller.currentDecision());
    }

    fn applyBlindGrowth(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
    ) void {
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width, self.retained_credits, self.dma_stage_requests);
        self.reported_width = decision.width;
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        if (self.workers) |pool| pool.ensure(io, limits.workers());
        self.metrics.clearProbe(io);
        self.metrics.config_epoch.store(decision.generation, .release);
        self.measurement = .blind;
    }

    fn evidenceFrom(
        self: *const SourceReadRuntime,
        probe: VectoredLoadMetrics.Snapshot,
        now_ns: u64,
    ) ?SourceReadWidthController.Evidence {
        if (probe.probe_epoch != self.controller.generation) return null;
        // No window before the generation's first completion.
        if (probe.probe_window_start_ns == 0) return null;
        const evidence: SourceReadWidthController.Evidence = .{
            .completed_requests = @intCast(probe.probe_read_operations),
            .elapsed_ns = self.clock.busyNs(probe.probe_window_start_ns, now_ns),
            .bytes = probe.probe_read_bytes,
            .exercised_width = probe.probe_peak_reads,
        };
        return if (evidence.scoreable(self.controller.width())) evidence else null;
    }

    fn currentEvidence(
        self: *SourceReadRuntime,
        io: std.Io,
        now_ns: u64,
    ) ?SourceReadWidthController.Evidence {
        return self.evidenceFrom(self.metrics.snapshot(io), now_ns);
    }

    /// Once per load: the first scoreable window is dropped when the
    /// requests completing inside it spent at least as long in the DMA
    /// stage as reading.
    fn discardWarmup(self: *SourceReadRuntime) bool {
        const pending = self.warmup_pending;
        self.warmup_pending = false;
        if (!pending) return false;
        const staged = self.metrics.dma_stage_ns.load(.monotonic) -| self.window_dma_base_ns;
        const read = self.metrics.read_ns.load(.monotonic) -| self.window_read_base_ns;
        return staged != 0 and staged >= read;
    }

    fn finalize(self: *SourceReadRuntime, io: std.Io) void {
        std.debug.assert(self.read_gate.inUse(io) == 0);
        _ = self.takeRemoteBackpressure();
        self.metrics.clearProbe(io);
    }

    fn run(self: *SourceReadRuntime, io: std.Io) std.Io.Cancelable!void {
        self.clock.tick(awakeNs(io), false);
        while (true) {
            self.control.waitTimeout(io, .{ .duration = .{
                .raw = .fromMilliseconds(if (self.source_response_observed) 25 else 10),
                .clock = .awake,
            } }) catch |err| switch (err) {
                error.Timeout => {},
                error.Canceled => return error.Canceled,
            };
            if (self.control.isSet()) self.control.reset();
            if (self.done.isSet()) {
                self.finalize(io);
                break;
            }
            const now_ns = awakeNs(io);

            const backpressure = self.takeRemoteBackpressure();
            if (backpressure.any()) {
                // A read admitted under the current generation has begun
                // once the window fenced at its start saw a read.
                const fresh_admissions = self.metrics.snapshot(io).probe_peak_reads != 0;
                const decision = if (backpressure.throttle)
                    self.controller.backoff(fresh_admissions)
                else
                    self.controller.stepDownTransient(fresh_admissions);
                if (decision) |value| {
                    load_log.debug("source width {s}: generation={d}, width={d}, fresh_admissions={}", .{
                        if (backpressure.throttle) "backoff" else "transient step-down",
                        value.generation,
                        value.width,
                        fresh_admissions,
                    });
                    self.applyDecision(io, value);
                }
                self.clock.tick(now_ns, false);
                continue;
            }
            if (self.metrics.read_bytes.load(.acquire) != 0) self.source_response_observed = true;
            const scheduler_snapshot = self.scheduler.snapshot(io);
            if (scheduler_snapshot.remaining_jobs != 0 and self.read_gate.currentLimit(io) == 0)
                self.gate_closed_ticks += 1;

            if (!self.source_response_observed) {
                self.clock.tick(now_ns, false);
                if (now_ns -| self.last_blind_growth_ns >= 10 * std.time.ns_per_ms and
                    shouldBootstrapSource(
                        self.source_bootstrap_enabled,
                        false,
                        self.metrics.read_bytes.load(.acquire),
                        self.request_gate.inUse(io),
                        self.controller.width(),
                        scheduler_snapshot.remaining_jobs,
                    ))
                {
                    self.last_blind_growth_ns = now_ns;
                    if (self.controller.blindGrow()) |decision| {
                        self.applyBlindGrowth(io, decision);
                    }
                }
                continue;
            }

            switch (self.measurement) {
                // Blind admissions overlap generations. The first response
                // opens the first measured window at the reached width; the
                // fence excludes what was admitted before it.
                .blind => {
                    self.applyDecision(io, self.controller.newGeneration());
                    self.clock.tick(now_ns, false);
                    continue;
                },
                .inactive => {
                    self.clock.tick(now_ns, false);
                    continue;
                },
                .measuring => {},
            }

            const probe = self.metrics.snapshot(io);
            const idle = scheduler_snapshot.remaining_jobs == 0 and
                self.metrics.pending_source_jobs.load(.acquire) == 0 and
                self.read_gate.inUse(io) == 0;
            // Idle before the window's first admission is not its time.
            self.clock.tick(now_ns, idle and probe.probe_window_start_ns != 0);
            if (idle) continue;
            const evidence = self.evidenceFrom(probe, now_ns) orelse continue;
            if (self.discardWarmup()) {
                load_log.debug("source width warm-up window discarded: generation={d}, width={d}, rate={Bi:.2}/s", .{
                    probe.probe_epoch,
                    self.controller.width(),
                    @as(u64, @intFromFloat(evidence.bytesPerSecond())),
                });
                self.applyDecision(io, self.controller.newGeneration());
                continue;
            }
            const scored_index = self.controller.index;
            const decision = self.controller.observe(evidence);
            load_log.debug("source width window: generation={d}, width={d}, rate={Bi:.2}/s, busy_ms={d:.1}, completed={d}, exercised={d}, samples={d}, next_width={d}, state={s}", .{
                probe.probe_epoch,
                read_width_ladder[scored_index],
                @as(u64, @intFromFloat(evidence.bytesPerSecond())),
                @as(f64, @floatFromInt(evidence.elapsed_ns)) / std.time.ns_per_ms,
                evidence.completed_requests,
                evidence.exercised_width,
                self.controller.samples[scored_index],
                decision.width,
                @tagName(self.controller.state),
            });
            self.applyDecision(io, decision);
        }
    }
};
/// Blind growth is warranted while a high-latency source has not answered
/// and the read gate is the limiter: at least `read_limit` workers hold a
/// lifecycle credit (taken before the claim, returned after the last DMA
/// callback), no read has returned, so the admitted reads are all still
/// pending and the other credit holders wait for a read permit.
fn shouldBootstrapSource(
    enabled: bool,
    response_observed: bool,
    read_bytes: u64,
    lifecycle_in_use: usize,
    read_limit: usize,
    remaining_jobs: usize,
) bool {
    return enabled and !response_observed and read_bytes == 0 and
        lifecycle_in_use >= read_limit and remaining_jobs != 0;
}

fn secondsBetween(from: std.Io.Timestamp, to: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.durationTo(to).nanoseconds)) / std.time.ns_per_s;
}

/// Mean milliseconds per unit of a summed duration (zero units: the total).
fn millisecondsPer(total_ns: u64, units: u64) f64 {
    return @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(@max(units, 1)));
}

fn awakeNs(io: std.Io) u64 {
    return @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
}

/// The direct DMA backend. Submissions and awaits come from one task at a
/// time; the workers, pump and controller run concurrently with them.
pub const DirectLoader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    load_profile: VFS.LoadProfile,
    progress: ?*std.Progress.Node,
    dma_resources: *dma.BenchmarkResult,
    owns_dma_resources: bool,
    pool: mem.DmaBlockPool,
    scheduler: FairVectoredReadScheduler,
    metrics: VectoredLoadMetrics = .{},
    read_gate: AdaptiveRequestGate,
    request_gate: AdaptiveRequestGate,
    pipeline: VectoredLoadPipeline,
    controller_runtime: SourceReadRuntime,
    worker_pool: WorkerPool,
    worker_group: std.Io.Group = .init,
    controller_group: std.Io.Group = .init,
    source_slots: std.StringHashMapUnmanaged(*LoaderSourceSlot) = .empty,
    bytes_loaded: std.atomic.Value(usize) = .init(0),
    created_at: std.Io.Timestamp,
    /// Submissions so far; the next batch's sequence number.
    batch_count: usize = 0,
    source_request_size: usize,
    maximum_blocks_per_job: usize,
    effective_pinned_feasible_width: usize,
    workers_started: bool = false,
    controller_started: bool = false,
    cleaned: bool = false,

    fn providedDmaBenchmark(opts: LoaderOptions) ?*dma.BenchmarkResult {
        const optional = opts.dma orelse return null;
        return if (optional.*) |*result| result else null;
    }

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: LoaderOptions,
    ) !*DirectLoader {
        if (platform.devices.len == 0 or platform.devices.len > 64)
            return error.DmaDeviceMismatch;
        var owned_resources: ?*dma.BenchmarkResult = null;
        const resources = providedDmaBenchmark(opts) orelse resources: {
            const owned = try allocator.create(dma.BenchmarkResult);
            errdefer allocator.destroy(owned);
            owned.* = try dma.benchmark(allocator, io, platform, .{});
            owned_resources = owned;
            break :resources owned;
        };
        errdefer if (owned_resources) |owned| {
            owned.deinit();
            allocator.destroy(owned);
        };
        try resources.acquire();
        errdefer resources.release();
        const calibration = resources.calibration;
        if (resources.workspace.device_pool_indices.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        if (calibration.block_size > max_load_read_request_size or
            resources.maxMappedBytes() < max_load_read_request_size)
            return error.InvalidDmaLoadConfig;

        const request_size = try effectiveSourceRequestSize(
            opts.load_profile.read_chunk_size,
            calibration.block_size,
        );
        const maximum_blocks_per_job = try maximumCoalescedJobBlocks(
            request_size,
            calibration.block_size,
        );
        const node_reserves = try allocator.alloc(usize, resources.workspace.pools.len);
        defer allocator.free(node_reserves);
        @memset(node_reserves, 0);
        for (platform.devices, 0..) |_, device_index| {
            const node_index = resources.workspace.device_pool_indices[device_index];
            node_reserves[node_index] = try std.math.add(
                usize,
                node_reserves[node_index],
                calibration.max_in_flight_per_device,
            );
        }
        // Calibration already mapped the working set for a 16 MiB request;
        // this grows the remainder for larger request sizes (HF profiles)
        // so no pinned slab grows inside a scored window. The arenas stay
        // with the benchmark result for later loaders.
        const pregrowth_started: std.Io.Timestamp = .now(io, .awake);
        const retained_before = resources.retainedMappedBytes();
        try resources.ensureSourceWorkingSet(
            maximum_blocks_per_job,
            dma.BenchmarkResult.preallocated_source_width,
            node_reserves,
        );
        const pregrown_bytes = resources.retainedMappedBytes() - retained_before;
        const pregrowth_ns: u64 = @intCast(@max(pregrowth_started.untilNow(io, .awake).nanoseconds, 0));
        // The per-node reserves were materialized by calibration (the DMA
        // stage's blocks); the pool keeps them as the growth floor of each
        // node for devices that join a later submission.
        var pool = try mem.DmaBlockPool.initFromProvider(
            allocator,
            resources.workspace.blockPoolArenaProvider(),
            calibration.block_size,
            resources.maxMappedBytes(),
            node_reserves,
        );
        var pool_moved = false;
        errdefer if (!pool_moved) pool.deinit();
        const aggregate_width = try pool.aggregatePotentialRequestWidth(maximum_blocks_per_job);
        const strict_width = try pool.minimumStrictAffinityRequestWidth(maximum_blocks_per_job);
        const strict_affinity = resources.hasStrictAffinity();
        const feasible_width = if (strict_affinity)
            @min(aggregate_width, strict_width)
        else
            aggregate_width;
        if (feasible_width == 0) return error.DmaMappedBudgetExceeded;

        const source_parallelism = opts.read_parallelism;
        const retained_credits = try pool.retainedRequestWidth(maximum_blocks_per_job, strict_affinity);
        const dma_stage_requests = dmaStageRequests(
            calibration.max_in_flight_per_device,
            platform.devices.len,
            calibration.block_size,
            request_size,
        );
        const controller = SourceReadWidthController.init(
            source_parallelism,
            feasible_width,
            retained_credits -| dma_stage_requests,
        );
        const limits: RequestGateLimits = .init(controller.width(), feasible_width, retained_credits, dma_stage_requests);
        const read_stats: ?ReadStatsCursor = if (opts.load_profile.stats) |provider| cursor: {
            const initial = provider.snapshot();
            break :cursor .{ .provider = provider, .previous = initial };
        } else null;
        var scheduler: FairVectoredReadScheduler = .init(allocator);
        var scheduler_moved = false;
        errdefer if (!scheduler_moved) scheduler.deinit();

        const self = try allocator.create(DirectLoader);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .load_profile = opts.load_profile,
            .progress = opts.progress,
            .dma_resources = resources,
            .owns_dma_resources = owned_resources != null,
            .pool = pool,
            .scheduler = scheduler,
            .read_gate = .init(limits.read),
            .request_gate = .init(limits.lifecycle),
            .pipeline = undefined,
            .controller_runtime = undefined,
            .worker_pool = undefined,
            .created_at = .now(io, .awake),
            .source_request_size = request_size,
            .maximum_blocks_per_job = maximum_blocks_per_job,
            .effective_pinned_feasible_width = feasible_width,
        };
        // Ownership moved into the stable heap object.
        pool_moved = true;
        scheduler_moved = true;
        errdefer {
            self.scheduler.deinit();
            self.pool.deinit();
        }

        self.pipeline = try VectoredLoadPipeline.init(
            allocator,
            io,
            platform,
            &self.pool,
            &self.read_gate,
            &self.request_gate,
            calibration.block_size,
            resources.workspace.device_pool_indices,
            strict_affinity,
            &self.metrics,
            &self.scheduler,
            calibration.max_in_flight_per_device * calibration.block_size,
        );
        errdefer self.pipeline.deinit();

        self.worker_pool = .{
            .loader = self,
            .maximum = RequestGateLimits.init(source_parallelism.maximum(), feasible_width, retained_credits, dma_stage_requests).workers(),
        };
        self.controller_runtime = .{
            .controller = controller,
            .read_gate = &self.read_gate,
            .request_gate = &self.request_gate,
            .metrics = &self.metrics,
            .next_read_admission = &self.pipeline.next_read_admission,
            .workers = &self.worker_pool,
            .scheduler = &self.scheduler,
            .pinned_feasible_width = feasible_width,
            .read_stats = read_stats,
            .source_bootstrap_enabled = opts.load_profile.high_latency,
            .retained_credits = retained_credits,
            .dma_stage_requests = dma_stage_requests,
            .reported_width = controller.width(),
        };
        // Born busy: both gates are open at the controller's width, the
        // first window is fenced before any worker can admit a read, and the
        // initial workers are spawned by the decision that opened the gates.
        errdefer self.stopWorkers();
        self.workers_started = true;
        self.controller_runtime.start(io);
        try self.startController();
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, dma_block_size={Bi:.2}, dma_budget_per_device={Bi:.2}, lifecycle_credits={d}, workers={d}, max_workers={d}, feasible_width={d}, width_ceiling={d}, retained={Bi:.2}, pregrown={Bi:.2}, pregrowth_ms={d:.3}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            request_size,
            calibration.block_size,
            self.pipeline.dma_budget_bytes,
            limits.lifecycle,
            self.worker_pool.spawned,
            self.worker_pool.maximum,
            feasible_width,
            read_width_ladder[self.controller_runtime.controller.max_index],
            self.pool.mappedBytes(),
            pregrown_bytes,
            @as(f64, @floatFromInt(pregrowth_ns)) / std.time.ns_per_ms,
        });
        return self;
    }

    fn startController(self: *DirectLoader) !void {
        try self.controller_group.concurrent(
            self.io,
            SourceReadRuntime.run,
            .{ &self.controller_runtime, self.io },
        );
        self.controller_started = true;
    }

    fn workerMain(self: *DirectLoader, index: usize) void {
        var scratch = VectoredReadRequest.Scratch.init(
            self.allocator,
            &self.pool,
            self.maximum_blocks_per_job,
            self.platform.devices.len,
        ) catch |err| {
            self.pipeline.recordError(err);
            return;
        };
        defer scratch.deinit();
        while (true) {
            if (!self.worker_pool.admit(self.io, index)) return;
            if (!self.scheduler.waitForWork(self.io)) return;
            if (self.pipeline.failed()) return;
            const credit_wait_started = awakeNs(self.io);
            if (!self.request_gate.acquire(self.io)) return;
            const credit_wait_ns = awakeNs(self.io) -| credit_wait_started;
            const claim = self.scheduler.claim(self.io) orelse {
                self.request_gate.release(self.io);
                continue;
            };
            _ = self.pipeline.metrics.lifecycle_wait_ns.fetchAdd(credit_wait_ns, .monotonic);
            self.pipeline.reserveSourceJob();
            const request = self.pipeline.registerRequest(claim);
            // The request's scheduling sentinel keeps the batch, and with it
            // `job.transfers`, alive until `runCoalesced` returns.
            VectoredReadRequest.runCoalesced(
                request,
                claim.job.source_slot,
                &self.pipeline,
                claim.job.file_offset,
                claim.job.len,
                claim.job.transfers,
                self,
                &scratch,
            );
        }
    }

    fn stopWorkers(self: *DirectLoader) void {
        self.scheduler.stop(self.io);
        self.worker_pool.stop(self.io);
        self.read_gate.close(self.io);
        self.request_gate.close(self.io);
        if (self.controller_started) {
            self.controller_runtime.done.set(self.io);
            self.controller_runtime.control.set(self.io);
        }
        if (self.workers_started) self.worker_group.await(self.io) catch {};
        if (self.controller_started) self.controller_group.await(self.io) catch {};
        self.workers_started = false;
        self.controller_started = false;
    }

    pub fn checkOpen(self: *DirectLoader) !void {
        if (self.cleaned) return error.LoaderShuttingDown;
        if (self.pipeline.errorValue()) |err| return err;
    }

    fn sourceSlot(self: *DirectLoader, uri: []const u8) !*LoaderSourceSlot {
        if (self.source_slots.get(uri)) |slot| return slot;
        const slot = try self.allocator.create(LoaderSourceSlot);
        errdefer self.allocator.destroy(slot);
        slot.* = .{ .uri = uri };
        try self.source_slots.putNoClobber(self.allocator, uri, slot);
        return slot;
    }

    fn createItem(
        self: *DirectLoader,
        source: *safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
    ) !*LoaderLoadItem {
        const item = try self.allocator.create(LoaderLoadItem);
        errdefer self.allocator.destroy(item);
        item.* = .{
            .source = source,
            .source_slot = try self.sourceSlot(source.file_uri),
            .shape = shape,
            .sharding = sharding.resolve(self.platform),
            .output = output,
        };
        return item;
    }

    /// Creates the batch's items; on failure nothing stays allocated.
    fn createItems(
        self: *DirectLoader,
        specs: []const LoadSpec,
        logical_bytes: *usize,
    ) ![]*LoaderLoadItem {
        const items = try self.allocator.alloc(*LoaderLoadItem, specs.len);
        errdefer self.allocator.free(items);
        var initialized: usize = 0;
        errdefer for (items[0..initialized]) |item| item.deinit(self.allocator);
        for (specs, items) |spec, *item| {
            // An empty source has no transfer, so its output would never be
            // written; the front ends reject it too.
            if (spec.source.byteSize() == 0) return error.EmptyTensor;
            item.* = try self.createItem(spec.source, spec.shape, spec.sharding, spec.output);
            initialized += 1;
            logical_bytes.* = try std.math.add(usize, logical_bytes.*, spec.source.shape.byteSize());
        }
        return items;
    }

    /// Plans `specs` one source file at a time, publishes each file's plan
    /// as soon as it exists behind every earlier submission, then seals the
    /// batch: work on the first file starts while the rest is planned. The
    /// batch owns its items until `awaitBatch` retires it. Nothing is
    /// published when a failure precedes the first plan; a later planning
    /// failure fails the loader (a partial submission can never complete),
    /// the batch is awaited here and the caller sees only the error.
    pub fn submit(self: *DirectLoader, specs: []const LoadSpec) !*Batch {
        try self.checkOpen();
        const batch = try Batch.create(self.allocator, self.io, .{
            .sequence = self.batch_count,
            .source_items = specs.len,
            .source_stats = if (self.load_profile.stats) |provider| provider.snapshot() else null,
        });
        errdefer if (batch.diagnostics.plans == 0) batch.destroy();
        batch.items = try self.createItems(specs, &batch.diagnostics.logical_bytes);
        self.scheduler.publishFiles(
            self.io,
            batch,
            self.platform.devices.len,
            batch.items,
            self.dma_resources.calibration.block_size,
            self.source_request_size,
        ) catch |err| {
            if (batch.diagnostics.plans == 0) return err;
            return self.failPublished(batch, err);
        };
        self.scheduler.seal(self.io, batch);
        self.batch_count += 1;
        // Every plan is visible: drop the publish sentinel. A batch without
        // jobs completes right here.
        batch.finishJobs(1);
        return batch;
    }

    /// Planning failed after part of the batch was published: the pipeline
    /// fails with that error, the batch is sealed and awaited here, and the
    /// loader's sticky error goes to the caller instead of a batch.
    fn failPublished(self: *DirectLoader, batch: *Batch, err: anyerror) anyerror {
        self.pipeline.recordError(err);
        self.scheduler.seal(self.io, batch);
        self.batch_count += 1;
        batch.finishJobs(1);
        self.awaitBatch(batch) catch |sticky| return sticky;
        return err;
    }
    /// Waits for the batch's last completion unit, retires it and returns
    /// the loader's sticky error if the pipeline failed. Targets the failure
    /// left open are marked so their buffers never report ready; a target
    /// PJRT already closed (its last call went out, accepted or not) is left
    /// to PJRT, whose shared transfer manager has dropped the definition
    /// event and aborts on a further call. The same manager drops the event
    /// when a transfer fails asynchronously, which the loader cannot see
    /// (the done event carries no error), so a buffer that failed that way
    /// and is then marked here aborts too: two failures in one load. The
    /// outputs of a failed submission are undefined either way. A batch that
    /// completed without an error must have closed every target (the pump
    /// flagged the submission that completed its bytes); one that did not
    /// would leave a buffer that never becomes ready, so it fails the
    /// loader instead.
    pub fn awaitBatch(self: *DirectLoader, batch: *Batch) !void {
        batch.done.waitUncancelable(self.io);
        const done_at: std.Io.Timestamp = .now(self.io, .awake);
        // Every request of this batch has completed: no worker or callback
        // touches its managers or contexts any more.
        var load_error = self.pipeline.errorValue();
        if (load_error == null and !batch.fullySubmitted()) {
            self.pipeline.recordError(error.IncompleteTransfer);
            load_error = self.pipeline.errorValue();
        }
        if (load_error != null) {
            for (batch.items) |item| {
                const state = item.state.readyValue() orelse continue;
                for (state.targets) |*target| {
                    if (target.canFail()) {
                        target.manager.setBufferErrorUnknown(
                            self.platform.pjrt_api,
                            0,
                            "live loader failed",
                        ) catch {};
                    }
                }
            }
        }
        self.pipeline.retireBatch(batch);
        self.logBatch(batch, done_at, load_error == null);
        batch.destroy();
        if (load_error) |err| return err;
    }

    fn logBatch(self: *DirectLoader, batch: *const Batch, done_at: std.Io.Timestamp, successful: bool) void {
        const diagnostics = &batch.diagnostics;
        var source_requests: u64 = 0;
        var source_bytes: u64 = 0;
        var source_retries: u64 = 0;
        var source_throttles: u64 = 0;
        if (self.load_profile.stats) |provider| {
            if (diagnostics.source_stats) |previous| {
                const delta = provider.snapshot().sub(previous);
                source_requests = delta.physical_requests;
                source_bytes = delta.physical_bytes;
                source_retries = delta.retries;
                source_throttles = delta.throttles;
            }
        }
        const published_at = diagnostics.published_at orelse self.created_at;
        const sealed_at = diagnostics.sealed_at orelse published_at;
        const first_claim_at = diagnostics.first_claim_at orelse published_at;
        const first_read_ns = diagnostics.first_read_ns.load(.acquire);
        const first_read_at: std.Io.Timestamp = if (first_read_ns == 0)
            first_claim_at
        else
            std.Io.Timestamp.fromNanoseconds(@intCast(first_read_ns));
        var longest_planning_ns: u64 = 0;
        for (batch.plans.items) |plan| longest_planning_ns = @max(longest_planning_ns, plan.planning_ns);
        const average_read_size = if (diagnostics.source_jobs == 0)
            0
        else
            diagnostics.source_bytes / diagnostics.source_jobs;
        const coalescing_ratio = if (diagnostics.source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(diagnostics.source_items)) /
                @as(f64, @floatFromInt(diagnostics.source_jobs));
        load_log.debug("batch completed: batch={d}, successful={}, logical_bytes={Bi:.2}, planned_source_bytes={Bi:.2}, published=+{d:.3}s, sealed=+{d:.3}s, first_claim=+{d:.3}s, first_read=+{d:.3}s, done=+{d:.3}s, elapsed={d:.3}s, plans={d}, planning_elapsed={d:.3}s, longest_planning={d:.3}s, planned_source_jobs={d}, source_runs={d}, source_items={d}, planned_transfers={d}, planned_dma_submissions={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}, selected_source_width={d}, request_size={Bi:.2}, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}", .{
            diagnostics.sequence,
            successful,
            diagnostics.logical_bytes,
            diagnostics.source_bytes,
            secondsBetween(self.created_at, published_at),
            secondsBetween(self.created_at, sealed_at),
            secondsBetween(self.created_at, first_claim_at),
            secondsBetween(self.created_at, first_read_at),
            secondsBetween(self.created_at, done_at),
            secondsBetween(published_at, done_at),
            diagnostics.plans,
            @as(f64, @floatFromInt(diagnostics.planning_ns)) / std.time.ns_per_s,
            @as(f64, @floatFromInt(longest_planning_ns)) / std.time.ns_per_s,
            diagnostics.source_jobs,
            diagnostics.source_runs,
            diagnostics.source_items,
            diagnostics.planned_transfers,
            diagnostics.planned_dma_submissions,
            coalescing_ratio,
            average_read_size,
            self.controller_runtime.reported_width,
            self.source_request_size,
            source_requests,
            source_bytes,
            source_retries,
            source_throttles,
        });
    }
    fn logSummary(self: *DirectLoader) void {
        const reads = self.metrics.read_operations.load(.acquire);
        load_log.debug("loader summary: batches={d}, successful={}, bytes_loaded={Bi:.2}, elapsed={d:.3}s, reads={d}, physical_source_calls={d}, tensor_transfer_pieces={d}, dma_submissions={d}, selected_source_width={d}, gate_closed_ticks={d}, request_size={Bi:.2}, pinned_high_water={Bi:.2}, pinned_mapped={Bi:.2}, credit_wait_ms_per_read={d:.3}, block_wait_ms_per_read={d:.3}, read_ms_per_read={d:.3}, dma_stage_ms_per_read={d:.3}, tensor_init_ms_per_read={d:.3}", .{
            self.batch_count,
            !self.pipeline.failed(),
            self.bytesLoaded(),
            secondsBetween(self.created_at, .now(self.io, .awake)),
            reads,
            self.metrics.source_calls.load(.acquire),
            self.metrics.transfer_pieces.load(.acquire),
            self.metrics.dma_submissions.load(.acquire),
            self.controller_runtime.reported_width,
            self.controller_runtime.gate_closed_ticks,
            self.source_request_size,
            self.pool.highWaterBytes(),
            self.pool.mappedBytes(),
            millisecondsPer(self.metrics.lifecycle_wait_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.block_wait_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.read_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.dma_stage_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.tensor_init_ns.load(.acquire), reads),
        });
    }

    /// Counts a submission's logical bytes once its await succeeded.
    pub fn commitBytes(self: *DirectLoader, logical_bytes: usize) void {
        _ = self.bytes_loaded.fetchAdd(logical_bytes, .monotonic);
    }

    pub fn bytesLoaded(self: *const DirectLoader) usize {
        return self.bytes_loaded.load(.acquire);
    }

    /// The front end awaited every batch before this; nothing is queued or
    /// in flight, so stopping the workers is a plain shutdown.
    pub fn destroy(self: *DirectLoader) void {
        if (!self.cleaned) {
            self.stopWorkers();
            self.logSummary();
            var slots = self.source_slots.valueIterator();
            while (slots.next()) |slot| {
                slot.*.deinit(self.io);
                self.allocator.destroy(slot.*);
            }
            self.source_slots.deinit(self.allocator);
            self.pipeline.deinit();
            self.scheduler.deinit();
            self.pool.deinit();
            self.dma_resources.release();
            if (self.owns_dma_resources) {
                self.dma_resources.deinit();
                self.allocator.destroy(self.dma_resources);
            }
            self.cleaned = true;
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }
};

test "loader DMA admission rotates and respects per-device budgets" {
    const all_ready: u64 = 0b1111;
    var active = [_]usize{ 0, 0, 0, 0 };
    var pieces = [_]usize{ 0, 0, 0, 0 };
    var next_device: usize = 0;

    for ([_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 }) |expected| {
        const selected = selectLoaderDmaDevice(
            &active,
            &pieces,
            8,
            all_ready,
            next_device,
        ).?;
        try std.testing.expectEqual(expected, selected);
        next_device = (selected + 1) % active.len;
    }

    // Bytes at the budget close a device; so does the piece cap.
    active = .{ 8, 0, 8, 0 };
    try std.testing.expectEqual(
        @as(?usize, 1),
        selectLoaderDmaDevice(&active, &pieces, 8, all_ready, 0),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        selectLoaderDmaDevice(&active, &pieces, 8, 0b0101, 0),
    );
    pieces = .{ 0, max_dma_pieces_per_device, 0, 0 };
    try std.testing.expectEqual(
        @as(?usize, 3),
        selectLoaderDmaDevice(&active, &pieces, 8, all_ready, 0),
    );
}

test "source bootstrap requires a high-latency source with no observed response" {
    try std.testing.expect(shouldBootstrapSource(true, false, 0, 12, 12, 1));
    try std.testing.expect(!shouldBootstrapSource(false, false, 0, 12, 12, 1));
    try std.testing.expect(!shouldBootstrapSource(true, true, 0, 12, 12, 1));
    try std.testing.expect(!shouldBootstrapSource(true, false, 1, 12, 12, 1));
    try std.testing.expect(!shouldBootstrapSource(true, false, 0, 12, 12, 0));
}

test "source request size combines the VFS floor with DMA granularity" {
    try std.testing.expectEqual(
        @as(usize, 8 * 1024 * 1024),
        try effectiveSourceRequestSize(8 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try effectiveSourceRequestSize(8 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try effectiveSourceRequestSize(16 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        max_load_read_request_size,
        try effectiveSourceRequestSize(32 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectError(error.InvalidLoadProfile, effectiveSourceRequestSize(0, 8 * 1024 * 1024));
    try std.testing.expectError(
        error.InvalidLoadProfile,
        effectiveSourceRequestSize(max_load_read_request_size + 1, 8 * 1024 * 1024),
    );
}

test "coalesced job block bound is independent of device count" {
    try std.testing.expectEqual(
        @as(usize, 2),
        try maximumCoalescedJobBlocks(32 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 3),
        try maximumCoalescedJobBlocks(17 * 1024 * 1024, 8 * 1024 * 1024),
    );
}

test "probe source capacity counts active reads and keeps their peak" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 7, 10);
    for (0..8) |index| metrics.beginRead(io, 7, 10 + @as(u64, @intCast(index)));

    const active = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 8), active.probe_peak_reads);
    try std.testing.expectEqual(@as(usize, 8), active.probe_active_reads);

    for (0..4) |index| metrics.endRead(io, 7, 10 + @as(u64, @intCast(index)));
    const draining = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 8), draining.probe_peak_reads);
    try std.testing.expectEqual(@as(usize, 4), draining.probe_active_reads);
    for (4..8) |index| metrics.endRead(io, 7, 10 + @as(u64, @intCast(index)));
    metrics.clearProbe(io);
}

test "source probe excludes pre-boundary admissions" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.beginRead(io, 6, 40);
    metrics.prepareProbe(io, 7, 41);
    metrics.beginRead(io, 7, 40);
    metrics.recordProbeRead(io, 7, 40, max_load_read_request_size);
    metrics.beginRead(io, 7, 41);
    // The first in-generation completion opens the window and is not counted.
    metrics.recordProbeRead(io, 7, 41, max_load_read_request_size);
    metrics.beginRead(io, 7, 42);
    metrics.recordProbeRead(io, 7, 42, max_load_read_request_size);
    const admitted = metrics.snapshot(io);
    try std.testing.expect(admitted.probe_window_start_ns != 0);
    try std.testing.expectEqual(@as(usize, 2), admitted.probe_active_reads);
    try std.testing.expectEqual(@as(u64, 1), admitted.probe_read_operations);
    try std.testing.expectEqual(@as(u64, max_load_read_request_size), admitted.probe_read_bytes);
    metrics.endRead(io, 6, 40);
    metrics.endRead(io, 7, 40);
    metrics.endRead(io, 7, 41);
    const draining = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 1), draining.probe_active_reads);
    metrics.endRead(io, 7, 42);
    const drained = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 0), drained.probe_active_reads);
    metrics.clearProbe(io);
}

test "partial source jobs contribute adaptive evidence" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 3, 1);
    // A full read opens the window; the partial tail read that follows
    // contributes its actual byte count.
    metrics.beginRead(io, 3, 1);
    metrics.recordProbeRead(io, 3, 1, max_load_read_request_size);
    metrics.endRead(io, 3, 1);
    metrics.beginRead(io, 3, 2);
    metrics.recordProbeRead(io, 3, 2, 256 * 1024);
    metrics.endRead(io, 3, 2);
    const snapshot = metrics.snapshot(io);
    try std.testing.expectEqual(@as(u64, 1), snapshot.probe_read_operations);
    try std.testing.expectEqual(@as(u64, 256 * 1024), snapshot.probe_read_bytes);
}

test "request lifecycle gate holds the DMA stage beyond the read width" {
    const normal: RequestGateLimits = .init(12, 64, 41, 8);
    try std.testing.expectEqual(@as(usize, 12), normal.read);
    try std.testing.expectEqual(@as(usize, 41), normal.lifecycle);
    try std.testing.expectEqual(@as(usize, 13), normal.workers());

    // Above the retained capacity the stage keeps its calibrated depth.
    const wide: RequestGateLimits = .init(48, 128, 41, 8);
    try std.testing.expectEqual(@as(usize, 48), wide.read);
    try std.testing.expectEqual(@as(usize, 56), wide.lifecycle);
    try std.testing.expectEqual(@as(usize, 49), wide.workers());
    // The pinned ceiling clips everything.
    const clipped: RequestGateLimits = .init(32, 32, 41, 8);
    try std.testing.expectEqual(@as(usize, 32), clipped.read);
    try std.testing.expectEqual(@as(usize, 32), clipped.lifecycle);
    try std.testing.expectEqual(@as(usize, 32), clipped.workers());
}

test "request lifecycle gate waits for every active request" {
    const io = std.testing.io;
    var gate: AdaptiveRequestGate = .init(2);
    try std.testing.expect(gate.acquire(io));
    try std.testing.expect(gate.acquire(io));

    var drained: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(gate_: *AdaptiveRequestGate, io_: std.Io, drained_: *std.Io.Event) void {
            gate_.waitEmpty(io_);
            drained_.set(io_);
        }
    }.run, .{ &gate, io, &drained });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!drained.isSet());

    gate.release(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!drained.isSet());
    gate.release(io);
    try group.await(io);
    try std.testing.expect(drained.isSet());
}

fn buildMesh2x2(
    allocator: std.mem.Allocator,
    target: platform_mod.Target,
    devices: []const platform_mod.Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 4) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[0]),
            .device(devices[1]),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[2]),
            .device(devices[3]),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

test "adaptive request gate reductions drain without cancelling active requests" {
    const io = std.testing.io;
    var gate: AdaptiveRequestGate = .init(2);
    try std.testing.expect(gate.acquire(io));
    try std.testing.expect(gate.acquire(io));

    gate.setLimit(io, 1);
    var admitted: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(gate_: *AdaptiveRequestGate, io_: std.Io, admitted_: *std.Io.Event) void {
            if (!gate_.acquire(io_)) return;
            admitted_.set(io_);
            gate_.release(io_);
        }
    }.run, .{ &gate, io, &admitted });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!admitted.isSet());

    gate.release(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!admitted.isSet());
    gate.release(io);
    try group.await(io);
    try std.testing.expect(admitted.isSet());
    try std.testing.expectEqual(@as(usize, 0), gate.inUse(io));
}

test "source read runtime never closes the read gate across decisions" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    var read_gate: AdaptiveRequestGate = .init(12);
    var request_gate: AdaptiveRequestGate = .init(13);
    var next_admission: std.atomic.Value(u64) = .init(41);
    var runtime: SourceReadRuntime = .{
        .controller = SourceReadWidthController.init(
            .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
            64,
            64,
        ),
        .read_gate = &read_gate,
        .request_gate = &request_gate,
        .metrics = &metrics,
        .next_read_admission = &next_admission,
        .scheduler = undefined,
        .pinned_feasible_width = 64,
        .read_stats = null,
        .source_bootstrap_enabled = false,
    };

    // Born busy: the first window is fenced at the next admission and
    // measures at the initial width with both gates open.
    runtime.start(io);
    try std.testing.expect(runtime.measurement == .measuring);
    try std.testing.expectEqual(@as(usize, 12), read_gate.currentLimit(io));
    try std.testing.expectEqual(@as(usize, 13), request_gate.currentLimit(io));
    try std.testing.expectEqual(runtime.controller.generation, metrics.snapshot(io).probe_epoch);
    metrics.beginRead(io, runtime.controller.generation, 40);
    try std.testing.expectEqual(@as(usize, 0), metrics.snapshot(io).probe_active_reads);
    metrics.beginRead(io, runtime.controller.generation, 41);
    try std.testing.expectEqual(@as(usize, 1), metrics.snapshot(io).probe_active_reads);
    metrics.endRead(io, runtime.controller.generation, 40);
    metrics.endRead(io, runtime.controller.generation, 41);

    // A scored window moves one rung up without touching the gate limit
    // below the new width; a hold at another width does the same.
    var expected_generation = runtime.controller.generation;
    for ([_]f64{ 100, 100, 90, 90 }) |rate| {
        next_admission.store(next_admission.load(.acquire) + 5, .release);
        const decision = runtime.controller.observe(sourceReadTestEvidence(&runtime.controller, rate));
        runtime.applyDecision(io, decision);
        expected_generation += 1;
        try std.testing.expectEqual(expected_generation, decision.generation);
        try std.testing.expect(read_gate.currentLimit(io) > 0);
        try std.testing.expect(request_gate.currentLimit(io) > read_gate.currentLimit(io));
        try std.testing.expectEqual(decision.width, read_gate.currentLimit(io));
        try std.testing.expectEqual(decision.width, runtime.reported_width);
        try std.testing.expectEqual(decision.generation, metrics.snapshot(io).probe_epoch);
        try std.testing.expectEqual(decision.generation, metrics.config_epoch.load(.acquire));
        try std.testing.expectEqual(next_admission.load(.acquire), metrics.probe_admission_start);
        try std.testing.expect((runtime.measurement == .measuring) == (runtime.controller.state == .climbing));
    }
    // 12 -> 16 and 24 (not better) -> the downward probe of 8 (10% below the
    // best) -> hold 12.
    try std.testing.expect(runtime.controller.state == .holding);
    try std.testing.expect(runtime.measurement == .inactive);
    try std.testing.expectEqual(@as(usize, 12), read_gate.currentLimit(io));
    try std.testing.expectEqual(@as(usize, 13), request_gate.currentLimit(io));
    try std.testing.expectEqual(@as(u64, 0), runtime.gate_closed_ticks);

    // Backoff while holding: one rung down, gate still open, window fenced
    // so a fresh admission can be told from delayed old-width feedback.
    try std.testing.expectEqual(@as(usize, 0), metrics.snapshot(io).probe_peak_reads);
    const backoff = runtime.controller.backoff(false).?;
    runtime.applyDecision(io, backoff);
    try std.testing.expectEqual(@as(usize, 8), read_gate.currentLimit(io));
    try std.testing.expectEqual(@as(usize, 9), request_gate.currentLimit(io));
    try std.testing.expect(runtime.measurement == .inactive);
    try std.testing.expect(runtime.controller.backoff(false) == null);
    metrics.beginRead(io, runtime.controller.generation, next_admission.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), metrics.snapshot(io).probe_peak_reads);
    try std.testing.expectEqual(@as(usize, 4), runtime.controller.backoff(true).?.width);
}

test "source read runtime measures from the reached width after a blind bootstrap" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    var read_gate: AdaptiveRequestGate = .init(12);
    var request_gate: AdaptiveRequestGate = .init(13);
    var next_admission: std.atomic.Value(u64) = .init(1);
    var runtime: SourceReadRuntime = .{
        .controller = SourceReadWidthController.init(
            .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
            128,
            128,
        ),
        .read_gate = &read_gate,
        .request_gate = &request_gate,
        .metrics = &metrics,
        .next_read_admission = &next_admission,
        .scheduler = undefined,
        .pinned_feasible_width = 128,
        .read_stats = null,
        .source_bootstrap_enabled = true,
    };
    runtime.start(io);
    runtime.applyBlindGrowth(io, runtime.controller.blindGrow().?);
    try std.testing.expect(runtime.measurement == .blind);
    try std.testing.expectEqual(@as(usize, 24), read_gate.currentLimit(io));
    try std.testing.expectEqual(std.math.maxInt(u64), metrics.snapshot(io).probe_epoch);
    runtime.applyBlindGrowth(io, runtime.controller.blindGrow().?);
    try std.testing.expectEqual(@as(usize, 32), read_gate.currentLimit(io));

    // The first response opens a measured window at 32 without a drain.
    next_admission.store(33, .release);
    runtime.applyDecision(io, runtime.controller.newGeneration());
    try std.testing.expect(runtime.measurement == .measuring);
    try std.testing.expectEqual(@as(usize, 32), read_gate.currentLimit(io));
    try std.testing.expectEqual(@as(usize, 33), request_gate.currentLimit(io));
    try std.testing.expectEqual(runtime.controller.generation, metrics.snapshot(io).probe_epoch);
    try std.testing.expectEqual(@as(u64, 33), metrics.probe_admission_start);
    try std.testing.expectEqual(@as(usize, 32), read_width_ladder[runtime.controller.start_index]);
}

test "source read runtime scores a window from its first admission on busy time" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    var runtime: SourceReadRuntime = undefined;
    runtime.controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    runtime.metrics = &metrics;
    runtime.clock = .{};
    metrics.prepareProbe(io, runtime.controller.generation, 1);
    // No admission yet: nothing to score however long the window has been open.
    try std.testing.expect(runtime.currentEvidence(io, std.math.maxInt(u64)) == null);
    // Thirteen reads: the first completion opens the window uncounted.
    for (1..14) |admission| {
        metrics.beginRead(io, runtime.controller.generation, admission);
    }
    for (1..14) |admission| {
        metrics.recordProbeRead(io, runtime.controller.generation, admission, max_load_read_request_size);
        metrics.endRead(io, runtime.controller.generation, admission);
    }
    const first_read_ns = metrics.snapshot(io).probe_window_start_ns;
    try std.testing.expect(first_read_ns != 0);
    runtime.clock.tick(first_read_ns, false);
    // 40 ms busy, 200 ms idle, 40 ms busy: 80 ms of busy time is too short.
    runtime.clock.tick(first_read_ns + 40 * std.time.ns_per_ms, false);
    runtime.clock.tick(first_read_ns + 240 * std.time.ns_per_ms, true);
    const short_ns = first_read_ns + 280 * std.time.ns_per_ms;
    try std.testing.expectEqual(80 * std.time.ns_per_ms, runtime.clock.busyNs(first_read_ns, short_ns));
    try std.testing.expect(runtime.currentEvidence(io, short_ns) == null);
    // Another 20 ms of busy time completes the 100 ms window.
    const scored_ns = short_ns + 20 * std.time.ns_per_ms;
    runtime.clock.tick(scored_ns, false);
    const evidence = runtime.currentEvidence(io, scored_ns).?;
    try std.testing.expectEqual(100 * std.time.ns_per_ms, evidence.elapsed_ns);
    try std.testing.expectEqual(@as(usize, 12), evidence.completed_requests);
    try std.testing.expectEqual(@as(usize, 13), evidence.exercised_width);
    try std.testing.expectEqual(@as(u64, 12 * max_load_read_request_size), evidence.bytes);
}

test "the submission that completes a target's bytes carries the last flag" {
    var target: VectoredTensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 100 };
    try std.testing.expect(!target.fullySubmitted());
    // Pieces arrive in any order; only the byte total matters.
    try std.testing.expect(!target.nextIsLast(20));
    target.noteSubmitted(20);
    try std.testing.expect(!target.nextIsLast(30));
    target.noteSubmitted(30);
    try std.testing.expect(!target.nextIsLast(10));
    try std.testing.expect(target.nextIsLast(50));
    try std.testing.expect(!target.fullySubmitted());
    target.noteSubmitted(50);
    try std.testing.expect(target.fullySubmitted());
}

/// A single-device pipeline without PJRT: enough for request, block and
/// batch lifecycle tests. Must not move after `init`.
const TestPipeline = struct {
    metrics: VectoredLoadMetrics = .{},
    gate: AdaptiveRequestGate,
    queues: [1]VectoredLoadPipeline.ReadyQueue = .{.empty},
    active: [1]usize = .{0},
    active_pieces: [1]usize = .{0},
    pipeline: VectoredLoadPipeline,

    fn init(
        self: *TestPipeline,
        gate_limit: usize,
        pool: ?*mem.DmaBlockPool,
        scheduler: *FairVectoredReadScheduler,
    ) void {
        self.* = .{ .gate = .init(gate_limit), .pipeline = undefined };
        self.pipeline = .{
            .allocator = std.testing.allocator,
            .io = std.testing.io,
            .platform = undefined,
            .pool = if (pool) |value| value else undefined,
            .read_gate = undefined,
            .request_gate = &self.gate,
            .block_size = 64,
            .device_pool_indices = &.{0},
            .numa_explicit = false,
            .metrics = &self.metrics,
            .scheduler = scheduler,
            .ready_queues = &self.queues,
            .active_bytes_by_device = &self.active,
            .active_pieces_by_device = &self.active_pieces,
            .dma_budget_bytes = 64,
        };
    }

    fn deinit(self: *TestPipeline) void {
        self.queues[0].deinit(std.testing.allocator);
    }

    /// The worker's claim-to-request sequence.
    fn claimRequest(
        self: *TestPipeline,
        scheduler: *FairVectoredReadScheduler,
    ) !*VectoredLoadPipeline.RequestContext {
        const io = std.testing.io;
        const claim = scheduler.claim(io) orelse return error.NoJob;
        try std.testing.expect(self.gate.acquire(io));
        self.pipeline.reserveSourceJob();
        return self.pipeline.registerRequest(claim);
    }
};

/// One pre-mapped arena; growth is refused.
const TestDmaArena = struct {
    storage: []u8,

    fn provider(self: *TestDmaArena) mem.DmaBlockPool.ArenaProvider {
        return .{
            .context = self,
            .node_count = 1,
            .arenaCountFn = arenaCount,
            .arenaFn = arenaAt,
            .allocateFn = allocate,
            .mappedBytesFn = mappedBytes,
        };
    }

    fn arenaCount(_: *anyopaque, _: usize) usize {
        return 1;
    }

    fn arenaAt(context: *anyopaque, _: usize, _: usize) []u8 {
        const self: *TestDmaArena = @ptrCast(@alignCast(context));
        return self.storage;
    }

    fn allocate(_: *anyopaque, _: usize, _: usize) anyerror![]u8 {
        return error.RequestExceedsCapacity;
    }

    fn mappedBytes(context: *anyopaque) usize {
        const self: *TestDmaArena = @ptrCast(@alignCast(context));
        return self.storage.len;
    }
};

test "late vectored callback failure drains and signals completion" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var arena: TestDmaArena = .{ .storage = try allocator.alloc(u8, 64) };
    defer allocator.free(arena.storage);
    var pool = try mem.DmaBlockPool.initFromProvider(allocator, arena.provider(), 64, 64, &.{0});
    defer pool.deinit();
    var scheduler: FairVectoredReadScheduler = .init(allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(1, &pool, &scheduler);
    defer fixture.deinit();
    const pipeline = &fixture.pipeline;

    // One request whose only block is queued behind a DMA slot that a failed
    // callback is about to free.
    const batch = try publishTestBatch(&scheduler, 1);
    const request = try fixture.claimRequest(&scheduler);
    var leased: [1]mem.DmaBlockPool.Block = undefined;
    var scratch = try pool.acquireScratch(allocator, 1);
    defer scratch.deinit();
    try pool.acquireMany(io, &leased, &.{.{}}, &scratch);
    const block = pipeline.registerBlock(request, leased[0], 1);
    try std.testing.expectEqual(&batch.plans.items[0].blocks[0], block);
    var target: VectoredTensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 64 };
    try fixture.queues[0].pushBack(allocator, .{
        .target = &target,
        .block = block,
        .source_offset = 0,
        .destination_offset = 0,
        .len = 64,
    });
    pipeline.ready_entries = 1;
    pipeline.active_events = 1;
    fixture.active[0] = 64;
    fixture.active_pieces[0] = 1;
    request.finishScheduling();
    try std.testing.expect(!batch.done.isSet());
    pipeline.first_error.store(@intFromError(error.Unknown), .release);

    pipeline.eventCompleted(0, 64);
    try std.testing.expectEqual(@as(usize, 0), pipeline.active_events);
    try std.testing.expectEqual(@as(usize, 0), pipeline.ready_entries);
    try std.testing.expect(block.lease.isComplete());
    try std.testing.expect(request.completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), fixture.gate.inUse(io));
    try std.testing.expect(batch.done.isSet());

    pipeline.retireBatch(batch);
    batch.destroy();
}

test "batch completes when every claimed request completes" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(3, null, &scheduler);
    defer fixture.deinit();

    const batch = try publishTestBatch(&scheduler, 3);
    try std.testing.expectEqual(@as(usize, 3), batch.remaining.load(.acquire));
    var requests: [3]*VectoredLoadPipeline.RequestContext = undefined;
    for (&requests) |*request| request.* = try fixture.claimRequest(&scheduler);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 3), fixture.gate.inUse(io));

    for (requests, 0..) |request, index| {
        try std.testing.expect(!batch.done.isSet());
        try std.testing.expectEqual(requests.len - index, batch.remaining.load(.acquire));
        request.finishScheduling();
    }
    try std.testing.expect(batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), fixture.gate.inUse(io));
    // Each claim took its job's slot of the plan's request array.
    for (requests, batch.plans.items[0].requests) |request, *slot| {
        try std.testing.expectEqual(slot, request);
    }

    fixture.pipeline.retireBatch(batch);
    batch.destroy();
}

test "retirement accepts the idle slots of jobs a failure retired" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(1, null, &scheduler);
    defer fixture.deinit();

    const batch = try publishTestBatch(&scheduler, 3);
    const request = try fixture.claimRequest(&scheduler);
    // `fail` retires the two unclaimed jobs; their request slots stay idle.
    scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    request.finishScheduling();
    try std.testing.expect(batch.done.isSet());
    const plan = batch.plans.items[0];
    try std.testing.expectEqual(&plan.requests[0], request);
    for (plan.requests[1..]) |*idle| {
        try std.testing.expectEqual(@as(usize, 0), idle.pending.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), idle.blocks.len);
    }
    // Every slot passes the retirement checks, claimed or idle.
    fixture.pipeline.retireBatch(batch);
    batch.destroy();
}

test "retired events are destroyed by the next pump or unlinked by the batch retirement" {
    // Without PJRT, contexts with a null event stand in for destroyed ones;
    // the list mechanics are the subject.
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(2, null, &scheduler);
    defer fixture.deinit();
    const pipeline = &fixture.pipeline;

    const batch = try publishTestBatch(&scheduler, 2);
    const requests = [_]*VectoredLoadPipeline.RequestContext{
        try fixture.claimRequest(&scheduler),
        try fixture.claimRequest(&scheduler),
    };
    const plan = batch.plans.items[0];
    plan.events_used = 2;
    for (plan.events, requests) |*ctx, request| ctx.* = .{
        .pipeline = pipeline,
        .block = &request.blocks[0],
        .pjrt_event = null,
        .device_index = 0,
        .len = 64,
    };
    // Two callbacks fired: the next pump destroys both, newest first.
    pipeline.retireEvent(&plan.events[0]);
    pipeline.retireEvent(&plan.events[1]);
    try std.testing.expectEqual(&plan.events[1], pipeline.retired);
    try std.testing.expectEqual(&plan.events[0], plan.events[1].next_retired);
    pipeline.metadata_mutex.lockUncancelable(io);
    pipeline.destroyRetired();
    pipeline.metadata_mutex.unlock(io);
    try std.testing.expect(pipeline.retired == null);
    try std.testing.expect(plan.events[0].next_retired == null);
    try std.testing.expect(plan.events[1].next_retired == null);

    // A callback that fires after the last pump leaves its context linked;
    // the batch's retirement unlinks it before the batch is freed.
    pipeline.retireEvent(&plan.events[0]);
    for (requests) |request| request.finishScheduling();
    try std.testing.expect(batch.done.isSet());
    pipeline.retireBatch(batch);
    try std.testing.expect(pipeline.retired == null);
    batch.destroy();
}

test "overlapping batches complete under concurrent claims and retirement" {
    const io = std.testing.io;
    const worker_count = 4;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(worker_count, null, &scheduler);
    defer fixture.deinit();

    const Worker = struct {
        fn run(fixture_: *TestPipeline, scheduler_: *FairVectoredReadScheduler) void {
            const io_ = std.testing.io;
            while (scheduler_.waitForWork(io_)) {
                if (!fixture_.gate.acquire(io_)) return;
                const claim = scheduler_.claim(io_) orelse {
                    fixture_.gate.release(io_);
                    continue;
                };
                fixture_.pipeline.reserveSourceJob();
                const request = fixture_.pipeline.registerRequest(claim);
                request.finishScheduling();
            }
        }
    };
    var group: std.Io.Group = .init;
    defer {
        scheduler.stop(io);
        fixture.gate.close(io);
        group.await(io) catch {};
    }
    for (0..worker_count) |_| try group.concurrent(io, Worker.run, .{ &fixture, &scheduler });

    // Three batches in flight at a time, awaited newest first: a batch is
    // retired while later ones are still being claimed.
    for (0..70) |round| {
        var batches: [3]*Batch = undefined;
        var job_counts: [3]usize = undefined;
        for (&batches, &job_counts, 0..) |*batch, *job_count, offset| {
            job_count.* = (round * 3 + offset) % 7;
            batch.* = try publishTestBatch(&scheduler, job_count.*);
        }
        var index = batches.len;
        while (index > 0) {
            index -= 1;
            const batch = batches[index];
            batch.done.waitUncancelable(io);
            fixture.pipeline.retireBatch(batch);
            batch.destroy();
        }
        try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);
    }
    fixture.gate.waitEmpty(io);
    try std.testing.expectEqual(@as(usize, 0), fixture.metrics.pending_source_jobs.load(.acquire));
}

fn buildMesh2x2x2(
    allocator: std.mem.Allocator,
    target: platform_mod.Target,
    devices: []const platform_mod.Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[0]),
                .device(devices[1]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[2]),
                .device(devices[3]),
            }),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[4]),
                .device(devices[5]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[6]),
                .device(devices[7]),
            }),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

const DispatchSpansTest = struct {
    const Scenario = struct {
        name: []const u8,
        device_count: u32,
        physical_mesh: CreateOptions.PhysicalMesh = .auto,
        shape: Shape,
        logical_mesh: Sharding.LogicalMesh,
        strategy: Sharding.Strategy,
        request_size: usize,
        block_size: usize,
    };

    fn run(scenario: Scenario) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        var platform = Platform.auto(allocator, io, .{
            .physical_mesh = scenario.physical_mesh,
            .cpu = .{ .device_count = scenario.device_count },
        }) catch return error.SkipZigTest;
        defer platform.deinit(allocator, io);

        const sharding_data: Sharding.Data = try .init(
            scenario.name,
            &platform.physical_mesh,
            scenario.logical_mesh,
            scenario.strategy,
        );
        try expectLayout(allocator, scenario.shape, .{ .data = &sharding_data }, scenario.request_size, scenario.block_size);
    }

    fn expectLayout(
        allocator: std.mem.Allocator,
        shape: Shape,
        sharding: Sharding,
        request_size: usize,
        block_size: usize,
    ) !void {
        const dispatch_spans: DispatchSpans = try .init(allocator, shape, sharding);
        defer dispatch_spans.deinit(allocator);

        const ordered_devices = sharding.devicesInCanonicalOrder();
        const writer_count = ordered_devices.len;
        const device_indices = try allocator.alloc(usize, writer_count);
        defer allocator.free(device_indices);
        var device_count: usize = 0;
        for (ordered_devices, device_indices) |device, *device_index| {
            device_index.* = @intCast(device.id);
            device_count = @max(device_count, device_index.* + 1);
        }
        const placement = try sharding.placement(shape);
        const writer_size = placement.shape.byteSize();
        const source = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(source);
        for (source, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 17);

        const expected = try allocator.alloc(u8, writer_count * writer_size);
        defer allocator.free(expected);
        @memset(expected, 0);
        for (dispatch_spans.spans) |span| {
            var mask = dispatch_spans.writerMask(span);
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const len = span.end - span.start;
                @memcpy(expected[writer_index * writer_size + span.writer_offset ..][0..len], source[span.start..span.end]);
            }
        }

        const actual = try allocator.alloc(u8, expected.len);
        defer allocator.free(actual);
        @memset(actual, 0);
        var source_tensor: safetensors.Tensor = .{
            .file_uri = "unused",
            .name = "value",
            .shape = shape,
            .offset = 0,
        };
        var item: LoaderLoadItem = .{
            .source = &source_tensor,
            .source_slot = undefined,
            .shape = shape,
            .sharding = sharding,
            .output = undefined,
        };
        var transfers: std.ArrayList(VectoredLoadPipeline.PlannedTransfer) = .empty;
        defer transfers.deinit(allocator);
        const physical_bytes = try allocator.alloc(usize, device_count);
        defer allocator.free(physical_bytes);
        // The pump flags a target's last transfer when its pieces reach the
        // placement's bytes, so every writer's pieces must sum to it.
        const written_bytes = try allocator.alloc(usize, writer_count);
        defer allocator.free(written_bytes);
        @memset(written_bytes, 0);

        const request_count = std.math.divCeil(usize, source.len, request_size) catch unreachable;
        var reverse_index = request_count;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const source_offset = reverse_index * request_size;
            const request_len = @min(request_size, source.len - source_offset);
            transfers.clearRetainingCapacity();
            @memset(physical_bytes, 0);
            try FairVectoredReadScheduler.appendTransfers(
                allocator,
                &transfers,
                0,
                &item,
                source_offset,
                request_len,
                source_offset,
                block_size,
                dispatch_spans,
                device_indices,
                physical_bytes,
            );
            for (transfers.items) |transfer| {
                try std.testing.expectEqual(&item, transfer.item);
                const block_source_offset = source_offset +
                    transfer.block_index * block_size + transfer.block_offset;
                var mask = transfer.writer_mask;
                while (mask != 0) {
                    const writer_index: usize = @intCast(@ctz(mask));
                    mask &= mask - 1;
                    try std.testing.expect(transfer.destination_offset + transfer.len <= writer_size);
                    @memcpy(
                        actual[writer_index * writer_size + transfer.destination_offset ..][0..transfer.len],
                        source[block_source_offset..][0..transfer.len],
                    );
                    written_bytes[writer_index] += transfer.len;
                }
            }
        }
        try std.testing.expectEqualSlices(u8, expected, actual);
        for (written_bytes) |bytes| try std.testing.expectEqual(writer_size, bytes);
    }
};

test "dispatch spans handle replication and block/request boundaries" {
    try DispatchSpansTest.run(.{
        .name = "replicated_boundaries",
        .device_count = 4,
        .shape = Shape.init(.{ .rows = 9, .cols = 257 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .request_size = 773,
        .block_size = 257,
    });
}

test "dispatch spans handle packed sub-byte storage" {
    const logical = Shape.init(.{ .rows = 9, .cols = 256 }, .u2)
        .withPartitioning(.{ .rows = .replicated, .cols = .replicated });
    const packed_shape = logical.packedShape();
    try std.testing.expectEqual(@as(usize, logical.byteSize()), packed_shape.byteSize());
    try DispatchSpansTest.run(.{
        .name = "packed_u2",
        .device_count = 4,
        .shape = packed_shape,
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .request_size = 131,
        .block_size = 67,
    });
}

test "dispatch spans handle 1D mirrored and folded sharding" {
    try DispatchSpansTest.run(.{
        .name = "mirrored_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .rows = 7, .model = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .model = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
        .request_size = 2053,
        .block_size = 509,
    });
    try DispatchSpansTest.run(.{
        .name = "folded_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .model = 4096 }, .f32).withPartitioning(.{ .model = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_y });
            break :blk strategy;
        },
        .request_size = 3001,
        .block_size = 997,
    });
}

test "dispatch spans handle 2D and 3D sharding" {
    try DispatchSpansTest.run(.{
        .name = "batch_model_2d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .batch = 8, .model = 1024 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model }),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y }),
        .request_size = 4093,
        .block_size = 1021,
    });
    try DispatchSpansTest.run(.{
        .name = "folded_model_3d",
        .device_count = 8,
        .physical_mesh = .{ .custom = buildMesh2x2x2 },
        .shape = Shape.init(.{ .batch = 16, .model = 4096 }, .f32)
            .withPartitioning(.{ .batch = .replicated, .model = .model }),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_z });
            break :blk strategy;
        },
        .request_size = 8191,
        .block_size = 2039,
    });
}
