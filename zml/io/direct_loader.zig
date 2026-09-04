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

const Parallelism = loader_types.Parallelism;
const LoaderOptions = loader_types.LoaderOptions;
const max_load_read_parallelism = load_limits.max_read_parallelism;
const max_load_read_request_size = load_limits.max_read_request_size;
const max_load_positional_iovecs = load_limits.max_positional_iovecs;
const maximumCoalescedJobBlocks = load_limits.maximumCoalescedJobBlocks;
const effectiveSourceRequestSize = load_limits.effectiveSourceRequestSize;
const DmaLoadConfig = dma.DmaLoadConfig;
const DmaPlatformSettings = dma.DmaPlatformSettings;
const requiredDmaWorkspaceBytes = dma.requiredDmaWorkspaceBytes;
const acquirePlatformDmaSettings = dma.acquirePlatformDmaSettings;
const releasePlatformDmaSettings = dma.releasePlatformDmaSettings;

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
    config_epoch: std.atomic.Value(u64) = .init(0),
    probe_epoch: u64 = std.math.maxInt(u64),
    probe_admission_start: u64 = std.math.maxInt(u64),
    probe_first_read_ns: u64 = 0,
    probe_active_reads: usize = 0,
    probe_peak_reads: usize = 0,
    probe_read_operations: u64 = 0,
    probe_read_bytes: u64 = 0,
    probe_mutex: std.Io.Mutex = .init,

    const Snapshot = struct {
        probe_epoch: u64,
        probe_first_read_ns: u64,
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
            .probe_first_read_ns = self.probe_first_read_ns,
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
        if (self.probe_first_read_ns == 0)
            self.probe_first_read_ns = @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
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
        self.probe_first_read_ns = 0;
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
        self.probe_first_read_ns = 0;
        self.probe_active_reads = 0;
        self.probe_peak_reads = 0;
        self.probe_read_operations = 0;
        self.probe_read_bytes = 0;
    }
};

const VectoredTensorTransfer = struct {
    const Target = struct {
        manager: *pjrt.AsyncHostToDeviceTransferManager,
        device_index: usize,
        total: usize,
        submitted_bytes: std.atomic.Value(usize) = .init(0),
        final_submitted: bool = false,
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
        const uninitialized = 0;
        const initializing = 1;
        const ready = 2;
        const failed = 3;

        value: T = undefined,
        status: std.atomic.Value(u8) = .init(uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(self: *Self, io: std.Io, ctx: Ctx) !*T {
            while (true) switch (self.status.load(.acquire)) {
                uninitialized => {
                    if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                    self.value = initFn(ctx) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(io);
                        return err;
                    };
                    self.status.store(ready, .release);
                    self.initialized.set(io);
                    return &self.value;
                },
                initializing => self.initialized.waitUncancelable(io),
                ready => return &self.value,
                failed => return @errorFromInt(self.error_code.load(.acquire)),
                else => unreachable,
            };
        }

        /// The value when initialization has completed successfully.
        fn readyValue(self: *Self) ?*T {
            return if (self.status.load(.acquire) == ready) &self.value else null;
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
    condition: std.Io.Condition = .init,

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
        if (self.in_use == 0) {
            self.condition.broadcast(io);
            return;
        }
        // One release creates one admission slot. Waking every worker here
        // turns a high adaptive cap into a thundering herd even when the
        // active limit is small.
        self.condition.signal(io);
    }

    fn waitEmpty(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.in_use != 0) self.condition.waitUncancelable(io, &self.mutex);
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

const RequestGateLimits = struct {
    read: usize,
    lifecycle: usize,

    fn init(read: usize, feasible_width: usize) RequestGateLimits {
        std.debug.assert(feasible_width > 0);
        const effective_read = @min(read, feasible_width);
        return .{
            .read = effective_read,
            .lifecycle = @min(feasible_width, effective_read +| 1),
        };
    }
};

fn selectLoaderDmaDevice(
    active: []const usize,
    per_device_limit: usize,
    ready_mask: u64,
    next_device: usize,
) ?usize {
    std.debug.assert(active.len > 0 and active.len <= 64);
    std.debug.assert(per_device_limit > 0 and next_device < active.len);
    for (0..active.len) |offset| {
        const device_index = (next_device + offset) % active.len;
        if (ready_mask & (@as(u64, 1) << @intCast(device_index)) == 0 or
            active[device_index] >= per_device_limit)
        {
            continue;
        }
        return device_index;
    }
    return null;
}

/// One planned submission: the immutable plan the scheduler hands out, the
/// per-tensor items it writes, and every request, block and event context
/// created while loading it. `remaining` counts completion units: one per
/// job plus a publish sentinel. A job's unit is released exactly once, by
/// whichever of these happens: its request's last reference drops (final DMA
/// callback or abandonment), a worker abandons the claimed job before a
/// request exists, or `FairVectoredReadScheduler.fail` retires it unclaimed.
/// The batch is done when `remaining` reaches zero; the awaiting task then
/// retires it, so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    const Diagnostics = struct {
        /// Submission number within the loader, for log correlation.
        sequence: usize = 0,
        logical_bytes: usize = 0,
        source_bytes: u64 = 0,
        source_jobs: usize = 0,
        source_runs: usize = 0,
        source_items: usize = 0,
        planned_transfers: usize = 0,
        planning_ns: u64 = 0,
        published_at: ?std.Io.Timestamp = null,
        /// Stamped by the scheduler when the first job is claimed.
        first_claim_at: ?std.Io.Timestamp = null,
        /// Aggregate source statistics at publish; the completion log reports
        /// the delta against them (loader-wide while batches overlap).
        source_stats: ?VFS.ReadStats = null,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    jobs: []FairVectoredReadScheduler.Job,
    transfers: []VectoredLoadPipeline.PlannedTransfer,
    /// Owned once published; empty before so an unpublished batch can be
    /// destroyed while the caller still owns the items.
    items: []*LoaderLoadItem = &.{},
    /// Next job to claim; owned by the scheduler mutex.
    cursor: usize = 0,
    remaining: std.atomic.Value(usize),
    done: std.Io.Event = .unset,
    requests: std.ArrayListUnmanaged(*VectoredLoadPipeline.RequestContext) = .empty,
    blocks: std.ArrayListUnmanaged(*VectoredLoadPipeline.BlockContext) = .empty,
    events: std.ArrayListUnmanaged(*VectoredLoadPipeline.EventContext) = .empty,
    diagnostics: Diagnostics,
    /// Set by retirement so a unit released after `done` trips `finishJobs`.
    freeing: if (builtin.mode == .Debug) bool else void = if (builtin.mode == .Debug) false else {},

    /// Takes ownership of the plan's jobs and transfers. The sentinel keeps
    /// the batch from completing before the publisher has made it visible.
    fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        plan: *FairVectoredReadScheduler.PreparedBatch,
        diagnostics: Diagnostics,
    ) !*Batch {
        const self = try allocator.create(Batch);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .jobs = plan.jobs,
            .transfers = plan.transfers,
            .remaining = .init(1 + plan.jobs.len),
            .diagnostics = diagnostics,
        };
        plan.* = undefined;
        return self;
    }

    /// Releases `count` completion units. MEMORY-ORDER RULE: this must be the
    /// caller's final access to the batch and to anything it owns (requests,
    /// blocks, events, items, jobs, transfers). The last unit sets `done`,
    /// and the awaiting task frees all of it as soon as it observes the event.
    fn finishJobs(self: *Batch, count: usize) void {
        if (count == 0) return;
        if (builtin.mode == .Debug) std.debug.assert(!self.freeing);
        const previous = self.remaining.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.done.set(self.io);
    }

    /// Frees the items and the plan. Contexts must already have been retired
    /// by `VectoredLoadPipeline.retireBatch`.
    fn destroy(self: *Batch) void {
        std.debug.assert(self.requests.items.len == 0);
        std.debug.assert(self.blocks.items.len == 0);
        std.debug.assert(self.events.items.len == 0);
        for (self.items) |item| item.deinit(self.allocator);
        self.allocator.free(self.items);
        self.allocator.free(self.transfers);
        self.allocator.free(self.jobs);
        self.requests.deinit(self.allocator);
        self.blocks.deinit(self.allocator);
        self.events.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
        batch: *Batch,
        pending: std.atomic.Value(usize) = .init(1), // scheduling sentinel
        completed: std.atomic.Value(bool) = .init(false),
        source_finished: std.atomic.Value(bool) = .init(false),
        read_epoch: u64,
        admission_id: u64 = 0,

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

    const EventContext = struct {
        pipeline: *VectoredLoadPipeline,
        block: *BlockContext,
        pjrt_event: *pjrt.Event,
        err: ?*pjrt.Error = null,
        device_index: usize,
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
    ready_queues: []std.ArrayListUnmanaged(ReadyTransfer),
    active_by_device: []usize,
    dma_limit: usize,
    next_device: usize = 0,
    pumping: bool = false,
    active_events: usize = 0,
    ready_entries: usize = 0,

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
        dma_limit: usize,
    ) !VectoredLoadPipeline {
        std.debug.assert(platform.devices.len <= 64);
        const ready_queues = try allocator.alloc(std.ArrayListUnmanaged(ReadyTransfer), platform.devices.len);
        errdefer allocator.free(ready_queues);
        @memset(ready_queues, .empty);
        const active_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(active_by_device);
        @memset(active_by_device, 0);
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
            .active_by_device = active_by_device,
            .dma_limit = dma_limit,
        };
    }

    fn deinit(self: *VectoredLoadPipeline) void {
        // Every batch was retired: no DMA event, queued transfer or request
        // context may outlive its batch.
        std.debug.assert(self.active_events == 0);
        std.debug.assert(self.ready_entries == 0);
        std.debug.assert(self.request_gate.inUse(self.io) == 0);
        for (self.ready_queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.ready_queues);
        self.allocator.free(self.active_by_device);
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

    fn registerRequest(self: *VectoredLoadPipeline, batch: *Batch) !*RequestContext {
        const request = try self.allocator.create(RequestContext);
        errdefer self.allocator.destroy(request);
        request.* = .{
            .pipeline = self,
            .batch = batch,
            .read_epoch = 0,
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try batch.requests.append(self.allocator, request);
        return request;
    }

    /// Destroys every context a done batch owns. Runs under `metadata_mutex`
    /// so an `abortReady` still iterating queued entries cannot race the free.
    /// PJRT events are destroyed here, from the awaiting task after their
    /// callbacks fired, exactly as `pjrt.Event.await` does.
    fn retireBatch(self: *VectoredLoadPipeline, batch: *Batch) void {
        std.debug.assert(batch.done.isSet());
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        if (builtin.mode == .Debug) batch.freeing = true;
        for (batch.events.items) |ctx| {
            ctx.pjrt_event.deinit(self.platform.pjrt_api);
            if (ctx.err) |err| err.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
        }
        for (batch.blocks.items) |block| {
            std.debug.assert(block.lease.isComplete());
            self.allocator.destroy(block);
        }
        for (batch.requests.items) |request| {
            std.debug.assert(request.completed.load(.acquire));
            self.allocator.destroy(request);
        }
        batch.events.clearRetainingCapacity();
        batch.blocks.clearRetainingCapacity();
        batch.requests.clearRetainingCapacity();
    }

    fn reserveSourceJob(self: *VectoredLoadPipeline) void {
        _ = self.metrics.pending_source_jobs.fetchAdd(1, .acq_rel);
    }

    fn abandonSourceJob(self: *VectoredLoadPipeline) void {
        const previous = self.metrics.pending_source_jobs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
    }

    fn registerBlock(
        self: *VectoredLoadPipeline,
        request: *RequestContext,
        dma_block: mem.DmaBlockPool.Block,
        references: usize,
    ) !*BlockContext {
        const block = try self.allocator.create(BlockContext);
        errdefer self.allocator.destroy(block);
        block.* = .{
            .pipeline = self,
            .request = request,
            .lease = .init(self.pool, self.io, dma_block, references),
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try request.batch.blocks.append(self.allocator, block);
        request.addBlock();
        return block;
    }

    fn transferReady(self: *const VectoredLoadPipeline, transfer: ReadyTransfer) bool {
        _ = self;
        if (transfer.destination_offset + transfer.len == transfer.target.total and
            transfer.target.submitted_bytes.load(.acquire) != transfer.destination_offset)
        {
            return false;
        }
        return true;
    }

    fn enqueueBlocks(
        self: *VectoredLoadPipeline,
        transfers: []const PlannedTransfer,
        blocks: []const *BlockContext,
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
            const block = blocks[transfer.block_index];
            const tensor = &transfer.item.state.value;
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const target = &tensor.targets[writer_index];
                self.ready_queues[target.device_index].appendAssumeCapacity(.{
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

    fn reserveBlockCapacity(self: *VectoredLoadPipeline, batch: *Batch, count: usize) !void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try batch.blocks.ensureUnusedCapacity(self.allocator, count);
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
            if (!self.failed()) {
                const limit = self.dma_limit;
                var ready_mask: u64 = 0;
                for (self.ready_queues, 0..) |queue, device_index| {
                    if (self.active_by_device[device_index] >= limit) continue;
                    for (queue.items) |transfer| {
                        if (self.transferReady(transfer)) {
                            ready_mask |= @as(u64, 1) << @intCast(device_index);
                            break;
                        }
                    }
                }
                const device_index = selectLoaderDmaDevice(
                    self.active_by_device,
                    limit,
                    ready_mask,
                    self.next_device,
                );
                if (device_index) |index| {
                    const queue = &self.ready_queues[index];
                    for (queue.items, 0..) |transfer, i| {
                        if (!self.transferReady(transfer)) continue;
                        selected = queue.swapRemove(i);
                        break;
                    }
                    std.debug.assert(selected != null);
                    self.next_device = (index + 1) % self.ready_queues.len;
                    self.active_by_device[index] += 1;
                    std.debug.assert(self.active_by_device[index] <= limit);
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
        const block = transfer.block;
        self.submitTransfer(transfer) catch |err| {
            self.recordError(err);
            self.eventCompleted(device_index);
            block.complete();
        };
    }

    fn submitTransfer(self: *VectoredLoadPipeline, transfer: ReadyTransfer) !void {
        const api = self.platform.pjrt_api;
        const is_last = transfer.destination_offset + transfer.len == transfer.target.total;
        const event = try transfer.target.manager.transferData(
            api,
            0,
            transfer.block.lease.data[transfer.source_offset..][0..transfer.len],
            @intCast(transfer.destination_offset),
            is_last,
        );
        if (is_last) transfer.target.final_submitted = true;
        _ = transfer.target.submitted_bytes.fetchAdd(transfer.len, .release);

        const ctx = self.allocator.create(EventContext) catch |err| {
            event.awaitRaw(api) catch {};
            event.deinit(api);
            return err;
        };
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .device_index = transfer.target.device_index,
        };

        const batch = transfer.block.request.batch;
        self.metadata_mutex.lockUncancelable(self.io);
        batch.events.append(self.allocator, ctx) catch |err| {
            self.metadata_mutex.unlock(self.io);
            event.awaitRaw(api) catch {};
            event.deinit(api);
            self.allocator.destroy(ctx);
            return err;
        };
        self.metadata_mutex.unlock(self.io);

        _ = self.metrics.dma_submissions.fetchAdd(1, .monotonic);
        event.onReady(api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                // MEMORY-ORDER RULE: the batch owns `ctx_` and its block, and
                // `block.complete()` may complete that batch, after which the
                // awaiting task frees the context, the block and the batch.
                // Load every field first, store the error, retire the DMA
                // slot, and complete the block last.
                const pipeline = ctx_.pipeline;
                const device_index = ctx_.device_index;
                const block = ctx_.block;
                ctx_.err = err;
                if (err) |pjrt_error| {
                    pipeline.recordError(pjrt_error.getCode(pipeline.platform.pjrt_api).toApiError());
                }
                pipeline.eventCompleted(device_index);
                block.complete();
            }
        }.call, ctx) catch |err| {
            // The batch owns `ctx` and destroys the event at retirement.
            event.awaitRaw(api) catch {};
            return err;
        };
    }

    fn eventCompleted(self: *VectoredLoadPipeline, device_index: usize) void {
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(self.active_events > 0);
        std.debug.assert(self.active_by_device[device_index] > 0);
        self.active_events -= 1;
        self.active_by_device[device_index] -= 1;
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
            for (queue.items) |transfer| {
                transfer.block.complete();
                self.ready_entries -= 1;
            }
            queue.clearRetainingCapacity();
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
        blocks: []*VectoredLoadPipeline.BlockContext,
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
            const blocks = try allocator.alloc(*VectoredLoadPipeline.BlockContext, maximum_blocks);
            errdefer allocator.free(blocks);
            const queue_counts = try allocator.alloc(usize, device_count);
            errdefer allocator.free(queue_counts);
            const pool_scratch = try pool.acquireScratch(allocator, maximum_blocks);
            return .{
                .allocator = allocator,
                .leased = leased,
                .affinities = affinities,
                .references = references,
                .iovecs = iovecs,
                .blocks = blocks,
                .queue_counts = queue_counts,
                .pool = pool_scratch,
            };
        }

        fn deinit(self: *Scratch) void {
            self.pool.deinit();
            self.allocator.free(self.queue_counts);
            self.allocator.free(self.blocks);
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
            const tensor = transfer.item.ensureState(direct) catch |err| {
                pipeline.recordError(err);
                return;
            };
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

        pipeline.reserveBlockCapacity(request.batch, block_count) catch |err| {
            pipeline.recordError(err);
            return;
        };
        pipeline.pool.acquireMany(pipeline.io, leased, affinities, &scratch.pool) catch |err| {
            pipeline.recordError(err);
            return;
        };
        if (pipeline.failed()) return;

        const iovecs = scratch.iovecs[0..block_count];
        for (iovecs, leased, 0..) |*iovec, block, block_index| {
            const consumed = block_index * pipeline.block_size;
            const len = @min(pipeline.block_size, request_len - consumed);
            iovec.* = block.data[0..len];
        }

        if (!beginRead(request, pipeline)) return;
        const read_result = readAbsoluteAllV(
            pipeline.io,
            file,
            iovecs,
            file_offset,
            pipeline.metrics,
        );
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

        const blocks = scratch.blocks[0..block_count];
        var initialized_blocks: usize = 0;
        for (blocks, leased, references) |*block, *lease, refs| {
            if (refs == 0) {
                for (blocks[0..initialized_blocks], references[0..initialized_blocks]) |initialized, initialized_refs| {
                    VectoredLoadPipeline.abandonSubmissions(initialized, initialized_refs);
                }
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            }
            block.* = pipeline.registerBlock(request, lease.*, refs) catch |err| {
                for (blocks[0..initialized_blocks], references[0..initialized_blocks]) |initialized, initialized_refs| {
                    VectoredLoadPipeline.abandonSubmissions(initialized, initialized_refs);
                }
                pipeline.recordError(err);
                return;
            };
            lease.data = &.{};
            initialized_blocks += 1;
        }
        pipeline.enqueueBlocks(transfers, blocks, queue_counts) catch |err| {
            for (transfers) |transfer| {
                VectoredLoadPipeline.abandonSubmissions(
                    blocks[transfer.block_index],
                    @popCount(transfer.writer_mask),
                );
            }
            pipeline.recordError(err);
            return;
        };
    }
};

/// A strict FIFO of published batches. Within a batch, jobs are handed out in
/// the planned order (fair by destination-device bytes, predecessor-safe);
/// a later batch's first job is claimed only after every job of the earlier
/// ones. The queue holds only batches with unclaimed jobs: a batch is popped
/// with its last claim (or by `fail`), and it can only be freed after `done`,
/// which needs every job claimed or retired, so a queued batch is never freed.
const FairVectoredReadScheduler = struct {
    const Job = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfers: []const VectoredLoadPipeline.PlannedTransfer,
    };

    const PlanningJob = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfer_start: usize,
        transfer_len: usize,
        predecessor: ?usize,
    };

    const Snapshot = struct {
        remaining_jobs: usize,
    };

    /// A claimed job and the batch that owns it. The claim holds one of the
    /// batch's completion units until the worker releases it.
    const Claim = struct {
        batch: *Batch,
        job: Job,
    };

    /// The planner's output; `Batch.create` takes ownership of it.
    const PreparedBatch = struct {
        allocator: std.mem.Allocator,
        jobs: []Job,
        transfers: []VectoredLoadPipeline.PlannedTransfer,
        source_bytes: u64,
        source_runs: usize,

        fn deinit(self: *PreparedBatch) void {
            self.allocator.free(self.jobs);
            self.allocator.free(self.transfers);
            self.* = undefined;
        }
    };

    allocator: std.mem.Allocator,
    /// Batches with unclaimed jobs in publish order; `head` is the first.
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
                var candidate: ?usize = null;
                for (queue.items[cursors[device_index]..]) |job_index| {
                    if (claimed[job_index]) continue;
                    const predecessor = jobs[job_index].predecessor;
                    if (predecessor == null or claimed[predecessor.?]) {
                        candidate = job_index;
                        break;
                    }
                }
                if (candidate == null) continue;
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

    fn prepareBatch(
        allocator: std.mem.Allocator,
        device_count: usize,
        items: []const *LoaderLoadItem,
        block_size: usize,
        request_size: usize,
    ) !PreparedBatch {
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
        const plans = try allocator.alloc(TensorPlan, items.len);
        var initialized_plans: usize = 0;
        defer {
            for (plans[0..initialized_plans]) |*plan| {
                plan.dispatch_spans.deinit(allocator);
                allocator.free(plan.device_indices);
            }
            allocator.free(plans);
        }
        for (items, plans) |item, *plan| {
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

        const order = try allocator.alloc(usize, items.len);
        defer allocator.free(order);
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

        var jobs_list: std.ArrayList(PlanningJob) = .empty;
        defer jobs_list.deinit(allocator);
        var transfers_list: std.ArrayList(VectoredLoadPipeline.PlannedTransfer) = .empty;
        defer transfers_list.deinit(allocator);
        var physical_list: std.ArrayList(usize) = .empty;
        defer physical_list.deinit(allocator);
        var safe_boundaries: std.ArrayList(u64) = .empty;
        defer safe_boundaries.deinit(allocator);
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        errdefer allocator.free(queues);
        @memset(queues, .empty);
        errdefer for (queues) |*queue| queue.deinit(allocator);
        var source_bytes: u64 = 0;
        var source_runs: usize = 0;
        var file_start: usize = 0;
        while (file_start < order.len) {
            const first_item = items[order[file_start]];
            var file_end = file_start + 1;
            while (file_end < order.len and std.mem.eql(
                u8,
                first_item.source.file_uri,
                items[order[file_end]].source.file_uri,
            )) : (file_end += 1) {}

            var previous_job: ?usize = null;
            var run_cursor = file_start;
            while (run_cursor < file_end) {
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
                while (run_item_end < file_end) : (run_item_end += 1) {
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
                    for (order[candidate_start..run_item_end]) |item_index| {
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
                            plans[item_index].dispatch_spans,
                            plans[item_index].device_indices,
                            row,
                        );
                    }
                    std.debug.assert(transfers_list.items.len > transfer_start);
                    try jobs_list.append(allocator, .{
                        .source_slot = first_item.source_slot,
                        .file_offset = job_start,
                        .len = job_len,
                        .transfer_start = transfer_start,
                        .transfer_len = transfers_list.items.len - transfer_start,
                        .predecessor = previous_job,
                    });
                    source_bytes +|= @intCast(job_len);
                    previous_job = job_index;
                    for (row, queues) |bytes, *queue| {
                        if (bytes != 0) try queue.append(allocator, job_index);
                    }
                    job_start = job_end;
                    jobs_remaining -= 1;
                }
                std.debug.assert(job_start == run_end);
                run_cursor = run_item_end;
            }
            file_start = file_end;
        }

        const planning_jobs = try jobs_list.toOwnedSlice(allocator);
        defer allocator.free(planning_jobs);
        const transfers = try transfers_list.toOwnedSlice(allocator);
        errdefer allocator.free(transfers);
        const physical_bytes = try physical_list.toOwnedSlice(allocator);
        defer allocator.free(physical_bytes);
        const fair_order = try fairOrder(allocator, planning_jobs, physical_bytes, queues);
        defer allocator.free(fair_order);
        const jobs = try allocator.alloc(Job, planning_jobs.len);
        errdefer allocator.free(jobs);
        for (jobs, fair_order) |*job, planning_index| {
            const planned = planning_jobs[planning_index];
            job.* = .{
                .source_slot = planned.source_slot,
                .file_offset = planned.file_offset,
                .len = planned.len,
                .transfers = transfers[planned.transfer_start..][0..planned.transfer_len],
            };
        }
        for (queues) |*queue| queue.deinit(allocator);
        allocator.free(queues);
        return .{
            .allocator = allocator,
            .jobs = jobs,
            .transfers = transfers,
            .source_bytes = source_bytes,
            .source_runs = source_runs,
        };
    }

    /// Appends a batch behind every earlier one. A batch without jobs is not
    /// queued: it completes when its publisher drops the sentinel.
    fn publish(self: *FairVectoredReadScheduler, io: std.Io, batch: *Batch) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.LoaderShuttingDown;
        if (batch.jobs.len == 0) return;
        std.debug.assert(batch.cursor == 0);
        try self.queue.append(self.allocator, batch);
        self.unclaimed_total += batch.jobs.len;
        self.condition.broadcast(io);
    }

    fn stop(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
    }

    /// Stops claims and retires the unclaimed units of every queued batch so
    /// each still reaches `done` through its claimed requests. Claims and
    /// this pass both move cursors under the mutex, so they partition the
    /// jobs exactly: every claimed job keeps its unit with its worker.
    fn fail(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
        for (self.queue.items[self.head..]) |batch| {
            const unclaimed = batch.jobs.len - batch.cursor;
            batch.cursor = batch.jobs.len;
            // Last access: the retired units may complete the batch.
            batch.finishJobs(unclaimed);
        }
        self.queue.clearRetainingCapacity();
        self.head = 0;
        self.unclaimed_total = 0;
    }

    /// Hands out the head batch's next job; the batch leaves the queue with
    /// its last one.
    fn claim(self: *FairVectoredReadScheduler, io: std.Io) ?Claim {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.head == self.queue.items.len) return null;
        const batch = self.queue.items[self.head];
        std.debug.assert(batch.cursor < batch.jobs.len);
        if (batch.cursor == 0) batch.diagnostics.first_claim_at = .now(io, .awake);
        const job = batch.jobs[batch.cursor];
        batch.cursor += 1;
        self.unclaimed_total -= 1;
        if (batch.cursor == batch.jobs.len) {
            self.head += 1;
            if (self.head == self.queue.items.len) {
                self.queue.clearRetainingCapacity();
                self.head = 0;
            }
        }
        return .{ .batch = batch, .job = job };
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

/// Planning input for the fair-order tests: one job per entry, chained to the
/// previous job of the same tensor; `file_offset` is the entry index.
const FairOrderJob = struct {
    tensor_index: usize,
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
    const previous_jobs = try allocator.alloc(?usize, jobs.len);
    defer allocator.free(previous_jobs);
    @memset(previous_jobs, null);
    for (jobs, planning_jobs, 0..) |job, *planned, job_index| {
        if (job.physical_bytes.len != device_count) return error.InvalidTestJob;
        planned.* = .{
            .source_slot = undefined,
            .file_offset = job_index,
            .len = 1,
            .transfer_start = 0,
            .transfer_len = 0,
            .predecessor = if (job.tensor_index < jobs.len) previous_jobs[job.tensor_index] else null,
        };
        if (job.tensor_index < jobs.len) previous_jobs[job.tensor_index] = job_index;
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

/// A batch of `job_count` unit jobs (`file_offset` = index) whose sentinel
/// is still held; publish it with `publishTestBatch`.
fn testBatch(allocator: std.mem.Allocator, job_count: usize) !*Batch {
    const jobs = try allocator.alloc(FairVectoredReadScheduler.Job, job_count);
    errdefer allocator.free(jobs);
    for (jobs, 0..) |*job, index| job.* = .{
        .source_slot = undefined,
        .file_offset = index,
        .len = 1,
        .transfers = &.{},
    };
    var plan: FairVectoredReadScheduler.PreparedBatch = .{
        .allocator = allocator,
        .jobs = jobs,
        .transfers = &.{},
        .source_bytes = job_count,
        .source_runs = job_count,
    };
    return Batch.create(allocator, std.testing.io, &plan, .{});
}

/// `DirectLoader.submit`'s publish sequence.
fn publishTestBatch(scheduler: *FairVectoredReadScheduler, job_count: usize) !*Batch {
    const io = std.testing.io;
    const batch = try testBatch(std.testing.allocator, job_count);
    scheduler.publish(io, batch) catch |err| {
        batch.destroy();
        return err;
    };
    batch.finishJobs(1);
    return batch;
}

test "fair order rotates sharded devices by scheduled bytes" {
    try expectFairOrder(2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 1, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 3, .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 2, 1, 3 });
}

test "source batch coalesces exact adjacent and overlapping tensor ranges" {
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

    var batch = try FairVectoredReadScheduler.prepareBatch(
        allocator,
        device_count,
        &item_ptrs,
        4,
        8,
    );
    defer batch.deinit();

    // a:[10,18) merges adjacency and the duplicate, a:[20,24) remains exact,
    // and b:[3,15) is split at the request-size boundary.
    try std.testing.expectEqual(@as(usize, 3), batch.source_runs);
    try std.testing.expectEqual(@as(usize, 4), batch.jobs.len);
    try std.testing.expectEqual(@as(u64, 24), batch.source_bytes);
    try std.testing.expectEqual(@as(usize, 7), batch.transfers.len);
    try std.testing.expectEqual(@as(u64, 10), batch.jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), batch.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 3), batch.jobs[0].transfers.len);
    try std.testing.expectEqual(@as(u64, 20), batch.jobs[1].file_offset);
    try std.testing.expectEqual(@as(u64, 3), batch.jobs[2].file_offset);
    try std.testing.expectEqual(@as(usize, 8), batch.jobs[2].len);
    try std.testing.expectEqual(@as(usize, 4), batch.jobs[3].len);

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
    var iov_batch = try FairVectoredReadScheduler.prepareBatch(
        allocator,
        device_count,
        &.{&iov_item},
        1,
        max_load_positional_iovecs + 1,
    );
    defer iov_batch.deinit();
    try std.testing.expectEqual(@as(usize, 2), iov_batch.jobs.len);
    try std.testing.expectEqual(max_load_positional_iovecs, iov_batch.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 1), iov_batch.jobs[1].len);

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
    var aligned_batch = try FairVectoredReadScheduler.prepareBatch(
        allocator,
        device_count,
        &aligned_item_ptrs,
        4,
        8,
    );
    defer aligned_batch.deinit();
    try std.testing.expectEqual(@as(usize, 3), aligned_batch.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_batch.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_batch.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 8), aligned_batch.jobs[1].len);
    try std.testing.expectEqual(@as(usize, 5), aligned_batch.jobs[2].len);
}

test "fair order preserves per-tensor request order" {
    try expectFairOrder(2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{ 1, 0 } },
        .{ .tensor_index = 0, .physical_bytes = &.{ 0, 2 } },
        .{ .tensor_index = 1, .physical_bytes = &.{ 0, 3 } },
    }, &.{ 0, 1, 2 });
}

test "fair order places a replicated job once and credits every replica" {
    // The replicated entry is skipped in device 1's queue; tie rotation gives
    // that device the next scheduling turn.
    try expectFairOrder(2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{ 20, 20 } },
        .{ .tensor_index = 1, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 2, 1 });
}

test "fair order compares physical bytes rather than scheduling turns" {
    // Device 0 receives a third turn because it has 8 scheduled bytes while
    // device 1 has 10; a turn-count scheduler would alternate.
    try expectFairOrder(2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 1, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 2, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 3, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 4, .physical_bytes = &.{ 0, 10 } },
    }, &.{ 0, 3, 1, 2, 4 });
}

test "fair order validates jobs and cleans up allocation failures" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidTestJob, testFairOrder(allocator, 2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{1} },
    }));
    // A job that no device queue lists can never be selected.
    try std.testing.expectError(error.InvalidLoaderJob, testFairOrder(allocator, 2, &.{
        .{ .tensor_index = 0, .physical_bytes = &.{ 0, 0 } },
    }));
    try std.testing.expectError(error.DmaDeviceMismatch, testFairOrder(allocator, 0, &.{}));
    var planning = [_]FairVectoredReadScheduler.PlanningJob{.{
        .source_slot = undefined,
        .file_offset = 0,
        .len = 1,
        .transfer_start = 0,
        .transfer_len = 0,
        .predecessor = null,
    }};
    const queues = [_]std.ArrayListUnmanaged(usize){ .empty, .empty };
    try std.testing.expectError(
        error.InvalidLoaderJob,
        FairVectoredReadScheduler.fairOrder(allocator, &planning, &.{1}, &queues),
    );

    const AllocationTest = struct {
        fn run(allocator_: std.mem.Allocator) !void {
            const order = try testFairOrder(allocator_, 2, &.{
                .{ .tensor_index = 0, .physical_bytes = &.{ 1, 1 } },
                .{ .tensor_index = 1, .physical_bytes = &.{ 1, 0 } },
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
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();

    var woke: std.atomic.Value(u8) = .init(0);
    var group: std.Io.Group = .init;
    const Waiter = struct {
        fn run(scheduler_: *FairVectoredReadScheduler, io_: std.Io, woke_: *std.atomic.Value(u8)) void {
            woke_.store(if (scheduler_.waitForWork(io_)) 1 else 2, .release);
        }
    };
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(@as(u8, 0), woke.load(.acquire));
    const batch = try publishTestBatch(&scheduler, 1);
    try group.await(io);
    try std.testing.expectEqual(@as(u8, 1), woke.load(.acquire));
    _ = scheduler.claim(io).?;
    batch.finishJobs(1);
    batch.destroy();

    woke.store(0, .release);
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(@as(u8, 0), woke.load(.acquire));
    scheduler.stop(io);
    try group.await(io);
    try std.testing.expectEqual(@as(u8, 2), woke.load(.acquire));
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

const read_width_ladder = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

/// Source-only adaptive state. DMA width and request size never enter its
/// evidence or decisions.
/// Climb-and-hold source width policy. Every scored window is attributed by
/// the admission fence to the rung in effect when its reads were admitted,
/// so a rung change never drains the read gate. The controller climbs the
/// ladder one rung per window while each rung beats the best rate seen by
/// 3%, then holds at the lowest rung within 3% of the best. One borderline
/// re-measure of the hold rung and one downward probe below the start rung
/// bound the number of windows a load spends away from its final width.
const SourceReadWidthController = struct {
    const State = enum { climbing, holding };

    /// A rung must beat the best rate by this factor to keep the climb going.
    const improvement_ratio = 1.03;
    /// The hold rung is the lowest rung retaining this fraction of the best.
    const hold_ratio = 0.97;
    /// A hold rung whose retention is this close to `hold_ratio` is measured
    /// a second time before the decision stands.
    const borderline_band = 0.02;

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
    borderline_used: bool = false,
    probed_down: bool = false,
    generation: u64 = 0,
    last_backoff_generation: u64 = std.math.maxInt(u64),

    fn init(configured: Parallelism, pinned_feasible_width: usize) SourceReadWidthController {
        const configured_max = @min(configured.maximum(), pinned_feasible_width);
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
        if (climb_sample and improved) {
            if (self.index < self.max_index) return self.moveTo(self.index + 1);
            return self.hold(self.index);
        }

        const hold_index = self.holdIndex();
        if (!self.borderline_used and self.isBorderline(hold_index)) {
            self.borderline_used = true;
            return self.moveTo(hold_index);
        }
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

    fn isBorderline(self: *const SourceReadWidthController, index: usize) bool {
        if (index == self.best_index) return false;
        const retention = self.rates[index].? / self.rates[self.best_index].?;
        return @abs(retention - hold_ratio) <= borderline_band;
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
    );
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());
    try std.testing.expect(adaptive.blindGrow() == null);
    // At the clip the first window holds the only rung it can use.
    _ = adaptive.observe(sourceReadTestEvidence(&adaptive, 100));
    try std.testing.expectEqual(SourceReadWidthController.State.holding, adaptive.state);
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());

    const fixed = SourceReadWidthController.init(.{ .fixed = 20 }, 7);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expectEqual(SourceReadWidthController.State.holding, fixed.state);

    const configured_initial = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 48, .maximum = 128 } },
        128,
    );
    try std.testing.expectEqual(@as(usize, 48), configured_initial.width());
}

test "source read evidence requires enough concurrency and duration" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
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
    // below the start rung gets a value below the 3% band.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 8, .rate = 19.90 },
        .{ .width = 12, .rate = 21.33 },
        .{ .width = 16, .rate = 20.69 },
        .{ .width = 24, .rate = 18.90 },
        .{ .width = 32, .rate = 17.33 },
    });
    // 12, 16 (0.970 of 12: the climb stops), the downward probe of 8.
    try std.testing.expectEqual(@as(usize, 3), windows);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expect(controller.probed_down);
    try std.testing.expect(!controller.borderline_used);
    try std.testing.expectEqual(@as(u8, 0), controller.samples[SourceReadWidthController.widthIndexAtMost(24)]);
    // Holding: further evidence and blind growth change nothing.
    const held = controller.observe(sourceReadTestEvidence(&controller, 30));
    try std.testing.expectEqual(@as(usize, 12), held.width);
    try std.testing.expectEqual(controller.generation, held.generation);
    try std.testing.expect(controller.blindGrow() == null);
}

test "source read controller re-measures a borderline hold rung once" {
    // 8 retains 0.975 of 12 on the first window and 0.960 on the mean of two,
    // so the second window moves the hold to 12.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    const width8 = SourceReadWidthController.widthIndexAtMost(8);
    _ = controller.observe(sourceReadTestEvidence(&controller, 100));
    _ = controller.observe(sourceReadTestEvidence(&controller, 101));
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 97.5));
    try std.testing.expect(controller.borderline_used);
    try std.testing.expectEqual(SourceReadWidthController.State.climbing, controller.state);
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 94.5));
    try std.testing.expectEqual(@as(u8, 2), controller.samples[width8]);
    try std.testing.expectEqual(SourceReadWidthController.State.holding, controller.state);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
}

test "source read controller holds at the lowest rung within 3% on a flat curve" {
    // Real AWS shape: 16 MiB requests plateau near 950 MiB/s from 16 up.
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 12, .rate = 900 },
        .{ .width = 16, .rate = 940 },
        .{ .width = 24, .rate = 948 },
        .{ .width = 32, .rate = 950 },
        .{ .width = 48, .rate = 950 },
    });
    // 12, 16 (better by 4.4%), 24 (not better by 3%): hold at 16, the lowest
    // rung within 3% of it; 12 at 0.957 is below the band.
    try std.testing.expectEqual(@as(usize, 3), windows);
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    try std.testing.expectEqual(@as(usize, 16), read_width_ladder[controller.best_index]);
    try std.testing.expect(!controller.probed_down);
}

test "source read controller probes below the start rung once" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    const windows = try replaySourceReadCurve(&controller, &.{
        .{ .width = 8, .rate = 105 },
        .{ .width = 12, .rate = 100 },
        .{ .width = 16, .rate = 100 },
    });
    // 12, 16 (flat), 8 (better by 5%): hold 8 without climbing further down.
    try std.testing.expectEqual(@as(usize, 3), windows);
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    try std.testing.expectEqual(@as(usize, 8), read_width_ladder[controller.best_index]);

    var from_one = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 1, .maximum = 128 } },
        128,
    );
    // Nothing below the lowest rung to probe.
    _ = try replaySourceReadCurve(&from_one, &.{
        .{ .width = 1, .rate = 100 },
        .{ .width = 2, .rate = 100 },
    });
    try std.testing.expectEqual(@as(usize, 1), from_one.width());
    try std.testing.expect(!from_one.probed_down);
}

test "source read controller backs off once per generation of fresh admissions" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
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
    );
    try std.testing.expectEqual(@as(usize, 1), floor.backoff(false).?.width);
}

test "source read controller keeps a fixed width" {
    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.backoff(true) == null);
    try std.testing.expect(fixed.blindGrow() == null);
    const observed = fixed.observe(sourceReadTestEvidence(&fixed, 100));
    try std.testing.expectEqual(@as(usize, 7), observed.width);
    try std.testing.expectEqual(@as(u64, 0), fixed.generation);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
}

const ReadStatsCursor = struct {
    provider: VFS.ReadStatsProvider,
    previous: VFS.ReadStats,

    fn takeBackpressure(self: *ReadStatsCursor) bool {
        const current = self.provider.snapshot();
        const delta = current.sub(self.previous);
        self.previous = current;
        return delta.retries != 0 or delta.transient_retries != 0 or
            delta.timeouts != 0 or delta.server_failures != 0 or
            delta.throttles != 0;
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

    fake.stats.retries = 2;
    fake.stats.throttles = 1;
    try std.testing.expect(cursor.takeBackpressure());
    try std.testing.expect(!cursor.takeBackpressure());
}

test "source measurement rejects another controller generation" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    var runtime: SourceReadRuntime = undefined;
    runtime.controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
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

/// Worker tasks are spawned on demand: enough to fill the lifecycle gate at
/// the current width, never more than the configured maximum. Idle tasks
/// above the gate limit only add scheduling noise (128 persistent workers
/// cost about 7% on one MI300X while the controller held width 12).
const WorkerPool = struct {
    loader: *DirectLoader,
    maximum: usize,
    mutex: std.Io.Mutex = .init,
    spawned: usize = 0,

    fn ensure(self: *WorkerPool, io: std.Io, wanted: usize) void {
        const target = @min(wanted, self.maximum);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.spawned < target) : (self.spawned += 1) {
            self.loader.worker_group.concurrent(io, DirectLoader.workerMain, .{self.loader}) catch |err| {
                load_log.err("cannot spawn source worker {d}: {}", .{ self.spawned + 1, err });
                return;
            };
        }
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

    fn takeRemoteBackpressure(self: *SourceReadRuntime) bool {
        const cursor = if (self.read_stats) |*value| value else return false;
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
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width);
        std.debug.assert(limits.read > 0);
        self.reported_width = decision.width;
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        if (self.workers) |pool| pool.ensure(io, limits.lifecycle);
        // Advance the diagnostic baseline at the generation boundary.
        _ = self.takeRemoteBackpressure();
        self.metrics.prepareProbe(io, decision.generation, self.next_read_admission.load(.acquire));
        self.clock.reset();
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
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width);
        self.reported_width = decision.width;
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        if (self.workers) |pool| pool.ensure(io, limits.lifecycle);
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
        // Do not charge a rung for prior-generation DMA drain before its
        // first source admission can begin.
        if (probe.probe_first_read_ns == 0) return null;
        const evidence: SourceReadWidthController.Evidence = .{
            .completed_requests = @intCast(probe.probe_read_operations),
            .elapsed_ns = self.clock.busyNs(probe.probe_first_read_ns, now_ns),
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

    fn finalize(self: *SourceReadRuntime, io: std.Io) void {
        std.debug.assert(self.read_gate.inUse(io) == 0);
        _ = self.takeRemoteBackpressure();
        self.metrics.clearProbe(io);
    }

    fn awakeNs(io: std.Io) u64 {
        return @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
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

            if (self.takeRemoteBackpressure()) {
                // A read admitted under the current generation has begun
                // once the window fenced at its start saw a read.
                const fresh_admissions = self.metrics.snapshot(io).probe_peak_reads != 0;
                if (self.controller.backoff(fresh_admissions)) |decision| {
                    load_log.debug("source width backoff: generation={d}, width={d}, fresh_admissions={}", .{
                        decision.generation,
                        decision.width,
                        fresh_admissions,
                    });
                    self.applyDecision(io, decision);
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
            self.clock.tick(now_ns, idle and probe.probe_first_read_ns != 0);
            if (idle) continue;
            const evidence = self.evidenceFrom(probe, now_ns) orelse continue;
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

/// The direct DMA backend. Submissions and awaits come from one task at a
/// time; the workers, pump and controller run concurrently with them.
pub const DirectLoader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    load_profile: VFS.LoadProfile,
    progress: ?*std.Progress.Node,
    dma_resources: *DmaPlatformSettings,
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

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: LoaderOptions,
    ) !*DirectLoader {
        if (platform.devices.len == 0 or platform.devices.len > 64)
            return error.DmaDeviceMismatch;
        const resources = try acquirePlatformDmaSettings(platform);
        errdefer releasePlatformDmaSettings(platform);
        const config = resources.config;
        if (config.block_size > max_load_read_request_size or
            config.max_mapped_bytes < max_load_read_request_size)
            return error.InvalidDmaLoadConfig;

        const request_size = try effectiveSourceRequestSize(
            opts.load_profile.read_chunk_size,
            config.block_size,
        );
        const maximum_blocks_per_job = try maximumCoalescedJobBlocks(
            request_size,
            config.block_size,
        );
        const node_reserves = try allocator.alloc(usize, resources.workspace.pools.len);
        defer allocator.free(node_reserves);
        @memset(node_reserves, 0);
        for (platform.devices, 0..) |_, device_index| {
            const node_index = resources.workspace.device_pool_indices[device_index];
            node_reserves[node_index] = try std.math.add(
                usize,
                node_reserves[node_index],
                config.max_in_flight_per_device,
            );
        }
        // Calibration already mapped the working set for a 16 MiB request;
        // this grows the remainder for larger request sizes (HF profiles)
        // so no pinned slab grows inside a scored window. The arenas stay
        // with the platform for later loaders.
        const pregrowth_started: std.Io.Timestamp = .now(io, .awake);
        const retained_before = resources.retainedMappedBytes();
        try resources.ensureSourceWorkingSet(
            maximum_blocks_per_job,
            DmaPlatformSettings.preallocated_source_width,
            node_reserves,
        );
        const pregrown_bytes = resources.retainedMappedBytes() - retained_before;
        const pregrowth_ns: u64 = @intCast(@max(pregrowth_started.untilNow(io, .awake).nanoseconds, 0));
        // The per-node reserves are deliberately non-materialized. They keep
        // enough mapped-budget capacity available for devices that join a
        // later submission without paying their allocation cost at init.
        var pool = try mem.DmaBlockPool.initFromProvider(
            allocator,
            resources.workspace.blockPoolArenaProvider(),
            config.block_size,
            config.max_mapped_bytes,
            node_reserves,
        );
        var pool_moved = false;
        errdefer if (!pool_moved) pool.deinit();
        const aggregate_width = try pool.aggregatePotentialRequestWidth(maximum_blocks_per_job);
        const strict_width = try pool.minimumStrictAffinityRequestWidth(maximum_blocks_per_job);
        const strict_affinity = for (config.device_numa_nodes) |node| {
            if (node != null) break true;
        } else false;
        const feasible_width = if (strict_affinity)
            @min(aggregate_width, strict_width)
        else
            aggregate_width;
        if (feasible_width == 0) return error.DmaMappedBudgetExceeded;

        const source_parallelism = opts.read_parallelism;
        const controller = SourceReadWidthController.init(source_parallelism, feasible_width);
        const limits: RequestGateLimits = .init(controller.width(), feasible_width);
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
            config.block_size,
            resources.workspace.device_pool_indices,
            strict_affinity,
            &self.metrics,
            &self.scheduler,
            config.max_in_flight_per_device,
        );
        errdefer self.pipeline.deinit();

        self.worker_pool = .{ .loader = self, .maximum = source_parallelism.maximum() };
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
            .reported_width = controller.width(),
        };
        // Born busy: both gates are open at the controller's width, the
        // first window is fenced before any worker can admit a read, and the
        // initial workers are spawned by the decision that opened the gates.
        errdefer self.stopWorkers();
        self.workers_started = true;
        self.controller_runtime.start(io);
        try self.startController();
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, dma_block_size={Bi:.2}, workers={d}, max_workers={d}, feasible_width={d}, retained={Bi:.2}, pregrown={Bi:.2}, pregrowth_ms={d:.3}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            request_size,
            config.block_size,
            self.worker_pool.spawned,
            self.worker_pool.maximum,
            feasible_width,
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

    fn workerMain(self: *DirectLoader) void {
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
            if (!self.scheduler.waitForWork(self.io)) return;
            if (self.pipeline.failed()) return;
            if (!self.request_gate.acquire(self.io)) return;
            const claim = self.scheduler.claim(self.io) orelse {
                self.request_gate.release(self.io);
                continue;
            };
            self.pipeline.reserveSourceJob();
            const request = self.pipeline.registerRequest(claim.batch) catch |err| {
                self.pipeline.abandonSourceJob();
                self.request_gate.release(self.io);
                // The job is claimed, so `fail` will not retire its unit;
                // release it here (last access to the batch) before failing.
                claim.batch.finishJobs(1);
                self.pipeline.recordError(err);
                return;
            };
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

    /// Plans and publishes `specs` as one batch behind every earlier one.
    /// Work may start before this returns. The batch owns its items until
    /// `awaitBatch` retires it; nothing is published when this fails.
    pub fn submit(self: *DirectLoader, specs: []const LoadSpec) !*Batch {
        try self.checkOpen();
        const items = try self.allocator.alloc(*LoaderLoadItem, specs.len);
        errdefer self.allocator.free(items);
        var initialized: usize = 0;
        errdefer for (items[0..initialized]) |item| item.deinit(self.allocator);
        var logical_bytes: usize = 0;
        for (specs, items) |spec, *item| {
            item.* = try self.createItem(spec.source, spec.shape, spec.sharding, spec.output);
            initialized += 1;
            logical_bytes = try std.math.add(usize, logical_bytes, spec.source.shape.byteSize());
        }
        const planning_started: std.Io.Timestamp = .now(self.io, .awake);
        var plan = try FairVectoredReadScheduler.prepareBatch(
            self.allocator,
            self.platform.devices.len,
            items,
            self.dma_resources.config.block_size,
            self.source_request_size,
        );
        const planning_elapsed = planning_started.untilNow(self.io, .awake);
        const batch = Batch.create(self.allocator, self.io, &plan, .{
            .sequence = self.batch_count,
            .logical_bytes = logical_bytes,
            .source_bytes = plan.source_bytes,
            .source_jobs = plan.jobs.len,
            .source_runs = plan.source_runs,
            .source_items = items.len,
            .planned_transfers = plan.transfers.len,
            .planning_ns = @intCast(@max(planning_elapsed.nanoseconds, 0)),
            .published_at = .now(self.io, .awake),
            .source_stats = if (self.load_profile.stats) |provider| provider.snapshot() else null,
        }) catch |err| {
            plan.deinit();
            return err;
        };
        errdefer batch.destroy();
        try self.scheduler.publish(self.io, batch);
        batch.items = items;
        self.batch_count += 1;
        // The plan is visible: drop the publish sentinel. A batch without
        // jobs completes right here.
        batch.finishJobs(1);
        return batch;
    }

    /// Waits for the batch's last completion unit, retires it and returns
    /// the loader's sticky error if the pipeline failed. Targets the failure
    /// left unsubmitted are marked so their buffers never report ready.
    pub fn awaitBatch(self: *DirectLoader, batch: *Batch) !void {
        batch.done.waitUncancelable(self.io);
        const done_at: std.Io.Timestamp = .now(self.io, .awake);
        // Every request of this batch has completed: no worker or callback
        // touches its managers or contexts any more.
        const load_error = self.pipeline.errorValue();
        if (load_error != null) {
            for (batch.items) |item| {
                const state = item.state.readyValue() orelse continue;
                for (state.targets) |*target| {
                    if (!target.final_submitted) {
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
        const first_claim_at = diagnostics.first_claim_at orelse published_at;
        const average_read_size = if (diagnostics.source_jobs == 0)
            0
        else
            diagnostics.source_bytes / diagnostics.source_jobs;
        const coalescing_ratio = if (diagnostics.source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(diagnostics.source_items)) /
                @as(f64, @floatFromInt(diagnostics.source_jobs));
        load_log.debug("batch completed: batch={d}, successful={}, logical_bytes={Bi:.2}, planned_source_bytes={Bi:.2}, published=+{d:.3}s, first_claim=+{d:.3}s, done=+{d:.3}s, elapsed={d:.3}s, planning_elapsed={d:.3}s, planned_source_jobs={d}, source_runs={d}, source_items={d}, planned_transfers={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}, selected_source_width={d}, request_size={Bi:.2}, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}", .{
            diagnostics.sequence,
            successful,
            diagnostics.logical_bytes,
            diagnostics.source_bytes,
            secondsBetween(self.created_at, published_at),
            secondsBetween(self.created_at, first_claim_at),
            secondsBetween(self.created_at, done_at),
            secondsBetween(published_at, done_at),
            @as(f64, @floatFromInt(diagnostics.planning_ns)) / std.time.ns_per_s,
            diagnostics.source_jobs,
            diagnostics.source_runs,
            diagnostics.source_items,
            diagnostics.planned_transfers,
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
        load_log.debug("loader summary: batches={d}, successful={}, bytes_loaded={Bi:.2}, elapsed={d:.3}s, reads={d}, physical_source_calls={d}, tensor_transfer_pieces={d}, dma_submissions={d}, selected_source_width={d}, gate_closed_ticks={d}, request_size={Bi:.2}, pinned_high_water={Bi:.2}, pinned_mapped={Bi:.2}", .{
            self.batch_count,
            !self.pipeline.failed(),
            self.bytesLoaded(),
            secondsBetween(self.created_at, .now(self.io, .awake)),
            self.metrics.read_operations.load(.acquire),
            self.metrics.source_calls.load(.acquire),
            self.metrics.transfer_pieces.load(.acquire),
            self.metrics.dma_submissions.load(.acquire),
            self.controller_runtime.reported_width,
            self.controller_runtime.gate_closed_ticks,
            self.source_request_size,
            self.pool.highWaterBytes(),
            self.pool.mappedBytes(),
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
            releasePlatformDmaSettings(self.platform);
            self.cleaned = true;
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }
};

test "loader DMA admission rotates and respects per-device limits" {
    const all_ready: u64 = 0b1111;
    var active = [_]usize{ 0, 0, 0, 0 };
    var next_device: usize = 0;

    for ([_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 }) |expected| {
        const selected = selectLoaderDmaDevice(
            &active,
            8,
            all_ready,
            next_device,
        ).?;
        try std.testing.expectEqual(expected, selected);
        next_device = (selected + 1) % active.len;
    }

    active = .{ 8, 0, 8, 0 };
    try std.testing.expectEqual(
        @as(?usize, 1),
        selectLoaderDmaDevice(&active, 8, all_ready, 0),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        selectLoaderDmaDevice(&active, 8, 0b0101, 0),
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

    const shared_numa = [_]?usize{null} ** 8;
    const config: DmaLoadConfig = .{
        .device_numa_nodes = &shared_numa,
        .block_size = 16 * 1024 * 1024,
        .max_in_flight_per_device = 1,
        .max_mapped_bytes = 1024 * 1024 * 1024,
    };
    // Eight device feed blocks dominate the two blocks required by a request.
    // The obsolete writer-boundary formula incorrectly required nine blocks.
    try std.testing.expectEqual(
        @as(usize, 8 * 16 * 1024 * 1024),
        try requiredDmaWorkspaceBytes(config),
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
    metrics.recordProbeRead(io, 7, 41, max_load_read_request_size);
    const admitted = metrics.snapshot(io);
    try std.testing.expect(admitted.probe_first_read_ns != 0);
    try std.testing.expectEqual(@as(usize, 1), admitted.probe_active_reads);
    try std.testing.expectEqual(@as(u64, 1), admitted.probe_read_operations);
    try std.testing.expectEqual(@as(u64, max_load_read_request_size), admitted.probe_read_bytes);
    metrics.endRead(io, 6, 40);
    metrics.endRead(io, 7, 40);
    const draining = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 1), draining.probe_active_reads);
    metrics.endRead(io, 7, 41);
    const drained = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 0), drained.probe_active_reads);
    metrics.clearProbe(io);
}

test "partial source jobs contribute adaptive evidence" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 3, 1);
    metrics.beginRead(io, 3, 1);
    metrics.recordProbeRead(io, 3, 1, 256 * 1024);
    metrics.endRead(io, 3, 1);
    const snapshot = metrics.snapshot(io);
    try std.testing.expectEqual(@as(u64, 1), snapshot.probe_read_operations);
    try std.testing.expectEqual(@as(u64, 256 * 1024), snapshot.probe_read_bytes);
}

test "request lifecycle gate permits one shared spare request" {
    const normal: RequestGateLimits = .init(12, 64);
    try std.testing.expectEqual(@as(usize, 12), normal.read);
    try std.testing.expectEqual(@as(usize, 13), normal.lifecycle);

    const clipped: RequestGateLimits = .init(32, 32);
    try std.testing.expectEqual(@as(usize, 32), clipped.read);
    try std.testing.expectEqual(@as(usize, 32), clipped.lifecycle);
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
    for ([_]f64{ 100, 100, 90 }) |rate| {
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
    // 12 -> 16 (not better) -> the downward probe of 8 (10% below) -> hold 12.
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
    );
    runtime.metrics = &metrics;
    runtime.clock = .{};
    metrics.prepareProbe(io, runtime.controller.generation, 1);
    // No admission yet: nothing to score however long the window has been open.
    try std.testing.expect(runtime.currentEvidence(io, std.math.maxInt(u64)) == null);
    for (1..13) |admission| {
        metrics.beginRead(io, runtime.controller.generation, admission);
    }
    for (1..13) |admission| {
        metrics.recordProbeRead(io, runtime.controller.generation, admission, max_load_read_request_size);
        metrics.endRead(io, runtime.controller.generation, admission);
    }
    const first_read_ns = metrics.snapshot(io).probe_first_read_ns;
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
    try std.testing.expectEqual(@as(usize, 12), evidence.exercised_width);
    try std.testing.expectEqual(@as(u64, 12 * max_load_read_request_size), evidence.bytes);
}

test "vectored final transfers wait for every prior destination submission" {
    var targets = [_]VectoredTensorTransfer.Target{
        .{ .manager = undefined, .device_index = 0, .total = 100 },
        .{ .manager = undefined, .device_index = 1, .total = 100 },
    };
    var block: VectoredLoadPipeline.BlockContext = undefined;
    var pipeline: VectoredLoadPipeline = undefined;
    var final: VectoredLoadPipeline.ReadyTransfer = .{
        .target = &targets[0],
        .block = &block,
        .source_offset = 0,
        .destination_offset = 80,
        .len = 20,
    };

    try std.testing.expect(!pipeline.transferReady(final));
    final.target = &targets[1];
    targets[1].submitted_bytes.store(80, .release);
    try std.testing.expect(pipeline.transferReady(final));
    final.target = &targets[0];
    targets[0].submitted_bytes.store(60, .release);
    try std.testing.expect(!pipeline.transferReady(final));
    _ = targets[0].submitted_bytes.fetchAdd(20, .release);
    try std.testing.expect(pipeline.transferReady(final));

    const non_final: VectoredLoadPipeline.ReadyTransfer = .{
        .target = &targets[0],
        .block = &block,
        .source_offset = 0,
        .destination_offset = 20,
        .len = 20,
    };
    targets[0].submitted_bytes.store(0, .release);
    try std.testing.expect(pipeline.transferReady(non_final));
}

/// A single-device pipeline without PJRT: enough for request, block and
/// batch lifecycle tests. Must not move after `init`.
const TestPipeline = struct {
    metrics: VectoredLoadMetrics = .{},
    gate: AdaptiveRequestGate,
    queues: [1]std.ArrayListUnmanaged(VectoredLoadPipeline.ReadyTransfer) = .{.empty},
    active: [1]usize = .{0},
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
            .active_by_device = &self.active,
            .dma_limit = 1,
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
        return self.pipeline.registerRequest(claim.batch);
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
    try pipeline.reserveBlockCapacity(batch, 1);
    const block = try pipeline.registerBlock(request, leased[0], 1);
    var target: VectoredTensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 64 };
    try fixture.queues[0].append(allocator, .{
        .target = &target,
        .block = block,
        .source_offset = 0,
        .destination_offset = 0,
        .len = 64,
    });
    pipeline.ready_entries = 1;
    pipeline.active_events = 1;
    fixture.active[0] = 1;
    request.finishScheduling();
    try std.testing.expect(!batch.done.isSet());
    pipeline.first_error.store(@intFromError(error.Unknown), .release);

    pipeline.eventCompleted(0);
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
    try std.testing.expectEqual(@as(usize, 3), batch.requests.items.len);

    fixture.pipeline.retireBatch(batch);
    batch.destroy();
}

test "batch completes when a claimed job is abandoned before its request" {
    const io = std.testing.io;
    var scheduler: FairVectoredReadScheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(2, null, &scheduler);
    defer fixture.deinit();

    const batch = try publishTestBatch(&scheduler, 2);
    const request = try fixture.claimRequest(&scheduler);
    // The worker's `registerRequest` failure path: the claim holds the unit,
    // so the worker must release it itself.
    const abandoned = scheduler.claim(io).?;
    try std.testing.expect(fixture.gate.acquire(io));
    fixture.pipeline.reserveSourceJob();
    fixture.pipeline.abandonSourceJob();
    fixture.gate.release(io);
    try std.testing.expect(!batch.done.isSet());
    abandoned.batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());

    request.finishScheduling();
    try std.testing.expect(batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), fixture.metrics.pending_source_jobs.load(.acquire));
    fixture.pipeline.retireBatch(batch);
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
                if (claim.job.file_offset % 3 == 2) {
                    // The `registerRequest` failure path.
                    fixture_.pipeline.abandonSourceJob();
                    fixture_.gate.release(io_);
                    claim.batch.finishJobs(1);
                    continue;
                }
                const request = fixture_.pipeline.registerRequest(claim.batch) catch unreachable;
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
            var expected_requests: usize = 0;
            for (0..job_counts[index]) |job| {
                if (job % 3 != 2) expected_requests += 1;
            }
            try std.testing.expectEqual(expected_requests, batch.requests.items.len);
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
                }
            }
        }
        try std.testing.expectEqualSlices(u8, expected, actual);
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
