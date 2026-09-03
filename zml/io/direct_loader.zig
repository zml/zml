const std = @import("std");

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
const Exe = @import("../exe.zig").Exe;
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

fn effectiveSourceRequestSize(read_chunk_size: usize, dma_block_size: usize) !usize {
    if (read_chunk_size == 0 or read_chunk_size > max_load_read_request_size)
        return error.InvalidLoadProfile;
    const selected = @max(read_chunk_size, dma_block_size);
    if (selected > max_load_read_request_size) return error.InvalidLoadProfile;
    return selected;
}

const VectoredLoadMetrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    source_calls: std.atomic.Value(u64) = .init(0),
    transfer_pieces: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    outstanding_requests: std.atomic.Value(usize) = .init(0),
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

    fn beginRequest(self: *VectoredLoadMetrics) void {
        _ = self.outstanding_requests.fetchAdd(1, .acq_rel);
    }

    fn endRequest(self: *VectoredLoadMetrics) void {
        _ = self.outstanding_requests.fetchSub(1, .acq_rel);
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
    total: usize,
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
            .total = packed_shape.byteSize(),
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

const LoaderSourceSlot = struct {
    const uninitialized = 0;
    const initializing = 1;
    const ready = 2;
    const failed = 3;

    uri: []const u8,
    file: std.Io.File = undefined,
    status: std.atomic.Value(u8) = .init(uninitialized),
    error_code: std.atomic.Value(u16) = .init(0),
    initialized: std.Io.Event = .unset,

    fn ensure(self: *LoaderSourceSlot, io: std.Io) !std.Io.File {
        while (true) switch (self.status.load(.acquire)) {
            uninitialized => {
                if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                self.file = std.Io.Dir.openFile(.cwd(), io, self.uri, .{ .mode = .read_only }) catch |err| {
                    self.error_code.store(@intFromError(err), .release);
                    self.status.store(failed, .release);
                    self.initialized.set(io);
                    return err;
                };
                self.status.store(ready, .release);
                self.initialized.set(io);
                return self.file;
            },
            initializing => self.initialized.waitUncancelable(io),
            ready => return self.file,
            failed => return @errorFromInt(self.error_code.load(.acquire)),
            else => unreachable,
        };
    }

    fn deinit(self: *LoaderSourceSlot, io: std.Io) void {
        if (self.status.load(.acquire) == ready) self.file.close(io);
    }
};

const LoaderLoadItem = struct {
    const StateSlot = struct {
        const uninitialized = 0;
        const initializing = 1;
        const ready = 2;
        const failed = 3;

        state: VectoredTensorTransfer = undefined,
        status: std.atomic.Value(u8) = .init(uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(self: *StateSlot, item: *LoaderLoadItem, direct: *DirectLoader) !*VectoredTensorTransfer {
            while (true) switch (self.status.load(.acquire)) {
                uninitialized => {
                    if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                    self.state = VectoredTensorTransfer.initResolved(
                        direct.allocator,
                        direct.platform,
                        item.source,
                        item.shape,
                        item.sharding,
                        item.output,
                        direct.progress,
                    ) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(direct.io);
                        return err;
                    };
                    self.status.store(ready, .release);
                    self.initialized.set(direct.io);
                    return &self.state;
                },
                initializing => self.initialized.waitUncancelable(direct.io),
                ready => return &self.state,
                failed => return @errorFromInt(self.error_code.load(.acquire)),
                else => unreachable,
            };
        }

        fn deinit(self: *StateSlot) void {
            if (self.status.load(.acquire) == ready) self.state.deinit();
        }
    };

    source: *const safetensors.Tensor,
    source_slot: *LoaderSourceSlot,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
    state: StateSlot = .{},

    fn deinit(self: *LoaderLoadItem, allocator: std.mem.Allocator) void {
        self.state.deinit();
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

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
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

            self.pipeline.metrics.endRequest();
            self.completed.store(true, .release);
            self.pipeline.request_gate.release(self.pipeline.io);
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
    requests: std.ArrayListUnmanaged(*RequestContext) = .empty,
    blocks: std.ArrayListUnmanaged(*BlockContext) = .empty,
    ready_queues: []std.ArrayListUnmanaged(ReadyTransfer),
    events: std.ArrayListUnmanaged(*EventContext) = .empty,
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
        std.debug.assert(self.active_events == 0);
        std.debug.assert(self.ready_entries == 0);
        for (self.events.items) |ctx| {
            ctx.pjrt_event.deinit(self.platform.pjrt_api);
            if (ctx.err) |err| err.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
        }
        for (self.blocks.items) |block| {
            std.debug.assert(block.lease.isComplete());
            self.allocator.destroy(block);
        }
        for (self.requests.items) |request| {
            std.debug.assert(request.completed.load(.acquire));
            self.allocator.destroy(request);
        }
        for (self.ready_queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.ready_queues);
        self.allocator.free(self.active_by_device);
        self.events.deinit(self.allocator);
        self.blocks.deinit(self.allocator);
        self.requests.deinit(self.allocator);
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

    fn registerRequest(self: *VectoredLoadPipeline) !*RequestContext {
        const request = try self.allocator.create(RequestContext);
        errdefer self.allocator.destroy(request);
        request.* = .{
            .pipeline = self,
            .read_epoch = 0,
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.requests.append(self.allocator, request);
        self.metrics.beginRequest();
        return request;
    }

    fn reapCompleted(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        std.debug.assert(self.active_events == 0 and self.ready_entries == 0);
        for (self.events.items) |ctx| {
            ctx.pjrt_event.deinit(self.platform.pjrt_api);
            if (ctx.err) |err| err.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
        }
        for (self.blocks.items) |block| {
            std.debug.assert(block.lease.isComplete());
            self.allocator.destroy(block);
        }
        for (self.requests.items) |request| {
            std.debug.assert(request.completed.load(.acquire));
            self.allocator.destroy(request);
        }
        self.events.clearRetainingCapacity();
        self.blocks.clearRetainingCapacity();
        self.requests.clearRetainingCapacity();
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
        try self.blocks.append(self.allocator, block);
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
            const tensor = &transfer.item.state.state;
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

    fn reserveBlockCapacity(self: *VectoredLoadPipeline, count: usize) !void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.blocks.ensureUnusedCapacity(self.allocator, count);
    }

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

    fn submitOne(self: *VectoredLoadPipeline, transfer: ReadyTransfer) void {
        const is_last = transfer.destination_offset + transfer.len == transfer.target.total;
        const event = transfer.target.manager.transferData(
            self.platform.pjrt_api,
            0,
            transfer.block.lease.data[transfer.source_offset..][0..transfer.len],
            @intCast(transfer.destination_offset),
            is_last,
        ) catch |err| {
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
            return;
        };
        if (is_last) transfer.target.final_submitted = true;
        _ = transfer.target.submitted_bytes.fetchAdd(transfer.len, .release);

        const ctx = self.allocator.create(EventContext) catch {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
            return;
        };
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .device_index = transfer.target.device_index,
        };

        self.metadata_mutex.lockUncancelable(self.io);
        self.events.append(self.allocator, ctx) catch {
            self.metadata_mutex.unlock(self.io);
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
            return;
        };
        self.metadata_mutex.unlock(self.io);

        _ = self.metrics.dma_submissions.fetchAdd(1, .monotonic);
        event.onReady(self.platform.pjrt_api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                ctx_.err = err;
                if (err) |pjrt_error| {
                    ctx_.pipeline.recordError(pjrt_error.getCode(ctx_.pipeline.platform.pjrt_api).toApiError());
                }
                ctx_.block.complete();
                ctx_.pipeline.eventCompleted(ctx_.device_index);
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
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
            const tensor = transfer.item.state.ensure(transfer.item, direct) catch |err| {
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

        pipeline.reserveBlockCapacity(block_count) catch |err| {
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
        for (transfers) |transfer| transfer.item.state.state.recordReadProgress(transfer.len);
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

/// Immutable source-job epochs ordered once by destination-device debt.
/// Replicated jobs credit every device they serve but occur once in the order.
const FairVectoredReadScheduler = struct {
    const Job = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfers: []const VectoredLoadPipeline.PlannedTransfer = &.{},
    };

    const PlanningJob = struct {
        source_slot: *LoaderSourceSlot,
        file_offset: u64,
        len: usize,
        transfer_start: usize,
        transfer_len: usize,
        predecessor: ?usize,
    };

    const TestJob = struct {
        tensor_index: usize,
        len: usize,
        physical_bytes: []const usize,
        block_count: usize = 1,
    };

    const Snapshot = struct {
        remaining_jobs: usize,
        has_unscheduled: bool,
    };

    const PreparedBatch = struct {
        allocator: std.mem.Allocator,
        jobs: []Job,
        transfers: []VectoredLoadPipeline.PlannedTransfer,
        source_bytes: u64,
        source_runs: usize,

        fn deinit(self: *PreparedBatch) void {
            if (self.jobs.len != 0) self.allocator.free(self.jobs);
            if (self.transfers.len != 0) self.allocator.free(self.transfers);
            self.* = undefined;
        }
    };

    worker_count: usize,
    plan: ?PreparedBatch = null,
    cursor: std.atomic.Value(usize) = .init(0),
    waiting_workers: usize = 0,
    stopping: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn init(worker_count: usize) FairVectoredReadScheduler {
        return .{ .worker_count = worker_count };
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
            if (writer_mask == 0) return error.InvalidLoaderJob;
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

    fn publish(self: *FairVectoredReadScheduler, io: std.Io, batch: *PreparedBatch) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.LoaderShuttingDown;
        if (self.plan != null) return error.LoaderEpochActive;
        self.plan = batch.*;
        batch.* = undefined;
        self.cursor.store(0, .release);
        self.condition.broadcast(io);
    }

    fn finishEpoch(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and self.waiting_workers != self.worker_count) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.plan) |*plan| plan.deinit();
        self.plan = null;
        self.cursor.store(0, .release);
    }

    fn stop(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
    }

    fn fail(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.plan) |plan| self.cursor.store(plan.jobs.len, .release);
        self.stopping = true;
        self.condition.broadcast(io);
    }

    fn initForTest(
        allocator: std.mem.Allocator,
        device_count: usize,
        test_jobs: []const TestJob,
    ) !FairVectoredReadScheduler {
        if (device_count == 0 or device_count > 64) return error.DmaDeviceMismatch;
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        errdefer allocator.free(queues);
        @memset(queues, .empty);
        errdefer for (queues) |*queue| queue.deinit(allocator);
        const planning_jobs = try allocator.alloc(PlanningJob, test_jobs.len);
        defer allocator.free(planning_jobs);
        const physical_bytes = try allocator.alloc(usize, test_jobs.len * device_count);
        defer allocator.free(physical_bytes);
        var previous_jobs = try allocator.alloc(?usize, test_jobs.len);
        defer allocator.free(previous_jobs);
        @memset(previous_jobs, null);
        var source_bytes: u64 = 0;
        for (test_jobs, planning_jobs, 0..) |job, *stored, job_index| {
            if (job.physical_bytes.len != device_count or job.block_count == 0)
                return error.InvalidTestJob;
            stored.* = .{
                .source_slot = undefined,
                .file_offset = 0,
                .len = job.len,
                .transfer_start = 0,
                .transfer_len = 0,
                .predecessor = if (job.tensor_index < previous_jobs.len)
                    previous_jobs[job.tensor_index]
                else
                    null,
            };
            if (job.tensor_index < previous_jobs.len)
                previous_jobs[job.tensor_index] = job_index;
            @memcpy(
                physical_bytes[job_index * device_count ..][0..device_count],
                job.physical_bytes,
            );
            var destinations: usize = 0;
            for (job.physical_bytes, queues) |bytes, *queue| {
                if (bytes == 0) continue;
                try queue.append(allocator, job_index);
                destinations += 1;
            }
            if (destinations == 0) return error.InvalidTestJob;
            source_bytes +|= @intCast(job.len);
        }
        const order = try fairOrder(allocator, planning_jobs, physical_bytes, queues);
        defer allocator.free(order);
        const jobs = try allocator.alloc(Job, test_jobs.len);
        errdefer allocator.free(jobs);
        for (jobs, order) |*stored, planning_index| {
            const test_job = test_jobs[planning_index];
            stored.* = .{
                .source_slot = undefined,
                .file_offset = test_job.tensor_index,
                .len = test_job.len,
            };
        }
        for (queues) |*queue| queue.deinit(allocator);
        allocator.free(queues);
        return .{
            .worker_count = 0,
            .plan = .{
                .allocator = allocator,
                .jobs = jobs,
                .transfers = &.{},
                .source_bytes = source_bytes,
                .source_runs = test_jobs.len,
            },
        };
    }

    fn deinit(self: *FairVectoredReadScheduler) void {
        if (self.plan) |*plan| plan.deinit();
        self.* = undefined;
    }

    fn claim(self: *FairVectoredReadScheduler, io: std.Io) ?Job {
        const plan = if (self.plan) |*value| value else return null;
        const position = self.cursor.fetchAdd(1, .monotonic);
        if (position >= plan.jobs.len) return null;
        if (position + 1 == plan.jobs.len) {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.condition.broadcast(io);
        }
        return plan.jobs[position];
    }

    fn waitExhausted(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.plan) |plan| {
            if (self.cursor.load(.acquire) >= plan.jobs.len) return;
            self.condition.waitUncancelable(io, &self.mutex);
        }
    }

    fn waitForWork(self: *FairVectoredReadScheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and (self.plan == null or
            self.cursor.load(.acquire) >= self.plan.?.jobs.len))
        {
            self.waiting_workers += 1;
            self.condition.broadcast(io);
            self.condition.waitUncancelable(io, &self.mutex);
            self.waiting_workers -= 1;
        }
        return !self.stopping;
    }

    fn snapshot(self: *FairVectoredReadScheduler, io: std.Io) Snapshot {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const plan = self.plan orelse return .{
            .remaining_jobs = 0,
            .has_unscheduled = false,
        };
        const position = @min(self.cursor.load(.acquire), plan.jobs.len);
        return .{
            .remaining_jobs = plan.jobs.len - position,
            .has_unscheduled = position != plan.jobs.len,
        };
    }
};

test "fair read scheduler rotates sharded devices by scheduled bytes" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .len = 10, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 3, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(u64, 0), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 2), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 1), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 3), scheduler.claim(io).?.file_offset);
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

test "fair read scheduler preserves per-tensor request order" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 1, 0 } },
        .{ .tensor_index = 0, .len = 2, .physical_bytes = &.{ 0, 2 } },
        .{ .tensor_index = 1, .len = 3, .physical_bytes = &.{ 0, 3 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.len);
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.len);
    try std.testing.expectEqual(@as(usize, 3), scheduler.claim(io).?.len);
}

test "fair read scheduler claims a replicated job once and credits every replica" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 20, .physical_bytes = &.{ 20, 20 } },
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(u64, 0), scheduler.claim(io).?.file_offset);
    // The replicated entry is skipped in device 1's queue; tie rotation gives
    // that device the next scheduling turn.
    try std.testing.expectEqual(@as(u64, 2), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 1), scheduler.claim(io).?.file_offset);
    try std.testing.expect(scheduler.claim(io) == null);
}

test "fair read scheduler compares physical bytes rather than scheduling turns" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 1, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 2, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 3, .len = 10, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 4, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(u64, 0), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 3), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 1), scheduler.claim(io).?.file_offset);
    // Device 0 receives another turn because it has 8 scheduled bytes while
    // device 1 has 10; a turn-count scheduler would alternate here.
    try std.testing.expectEqual(@as(u64, 2), scheduler.claim(io).?.file_offset);
    try std.testing.expectEqual(@as(u64, 4), scheduler.claim(io).?.file_offset);
}

test "fair read scheduler tracks fixed jobs and tails" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = max_load_read_request_size, .physical_bytes = &.{max_load_read_request_size} },
        .{ .tensor_index = 1, .len = max_load_read_request_size, .physical_bytes = &.{max_load_read_request_size} },
        .{ .tensor_index = 2, .len = 7, .physical_bytes = &.{7} },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 1, &jobs);
    defer scheduler.deinit();
    const initial = scheduler.snapshot(std.testing.io);
    try std.testing.expectEqual(@as(usize, 3), initial.remaining_jobs);
    try std.testing.expectEqual(@as(usize, 3), initial.remaining_jobs);
    _ = scheduler.claim(std.testing.io).?;
    const after = scheduler.snapshot(std.testing.io);
    try std.testing.expectEqual(@as(usize, 2), after.remaining_jobs);
    try std.testing.expectEqual(@as(usize, 2), after.remaining_jobs);
}

test "fair read scheduler exhaustion waits for the final claim" {
    const io = std.testing.io;
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 1, &.{
        .{ .tensor_index = 0, .len = 8, .physical_bytes = &.{8} },
        .{ .tensor_index = 1, .len = 8, .physical_bytes = &.{8} },
    });
    defer scheduler.deinit();

    var exhausted: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(scheduler_: *FairVectoredReadScheduler, io_: std.Io, exhausted_: *std.Io.Event) void {
            scheduler_.waitExhausted(io_);
            exhausted_.set(io_);
        }
    }.run, .{ &scheduler, io, &exhausted });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!exhausted.isSet());

    _ = scheduler.claim(io).?;
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!exhausted.isSet());
    _ = scheduler.claim(io).?;
    try group.await(io);
    try std.testing.expect(exhausted.isSet());
}

test "fair read scheduler concurrent claims return every logical job once" {
    var job_storage: [32]FairVectoredReadScheduler.TestJob = undefined;
    for (&job_storage, 0..) |*job, index| job.* = .{
        .tensor_index = index,
        .len = 1,
        .physical_bytes = if (index % 3 == 0) &.{ 1, 1 } else if (index % 2 == 0) &.{ 1, 0 } else &.{ 0, 1 },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &job_storage);
    defer scheduler.deinit();
    var seen: std.atomic.Value(u64) = .init(0);
    var claim_count: std.atomic.Value(usize) = .init(0);
    var duplicate: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    for (0..8) |_| try group.concurrent(std.testing.io, struct {
        fn run(
            scheduler_: *FairVectoredReadScheduler,
            seen_: *std.atomic.Value(u64),
            claim_count_: *std.atomic.Value(usize),
            duplicate_: *std.atomic.Value(bool),
        ) void {
            while (scheduler_.claim(std.testing.io)) |job| {
                const mask = @as(u64, 1) << @intCast(job.file_offset);
                if (seen_.fetchOr(mask, .acq_rel) & mask != 0) duplicate_.store(true, .release);
                _ = claim_count_.fetchAdd(1, .monotonic);
            }
        }
    }.run, .{ &scheduler, &seen, &claim_count, &duplicate });
    try group.await(std.testing.io);
    try std.testing.expectEqual(std.math.maxInt(u32), @as(u32, @truncate(seen.load(.acquire))));
    try std.testing.expectEqual(job_storage.len, claim_count.load(.acquire));
    try std.testing.expect(!duplicate.load(.acquire));
}

test "fair read scheduler validates jobs and cleans up allocation failures" {
    const wrong_width = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{1} },
    };
    try std.testing.expectError(
        error.InvalidTestJob,
        FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &wrong_width),
    );
    const no_destination = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 0, 0 } },
    };
    try std.testing.expectError(
        error.InvalidTestJob,
        FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &no_destination),
    );

    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const jobs = [_]FairVectoredReadScheduler.TestJob{
                .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 1, 1 } },
                .{ .tensor_index = 1, .len = 1, .physical_bytes = &.{ 1, 0 } },
            };
            var scheduler = try FairVectoredReadScheduler.initForTest(allocator, 2, &jobs);
            defer scheduler.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "fair scheduler rejects a second immutable epoch plan" {
    const io = std.testing.io;
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &.{
        .{ .tensor_index = 0, .len = 10, .physical_bytes = &.{ 10, 0 } },
    });
    defer scheduler.deinit();
    var pending = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &.{
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 0, 10 } },
    });
    defer pending.deinit();
    var plan = pending.plan.?;
    pending.plan = null;
    defer plan.deinit();
    try std.testing.expectError(error.LoaderEpochActive, scheduler.publish(io, &plan));
}

test "fair scheduler publishes a new plan only after the epoch barrier" {
    const io = std.testing.io;
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &.{
        .{ .tensor_index = 0, .len = 8, .physical_bytes = &.{ 8, 0 } },
        .{ .tensor_index = 1, .len = 8, .physical_bytes = &.{ 0, 8 } },
    });
    defer scheduler.deinit();
    _ = scheduler.claim(io).?;
    _ = scheduler.claim(io).?;
    scheduler.finishEpoch(io);
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_jobs);

    var next = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &.{
        .{ .tensor_index = 2, .len = 4, .physical_bytes = &.{ 0, 4 } },
    });
    defer next.deinit();
    var plan = next.plan.?;
    next.plan = null;
    try scheduler.publish(io, &plan);
    try std.testing.expectEqual(@as(u64, 2), scheduler.claim(io).?.file_offset);
}

const read_width_ladder = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

/// Source-only adaptive state. DMA width and request size never enter its
/// evidence or decisions.
const SourceReadWidthController = struct {
    const Phase = enum { ramp_up, refine_down, settled };

    const Evidence = struct {
        completed_requests: usize,
        elapsed_ns: u64,
        bytes: u64,
        exercised_width: usize,
        remaining_full_jobs: usize,

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

    const Decision = struct {
        width: usize,
        generation: u64,
        changed: bool = false,
        settled: bool = false,
    };

    const Confirmation = struct {
        index: usize,
        resume_phase: Phase,
        resume_index: usize,
        prior_selected_index: usize,
    };

    fixed_width: ?usize = null,
    maximum_index: usize,
    current_index: usize,
    selected_index: usize,
    peak_index: usize,
    generation: u64 = 0,
    phase: Phase,
    rates: [read_width_ladder.len]?f64 = @splat(null),
    ramp_scores: usize = 0,
    unchanged_candidates: usize = 0,
    confirmation: ?Confirmation = null,
    confirmation_used: bool = false,

    fn init(configured: Parallelism, pinned_feasible_width: usize) SourceReadWidthController {
        const configured_max = @min(configured.maximum(), pinned_feasible_width);
        var maximum_index: usize = 0;
        for (read_width_ladder, 0..) |candidate_width, index| {
            if (candidate_width > configured_max) break;
            maximum_index = index;
        }
        if (!configured.isAdaptive()) {
            const fixed = @min(configured.initial(), pinned_feasible_width);
            const fixed_index = widthIndexAtMost(fixed);
            return .{
                .fixed_width = @max(@as(usize, 1), fixed),
                .maximum_index = fixed_index,
                .current_index = fixed_index,
                .selected_index = fixed_index,
                .peak_index = fixed_index,
                .phase = .settled,
            };
        }
        const initial_index = @min(widthIndexAtMost(configured.initial()), maximum_index);
        return .{
            .maximum_index = maximum_index,
            .current_index = initial_index,
            .selected_index = initial_index,
            .peak_index = initial_index,
            .phase = .ramp_up,
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
        return self.fixed_width orelse read_width_ladder[self.current_index];
    }

    fn isAdaptive(self: *const SourceReadWidthController) bool {
        return self.fixed_width == null;
    }

    fn selectedWidth(self: *const SourceReadWidthController) usize {
        return self.fixed_width orelse read_width_ladder[self.selected_index];
    }

    fn currentDecision(self: *const SourceReadWidthController) Decision {
        return .{
            .width = self.width(),
            .generation = self.generation,
            .settled = self.phase == .settled,
        };
    }

    fn probeCost(index: usize) usize {
        const candidate_width = read_width_ladder[index];
        return candidate_width +| @max(@as(usize, 8), candidate_width);
    }

    fn probeFitsTail(index: usize, remaining_full_jobs: usize) bool {
        return probeCost(index) *| 4 <= remaining_full_jobs;
    }

    fn restartFitsTail(self: *const SourceReadWidthController, remaining_full_jobs: usize) bool {
        if (self.phase == .settled) return true;
        return probeFitsTail(self.current_index, remaining_full_jobs);
    }

    fn blindGrow(
        self: *SourceReadWidthController,
        remaining_full_jobs: usize,
    ) ?Decision {
        if (!self.isAdaptive() or self.phase != .ramp_up or self.ramp_scores != 0 or
            self.current_index >= self.maximum_index)
            return null;
        const ceiling: usize = if (self.width() < 24) 24 else if (self.width() < 32) 32 else return null;
        const target = @min(widthIndexAtMost(ceiling), self.maximum_index);
        if (target <= self.current_index or !probeFitsTail(target, remaining_full_jobs)) return null;
        return self.changeTo(target);
    }

    fn observe(self: *SourceReadWidthController, evidence: Evidence) Decision {
        if (!self.isAdaptive() or self.phase == .settled)
            return self.currentDecision();
        std.debug.assert(evidence.scoreable(self.width()));
        const rate = evidence.bytesPerSecond();
        return self.finishScore(self.current_index, rate, evidence.remaining_full_jobs);
    }

    fn finishScore(
        self: *SourceReadWidthController,
        index: usize,
        rate: f64,
        remaining_full_jobs: usize,
    ) Decision {
        if (self.confirmation) |confirmation| {
            std.debug.assert(index == confirmation.index);
            self.rates[index] = ((self.rates[index] orelse rate) + rate) / 2;
            self.confirmation = null;
            self.recomputePeakAndSelection();
            self.phase = confirmation.resume_phase;
            return self.advanceAfterScore(
                confirmation.resume_index,
                confirmation.prior_selected_index,
                remaining_full_jobs,
            );
        }

        const prior_selected = self.selected_index;
        self.rates[index] = rate;
        self.recomputePeakAndSelection();
        const peak_rate = self.rates[self.peak_index] orelse rate;
        const confirmation_candidate: ?usize = blk: {
            for ([_]usize{ index, prior_selected, self.selected_index }) |candidate| {
                if (candidate == self.peak_index) continue;
                const candidate_rate = self.rates[candidate] orelse continue;
                const retention = if (peak_rate == 0) 0 else candidate_rate / peak_rate;
                const adjacent = candidate + 1 == self.peak_index or
                    self.peak_index + 1 == candidate;
                if (adjacent and @abs(retention - 0.97) <= 0.02) break :blk candidate;
            }
            break :blk null;
        };
        if (!self.confirmation_used and confirmation_candidate != null and
            probeFitsTail(confirmation_candidate.?, remaining_full_jobs))
        {
            self.confirmation_used = true;
            self.confirmation = .{
                .index = confirmation_candidate.?,
                .resume_phase = self.phase,
                .resume_index = index,
                .prior_selected_index = prior_selected,
            };
            return self.restartAt(confirmation_candidate.?);
        }

        return self.advanceAfterScore(index, prior_selected, remaining_full_jobs);
    }

    fn advanceAfterScore(
        self: *SourceReadWidthController,
        index: usize,
        prior_selected: usize,
        remaining_full_jobs: usize,
    ) Decision {
        const peak_rate = self.rates[self.peak_index] orelse 0;
        const index_rate = self.rates[index] orelse 0;
        const retention = if (peak_rate == 0) 0 else index_rate / peak_rate;
        return switch (self.phase) {
            .ramp_up => blk: {
                self.ramp_scores += 1;
                if (self.ramp_scores > 1) {
                    if (self.selected_index == prior_selected)
                        self.unchanged_candidates += 1
                    else
                        self.unchanged_candidates = 0;
                }
                if ((self.ramp_scores > 1 and self.unchanged_candidates >= 2) or
                    index == self.maximum_index)
                    break :blk self.beginRefineOrSettle(remaining_full_jobs);
                if (probeFitsTail(index + 1, remaining_full_jobs))
                    break :blk self.changeTo(index + 1);
                break :blk self.beginRefineOrSettle(remaining_full_jobs);
            },
            .refine_down => blk: {
                if (retention < 0.97 or index == 0 or
                    !probeFitsTail(index - 1, remaining_full_jobs))
                    break :blk self.settle();
                break :blk self.changeTo(index - 1);
            },
            else => self.currentDecision(),
        };
    }

    fn recomputePeakAndSelection(self: *SourceReadWidthController) void {
        var peak_index = self.peak_index;
        var peak_rate: f64 = self.rates[peak_index] orelse 0;
        for (self.rates, 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate > peak_rate) {
                peak_rate = rate;
                peak_index = index;
            }
        }
        self.peak_index = peak_index;
        var selected = peak_index;
        for (self.rates, 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate >= peak_rate * 0.97) {
                selected = index;
                break;
            }
        }
        self.selected_index = selected;
    }

    fn beginRefineOrSettle(
        self: *SourceReadWidthController,
        remaining_full_jobs: usize,
    ) Decision {
        self.phase = .refine_down;
        if (self.selected_index > 0 and probeFitsTail(self.selected_index - 1, remaining_full_jobs))
            return self.changeTo(self.selected_index - 1);
        return self.settle();
    }

    fn changeTo(self: *SourceReadWidthController, index: usize) Decision {
        const changed = index != self.current_index;
        self.current_index = index;
        if (changed) self.generation +|= 1;
        return .{
            .width = self.width(),
            .generation = self.generation,
            .changed = changed,
            .settled = self.phase == .settled,
        };
    }

    fn restartAt(self: *SourceReadWidthController, index: usize) Decision {
        self.current_index = index;
        self.generation +|= 1;
        return .{
            .width = self.width(),
            .generation = self.generation,
            .changed = true,
            .settled = self.phase == .settled,
        };
    }

    fn settle(self: *SourceReadWidthController) Decision {
        self.phase = .settled;
        self.confirmation = null;
        return self.changeTo(self.selected_index);
    }

    fn rollbackTail(self: *SourceReadWidthController) Decision {
        if (self.phase == .settled) return self.currentDecision();
        if (self.confirmation) |confirmation|
            self.selected_index = confirmation.prior_selected_index;
        return self.settle();
    }

    fn backoff(self: *SourceReadWidthController) Decision {
        if (!self.isAdaptive()) return self.currentDecision();
        if (self.current_index == 0) {
            self.selected_index = 0;
            return self.settle();
        }

        // Never retain a width above the last clean selection after the source
        // reports pressure. Further feedback can keep walking a settled
        // controller down one rung at a time.
        self.selected_index = @min(self.current_index - 1, self.selected_index);
        self.peak_index = self.selected_index;
        self.phase = .settled;
        return self.changeTo(self.selected_index);
    }
};

fn sourceReadTestEvidence(
    controller: *const SourceReadWidthController,
    rate: u64,
    remaining_full_jobs: usize,
) SourceReadWidthController.Evidence {
    return .{
        .completed_requests = @max(@as(usize, 8), controller.width()),
        .elapsed_ns = std.time.ns_per_s,
        .bytes = rate,
        .exercised_width = controller.width(),
        .remaining_full_jobs = remaining_full_jobs,
    };
}

test "source read controller bounds blind growth at 32" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expectEqual(@as(usize, 24), controller.blindGrow(10_000).?.width);
    try std.testing.expectEqual(@as(usize, 32), controller.blindGrow(10_000).?.width);
    try std.testing.expect(controller.blindGrow(10_000) == null);
}

test "source read controller clips infeasible adaptive and fixed widths" {
    var adaptive = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        10,
    );
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());
    try std.testing.expect(adaptive.blindGrow(10_000) == null);

    const fixed = SourceReadWidthController.init(.{ .fixed = 20 }, 7);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.currentDecision().settled);

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
    var short = sourceReadTestEvidence(&controller, 100, 10_000);
    short.elapsed_ns = 99 * std.time.ns_per_ms;
    try std.testing.expect(!short.scoreable(controller.width()));
    var unexercised = sourceReadTestEvidence(&controller, 100, 10_000);
    unexercised.exercised_width -= 1;
    try std.testing.expect(!unexercised.scoreable(controller.width()));
    try std.testing.expectEqual(
        @as(usize, 16),
        controller.observe(sourceReadTestEvidence(&controller, 100, 10_000)).width,
    );
}

test "source read controller selects plateau then refines downward" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 100_000));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 94, 100_000));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 94, 100_000));
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    const settled = controller.observe(sourceReadTestEvidence(&controller, 80, 100_000));
    try std.testing.expect(settled.settled);
    try std.testing.expectEqual(@as(usize, 12), controller.selectedWidth());
}

test "source read controller confirms an adjacent boundary once" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.widthIndexAtMost(12), controller.confirmation.?.index);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    try std.testing.expect(controller.confirmation == null);
    try std.testing.expect(controller.confirmation_used);
    try std.testing.expectEqual(@as(usize, 24), controller.width());
}

test "source read controller confirms a borderline candidate in place" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 96, 1_000_000));
    const width16 = SourceReadWidthController.widthIndexAtMost(16);
    try std.testing.expectEqual(width16, controller.confirmation.?.index);
    const generation = controller.generation;
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    try std.testing.expect(generation > 1);
}

test "source read controller rolls back an unfinished confirmation" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    try std.testing.expect(controller.confirmation != null);
    const rollback = controller.rollbackTail();
    try std.testing.expect(rollback.settled);
    try std.testing.expectEqual(@as(usize, 12), rollback.width);
}

test "source read controller charges one unfinished confirmation on restart" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    const remaining_cost = SourceReadWidthController.probeCost(controller.confirmation.?.index);
    try std.testing.expect(controller.restartFitsTail(remaining_cost * 4));
    try std.testing.expect(!controller.restartFitsTail(remaining_cost * 4 - 1));
}

test "source read controller refines downward when an upward tail no longer fits" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 100_000));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const downward = controller.observe(sourceReadTestEvidence(&controller, 120, 100));
    try std.testing.expectEqual(SourceReadWidthController.Phase.refine_down, controller.phase);
    try std.testing.expectEqual(@as(usize, 12), downward.width);
}

test "source read controller rejects dirty probe restarts at a short tail" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    try std.testing.expect(controller.restartFitsTail(96));
    try std.testing.expect(!controller.restartFitsTail(95));
}

test "source read controller keeps fixed width and rolls back a short tail" {
    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.currentDecision().settled);

    var adaptive = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    const tail = adaptive.observe(sourceReadTestEvidence(&adaptive, 100, 10));
    try std.testing.expect(tail.settled);
    try std.testing.expectEqual(@as(usize, 12), tail.width);
}

test "source read controller backs off before and after convergence" {
    var probing = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    try std.testing.expectEqual(@as(usize, 24), probing.blindGrow(10_000).?.width);
    const probe_backoff = probing.backoff();
    try std.testing.expect(probe_backoff.changed);
    try std.testing.expect(probe_backoff.settled);
    try std.testing.expectEqual(@as(usize, 12), probe_backoff.width);

    const settled_backoff = probing.backoff();
    try std.testing.expect(settled_backoff.changed);
    try std.testing.expect(settled_backoff.settled);
    try std.testing.expectEqual(@as(usize, 8), settled_backoff.width);

    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64);
    const fixed_backoff = fixed.backoff();
    try std.testing.expect(!fixed_backoff.changed);
    try std.testing.expectEqual(@as(usize, 7), fixed_backoff.width);
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
    metrics.prepareProbe(io, runtime.controller.generation + 1, 1);
    try std.testing.expect(runtime.currentEvidence(io, 1_000) == null);
}

const SourceReadRuntime = struct {
    const Measurement = union(enum) {
        inactive,
        transitioning: usize,
        measuring,
        scoring: SourceReadWidthController.Evidence,
        blind,
    };

    controller: SourceReadWidthController,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    metrics: *VectoredLoadMetrics,
    next_read_admission: *std.atomic.Value(u64),
    scheduler: *FairVectoredReadScheduler,
    pinned_feasible_width: usize,
    read_stats: ?ReadStatsCursor,
    source_bootstrap_enabled: bool,
    source_response_observed: bool = false,
    measurement: Measurement = .inactive,
    last_blind_growth_ns: u64 = 0,
    scheduler_idle: bool = false,
    backoff_admission_start: ?u64 = null,
    reported_width: usize = 1,
    epoch_barrier_requested: std.atomic.Value(bool) = .init(false),
    epoch_barrier_done: std.Io.Event = .unset,
    control: std.Io.Event = .unset,
    done: std.Io.Event = .unset,

    fn takeRemoteBackpressure(self: *SourceReadRuntime) bool {
        const cursor = if (self.read_stats) |*value| value else return false;
        return cursor.takeBackpressure();
    }

    fn applyDecision(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
        force_probe: bool,
    ) void {
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width);
        self.reported_width = decision.width;
        self.request_gate.setLimit(io, limits.lifecycle);
        if (decision.changed or (!decision.settled and force_probe)) {
            self.read_gate.setLimit(io, 0);
            self.metrics.clearProbe(io);
            self.measurement = .{ .transitioning = limits.read };
            _ = self.activatePendingProbe(io);
        } else if (decision.settled) {
            self.read_gate.setLimit(io, limits.read);
            self.metrics.clearProbe(io);
            self.metrics.config_epoch.store(decision.generation, .release);
            self.measurement = .inactive;
        }
    }

    fn activatePendingProbe(self: *SourceReadRuntime, io: std.Io) bool {
        const read_limit = switch (self.measurement) {
            .transitioning => |limit| limit,
            else => return false,
        };
        if (self.read_gate.inUse(io) != 0) return false;
        // Advance the diagnostic baseline at a generation boundary.
        _ = self.takeRemoteBackpressure();
        const admission_start = self.next_read_admission.load(.acquire);
        if (self.controller.phase == .settled) {
            self.metrics.config_epoch.store(self.controller.generation, .release);
            self.backoff_admission_start = admission_start;
            self.measurement = .inactive;
        } else {
            self.metrics.prepareProbe(io, self.controller.generation, admission_start);
            self.measurement = .measuring;
        }
        self.read_gate.setLimit(io, read_limit);
        return true;
    }

    fn backoffReady(self: *SourceReadRuntime) bool {
        const boundary = self.backoff_admission_start orelse return true;
        if (self.next_read_admission.load(.acquire) <= boundary) return false;
        self.backoff_admission_start = null;
        return true;
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
        self.metrics.clearProbe(io);
        self.metrics.config_epoch.store(decision.generation, .release);
        self.measurement = .blind;
    }

    fn currentEvidence(
        self: *SourceReadRuntime,
        io: std.Io,
        remaining_full_jobs: usize,
    ) ?SourceReadWidthController.Evidence {
        const probe = self.metrics.snapshot(io);
        if (probe.probe_epoch != self.controller.generation) return null;
        const now_ns: u64 = @intCast(@max(
            std.Io.Timestamp.now(io, .awake).nanoseconds,
            1,
        ));
        const evidence: SourceReadWidthController.Evidence = .{
            .completed_requests = @intCast(probe.probe_read_operations),
            // Do not charge a candidate for prior-generation DMA drain before
            // its first source admission can begin.
            .elapsed_ns = if (probe.probe_first_read_ns == 0)
                0
            else
                now_ns -| probe.probe_first_read_ns,
            .bytes = probe.probe_read_bytes,
            .exercised_width = probe.probe_peak_reads,
            .remaining_full_jobs = remaining_full_jobs,
        };
        return if (evidence.scoreable(self.controller.width())) evidence else null;
    }

    fn finalize(self: *SourceReadRuntime, io: std.Io) void {
        std.debug.assert(self.read_gate.inUse(io) == 0);
        _ = self.takeRemoteBackpressure();
        self.metrics.clearProbe(io);
    }

    fn finishIdleMeasurement(self: *SourceReadRuntime, io: std.Io) void {
        _ = self.takeRemoteBackpressure();
        if (self.controller.phase != .settled) {
            switch (self.measurement) {
                .scoring => |pending| {
                    var evidence = pending;
                    evidence.remaining_full_jobs = std.math.maxInt(usize);
                    _ = self.controller.observe(evidence);
                },
                .measuring => if (self.currentEvidence(io, std.math.maxInt(usize))) |evidence| {
                    _ = self.controller.observe(evidence);
                },
                else => {},
            }
        }
        self.metrics.clearProbe(io);
        self.measurement = .inactive;
        self.reported_width = self.controller.selectedWidth();
    }

    fn epochBarrier(self: *SourceReadRuntime, io: std.Io) void {
        self.epoch_barrier_done.reset();
        self.epoch_barrier_requested.store(true, .release);
        self.control.set(io);
        self.epoch_barrier_done.waitUncancelable(io);
    }

    fn run(self: *SourceReadRuntime, io: std.Io) std.Io.Cancelable!void {
        const started: std.Io.Timestamp = .now(io, .awake);
        self.applyDecision(io, self.controller.currentDecision(), self.controller.isAdaptive());
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

            if (self.takeRemoteBackpressure()) {
                // Feedback collected while the old generation drains belongs
                // to that transition and must not trigger another rung.
                switch (self.measurement) {
                    .transitioning => continue,
                    else => {},
                }
                if (!self.backoffReady()) continue;
                self.applyDecision(io, self.controller.backoff(), false);
                continue;
            }
            if (self.metrics.read_bytes.load(.acquire) != 0) self.source_response_observed = true;
            const scheduler_snapshot = self.scheduler.snapshot(io);
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));

            const idle = !scheduler_snapshot.has_unscheduled and
                self.metrics.pending_source_jobs.load(.acquire) == 0 and
                self.read_gate.inUse(io) == 0;
            if (idle) {
                if (!self.scheduler_idle) {
                    self.finishIdleMeasurement(io);
                    self.scheduler_idle = true;
                }
                if (self.epoch_barrier_requested.swap(false, .acq_rel)) {
                    _ = self.takeRemoteBackpressure();
                    self.epoch_barrier_done.set(io);
                }
                continue;
            }
            if (self.scheduler_idle) {
                self.scheduler_idle = false;
                self.applyDecision(
                    io,
                    self.controller.currentDecision(),
                    self.controller.phase != .settled,
                );
                continue;
            }

            // Blind admissions deliberately overlap generations so a remote
            // source can ramp before its first response. Once any response is
            // visible, close the read gate and start a clean generation only
            // after every blind admission has returned.
            switch (self.measurement) {
                .blind => if (self.source_response_observed) {
                    self.controller.generation +|= 1;
                    var decision = self.controller.currentDecision();
                    decision.changed = true;
                    self.applyDecision(io, decision, true);
                    continue;
                },
                else => {},
            }

            if (!self.source_response_observed) {
                if (now_ns -| self.last_blind_growth_ns >= 10 * std.time.ns_per_ms and
                    shouldBootstrapSource(
                        self.source_bootstrap_enabled,
                        false,
                        self.metrics.read_bytes.load(.acquire),
                        self.metrics.outstanding_requests.load(.acquire),
                        self.controller.width(),
                        scheduler_snapshot.has_unscheduled,
                    ))
                {
                    self.last_blind_growth_ns = now_ns;
                    if (self.controller.blindGrow(scheduler_snapshot.remaining_jobs)) |decision| {
                        self.applyBlindGrowth(io, decision);
                    }
                }
                continue;
            }

            if (self.controller.phase == .settled) continue;

            // Hold a completed score until all calls admitted at that width
            // have drained, keeping generation attribution unambiguous.
            switch (self.measurement) {
                .scoring => |pending| {
                    if (self.read_gate.inUse(io) != 0) continue;
                    _ = self.takeRemoteBackpressure();
                    var evidence = pending;
                    evidence.remaining_full_jobs = scheduler_snapshot.remaining_jobs;
                    const decision = self.controller.observe(evidence);
                    self.measurement = .inactive;
                    self.applyDecision(io, decision, !decision.settled);
                    continue;
                },
                else => {},
            }

            switch (self.measurement) {
                .transitioning => {
                    _ = self.activatePendingProbe(io);
                    continue;
                },
                else => {},
            }

            switch (self.measurement) {
                .measuring => if (self.currentEvidence(
                    io,
                    scheduler_snapshot.remaining_jobs,
                )) |evidence| {
                    // Freeze a complete interval, then drain admissions that
                    // raced with the snapshot. Their bytes are excluded.
                    self.read_gate.setLimit(io, 0);
                    self.measurement = .{ .scoring = evidence };
                    continue;
                },
                else => {},
            }
        }
    }
};

fn shouldBootstrapSource(
    enabled: bool,
    response_observed: bool,
    read_bytes: u64,
    outstanding_requests: usize,
    read_limit: usize,
    has_unscheduled: bool,
) bool {
    return enabled and !response_observed and read_bytes == 0 and
        outstanding_requests >= read_limit and has_unscheduled;
}

const DirectLoaderDiagnostics = struct {
    logical_bytes: usize = 0,
    source_bytes: u64 = 0,
    source_jobs: usize = 0,
    source_runs: usize = 0,
    source_items: usize = 0,
    planned_transfers: usize = 0,
    planning_ns: u64 = 0,
    started_at: ?std.Io.Timestamp = null,
    epoch_number: usize = 0,
    logged_read_operations: u64 = 0,
    logged_source_calls: u64 = 0,
    logged_transfer_pieces: u64 = 0,
    logged_dma_submissions: u64 = 0,
    source_stats: ?VFS.ReadStats = null,

    fn resetEpoch(self: *DirectLoaderDiagnostics) void {
        self.logical_bytes = 0;
        self.source_bytes = 0;
        self.source_jobs = 0;
        self.source_runs = 0;
        self.source_items = 0;
        self.planned_transfers = 0;
        self.planning_ns = 0;
        self.started_at = null;
    }
};

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
    worker_group: std.Io.Group = .init,
    controller_group: std.Io.Group = .init,
    source_slots: std.ArrayListUnmanaged(*LoaderSourceSlot) = .empty,
    epoch_items: std.ArrayListUnmanaged(*LoaderLoadItem) = .empty,
    bytes_loaded: std.atomic.Value(usize) = .init(0),
    diagnostics: DirectLoaderDiagnostics = .{},
    epoch_active: bool = false,
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
        var scheduler = FairVectoredReadScheduler.init(source_parallelism.maximum());
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
            .source_request_size = request_size,
            .maximum_blocks_per_job = maximum_blocks_per_job,
            .effective_pinned_feasible_width = feasible_width,
            .diagnostics = .{
                .source_stats = if (read_stats) |cursor| cursor.previous else null,
            },
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

        self.controller_runtime = .{
            .controller = controller,
            .read_gate = &self.read_gate,
            .request_gate = &self.request_gate,
            .metrics = &self.metrics,
            .next_read_admission = &self.pipeline.next_read_admission,
            .scheduler = &self.scheduler,
            .pinned_feasible_width = feasible_width,
            .read_stats = read_stats,
            .source_bootstrap_enabled = opts.load_profile.high_latency,
        };
        errdefer self.stopWorkers();
        try self.startWorkers(source_parallelism.maximum());
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, dma_block_size={Bi:.2}, workers={d}, feasible_width={d}, mapped={Bi:.2}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            request_size,
            config.block_size,
            source_parallelism.maximum(),
            feasible_width,
            self.pool.mappedBytes(),
        });
        return self;
    }

    fn startWorkers(self: *DirectLoader, worker_count: usize) !void {
        try self.controller_group.concurrent(
            self.io,
            SourceReadRuntime.run,
            .{ &self.controller_runtime, self.io },
        );
        self.controller_started = true;
        self.workers_started = true;
        for (0..worker_count) |_| {
            self.worker_group.concurrent(self.io, workerMain, .{self}) catch |err| {
                self.stopWorkers();
                return err;
            };
        }
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
            const job = self.scheduler.claim(self.io) orelse {
                self.request_gate.release(self.io);
                continue;
            };
            self.pipeline.reserveSourceJob();
            const request = self.pipeline.registerRequest() catch |err| {
                self.pipeline.abandonSourceJob();
                self.request_gate.release(self.io);
                self.pipeline.recordError(err);
                return;
            };
            VectoredReadRequest.runCoalesced(
                request,
                job.source_slot,
                &self.pipeline,
                job.file_offset,
                job.len,
                job.transfers,
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
        if (self.epoch_active) return error.LoaderEpochActive;
    }

    fn sourceSlot(self: *DirectLoader, uri: []const u8) !*LoaderSourceSlot {
        for (self.source_slots.items) |slot| {
            if (std.mem.eql(u8, slot.uri, uri)) return slot;
        }
        const slot = try self.allocator.create(LoaderSourceSlot);
        errdefer self.allocator.destroy(slot);
        slot.* = .{ .uri = uri };
        try self.source_slots.append(self.allocator, slot);
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

    fn appendItems(self: *DirectLoader, items: []const *LoaderLoadItem) !void {
        try self.checkOpen();
        const planning_started: std.Io.Timestamp = .now(self.io, .awake);
        var batch = try FairVectoredReadScheduler.prepareBatch(
            self.allocator,
            self.platform.devices.len,
            items,
            self.dma_resources.config.block_size,
            self.source_request_size,
        );
        const planning_elapsed = planning_started.untilNow(self.io, .awake);
        var batch_owned = true;
        defer if (batch_owned) batch.deinit();
        const batch_transfer_count = batch.transfers.len;
        const batch_job_count = batch.jobs.len;
        const batch_source_bytes = batch.source_bytes;
        const batch_source_runs = batch.source_runs;
        try self.epoch_items.ensureUnusedCapacity(self.allocator, items.len);

        var logical_bytes: usize = 0;
        for (items) |item| {
            const placement = try item.sharding.placement(item.shape.packedShape());
            if (placement.shape.byteSize() != 0) {
                for (item.sharding.devicesInCanonicalOrder()) |device| {
                    if (device.id >= self.platform.devices.len) return error.DmaDeviceMismatch;
                }
            }
            logical_bytes = try std.math.add(usize, logical_bytes, item.source.shape.byteSize());
        }
        self.diagnostics.started_at = .now(self.io, .awake);
        if (self.load_profile.stats) |provider| {
            // Exclude aggregate backend traffic that happened while this
            // loader had no active epoch from the next diagnostic delta.
            self.diagnostics.source_stats = provider.snapshot();
        }
        try self.scheduler.publish(self.io, &batch);
        batch_owned = false;
        for (items) |item| self.epoch_items.appendAssumeCapacity(item);
        self.diagnostics.logical_bytes = logical_bytes;
        self.diagnostics.source_bytes = batch_source_bytes;
        self.diagnostics.source_jobs = batch_job_count;
        self.diagnostics.source_runs = batch_source_runs;
        self.diagnostics.source_items = items.len;
        self.diagnostics.planned_transfers = batch_transfer_count;
        self.diagnostics.planning_ns = @intCast(@max(planning_elapsed.nanoseconds, 0));
        self.epoch_active = true;
    }

    pub fn loadPrepared(self: *DirectLoader, specs: []const LoadSpec) !void {
        try self.checkOpen();
        var items: std.ArrayListUnmanaged(*LoaderLoadItem) = .empty;
        defer items.deinit(self.allocator);
        errdefer for (items.items) |item| item.deinit(self.allocator);
        try items.ensureTotalCapacityPrecise(self.allocator, specs.len);
        for (specs) |spec| {
            items.appendAssumeCapacity(try self.createItem(
                spec.source,
                spec.shape,
                spec.sharding,
                spec.output,
            ));
        }
        try self.appendItems(items.items);
        items.clearRetainingCapacity();
    }

    pub fn await(self: *DirectLoader) !void {
        if (self.cleaned) return error.LoaderShuttingDown;
        if (!self.epoch_active) {
            if (self.pipeline.errorValue()) |err| return err;
            return;
        }
        self.scheduler.waitExhausted(self.io);
        self.request_gate.waitEmpty(self.io);
        self.controller_runtime.epochBarrier(self.io);
        const load_error = self.pipeline.errorValue();
        if (load_error != null) {
            self.stopWorkers();
            for (self.epoch_items.items) |item| {
                if (item.state.status.load(.acquire) != LoaderLoadItem.StateSlot.ready) continue;
                for (item.state.state.targets) |*target| {
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
        self.scheduler.finishEpoch(self.io);
        self.pipeline.reapCompleted();
        for (self.epoch_items.items) |item| item.deinit(self.allocator);
        self.epoch_items.clearRetainingCapacity();
        self.logEpoch(load_error == null);
        if (load_error) |err| {
            self.diagnostics.resetEpoch();
            self.epoch_active = false;
            return err;
        }
        _ = self.bytes_loaded.fetchAdd(self.diagnostics.logical_bytes, .monotonic);
        self.diagnostics.resetEpoch();
        self.epoch_active = false;
    }

    fn logEpoch(self: *DirectLoader, successful: bool) void {
        const read_operations = self.metrics.read_operations.load(.acquire);
        const source_calls = self.metrics.source_calls.load(.acquire);
        const transfer_pieces = self.metrics.transfer_pieces.load(.acquire);
        const dma_submissions = self.metrics.dma_submissions.load(.acquire);
        const epoch_reads = read_operations -| self.diagnostics.logged_read_operations;
        const epoch_source_calls = source_calls -| self.diagnostics.logged_source_calls;
        const epoch_transfer_pieces = transfer_pieces -| self.diagnostics.logged_transfer_pieces;
        const epoch_dma = dma_submissions -| self.diagnostics.logged_dma_submissions;
        self.diagnostics.logged_read_operations = read_operations;
        self.diagnostics.logged_source_calls = source_calls;
        self.diagnostics.logged_transfer_pieces = transfer_pieces;
        self.diagnostics.logged_dma_submissions = dma_submissions;
        var source_requests: u64 = 0;
        var source_bytes: u64 = 0;
        var source_retries: u64 = 0;
        var source_throttles: u64 = 0;
        if (self.load_profile.stats) |provider| {
            const current = provider.snapshot();
            if (self.diagnostics.source_stats) |previous| {
                const delta = current.sub(previous);
                source_requests = delta.physical_requests;
                source_bytes = delta.physical_bytes;
                source_retries = delta.retries;
                source_throttles = delta.throttles;
            }
            self.diagnostics.source_stats = current;
        }
        const elapsed_seconds: f64 = if (self.diagnostics.started_at) |started|
            @as(f64, @floatFromInt(started.untilNow(self.io, .awake).nanoseconds)) /
                std.time.ns_per_s
        else
            0;
        const average_read_size = if (self.diagnostics.source_jobs == 0)
            0
        else
            self.diagnostics.source_bytes / self.diagnostics.source_jobs;
        const coalescing_ratio = if (self.diagnostics.source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(self.diagnostics.source_items)) /
                @as(f64, @floatFromInt(self.diagnostics.source_jobs));
        load_log.debug("epoch completed: epoch={d}, successful={}, logical_bytes={Bi:.2}, planned_source_bytes={Bi:.2}, elapsed={d:.3}s, planning_elapsed={d:.3}s, reads={d}, physical_source_calls={d}, planned_source_jobs={d}, source_runs={d}, source_items={d}, planned_transfers={d}, tensor_transfer_pieces={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}, selected_source_width={d}, request_size={Bi:.2}, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}, dma_submissions={d}, pinned_high_water={Bi:.2}, pinned_mapped={Bi:.2}", .{
            self.diagnostics.epoch_number,
            successful,
            self.diagnostics.logical_bytes,
            self.diagnostics.source_bytes,
            elapsed_seconds,
            @as(f64, @floatFromInt(self.diagnostics.planning_ns)) / std.time.ns_per_s,
            epoch_reads,
            epoch_source_calls,
            self.diagnostics.source_jobs,
            self.diagnostics.source_runs,
            self.diagnostics.source_items,
            self.diagnostics.planned_transfers,
            epoch_transfer_pieces,
            coalescing_ratio,
            average_read_size,
            self.controller_runtime.reported_width,
            self.source_request_size,
            source_requests,
            source_bytes,
            source_retries,
            source_throttles,
            epoch_dma,
            self.pool.highWaterBytes(),
            self.pool.mappedBytes(),
        });
        self.diagnostics.epoch_number += 1;
    }

    pub fn loadBinding(
        self: *DirectLoader,
        sources: []const *safetensors.Tensor,
        inputs: []Buffer,
        exe: *const Exe,
    ) !void {
        try self.checkOpen();
        const items = try self.allocator.alloc(*LoaderLoadItem, sources.len);
        defer self.allocator.free(items);
        var initialized: usize = 0;
        errdefer for (items[0..initialized]) |item| item.deinit(self.allocator);
        for (sources, exe.input_shapes, exe.input_shardings, inputs, items) |source, shape, sharding, *input, *item| {
            item.* = try self.createItem(source, shape, sharding, input);
            initialized += 1;
        }
        try self.appendItems(items);
        initialized = 0;
        try self.await();
    }

    pub fn bytesLoaded(self: *const DirectLoader) usize {
        return self.bytes_loaded.load(.acquire);
    }

    pub fn destroy(self: *DirectLoader) void {
        if (!self.cleaned) {
            if (self.epoch_active) self.await() catch {};
            self.stopWorkers();
            self.pipeline.reapCompleted();
            for (self.epoch_items.items) |item| item.deinit(self.allocator);
            self.epoch_items.deinit(self.allocator);
            for (self.source_slots.items) |slot| {
                slot.deinit(self.io);
                self.allocator.destroy(slot);
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
    try std.testing.expect(shouldBootstrapSource(true, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(false, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, true, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, false, 1, 12, 12, true));
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

test "probe source capacity counts active reads rather than retained requests" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 7, 10);
    for (0..48) |_| metrics.beginRequest();
    for (0..8) |index| metrics.beginRead(io, 7, 10 + @as(u64, @intCast(index)));

    try std.testing.expectEqual(@as(usize, 48), metrics.outstanding_requests.load(.acquire));
    const active = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 8), active.probe_peak_reads);
    try std.testing.expectEqual(@as(usize, 8), active.probe_active_reads);

    for (0..8) |index| metrics.endRead(io, 7, 10 + @as(u64, @intCast(index)));
    for (0..48) |_| metrics.endRequest();
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

test "settled backoff waits for a new-generation admission" {
    var next_admission: std.atomic.Value(u64) = .init(41);
    var runtime: SourceReadRuntime = undefined;
    runtime.next_read_admission = &next_admission;
    runtime.backoff_admission_start = 41;

    try std.testing.expect(!runtime.backoffReady());
    next_admission.store(42, .release);
    try std.testing.expect(runtime.backoffReady());
    try std.testing.expect(runtime.backoffReady());
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

test "late vectored callback failure drains and signals completion" {
    const io = std.testing.io;
    var queues = [_]std.ArrayListUnmanaged(VectoredLoadPipeline.ReadyTransfer){.empty};
    var active = [_]usize{1};
    var pipeline: VectoredLoadPipeline = .{
        .allocator = std.testing.allocator,
        .io = io,
        .platform = undefined,
        .pool = undefined,
        .read_gate = undefined,
        .request_gate = undefined,
        .block_size = 1,
        .device_pool_indices = &.{0},
        .numa_explicit = false,
        .metrics = undefined,
        .scheduler = undefined,
        .ready_queues = &queues,
        .active_by_device = &active,
        .dma_limit = 1,
        .active_events = 1,
    };
    pipeline.first_error.store(@intFromError(error.Unknown), .release);

    pipeline.eventCompleted(0);
    try std.testing.expectEqual(@as(usize, 0), pipeline.active_events);
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
