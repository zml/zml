//! Direct loading follows a batch from per-file planning through FIFO claims,
//! source reads and DMA completion. Runtime ownership and synchronization
//! live here. CUDA, ROCm, oneAPI, Furiosa and CPU load this way
//! (`backend.loadsDirect`).
//!
//! Sizing, fixed by `Loader.create`. Calibration picks the DMA block and the
//! per-device depth (dma_calibration.zig). A source request is the larger of
//! the checkpoint backend's minimum read (8 MiB for local files, 16 MiB over
//! https, s3 and gs, 32 MiB on hf) and the block: at most 32 MiB, and a
//! whole number of blocks. The source width is `read_parallelism`, or 16
//! for a local checkpoint and 32 for a high-latency one
//! (`backend.Options.readWidth`), clipped to what the mapped ceiling fits.
//! The workspace then maps, once and before the first read, `width + 1`
//! requests, the DMA reserve and the zero block, and cuts them into blocks
//! (`Workspace.carve`): nothing maps during a load. A read gate holds the
//! reads to the width. The lifecycle credits are the whole pinned set in
//! requests, so transfers that still hold blocks keep the DMA stage fed and
//! never want a block the reads did not leave free. The width never
//! changes: a source that answers 429 (or 503 on the object stores) is held
//! one layer down, where the VFS holds every request of that backend
//! (vfs/request.zig), so the traffic falls while the memory bounds stay.
//!
//! Planning and claims. `submit` sorts a submission's parts by file and
//! offset, plans one file at a time and publishes each plan at once, so the
//! workers read the first file while the next is planned. A plan merges
//! touching or overlapping ranges into reads, never across a gap or a file,
//! and cuts them into jobs of at most a request; jobs keep that file order,
//! and the `Scheduler` hands them out in submission order. A worker takes a
//! credit, claims a job, reads it into its slot's pinned blocks and queues
//! its preplanned transfer pieces on per-device pumps. A block may feed
//! several outputs or devices; the request keeps its credit until every
//! piece it queued completed or was abandoned. The last-transfer flag goes
//! on the piece that completes a destination's bytes, so reads and DMA
//! complete in any order. A batch owns its plans and callback contexts
//! until every job completes; a failure retires the unclaimed jobs and
//! drains what is in flight before the batch goes.
//!
//! Destinations. A part's source bytes map to its destination's logical
//! offsets, which the destination's `ShardGrid` places on the devices by
//! arithmetic. Pieces of one destination merge across parts when block,
//! writers and both offsets are contiguous, and its parts may come from
//! different files and plans. Bytes no part covers go out with the plan of
//! its first part, as a last fill job that reads nothing and transfers from
//! the zero block, which no credit owns. The job of a tensor sharded along
//! any axis but its first is staged by destination where that saves
//! submissions (`JobCut.placeDma`); Llama's plans take 1-4 ms, against
//! 73-112 ms when every row and device was a span. Strided parts are
//! gathered (`JobCut.appendGatherJobs`), their runs copied out of the
//! worker's bounce buffer by `Lattice.copyRuns`. Block-scale swizzles gather
//! this way, as runs of four bytes. Runs under 4 KiB cost host copies, where
//! a program costs a compile and staging memory: on DeepSeek-V4.1 over two
//! GB300, laying the MXFP4 experts out here (gate and up rows interleaved,
//! scales swizzled) copied 77 to 189 GiB through bounce buffers, 17 to 36 s
//! of worker time; as programs it copied 1 to 3 GiB and loaded 0.1 to 1.1 s
//! faster, for 4 to 6 GiB more device memory at the peak.

const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const pjrtx = @import("../pjrtx.zig");
const platform_mod = @import("../platform.zig");
const CreateOptions = platform_mod.CreateOptions;
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const backend = @import("backend.zig");
const BackendOptions = backend.Options;
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const ShardGrid = @import("ShardGrid.zig");
const Memory = @import("TensorStore.zig").Memory;
const dma_calibration = @import("dma_calibration.zig");
const host_memory = @import("host_memory.zig");
const load_log = @import("log.zig").load;
const load_testing = @import("testing.zig");

/// Creating a tensor's device state; `LazyOnce` replays it to every later
/// reader of the tensor.
const TransferInitError = std.mem.Allocator.Error || pjrt.ApiError || error{ IncompatibleSharding, UnsupportedMemoryKind };
/// Everything a worker, a pump callback or a partially published plan can
/// record as the loader's sticky error (`Loader.recordError`).
const StickyError = TransferInitError || std.Io.File.OpenError || safetensors.ExactPositionalError || error{ InvalidOptions, UnsupportedLayout, InvalidTensorRange, Internal, Closed };

/// Bounds per-device event overhead for tiny tensors. This is not a measured
/// optimum: 64 pieces smaller than block_size / 8 cannot fill an eight-block
/// byte budget. Raising the stage also costs host memory: replicated DeepSeek
/// on four GB300 improved 5.536 -> 5.024 s at depth 8 -> 32, but pinned
/// high-water grew 0.9-1.0 -> 2.36-2.50 GiB over six paired runs. That
/// tradeoff did not justify a larger default stage.
const max_dma_pieces_per_device: usize = 64;

/// The direct DMA backend. Submissions and awaits come from one task at a
/// time; the workers and the pumps run concurrently with them.
pub const Loader = struct {
    pub const AwaitError = StickyError;
    pub const SubmitError = StickyError;

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// The pinned arenas, grown once at creation and cut into `blocks`.
    workspace: host_memory.Workspace,
    /// The arenas cut into blocks: the zero block first, then the credits'
    /// slots (`Credits`). The zero block is the source of every DMA fill
    /// transfer, in no credit's slot, never written again.
    blocks: []host_memory.Block,
    scheduler: Scheduler,
    metrics: Metrics = .{},
    /// Source reads in flight, at most the load's width (`RequestGate.limit`):
    /// the configured width clipped to what the pinned set holds. A
    /// throttled source is the VFS's business, which holds every request of
    /// that backend (`vfs/request.zig`); the loader never changes its width.
    read_gate: RequestGate,
    /// Lifecycle credits, the pre-grown pinned capacity in requests, each a
    /// slot of the blocks one job spans at most (`Credits`). A request
    /// holds one from its claim to its last DMA callback, so the credits
    /// beyond the read width are the requests the DMA stage may hold. The
    /// capacity is the source working set of `width + 1` requests plus the
    /// calibrated DMA depth of every device, which is at least the width
    /// plus that depth because a request is a whole number of blocks. One
    /// credit beyond the width fed the DMA engines one request at a time:
    /// on a GB300 loading DeepSeek-V4 (pieces of 4 MiB and 256 KiB) that
    /// was 24.3 GiB/s at width 16 against 43.8 GiB/s with eight requests
    /// queued, and the workers spent 9 ms per request waiting for a credit
    /// against 3 ms reading.
    credits: Credits,
    /// The sticky first error (`recordError`); zero while the loader works.
    first_error: std.atomic.Value(u16) = .init(0),
    /// One per device (`Dma.DevicePump`). Where several are locked at once
    /// (`Dma.enqueueBlocks`, `Dma.retireBatch`) they are taken in device
    /// order; a pump, a completion and `Dma.abortReady` hold one at a time.
    pumps: []Dma.DevicePump,
    /// Per-device in-flight budget: the calibrated depth in blocks, in bytes.
    dma_budget_bytes: usize,
    worker_group: std.Io.Group = .init,
    /// One per source path. Its file opens at first use (during planning
    /// when a VFS decides direct IO for a local file, otherwise at the first
    /// read) and stays open until `destroy`. Keyed by the slot's own copy of
    /// the path: the registry that named it may go once the batch retired.
    source_slots: std.StringHashMapUnmanaged(*SourceSlot) = .empty,
    /// Device bytes of the outputs of the sealed batches, per
    /// `platform.devices` index, cumulative, as `TensorTransfer.init`
    /// allocates them; counted by the submitting task.
    submitted_bytes: []u64,
    /// Device bytes allocated for outputs so far, per `platform.devices`
    /// index, cumulative.
    allocated_bytes: []std.atomic.Value(u64),
    created_at: std.Io.Timestamp,
    /// Submissions so far; the next batch's sequence number.
    batch_count: usize = 0,
    /// The block size, the request size and the read alignment, for the
    /// whole load.
    plan_config: Planner.Config,
    /// See `Config.direct_io`.
    direct_io: VFS.DirectIo,

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: BackendOptions,
    ) !*Loader {
        // A destination's shards and a transfer's writers (a u64 mask) are
        // per device; Platform keeps devices past its maximum, with a warning.
        comptime std.debug.assert(Platform.MAX_NUM_DEVICES <= ShardGrid.max_boxes);
        if (platform.devices.len > Platform.MAX_NUM_DEVICES) {
            load_log.err("the direct loader addresses up to {d} devices, got {d}: UnsupportedPlatform", .{ Platform.MAX_NUM_DEVICES, platform.devices.len });
            return error.UnsupportedPlatform;
        }
        const self = try allocator.create(Loader);
        errdefer allocator.destroy(self);
        const submitted_bytes = try allocator.alloc(u64, platform.devices.len);
        errdefer allocator.free(submitted_bytes);
        @memset(submitted_bytes, 0);
        const allocated_bytes = try allocator.alloc(std.atomic.Value(u64), platform.devices.len);
        errdefer allocator.free(allocated_bytes);
        @memset(allocated_bytes, .init(0));

        var workspace: host_memory.Workspace = try .init(allocator, io, platform);
        errdefer workspace.deinit();
        // At debug when canceled: `Loader.deinit` cancels a set-up no `start`
        // waited for.
        const calibration = dma_calibration.calibrate(&workspace, platform, .{ .block_sizes = opts.dma_block_sizes }) catch |err| {
            load_log.failure(err, "calibrate DMA: target={t}, mapped_ceiling={d}", .{
                platform.target,
                workspace.max_mapped_bytes,
            });
            return err;
        };
        const block_size = calibration.block_size;
        // The larger of the profile's minimum read and the block. A request
        // is a whole number of blocks, so the pinned set is an exact number
        // of requests and the lifecycle credits are the capacity in
        // requests: `Loader.init` checked it for every block calibration may
        // select, and every shipped profile satisfies it (8, 16 and 32 MiB
        // chunks over power-of-two blocks).
        const request_size = @max(opts.load_profile.read_chunk_size, block_size);
        std.debug.assert(request_size <= backend.max_read_request_size and request_size % block_size == 0);
        const blocks_per_request = request_size / block_size;
        // The DMA stage of every device, kept mapped beside the source
        // working set; at least one maximal source request, so a single
        // request always fits beside it.
        const dma_reserve = @max(
            calibration.max_in_flight_per_device * platform.devices.len,
            Planner.blockCount(backend.max_read_request_size, block_size),
        );
        // The set is sized once, here: the width is what the mapped ceiling
        // fits, and the whole set is carved before the first read, so
        // nothing maps inside a load. One more block is mapped for the zero
        // fill, which the credits do not pay for.
        const zero_blocks = 1;
        const available = workspace.capacity(block_size);
        const width = @min(
            opts.readWidth(),
            (available -| dma_reserve -| zero_blocks) / blocks_per_request -| 1,
        );
        if (width == 0) {
            load_log.err("host memory ceiling {Bi:.2} fits no source request beside the DMA reserve: available_blocks={d}, reserve_blocks={d}, blocks_per_request={d}: OutOfMemory", .{ workspace.max_mapped_bytes, available, dma_reserve, blocks_per_request });
            return error.OutOfMemory;
        }
        if (width < opts.readWidth()) {
            load_log.debug("DMA source working set clipped by the mapped ceiling: width={d} of {d}", .{
                width,
                opts.readWidth(),
            });
        }
        // Two arenas, the reserve first: ROCm balances bytes across its host
        // nodes per allocation, and that split is recorded.
        const pregrowth_started: std.Io.Timestamp = .now(io, .awake);
        const retained_before = workspace.mapped_bytes;
        const blocks = workspace.carve(block_size, dma_reserve, (width + 1) * blocks_per_request + dma_reserve + zero_blocks) catch |err| {
            load_log.err("grow and carve host workspace: block_size={d}, source_width={d}, reserve_blocks={d}, mapped={d}, ceiling={d}: {s}", .{ block_size, width, dma_reserve, workspace.mapped_bytes, workspace.max_mapped_bytes, @errorName(err) });
            return err;
        };
        errdefer allocator.free(blocks);
        const pregrowth_ns: u64 = @intCast(@max(pregrowth_started.untilNow(io, .awake).nanoseconds, 0));
        load_log.debug("host workspace pregrown: retained={Bi:.2}, pregrown={Bi:.2}, pregrowth_ms={d:.3}", .{
            workspace.mapped_bytes,
            workspace.mapped_bytes - retained_before,
            @as(f64, @floatFromInt(pregrowth_ns)) / std.time.ns_per_ms,
        });
        std.debug.assert(blocks.len != 0);
        @memset(blocks[0], 0);
        var credits: Credits = try .init(allocator, blocks[1..], blocks_per_request);
        errdefer credits.deinit(allocator);
        // Every credit the pinned set holds is granted: the pre-growth
        // fitted `width + 1` requests beside the DMA reserve, so the stage
        // takes every block the reads leave free.
        const workers = width + 1;
        std.debug.assert(credits.count() >= workers);
        const pumps = try allocator.alloc(Dma.DevicePump, platform.devices.len);
        errdefer allocator.free(pumps);
        @memset(pumps, .{});

        const source_alignment = backend.readAlignment(io, opts.direct_io);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .workspace = workspace,
            .blocks = blocks,
            .scheduler = .init(allocator),
            .read_gate = .init(width),
            .credits = credits,
            .pumps = pumps,
            .dma_budget_bytes = calibration.max_in_flight_per_device * block_size,
            .submitted_bytes = submitted_bytes,
            .allocated_bytes = allocated_bytes,
            .created_at = .now(io, .awake),
            .plan_config = .{
                .block_size = block_size,
                .request_size = request_size,
                .alignment = source_alignment,
            },
            .direct_io = opts.direct_io,
        };
        std.debug.assert(self.dma_budget_bytes > 0);
        errdefer self.scheduler.deinit();
        _ = Planner.maximumJobLen(self.plan_config) catch |err| {
            load_log.err("invalid read alignment: alignment={d}, request_size={d}, block_size={d}: {s}", .{ self.plan_config.alignment, self.plan_config.request_size, self.plan_config.block_size, @errorName(err) });
            return err;
        };

        errdefer self.stopWorkers();
        for (0..workers) |_| try self.worker_group.concurrent(io, workerMain, .{self});
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, direct_io={t}, source_alignment={d}, dma_block_size={Bi:.2}, dma_budget_per_device={Bi:.2}, source_width={d}, lifecycle_credits={d}, workers={d}, retained={Bi:.2}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            request_size,
            opts.direct_io,
            source_alignment,
            block_size,
            self.dma_budget_bytes,
            width,
            self.credits.count(),
            workers,
            self.workspace.mapped_bytes,
        });
        return self;
    }

    /// Plans `specs` one source file at a time, publishes each file's plan
    /// as soon as it exists behind every earlier submission, then seals the
    /// batch: work on the first file starts while the rest is planned. The
    /// batch owns its destinations and parts until `awaitBatch` retires it.
    /// The specs keep the `LoadSpec` contract, which the front end checked;
    /// a strided part that does not fit one read fails before anything is
    /// published (`Planner.checkGatherReads`). Nothing is published when a
    /// failure precedes the first plan; a later planning
    /// failure fails the loader (a partial submission can never complete),
    /// the batch is awaited here and the caller sees only the error. The
    /// batch's host-resident destinations are then created ahead of their
    /// reads (`startAllocations`).
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: std.Progress.Node) SubmitError!*Batch {
        try self.checkOpen();
        const batch = batch: {
            const batch = try Batch.create(self.allocator, self.io, .{ .sequence = self.batch_count });
            errdefer batch.destroy();
            try self.createDestinations(batch, specs, progress);
            break :batch batch;
        };
        Planner.publishFiles(&self.scheduler, self.io, batch, batch.parts, self.plan_config, self.direct_io) catch |err| {
            if (batch.plans.items.len == 0) {
                load_log.debug("reject batch {d} before publication: source={s}: {s}", .{ batch.diagnostics.sequence, batch.planning_source, @errorName(err) });
                self.destroyBatch(batch);
                // An older batch that failed since `checkOpen` stopped the
                // scheduler, whose publish then says only `Closed`.
                return self.errorValue() orelse err;
            }
            // Planning failed after part of the batch was published: fail
            // the loader, seal and await this batch, and return the sticky
            // error instead of a batch the caller could not complete.
            self.recordError(err, "plan partially published batch {d}, source={s}, published_plans={d}", .{ batch.diagnostics.sequence, batch.planning_source, batch.plans.items.len });
            batch.diagnostics.sealed_at = .now(self.io, .awake);
            self.batch_count += 1;
            batch.finishJobs(1);
            self.awaitBatch(batch) catch |sticky| return sticky;
            return err;
        };
        // Stamped before the sentinel goes, so before the batch can complete.
        batch.diagnostics.sealed_at = .now(self.io, .awake);
        self.batch_count += 1;
        self.countSubmitted(batch);
        self.startAllocations(batch);
        // Every plan is visible: drop the publish sentinel. A batch without
        // jobs completes right here.
        batch.finishJobs(1);
        return batch;
    }

    /// Creates the batch's host-resident destinations ahead of their reads,
    /// in claim order, on a task of its own. A first-time pinned allocation
    /// runs at about 14-15 GiB/s on four B70 (oneAPI), serialized whatever
    /// the number of callers (one to sixteen threads measured alike; the
    /// time is the xe driver's VM_BIND), so on the workers' read path it
    /// stalled them for 10-12 ms per tensor; ahead of them it overlaps the
    /// reads. A worker that still finds its destination missing creates it
    /// itself (`LazyOnce`). The task holds one completion unit, so the batch
    /// outlives it; without a task the workers create everything.
    fn startAllocations(self: *Loader, batch: *Batch) void {
        var host_destinations: usize = 0;
        for (batch.destinations) |*destination| host_destinations += @intFromBool(destination.hostResident());
        if (host_destinations == 0) return;
        _ = batch.remaining.fetchAdd(1, .acq_rel);
        self.worker_group.concurrent(self.io, allocateMain, .{ self, batch, host_destinations }) catch |err| {
            load_log.debug("batch {d}: no allocation task, workers create its {d} host destinations: {s}", .{ batch.diagnostics.sequence, host_destinations, @errorName(err) });
            batch.finishJobs(1);
        };
    }

    /// Walks the host jobs in claim order (the plans in publish order, each
    /// plan's jobs in order) and their transfers as `runHost` does,
    /// skipping a destination already created. `ensured` also counts one a
    /// worker was still creating, which the task waited for.
    fn allocateMain(self: *Loader, batch: *Batch, host_destinations: usize) void {
        var ensured: usize = 0;
        var previous: ?*Destination = null;
        walk: for (batch.plans.items) |plan| for (plan.jobs) |job| {
            switch (job.kind) {
                .host_direct, .host_bounce, .host_fill => {},
                .dma_read, .dma_staged, .dma_gather, .dma_fill => continue,
            }
            for (plan.transfersOf(job)) |transfer| {
                const destination = transfer.destination;
                if (destination == previous or destination.state.readyValue() != null) continue;
                previous = destination;
                if (self.failed()) break :walk;
                const started = awakeNs(self.io);
                _ = destination.ensureState(self) catch |err| {
                    self.recordError(err, "initialize tensor {s} from {s} ahead of its reads, shape={f}", .{ destination.name(), destination.parts[0].source.file_uri, destination.shape });
                    break :walk;
                };
                _ = self.metrics.ahead_init_ns.fetchAdd(awakeNs(self.io) -| started, .monotonic);
                ensured += 1;
            }
        };
        load_log.debug("batch allocations: batch={d}, host_destinations={d}, ensured={d}, done=+{d:.3}s", .{ batch.diagnostics.sequence, host_destinations, ensured, secondsBetween(self.created_at, .now(self.io, .awake)) });
        batch.finishJobs(1);
    }

    /// Waits for the batch's last completion unit, retires it and returns
    /// the loader's sticky error if the loader failed. Targets the failure
    /// left open are marked so their buffers never report ready; a target
    /// PJRT already closed (its last call went out, accepted or not) is left
    /// to PJRT, whose shared transfer manager has dropped the definition
    /// event and aborts on a further call. The same manager drops the event
    /// when a transfer fails asynchronously, which the loader does not check
    /// (the done event carries no error, only the buffer's definition event
    /// does), so a buffer that failed that way and is then marked here
    /// aborts too: two failures in one load. A program reading such a
    /// buffer fails its execution, which the front end returns; a bulk
    /// buffer shows the error at its first use. Either way an output of a
    /// failed submission holds buffers whose contents are unspecified, or
    /// was never written. A batch that
    /// completed without an error must have closed every target (the pump
    /// flagged the submission that completed its bytes); one that did not
    /// would leave a buffer that never becomes ready, so it fails the
    /// loader instead.
    pub fn awaitBatch(self: *Loader, batch: *Batch) AwaitError!void {
        batch.done.waitUncancelable(self.io);
        const done_at: std.Io.Timestamp = .now(self.io, .awake);
        // Every request of this batch has completed: no worker or callback
        // touches its managers or contexts any more.
        var load_error = self.errorValue();
        if (load_error == null and !batch.fullySubmitted()) {
            self.recordError(error.Internal, "batch {d} completed without submitting every target's bytes", .{batch.diagnostics.sequence});
            load_error = self.errorValue();
        }
        // Each destination completed its progress item.
        if (load_error == null) for (batch.destinations) |*destination| std.debug.assert(destination.unread_bytes.load(.acquire) == 0);
        if (load_error != null) {
            for (batch.destinations) |*destination| {
                const state = destination.state.readyValue() orelse continue;
                for (state.targets) |*target| {
                    if (!target.closed) {
                        target.manager.setBufferErrorUnknown(
                            self.platform.pjrt_api,
                            0,
                            "live loader failed",
                        ) catch {};
                    }
                }
            }
        }
        Dma.retireBatch(self, batch);
        self.logBatch(batch, done_at, load_error == null);
        self.destroyBatch(batch);
        if (load_error) |err| return err;
    }

    /// Fails the loader unless it already failed (`recordError`): unclaimed
    /// jobs retire, queued transfers drain, and requests in flight finish
    /// their read.
    pub fn abort(self: *Loader) void {
        self.recordError(error.Canceled, "loader aborted", .{});
    }

    /// The front end awaited every batch before this; nothing is queued or
    /// in flight, so stopping the workers is a plain shutdown.
    pub fn destroy(self: *Loader) void {
        self.stopWorkers();
        self.logSummary();
        var slots = self.source_slots.valueIterator();
        while (slots.next()) |slot| {
            slot.*.deinit(self.io);
            self.allocator.free(slot.*.uri);
            self.allocator.destroy(slot.*);
        }
        self.source_slots.deinit(self.allocator);
        // Every batch was retired: no DMA event, queued transfer or request
        // context may outlive its batch.
        std.debug.assert(self.credits.inUse(self.io) == 0);
        for (self.pumps) |*device_pump| {
            std.debug.assert(device_pump.active_pieces == 0);
            std.debug.assert(device_pump.queue.len == 0);
            std.debug.assert(device_pump.retired == null);
            device_pump.queue.deinit(self.allocator);
        }
        self.allocator.free(self.pumps);
        self.scheduler.deinit();
        self.credits.deinit(self.allocator);
        self.allocator.free(self.blocks);
        self.workspace.deinit();

        const allocator = self.allocator;
        allocator.free(self.allocated_bytes);
        allocator.free(self.submitted_bytes);
        allocator.destroy(self);
    }

    /// Cumulative output bytes allocated since creation, per `platform.devices`
    /// index. Freeing an output does not decrease these counters.
    pub fn allocatedBytesPerDevice(self: *const Loader, out: []u64) void {
        for (out, self.allocated_bytes) |*bytes, *counter| bytes.* = counter.load(.acquire);
    }

    /// Output bytes of the sealed batches not allocated yet, per
    /// `platform.devices` index: what is still to land on each device.
    /// Workers allocate as they read, possibly before the batch is sealed,
    /// so the difference saturates.
    pub fn unlandedBytesPerDevice(self: *const Loader, out: []u64) void {
        for (out, self.submitted_bytes, self.allocated_bytes) |*bytes, submitted, *allocated| bytes.* = submitted -| allocated.load(.acquire);
    }

    /// Counts the device bytes of the sealed `batch`: a box of each
    /// destination outside host memory on each of its devices, as
    /// `TensorTransfer.init` allocates them.
    fn countSubmitted(self: *Loader, batch: *const Batch) void {
        for (batch.destinations) |*destination| {
            if (destination.hostResident()) continue;
            for (destination.placed_shape._sharding.mesh.?.devicesInCanonicalOrder()) |device| self.submitted_bytes[device.id] += destination.grid.box_bytes;
        }
    }

    fn checkOpen(self: *Loader) !void {
        if (self.errorValue()) |err| return err;
    }

    /// Creates the batch's destinations, one per spec, with copies of their
    /// parts and dimensions; on failure nothing stays allocated. A
    /// destination's device state is created lazily by the first worker that
    /// reads or fills for it, or by the allocation task.
    fn createDestinations(
        self: *Loader,
        batch: *Batch,
        specs: []const LoadSpec,
        progress: std.Progress.Node,
    ) !void {
        var part_count: usize = 0;
        var dim_count: usize = 0;
        for (specs) |spec| {
            part_count += spec.parts.len;
            for (spec.parts) |part| dim_count += part.dims.len;
        }
        const destinations = try self.allocator.alloc(Destination, specs.len);
        errdefer self.allocator.free(destinations);
        const parts = try self.allocator.alloc(Part, part_count);
        errdefer self.allocator.free(parts);
        const dims = try self.allocator.alloc(LoadPart.Dim, dim_count);
        errdefer self.allocator.free(dims);
        var next_part: usize = 0;
        var next_dim: usize = 0;
        var logical_bytes: usize = 0;
        for (specs, destinations) |spec, *destination| {
            const placed_shape = self.platform.placedShape(spec.shape).packedShape();
            const destination_parts = parts[next_part..][0..spec.parts.len];
            next_part += spec.parts.len;
            var covered_bytes: u64 = 0;
            var gathered = false;
            for (spec.parts, destination_parts) |part, *slot| {
                const part_dims = dims[next_dim..][0..part.dims.len];
                next_dim += part.dims.len;
                @memcpy(part_dims, part.dims);
                slot.* = .{
                    .source = part.source,
                    .source_slot = try self.sourceSlot(part.source.file_uri),
                    .destination = destination,
                    .src_offset = part.src_offset,
                    .dst_offset = part.dst_offset,
                    .width = part.width,
                    .dims = part_dims,
                };
                covered_bytes += part.width;
                gathered = gathered or part.dims.len != 0;
            }
            destination.* = .{
                .shape = spec.shape,
                .placed_shape = placed_shape,
                .grid = try .init(placed_shape),
                .memory = spec.memory,
                .output = spec.output,
                .parts = destination_parts,
                .gathered = gathered,
                // A gathered destination's jobs land every byte, zeros too.
                .unread_bytes = .init(if (gathered) placed_shape.byteSize() else covered_bytes),
                .progress = progress,
            };
            if (gathered) for (destination_parts) |*part| {
                // A remote file is read exact (`Planner.Reads.alignmentOf`);
                // a local one may be read directly, its range widened.
                var config = self.plan_config;
                if (VFS.fromIo(self.io)) |vfs| {
                    if (!vfs.isLocal(part.source.file_uri)) config.alignment = 0;
                }
                try Planner.checkGatherReads(config, part);
            };
            logical_bytes += spec.shape.byteSize();
        }
        batch.destinations = destinations;
        batch.parts = parts;
        batch.dims = dims;
        batch.diagnostics.source_items = parts.len;
        batch.diagnostics.logical_bytes = logical_bytes;
    }

    /// Releases the destinations' device state, then the batch's memory.
    fn destroyBatch(self: *Loader, batch: *Batch) void {
        for (batch.destinations) |*destination| destination.deinit(self.allocator, self.platform.pjrt_api);
        batch.destroy();
    }

    fn sourceSlot(self: *Loader, uri: []const u8) !*SourceSlot {
        if (self.source_slots.get(uri)) |slot| return slot;
        const owned = try self.allocator.dupe(u8, uri);
        errdefer self.allocator.free(owned);
        const slot = try self.allocator.create(SourceSlot);
        errdefer self.allocator.destroy(slot);
        slot.* = .{ .uri = owned };
        try self.source_slots.putNoClobber(self.allocator, owned, slot);
        return slot;
    }

    fn workerMain(self: *Loader) void {
        var scratch = ReadRequest.Scratch.init(
            self.allocator,
            self.credits.blocks_per_slot,
            self.plan_config.block_size,
            self.platform.devices.len,
        ) catch |err| {
            self.recordError(err, "allocate source worker scratch: blocks={d}, devices={d}", .{ self.credits.blocks_per_slot, self.platform.devices.len });
            return;
        };
        defer scratch.deinit();
        while (self.scheduler.waitForWork(self.io)) {
            if (self.failed()) return;
            const credit_wait_started = awakeNs(self.io);
            const slot = self.credits.acquire(self.io) orelse return;
            const credit_wait_ns = awakeNs(self.io) -| credit_wait_started;
            const claim = self.scheduler.claim(self.io) orelse {
                self.credits.release(self.io, slot);
                continue;
            };
            _ = self.metrics.lifecycle_wait_ns.fetchAdd(credit_wait_ns, .monotonic);
            const request = ReadRequest.init(self, claim, slot);
            // The scheduling sentinel keeps the batch, and with it the
            // request's plan, alive through `run` and error reporting.
            defer request.release();
            request.run(&scratch) catch |err| {
                const job = request.plan.jobs[request.index];
                const bytes: usize = switch (job.kind) {
                    .dma_read, .host_direct, .host_bounce => |read| read.len,
                    .dma_staged => |staged| staged.read.len,
                    .dma_gather => |gather| gather.len,
                    .dma_fill, .host_fill => 0,
                };
                self.recordError(err, "source request: file={s}, offset={d}, bytes={d}, job={t}", .{ request.plan.source_slot.uri, job.file_offset, bytes, job.kind });
            };
        }
    }

    /// Stops the workers; awaiting a group that never spawned returns at
    /// once, so this also cleans up a failed `create`.
    fn stopWorkers(self: *Loader) void {
        self.scheduler.stop(self.io);
        self.read_gate.close(self.io);
        self.credits.close(self.io);
        self.worker_group.await(self.io) catch {};
    }

    fn failed(self: *const Loader) bool {
        return self.first_error.load(.acquire) != 0;
    }

    fn errorValue(self: *const Loader) ?StickyError {
        const value = self.first_error.load(.acquire);
        return if (value == 0) null else @errorCast(@errorFromInt(value));
    }

    /// Keeps `err` as the sticky error unless one is kept already and logs
    /// it, then fails the load without waiting: unclaimed jobs retire
    /// (`Scheduler.fail`), reads and credits close, and queued transfers are
    /// released (`Dma.abortReady`). Any thread may call it, a DMA callback
    /// included; requests in flight finish and release their units.
    /// Failing other batches than the caller's is safe from any thread:
    /// claims are FIFO and a queued plan holds a unit of its batch, so
    /// `fail` reaches only batches still alive; a batch retires only once
    /// its last unit is released, and `retireBatch` holds every pump lock
    /// while each queued transfer holds its request, so `abortReady` also
    /// reaches only live batches; and `destroy` waits for the worker group
    /// before it frees what the workers use.
    fn recordError(self: *Loader, err: StickyError, comptime fmt: []const u8, args: anytype) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            load_log.failure(err, fmt, args);
            self.scheduler.fail(self.io);
            self.read_gate.close(self.io);
            self.credits.close(self.io);
            Dma.abortReady(self);
        }
    }

    /// Counts the plans' jobs, transfers and events only when the lines
    /// would be logged.
    fn logBatch(self: *Loader, batch: *const Batch, done_at: std.Io.Timestamp, successful: bool) void {
        if (!std.log.logEnabled(.debug, .@"zml/io/load")) return;
        const diagnostics = &batch.diagnostics;
        const sealed_at = diagnostics.sealed_at orelse self.created_at;
        const published_at = diagnostics.published_at orelse sealed_at;
        load_log.debug("batch completed: batch={d}, successful={}, logical_bytes={Bi:.2}, published=+{d:.3}s, sealed=+{d:.3}s, done=+{d:.3}s, elapsed={d:.3}s, source_width={d}, request_size={Bi:.2}", .{
            diagnostics.sequence,
            successful,
            diagnostics.logical_bytes,
            secondsBetween(self.created_at, published_at),
            secondsBetween(self.created_at, sealed_at),
            secondsBetween(self.created_at, done_at),
            secondsBetween(published_at, done_at),
            self.read_gate.limit,
            self.plan_config.request_size,
        });

        var planning_ns: u64 = 0;
        var source_bytes: u64 = 0;
        var jobs: usize = 0;
        var transfers: usize = 0;
        var dma_submissions: usize = 0;
        var kinds: std.EnumArray(std.meta.Tag(Batch.Plan.Job.Kind), usize) = .initFill(0);
        for (batch.plans.items) |plan| {
            planning_ns += plan.planning_ns;
            source_bytes += plan.source_bytes;
            jobs += plan.jobs.len;
            transfers += plan.transfers.len;
            dma_submissions += plan.events.len;
            for (plan.jobs) |job| kinds.getPtr(job.kind).* += 1;
        }
        const fill_jobs = kinds.get(.dma_fill) + kinds.get(.host_fill);
        const host_jobs = kinds.get(.host_direct) + kinds.get(.host_bounce) + kinds.get(.host_fill);
        const source_jobs = jobs - fill_jobs;
        const average_read_size = if (source_jobs == 0) 0 else source_bytes / source_jobs;
        const coalescing_ratio = if (source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(diagnostics.source_items)) /
                @as(f64, @floatFromInt(source_jobs));
        load_log.debug("batch planning: batch={d}, plans={d}, planning_elapsed={d:.3}s, planned_source_bytes={Bi:.2}, planned_source_jobs={d}, fill_jobs={d}, host_jobs={d}, source_items={d}, planned_transfers={d}, planned_dma_submissions={d}, staged_jobs={d}, gather_jobs={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}", .{
            diagnostics.sequence,
            batch.plans.items.len,
            @as(f64, @floatFromInt(planning_ns)) / std.time.ns_per_s,
            source_bytes,
            source_jobs,
            fill_jobs,
            host_jobs,
            diagnostics.source_items,
            transfers,
            dma_submissions,
            kinds.get(.dma_staged),
            kinds.get(.dma_gather),
            coalescing_ratio,
            average_read_size,
        });
    }

    fn logSummary(self: *Loader) void {
        const reads = self.metrics.read_operations.load(.acquire);
        load_log.debug("loader summary: batches={d}, successful={}, read_bytes={Bi:.2}, elapsed={d:.3}s, reads={d}, source_width={d}, request_size={Bi:.2}, credits_high_water={d} of {d}, pinned_mapped={Bi:.2}", .{
            self.batch_count,
            !self.failed(),
            self.metrics.read_bytes.load(.acquire),
            secondsBetween(self.created_at, .now(self.io, .awake)),
            reads,
            self.read_gate.limit,
            self.plan_config.request_size,
            self.credits.high_water,
            self.credits.count(),
            self.workspace.mapped_bytes,
        });
        load_log.debug("loader waits: credit_wait_ms_per_read={d:.3}, read_ms_per_read={d:.3}, dma_stage_ms_per_read={d:.3}, tensor_init_ms_per_read={d:.3}, ahead_init_ms={d:.3}", .{
            millisecondsPer(self.metrics.lifecycle_wait_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.read_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.dma_stage_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.tensor_init_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.ahead_init_ns.load(.acquire), 1),
        });
        const submissions = self.metrics.dma_submissions.load(.acquire);
        load_log.debug("loader DMA: dma_submissions={d}, dma_submit_us_per_piece={d:.2}, dma_piece_latency_ms={d:.3}, pump_stops_empty={d}, pump_stops_full={d}", .{
            submissions,
            millisecondsPer(self.metrics.dma_submit_ns.load(.acquire), submissions) * 1000,
            millisecondsPer(self.metrics.dma_piece_ns.load(.acquire), submissions),
            self.metrics.pump_stops_empty.load(.acquire),
            self.metrics.pump_stops_full.load(.acquire),
        });
    }
};

/// One submission: its plans (one per source file, published as each file
/// is planned), the destinations it writes and their parts, and every
/// request and event context created while loading it. `remaining` counts completion
/// units: one per published job plus a publish sentinel held until the
/// submission is sealed, and one for the allocation task while it runs
/// (`Loader.startAllocations`), taken before the sentinel is dropped. A
/// job's unit is released exactly once, by whichever of these happens: its
/// request's last reference drops (final DMA callback, a host job's end, or
/// abandonment), or `Scheduler.fail` retires it unclaimed. The
/// batch is done when `remaining` reaches zero; the awaiting task then
/// retires it, so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    /// One file's jobs in claim order with their transfer records and every
    /// context the jobs can need, allocated by the planner: one request per
    /// job and one event per planned DMA submission, handed out in
    /// submission order; a host job needs only its request. Contexts hold the
    /// plan's address, so plans are heap objects freed with the batch.
    const Plan = struct {
        /// A claimable job: its slice of the plan's transfers and what its
        /// worker does for them. Its request slot is its index.
        const Job = struct {
            /// Where the job reads first: its read's start, a gather job's
            /// first extent's; zero for a job that reads nothing.
            /// `Planner.preparePlan` sorts the read and gather jobs by it and
            /// appends the fill jobs after them; a failed request logs it.
            file_offset: u64,
            transfer_start: usize,
            transfer_len: usize,
            kind: Kind,

            /// How the job's bytes reach their destinations. A destination
            /// in a host memory space (`Destination.hostResident`) is written
            /// by the worker that reads it: going through pinned blocks and
            /// a host-to-host PJRT transfer copied every byte twice and held
            /// the DMA pumps, which on four B70 made Llama-3.1-8B take 2.0 s
            /// into pinned_host against 0.67 s into device memory.
            const Kind = union(enum) {
                /// Read into the request's slot of pinned blocks (`Credits`)
                /// in file order, where its transfers address it, then
                /// copied by the devices through PJRT transfer managers
                /// (`Dma`).
                dma_read: Read,
                /// Read, then landed in its blocks grouped by destination box
                /// (`Planner.JobCut.stage`).
                dma_staged: Staged,
                /// Reads its extents, which its items land in its blocks
                /// (`Planner.JobCut.appendGatherJobs`).
                dma_gather: Gather,
                /// Reads nothing: every transfer goes from the loader's zero
                /// block (`Planner.appendFillTransfers`).
                dma_fill,
                /// Read straight into host-resident destinations: the read's
                /// iovecs are the destination ranges, so the read is the only
                /// copy. Needs a read without alignment padding whose
                /// transfers address each of its bytes once, in order
                /// (`Planner.JobCut.coversReadInOrder`).
                host_direct: Read,
                /// Read into the request's slot of blocks, then copied into
                /// host-resident destinations by the worker: reads widened
                /// for direct I/O, whose buffers must be aligned, and
                /// overlapping parts.
                host_bounce: Read,
                /// Writes zeros in place, reading nothing.
                host_fill,
            };

            /// `len` bytes from the job's `file_offset`, of which the first
            /// `minimum_len` hold tensor data and must exist. The rest is
            /// alignment padding the end of the file may cut.
            const Read = struct {
                len: usize,
                minimum_len: usize,
            };

            /// A staged job's read, whose tensor bytes, from `data_offset`
            /// to `read.minimum_len`, the `item_len` items of the plan's
            /// `items` from `item_start` place in its blocks.
            const Staged = struct {
                read: Read,
                /// Alignment padding before the job's first tensor byte.
                data_offset: usize,
                item_start: usize,
                item_len: usize,
            };

            /// A gather job: `len` bytes of its destination's logical range,
            /// placed in its blocks by the `item_len` items of the plan's
            /// `items` from `item_start`, one from each of its `extent_len`
            /// extents from `extent_start`, in order. A range no part covers
            /// has neither.
            const Gather = struct {
                len: usize,
                item_start: usize,
                item_len: usize,
                extent_start: usize,
                extent_len: usize,
                /// Its blocks start zeroed: bytes no part covers.
                zeroed: bool,
            };

            /// The blocks of its credit's slot the job reads into or lands
            /// in; none for a direct host job or a fill job.
            fn blockLen(self: Job, block_size: usize) usize {
                return switch (self.kind) {
                    .dma_read, .host_bounce => |read| Planner.blockCount(read.len, block_size),
                    .dma_staged => |staged| Planner.blockCount(staged.read.minimum_len - staged.data_offset, block_size),
                    .dma_gather => |gather| Planner.blockCount(gather.len, block_size),
                    .dma_fill, .host_direct, .host_fill => 0,
                };
            }
        };

        /// Bytes a staged or gather job places in its blocks: the runs
        /// `first` to `end` of `lattice`, its source offsets in the file
        /// and its destination offsets logical, clipped to the logical
        /// range `[clip_start, clip_end)`. A byte at logical offset `l` in
        /// box `b` of the destination's grid lands at
        /// `landings[landing_start + b] +% grid.boxOffset(l)` of the job's
        /// blocks taken end to end (`walkItem`). A staged job's items come
        /// from its read, a gather job's from the extent of the same index.
        const Item = struct {
            destination: *Destination,
            lattice: backend.Lattice,
            first: u64 = 0,
            end: u64 = 1,
            clip_start: u64,
            clip_end: u64,
            landing_start: usize,
        };

        /// One read of a gather job: `len` bytes from `file_offset` of
        /// `source_slot`'s file, of which the first `minimum_len` exist.
        const Extent = struct {
            source_slot: *SourceSlot,
            file_offset: u64,
            len: usize,
            minimum_len: usize,
            /// Read straight into the job's blocks: an exact read whose
            /// items' runs tile it, in order. Otherwise read into the
            /// worker's bounce buffer and copied from there.
            scatter: bool,
        };

        /// `len` bytes of the job's read, at `block_offset` of its block
        /// `block_index`, for every device of `writer_mask` at
        /// `destination_offset` of its shard. A DMA transfer stays inside
        /// one block; a host one is not cut by blocks: its `block_index` is
        /// zero and its `block_offset` counts from the read's start.
        const Transfer = struct {
            destination: *Destination,
            block_index: usize,
            block_offset: usize,
            writer_mask: u64,
            destination_offset: usize,
            len: usize,
        };

        allocator: std.mem.Allocator,
        /// The submission this plan belongs to; set when it is published,
        /// under the scheduler mutex, with its completion units.
        batch: *Batch = undefined,
        /// The file every job of the plan reads.
        source_slot: *SourceSlot,
        jobs: []Job,
        transfers: []Transfer,
        /// The staged and gather jobs' items, their landing tables (one
        /// entry per box of the item's destination) and the gather jobs'
        /// extents, owned by the plan.
        items: []Item = &.{},
        landings: []u64 = &.{},
        extents: []Extent = &.{},
        /// The file is read directly: every read is aligned, and a staged
        /// job goes through the worker's bounce buffer.
        direct_reads: bool = false,
        requests: []ReadRequest,
        events: []Dma.EventContext,
        /// Event slots handed out so far, one `fetchAdd` per submission from
        /// any device's pump.
        events_used: std.atomic.Value(usize) = .init(0),
        source_bytes: u64,
        planning_ns: u64 = 0,
        /// Next job to claim; owned by the scheduler mutex.
        cursor: usize = 0,

        /// Takes ownership of transfers on success. All callback storage is
        /// allocated here, before publication makes the plan visible to
        /// workers: `dma_submissions` events, one per writer of a DMA
        /// transfer.
        fn create(
            allocator: std.mem.Allocator,
            source_slot: *SourceSlot,
            job_count: usize,
            transfers: []Transfer,
            dma_submissions: usize,
            source_bytes: u64,
        ) !*Plan {
            const jobs = try allocator.alloc(Job, job_count);
            errdefer allocator.free(jobs);
            const requests = try allocator.alloc(ReadRequest, job_count);
            errdefer allocator.free(requests);
            @memset(requests, ReadRequest.idle);
            const events = try allocator.alloc(Dma.EventContext, dma_submissions);
            errdefer allocator.free(events);
            const self = try allocator.create(Plan);
            self.* = .{
                .allocator = allocator,
                .source_slot = source_slot,
                .jobs = jobs,
                .transfers = transfers,
                .requests = requests,
                .events = events,
                .source_bytes = source_bytes,
            };
            return self;
        }

        fn transfersOf(self: *const Plan, job: Job) []const Transfer {
            return self.transfers[job.transfer_start..][0..job.transfer_len];
        }

        /// Frees the plan; `Dma.retireBatch` retired its contexts first.
        fn destroy(self: *Plan) void {
            const allocator = self.allocator;
            allocator.free(self.events);
            allocator.free(self.requests);
            allocator.free(self.extents);
            allocator.free(self.landings);
            allocator.free(self.items);
            allocator.free(self.transfers);
            allocator.free(self.jobs);
            allocator.destroy(self);
        }
    };

    /// What `Loader.logBatch` logs beside the counts it takes from the plans.
    const Diagnostics = struct {
        /// Submission number within the loader, for log correlation.
        sequence: usize = 0,
        logical_bytes: usize = 0,
        /// Parts planned: one per plain tensor, one per source of a
        /// composite one.
        source_items: usize = 0,
        /// The first publish; null for a submission without plans.
        published_at: ?std.Io.Timestamp = null,
        /// Once every plan is published, before the publish sentinel goes.
        sealed_at: ?std.Io.Timestamp = null,
    };

    // Borrowed source URI of the file currently being planned.
    planning_source: []const u8 = "",
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Published plans in file order: appended by the submitting task under
    /// the scheduler mutex; freed at `destroy`.
    plans: std.ArrayList(*Plan) = .empty,
    /// Owned by the batch; their device state by the loader, released
    /// before `destroy`.
    destinations: []Destination = &.{},
    /// The destinations' parts, each destination's consecutive and in its
    /// destination order: what the planner sorts by file offset.
    parts: []Part = &.{},
    /// The strided parts' dimensions.
    dims: []LoadPart.Dim = &.{},
    remaining: std.atomic.Value(usize),
    done: std.Io.Event = .unset,
    diagnostics: Diagnostics,
    /// Set by retirement so a unit released after `done` trips `finishJobs`.
    freeing: if (builtin.mode == .debug) bool else void = if (builtin.mode == .debug) false else {},

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

    /// Releases `count` completion units. MEMORY-ORDER RULE: this must be the
    /// caller's final access to the batch and to anything it owns (requests,
    /// events, destinations, parts, plans; a request's slot of blocks is the
    /// loader's and went back with its credit just before). The last unit
    /// sets `done`, and the awaiting task frees all of it as soon as it
    /// observes the event.
    fn finishJobs(self: *Batch, count: usize) void {
        if (count == 0) return;
        if (builtin.mode == .debug) std.debug.assert(!self.freeing);
        const previous = self.remaining.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.done.set(self.io);
    }

    /// Frees the destinations, the parts and the plans with their contexts,
    /// which `Dma.retireBatch` must already have retired.
    fn destroy(self: *Batch) void {
        self.allocator.free(self.dims);
        self.allocator.free(self.parts);
        self.allocator.free(self.destinations);
        for (self.plans.items) |plan| plan.destroy();
        self.plans.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// After `done`: every destination was touched and every target
    /// received its last transfer.
    fn fullySubmitted(self: *const Batch) bool {
        for (self.destinations) |*destination| {
            const state = destination.state.readyValue() orelse return false;
            if (!state.fullySubmitted()) return false;
        }
        return true;
    }
};

/// One output buffer and what places it. Owned by the batch. Its parts may
/// come from several files, so several plans can write it: their pieces,
/// and the zero fill, all count toward the same targets (`Target.total` is
/// the placement's bytes, whichever plan submits them; `host_total` for a
/// host-resident one), and its device state is created once, by the first
/// worker that reads or fills for it or, for a host-resident one, normally
/// by the allocation task ahead of them.
const Destination = struct {
    const InitContext = struct { destination: *const Destination, direct: *Loader };

    fn initTransfer(ctx: InitContext) TransferInitError!TensorTransfer {
        return TensorTransfer.init(ctx.direct, ctx.destination);
    }

    /// The caller's shape, which the output buffer keeps.
    shape: Shape,
    /// What PJRT allocates and the planner cuts: `shape` packed, its
    /// sharding naming a mesh (`Platform.placedShape`).
    placed_shape: Shape,
    /// Where `placed_shape`'s logical bytes land on its devices.
    grid: ShardGrid,
    /// Where its bytes go (`TensorStore.Memory`).
    memory: Memory = .default,
    output: *Buffer,
    /// What fills it, in ascending, disjoint destination order
    /// (`backend.LoadSpec`): a slice of the batch's parts. Bytes no part
    /// covers are written from the loader's zero block, by the fill job of
    /// the plan holding `parts[0]`.
    parts: []Part,
    /// Fed by strided parts: planned whole, by logical ranges, with the
    /// plan holding `parts[0]` (`Planner.JobCut.appendGatherJobs`), whose
    /// jobs also write its zero fill.
    gathered: bool = false,
    /// Bytes not read yet: the parts' widths, or for a gathered destination
    /// its whole placed size (its gather jobs land the zeros too). The read
    /// that brings it to zero completes the destination's item on
    /// `progress`.
    unread_bytes: std.atomic.Value(u64),
    progress: std.Progress.Node = .none,
    state: LazyOnce(TensorTransfer, InitContext, TransferInitError, initTransfer) = .{},

    /// In host memory (`Memory.isHost`): written by the CPU, not by DMA.
    fn hostResident(self: *const Destination) bool {
        return self.memory.isHost();
    }

    fn ensureState(self: *Destination, direct: *Loader) TransferInitError!*TensorTransfer {
        return self.state.ensure(direct.io, .{ .destination = self, .direct = direct });
    }

    fn deinit(self: *Destination, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        if (self.state.readyValue()) |state| state.deinit(allocator, api);
    }

    /// The first source's name, for logs.
    fn name(self: *const Destination) []const u8 {
        return self.parts[0].source.name;
    }

    /// Counts `bytes` as read (`unread_bytes`); the last read completes the
    /// destination's one progress item.
    fn recordRead(self: *Destination, bytes: u64) void {
        if (self.unread_bytes.fetchSub(bytes, .acq_rel) == bytes) self.progress.completeOne();
    }
};

/// One source range of a destination: what the planner sorts by file offset
/// and cuts into jobs, as it did whole tensors. A plain tensor is one part
/// covering its whole destination.
const Part = struct {
    source: *const safetensors.Tensor,
    source_slot: *SourceSlot,
    destination: *Destination,
    /// `width` bytes of the source from `src_offset` land at `dst_offset`
    /// of the destination's logical bytes, repeated over `dims` for a
    /// strided part (`backend.Lattice`).
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
    width: u64,
    dims: []const LoadPart.Dim = &.{},

    fn lattice(self: *const Part) backend.Lattice {
        return .{ .src_offset = self.src_offset, .dst_offset = self.dst_offset, .width = self.width, .dims = self.dims };
    }

    /// The part's bytes in its file.
    fn fileRange(self: *const Part) error{InvalidTensorRange}!struct { start: u64, end: u64 } {
        const start = std.math.add(u64, self.source.offset, self.src_offset) catch return self.rangeError(self.src_offset);
        const end = std.math.add(u64, start, self.width) catch return self.rangeError(self.src_offset +| self.width);
        return .{ .start = start, .end = end };
    }

    /// The part's source bytes up to `source_end`, from the source's file
    /// offset, pass the largest file offset.
    fn rangeError(self: *const Part, source_end: u64) error{InvalidTensorRange} {
        load_log.err("destination {f} from {s} in {s}: {d} source bytes from offset {d} pass the largest file offset", .{ self.destination.shape, self.source.name, self.source.file_uri, source_end, self.source.offset });
        return error.InvalidTensorRange;
    }
};

const SourceSlot = struct {
    const OpenContext = struct { io: std.Io, uri: []const u8 };

    fn openFile(ctx: OpenContext) std.Io.File.OpenError!std.Io.File {
        return std.Io.Dir.openFile(.cwd(), ctx.io, ctx.uri, .{ .mode = .read_only });
    }

    uri: []const u8,
    file: LazyOnce(std.Io.File, OpenContext, std.Io.File.OpenError, openFile) = .{},
    /// The file's read alignment once the planner decided it
    /// (`Planner.Reads`): 0 when it is read exactly.
    alignment: ?usize = null,

    fn ensure(self: *SourceSlot, io: std.Io) std.Io.File.OpenError!std.Io.File {
        const file = try self.file.ensure(io, .{ .io = io, .uri = self.uri });
        return file.*;
    }

    fn deinit(self: *SourceSlot, io: std.Io) void {
        if (self.file.readyValue()) |file| file.close(io);
    }
};

const TensorTransfer = struct {
    /// One destination buffer of a tensor. PJRT makes the buffer ready once
    /// every transfer submitted to it has completed and one of them carried
    /// the last-transfer flag, whatever their completion order: the flag
    /// only closes the buffer to further calls. The pump, the only
    /// submitter, therefore flags the submission that completes the
    /// placement's bytes, and no piece ever waits for another one.
    /// The former highest-offset tail waited for its prefixes. On a sharded
    /// multi-device load, tails could occupy every lifecycle credit while
    /// their prefixes remained unclaimed: more pinned memory did not fix it.
    /// Closing by submitted bytes removes that dependency. This follows
    /// XLA's CommonAsyncHostToDeviceTransferManager: readiness requires zero
    /// in-flight transfers and a last-call flag, not destination-offset order.
    const Target = struct {
        manager: *pjrt.AsyncHostToDeviceTransferManager,
        device_index: usize,
        /// Bytes of the placement on this device. The pieces partition them:
        /// those of every part, from whichever file and plan, and those of
        /// the zero fill.
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

    /// A DMA destination's transfer targets, one per device in canonical
    /// order; empty for a host-resident one.
    targets: []Target = &.{},
    /// A host-resident destination's shards in host memory, which workers
    /// write in place, one per device in canonical order; empty for a DMA
    /// one. The devices of one box share theirs (`initHost`).
    host: [][*]u8 = &.{},
    /// Device bytes written into the host shards so far, summed over the
    /// devices, against `host_total`: the host side of `fullySubmitted`.
    host_written: std.atomic.Value(usize) = .init(0),
    host_total: usize = 0,

    /// Creates the destination's buffers and writes the output shell: for a
    /// DMA destination, the device buffers and transfer managers; for a
    /// host-resident one, the buffers workers write (`initHost`). Runs once
    /// per destination (`LazyOnce`), however many parts feed it, on the
    /// allocation task or the worker that first reads or fills for it,
    /// which also counts the allocation once.
    /// A host destination (`Memory.isHost`) is created in each device's
    /// pinned host memory space and takes no device memory, so it is not
    /// counted in `allocated_bytes`.
    fn init(direct: *Loader, destination: *const Destination) TransferInitError!TensorTransfer {
        if (destination.hostResident()) return initHost(direct, destination);
        const allocator = direct.allocator;
        const platform = direct.platform;
        const packed_placement: Sharding.Placement = try .init(&destination.placed_shape);
        const packed_local = packed_placement.localShape(destination.placed_shape);
        const ordered_devices = packed_placement.mesh().?.devicesInCanonicalOrder();
        const targets = try allocator.alloc(Target, ordered_devices.len);
        errdefer allocator.free(targets);

        var pjrt_buffers: Buffer.Shards = .empty;
        var initialized: usize = 0;
        errdefer {
            for (targets[0..initialized]) |target| target.manager.deinit(platform.pjrt_api);
            for (pjrt_buffers.constSlice()) |buffer| buffer.deinit(platform.pjrt_api);
        }

        const shape_spec: pjrt.ShapeSpec = .init(
            packed_local.dims(),
            pjrtx.bufferTypeFromDtype(packed_local.dtype()),
        );
        for (ordered_devices, 0..) |device, i| {
            const memory = platform.devices[device.id].memory(.default) orelse return error.UnsupportedMemoryKind;
            const manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(platform.pjrt_api);
            const pjrt_buffer = try manager.retrieveBuffer(platform.pjrt_api, 0);
            targets[i] = .{
                .manager = manager,
                .device_index = device.id,
                .total = packed_local.byteSize(),
            };
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(pjrt_buffer);
        }
        for (targets) |target| {
            _ = direct.allocated_bytes[target.device_index].fetchAdd(target.total, .monotonic);
        }

        destination.output.* = .fromPjrtBuffers(platform, destination.shape, pjrt_buffers.constSlice());
        return .{ .targets = targets };
    }

    /// Creates a host-resident destination's buffers and reads their host
    /// pointers for the workers. The devices holding one box
    /// (`ShardGrid.writers`) share one allocation: every device's shard is
    /// a view of it, and the allocation lives until the last view is gone
    /// (`SharedHostAllocation`). A transfer is then written once, through
    /// its first writer's shard (`hostLanding`). A `host_mapped` shard is
    /// always such a view, on the device's default memory space, so
    /// kernels read the allocation in place. Every direct target lets a
    /// host allocation of one device back a view on another, for transfers
    /// and kernels alike: CPU, CUDA and ROCm allocate portable pinned memory
    /// (`Buffer.HostAccessible.init`), and on oneAPI the four B70 of the test
    /// "a replicated host-pinned table is one allocation that every device
    /// reads through device views" gather from an allocation made in the
    /// first one's host memory space, and read it back. A target that could
    /// not would need a copy per device.
    /// The buffers are complete before any worker writes them, so they
    /// report ready from the start; the front end hands them out only once
    /// the batch is done.
    fn initHost(direct: *Loader, destination: *const Destination) TransferInitError!TensorTransfer {
        const allocator = direct.allocator;
        const platform = direct.platform;
        const api = platform.pjrt_api;
        const placement: Sharding.Placement = try .init(&destination.placed_shape);
        const local = placement.localShape(destination.placed_shape);
        const ordered_devices = placement.mesh().?.devicesInCanonicalOrder();
        const shard_count = ordered_devices.len;
        std.debug.assert(shard_count <= Platform.MAX_NUM_DEVICES);
        const shards = try allocator.alloc([*]u8, shard_count);
        errdefer allocator.free(shards);
        const dims = local.dims();
        const element_type = pjrtx.bufferTypeFromDtype(local.dtype());
        const layout = platform.defaultMemoryLayout(dims, local.dtype());

        // The writers of each device's box; the first of them creates the
        // allocation the others view.
        const grid = &destination.grid;
        var box_writers: [Platform.MAX_NUM_DEVICES]u64 = undefined;
        for (grid.writers[0..grid.box_count]) |writers| {
            var mask = writers;
            while (mask != 0) : (mask &= mask - 1) box_writers[@ctz(mask)] = writers;
        }

        // Creation holds a reference to every shared allocation and its
        // views one each. Declared before the buffers' cleanup, so a failure
        // destroys the views first and this frees the owners.
        var shared: [Platform.MAX_NUM_DEVICES]?*SharedHostAllocation = @splat(null);
        defer for (shared[0..shard_count]) |allocation| if (allocation) |a| a.release();
        var pjrt_buffers: Buffer.Shards = .empty;
        errdefer for (pjrt_buffers.constSlice()) |buffer| buffer.deinit(api);
        for (ordered_devices, shards, 0..) |device, *shard, index| {
            // The front end only submits host kinds when every device has
            // a pinned host space.
            const host = (platform.devices[device.id].memory(.host_pinned) orelse return error.UnsupportedMemoryKind).pjrt_memory;
            if (grid.replicas() > 1 or destination.memory == .host_mapped) {
                const first: usize = @ctz(box_writers[index]);
                if (first == index) shared[index] = try .create(direct.io, platform, dims, element_type, layout, host);
                const allocation = shared[first].?;
                const space = (platform.devices[device.id].memory(destination.memory.space()) orelse return error.UnsupportedMemoryKind).pjrt_memory;
                pjrt_buffers.appendAssumeCapacity(try allocation.view(platform, dims, element_type, layout, space));
                shard.* = allocation.data;
            } else {
                const buffer = try createHostBuffer(direct.io, platform, dims, element_type, layout, host);
                pjrt_buffers.appendAssumeCapacity(buffer);
                shard.* = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
            }
        }

        destination.output.* = .fromPjrtBuffers(platform, destination.shape, pjrt_buffers.constSlice());
        return .{
            .host = shards,
            .host_total = local.byteSize() * shard_count,
        };
    }

    fn deinit(self: *TensorTransfer, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        for (self.targets) |target| target.manager.deinit(api);
        allocator.free(self.targets);
        allocator.free(self.host);
    }

    /// Every byte of every device went out, or was written in place.
    fn fullySubmitted(self: *const TensorTransfer) bool {
        for (self.targets) |target| {
            if (!target.fullySubmitted()) return false;
        }
        return self.host_written.load(.acquire) == self.host_total;
    }

    /// Where a host transfer's bytes go: the allocation its writers, the
    /// devices of one box, share. Writing it once writes every one of them.
    fn hostLanding(self: *const TensorTransfer, transfer: Batch.Plan.Transfer) []u8 {
        const first: usize = @ctz(transfer.writer_mask);
        std.debug.assert(first < self.host.len);
        return self.host[first][transfer.destination_offset..][0..transfer.len];
    }
};

/// One host allocation that the devices of a region read through views of
/// it, in their pinned host memory space or, for `host_mapped`, their
/// default one: the owner buffer is not a shard, every view holds a
/// reference that its deletion callback drops, and the last drop destroys
/// the owner. PJRT runs a view's callback once the view's memory is out of
/// use, after any execution still reading it, so the allocation outlives
/// its readers whatever order the shards are freed in, including views of
/// the shards (`Buffer.createView`) that their owner outlives. On the
/// process heap: the last view may go long after the loader.
/// Writing through one view shows on every device, so a shard of a
/// replicated host-resident destination must not be donated to an
/// execution that writes it, as with `Buffer.HostAccessible.init(.undonatable)`.
const SharedHostAllocation = struct {
    /// Allocations not released yet, counted in tests only.
    var live: if (builtin.is_test) std.atomic.Value(usize) else void = if (builtin.is_test) .init(0) else {};

    api: *const pjrt.Api,
    owner: *pjrt.Buffer,
    data: [*]u8,
    /// One per live view, and one for `TensorTransfer.initHost` while it
    /// creates them.
    references: std.atomic.Value(usize) = .init(1),

    fn create(
        io: std.Io,
        platform: *const Platform,
        dims: []const i64,
        element_type: pjrt.BufferType,
        layout: pjrt.MemoryLayout,
        memory: *const pjrt.Memory,
    ) TransferInitError!*SharedHostAllocation {
        const api = platform.pjrt_api;
        const self = try std.heap.smp_allocator.create(SharedHostAllocation);
        errdefer std.heap.smp_allocator.destroy(self);
        const owner = try createHostBuffer(io, platform, dims, element_type, layout, memory);
        errdefer owner.deinit(api);
        self.* = .{
            .api = api,
            .owner = owner,
            .data = @ptrCast(try owner.opaqueDeviceMemoryDataPointer(api)),
        };
        if (builtin.is_test) _ = live.fetchAdd(1, .monotonic);
        return self;
    }

    /// A shard in `memory`, holding a reference: the host space of its
    /// device, or the default one, where kernels read the allocation in
    /// place. Every device can read it.
    fn view(
        self: *SharedHostAllocation,
        platform: *const Platform,
        dims: []const i64,
        element_type: pjrt.BufferType,
        layout: pjrt.MemoryLayout,
        memory: *const pjrt.Memory,
    ) TransferInitError!*pjrt.Buffer {
        _ = self.references.fetchAdd(1, .monotonic);
        errdefer self.release();
        return platform.pjrt_client.createViewOfDeviceBuffer(platform.pjrt_api, .{
            .device_buffer_ptr = self.data,
            .dims = dims,
            .element_type = element_type,
            .layout = layout,
            .memory = memory,
            .on_delete_callback = &onViewDeleted,
            .on_delete_callback_arg = self,
        });
    }

    fn onViewDeleted(_: ?*anyopaque, context: ?*anyopaque) callconv(.c) void {
        const self: *SharedHostAllocation = @ptrCast(@alignCast(context.?));
        self.release();
    }

    fn release(self: *SharedHostAllocation) void {
        if (self.references.fetchSub(1, .acq_rel) != 1) return;
        self.owner.deinit(self.api);
        std.heap.smp_allocator.destroy(self);
        if (builtin.is_test) _ = live.fetchSub(1, .monotonic);
    }
};

/// Shared host allocations not released yet, in tests: the front end's
/// tests count the ones their views keep.
pub fn liveHostAllocations() usize {
    comptime std.debug.assert(builtin.is_test);
    return SharedHostAllocation.live.load(.acquire);
}

/// An uninitialized buffer in a host memory space, once its allocation
/// completed: its host pointer is valid only then.
fn createHostBuffer(
    io: std.Io,
    platform: *const Platform,
    dims: []const i64,
    element_type: pjrt.BufferType,
    layout: pjrt.MemoryLayout,
    memory: *const pjrt.Memory,
) TransferInitError!*pjrt.Buffer {
    const api = platform.pjrt_api;
    const buffer = try platform.pjrt_client.createUninitializedBuffer(api, .{
        .dims = dims,
        .element_type = element_type,
        .layout = layout,
        .dst = .{ .memory = memory },
    });
    errdefer buffer.deinit(api);
    const ready = buffer.readyEvent(api);
    defer ready.deinit(api);
    try ready.await(api, io);
    return buffer;
}

/// Builds immutable per-file source jobs and transfer records. Fairness is
/// decided here once; the scheduler only publishes and claims those jobs.
/// For DeepSeek-V4-Flash (148.65 GiB, 69,187 tensors), tensor-local reads made
/// about 69,445 source calls despite a 16 MiB request limit. Coalescing reduced
/// that to 9,524. A rigid grid still produced 78,665 DMA pieces; tensor-safe
/// cuts kept the same read count with 69,572 pieces. Per-tensor device buffers
/// remain intentional: reducing submissions below roughly tensors x destinations would require
/// packed device allocations or device-side scattering, not another read cut.
/// A composite destination is planned through its parts: each part maps its
/// source bytes to the destination's logical bytes, which the destination's
/// grid places (`ShardGrid`), and bytes no part covers come from the zero
/// block. A destination fed by strided parts is planned by its logical
/// ranges instead (`JobCut.appendGatherJobs`).
const Planner = struct {
    /// What every plan of a loader is cut by.
    const Config = struct {
        block_size: usize,
        request_size: usize,
        /// Every read of a file read directly starts and ends at a multiple
        /// of this; 0 reads exact tensor ranges.
        alignment: usize,
        /// Submissions a DMA job must save to be staged (`JobCut.stage`):
        /// one device's in-flight pieces.
        minimum_staging_gain: usize = max_dma_pieces_per_device,
    };

    /// How each file is read: aligned when it is local and the VFS that
    /// opens it reads it directly under `policy`, decided once per file and
    /// kept in its slot. A remote file is read exact without being opened
    /// here. Without a VFS (`io` is none, or no alignment) every file takes
    /// `alignment`. A gather job reads files other than its plan's, so it
    /// asks here for each.
    const Reads = struct {
        io: std.Io,
        vfs: ?*VFS = null,
        policy: VFS.DirectIo = .auto,
        alignment: usize,

        fn alignmentOf(self: Reads, slot: *SourceSlot) !usize {
            const vfs = self.vfs orelse return self.alignment;
            if (slot.alignment) |known| return known;
            const direct = vfs.isLocal(slot.uri) and vfs.useDirectIo(try slot.ensure(self.io), self.policy);
            const alignment = if (direct) self.alignment else 0;
            slot.alignment = alignment;
            return alignment;
        }
    };

    /// Plans `parts` one file at a time and publishes each plan as soon as
    /// it exists, so workers claim the first file while the rest is planned.
    /// Stops at the first error; the caller seals or fails the batch. With
    /// an alignment, a file is planned with widened reads only when the
    /// VFS the files are opened through reads it directly under `policy`: a
    /// widened read served from the page cache costs the DMA sources their
    /// block alignment for nothing, and an `io` that is no VFS never reads
    /// directly.
    /// Whole-model planning delayed the first read by 0.32 s for DeepSeek's
    /// 46 files on one MI300X. Publishing per file overlaps the rest with IO;
    /// Llama's four files took only 1-2 ms total on one B70, so this was
    /// throughput-neutral there.
    /// Widening buffered reads is not free: warm replicated Llama on four
    /// GB300 lost 3-6% when its first tensor no longer landed at block offset
    /// zero. Ask the VFS actually owning the handle (`VFS.fromIo`), both to
    /// avoid that cost and to avoid indexing another VFS's handle table.
    fn publishFiles(
        scheduler: *Scheduler,
        io: std.Io,
        batch: *Batch,
        parts: []Part,
        config: Config,
        policy: VFS.DirectIo,
    ) !void {
        const order = try sortedPartOrder(scheduler.allocator, parts);
        defer scheduler.allocator.free(order);
        const reads: Reads = .{
            .io = io,
            .vfs = if (config.alignment != 0) VFS.fromIo(io) else null,
            .policy = policy,
            .alignment = config.alignment,
        };
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(parts, order, file_start);
            batch.planning_source = parts[order[file_start]].source.file_uri;
            const planning_started: std.Io.Timestamp = .now(io, .awake);
            var file_config = config;
            file_config.alignment = try reads.alignmentOf(parts[order[file_start]].source_slot);
            const plan = try preparePlan(scheduler.allocator, parts, order[file_start..file_end], file_config, reads);
            plan.planning_ns = @intCast(@max(planning_started.untilNow(io, .awake).nanoseconds, 0));
            scheduler.publish(io, batch, plan) catch |err| {
                plan.destroy();
                return err;
            };
            file_start = file_end;
        }
    }

    /// Plans one file: `order` indexes `parts` sorted by offset then size,
    /// all on the same file. Runs of touching ranges are cut into the
    /// minimum number of jobs at part-safe boundaries; no job depends on
    /// another, since every DMA piece is submitted as soon as its block is
    /// read. The plan also carries the contexts its jobs need: one request
    /// per job, one block per job block, one event per DMA submission (a
    /// transfer's writer count). With an alignment, a job's read is widened
    /// to aligned bounds around its tensor range (`maximumJobLen` leaves
    /// room for it); the transfers address the widened read. A last fill
    /// job writes the zero fill of every destination whose first part is in
    /// this file.
    /// Host-resident destinations are written by the worker that reads
    /// them, not by DMA, so their parts are cut into jobs of their own
    /// (`Batch.Plan.Job.Kind`), merged back into file order with the DMA
    /// jobs, and their zero fill is a fill job of its own too. A gathered
    /// destination is planned whole, by logical ranges, with the file of
    /// its first part (`JobCut.appendGatherJobs`); `reads` gives the
    /// alignment of the other files its jobs read.
    fn preparePlan(
        allocator: std.mem.Allocator,
        parts: []Part,
        order: []const usize,
        config: Config,
        reads: Reads,
    ) !*Batch.Plan {
        const source_slot = parts[order[0]].source_slot;
        var host_parts: usize = 0;
        var gathered_parts: usize = 0;
        for (order) |part_index| {
            const part = &parts[part_index];
            std.debug.assert(std.mem.eql(u8, part.source.file_uri, parts[order[0]].source.file_uri));
            const destination = part.destination;
            if (destination.gathered) {
                gathered_parts += 1;
            } else if (destination.hostResident()) {
                host_parts += 1;
            }
        }

        // The DMA parts and the host-resident parts, each in file order;
        // gathered parts are planned by destination.
        const read_parts = order.len - gathered_parts;
        const class_order = try allocator.alloc(usize, read_parts);
        defer allocator.free(class_order);
        var device_cursor: usize = 0;
        var host_cursor: usize = read_parts - host_parts;
        for (order) |part_index| {
            const destination = parts[part_index].destination;
            if (destination.gathered) continue;
            if (destination.hostResident()) {
                class_order[host_cursor] = part_index;
                host_cursor += 1;
            } else {
                class_order[device_cursor] = part_index;
                device_cursor += 1;
            }
        }

        // A gather job's extents may be in files read directly while this
        // one is not, so its ranges leave room for the widest alignment.
        var gather_config = config;
        gather_config.alignment = reads.alignment;
        var cut: JobCut = .{
            .allocator = allocator,
            .parts = parts,
            .config = config,
            .reads = reads,
            .maximum_job_len = try maximumJobLen(config),
            .gather_len = try maximumJobLen(gather_config),
            .read_limit = readLimit(config),
        };
        defer cut.deinit();
        try cut.cutRuns(class_order[0 .. read_parts - host_parts], false);
        try cut.cutRuns(class_order[read_parts - host_parts ..], true);
        for (order) |part_index| {
            const part = &parts[part_index];
            if (part.destination.gathered and part == &part.destination.parts[0]) try cut.appendGatherJobs(part.destination);
        }
        // Jobs are claimed in planning order, which is file order (a
        // gather job by its first read). A byte-fair order across
        // destination devices was measured against it on the two hosts
        // where the DMA pump is not the ceiling and made no difference
        // (sixteenth pass). Each class is in file order already; the
        // stable sort interleaves them.
        std.mem.sort(Batch.Plan.Job, cut.jobs.items, {}, struct {
            fn lessThan(_: void, lhs: Batch.Plan.Job, rhs: Batch.Plan.Job) bool {
                return lhs.file_offset < rhs.file_offset;
            }
        }.lessThan);

        // The zero fill of each destination goes out once, with the plan
        // of its first part; it depends on no read, so one job after this
        // file's reads carries all of it, one per delivery class.
        try cut.appendFillJob(order, false);
        try cut.appendFillJob(order, true);

        const plan = plan: {
            const transfers = try cut.transfers.toOwnedSlice(allocator);
            errdefer allocator.free(transfers);
            break :plan try Batch.Plan.create(
                allocator,
                source_slot,
                cut.jobs.items.len,
                transfers,
                cut.dma_submissions,
                cut.source_bytes,
            );
        };
        errdefer plan.destroy();
        plan.items = try cut.items.toOwnedSlice(allocator);
        plan.landings = try cut.landings.toOwnedSlice(allocator);
        plan.extents = try cut.extents.toOwnedSlice(allocator);
        @memcpy(plan.jobs, cut.jobs.items);
        plan.direct_reads = config.alignment != 0;
        return plan;
    }

    /// `preparePlan` with every file read at `config.alignment`: for tests.
    fn prepareTestPlan(allocator: std.mem.Allocator, parts: []Part, order: []const usize, config: Config) !*Batch.Plan {
        return preparePlan(allocator, parts, order, config, .{ .io = undefined, .alignment = config.alignment });
    }

    /// The jobs and transfers of one plan as `preparePlan` cuts them.
    const JobCut = struct {
        /// Host-resident transfers are written by the worker, not cut by
        /// blocks: one transfer per destination run of a job, its
        /// `block_offset` relative to the job's read start, `block_index`
        /// zero.
        const unblocked = std.math.maxInt(usize);

        /// One part's bytes in a read job: the logical range `[start, end)`
        /// of its destination, from `file_offset` of the file on.
        const Piece = struct { destination: *Destination, start: u64, end: u64, file_offset: u64 };
        /// A piece's bytes on box `box` of its destination: `len` from
        /// `box_offset`, and where a staged job lands them.
        const Entry = struct { piece: u32, box: u32, group: u32 = 0, box_offset: u64, len: u64, landing: u64 = 0 };
        /// What a staged job lands together.
        const Group = struct { destination: *Destination, box: u32 };

        allocator: std.mem.Allocator,
        parts: []Part,
        config: Config,
        reads: Reads,
        maximum_job_len: usize,
        /// The longest logical range of a gather job, and its longest read
        /// (`readLimit`).
        gather_len: usize,
        read_limit: usize,
        jobs: std.ArrayList(Batch.Plan.Job) = .empty,
        transfers: std.ArrayList(Batch.Plan.Transfer) = .empty,
        items: std.ArrayList(Batch.Plan.Item) = .empty,
        landings: std.ArrayList(u64) = .empty,
        extents: std.ArrayList(Batch.Plan.Extent) = .empty,
        safe_boundaries: std.ArrayList(u64) = .empty,
        /// A read job's pieces before they are placed, and the staging
        /// scratch.
        pieces: std.ArrayList(Piece) = .empty,
        entries: std.ArrayList(Entry) = .empty,
        groups: std.AutoHashMapUnmanaged(Group, u32) = .empty,
        source_bytes: u64 = 0,
        /// One event per writer of a DMA transfer.
        dma_submissions: usize = 0,

        fn deinit(self: *JobCut) void {
            const allocator = self.allocator;
            self.groups.deinit(allocator);
            self.entries.deinit(allocator);
            self.pieces.deinit(allocator);
            self.safe_boundaries.deinit(allocator);
            self.extents.deinit(allocator);
            self.landings.deinit(allocator);
            self.items.deinit(allocator);
            self.transfers.deinit(allocator);
            self.jobs.deinit(allocator);
        }

        /// Cuts the runs of `order` (part indices of one delivery class in
        /// file order) into jobs.
        fn cutRuns(self: *JobCut, order: []const usize, host: bool) !void {
            const allocator = self.allocator;
            const parts = self.parts;
            const alignment = self.config.alignment;
            const maximum_job_len = self.maximum_job_len;
            var run_cursor: usize = 0;
            while (run_cursor < order.len) {
                self.safe_boundaries.clearRetainingCapacity();
                const first_range = try parts[order[run_cursor]].fileRange();
                const first_offset = first_range.start;
                var run_end = first_range.end;
                if (run_end == first_offset) {
                    run_cursor += 1;
                    continue;
                }
                var run_part_end = run_cursor + 1;
                while (run_part_end < order.len) : (run_part_end += 1) {
                    const candidate = try parts[order[run_part_end]].fileRange();
                    if (candidate.start > run_end) break;
                    // A touching range starts where every preceding range has
                    // ended. Unlike an arbitrary part end, this is safe even
                    // when the batch contains overlapping or duplicate ranges.
                    if (candidate.end != candidate.start and candidate.start == run_end and
                        (self.safe_boundaries.items.len == 0 or
                            self.safe_boundaries.items[self.safe_boundaries.items.len - 1] != candidate.start))
                    {
                        try self.safe_boundaries.append(allocator, candidate.start);
                    }
                    run_end = @max(run_end, candidate.end);
                }

                var job_start = first_offset;
                var candidate_start = run_cursor;
                var boundary_cursor: usize = 0;
                const run_len = run_end - first_offset;
                const maximum_job_len_u64: u64 = @intCast(maximum_job_len);
                var jobs_remaining: usize = @intCast(run_len / maximum_job_len_u64 +
                    @intFromBool(run_len % maximum_job_len_u64 != 0));
                while (jobs_remaining != 0) {
                    const safe_boundaries = self.safe_boundaries.items;
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
                        ) catch return error.Internal;
                        const minimum_end = @max(job_start + 1, run_end - remaining_capacity);
                        const maximum_end = @min(
                            hard_end,
                            run_end - @as(u64, @intCast(jobs_remaining - 1)),
                        );
                        while (boundary_cursor < safe_boundaries.len and
                            safe_boundaries[boundary_cursor] <= job_start)
                        {
                            boundary_cursor += 1;
                        }
                        var scan = boundary_cursor;
                        var selected: ?u64 = null;
                        while (scan < safe_boundaries.len and
                            safe_boundaries[scan] <= maximum_end) : (scan += 1)
                        {
                            if (safe_boundaries[scan] >= minimum_end)
                                selected = safe_boundaries[scan];
                        }
                        boundary_cursor = scan;
                        break :boundary selected orelse maximum_end;
                    };
                    const job_len: usize = @intCast(job_end - job_start);
                    const read_start = if (alignment == 0) job_start else std.mem.alignBackward(u64, job_start, alignment);
                    const read_end = if (alignment == 0) job_end else std.mem.alignBackward(
                        u64,
                        std.math.add(u64, job_end, alignment - 1) catch {
                            load_log.err("{s}: a read ending at byte {d} overflows when widened to the {d}-byte alignment", .{ parts[order[run_cursor]].source.file_uri, job_end, alignment });
                            return error.InvalidTensorRange;
                        },
                        alignment,
                    );
                    const read_len: usize = @intCast(read_end - read_start);
                    const transfer_start = self.transfers.items.len;
                    // The job's pieces first: where they land in its
                    // blocks depends on all of them.
                    self.pieces.clearRetainingCapacity();
                    while (candidate_start < run_part_end) {
                        const candidate = try parts[order[candidate_start]].fileRange();
                        if (candidate.end > job_start) break;
                        candidate_start += 1;
                    }
                    for (order[candidate_start..run_part_end]) |part_index| {
                        const part = &parts[part_index];
                        const range = try part.fileRange();
                        if (range.start >= job_end) break;
                        const intersection_start = @max(job_start, range.start);
                        const intersection_end = @min(job_end, range.end);
                        if (intersection_start >= intersection_end) continue;
                        const logical_start = part.dst_offset + (intersection_start - range.start);
                        try self.pieces.append(allocator, .{
                            .destination = part.destination,
                            .start = logical_start,
                            .end = logical_start + (intersection_end - intersection_start),
                            .file_offset = intersection_start,
                        });
                    }
                    std.debug.assert(self.pieces.items.len != 0);
                    const read: Batch.Plan.Job.Read = .{ .len = read_len, .minimum_len = @intCast(job_end - read_start) };
                    const kind: Batch.Plan.Job.Kind = if (host) host: {
                        for (self.pieces.items) |piece| try self.appendRunTransfers(transfer_start, piece, read_start, unblocked);
                        break :host if (alignment == 0 and coversReadInOrder(self.transfers.items[transfer_start..], read_len))
                            .{ .host_direct = read }
                        else
                            .{ .host_bounce = read };
                    } else try self.placeDma(read, read_start, job_start, job_len);
                    const job_transfers = self.transfers.items[transfer_start..];
                    if (!host) self.dma_submissions += submissionCount(job_transfers);
                    try self.jobs.append(allocator, .{
                        .file_offset = read_start,
                        .transfer_start = transfer_start,
                        .transfer_len = job_transfers.len,
                        .kind = kind,
                    });
                    self.source_bytes +|= @intCast(job_len);
                    job_start = job_end;
                    jobs_remaining -= 1;
                }
                std.debug.assert(job_start == run_end);
                run_cursor = run_part_end;
            }
        }

        /// Places a DMA job's pieces (`pieces`, in file order) in its
        /// blocks. Blocks holding the read in file order give one transfer
        /// per run (`ShardGrid.run`): a tensor sharded along any axis but
        /// its first takes one per row and device, as a column-sharded
        /// `o_proj` or `down_proj` does. A staged job lands its read grouped
        /// by destination box instead (`stage`), so a box takes one piece
        /// per block. Llama-3.1-8B sharded like that on four B70, page
        /// cache warm, planned 1.05M submissions and loaded in 5.1 s (~1.2
        /// ms per piece, 64 in flight per device); staged, 5,468 and 0.71
        /// s, against 0.67 s sharded by rows. Staging costs a read
        /// scattered over the runs, or for a direct read a copy through the
        /// worker's bounce buffer (`landJob`), so a job is staged only where
        /// it saves `Config.minimum_staging_gain` submissions, and only when
        /// its pieces tile its bytes once: overlapping parts send a source
        /// byte to two pieces, and one read cannot land it twice.
        fn placeDma(self: *JobCut, read: Batch.Plan.Job.Read, read_start: u64, data_start: u64, job_len: usize) !Batch.Plan.Job.Kind {
            const block_size = self.config.block_size;
            const transfer_start = self.transfers.items.len;
            // File order takes a submission per writer of every run, cut by
            // blocks, less where runs of adjacent pieces continue each other
            // or a run ends where a block does: at most `slack` fewer.
            var in_file_order: usize = 0;
            var slack: usize = 0;
            var tiles = true;
            var next = data_start;
            for (self.pieces.items) |piece| {
                if (piece.file_offset != next) tiles = false;
                next = piece.file_offset + (piece.end - piece.start);
                const grid = &piece.destination.grid;
                const landing = piece.file_offset - read_start;
                const crossings = (landing + piece.end - piece.start - 1) / block_size - landing / block_size;
                in_file_order += (grid.runCount(piece.start, piece.end) + crossings) * grid.replicas();
                slack += (1 + crossings) * grid.replicas();
            }
            if (tiles and next == data_start + job_len) {
                if (try self.stage(read, read_start, data_start, job_len, in_file_order, slack)) |staged| return .{ .dma_staged = staged };
            }
            for (self.pieces.items) |piece| try self.appendRunTransfers(transfer_start, piece, read_start, block_size);
            return .{ .dma_read = read };
        }

        /// Lands the job's pieces box by box: their bytes on each box of a
        /// destination, grouped by (destination, box) in order of first
        /// appearance, each group in box order. One `Item` per piece tells
        /// the worker where its bytes go (`landJob`). Null, with nothing
        /// appended, when that saves fewer than
        /// `Config.minimum_staging_gain` submissions against file order,
        /// estimated at `in_file_order` and at most `slack` fewer: counted
        /// exactly, by listing the runs, only when that decides.
        fn stage(self: *JobCut, read: Batch.Plan.Job.Read, read_start: u64, data_start: u64, job_len: usize, in_file_order: usize, slack: usize) !?Batch.Plan.Job.Staged {
            const allocator = self.allocator;
            const block_size = self.config.block_size;
            const pieces = self.pieces.items;
            self.entries.clearRetainingCapacity();
            self.groups.clearRetainingCapacity();
            for (pieces, 0..) |piece, index| {
                const grid = &piece.destination.grid;
                for (0..grid.box_count) |box| {
                    const low = grid.rankIn(box, piece.start);
                    const high = grid.rankIn(box, piece.end);
                    if (high == low) continue;
                    const group = try self.groups.getOrPut(allocator, .{ .destination = piece.destination, .box = @intCast(box) });
                    if (!group.found_existing) group.value_ptr.* = @intCast(self.groups.count() - 1);
                    try self.entries.append(allocator, .{ .piece = @intCast(index), .box = @intCast(box), .group = group.value_ptr.*, .box_offset = low, .len = high - low });
                }
            }
            const entries = self.entries.items;
            std.mem.sort(Entry, entries, {}, struct {
                fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
                    if (lhs.group != rhs.group) return lhs.group < rhs.group;
                    if (lhs.box_offset != rhs.box_offset) return lhs.box_offset < rhs.box_offset;
                    return lhs.piece < rhs.piece;
                }
            }.lessThan);
            // The read's own order when every box of a piece is one run and
            // the groups come in the order of the runs.
            var runs: u64 = 0;
            for (pieces) |piece| runs += piece.destination.grid.runCount(piece.start, piece.end);
            if (runs == entries.len) {
                const in_order = for (entries[1..], entries[0 .. entries.len - 1]) |entry, previous| {
                    if (entry.piece < previous.piece or (entry.piece == previous.piece and entry.box < previous.box)) break false;
                } else true;
                if (in_order) return null;
            }
            const staged_start = self.transfers.items.len;
            var landing: u64 = 0;
            for (entries) |*entry| {
                const destination = pieces[entry.piece].destination;
                entry.landing = landing;
                try appendBlocked(allocator, &self.transfers, staged_start, .{
                    .destination = destination,
                    .block_index = 0,
                    .block_offset = 0,
                    .writer_mask = destination.grid.writers[entry.box],
                    .destination_offset = @intCast(entry.box_offset),
                    .len = @intCast(entry.len),
                }, @intCast(landing), block_size);
                landing += entry.len;
            }
            std.debug.assert(landing == job_len);
            const needed = submissionCount(self.transfers.items[staged_start..]) +| self.config.minimum_staging_gain;
            var file_order_wins = needed > in_file_order;
            if (!file_order_wins and needed +| slack > in_file_order) {
                const exact_start = self.transfers.items.len;
                for (pieces) |piece| try self.appendRunTransfers(exact_start, piece, read_start, block_size);
                file_order_wins = needed > submissionCount(self.transfers.items[exact_start..]);
                self.transfers.shrinkRetainingCapacity(exact_start);
            }
            if (file_order_wins) {
                self.transfers.shrinkRetainingCapacity(staged_start);
                return null;
            }

            // A landing table per piece, one entry per box of its destination.
            const item_start = self.items.items.len;
            for (pieces) |piece| {
                const landing_start = self.landings.items.len;
                try self.landings.appendNTimes(allocator, 0, piece.destination.grid.box_count);
                try self.items.append(allocator, .{
                    .destination = piece.destination,
                    .lattice = .{ .src_offset = piece.file_offset, .dst_offset = piece.start, .width = piece.end - piece.start },
                    .clip_start = piece.start,
                    .clip_end = piece.end,
                    .landing_start = landing_start,
                });
            }
            for (entries) |entry| {
                const item = self.items.items[item_start + entry.piece];
                self.landings.items[item.landing_start + entry.box] = entry.landing -% entry.box_offset;
            }
            return .{
                .read = read,
                .data_offset = @intCast(data_start - read_start),
                .item_start = item_start,
                .item_len = pieces.len,
            };
        }

        /// Appends `piece`'s transfers with its bytes where the read holds
        /// them, one per run (`ShardGrid.run`), merged with the previous
        /// transfer from `transfer_start` on where block, writers and both
        /// offsets continue it: parts adjacent in the file and in the
        /// destination go out as one piece. `block_size` cuts them, or
        /// `unblocked` for a host job.
        fn appendRunTransfers(self: *JobCut, transfer_start: usize, piece: Piece, read_start: u64, block_size: usize) !void {
            const destination = piece.destination;
            const grid = &destination.grid;
            var at = piece.start;
            while (at < piece.end) {
                const stop = grid.runEnd(at, piece.end);
                try appendBlocked(self.allocator, &self.transfers, transfer_start, .{
                    .destination = destination,
                    .block_index = 0,
                    .block_offset = 0,
                    .writer_mask = grid.writers[grid.boxOf(at)],
                    .destination_offset = @intCast(grid.boxOffset(at)),
                    .len = @intCast(stop - at),
                }, @intCast(piece.file_offset + (at - piece.start) - read_start), block_size);
                at = stop;
            }
        }

        /// Plans a gathered destination (`Destination.gathered`) whole: its
        /// logical bytes cut into ranges of at most `gather_len`, each one
        /// job. For every part with runs in the range the job reads the
        /// source range those runs come from (an `Extent`, from whichever
        /// file holds it), and it lands the range in its blocks box by box,
        /// one transfer per box and block, its bytes no part covers zero.
        /// A part whose source runs are out of order (a broadcast repeats
        /// them, a transpose permutes them) reads its whole source range
        /// in every job, which must fit one read: `UnsupportedLayout`
        /// otherwise. An interleave of members reads each member's rows of
        /// the range, so the range lands in one piece per box and block
        /// whatever the interleave's granularity.
        fn appendGatherJobs(self: *JobCut, destination: *Destination) !void {
            const allocator = self.allocator;
            const block_size = self.config.block_size;
            const grid = &destination.grid;
            const parts = destination.parts;
            // reach[i]: the furthest destination byte parts[0..=i] reach, so
            // a range's first candidate is found by bisection.
            const reach = try allocator.alloc(u64, parts.len);
            defer allocator.free(reach);
            var furthest: u64 = 0;
            for (parts, reach) |*part, *at| {
                const spans = part.lattice().spans().?;
                furthest = @max(furthest, part.dst_offset + spans.dst);
                at.* = furthest;
            }
            var start: u64 = 0;
            while (start < grid.total) {
                const end = @min(grid.total, start + self.gather_len);
                const transfer_start = self.transfers.items.len;
                const item_start = self.items.items.len;
                const extent_start = self.extents.items.len;
                const landing_start = self.landings.items.len;
                try self.landings.appendNTimes(allocator, 0, grid.box_count);
                var landing: u64 = 0;
                for (0..grid.box_count) |box| {
                    const low = grid.rankIn(box, start);
                    const high = grid.rankIn(box, end);
                    if (high == low) continue;
                    self.landings.items[landing_start + box] = landing -% low;
                    try appendBlocked(allocator, &self.transfers, transfer_start, .{
                        .destination = destination,
                        .block_index = 0,
                        .block_offset = 0,
                        .writer_mask = grid.writers[box],
                        .destination_offset = @intCast(low),
                        .len = @intCast(high - low),
                    }, @intCast(landing), block_size);
                    landing += high - low;
                }
                std.debug.assert(landing == end - start);

                var covered: u64 = 0;
                var index = std.sort.partitionPoint(u64, reach, start, struct {
                    fn below(limit: u64, at: u64) bool {
                        return at <= limit;
                    }
                }.below);
                while (index < parts.len and parts[index].dst_offset < end) : (index += 1) {
                    const part = &parts[index];
                    const lattice = part.lattice();
                    const first = lattice.firstEndingAfter(start);
                    const last = lattice.firstStartingFrom(end);
                    if (first >= last) continue;
                    covered += lattice.coveredIn(start, end);
                    try self.appendExtent(part, lattice, first, last, start, end, landing_start);
                }
                std.debug.assert(covered <= end - start);
                const job_len: usize = @intCast(end - start);
                const extent_len = self.extents.items.len - extent_start;
                try self.jobs.append(allocator, .{
                    .file_offset = if (extent_len != 0) self.extents.items[extent_start].file_offset else 0,
                    .transfer_start = transfer_start,
                    .transfer_len = self.transfers.items.len - transfer_start,
                    .kind = .{ .dma_gather = .{
                        .len = job_len,
                        .item_start = item_start,
                        .item_len = self.items.items.len - item_start,
                        .extent_start = extent_start,
                        .extent_len = extent_len,
                        .zeroed = covered < end - start,
                    } },
                });
                self.dma_submissions += submissionCount(self.transfers.items[transfer_start..]);
                start = end;
            }
        }

        /// The read of `part`'s runs `first` to `last` (exclusive) clipped
        /// to `[start, end)`, and the item that places them: from the first
        /// run's first byte in the range to the last run's last one when
        /// the part's source runs are in order, else the part's whole
        /// source range.
        fn appendExtent(self: *JobCut, part: *const Part, lattice: backend.Lattice, first: u64, last: u64, start: u64, end: u64, landing_start: usize) !void {
            const width = part.width;
            var source_start = part.src_offset;
            var source_end = part.src_offset + lattice.spans().?.src;
            const ordered = lattice.srcOrdered();
            if (ordered) {
                const head = lattice.at(first);
                const tail = lattice.at(last - 1);
                source_start = head.src + (start -| head.dst);
                source_end = tail.src + width - ((tail.dst + width) -| end);
            }
            const file_start = std.math.add(u64, part.source.offset, source_start) catch return part.rangeError(source_start);
            const file_end = std.math.add(u64, part.source.offset, source_end) catch return part.rangeError(source_end);
            const alignment = try self.reads.alignmentOf(part.source_slot);
            const read_start = if (alignment == 0) file_start else std.mem.alignBackward(u64, file_start, alignment);
            const read_end = if (alignment == 0) file_end else std.mem.alignForward(u64, file_end, alignment);
            // A range in order is clipped to the job's, and `checkGatherReads`
            // refused a part whose whole range does not fit.
            std.debug.assert(read_end - read_start <= self.read_limit);
            try self.extents.append(self.allocator, .{
                .source_slot = part.source_slot,
                .file_offset = read_start,
                .len = @intCast(read_end - read_start),
                .minimum_len = @intCast(file_end - read_start),
                .scatter = alignment == 0 and ordered and lattice.srcTiled(),
            });
            try self.items.append(self.allocator, .{
                .destination = part.destination,
                .lattice = .{ .src_offset = part.source.offset + part.src_offset, .dst_offset = part.dst_offset, .width = width, .dims = part.dims },
                .first = first,
                .end = last,
                .clip_start = start,
                .clip_end = end,
                .landing_start = landing_start,
            });
            self.source_bytes +|= file_end - file_start;
        }

        /// Appends `piece` landing at `landing` of the job's blocks taken
        /// end to end, cut at block boundaries, merged with the previous
        /// transfer from `transfer_start` on where block, writers and both
        /// offsets continue it.
        fn appendBlocked(
            allocator: std.mem.Allocator,
            output: *std.ArrayList(Batch.Plan.Transfer),
            transfer_start: usize,
            piece: Batch.Plan.Transfer,
            landing: usize,
            block_size: usize,
        ) !void {
            var cursor: usize = 0;
            while (cursor < piece.len) {
                const at = landing + cursor;
                const block_index = at / block_size;
                const block_offset = at % block_size;
                const take = @min(piece.len - cursor, block_size - block_offset);
                const destination_offset = piece.destination_offset + cursor;
                cursor += take;
                if (output.items.len > transfer_start) {
                    const previous = &output.items[output.items.len - 1];
                    if (previous.destination == piece.destination and previous.block_index == block_index and
                        previous.writer_mask == piece.writer_mask and
                        previous.block_offset + previous.len == block_offset and
                        previous.destination_offset + previous.len == destination_offset)
                    {
                        previous.len += take;
                        continue;
                    }
                }
                try output.append(allocator, .{
                    .destination = piece.destination,
                    .block_index = block_index,
                    .block_offset = block_offset,
                    .writer_mask = piece.writer_mask,
                    .destination_offset = destination_offset,
                    .len = take,
                });
            }
        }

        fn submissionCount(transfers: []const Batch.Plan.Transfer) usize {
            var count: usize = 0;
            for (transfers) |transfer| count += @popCount(transfer.writer_mask);
            return count;
        }

        /// The zero fill of the destinations of one delivery class whose
        /// first part is among this file's (`order`), as one job. A DMA
        /// fill job reads one block, the zero block; a host one writes zeros
        /// in place. A gathered destination's jobs write its zeros
        /// themselves.
        fn appendFillJob(self: *JobCut, order: []const usize, host: bool) !void {
            const fill_start = self.transfers.items.len;
            for (order) |part_index| {
                const part = &self.parts[part_index];
                const destination = part.destination;
                if (part != &destination.parts[0] or destination.gathered or destination.hostResident() != host) continue;
                try appendFillTransfers(self.allocator, &self.transfers, fill_start, destination, if (host) unblocked else self.config.block_size);
            }
            const fill_transfers = self.transfers.items[fill_start..];
            if (fill_transfers.len == 0) return;
            try self.jobs.append(self.allocator, .{
                .file_offset = 0,
                .transfer_start = fill_start,
                .transfer_len = fill_transfers.len,
                .kind = if (host) .host_fill else .dma_fill,
            });
            if (!host) self.dma_submissions += submissionCount(fill_transfers);
        }

        /// Whether a host job's transfers address its read from byte zero
        /// to `read_len` in order, each source byte once: then the read
        /// can scatter straight into the destinations. Overlapping or
        /// duplicate parts address some source bytes twice and go through
        /// blocks instead.
        fn coversReadInOrder(transfers: []const Batch.Plan.Transfer, read_len: usize) bool {
            var next: usize = 0;
            for (transfers) |transfer| {
                std.debug.assert(transfer.block_index == 0);
                if (transfer.block_offset != next) return false;
                next += transfer.len;
            }
            return next == read_len;
        }
    };
    fn blockCount(len: usize, block_size: usize) usize {
        return len / block_size + @intFromBool(len % block_size != 0);
    }

    /// The longest read of a gather job's extent: a request, unwidened.
    fn readLimit(config: Config) usize {
        return config.request_size;
    }

    /// A strided part whose source runs are not back to back reads its whole
    /// source range in every job (`JobCut.appendExtent`): refused before
    /// anything is published when that range does not fit one read.
    fn checkGatherReads(config: Config, part: *const Part) error{UnsupportedLayout}!void {
        const lattice = part.lattice();
        if (lattice.srcTiled()) return;
        const read_limit = readLimit(config);
        const alignment = @max(config.alignment, 1);
        const start = part.source.offset + part.src_offset;
        const end = start + lattice.spans().?.src;
        if (std.mem.alignForward(u64, end, alignment) - std.mem.alignBackward(u64, start, alignment) > read_limit) {
            load_log.err("destination {f} from {s}: a strided part reads {d} source bytes for every job, more than a read of {d}", .{ part.destination.shape, part.source.name, end - start, read_limit });
            return error.UnsupportedLayout;
        }
    }

    /// The longest tensor range one job may cover: the request size, less
    /// the two alignment units its widened read can add, so a widened job
    /// still fits a request (a credit's slot).
    fn maximumJobLen(config: Config) error{InvalidOptions}!usize {
        const unpadded = config.request_size;
        if (unpadded == 0) return error.InvalidOptions;
        if (config.alignment == 0) return unpadded;
        if (!std.math.isPowerOfTwo(config.alignment) or config.alignment > (unpadded - 1) / 2) return error.InvalidOptions;
        return unpadded - 2 * config.alignment;
    }

    /// Part indices sorted by file URI, offset, size and index: the planner's
    /// input, taken one file group at a time (`fileGroupEnd`). An offset
    /// that overflows sorts last; the planner reports it.
    fn sortedPartOrder(allocator: std.mem.Allocator, parts: []const Part) ![]usize {
        const order = try allocator.alloc(usize, parts.len);
        for (order, 0..) |*index, i| index.* = i;
        const SortContext = struct {
            parts: []const Part,

            fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                const left = ctx.parts[lhs];
                const right = ctx.parts[rhs];
                const uri_order = std.mem.order(u8, left.source.file_uri, right.source.file_uri);
                if (uri_order != .eq) return uri_order == .lt;
                const left_offset = left.source.offset +| left.src_offset;
                const right_offset = right.source.offset +| right.src_offset;
                if (left_offset != right_offset) return left_offset < right_offset;
                if (left.width != right.width) return left.width < right.width;
                return lhs < rhs;
            }
        };
        std.mem.sort(usize, order, SortContext{ .parts = parts }, SortContext.lessThan);
        return order;
    }

    /// The end of the file group that begins at `order[start]`.
    fn fileGroupEnd(parts: []const Part, order: []const usize, start: usize) usize {
        const uri = parts[order[start]].source.file_uri;
        var end = start + 1;
        while (end < order.len and std.mem.eql(u8, uri, parts[order[end]].source.file_uri)) : (end += 1) {}
        return end;
    }

    /// Appends the zero-fill transfers of `destination`: every logical
    /// range no part covers (the gaps between its parts and the tail),
    /// placed by its grid like read bytes, each piece read from the zero
    /// block at offset 0. Pieces count toward their targets' totals, so the
    /// fill completes each device's placement bytes alongside the parts. A
    /// piece is at most the zero block, one block of the pinned set.
    fn appendFillTransfers(
        allocator: std.mem.Allocator,
        output: *std.ArrayList(Batch.Plan.Transfer),
        transfer_start: usize,
        destination: *Destination,
        block_size: usize,
    ) !void {
        var covered_end: u64 = 0;
        for (destination.parts) |part| {
            if (part.dst_offset > covered_end) try appendFillRange(allocator, output, transfer_start, destination, covered_end, part.dst_offset, block_size);
            covered_end = part.dst_offset + part.width;
        }
        if (covered_end < destination.grid.total) try appendFillRange(allocator, output, transfer_start, destination, covered_end, destination.grid.total, block_size);
    }

    /// The fill of the logical range `[start, end)`: one range of each box
    /// it touches, cut by the zero block, merged where one writer set's
    /// pieces are contiguous.
    fn appendFillRange(
        allocator: std.mem.Allocator,
        output: *std.ArrayList(Batch.Plan.Transfer),
        transfer_start: usize,
        destination: *Destination,
        start: u64,
        end: u64,
        block_size: usize,
    ) !void {
        const grid = &destination.grid;
        for (0..grid.box_count) |box| {
            var destination_offset: usize = @intCast(grid.rankIn(box, start));
            var remaining: usize = @intCast(grid.rankIn(box, end) - destination_offset);
            const writer_mask = grid.writers[box];
            while (remaining != 0) {
                if (output.items.len > transfer_start) merge: {
                    const previous = &output.items[output.items.len - 1];
                    if (previous.destination != destination or previous.writer_mask != writer_mask or
                        previous.destination_offset + previous.len != destination_offset or
                        previous.len == block_size)
                        break :merge;
                    const grown = @min(remaining, block_size - previous.len);
                    previous.len += grown;
                    destination_offset += grown;
                    remaining -= grown;
                    continue;
                }
                const take = @min(remaining, block_size);
                try output.append(allocator, .{
                    .destination = destination,
                    .block_index = 0,
                    .block_offset = 0,
                    .writer_mask = writer_mask,
                    .destination_offset = destination_offset,
                    .len = take,
                });
                destination_offset += take;
                remaining -= take;
            }
        }
    }
};

/// A strict FIFO of published plans. A submission holds one plan per source
/// file, published as soon as that file is planned, so a plan reaches the
/// queue behind every earlier plan of its own submission and of every
/// earlier one (submissions publish and seal on one task). Within a plan,
/// jobs are handed out in the planned order (fair by destination-device
/// bytes); a plan leaves the queue with its last claim or through `fail`.
/// A plan is queued only with jobs left and its completion units are added
/// in the same critical section, so a queued plan's batch still holds at
/// least one unit and can never be freed under the scheduler.
const Scheduler = struct {
    /// A claimed job: its position in the plan that owns it. The claim holds
    /// one of the batch's completion units until the request releases it.
    const Claim = struct {
        plan: *Batch.Plan,
        index: usize,
    };

    allocator: std.mem.Allocator,
    /// Plans with unclaimed jobs in publish order.
    queue: std.Deque(*Batch.Plan) = .empty,
    unclaimed_total: usize = 0,
    stopping: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn init(allocator: std.mem.Allocator) Scheduler {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Scheduler) void {
        // Every batch was awaited (and so claimed out or retired) first.
        std.debug.assert(self.queue.len == 0);
        self.queue.deinit(self.allocator);
        self.* = undefined;
    }

    /// Publishes one plan of an open submission behind every earlier plan.
    /// A plan without jobs never joins the queue; it only counts in the
    /// diagnostics.
    fn publish(self: *Scheduler, io: std.Io, batch: *Batch, plan: *Batch.Plan) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.Closed;
        std.debug.assert(batch.diagnostics.sealed_at == null);
        const job_count = plan.jobs.len;
        try batch.plans.ensureUnusedCapacity(batch.allocator, 1);
        if (job_count != 0) try self.queue.ensureUnusedCapacity(self.allocator, 1);
        // Nothing below fails: the plan, its units and the queue entry
        // appear together.
        if (batch.diagnostics.published_at == null) batch.diagnostics.published_at = .now(io, .awake);
        plan.batch = batch;
        batch.plans.appendAssumeCapacity(plan);
        _ = batch.remaining.fetchAdd(job_count, .acq_rel);
        if (job_count != 0) {
            self.queue.pushBackAssumeCapacity(plan);
            self.unclaimed_total += job_count;
            self.condition.broadcast(io);
        }
    }

    fn stop(self: *Scheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
    }

    /// Stops claims and retires the unclaimed units of every queued plan so
    /// its batch still reaches `done` through its claimed requests (an open
    /// one through its seal and sentinel as well). Claims and this pass both
    /// move cursors under the mutex, so they partition the jobs exactly:
    /// every claimed job keeps its unit with its worker. A batch's queued
    /// plans are contiguous here, so the retire that may complete it is the
    /// last access to it and to anything it owns.
    fn fail(self: *Scheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
        while (self.queue.popFront()) |plan| {
            const unclaimed = plan.jobs.len - plan.cursor;
            plan.cursor = plan.jobs.len;
            plan.batch.finishJobs(unclaimed);
        }
        self.unclaimed_total = 0;
    }

    /// Hands out the head plan's next job; the plan leaves the queue with
    /// its last one. Null only when nothing is queued: a queued plan always
    /// has a job left.
    fn claim(self: *Scheduler, io: std.Io) ?Claim {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const plan = self.queue.front() orelse return null;
        const index = plan.cursor;
        std.debug.assert(index < plan.jobs.len);
        plan.cursor += 1;
        self.unclaimed_total -= 1;
        if (plan.cursor == plan.jobs.len) _ = self.queue.popFront();
        return .{ .plan = plan, .index = index };
    }

    fn waitForWork(self: *Scheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and self.unclaimed_total == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }
};

/// A claimed source job and its completion state. The plan owns its storage.
const ReadRequest = struct {
    const Scratch = struct {
        allocator: std.mem.Allocator,
        iovecs: [][]u8,
        queue_counts: []usize,
        /// A direct host read's destination ranges, at most one vectored
        /// read's worth (`HostScatter`).
        host_iovecs: [][]u8,
        /// A staged job's direct read and a gather job's reads that do not
        /// scatter (`landJob`), allocated on first use: as large as a job's
        /// blocks, page aligned for O_DIRECT.
        bounce: []align(std.heap.page_size_min) u8 = &.{},
        bounce_capacity: usize,

        fn init(
            allocator: std.mem.Allocator,
            maximum_blocks: usize,
            block_size: usize,
            device_count: usize,
        ) !Scratch {
            const iovecs = try allocator.alloc([]u8, maximum_blocks);
            errdefer allocator.free(iovecs);
            const queue_counts = try allocator.alloc(usize, device_count);
            errdefer allocator.free(queue_counts);
            const host_iovecs = try allocator.alloc([]u8, safetensors.max_positional_iovecs);
            return .{
                .allocator = allocator,
                .iovecs = iovecs,
                .queue_counts = queue_counts,
                .host_iovecs = host_iovecs,
                .bounce_capacity = maximum_blocks * block_size,
            };
        }

        fn bounceBuffer(self: *Scratch, len: usize) ![]u8 {
            std.debug.assert(len <= self.bounce_capacity);
            if (self.bounce.len == 0) self.bounce = try self.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), self.bounce_capacity);
            return self.bounce[0..len];
        }

        fn deinit(self: *Scratch) void {
            self.allocator.free(self.bounce);
            self.allocator.free(self.host_iovecs);
            self.allocator.free(self.queue_counts);
            self.allocator.free(self.iovecs);
            self.* = undefined;
        }
    };

    loader: *Loader,
    plan: *Batch.Plan,
    /// The job's index in its plan, which holds this request at the same
    /// index.
    index: usize,
    /// The request's lifecycle credit: its slot of blocks (`Credits`), which
    /// only this request reads into and transfers from until it ends.
    slot: u32,
    /// The scheduling sentinel and one reference per DMA piece queued; the
    /// last release gives the credit back and finishes the job.
    pending: std.atomic.Value(usize) = .init(1),
    /// Awake-clock nanoseconds of the enqueue; 0 until then.
    enqueued_ns: u64 = 0,

    /// The slot of a job that was never claimed: nothing pending, so the
    /// retirement checks hold.
    const idle: ReadRequest = .{
        .loader = undefined,
        .plan = undefined,
        .index = undefined,
        .slot = undefined,
        .pending = .init(0),
    };

    /// Takes the claimed job's request context with the worker's credit.
    /// The claim holds the batch's completion unit; the request's final
    /// reference drop releases it.
    fn init(loader: *Loader, claim: Scheduler.Claim, slot: u32) *ReadRequest {
        const self = &claim.plan.requests[claim.index];
        self.* = .{
            .loader = loader,
            .plan = claim.plan,
            .index = claim.index,
            .slot = slot,
        };
        return self;
    }

    fn run(self: *ReadRequest, scratch: *Scratch) !void {
        const loader = self.loader;
        const io = loader.io;
        if (loader.failed()) return;

        const job = self.plan.jobs[self.index];
        switch (job.kind) {
            .dma_read, .dma_staged, .dma_gather, .dma_fill => {},
            .host_direct, .host_bounce, .host_fill => return self.runHost(scratch),
        }
        const transfers = self.plan.transfersOf(job);
        const block_size = loader.plan_config.block_size;
        // A fill job's transfers read the zero block, nothing else.
        const blocks: []const host_memory.Block = if (job.kind == .dma_fill)
            loader.blocks[0..1]
        else
            loader.credits.slotBlocks(self.slot)[0..job.blockLen(block_size)];
        const queue_counts = scratch.queue_counts;
        @memset(queue_counts, 0);
        var pieces: usize = 0;

        // The planner cut the transfers to this job's blocks and the tensor's
        // writers are the devices it planned for (both come from the
        // sharding's canonical device order), so these are invariants.
        for (transfers) |transfer| {
            const init_started = awakeNs(io);
            const destination = transfer.destination;
            const tensor = destination.ensureState(loader) catch |err| {
                loader.recordError(err, "initialize tensor {s} from {s}, shape={f}", .{ destination.name(), destination.parts[0].source.file_uri, destination.shape });
                return err;
            };
            _ = loader.metrics.tensor_init_ns.fetchAdd(awakeNs(io) -| init_started, .monotonic);
            std.debug.assert(transfer.block_index < blocks.len and
                transfer.block_offset < block_size and
                transfer.len <= block_size - transfer.block_offset);
            pieces += @popCount(transfer.writer_mask);
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                std.debug.assert(writer_index < tensor.targets.len);
                queue_counts[tensor.targets[writer_index].device_index] += 1;
            }
        }
        if (loader.errorValue()) |err| return err;
        switch (job.kind) {
            .dma_read => |read| {
                const iovecs = scratch.iovecs[0..blocks.len];
                for (iovecs, blocks, 0..) |*iovec, block, block_index| {
                    const consumed = block_index * block_size;
                    iovec.* = block[0..@min(block_size, read.len - consumed)];
                }
                try readGated(loader, try self.plan.source_slot.ensure(io), iovecs, job.file_offset, read.minimum_len);
            },
            .dma_staged, .dma_gather => try landJob(FileReader{ .loader = loader }, self.plan, job, blocks, block_size, scratch),
            .dma_fill => {},
            .host_direct, .host_bounce, .host_fill => unreachable,
        }
        if (job.kind != .dma_fill) {
            for (transfers) |transfer| transfer.destination.recordRead(transfer.len);
            if (loader.errorValue()) |err| return err;
        }
        // Every piece holds the request, and with it the slot, until its
        // callback.
        _ = self.pending.fetchAdd(pieces, .acq_rel);
        self.enqueued_ns = awakeNs(io);
        Dma.enqueueBlocks(loader, transfers, self, blocks, queue_counts) catch |err| {
            self.enqueued_ns = 0;
            // Nothing was queued. The worker still holds the scheduling
            // reference, so this cannot end the request.
            _ = self.pending.fetchSub(pieces, .acq_rel);
            return err;
        };
    }

    /// A host job: the worker writes its destinations itself, so the job
    /// ends with the worker's scheduling reference; its slot's blocks, if it
    /// reads through them, are done when it returns. Its read takes the read
    /// gate like any other.
    fn runHost(self: *ReadRequest, scratch: *Scratch) !void {
        const loader = self.loader;
        const io = loader.io;
        const job = self.plan.jobs[self.index];
        const transfers = self.plan.transfersOf(job);
        // Normally created ahead by the allocation task; a worker that gets
        // here first creates the destination itself.
        var previous: ?*Destination = null;
        for (transfers) |transfer| {
            const destination = transfer.destination;
            if (destination == previous) continue;
            previous = destination;
            std.debug.assert(destination.hostResident());
            const init_started = awakeNs(io);
            _ = destination.ensureState(loader) catch |err| {
                loader.recordError(err, "initialize tensor {s} from {s}, shape={f}", .{ destination.name(), destination.parts[0].source.file_uri, destination.shape });
                return err;
            };
            _ = loader.metrics.tensor_init_ns.fetchAdd(awakeNs(io) -| init_started, .monotonic);
        }
        if (loader.errorValue()) |err| return err;

        // One write lands a transfer on all its writers (`hostLanding`).
        var written: HostWritten = .{};
        defer written.flush();
        if (job.kind == .host_fill) {
            for (transfers) |transfer| {
                const tensor = &transfer.destination.state.value;
                @memset(tensor.hostLanding(transfer), 0);
                written.add(tensor, transfer.len * @popCount(transfer.writer_mask));
            }
            return;
        }

        const file = try self.plan.source_slot.ensure(io);
        switch (job.kind) {
            .host_direct => |read| {
                var scatter: HostScatter = .{
                    .iovecs = scratch.host_iovecs,
                    .file_offset = job.file_offset,
                };
                for (transfers) |transfer| {
                    const tensor = &transfer.destination.state.value;
                    // The planner checked that the transfers cover the read
                    // in order, each source byte once.
                    std.debug.assert(transfer.block_index == 0 and transfer.block_offset == read.len - scatter.remaining(read.len));
                    if (!scatter.hasRoom()) try scatter.flush(loader, file);
                    scatter.addRead(tensor.hostLanding(transfer));
                    written.add(tensor, transfer.len * @popCount(transfer.writer_mask));
                }
                try scatter.flush(loader, file);
                std.debug.assert(scatter.read_len == read.len);
            },
            .host_bounce => |read| {
                const block_size = loader.plan_config.block_size;
                const blocks = loader.credits.slotBlocks(self.slot)[0..job.blockLen(block_size)];
                const iovecs = scratch.iovecs[0..blocks.len];
                for (iovecs, blocks, 0..) |*iovec, block, block_index| {
                    iovec.* = block[0..@min(block_size, read.len - block_index * block_size)];
                }
                try readGated(loader, file, iovecs, job.file_offset, read.minimum_len);
                for (transfers) |transfer| {
                    const tensor = &transfer.destination.state.value;
                    std.debug.assert(transfer.block_index == 0 and transfer.block_offset + transfer.len <= read.len);
                    copyFromBlocks(blocks, block_size, transfer.block_offset, tensor.hostLanding(transfer));
                    written.add(tensor, transfer.len * @popCount(transfer.writer_mask));
                }
            },
            .dma_read, .dma_staged, .dma_gather, .dma_fill, .host_fill => unreachable,
        }
        for (transfers) |transfer| transfer.destination.recordRead(transfer.len);
    }

    /// Releases a DMA piece or the scheduling reference. The final release
    /// gives the credit back and may let the awaiting task free this request
    /// and everything its batch owns.
    fn release(self: *ReadRequest) void {
        const previous = self.pending.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;

        // Locals first: releasing the batch unit may complete the batch
        // that owns this request, so nothing is touched after it.
        const loader = self.loader;
        const batch = self.plan.batch;
        if (self.enqueued_ns != 0) {
            _ = loader.metrics.dma_stage_ns.fetchAdd(awakeNs(loader.io) -| self.enqueued_ns, .monotonic);
        }
        loader.credits.release(loader.io, self.slot);
        batch.finishJobs(1);
    }
};

/// One source read under the read gate, counted in the loader's metrics.
fn readGated(loader: *Loader, file: std.Io.File, iovecs: []const []u8, file_offset: u64, minimum_len: usize) !void {
    const io = loader.io;
    if (!loader.read_gate.acquire(io)) return error.Closed;
    defer loader.read_gate.release(io);
    const read_started = awakeNs(io);
    const read_result = safetensors.readFilePositionalAllV(io, file, iovecs, file_offset, minimum_len);
    _ = loader.metrics.read_ns.fetchAdd(awakeNs(io) -| read_started, .monotonic);
    const bytes_read = try read_result;
    _ = loader.metrics.read_operations.fetchAdd(1, .monotonic);
    _ = loader.metrics.read_bytes.fetchAdd(@intCast(bytes_read), .monotonic);
}

/// What the worker reads a job's files through: the read gate and the
/// loader's metrics (`readGated`). Tests read from bytes instead.
const FileReader = struct {
    loader: *Loader,

    fn read(self: FileReader, slot: *SourceSlot, iovecs: []const []u8, file_offset: u64, minimum_len: usize) !void {
        try readGated(self.loader, try slot.ensure(self.loader.io), iovecs, file_offset, minimum_len);
    }
};

/// Reads from bytes standing for the files, as `FileReader` reads files:
/// for tests that replay a plan's jobs.
const BytesReader = struct {
    const File = struct { slot: *const SourceSlot, bytes: []const u8 };

    files: []const File,

    fn bytesOf(self: BytesReader, slot: *const SourceSlot) ?[]const u8 {
        for (self.files) |file| {
            if (file.slot == slot) return file.bytes;
        }
        return null;
    }

    fn read(self: BytesReader, slot: *SourceSlot, iovecs: []const []u8, file_offset: u64, minimum_len: usize) !void {
        const bytes = self.bytesOf(slot) orelse return error.FileNotFound;
        var at: usize = @intCast(file_offset);
        var read_len: usize = 0;
        for (iovecs) |iovec| {
            const take = @min(iovec.len, bytes.len -| at);
            @memcpy(iovec[0..take], bytes[at..][0..take]);
            at += take;
            read_len += take;
            if (take < iovec.len) break;
        }
        if (read_len < minimum_len) return error.EndOfStream;
    }
};

/// Reads a staged or gather job and lands its items' runs in its blocks
/// (`Batch.Plan.Item`, `walkItem`). A staged job reads its own read; a
/// gather job reads each of its extents. A read marked for it scatters
/// straight into the blocks, `max_positional_iovecs` ranges per call; any
/// other lands in the worker's bounce buffer and is copied from there: a
/// direct read, whose ranges must be aligned, and a gather read whose
/// source runs are out of order or repeat (a transpose, a broadcast).
fn landJob(
    reader: anytype,
    plan: *const Batch.Plan,
    job: Batch.Plan.Job,
    blocks: []const host_memory.Block,
    block_size: usize,
    scratch: *ReadRequest.Scratch,
) !void {
    switch (job.kind) {
        .dma_staged => |staged| {
            const items = plan.items[staged.item_start..][0..staged.item_len];
            std.debug.assert(plan.direct_reads or (staged.data_offset == 0 and staged.read.len == staged.read.minimum_len));
            const read: Batch.Plan.Extent = .{
                .source_slot = plan.source_slot,
                .file_offset = job.file_offset,
                .len = staged.read.len,
                .minimum_len = staged.read.minimum_len,
                .scatter = !plan.direct_reads,
            };
            return landExtent(reader, read, items, plan.landings, blocks, block_size, scratch);
        },
        .dma_gather => |gather| {
            if (gather.zeroed) {
                for (blocks) |block| @memset(block[0..block_size], 0);
            }
            const items = plan.items[gather.item_start..][0..gather.item_len];
            const extents = plan.extents[gather.extent_start..][0..gather.extent_len];
            for (extents, items) |extent, *item| {
                try landExtent(reader, extent, item[0..1], plan.landings, blocks, block_size, scratch);
            }
        },
        .dma_read, .dma_fill, .host_direct, .host_bounce, .host_fill => unreachable,
    }
}

fn landExtent(
    reader: anytype,
    extent: Batch.Plan.Extent,
    items: []const Batch.Plan.Item,
    landings: []const u64,
    blocks: []const host_memory.Block,
    block_size: usize,
    scratch: *ReadRequest.Scratch,
) !void {
    if (extent.scatter) {
        std.debug.assert(extent.len == extent.minimum_len);
        var sink: ScatterSink(@TypeOf(reader)) = .{
            .reader = reader,
            .slot = extent.source_slot,
            .iovecs = scratch.host_iovecs,
            .file_offset = extent.file_offset,
            .blocks = blocks,
            .block_size = block_size,
        };
        for (items) |item| try walkItem(item, landings, &sink);
        try sink.flush();
        std.debug.assert(sink.file_offset == extent.file_offset + extent.len);
        return;
    }
    const bounce = try scratch.bounceBuffer(extent.len);
    try reader.read(extent.source_slot, &.{bounce}, extent.file_offset, extent.minimum_len);
    var sink: CopySink = .{ .bounce = bounce, .file_offset = extent.file_offset, .blocks = blocks, .block_size = block_size };
    for (items) |item| try copyItem(item, landings, &sink);
}

/// Hands `sink` every run of `item` as `place(file offset, landing, len)`,
/// in run order, cut where the destination's box changes
/// (`ShardGrid.runEnd`).
fn walkItem(item: Batch.Plan.Item, landings: []const u64, sink: anytype) !void {
    const table = landings[item.landing_start..][0..item.destination.grid.box_count];
    try walkRuns(item.lattice, item.first, item.end, item.clip_start, item.clip_end, &item.destination.grid, table, sink);
}

/// `walkItem` for runs `first` to `end` of `lattice`, clipped to the
/// destination range `[clip_start, clip_end)`.
fn walkRuns(lattice: backend.Lattice, first: u64, end: u64, clip_start: u64, clip_end: u64, grid: *const ShardGrid, table: []const u64, sink: anytype) !void {
    if (first >= end) return;
    var runs: backend.Lattice.Cursor = .init(lattice, first, end);
    while (runs.next()) |run| {
        var at = @max(run.dst, clip_start);
        const stop = @min(run.dst + lattice.width, clip_end);
        var source = run.src + (at - run.dst);
        while (at < stop) {
            const piece_end = grid.runEnd(at, stop);
            try sink.place(source, table[grid.boxOf(at)] +% grid.boxOffset(at), piece_end - at);
            source += piece_end - at;
            at = piece_end;
        }
    }
}

/// `walkItem` from the bounce buffer, a stretch at a time where that pays:
/// a stretch of the destination landing in one piece, inside one run of
/// the grid and one block, has its whole runs copied by
/// `Lattice.copyRuns`, and only the runs its ends cut go one at a time. A
/// stretch costs a few bisections, so items with few runs per stretch (a
/// part of a few wide runs, a grid that cuts every row) are walked.
fn copyItem(item: Batch.Plan.Item, landings: []const u64, sink: *CopySink) !void {
    const lattice = item.lattice;
    const grid = &item.destination.grid;
    if (lattice.dims.len == 0 or item.first >= item.end) return walkItem(item, landings, sink);
    const start = @max(item.clip_start, lattice.at(item.first).dst);
    const stop = @min(item.clip_end, lattice.at(item.end - 1).dst + lattice.width);
    if (start >= stop) return;
    const stretches = grid.runCount(start, stop) + (stop - start) / sink.block_size + 1;
    if (item.end - item.first < 16 * stretches) return walkItem(item, landings, sink);
    const table = landings[item.landing_start..][0..grid.box_count];
    const source = @intFromPtr(sink.bounce.ptr) -% @as(usize, @intCast(sink.file_offset));
    var at = start;
    while (at < stop) {
        const landing: usize = @intCast(table[grid.boxOf(at)] +% grid.boxOffset(at));
        const in_block = landing % sink.block_size;
        const stretch_end = @min(grid.runEnd(at, stop), at + (sink.block_size - in_block));
        // The address logical offset zero would have, were the whole
        // destination laid out like this stretch.
        const destination = @intFromPtr(sink.blocks[landing / sink.block_size].ptr) + in_block -% @as(usize, @intCast(at));
        const touched = @max(item.first, lattice.firstEndingAfter(at));
        const whole = @max(item.first, lattice.firstStartingFrom(at));
        const whole_end = @min(item.end, lattice.firstEndingAfter(stretch_end));
        const touched_end = @min(item.end, lattice.firstStartingFrom(stretch_end));
        if (whole < whole_end) {
            try walkRuns(lattice, touched, whole, at, stretch_end, grid, table, sink);
            lattice.copyRuns(whole, whole_end, destination, source);
            try walkRuns(lattice, whole_end, touched_end, at, stretch_end, grid, table, sink);
        } else {
            try walkRuns(lattice, touched, touched_end, at, stretch_end, grid, table, sink);
        }
        at = stretch_end;
    }
}

/// A read scattered over the blocks as its runs are walked: each run's
/// landing becomes iovecs (split at block boundaries, merged where memory
/// continues), read in calls of at most `iovecs.len`. Runs must come in
/// file order, back to back.
fn ScatterSink(comptime Reader: type) type {
    return struct {
        reader: Reader,
        slot: *SourceSlot,
        iovecs: [][]u8,
        count: usize = 0,
        pending: usize = 0,
        /// The file offset of the first byte not read yet.
        file_offset: u64,
        blocks: []const host_memory.Block,
        block_size: usize,

        fn place(self: *@This(), source: u64, landing: u64, len: u64) !void {
            std.debug.assert(source == self.file_offset + self.pending);
            var at: usize = @intCast(landing);
            var rest: usize = @intCast(len);
            while (rest != 0) {
                const in_block = at % self.block_size;
                const take = @min(rest, self.block_size - in_block);
                const range = self.blocks[at / self.block_size][in_block..][0..take];
                const merged = self.count != 0 and self.iovecs[self.count - 1].ptr + self.iovecs[self.count - 1].len == range.ptr;
                if (merged) {
                    self.iovecs[self.count - 1].len += take;
                } else {
                    if (self.count == self.iovecs.len) try self.flush();
                    self.iovecs[self.count] = range;
                    self.count += 1;
                }
                self.pending += take;
                at += take;
                rest -= take;
            }
        }

        fn flush(self: *@This()) !void {
            if (self.count == 0) return;
            try self.reader.read(self.slot, self.iovecs[0..self.count], self.file_offset, self.pending);
            self.file_offset += self.pending;
            self.count = 0;
            self.pending = 0;
        }
    };
}

/// Runs copied from a read in the bounce buffer, which starts at
/// `file_offset`.
const CopySink = struct {
    bounce: []const u8,
    file_offset: u64,
    blocks: []const host_memory.Block,
    block_size: usize,

    fn place(self: *CopySink, source: u64, landing: u64, len: u64) !void {
        copyToBlocks(self.blocks, self.block_size, @intCast(landing), self.bounce[@intCast(source - self.file_offset)..][0..@intCast(len)]);
    }
};

/// A direct host read in progress: the destination ranges of a run of
/// transfers, which one vectored read fills. Flushed whenever the next
/// transfer might not fit, so a job of any number of transfers reads in as
/// many calls as it needs, each one contiguous in the file.
const HostScatter = struct {
    iovecs: [][]u8,
    iovec_count: usize = 0,
    /// File offset of the first byte not read yet.
    file_offset: u64,
    /// Bytes the pending iovecs cover, and every byte read so far.
    pending_len: usize = 0,
    read_len: usize = 0,

    /// Bytes of a job of `len` neither read nor pending.
    fn remaining(self: *const HostScatter, len: usize) usize {
        return len - self.read_len - self.pending_len;
    }

    /// Whether another read range fits.
    fn hasRoom(self: *const HostScatter) bool {
        return self.iovec_count < self.iovecs.len;
    }

    /// Appends a range to the read, merged with the previous one when
    /// contiguous in memory too.
    fn addRead(self: *HostScatter, destination: []u8) void {
        self.pending_len += destination.len;
        if (self.iovec_count != 0) {
            const last = &self.iovecs[self.iovec_count - 1];
            if (last.ptr + last.len == destination.ptr) {
                last.len += destination.len;
                return;
            }
        }
        self.iovecs[self.iovec_count] = destination;
        self.iovec_count += 1;
    }

    /// Reads the pending ranges, every byte of which must exist (a direct
    /// host read has no alignment padding).
    fn flush(self: *HostScatter, loader: *Loader, file: std.Io.File) !void {
        if (self.iovec_count == 0) return;
        try readGated(loader, file, self.iovecs[0..self.iovec_count], self.file_offset, self.pending_len);
        if (loader.errorValue()) |err| return err;
        self.file_offset += self.pending_len;
        self.read_len += self.pending_len;
        self.pending_len = 0;
        self.iovec_count = 0;
    }
};

/// Bytes a worker wrote into host-resident destinations, added to each
/// destination's `host_written` once per run of its transfers.
const HostWritten = struct {
    tensor: ?*TensorTransfer = null,
    bytes: usize = 0,

    fn add(self: *HostWritten, tensor: *TensorTransfer, bytes: usize) void {
        if (self.tensor != tensor) {
            self.flush();
            self.tensor = tensor;
        }
        self.bytes += bytes;
    }

    fn flush(self: *HostWritten) void {
        const tensor = self.tensor orelse return;
        _ = tensor.host_written.fetchAdd(self.bytes, .acq_rel);
        self.* = .{};
    }
};

/// Copies `source` to offset `offset` of `blocks` taken end to end,
/// `block_size` bytes each.
fn copyToBlocks(blocks: []const host_memory.Block, block_size: usize, offset: usize, source: []const u8) void {
    var destination = offset;
    var rest = source;
    while (rest.len != 0) {
        const block = blocks[destination / block_size];
        const in_block = destination % block_size;
        const take = @min(rest.len, block_size - in_block);
        @memcpy(block[in_block..][0..take], rest[0..take]);
        rest = rest[take..];
        destination += take;
    }
}

/// Copies `destination.len` bytes from offset `offset` of the read that
/// `blocks` hold, `block_size` bytes each.
fn copyFromBlocks(blocks: []const host_memory.Block, block_size: usize, offset: usize, destination: []u8) void {
    var source = offset;
    var rest = destination;
    while (rest.len != 0) {
        const block = blocks[source / block_size];
        const in_block = source % block_size;
        const take = @min(rest.len, block_size - in_block);
        @memcpy(rest[0..take], block[in_block..][0..take]);
        rest = rest[take..];
        source += take;
    }
}

/// The DMA stage, over the loader's pumps (`Loader.pumps`): workers queue a
/// request's transfers per device, a pump submits them under the device's
/// in-flight budget, and their callbacks release the request.
const Dma = struct {
    /// One DMA piece: `data`, in its request's slot or the zero block, for
    /// `target` at `destination_offset`. The piece holds one of its
    /// request's references until its callback. Replicas share their
    /// source bytes: copying them to another CPU socket first was rejected
    /// on MI300X, where raw local and remote H2D both reached ~49-50 GiB/s
    /// per GPU while one CPU thread copied across sockets at 5-10 GiB/s.
    const ReadyTransfer = struct {
        target: *TensorTransfer.Target,
        request: *ReadRequest,
        data: []const u8,
        destination_offset: usize,
    };

    /// Transfers ready for one device, submitted in arrival order so the
    /// oldest requests and submissions complete first. Owned by the
    /// device's `DevicePump.mutex`.
    const ReadyQueue = std.Deque(ReadyTransfer);

    /// One device's submission state. Its pump runs on whichever thread
    /// finds work -- a worker after enqueueing, a ready callback after a
    /// completion -- one at a time per device, and shares nothing with the
    /// other devices' pumps. On four GB300, a synthetic 1 MiB transfer probe
    /// with parallel submitters reached 174k-184k submissions/s, while the
    /// single-pump loader managed 51k-55k/s on replicated DeepSeek's mixed
    /// 256 KiB/4 MiB traffic. Those are different workloads, not a measured
    /// loader speedup. The end-to-end comparison was 5.41 -> 4.68 s (-13.5%)
    /// over five interleaved pairs at depth eight and 16 MiB blocks, with
    /// unchanged or lower pinned high-water.
    /// One submitter per device is a correctness requirement, not a
    /// performance choice: a tensor's pieces for one device are flagged
    /// last by `Target.nextIsLast` in submission order, and two threads
    /// submitting pieces of the same tensor concurrently left targets
    /// unclosed (`IncompleteTransfer` on Llama, whose tensors span blocks).
    /// Concurrent submitters measured no faster anyway: a submission blocks
    /// ~28 us in the driver once four GB300 each hold ~32 pieces in flight,
    /// and pumps mostly stop for lack of room. Synthetic probes attributed
    /// part of the remaining gap to source misalignment (-13%), concurrent
    /// CPU writes at 35 GB/s (-10%) and unplaced memory (-23%). Those are
    /// separate probe arms, not additive loader losses; roughly 1.3x still
    /// remained unattributed with the conditions stacked, so the gap
    /// is not evidence that more submitters would help. A dedicated task per
    /// device took 5.0-5.4 s with a wake per completion (a futex round trip
    /// per piece), or 4.4-5.1 s with half-budget wake hysteresis. Concurrent
    /// submitters took 4.56-4.58 s but failed two of four Llama runs with
    /// IncompleteTransfer; DeepSeek's mostly single-piece tensors hid the bug.
    const DevicePump = struct {
        mutex: std.Io.Mutex = .init,
        queue: ReadyQueue = .empty,
        /// In-flight DMA bytes and submissions; owned by `mutex`.
        active_bytes: usize = 0,
        active_pieces: usize = 0,
        pumping: bool = false,
        /// Contexts whose callback fired, for the next pump to destroy: an
        /// intrusive stack through `EventContext.next_retired`, owned by
        /// `mutex`. Destroying an event from the next pump instead of at
        /// its batch's retirement bounds live PJRT events by the DMA width
        /// plus one pump batch rather than by a submission's transfer
        /// count; every shipped plugin accepted destruction outside the
        /// event's own callback (16,384 fired events on two B70, CUDA and
        /// ROCm alike).
        retired: ?*EventContext = null,

        /// Whether the next queued transfer may go. The budget is in bytes,
        /// not submissions: the calibrated depth is `max_in_flight_per_device`
        /// blocks, and a piece is one tensor's slice of a block, so a model of
        /// small tensors needs many more pieces in flight to keep the same
        /// bytes moving: eight average DeepSeek pieces occupy only ~18 MiB
        /// against the 128 MiB calibrated with 16 MiB blocks. On one GB300,
        /// widening the stage AND adding lifecycle credits raised DeepSeek
        /// from 24 to 44 GiB/s; this was not a byte-budget-only experiment.
        /// The piece that crosses the budget is admitted: a device with room
        /// always has a transfer in flight.
        fn hasRoom(self: *const DevicePump, budget_bytes: usize) bool {
            return self.active_bytes < budget_bytes and self.active_pieces < max_dma_pieces_per_device;
        }

        /// `mutex`. Destroys every retired event: their callbacks have run,
        /// and only the pump or the batch's retirement destroys an event,
        /// never its own callback.
        fn destroyRetired(self: *DevicePump) void {
            while (self.retired) |ctx| {
                self.retired = ctx.next_retired;
                ctx.next_retired = null;
                ctx.destroyEvent();
            }
        }
    };

    const EventContext = struct {
        request: *ReadRequest,
        /// Null once destroyed, by a pump or by the batch's retirement.
        pjrt_event: ?*pjrt.Event,
        err: ?*pjrt.Error = null,
        device_index: usize,
        /// Bytes this submission holds against the device's in-flight budget.
        len: usize,
        submitted_ns: u64 = 0,
        /// Link of the device pump's `retired`; owned by that pump's mutex.
        next_retired: ?*EventContext = null,

        /// Pump mutex. Destroys the event and its error, once.
        /// The platform is reached through the request, and only when there
        /// is something to destroy: a context whose event and error are both
        /// null never touches it.
        fn destroyEvent(self: *EventContext) void {
            if (self.pjrt_event) |event| event.deinit(self.request.loader.platform.pjrt_api);
            self.pjrt_event = null;
            if (self.err) |err| err.deinit(self.request.loader.platform.pjrt_api);
            self.err = null;
        }
    };

    fn lockAllPumps(loader: *Loader) void {
        for (loader.pumps) |*device_pump| device_pump.mutex.lockUncancelable(loader.io);
    }

    fn unlockAllPumps(loader: *Loader) void {
        var index = loader.pumps.len;
        while (index > 0) {
            index -= 1;
            loader.pumps[index].mutex.unlock(loader.io);
        }
    }

    /// Retires the contexts of a done batch: destroys the PJRT events a pump
    /// has not destroyed yet (from the awaiting task after their callbacks
    /// fired, exactly as `pjrt.Event.await` does), unlinks the batch's
    /// contexts from `retired` and checks that every request completed. Runs with every pump locked so an `abortReady` still
    /// iterating queued entries or a pump draining its `retired` cannot race
    /// the free that follows.
    fn retireBatch(loader: *Loader, batch: *Batch) void {
        std.debug.assert(batch.done.isSet());
        lockAllPumps(loader);
        defer unlockAllPumps(loader);
        if (builtin.mode == .debug) batch.freeing = true;
        for (batch.plans.items) |plan| {
            for (plan.events[0..plan.events_used.load(.acquire)]) |*ctx| ctx.destroyEvent();
            for (plan.requests) |*request| std.debug.assert(request.pending.load(.acquire) == 0);
        }
        // Only this batch's contexts have a destroyed event while still
        // linked: a pump unlinks what it destroys.
        for (loader.pumps) |*device_pump| {
            var link = &device_pump.retired;
            while (link.*) |ctx| {
                if (ctx.pjrt_event == null) link.* = ctx.next_retired else link = &ctx.next_retired;
            }
        }
    }

    /// Hands a context whose callback fired to its device's next pump. Under
    /// that pump's mutex, so the batch's retirement sees it before the batch
    /// is freed.
    fn retireEvent(loader: *Loader, ctx: *EventContext) void {
        const device_pump = &loader.pumps[ctx.device_index];
        device_pump.mutex.lockUncancelable(loader.io);
        ctx.next_retired = device_pump.retired;
        device_pump.retired = ctx;
        device_pump.mutex.unlock(loader.io);
    }

    /// Queue the whole source job before pumping. The former per-piece path
    /// paid roughly 69k-79k mutex/pump trips per DeepSeek load; coalesced reads
    /// alone did not remove that cost. Reserve every destination first so
    /// allocation failure cannot publish a subset of the request's pieces.
    /// A transfer reads `blocks[transfer.block_index]`.
    fn enqueueBlocks(
        loader: *Loader,
        transfers: []const Batch.Plan.Transfer,
        request: *ReadRequest,
        blocks: []const host_memory.Block,
        queue_counts: []const usize,
    ) !void {
        std.debug.assert(queue_counts.len == loader.pumps.len);
        // Every pump is held while the request's transfers land, and every
        // destination is reserved before any queue changes, so the request
        // is either fully queued or not queued at all: the caller drops all
        // of its pieces on failure.
        lockAllPumps(loader);
        errdefer unlockAllPumps(loader);
        // A failure recorded before these locks may have drained the queues
        // already (`abortReady`), and nothing would complete what this
        // queued; one recorded after them drains it.
        if (loader.errorValue()) |err| return err;
        for (loader.pumps, queue_counts) |*device_pump, count| {
            try device_pump.queue.ensureUnusedCapacity(loader.allocator, count);
        }
        for (transfers) |transfer| {
            const data = blocks[transfer.block_index][transfer.block_offset..][0..transfer.len];
            const tensor = &transfer.destination.state.value;
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const target = &tensor.targets[writer_index];
                const device_pump = &loader.pumps[target.device_index];
                device_pump.queue.pushBackAssumeCapacity(.{
                    .target = target,
                    .request = request,
                    .data = data,
                    .destination_offset = transfer.destination_offset,
                });
            }
        }
        unlockAllPumps(loader);
        for (queue_counts, 0..) |count, device_index| {
            if (count != 0) requestPump(loader, device_index);
        }
    }

    /// Runs the device's pump on this thread unless one is already running.
    fn requestPump(loader: *Loader, device_index: usize) void {
        const device_pump = &loader.pumps[device_index];
        device_pump.mutex.lockUncancelable(loader.io);
        if (device_pump.pumping or loader.failed()) {
            device_pump.mutex.unlock(loader.io);
            return;
        }
        device_pump.pumping = true;
        device_pump.mutex.unlock(loader.io);
        pump(loader, device_index);
    }

    /// Submits from the device's queue, in arrival order, until the device
    /// has no room or nothing is queued. Any queued transfer can go: nothing
    /// waits for another piece. Exactly one pump runs per device at a time,
    /// see `DevicePump`.
    fn pump(loader: *Loader, device_index: usize) void {
        const device_pump = &loader.pumps[device_index];
        while (true) {
            var selected: ?ReadyTransfer = null;
            device_pump.mutex.lockUncancelable(loader.io);
            device_pump.destroyRetired();
            if (!loader.failed()) {
                if (!device_pump.hasRoom(loader.dma_budget_bytes)) {
                    _ = loader.metrics.pump_stops_full.fetchAdd(1, .monotonic);
                } else if (device_pump.queue.popFront()) |transfer| {
                    device_pump.active_bytes += transfer.data.len;
                    device_pump.active_pieces += 1;
                    selected = transfer;
                } else {
                    _ = loader.metrics.pump_stops_empty.fetchAdd(1, .monotonic);
                }
            }
            if (selected == null) device_pump.pumping = false;
            device_pump.mutex.unlock(loader.io);
            if (selected) |transfer| submitOne(loader, transfer) else return;
        }
    }

    /// MEMORY-ORDER RULE: `transfer.target` and `transfer.request` belong to
    /// a batch, and the request's final release may complete that batch,
    /// after which the awaiting task frees it. Every path therefore copies
    /// what it needs into locals, retires the DMA slot with
    /// `eventCompleted`, and calls `request.release()` last.
    fn submitOne(loader: *Loader, transfer: ReadyTransfer) void {
        const device_index = transfer.target.device_index;
        const len = transfer.data.len;
        const request = transfer.request;
        submitTransfer(loader, transfer) catch |err| {
            loader.recordError(err, "submit DMA transfer: device={d}, destination_offset={d}, bytes={d}", .{ device_index, transfer.destination_offset, len });
            eventCompleted(loader, device_index, len);
            request.release();
        };
    }

    fn submitTransfer(loader: *Loader, transfer: ReadyTransfer) !void {
        const api = loader.platform.pjrt_api;
        const target = transfer.target;
        const is_last = target.nextIsLast(transfer.data.len);
        // Even a rejected last call closes the buffer on PJRT's side.
        if (is_last) target.closed = true;
        const submit_started = awakeNs(loader.io);
        const event = try target.manager.transferData(
            api,
            0,
            transfer.data,
            @intCast(transfer.destination_offset),
            is_last,
        );
        target.noteSubmitted(transfer.data.len);

        // The plan holds one event slot per planned submission; the batch
        // owns it. A pump destroys the event once its callback has run, or
        // the batch's retirement does.
        const plan = transfer.request.plan;
        const event_index = plan.events_used.fetchAdd(1, .monotonic);
        std.debug.assert(event_index < plan.events.len);
        const ctx = &plan.events[event_index];
        ctx.* = .{
            .request = transfer.request,
            .pjrt_event = event,
            .device_index = target.device_index,
            .len = transfer.data.len,
            .submitted_ns = submit_started,
        };

        _ = loader.metrics.dma_submissions.fetchAdd(1, .monotonic);
        event.onReady(api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                // MEMORY-ORDER RULE: the batch owns `ctx_` and its request,
                // and `request.release()` may complete that batch, after
                // which the awaiting task frees the context, the request and
                // the batch. Load every field first, store the error, retire
                // the DMA slot (which may pump on this thread), hand the
                // context to the next pump, and release the request last.
                const request = ctx_.request;
                const loader_ = request.loader;
                const device_index = ctx_.device_index;
                const len = ctx_.len;
                _ = loader_.metrics.dma_piece_ns.fetchAdd(awakeNs(loader_.io) -| ctx_.submitted_ns, .monotonic);
                ctx_.err = err;
                // The shipped plugins resolve this event without an error
                // whatever happened to the copy (the C API wrapper sets the
                // promise with an OK status); kept for a plugin that reports.
                if (err) |pjrt_error| {
                    loader_.recordError(pjrt_error.getCode(loader_.platform.pjrt_api).toApiError(), "complete DMA transfer: device={d}, bytes={d}, message={s}", .{ device_index, len, pjrt_error.getMessage(loader_.platform.pjrt_api) });
                }
                eventCompleted(loader_, device_index, len);
                // After the pump this callback may have run, so the event is
                // destroyed by a later pump or by the batch's retirement,
                // never inside its own callback.
                retireEvent(loader_, ctx_);
                request.release();
            }
        }.call, ctx) catch |err| {
            // The batch owns `ctx` and destroys the event at retirement.
            event.awaitRaw(api) catch {};
            return err;
        };
        _ = loader.metrics.dma_submit_ns.fetchAdd(awakeNs(loader.io) -| submit_started, .monotonic);
    }

    fn eventCompleted(loader: *Loader, device_index: usize, len: usize) void {
        const device_pump = &loader.pumps[device_index];
        device_pump.mutex.lockUncancelable(loader.io);
        std.debug.assert(device_pump.active_bytes >= len);
        std.debug.assert(device_pump.active_pieces > 0);
        device_pump.active_bytes -= len;
        device_pump.active_pieces -= 1;
        device_pump.mutex.unlock(loader.io);
        // A ready callback can be the first place an asynchronous PJRT error
        // becomes visible. Once outside the pump lock, retire every queued
        // transfer so request lifecycles cannot wait forever on entries that
        // the failed pumps will no longer submit.
        if (loader.failed())
            abortReady(loader)
        else
            requestPump(loader, device_index);
    }

    /// Force-completes every queued transfer, one device at a time. A
    /// completion here may finish its batch, but a batch still has an entry
    /// queued until its last one is popped, and it is retired with every
    /// pump locked, so no queued entry is touched after its request ends.
    fn abortReady(loader: *Loader) void {
        for (loader.pumps) |*device_pump| {
            device_pump.mutex.lockUncancelable(loader.io);
            while (device_pump.queue.popFront()) |transfer| {
                transfer.request.release();
            }
            device_pump.mutex.unlock(loader.io);
        }
    }
};

const RequestGate = struct {
    limit: usize,
    in_use: usize = 0,
    closed: bool = false,
    mutex: std.Io.Mutex = .init,
    /// Admission waiters; one release wakes one of them.
    condition: std.Io.Condition = .init,

    fn init(limit: usize) RequestGate {
        return .{ .limit = limit };
    }

    fn acquire(self: *RequestGate, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed and self.in_use >= self.limit) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.closed) return false;
        self.in_use += 1;
        return true;
    }

    fn release(self: *RequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.in_use > 0);
        self.in_use -= 1;
        // One release creates one admission slot. Waking every worker here
        // turns a wide gate into a thundering herd even when the active
        // limit is small.
        self.condition.signal(io);
    }

    fn inUse(self: *RequestGate, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.in_use;
    }

    fn close(self: *RequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }
};

/// The lifecycle credits: one per request the pre-grown pinned set holds,
/// each a fixed slot of `blocks_per_slot` blocks. A worker takes a credit
/// before claiming a job, and the request gives it back with its last
/// reference: after its last DMA callback, or at the end of a host job. Only
/// the holder reads into, lands in and transfers from its slot's blocks, so
/// the credits are the one bound on pinned memory in use. A block whose
/// pieces are done stays with its request until the request ends; nobody
/// else could use it anyway, since every credit owns a whole slot.
const Credits = struct {
    /// Every slot's blocks, slot `s` at `blocks[s * blocks_per_slot ..]`;
    /// a tail shorter than a slot is left unused.
    blocks: []const host_memory.Block,
    blocks_per_slot: usize,
    /// Free slots, a stack of indices; owned by `mutex`.
    free: []u32,
    free_len: usize,
    closed: bool = false,
    /// The most slots held at once.
    high_water: usize = 0,
    mutex: std.Io.Mutex = .init,
    /// Waiters for a slot; one release wakes one of them.
    condition: std.Io.Condition = .init,

    fn init(allocator: std.mem.Allocator, blocks: []const host_memory.Block, blocks_per_slot: usize) std.mem.Allocator.Error!Credits {
        std.debug.assert(blocks_per_slot != 0);
        const free = try allocator.alloc(u32, blocks.len / blocks_per_slot);
        // Slot 0 first.
        for (free, 0..) |*slot, index| slot.* = @intCast(free.len - 1 - index);
        return .{ .blocks = blocks, .blocks_per_slot = blocks_per_slot, .free = free, .free_len = free.len };
    }

    fn deinit(self: *Credits, allocator: std.mem.Allocator) void {
        std.debug.assert(self.free_len == self.free.len);
        allocator.free(self.free);
    }

    fn count(self: *const Credits) usize {
        return self.free.len;
    }

    /// Waits for a free slot; null once closed.
    fn acquire(self: *Credits, io: std.Io) ?u32 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed and self.free_len == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.closed) return null;
        self.free_len -= 1;
        self.high_water = @max(self.high_water, self.free.len - self.free_len);
        return self.free[self.free_len];
    }

    fn release(self: *Credits, io: std.Io, slot: u32) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.free_len < self.free.len);
        self.free[self.free_len] = slot;
        self.free_len += 1;
        // One release frees one slot: waking every waiter here would be a
        // thundering herd.
        self.condition.signal(io);
    }

    fn slotBlocks(self: *const Credits, slot: u32) []const host_memory.Block {
        return self.blocks[slot * self.blocks_per_slot ..][0..self.blocks_per_slot];
    }

    fn inUse(self: *Credits, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.free.len - self.free_len;
    }

    fn close(self: *Credits, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }
};

const Metrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    /// Time workers spend waiting for a lifecycle credit and inside the
    /// source read, summed over requests: waits above the read time mean the
    /// load is DMA-completion bound.
    lifecycle_wait_ns: std.atomic.Value(u64) = .init(0),
    read_ns: std.atomic.Value(u64) = .init(0),
    /// From a request's enqueue to its last DMA callback: how long the DMA
    /// stage holds a lifecycle credit and pinned blocks per request.
    dma_stage_ns: std.atomic.Value(u64) = .init(0),
    /// Inside `submitTransfer` (the PJRT call and the callback registration)
    /// and from a submission to its ready callback, summed over submissions.
    dma_submit_ns: std.atomic.Value(u64) = .init(0),
    dma_piece_ns: std.atomic.Value(u64) = .init(0),
    /// Why a device pump stopped: nothing queued, or no room under the
    /// device's budget. A pump that mostly stops empty is starved by the
    /// reads; one that mostly stops full is waiting on the engine.
    pump_stops_empty: std.atomic.Value(u64) = .init(0),
    pump_stops_full: std.atomic.Value(u64) = .init(0),
    /// Inside `ensureState` for a claimed job's destinations: the first
    /// worker to touch a tensor creates its PJRT buffers and transfer
    /// managers there, and the other workers of the same tensor wait for it;
    /// a host job waits there for the allocation task.
    tensor_init_ns: std.atomic.Value(u64) = .init(0),
    /// The allocation task's time creating host-resident destinations ahead
    /// of their reads, summed over batches.
    ahead_init_ns: std.atomic.Value(u64) = .init(0),
};

/// A value initialized at most once by whichever task touches it first.
/// Concurrent callers wait on the event; a failed initialization keeps its
/// error code and re-materializes the same error for every later caller.
fn LazyOnce(comptime T: type, comptime Ctx: type, comptime ErrorSet: type, comptime initFn: fn (Ctx) ErrorSet!T) type {
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

        fn ensure(self: *Self, io: std.Io, ctx: Ctx) ErrorSet!*T {
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
                .failed => return @errorCast(@errorFromInt(self.error_code.load(.acquire))),
            };
        }

        /// The value when initialization has completed successfully.
        fn readyValue(self: *Self) ?*T {
            return if (self.status.load(.acquire) == .ready) &self.value else null;
        }
    };
}

fn secondsBetween(from: std.Io.Timestamp, to: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.durationTo(to).nanoseconds)) / std.time.ns_per_s;
}

/// Mean milliseconds per unit of a summed duration (zero units: the total).
fn millisecondsPer(total_ns: u64, units: u64) f64 {
    return @as(f64, @floatFromInt(total_ns)) / std.time.ns_per_ms / @as(f64, @floatFromInt(@max(units, 1)));
}

fn awakeNs(io: std.Io) u64 {
    const now: std.Io.Timestamp = .now(io, .awake);
    return @intCast(@max(now.nanoseconds, 1));
}

test "plan construction releases allocations and takes transfers only on success" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const plan = plan: {
                const transfers = try allocator.alloc(Batch.Plan.Transfer, 2);
                errdefer allocator.free(transfers);
                @memset(transfers, .{
                    .destination = undefined,
                    .block_index = 0,
                    .block_offset = 0,
                    .writer_mask = 1,
                    .destination_offset = 0,
                    .len = 64,
                });
                transfers[0].writer_mask = 0b11;
                break :plan try Batch.Plan.create(allocator, undefined, 2, transfers, 3, 128);
            };
            defer plan.destroy();
            try std.testing.expectEqual(@as(usize, 2), plan.jobs.len);
            try std.testing.expectEqual(@as(usize, 2), plan.requests.len);
            // The replicated piece needs two callbacks; the other needs one.
            try std.testing.expectEqual(@as(usize, 3), plan.events.len);
            for (plan.requests) |request| {
                try std.testing.expectEqual(@as(usize, 0), request.pending.load(.acquire));
            }
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "loader releases the pinned set when its workers cannot start" {
    const allocator = std.testing.allocator;
    const platform = Platform.init(allocator, std.testing.io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, std.testing.io);
    // The testing io, but no worker starts: `create` fails after the carve,
    // the credits and the pumps, and unwinds every one.
    var vtable = std.testing.io.vtable.*;
    vtable.groupConcurrent = std.Io.failingGroupConcurrent;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    const result: anyerror!void = if (Loader.create(allocator, io, platform, .{
        .read_parallelism = 2,
        .load_profile = .local,
        .direct_io = .off,
    })) |loader| unexpected: {
        loader.destroy();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.ConcurrencyUnavailable, result);
}

test "loader failures clean up before publication, after publication and during reading" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const contents = [_]u8{ 1, 2, 3, 4 };
    var path_buffer: [1024]u8 = undefined;
    const path = path: {
        const file = try tmp.dir.createFile(io, "weights.bin", .{ .read = true });
        defer file.close(io);
        try file.writePositionalAll(io, &contents, 0);
        const len = try file.realPath(io, &path_buffer);
        break :path path_buffer[0..len];
    };
    const later_path = try std.fmt.allocPrint(allocator, "{s}.later", .{path});
    defer allocator.free(later_path);

    const Failure = enum { before_publication, after_publication, reading };
    // Device memory goes through blocks and DMA; host memory is written in
    // place by the workers, into views of it for `host_mapped`. All clean
    // up the same way.
    for ([_]Memory{ .default, .host_pinned, .host_mapped }) |memory| for (std.enums.values(Failure)) |failure| {
        var sources = [_]safetensors.Tensor{
            .{ .file_uri = path, .name = "first", .shape = .init(.{4}, .u8), .offset = 0 },
            .{ .file_uri = later_path, .name = "later", .shape = .init(.{4}, .u8), .offset = 0 },
        };
        var outputs: [2]Buffer = @splat(.{
            ._platform = platform,
            ._shape = sources[0].shape,
            ._shards = .empty,
        });
        defer for (&outputs) |*output| output.deinit();
        const loader = try Loader.create(allocator, io, platform, .{
            .read_parallelism = 2,
            .load_profile = .local,
            .direct_io = .off,
        });
        defer loader.destroy();
        var parts = [_]LoadPart{ .whole(&sources[0]), .whole(&sources[1]) };
        var specs = [_]LoadSpec{
            .{ .shape = sources[0].shape, .memory = memory, .output = &outputs[0], .parts = parts[0..1] },
            .{ .shape = sources[1].shape, .memory = memory, .output = &outputs[1], .parts = parts[1..2] },
        };

        switch (failure) {
            .before_publication => {
                // The first file's range overflows, which fails its planning
                // before any plan was published.
                sources[0].offset = std.math.maxInt(u64) - 1;
                const result: anyerror!void = if (loader.submit(&specs, .none)) |batch| unexpected: {
                    loader.awaitBatch(batch) catch {};
                    break :unexpected {};
                } else |err| err;
                try std.testing.expectError(error.InvalidTensorRange, result);
                sources[0].offset = 0;
            },
            .after_publication => {
                // The later file's range overflows, which fails its planning
                // after the first file's plan was published.
                sources[1].offset = std.math.maxInt(u64) - 1;
                const result: anyerror!void = if (loader.submit(&specs, .none)) |batch| unexpected: {
                    loader.awaitBatch(batch) catch {};
                    break :unexpected {};
                } else |err| err;
                try std.testing.expectError(error.InvalidTensorRange, result);
            },
            .reading => {
                // Planning succeeds, but the source ends after four of five bytes.
                sources[0].shape = .init(.{5}, .u8);
                parts[0] = .whole(&sources[0]);
                specs[0].shape = sources[0].shape;
                const batch = try loader.submit(specs[0..1], .none);
                try std.testing.expectError(error.UnexpectedEndOfFile, loader.awaitBatch(batch));
            },
        }
        try std.testing.expectEqual(@as(usize, 0), loader.scheduler.unclaimed_total);
        try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
        try std.testing.expectEqual(@as(usize, 0), loader.read_gate.inUse(io));

        if (failure == .before_publication) {
            try std.testing.expectEqual(@as(usize, 0), loader.batch_count);
            try loader.checkOpen();
            const batch = try loader.submit(specs[0..1], .none);
            try loader.awaitBatch(batch);
            const loaded = try outputs[0].toSliceAlloc(allocator, io);
            defer loaded.free(allocator);
            try std.testing.expectEqualSlices(u8, &contents, loaded.constData());
        } else {
            try std.testing.expectEqual(@as(usize, 1), loader.batch_count);
            const expected = if (failure == .reading) error.UnexpectedEndOfFile else error.InvalidTensorRange;
            try std.testing.expectError(expected, loader.checkOpen());
            const result: anyerror!void = if (loader.submit(specs[0..1], .none)) |batch| unexpected: {
                loader.awaitBatch(batch) catch {};
                break :unexpected {};
            } else |err| err;
            try std.testing.expectError(expected, result);
        }
    };
}

test "a submit the loader fails between its open check and its first publish returns that failure, not Closed" {
    const io = std.testing.io;
    const platform = Platform.init(std.testing.allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(std.testing.allocator, io);
    // Fails the loader in the submitting task's next allocation: after
    // `submit`'s open check and before its first publish, where a worker of
    // an older batch can fail it and the scheduler then answers `Closed`.
    const FailNextAllocation = struct {
        parent: std.mem.Allocator,
        loader: *Loader = undefined,
        thread: std.Thread.Id = undefined,
        armed: std.atomic.Value(bool) = .init(false),

        fn arm(self: *@This(), loader: *Loader) void {
            self.loader = loader;
            self.thread = std.Thread.getCurrentId();
            self.armed.store(true, .release);
        }

        fn allocator(self: *@This()) std.mem.Allocator {
            return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
        }

        fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            // The workers allocate on threads of their own.
            if (self.armed.load(.acquire) and self.thread == std.Thread.getCurrentId()) {
                self.armed.store(false, .monotonic);
                self.loader.recordError(error.UnexpectedEndOfFile, "an older batch's read", .{});
            }
            return self.parent.rawAlloc(len, alignment, ret_addr);
        }

        fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.parent.rawResize(memory, alignment, new_len, ret_addr);
        }

        fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.parent.rawRemap(memory, alignment, new_len, ret_addr);
        }

        fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.parent.rawFree(memory, alignment, ret_addr);
        }
    };
    var failing: FailNextAllocation = .{ .parent = std.testing.allocator };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [1024]u8 = undefined;
    const path = path: {
        const file = try tmp.dir.createFile(io, "weights.bin", .{ .read = true });
        defer file.close(io);
        try file.writePositionalAll(io, &.{ 1, 2, 3, 4 }, 0);
        const len = try file.realPath(io, &path_buffer);
        break :path path_buffer[0..len];
    };
    var source: safetensors.Tensor = .{ .file_uri = path, .name = "x", .shape = .init(.{4}, .u8), .offset = 0 };
    var output: Buffer = .{ ._platform = platform, ._shape = source.shape, ._shards = .empty };
    defer output.deinit();
    const loader = try Loader.create(failing.allocator(), io, platform, .{
        .read_parallelism = 2,
        .load_profile = .local,
        .direct_io = .off,
    });
    defer loader.destroy();
    const parts = [_]LoadPart{.whole(&source)};
    failing.arm(loader);
    const result: anyerror!void = if (loader.submit(&.{.{ .shape = source.shape, .output = &output, .parts = &parts }}, .none)) |batch| unexpected: {
        loader.awaitBatch(batch) catch {};
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.UnexpectedEndOfFile, result);
    try std.testing.expectEqual(@as(usize, 0), loader.batch_count);
}

test "source slots keep their own copy of each path" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const contents = [_]u8{ 1, 2, 3, 4 };
    var path_buffer: [1024]u8 = undefined;
    const path = path: {
        const file = try tmp.dir.createFile(io, "weights.bin", .{ .read = true });
        defer file.close(io);
        try file.writePositionalAll(io, &contents, 0);
        const len = try file.realPath(io, &path_buffer);
        break :path path_buffer[0..len];
    };
    const loader = try Loader.create(allocator, io, platform, .{
        .read_parallelism = 2,
        .load_profile = .local,
        .direct_io = .off,
    });
    defer loader.destroy();
    // Each submission names the file through a string of its own, freed
    // right after the submit: workers open the file through its slot.
    for (0..2) |_| {
        const uri = try allocator.dupe(u8, path);
        var source: safetensors.Tensor = .{ .file_uri = uri, .name = "x", .shape = .init(.{4}, .u8), .offset = 0 };
        var output: Buffer = .{ ._platform = platform, ._shape = source.shape, ._shards = .empty };
        defer output.deinit();
        const parts = [_]LoadPart{.whole(&source)};
        const submitted = loader.submit(&.{.{ .shape = source.shape, .output = &output, .parts = &parts }}, .none);
        allocator.free(uri);
        try loader.awaitBatch(try submitted);
        const loaded = try output.toSliceAlloc(allocator, io);
        defer loaded.free(allocator);
        try std.testing.expectEqualSlices(u8, &contents, loaded.constData());
    }
    // The testing allocator overwrites freed bytes: a slot keyed by the
    // first string would miss the second lookup.
    try std.testing.expectEqual(@as(usize, 1), loader.source_slots.count());
}

/// One plain destination per source, as `createDestinations` builds them for
/// whole-tensor specs, without device state: for planner tests.
fn initPlainParts(
    platform: *const Platform,
    sources: []safetensors.Tensor,
    slots: []const *SourceSlot,
    outputs: []Buffer,
    destinations: []Destination,
    parts: []Part,
) void {
    for (sources, slots, outputs, destinations, parts) |*source, slot, *output, *destination, *part| {
        part.* = .{
            .source = source,
            .source_slot = slot,
            .destination = destination,
            .width = source.byteSize(),
        };
        const placed = platform.placedShape(source.shape).packedShape();
        destination.* = .{
            .shape = source.shape,
            .placed_shape = placed,
            .grid = ShardGrid.init(placed) catch unreachable,
            .output = output,
            .parts = @as(*[1]Part, part),
            .unread_bytes = .init(source.byteSize()),
        };
    }
}

/// A plan of `job_count` unit jobs (`file_offset` = index), each with one
/// block and one event slot, and no transfers.
fn testPlan(allocator: std.mem.Allocator, job_count: usize) !*Batch.Plan {
    const jobs = try allocator.alloc(Batch.Plan.Job, job_count);
    errdefer allocator.free(jobs);
    const requests = try allocator.alloc(ReadRequest, job_count);
    errdefer allocator.free(requests);
    @memset(requests, ReadRequest.idle);
    const events = try allocator.alloc(Dma.EventContext, job_count);
    errdefer allocator.free(events);
    for (jobs, 0..) |*job, index| job.* = .{
        .file_offset = index,
        .transfer_start = 0,
        .transfer_len = 0,
        .kind = .{ .dma_read = .{ .len = 1, .minimum_len = 1 } },
    };
    const plan = try allocator.create(Batch.Plan);
    plan.* = .{
        .allocator = allocator,
        .source_slot = undefined,
        .jobs = jobs,
        .transfers = &.{},
        .requests = requests,
        .events = events,
        .source_bytes = job_count,
    };
    return plan;
}

/// Publishes a plan of `job_count` unit jobs into an open batch.
fn publishTestPlan(scheduler: *Scheduler, batch: *Batch, job_count: usize) !void {
    const plan = try testPlan(std.testing.allocator, job_count);
    scheduler.publish(std.testing.io, batch, plan) catch |err| {
        plan.destroy();
        return err;
    };
}

/// `Loader.submit`'s sequence for a one-file submission: publish, then
/// drop the sentinel.
fn publishTestBatch(scheduler: *Scheduler, job_count: usize) !*Batch {
    const batch = try Batch.create(std.testing.allocator, std.testing.io, .{});
    errdefer batch.destroy();
    try publishTestPlan(scheduler, batch, job_count);
    batch.finishJobs(1);
    return batch;
}

/// Claims out and frees an open batch that a test only inspected; it must
/// be the only queued batch.
fn discardTestBatch(scheduler: *Scheduler, batch: *Batch) void {
    const io = std.testing.io;
    var claimed: usize = 0;
    while (scheduler.claim(io)) |claim| {
        std.debug.assert(claim.plan.batch == batch);
        claimed += 1;
    }
    batch.finishJobs(1 + claimed);
    std.debug.assert(batch.done.isSet());
    batch.destroy();
}

test "the planner opens only local files to ask about direct I/O" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const contents: [2 * 4096]u8 = @splat(7);
    var path_buffer: [1024]u8 = undefined;
    const path = path: {
        const file = try tmp.dir.createFile(io, "drafter.bin", .{ .read = true });
        defer file.close(io);
        try file.writePositionalAll(io, &contents, 0);
        const len = try file.realPath(io, &path_buffer);
        break :path path_buffer[0..len];
    };
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    // A scheme whose files the plain io cannot open: asking would fail.
    try vfs.registerBackend("remote", .{ .io = io });
    const vfs_io = vfs.io();
    const reads: Planner.Reads = .{ .io = vfs_io, .vfs = &vfs, .policy = .on, .alignment = 4096 };

    // A remote file is read exact, without an open on the planner.
    var remote: SourceSlot = .{ .uri = "remote://bucket/model.safetensors" };
    defer remote.deinit(vfs_io);
    try std.testing.expectEqual(@as(usize, 0), try reads.alignmentOf(&remote));
    try std.testing.expect(remote.file.readyValue() == null);

    // A local file beside it is opened and asked, whatever the checkpoint:
    // direct where the filesystem allows it.
    var local: SourceSlot = .{ .uri = path };
    defer local.deinit(vfs_io);
    const alignment = try reads.alignmentOf(&local);
    const file = local.file.readyValue().?.*;
    try std.testing.expectEqual(@as(usize, if (vfs.useDirectIo(file, .on)) 4096 else 0), alignment);
}

test "source planner coalesces exact adjacent and overlapping tensor ranges per file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "a", .name = "a0", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a1", .shape = .init(.{4}, .u8), .offset = 14 },
        .{ .file_uri = "a", .name = "a0-copy", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a-gap", .shape = .init(.{4}, .u8), .offset = 20 },
        .{ .file_uri = "b", .name = "b0", .shape = .init(.{12}, .u8), .offset = 3 },
    };
    var slots = [_]SourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var destinations: [sources.len]Destination = undefined;
    var parts: [sources.len]Part = undefined;
    initPlainParts(platform, &sources, &.{ &slots[0], &slots[0], &slots[0], &slots[0], &slots[1] }, &outputs, &destinations, &parts);

    var scheduler: Scheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 8,
        .alignment = 0,
    }, .off);

    // One plan per file. a:[10,18) merges adjacency and the duplicate,
    // a:[20,24) remains exact, and b:[3,15) is split at the request-size
    // boundary.
    const plans = batch.plans.items;
    try std.testing.expectEqual(@as(usize, 2), plans.len);
    try std.testing.expect(plans[0].source_slot == &slots[0]);
    try std.testing.expect(plans[1].source_slot == &slots[1]);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 4), plans[0].transfers.len);
    try std.testing.expectEqual(@as(u64, 10), plans[0].jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), plans[0].jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 3), plans[0].jobs[0].transfer_len);
    try std.testing.expectEqual(@as(u64, 20), plans[0].jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].transfers.len);
    try std.testing.expectEqual(@as(u64, 3), plans[1].jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), plans[1].jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 4), plans[1].jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(u64, 12), plans[0].source_bytes);
    try std.testing.expectEqual(@as(u64, 12), plans[1].source_bytes);
    // One writer per transfer on one device: one event per transfer. Jobs
    // span their 4-byte blocks.
    try std.testing.expectEqual(@as(usize, 4), plans[0].events.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].events.len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].requests.len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs[0].blockLen(4));
    try std.testing.expectEqual(@as(usize, 1), plans[0].jobs[1].blockLen(4));
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs[0].blockLen(4));
    try std.testing.expectEqual(@as(usize, 1), plans[1].jobs[1].blockLen(4));
    for (plans) |plan| {
        for (plan.requests) |*request| try std.testing.expectEqual(@as(usize, 0), request.pending.load(.acquire));
    }
    discardTestBatch(&scheduler, batch);

    var aligned_sources = [_]safetensors.Tensor{
        .{ .file_uri = "aligned", .name = "aligned0", .shape = .init(.{7}, .u8), .offset = 0 },
        .{ .file_uri = "aligned", .name = "aligned1", .shape = .init(.{8}, .u8), .offset = 7 },
        .{ .file_uri = "aligned", .name = "aligned2", .shape = .init(.{5}, .u8), .offset = 15 },
    };
    var aligned_slot: SourceSlot = .{ .uri = "aligned" };
    var aligned_outputs: [aligned_sources.len]Buffer = undefined;
    var aligned_destinations: [aligned_sources.len]Destination = undefined;
    var aligned_items: [aligned_sources.len]Part = undefined;
    initPlainParts(platform, &aligned_sources, &.{ &aligned_slot, &aligned_slot, &aligned_slot }, &aligned_outputs, &aligned_destinations, &aligned_items);
    const aligned_plan = try Planner.prepareTestPlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .block_size = 4,
        .request_size = 8,
        .alignment = 0,
    });
    defer aligned_plan.destroy();
    try std.testing.expectEqual(@as(usize, 3), aligned_plan.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_plan.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_plan.jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 8), aligned_plan.jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 5), aligned_plan.jobs[2].kind.dma_read.len);

    // Widened for direct I/O: the same three jobs (16 - 2 * 4 = 8 still cuts
    // at the tensor-safe boundaries 7 and 15), each read at 4-byte bounds,
    // its transfers relative to the widened start; the last read runs past
    // the tensor data, of which every byte is required.
    const widened_plan = try Planner.prepareTestPlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .block_size = 4,
        .request_size = 16,
        .alignment = 4,
    });
    defer widened_plan.destroy();
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs.len);
    try std.testing.expectEqual(@as(u64, 0), widened_plan.jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 7), widened_plan.jobs[0].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(u64, 4), widened_plan.jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 12), widened_plan.jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 11), widened_plan.jobs[1].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs[1].blockLen(4));
    const second_transfer = widened_plan.transfers[widened_plan.jobs[1].transfer_start];
    try std.testing.expectEqual(@as(usize, 0), second_transfer.block_index);
    try std.testing.expectEqual(@as(usize, 3), second_transfer.block_offset);
    try std.testing.expectEqual(@as(u64, 12), widened_plan.jobs[2].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.transfers[widened_plan.jobs[2].transfer_start].block_offset);
    try std.testing.expectEqual(@as(u64, 20), widened_plan.source_bytes);

    // No room for the widening: two alignment units must fit in a request.
    const invalid_plan: anyerror!void = if (Planner.prepareTestPlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .block_size = 4,
        .request_size = 8,
        .alignment = 4,
    })) |plan| unexpected: {
        plan.destroy();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.InvalidOptions, invalid_plan);
    try std.testing.expectError(error.InvalidOptions, Planner.maximumJobLen(.{
        .block_size = 4,
        .request_size = 16,
        .alignment = 3,
    }));
}

test "scheduler publishes a submission one file at a time and claims the files in order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    // Submitted out of file order: the sort groups "a" before "b".
    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "b", .name = "b0", .shape = .init(.{4}, .u8), .offset = 0 },
        .{ .file_uri = "a", .name = "a1", .shape = .init(.{4}, .u8), .offset = 4 },
        .{ .file_uri = "a", .name = "a0", .shape = .init(.{4}, .u8), .offset = 0 },
    };
    var slots = [_]SourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var destinations: [sources.len]Destination = undefined;
    var parts: [sources.len]Part = undefined;
    initPlainParts(platform, &sources, &.{ &slots[1], &slots[0], &slots[0] }, &outputs, &destinations, &parts);

    var scheduler: Scheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 4,
        .alignment = 0,
    }, .off);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), scheduler.unclaimed_total);
    try std.testing.expect(batch.diagnostics.published_at != null);

    // File a's jobs in offset order, then file b's.
    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[0]);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[0]);
    try std.testing.expectEqual(@as(u64, 4), claim.plan.jobs[claim.index].file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[1]);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    // Both plans left the queue with their last claim; the submission is
    // still open until its sentinel goes.
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    batch.finishJobs(3);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}

test "fifo scheduler claims batches in publish order" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 2);
    const second = try publishTestBatch(&scheduler, 1);
    try std.testing.expectEqual(@as(usize, 3), scheduler.unclaimed_total);

    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == first);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expect(claim.plan == first.plans.items[0] and claim.index == 0);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == first);
    try std.testing.expectEqual(@as(u64, 1), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expectEqual(@as(usize, 1), scheduler.unclaimed_total);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == second);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.unclaimed_total);

    first.finishJobs(2);
    second.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    try std.testing.expect(second.done.isSet());
    first.destroy();
    second.destroy();
}

test "fifo scheduler completes a batch while a later batch has unclaimed jobs" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 1);
    const second = try publishTestBatch(&scheduler, 2);

    try std.testing.expect(scheduler.claim(io).?.plan.batch == first);
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    try std.testing.expect(!second.done.isSet());
    try std.testing.expectEqual(@as(usize, 2), scheduler.unclaimed_total);
    // The completed batch left the queue with its last claim, so it can go
    // away while the other one is still being claimed.
    first.destroy();

    try std.testing.expect(scheduler.claim(io).?.plan.batch == second);
    try std.testing.expect(scheduler.claim(io).?.plan.batch == second);
    try std.testing.expect(scheduler.claim(io) == null);
    second.finishJobs(2);
    try std.testing.expect(second.done.isSet());
    second.destroy();
}

test "fifo scheduler failure retires the unclaimed units of every queued batch" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 2);
    const second = try publishTestBatch(&scheduler, 3);
    // A batch without jobs is never queued and completes at publish.
    const empty = try publishTestBatch(&scheduler, 0);
    try std.testing.expect(empty.done.isSet());
    empty.destroy();

    // One claim in flight: its unit stays with the worker.
    try std.testing.expect(scheduler.claim(io).?.plan.batch == first);
    scheduler.fail(io);
    try std.testing.expect(!first.done.isSet());
    try std.testing.expectEqual(@as(usize, 1), first.remaining.load(.acquire));
    try std.testing.expect(second.done.isSet());
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.unclaimed_total);
    try std.testing.expect(!scheduler.waitForWork(io));
    const publish_result: anyerror!void = if (publishTestBatch(&scheduler, 1)) |_| {} else |err| err;
    try std.testing.expectError(error.Closed, publish_result);
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    first.destroy();
    second.destroy();

    // Everything claimed before the failure: nothing to retire.
    var exhausted: Scheduler = .init(std.testing.allocator);
    defer exhausted.deinit();
    const claimed = try publishTestBatch(&exhausted, 1);
    try std.testing.expect(exhausted.claim(io).?.plan.batch == claimed);
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
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();

    var woke: std.atomic.Value(WaitResult) = .init(.waiting);
    var group: std.Io.Group = .init;
    const Waiter = struct {
        fn run(scheduler_: *Scheduler, io_: std.Io, woke_: *std.atomic.Value(WaitResult)) void {
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
    var scheduler: Scheduler = .init(std.testing.allocator);
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
            scheduler_: *Scheduler,
            batches_: []const *Batch,
            seen_: *std.atomic.Value(u64),
            claim_count_: *std.atomic.Value(usize),
            duplicate_: *std.atomic.Value(bool),
        ) void {
            while (scheduler_.claim(std.testing.io)) |claim| {
                const base: u64 = if (claim.plan.batch == batches_[0]) 0 else 16;
                const mask = @as(u64, 1) << @intCast(base + claim.plan.jobs[claim.index].file_offset);
                if (seen_.fetchOr(mask, .acq_rel) & mask != 0) duplicate_.store(true, .release);
                _ = claim_count_.fetchAdd(1, .monotonic);
            }
        }
    }.run, .{ &scheduler, &batches, &seen, &claim_count, &duplicate });
    try group.await(io);
    try std.testing.expectEqual(std.math.maxInt(u32), @as(u32, @truncate(seen.load(.acquire))));
    try std.testing.expectEqual(@as(usize, 32), claim_count.load(.acquire));
    try std.testing.expect(!duplicate.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.unclaimed_total);
    for (batches) |batch| {
        batch.finishJobs(16);
        try std.testing.expect(batch.done.isSet());
        batch.destroy();
    }
}

test "fifo scheduler queues plans, so an open submission holds no head" {
    const WaitResult = enum(u8) { waiting, work, stopped };
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 1);
    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == batch);
    // The plan left the queue with its last claim; the submission is still
    // open, and holds no place of its own.
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.unclaimed_total);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    try std.testing.expect(!batch.done.isSet());

    // A worker sleeps until the next plan is published.
    var woke: std.atomic.Value(WaitResult) = .init(.waiting);
    var group: std.Io.Group = .init;
    const Waiter = struct {
        fn run(scheduler_: *Scheduler, io_: std.Io, woke_: *std.atomic.Value(WaitResult)) void {
            woke_.store(if (scheduler_.waitForWork(io_)) .work else .stopped, .release);
        }
    };
    try group.concurrent(io, Waiter.run, .{ &scheduler, io, &woke });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(WaitResult.waiting, woke.load(.acquire));
    try publishTestPlan(&scheduler, batch, 2);
    try group.await(io);
    try std.testing.expectEqual(WaitResult.work, woke.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), scheduler.unclaimed_total);

    // The new plan's jobs follow within the same submission.
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == batch);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expectEqual(@as(usize, 1), scheduler.queue.len);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(@as(u64, 1), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    try std.testing.expect(scheduler.claim(io) == null);
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 3), batch.remaining.load(.acquire));
    batch.finishJobs(3);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}

test "fifo scheduler failure retires every published plan of an open batch" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 2);
    try publishTestPlan(&scheduler, batch, 3);
    try std.testing.expectEqual(@as(usize, 5), scheduler.unclaimed_total);
    try std.testing.expectEqual(@as(usize, 6), batch.remaining.load(.acquire));
    try std.testing.expect(scheduler.claim(io).?.plan.batch == batch);

    // The claim keeps its unit; the four unclaimed jobs of both plans are
    // retired; the sentinel is still held.
    scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    try std.testing.expectEqual(@as(usize, 2), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.unclaimed_total);
    try std.testing.expect(scheduler.claim(io) == null);
    // The submission goes on: its next plan is refused, and the sentinel
    // drop leaves only the claim to complete it.
    try std.testing.expectError(error.Closed, publishTestPlan(&scheduler, batch, 1));
    batch.finishJobs(1);
    try std.testing.expect(!batch.done.isSet());
    batch.finishJobs(1);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}

test "device pump admits by bytes and by pieces" {
    var pump: Dma.DevicePump = .{};
    try std.testing.expect(pump.hasRoom(8));
    // The piece that crosses the budget was admitted; the next is not.
    pump.active_bytes = 8;
    try std.testing.expect(!pump.hasRoom(8));
    pump.active_bytes = 7;
    try std.testing.expect(pump.hasRoom(8));
    pump.active_pieces = max_dma_pieces_per_device;
    try std.testing.expect(!pump.hasRoom(8));
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

test "the submission that completes a target's bytes carries the last flag" {
    var target: TensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 100 };
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

/// A single-device loader without PJRT, workers or pinned arenas: enough
/// for request, credit and batch lifecycle tests, with `slots` credits of
/// one 64-byte block. Its other fields are undefined. Must not move after
/// `init`.
const TestLoader = struct {
    memory: []u8,
    blocks: []host_memory.Block,
    pumps: [1]Dma.DevicePump = .{.{}},
    loader: Loader,

    fn init(self: *TestLoader, slots: usize) !void {
        const allocator = std.testing.allocator;
        const memory = try allocator.alloc(u8, slots * 64);
        errdefer allocator.free(memory);
        const blocks = try allocator.alloc(host_memory.Block, slots);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, index| block.* = memory[index * 64 ..][0..64];
        self.* = .{ .memory = memory, .blocks = blocks, .loader = undefined };
        self.loader = .{
            .allocator = allocator,
            .io = std.testing.io,
            .platform = undefined,
            .workspace = undefined,
            .blocks = undefined,
            .scheduler = .init(allocator),
            .read_gate = .init(1),
            .credits = try .init(allocator, blocks, 1),
            .pumps = &self.pumps,
            .dma_budget_bytes = 64,
            .submitted_bytes = undefined,
            .allocated_bytes = undefined,
            .created_at = undefined,
            .plan_config = .{ .block_size = 64, .request_size = 64, .alignment = 0 },
            .direct_io = undefined,
        };
    }

    fn deinit(self: *TestLoader) void {
        const allocator = std.testing.allocator;
        self.loader.scheduler.deinit();
        self.pumps[0].queue.deinit(allocator);
        self.loader.credits.deinit(allocator);
        allocator.free(self.blocks);
        allocator.free(self.memory);
    }

    /// The worker's credit-then-claim sequence.
    fn claimRequest(self: *TestLoader) !*ReadRequest {
        const io = std.testing.io;
        const loader = &self.loader;
        const slot = loader.credits.acquire(io) orelse return error.Closed;
        const claim = loader.scheduler.claim(io) orelse {
            loader.credits.release(io, slot);
            return error.NoJob;
        };
        return ReadRequest.init(loader, claim, slot);
    }
};

test "late vectored callback failure drains and signals completion" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;

    // One request whose only piece is queued behind a DMA slot that a failed
    // callback is about to free.
    const batch = try publishTestBatch(&loader.scheduler, 1);
    const request = try fixture.claimRequest();
    var target: TensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 64 };
    _ = request.pending.fetchAdd(1, .acq_rel);
    try fixture.pumps[0].queue.pushBack(allocator, .{
        .target = &target,
        .request = request,
        .data = loader.credits.slotBlocks(request.slot)[0],
        .destination_offset = 0,
    });
    fixture.pumps[0].active_bytes = 64;
    fixture.pumps[0].active_pieces = 1;
    request.release();
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 1), loader.credits.inUse(io));
    loader.first_error.store(@intFromError(error.Unknown), .release);

    Dma.eventCompleted(loader, 0, 64);
    try std.testing.expectEqual(@as(usize, 0), fixture.pumps[0].active_pieces);
    try std.testing.expectEqual(@as(usize, 0), request.pending.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
    try std.testing.expect(batch.done.isSet());

    Dma.retireBatch(loader, batch);
    batch.destroy();
}

test "batch completes when every claimed request completes" {
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(3);
    defer fixture.deinit();
    const loader = &fixture.loader;

    const batch = try publishTestBatch(&loader.scheduler, 3);
    try std.testing.expectEqual(@as(usize, 3), batch.remaining.load(.acquire));
    var requests: [3]*ReadRequest = undefined;
    for (&requests) |*request| request.* = try fixture.claimRequest();
    try std.testing.expect(loader.scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 3), loader.credits.inUse(io));

    for (requests, 0..) |request, index| {
        try std.testing.expect(!batch.done.isSet());
        try std.testing.expectEqual(requests.len - index, batch.remaining.load(.acquire));
        request.release();
    }
    try std.testing.expect(batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
    // Each claim took its job's slot of the plan's request array.
    for (requests, batch.plans.items[0].requests) |request, *slot| {
        try std.testing.expect(request == slot);
    }

    Dma.retireBatch(loader, batch);
    batch.destroy();
}

test "retirement accepts the idle slots of jobs a failure retired" {
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;

    const batch = try publishTestBatch(&loader.scheduler, 3);
    const request = try fixture.claimRequest();
    // `fail` retires the two unclaimed jobs; their request slots stay idle.
    loader.scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    request.release();
    try std.testing.expect(batch.done.isSet());
    const plan = batch.plans.items[0];
    try std.testing.expect(request == &plan.requests[0]);
    for (plan.requests[1..]) |*idle| {
        try std.testing.expectEqual(@as(usize, 0), idle.pending.load(.acquire));
    }
    // Every slot passes the retirement checks, claimed or idle.
    Dma.retireBatch(loader, batch);
    batch.destroy();
}

test "retired events are destroyed by the next pump or unlinked by the batch retirement" {
    // Without PJRT, contexts with a null event stand in for destroyed ones;
    // the list mechanics are the subject.
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(2);
    defer fixture.deinit();
    const loader = &fixture.loader;

    const batch = try publishTestBatch(&loader.scheduler, 2);
    const requests = [_]*ReadRequest{
        try fixture.claimRequest(),
        try fixture.claimRequest(),
    };
    const plan = batch.plans.items[0];
    plan.events_used.store(2, .release);
    for (plan.events, requests) |*ctx, request| ctx.* = .{
        .request = request,
        .pjrt_event = null,
        .device_index = 0,
        .len = 64,
    };
    // Two callbacks fired: the next pump destroys both, newest first.
    Dma.retireEvent(loader, &plan.events[0]);
    Dma.retireEvent(loader, &plan.events[1]);
    const device_pump = &fixture.pumps[0];
    try std.testing.expect(device_pump.retired == &plan.events[1]);
    try std.testing.expect(plan.events[1].next_retired == &plan.events[0]);
    device_pump.mutex.lockUncancelable(io);
    device_pump.destroyRetired();
    device_pump.mutex.unlock(io);
    try std.testing.expect(device_pump.retired == null);
    try std.testing.expect(plan.events[0].next_retired == null);
    try std.testing.expect(plan.events[1].next_retired == null);

    // A callback that fires after the last pump leaves its context linked;
    // the batch's retirement unlinks it before the batch is freed.
    Dma.retireEvent(loader, &plan.events[0]);
    for (requests) |request| request.release();
    try std.testing.expect(batch.done.isSet());
    Dma.retireBatch(loader, batch);
    try std.testing.expect(device_pump.retired == null);
    batch.destroy();
}

test "overlapping batches complete under concurrent claims and retirement" {
    const io = std.testing.io;
    const worker_count = 4;
    var fixture: TestLoader = undefined;
    try fixture.init(worker_count);
    defer fixture.deinit();
    const loader = &fixture.loader;

    const Worker = struct {
        fn run(loader_: *Loader) void {
            const io_ = std.testing.io;
            while (loader_.scheduler.waitForWork(io_)) {
                const slot = loader_.credits.acquire(io_) orelse return;
                const claim = loader_.scheduler.claim(io_) orelse {
                    loader_.credits.release(io_, slot);
                    continue;
                };
                const request = ReadRequest.init(loader_, claim, slot);
                request.release();
            }
        }
    };
    var group: std.Io.Group = .init;
    defer {
        loader.scheduler.stop(io);
        loader.credits.close(io);
        group.await(io) catch {};
    }
    for (0..worker_count) |_| try group.concurrent(io, Worker.run, .{loader});

    // Three batches in flight at a time, awaited newest first: a batch is
    // retired while later ones are still being claimed.
    for (0..70) |round| {
        var batches: [3]*Batch = undefined;
        var job_counts: [3]usize = undefined;
        for (&batches, &job_counts, 0..) |*batch, *job_count, offset| {
            job_count.* = (round * 3 + offset) % 7;
            batch.* = try publishTestBatch(&loader.scheduler, job_count.*);
        }
        var index = batches.len;
        while (index > 0) {
            index -= 1;
            const batch = batches[index];
            batch.done.waitUncancelable(io);
            Dma.retireBatch(loader, batch);
            batch.destroy();
        }
        try std.testing.expectEqual(@as(usize, 0), loader.scheduler.unclaimed_total);
    }
    // Every request released its credit before its batch completed; the
    // workers only hold one while a claim is in progress.
    loader.scheduler.stop(io);
    loader.credits.close(io);
    try group.await(io);
    try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
}

/// Plans a composite destination fed by three parts from two files, with a
/// gap and a tail of zero fill, beside a plain one, then replays the plans on
/// host memory: every device's placement bytes are written exactly once,
/// from the right source bytes or from zeros.
const CompositeTest = struct {
    /// With a host memory kind the destinations take host jobs: direct
    /// without `alignment`, through its slot's blocks with it. A DMA job is
    /// staged where it saves `staging_gain` submissions; the replay lays
    /// its blocks out as its worker would (`jobBlocks`).
    fn run(platform: *const Platform, shape: Shape, block_size: usize, request_size: usize, memory: Memory, alignment: usize, staging_gain: usize, expect_cross_part_piece: bool) !void {
        const allocator = std.testing.allocator;
        var first_bytes: [96]u8 = undefined;
        for (&first_bytes, 0..) |*byte, i| byte.* = @as(u8, @truncate(i *% 31 +% 7)) | 1;
        var second_bytes: [128]u8 = undefined;
        for (&second_bytes, 0..) |*byte, i| byte.* = @as(u8, @truncate(i *% 17 +% 3)) | 0x80;
        // Rows of 16 bytes. a and b are adjacent in "first" and in the
        // destination; c leaves a five-row gap before it and a two-row tail.
        var sources = [_]safetensors.Tensor{
            .{ .file_uri = "first", .name = "a", .shape = .init(.{ 3, 8 }, .u16), .offset = 4 },
            .{ .file_uri = "first", .name = "b", .shape = .init(.{ 2, 8 }, .u16), .offset = 52 },
            .{ .file_uri = "second", .name = "c", .shape = .init(.{ 4, 8 }, .u16), .offset = 0 },
            .{ .file_uri = "second", .name = "plain", .shape = .init(.{ 4, 8 }, .u16), .offset = 64 },
        };
        var slots = [_]SourceSlot{ .{ .uri = "first" }, .{ .uri = "second" } };
        var outputs: [2]Buffer = undefined;
        var destinations: [2]Destination = undefined;
        var parts = [_]Part{
            .{ .source = &sources[0], .source_slot = &slots[0], .destination = &destinations[0], .dst_offset = 0, .width = 48 },
            .{ .source = &sources[1], .source_slot = &slots[0], .destination = &destinations[0], .dst_offset = 48, .width = 32 },
            .{ .source = &sources[2], .source_slot = &slots[1], .destination = &destinations[0], .dst_offset = 160, .width = 64 },
            .{ .source = &sources[3], .source_slot = &slots[1], .destination = &destinations[1], .width = 64 },
        };
        const plain_shape = shape.setDim(0, 4);
        destinations = .{
            .{ .shape = shape, .placed_shape = platform.placedShape(shape).packedShape(), .grid = try .init(platform.placedShape(shape).packedShape()), .memory = memory, .output = &outputs[0], .parts = parts[0..3], .unread_bytes = .init(144) },
            .{ .shape = plain_shape, .placed_shape = platform.placedShape(plain_shape).packedShape(), .grid = try .init(platform.placedShape(plain_shape).packedShape()), .memory = memory, .output = &outputs[1], .parts = parts[3..4], .unread_bytes = .init(64) },
        };
        const host = destinations[0].hostResident();

        const order = try Planner.sortedPartOrder(allocator, &parts);
        defer allocator.free(order);
        var plans: [2]*Batch.Plan = undefined;
        var plan_count: usize = 0;
        defer for (plans[0..plan_count]) |plan| plan.destroy();
        var start: usize = 0;
        while (start < order.len) : (plan_count += 1) {
            const end = Planner.fileGroupEnd(&parts, order, start);
            plans[plan_count] = try Planner.prepareTestPlan(allocator, &parts, order[start..end], .{
                .block_size = block_size,
                .request_size = request_size,
                .alignment = alignment,
                .minimum_staging_gain = staging_gain,
            });
            start = end;
        }
        try std.testing.expectEqual(@as(usize, 2), plan_count);
        // The fill goes out once, with the plan of the destination's first
        // part, as its last job.
        const fill_kind: std.meta.Tag(Batch.Plan.Job.Kind) = if (host) .host_fill else .dma_fill;
        try std.testing.expectEqual(fill_kind, std.meta.activeTag(plans[0].jobs[plans[0].jobs.len - 1].kind));
        // Host jobs take no DMA event; DMA jobs one
        // event per writer.
        for (plans[0..plan_count], [_]usize{ 1, 0 }) |plan, fill_jobs| {
            var writers: usize = 0;
            var fills: usize = 0;
            for (plan.jobs) |job| {
                const kind = std.meta.activeTag(job.kind);
                if (kind == fill_kind) fills += 1;
                if (host) {
                    const read_kind: std.meta.Tag(Batch.Plan.Job.Kind) = if (alignment == 0) .host_direct else .host_bounce;
                    try std.testing.expect(kind == .host_fill or kind == read_kind);
                } else {
                    try std.testing.expect(kind == .dma_read or kind == .dma_staged or kind == .dma_fill);
                }
                for (plan.transfers[job.transfer_start..][0..job.transfer_len]) |transfer| writers += @popCount(transfer.writer_mask);
            }
            try std.testing.expectEqual(fill_jobs, fills);
            try std.testing.expectEqual(if (host) 0 else writers, plan.events.len);
        }

        const zeros = try allocator.alloc(u8, 256);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        var replays: [2]Replay = undefined;
        var initialized: usize = 0;
        defer for (replays[0..initialized]) |*replay| replay.deinit(allocator);
        for (&replays, &destinations) |*replay, *destination| {
            replay.* = try .init(allocator, destination.placed_shape);
            initialized += 1;
        }
        var cross_part_piece = false;
        const reader: BytesReader = .{ .files = &.{ .{ .slot = &slots[0], .bytes = &first_bytes }, .{ .slot = &slots[1], .bytes = &second_bytes } } };
        for (plans[0..plan_count]) |plan| {
            const file_bytes: []const u8 = if (plan.source_slot == &slots[0]) &first_bytes else &second_bytes;
            for (plan.jobs) |job| {
                const fill = job.kind == .dma_fill or job.kind == .host_fill;
                const blocks = if (host or fill) &.{} else try jobBlocks(allocator, plan, job, reader, block_size);
                defer allocator.free(blocks);
                for (plan.transfers[job.transfer_start..][0..job.transfer_len]) |transfer| {
                    if (host) {
                        // Host transfers address the read, not its blocks.
                        try std.testing.expectEqual(@as(usize, 0), transfer.block_index);
                        switch (job.kind) {
                            .host_direct, .host_bounce => |read| try std.testing.expect(transfer.block_offset + transfer.len <= read.len),
                            else => {},
                        }
                    } else {
                        // A fill job's transfers read the zero block.
                        try std.testing.expect(transfer.block_index < if (fill) 1 else job.blockLen(block_size));
                        try std.testing.expect(transfer.block_offset + transfer.len <= block_size);
                    }
                    const bytes = if (fill) fill: {
                        try std.testing.expectEqual(@as(usize, 0), transfer.block_offset);
                        break :fill zeros[0..transfer.len];
                    } else if (host) read: {
                        const file_offset = job.file_offset + transfer.block_offset;
                        // a's last byte and b's first one in the same piece.
                        if (plan.source_slot == &slots[0] and file_offset <= 51 and file_offset + transfer.len > 52) cross_part_piece = true;
                        break :read file_bytes[file_offset..][0..transfer.len];
                    } else read: {
                        const file_offset = job.file_offset + transfer.block_index * block_size + transfer.block_offset;
                        if (job.kind == .dma_read and plan.source_slot == &slots[0] and file_offset <= 51 and file_offset + transfer.len > 52) cross_part_piece = true;
                        break :read blocks[transfer.block_index * block_size + transfer.block_offset ..][0..transfer.len];
                    };
                    const index: usize = if (transfer.destination == &destinations[0]) 0 else 1;
                    try replays[index].write(transfer.writer_mask, transfer.destination_offset, bytes);
                }
            }
        }
        try std.testing.expectEqual(expect_cross_part_piece, cross_part_piece);

        var global: [256]u8 = @splat(0);
        @memcpy(global[0..80], first_bytes[4..84]);
        @memcpy(global[160..224], second_bytes[0..64]);
        try replays[0].expect(allocator, &global);
        try replays[1].expect(allocator, second_bytes[64..128]);
    }

    /// A DMA job's blocks taken end to end once its worker has read it:
    /// the read in file order, or as its worker lands a staged or gather
    /// job (`landJob`).
    fn jobBlocks(allocator: std.mem.Allocator, plan: *const Batch.Plan, job: Batch.Plan.Job, reader: BytesReader, block_size: usize) ![]u8 {
        const bytes = try allocator.alloc(u8, job.blockLen(block_size) * block_size);
        errdefer allocator.free(bytes);
        @memset(bytes, 0xaa);
        // A bounce buffer as large as any of the job's reads.
        var longest: usize = 0;
        switch (job.kind) {
            .dma_read => |read| {
                // The end of the file cuts a widened read's padding.
                const file_bytes = reader.bytesOf(plan.source_slot).?;
                const data = file_bytes[@intCast(job.file_offset)..][0..@min(read.len, file_bytes.len - job.file_offset)];
                try std.testing.expect(data.len >= read.minimum_len);
                @memcpy(bytes[0..data.len], data);
                return bytes;
            },
            .dma_staged => |staged| {
                // Only a direct read has padding around its tensor bytes.
                if (!plan.direct_reads) try std.testing.expect(staged.data_offset == 0 and staged.read.len == staged.read.minimum_len);
                longest = staged.read.len;
            },
            .dma_gather => |gather| {
                longest = gather.len;
                for (plan.extents[gather.extent_start..][0..gather.extent_len]) |extent| longest = @max(longest, extent.len);
            },
            .dma_fill, .host_direct, .host_bounce, .host_fill => unreachable,
        }
        const blocks = try allocator.alloc(host_memory.Block, job.blockLen(block_size));
        defer allocator.free(blocks);
        for (blocks, 0..) |*block, index| block.* = bytes[index * block_size ..][0..block_size];
        var scratch: ReadRequest.Scratch = try .init(allocator, Planner.blockCount(longest, block_size), block_size, 1);
        defer scratch.deinit();
        try landJob(reader, plan, job, blocks, block_size, &scratch);
        return bytes;
    }

    /// One destination's device buffers, written by replayed transfers.
    const Replay = struct {
        shape: Shape,
        placement: Sharding.Placement,
        writer_size: usize,
        bytes: []u8,
        written: []bool,

        fn init(allocator: std.mem.Allocator, placed_shape: Shape) !Replay {
            const placement: Sharding.Placement = try .init(&placed_shape);
            const writers = placement.mesh().?.devicesInCanonicalOrder().len;
            const writer_size = placement.localShape(placed_shape).byteSize();
            const bytes = try allocator.alloc(u8, writers * writer_size);
            errdefer allocator.free(bytes);
            @memset(bytes, 0xaa);
            const written = try allocator.alloc(bool, bytes.len);
            @memset(written, false);
            return .{ .shape = placed_shape, .placement = placement, .writer_size = writer_size, .bytes = bytes, .written = written };
        }

        fn deinit(self: *Replay, allocator: std.mem.Allocator) void {
            allocator.free(self.written);
            allocator.free(self.bytes);
        }

        /// Each byte of each device at most once.
        fn write(self: *Replay, writer_mask: u64, destination_offset: usize, source: []const u8) !void {
            var mask = writer_mask;
            while (mask != 0) {
                const writer: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                try std.testing.expect(destination_offset + source.len <= self.writer_size);
                const at = writer * self.writer_size + destination_offset;
                for (self.written[at..][0..source.len]) |*flag| {
                    try std.testing.expect(!flag.*);
                    flag.* = true;
                }
                @memcpy(self.bytes[at..][0..source.len], source);
            }
        }

        /// Every device holds its slice of the rank-2 `global`, entirely.
        fn expect(self: *const Replay, allocator: std.mem.Allocator, global: []const u8) !void {
            for (self.written) |flag| try std.testing.expect(flag);
            const shape = self.shape;
            const element_bytes = shape.dtype().sizeOf();
            const row_bytes: usize = @intCast(shape.dim(1) * element_bytes);
            const expected = try allocator.alloc(u8, self.writer_size);
            defer allocator.free(expected);
            for (0..self.placement.mesh().?.devicesInCanonicalOrder().len) |writer| {
                const slices = self.placement.slices(writer, shape);
                const rows = slices.get(0);
                const columns = slices.get(1);
                const len: usize = @intCast(columns.size * element_bytes);
                var cursor: usize = 0;
                for (@intCast(rows.start)..@intCast(rows.start + rows.size)) |row| {
                    @memcpy(expected[cursor..][0..len], global[row * row_bytes + @as(usize, @intCast(columns.start * element_bytes)) ..][0..len]);
                    cursor += len;
                }
                try std.testing.expectEqualSlices(u8, expected, self.bytes[writer * self.writer_size ..][0..self.writer_size]);
            }
        }
    };
};

test "composite destinations place each part and zero-fill the rest of every device" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.init(allocator, io, .cpu, .{
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .cpu = .{ .device_count = 4 },
    }) catch return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    const halves: Sharding.Mesh = try .init("halves", &platform.physical_mesh, .mesh(.{ .model = .high_bandwidth }), .parseBindings(.{ .model = .link_x }));
    const quarters: Sharding.Mesh = try .init("quarters", &platform.physical_mesh, .mesh(.{ .model = .high_bandwidth }), .parseBindings(.{ .model = .{ .link_x, .link_y } }));
    const shape = Shape.init(.{ .row = 16, .col = 8 }, .u16);
    const scenarios = .{
        // Two row shards, each on two devices: b and c straddle no
        // boundary, the gap does.
        .{ &halves, .{ .row = .model }, true },
        // Four row shards: b, c, the gap and the tail straddle shards.
        .{ &quarters, .{ .row = .model }, true },
        // Every row of every source split across two devices.
        .{ &halves, .{ .col = .model }, false },
        // Every byte on every device: one span, so a and b merge.
        .{ &quarters, .{}, true },
    };
    inline for (scenarios) |scenario| {
        const placed = shape.withPartitioning(scenario[0], scenario[1]);
        const never = std.math.maxInt(usize);
        // 64-byte blocks hold a's tail and b's head together; 16-byte ones
        // cut every row apart and every fill piece to one block.
        try CompositeTest.run(platform, placed, 64, 128, .default, 0, never, scenario[2]);
        try CompositeTest.run(platform, placed, 16, 32, .default, 0, never, false);
        // Staged wherever it saves a submission: the column-sharded jobs
        // land grouped by device, in reads exact or widened.
        for ([_]usize{ 0, 8 }) |alignment| {
            try CompositeTest.run(platform, placed, 64, 128, .default, alignment, 1, scenario[2]);
            try CompositeTest.run(platform, placed, 16, 48, .default, alignment, 1, false);
        }
        // Host jobs are not cut by blocks: in one job a and b share a
        // piece wherever the grid allows, read in place or through blocks
        // when the reads are widened. Jobs of 32 bytes cut them apart.
        for ([_]usize{ 0, 8 }) |alignment| {
            try CompositeTest.run(platform, placed, 64, 128, .host_pinned, alignment, never, scenario[2]);
            try CompositeTest.run(platform, placed, 16, 32, .host_pinned, alignment, never, false);
        }
    }
}

test "concurrent loader failures preserve the first error" {
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;
    var group: std.Io.Group = .init;
    defer group.await(io) catch {};
    const Worker = struct {
        fn run(loader_: *Loader, err: StickyError) void {
            loader_.recordError(err, "concurrent source failure", .{});
        }
    };
    for (0..8) |i| try group.concurrent(io, Worker.run, .{ loader, @as(StickyError, if (i % 2 == 0) error.FileNotFound else error.UnexpectedEndOfFile) });
    try group.await(io);
    const first = loader.errorValue().?;
    loader.recordError(error.Internal, "later failure", .{});
    try std.testing.expectEqual(first, loader.errorValue().?);
}

test "read alignment overflow is an invalid option" {
    try std.testing.expectError(error.InvalidOptions, Planner.maximumJobLen(.{
        .block_size = 4096,
        .request_size = 4096,
        .alignment = @as(usize, 1) << (@bitSizeOf(usize) - 1),
    }));
}

test "lazy initialization preserves its finite error across repeated callers" {
    const Context = struct {
        fn init(calls: *usize) error{FileNotFound}!u32 {
            calls.* += 1;
            return error.FileNotFound;
        }
    };
    var once: LazyOnce(u32, *usize, error{FileNotFound}, Context.init) = .{};
    var calls: usize = 0;
    try std.testing.expectError(error.FileNotFound, once.ensure(std.testing.io, &calls));
    try std.testing.expectError(error.FileNotFound, once.ensure(std.testing.io, &calls));
    try std.testing.expectEqual(1, calls);
}

test "host-resident and DMA parts never share a job, and overlapping host parts read through blocks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 2 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    // x, y and z end to end in one file. x lands in two host destinations,
    // y in device memory between them, z in a third host destination.
    const shape = Shape.init(.{ 8, 8 }, .u8);
    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "file", .name = "x", .shape = shape, .offset = 0 },
        .{ .file_uri = "file", .name = "y", .shape = shape, .offset = 64 },
        .{ .file_uri = "file", .name = "z", .shape = shape, .offset = 128 },
    };
    var slot: SourceSlot = .{ .uri = "file" };
    var outputs: [4]Buffer = undefined;
    var destinations: [4]Destination = undefined;
    var parts = [_]Part{
        .{ .source = &sources[0], .source_slot = &slot, .destination = &destinations[0], .width = 64 },
        .{ .source = &sources[1], .source_slot = &slot, .destination = &destinations[1], .width = 64 },
        .{ .source = &sources[0], .source_slot = &slot, .destination = &destinations[2], .width = 64 },
        .{ .source = &sources[2], .source_slot = &slot, .destination = &destinations[3], .width = 64 },
    };
    const memories = [_]Memory{ .host_pinned, .default, .host_pinned, .host_pinned };
    for (&destinations, &outputs, &parts, memories) |*destination, *output, *part, memory| {
        destination.* = .{ .shape = shape, .placed_shape = platform.placedShape(shape).packedShape(), .grid = try .init(platform.placedShape(shape).packedShape()), .memory = memory, .output = output, .parts = @as(*[1]Part, part), .unread_bytes = .init(64) };
    }

    const order = try Planner.sortedPartOrder(allocator, &parts);
    defer allocator.free(order);
    const plan = try Planner.prepareTestPlan(allocator, &parts, order, .{
        .block_size = 64,
        .request_size = 1024,
        .alignment = 0,
    });
    defer plan.destroy();
    // x's run reads its bytes once for two destinations, so through
    // blocks; y goes out by DMA; z is read in place. File order.
    const expected = [_]struct { offset: u64, kind: std.meta.Tag(Batch.Plan.Job.Kind) }{
        .{ .offset = 0, .kind = .host_bounce },
        .{ .offset = 64, .kind = .dma_read },
        .{ .offset = 128, .kind = .host_direct },
    };
    try std.testing.expectEqual(expected.len, plan.jobs.len);
    var dma_writers: usize = 0;
    for (plan.jobs, expected) |job, want| {
        try std.testing.expectEqual(want.offset, job.file_offset);
        try std.testing.expectEqual(want.kind, std.meta.activeTag(job.kind));
        for (plan.transfers[job.transfer_start..][0..job.transfer_len]) |transfer| {
            try std.testing.expectEqual(job.kind != .dma_read, transfer.destination.hostResident());
            if (job.kind == .dma_read) dma_writers += @popCount(transfer.writer_mask);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), plan.jobs[0].transfer_len);
    try std.testing.expectEqual(@as(usize, 1), plan.jobs[0].blockLen(64));
    try std.testing.expectEqual(dma_writers, plan.events.len);
}

test "column-sharded tensors load as one DMA piece per device and block, read exact or widened" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: load_testing.Devices = undefined;
    try devices.init(allocator, io, 4);
    defer devices.deinit();
    const mesh = try devices.mesh("tp");

    // Rows of 32 bytes, 8 per device; then 12,000 one-byte pieces, more
    // than one vectored read takes.
    var rows: [64 * 32]u8 = undefined;
    var cols: [1024 * 32]u8 = undefined;
    var narrow: [3000 * 4]u8 = undefined;
    load_testing.pattern(&rows, 1);
    load_testing.pattern(&cols, 2);
    load_testing.pattern(&narrow, 3);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "staged.safetensors", &registry, &.{
        .{ .name = "rows", .dtype = "U16", .dims = &.{ 64, 16 }, .bytes = &rows },
        .{ .name = "cols", .dtype = "U16", .dims = &.{ 1024, 16 }, .bytes = &cols },
        .{ .name = "narrow", .dtype = "U8", .dims = &.{ 3000, 4 }, .bytes = &narrow },
    });
    const names = [_][]const u8{ "rows", "cols", "narrow" };
    const shapes = [_]Shape{
        Shape.init(.{ .row = 64, .col = 16 }, .u16).withPartitioning(&mesh, .{ .row = .model }),
        Shape.init(.{ .row = 1024, .col = 16 }, .u16).withPartitioning(&mesh, .{ .col = .model }),
        Shape.init(.{ .row = 3000, .col = 4 }, .u8).withPartitioning(&mesh, .{ .col = .model }),
    };
    const globals = [_][]const u8{ &rows, &cols, &narrow };
    const row_pieces = 1024 * 4 + 3000 * 4;

    for ([_]usize{ 0, 64 }) |alignment| {
        var parts: [names.len]LoadPart = undefined;
        var specs: [names.len]LoadSpec = undefined;
        var outputs: [names.len]Buffer = undefined;
        for (&parts, &specs, &outputs, names, shapes) |*part, *spec, *output, name, shape| {
            part.* = .whole(registry.tensors.getPtr(name).?);
            spec.* = .{ .shape = shape, .output = output, .parts = part[0..1] };
        }
        const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 2, .direct_io = .off });
        defer loader.destroy();
        loader.plan_config.alignment = alignment;
        const batch = try loader.submit(&specs, .none);
        // Every plan is published once `submit` returns.
        var staged_jobs: usize = 0;
        var submissions: usize = 0;
        for (batch.plans.items) |plan| {
            for (plan.jobs) |job| staged_jobs += @intFromBool(job.kind == .dma_staged);
            submissions += plan.events.len;
        }
        try loader.awaitBatch(batch);
        defer for (&outputs) |*output| output.deinit();
        for (&outputs, globals) |*output, global| try load_testing.expectShards(allocator, io, output, global);
        try std.testing.expect(staged_jobs >= 1);
        try std.testing.expect(submissions < row_pieces / 100);
        try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
    }
}

test "a strided part whose source does not fit one read is refused before anything is published" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: load_testing.Devices = undefined;
    try devices.init(allocator, io, 1);
    defer devices.deinit();
    var bytes: [32 * 16]u8 = undefined;
    load_testing.pattern(&bytes, 1);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "wide.safetensors", &registry, &.{
        .{ .name = "wide", .dtype = "U8", .dims = &.{ 32, 16 }, .bytes = &bytes },
    });
    // [32, 16] transposed: 512 runs of one byte, its source read whole by
    // every job.
    const part: LoadPart = .{ .source = registry.tensors.getPtr("wide").?, .width = 1, .dims = &.{
        .{ .count = 16, .src_stride = 1, .dst_stride = 32 },
        .{ .count = 32, .src_stride = 16, .dst_stride = 1 },
    } };
    var output: Buffer = undefined;
    const spec: LoadSpec = .{ .shape = .init(.{ 16, 32 }, .u8), .output = &output, .parts = (&part)[0..1] };
    const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 2, .direct_io = .off });
    defer loader.destroy();
    loader.plan_config.request_size = 256;
    if (loader.submit(&.{spec}, .none)) |batch| {
        try loader.awaitBatch(batch);
        output.deinit();
        return error.TestUnexpectedResult;
    } else |err| try std.testing.expectEqual(error.UnsupportedLayout, err);
    // Nothing went out, so the loader takes the next submission.
    loader.plan_config.request_size = 1024;
    try loader.awaitBatch(try loader.submit(&.{spec}, .none));
    defer output.deinit();
    var transposed: [16 * 32]u8 = undefined;
    for (0..16) |column| {
        for (0..32) |row| transposed[column * 32 + row] = bytes[row * 16 + column];
    }
    try load_testing.expectShards(allocator, io, &output, &transposed);
}

test "a strided part of a remote file is checked against its exact source range" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: load_testing.Devices = undefined;
    try devices.init(allocator, io, 1);
    defer devices.deinit();
    var bytes: [128 * 128]u8 = undefined;
    load_testing.pattern(&bytes, 1);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "square.safetensors", &registry, &.{
        .{ .name = "square", .dtype = "U8", .dims = &.{ 128, 128 }, .bytes = &bytes },
    });
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    // The same file through a registered scheme: a remote file, read exact.
    try vfs.registerBackend("remote", .{ .io = io });
    const local = registry.tensors.get("square").?;
    var remote = local;
    var uri_buffer: [1100]u8 = undefined;
    remote.file_uri = try std.fmt.bufPrint(&uri_buffer, "remote://{s}", .{local.file_uri});
    std.debug.assert(local.offset % 4096 != 0);

    var transposed: [128 * 128]u8 = undefined;
    for (0..128) |column| {
        for (0..128) |row| transposed[column * 128 + row] = bytes[row * 128 + column];
    }
    // Transposed: its source read whole by every job, a request exactly,
    // at an offset off the direct I/O alignment.
    const dims = [_]LoadPart.Dim{
        .{ .count = 128, .src_stride = 1, .dst_stride = 128 },
        .{ .count = 128, .src_stride = 128, .dst_stride = 1 },
    };
    const loader = try Loader.create(allocator, vfs.io(), devices.platform, .{ .read_parallelism = 2 });
    defer loader.destroy();
    try std.testing.expectEqual(VFS.direct_io_alignment, loader.plan_config.alignment);
    loader.plan_config.request_size = bytes.len;

    var output: Buffer = undefined;
    const part: LoadPart = .{ .source = &remote, .width = 1, .dims = &dims };
    const spec: LoadSpec = .{ .shape = .init(.{ 128, 128 }, .u8), .output = &output, .parts = (&part)[0..1] };
    try loader.awaitBatch(try loader.submit(&.{spec}, .none));
    defer output.deinit();
    try load_testing.expectShards(allocator, io, &output, &transposed);

    // A local file may be read directly: its range widened to the alignment
    // no longer fits.
    if (VFS.direct_io_alignment != 0) {
        var refused: Buffer = undefined;
        const local_part: LoadPart = .{ .source = &local, .width = 1, .dims = &dims };
        const local_spec: LoadSpec = .{ .shape = .init(.{ 128, 128 }, .u8), .output = &refused, .parts = (&local_part)[0..1] };
        if (loader.submit(&.{local_spec}, .none)) |batch| {
            try loader.awaitBatch(batch);
            refused.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.UnsupportedLayout, err);
    }
}

/// A host-resident load on four devices: rows sharded; columns sharded, two
/// bytes per device per row, so one job has 8,192 transfers and scatters in
/// several reads; replicated; one source twice, replicated and row-sharded,
/// which overlaps; and a device-memory tensor in the same file.
const HostResidentTest = struct {
    /// Each run writes its own file from `seed`: a buffer that lands on
    /// memory an earlier run freed must not find the expected bytes there.
    fn run(devices: *const load_testing.Devices, seed: u8, alignment: usize) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        const platform = devices.platform;
        const mesh = try devices.mesh("tp");

        var rows: [64 * 16]u8 = undefined;
        var dev: [16 * 16]u8 = undefined;
        var cols: [2048 * 8]u8 = undefined;
        var rep: [96 * 4]u8 = undefined;
        var twice: [32 * 4]u8 = undefined;
        load_testing.pattern(&rows, seed);
        load_testing.pattern(&dev, seed +% 1);
        load_testing.pattern(&cols, seed +% 2);
        load_testing.pattern(&rep, seed +% 3);
        load_testing.pattern(&twice, seed +% 4);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var registry: safetensors.TensorRegistry = .init(allocator);
        defer registry.deinit();
        // The device tensor separates the host runs: an overlap sends its
        // whole job through blocks, so the twice-bound source must not share
        // a job with the others.
        try load_testing.writeSafetensors(allocator, io, tmp.dir, "host.safetensors", &registry, &.{
            .{ .name = "rows", .dtype = "U16", .dims = &.{ 64, 8 }, .bytes = &rows },
            .{ .name = "cols", .dtype = "U8", .dims = &.{ 2048, 8 }, .bytes = &cols },
            .{ .name = "rep", .dtype = "U8", .dims = &.{ 96, 4 }, .bytes = &rep },
            .{ .name = "dev", .dtype = "U16", .dims = &.{ 16, 8 }, .bytes = &dev },
            .{ .name = "twice", .dtype = "U8", .dims = &.{ 32, 4 }, .bytes = &twice },
        });

        const names = [_][]const u8{ "rows", "dev", "cols", "rep", "twice", "twice" };
        const shapes = [_]Shape{
            Shape.init(.{ .row = 64, .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model }),
            Shape.init(.{ .row = 16, .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model }),
            Shape.init(.{ .row = 2048, .col = 8 }, .u8).withPartitioning(&mesh, .{ .col = .model }),
            Shape.init(.{ .row = 96, .col = 4 }, .u8).withPartitioning(&mesh, .{}),
            Shape.init(.{ .row = 32, .col = 4 }, .u8).withPartitioning(&mesh, .{}),
            Shape.init(.{ .row = 32, .col = 4 }, .u8).withPartitioning(&mesh, .{ .row = .model }),
        };
        // `rows` as views of pinned memory in the devices' default space.
        const memories = [_]Memory{ .host_mapped, .default, .host_pinned, .host_pinned, .host_pinned, .host_pinned };
        const globals = [_][]const u8{ &rows, &dev, &cols, &rep, &twice, &twice };
        var parts: [names.len]LoadPart = undefined;
        var specs: [names.len]LoadSpec = undefined;
        var outputs: [names.len]Buffer = undefined;
        for (&parts, &specs, &outputs, names, shapes, memories) |*part, *spec, *output, name, shape, memory| {
            part.* = .whole(registry.tensors.getPtr(name).?);
            spec.* = .{ .shape = shape, .memory = memory, .output = output, .parts = part[0..1] };
        }

        const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off });
        defer loader.destroy();
        loader.plan_config.alignment = alignment;
        const live_before = SharedHostAllocation.live.load(.acquire);
        {
            const batch = try loader.submit(&specs, .none);
            try loader.awaitBatch(batch);
            defer for (&outputs) |*output| output.deinit();
            for (&outputs, globals) |*output, global| try load_testing.expectShards(allocator, io, output, global);
            // Every credit, with its blocks, came back.
            try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));

            // Host shards in each device's host memory, or its default
            // memory for views; the replicated ones one allocation.
            for (outputs, memories) |output, memory| {
                const ordered = platform.placedShape(output.shape())._sharding.mesh.?.devicesInCanonicalOrder();
                for (ordered, output._shards.constSlice()) |device, shard| {
                    try std.testing.expect(shard.memory(platform.pjrt_api) == platform.devices[device.id].memory(memory.space()).?.pjrt_memory);
                }
            }
            for (outputs[3..5]) |output| {
                const shards = output._shards.constSlice();
                const first = try shards[0].opaqueDeviceMemoryDataPointer(platform.pjrt_api);
                for (shards[1..]) |shard| {
                    const pointer = try shard.opaqueDeviceMemoryDataPointer(platform.pjrt_api);
                    try std.testing.expect(pointer == first);
                }
            }
            // The two replicated outputs, and one per device of `rows`.
            try std.testing.expectEqual(live_before + 2 + outputs[0]._shards.len, SharedHostAllocation.live.load(.acquire));
            // Shards may be freed in any order: free one replicated output's
            // last first; its allocation goes with its last view.
            std.mem.reverse(*pjrt.Buffer, outputs[3]._shards.slice());
        }
        try std.testing.expectEqual(live_before, SharedHostAllocation.live.load(.acquire));
    }
};

test "host-resident destinations are written in place, through blocks when reads are widened or overlap, and replicas share one allocation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: load_testing.Devices = undefined;
    try devices.init(allocator, io, 4);
    defer devices.deinit();
    if (!backend.supportsMemory(devices.platform, .host_pinned)) return error.SkipZigTest;
    // In place, then through blocks for every host job.
    for ([_]usize{ 0, 64 }, [_]u8{ 1, 5 }) |alignment, seed| {
        try HostResidentTest.run(&devices, seed, alignment);
    }
}
