//! Direct loading follows a batch from per-file planning through FIFO claims,
//! source reads and DMA completion. Runtime ownership and synchronization
//! live here. CUDA, ROCm, oneAPI, Furiosa and CPU load this way
//! (`backend.loadsDirect`).
//!
//! Sizing, fixed by `Loader.create`. Calibration picks the DMA block
//! (dma_calibration.zig); each device's DMA budget is
//! `dma_calibration.depth` blocks. A source request is the larger of
//! the checkpoint's minimum read (`VFS.readHints`) and the block, a whole
//! number of blocks. The source width is `read_parallelism`, clipped to what
//! the mapped ceiling fits.
//! The workspace then maps, once and before the first read, `width + 1`
//! requests, the DMA reserve and the zero block, and cuts them into blocks
//! (`Workspace.carve`): nothing maps during a load. `width + 1` workers
//! share `width` read permits, so a full width of reads is in flight while
//! one worker queues what it read. The lifecycle credits are the whole
//! pinned set in requests, so transfers that still hold blocks keep the DMA
//! stage fed and never want a block the reads did not leave free. The width
//! never changes: a source that answers 429 (or 503 on the object stores)
//! is held one layer down, where the VFS holds every request of that
//! backend (vfs/request.zig), so the traffic falls while the memory bounds
//! stay.
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
const stdx = @import("stdx");
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
const testing = @import("../testing.zig");

/// Creating a tensor's device state: host memory or PJRT, since the front
/// end checked its placement and memory kind. `LazyOnce` replays it to every
/// later reader of the tensor.
const TransferInitError = std.mem.Allocator.Error || pjrt.ApiError;

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
    /// One permit per source read in flight (`readGated`), `read_width` of
    /// them. A throttled source is the VFS's business, which holds every
    /// request of that backend (`vfs/request.zig`); the loader never changes
    /// its width.
    read_permits: std.Io.Semaphore,
    read_width: usize,
    /// Lifecycle credits (`Credits`). One credit beyond the width fed the
    /// DMA engines one request at a time: on a GB300 loading DeepSeek-V4
    /// (pieces of 4 MiB and 256 KiB) that was 24.3 GiB/s at width 16
    /// against 43.8 GiB/s with eight requests queued, and the workers spent
    /// 9 ms per request waiting for a credit against 3 ms reading.
    credits: Credits,
    /// The sticky first error (`recordError`).
    first_error: stdx.FirstError(anyerror) = .{},
    /// One per device (`Dma.DevicePump`). Where several are locked at once
    /// (`Dma.enqueue`, `Dma.retireBatch`) they are taken in device
    /// order; a pump, a completion and `Dma.abortReady` hold one at a time.
    pumps: []Dma.DevicePump,
    /// Per-device in-flight budget: `dma_calibration.depth` blocks, in bytes.
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
    /// The block size and the request size, for the whole load.
    plan_config: Planner.Config,
    /// How each file is read: exact, or aligned for direct I/O.
    reads: Planner.Reads,

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: BackendOptions,
    ) !*Loader {
        // A destination's shards and a transfer's writers (a u64 mask) are
        // per device; Platform refuses devices past its maximum.
        comptime std.debug.assert(Platform.MAX_NUM_DEVICES <= ShardGrid.max_boxes);
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
        // At debug when canceled (`Loader.deinit` cancels a set-up no
        // `start` waited for) or out of memory (the workspace logs what it
        // fails to map or pin).
        const block_size = dma_calibration.calibrate(&workspace, platform, .{ .block_sizes = opts.dma_block_sizes.constSlice() }) catch |err| {
            load_log.failure(err, "calibrate DMA: target={t}", .{platform.target});
            return err;
        };
        // `Loader.init` checked that a request is a whole number of every
        // block calibration may select.
        const request_size = @max(opts.read_hints.read_chunk_size, block_size);
        std.debug.assert(request_size <= backend.max_read_request_size and request_size % block_size == 0);
        const blocks_per_request = request_size / block_size;
        // The DMA stage of every device, kept mapped beside the source
        // working set; at least one maximal source request, so a single
        // request always fits beside it.
        const dma_reserve = @max(
            dma_calibration.depth * platform.devices.len,
            Planner.blockCount(backend.max_read_request_size, block_size),
        );
        // One more block for the zero fill, which no credit owns.
        const zero_blocks = 1;
        const available = workspace.capacity(block_size);
        const width = @min(
            opts.read_parallelism,
            (available -| dma_reserve -| zero_blocks) / blocks_per_request -| 1,
        );
        if (width == 0) {
            load_log.err("host memory ceiling {Bi:.2} fits no source request beside the DMA reserve: available_blocks={d}, reserve_blocks={d}, blocks_per_request={d}: OutOfMemory", .{ host_memory.Workspace.mapped_bytes_ceiling, available, dma_reserve, blocks_per_request });
            return error.OutOfMemory;
        }
        if (width < opts.read_parallelism) {
            load_log.debug("DMA source working set clipped by the mapped ceiling: width={d} of {d}", .{
                width,
                opts.read_parallelism,
            });
        }
        // Two arenas, the reserve first: ROCm balances bytes across its host
        // nodes per allocation, and that split is recorded.
        const pregrowth_started: std.Io.Timestamp = .now(io, .awake);
        const retained_before = workspace.mappedBytes();
        // The workspace logs what it fails to map.
        const blocks = try workspace.carve(block_size, dma_reserve, (width + 1) * blocks_per_request + dma_reserve + zero_blocks);
        errdefer allocator.free(blocks);
        const pregrowth_ns: u64 = @intCast(@max(pregrowth_started.untilNow(io, .awake).nanoseconds, 0));
        load_log.debug("host workspace pregrown: retained={Bi:.2}, pregrown={Bi:.2}, pregrowth_ms={d:.3}", .{
            workspace.mappedBytes(),
            workspace.mappedBytes() - retained_before,
            @as(f64, @floatFromInt(pregrowth_ns)) / std.time.ns_per_ms,
        });
        std.debug.assert(blocks.len != 0);
        @memset(blocks[0], 0);
        var credits: Credits = try .init(allocator, blocks[1..], blocks_per_request);
        errdefer credits.deinit(allocator);
        const workers = width + 1;
        std.debug.assert(credits.count() >= workers);
        const pumps = try allocator.alloc(Dma.DevicePump, platform.devices.len);
        errdefer allocator.free(pumps);
        @memset(pumps, .{});

        const reads: Planner.Reads = .init(io, opts.direct_io);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .workspace = workspace,
            .blocks = blocks,
            .scheduler = .init(allocator),
            .read_permits = .{ .permits = width },
            .read_width = width,
            .credits = credits,
            .pumps = pumps,
            .dma_budget_bytes = dma_calibration.depth * block_size,
            .submitted_bytes = submitted_bytes,
            .allocated_bytes = allocated_bytes,
            .created_at = .now(io, .awake),
            .plan_config = .{
                .block_size = block_size,
                .request_size = request_size,
            },
            .reads = reads,
        };
        std.debug.assert(self.dma_budget_bytes > 0);
        errdefer self.scheduler.deinit();

        errdefer self.stopWorkers();
        for (0..workers) |_| self.worker_group.concurrent(io, workerMain, .{self}) catch |err| {
            load_log.err("start {d} source workers: {s}", .{ workers, @errorName(err) });
            return err;
        };
        load_log.debug("live loader ready: target={s}, request_size={Bi:.2}, direct_io={t}, source_alignment={d}, dma_block_size={Bi:.2}, dma_budget_per_device={Bi:.2}, source_width={d}, lifecycle_credits={d}, workers={d}, retained={Bi:.2}", .{
            @tagName(platform.target),
            request_size,
            opts.direct_io,
            reads.alignment,
            block_size,
            self.dma_budget_bytes,
            width,
            self.credits.count(),
            workers,
            self.workspace.mappedBytes(),
        });
        return self;
    }

    /// Plans `specs` one source file at a time, publishes each file's plan
    /// as soon as it exists behind every earlier submission, then seals the
    /// batch: work on the first file starts while the rest is planned. The
    /// batch owns its destinations and parts until `awaitBatch` retires it.
    /// The specs keep the `LoadSpec` contract, which the front end checked;
    /// a strided part that does not fit one read is `InvalidBinding` before
    /// anything is published (`Planner.checkGatherReads`). A planning
    /// failure fails the loader (a partial submission can never complete):
    /// the batch is awaited here and the caller sees only the first error.
    /// The batch's host-resident destinations are then created ahead of
    /// their reads (`startAllocations`).
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: std.Progress.Node) !*Batch {
        try self.checkOpen();
        const batch = batch: {
            const batch = try Batch.create(self.allocator, self.io, .{ .sequence = self.batch_count });
            errdefer batch.destroy();
            try self.createDestinations(batch, specs, progress);
            break :batch batch;
        };
        var source: []const u8 = "";
        Planner.publishFiles(&self.scheduler, self.io, batch, batch.parts, self.plan_config, self.reads, &source) catch |err| {
            // Fail the loader, seal and await what was published of this
            // batch, and return the loader's first error.
            self.recordError(err, "plan batch {d}, source={s}, published_plans={d}", .{ batch.diagnostics.sequence, source, batch.plans.items.len });
            batch.diagnostics.sealed_at = .now(self.io, .awake);
            self.batch_count += 1;
            batch.finishJobs(1);
            try self.awaitBatch(batch);
            unreachable;
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
        for (batch.destinations) |*destination| host_destinations += @intFromBool(destination.memory.isHost());
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
                .dma_read, .dma_land, .dma_fill => continue,
            }
            for (plan.transfersOf(job)) |transfer| {
                const destination = transfer.destination;
                if (destination == previous or destination.state.readyValue() != null) continue;
                previous = destination;
                if (self.first_error.get() != null) break :walk;
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
    /// the loader's sticky error if the loader failed. Destroying the batch
    /// destroys its transfer managers, and PJRT's manager destructor errors
    /// every buffer whose last transfer never went out, so a buffer the
    /// failure left open reports that error, never success. A transfer
    /// that fails asynchronously errors its buffer's definition event too
    /// (the done event carries no error). A program reading such a buffer
    /// fails its execution, which the front end returns; a bulk buffer
    /// shows the error at its first use. Either way an output of a failed
    /// submission holds buffers whose contents are unspecified, or was
    /// never written. A batch that completed without an error submitted
    /// every byte of every target (the pump flagged the submission that
    /// completed its bytes); one that did not would leave a buffer PJRT
    /// then errors, a bookkeeping bug this panics on. A host-resident
    /// destination has no target: its workers write it in place, which
    /// only the debug check of its progress covers.
    pub fn awaitBatch(self: *Loader, batch: *Batch) !void {
        batch.done.waitUncancelable(self.io);
        const done_at: std.Io.Timestamp = .now(self.io, .awake);
        // Every request of this batch has completed: no worker or callback
        // touches its managers or contexts any more.
        const load_error = self.first_error.get();
        if (load_error == null) for (batch.destinations) |*destination| {
            const submitted = submitted: {
                const state = destination.state.readyValue() orelse break :submitted false;
                for (state.targets) |target| {
                    if (!target.fullySubmitted()) break :submitted false;
                }
                break :submitted true;
            };
            if (!submitted) std.debug.panic("loader: batch {d} completed without submitting every byte of tensor {s} {f}", .{ batch.diagnostics.sequence, destination.name(), destination.shape });
        };
        // Each destination completed its progress item.
        if (load_error == null) for (batch.destinations) |*destination| std.debug.assert(destination.unread_bytes.load(.acquire) == 0);
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
            if (destination.memory.isHost()) continue;
            for (destination.placed_shape._sharding.mesh.?.devicesInCanonicalOrder()) |device| self.submitted_bytes[device.id] += destination.grid.box_bytes;
        }
    }

    fn checkOpen(self: *Loader) !void {
        if (self.first_error.get()) |err| return err;
    }

    /// Creates the batch's destinations, one per spec, and their parts in
    /// its arena, their sources and dimensions staying the specs' until the
    /// batch is awaited (`backend.LoadSpec`). A destination's device state
    /// is created lazily by the first worker that reads or fills for it,
    /// or by the allocation task.
    fn createDestinations(
        self: *Loader,
        batch: *Batch,
        specs: []const LoadSpec,
        progress: std.Progress.Node,
    ) !void {
        const arena = batch.arena.allocator();
        var part_count: usize = 0;
        for (specs) |spec| part_count += spec.parts.len;
        const destinations = try arena.alloc(Destination, specs.len);
        const parts = try arena.alloc(Part, part_count);
        var next_part: usize = 0;
        var logical_bytes: usize = 0;
        for (specs, destinations) |spec, *destination| {
            const placed_shape = self.platform.placedShape(spec.shape).packedShape();
            const destination_parts = parts[next_part..][0..spec.parts.len];
            next_part += spec.parts.len;
            var covered_bytes: u64 = 0;
            var gathered = false;
            for (spec.parts, destination_parts) |part, *slot| {
                slot.* = .{
                    .source = part.source,
                    .source_slot = try self.sourceSlot(part.source.file_uri),
                    .destination = destination,
                    .lattice = part.lattice,
                };
                covered_bytes += part.lattice.width;
                gathered = gathered or part.lattice.dims.len != 0;
            }
            destination.* = .{
                .shape = spec.shape,
                .placed_shape = placed_shape,
                .grid = .init(placed_shape),
                .memory = spec.memory,
                .output = spec.output,
                .parts = destination_parts,
                .gathered = gathered,
                // A gathered destination's jobs land every byte, zeros too.
                .unread_bytes = .init(if (gathered) placed_shape.byteSize() else covered_bytes),
                .progress = progress,
            };
            if (gathered) for (destination_parts) |*part| try Planner.checkGatherReads(self.plan_config, self.reads, part);
            logical_bytes += spec.shape.byteSize();
        }
        batch.destinations = destinations;
        batch.parts = parts;
        batch.diagnostics.source_items = parts.len;
        batch.diagnostics.logical_bytes = logical_bytes;
    }

    /// Releases the destinations' device state, then the batch's arena.
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
        var scratch = ReadRequest.Scratch.init(self.allocator, self.credits.blocks_per_slot, self.plan_config.block_size) catch |err| {
            self.recordError(err, "allocate source worker scratch: blocks={d}", .{self.credits.blocks_per_slot});
            return;
        };
        defer scratch.deinit();
        while (self.scheduler.waitForWork(self.io)) {
            const credit_wait_started = awakeNs(self.io);
            const slot = self.credits.acquire(self.io);
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
                self.recordError(err, "source request: file={s}, job={t}", .{ request.plan.source_slot.uri, request.plan.jobs[request.index].kind });
            };
        }
    }

    /// Stops the workers: every batch was awaited, so nothing is queued and
    /// no credit or permit is held, and `Scheduler.fail` only wakes the
    /// workers waiting for work. Awaiting a group that never spawned
    /// returns at once, so this also cleans up a failed `create`.
    fn stopWorkers(self: *Loader) void {
        std.debug.assert(self.scheduler.queue.len == 0);
        self.scheduler.fail(self.io);
        self.worker_group.await(self.io) catch {};
    }

    /// Keeps `err` as the sticky error unless one is kept already and logs
    /// it, then fails the load without waiting: unclaimed jobs retire
    /// (`Scheduler.fail`) and queued transfers are released
    /// (`Dma.abortReady`). Any thread may call it, a DMA callback included;
    /// requests in flight stop at their next read (`readGated`) or enqueue
    /// and release their units.
    /// Failing other batches than the caller's is safe from any thread:
    /// claims are FIFO and a queued plan holds a unit of its batch, so
    /// `fail` reaches only batches still alive; a batch retires only once
    /// its last unit is released, and `retireBatch` holds every pump lock
    /// while each queued transfer holds its request, so `abortReady` also
    /// reaches only live batches; and `destroy` waits for the worker group
    /// before it frees what the workers use.
    fn recordError(self: *Loader, err: anyerror, comptime fmt: []const u8, args: anytype) void {
        if (self.first_error.set(err)) {
            load_log.failure(err, fmt, args);
            self.scheduler.fail(self.io);
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
            self.read_width,
            self.plan_config.request_size,
        });

        var planning_ns: u64 = 0;
        var source_bytes: u64 = 0;
        var jobs: usize = 0;
        var transfers: usize = 0;
        var dma_submissions: usize = 0;
        var kinds: std.EnumArray(std.meta.Tag(Batch.Plan.Job.Kind), usize) = .initFill(0);
        var gather_jobs: usize = 0;
        for (batch.plans.items) |plan| {
            planning_ns += plan.planning_ns;
            source_bytes += plan.source_bytes;
            jobs += plan.jobs.len;
            transfers += plan.transfers.len;
            dma_submissions += plan.events.len;
            for (plan.jobs) |job| {
                kinds.getPtr(job.kind).* += 1;
                // A gathered destination's landing jobs land it alone.
                if (job.kind == .dma_land and plan.transfersOf(job)[0].destination.gathered) gather_jobs += 1;
            }
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
            kinds.get(.dma_land) - gather_jobs,
            gather_jobs,
            coalescing_ratio,
            average_read_size,
        });
    }

    fn logSummary(self: *Loader) void {
        const reads = self.metrics.read_operations.load(.acquire);
        load_log.debug("loader summary: batches={d}, successful={}, read_bytes={Bi:.2}, elapsed={d:.3}s, reads={d}, source_width={d}, request_size={Bi:.2}, credits_high_water={d} of {d}, pinned_mapped={Bi:.2}", .{
            self.batch_count,
            self.first_error.get() == null,
            self.metrics.read_bytes.load(.acquire),
            secondsBetween(self.created_at, .now(self.io, .awake)),
            reads,
            self.read_width,
            self.plan_config.request_size,
            self.credits.high_water,
            self.credits.count(),
            self.workspace.mappedBytes(),
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
/// request and event context created while loading it, all in the batch's
/// arena until it retires. `remaining` counts completion units: one per
/// published job plus a publish sentinel held until the submission is
/// sealed, and one for the allocation task while it runs
/// (`Loader.startAllocations`), taken before the sentinel is dropped. A
/// job's unit is released exactly once, by whichever of these happens: its
/// request's last reference drops (final DMA callback, a host job's end, or
/// abandonment), or `Scheduler.fail` retires it unclaimed. The batch is
/// done when `remaining` reaches zero; the awaiting task then retires it,
/// so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    /// One file's jobs in claim order with their transfer records and every
    /// context the jobs can need, allocated by the planner: one request per
    /// job and one event per planned DMA submission, handed out in
    /// submission order; a host job needs only its request. Contexts hold the
    /// plan's address, which the batch's arena keeps until it retires.
    const Plan = struct {
        /// A claimable job: its slice of the plan's transfers and what its
        /// worker does for them. Its request slot is its index.
        const Job = struct {
            /// Where the job reads first: its read's start, a landing job's
            /// first extent's; zero for a job that reads nothing.
            /// `Planner.preparePlan` sorts the other jobs by it and appends
            /// the fill jobs after them.
            file_offset: u64,
            transfer_start: usize,
            transfer_len: usize,
            kind: Kind,

            /// How the job's bytes reach their destinations. A destination
            /// in a host memory space (`Memory.isHost`) is written
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
                /// Reads its extents, whose items land their bytes in its
                /// blocks: a read of its plan's file grouped by destination
                /// box (`Planner.JobCut.stage`), or a gathered destination's
                /// range (`Planner.JobCut.appendGatherJobs`).
                dma_land: Land,
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

            /// A landing job: `len` bytes in its blocks, landed by the
            /// `extent_len` extents of the plan's `extents` from
            /// `extent_start`, in order. A gathered range no part covers
            /// has none.
            const Land = struct {
                len: usize,
                extent_start: usize,
                extent_len: usize,
                /// Its blocks start zeroed: bytes no part covers.
                zeroed: bool = false,
            };

            /// The blocks of its credit's slot the job reads into or lands
            /// in; none for a direct host job or a fill job.
            fn blockLen(self: Job, block_size: usize) usize {
                return switch (self.kind) {
                    .dma_read, .host_bounce => |read| Planner.blockCount(read.len, block_size),
                    .dma_land => |land| Planner.blockCount(land.len, block_size),
                    .dma_fill, .host_direct, .host_fill => 0,
                };
            }
        };

        /// Bytes an extent of a landing job places in its blocks: the runs
        /// `first` to `end` of `lattice`, its source offsets in the file
        /// and its destination offsets logical, clipped to the logical
        /// range `[clip_start, clip_end)`. A byte at logical offset `l`,
        /// which `grid.locate(l)` puts at offset `o` of box `b` of the
        /// destination's grid, lands at `landings[landing_start + b] +% o`
        /// of the job's blocks taken end to end (`walkItem`).
        const Item = struct {
            destination: *Destination,
            lattice: backend.Lattice,
            first: u64 = 0,
            end: u64 = 1,
            clip_start: u64,
            clip_end: u64,
            landing_start: usize,
        };

        /// One read of a landing job: `len` bytes from `file_offset` of
        /// `source_slot`'s file, of which the first `minimum_len` exist,
        /// landed by the `item_len` items of the plan's `items` from
        /// `item_start`.
        const Extent = struct {
            source_slot: *SourceSlot,
            file_offset: u64,
            len: usize,
            minimum_len: usize,
            /// Read straight into the job's blocks: an exact read whose
            /// items' runs tile it, in order. Otherwise read into the
            /// worker's bounce buffer and copied from there.
            scatter: bool,
            item_start: usize,
            item_len: usize,
        };

        /// `len` bytes at `landing` of the job's blocks taken end to end,
        /// for every device of `writer_mask` at `destination_offset` of its
        /// shard. A DMA transfer stays inside one block, which `Dma.enqueue`
        /// finds from `landing` (`blockPiece`); a fill one reads the zero
        /// block from its start. A host one is not cut by blocks: its
        /// `landing` counts from the read's start, and a fill one's is
        /// unused.
        const Transfer = struct {
            destination: *Destination,
            landing: usize,
            writer_mask: u64,
            destination_offset: usize,
            len: usize,
        };

        /// The submission this plan belongs to; set when it is published,
        /// under the scheduler mutex, with its completion units.
        batch: *Batch = undefined,
        /// The file every job of the plan reads.
        source_slot: *SourceSlot,
        jobs: []Job,
        transfers: []Transfer,
        /// The landing jobs' extents, their items and the items' landing
        /// tables (one entry per box of the item's destination).
        extents: []Extent = &.{},
        items: []Item = &.{},
        landings: []u64 = &.{},
        requests: []ReadRequest,
        events: []Dma.EventContext,
        /// Event slots handed out so far, one `fetchAdd` per submission from
        /// any device's pump.
        events_used: std.atomic.Value(usize) = .init(0),
        source_bytes: u64,
        planning_ns: u64 = 0,
        /// Next job to claim; owned by the scheduler mutex.
        cursor: usize = 0,

        fn transfersOf(self: *const Plan, job: Job) []const Transfer {
            return self.transfers[job.transfer_start..][0..job.transfer_len];
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

    /// Holds the plans with their contexts, the destinations and the parts
    /// until `destroy`; only the submitting task allocates in it.
    arena: std.heap.ArenaAllocator,
    io: std.Io,
    /// Published plans in file order: appended by the submitting task under
    /// the scheduler mutex.
    plans: std.ArrayList(*Plan) = .empty,
    /// Their device state is the loader's, released before `destroy`.
    destinations: []Destination = &.{},
    /// The destinations' parts, each destination's consecutive and in its
    /// destination order: what the planner sorts by file offset.
    parts: []Part = &.{},
    remaining: std.atomic.Value(usize),
    done: std.Io.Event = .unset,
    diagnostics: Diagnostics,

    /// An open batch holding its publish sentinel and nothing else. The
    /// sentinel keeps the batch from completing before its last plan is
    /// visible.
    fn create(allocator: std.mem.Allocator, io: std.Io, diagnostics: Diagnostics) !*Batch {
        const self = try allocator.create(Batch);
        self.* = .{
            .arena = .init(allocator),
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
        const previous = self.remaining.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.done.set(self.io);
    }

    /// Frees the arena: the plans with their contexts, which
    /// `Dma.retireBatch` must already have retired, the destinations and
    /// the parts.
    fn destroy(self: *Batch) void {
        const allocator = self.arena.child_allocator;
        self.arena.deinit();
        allocator.destroy(self);
    }
};

/// One output buffer and what places it. Owned by the batch. Its parts may
/// come from several files, so several plans can write it: their pieces,
/// and the zero fill, all count toward the same targets (`Target.total` is
/// the placement's bytes, whichever plan submits them), and its device
/// state is created once, by the first worker that reads or fills for it
/// or, for a host-resident one, normally by the allocation task ahead of
/// them.
const Destination = struct {
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
    state: LazyOnce(TensorTransfer) = .{},

    /// How its parts are planned, a class at a time in this order
    /// (`Planner.sortedPartOrder`): DMA jobs, host jobs, or gathered whole.
    const Delivery = enum { dma, host, gathered };

    fn delivery(self: *const Destination) Delivery {
        if (self.gathered) return .gathered;
        return if (self.memory.isHost()) .host else .dma;
    }

    fn ensureState(self: *Destination, direct: *Loader) TransferInitError!*TensorTransfer {
        return self.state.ensure(direct.io, TensorTransfer.init, .{ direct, self });
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
    /// The spec part's; its dimensions are borrowed until the await
    /// (`backend.LoadSpec`).
    lattice: backend.Lattice,

    /// The part's bytes in its file.
    fn fileRange(self: *const Part) error{InvalidBinding}!struct { start: u64, end: u64 } {
        const lattice = self.lattice;
        const start = std.math.add(u64, self.source.offset, lattice.src_offset) catch return self.rangeError(lattice.src_offset);
        const end = std.math.add(u64, start, lattice.width) catch return self.rangeError(lattice.src_offset +| lattice.width);
        return .{ .start = start, .end = end };
    }

    /// The part's source bytes up to `source_end`, from the source's file
    /// offset, pass the largest file offset.
    fn rangeError(self: *const Part, source_end: u64) error{InvalidBinding} {
        load_log.err("destination {f} from {s} in {s}: {d} source bytes from offset {d} pass the largest file offset", .{ self.destination.shape, self.source.name, self.source.file_uri, source_end, self.source.offset });
        return error.InvalidBinding;
    }
};

const SourceSlot = struct {
    uri: []const u8,
    file: LazyOnce(std.Io.File) = .{},
    /// The file's read alignment once the planner decided it
    /// (`Planner.Reads`): 0 when it is read exactly.
    alignment: ?usize = null,

    fn ensure(self: *SourceSlot, io: std.Io) std.Io.File.OpenError!std.Io.File {
        const file = try self.file.ensure(io, std.Io.Dir.openFile, .{ .cwd(), io, self.uri, .{ .mode = .read_only } });
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
        if (destination.memory.isHost()) return initHost(direct, destination);
        const allocator = direct.allocator;
        const platform = direct.platform;
        // `ShardGrid.init` placed this shape already.
        const packed_placement = Sharding.Placement.init(&destination.placed_shape) catch unreachable;
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
        // `ShardGrid.init` placed this shape already.
        const placement = Sharding.Placement.init(&destination.placed_shape) catch unreachable;
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
            // a pinned host space (`backend.supportsMemory`).
            const host = platform.devices[device.id].memory(.host_pinned).?.pjrt_memory;
            if (grid.replicas() > 1 or destination.memory == .host_mapped) {
                const first: usize = @ctz(box_writers[index]);
                if (first == index) shared[index] = try .create(direct.io, platform, dims, element_type, layout, host);
                const allocation = shared[first].?;
                const space = platform.devices[device.id].memory(destination.memory.space()).?.pjrt_memory;
                pjrt_buffers.appendAssumeCapacity(try allocation.view(platform, dims, element_type, layout, space));
                shard.* = allocation.data;
            } else {
                const buffer = try createHostBuffer(direct.io, platform, dims, element_type, layout, host);
                pjrt_buffers.appendAssumeCapacity(buffer);
                shard.* = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
            }
        }

        destination.output.* = .fromPjrtBuffers(platform, destination.shape, pjrt_buffers.constSlice());
        return .{ .host = shards };
    }

    fn deinit(self: *TensorTransfer, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        for (self.targets) |target| target.manager.deinit(api);
        allocator.free(self.targets);
        allocator.free(self.host);
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

/// Builds immutable per-file source jobs and transfer records. Their claim
/// order (file order) is decided here once; the scheduler only publishes and
/// claims those jobs.
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
        /// The longest read: a credit's slot of blocks.
        request_size: usize,
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
    /// Widening buffered reads is not free: a widened read served from the
    /// page cache costs the DMA sources their block alignment for nothing,
    /// and warm replicated Llama on four GB300 lost 3-6% when its first
    /// tensor no longer landed at block offset zero. So a file asks the VFS
    /// actually owning its handle (`VFS.fromIo`, in `init`), both to avoid
    /// that cost and to avoid indexing another VFS's handle table; an `io`
    /// that is no VFS never reads directly.
    const Reads = struct {
        io: std.Io,
        vfs: ?*VFS,
        policy: VFS.DirectIo,
        /// Every read of a file read directly starts and ends at a multiple
        /// of this; 0 reads every file exact.
        alignment: usize,

        fn init(io: std.Io, policy: VFS.DirectIo) Reads {
            const alignment = backend.readAlignment(io, policy);
            return .{ .io = io, .vfs = if (alignment != 0) VFS.fromIo(io) else null, .policy = policy, .alignment = alignment };
        }

        /// Every file read at `alignment`, none opened to ask: for tests.
        fn fixed(alignment: usize) Reads {
            return .{ .io = undefined, .vfs = null, .policy = .off, .alignment = alignment };
        }

        /// The file's alignment, decided the first time it is asked: a
        /// local file is opened here.
        fn alignmentOf(self: Reads, slot: *SourceSlot) !usize {
            const vfs = self.vfs orelse return self.alignment;
            if (slot.alignment) |known| return known;
            const direct = vfs.isLocal(slot.uri) and vfs.useDirectIo(try slot.ensure(self.io), self.policy);
            const alignment = if (direct) self.alignment else 0;
            slot.alignment = alignment;
            return alignment;
        }

        /// The widest alignment `alignmentOf` can give the file, without
        /// opening it: 0 for a remote file.
        fn boundOf(self: Reads, uri: []const u8) usize {
            const vfs = self.vfs orelse return self.alignment;
            return if (vfs.isLocal(uri)) self.alignment else 0;
        }
    };

    /// Plans `parts` one file at a time and publishes each plan as soon as
    /// it exists, so workers claim the first file while the rest is planned,
    /// each plan in the batch's arena. Stops at the first error, with
    /// `source` set to the file being planned, or to the file a gather job
    /// of its plan failed to open: what the failure names. The caller seals
    /// or fails the batch. `reads` says how each file is read (`Reads`).
    /// Whole-model planning delayed the first read by 0.32 s for DeepSeek's
    /// 46 files on one MI300X. Publishing per file overlaps the rest with IO;
    /// Llama's four files took only 1-2 ms total on one B70, so this was
    /// throughput-neutral there.
    fn publishFiles(
        scheduler: *Scheduler,
        io: std.Io,
        batch: *Batch,
        parts: []Part,
        config: Config,
        reads: Reads,
        source: *[]const u8,
    ) !void {
        const order = try sortedPartOrder(scheduler.allocator, parts);
        defer scheduler.allocator.free(order);
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(parts, order, file_start);
            source.* = parts[order[file_start]].source.file_uri;
            const planning_started: std.Io.Timestamp = .now(io, .awake);
            const plan = try preparePlan(scheduler.allocator, batch.arena.allocator(), parts, order[file_start..file_end], config, reads, source);
            plan.planning_ns = @intCast(@max(planning_started.untilNow(io, .awake).nanoseconds, 0));
            try scheduler.publish(io, batch, plan);
            file_start = file_end;
        }
    }

    /// Plans one file: `order` indexes `parts` on the same file, sorted by
    /// delivery class, then offset and size (`sortedPartOrder`). Runs of
    /// touching ranges are cut into the minimum number of jobs at part-safe
    /// boundaries; no job depends on another, since every DMA piece is
    /// submitted as soon as its block is read. With an alignment
    /// (`Reads.alignmentOf`), a job's read is widened to aligned bounds
    /// around its tensor range (`maximumJobLen` leaves room for it); the
    /// transfers address the widened read. A last fill job writes the zero
    /// fill of every destination whose first part is in this file.
    /// Host-resident destinations are written by the worker that reads
    /// them, not by DMA, so their parts are cut into jobs of their own
    /// (`Batch.Plan.Job.Kind`), merged back into file order with the DMA
    /// jobs, and their zero fill is a fill job of its own too. A gathered
    /// destination is planned whole, by logical ranges, with the file of
    /// its first part (`JobCut.appendGatherJobs`); `reads` gives the
    /// alignment of the other files its jobs read, and `source` is set to
    /// one that fails to open. The plan goes in `arena` with its contexts
    /// (`JobCut.finish`); `allocator` holds the cut while it grows.
    fn preparePlan(
        allocator: std.mem.Allocator,
        arena: std.mem.Allocator,
        parts: []Part,
        order: []const usize,
        config: Config,
        reads: Reads,
        source: *[]const u8,
    ) !*Batch.Plan {
        const source_slot = parts[order[0]].source_slot;
        for (order) |part_index| std.debug.assert(std.mem.eql(u8, parts[part_index].source.file_uri, parts[order[0]].source.file_uri));
        const dma_end = deliveryEnd(parts, order, 0, .dma);
        const host_end = deliveryEnd(parts, order, dma_end, .host);
        const alignment = try reads.alignmentOf(source_slot);
        var cut: JobCut = .{
            .allocator = allocator,
            .parts = parts,
            .source_slot = source_slot,
            .config = config,
            .reads = reads,
            .source = source,
            .alignment = alignment,
            .maximum_job_len = maximumJobLen(config.request_size, alignment),
            // A gather job's extents may be in files read directly while
            // this one is not, so its ranges leave room for the widest
            // alignment.
            .gather_len = maximumJobLen(config.request_size, reads.alignment),
        };
        defer cut.deinit();
        try cut.cutRuns(order[0..dma_end], false);
        try cut.cutRuns(order[dma_end..host_end], true);
        for (order[host_end..]) |part_index| {
            const part = &parts[part_index];
            std.debug.assert(part.destination.gathered);
            if (part == &part.destination.parts[0]) try cut.appendGatherJobs(part.destination);
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
        try cut.appendFillJob(order[0..dma_end], false);
        try cut.appendFillJob(order[dma_end..host_end], true);
        return cut.finish(arena);
    }

    /// The end of the parts of `delivery` in `order` from `start` on.
    fn deliveryEnd(parts: []const Part, order: []const usize, start: usize, delivery: Destination.Delivery) usize {
        var end = start;
        while (end < order.len and parts[order[end]].destination.delivery() == delivery) : (end += 1) {}
        return end;
    }

    /// `preparePlan` with every file read at `alignment`, kept in `arena`:
    /// for tests, whose allocator checks the cut for leaks.
    fn prepareTestPlan(arena: std.mem.Allocator, parts: []Part, order: []const usize, config: Config, alignment: usize) !*Batch.Plan {
        var source: []const u8 = "";
        return preparePlan(std.testing.allocator, arena, parts, order, config, .fixed(alignment), &source);
    }

    /// The jobs and transfers of one plan as `preparePlan` cuts them.
    const JobCut = struct {
        /// Host-resident transfers are written by the worker, not cut by
        /// blocks: one transfer per destination run of a job, its
        /// `landing` relative to the job's read start.
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
        /// The file every job of the plan reads.
        source_slot: *SourceSlot,
        config: Config,
        reads: Reads,
        /// Set to a file a gather job fails to open.
        source: *[]const u8,
        /// The alignment of the plan's file (`Reads.alignmentOf`).
        alignment: usize,
        maximum_job_len: usize,
        /// The longest logical range of a gather job.
        gather_len: usize,
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

        /// The plan of the jobs cut, copied into `arena` with every context
        /// its jobs can need, allocated before publication makes it visible
        /// to workers: a request per job and an event per writer of a DMA
        /// job's transfer.
        fn finish(self: *JobCut, arena: std.mem.Allocator) !*Batch.Plan {
            const jobs = self.jobs.items;
            const transfers = self.transfers.items;
            var dma_submissions: usize = 0;
            for (jobs) |job| switch (job.kind) {
                .dma_read, .dma_land, .dma_fill => dma_submissions += submissionCount(transfers[job.transfer_start..][0..job.transfer_len]),
                .host_direct, .host_bounce, .host_fill => {},
            };
            const requests = try arena.alloc(ReadRequest, jobs.len);
            @memset(requests, ReadRequest.idle);
            const plan = try arena.create(Batch.Plan);
            plan.* = .{
                .source_slot = self.source_slot,
                .jobs = try arena.dupe(Batch.Plan.Job, jobs),
                .transfers = try arena.dupe(Batch.Plan.Transfer, transfers),
                .extents = try arena.dupe(Batch.Plan.Extent, self.extents.items),
                .items = try arena.dupe(Batch.Plan.Item, self.items.items),
                .landings = try arena.dupe(u64, self.landings.items),
                .requests = requests,
                .events = try arena.alloc(Dma.EventContext, dma_submissions),
                .source_bytes = self.source_bytes,
            };
            return plan;
        }

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
            const alignment = self.alignment;
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
                        // Less than the run's length, which is a u64.
                        const remaining_capacity = @as(u64, @intCast(jobs_remaining - 1)) * maximum_job_len_u64;
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
                            return error.InvalidBinding;
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
                        const logical_start = part.lattice.dst_offset + (intersection_start - range.start);
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
                    try self.jobs.append(allocator, .{
                        .file_offset = read_start,
                        .transfer_start = transfer_start,
                        .transfer_len = self.transfers.items.len - transfer_start,
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
                if (try self.stage(read, read_start, job_len, in_file_order, slack)) |land| return .{ .dma_land = land };
            }
            for (self.pieces.items) |piece| try self.appendRunTransfers(transfer_start, piece, read_start, block_size);
            return .{ .dma_read = read };
        }

        /// Lands the job's pieces box by box: their bytes on each box of a
        /// destination, grouped by (destination, box) in order of first
        /// appearance, each group in box order. The job's read is its one
        /// extent, and one `Item` per piece tells the worker where its bytes
        /// go (`landJob`). Null, with nothing appended, when that saves
        /// fewer than `Config.minimum_staging_gain` submissions against file
        /// order, estimated at `in_file_order` and at most `slack` fewer:
        /// counted exactly, by listing the runs, only when that decides.
        fn stage(self: *JobCut, read: Batch.Plan.Job.Read, read_start: u64, job_len: usize, in_file_order: usize, slack: usize) !?Batch.Plan.Job.Land {
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
                    .landing = @intCast(landing),
                    .writer_mask = destination.grid.writers[entry.box],
                    .destination_offset = @intCast(entry.box_offset),
                    .len = @intCast(entry.len),
                }, block_size);
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
            // Only a direct read has padding around its tensor bytes, which
            // goes through the worker's bounce buffer.
            const extent_start = self.extents.items.len;
            try self.extents.append(allocator, .{
                .source_slot = self.source_slot,
                .file_offset = read_start,
                .len = read.len,
                .minimum_len = read.minimum_len,
                .scatter = self.alignment == 0,
                .item_start = item_start,
                .item_len = pieces.len,
            });
            return .{ .len = job_len, .extent_start = extent_start, .extent_len = 1 };
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
                const box, const box_offset = grid.locate(at);
                try appendBlocked(self.allocator, &self.transfers, transfer_start, .{
                    .destination = destination,
                    .landing = @intCast(piece.file_offset + (at - piece.start) - read_start),
                    .writer_mask = grid.writers[box],
                    .destination_offset = @intCast(box_offset),
                    .len = @intCast(stop - at),
                }, block_size);
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
        /// in every job, which must fit one read: `InvalidBinding`
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
                const spans = part.lattice.spans().?;
                furthest = @max(furthest, part.lattice.dst_offset + spans.dst);
                at.* = furthest;
            }
            var start: u64 = 0;
            while (start < grid.total) {
                const end = @min(grid.total, start + self.gather_len);
                const transfer_start = self.transfers.items.len;
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
                        .landing = @intCast(landing),
                        .writer_mask = grid.writers[box],
                        .destination_offset = @intCast(low),
                        .len = @intCast(high - low),
                    }, block_size);
                    landing += high - low;
                }
                std.debug.assert(landing == end - start);

                var covered: u64 = 0;
                var index = std.sort.partitionPoint(u64, reach, start, struct {
                    fn below(limit: u64, at: u64) bool {
                        return at <= limit;
                    }
                }.below);
                while (index < parts.len and parts[index].lattice.dst_offset < end) : (index += 1) {
                    const part = &parts[index];
                    const first = part.lattice.firstEndingAfter(start);
                    const last = part.lattice.firstStartingFrom(end);
                    if (first >= last) continue;
                    covered += part.lattice.coveredIn(start, end);
                    try self.appendExtent(part, first, last, start, end, landing_start);
                }
                std.debug.assert(covered <= end - start);
                const job_len: usize = @intCast(end - start);
                const extent_len = self.extents.items.len - extent_start;
                try self.jobs.append(allocator, .{
                    .file_offset = if (extent_len != 0) self.extents.items[extent_start].file_offset else 0,
                    .transfer_start = transfer_start,
                    .transfer_len = self.transfers.items.len - transfer_start,
                    .kind = .{ .dma_land = .{
                        .len = job_len,
                        .extent_start = extent_start,
                        .extent_len = extent_len,
                        .zeroed = covered < end - start,
                    } },
                });
                start = end;
            }
        }

        /// The extent reading `part`'s runs `first` to `last` (exclusive)
        /// clipped to `[start, end)`, with the one item that places them:
        /// from the first run's first byte in the range to the last run's
        /// last one when the part's source runs are in order, else the
        /// part's whole source range.
        fn appendExtent(self: *JobCut, part: *const Part, first: u64, last: u64, start: u64, end: u64, landing_start: usize) !void {
            const lattice = part.lattice;
            const width = lattice.width;
            var source_start = lattice.src_offset;
            var source_end = lattice.src_offset + lattice.spans().?.src;
            const ordered = lattice.srcOrdered();
            if (ordered) {
                const head = lattice.at(first);
                const tail = lattice.at(last - 1);
                source_start = head.src + (start -| head.dst);
                source_end = tail.src + width - ((tail.dst + width) -| end);
            }
            const file_start = std.math.add(u64, part.source.offset, source_start) catch return part.rangeError(source_start);
            const file_end = std.math.add(u64, part.source.offset, source_end) catch return part.rangeError(source_end);
            const alignment = self.reads.alignmentOf(part.source_slot) catch |err| {
                self.source.* = part.source_slot.uri;
                return err;
            };
            const read_start = if (alignment == 0) file_start else std.mem.alignBackward(u64, file_start, alignment);
            const read_end = if (alignment == 0) file_end else std.mem.alignForward(u64, file_end, alignment);
            // A range in order is clipped to the job's, and `checkGatherReads`
            // refused a part whose whole range does not fit.
            std.debug.assert(read_end - read_start <= self.config.request_size);
            try self.extents.append(self.allocator, .{
                .source_slot = part.source_slot,
                .file_offset = read_start,
                .len = @intCast(read_end - read_start),
                .minimum_len = @intCast(file_end - read_start),
                .scatter = alignment == 0 and ordered and lattice.srcTiled(),
                .item_start = self.items.items.len,
                .item_len = 1,
            });
            try self.items.append(self.allocator, .{
                .destination = part.destination,
                .lattice = .{ .src_offset = part.source.offset + lattice.src_offset, .dst_offset = lattice.dst_offset, .width = width, .dims = lattice.dims },
                .first = first,
                .end = last,
                .clip_start = start,
                .clip_end = end,
                .landing_start = landing_start,
            });
            self.source_bytes +|= file_end - file_start;
        }

        /// Appends `piece`, cut at block boundaries, merged with the
        /// previous transfer from `transfer_start` on where block, writers
        /// and both offsets continue it.
        fn appendBlocked(
            allocator: std.mem.Allocator,
            output: *std.ArrayList(Batch.Plan.Transfer),
            transfer_start: usize,
            piece: Batch.Plan.Transfer,
            block_size: usize,
        ) !void {
            var cursor: usize = 0;
            while (cursor < piece.len) {
                const at = piece.landing + cursor;
                const in_block = at % block_size;
                const take = @min(piece.len - cursor, block_size - in_block);
                const destination_offset = piece.destination_offset + cursor;
                cursor += take;
                // A transfer that ends where this one starts is in its
                // block unless a block starts here.
                if (output.items.len > transfer_start and in_block != 0) {
                    const previous = &output.items[output.items.len - 1];
                    if (previous.destination == piece.destination and
                        previous.writer_mask == piece.writer_mask and
                        previous.landing + previous.len == at and
                        previous.destination_offset + previous.len == destination_offset)
                    {
                        previous.len += take;
                        continue;
                    }
                }
                try output.append(allocator, .{
                    .destination = piece.destination,
                    .landing = at,
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

        /// The zero fill of the destinations whose first part is among
        /// `order`, the file's parts of one delivery class, as one job. A DMA
        /// fill job reads one block, the zero block; a host one writes zeros
        /// in place. A gathered destination's jobs write its zeros
        /// themselves.
        fn appendFillJob(self: *JobCut, order: []const usize, host: bool) !void {
            const fill_start = self.transfers.items.len;
            for (order) |part_index| {
                const part = &self.parts[part_index];
                const destination = part.destination;
                if (part != &destination.parts[0]) continue;
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
        }

        /// Whether a host job's transfers address its read from byte zero
        /// to `read_len` in order, each source byte once: then the read
        /// can scatter straight into the destinations. Overlapping or
        /// duplicate parts address some source bytes twice and go through
        /// blocks instead.
        fn coversReadInOrder(transfers: []const Batch.Plan.Transfer, read_len: usize) bool {
            var next: usize = 0;
            for (transfers) |transfer| {
                if (transfer.landing != next) return false;
                next += transfer.len;
            }
            return next == read_len;
        }
    };
    fn blockCount(len: usize, block_size: usize) usize {
        return len / block_size + @intFromBool(len % block_size != 0);
    }

    /// A strided part whose source runs are not back to back reads its whole
    /// source range in every job (`JobCut.appendExtent`): refused before
    /// anything is published when that range, widened to the widest
    /// alignment its file may be read at (`Reads.boundOf`), does not fit
    /// one read.
    fn checkGatherReads(config: Config, reads: Reads, part: *const Part) error{InvalidBinding}!void {
        const lattice = part.lattice;
        if (lattice.srcTiled()) return;
        const alignment = @max(reads.boundOf(part.source.file_uri), 1);
        const start = part.source.offset + lattice.src_offset;
        const end = start + lattice.spans().?.src;
        if (std.mem.alignForward(u64, end, alignment) - std.mem.alignBackward(u64, start, alignment) > config.request_size) {
            load_log.err("destination {f} from {s}: a strided part reads {d} source bytes for every job, more than a read of {d}", .{ part.destination.shape, part.source.name, end - start, config.request_size });
            return error.InvalidBinding;
        }
    }

    /// The longest tensor range one job may cover: the request size, less
    /// the two alignment units its widened read can add, so a widened job
    /// still fits a request (a credit's slot). `Loader.init` refused a
    /// request that cannot hold them.
    fn maximumJobLen(request_size: usize, alignment: usize) usize {
        std.debug.assert(request_size != 0);
        if (alignment == 0) return request_size;
        std.debug.assert(std.math.isPowerOfTwo(alignment) and 2 * alignment < request_size);
        return request_size - 2 * alignment;
    }

    /// Part indices sorted by file URI, delivery class
    /// (`Destination.Delivery`), offset, size and index: the planner's
    /// input, taken one file group at a time (`fileGroupEnd`), each class
    /// in file order. An offset that overflows sorts last in its class;
    /// the planner reports it.
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
                const left_delivery = left.destination.delivery();
                const right_delivery = right.destination.delivery();
                if (left_delivery != right_delivery) return @backingInt(left_delivery) < @backingInt(right_delivery);
                const left_offset = left.source.offset +| left.lattice.src_offset;
                const right_offset = right.source.offset +| right.lattice.src_offset;
                if (left_offset != right_offset) return left_offset < right_offset;
                if (left.lattice.width != right.lattice.width) return left.lattice.width < right.lattice.width;
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
            const lattice = part.lattice;
            if (lattice.dst_offset > covered_end) try appendFillRange(allocator, output, transfer_start, destination, covered_end, lattice.dst_offset, block_size);
            covered_end = lattice.dst_offset + lattice.width;
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
                    .landing = 0,
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
/// jobs are handed out in the planned order (file order); a plan leaves the
/// queue with its last claim or through `fail`.
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
    /// Set by `fail`: `publish` refuses plans (`Canceled`) and `waitForWork`
    /// returns false.
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
    /// diagnostics. `Canceled` once the loader failed, whose first error
    /// `submit` then returns.
    fn publish(self: *Scheduler, io: std.Io, batch: *Batch, plan: *Batch.Plan) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.Canceled;
        const job_count = plan.jobs.len;
        try batch.plans.ensureUnusedCapacity(batch.arena.allocator(), 1);
        if (job_count != 0) try self.queue.ensureUnusedCapacity(self.allocator, 1);
        // Nothing below fails: the plan, its units and the queue entry
        // appear together.
        if (batch.diagnostics.published_at == null) batch.diagnostics.published_at = .now(io, .awake);
        plan.batch = batch;
        batch.plans.appendAssumeCapacity(plan);
        _ = batch.remaining.fetchAdd(job_count, .acq_rel);
        if (job_count != 0) {
            self.queue.pushBackAssumeCapacity(plan);
            self.condition.broadcast(io);
        }
    }

    /// Stops claims and publishes, wakes every worker waiting for work, and
    /// retires the unclaimed units of every queued plan so its batch still
    /// reaches `done` through its claimed requests (an open one through its
    /// seal and sentinel as well). Claims and this pass both move cursors
    /// under the mutex, so they partition the jobs exactly: every claimed
    /// job keeps its unit with its worker. A batch's queued plans are
    /// contiguous here, so the retire that may complete it is the last
    /// access to it and to anything it owns.
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
        if (plan.cursor == plan.jobs.len) _ = self.queue.popFront();
        return .{ .plan = plan, .index = index };
    }

    fn waitForWork(self: *Scheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and self.queue.len == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }
};

/// A claimed source job and its completion state. The plan owns its storage.
const ReadRequest = struct {
    const Scratch = struct {
        allocator: std.mem.Allocator,
        /// A read's ranges: one per block of a job (`readBlocks`), or a
        /// scattered read's, at most one vectored read's worth (`Scatter`).
        iovecs: [][]u8,
        /// A landing job's reads that do not scatter (`landJob`), allocated
        /// on first use: as large as a job's blocks, page aligned for
        /// O_DIRECT.
        bounce: []align(std.heap.page_size_min) u8 = &.{},
        bounce_capacity: usize,

        fn init(allocator: std.mem.Allocator, maximum_blocks: usize, block_size: usize) !Scratch {
            return .{
                .allocator = allocator,
                .iovecs = try allocator.alloc([]u8, @max(maximum_blocks, max_positional_iovecs)),
                .bounce_capacity = maximum_blocks * block_size,
            };
        }

        /// The ranges a `Scatter` gathers before it reads.
        fn scatterIovecs(self: *const Scratch) [][]u8 {
            return self.iovecs[0..max_positional_iovecs];
        }

        fn bounceBuffer(self: *Scratch, len: usize) ![]u8 {
            std.debug.assert(len <= self.bounce_capacity);
            if (self.bounce.len == 0) self.bounce = try self.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), self.bounce_capacity);
            return self.bounce[0..len];
        }

        fn deinit(self: *Scratch) void {
            self.allocator.free(self.bounce);
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
        const job = self.plan.jobs[self.index];
        const transfers = self.plan.transfersOf(job);
        // The first worker to touch a destination creates its state; a
        // host-resident one's is normally created ahead by the allocation
        // task.
        var previous: ?*Destination = null;
        for (transfers) |transfer| {
            const destination = transfer.destination;
            if (destination == previous) continue;
            previous = destination;
            const init_started = awakeNs(io);
            _ = destination.ensureState(loader) catch |err| {
                loader.recordError(err, "initialize tensor {s} from {s}, shape={f}", .{ destination.name(), destination.parts[0].source.file_uri, destination.shape });
                return err;
            };
            _ = loader.metrics.tensor_init_ns.fetchAdd(awakeNs(io) -| init_started, .monotonic);
        }
        switch (job.kind) {
            .dma_read, .dma_land, .dma_fill => {},
            .host_direct, .host_bounce, .host_fill => return self.runHost(scratch),
        }

        const block_size = loader.plan_config.block_size;
        // A fill job's transfers read the zero block, nothing else.
        const blocks: []const host_memory.Block = if (job.kind == .dma_fill)
            loader.blocks[0..1]
        else
            loader.credits.slotBlocks(self.slot)[0..job.blockLen(block_size)];
        switch (job.kind) {
            .dma_read => |read| try self.readBlocks(scratch, blocks, read),
            .dma_land => |land| try landJob(loader, self.plan, land, blocks, scratch),
            .dma_fill => {},
            .host_direct, .host_bounce, .host_fill => unreachable,
        }
        if (job.kind != .dma_fill) {
            for (transfers) |transfer| transfer.destination.recordRead(transfer.len);
        }
        try Dma.enqueue(self, blocks);
    }

    /// A host job: the worker writes its destinations itself, so the job
    /// ends with the worker's scheduling reference; its slot's blocks, if it
    /// reads through them, are done when it returns. Its reads take a read
    /// permit like any other.
    fn runHost(self: *ReadRequest, scratch: *Scratch) !void {
        const loader = self.loader;
        const job = self.plan.jobs[self.index];
        const transfers = self.plan.transfersOf(job);

        if (job.kind == .host_fill) {
            for (transfers) |transfer| {
                const tensor = transfer.destination.state.readyValue().?;
                @memset(tensor.hostLanding(transfer), 0);
            }
            return;
        }

        switch (job.kind) {
            .host_direct => |read| {
                var scatter: Scatter = .{
                    .loader = loader,
                    .slot = self.plan.source_slot,
                    .iovecs = scratch.scatterIovecs(),
                    .file_offset = job.file_offset,
                };
                for (transfers) |transfer| {
                    const tensor = transfer.destination.state.readyValue().?;
                    // The planner checked that the transfers cover the read
                    // in order, each source byte once.
                    try scatter.add(tensor.hostLanding(transfer));
                }
                try scatter.flush();
                std.debug.assert(scatter.file_offset == job.file_offset + read.len);
            },
            .host_bounce => |read| {
                const block_size = loader.plan_config.block_size;
                const blocks = loader.credits.slotBlocks(self.slot)[0..job.blockLen(block_size)];
                try self.readBlocks(scratch, blocks, read);
                for (transfers) |transfer| {
                    const tensor = transfer.destination.state.readyValue().?;
                    // The last block holds stale bytes past the read, which
                    // no bounds check would stop a copy from taking.
                    std.debug.assert(transfer.landing + transfer.len <= read.len);
                    var at = transfer.landing;
                    var rest = tensor.hostLanding(transfer);
                    while (rest.len != 0) {
                        const piece = blockPiece(blocks, block_size, at, rest.len);
                        @memcpy(rest[0..piece.len], piece);
                        rest = rest[piece.len..];
                        at += piece.len;
                    }
                }
            },
            .dma_read, .dma_land, .dma_fill, .host_fill => unreachable,
        }
        for (transfers) |transfer| transfer.destination.recordRead(transfer.len);
    }

    /// Reads the job's `read` into `blocks`, its slot's, in file order.
    fn readBlocks(self: *ReadRequest, scratch: *Scratch, blocks: []const host_memory.Block, read: Batch.Plan.Job.Read) !void {
        const loader = self.loader;
        const block_size = loader.plan_config.block_size;
        const iovecs = scratch.iovecs[0..blocks.len];
        for (iovecs, blocks, 0..) |*iovec, block, block_index| {
            iovec.* = block[0..@min(block_size, read.len - block_index * block_size)];
        }
        try readSlot(loader, self.plan.source_slot, iovecs, self.plan.jobs[self.index].file_offset, read.minimum_len);
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

/// The most ranges one positional read takes.
const max_positional_iovecs: usize = if (@TypeOf(std.posix.IOV_MAX) == void) 64 else std.posix.IOV_MAX;

/// Completes an exact absolute positional scatter read, resuming short reads
/// and respecting the platform iovec limit.
/// Scatters `[file_offset, file_offset + total)` into `buffers`, where
/// `total` is their combined length. The file may end early only past
/// `minimum` bytes: the bytes after it are padding a caller added to align
/// the read, and the end of the file cuts them. Returns the bytes read.
fn readFilePositionalAllV(
    io: std.Io,
    file: std.Io.File,
    buffers: []const []u8,
    file_offset: u64,
    minimum: u64,
) !u64 {
    var read_size: u64 = 0;
    for (buffers) |buffer| {
        const len = std.math.cast(u64, buffer.len) orelse return error.OutOfBounds;
        read_size = std.math.add(u64, read_size, len) catch return error.OutOfBounds;
    }
    _ = std.math.add(u64, file_offset, read_size) catch return error.OutOfBounds;
    if (minimum > read_size) return error.OutOfBounds;
    if (read_size == 0) return 0;

    var batch: [max_positional_iovecs][]u8 = undefined;
    var buffer_index: usize = 0;
    var buffer_offset: usize = 0;
    var completed: u64 = 0;
    while (completed < read_size) {
        var batch_len: usize = 0;
        var batch_size: u64 = 0;
        var scan_index = buffer_index;
        var scan_offset = buffer_offset;
        while (scan_index < buffers.len and batch_len < batch.len) : (scan_index += 1) {
            const buffer = buffers[scan_index];
            if (scan_offset >= buffer.len) {
                scan_offset = 0;
                continue;
            }
            batch[batch_len] = buffer[scan_offset..];
            batch_size += batch[batch_len].len;
            batch_len += 1;
            scan_offset = 0;
        }
        if (batch_len == 0) return error.UnexpectedEndOfFile;

        const absolute_offset = std.math.add(u64, file_offset, completed) catch return error.OutOfBounds;
        const bytes_read = try file.readPositional(io, batch[0..batch_len], absolute_offset);
        if (bytes_read == 0) {
            if (completed >= minimum) return completed;
            return error.UnexpectedEndOfFile;
        }
        completed += @intCast(bytes_read);
        // A short read past `minimum` leaves only padding unread, which no
        // caller addresses; the call that would fetch it (or confirm the
        // end of the file) starts at an unaligned offset, which a file read
        // directly rejects.
        if (bytes_read < batch_size and completed >= minimum) return completed;

        try advanceScatter(buffers, &buffer_index, &buffer_offset, bytes_read);
    }
    return completed;
}

fn advanceScatter(
    buffers: []const []u8,
    buffer_index: *usize,
    buffer_offset: *usize,
    bytes: usize,
) error{UnexpectedEndOfFile}!void {
    var remaining = bytes;
    while (remaining > 0) {
        while (buffer_index.* < buffers.len and buffer_offset.* == buffers[buffer_index.*].len) {
            buffer_index.* += 1;
            buffer_offset.* = 0;
        }
        if (buffer_index.* >= buffers.len) return error.UnexpectedEndOfFile;
        const available = buffers[buffer_index.*].len - buffer_offset.*;
        if (remaining < available) {
            buffer_offset.* += remaining;
            return;
        }
        remaining -= available;
        buffer_index.* += 1;
        buffer_offset.* = 0;
    }
}

/// One source read under a read permit, counted in the loader's metrics.
/// A request claimed before the loader failed stops here, with the
/// loader's first error.
fn readGated(loader: *Loader, file: std.Io.File, iovecs: []const []u8, file_offset: u64, minimum_len: usize) !void {
    const io = loader.io;
    loader.read_permits.waitUncancelable(io);
    defer loader.read_permits.post(io);
    if (loader.first_error.get()) |err| return err;
    const read_started = awakeNs(io);
    const read_result = readFilePositionalAllV(io, file, iovecs, file_offset, minimum_len);
    _ = loader.metrics.read_ns.fetchAdd(awakeNs(io) -| read_started, .monotonic);
    const bytes_read = try read_result;
    _ = loader.metrics.read_operations.fetchAdd(1, .monotonic);
    _ = loader.metrics.read_bytes.fetchAdd(@intCast(bytes_read), .monotonic);
}

/// `readGated` from `slot`'s file, recording a failure with that file: every
/// read a worker makes goes through here, since a landing job's extents may
/// sit in other files than its plan's, and the worker's record of the
/// failure then changes nothing.
fn readSlot(loader: *Loader, slot: *SourceSlot, iovecs: []const []u8, file_offset: u64, minimum_len: usize) !void {
    const file = slot.ensure(loader.io) catch |err| {
        loader.recordError(err, "open {s}", .{slot.uri});
        return err;
    };
    readGated(loader, file, iovecs, file_offset, minimum_len) catch |err| {
        loader.recordError(err, "read {s} at {d}, {d} bytes at least", .{ slot.uri, file_offset, minimum_len });
        return err;
    };
}

/// Reads a landing job's extents and lands their items' runs in its blocks
/// (`Batch.Plan.Item`, `walkItem`). An extent marked for it scatters
/// straight into the blocks, `max_positional_iovecs` ranges per call; any
/// other lands in the worker's bounce buffer and is copied from there: a
/// direct read, whose ranges must be aligned, and a gather read whose
/// source runs are out of order or repeat (a transpose, a broadcast).
fn landJob(
    loader: *Loader,
    plan: *const Batch.Plan,
    land: Batch.Plan.Job.Land,
    blocks: []const host_memory.Block,
    scratch: *ReadRequest.Scratch,
) !void {
    if (land.zeroed) {
        for (blocks) |block| @memset(block[0..loader.plan_config.block_size], 0);
    }
    for (plan.extents[land.extent_start..][0..land.extent_len]) |extent| {
        try landExtent(loader, extent, plan.items[extent.item_start..][0..extent.item_len], plan.landings, blocks, scratch);
    }
}

fn landExtent(
    loader: *Loader,
    extent: Batch.Plan.Extent,
    items: []const Batch.Plan.Item,
    landings: []const u64,
    blocks: []const host_memory.Block,
    scratch: *ReadRequest.Scratch,
) !void {
    const block_size = loader.plan_config.block_size;
    if (extent.scatter) {
        std.debug.assert(extent.len == extent.minimum_len);
        var scatter: Scatter = .{
            .loader = loader,
            .slot = extent.source_slot,
            .iovecs = scratch.scatterIovecs(),
            .file_offset = extent.file_offset,
            .blocks = blocks,
        };
        for (items) |item| try walkItem(item, landings, &scatter);
        try scatter.flush();
        std.debug.assert(scatter.file_offset == extent.file_offset + extent.len);
        return;
    }
    const bounce = try scratch.bounceBuffer(extent.len);
    try readSlot(loader, extent.source_slot, &.{bounce}, extent.file_offset, extent.minimum_len);
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
            const box, const box_offset = grid.locate(at);
            try sink.place(source, table[box] +% box_offset, piece_end - at);
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
        const box, const box_offset = grid.locate(at);
        const landing: usize = @intCast(table[box] +% box_offset);
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

/// A read scattered over memory ranges as they come, in file order and
/// back to back from `file_offset` of `slot`'s file: a range that continues
/// the previous one in memory merges with it, and one call reads at most
/// `iovecs.len` ranges. Fed destination ranges directly (`add`), or the runs
/// of a walk landing in `blocks` (`place`). Every byte read must exist: a
/// scattered read has no alignment padding.
const Scatter = struct {
    loader: *Loader,
    slot: *SourceSlot,
    iovecs: [][]u8,
    count: usize = 0,
    /// Bytes the pending ranges cover.
    pending: usize = 0,
    /// The file offset of the first byte not read yet.
    file_offset: u64,
    blocks: []const host_memory.Block = &.{},

    fn add(self: *Scatter, range: []u8) !void {
        if (self.count != 0) {
            const last = &self.iovecs[self.count - 1];
            if (last.ptr + last.len == range.ptr) {
                last.len += range.len;
                self.pending += range.len;
                return;
            }
        }
        if (self.count == self.iovecs.len) try self.flush();
        self.iovecs[self.count] = range;
        self.count += 1;
        self.pending += range.len;
    }

    /// A run's landing in the blocks, split at block boundaries.
    fn place(self: *Scatter, source: u64, landing: u64, len: u64) !void {
        std.debug.assert(source == self.file_offset + self.pending);
        var at: usize = @intCast(landing);
        var rest: usize = @intCast(len);
        while (rest != 0) {
            const piece = blockPiece(self.blocks, self.loader.plan_config.block_size, at, rest);
            try self.add(piece);
            at += piece.len;
            rest -= piece.len;
        }
    }

    fn flush(self: *Scatter) !void {
        if (self.count == 0) return;
        try readSlot(self.loader, self.slot, self.iovecs[0..self.count], self.file_offset, self.pending);
        self.file_offset += self.pending;
        self.count = 0;
        self.pending = 0;
    }
};

/// Runs copied from a read in the bounce buffer, which starts at
/// `file_offset`.
const CopySink = struct {
    bounce: []const u8,
    file_offset: u64,
    blocks: []const host_memory.Block,
    block_size: usize,

    fn place(self: *CopySink, source: u64, landing: u64, len: u64) !void {
        var at: usize = @intCast(landing);
        var rest = self.bounce[@intCast(source - self.file_offset)..][0..@intCast(len)];
        while (rest.len != 0) {
            const piece = blockPiece(self.blocks, self.block_size, at, rest.len);
            @memcpy(piece, rest[0..piece.len]);
            rest = rest[piece.len..];
            at += piece.len;
        }
    }
};

/// The bytes from offset `at` of `blocks` taken end to end, `block_size`
/// bytes each, up to `len` of them or the end of that block.
fn blockPiece(blocks: []const host_memory.Block, block_size: usize, at: usize, len: usize) []u8 {
    const in_block = at % block_size;
    return blocks[at / block_size][in_block..][0..@min(len, block_size - in_block)];
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
    /// unclosed (on Llama, whose tensors span blocks).
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
    /// submitters took 4.56-4.58 s but left targets unclosed in two of four
    /// Llama runs; DeepSeek's mostly single-piece tensors hid the bug.
    const DevicePump = struct {
        mutex: std.Io.Mutex = .init,
        queue: ReadyQueue = .empty,
        /// In-flight DMA bytes and submissions; owned by `mutex`.
        active_bytes: usize = 0,
        active_pieces: usize = 0,
        pumping: bool = false,
        /// Contexts whose callback fired, for the next pump or a batch's
        /// retirement to destroy: an intrusive stack through
        /// `EventContext.next_retired`, owned by `mutex`. Destroying an
        /// event from the next pump instead of at its batch's retirement
        /// bounds live PJRT events by the DMA width plus one pump batch
        /// rather than by a submission's transfer count; every shipped
        /// plugin accepted destruction outside the event's own callback
        /// (16,384 fired events on two B70, CUDA and ROCm alike).
        retired: ?*EventContext = null,

        /// Whether the next queued transfer may go. The budget is in bytes,
        /// not submissions: `dma_calibration.depth` blocks, and a piece is
        /// one tensor's slice of a block, so a model of small tensors needs
        /// many more pieces in flight to keep the same bytes moving: eight
        /// average DeepSeek pieces occupy only ~18 MiB against the 128 MiB
        /// budget of 16 MiB blocks. On one GB300, widening the stage AND
        /// adding lifecycle credits raised DeepSeek from 24 to 44 GiB/s;
        /// this was not a byte-budget-only experiment. The piece that
        /// crosses the budget is admitted: a device with room always has a
        /// transfer in flight.
        fn hasRoom(self: *const DevicePump, budget_bytes: usize) bool {
            return self.active_bytes < budget_bytes and self.active_pieces < max_dma_pieces_per_device;
        }

        /// `mutex`. Destroys every retired event: their callbacks have run,
        /// and only a pump or a batch's retirement destroys an event,
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
        /// Null once destroyed: by a pump or a batch's retirement after its
        /// callback ran, or at once by a failed `onReady`.
        pjrt_event: ?*pjrt.Event,
        device_index: usize,
        /// Bytes this submission holds against the device's in-flight budget.
        len: usize,
        submitted_ns: u64 = 0,
        /// Link of the device pump's `retired`; owned by that pump's mutex.
        next_retired: ?*EventContext = null,

        /// Pump mutex, unless no callback was registered. Destroys the
        /// event, once. The platform is reached through the request, and
        /// only when there is something to destroy: a context whose event
        /// is null never touches it.
        fn destroyEvent(self: *EventContext) void {
            if (self.pjrt_event) |event| event.deinit(self.request.loader.platform.pjrt_api);
            self.pjrt_event = null;
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

    /// Retires the contexts of a done batch from the awaiting task, after
    /// their callbacks fired, exactly as `pjrt.Event.await` does: drains
    /// every pump's `retired`, whichever batch its contexts belong to, and
    /// checks that every request completed. Runs with every pump locked so
    /// an `abortReady` still iterating queued entries or a pump draining its
    /// `retired` cannot race the free that follows.
    fn retireBatch(loader: *Loader, batch: *Batch) void {
        std.debug.assert(batch.done.isSet());
        lockAllPumps(loader);
        defer unlockAllPumps(loader);
        for (loader.pumps) |*device_pump| device_pump.destroyRetired();
        for (batch.plans.items) |plan| {
            for (plan.requests) |*request| std.debug.assert(request.pending.load(.acquire) == 0);
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

    /// Queues every piece of the request's job before pumping. The former
    /// per-piece path paid roughly 69k-79k mutex/pump trips per DeepSeek
    /// load; coalesced reads alone did not remove that cost. A transfer
    /// reads `len` bytes from its `landing` of `blocks` taken end to end,
    /// inside one block, once for each of its writers: the devices the
    /// planner planned for (both come from the sharding's canonical device
    /// order).
    fn enqueue(request: *ReadRequest, blocks: []const host_memory.Block) !void {
        const loader = request.loader;
        const block_size = loader.plan_config.block_size;
        const transfers = request.plan.transfersOf(request.plan.jobs[request.index]);
        var counts: [Platform.MAX_NUM_DEVICES]usize = @splat(0);
        var pieces: usize = 0;
        for (transfers) |transfer| {
            const tensor = transfer.destination.state.readyValue().?;
            pieces += @popCount(transfer.writer_mask);
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                counts[tensor.targets[writer_index].device_index] += 1;
            }
        }
        const enqueued_ns = awakeNs(loader.io);
        // Every pump is held while the request's pieces land, and every
        // queue is reserved before any changes, so the request is either
        // fully queued or not queued at all.
        lockAllPumps(loader);
        {
            errdefer unlockAllPumps(loader);
            // A failure recorded before these locks may have drained the
            // queues already (`abortReady`), and nothing would complete what
            // this queued; one recorded after them drains it.
            if (loader.first_error.get()) |err| return err;
            for (loader.pumps, counts[0..loader.pumps.len]) |*device_pump, count| {
                try device_pump.queue.ensureUnusedCapacity(loader.allocator, count);
            }
        }
        // Every piece holds the request, and with it the slot, until its
        // callback.
        _ = request.pending.fetchAdd(pieces, .acq_rel);
        request.enqueued_ns = enqueued_ns;
        for (transfers) |transfer| {
            // The planner cut the transfers to the job's blocks.
            const data = blockPiece(blocks, block_size, transfer.landing, transfer.len);
            std.debug.assert(data.len == transfer.len);
            const tensor = transfer.destination.state.readyValue().?;
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const target = &tensor.targets[writer_index];
                loader.pumps[target.device_index].queue.pushBackAssumeCapacity(.{
                    .target = target,
                    .request = request,
                    .data = data,
                    .destination_offset = transfer.destination_offset,
                });
            }
        }
        unlockAllPumps(loader);
        for (counts[0..loader.pumps.len], 0..) |count, device_index| {
            if (count != 0) requestPump(loader, device_index);
        }
    }

    /// Runs the device's pump on this thread unless one is already running.
    fn requestPump(loader: *Loader, device_index: usize) void {
        const device_pump = &loader.pumps[device_index];
        device_pump.mutex.lockUncancelable(loader.io);
        if (device_pump.pumping or loader.first_error.get() != null) {
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
            if (loader.first_error.get() == null) {
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
        // owns it. A pump or a batch's retirement destroys the event once
        // its callback has run; a failed `onReady` destroys it at once.
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
                // the batch. Load every field first, retire the DMA slot
                // (which may pump on this thread), hand the context to the
                // next pump, and release the request last.
                const request = ctx_.request;
                const loader_ = request.loader;
                const api_ = loader_.platform.pjrt_api;
                const device_index = ctx_.device_index;
                const len = ctx_.len;
                _ = loader_.metrics.dma_piece_ns.fetchAdd(awakeNs(loader_.io) -| ctx_.submitted_ns, .monotonic);
                // The shipped plugins resolve this event without an error
                // whatever happened to the copy (the C API wrapper sets the
                // promise with an OK status); kept for a plugin that reports.
                // The callback owns the error.
                if (err) |pjrt_error| {
                    loader_.recordError(pjrt_error.getCode(api_).toApiError(), "complete DMA transfer: device={d}, bytes={d}, message={s}", .{ device_index, len, pjrt_error.getMessage(api_) });
                    pjrt_error.deinit(api_);
                }
                eventCompleted(loader_, device_index, len);
                // After the pump this callback may have run, so the event is
                // destroyed by a later pump or by a batch's retirement,
                // never inside its own callback.
                retireEvent(loader_, ctx_);
                request.release();
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(api) catch {};
            ctx.destroyEvent();
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
        if (loader.first_error.get() != null)
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

    /// Waits for a free slot. Requests in flight give theirs back even after
    /// a failure, so a waiter always gets one.
    fn acquire(self: *Credits, io: std.Io) u32 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.free_len == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
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

/// A value initialized at most once by whichever task touches it first,
/// with the initializer that task passes. Concurrent callers wait on the
/// event; a failed initialization keeps its error code and re-materializes
/// the same error for every later caller.
fn LazyOnce(comptime T: type) type {
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

        /// The error set of `initFn`, which returns `T`.
        fn InitError(comptime initFn: anytype) type {
            return @typeInfo(@typeInfo(@TypeOf(initFn)).@"fn".return_type.?).error_union.error_set;
        }

        fn ensure(self: *Self, io: std.Io, comptime initFn: anytype, args: std.meta.ArgsTuple(@TypeOf(initFn))) InitError(initFn)!*T {
            while (true) switch (self.status.load(.acquire)) {
                .uninitialized => {
                    if (self.status.cmpxchgStrong(.uninitialized, .initializing, .acq_rel, .acquire) != null) continue;
                    self.value = @call(.auto, initFn, args) catch |err| {
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
    const path = try writeTestFile(tmp.dir, io, "weights.bin", &contents, &path_buffer);
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
                try std.testing.expectError(error.InvalidBinding, result);
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
                try std.testing.expectError(error.InvalidBinding, result);
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
        try std.testing.expectEqual(@as(usize, 0), loader.scheduler.queue.len);
        try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
        try std.testing.expectEqual(loader.read_width, loader.read_permits.permits);

        // Each failure fails the loader: a later submission gets it.
        try std.testing.expectEqual(@as(usize, 1), loader.batch_count);
        const expected = if (failure == .reading) error.UnexpectedEndOfFile else error.InvalidBinding;
        try std.testing.expectError(expected, loader.checkOpen());
        const result: anyerror!void = if (loader.submit(specs[0..1], .none)) |batch| unexpected: {
            loader.awaitBatch(batch) catch {};
            break :unexpected {};
        } else |err| err;
        try std.testing.expectError(expected, result);
    };
}

test "exactly the width of reads are in flight, and a read failing mid-load leaves nothing held" {
    const allocator = std.testing.allocator;
    const platform = Platform.init(allocator, std.testing.io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, std.testing.io);
    // Every source read holds for a moment and is counted; the one numbered
    // `fail_at` fails.
    const Reads = struct {
        var in_flight: std.atomic.Value(usize) = .init(0);
        var peak: std.atomic.Value(usize) = .init(0);
        var started: std.atomic.Value(usize) = .init(0);
        var fail_at: usize = std.math.maxInt(usize);

        fn readPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
            _ = peak.fetchMax(in_flight.fetchAdd(1, .acq_rel) + 1, .acq_rel);
            defer _ = in_flight.fetchSub(1, .acq_rel);
            std.testing.io.sleep(.fromMilliseconds(5), .awake) catch {};
            if (started.fetchAdd(1, .acq_rel) == fail_at) return error.InputOutput;
            return std.testing.io.vtable.fileReadPositional(userdata, file, data, offset);
        }
    };
    var vtable = std.testing.io.vtable.*;
    vtable.fileReadPositional = Reads.readPositional;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };

    // Sixteen tensors with gaps between them: a read, and a job, each.
    const tensors = 16;
    var contents: [tensors * 8]u8 = undefined;
    for (&contents, 0..) |*byte, index| byte.* = @truncate(index);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [1024]u8 = undefined;
    const path = try writeTestFile(tmp.dir, std.testing.io, "weights.bin", &contents, &path_buffer);
    var sources: [tensors]safetensors.Tensor = undefined;
    var parts: [tensors]LoadPart = undefined;
    var specs: [tensors]LoadSpec = undefined;
    var outputs: [tensors]Buffer = undefined;
    for (&sources, &parts, &specs, &outputs, 0..) |*source, *part, *spec, *output, index| {
        source.* = .{ .file_uri = path, .name = "t", .shape = .init(.{4}, .u8), .offset = index * 8 };
        part.* = .whole(source);
        spec.* = .{ .shape = source.shape, .output = output, .parts = part[0..1] };
    }

    for ([_]?usize{ null, 5 }) |fail_at| {
        for (&outputs) |*output| output.* = .{ ._platform = platform, ._shape = sources[0].shape, ._shards = .empty };
        defer for (&outputs) |*output| output.deinit();
        // `destroy` returns: no worker waits for a credit or a permit.
        const loader = try Loader.create(allocator, io, platform, .{
            .read_parallelism = 2,
            .direct_io = .off,
        });
        defer loader.destroy();
        try std.testing.expectEqual(@as(usize, 2), loader.read_width);
        // `create` reads too, the host's memory layout.
        Reads.in_flight = .init(0);
        Reads.peak = .init(0);
        Reads.started = .init(0);
        Reads.fail_at = fail_at orelse std.math.maxInt(usize);
        const batch = try loader.submit(&specs, .none);
        if (fail_at == null) {
            try loader.awaitBatch(batch);
            // Three workers, two permits.
            try std.testing.expectEqual(@as(usize, tensors), Reads.started.load(.acquire));
            try std.testing.expectEqual(loader.read_width, Reads.peak.load(.acquire));
            try testing.expectShards(allocator, std.testing.io, &outputs[tensors - 1], contents[(tensors - 1) * 8 ..][0..4], .default);
        } else {
            try std.testing.expectError(error.InputOutput, loader.awaitBatch(batch));
            // The reads stopped at the failure.
            try std.testing.expect(Reads.started.load(.acquire) < tensors);
        }
        try std.testing.expectEqual(@as(usize, 0), loader.scheduler.queue.len);
        try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
        try std.testing.expectEqual(loader.read_width, loader.read_permits.permits);
    }
}

test "a default-memory output a failed load left unfinished reports an error on every device" {
    const allocator = std.testing.allocator;
    // On the accelerator when it has two devices: each plugin's transfer
    // manager errors the buffers it leaves unfinished.
    var devices: testing.Devices = undefined;
    try devices.init(allocator, std.testing.io, 2);
    defer devices.deinit();
    const platform = devices.platform;
    // The read reaching the last request fails.
    const Reads = struct {
        var fail_from: u64 = std.math.maxInt(u64);

        fn readPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
            if (offset >= fail_from) return error.InputOutput;
            return std.testing.io.vtable.fileReadPositional(userdata, file, data, offset);
        }
    };
    var vtable = std.testing.io.vtable.*;
    vtable.fileReadPositional = Reads.readPositional;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off });
    defer loader.destroy();

    // A replicated tensor of three requests, a job each: no device's last
    // piece goes out.
    const request_size = loader.plan_config.request_size;
    const contents = try allocator.alloc(u8, 3 * request_size);
    defer allocator.free(contents);
    testing.pattern(contents, 1);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [1024]u8 = undefined;
    const path = try writeTestFile(tmp.dir, std.testing.io, "weights.bin", contents, &path_buffer);
    var source: safetensors.Tensor = .{ .file_uri = path, .name = "t", .shape = .init(.{contents.len}, .u8), .offset = 0 };
    var part: LoadPart = .whole(&source);
    var output: Buffer = .{ ._platform = platform, ._shape = source.shape, ._shards = .empty };
    defer output.deinit();
    const spec: LoadSpec = .{ .shape = source.shape, .output = &output, .parts = (&part)[0..1] };

    Reads.fail_from = 2 * request_size;
    defer Reads.fail_from = std.math.maxInt(u64);
    const batch = try loader.submit((&spec)[0..1], .none);
    try std.testing.expectError(error.InputOutput, loader.awaitBatch(batch));
    // The failing job created the buffers before its read. Each one is
    // ready, with an error, once the batch is retired: a success would
    // hand out a buffer missing its last request, a pending one would hang
    // its first user.
    const placed = platform.placedShape(source.shape);
    try std.testing.expectEqual(placed._sharding.mesh.?.devicesInCanonicalOrder().len, output._shards.len);
    for (output._shards.constSlice()) |shard| {
        const event = shard.readyEvent(platform.pjrt_api);
        defer event.deinit(platform.pjrt_api);
        try std.testing.expect(event.isReady(platform.pjrt_api));
        const err = event.getEventError(platform.pjrt_api) orelse return error.TestUnexpectedResult;
        err.deinit(platform.pjrt_api);
    }
}

test "a submit the loader fails between its open check and its first publish returns that failure" {
    const io = std.testing.io;
    const platform = Platform.init(std.testing.allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(std.testing.allocator, io);
    // Fails the loader in the submitting task's next allocation: after
    // `submit`'s open check and before its first publish, where a worker of
    // an older batch can fail it and the scheduler then answers `Canceled`.
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
    const path = try writeTestFile(tmp.dir, io, "weights.bin", &.{ 1, 2, 3, 4 }, &path_buffer);
    var source: safetensors.Tensor = .{ .file_uri = path, .name = "x", .shape = .init(.{4}, .u8), .offset = 0 };
    var output: Buffer = .{ ._platform = platform, ._shape = source.shape, ._shards = .empty };
    defer output.deinit();
    const loader = try Loader.create(failing.allocator(), io, platform, .{
        .read_parallelism = 2,
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
    const path = try writeTestFile(tmp.dir, io, "weights.bin", &contents, &path_buffer);
    const loader = try Loader.create(allocator, io, platform, .{
        .read_parallelism = 2,
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
        try testing.expectShards(allocator, io, &output, &contents, .default);
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
            .lattice = .{ .width = source.byteSize() },
        };
        const placed = platform.placedShape(source.shape).packedShape();
        destination.* = .{
            .shape = source.shape,
            .placed_shape = placed,
            .grid = .init(placed),
            .output = output,
            .parts = @as(*[1]Part, part),
            .unread_bytes = .init(source.byteSize()),
        };
    }
}

/// A plan of `job_count` unit jobs (`file_offset` = index), each with one
/// block and one event slot, and no transfers, in `arena`.
fn testPlan(arena: std.mem.Allocator, job_count: usize) !*Batch.Plan {
    const jobs = try arena.alloc(Batch.Plan.Job, job_count);
    const requests = try arena.alloc(ReadRequest, job_count);
    @memset(requests, ReadRequest.idle);
    const events = try arena.alloc(Dma.EventContext, job_count);
    for (jobs, 0..) |*job, index| job.* = .{
        .file_offset = index,
        .transfer_start = 0,
        .transfer_len = 0,
        .kind = .{ .dma_read = .{ .len = 1, .minimum_len = 1 } },
    };
    const plan = try arena.create(Batch.Plan);
    plan.* = .{
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
    try scheduler.publish(std.testing.io, batch, try testPlan(batch.arena.allocator(), job_count));
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

/// Writes `contents` to a new file `name` in `dir` and returns its real
/// path, in `path_buffer`.
fn writeTestFile(dir: std.Io.Dir, io: std.Io, name: []const u8, contents: []const u8, path_buffer: []u8) ![]const u8 {
    const file = try dir.createFile(io, name, .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, contents, 0);
    return path_buffer[0..try file.realPath(io, path_buffer)];
}

/// The jobs of the queued plans no worker claimed yet.
fn unclaimedJobs(scheduler: *const Scheduler) usize {
    var jobs: usize = 0;
    for (0..scheduler.queue.len) |index| {
        const plan = scheduler.queue.at(index);
        jobs += plan.jobs.len - plan.cursor;
    }
    return jobs;
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
    const path = try writeTestFile(tmp.dir, io, "drafter.bin", &contents, &path_buffer);
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    // A scheme whose files the plain io cannot open: asking would fail.
    try vfs.registerBackend("remote", .{ .io = io });
    const vfs_io = vfs.io();
    const reads: Planner.Reads = .{ .io = vfs_io, .vfs = &vfs, .policy = .on, .alignment = 4096 };
    try std.testing.expectEqual(@as(usize, 0), reads.boundOf("remote://bucket/model.safetensors"));
    try std.testing.expectEqual(@as(usize, 4096), reads.boundOf(path));

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
    var source: []const u8 = "";
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 8,
    }, .fixed(0), &source);

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
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const aligned_plan = try Planner.prepareTestPlan(arena.allocator(), &aligned_items, &.{ 0, 1, 2 }, .{
        .block_size = 4,
        .request_size = 8,
    }, 0);
    try std.testing.expectEqual(@as(usize, 3), aligned_plan.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_plan.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_plan.jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 8), aligned_plan.jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 5), aligned_plan.jobs[2].kind.dma_read.len);

    // Widened for direct I/O: the same three jobs (16 - 2 * 4 = 8 still cuts
    // at the tensor-safe boundaries 7 and 15), each read at 4-byte bounds,
    // its transfers relative to the widened start; the last read runs past
    // the tensor data, of which every byte is required.
    const widened_plan = try Planner.prepareTestPlan(arena.allocator(), &aligned_items, &.{ 0, 1, 2 }, .{
        .block_size = 4,
        .request_size = 16,
    }, 4);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs.len);
    try std.testing.expectEqual(@as(u64, 0), widened_plan.jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 7), widened_plan.jobs[0].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(u64, 4), widened_plan.jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 12), widened_plan.jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 11), widened_plan.jobs[1].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs[1].blockLen(4));
    try std.testing.expectEqual(@as(usize, 3), widened_plan.transfers[widened_plan.jobs[1].transfer_start].landing);
    try std.testing.expectEqual(@as(u64, 12), widened_plan.jobs[2].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].kind.dma_read.minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.transfers[widened_plan.jobs[2].transfer_start].landing);
    try std.testing.expectEqual(@as(u64, 20), widened_plan.source_bytes);
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
    var source: []const u8 = "";
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 4,
    }, .fixed(0), &source);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), unclaimedJobs(&scheduler));
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
    try std.testing.expectEqual(@as(usize, 3), unclaimedJobs(&scheduler));

    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == first);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expect(claim.plan == first.plans.items[0] and claim.index == 0);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == first);
    try std.testing.expectEqual(@as(u64, 1), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expectEqual(@as(usize, 1), unclaimedJobs(&scheduler));
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.batch == second);
    try std.testing.expectEqual(@as(u64, 0), claim.plan.jobs[claim.index].file_offset);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&scheduler));

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
    try std.testing.expectEqual(@as(usize, 2), unclaimedJobs(&scheduler));
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
    try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&scheduler));
    try std.testing.expect(!scheduler.waitForWork(io));
    const publish_result: anyerror!void = if (publishTestBatch(&scheduler, 1)) |_| {} else |err| err;
    try std.testing.expectError(error.Canceled, publish_result);
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

test "fifo scheduler wakes waiting workers on publish and releases them on failure" {
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
    scheduler.fail(io);
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
    try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&scheduler));
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
    try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&scheduler));
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
    try std.testing.expectEqual(@as(usize, 2), unclaimedJobs(&scheduler));

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
    try std.testing.expectEqual(@as(usize, 5), unclaimedJobs(&scheduler));
    try std.testing.expectEqual(@as(usize, 6), batch.remaining.load(.acquire));
    try std.testing.expect(scheduler.claim(io).?.plan.batch == batch);

    // The claim keeps its unit; the four unclaimed jobs of both plans are
    // retired; the sentinel is still held.
    scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    try std.testing.expectEqual(@as(usize, 2), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&scheduler));
    try std.testing.expect(scheduler.claim(io) == null);
    // The submission goes on: its next plan is refused, and the sentinel
    // drop leaves only the claim to complete it.
    try std.testing.expectError(error.Canceled, publishTestPlan(&scheduler, batch, 1));
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
            .read_permits = .{ .permits = 1 },
            .read_width = 1,
            .credits = try .init(allocator, blocks, 1),
            .pumps = &self.pumps,
            .dma_budget_bytes = 64,
            .submitted_bytes = undefined,
            .allocated_bytes = undefined,
            .created_at = undefined,
            .plan_config = .{ .block_size = 64, .request_size = 64 },
            .reads = .fixed(0),
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
        const slot = loader.credits.acquire(io);
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
    _ = loader.first_error.set(error.Unknown);

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

test "a worker waiting for a credit when the loader fails gets one back and stops" {
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;

    // A request in flight holds the only credit; a worker that saw the
    // second job waits on the credits.
    const batch = try publishTestBatch(&loader.scheduler, 2);
    const request = try fixture.claimRequest();
    try loader.worker_group.concurrent(io, Loader.workerMain, .{loader});
    while (loader.credits.condition.state.load(.monotonic).waiters == 0) io.sleep(.fromMilliseconds(1), .awake) catch {};

    // The failure retires the second job and closes nothing: the request
    // gives its credit back as it ends, the worker claims nothing with it,
    // and `destroy`'s shutdown stops it.
    loader.recordError(error.InputOutput, "injected source failure", .{});
    request.release();
    loader.stopWorkers();
    try std.testing.expect(batch.done.isSet());
    try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
    try std.testing.expectEqual(@as(usize, 0), batch.plans.items[0].requests[1].pending.load(.acquire));
    Dma.retireBatch(loader, batch);
    batch.destroy();
}

test "retired events are destroyed by the next pump or by a batch's retirement" {
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
    // the batch's retirement destroys it before the batch is freed, with
    // the retired contexts of a batch still in flight.
    var other: Dma.EventContext = .{ .request = undefined, .pjrt_event = null, .device_index = 0, .len = 64 };
    Dma.retireEvent(loader, &other);
    Dma.retireEvent(loader, &plan.events[0]);
    for (requests) |request| request.release();
    try std.testing.expect(batch.done.isSet());
    Dma.retireBatch(loader, batch);
    try std.testing.expect(device_pump.retired == null);
    try std.testing.expect(plan.events[0].next_retired == null);
    try std.testing.expect(other.next_retired == null);
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
                const slot = loader_.credits.acquire(io_);
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
        loader.scheduler.fail(io);
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
        try std.testing.expectEqual(@as(usize, 0), unclaimedJobs(&loader.scheduler));
    }
    // Every request released its credit before its batch completed; the
    // workers only hold one while a claim is in progress.
    loader.scheduler.fail(io);
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
    /// staged where it saves `staging_gain` submissions; the replay reads
    /// its blocks from the two files as its worker would (`jobBlocks`).
    fn run(platform: *const Platform, shape: Shape, block_size: usize, request_size: usize, memory: Memory, alignment: usize, staging_gain: usize, expect_cross_part_piece: bool) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
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
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var paths: [2][std.Io.Dir.max_path_bytes]u8 = undefined;
        var slots: [2]SourceSlot = undefined;
        var opened: usize = 0;
        defer for (slots[0..opened]) |*slot| slot.deinit(io);
        for (&slots, &paths, [_][]const u8{ "first", "second" }, [_][]const u8{ &first_bytes, &second_bytes }) |*slot, *path, name, bytes| {
            slot.* = .{ .uri = try writeTestFile(tmp.dir, io, name, bytes, path) };
            opened += 1;
        }
        var fixture: TestLoader = undefined;
        try fixture.init(1);
        defer fixture.deinit();
        const loader = &fixture.loader;
        loader.plan_config = .{ .block_size = block_size, .request_size = request_size };
        var outputs: [2]Buffer = undefined;
        var destinations: [2]Destination = undefined;
        var parts = [_]Part{
            .{ .source = &sources[0], .source_slot = &slots[0], .destination = &destinations[0], .lattice = .{ .dst_offset = 0, .width = 48 } },
            .{ .source = &sources[1], .source_slot = &slots[0], .destination = &destinations[0], .lattice = .{ .dst_offset = 48, .width = 32 } },
            .{ .source = &sources[2], .source_slot = &slots[1], .destination = &destinations[0], .lattice = .{ .dst_offset = 160, .width = 64 } },
            .{ .source = &sources[3], .source_slot = &slots[1], .destination = &destinations[1], .lattice = .{ .width = 64 } },
        };
        const plain_shape = shape.setDim(0, 4);
        destinations = .{
            .{ .shape = shape, .placed_shape = platform.placedShape(shape).packedShape(), .grid = .init(platform.placedShape(shape).packedShape()), .memory = memory, .output = &outputs[0], .parts = parts[0..3], .unread_bytes = .init(144) },
            .{ .shape = plain_shape, .placed_shape = platform.placedShape(plain_shape).packedShape(), .grid = .init(platform.placedShape(plain_shape).packedShape()), .memory = memory, .output = &outputs[1], .parts = parts[3..4], .unread_bytes = .init(64) },
        };
        const host = destinations[0].memory.isHost();

        const order = try Planner.sortedPartOrder(allocator, &parts);
        defer allocator.free(order);
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        var plans: [2]*Batch.Plan = undefined;
        var plan_count: usize = 0;
        var start: usize = 0;
        while (start < order.len) : (plan_count += 1) {
            const end = Planner.fileGroupEnd(&parts, order, start);
            plans[plan_count] = try Planner.prepareTestPlan(arena.allocator(), &parts, order[start..end], .{
                .block_size = block_size,
                .request_size = request_size,
                .minimum_staging_gain = staging_gain,
            }, alignment);
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
                    try std.testing.expect(kind == .dma_read or kind == .dma_land or kind == .dma_fill);
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
        for (plans[0..plan_count]) |plan| {
            const file_bytes: []const u8 = if (plan.source_slot == &slots[0]) &first_bytes else &second_bytes;
            for (plan.jobs) |job| {
                const fill = job.kind == .dma_fill or job.kind == .host_fill;
                const blocks = if (host or fill) &.{} else try jobBlocks(allocator, loader, plan, job);
                defer allocator.free(blocks);
                for (plan.transfers[job.transfer_start..][0..job.transfer_len]) |transfer| {
                    if (host) {
                        // Host transfers address the read, not its blocks.
                        switch (job.kind) {
                            .host_direct, .host_bounce => |read| try std.testing.expect(transfer.landing + transfer.len <= read.len),
                            else => {},
                        }
                    } else {
                        // A fill job's transfers read the zero block.
                        try std.testing.expect(transfer.landing < (if (fill) 1 else job.blockLen(block_size)) * block_size);
                        try std.testing.expect(transfer.landing % block_size + transfer.len <= block_size);
                    }
                    const bytes = if (fill) fill: {
                        try std.testing.expectEqual(@as(usize, 0), transfer.landing);
                        break :fill zeros[0..transfer.len];
                    } else if (host) read: {
                        const file_offset = job.file_offset + transfer.landing;
                        // a's last byte and b's first one in the same piece.
                        if (plan.source_slot == &slots[0] and file_offset <= 51 and file_offset + transfer.len > 52) cross_part_piece = true;
                        break :read file_bytes[file_offset..][0..transfer.len];
                    } else read: {
                        const file_offset = job.file_offset + transfer.landing;
                        if (job.kind == .dma_read and plan.source_slot == &slots[0] and file_offset <= 51 and file_offset + transfer.len > 52) cross_part_piece = true;
                        break :read blocks[transfer.landing..][0..transfer.len];
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
    /// the read in file order, or as its worker lands a landing job
    /// (`landJob`).
    fn jobBlocks(allocator: std.mem.Allocator, loader: *Loader, plan: *const Batch.Plan, job: Batch.Plan.Job) ![]u8 {
        const block_size = loader.plan_config.block_size;
        const bytes = try allocator.alloc(u8, job.blockLen(block_size) * block_size);
        errdefer allocator.free(bytes);
        @memset(bytes, 0xaa);
        const land = switch (job.kind) {
            .dma_read => |read| {
                // The end of the file cuts a widened read's padding.
                try readSlot(loader, plan.source_slot, &.{bytes[0..read.len]}, job.file_offset, read.minimum_len);
                return bytes;
            },
            .dma_land => |land| land,
            .dma_fill, .host_direct, .host_bounce, .host_fill => unreachable,
        };
        // A bounce buffer as large as any of the job's reads.
        var longest: usize = land.len;
        for (plan.extents[land.extent_start..][0..land.extent_len]) |extent| longest = @max(longest, extent.len);
        const blocks = try allocator.alloc(host_memory.Block, job.blockLen(block_size));
        defer allocator.free(blocks);
        for (blocks, 0..) |*block, index| block.* = bytes[index * block_size ..][0..block_size];
        var scratch: ReadRequest.Scratch = try .init(allocator, Planner.blockCount(longest, block_size), block_size);
        defer scratch.deinit();
        try landJob(loader, plan, land, blocks, &scratch);
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
        fn run(loader_: *Loader, err: anyerror) void {
            loader_.recordError(err, "concurrent source failure", .{});
        }
    };
    for (0..8) |i| try group.concurrent(io, Worker.run, .{ loader, @as(anyerror, if (i % 2 == 0) error.FileNotFound else error.UnexpectedEndOfFile) });
    try group.await(io);
    const first = loader.first_error.get().?;
    loader.recordError(error.Internal, "later failure", .{});
    try std.testing.expectEqual(first, loader.first_error.get().?);
}

test "lazy initialization preserves its finite error across repeated callers" {
    const Context = struct {
        fn init(calls: *usize) error{FileNotFound}!u32 {
            calls.* += 1;
            return error.FileNotFound;
        }
    };
    var once: LazyOnce(u32) = .{};
    var calls: usize = 0;
    try std.testing.expectError(error.FileNotFound, once.ensure(std.testing.io, Context.init, .{&calls}));
    try std.testing.expectError(error.FileNotFound, once.ensure(std.testing.io, Context.init, .{&calls}));
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
        .{ .source = &sources[0], .source_slot = &slot, .destination = &destinations[0], .lattice = .{ .width = 64 } },
        .{ .source = &sources[1], .source_slot = &slot, .destination = &destinations[1], .lattice = .{ .width = 64 } },
        .{ .source = &sources[0], .source_slot = &slot, .destination = &destinations[2], .lattice = .{ .width = 64 } },
        .{ .source = &sources[2], .source_slot = &slot, .destination = &destinations[3], .lattice = .{ .width = 64 } },
    };
    const memories = [_]Memory{ .host_pinned, .default, .host_pinned, .host_pinned };
    for (&destinations, &outputs, &parts, memories) |*destination, *output, *part, memory| {
        destination.* = .{ .shape = shape, .placed_shape = platform.placedShape(shape).packedShape(), .grid = .init(platform.placedShape(shape).packedShape()), .memory = memory, .output = output, .parts = @as(*[1]Part, part), .unread_bytes = .init(64) };
    }

    const order = try Planner.sortedPartOrder(allocator, &parts);
    defer allocator.free(order);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const plan = try Planner.prepareTestPlan(arena.allocator(), &parts, order, .{
        .block_size = 64,
        .request_size = 1024,
    }, 0);
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
            try std.testing.expectEqual(job.kind != .dma_read, transfer.destination.memory.isHost());
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
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 4);
    defer devices.deinit();
    const mesh = try devices.mesh("tp");

    // Rows of 32 bytes, 8 per device; then 12,000 one-byte pieces, more
    // than one vectored read takes.
    var rows: [64 * 32]u8 = undefined;
    var cols: [1024 * 32]u8 = undefined;
    var narrow: [3000 * 4]u8 = undefined;
    testing.pattern(&rows, 1);
    testing.pattern(&cols, 2);
    testing.pattern(&narrow, 3);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try testing.writeSafetensors(allocator, io, tmp.dir, "staged.safetensors", &registry, &.{
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
        loader.reads.alignment = alignment;
        const batch = try loader.submit(&specs, .none);
        // Every plan is published once `submit` returns.
        var staged_jobs: usize = 0;
        var submissions: usize = 0;
        for (batch.plans.items) |plan| {
            for (plan.jobs) |job| staged_jobs += @intFromBool(job.kind == .dma_land);
            submissions += plan.events.len;
        }
        try loader.awaitBatch(batch);
        defer for (&outputs) |*output| output.deinit();
        for (&outputs, globals) |*output, global| try testing.expectShards(allocator, io, output, global, .default);
        try std.testing.expect(staged_jobs >= 1);
        try std.testing.expect(submissions < row_pieces / 100);
        try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
    }
}

test "a strided part whose source does not fit one read is refused before anything is published" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 1);
    defer devices.deinit();
    var bytes: [32 * 16]u8 = undefined;
    testing.pattern(&bytes, 1);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try testing.writeSafetensors(allocator, io, tmp.dir, "wide.safetensors", &registry, &.{
        .{ .name = "wide", .dtype = "U8", .dims = &.{ 32, 16 }, .bytes = &bytes },
    });
    // [32, 16] transposed: 512 runs of one byte, its source read whole by
    // every job.
    const part: LoadPart = .{ .source = registry.tensors.getPtr("wide").?, .lattice = .{ .width = 1, .dims = &.{
        .{ .count = 16, .src_stride = 1, .dst_stride = 32 },
        .{ .count = 32, .src_stride = 16, .dst_stride = 1 },
    } } };
    var output: Buffer = undefined;
    const spec: LoadSpec = .{ .shape = .init(.{ 16, 32 }, .u8), .output = &output, .parts = (&part)[0..1] };
    const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 2, .direct_io = .off });
    defer loader.destroy();
    loader.plan_config.request_size = 256;
    if (loader.submit(&.{spec}, .none)) |batch| {
        try loader.awaitBatch(batch);
        output.deinit();
        return error.TestUnexpectedResult;
    } else |err| try std.testing.expectEqual(error.InvalidBinding, err);
    // Nothing went out, so the loader takes the next submission.
    loader.plan_config.request_size = 1024;
    try loader.awaitBatch(try loader.submit(&.{spec}, .none));
    defer output.deinit();
    var transposed: [16 * 32]u8 = undefined;
    for (0..16) |column| {
        for (0..32) |row| transposed[column * 32 + row] = bytes[row * 16 + column];
    }
    try testing.expectShards(allocator, io, &output, &transposed, .default);
}

test "a strided part of a remote file is checked against its exact source range" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 1);
    defer devices.deinit();
    var bytes: [128 * 128]u8 = undefined;
    testing.pattern(&bytes, 1);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try testing.writeSafetensors(allocator, io, tmp.dir, "square.safetensors", &registry, &.{
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
    const dims = [_]backend.Lattice.Dim{
        .{ .count = 128, .src_stride = 1, .dst_stride = 128 },
        .{ .count = 128, .src_stride = 128, .dst_stride = 1 },
    };
    const loader = try Loader.create(allocator, vfs.io(), devices.platform, .{ .read_parallelism = 2 });
    defer loader.destroy();
    try std.testing.expectEqual(VFS.direct_io_alignment, loader.reads.alignment);
    loader.plan_config.request_size = bytes.len;

    var output: Buffer = undefined;
    const part: LoadPart = .{ .source = &remote, .lattice = .{ .width = 1, .dims = &dims } };
    const spec: LoadSpec = .{ .shape = .init(.{ 128, 128 }, .u8), .output = &output, .parts = (&part)[0..1] };
    try loader.awaitBatch(try loader.submit(&.{spec}, .none));
    defer output.deinit();
    try testing.expectShards(allocator, io, &output, &transposed, .default);

    // A local file may be read directly: its range widened to the alignment
    // no longer fits.
    if (VFS.direct_io_alignment != 0) {
        var refused: Buffer = undefined;
        const local_part: LoadPart = .{ .source = &local, .lattice = .{ .width = 1, .dims = &dims } };
        const local_spec: LoadSpec = .{ .shape = .init(.{ 128, 128 }, .u8), .output = &refused, .parts = (&local_part)[0..1] };
        if (loader.submit(&.{local_spec}, .none)) |batch| {
            try loader.awaitBatch(batch);
            refused.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidBinding, err);
    }
}

/// A host-resident load on four devices: rows sharded; columns sharded, two
/// bytes per device per row, so one job has 8,192 transfers and scatters in
/// several reads; replicated; one source twice, replicated and row-sharded,
/// which overlaps; and a device-memory tensor in the same file.
const HostResidentTest = struct {
    /// Each run writes its own file from `seed`: a buffer that lands on
    /// memory an earlier run freed must not find the expected bytes there.
    fn run(devices: *const testing.Devices, seed: u8, alignment: usize) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        const platform = devices.platform;
        const mesh = try devices.mesh("tp");

        var rows: [64 * 16]u8 = undefined;
        var dev: [16 * 16]u8 = undefined;
        var cols: [2048 * 8]u8 = undefined;
        var rep: [96 * 4]u8 = undefined;
        var twice: [32 * 4]u8 = undefined;
        testing.pattern(&rows, seed);
        testing.pattern(&dev, seed +% 1);
        testing.pattern(&cols, seed +% 2);
        testing.pattern(&rep, seed +% 3);
        testing.pattern(&twice, seed +% 4);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var registry: safetensors.TensorRegistry = .init(allocator);
        defer registry.deinit();
        // The device tensor separates the host runs: an overlap sends its
        // whole job through blocks, so the twice-bound source must not share
        // a job with the others.
        try testing.writeSafetensors(allocator, io, tmp.dir, "host.safetensors", &registry, &.{
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
        loader.reads.alignment = alignment;
        const live_before = SharedHostAllocation.live.load(.acquire);
        {
            const batch = try loader.submit(&specs, .none);
            try loader.awaitBatch(batch);
            defer for (&outputs) |*output| output.deinit();
            // Host shards in each device's host memory, or its default
            // memory for views.
            for (&outputs, globals, memories) |*output, global, memory| try testing.expectShards(allocator, io, output, global, memory);
            // Every credit, with its blocks, came back.
            try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));

            // The replicated host ones are one allocation.
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
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 4);
    defer devices.deinit();
    if (!backend.supportsMemory(devices.platform, .host_pinned)) return error.SkipZigTest;
    // In place, then through blocks for every host job.
    for ([_]usize{ 0, 64 }, [_]u8{ 1, 5 }) |alignment, seed| {
        try HostResidentTest.run(&devices, seed, alignment);
    }
}

const CountingPositionalIo = struct {
    calls: usize = 0,
    max_result: ?usize = null,

    const vtable: std.Io.VTable = blk: {
        var result = std.testing.io.vtable.*;
        result.fileReadPositional = fileReadPositional;
        break :blk result;
    };

    fn io(self: *CountingPositionalIo) std.Io {
        return .{
            .userdata = self,
            .vtable = &vtable,
        };
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        _: std.Io.File,
        data: []const []u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *CountingPositionalIo = @ptrCast(@alignCast(userdata.?));
        self.calls += 1;

        var requested: usize = 0;
        for (data) |buffer| requested +|= buffer.len;
        var remaining = @min(requested, self.max_result orelse requested);
        var position: usize = 0;
        for (data) |buffer| {
            const take = @min(remaining, buffer.len);
            for (buffer[0..take]) |*byte| {
                byte.* = @truncate(offset + position);
                position += 1;
            }
            remaining -= take;
            if (remaining == 0) break;
        }
        return position;
    }
};

test "readFilePositionalAllV batches more than IOV_MAX buffers" {
    const io = std.testing.io;
    const len = max_positional_iovecs + 1;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    defer file.close(io);
    var contents: [len]u8 = undefined;
    for (&contents, 0..) |*byte, i| byte.* = @truncate(i);
    try file.writePositionalAll(io, &contents, 0);

    var output: [len]u8 = undefined;
    var buffers: [len][]u8 = undefined;
    for (&buffers, 0..) |*buffer, i| buffer.* = output[i..][0..1];
    try std.testing.expectEqual(@as(u64, len), try readFilePositionalAllV(io, file, &buffers, 0, len));
    try std.testing.expectEqualSlices(u8, &contents, &output);
}

test "readFilePositionalAllV resumes short reads" {
    var counting: CountingPositionalIo = .{ .max_result = 1 };
    const io = counting.io();
    const file: std.Io.File = .{ .handle = 0, .flags = .{ .nonblocking = false } };

    var output: [4]u8 = undefined;
    _ = try readFilePositionalAllV(io, file, &.{ output[0..1], output[1..3], output[3..4] }, 7, output.len);
    try std.testing.expectEqual(@as(usize, 4), counting.calls);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 10 }, &output);
}

test "advanceScatter resumes partial reads across boundaries" {
    var storage: [6]u8 = undefined;
    const buffers = [_][]u8{ storage[0..2], storage[2..2], storage[2..5], storage[5..6] };
    var index: usize = 0;
    var offset: usize = 0;
    try advanceScatter(&buffers, &index, &offset, 1);
    try std.testing.expectEqual(@as(usize, 0), index);
    try std.testing.expectEqual(@as(usize, 1), offset);
    try advanceScatter(&buffers, &index, &offset, 2);
    try std.testing.expectEqual(@as(usize, 2), index);
    try std.testing.expectEqual(@as(usize, 1), offset);
    try advanceScatter(&buffers, &index, &offset, 3);
    try std.testing.expectEqual(@as(usize, 4), index);
    try std.testing.expectEqual(@as(usize, 0), offset);
    try std.testing.expectError(error.UnexpectedEndOfFile, advanceScatter(&buffers, &index, &offset, 1));
}
