//! Direct loading follows a batch from per-file planning through FIFO claims,
//! source reads and DMA completion. Runtime ownership and synchronization
//! live here. CUDA, ROCm, oneAPI, Furiosa and CPU load this way
//! (`backend.loadsDirect`).
//!
//! Sizing, fixed by `Loader.create`. Calibration picks the DMA block
//! (dma_calibration.zig) and maps the ring it measures in; each device's
//! DMA budget is `dma_calibration.depth` blocks. A source request is the
//! checkpoint's minimum read (`VFS.readHints`) rounded up to whole blocks,
//! one block at least (`backend.requestSize`). The source width is
//! `read_parallelism`, clipped to what the mapped ceiling fits. The
//! workspace then maps, once and before the first read, what the ring
//! leaves of `width + 1` requests, the DMA reserve and the zero block, and
//! cuts them into blocks (`Workspace.carve`): nothing maps during a load.
//! `width + 1` workers share `width` read permits, so a full width of reads
//! is in flight while one worker queues what it read. The lifecycle credits
//! are the whole pinned set in requests, so transfers that still hold
//! blocks keep the DMA stage fed and never want a block the reads did not
//! leave free.
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
//! Direct I/O. A local file the VFS reads past the page cache
//! (`VFS.useDirectIo`, asked once per file: `Planner.Reads`) is read in
//! whole alignment units: each job's read widens to aligned bounds around
//! its range, into blocks that are whole units, and its transfers address
//! the widened read. A job's range is at most a request less two units, so
//! its widened read still fits its credit's slot. Every other file is read
//! exact.
//!
//! Destinations. A part's source bytes map to its destination's logical
//! offsets, which the destination's `ShardGrid` places on the devices by
//! arithmetic. Pieces of one destination merge across parts when block,
//! writers and both offsets are contiguous, and its parts may come from
//! different files and plans.
//!
//! Locks. Where several pumps are held, they are taken in device order. A
//! pump's mutex may be held while a released request gives its credit back
//! (`Credits`), never the reverse, and no other lock is taken under the
//! scheduler's mutex. A DMA callback, on a plugin thread, holds one pump at
//! a time.

const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const pjrtx = @import("../pjrtx.zig");
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const backend = @import("backend.zig");
const BackendOptions = backend.Options;
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const ShardGrid = @import("ShardGrid.zig");
const dma_calibration = @import("dma_calibration.zig");
const Memory = @import("TensorStore.zig").Memory;
const host_memory = @import("host_memory.zig");
const load_log = @import("log.zig").load;
const testing = @import("../testing.zig");

/// Creating a tensor's device state: host memory or PJRT, since the front
/// end checked its placement. `LazyOnce` replays it to every later reader
/// of the tensor.
const StateInitError = std.mem.Allocator.Error || pjrt.ApiError;

/// Bounds per-device event overhead for tiny tensors. This is not a measured
/// optimum: 64 pieces smaller than block_size / 8 cannot fill an eight-block
/// byte budget.
const max_dma_pieces_per_device: usize = 64;

/// The direct DMA backend. Submissions and awaits come from one task at a
/// time; the workers and the pumps run concurrently with them.
pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// The pinned arenas, grown once at creation and cut into `blocks`.
    workspace: host_memory.Workspace,
    /// The workspace's blocks (`carve`): the zero block first, then the
    /// credits' slots (`Credits`). The zero block is in no credit's slot
    /// and is never written after `create`.
    blocks: []const host_memory.Block,
    scheduler: Scheduler,
    metrics: Metrics = .{},
    /// One permit per source read in flight (`readSlot`), `read_width` of
    /// them. The width never changes: a source that answers 429 (or 503 on
    /// the object stores) is held one layer down, where the VFS holds every
    /// request of that backend (`vfs/request.zig`), so the traffic falls
    /// while the memory bounds stay.
    read_permits: std.Io.Semaphore,
    read_width: usize,
    /// Lifecycle credits (`Credits`), beyond the workers' through the DMA
    /// reserve. One credit beyond the width fed the DMA engines one request
    /// at a time: on a GB300 loading DeepSeek-V4 (pieces of 4 MiB and 256
    /// KiB) that was 24.3 GiB/s at width 16 against 43.8 GiB/s with eight
    /// requests queued, and the workers spent 9 ms per request waiting for a
    /// credit against 3 ms reading.
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
    /// What the debug lines time from (`logBatch`, `logSummary`).
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

        var workspace: host_memory.Workspace = try .init(allocator, io, platform);
        errdefer workspace.deinit();
        // At debug when canceled (`zml.io.next.Loader.deinit` cancels a
        // set-up no `start` waited for) or out of memory (the workspace logs
        // what it fails to map or pin).
        const block_size = dma_calibration.calibrate(&workspace, platform, .{ .block_sizes = opts.dmaBlockSizes() }) catch |err| {
            load_log.failure(err, "calibrate DMA: target={t}", .{platform.target});
            return err;
        };
        const reads: Planner.Reads = .init(io, opts.direct_io);
        const request_size = backend.requestSize(opts.read_hints.read_chunk_size, block_size, reads.alignment).?;
        const blocks_per_request = request_size / block_size;
        // The DMA stage of every device, kept mapped beside the source
        // working set; at least one maximal source request, so a single
        // request always fits beside it.
        const dma_reserve = @max(
            dma_calibration.depth * platform.devices.len,
            Planner.blockCount(backend.max_read_request_size, block_size),
        );
        // One more block for the zeros, which no credit owns.
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
    /// The specs keep the `LoadSpec` contract, which the front end checked,
    /// and this backend's: device memory, and contiguous parts covering
    /// every byte of their destination. A planning failure fails the
    /// loader (a partial submission can never complete): the batch is
    /// awaited here and the caller sees only the first error. A failed
    /// loader refuses every submit with its first error.
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: std.Progress.Node) !*Batch {
        try self.checkOpen();
        const batch = batch: {
            const batch = try Batch.create(self.allocator, self.io, self.batch_count);
            errdefer batch.destroy();
            try self.createDestinations(batch, specs, progress);
            break :batch batch;
        };
        Planner.publishFiles(&self.scheduler, self.io, batch, batch.parts, self.plan_config, self.reads) catch |err| {
            // Fail the loader, seal and await what was published of this
            // batch, and return the loader's first error.
            self.recordError(err, "plan batch {d}, published_plans={d}", .{ batch.sequence, batch.plans.items.len });
            batch.sealed_at = .now(self.io, .awake);
            self.batch_count += 1;
            batch.releaseUnits(1);
            try self.awaitBatch(batch);
            unreachable;
        };
        // Stamped before the sentinel goes, so before the batch can complete.
        batch.sealed_at = .now(self.io, .awake);
        self.batch_count += 1;
        // Every plan is visible: drop the publish sentinel. A batch without
        // jobs completes right here.
        batch.releaseUnits(1);
        return batch;
    }

    /// Waits for the batch's last completion unit, retires it and returns
    /// the loader's sticky error if the loader failed. Destroying the batch
    /// destroys its transfer managers, and PJRT's manager destructor errors
    /// every buffer whose last transfer never went out, so a buffer the
    /// failure left open reports that error, never success. A transfer
    /// that fails asynchronously errors its buffer's definition event too
    /// (the done event carries no error). A program reading such a buffer
    /// fails its execution; a bulk buffer shows the error at its first use.
    /// Either way an output of a failed submission holds buffers whose
    /// contents are unspecified, or was never written. A batch that
    /// completed without an error submitted every byte of every target (the
    /// pump flagged the submission that completed its bytes); one that did
    /// not would leave a buffer PJRT then errors, a bookkeeping bug this
    /// panics on.
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
            if (!submitted) std.debug.panic("loader: batch {d} completed without submitting every byte of tensor {s} {f}", .{ batch.sequence, destination.name(), destination.shape });
        };
        // Each destination completed its progress item.
        if (load_error == null) for (batch.destinations) |*destination| std.debug.assert(destination.unread_bytes.load(.acquire) == 0);
        Dma.retireBatch(self, batch);
        self.logBatch(batch, done_at, load_error == null);
        self.destroyBatch(batch);
        if (load_error) |err| return err;
    }

    /// Fails the loader unless it already failed (`recordError`): unclaimed
    /// jobs retire, queued pieces drain, and requests in flight finish
    /// their read.
    pub fn abort(self: *Loader) void {
        self.recordError(error.Canceled, "loader aborted", .{});
    }

    /// The caller awaited every batch before this; nothing is queued or in
    /// flight, so stopping the workers is a plain shutdown.
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
        self.workspace.deinit();
        self.allocator.destroy(self);
    }

    fn checkOpen(self: *Loader) !void {
        if (self.first_error.get()) |err| return err;
    }

    /// Creates the batch's destinations, one per spec, and their parts in
    /// its arena, their sources and dimensions staying the specs' until the
    /// batch is awaited (`backend.LoadSpec`). A destination's device state
    /// is created lazily by the first worker that reads for it.
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
        for (specs, destinations) |spec, *destination| {
            std.debug.assert(spec.memory == .default); // submit
            const grid: ShardGrid = .init(self.platform, spec.shape);
            const destination_parts = parts[next_part..][0..spec.parts.len];
            next_part += spec.parts.len;
            var covered_bytes: u64 = 0;
            for (spec.parts, destination_parts) |part, *slot| {
                std.debug.assert(part.lattice.dims.len == 0); // submit
                slot.* = .{
                    .source = part.source,
                    .source_slot = try self.sourceSlot(part.source.file_uri),
                    .destination = destination,
                    .lattice = part.lattice,
                };
                covered_bytes += part.lattice.width;
            }
            // The parts are disjoint (`backend.LoadSpec`): they cover the
            // destination when their widths add up to it.
            std.debug.assert(covered_bytes == grid.total); // submit
            destination.* = .{
                .shape = spec.shape,
                .grid = grid,
                .output = spec.output,
                .parts = destination_parts,
                .unread_bytes = .init(covered_bytes),
                .progress = progress,
            };
        }
        batch.destinations = destinations;
        batch.parts = parts;
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
        var scratch = ReadRequest.Scratch.init(self.allocator, self.credits.blocks_per_slot) catch |err| {
            self.recordError(err, "allocate source worker scratch: blocks={d}", .{self.credits.blocks_per_slot});
            return;
        };
        defer scratch.deinit();
        while (self.scheduler.waitForWork(self.io)) {
            const request = self.claimRequest() orelse continue;
            // The scheduling sentinel keeps the batch, and with it the
            // request's plan, alive through `run` and error reporting.
            defer request.release();
            request.run(&scratch) catch |err| {
                self.recordError(err, "source request: file={s}, job={t}", .{ request.plan.source_slot.uri, request.plan.jobs[request.index].kind });
            };
        }
    }

    /// Takes a credit, then the next job (`Credits`): a claimed job never
    /// waits for its slot. Null, with the credit given back, when nothing
    /// was left to claim.
    fn claimRequest(self: *Loader) ?*ReadRequest {
        const credit_wait_started = awakeNs(self.io);
        const slot = self.credits.acquire(self.io);
        const credit_wait_ns = awakeNs(self.io) -| credit_wait_started;
        const claim = self.scheduler.claim(self.io) orelse {
            self.credits.release(self.io, slot);
            return null;
        };
        _ = self.metrics.credit_wait_ns.fetchAdd(credit_wait_ns, .monotonic);
        return ReadRequest.init(self, claim, slot);
    }

    /// Stops the workers: every batch was awaited, so nothing is queued and
    /// no request holds a credit or permit, and `Scheduler.fail` only wakes
    /// the workers waiting for work. A worker may still hold the credit it
    /// took for a claim that finds nothing (`claimRequest`); it gives it
    /// back before it sees `stopping`, so every credit is free once this
    /// returns. Awaiting a group that never spawned returns at once, so
    /// this also cleans up a failed `create`.
    fn stopWorkers(self: *Loader) void {
        std.debug.assert(self.scheduler.queue.len == 0);
        self.scheduler.fail(self.io);
        self.worker_group.await(self.io) catch {};
    }

    /// Keeps `err` as the sticky error unless one is kept already and logs
    /// it, then fails the load without waiting: unclaimed jobs retire
    /// (`Scheduler.fail`) and queued pieces are released
    /// (`Dma.abortReady`). Any thread may call it, a DMA callback included;
    /// requests in flight stop at their next read (`readSlot`) or enqueue
    /// and release their units.
    /// Failing other batches than the caller's is safe from any thread:
    /// `Scheduler.fail` and `Dma.abortReady` reach only live batches (their
    /// docs), and `destroy` waits for the worker group before it frees what
    /// the workers use.
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
        const sealed_at = batch.sealed_at orelse self.created_at;
        const published_at = batch.published_at orelse sealed_at;
        var logical_bytes: usize = 0;
        for (batch.destinations) |*destination| logical_bytes += destination.shape.byteSize();
        load_log.debug("batch completed: batch={d}, successful={}, logical_bytes={Bi:.2}, published=+{d:.3}s, sealed=+{d:.3}s, done=+{d:.3}s, elapsed={d:.3}s, source_width={d}, request_size={Bi:.2}", .{
            batch.sequence,
            successful,
            logical_bytes,
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
        for (batch.plans.items) |plan| {
            planning_ns += plan.planning_ns;
            source_bytes += plan.source_bytes;
            jobs += plan.jobs.len;
            transfers += plan.transfers.len;
            dma_submissions += plan.events.len;
        }
        const average_read_size = if (jobs == 0) 0 else source_bytes / jobs;
        const coalescing_ratio = if (jobs == 0)
            0
        else
            @as(f64, @floatFromInt(batch.parts.len)) /
                @as(f64, @floatFromInt(jobs));
        load_log.debug("batch planning: batch={d}, plans={d}, planning_elapsed={d:.3}s, planned_source_bytes={Bi:.2}, planned_source_jobs={d}, source_items={d}, planned_transfers={d}, planned_dma_submissions={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}", .{
            batch.sequence,
            batch.plans.items.len,
            @as(f64, @floatFromInt(planning_ns)) / std.time.ns_per_s,
            source_bytes,
            jobs,
            batch.parts.len,
            transfers,
            dma_submissions,
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
        load_log.debug("loader waits: credit_wait_ms_per_read={d:.3}, read_ms_per_read={d:.3}, dma_stage_ms_per_read={d:.3}, tensor_init_ms_per_read={d:.3}", .{
            millisecondsPer(self.metrics.credit_wait_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.read_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.dma_stage_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.tensor_init_ns.load(.acquire), reads),
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
/// arena until it retires. `remaining` counts completion units: one
/// per published job plus a publish sentinel held until the submission is
/// sealed. A job's unit is released exactly once, by whichever of these
/// happens: its request's last reference drops (final DMA callback or
/// abandonment), or `Scheduler.fail` retires it unclaimed. The batch is
/// done when `remaining` reaches zero; the awaiting task then retires it,
/// so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
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
    /// Submission number within the loader, for log correlation.
    sequence: usize,
    /// The first publish; null for a submission without plans.
    published_at: ?std.Io.Timestamp = null,
    /// Once every plan is published, before the publish sentinel goes.
    sealed_at: ?std.Io.Timestamp = null,

    /// An open batch holding its publish sentinel and nothing else. The
    /// sentinel keeps the batch from completing before its last plan is
    /// visible.
    fn create(allocator: std.mem.Allocator, io: std.Io, sequence: usize) !*Batch {
        const self = try allocator.create(Batch);
        self.* = .{
            .arena = .init(allocator),
            .io = io,
            .remaining = .init(1),
            .sequence = sequence,
        };
        return self;
    }

    /// Releases `count` completion units. MEMORY-ORDER RULE: this must be the
    /// caller's final access to the batch and to anything it owns (requests,
    /// events, destinations, parts, plans; a request's slot of blocks is the
    /// loader's and went back with its credit just before). The last unit
    /// sets `done`, and the awaiting task frees all of it as soon as it
    /// observes the event.
    fn releaseUnits(self: *Batch, count: usize) void {
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
/// come from several files, so several plans can write it: their pieces all
/// count toward the same targets (`Target.total` is the placement's bytes,
/// whichever plan submits them), and its device state is created once, by
/// the first worker that reads for it.
const Destination = struct {
    /// The caller's shape, which the output buffer keeps.
    shape: Shape,
    /// Where `shape`, packed and placed (`Platform.placedShape`), lands on
    /// its devices: what PJRT allocates and the planner cuts.
    grid: ShardGrid,
    output: *Buffer,
    /// What fills it, in ascending, disjoint destination order
    /// (`backend.LoadSpec`).
    parts: []Part,
    /// Bytes not read yet, the parts' widths at first. The read that
    /// brings it to zero completes the destination's item on `progress`.
    unread_bytes: std.atomic.Value(u64),
    progress: std.Progress.Node = .none,
    state: LazyOnce(DestinationState) = .{},

    fn ensureState(self: *Destination, allocator: std.mem.Allocator, io: std.Io, platform: *const Platform) StateInitError!*DestinationState {
        return self.state.ensure(io, DestinationState.init, .{ allocator, platform, self });
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
/// and cuts into jobs. A plain tensor is one part covering its whole
/// destination.
const Part = struct {
    source: *const safetensors.Tensor,
    source_slot: *SourceSlot,
    destination: *Destination,
    /// The spec part's; its dimensions are borrowed until the await
    /// (`backend.LoadSpec`).
    lattice: backend.Lattice,

    /// The part's bytes in its file, which `backend.checkParts` bounds.
    fn fileRange(self: *const Part) struct { start: u64, end: u64 } {
        const start = self.source.offset + self.lattice.src_offset;
        return .{ .start = start, .end = start + self.lattice.width };
    }
};

/// One source file: every job of a plan reads the file of its slot, which
/// the first read opens, or the planner when it asks the VFS whether the
/// file is read directly (`Planner.Reads`).
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

/// One file's jobs in claim order with their transfer records and every
/// context the jobs can need, allocated by the planner: one request per job
/// and one event per planned DMA submission, handed out in submission
/// order. Contexts hold the plan's address, which the batch's arena keeps
/// until it retires.
const Plan = struct {
    /// A claimable job: its slice of the plan's transfers and what its
    /// worker does for them. Its request slot is its index.
    const Job = struct {
        /// Where the job's read starts.
        file_offset: u64,
        transfer_start: usize,
        transfer_len: usize,
        kind: Kind,

        /// How the job's bytes reach their destinations.
        const Kind = union(enum) {
            /// Read into pinned blocks in file order, where its transfers
            /// address it, then copied to the devices by DMA.
            dma_read: Read,
        };

        /// `len` bytes from the job's `file_offset`, of which the first
        /// `minimum_len` hold tensor data and must exist. The rest is
        /// alignment padding the end of the file may cut.
        const Read = struct {
            len: usize,
            minimum_len: usize,
        };

        /// The blocks the job reads into.
        fn blockLen(self: Job, block_size: usize) usize {
            return switch (self.kind) {
                .dma_read => |read| Planner.blockCount(read.len, block_size),
            };
        }
    };

    /// `len` bytes at `landing` of the job's blocks taken end to end, for
    /// every device of `writer_mask` at `destination_offset` of its shard.
    /// Every transfer stays inside one block.
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
    requests: []ReadRequest,
    events: []Dma.EventContext,
    /// Event slots handed out so far, one `fetchAdd` per submission from
    /// any device's pump.
    events_used: std.atomic.Value(usize) = .init(0),
    /// The tensor bytes the jobs read.
    source_bytes: u64,
    /// How long the plan took to cut.
    planning_ns: u64 = 0,
    /// Next job to claim; owned by the scheduler mutex.
    cursor: usize = 0,

    fn transfersOf(self: *const Plan, job: Job) []const Transfer {
        return self.transfers[job.transfer_start..][0..job.transfer_len];
    }
};

/// Builds immutable per-file source jobs and transfer records. Their claim
/// order (file order) is decided here once.
/// For DeepSeek-V4-Flash (148.65 GiB, 69,187 tensors), tensor-local reads made
/// about 69,445 source calls despite a 16 MiB request limit. Coalescing reduced
/// that to 9,524. A rigid grid still produced 78,665 DMA pieces; tensor-safe
/// cuts kept the same read count with 69,572 pieces. Per-tensor device buffers
/// remain intentional: reducing submissions below roughly tensors x destinations would require
/// packed device allocations or device-side scattering, not another read cut.
const Planner = struct {
    /// What every plan of a loader is cut by.
    const Config = struct {
        block_size: usize,
        /// The longest read, a whole number of blocks
        /// (`backend.requestSize`).
        request_size: usize,
    };

    /// How each file is read: aligned when it is local and the VFS that
    /// opens it reads it directly under `policy`, decided once per file and
    /// kept in its slot. A remote file is read exact without being opened
    /// here. Without a VFS (`io` is none, or no alignment) every file takes
    /// `alignment`.
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

        /// The file's alignment, decided the first time it is asked: a
        /// local file is opened here. One that fails to open is read exact:
        /// its reads fail with the same error (`LazyOnce`), which `readSlot`
        /// logs with the path.
        fn alignmentOf(self: Reads, slot: *SourceSlot) error{Canceled}!usize {
            const vfs = self.vfs orelse return self.alignment;
            if (slot.alignment) |known| return known;
            const direct = vfs.isLocal(slot.uri) and vfs.useDirectIo(slot.ensure(self.io) catch |err| switch (err) {
                error.Canceled => return error.Canceled,
                else => return 0,
            }, self.policy);
            const alignment = if (direct) self.alignment else 0;
            slot.alignment = alignment;
            return alignment;
        }
    };

    /// Plans `parts` one file at a time and publishes each plan as soon as
    /// it exists, so workers claim the first file while the rest is planned,
    /// each plan in the batch's arena. Stops at the first error; the caller
    /// seals or fails the batch. `reads` says how each file is read
    /// (`Reads`).
    /// Whole-model planning delayed the first read by 0.32 s for DeepSeek's
    /// 46 files on one MI300X; publishing per file overlaps the rest with IO.
    fn publishFiles(
        scheduler: *Scheduler,
        io: std.Io,
        batch: *Batch,
        parts: []Part,
        config: Config,
        reads: Reads,
    ) !void {
        const order = try sortedPartOrder(scheduler.allocator, parts);
        defer scheduler.allocator.free(order);
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(parts, order, file_start);
            const planning_started: std.Io.Timestamp = .now(io, .awake);
            const plan = try preparePlan(scheduler.allocator, batch.arena.allocator(), parts, order[file_start..file_end], config, reads);
            plan.planning_ns = @intCast(@max(planning_started.untilNow(io, .awake).nanoseconds, 0));
            try scheduler.publish(io, batch, plan);
            if (batch.published_at == null) batch.published_at = .now(io, .awake);
            file_start = file_end;
        }
    }

    /// Plans one file: `order` indexes its `parts` by offset and size
    /// (`sortedPartOrder`), and `JobCut.cutRuns` cuts them at the file's
    /// alignment (`Reads.alignmentOf`). The plan goes in `arena`
    /// (`JobCut.finish`), the cut in `allocator`.
    fn preparePlan(
        allocator: std.mem.Allocator,
        arena: std.mem.Allocator,
        parts: []Part,
        order: []const usize,
        config: Config,
        reads: Reads,
    ) !*Plan {
        const source_slot = parts[order[0]].source_slot;
        for (order) |part_index| std.debug.assert(std.mem.eql(u8, parts[part_index].source.file_uri, parts[order[0]].source.file_uri));
        const alignment = try reads.alignmentOf(source_slot);
        var cut: JobCut = .{
            .allocator = allocator,
            .parts = parts,
            .source_slot = source_slot,
            .config = config,
            .alignment = alignment,
            .maximum_job_len = maximumJobLen(config.request_size, alignment),
        };
        defer cut.deinit();
        // Jobs are claimed in planning order, which is file order. A
        // byte-fair order across destination devices was measured against
        // it on the two hosts where the DMA pump is not the ceiling and made
        // no difference.
        try cut.cutRuns(order);
        return cut.finish(arena);
    }

    /// The jobs and transfers of one plan as `preparePlan` cuts them.
    const JobCut = struct {
        /// One part's bytes in a read job: the logical range `[start, end)`
        /// of its destination, from `file_offset` of the file on.
        const Segment = struct { destination: *Destination, start: u64, end: u64, file_offset: u64 };

        allocator: std.mem.Allocator,
        parts: []Part,
        /// The file every job of the plan reads.
        source_slot: *SourceSlot,
        config: Config,
        /// The alignment of the plan's file (`Reads.alignmentOf`).
        alignment: usize,
        maximum_job_len: usize,
        jobs: std.ArrayList(Plan.Job) = .empty,
        transfers: std.ArrayList(Plan.Transfer) = .empty,
        safe_boundaries: std.ArrayList(u64) = .empty,
        source_bytes: u64 = 0,

        /// The plan of the jobs cut, copied into `arena` with every context
        /// its jobs can need, allocated before publication makes it visible
        /// to workers: a request per job and an event per writer of a
        /// transfer.
        fn finish(self: *JobCut, arena: std.mem.Allocator) !*Plan {
            const jobs = self.jobs.items;
            const transfers = self.transfers.items;
            const requests = try arena.alloc(ReadRequest, jobs.len);
            @memset(requests, ReadRequest.idle);
            const plan = try arena.create(Plan);
            plan.* = .{
                .source_slot = self.source_slot,
                .jobs = try arena.dupe(Plan.Job, jobs),
                .transfers = try arena.dupe(Plan.Transfer, transfers),
                .requests = requests,
                .events = try arena.alloc(Dma.EventContext, submissionCount(transfers)),
                .source_bytes = self.source_bytes,
            };
            return plan;
        }

        fn deinit(self: *JobCut) void {
            const allocator = self.allocator;
            self.safe_boundaries.deinit(allocator);
            self.transfers.deinit(allocator);
            self.jobs.deinit(allocator);
        }

        /// Cuts the runs of `order` (part indices in file order) into jobs:
        /// each run of touching or overlapping ranges into the minimum number
        /// of jobs, cut at part-safe boundaries where that number allows. No
        /// job depends on another, since every DMA piece is submitted as soon
        /// as its block is read. With an alignment (`Reads.alignmentOf`), a
        /// job's read widens to aligned bounds around its range
        /// (`maximumJobLen` leaves room for it), and the transfers address
        /// the widened read.
        fn cutRuns(self: *JobCut, order: []const usize) !void {
            const allocator = self.allocator;
            const parts = self.parts;
            const alignment = self.alignment;
            const maximum_job_len = self.maximum_job_len;
            var run_cursor: usize = 0;
            while (run_cursor < order.len) {
                self.safe_boundaries.clearRetainingCapacity();
                const first_range = parts[order[run_cursor]].fileRange();
                std.debug.assert(first_range.end > first_range.start); // backend.LoadSpec: parts are non-empty
                const first_offset = first_range.start;
                var run_end = first_range.end;
                var run_part_end = run_cursor + 1;
                while (run_part_end < order.len) : (run_part_end += 1) {
                    const candidate = parts[order[run_part_end]].fileRange();
                    if (candidate.start > run_end) break;
                    // A touching range starts where every preceding range has
                    // ended. Unlike an arbitrary part end, this is safe even
                    // when the batch contains overlapping or duplicate ranges.
                    if (candidate.start == run_end and
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
                    const read_end = if (alignment == 0) job_end else std.mem.alignForward(u64, job_end, alignment);
                    const read_len: usize = @intCast(read_end - read_start);
                    const transfer_start = self.transfers.items.len;
                    while (candidate_start < run_part_end) {
                        const candidate = parts[order[candidate_start]].fileRange();
                        if (candidate.end > job_start) break;
                        candidate_start += 1;
                    }
                    for (order[candidate_start..run_part_end]) |part_index| {
                        const part = &parts[part_index];
                        const range = part.fileRange();
                        if (range.start >= job_end) break;
                        const intersection_start = @max(job_start, range.start);
                        const intersection_end = @min(job_end, range.end);
                        if (intersection_start >= intersection_end) continue;
                        const logical_start = part.lattice.dst_offset + (intersection_start - range.start);
                        try self.appendRunTransfers(transfer_start, .{
                            .destination = part.destination,
                            .start = logical_start,
                            .end = logical_start + (intersection_end - intersection_start),
                            .file_offset = intersection_start,
                        }, read_start);
                    }
                    std.debug.assert(self.transfers.items.len != transfer_start);
                    try self.jobs.append(allocator, .{
                        .file_offset = read_start,
                        .transfer_start = transfer_start,
                        .transfer_len = self.transfers.items.len - transfer_start,
                        .kind = .{ .dma_read = .{ .len = read_len, .minimum_len = @intCast(job_end - read_start) } },
                    });
                    self.source_bytes +|= @intCast(job_len);
                    job_start = job_end;
                    jobs_remaining -= 1;
                }
                std.debug.assert(job_start == run_end);
                run_cursor = run_part_end;
            }
        }

        /// Appends `segment`'s transfers with its bytes where the read holds
        /// them, one per run (`ShardGrid.run`) and block, merged with the
        /// previous transfer from `transfer_start` on where block, writers
        /// and both offsets continue it: parts adjacent in the file and in
        /// the destination go out as one transfer.
        fn appendRunTransfers(self: *JobCut, transfer_start: usize, segment: Segment, read_start: u64) !void {
            const destination = segment.destination;
            const grid = &destination.grid;
            var at = segment.start;
            while (at < segment.end) {
                const stop = grid.runEnd(at, segment.end);
                const box, const box_offset = grid.locate(at);
                try self.appendBlocked(transfer_start, .{
                    .destination = destination,
                    .landing = @intCast(segment.file_offset + (at - segment.start) - read_start),
                    .writer_mask = grid.writers[box],
                    .destination_offset = @intCast(box_offset),
                    .len = @intCast(stop - at),
                });
                at = stop;
            }
        }

        /// Appends `transfer`, cut at block boundaries, merged with the
        /// previous transfer from `transfer_start` on where block, writers
        /// and both offsets continue it.
        fn appendBlocked(self: *JobCut, transfer_start: usize, transfer: Plan.Transfer) !void {
            const block_size = self.config.block_size;
            var cursor: usize = 0;
            while (cursor < transfer.len) {
                const at = transfer.landing + cursor;
                const in_block = at % block_size;
                const take = @min(transfer.len - cursor, block_size - in_block);
                const destination_offset = transfer.destination_offset + cursor;
                cursor += take;
                // A transfer that ends where this one starts is in its
                // block unless a block starts here.
                if (self.transfers.items.len > transfer_start and in_block != 0) {
                    const previous = &self.transfers.items[self.transfers.items.len - 1];
                    if (previous.destination == transfer.destination and
                        previous.writer_mask == transfer.writer_mask and
                        previous.landing + previous.len == at and
                        previous.destination_offset + previous.len == destination_offset)
                    {
                        previous.len += take;
                        continue;
                    }
                }
                try self.transfers.append(self.allocator, .{
                    .destination = transfer.destination,
                    .landing = at,
                    .writer_mask = transfer.writer_mask,
                    .destination_offset = destination_offset,
                    .len = take,
                });
            }
        }

        fn submissionCount(transfers: []const Plan.Transfer) usize {
            var count: usize = 0;
            for (transfers) |transfer| count += @popCount(transfer.writer_mask);
            return count;
        }
    };
    fn blockCount(len: usize, block_size: usize) usize {
        return len / block_size + @intFromBool(len % block_size != 0);
    }

    /// The longest tensor range one job may cover: the request size, less
    /// the two alignment units its widened read can add, so a widened job
    /// still fits a request (a credit's slot). `backend.requestSize` refuses
    /// a request that cannot hold them.
    fn maximumJobLen(request_size: usize, alignment: usize) usize {
        std.debug.assert(request_size != 0);
        if (alignment == 0) return request_size;
        std.debug.assert(std.math.isPowerOfTwo(alignment) and 2 * alignment < request_size);
        return request_size - 2 * alignment;
    }

    /// Part indices sorted by file URI, offset, size and index: the
    /// planner's input, taken one file group at a time (`fileGroupEnd`), in
    /// file order.
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
                const left_offset = left.fileRange().start;
                const right_offset = right.fileRange().start;
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
        plan: *Plan,
        index: usize,
    };

    allocator: std.mem.Allocator,
    /// Plans with unclaimed jobs in publish order.
    queue: std.Deque(*Plan) = .empty,
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
    /// A plan without jobs never joins the queue. `Canceled` once the
    /// scheduler failed (`fail`).
    fn publish(self: *Scheduler, io: std.Io, batch: *Batch, plan: *Plan) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.Canceled;
        const job_count = plan.jobs.len;
        try batch.plans.ensureUnusedCapacity(batch.arena.allocator(), 1);
        if (job_count != 0) try self.queue.ensureUnusedCapacity(self.allocator, 1);
        // Nothing below fails: the plan, its units and the queue entry
        // appear together.
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
    /// job keeps its unit with its worker. Each queued plan still holds an
    /// unclaimed unit of its batch, so only the retire of a batch's last
    /// queued plan can complete it, and nothing touches the batch after that.
    fn fail(self: *Scheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.condition.broadcast(io);
        while (self.queue.popFront()) |plan| {
            const unclaimed = plan.jobs.len - plan.cursor;
            plan.cursor = plan.jobs.len;
            plan.batch.releaseUnits(unclaimed);
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
        /// A read's ranges: one per block of a job (`readBlocks`).
        iovecs: [][]u8,

        fn init(allocator: std.mem.Allocator, maximum_blocks: usize) !Scratch {
            return .{
                .allocator = allocator,
                .iovecs = try allocator.alloc([]u8, maximum_blocks),
            };
        }

        fn deinit(self: *Scratch) void {
            self.allocator.free(self.iovecs);
            self.* = undefined;
        }
    };

    loader: *Loader,
    plan: *Plan,
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
        const block_size = loader.plan_config.block_size;
        const job = self.plan.jobs[self.index];
        const transfers = self.plan.transfersOf(job);
        // The first worker to touch a destination creates its state.
        var previous: ?*Destination = null;
        for (transfers) |transfer| {
            const destination = transfer.destination;
            if (destination == previous) continue;
            previous = destination;
            const init_started = awakeNs(loader.io);
            _ = destination.ensureState(loader.allocator, loader.io, loader.platform) catch |err| {
                loader.recordError(err, "initialize tensor {s} from {s}, shape={f}", .{ destination.name(), destination.parts[0].source.file_uri, destination.shape });
                return err;
            };
            _ = loader.metrics.tensor_init_ns.fetchAdd(awakeNs(loader.io) -| init_started, .monotonic);
        }
        const blocks = loader.credits.slotBlocks(self.slot)[0..job.blockLen(block_size)];
        switch (job.kind) {
            .dma_read => |read| try self.readBlocks(scratch, blocks, read),
        }
        for (transfers) |transfer| transfer.destination.recordRead(transfer.len);
        try Dma.enqueue(self, blocks);
    }

    /// Reads the job's `read` into `blocks`, its slot's, in file order.
    fn readBlocks(self: *ReadRequest, scratch: *Scratch, blocks: []const host_memory.Block, read: Plan.Job.Read) !void {
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
        batch.releaseUnits(1);
    }
};

/// Reads `[file_offset, file_offset + minimum)`, and possibly more, into
/// `buffers` in order, consuming them as it reads; bytes past `minimum` may
/// stay unread. The io takes as many buffers per call as it can (VFS's
/// direct read `IOV_MAX`, std's and zio's 8, a remote read all); a short
/// read resumes where it stopped. Reading stops once `minimum` bytes are
/// in, so the file may end only past `minimum`: the bytes after it are
/// padding a caller added to align the read, and the read that would fetch
/// them can start unaligned, which a file read directly refuses. Returns
/// the bytes read.
fn readFilePositionalAllV(
    io: std.Io,
    file: std.Io.File,
    buffers: [][]u8,
    file_offset: u64,
    minimum: u64,
) !u64 {
    var index: usize = 0;
    var completed: u64 = 0;
    while (completed < minimum) {
        std.debug.assert(index < buffers.len);
        const bytes_read = try file.readPositional(io, buffers[index..], file_offset + completed);
        if (bytes_read == 0) return error.UnexpectedEndOfFile;
        completed += bytes_read;
        var rest = bytes_read;
        while (index < buffers.len and rest >= buffers[index].len) : (index += 1) rest -= buffers[index].len;
        if (rest != 0) buffers[index] = buffers[index][rest..];
    }
    return completed;
}

/// Every worker read goes through here, under a read permit: a failure is
/// recorded with the file. Once the loader failed, returns its first error.
/// Consumes `iovecs` (`readFilePositionalAllV`).
fn readSlot(loader: *Loader, slot: *SourceSlot, iovecs: [][]u8, file_offset: u64, minimum_len: usize) !void {
    const io = loader.io;
    const file = slot.ensure(io) catch |err| {
        loader.recordError(err, "open {s}", .{slot.uri});
        return err;
    };
    loader.read_permits.waitUncancelable(io);
    defer loader.read_permits.post(io);
    if (loader.first_error.get()) |err| return err;
    const read_started = awakeNs(io);
    const read_result = readFilePositionalAllV(io, file, iovecs, file_offset, minimum_len);
    _ = loader.metrics.read_ns.fetchAdd(awakeNs(io) -| read_started, .monotonic);
    const bytes_read = read_result catch |err| {
        loader.recordError(err, "read {s} at {d}, {d} bytes at least", .{ slot.uri, file_offset, minimum_len });
        return err;
    };
    _ = loader.metrics.read_operations.fetchAdd(1, .monotonic);
    _ = loader.metrics.read_bytes.fetchAdd(@intCast(bytes_read), .monotonic);
}

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
    /// One DMA piece: `data`, in its request's slot, for `target` at
    /// `destination_offset`. The piece holds one of its request's
    /// references until its callback. Replicas share their source bytes:
    /// copying them to another CPU socket first was rejected on MI300X,
    /// where raw local and remote H2D both reached ~49-50 GiB/s per GPU
    /// while one CPU thread copied across sockets at 5-10 GiB/s.
    const Piece = struct {
        target: *DestinationState.Target,
        request: *ReadRequest,
        data: []const u8,
        destination_offset: usize,
    };

    /// Pieces ready for one device, submitted in arrival order so the
    /// oldest requests and submissions complete first. Owned by the
    /// device's `DevicePump.mutex`.
    const ReadyQueue = std.Deque(Piece);

    /// One device's submission state. Its pump runs on whichever thread
    /// finds work -- a worker after enqueueing, a ready callback after a
    /// completion -- one at a time per device (`pumping`), and shares
    /// nothing with the other devices' pumps. Per-device pumps load
    /// replicated DeepSeek on four GB300 in 4.68 s, against 5.41 s with one
    /// pump for every device, at unchanged pinned high-water. A dedicated
    /// task per device took 5.0-5.4 s with a wake per completion (a futex
    /// round trip per piece), or 4.4-5.1 s with half-budget wake hysteresis.
    const DevicePump = struct {
        mutex: std.Io.Mutex = .init,
        queue: ReadyQueue = .empty,
        /// In-flight DMA bytes and submissions; owned by `mutex`.
        active_bytes: usize = 0,
        active_pieces: usize = 0,
        /// True while one thread runs this device's pump; owned by `mutex`.
        /// One submitter per device is a correctness requirement:
        /// `Target.nextIsLast` flags a tensor's last piece in submission
        /// order.
        pumping: bool = false,
        /// Contexts whose callback fired, for the next pump or a batch's
        /// retirement to destroy, never the event's own callback: an
        /// intrusive stack through `EventContext.next_retired`, owned by
        /// `mutex`. Destroying an event from the next pump instead of at its
        /// batch's retirement bounds live PJRT events by the DMA width plus
        /// one pump batch rather than by a submission's transfer count;
        /// every shipped plugin accepted destruction outside the event's own
        /// callback (16,384 fired events on two B70, CUDA and ROCm alike).
        retired: ?*EventContext = null,

        /// Whether the next queued piece may go. The budget is in bytes,
        /// not submissions: `dma_calibration.depth` blocks, and a piece is
        /// one tensor's slice of a block, so a model of small tensors needs
        /// many more pieces in flight to keep the same bytes moving: eight
        /// average DeepSeek pieces occupy only ~18 MiB against the 128 MiB
        /// budget of 16 MiB blocks. The piece that crosses the budget is
        /// admitted: a device with room always has a piece in flight.
        fn hasRoom(self: *const DevicePump, budget_bytes: usize) bool {
            return self.active_bytes < budget_bytes and self.active_pieces < max_dma_pieces_per_device;
        }

        /// `mutex`. Destroys every retired event.
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
        /// Awake-clock nanoseconds of the submission.
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

    /// Queues every piece of the request's job before pumping: queuing piece
    /// by piece costs roughly 69k-79k mutex and pump trips per DeepSeek load,
    /// which coalesced reads alone do not remove. A transfer reads `len`
    /// bytes from its `landing` of `blocks` taken end to end, inside one
    /// block, once for each of its writers: the devices the planner planned
    /// for (both come from the sharding's canonical device order).
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
    /// has no room or nothing is queued. Any queued piece can go: nothing
    /// waits for another piece. Exactly one pump runs per device at a time,
    /// see `DevicePump.pumping`.
    fn pump(loader: *Loader, device_index: usize) void {
        const device_pump = &loader.pumps[device_index];
        while (true) {
            var selected: ?Piece = null;
            device_pump.mutex.lockUncancelable(loader.io);
            device_pump.destroyRetired();
            if (loader.first_error.get() == null) {
                if (!device_pump.hasRoom(loader.dma_budget_bytes)) {
                    _ = loader.metrics.pump_stops_full.fetchAdd(1, .monotonic);
                } else if (device_pump.queue.popFront()) |piece| {
                    device_pump.active_bytes += piece.data.len;
                    device_pump.active_pieces += 1;
                    selected = piece;
                } else {
                    _ = loader.metrics.pump_stops_empty.fetchAdd(1, .monotonic);
                }
            }
            if (selected == null) device_pump.pumping = false;
            device_pump.mutex.unlock(loader.io);
            if (selected) |piece| submitOne(loader, piece) else return;
        }
    }

    fn submitOne(loader: *Loader, piece: Piece) void {
        const device_index = piece.target.device_index;
        const len = piece.data.len;
        submitTransfer(loader, piece) catch |err| {
            loader.recordError(err, "submit DMA transfer: device={d}, destination_offset={d}, bytes={d}", .{ device_index, piece.destination_offset, len });
            finishPiece(loader, device_index, len, null, piece.request);
        };
    }

    fn submitTransfer(loader: *Loader, piece: Piece) !void {
        const api = loader.platform.pjrt_api;
        const target = piece.target;
        const is_last = target.nextIsLast(piece.data.len);
        const submit_started = awakeNs(loader.io);
        const event = try target.manager.transferData(
            api,
            0,
            piece.data,
            @intCast(piece.destination_offset),
            is_last,
        );
        target.noteSubmitted(piece.data.len);

        // The plan holds one event slot per planned submission; the batch
        // owns it.
        const plan = piece.request.plan;
        const event_index = plan.events_used.fetchAdd(1, .monotonic);
        std.debug.assert(event_index < plan.events.len);
        const ctx = &plan.events[event_index];
        ctx.* = .{
            .request = piece.request,
            .pjrt_event = event,
            .device_index = target.device_index,
            .len = piece.data.len,
            .submitted_ns = submit_started,
        };

        _ = loader.metrics.dma_submissions.fetchAdd(1, .monotonic);
        event.onReady(api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
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
                finishPiece(loader_, device_index, len, ctx_, request);
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(api) catch {};
            ctx.destroyEvent();
            return err;
        };
        _ = loader.metrics.dma_submit_ns.fetchAdd(awakeNs(loader.io) -| submit_started, .monotonic);
    }

    /// Ends a submitted piece: frees its room, which may pump on this thread;
    /// then retires a fired context, after that pump, so its event is never
    /// destroyed inside its own callback; then releases the request last,
    /// since that release may complete and free the batch.
    fn finishPiece(loader: *Loader, device_index: usize, len: usize, fired: ?*EventContext, request: *ReadRequest) void {
        eventCompleted(loader, device_index, len);
        if (fired) |ctx| retireEvent(loader, ctx);
        request.release();
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
        // piece so request lifecycles cannot wait forever on entries that
        // the failed pumps will no longer submit.
        if (loader.first_error.get() != null)
            abortReady(loader)
        else
            requestPump(loader, device_index);
    }

    /// Force-completes every queued piece, one device at a time. A
    /// completion here may finish its batch, but a batch still has an entry
    /// queued until its last one is popped, and it is retired with every
    /// pump locked, so no queued entry is touched after its request ends.
    fn abortReady(loader: *Loader) void {
        for (loader.pumps) |*device_pump| {
            device_pump.mutex.lockUncancelable(loader.io);
            while (device_pump.queue.popFront()) |piece| {
                piece.request.release();
            }
            device_pump.mutex.unlock(loader.io);
        }
    }
};

const DestinationState = struct {
    /// One destination buffer of a tensor. PJRT makes the buffer ready once
    /// every transfer submitted to it has completed and one of them carried
    /// the last-transfer flag, whatever their completion order: the flag
    /// only closes the buffer to further calls. The pump, the only
    /// submitter, therefore flags the submission that completes the
    /// placement's bytes, and no piece ever waits for another one.
    /// Flagging the highest-offset piece instead makes a tail wait for its
    /// prefixes, which can deadlock a sharded multi-device load: tails can
    /// hold every lifecycle credit while their prefixes stay unclaimed,
    /// whatever the pinned memory. This follows XLA's
    /// CommonAsyncHostToDeviceTransferManager: readiness requires zero
    /// in-flight transfers and a last-call flag, not destination-offset order.
    const Target = struct {
        manager: *pjrt.AsyncHostToDeviceTransferManager,
        device_index: usize,
        /// Bytes of the placement on this device. The pieces partition them:
        /// those of every part, from whichever file and plan.
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

    /// The destination's transfer targets, one per device in canonical
    /// order.
    targets: []Target,

    /// Creates the destination's device buffers and transfer managers, one
    /// per device of its grid in canonical order, and writes the output
    /// shell. Runs once per destination (`LazyOnce`), however many parts
    /// feed it, on the worker that first reads for it.
    fn init(allocator: std.mem.Allocator, platform: *const Platform, destination: *const Destination) StateInitError!DestinationState {
        const grid = &destination.grid;
        const targets = try allocator.alloc(Target, grid.devices.len);
        errdefer allocator.free(targets);

        var pjrt_buffers: Buffer.Shards = .empty;
        var initialized: usize = 0;
        errdefer {
            for (targets[0..initialized]) |target| target.manager.deinit(platform.pjrt_api);
            for (pjrt_buffers.constSlice()) |buffer| buffer.deinit(platform.pjrt_api);
        }

        const shape_spec: pjrt.ShapeSpec = .init(
            grid.local.dims(),
            pjrtx.bufferTypeFromDtype(grid.local.dtype()),
        );
        for (grid.devices, 0..) |device, i| {
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
                .total = grid.local.byteSize(),
            };
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(pjrt_buffer);
        }

        destination.output.* = .fromPjrtBuffers(platform, destination.shape, pjrt_buffers.constSlice());
        return .{ .targets = targets };
    }

    fn deinit(self: *DestinationState, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        for (self.targets) |target| target.manager.deinit(api);
        allocator.free(self.targets);
    }
};

/// The lifecycle credits: one per request the pre-grown pinned set holds,
/// each a fixed slot of `blocks_per_slot` blocks. A worker takes a credit
/// before claiming a job, and the request gives it back with its last
/// reference, after its last DMA callback. Only the holder reads into and
/// transfers from its slot's blocks, so the credits are the one bound on
/// pinned memory in use. A block whose pieces are done stays with its
/// request until the request ends; nobody else could use it anyway, since
/// every credit owns a whole slot.
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

/// The debug summary's counters (`Loader.logSummary`): read only by logs;
/// nothing here decides.
const Metrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    /// Time workers spend waiting for a lifecycle credit and inside the
    /// source read, summed over requests: waits above the read time mean the
    /// load is DMA-completion bound.
    credit_wait_ns: std.atomic.Value(u64) = .init(0),
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
    /// managers there, and the other workers of the same tensor wait for it.
    tensor_init_ns: std.atomic.Value(u64) = .init(0),
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

/// Writes `contents` to a new file `name` in `dir` and returns its real
/// path, in `path_buffer`.
fn writeTestFile(dir: std.Io.Dir, io: std.Io, name: []const u8, contents: []const u8, path_buffer: []u8) ![]const u8 {
    const file = try dir.createFile(io, name, .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, contents, 0);
    return path_buffer[0..try file.realPath(io, path_buffer)];
}

// Planner tests.

/// Every file read at `alignment`, none opened to ask.
fn fixedReads(alignment: usize) Planner.Reads {
    return .{ .io = undefined, .vfs = null, .policy = .off, .alignment = alignment };
}

/// `Planner.preparePlan` with every file read at `alignment`, kept in
/// `arena`; the testing allocator checks the cut for leaks.
fn prepareTestPlan(arena: std.mem.Allocator, parts: []Part, order: []const usize, config: Planner.Config, alignment: usize) !*Plan {
    return Planner.preparePlan(std.testing.allocator, arena, parts, order, config, fixedReads(alignment));
}

/// One plain destination per source, as `createDestinations` builds them
/// for whole-tensor specs, without device state: for planner tests.
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
        destination.* = .{
            .shape = source.shape,
            .grid = .init(platform, source.shape),
            .output = output,
            .parts = @as(*[1]Part, part),
            .unread_bytes = .init(source.byteSize()),
        };
    }
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
    batch.releaseUnits(1 + claimed);
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
    const batch = try Batch.create(allocator, io, 0);
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 8,
    }, fixedReads(0));

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
    const aligned_plan = try prepareTestPlan(arena.allocator(), &aligned_items, &.{ 0, 1, 2 }, .{
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
    const widened_plan = try prepareTestPlan(arena.allocator(), &aligned_items, &.{ 0, 1, 2 }, .{
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

// Scheduler tests.

/// A plan of `job_count` unit jobs (`file_offset` = index), each with one
/// block and one event slot, and no transfers, in `arena`.
fn testPlan(arena: std.mem.Allocator, job_count: usize) !*Plan {
    const jobs = try arena.alloc(Plan.Job, job_count);
    const requests = try arena.alloc(ReadRequest, job_count);
    @memset(requests, ReadRequest.idle);
    const events = try arena.alloc(Dma.EventContext, job_count);
    for (jobs, 0..) |*job, index| job.* = .{
        .file_offset = index,
        .transfer_start = 0,
        .transfer_len = 0,
        .kind = .{ .dma_read = .{ .len = 1, .minimum_len = 1 } },
    };
    const plan = try arena.create(Plan);
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

/// A one-file submission's sequence: publish, then drop the sentinel.
fn publishTestBatch(scheduler: *Scheduler, job_count: usize) !*Batch {
    const batch = try Batch.create(std.testing.allocator, std.testing.io, 0);
    errdefer batch.destroy();
    try publishTestPlan(scheduler, batch, job_count);
    batch.releaseUnits(1);
    return batch;
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
    const batch = try Batch.create(allocator, io, 0);
    try Planner.publishFiles(&scheduler, io, batch, &parts, .{
        .block_size = 4,
        .request_size = 4,
    }, fixedReads(0));
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), unclaimedJobs(&scheduler));
    try std.testing.expect(batch.published_at != null);

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
    batch.releaseUnits(1);
    try std.testing.expect(!batch.done.isSet());
    batch.releaseUnits(3);
    try std.testing.expect(batch.done.isSet());
    batch.destroy();
}

test "fifo scheduler claims plan by plan in publish order, and a done batch leaves while a later plan is queued" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    // An open submission's first plan, a sealed batch, then the open
    // submission's second plan: an open submission holds no place of its
    // own.
    const open = try Batch.create(std.testing.allocator, io, 0);
    try publishTestPlan(&scheduler, open, 1);
    const sealed = try publishTestBatch(&scheduler, 2);
    try publishTestPlan(&scheduler, open, 1);
    try std.testing.expectEqual(@as(usize, 4), unclaimedJobs(&scheduler));

    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan == open.plans.items[0] and claim.index == 0);
    for (0..2) |index| {
        claim = scheduler.claim(io).?;
        try std.testing.expect(claim.plan == sealed.plans.items[0] and claim.index == index);
    }
    // The sealed batch left the queue with its last claim, so it completes
    // and goes away while the open one's second plan is still queued.
    sealed.releaseUnits(2);
    try std.testing.expect(sealed.done.isSet());
    sealed.destroy();
    try std.testing.expectEqual(@as(usize, 1), scheduler.queue.len);

    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan == open.plans.items[1] and claim.index == 0);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    // Claimed out, the open submission completes only once its sentinel
    // goes.
    open.releaseUnits(2);
    try std.testing.expect(!open.done.isSet());
    open.releaseUnits(1);
    try std.testing.expect(open.done.isSet());
    open.destroy();
}

test "fifo scheduler failure retires the unclaimed jobs of every queued plan and refuses later publishes" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    // A batch without jobs is never queued and completes at publish.
    const empty = try publishTestBatch(&scheduler, 0);
    try std.testing.expect(empty.done.isSet());
    empty.destroy();
    // An open submission of two plans, then a sealed batch.
    const open = try Batch.create(std.testing.allocator, io, 0);
    try publishTestPlan(&scheduler, open, 2);
    try publishTestPlan(&scheduler, open, 3);
    const sealed = try publishTestBatch(&scheduler, 3);
    try std.testing.expectEqual(@as(usize, 8), unclaimedJobs(&scheduler));
    try std.testing.expectEqual(@as(usize, 6), open.remaining.load(.acquire));

    // The claim keeps its unit and the open submission its sentinel; every
    // other job is retired.
    try std.testing.expect(scheduler.claim(io).?.plan.batch == open);
    scheduler.fail(io);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.len);
    try std.testing.expectEqual(@as(usize, 2), open.remaining.load(.acquire));
    try std.testing.expect(sealed.done.isSet());
    sealed.destroy();
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expect(!scheduler.waitForWork(io));
    // The submission goes on: its next plan is refused, and the sentinel
    // drop leaves only the claim to complete it.
    try std.testing.expectError(error.Canceled, publishTestPlan(&scheduler, open, 1));
    open.releaseUnits(1);
    try std.testing.expect(!open.done.isSet());
    open.releaseUnits(1);
    try std.testing.expect(open.done.isSet());
    open.destroy();
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
    batch.releaseUnits(1);
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
        batch.releaseUnits(16);
        try std.testing.expect(batch.done.isSet());
        batch.destroy();
    }
}

// DMA pump and transfer target tests.

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

test "the submission that completes a target's bytes carries the last flag" {
    var target: DestinationState.Target = .{ .manager = undefined, .device_index = 0, .total = 100 };
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

// Request, credit and batch lifecycle tests.

/// A single-device loader without PJRT, workers or pinned arenas: enough
/// for request, credit and batch lifecycle tests, with 64-byte blocks cut
/// as `create` cuts them, the zero block then `slots` credits of one block.
/// `platform` and `workspace` are its only undefined fields: what these
/// tests run reaches the platform only through a non-null PJRT event
/// (`Dma.EventContext.destroyEvent`). Must not move after `init`.
const TestLoader = struct {
    memory: []u8,
    pumps: [1]Dma.DevicePump = .{.{}},
    loader: Loader,

    fn init(self: *TestLoader, slots: usize) !void {
        const allocator = std.testing.allocator;
        const memory = try allocator.alloc(u8, (slots + 1) * 64);
        errdefer allocator.free(memory);
        const blocks = try allocator.alloc(host_memory.Block, slots + 1);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, index| block.* = memory[index * 64 ..][0..64];
        @memset(blocks[0], 0);
        self.* = .{ .memory = memory, .loader = undefined };
        self.loader = .{
            .allocator = allocator,
            .io = std.testing.io,
            .platform = undefined,
            .workspace = undefined,
            .blocks = blocks,
            .scheduler = .init(allocator),
            .read_permits = .{ .permits = 1 },
            .read_width = 1,
            .credits = try .init(allocator, blocks[1..], 1),
            .pumps = &self.pumps,
            .dma_budget_bytes = 64,
            .created_at = .now(std.testing.io, .awake),
            .plan_config = .{ .block_size = 64, .request_size = 64 },
            .reads = fixedReads(0),
        };
    }

    fn deinit(self: *TestLoader) void {
        const allocator = std.testing.allocator;
        self.loader.scheduler.deinit();
        self.pumps[0].queue.deinit(allocator);
        self.loader.credits.deinit(allocator);
        allocator.free(self.loader.blocks);
        allocator.free(self.memory);
    }
};

test "a callback failing while a piece is queued drains it and completes the batch" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;

    // One request whose only piece is queued behind a DMA slot that a failed
    // callback is about to free.
    const batch = try publishTestBatch(&loader.scheduler, 1);
    const request = loader.claimRequest().?;
    var target: DestinationState.Target = .{ .manager = undefined, .device_index = 0, .total = 64 };
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
    for (&requests) |*request| request.* = loader.claimRequest().?;
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

test "a worker waiting for a credit when the loader fails gets one back and stops" {
    const io = std.testing.io;
    var fixture: TestLoader = undefined;
    try fixture.init(1);
    defer fixture.deinit();
    const loader = &fixture.loader;

    // A request in flight holds the only credit; a worker that saw the
    // second job waits on the credits.
    const batch = try publishTestBatch(&loader.scheduler, 2);
    const request = loader.claimRequest().?;
    try loader.worker_group.concurrent(io, Loader.workerMain, .{loader});
    while (loader.credits.condition.state.load(.monotonic).waiters == 0) io.sleep(.fromMilliseconds(1), .awake) catch {};

    // The failure retires the second job and closes nothing: the request
    // gives its credit back as it ends, the worker claims nothing with it,
    // and `destroy`'s shutdown stops it.
    loader.recordError(error.InputOutput, "injected source failure", .{});
    try std.testing.expect(!batch.done.isSet());
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
        loader.claimRequest().?,
        loader.claimRequest().?,
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
            while (loader_.scheduler.waitForWork(loader_.io)) if (loader_.claimRequest()) |request| request.release();
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

// Source read tests.

test "readFilePositionalAllV resumes short reads and stops at minimum" {
    const OneByteIo = struct {
        const file_end = 11;

        /// Reads one byte per call, the low byte of its offset, from a file
        /// that ends at `file_end`, and counts the calls in `userdata`.
        fn fileReadPositional(userdata: ?*anyopaque, _: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
            const calls: *usize = @ptrCast(@alignCast(userdata.?));
            calls.* += 1;
            if (offset >= file_end) return 0;
            for (data) |buffer| {
                if (buffer.len == 0) continue;
                buffer[0] = @truncate(offset);
                return 1;
            }
            return 0;
        }
    };
    var vtable = std.testing.io.vtable.*;
    vtable.fileReadPositional = OneByteIo.fileReadPositional;
    var calls: usize = 0;
    const io: std.Io = .{ .userdata = &calls, .vtable = &vtable };
    const file: std.Io.File = .{ .handle = 0, .flags = .{ .nonblocking = false } };

    var output: [5]u8 = @splat(0xaa);
    var buffers = [_][]u8{ output[0..1], output[1..1], output[1..3], output[3..5] };
    try std.testing.expectEqual(@as(u64, 4), try readFilePositionalAllV(io, file, &buffers, 7, 4));
    try std.testing.expectEqual(@as(usize, 4), calls);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 10, 0xaa }, &output);

    var past_end = [_][]u8{&output};
    try std.testing.expectError(error.UnexpectedEndOfFile, readFilePositionalAllV(io, file, &past_end, 7, output.len));
}

// LazyOnce tests.

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

// Whole-loader tests.

/// Submits `specs`, which must fail with `expected`; a batch that went out
/// anyway is awaited before the test fails.
fn expectSubmitError(loader: *Loader, specs: []const LoadSpec, expected: anyerror) !void {
    const batch = loader.submit(specs, .none) catch |err| return std.testing.expectEqual(expected, err);
    loader.awaitBatch(batch) catch {};
    return error.TestUnexpectedResult;
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
    const result: anyerror!void = if (Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off })) |loader| unexpected: {
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
    var later_buffer: [1024]u8 = undefined;
    const later_path = try writeTestFile(tmp.dir, io, "weights.bin.later", &contents, &later_buffer);
    const sources = [_]safetensors.Tensor{
        .{ .file_uri = path, .name = "first", .shape = .init(.{4}, .u8), .offset = 0 },
        .{ .file_uri = later_path, .name = "later", .shape = .init(.{5}, .u8), .offset = 0 },
    };
    const parts = [_]LoadPart{ .whole(&sources[0]), .whole(&sources[1]) };

    // Device memory, the one kind this backend loads, goes through blocks
    // and DMA.
    for ([_]Memory{.default}) |memory| {
        // Planning allocates from the scheduler's allocator, whose
        // allocations fail in turn: the first file's planning fails before
        // any plan is published, then the later file's after the first
        // file's plan was. Past the last, planning succeeds and the read
        // fails: the later source ends after four of five bytes.
        var fail_index: usize = 0;
        while (true) : (fail_index += 1) {
            var outputs: [2]Buffer = @splat(.{
                ._platform = platform,
                ._shape = sources[0].shape,
                ._shards = .empty,
            });
            defer for (&outputs) |*output| output.deinit();
            var failing: std.testing.FailingAllocator = .init(allocator, .{ .fail_index = fail_index });
            const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off });
            defer loader.destroy();
            loader.scheduler.allocator = failing.allocator();
            const specs = [_]LoadSpec{
                .{ .shape = sources[0].shape, .memory = memory, .output = &outputs[0], .parts = parts[0..1] },
                .{ .shape = sources[1].shape, .memory = memory, .output = &outputs[1], .parts = parts[1..2] },
            };

            if (loader.submit(&specs, .none)) |batch| {
                try std.testing.expectError(error.UnexpectedEndOfFile, loader.awaitBatch(batch));
            } else |err| try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(@as(usize, 0), loader.scheduler.queue.len);
            // A worker the batch woke may still hold the credit it took for
            // a claim that finds nothing: every credit is free once the
            // workers stopped (`stopWorkers`).
            loader.stopWorkers();
            try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
            try std.testing.expectEqual(loader.read_width, loader.read_permits.permits);

            // Each failure fails the loader: a later submission gets it.
            try std.testing.expectEqual(@as(usize, 1), loader.batch_count);
            const expected: anyerror = if (failing.has_induced_failure) error.OutOfMemory else error.UnexpectedEndOfFile;
            try std.testing.expectError(expected, loader.checkOpen());
            try expectSubmitError(loader, specs[0..1], expected);
            if (!failing.has_induced_failure) break;
        }
    }
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
        const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off });
        defer loader.destroy();
        try std.testing.expectEqual(@as(usize, 2), loader.read_width);
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
        // A worker the batch woke may still hold the credit it took for a
        // claim that finds nothing: every credit is free once the workers
        // stopped (`stopWorkers`).
        loader.stopWorkers();
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

test "whole tensors load on four devices, sharded by rows or columns, replicated, or joined from two files" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 4);
    defer devices.deinit();
    const mesh = try devices.mesh("tp");

    // Rows of 32 bytes, 16 per device; rows cut into four pieces of 8
    // bytes, one per device; a table on every device; and a tensor whose
    // rows continue in a second file.
    var rows: [64 * 32]u8 = undefined;
    var cols: [1024 * 32]u8 = undefined;
    var table: [3000 * 4]u8 = undefined;
    var tail: [64 * 32]u8 = undefined;
    testing.pattern(&rows, 1);
    testing.pattern(&cols, 2);
    testing.pattern(&table, 3);
    testing.pattern(&tail, 4);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try testing.writeSafetensors(allocator, io, tmp.dir, "first.safetensors", &registry, &.{
        .{ .name = "rows", .dtype = "U16", .dims = &.{ 64, 16 }, .bytes = &rows },
        .{ .name = "cols", .dtype = "U16", .dims = &.{ 1024, 16 }, .bytes = &cols },
        .{ .name = "table", .dtype = "U8", .dims = &.{ 3000, 4 }, .bytes = &table },
    });
    try testing.writeSafetensors(allocator, io, tmp.dir, "second.safetensors", &registry, &.{
        .{ .name = "tail", .dtype = "U16", .dims = &.{ 64, 16 }, .bytes = &tail },
    });
    var joined: [rows.len + tail.len]u8 = undefined;
    @memcpy(joined[0..rows.len], &rows);
    @memcpy(joined[rows.len..], &tail);
    const parts = [_]LoadPart{
        .whole(registry.tensors.getPtr("rows").?),
        .whole(registry.tensors.getPtr("cols").?),
        .whole(registry.tensors.getPtr("table").?),
        // rows again, then tail after it: the first file's run overlaps.
        .whole(registry.tensors.getPtr("rows").?),
        .{ .source = registry.tensors.getPtr("tail").?, .lattice = .{ .dst_offset = rows.len, .width = tail.len } },
    };
    var outputs: [4]Buffer = undefined;
    const specs = [_]LoadSpec{
        .{ .shape = Shape.init(.{ .row = 64, .col = 16 }, .u16).withPartitioning(&mesh, .{ .row = .model }), .output = &outputs[0], .parts = parts[0..1] },
        .{ .shape = Shape.init(.{ .row = 1024, .col = 16 }, .u16).withPartitioning(&mesh, .{ .col = .model }), .output = &outputs[1], .parts = parts[1..2] },
        .{ .shape = Shape.init(.{ .row = 3000, .col = 4 }, .u8).withPartitioning(&mesh, .{}), .output = &outputs[2], .parts = parts[2..3] },
        .{ .shape = Shape.init(.{ .row = 128, .col = 16 }, .u16).withPartitioning(&mesh, .{ .row = .model }), .output = &outputs[3], .parts = parts[3..5] },
    };
    const globals = [_][]const u8{ &rows, &cols, &table, &joined };

    // Blocks of 4 KiB and requests of two: the first file's run is cut into
    // jobs, and the jobs' transfers at block ends.
    const loader = try Loader.create(allocator, io, devices.platform, .{
        .read_parallelism = 2,
        .read_hints = .{ .read_chunk_size = 8192 },
        .dma_block_size = 4096,
        .direct_io = .off,
    });
    defer loader.destroy();
    const batch = try loader.submit(&specs, .none);
    var jobs: usize = 0;
    for (batch.plans.items) |plan| jobs += plan.jobs.len;
    try loader.awaitBatch(batch);
    defer for (&outputs) |*output| output.deinit();
    for (&outputs, globals) |*output, global| try testing.expectShards(allocator, io, output, global, .default);
    // 46,816 bytes in the first file, 2,048 in the second.
    try std.testing.expectEqual(@as(usize, 6 + 1), jobs);
    // Every credit is free once the workers stopped (`stopWorkers`).
    loader.stopWorkers();
    try std.testing.expectEqual(@as(usize, 0), loader.credits.inUse(io));
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
    const loader = try Loader.create(failing.allocator(), io, platform, .{ .read_parallelism = 2, .direct_io = .off });
    defer loader.destroy();
    const parts = [_]LoadPart{.whole(&source)};
    failing.arm(loader);
    try expectSubmitError(loader, &.{.{ .shape = source.shape, .output = &output, .parts = &parts }}, error.UnexpectedEndOfFile);
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
    const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2, .direct_io = .off });
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
