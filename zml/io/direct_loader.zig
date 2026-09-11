//! Direct loading follows a batch from per-file planning through FIFO claims,
//! source reads and DMA completion. The source width is fixed per load
//! profile (`limits.defaultReadParallelism`) and only steps down when the
//! source throttles; runtime ownership and synchronization live here.

const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const backend = @import("backend.zig");
const host_memory = @import("host_memory.zig");
const dma_calibration = @import("dma_calibration.zig");
const DispatchSpans = @import("DispatchSpans.zig");
const load_limits = @import("limits.zig");
const platform_mod = @import("../platform.zig");
const pjrtx = @import("../pjrtx.zig");
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");

const CreateOptions = platform_mod.CreateOptions;
const BackendConfig = backend.Config;
const LoadSpec = backend.LoadSpec;
const Platform = platform_mod.Platform;

const load_log = std.log.scoped(.@"zml/io/load");

/// Bounds per-device event overhead for tiny tensors. This is not a measured
/// optimum: 64 pieces smaller than block_size / 8 cannot fill an eight-block
/// byte budget. Raising the stage also costs host memory: replicated DeepSeek
/// on four GB300 improved 5.536 -> 5.024 s at depth 8 -> 32, but pinned
/// high-water grew 0.9-1.0 -> 2.36-2.50 GiB over six paired runs. That
/// tradeoff did not justify a larger default stage.
const max_dma_pieces_per_device: usize = 64;

/// The direct DMA backend. Submissions and awaits come from one task at a
/// time; the workers, the pumps and the throttle watch run concurrently
/// with them.
pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    load_profile: VFS.LoadProfile,
    calibration: dma_calibration.Result,
    pool: host_memory.BlockPool,
    scheduler: Scheduler,
    metrics: Metrics = .{},
    read_gate: RequestGate,
    request_gate: RequestGate,
    pipeline: Pipeline,
    /// Present when the profile reports read statistics (the remote VFS
    /// backends): the one thing that changes the width during a load.
    throttle: ?ThrottleWatch = null,
    worker_group: std.Io.Group = .init,
    throttle_group: std.Io.Group = .init,
    source_slots: std.StringHashMapUnmanaged(*SourceSlot) = .empty,
    /// Device bytes allocated for outputs so far, per `platform.devices`
    /// index, cumulative: the front end subtracts it from what it submitted
    /// to know what is still to land.
    allocated_bytes: []std.atomic.Value(u64),
    created_at: std.Io.Timestamp,
    /// Submissions so far; the next batch's sequence number.
    batch_count: usize = 0,
    /// Concurrent source reads: the configured width clipped to what the
    /// pinned budget holds, halved by the throttle watch while it runs.
    width: usize,
    limits: RequestGateLimits.Config,
    plan_config: Planner.Config,
    /// See `Config.direct_io`.
    direct_io: VFS.DirectIo,
    maximum_blocks_per_job: usize,

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: BackendConfig,
    ) !*Loader {
        const self = try allocator.create(Loader);
        errdefer allocator.destroy(self);
        const allocated_bytes = try allocator.alloc(std.atomic.Value(u64), platform.devices.len);
        errdefer allocator.free(allocated_bytes);
        @memset(allocated_bytes, .init(0));
        const sizing = try Sizing.init(allocator, io, platform, opts);
        const calibration = sizing.calibration;
        const source_alignment = if (opts.direct_io != .off) opts.load_profile.direct_io_alignment orelse 0 else 0;
        const width = @min(opts.read_parallelism, sizing.feasible_width);
        const limits_config: RequestGateLimits.Config = .{
            .feasible_width = sizing.feasible_width,
            .retained = sizing.retained_credits,
            .dma_stage = sizing.dma_stage_requests,
        };
        const limits = limits_config.at(width);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .load_profile = opts.load_profile,
            .calibration = calibration,
            .pool = sizing.pool,
            .scheduler = .init(allocator),
            .read_gate = .init(limits.read),
            .request_gate = .init(limits.lifecycle),
            .pipeline = undefined,
            .allocated_bytes = allocated_bytes,
            .created_at = .now(io, .awake),
            .width = width,
            .limits = limits_config,
            .plan_config = .{
                .device_count = platform.devices.len,
                .block_size = calibration.block_size,
                .request_size = sizing.request_size,
                .alignment = source_alignment,
            },
            .direct_io = opts.direct_io,
            .maximum_blocks_per_job = sizing.maximum_blocks_per_job,
        };
        errdefer {
            self.scheduler.deinit();
            self.pool.deinit();
        }
        _ = Planner.maximumJobLen(self.plan_config) catch return error.InvalidLoadProfile;

        self.pipeline = try Pipeline.init(
            allocator,
            io,
            platform,
            &self.pool,
            &self.read_gate,
            &self.request_gate,
            calibration.block_size,
            &self.metrics,
            &self.scheduler,
            calibration.max_in_flight_per_device * calibration.block_size,
        );
        errdefer self.pipeline.deinit();
        if (opts.load_profile.stats) |provider| {
            self.throttle = .{
                .cursor = .{ .provider = provider, .previous = provider.snapshot() },
                .metrics = &self.metrics,
                .read_gate = &self.read_gate,
                .request_gate = &self.request_gate,
                .limits = limits_config,
                .width = &self.width,
            };
        }
        errdefer self.stopWorkers();
        for (0..limits.workers()) |_| try self.worker_group.concurrent(io, workerMain, .{self});
        if (self.throttle) |*watch| try self.throttle_group.concurrent(io, ThrottleWatch.run, .{ watch, io });
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, direct_io={t}, source_alignment={d}, dma_block_size={Bi:.2}, dma_budget_per_device={Bi:.2}, source_width={d}, lifecycle_credits={d}, workers={d}, feasible_width={d}, retained={Bi:.2}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            sizing.request_size,
            opts.direct_io,
            source_alignment,
            calibration.block_size,
            self.pipeline.dma_budget_bytes,
            width,
            limits.lifecycle,
            limits.workers(),
            sizing.feasible_width,
            self.pool.workspace.mapped_bytes,
        });
        return self;
    }

    /// Plans `specs` one source file at a time, publishes each file's plan
    /// as soon as it exists behind every earlier submission, then seals the
    /// batch: work on the first file starts while the rest is planned. The
    /// batch owns its items until `awaitBatch` retires it. Nothing is
    /// published when a failure precedes the first plan; a later planning
    /// failure fails the loader (a partial submission can never complete),
    /// the batch is awaited here and the caller sees only the error.
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: ?*std.Progress.Node) !*Batch {
        try self.checkOpen();
        const batch = batch: {
            const batch = try Batch.create(self.allocator, self.io, .{
                .sequence = self.batch_count,
                .source_items = specs.len,
                .source_stats = if (self.load_profile.stats) |provider| provider.snapshot() else null,
            });
            errdefer batch.destroy();
            batch.items = try self.createItems(specs, &batch.diagnostics.logical_bytes, progress);
            break :batch batch;
        };
        Planner.publishFiles(&self.scheduler, self.io, batch, batch.items, self.plan_config, self.direct_io) catch |err| {
            if (batch.plans.items.len == 0) {
                self.destroyBatch(batch);
                return err;
            }
            // Planning failed after part of the batch was published: fail
            // the pipeline, seal and await this batch, and return the sticky
            // error instead of a batch the caller could not complete.
            self.pipeline.recordError(err);
            self.scheduler.seal(self.io, batch);
            self.batch_count += 1;
            batch.finishJobs(1);
            self.awaitBatch(batch) catch |sticky| return sticky;
            return err;
        };
        self.scheduler.seal(self.io, batch);
        self.batch_count += 1;
        // Every plan is visible: drop the publish sentinel. A batch without
        // jobs completes right here.
        batch.finishJobs(1);
        return batch;
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
    pub fn awaitBatch(self: *Loader, batch: *Batch) !void {
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
            for (batch.items) |*item| {
                const state = item.state.readyValue() orelse continue;
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
        self.pipeline.retireBatch(batch);
        self.logBatch(batch, done_at, load_error == null);
        self.destroyBatch(batch);
        if (load_error) |err| return err;
    }

    /// The front end awaited every batch before this; nothing is queued or
    /// in flight, so stopping the workers is a plain shutdown.
    pub fn destroy(self: *Loader) void {
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

        const allocator = self.allocator;
        allocator.free(self.allocated_bytes);
        allocator.destroy(self);
    }

    fn checkOpen(self: *Loader) !void {
        if (self.pipeline.errorValue()) |err| return err;
    }

    /// The calibrated block pool and the request sizing derived from it.
    const Sizing = struct {
        calibration: dma_calibration.Result,
        pool: host_memory.BlockPool,
        request_size: usize,
        maximum_blocks_per_job: usize,
        feasible_width: usize,
        retained_credits: usize,
        dma_stage_requests: usize,

        fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: BackendConfig,
        ) !Sizing {
            const calibration, const request_size, const maximum_blocks_per_job, var pool = pool: {
                var workspace = try host_memory.Workspace.init(allocator, io, platform, .{
                    .max_mapped_bytes = opts.max_host_bytes,
                });
                errdefer workspace.deinit();
                const calibration = try dma_calibration.calibrate(&workspace, platform, opts.dma);

                const request_size = try load_limits.effectiveSourceRequestSize(
                    opts.load_profile.read_chunk_size,
                    calibration.block_size,
                );
                const maximum_blocks_per_job = try load_limits.maximumCoalescedJobBlocks(
                    request_size,
                    calibration.block_size,
                );
                // The DMA stage of every device, kept mapped as the pool's growth floor.
                const dma_reserve = calibration.max_in_flight_per_device * platform.devices.len;
                // Grow the DMA stage reserve and the source working set of
                // the configured width before reads begin (mapping a slab
                // during a load cost 146 ms on one MI300X); calibration
                // arenas become the load's initial capacity.
                const pregrowth_started: std.Io.Timestamp = .now(io, .awake);
                const retained_before = workspace.mapped_bytes;
                try ensureLoadBlockReserve(&workspace, calibration.block_size, dma_reserve);
                try ensureSourceWorkingSet(
                    &workspace,
                    calibration.block_size,
                    maximum_blocks_per_job,
                    opts.read_parallelism,
                    dma_reserve,
                );
                const pregrown_bytes = workspace.mapped_bytes - retained_before;
                const pregrowth_ns: u64 = @intCast(@max(pregrowth_started.untilNow(io, .awake).nanoseconds, 0));
                load_log.debug("host workspace pregrown: retained={Bi:.2}, pregrown={Bi:.2}, pregrowth_ms={d:.3}", .{
                    workspace.mapped_bytes,
                    pregrown_bytes,
                    @as(f64, @floatFromInt(pregrowth_ns)) / std.time.ns_per_ms,
                });
                const pool = try host_memory.BlockPool.init(allocator, &workspace, calibration.block_size, dma_reserve);
                break :pool .{ calibration, request_size, maximum_blocks_per_job, pool };
            };
            errdefer pool.deinit();
            const feasible_width = try pool.potentialRequestWidth(maximum_blocks_per_job);
            if (feasible_width == 0) return error.DmaMappedBudgetExceeded;

            return .{
                .calibration = calibration,
                .pool = pool,
                .request_size = request_size,
                .maximum_blocks_per_job = maximum_blocks_per_job,
                .feasible_width = feasible_width,
                .retained_credits = try pool.retainedRequestWidth(maximum_blocks_per_job),
                .dma_stage_requests = dmaStageRequests(
                    calibration.max_in_flight_per_device,
                    platform.devices.len,
                    calibration.block_size,
                    request_size,
                ),
            };
        }
    };

    /// Creates the batch's items; on failure nothing stays allocated. An
    /// item's device state is created lazily by the first worker that
    /// reads for it.
    fn createItems(
        self: *Loader,
        specs: []const LoadSpec,
        logical_bytes: *usize,
        progress: ?*std.Progress.Node,
    ) ![]Item {
        const items = try self.allocator.alloc(Item, specs.len);
        errdefer self.allocator.free(items);
        for (specs, items) |spec, *item| {
            // An empty source has no transfer, so its output would never be
            // written; the front ends reject it too.
            if (spec.source.byteSize() == 0) return error.EmptyTensor;
            item.* = .{
                .source = spec.source,
                .source_slot = try self.sourceSlot(spec.source.file_uri),
                .shape = spec.shape,
                .sharding = spec.sharding,
                .output = spec.output,
                .progress = progress,
            };
            logical_bytes.* += spec.source.shape.byteSize();
        }
        return items;
    }

    /// Releases the items' device state, then the batch's memory.
    fn destroyBatch(self: *Loader, batch: *Batch) void {
        for (batch.items) |*item| item.deinit(self.allocator, self.platform.pjrt_api);
        batch.destroy();
    }

    fn sourceSlot(self: *Loader, uri: []const u8) !*SourceSlot {
        if (self.source_slots.get(uri)) |slot| return slot;
        const slot = try self.allocator.create(SourceSlot);
        errdefer self.allocator.destroy(slot);
        slot.* = .{ .uri = uri };
        try self.source_slots.putNoClobber(self.allocator, uri, slot);
        return slot;
    }

    fn workerMain(self: *Loader) void {
        var scratch = ReadRequest.Scratch.init(
            self.allocator,
            self.maximum_blocks_per_job,
            self.platform.devices.len,
        ) catch |err| {
            self.pipeline.recordError(err);
            return;
        };
        defer scratch.deinit();
        while (self.scheduler.waitForWork(self.io)) {
            if (self.pipeline.failed()) return;
            const credit_wait_started = awakeNs(self.io);
            if (!self.request_gate.acquire(self.io)) return;
            const credit_wait_ns = awakeNs(self.io) -| credit_wait_started;
            const claim = self.scheduler.claim(self.io) orelse {
                self.request_gate.release(self.io);
                continue;
            };
            _ = self.metrics.lifecycle_wait_ns.fetchAdd(credit_wait_ns, .monotonic);
            const request = ReadRequest.init(&self.pipeline, claim);
            // The scheduling sentinel keeps the batch, and with it the
            // claim's plan, alive through `run` and error reporting.
            defer request.finishScheduling();
            request.run(self, claim, &scratch) catch |err| self.pipeline.recordError(err);
        }
    }

    /// Stops the workers and the throttle watch; awaiting a group that never
    /// spawned returns at once, so this also cleans up a failed `create`.
    fn stopWorkers(self: *Loader) void {
        self.scheduler.stop(self.io);
        self.read_gate.close(self.io);
        self.request_gate.close(self.io);
        if (self.throttle) |*watch| watch.done.set(self.io);
        self.worker_group.await(self.io) catch {};
        self.throttle_group.await(self.io) catch {};
    }

    fn logBatch(self: *Loader, batch: *const Batch, done_at: std.Io.Timestamp, successful: bool) void {
        const diagnostics = &batch.diagnostics;
        const published_at = diagnostics.published_at orelse self.created_at;
        const sealed_at = diagnostics.sealed_at orelse published_at;
        load_log.debug("batch completed: batch={d}, successful={}, logical_bytes={Bi:.2}, published=+{d:.3}s, sealed=+{d:.3}s, done=+{d:.3}s, elapsed={d:.3}s, source_width={d}, request_size={Bi:.2}", .{
            diagnostics.sequence,
            successful,
            diagnostics.logical_bytes,
            secondsBetween(self.created_at, published_at),
            secondsBetween(self.created_at, sealed_at),
            secondsBetween(self.created_at, done_at),
            secondsBetween(published_at, done_at),
            self.width,
            self.plan_config.request_size,
        });

        const average_read_size = if (diagnostics.source_jobs == 0)
            0
        else
            diagnostics.source_bytes / diagnostics.source_jobs;
        const coalescing_ratio = if (diagnostics.source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(diagnostics.source_items)) /
                @as(f64, @floatFromInt(diagnostics.source_jobs));
        load_log.debug("batch planning: batch={d}, plans={d}, planning_elapsed={d:.3}s, planned_source_bytes={Bi:.2}, planned_source_jobs={d}, source_items={d}, planned_transfers={d}, planned_dma_submissions={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}", .{
            diagnostics.sequence,
            diagnostics.plans,
            @as(f64, @floatFromInt(diagnostics.planning_ns)) / std.time.ns_per_s,
            diagnostics.source_bytes,
            diagnostics.source_jobs,
            diagnostics.source_items,
            diagnostics.planned_transfers,
            diagnostics.planned_dma_submissions,
            coalescing_ratio,
            average_read_size,
        });
        if (self.load_profile.stats) |provider| {
            if (diagnostics.source_stats) |previous| {
                const delta = provider.snapshot().sub(previous);
                load_log.debug("batch source: batch={d}, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}", .{
                    diagnostics.sequence,
                    delta.physical_requests,
                    delta.physical_bytes,
                    delta.retries,
                    delta.throttles,
                });
            }
        }
    }

    fn logSummary(self: *Loader) void {
        const reads = self.metrics.read_operations.load(.acquire);
        load_log.debug("loader summary: batches={d}, successful={}, read_bytes={Bi:.2}, elapsed={d:.3}s, reads={d}, source_width={d}, request_size={Bi:.2}, pinned_high_water={Bi:.2}, pinned_mapped={Bi:.2}", .{
            self.batch_count,
            !self.pipeline.failed(),
            self.metrics.read_bytes.load(.acquire),
            secondsBetween(self.created_at, .now(self.io, .awake)),
            reads,
            self.width,
            self.plan_config.request_size,
            self.pool.high_water * self.pool.block_size,
            self.pool.workspace.mapped_bytes,
        });
        load_log.debug("loader waits: credit_wait_ms_per_read={d:.3}, block_wait_ms_per_read={d:.3}, read_ms_per_read={d:.3}, dma_stage_ms_per_read={d:.3}, tensor_init_ms_per_read={d:.3}", .{
            millisecondsPer(self.metrics.lifecycle_wait_ns.load(.acquire), reads),
            millisecondsPer(self.metrics.block_wait_ns.load(.acquire), reads),
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
/// is planned), the per-tensor items it writes, and every request, block and
/// event context created while loading it. `remaining` counts completion
/// units: one per published job plus a publish sentinel held until the
/// submission is sealed. A job's unit is released exactly once, by whichever
/// of these happens: its request's last reference drops (final DMA callback
/// or abandonment), or `Scheduler.fail` retires it unclaimed. The
/// batch is done when `remaining` reaches zero; the awaiting task then
/// retires it, so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    /// One file's jobs in claim order with their transfer records and every
    /// context the jobs can need, allocated by the planner: one request per
    /// job, the job's blocks and one event per planned DMA submission,
    /// handed out in submission order. Contexts hold the plan's address, so
    /// plans are heap objects freed with the batch.
    const Plan = struct {
        /// A claimable job: its source range and its slices of the plan's
        /// transfers and block contexts. Its request slot is its index.
        const Job = struct {
            /// The read: `len` bytes from `file_offset`, of which the first
            /// `minimum_len` hold tensor data and must exist. The rest is
            /// alignment padding the end of the file may cut.
            file_offset: u64,
            len: usize,
            minimum_len: usize,
            transfer_start: usize,
            transfer_len: usize,
            block_start: usize,
            block_len: usize,
        };

        const Transfer = struct {
            item: *Item,
            block_index: usize,
            block_offset: usize,
            writer_mask: u64,
            destination_offset: usize,
            len: usize,
        };

        allocator: std.mem.Allocator,
        /// The file every job of the plan reads.
        source_slot: *SourceSlot,
        jobs: []Job,
        transfers: []Transfer,
        requests: []ReadRequest,
        blocks: []Pipeline.BlockContext,
        events: []Pipeline.EventContext,
        /// Event slots handed out so far, one `fetchAdd` per submission from
        /// any device's pump.
        events_used: std.atomic.Value(usize) = .init(0),
        source_bytes: u64,
        planning_ns: u64 = 0,
        /// Next job to claim; owned by the scheduler mutex.
        cursor: usize = 0,

        /// Takes ownership of transfers on success. All callback storage is
        /// allocated here, before publication makes the plan visible to workers.
        fn create(
            allocator: std.mem.Allocator,
            source_slot: *SourceSlot,
            job_count: usize,
            block_count: usize,
            transfers: []Transfer,
            source_bytes: u64,
        ) !*Plan {
            var dma_submissions: usize = 0;
            for (transfers) |transfer| dma_submissions += @popCount(transfer.writer_mask);
            const jobs = try allocator.alloc(Job, job_count);
            errdefer allocator.free(jobs);
            const requests = try allocator.alloc(ReadRequest, job_count);
            errdefer allocator.free(requests);
            @memset(requests, ReadRequest.idle);
            const blocks = try allocator.alloc(Pipeline.BlockContext, block_count);
            errdefer allocator.free(blocks);
            const events = try allocator.alloc(Pipeline.EventContext, dma_submissions);
            errdefer allocator.free(events);
            const self = try allocator.create(Plan);
            self.* = .{
                .allocator = allocator,
                .source_slot = source_slot,
                .jobs = jobs,
                .transfers = transfers,
                .requests = requests,
                .blocks = blocks,
                .events = events,
                .source_bytes = source_bytes,
            };
            return self;
        }

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
        /// Aggregate source statistics at publish; the completion log reports
        /// the delta against them (loader-wide while batches overlap).
        source_stats: ?VFS.ReadStats = null,
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
    /// Owned by the batch; their device state by the loader, released
    /// before `destroy`.
    items: []Item = &.{},
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
    fn appendPlanAssumeCapacity(self: *Batch, plan: *Plan) void {
        self.plans.appendAssumeCapacity(plan);
        _ = self.remaining.fetchAdd(plan.jobs.len, .acq_rel);
    }

    /// Scheduler mutex. The next unclaimed job in plan order, or null when
    /// every published plan is exhausted.
    fn claimJob(self: *Batch) ?Scheduler.Claim {
        while (self.plan_cursor < self.plans.items.len) : (self.plan_cursor += 1) {
            const plan = self.plans.items[self.plan_cursor];
            if (plan.cursor == plan.jobs.len) continue;
            const index = plan.cursor;
            plan.cursor += 1;
            return .{ .batch = self, .plan = plan, .index = index };
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

    /// Frees the items' memory and the plans with their contexts, which
    /// `Pipeline.retireBatch` must already have retired.
    fn destroy(self: *Batch) void {
        self.allocator.free(self.items);
        for (self.plans.items) |plan| plan.destroy();
        self.plans.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// After `done`: every item was touched and every target received its
    /// last transfer.
    fn fullySubmitted(self: *const Batch) bool {
        for (self.items) |*item| {
            const state = item.state.readyValue() orelse return false;
            for (state.targets) |target| {
                if (!target.fullySubmitted()) return false;
            }
        }
        return true;
    }
};

const Item = struct {
    const InitContext = struct { item: *const Item, direct: *Loader };

    fn initTransfer(ctx: InitContext) !TensorTransfer {
        return TensorTransfer.init(ctx.direct, ctx.item);
    }

    source: *const safetensors.Tensor,
    source_slot: *SourceSlot,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
    progress: ?*std.Progress.Node = null,
    state: LazyOnce(TensorTransfer, InitContext, initTransfer) = .{},

    fn ensureState(self: *Item, direct: *Loader) !*TensorTransfer {
        return self.state.ensure(direct.io, .{ .item = self, .direct = direct });
    }

    fn deinit(self: *Item, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        if (self.state.readyValue()) |state| state.deinit(allocator, api);
    }
};

const SourceSlot = struct {
    const OpenContext = struct { io: std.Io, uri: []const u8 };

    fn openFile(ctx: OpenContext) !std.Io.File {
        return std.Io.Dir.openFile(.cwd(), ctx.io, ctx.uri, .{ .mode = .read_only });
    }

    uri: []const u8,
    file: LazyOnce(std.Io.File, OpenContext, openFile) = .{},

    fn ensure(self: *SourceSlot, io: std.Io) !std.Io.File {
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

    targets: []Target,
    completed_read_bytes: std.atomic.Value(usize) = .init(0),
    progress: ?std.Progress.Node = null,

    /// Creates the item's device buffers and transfer managers and writes
    /// the output shell. Runs once per tensor (`LazyOnce`), on the worker
    /// that first reads for it, which also counts the allocation.
    fn init(direct: *Loader, item: *const Item) !TensorTransfer {
        const allocator = direct.allocator;
        const platform = direct.platform;
        const packed_shape = item.shape.packedShape();
        const packed_placement = try item.sharding.placement(packed_shape);
        const ordered_devices = item.sharding.devicesInCanonicalOrder();
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
        for (targets) |target| {
            _ = direct.allocated_bytes[target.device_index].fetchAdd(target.total, .monotonic);
        }

        item.output.* = .fromPjrtBuffers(platform, item.shape, item.sharding, pjrt_buffers.constSlice());
        const progress = if (item.progress) |parent|
            parent.start(
                item.source.name,
                item.shape.byteSize() / 1024 + @intFromBool(item.shape.byteSize() % 1024 != 0),
            )
        else
            null;

        return .{
            .targets = targets,
            .progress = progress,
        };
    }

    fn deinit(self: *TensorTransfer, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        if (self.progress) |*progress| progress.end();
        for (self.targets) |target| target.manager.deinit(api);
        allocator.free(self.targets);
    }

    fn recordReadProgress(self: *TensorTransfer, bytes: usize) void {
        const completed = self.completed_read_bytes.fetchAdd(bytes, .acq_rel) + bytes;
        if (self.progress) |*progress| {
            progress.setCompletedItems(completed / 1024 + @intFromBool(completed % 1024 != 0));
        }
    }
};

/// Builds immutable per-file source jobs and transfer records. Fairness is
/// decided here once; the scheduler only publishes and claims those jobs.
/// For DeepSeek-V4-Flash (148.65 GiB, 69,187 tensors), tensor-local reads made
/// about 69,445 source calls despite a 16 MiB request limit. Coalescing reduced
/// that to 9,524. A rigid grid still produced 78,665 DMA pieces; tensor-safe
/// cuts kept the same read count with 69,572 pieces. Per-tensor device buffers
/// remain intentional: reducing submissions below roughly tensors x destinations would require
/// packed device allocations or device-side scattering, not another read cut.
const Planner = struct {
    /// What every plan of a loader is cut by.
    const Config = struct {
        device_count: usize,
        block_size: usize,
        request_size: usize,
        /// Every read of a file read directly starts and ends at a multiple
        /// of this; 0 reads exact tensor ranges.
        alignment: usize,
    };

    /// One item's placement, expanded once per plan.
    const TensorPlan = struct {
        item: *Item,
        dispatch_spans: DispatchSpans,
        device_indices: []usize,
        total: usize,
    };

    /// Plans `items` one file at a time and publishes each plan as soon as
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
    /// zero. Ask the VFS actually owning the handle, not one remembered by
    /// the profile, both to avoid that cost and to avoid indexing another
    /// VFS's handle table.
    fn publishFiles(
        scheduler: *Scheduler,
        io: std.Io,
        batch: *Batch,
        items: []Item,
        config: Config,
        policy: VFS.DirectIo,
    ) !void {
        const order = try sortedItemOrder(scheduler.allocator, items);
        defer scheduler.allocator.free(order);
        const vfs = if (config.alignment != 0) VFS.fromIo(io) else null;
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(items, order, file_start);
            const planning_started: std.Io.Timestamp = .now(io, .awake);
            var file_config = config;
            if (vfs) |v| {
                const file = try items[order[file_start]].source_slot.ensure(io);
                if (!v.useDirectIo(file, policy)) file_config.alignment = 0;
            }
            const plan = try preparePlan(scheduler.allocator, items, order[file_start..file_end], file_config);
            plan.planning_ns = @intCast(@max(planning_started.untilNow(io, .awake).nanoseconds, 0));
            scheduler.publish(io, batch, plan) catch |err| {
                plan.destroy();
                return err;
            };
            file_start = file_end;
        }
    }

    /// Plans one file: `order` indexes `items` sorted by offset then size,
    /// all on the same file. Runs of touching ranges are cut into the
    /// minimum number of jobs at tensor-safe boundaries, in a fair order
    /// across the destination devices (the planning order when there is one
    /// device); no job depends on another, since every DMA piece is
    /// submitted as soon as its block is read. The plan also carries the
    /// contexts its jobs need: one request per job, one block per job block,
    /// one event per DMA submission (a transfer's writer count). With an
    /// alignment, a job's read is widened to aligned bounds around its
    /// tensor range (`maximumJobLen` leaves room for it); the transfers
    /// address the widened read.
    fn preparePlan(
        allocator: std.mem.Allocator,
        items: []Item,
        order: []const usize,
        config: Config,
    ) !*Batch.Plan {
        const device_count = config.device_count;
        const block_size = config.block_size;
        const alignment = config.alignment;
        const maximum_job_len = try maximumJobLen(config);
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
            const item = &items[item_index];
            std.debug.assert(std.mem.eql(u8, item.source.file_uri, items[order[0]].source.file_uri));
            const packed_shape = item.shape.packedShape();
            plan.* = .{
                .item = item,
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
        var jobs_list: std.ArrayList(Batch.Plan.Job) = .empty;
        defer jobs_list.deinit(allocator);
        var transfers_list: std.ArrayList(Batch.Plan.Transfer) = .empty;
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

            var job_start = first_offset;
            var candidate_start = run_cursor;
            var boundary_cursor: usize = 0;
            const run_len = run_end - first_offset;
            const maximum_job_len_u64: u64 = @intCast(maximum_job_len);
            var jobs_remaining: usize = @intCast(run_len / maximum_job_len_u64 +
                @intFromBool(run_len % maximum_job_len_u64 != 0));
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
                const read_start = if (alignment == 0) job_start else std.mem.alignBackward(u64, job_start, alignment);
                const read_end = if (alignment == 0) job_end else std.mem.alignBackward(
                    u64,
                    std.math.add(u64, job_end, alignment - 1) catch return error.InvalidLoaderJob,
                    alignment,
                );
                const read_len: usize = @intCast(read_end - read_start);
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
                    const item = &items[item_index];
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
                        &tensor_plans[position],
                        @intCast(intersection_start - item.source.offset),
                        @intCast(intersection_end - intersection_start),
                        read_start,
                        block_size,
                        row,
                    );
                }
                std.debug.assert(transfers_list.items.len > transfer_start);
                const block_len = read_len / block_size + @intFromBool(read_len % block_size != 0);
                try jobs_list.append(allocator, .{
                    .file_offset = read_start,
                    .len = read_len,
                    .minimum_len = @intCast(job_end - read_start),
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
        const plan = plan: {
            const transfers = try transfers_list.toOwnedSlice(allocator);
            errdefer allocator.free(transfers);
            break :plan try Batch.Plan.create(
                allocator,
                items[order[0]].source_slot,
                planning_jobs.len,
                block_total,
                transfers,
                source_bytes,
            );
        };
        errdefer plan.destroy();
        if (device_count == 1) {
            @memcpy(plan.jobs, planning_jobs);
        } else {
            const fair_order = try fairOrder(allocator, planning_jobs.len, physical_list.items, queues);
            defer allocator.free(fair_order);
            for (plan.jobs, fair_order) |*job, planning_index| job.* = planning_jobs[planning_index];
        }
        return plan;
    }

    /// The longest tensor range one job may cover: the request size within
    /// the scatter limit, less the two alignment units its widened read
    /// can add, so a widened job still fits `maximumCoalescedJobBlocks`.
    fn maximumJobLen(config: Config) !usize {
        const scatter_limit = config.block_size *| load_limits.max_positional_iovecs;
        const unpadded = @min(config.request_size, scatter_limit);
        if (unpadded == 0) return error.InvalidLoaderJob;
        if (config.alignment == 0) return unpadded;
        if (!std.math.isPowerOfTwo(config.alignment) or 2 * config.alignment >= unpadded) return error.InvalidLoaderJob;
        return unpadded - 2 * config.alignment;
    }

    /// Item indices sorted by file URI, offset, size and index: the planner's
    /// input, taken one file group at a time (`fileGroupEnd`).
    fn sortedItemOrder(allocator: std.mem.Allocator, items: []const Item) ![]usize {
        const order = try allocator.alloc(usize, items.len);
        for (order, 0..) |*index, i| index.* = i;
        const SortContext = struct {
            items: []const Item,

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
    fn fileGroupEnd(items: []const Item, order: []const usize, start: usize) usize {
        const uri = items[order[start]].source.file_uri;
        var end = start + 1;
        while (end < order.len and std.mem.eql(u8, uri, items[order[end]].source.file_uri)) : (end += 1) {}
        return end;
    }

    /// Appends the transfers of `len` bytes of `tensor` from `tensor_offset`
    /// read by the job at `job_file_offset`, merging with the previous
    /// transfer where contiguous, and charges the bytes to each destination
    /// device in `physical_bytes`.
    fn appendTransfers(
        allocator: std.mem.Allocator,
        output: *std.ArrayList(Batch.Plan.Transfer),
        transfer_start: usize,
        tensor: *const TensorPlan,
        tensor_offset: usize,
        len: usize,
        job_file_offset: u64,
        block_size: usize,
        physical_bytes: []usize,
    ) !void {
        const item = tensor.item;
        const spans = tensor.dispatch_spans;
        const piece_end = tensor_offset + len;
        var cursor = tensor_offset;
        var span_index = spans.spanIndexAt(cursor) orelse return error.InvalidLoaderJob;
        while (cursor < piece_end) {
            const span = spans.spans[span_index];
            const absolute = item.source.offset + @as(u64, @intCast(cursor));
            if (absolute < job_file_offset) return error.InvalidLoaderJob;
            const source_relative = std.math.cast(usize, absolute - job_file_offset) orelse
                return error.InvalidLoaderJob;
            const block_index = source_relative / block_size;
            const block_offset = source_relative % block_size;
            const take = @min(
                @min(piece_end - cursor, span.end - cursor),
                block_size - block_offset,
            );
            const writer_mask = span.writer_mask;
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
                if (writer_index >= tensor.device_indices.len) return error.InvalidLoaderJob;
                physical_bytes[tensor.device_indices[writer_index]] += take;
            }
            cursor += take;
            if (cursor == span.end) span_index += 1;
        }
    }

    /// The claim order of `job_count` jobs: each turn goes to the device
    /// with the fewest scheduled bytes that still has a queued job, ties
    /// rotating, so every device's DMA engine is fed early.
    fn fairOrder(
        allocator: std.mem.Allocator,
        job_count: usize,
        physical_bytes: []const usize,
        queues: []const std.ArrayListUnmanaged(usize),
    ) ![]usize {
        const device_count = queues.len;
        if (device_count == 0 or device_count > 64) return error.DmaDeviceMismatch;
        if (physical_bytes.len != job_count * device_count) return error.InvalidLoaderJob;

        const order = try allocator.alloc(usize, job_count);
        errdefer allocator.free(order);
        const cursors = try allocator.alloc(usize, device_count);
        defer allocator.free(cursors);
        @memset(cursors, 0);
        const scheduled = try allocator.alloc(u64, device_count);
        defer allocator.free(scheduled);
        @memset(scheduled, 0);
        const claimed = try allocator.alloc(bool, job_count);
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
const Scheduler = struct {
    /// A claimed job: its position in the plan and batch that own it. The
    /// claim holds one of the batch's completion units until the request
    /// releases it.
    const Claim = struct {
        batch: *Batch,
        plan: *Batch.Plan,
        index: usize,

        fn job(self: Claim) Batch.Plan.Job {
            return self.plan.jobs[self.index];
        }

        /// The job's request slot, one per job in the plan.
        fn request(self: Claim) *ReadRequest {
            return &self.plan.requests[self.index];
        }

        fn transfers(self: Claim) []const Batch.Plan.Transfer {
            const claimed = self.job();
            return self.plan.transfers[claimed.transfer_start..][0..claimed.transfer_len];
        }

        fn blocks(self: Claim) []Pipeline.BlockContext {
            const claimed = self.job();
            return self.plan.blocks[claimed.block_start..][0..claimed.block_len];
        }
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

    fn init(allocator: std.mem.Allocator) Scheduler {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Scheduler) void {
        // Every batch was awaited (and so claimed out or retired) first.
        std.debug.assert(self.head == self.queue.items.len);
        self.queue.deinit(self.allocator);
        self.* = undefined;
    }

    /// Publishes one plan of an open batch behind every earlier plan and
    /// batch. The batch joins the queue with its first plan that has jobs;
    /// a plan without jobs only counts in the diagnostics.
    fn publish(self: *Scheduler, io: std.Io, batch: *Batch, plan: *Batch.Plan) !void {
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
        diagnostics.planning_ns += plan.planning_ns;
        diagnostics.source_bytes += plan.source_bytes;
        diagnostics.source_jobs += job_count;
        diagnostics.planned_transfers += plan.transfers.len;
        diagnostics.planned_dma_submissions += plan.events.len;
        batch.appendPlanAssumeCapacity(plan);
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
    fn seal(self: *Scheduler, io: std.Io, batch: *Batch) void {
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
    fn popHead(self: *Scheduler) void {
        self.queue.items[self.head].queued = false;
        self.head += 1;
        if (self.head == self.queue.items.len) {
            self.queue.clearRetainingCapacity();
            self.head = 0;
        }
    }

    fn stop(self: *Scheduler, io: std.Io) void {
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
    fn fail(self: *Scheduler, io: std.Io) void {
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
    fn claim(self: *Scheduler, io: std.Io) ?Claim {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.head == self.queue.items.len) return null;
        const batch = self.queue.items[self.head];
        const claimed = batch.claimJob() orelse {
            std.debug.assert(!batch.sealed);
            return null;
        };
        self.unclaimed_total -= 1;
        if (batch.sealed and batch.exhausted()) self.popHead();
        return claimed;
    }

    fn waitForWork(self: *Scheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.stopping and self.unclaimed_total == 0) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }

    fn remainingJobs(self: *Scheduler, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.unclaimed_total;
    }
};

/// A claimed source job and its completion state. The plan owns its storage;
/// the worker holds a scheduling reference until all blocks have been handed off.
const ReadRequest = struct {
    const Scratch = struct {
        allocator: std.mem.Allocator,
        leased: []host_memory.BlockPool.Block,
        references: []usize,
        iovecs: [][]u8,
        queue_counts: []usize,

        fn init(
            allocator: std.mem.Allocator,
            maximum_blocks: usize,
            device_count: usize,
        ) !Scratch {
            const leased = try allocator.alloc(host_memory.BlockPool.Block, maximum_blocks);
            errdefer allocator.free(leased);
            const references = try allocator.alloc(usize, maximum_blocks);
            errdefer allocator.free(references);
            const iovecs = try allocator.alloc([]u8, maximum_blocks);
            errdefer allocator.free(iovecs);
            const queue_counts = try allocator.alloc(usize, device_count);
            return .{
                .allocator = allocator,
                .leased = leased,
                .references = references,
                .iovecs = iovecs,
                .queue_counts = queue_counts,
            };
        }

        fn deinit(self: *Scratch) void {
            self.allocator.free(self.queue_counts);
            self.allocator.free(self.iovecs);
            self.allocator.free(self.references);
            self.allocator.free(self.leased);
            self.* = undefined;
        }
    };

    pipeline: *Pipeline,
    batch: *Batch,
    plan: *Batch.Plan,
    /// The job's block contexts; its worker registers them in order.
    blocks: []Pipeline.BlockContext,
    blocks_registered: usize = 0,
    pending: std.atomic.Value(usize) = .init(1), // scheduling sentinel
    completed: std.atomic.Value(bool) = .init(false),
    /// Awake-clock nanoseconds of the enqueue; 0 until then.
    enqueued_ns: u64 = 0,

    /// The slot of a job that was never claimed: nothing pending, so the
    /// retirement checks hold.
    const idle: ReadRequest = .{
        .pipeline = undefined,
        .batch = undefined,
        .plan = undefined,
        .blocks = &.{},
        .pending = .init(0),
        .completed = .init(true),
    };

    /// Takes the claimed job's request context. The claim holds the batch's
    /// completion unit; the request's final reference drop releases it.
    fn init(pipeline: *Pipeline, claim: Scheduler.Claim) *ReadRequest {
        const self = claim.request();
        self.* = .{
            .pipeline = pipeline,
            .batch = claim.batch,
            .plan = claim.plan,
            .blocks = claim.blocks(),
        };
        return self;
    }

    fn run(self: *ReadRequest, loader: *Loader, claim: Scheduler.Claim, scratch: *Scratch) !void {
        const pipeline = self.pipeline;
        const io = pipeline.io;
        if (pipeline.failed()) return;

        const job = claim.job();
        const transfers = claim.transfers();
        const file = try claim.plan.source_slot.ensure(io);
        const block_count = job.len / pipeline.block_size +
            @intFromBool(job.len % pipeline.block_size != 0);
        if (block_count == 0) return;

        std.debug.assert(block_count <= scratch.leased.len);
        std.debug.assert(block_count == self.blocks.len);
        const leased = scratch.leased[0..block_count];
        const references = scratch.references[0..block_count];
        @memset(references, 0);
        const queue_counts = scratch.queue_counts;
        @memset(queue_counts, 0);

        // The planner cut the transfers to this job's blocks and the tensor's
        // writers are the devices it planned for (both come from the
        // sharding's canonical device order), so these are invariants.
        for (transfers) |transfer| {
            const init_started = awakeNs(io);
            const tensor = try transfer.item.ensureState(loader);
            _ = pipeline.metrics.tensor_init_ns.fetchAdd(awakeNs(io) -| init_started, .monotonic);
            std.debug.assert(transfer.block_index < block_count and
                transfer.block_offset < pipeline.block_size and
                transfer.len <= pipeline.block_size - transfer.block_offset);
            references[transfer.block_index] += @popCount(transfer.writer_mask);
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                std.debug.assert(writer_index < tensor.targets.len);
                queue_counts[tensor.targets[writer_index].device_index] += 1;
            }
        }
        // Every block of a job is covered by a transfer: a block without a
        // reference would never be released.
        for (references) |refs| std.debug.assert(refs != 0);
        {
            const block_wait_started = awakeNs(io);
            try pipeline.pool.acquireMany(io, leased);
            errdefer pipeline.pool.releaseMany(io, leased);
            _ = pipeline.metrics.block_wait_ns.fetchAdd(awakeNs(io) -| block_wait_started, .monotonic);
            if (pipeline.errorValue()) |err| return err;

            const iovecs = scratch.iovecs[0..block_count];
            for (iovecs, leased, 0..) |*iovec, block, block_index| {
                const consumed = block_index * pipeline.block_size;
                iovec.* = block[0..@min(pipeline.block_size, job.len - consumed)];
            }

            {
                if (!pipeline.read_gate.acquire(io)) return error.LoaderShuttingDown;
                defer pipeline.read_gate.release(io);
                const read_started = awakeNs(io);
                const read_result = safetensors.readFilePositionalAllV(
                    io,
                    file,
                    iovecs,
                    job.file_offset,
                    job.minimum_len,
                );
                _ = pipeline.metrics.read_ns.fetchAdd(awakeNs(io) -| read_started, .monotonic);
                const bytes_read = try read_result;
                _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
                _ = pipeline.metrics.read_bytes.fetchAdd(@intCast(bytes_read), .monotonic);
                for (transfers) |transfer| transfer.item.state.value.recordReadProgress(transfer.len);
            }
            if (pipeline.errorValue()) |err| return err;

            // From here the request's block contexts release the leases.
            for (leased, references) |lease, refs| _ = self.registerBlock(lease, refs);
        }
        self.enqueued_ns = awakeNs(io);
        pipeline.enqueueBlocks(transfers, self.blocks, queue_counts) catch |err| {
            self.enqueued_ns = 0;
            // Nothing was queued. The worker still holds the scheduling reference.
            for (transfers) |transfer| {
                Pipeline.abandonSubmissions(&self.blocks[transfer.block_index], @popCount(transfer.writer_mask));
            }
            return err;
        };
    }

    /// Takes the request's next block context for a leased block. Only the
    /// request's worker touches its slots, in order.
    fn registerBlock(self: *ReadRequest, data: host_memory.BlockPool.Block, references: usize) *Pipeline.BlockContext {
        const block = &self.blocks[self.blocks_registered];
        block.* = .{
            .pipeline = self.pipeline,
            .request = self,
            .lease = .init(self.pipeline.pool, self.pipeline.io, data, references),
        };
        self.blocks_registered += 1;
        _ = self.pending.fetchAdd(1, .acq_rel);
        return block;
    }

    /// Final worker access, after any error was recorded on the pipeline.
    fn finishScheduling(self: *ReadRequest) void {
        self.release();
    }

    /// Releases a block or the scheduling reference. The final release may
    /// let the awaiting task free this request and everything its batch owns.
    fn release(self: *ReadRequest) void {
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

const Pipeline = struct {
    const BlockContext = struct {
        pipeline: *Pipeline,
        request: *ReadRequest,
        lease: host_memory.BlockPool.Lease,

        fn complete(self: *BlockContext) void {
            if (self.lease.complete()) self.request.release();
        }
    };

    const ReadyTransfer = struct {
        target: *TensorTransfer.Target,
        block: *BlockContext,
        source_offset: usize,
        destination_offset: usize,
        len: usize,
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
        ready_entries: usize = 0,
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
        pipeline: *Pipeline,
        block: *BlockContext,
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
    pool: *host_memory.BlockPool,
    read_gate: *RequestGate,
    request_gate: *RequestGate,
    block_size: usize,
    metrics: *Metrics,
    scheduler: *Scheduler,
    first_error: std.atomic.Value(u16) = .init(0),
    /// One per device. Where several are locked at once (`enqueueBlocks`,
    /// `retireBatch`) they are taken in device order; a pump, a completion
    /// and `abortReady` hold one at a time.
    pumps: []DevicePump,
    /// Per-device in-flight budget: the calibrated depth in blocks, in bytes.
    dma_budget_bytes: usize,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *host_memory.BlockPool,
        read_gate: *RequestGate,
        request_gate: *RequestGate,
        block_size: usize,
        metrics: *Metrics,
        scheduler: *Scheduler,
        dma_budget_bytes: usize,
    ) !Pipeline {
        std.debug.assert(platform.devices.len <= 64);
        std.debug.assert(dma_budget_bytes > 0);
        const pumps = try allocator.alloc(DevicePump, platform.devices.len);
        errdefer allocator.free(pumps);
        @memset(pumps, .{});
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .pool = pool,
            .read_gate = read_gate,
            .request_gate = request_gate,
            .block_size = block_size,
            .metrics = metrics,
            .scheduler = scheduler,
            .pumps = pumps,
            .dma_budget_bytes = dma_budget_bytes,
        };
    }

    fn deinit(self: *Pipeline) void {
        // Every batch was retired: no DMA event, queued transfer or request
        // context may outlive its batch.
        std.debug.assert(self.request_gate.inUse(self.io) == 0);
        for (self.pumps) |*device_pump| {
            std.debug.assert(device_pump.active_pieces == 0);
            std.debug.assert(device_pump.ready_entries == 0);
            std.debug.assert(device_pump.retired == null);
            device_pump.queue.deinit(self.allocator);
        }
        self.allocator.free(self.pumps);
    }

    fn lockAllPumps(self: *Pipeline) void {
        for (self.pumps) |*device_pump| device_pump.mutex.lockUncancelable(self.io);
    }

    fn unlockAllPumps(self: *Pipeline) void {
        var index = self.pumps.len;
        while (index > 0) {
            index -= 1;
            self.pumps[index].mutex.unlock(self.io);
        }
    }

    fn failed(self: *const Pipeline) bool {
        return self.first_error.load(.acquire) != 0;
    }

    fn errorValue(self: *const Pipeline) ?anyerror {
        const value = self.first_error.load(.acquire);
        return if (value == 0) null else @errorFromInt(value);
    }

    fn recordError(self: *Pipeline, err: anyerror) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            self.scheduler.fail(self.io);
            self.pool.close(self.io);
            self.read_gate.close(self.io);
            self.request_gate.close(self.io);
            self.abortReady();
        }
    }

    /// Retires the contexts of a done batch: destroys the PJRT events a pump
    /// has not destroyed yet (from the awaiting task after their callbacks
    /// fired, exactly as `pjrt.Event.await` does), unlinks the batch's
    /// contexts from `retired` and checks that every request and block
    /// completed. Runs with every pump locked so an `abortReady` still
    /// iterating queued entries or a pump draining its `retired` cannot race
    /// the free that follows.
    fn retireBatch(self: *Pipeline, batch: *Batch) void {
        std.debug.assert(batch.done.isSet());
        self.lockAllPumps();
        defer self.unlockAllPumps();
        if (builtin.mode == .Debug) batch.freeing = true;
        for (batch.plans.items) |plan| {
            for (plan.events[0..plan.events_used.load(.acquire)]) |*ctx| ctx.destroyEvent();
            for (plan.requests) |*request| {
                std.debug.assert(request.completed.load(.acquire));
                for (request.blocks[0..request.blocks_registered]) |*block| {
                    std.debug.assert(block.lease.remaining.load(.acquire) == 0);
                }
            }
        }
        // Only this batch's contexts have a destroyed event while still
        // linked: a pump unlinks what it destroys.
        for (self.pumps) |*device_pump| {
            var link = &device_pump.retired;
            while (link.*) |ctx| {
                if (ctx.pjrt_event == null) link.* = ctx.next_retired else link = &ctx.next_retired;
            }
        }
    }

    /// Hands a context whose callback fired to its device's next pump. Under
    /// that pump's mutex, so the batch's retirement sees it before the batch
    /// is freed.
    fn retireEvent(self: *Pipeline, ctx: *EventContext) void {
        const device_pump = &self.pumps[ctx.device_index];
        device_pump.mutex.lockUncancelable(self.io);
        ctx.next_retired = device_pump.retired;
        device_pump.retired = ctx;
        device_pump.mutex.unlock(self.io);
    }

    /// Queue the whole source job before pumping. The former per-piece path
    /// paid roughly 69k-79k mutex/pump trips per DeepSeek load; coalesced reads
    /// alone did not remove that cost. Reserve every destination first so
    /// allocation failure cannot publish a subset of the block references.
    fn enqueueBlocks(
        self: *Pipeline,
        transfers: []const Batch.Plan.Transfer,
        blocks: []BlockContext,
        queue_counts: []const usize,
    ) !void {
        std.debug.assert(queue_counts.len == self.pumps.len);
        // Every pump is held while the request's transfers land, and every
        // destination is reserved before any queue changes, so the request
        // is either fully queued or not queued at all: the caller abandons
        // all of its submissions on failure.
        self.lockAllPumps();
        errdefer self.unlockAllPumps();
        for (self.pumps, queue_counts) |*device_pump, count| {
            try device_pump.queue.ensureUnusedCapacity(self.allocator, count);
        }
        for (transfers) |transfer| {
            const block = &blocks[transfer.block_index];
            const tensor = &transfer.item.state.value;
            var mask = transfer.writer_mask;
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const target = &tensor.targets[writer_index];
                const device_pump = &self.pumps[target.device_index];
                device_pump.queue.pushBackAssumeCapacity(.{
                    .target = target,
                    .block = block,
                    .source_offset = transfer.block_offset,
                    .destination_offset = transfer.destination_offset,
                    .len = transfer.len,
                });
                device_pump.ready_entries += 1;
            }
        }
        self.unlockAllPumps();
        for (queue_counts, 0..) |count, device_index| {
            if (count != 0) self.requestPump(device_index);
        }
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

    /// Runs the device's pump on this thread unless one is already running.
    fn requestPump(self: *Pipeline, device_index: usize) void {
        const device_pump = &self.pumps[device_index];
        device_pump.mutex.lockUncancelable(self.io);
        if (device_pump.pumping or self.failed()) {
            device_pump.mutex.unlock(self.io);
            return;
        }
        device_pump.pumping = true;
        device_pump.mutex.unlock(self.io);
        self.pump(device_index);
    }

    /// Submits from the device's queue, in arrival order, until the device
    /// has no room or nothing is queued. Any queued transfer can go: nothing
    /// waits for another piece. Exactly one pump runs per device at a time,
    /// see `DevicePump`.
    fn pump(self: *Pipeline, device_index: usize) void {
        const device_pump = &self.pumps[device_index];
        while (true) {
            var selected: ?ReadyTransfer = null;
            device_pump.mutex.lockUncancelable(self.io);
            device_pump.destroyRetired();
            if (!self.failed()) {
                if (!device_pump.hasRoom(self.dma_budget_bytes)) {
                    _ = self.metrics.pump_stops_full.fetchAdd(1, .monotonic);
                } else if (device_pump.queue.popFront()) |transfer| {
                    device_pump.active_bytes += transfer.len;
                    device_pump.active_pieces += 1;
                    device_pump.ready_entries -= 1;
                    selected = transfer;
                } else {
                    _ = self.metrics.pump_stops_empty.fetchAdd(1, .monotonic);
                }
            }
            if (selected == null) device_pump.pumping = false;
            device_pump.mutex.unlock(self.io);
            if (selected) |transfer| self.submitOne(transfer) else return;
        }
    }

    /// MEMORY-ORDER RULE: `transfer.target` and `transfer.block` belong to a
    /// batch, and the block's final completion may complete that batch, after
    /// which the awaiting task frees it. Every path therefore copies what it
    /// needs into locals, retires the DMA slot with `eventCompleted`, and
    /// calls `block.complete()` last.
    fn submitOne(self: *Pipeline, transfer: ReadyTransfer) void {
        const device_index = transfer.target.device_index;
        const len = transfer.len;
        const block = transfer.block;
        self.submitTransfer(transfer) catch |err| {
            self.recordError(err);
            self.eventCompleted(device_index, len);
            block.complete();
        };
    }

    fn submitTransfer(self: *Pipeline, transfer: ReadyTransfer) !void {
        const api = self.platform.pjrt_api;
        const target = transfer.target;
        const is_last = target.nextIsLast(transfer.len);
        // Even a rejected last call closes the buffer on PJRT's side.
        if (is_last) target.closed = true;
        const submit_started = awakeNs(self.io);
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
        const event_index = plan.events_used.fetchAdd(1, .monotonic);
        std.debug.assert(event_index < plan.events.len);
        const ctx = &plan.events[event_index];
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .device_index = target.device_index,
            .len = transfer.len,
            .submitted_ns = submit_started,
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
                _ = pipeline.metrics.dma_piece_ns.fetchAdd(awakeNs(pipeline.io) -| ctx_.submitted_ns, .monotonic);
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
                pipeline.retireEvent(ctx_);
                block.complete();
            }
        }.call, ctx) catch |err| {
            // The batch owns `ctx` and destroys the event at retirement.
            event.awaitRaw(api) catch {};
            return err;
        };
        _ = self.metrics.dma_submit_ns.fetchAdd(awakeNs(self.io) -| submit_started, .monotonic);
    }

    fn eventCompleted(self: *Pipeline, device_index: usize, len: usize) void {
        const device_pump = &self.pumps[device_index];
        device_pump.mutex.lockUncancelable(self.io);
        std.debug.assert(device_pump.active_bytes >= len);
        std.debug.assert(device_pump.active_pieces > 0);
        device_pump.active_bytes -= len;
        device_pump.active_pieces -= 1;
        device_pump.mutex.unlock(self.io);
        // A ready callback can be the first place an asynchronous PJRT error
        // becomes visible. Once outside the pump lock, retire every queued
        // transfer so request lifecycles cannot wait forever on entries that
        // the failed pumps will no longer submit.
        if (self.failed())
            self.abortReady()
        else
            self.requestPump(device_index);
    }

    /// Force-completes every queued transfer, one device at a time. A
    /// completion here may finish its batch, but a batch still has an entry
    /// queued until its last one is popped, and it is retired with every
    /// pump locked, so no queued entry is touched after its block completes.
    fn abortReady(self: *Pipeline) void {
        for (self.pumps) |*device_pump| {
            device_pump.mutex.lockUncancelable(self.io);
            while (device_pump.queue.popFront()) |transfer| {
                transfer.block.complete();
                device_pump.ready_entries -= 1;
            }
            device_pump.mutex.unlock(self.io);
        }
    }
};

/// Halves the source width when the source throttles: the one width change
/// left after the adaptive controller went (CTX.md, fifteenth pass). Runs as
/// a task only when the profile has a statistics side channel, which the
/// local backend has not, and samples it every 25 ms so a throttle is seen
/// while the workers sleep in the backend's retries.
const ThrottleWatch = struct {
    cursor: ReadStatsCursor,
    metrics: *const Metrics,
    read_gate: *RequestGate,
    request_gate: *RequestGate,
    limits: RequestGateLimits.Config,
    /// The loader's width; this task is its only writer while it runs.
    width: *usize,
    /// Reads completed at the last step and the width in flight then: the
    /// next step waits until that many reads have completed since, so the
    /// delayed feedback of the old width cannot ratchet through several
    /// steps.
    reads_at_step: u64 = 0,
    settle_reads: u64 = 0,
    done: std.Io.Event = .unset,

    fn run(self: *ThrottleWatch, io: std.Io) std.Io.Cancelable!void {
        while (true) {
            self.done.waitTimeout(io, .{ .duration = .{
                .raw = .fromMilliseconds(25),
                .clock = .awake,
            } }) catch |err| switch (err) {
                error.Timeout => {},
                error.Canceled => return error.Canceled,
            };
            if (self.done.isSet()) return;
            self.tick(io);
        }
    }

    /// One sample: on a throttle, halve the width once everything in flight
    /// at the previous step has returned.
    fn tick(self: *ThrottleWatch, io: std.Io) void {
        if (!self.cursor.takeThrottle()) return;
        const completed = self.metrics.read_operations.load(.acquire);
        if (completed -| self.reads_at_step < self.settle_reads) return;
        const width = self.width.*;
        if (width == 1) return;
        const narrower = width / 2;
        const limits = self.limits.at(narrower);
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        self.width.* = narrower;
        self.reads_at_step = completed;
        self.settle_reads = width;
        load_log.debug("source throttled: width {d} -> {d}, lifecycle_credits={d}", .{ width, narrower, limits.lifecycle });
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

    /// Requests admitted under the old limit keep their permits; a lower
    /// limit only holds back new admissions.
    fn setLimit(self: *RequestGate, io: std.Io, new_limit: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.limit = new_limit;
        self.condition.broadcast(io);
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
    /// What the limits at any width derive from, fixed at creation.
    const Config = struct {
        feasible_width: usize,
        retained: usize,
        dma_stage: usize,

        fn at(self: Config, width: usize) RequestGateLimits {
            return .init(width, self.feasible_width, self.retained, self.dma_stage);
        }
    };

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

const Metrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    /// Time workers spend waiting for a lifecycle credit, for pinned
    /// blocks, and inside the source read, summed over requests: waits
    /// above the read time mean the load is DMA-completion bound.
    lifecycle_wait_ns: std.atomic.Value(u64) = .init(0),
    block_wait_ns: std.atomic.Value(u64) = .init(0),
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
    /// Inside `ensureState` for a claimed job's items: the first worker to
    /// touch a tensor creates its PJRT buffers and transfer managers there,
    /// and the other workers of the same tensor wait for it.
    tensor_init_ns: std.atomic.Value(u64) = .init(0),
};

const ReadStatsCursor = struct {
    /// Backend-wide, not tagged by loader or submission: the watch assumes
    /// this load is the backend's only material user. Concurrent unrelated
    /// traffic can otherwise look like this loader's throttling.
    provider: VFS.ReadStatsProvider,
    previous: VFS.ReadStats,

    /// Whether the source throttled or timed out a request since the last
    /// call. Retries, connection failures and other 5xx are the backend's
    /// retry loop's business: they say nothing about the width, and the
    /// width cannot climb back.
    fn takeThrottle(self: *ReadStatsCursor) bool {
        const current = self.provider.snapshot();
        const delta = current.sub(self.previous);
        self.previous = current;
        return delta.throttles != 0 or delta.timeouts != 0;
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

/// Ensures the workspace can feed every calibrated device and hold one
/// complete fixed-size source request; retained arenas are reused first.
fn ensureLoadBlockReserve(
    self: *host_memory.Workspace,
    block_size: usize,
    calibrated_reserve: usize,
) !void {
    if (block_size == 0) return error.InvalidDmaLoadConfig;
    const request_blocks = try load_limits.maximumCoalescedJobBlocks(
        load_limits.max_read_request_size,
        block_size,
    );
    return self.growToBlocks(block_size, @max(calibrated_reserve, request_blocks));
}

/// Pre-grows `width + 1` source requests beside the DMA reserve.
fn ensureSourceWorkingSet(
    self: *host_memory.Workspace,
    block_size: usize,
    request_blocks: usize,
    width: usize,
    feed_reserve: usize,
) !void {
    if (block_size == 0 or request_blocks == 0) return error.InvalidDmaLoadConfig;
    const usable = self.usableBlocks(block_size);
    // Reserve first; when not even one request fits beside it the reserve
    // stays non-materialized and the source set alone is fitted.
    var with_reserve = true;
    var fitted_width = width;
    var target: usize = 0;
    while (true) : (fitted_width -= 1) {
        const source_blocks = (fitted_width + 1) * request_blocks;
        target = source_blocks + if (with_reserve) feed_reserve else 0;
        const growth_bytes = (target -| usable) * block_size;
        if (self.mapped_bytes + growth_bytes <= self.max_mapped_bytes) break;
        if (fitted_width == 0) {
            if (!with_reserve) return; // Leave growth to the load.
            with_reserve = false;
            fitted_width = width + 1;
        }
    }
    if (fitted_width < width or !with_reserve) {
        load_log.debug("DMA source working set clipped by the mapped ceiling: width={d} of {d}, reserve_materialized={}", .{
            fitted_width,
            width,
            with_reserve,
        });
    }
    return self.growToBlocks(block_size, target);
}

/// Requests whose pieces fill the DMA stage: `per_device` blocks of
/// in-flight bytes on every device, counted in requests of `request_size`.
/// Calibration pre-grows the same blocks per device as the stage's floor;
/// the lifecycle credits bound what the stage holds beyond it.
fn dmaStageRequests(per_device: usize, devices: usize, block_size: usize, request_size: usize) usize {
    std.debug.assert(request_size > 0);
    const bytes = per_device * devices * block_size;
    return @max(@as(usize, 1), bytes / request_size + @intFromBool(bytes % request_size != 0));
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
                    .item = undefined,
                    .block_index = 0,
                    .block_offset = 0,
                    .writer_mask = 1,
                    .destination_offset = 0,
                    .len = 64,
                });
                transfers[0].writer_mask = 0b11;
                break :plan try Batch.Plan.create(allocator, undefined, 2, 3, transfers, 128);
            };
            defer plan.destroy();
            try std.testing.expectEqual(@as(usize, 2), plan.jobs.len);
            try std.testing.expectEqual(@as(usize, 2), plan.requests.len);
            try std.testing.expectEqual(@as(usize, 3), plan.blocks.len);
            // The replicated piece needs two callbacks; the other needs one.
            try std.testing.expectEqual(@as(usize, 3), plan.events.len);
            for (plan.requests) |request| {
                try std.testing.expect(request.completed.load(.acquire));
                try std.testing.expectEqual(@as(usize, 0), request.pending.load(.acquire));
            }
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "loader releases the calibrated pool when alignment validation fails" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    var profile: VFS.LoadProfile = .local;
    profile.direct_io_alignment = 3;
    const result: anyerror!void = if (Loader.create(allocator, io, platform, .{
        .read_parallelism = 2,
        .load_profile = profile,
        .dma = .{},
        .max_host_bytes = 64 * 1024 * 1024,
        .direct_io = .on,
    })) |loader| unexpected: {
        loader.destroy();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.InvalidLoadProfile, result);
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
    for (std.enums.values(Failure)) |failure| {
        var sources = [_]safetensors.Tensor{
            .{ .file_uri = path, .name = "first", .shape = .init(.{4}, .u8), .offset = 0 },
            .{ .file_uri = later_path, .name = "later", .shape = .init(.{4}, .u8), .offset = 0 },
        };
        var outputs: [2]Buffer = @splat(.{
            ._platform = platform,
            ._shape = sources[0].shape,
            ._sharding = platform.replicated_sharding,
            ._shards = .empty,
        });
        defer for (&outputs) |*output| output.deinit();
        const loader = try Loader.create(allocator, io, platform, .{
            .read_parallelism = 2,
            .load_profile = .local,
            .dma = .{},
            .max_host_bytes = 64 * 1024 * 1024,
            .direct_io = .off,
        });
        defer loader.destroy();
        var specs = [_]LoadSpec{
            .{ .source = &sources[0], .shape = sources[0].shape, .sharding = platform.replicated_sharding, .output = &outputs[0] },
            .{ .source = &sources[1], .shape = .init(.{8}, .u8), .sharding = platform.replicated_sharding, .output = &outputs[1] },
        };

        switch (failure) {
            .before_publication, .after_publication => {
                // The later file's shape mismatch fails planning. Including the
                // first file forces that failure past one successful publication.
                const submitted = if (failure == .before_publication) specs[1..] else &specs;
                const result: anyerror!void = if (loader.submit(submitted, null)) |batch| unexpected: {
                    loader.awaitBatch(batch) catch {};
                    break :unexpected {};
                } else |err| err;
                try std.testing.expectError(error.InvalidLoaderJob, result);
            },
            .reading => {
                // Planning succeeds, but the source ends after four of five bytes.
                sources[0].shape = .init(.{5}, .u8);
                specs[0].shape = sources[0].shape;
                const batch = try loader.submit(specs[0..1], null);
                try std.testing.expectError(error.UnexpectedEndOfFile, loader.awaitBatch(batch));
                try std.testing.expect(loader.pool.high_water > 0);
            },
        }
        try std.testing.expectEqual(@as(usize, 0), loader.scheduler.remainingJobs(io));
        try std.testing.expectEqual(@as(usize, 0), loader.request_gate.inUse(io));
        try std.testing.expectEqual(@as(usize, 0), loader.read_gate.inUse(io));
        try std.testing.expectEqual(@as(usize, 0), loader.pool.in_use);

        if (failure == .before_publication) {
            try std.testing.expectEqual(@as(usize, 0), loader.batch_count);
            try loader.checkOpen();
            const batch = try loader.submit(specs[0..1], null);
            try loader.awaitBatch(batch);
            const loaded = try outputs[0].toSliceAlloc(allocator, io);
            defer loaded.free(allocator);
            try std.testing.expectEqualSlices(u8, &contents, loaded.constData());
        } else {
            try std.testing.expectEqual(@as(usize, 1), loader.batch_count);
            const expected = if (failure == .reading) error.UnexpectedEndOfFile else error.InvalidLoaderJob;
            try std.testing.expectError(expected, loader.checkOpen());
            const result: anyerror!void = if (loader.submit(specs[0..1], null)) |batch| unexpected: {
                loader.awaitBatch(batch) catch {};
                break :unexpected {};
            } else |err| err;
            try std.testing.expectError(expected, result);
        }
    }
}

test "DMA stage requests cover the per-device in-flight bytes" {
    const mib = 1024 * 1024;
    try std.testing.expectEqual(@as(usize, 8), dmaStageRequests(8, 1, 16 * mib, 16 * mib));
    try std.testing.expectEqual(@as(usize, 32), dmaStageRequests(8, 4, 16 * mib, 16 * mib));
    // A 32 MiB HF request holds two blocks: half as many requests.
    try std.testing.expectEqual(@as(usize, 4), dmaStageRequests(8, 1, 16 * mib, 32 * mib));
    try std.testing.expectEqual(@as(usize, 1), dmaStageRequests(1, 1, 16 * mib, 64 * mib));
}

/// Planning input for the fair-order tests: one job per entry, charged to
/// the devices its `physical_bytes` names.
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
    const physical_bytes = try allocator.alloc(usize, jobs.len * device_count);
    defer allocator.free(physical_bytes);
    for (jobs, 0..) |job, job_index| {
        if (job.physical_bytes.len != device_count) return error.InvalidTestJob;
        for (job.physical_bytes, queues, 0..) |bytes, *queue, device_index| {
            physical_bytes[job_index * device_count + device_index] = bytes;
            if (bytes != 0) try queue.append(allocator, job_index);
        }
    }
    return Planner.fairOrder(allocator, jobs.len, physical_bytes, queues);
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
    const jobs = try allocator.alloc(Batch.Plan.Job, job_count);
    errdefer allocator.free(jobs);
    const requests = try allocator.alloc(ReadRequest, job_count);
    errdefer allocator.free(requests);
    @memset(requests, ReadRequest.idle);
    const blocks = try allocator.alloc(Pipeline.BlockContext, job_count);
    errdefer allocator.free(blocks);
    const events = try allocator.alloc(Pipeline.EventContext, job_count);
    errdefer allocator.free(events);
    for (jobs, 0..) |*job, index| job.* = .{
        .file_offset = index,
        .len = 1,
        .minimum_len = 1,
        .transfer_start = 0,
        .transfer_len = 0,
        .block_start = index,
        .block_len = 1,
    };
    const plan = try allocator.create(Batch.Plan);
    plan.* = .{
        .allocator = allocator,
        .source_slot = undefined,
        .jobs = jobs,
        .transfers = &.{},
        .requests = requests,
        .blocks = blocks,
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

/// `Loader.submit`'s sequence for a one-file submission: publish,
/// seal, drop the sentinel.
fn publishTestBatch(scheduler: *Scheduler, job_count: usize) !*Batch {
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
fn discardTestBatch(scheduler: *Scheduler, batch: *Batch) void {
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
    var slots = [_]SourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var items: [sources.len]Item = undefined;
    for (&items, 0..) |*item, i| {
        item.* = .{
            .source = &sources[i],
            .source_slot = if (i == sources.len - 1) &slots[1] else &slots[0],
            .shape = sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &outputs[i],
        };
    }
    var device_count: usize = 0;
    for (platform.replicated_sharding.devicesInCanonicalOrder()) |device| {
        device_count = @max(device_count, @as(usize, @intCast(device.id)) + 1);
    }

    var scheduler: Scheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try Planner.publishFiles(&scheduler, io, batch, &items, .{
        .device_count = device_count,
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
    try std.testing.expectEqual(@as(usize, 8), plans[0].jobs[0].len);
    try std.testing.expectEqual(@as(usize, 3), plans[0].jobs[0].transfer_len);
    try std.testing.expectEqual(@as(u64, 20), plans[0].jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].transfers.len);
    try std.testing.expectEqual(@as(u64, 3), plans[1].jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), plans[1].jobs[0].len);
    try std.testing.expectEqual(@as(usize, 4), plans[1].jobs[1].len);
    // The totals equal the former single-plan numbers for the same inputs.
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 4), batch.diagnostics.source_jobs);
    try std.testing.expectEqual(@as(u64, 24), batch.diagnostics.source_bytes);
    try std.testing.expectEqual(@as(usize, 7), batch.diagnostics.planned_transfers);
    // One writer per transfer on one device: one event per transfer. Each
    // job's contexts are ranges of its plan's arrays, blocks by 4-byte block.
    try std.testing.expectEqual(@as(usize, 7), batch.diagnostics.planned_dma_submissions);
    try std.testing.expectEqual(@as(usize, 4), plans[0].events.len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].events.len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].requests.len);
    try std.testing.expectEqual(@as(usize, 3), plans[0].blocks.len);
    try std.testing.expectEqual(@as(usize, 0), plans[0].jobs[0].block_start);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs[0].block_len);
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs[1].block_start);
    try std.testing.expectEqual(@as(usize, 1), plans[0].jobs[1].block_len);
    try std.testing.expectEqual(@as(usize, 3), plans[1].blocks.len);
    try std.testing.expectEqual(@as(usize, 0), plans[1].jobs[0].block_start);
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs[0].block_len);
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs[1].block_start);
    try std.testing.expectEqual(@as(usize, 1), plans[1].jobs[1].block_len);
    for (plans) |plan| {
        for (plan.requests) |*request| try std.testing.expect(request.completed.load(.acquire));
    }
    discardTestBatch(&scheduler, batch);

    var iov_source: safetensors.Tensor = .{
        .file_uri = "iov",
        .name = "iov0",
        .shape = .init(.{@as(i64, @intCast(load_limits.max_positional_iovecs + 1))}, .u8),
        .offset = 0,
    };
    var iov_slot: SourceSlot = .{ .uri = "iov" };
    var iov_output: Buffer = undefined;
    var iov_items = [_]Item{.{
        .source = &iov_source,
        .source_slot = &iov_slot,
        .shape = iov_source.shape,
        .sharding = platform.replicated_sharding,
        .output = &iov_output,
    }};
    const iov_plan = try Planner.preparePlan(allocator, &iov_items, &.{0}, .{
        .device_count = device_count,
        .block_size = 1,
        .request_size = load_limits.max_positional_iovecs + 1,
        .alignment = 0,
    });
    defer iov_plan.destroy();
    try std.testing.expectEqual(@as(usize, 2), iov_plan.jobs.len);
    try std.testing.expectEqual(load_limits.max_positional_iovecs, iov_plan.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 1), iov_plan.jobs[1].len);

    var aligned_sources = [_]safetensors.Tensor{
        .{ .file_uri = "aligned", .name = "aligned0", .shape = .init(.{7}, .u8), .offset = 0 },
        .{ .file_uri = "aligned", .name = "aligned1", .shape = .init(.{8}, .u8), .offset = 7 },
        .{ .file_uri = "aligned", .name = "aligned2", .shape = .init(.{5}, .u8), .offset = 15 },
    };
    var aligned_slot: SourceSlot = .{ .uri = "aligned" };
    var aligned_outputs: [aligned_sources.len]Buffer = undefined;
    var aligned_items: [aligned_sources.len]Item = undefined;
    for (&aligned_items, 0..) |*item, i| {
        item.* = .{
            .source = &aligned_sources[i],
            .source_slot = &aligned_slot,
            .shape = aligned_sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &aligned_outputs[i],
        };
    }
    const aligned_plan = try Planner.preparePlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .device_count = device_count,
        .block_size = 4,
        .request_size = 8,
        .alignment = 0,
    });
    defer aligned_plan.destroy();
    try std.testing.expectEqual(@as(usize, 3), aligned_plan.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_plan.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_plan.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 8), aligned_plan.jobs[1].len);
    try std.testing.expectEqual(@as(usize, 5), aligned_plan.jobs[2].len);

    // Widened for direct I/O: the same three jobs (16 - 2 * 4 = 8 still cuts
    // at the tensor-safe boundaries 7 and 15), each read at 4-byte bounds,
    // its transfers relative to the widened start; the last read runs past
    // the tensor data, of which every byte is required.
    const widened_plan = try Planner.preparePlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .device_count = device_count,
        .block_size = 4,
        .request_size = 16,
        .alignment = 4,
    });
    defer widened_plan.destroy();
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs.len);
    try std.testing.expectEqual(@as(u64, 0), widened_plan.jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 7), widened_plan.jobs[0].minimum_len);
    try std.testing.expectEqual(@as(u64, 4), widened_plan.jobs[1].file_offset);
    try std.testing.expectEqual(@as(usize, 12), widened_plan.jobs[1].len);
    try std.testing.expectEqual(@as(usize, 11), widened_plan.jobs[1].minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.jobs[1].block_len);
    const second_transfer = widened_plan.transfers[widened_plan.jobs[1].transfer_start];
    try std.testing.expectEqual(@as(usize, 0), second_transfer.block_index);
    try std.testing.expectEqual(@as(usize, 3), second_transfer.block_offset);
    try std.testing.expectEqual(@as(u64, 12), widened_plan.jobs[2].file_offset);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].len);
    try std.testing.expectEqual(@as(usize, 8), widened_plan.jobs[2].minimum_len);
    try std.testing.expectEqual(@as(usize, 3), widened_plan.transfers[widened_plan.jobs[2].transfer_start].block_offset);
    try std.testing.expectEqual(@as(u64, 20), widened_plan.source_bytes);

    // No room for the widening: two alignment units must fit in a request.
    const invalid_plan: anyerror!void = if (Planner.preparePlan(allocator, &aligned_items, &.{ 0, 1, 2 }, .{
        .device_count = device_count,
        .block_size = 4,
        .request_size = 8,
        .alignment = 4,
    })) |plan| unexpected: {
        plan.destroy();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.InvalidLoaderJob, invalid_plan);
    try std.testing.expectError(error.InvalidLoaderJob, Planner.maximumJobLen(.{
        .device_count = 1,
        .block_size = 4,
        .request_size = 16,
        .alignment = 3,
    }));
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
    var slots = [_]SourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var items: [sources.len]Item = undefined;
    for (&items, 0..) |*item, i| {
        item.* = .{
            .source = &sources[i],
            .source_slot = if (i == 0) &slots[1] else &slots[0],
            .shape = sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &outputs[i],
        };
    }
    var device_count: usize = 0;
    for (platform.replicated_sharding.devicesInCanonicalOrder()) |device| {
        device_count = @max(device_count, @as(usize, @intCast(device.id)) + 1);
    }

    var scheduler: Scheduler = .init(allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(allocator, io, .{});
    try Planner.publishFiles(&scheduler, io, batch, &items, .{
        .device_count = device_count,
        .block_size = 4,
        .request_size = 4,
        .alignment = 0,
    }, .off);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 3), batch.diagnostics.source_jobs);
    try std.testing.expectEqual(@as(usize, 3), scheduler.remainingJobs(io));
    try std.testing.expect(batch.diagnostics.published_at != null);
    try std.testing.expect(batch.diagnostics.sealed_at == null);

    // File a's jobs in offset order, then file b's.
    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[0]);
    try std.testing.expectEqual(@as(u64, 0), claim.job().file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[0]);
    try std.testing.expectEqual(@as(u64, 4), claim.job().file_offset);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.plan.source_slot == &slots[1]);
    try std.testing.expectEqual(@as(u64, 0), claim.job().file_offset);
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
    const queues = [_]std.ArrayListUnmanaged(usize){ .empty, .empty };
    try std.testing.expectError(
        error.InvalidLoaderJob,
        Planner.fairOrder(allocator, 1, &.{1}, &queues),
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
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const first = try publishTestBatch(&scheduler, 2);
    const second = try publishTestBatch(&scheduler, 1);
    try std.testing.expectEqual(@as(usize, 3), scheduler.remainingJobs(io));

    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.batch == first);
    try std.testing.expectEqual(@as(u64, 0), claim.job().file_offset);
    try std.testing.expect(claim.request() == &first.plans.items[0].requests[0]);
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.batch == first);
    try std.testing.expectEqual(@as(u64, 1), claim.job().file_offset);
    try std.testing.expectEqual(@as(usize, 1), scheduler.remainingJobs(io));
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.batch == second);
    try std.testing.expectEqual(@as(u64, 0), claim.job().file_offset);
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));

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

    try std.testing.expect(scheduler.claim(io).?.batch == first);
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    try std.testing.expect(!second.done.isSet());
    try std.testing.expectEqual(@as(usize, 2), scheduler.remainingJobs(io));
    // The completed batch left the queue with its last claim, so it can go
    // away while the other one is still being claimed.
    first.destroy();

    try std.testing.expect(scheduler.claim(io).?.batch == second);
    try std.testing.expect(scheduler.claim(io).?.batch == second);
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
    try std.testing.expect(scheduler.claim(io).?.batch == first);
    scheduler.fail(io);
    try std.testing.expect(!first.done.isSet());
    try std.testing.expectEqual(@as(usize, 1), first.remaining.load(.acquire));
    try std.testing.expect(second.done.isSet());
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));
    try std.testing.expect(!scheduler.waitForWork(io));
    const publish_result: anyerror!void = if (publishTestBatch(&scheduler, 1)) |_| {} else |err| err;
    try std.testing.expectError(error.LoaderShuttingDown, publish_result);
    first.finishJobs(1);
    try std.testing.expect(first.done.isSet());
    first.destroy();
    second.destroy();

    // Everything claimed before the failure: nothing to retire.
    var exhausted: Scheduler = .init(std.testing.allocator);
    defer exhausted.deinit();
    const claimed = try publishTestBatch(&exhausted, 1);
    try std.testing.expect(exhausted.claim(io).?.batch == claimed);
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
                const base: u64 = if (claim.batch == batches_[0]) 0 else 16;
                const mask = @as(u64, 1) << @intCast(base + claim.job().file_offset);
                if (seen_.fetchOr(mask, .acq_rel) & mask != 0) duplicate_.store(true, .release);
                _ = claim_count_.fetchAdd(1, .monotonic);
            }
        }
    }.run, .{ &scheduler, &batches, &seen, &claim_count, &duplicate });
    try group.await(io);
    try std.testing.expectEqual(std.math.maxInt(u32), @as(u32, @truncate(seen.load(.acquire))));
    try std.testing.expectEqual(@as(usize, 32), claim_count.load(.acquire));
    try std.testing.expect(!duplicate.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));
    for (batches) |batch| {
        batch.finishJobs(16);
        try std.testing.expect(batch.done.isSet());
        batch.destroy();
    }
}

test "fifo scheduler keeps an open batch at the head until its next plan is published" {
    const WaitResult = enum(u8) { waiting, work, stopped };
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 1);
    var claim = scheduler.claim(io).?;
    try std.testing.expect(claim.batch == batch);
    // The published plan is exhausted but the batch is open: nothing to
    // claim, and the head is kept.
    try std.testing.expect(scheduler.claim(io) == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));
    try std.testing.expect(batch.queued);
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
    try std.testing.expectEqual(@as(usize, 2), batch.diagnostics.plans);
    try std.testing.expectEqual(@as(usize, 3), batch.diagnostics.source_jobs);

    // The new plan's jobs follow within the same batch; sealed with a job
    // left, the last claim pops it.
    claim = scheduler.claim(io).?;
    try std.testing.expect(claim.batch == batch);
    try std.testing.expectEqual(@as(u64, 0), claim.job().file_offset);
    scheduler.seal(io, batch);
    try std.testing.expect(batch.queued);
    claim = scheduler.claim(io).?;
    try std.testing.expectEqual(@as(u64, 1), claim.job().file_offset);
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
    try std.testing.expect(scheduler.claim(io).?.batch == exhausted);
    scheduler.seal(io, exhausted);
    try std.testing.expect(!exhausted.queued);
    try std.testing.expectEqual(@as(usize, 0), scheduler.queue.items.len);
    exhausted.finishJobs(2);
    try std.testing.expect(exhausted.done.isSet());
    exhausted.destroy();
}

test "fifo scheduler failure retires every published plan of an open batch" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    const batch = try Batch.create(std.testing.allocator, io, .{});
    try publishTestPlan(&scheduler, batch, 2);
    try publishTestPlan(&scheduler, batch, 3);
    try std.testing.expectEqual(@as(usize, 5), scheduler.remainingJobs(io));
    try std.testing.expectEqual(@as(usize, 6), batch.remaining.load(.acquire));
    try std.testing.expect(scheduler.claim(io).?.batch == batch);

    // The claim keeps its unit; the four unclaimed jobs of both plans are
    // retired; the sentinel is still held.
    scheduler.fail(io);
    try std.testing.expect(!batch.done.isSet());
    try std.testing.expect(!batch.queued);
    try std.testing.expectEqual(@as(usize, 2), batch.remaining.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));
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

/// A read statistics provider the tests move by hand.
const FakeStatsProvider = struct {
    stats: VFS.ReadStats = .{},

    fn snapshot(userdata: *anyopaque) VFS.ReadStats {
        const self: *@This() = @ptrCast(@alignCast(userdata));
        return self.stats;
    }

    fn provider(self: *FakeStatsProvider) VFS.ReadStatsProvider {
        return .{ .userdata = self, .snapshotFn = snapshot };
    }
};

test "one load-profile feedback cursor reports only new throttles" {
    var fake: FakeStatsProvider = .{};
    var cursor: ReadStatsCursor = .{
        .provider = fake.provider(),
        .previous = fake.provider().snapshot(),
    };

    try std.testing.expect(!cursor.takeThrottle());
    // Retries and failures are the backend's retry loop's business.
    fake.stats.retries = 2;
    fake.stats.server_failures = 1;
    fake.stats.transient_retries = 1;
    try std.testing.expect(!cursor.takeThrottle());
    fake.stats.throttles = 1;
    try std.testing.expect(cursor.takeThrottle());
    // Only what moved since the last call.
    try std.testing.expect(!cursor.takeThrottle());
    fake.stats.timeouts = 1;
    try std.testing.expect(cursor.takeThrottle());
    try std.testing.expect(!cursor.takeThrottle());
}

test "throttle watch halves the width once the reads in flight at the last step returned" {
    const io = std.testing.io;
    var fake: FakeStatsProvider = .{};
    var metrics: Metrics = .{};
    const limits: RequestGateLimits.Config = .{ .feasible_width = 64, .retained = 41, .dma_stage = 8 };
    var width: usize = 32;
    var read_gate: RequestGate = .init(limits.at(width).read);
    var request_gate: RequestGate = .init(limits.at(width).lifecycle);
    var watch: ThrottleWatch = .{
        .cursor = .{ .provider = fake.provider(), .previous = fake.provider().snapshot() },
        .metrics = &metrics,
        .read_gate = &read_gate,
        .request_gate = &request_gate,
        .limits = limits,
        .width = &width,
    };

    // Nothing moved, then retries alone: the width stays.
    watch.tick(io);
    fake.stats.retries = 3;
    fake.stats.server_failures = 1;
    watch.tick(io);
    try std.testing.expectEqual(@as(usize, 32), width);
    try std.testing.expectEqual(@as(usize, 32), read_gate.limit);

    // A throttle halves the width; both gates follow.
    fake.stats.throttles = 1;
    watch.tick(io);
    try std.testing.expectEqual(@as(usize, 16), width);
    try std.testing.expectEqual(@as(usize, 16), read_gate.limit);
    try std.testing.expectEqual(@as(usize, 41), request_gate.limit);

    // A throttle before the 32 reads in flight at the step have returned is
    // the old width's feedback.
    fake.stats.throttles = 2;
    watch.tick(io);
    try std.testing.expectEqual(@as(usize, 16), width);
    metrics.read_operations.store(32, .release);
    fake.stats.timeouts = 1;
    watch.tick(io);
    try std.testing.expectEqual(@as(usize, 8), width);
    try std.testing.expectEqual(@as(usize, 8), read_gate.limit);
    try std.testing.expectEqual(@as(usize, 41), request_gate.limit);

    // One read at a time is the floor.
    width = 1;
    watch.settle_reads = 0;
    fake.stats.throttles = 3;
    watch.tick(io);
    try std.testing.expectEqual(@as(usize, 1), width);
    try std.testing.expectEqual(@as(usize, 8), read_gate.limit);
}

test "device pump admits by bytes and by pieces" {
    var pump: Pipeline.DevicePump = .{};
    try std.testing.expect(pump.hasRoom(8));
    // The piece that crosses the budget was admitted; the next is not.
    pump.active_bytes = 8;
    try std.testing.expect(!pump.hasRoom(8));
    pump.active_bytes = 7;
    try std.testing.expect(pump.hasRoom(8));
    pump.active_pieces = max_dma_pieces_per_device;
    try std.testing.expect(!pump.hasRoom(8));
}

test "source request size combines the VFS floor with DMA granularity" {
    try std.testing.expectEqual(
        @as(usize, 8 * 1024 * 1024),
        try load_limits.effectiveSourceRequestSize(8 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try load_limits.effectiveSourceRequestSize(8 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try load_limits.effectiveSourceRequestSize(16 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        load_limits.max_read_request_size,
        try load_limits.effectiveSourceRequestSize(32 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectError(error.InvalidLoadProfile, load_limits.effectiveSourceRequestSize(0, 8 * 1024 * 1024));
    try std.testing.expectError(
        error.InvalidLoadProfile,
        load_limits.effectiveSourceRequestSize(load_limits.max_read_request_size + 1, 8 * 1024 * 1024),
    );
}

test "coalesced job block bound is independent of device count" {
    try std.testing.expectEqual(
        @as(usize, 2),
        try load_limits.maximumCoalescedJobBlocks(32 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 3),
        try load_limits.maximumCoalescedJobBlocks(17 * 1024 * 1024, 8 * 1024 * 1024),
    );
}

test "request lifecycle gate holds the DMA stage beyond the read width" {
    const config: RequestGateLimits.Config = .{ .feasible_width = 64, .retained = 41, .dma_stage = 8 };
    const normal = config.at(12);
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

test "request gate reductions drain without cancelling active requests" {
    const io = std.testing.io;
    var gate: RequestGate = .init(2);
    try std.testing.expect(gate.acquire(io));
    try std.testing.expect(gate.acquire(io));

    gate.setLimit(io, 1);
    var admitted: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(gate_: *RequestGate, io_: std.Io, admitted_: *std.Io.Event) void {
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

/// A single-device pipeline without PJRT: enough for request, block and
/// batch lifecycle tests. Must not move after `init`.
const TestPipeline = struct {
    metrics: Metrics = .{},
    gate: RequestGate,
    pumps: [1]Pipeline.DevicePump = .{.{}},
    pipeline: Pipeline,

    fn init(
        self: *TestPipeline,
        gate_limit: usize,
        pool: ?*host_memory.BlockPool,
        scheduler: *Scheduler,
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
            .metrics = &self.metrics,
            .scheduler = scheduler,
            .pumps = &self.pumps,
            .dma_budget_bytes = 64,
        };
    }

    fn deinit(self: *TestPipeline) void {
        self.pumps[0].queue.deinit(std.testing.allocator);
    }

    /// The worker's claim-to-request sequence.
    fn claimRequest(
        self: *TestPipeline,
        scheduler: *Scheduler,
    ) !*ReadRequest {
        const io = std.testing.io;
        const claim = scheduler.claim(io) orelse return error.NoJob;
        try std.testing.expect(self.gate.acquire(io));
        return ReadRequest.init(&self.pipeline, claim);
    }
};

test "late vectored callback failure drains and signals completion" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var pool = pool_init: {
        var workspace = try host_memory.Workspace.initForTesting(allocator, io, 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(64);
        break :pool_init try host_memory.BlockPool.init(allocator, &workspace, 64, 0);
    };
    defer pool.deinit();
    var scheduler: Scheduler = .init(allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(1, &pool, &scheduler);
    defer fixture.deinit();
    const pipeline = &fixture.pipeline;

    // One request whose only block is queued behind a DMA slot that a failed
    // callback is about to free.
    const batch = try publishTestBatch(&scheduler, 1);
    const request = try fixture.claimRequest(&scheduler);
    var leased: [1]host_memory.BlockPool.Block = undefined;
    try pool.acquireMany(io, &leased);
    const block = request.registerBlock(leased[0], 1);
    try std.testing.expect(block == &batch.plans.items[0].blocks[0]);
    var target: TensorTransfer.Target = .{ .manager = undefined, .device_index = 0, .total = 64 };
    try fixture.pumps[0].queue.pushBack(allocator, .{
        .target = &target,
        .block = block,
        .source_offset = 0,
        .destination_offset = 0,
        .len = 64,
    });
    fixture.pumps[0].ready_entries = 1;
    fixture.pumps[0].active_bytes = 64;
    fixture.pumps[0].active_pieces = 1;
    request.finishScheduling();
    try std.testing.expect(!batch.done.isSet());
    pipeline.first_error.store(@intFromError(error.Unknown), .release);

    pipeline.eventCompleted(0, 64);
    try std.testing.expectEqual(@as(usize, 0), fixture.pumps[0].active_pieces);
    try std.testing.expectEqual(@as(usize, 0), fixture.pumps[0].ready_entries);
    try std.testing.expect(block.lease.remaining.load(.acquire) == 0);
    try std.testing.expect(request.completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), fixture.gate.inUse(io));
    try std.testing.expect(batch.done.isSet());

    pipeline.retireBatch(batch);
    batch.destroy();
}

test "batch completes when every claimed request completes" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(3, null, &scheduler);
    defer fixture.deinit();

    const batch = try publishTestBatch(&scheduler, 3);
    try std.testing.expectEqual(@as(usize, 3), batch.remaining.load(.acquire));
    var requests: [3]*ReadRequest = undefined;
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
        try std.testing.expect(request == slot);
    }

    fixture.pipeline.retireBatch(batch);
    batch.destroy();
}

test "retirement accepts the idle slots of jobs a failure retired" {
    const io = std.testing.io;
    var scheduler: Scheduler = .init(std.testing.allocator);
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
    try std.testing.expect(request == &plan.requests[0]);
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
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(2, null, &scheduler);
    defer fixture.deinit();
    const pipeline = &fixture.pipeline;

    const batch = try publishTestBatch(&scheduler, 2);
    const requests = [_]*ReadRequest{
        try fixture.claimRequest(&scheduler),
        try fixture.claimRequest(&scheduler),
    };
    const plan = batch.plans.items[0];
    plan.events_used.store(2, .release);
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
    pipeline.retireEvent(&plan.events[0]);
    for (requests) |request| request.finishScheduling();
    try std.testing.expect(batch.done.isSet());
    pipeline.retireBatch(batch);
    try std.testing.expect(device_pump.retired == null);
    batch.destroy();
}

test "overlapping batches complete under concurrent claims and retirement" {
    const io = std.testing.io;
    const worker_count = 4;
    var scheduler: Scheduler = .init(std.testing.allocator);
    defer scheduler.deinit();
    var fixture: TestPipeline = undefined;
    fixture.init(worker_count, null, &scheduler);
    defer fixture.deinit();

    const Worker = struct {
        fn run(fixture_: *TestPipeline, scheduler_: *Scheduler) void {
            const io_ = std.testing.io;
            while (scheduler_.waitForWork(io_)) {
                if (!fixture_.gate.acquire(io_)) return;
                const claim = scheduler_.claim(io_) orelse {
                    fixture_.gate.release(io_);
                    continue;
                };
                const request = ReadRequest.init(&fixture_.pipeline, claim);
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
        try std.testing.expectEqual(@as(usize, 0), scheduler.remainingJobs(io));
    }
    // Every request released its credit before its batch completed; the
    // workers only hold one while a claim is in progress.
    scheduler.stop(io);
    fixture.gate.close(io);
    try group.await(io);
    try std.testing.expectEqual(@as(usize, 0), fixture.gate.inUse(io));
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

const DispatchTest = struct {
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
            var mask = span.writer_mask;
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
        var item: Item = .{
            .source = &source_tensor,
            .source_slot = undefined,
            .shape = shape,
            .sharding = sharding,
            .output = undefined,
        };
        const tensor_plan: Planner.TensorPlan = .{
            .item = &item,
            .dispatch_spans = dispatch_spans,
            .device_indices = device_indices,
            .total = shape.byteSize(),
        };
        var transfers: std.ArrayList(Batch.Plan.Transfer) = .empty;
        defer transfers.deinit(allocator);
        const physical_bytes = try allocator.alloc(usize, device_count);
        defer allocator.free(physical_bytes);
        // The pump flags a target's last transfer when its pieces reach the
        // placement's bytes, so every writer's pieces must sum to it.
        const written_bytes = try allocator.alloc(usize, writer_count);
        defer allocator.free(written_bytes);
        @memset(written_bytes, 0);

        const request_count = source.len / request_size + @intFromBool(source.len % request_size != 0);
        var reverse_index = request_count;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const source_offset = reverse_index * request_size;
            const request_len = @min(request_size, source.len - source_offset);
            transfers.clearRetainingCapacity();
            @memset(physical_bytes, 0);
            try Planner.appendTransfers(
                allocator,
                &transfers,
                0,
                &tensor_plan,
                source_offset,
                request_len,
                source_offset,
                block_size,
                physical_bytes,
            );
            for (transfers.items) |transfer| {
                try std.testing.expect(transfer.item == &item);
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
    try DispatchTest.run(.{
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
    try DispatchTest.run(.{
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
    try DispatchTest.run(.{
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
    try DispatchTest.run(.{
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
    try DispatchTest.run(.{
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
    try DispatchTest.run(.{
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
