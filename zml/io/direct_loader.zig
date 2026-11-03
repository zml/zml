const std = @import("std");

const pjrt = @import("pjrt");

const Buffer = @import("../buffer.zig").Buffer;
const pjrtx = @import("../pjrtx.zig");
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const backend = @import("backend.zig");
const ShardGrid = @import("ShardGrid.zig");
const host_memory = @import("host_memory.zig");

/// Creating a tensor's device state: host memory or PJRT, since the front
/// end checked its placement. `LazyOnce` replays it to every later reader
/// of the tensor.
const StateInitError = std.mem.Allocator.Error || pjrt.ApiError;

/// One submission: its plans (one per source file, published as each file
/// is planned), the destinations it writes and their parts, all in the
/// batch's arena until it retires. `remaining` counts completion units: one
/// per published job plus a publish sentinel held until the submission is
/// sealed. A job's unit is released exactly once, by whichever of these
/// happens: its request's last reference drops (final DMA callback or
/// abandonment), or `Scheduler.fail` retires it unclaimed. The batch is
/// done when `remaining` reaches zero; the awaiting task then retires it,
/// so releasing a unit is the last permitted access to the batch.
pub const Batch = struct {
    /// Holds the plans, the destinations and the parts until `destroy`;
    /// only the submitting task allocates in it.
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
    /// caller's final access to the batch and to anything it owns (plans,
    /// destinations, parts). The last unit sets `done`, and the awaiting
    /// task frees all of it as soon as it observes the event.
    fn releaseUnits(self: *Batch, count: usize) void {
        if (count == 0) return;
        const previous = self.remaining.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.done.set(self.io);
    }

    /// Frees the arena: the plans, the destinations and the parts.
    fn destroy(self: *Batch) void {
        const allocator = self.arena.child_allocator;
        self.arena.deinit();
        allocator.destroy(self);
    }
};

/// One output buffer and what places it. Its parts may come from several
/// files, so several plans can write it: their pieces all count toward the
/// same targets (`Target.total` is the placement's bytes, whichever plan
/// submits them), and its device state is created once, by the first
/// worker that reads for it.
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
    state: LazyOnce(DestinationState) = .{},

    fn ensureState(self: *Destination, allocator: std.mem.Allocator, io: std.Io, platform: *const Platform) StateInitError!*DestinationState {
        return self.state.ensure(io, DestinationState.init, .{ allocator, platform, self });
    }

    fn deinit(self: *Destination, allocator: std.mem.Allocator, api: *const pjrt.Api) void {
        if (self.state.readyValue()) |state| state.deinit(allocator, api);
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

/// One source file: every job of a plan reads the file of its slot.
const SourceSlot = struct {
    uri: []const u8,
};

/// One file's jobs in claim order with their transfer records, allocated by
/// the planner.
const Plan = struct {
    /// A claimable job: its slice of the plan's transfers and what its
    /// worker does for them.
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
        /// `minimum_len` hold tensor data and must exist. Reads are exact:
        /// the two are equal.
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
    /// The tensor bytes the jobs read.
    source_bytes: u64,
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

    /// Plans `parts` one file at a time and publishes each plan as soon as
    /// it exists, so workers claim the first file while the rest is planned,
    /// each plan in the batch's arena. Stops at the first error; the caller
    /// seals or fails the batch.
    /// Whole-model planning delayed the first read by 0.32 s for DeepSeek's
    /// 46 files on one MI300X; publishing per file overlaps the rest with IO.
    fn publishFiles(
        scheduler: *Scheduler,
        io: std.Io,
        batch: *Batch,
        parts: []Part,
        config: Config,
    ) !void {
        const order = try sortedPartOrder(scheduler.allocator, parts);
        defer scheduler.allocator.free(order);
        var file_start: usize = 0;
        while (file_start < order.len) {
            const file_end = fileGroupEnd(parts, order, file_start);
            const plan = try preparePlan(scheduler.allocator, batch.arena.allocator(), parts, order[file_start..file_end], config);
            try scheduler.publish(io, batch, plan);
            file_start = file_end;
        }
    }

    /// Plans one file: `order` indexes its `parts` by offset and size
    /// (`sortedPartOrder`), and `JobCut.cutRuns` cuts them. The plan goes in
    /// `arena` (`JobCut.finish`), the cut in `allocator`.
    fn preparePlan(
        allocator: std.mem.Allocator,
        arena: std.mem.Allocator,
        parts: []Part,
        order: []const usize,
        config: Config,
    ) !*Plan {
        const source_slot = parts[order[0]].source_slot;
        for (order) |part_index| std.debug.assert(std.mem.eql(u8, parts[part_index].source.file_uri, parts[order[0]].source.file_uri));
        var cut: JobCut = .{
            .allocator = allocator,
            .parts = parts,
            .source_slot = source_slot,
            .config = config,
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
        jobs: std.ArrayList(Plan.Job) = .empty,
        transfers: std.ArrayList(Plan.Transfer) = .empty,
        safe_boundaries: std.ArrayList(u64) = .empty,
        source_bytes: u64 = 0,

        /// The plan of the jobs cut, copied into `arena`.
        fn finish(self: *JobCut, arena: std.mem.Allocator) !*Plan {
            const plan = try arena.create(Plan);
            plan.* = .{
                .source_slot = self.source_slot,
                .jobs = try arena.dupe(Plan.Job, self.jobs.items),
                .transfers = try arena.dupe(Plan.Transfer, self.transfers.items),
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
        /// of jobs of at most a request, cut at part-safe boundaries where
        /// that number allows. No job depends on another, since every DMA
        /// piece is submitted as soon as its block is read.
        fn cutRuns(self: *JobCut, order: []const usize) !void {
            const allocator = self.allocator;
            const parts = self.parts;
            const maximum_job_len = self.config.request_size;
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
                        }, job_start);
                    }
                    std.debug.assert(self.transfers.items.len != transfer_start);
                    try self.jobs.append(allocator, .{
                        .file_offset = job_start,
                        .transfer_start = transfer_start,
                        .transfer_len = self.transfers.items.len - transfer_start,
                        .kind = .{ .dma_read = .{ .len = job_len, .minimum_len = job_len } },
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
    };
    fn blockCount(len: usize, block_size: usize) usize {
        return len / block_size + @intFromBool(len % block_size != 0);
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

// Planner tests.

/// `Planner.preparePlan`, kept in `arena`; the testing allocator checks the
/// cut for leaks.
fn prepareTestPlan(arena: std.mem.Allocator, parts: []Part, order: []const usize, config: Planner.Config) !*Plan {
    return Planner.preparePlan(std.testing.allocator, arena, parts, order, config);
}

/// One plain destination per source, as whole-tensor specs make them: for
/// planner tests.
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
    });

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
    // Jobs span their 4-byte blocks.
    try std.testing.expectEqual(@as(usize, 2), plans[0].jobs[0].blockLen(4));
    try std.testing.expectEqual(@as(usize, 1), plans[0].jobs[1].blockLen(4));
    try std.testing.expectEqual(@as(usize, 2), plans[1].jobs[0].blockLen(4));
    try std.testing.expectEqual(@as(usize, 1), plans[1].jobs[1].blockLen(4));
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
    });
    try std.testing.expectEqual(@as(usize, 3), aligned_plan.jobs.len);
    try std.testing.expectEqual(@as(usize, 6), aligned_plan.transfers.len);
    try std.testing.expectEqual(@as(usize, 7), aligned_plan.jobs[0].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 8), aligned_plan.jobs[1].kind.dma_read.len);
    try std.testing.expectEqual(@as(usize, 5), aligned_plan.jobs[2].kind.dma_read.len);
}

// Scheduler tests.

/// A plan of `job_count` unit jobs (`file_offset` = index), each with one
/// block, and no transfers, in `arena`.
fn testPlan(arena: std.mem.Allocator, job_count: usize) !*Plan {
    const jobs = try arena.alloc(Plan.Job, job_count);
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
    });
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items.len);
    try std.testing.expectEqual(@as(usize, 2), batch.plans.items[0].jobs.len);
    try std.testing.expectEqual(@as(usize, 1), batch.plans.items[1].jobs.len);
    try std.testing.expectEqual(@as(usize, 3), unclaimedJobs(&scheduler));

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
