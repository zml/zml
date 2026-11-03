const std = @import("std");

const Buffer = @import("../buffer.zig").Buffer;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const backend = @import("backend.zig");
const ShardGrid = @import("ShardGrid.zig");
const host_memory = @import("host_memory.zig");

/// One output buffer and what places it. Its parts may come from several
/// files, so several plans can write it.
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

    /// The file every job of the plan reads.
    source_slot: *SourceSlot,
    jobs: []Job,
    transfers: []Transfer,
    /// The tensor bytes the jobs read.
    source_bytes: u64,

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

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const order = try Planner.sortedPartOrder(allocator, &parts);
    defer allocator.free(order);
    var plans: [2]*Plan = undefined;
    var file_start: usize = 0;
    for (&plans) |*plan| {
        const file_end = Planner.fileGroupEnd(&parts, order, file_start);
        plan.* = try prepareTestPlan(arena.allocator(), &parts, order[file_start..file_end], .{
            .block_size = 4,
            .request_size = 8,
        });
        file_start = file_end;
    }
    try std.testing.expectEqual(order.len, file_start);

    // One plan per file. a:[10,18) merges adjacency and the duplicate,
    // a:[20,24) remains exact, and b:[3,15) is split at the request-size
    // boundary.
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
