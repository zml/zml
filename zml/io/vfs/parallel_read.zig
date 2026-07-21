const std = @import("std");

const stdx = @import("stdx");
const ReadStats = @import("base.zig").ReadStats;

pub const InitOpts = struct {
    chunk_size: usize,
    num_workers: usize,
    queue_capacity: usize,
    max_retries: usize,
    retry_initial_delay: std.Io.Duration,
    retry_max_delay: std.Io.Duration,
};

pub const Status = union(enum) {
    success,
    retry_after: struct {
        delay: ?std.Io.Duration = null,
        throttled: bool = false,
    },

    pub fn retry() Status {
        return .{ .retry_after = .{} };
    }

    pub fn throttle(delay: ?std.Io.Duration) Status {
        return .{ .retry_after = .{ .delay = delay, .throttled = true } };
    }
};

pub const BatchState = struct {
    pending: std.atomic.Value(u32),
    err: std.atomic.Value(u16) = .init(0),

    pub fn failed(batch: *BatchState) bool {
        return batch.err.load(.acquire) != 0;
    }

    pub fn fail(batch: *BatchState, err: anyerror) void {
        _ = batch.err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    pub fn anyError(batch: *BatchState) ?anyerror {
        const err = batch.err.load(.acquire);
        return if (err == 0) null else @errorFromInt(err);
    }

    pub fn completeOne(batch: *BatchState, io: std.Io) void {
        const previous = batch.pending.fetchSub(1, .release);
        std.debug.assert(previous > 0);
        if (previous == 1) io.futexWake(u32, &batch.pending.raw, 1);
    }

    pub fn waitUncancelable(batch: *BatchState, io: std.Io) void {
        while (true) {
            const pending = batch.pending.load(.acquire);
            if (pending == 0) return;
            io.futexWaitUncancelable(u32, &batch.pending.raw, pending);
        }
    }
};

pub fn Pool(comptime Job: type) type {
    return struct {
        const Self = @This();

        group: std.Io.Group = .init,
        job_queue: std.Io.Queue(Job),
        queue_buf: []Job,
        client: *std.http.Client,
        prng: std.Random.DefaultPrng,
        chunk_size: usize,
        max_retries: usize,
        retry_initial_delay: std.Io.Duration,
        retry_max_delay: std.Io.Duration,
        physical_requests: std.atomic.Value(u64) = .init(0),
        physical_bytes: std.atomic.Value(u64) = .init(0),
        retries: std.atomic.Value(u64) = .init(0),
        throttles: std.atomic.Value(u64) = .init(0),
        retry_delay_ns: std.atomic.Value(u64) = .init(0),

        pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io, client: *std.http.Client, opts: InitOpts) !void {
            stdx.debug.assert(opts.num_workers > 0, "Pool must have at least one worker", .{});
            stdx.debug.assert(opts.chunk_size > 0, "Pool must have a positive download chunk size", .{});
            stdx.debug.assert(opts.queue_capacity > 0, "Pool must have a positive queue capacity", .{});
            stdx.debug.assert(opts.retry_initial_delay.nanoseconds >= 0, "Pool retry initial delay must not be negative", .{});
            stdx.debug.assert(opts.retry_max_delay.nanoseconds >= 0, "Pool retry max delay must not be negative", .{});
            stdx.debug.assert(opts.retry_max_delay.nanoseconds >= opts.retry_initial_delay.nanoseconds, "Pool retry max delay must be at least the initial delay", .{});

            const queue_buf = try allocator.alloc(Job, opts.queue_capacity);
            errdefer allocator.free(queue_buf);

            self.* = .{
                .client = client,
                .chunk_size = opts.chunk_size,
                .max_retries = opts.max_retries,
                .retry_initial_delay = opts.retry_initial_delay,
                .retry_max_delay = opts.retry_max_delay,
                .prng = .init(blk: {
                    var seed: u64 = undefined;
                    io.random(@ptrCast(&seed));
                    break :blk seed;
                }),
                .job_queue = .init(queue_buf),
                .queue_buf = queue_buf,
            };
            errdefer self.group.cancel(io);

            for (0..opts.num_workers) |_| {
                try self.group.concurrent(io, worker, .{ self, io });
            }
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator, io: std.Io) void {
            self.job_queue.close(io);
            self.group.cancel(io);

            allocator.free(self.queue_buf);
        }

        pub fn readStats(self: *const Self) ReadStats {
            return .{
                .physical_requests = self.physical_requests.load(.acquire),
                .physical_bytes = self.physical_bytes.load(.acquire),
                .retries = self.retries.load(.acquire),
                .throttles = self.throttles.load(.acquire),
                .retry_delay_ns = self.retry_delay_ns.load(.acquire),
            };
        }

        fn worker(pool: *Self, io: std.Io) std.Io.Cancelable!void {
            while (true) {
                const job = pool.job_queue.getOne(io) catch break;

                var attempt: usize = 0;
                while (true) {
                    if (job.batch.state.failed()) break;
                    _ = pool.physical_requests.fetchAdd(1, .monotonic);
                    const status = job.perform(pool.client) catch |err| {
                        job.batch.state.fail(err);
                        break;
                    };

                    switch (status) {
                        .success => {
                            if (@hasField(Job, "chunk_len")) {
                                _ = pool.physical_bytes.fetchAdd(@intCast(job.chunk_len), .monotonic);
                            }
                            break;
                        },
                        .retry_after => |retry| {
                            if (attempt >= pool.max_retries) {
                                job.batch.state.fail(error.RetriesExhausted);
                                break;
                            }

                            _ = pool.retries.fetchAdd(1, .monotonic);
                            if (retry.throttled) _ = pool.throttles.fetchAdd(1, .monotonic);
                            const delay = retry.delay orelse pool.backoffDuration(attempt);
                            _ = pool.retry_delay_ns.fetchAdd(@intCast(@max(delay.nanoseconds, 0)), .monotonic);
                            io.sleep(delay, .awake) catch {
                                job.batch.state.fail(error.RetriesExhausted);
                                break;
                            };

                            attempt += 1;
                        },
                    }
                    continue;
                }

                job.batch.state.completeOne(io);
            }
        }

        fn backoffDuration(pool: *Self, attempt: usize) std.Io.Duration {
            // FullJitter algorithm
            const max_delay_ns: i96 = @min(
                pool.retry_max_delay.toNanoseconds(),
                pool.retry_initial_delay.toNanoseconds() *| (@as(i96, 1) << @as(u7, @intCast(@min(attempt, std.math.maxInt(u7))))),
            );
            const delay_ns = pool.prng.random().intRangeAtMost(i96, 0, max_delay_ns);
            return .fromNanoseconds(delay_ns);
        }
    };
}

pub const ContentRange = struct {
    start: u64,
    end: u64,
    total: u64,
};

pub fn parseContentRange(value: []const u8) ?ContentRange {
    const space = std.mem.indexOfScalar(u8, value, ' ') orelse return null;
    const dash = std.mem.indexOfScalar(u8, value, '-') orelse return null;
    const slash = std.mem.indexOfScalar(u8, value, '/') orelse return null;

    return .{
        .start = std.fmt.parseInt(u64, value[space + 1 .. dash], 10) catch return null,
        .end = std.fmt.parseInt(u64, value[dash + 1 .. slash], 10) catch return null,
        .total = std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch return null,
    };
}

pub fn readSize(file_size: u64, offset: u64, data: []const []u8) usize {
    if (offset >= file_size) return 0;

    var requested: usize = 0;
    for (data) |buf| requested +|= buf.len;
    return @intCast(@min(file_size - offset, requested));
}

pub fn readChunk(
    reader: *std.Io.Reader,
    content_range: ?ContentRange,
    file_offset: u64,
    data: []const []u8,
    chunk_offset: usize,
    chunk_len: usize,
) !void {
    if (content_range) |cr| {
        if (cr.start > file_offset) return error.InvalidContentRange;
        if (cr.start < file_offset) {
            try reader.discardAll(file_offset - cr.start);
        }
    }

    var remaining = chunk_len;
    var skip = chunk_offset;

    for (data) |buf| {
        if (skip >= buf.len) {
            skip -= buf.len;
            continue;
        }

        const destination = buf[skip..][0..@min(remaining, buf.len - skip)];
        try reader.readSliceAll(destination);
        remaining -= destination.len;
        if (remaining == 0) break;
        skip = 0;
    }
    if (remaining != 0) return error.UnexpectedEndOfOutput;
}

test "readChunk scatters a chunk across destination boundaries" {
    var reader: std.Io.Reader = .fixed("abcdef");
    var first: [3]u8 = @splat(0);
    var second: [4]u8 = @splat(0);
    try readChunk(&reader, null, 0, &.{ &first, &second }, 2, 5);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 'a' }, &first);
    try std.testing.expectEqualStrings("bcde", &second);
}

test "readChunk rejects an undersized destination scatter list" {
    var reader: std.Io.Reader = .fixed("abcd");
    var output: [3]u8 = undefined;
    try std.testing.expectError(error.UnexpectedEndOfOutput, readChunk(&reader, null, 0, &.{&output}, 0, 4));
}
