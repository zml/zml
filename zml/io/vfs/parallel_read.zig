const std = @import("std");

const stdx = @import("stdx");

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
    retry_after: ?std.Io.Duration,

    pub fn retry() Status {
        return .{ .retry_after = null };
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
        num_workers: usize,
        chunk_size: usize,
        max_retries: usize,
        retry_initial_delay: std.Io.Duration,
        retry_max_delay: std.Io.Duration,

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
                .num_workers = opts.num_workers,
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

        fn worker(pool: *Self, io: std.Io) std.Io.Cancelable!void {
            while (true) {
                const job = pool.job_queue.getOne(io) catch break;

                var attempt: usize = 0;
                while (true) {
                    if (job.batch.state.failed()) break;
                    const status = job.perform(pool.client) catch |err| {
                        job.batch.state.fail(err);
                        break;
                    };

                    switch (status) {
                        .success => break,
                        .retry_after => |duration| {
                            if (attempt >= pool.max_retries) {
                                job.batch.state.fail(error.RetriesExhausted);
                                break;
                            }

                            io.sleep(duration orelse pool.backoffDuration(attempt), .awake) catch {
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
    for (data) |buf| requested += buf.len;
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
}
