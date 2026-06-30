const std = @import("std");

const stdx = @import("stdx");

pub const InitOpts = struct {
    chunk_size: usize = 16 * 1024 * 1024,
    num_workers: usize = 32,
    queue_capacity: usize = 64,
};

pub const BatchExecutor = struct {
    pending: std.atomic.Value(u32),
    err: std.atomic.Value(u16) = .init(0),

    pub fn failed(batch: *BatchExecutor) bool {
        return batch.err.load(.acquire) != 0;
    }

    pub fn fail(batch: *BatchExecutor, err: anyerror) void {
        _ = batch.err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    pub fn firstError(batch: *BatchExecutor) ?anyerror {
        const err = batch.err.load(.acquire);
        return if (err == 0) null else @errorFromInt(err);
    }

    pub fn completeOne(batch: *BatchExecutor, io: std.Io) void {
        const previous = batch.pending.fetchSub(1, .release);
        std.debug.assert(previous > 0);
        if (previous == 1) io.futexWake(u32, &batch.pending.raw, 1);
    }

    pub fn waitUncancelable(batch: *BatchExecutor, io: std.Io) void {
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
        chunk_size: usize,

        pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io, client: *std.http.Client, opts: InitOpts) !void {
            stdx.debug.assert(opts.num_workers > 0, "Pool must have at least one worker", .{});
            stdx.debug.assert(opts.chunk_size > 0, "Pool must have a positive download chunk size", .{});
            stdx.debug.assert(opts.queue_capacity > 0, "Pool must have a positive queue capacity", .{});

            const queue_buf = try allocator.alloc(Job, opts.queue_capacity);
            errdefer allocator.free(queue_buf);

            self.* = .{
                .client = client,
                .chunk_size = opts.chunk_size,
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

                if (!job.batch.executor.failed()) {
                    if (job.perform(pool.client)) |err| job.batch.executor.fail(err);
                }

                job.batch.executor.completeOne(io);
            }
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
