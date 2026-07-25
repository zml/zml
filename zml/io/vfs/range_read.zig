const std = @import("std");

const ReadFailure = @import("base.zig").ReadFailure;

pub const ContentRange = struct {
    start: u64,
    end: u64,
    total: u64,
};

pub const ResponseTiming = struct {
    first_body_byte: std.Io.Timestamp,
    completed: std.Io.Timestamp,

    pub fn ttfbNanoseconds(self: ResponseTiming, attempt_started: std.Io.Timestamp) u64 {
        return elapsedNanoseconds(attempt_started, self.first_body_byte);
    }

    pub fn bodyNanoseconds(self: ResponseTiming) u64 {
        return elapsedNanoseconds(self.first_body_byte, self.completed);
    }
};

pub const AttemptResult = union(enum) {
    success: ResponseTiming,
    retry: Retry,
};

pub const Retry = struct {
    failure: ReadFailure,
    delay: ?std.Io.Duration = null,
};

pub fn parseContentRange(value: []const u8) ?ContentRange {
    const prefix = "bytes ";
    if (value.len < prefix.len or !std.ascii.eqlIgnoreCase(value[0..prefix.len], prefix)) return null;
    const range_and_total = value[prefix.len..];
    const dash = std.mem.indexOfScalar(u8, range_and_total, '-') orelse return null;
    const slash = dash + 1 + (std.mem.indexOfScalar(u8, range_and_total[dash + 1 ..], '/') orelse return null);
    if (dash == 0 or slash == dash + 1 or slash + 1 == range_and_total.len) return null;
    if (std.mem.indexOfScalar(u8, range_and_total[slash + 1 ..], '/') != null) return null;

    const result: ContentRange = .{
        .start = std.fmt.parseInt(u64, range_and_total[0..dash], 10) catch return null,
        .end = std.fmt.parseInt(u64, range_and_total[dash + 1 .. slash], 10) catch return null,
        .total = std.fmt.parseInt(u64, range_and_total[slash + 1 ..], 10) catch return null,
    };
    if (result.end < result.start or result.end >= result.total) return null;
    return result;
}

pub fn readSize(file_size: u64, offset: u64, data: []const []u8) usize {
    if (offset >= file_size) return 0;

    var requested: usize = 0;
    for (data) |buf| requested +|= buf.len;
    return @intCast(@min(file_size - offset, requested));
}

pub fn readResponse(
    io: std.Io,
    reader: *std.Io.Reader,
    status: std.http.Status,
    content_range: ?ContentRange,
    content_length: ?u64,
    file_size: u64,
    offset: u64,
    data: []const []u8,
    read_size: usize,
) !ResponseTiming {
    if (read_size == 0) return error.EmptyRangeRead;
    if (status != .partial_content) return error.UnexpectedStatus;
    const cr = content_range orelse return error.InvalidContentRange;
    const requested_end = std.math.add(u64, offset, read_size - 1) catch return error.InvalidContentRange;
    if (cr.start != offset or cr.end != requested_end or cr.total != file_size) {
        return error.InvalidContentRange;
    }
    if (content_length == null or content_length.? != read_size) return error.InvalidContentLength;

    var first: [1]u8 = undefined;
    try reader.readSliceAll(&first);
    const first_body_byte: std.Io.Timestamp = .now(io, .awake);
    try writeFirstAndReadScatter(reader, data, first[0], read_size);

    return .{
        .first_body_byte = first_body_byte,
        .completed = .now(io, .awake),
    };
}

fn writeFirstAndReadScatter(
    reader: *std.Io.Reader,
    data: []const []u8,
    first: u8,
    read_size: usize,
) !void {
    var wrote_first = false;
    for (data) |buf| {
        if (buf.len == 0) continue;
        buf[0] = first;
        wrote_first = true;
        break;
    }
    if (!wrote_first) return error.UnexpectedEndOfOutput;
    try readScatter(reader, data, 1, read_size - 1);
}

pub fn readScatter(
    reader: *std.Io.Reader,
    data: []const []u8,
    destination_offset: usize,
    len: usize,
) !void {
    var remaining = len;
    var skip = destination_offset;

    for (data) |buf| {
        if (skip >= buf.len) {
            skip -= buf.len;
            continue;
        }
        if (remaining == 0) break;

        const destination = buf[skip..][0..@min(remaining, buf.len - skip)];
        try reader.readSliceAll(destination);
        remaining -= destination.len;
        skip = 0;
    }
    if (remaining != 0) return error.UnexpectedEndOfOutput;
}

pub fn fullJitterDelay(
    io: std.Io,
    initial: std.Io.Duration,
    maximum: std.Io.Duration,
    attempt: usize,
) std.Io.Duration {
    const max_delay_ns: i96 = @min(
        maximum.toNanoseconds(),
        initial.toNanoseconds() *| (@as(i96, 1) << @as(u7, @intCast(@min(attempt, std.math.maxInt(u7))))),
    );
    if (max_delay_ns <= 0) return .fromNanoseconds(0);

    var seed: u64 = undefined;
    io.random(@ptrCast(&seed));
    var prng: std.Random.DefaultPrng = .init(seed);
    return .fromNanoseconds(prng.random().intRangeAtMost(i96, 0, max_delay_ns));
}

pub fn assertValidOptions(
    minimum_request_size: usize,
    retry_initial_delay: std.Io.Duration,
    retry_max_delay: std.Io.Duration,
) void {
    std.debug.assert(minimum_request_size > 0);
    std.debug.assert(retry_initial_delay.nanoseconds >= 0);
    std.debug.assert(retry_max_delay.nanoseconds >= retry_initial_delay.nanoseconds);
}

fn elapsedNanoseconds(start: std.Io.Timestamp, end: std.Io.Timestamp) u64 {
    return @intCast(@max(end.nanoseconds - start.nanoseconds, 0));
}

test "Content-Range parsing is strict" {
    try std.testing.expectEqual(
        ContentRange{ .start = 2, .end = 9, .total = 10 },
        parseContentRange("bytes 2-9/10").?,
    );
    try std.testing.expectEqual(
        ContentRange{ .start = 2, .end = 9, .total = 10 },
        parseContentRange("ByTeS 2-9/10").?,
    );
    try std.testing.expect(parseContentRange("items 2-9/10") == null);
    try std.testing.expect(parseContentRange("bytes 2-9/*") == null);
    try std.testing.expect(parseContentRange("bytes 9-2/10") == null);
    try std.testing.expect(parseContentRange("bytes 2-10/10") == null);
    try std.testing.expect(parseContentRange("bytes 2-9/10 trailing") == null);
}

test "range responses fill scatter buffers" {
    var reader: std.Io.Reader = .fixed("23456");
    var first: [2]u8 = undefined;
    var second: [3]u8 = undefined;
    _ = try readResponse(
        std.testing.io,
        &reader,
        .partial_content,
        .{ .start = 2, .end = 6, .total = 10 },
        5,
        10,
        2,
        &.{ &first, &second },
        5,
    );
    try std.testing.expectEqualStrings("23", &first);
    try std.testing.expectEqualStrings("456", &second);
}

test "partial responses require the exact Content-Range and length" {
    var reader: std.Io.Reader = .fixed("3456");
    var output: [4]u8 = undefined;
    try std.testing.expectError(error.UnexpectedStatus, readResponse(std.testing.io, &reader, .ok, null, 4, 10, 3, &.{&output}, 4));
    try std.testing.expectError(error.InvalidContentRange, readResponse(std.testing.io, &reader, .partial_content, null, 4, 10, 3, &.{&output}, 4));

    reader = .fixed("4567");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(std.testing.io, &reader, .partial_content, .{ .start = 4, .end = 7, .total = 10 }, 4, 10, 3, &.{&output}, 4),
    );

    reader = .fixed("3456");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(std.testing.io, &reader, .partial_content, .{ .start = 3, .end = 6, .total = 11 }, 4, 10, 3, &.{&output}, 4),
    );

    reader = .fixed("3456");
    try std.testing.expectError(
        error.InvalidContentLength,
        readResponse(std.testing.io, &reader, .partial_content, .{ .start = 3, .end = 6, .total = 10 }, 3, 10, 3, &.{&output}, 4),
    );
}
