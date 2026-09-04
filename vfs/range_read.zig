const std = @import("std");

const ReadFailure = @import("base.zig").ReadFailure;

pub const ContentRange = struct {
    start: u64,
    end: u64,
    total: u64,
};

pub const AttemptResult = union(enum) {
    success,
    retry: Retry,
};

pub const Retry = struct {
    failure: ReadFailure,
    delay: ?std.Io.Duration = null,
};

/// Retry classification of a non-2xx status; null when the status is not
/// retried. `unavailable` is what a 503 means for the backend: rate limiting
/// on the object stores (AWS `SlowDown`, GCS), a server failure elsewhere.
pub fn classifyStatus(status: std.http.Status, unavailable: ReadFailure) ?ReadFailure {
    return switch (status) {
        .request_timeout => .timeout,
        .too_many_requests => .throttle,
        .service_unavailable => unavailable,
        else => if (status.class() == .server_error) .server_failure else null,
    };
}

/// The retry delay the server names, if any: `Retry-After` delta-seconds
/// (the HTTP-date form is not parsed and falls back to the jittered delay)
/// or the `t=` reset of the `RateLimit` header Hugging Face sends.
pub fn serverRetryDelay(head: std.http.Client.Response.Head) ?std.Io.Duration {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "Retry-After")) {
            return delaySeconds(header.value) orelse continue;
        }
        if (std.ascii.eqlIgnoreCase(header.name, "RateLimit")) {
            var parts = std.mem.splitScalar(u8, header.value, ';');
            while (parts.next()) |part| {
                const trimmed = std.mem.trim(u8, part, " \t");
                if (std.mem.startsWith(u8, trimmed, "t=")) return delaySeconds(trimmed[2..]) orelse continue;
            }
        }
    }
    return null;
}

fn delaySeconds(value: []const u8) ?std.Io.Duration {
    const seconds = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch return null;
    return .fromSeconds(seconds);
}

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

/// Validates the response against the requested range, discards the prefix
/// of a `200` that ignored `Range`, then scatters `read_size` bytes.
pub fn readResponse(
    reader: *std.Io.Reader,
    status: std.http.Status,
    content_range: ?ContentRange,
    offset: u64,
    data: []const []u8,
    read_size: usize,
) !void {
    if (read_size == 0) return error.EmptyRangeRead;

    const response_start = switch (status) {
        .partial_content => blk: {
            const cr = content_range orelse return error.InvalidContentRange;
            const requested_end = std.math.add(u64, offset, read_size - 1) catch return error.InvalidContentRange;
            if (cr.start > offset or cr.end < requested_end) return error.InvalidContentRange;
            break :blk cr.start;
        },
        .ok => 0,
        else => return error.UnexpectedStatus,
    };

    try reader.discardAll(offset - response_start);
    try readScatter(reader, data, read_size);
}

pub fn readScatter(reader: *std.Io.Reader, data: []const []u8, len: usize) !void {
    var remaining = len;
    for (data) |buf| {
        if (remaining == 0) break;
        const destination = buf[0..@min(remaining, buf.len)];
        try reader.readSliceAll(destination);
        remaining -= destination.len;
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
    retry_initial_delay: std.Io.Duration,
    retry_max_delay: std.Io.Duration,
) void {
    std.debug.assert(retry_initial_delay.nanoseconds >= 0);
    std.debug.assert(retry_max_delay.nanoseconds >= retry_initial_delay.nanoseconds);
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
    var reader: std.Io.Reader = .fixed("23456789");
    var first: [2]u8 = undefined;
    var second: [3]u8 = undefined;
    try readResponse(
        &reader,
        .partial_content,
        .{ .start = 2, .end = 9, .total = 10 },
        3,
        &.{ &first, &second },
        5,
    );
    try std.testing.expectEqualStrings("34", &first);
    try std.testing.expectEqualStrings("567", &second);
}

test "200 responses that ignore Range are positioned and scattered" {
    var reader: std.Io.Reader = .fixed("0123456789");
    var first: [1]u8 = undefined;
    var second: [4]u8 = undefined;
    try readResponse(&reader, .ok, null, 3, &.{ &first, &second }, 5);
    try std.testing.expectEqualStrings("3", &first);
    try std.testing.expectEqualStrings("4567", &second);
}

test "partial responses require a covering Content-Range" {
    var reader: std.Io.Reader = .fixed("3456");
    var output: [4]u8 = undefined;
    try std.testing.expectError(error.InvalidContentRange, readResponse(&reader, .partial_content, null, 3, &.{&output}, 4));

    reader = .fixed("4567");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(&reader, .partial_content, .{ .start = 4, .end = 7, .total = 10 }, 3, &.{&output}, 4),
    );

    reader = .fixed("345");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(&reader, .partial_content, .{ .start = 3, .end = 5, .total = 10 }, 3, &.{&output}, 4),
    );
}

test "retry status classification is typed and 503 depends on the backend" {
    try std.testing.expectEqual(ReadFailure.timeout, classifyStatus(.request_timeout, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.throttle, classifyStatus(.too_many_requests, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.server_failure, classifyStatus(.bad_gateway, .throttle).?);
    try std.testing.expectEqual(ReadFailure.server_failure, classifyStatus(.service_unavailable, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.throttle, classifyStatus(.service_unavailable, .throttle).?);
    try std.testing.expect(classifyStatus(.not_found, .throttle) == null);
}

test "server retry delay comes from Retry-After seconds or a RateLimit reset" {
    const Head = std.http.Client.Response.Head;
    const retry_after = try Head.parse("HTTP/1.1 503 Service Unavailable\r\nretry-after: 3\r\n\r\n");
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(3), serverRetryDelay(retry_after).?);

    const rate_limit = try Head.parse("HTTP/1.1 429 Too Many Requests\r\nRateLimit: \"default\"; r=0; t=7\r\n\r\n");
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(7), serverRetryDelay(rate_limit).?);

    const http_date = try Head.parse("HTTP/1.1 503 Service Unavailable\r\nRetry-After: Wed, 21 Oct 2015 07:28:00 GMT\r\n\r\n");
    try std.testing.expect(serverRetryDelay(http_date) == null);
    const none = try Head.parse("HTTP/1.1 500 Internal Server Error\r\n\r\n");
    try std.testing.expect(serverRetryDelay(none) == null);
}
