const std = @import("std");

const AtomicReadStats = @import("base.zig").AtomicReadStats;
const ReadFailure = @import("base.zig").ReadFailure;

const log = std.log.scoped(.@"zml/vfs/range_read");

pub const ContentRange = struct {
    start: u64,
    end: u64,
    total: u64,
};

pub const RetryConfig = struct {
    max_retries: usize,
    initial_delay: std.Io.Duration,
    max_delay: std.Io.Duration,

    /// Reads the `max_retries`, `retry_initial_delay` and `retry_max_delay`
    /// fields every backend's `InitOpts` declares.
    pub fn fromOptions(opts: anytype) RetryConfig {
        const self: RetryConfig = .{
            .max_retries = opts.max_retries,
            .initial_delay = opts.retry_initial_delay,
            .max_delay = opts.retry_max_delay,
        };
        std.debug.assert(self.initial_delay.nanoseconds >= 0);
        std.debug.assert(self.max_delay.nanoseconds >= self.initial_delay.nanoseconds);
        return self;
    }
};

/// One attempt of a range read as the backend's request hook sees it.
pub const Attempt = struct {
    /// Zero-based; retries are attempt 1 and above.
    ordinal: usize,
    /// The `Range` header the loop sends, for backends that sign it.
    range: std.http.Header,
};

/// The backend's part of one GET: the URI, the authorization value and up to
/// `max_extra_headers` headers. The loop appends `Range`. Every slice must
/// stay valid until the attempt completes, so hooks write into storage that
/// outlives the call (the backend's per-read context).
pub const PreparedRequest = struct {
    uri: std.Uri,
    authorization: std.http.Client.Request.Headers.Value = .default,
    extra_headers: []const std.http.Header = &.{},
};

pub const max_extra_headers = 4;

pub const RequestSpec = struct {
    /// Backend name for log lines.
    backend: []const u8,
    /// Object identity for log lines.
    target: []const u8,
    /// What a 503 means for this backend: rate limiting on the object stores
    /// (AWS `SlowDown`, GCS), a server failure elsewhere.
    unavailable: ReadFailure,
    context: *anyopaque,
    /// Called once per attempt: S3's SigV4 signature carries a timestamp and
    /// a bearer token may have expired during the retry delay.
    prepare: *const fn (context: *anyopaque, attempt: Attempt) anyerror!PreparedRequest,
};

/// `RequestSpec.prepare` for backends whose request is the same on every
/// attempt; `context` points at the `PreparedRequest`.
pub fn prepareStatic(context: *anyopaque, _: Attempt) anyerror!PreparedRequest {
    const request: *const PreparedRequest = @ptrCast(@alignCast(context));
    return request.*;
}

const AttemptResult = union(enum) {
    success,
    retry: Retry,
};

const Retry = struct {
    failure: ReadFailure,
    delay: ?std.Io.Duration = null,
};

/// The one range read loop: one `GET` with `Range: bytes=offset-(offset+size-1)`
/// per attempt, prepared by `spec.prepare`, with serial retries inside the
/// caller's call. A retried failure is counted, then either the server-named
/// delay or full-jitter exponential backoff is slept before the next attempt;
/// `error.RetriesExhausted` after `retry.max_retries` retries. Statuses that
/// are not retried fail immediately.
pub fn performRangeRead(
    io: std.Io,
    client: *std.http.Client,
    stats: *AtomicReadStats,
    retry: RetryConfig,
    spec: RequestSpec,
    data: []const []u8,
    offset: u64,
    size: usize,
) !usize {
    if (size == 0) return 0;

    var range_buffer: [64]u8 = undefined;
    const range: std.http.Header = .{
        .name = "Range",
        .value = std.fmt.bufPrint(
            &range_buffer,
            "bytes={d}-{d}",
            .{ offset, offset + @as(u64, @intCast(size - 1)) },
        ) catch unreachable,
    };

    var attempt: usize = 0;
    while (true) : (attempt += 1) {
        const failure = switch (try performAttempt(client, stats, spec, .{ .ordinal = attempt, .range = range }, data, offset, size)) {
            .success => return size,
            .retry => |failure| failure,
        };
        stats.recordFailure(failure.failure);
        if (attempt >= retry.max_retries) return error.RetriesExhausted;

        stats.recordRetry();
        const delay = failure.delay orelse fullJitterDelay(io, retry.initial_delay, retry.max_delay, attempt);
        stats.recordRetryDelay(delay);
        io.sleep(delay, .awake) catch return error.RetriesExhausted;
    }
}

fn performAttempt(
    client: *std.http.Client,
    stats: *AtomicReadStats,
    spec: RequestSpec,
    attempt: Attempt,
    data: []const []u8,
    offset: u64,
    size: usize,
) !AttemptResult {
    const prepared = try spec.prepare(spec.context, attempt);
    std.debug.assert(prepared.extra_headers.len <= max_extra_headers);
    var headers: [max_extra_headers + 1]std.http.Header = undefined;
    @memcpy(headers[0..prepared.extra_headers.len], prepared.extra_headers);
    headers[prepared.extra_headers.len] = attempt.range;

    stats.recordAttempt();
    var req = client.request(.GET, prepared.uri, .{
        .redirect_behavior = .not_allowed,
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = prepared.authorization,
        },
        .extra_headers = headers[0 .. prepared.extra_headers.len + 1],
    }) catch |err| switch (err) {
        error.Timeout => return retryable(spec, "connect", err, .timeout),
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.HostUnreachable,
        error.NetworkUnreachable,
        error.NetworkDown,
        error.NameServerFailure,
        => return retryable(spec, "connect", err, .transient),
        else => return fatal(spec, "connect", err),
    };
    defer req.deinit();

    req.sendBodiless() catch |err| switch (err) {
        error.WriteFailed => return retryable(spec, "send headers", err, .transient),
    };

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = req.receiveHead(&redirect_buffer) catch |err| switch (err) {
        error.Timeout => return retryable(spec, "receive headers", err, .timeout),
        error.HttpConnectionClosing,
        error.HttpRequestTruncated,
        error.ReadFailed,
        error.WriteFailed,
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.HostUnreachable,
        error.NetworkUnreachable,
        error.NetworkDown,
        error.NameServerFailure,
        => return retryable(spec, "receive headers", err, .transient),
        else => return fatal(spec, "receive headers", err),
    };

    if (res.head.status != .partial_content and res.head.status != .ok) {
        const failure = classifyStatus(res.head.status, spec.unavailable) orelse {
            log.err("{s}: read of {s} failed: {s}", .{ spec.backend, spec.target, res.head.bytes });
            return error.RequestFailed;
        };
        log.warn("{s}: read of {s} failed: {s}", .{ spec.backend, spec.target, res.head.bytes });
        return .{ .retry = .{ .failure = failure, .delay = serverRetryDelay(res.head) } };
    }

    // The head bytes are released when the body reader is taken.
    const content_range = contentRange(res.head);
    readResponse(res.reader(&.{}), res.head.status, content_range, offset, data, size) catch |err| switch (err) {
        error.EndOfStream, error.ReadFailed => return retryable(spec, "read body", err, .transient),
        else => return fatal(spec, "read body", err),
    };
    stats.recordSuccess(size);
    return .success;
}

fn retryable(spec: RequestSpec, stage: []const u8, err: anyerror, failure: ReadFailure) AttemptResult {
    log.warn("{s}: {s} for {s} failed: {}", .{ spec.backend, stage, spec.target, err });
    return .{ .retry = .{ .failure = failure } };
}

fn fatal(spec: RequestSpec, stage: []const u8, err: anyerror) anyerror {
    log.err("{s}: {s} for {s} failed: {}", .{ spec.backend, stage, spec.target, err });
    return err;
}

fn contentRange(head: std.http.Client.Response.Head) ?ContentRange {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) return parseContentRange(header.value);
    }
    return null;
}

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

test "retry configuration comes from the backend init options" {
    const retry: RetryConfig = .fromOptions(.{
        .max_retries = @as(usize, 2),
        .retry_initial_delay = std.Io.Duration.fromMilliseconds(5),
        .retry_max_delay = std.Io.Duration.fromSeconds(1),
    });
    try std.testing.expectEqual(@as(usize, 2), retry.max_retries);
    try std.testing.expectEqual(std.Io.Duration.fromMilliseconds(5), retry.initial_delay);
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(1), retry.max_delay);
}
