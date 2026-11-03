//! The Range specifics of a governed request: the `Range` header, the
//! `Content-Range` validation and the scatter into the caller's buffers.
//! Everything else (the hold, the retries, the classification) is in
//! `request.zig`.

const std = @import("std");

const request = @import("request.zig");

const ContentRange = struct {
    start: u64,
    end: u64,
    total: u64,
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

/// One range read: what identifies it, and whether it follows a redirect.
pub const RangeSpec = struct {
    request: request.RequestSpec,
    /// The object stores and a resolved download URL never redirect a
    /// read; a plain HTTP server may.
    redirects: request.Redirects = .forbid,
};

/// The `prepare` of a request that is the same on every attempt: a plain
/// GET of `uri`.
pub fn static(uri: *const std.Uri, _: request.Attempt, _: std.http.Header) anyerror!PreparedRequest {
    return .{ .uri = uri.* };
}

/// One `GET` with `Range: bytes=offset-(offset+size-1)` per attempt through
/// the governed loop, where `size` is what `data` holds of the `file_size`
/// bytes past `offset`; nothing is sent when it is 0. `prepare` is called
/// once per attempt: S3's SigV4 signature covers the `Range` header and a
/// timestamp, a bearer token may have expired, and a signed download URL
/// may need re-resolving after a 403.
/// The caller holds its read slot through every retry and sleep;
/// backend-local parallel readers would multiply the caller's width and
/// defeat its memory bounds.
pub fn performRangeRead(
    governor: *request.Governor,
    spec: RangeSpec,
    context: anytype,
    comptime prepare: fn (@TypeOf(context), request.Attempt, std.http.Header) anyerror!PreparedRequest,
    data: []const []u8,
    offset: u64,
    file_size: u64,
) !usize {
    const size = readSize(file_size, offset, data);
    if (size == 0) return 0;

    const RangeRead = struct {
        context: @TypeOf(context),
        redirects: request.Redirects,
        range: std.http.Header,
        data: []const []u8,
        offset: u64,
        size: usize,
        progress: *std.atomic.Value(u64) = undefined,

        fn attempt(self: *@This(), current: request.Attempt) anyerror!request.Outcome(void) {
            self.progress = current.progress;
            const prepared = try prepare(self.context, current, self.range);
            std.debug.assert(prepared.extra_headers.len <= max_extra_headers);
            var headers: [max_extra_headers + 1]std.http.Header = undefined;
            @memcpy(headers[0..prepared.extra_headers.len], prepared.extra_headers);
            headers[prepared.extra_headers.len] = self.range;
            return request.exchange(void, current, prepared.uri, .{
                .authorization = prepared.authorization,
                .extra_headers = headers[0 .. prepared.extra_headers.len + 1],
                .redirects = self.redirects,
            }, self, consume);
        }

        fn consume(self: *@This(), res: *std.http.Client.Response) anyerror!void {
            // The head bytes are released when the body reader is taken.
            const content_range = contentRange(res.head);
            return readResponse(res.reader(&.{}), res.head.status, content_range, self.offset, self.data, self.size, self.progress);
        }
    };

    var range_buffer: [64]u8 = undefined;
    var state: RangeRead = .{
        .context = context,
        .redirects = spec.redirects,
        .range = .{
            .name = "Range",
            .value = std.fmt.bufPrint(
                &range_buffer,
                "bytes={d}-{d}",
                .{ offset, offset + @as(u64, @intCast(size - 1)) },
            ) catch unreachable,
        },
        .data = data,
        .offset = offset,
        .size = size,
    };
    try request.perform(void, governor, spec.request, &state, RangeRead.attempt);
    governor.stats.recordSuccess(size);
    return size;
}

fn contentRange(head: std.http.Client.Response.Head) ?ContentRange {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) return parseContentRange(header.value);
    }
    return null;
}

fn parseContentRange(value: []const u8) ?ContentRange {
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

fn readSize(file_size: u64, offset: u64, data: []const []u8) usize {
    if (offset >= file_size) return 0;

    var requested: usize = 0;
    for (data) |buf| requested +|= buf.len;
    return @intCast(@min(file_size - offset, requested));
}

/// Validates the response against the requested range, discards the prefix
/// of a `200` that ignored `Range`, then scatters `read_size` bytes, counting
/// what it receives in `progress`.
fn readResponse(
    reader: *std.Io.Reader,
    status: std.http.Status,
    content_range: ?ContentRange,
    offset: u64,
    data: []const []u8,
    read_size: usize,
    progress: ?*std.atomic.Value(u64),
) !void {
    std.debug.assert(read_size != 0);

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

    var skip = offset - response_start;
    while (skip != 0) {
        const len = @min(skip, progress_slice);
        try reader.discardAll(len);
        count(progress, len);
        skip -= len;
    }
    try readScatter(reader, data, read_size, progress);
}

/// Bytes read between two progress counts.
const progress_slice = 1 << 20;

fn count(progress: ?*std.atomic.Value(u64), len: u64) void {
    if (progress) |counter| _ = counter.fetchAdd(len, .monotonic);
}

fn readScatter(reader: *std.Io.Reader, data: []const []u8, len: usize, progress: ?*std.atomic.Value(u64)) !void {
    var remaining = len;
    for (data) |buf| {
        var destination = buf[0..@min(remaining, buf.len)];
        remaining -= destination.len;
        while (destination.len != 0) {
            const slice = destination[0..@min(destination.len, progress_slice)];
            try reader.readSliceAll(slice);
            count(progress, slice.len);
            destination = destination[slice.len..];
        }
    }
    if (remaining != 0) return error.UnexpectedEndOfOutput;
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
        null,
    );
    try std.testing.expectEqualStrings("34", &first);
    try std.testing.expectEqualStrings("567", &second);
}

test "200 responses that ignore Range are positioned and scattered" {
    var reader: std.Io.Reader = .fixed("0123456789");
    var first: [1]u8 = undefined;
    var second: [4]u8 = undefined;
    try readResponse(&reader, .ok, null, 3, &.{ &first, &second }, 5, null);
    try std.testing.expectEqualStrings("3", &first);
    try std.testing.expectEqualStrings("4567", &second);
}

test "partial responses require a covering Content-Range" {
    var reader: std.Io.Reader = .fixed("3456");
    var output: [4]u8 = undefined;
    try std.testing.expectError(error.InvalidContentRange, readResponse(&reader, .partial_content, null, 3, &.{&output}, 4, null));

    reader = .fixed("4567");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(&reader, .partial_content, .{ .start = 4, .end = 7, .total = 10 }, 3, &.{&output}, 4, null),
    );

    reader = .fixed("345");
    try std.testing.expectError(
        error.InvalidContentRange,
        readResponse(&reader, .partial_content, .{ .start = 3, .end = 5, .total = 10 }, 3, &.{&output}, 4, null),
    );
}
