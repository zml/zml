//! The Range specifics of a governed request: the `Range` header, the
//! `Content-Range` validation and the scatter into the caller's buffers.
//! Everything else (the hold, the retries, the classification) is in
//! `request.zig`.

const std = @import("std");

const request = @import("request.zig");

const log = std.log.scoped(.@"zml/vfs/range_read");

pub const ContentRange = struct {
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

/// One range read: what identifies it, and the hook that prepares each
/// attempt. The hook is called once per attempt: S3's SigV4 signature covers
/// the `Range` header and a timestamp, a bearer token may have expired during
/// a hold, and a signed download URL may need re-resolving after a 403.
pub const RangeSpec = struct {
    request: request.RequestSpec,
    context: *anyopaque,
    prepare: *const fn (
        context: *anyopaque,
        attempt: request.Attempt,
        range: std.http.Header,
    ) anyerror!PreparedRequest,
};

/// `RangeSpec.prepare` for backends whose request is the same on every
/// attempt; `context` points at the `PreparedRequest`.
pub fn prepareStatic(context: *anyopaque, _: request.Attempt, _: std.http.Header) anyerror!PreparedRequest {
    const prepared: *const PreparedRequest = @ptrCast(@alignCast(context));
    return prepared.*;
}

/// One `GET` with `Range: bytes=offset-(offset+size-1)` per attempt through
/// the governed loop.
/// One caller owns one source credit through every retry, hold and sleep.
/// Adding backend-local parallel readers would multiply the loader's chosen
/// width and defeat its memory/backpressure bounds. Artificial S3Proxy tests
/// with a per-request bandwidth cap rewarded very high concurrency (16 MiB at
/// width 96 reached ~11.5 GiB/s), but real AWS plateaued near 950 MiB/s at
/// widths 24-128; the proxy is not evidence for production defaults.
pub fn performRangeRead(
    ctx: request.Context,
    spec: RangeSpec,
    data: []const []u8,
    offset: u64,
    size: usize,
) !usize {
    if (size == 0) return 0;

    var range_buffer: [64]u8 = undefined;
    var state: RangeRead = .{
        .ctx = ctx,
        .spec = spec,
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
    try request.perform(void, ctx, spec.request, &state, RangeRead.attempt);
    ctx.stats.recordSuccess(size);
    return size;
}

const RangeRead = struct {
    ctx: request.Context,
    spec: RangeSpec,
    range: std.http.Header,
    data: []const []u8,
    offset: u64,
    size: usize,

    fn attempt(self: *RangeRead, current: request.Attempt) anyerror!request.Outcome(void) {
        const prepared = try self.spec.prepare(self.spec.context, current, self.range);
        std.debug.assert(prepared.extra_headers.len <= max_extra_headers);
        var headers: [max_extra_headers + 1]std.http.Header = undefined;
        @memcpy(headers[0..prepared.extra_headers.len], prepared.extra_headers);
        headers[prepared.extra_headers.len] = self.range;

        var head_buffer: [8 * 1024]u8 = undefined;
        return request.exchange(void, self.ctx, prepared.uri, .{
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = prepared.authorization,
            },
            .extra_headers = headers[0 .. prepared.extra_headers.len + 1],
            .head_buffer = &head_buffer,
        }, self.spec.request, self, RangeRead.consume);
    }

    fn consume(self: *RangeRead, res: *std.http.Client.Response) anyerror!void {
        // The head bytes are released when the body reader is taken.
        const content_range = contentRange(res.head);
        return readResponse(
            res.reader(&.{}),
            res.head.status,
            content_range,
            self.offset,
            self.data,
            self.size,
        ) catch |err| switch (err) {
            // The loop retries a truncated body; anything else is a
            // malformed answer to a valid request.
            error.EndOfStream, error.ReadFailed => err,
            else => {
                log.err("{s}: read of {s} returned an unusable body: {}", .{
                    self.spec.request.backend,
                    self.spec.request.target,
                    err,
                });
                return err;
            },
        };
    }
};

fn contentRange(head: std.http.Client.Response.Head) ?ContentRange {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) return parseContentRange(header.value);
    }
    return null;
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
