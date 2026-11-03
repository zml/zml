const std = @import("std");

const MockServer = @import("mock_server.zig").MockServer;
const range_read = @import("range_read.zig");
const vfs_request = @import("request.zig");

/// A retry configuration that never sleeps, for the tests that count
/// attempts rather than time.
fn instantRetries(max_retries: usize) vfs_request.Retry {
    return .{
        .max_retries = max_retries,
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
    };
}

test "the shared range loop prepares the request once per attempt" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const request_size = 3 * 1024;

    var object: [request_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 7);
    var output: [request_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{ .fail_first_gets = 2 });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    // An S3-style request: as a SigV4 signature must be, it is prepared
    // again by the hook for every attempt.
    const Signed = struct {
        uri: std.Uri,
        prepared: usize = 0,

        fn prepare(self: *@This(), _: vfs_request.Attempt, range: std.http.Header) anyerror!range_read.PreparedRequest {
            try std.testing.expectEqualStrings("Range", range.name);
            self.prepared += 1;
            return .{ .uri = self.uri };
        }
    };
    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    var signed: Signed = .{ .uri = try .parse(url) };
    const spec: range_read.RangeSpec = .{ .request = .{ .target = url } };
    const buffers = [_][]u8{ output[0..1024], output[1024..] };

    // No retry budget: one attempt, one hook call, the failure is counted.
    var no_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(0));
    try std.testing.expectError(
        error.RetriesExhausted,
        range_read.performRangeRead(&no_retry, spec, &signed, Signed.prepare, &buffers, 0, object.len),
    );
    try std.testing.expectEqual(@as(usize, 1), signed.prepared);
    try std.testing.expectEqual(@as(u64, 1), no_retry.stats.server_failures.load(.acquire));

    signed.prepared = 0;
    var one_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(1));
    try std.testing.expectEqual(
        object.len,
        try range_read.performRangeRead(&one_retry, spec, &signed, Signed.prepare, &buffers, 0, object.len),
    );
    try std.testing.expectEqual(@as(usize, 2), signed.prepared);
    try std.testing.expectEqualSlices(u8, &object, &output);

    try server.stop(io);

    // One GET per hook call.
    try std.testing.expectEqual(@as(usize, 3), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));
    const stats = &one_retry.stats;
    try std.testing.expectEqual(@as(u64, 2), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.server_failures.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
}
