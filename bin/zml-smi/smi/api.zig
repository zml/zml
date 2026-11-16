const std = @import("std");
const smi = @import("smi.zig");

const log = std.log.scoped(.@"zml-smi/api");

pub const Format = enum {
    /// The API: what --json prints, which --remotes reads.
    json,
    prometheus,
};

/// Answers every request on `address` with the latest values, until canceled.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, address: std.Io.net.IpAddress, collector: *smi.Collector, format: Format) void {
    var server = address.listen(io, .{ .reuse_address = true }) catch |err| {
        log.err("{t} server cannot listen on {f}: {t}", .{ format, address, err });
        return;
    };
    defer server.deinit(io);

    var connections: std.Io.Group = .init;
    defer connections.cancel(io);
    while (true) {
        const stream = server.accept(io) catch return;
        connections.concurrent(io, answer, .{ gpa, io, stream, collector, format }) catch {
            stream.close(io);
            return;
        };
    }
}

fn answer(gpa: std.mem.Allocator, io: std.Io, stream: std.Io.net.Stream, collector: *smi.Collector, format: Format) void {
    defer stream.close(io);
    var read_buffer: [8192]u8 = undefined;
    var write_buffer: [4096]u8 = undefined;
    var reader = stream.reader(io, &read_buffer);
    var writer = stream.writer(io, &write_buffer);
    var http: std.http.Server = .init(&reader.interface, &writer.interface);

    // The body is written to memory first, so a slow client never holds a lock.
    var body: std.Io.Writer.Allocating = .init(gpa);
    defer body.deinit();

    while (true) {
        var request = http.receiveHead() catch return;
        if (format == .prometheus and request.head.method != .GET) {
            request.respond("Method Not Allowed\n", .{ .status = .method_not_allowed }) catch return;
            continue;
        }

        body.clearRetainingCapacity();
        smi.write(&body.writer, collector, switch (format) {
            .json => smi.json.write,
            .prometheus => smi.prometheus.write,
        }) catch return;

        request.respond(body.written(), .{ .extra_headers = switch (format) {
            .json => &.{
                .{ .name = "content-type", .value = "application/json" },
                .{ .name = "access-control-allow-origin", .value = "*" },
            },
            .prometheus => &.{.{ .name = "content-type", .value = "text/plain" }},
        } }) catch return;
    }
}
