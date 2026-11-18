const std = @import("std");
const formats = @import("zml-smi-2/formats");
const Host = @import("zml-smi-2/host").Host;
const Collector = @import("zml-smi-2/collector").Collector(Host);

const log = std.log.scoped(.@"zml-smi-2/api");

pub const Format = enum {
    /// The API: what --json prints, which --remotes reads.
    json,
    prometheus,
};

const Serializer = *const fn (*std.Io.Writer, formats.Snapshot) std.Io.Writer.Error!void;

pub fn write(writer: *std.Io.Writer, collector: *Collector, serialize: Serializer) !void {
    var arena: std.heap.ArenaAllocator = .init(collector.gpa);
    defer arena.deinit();
    const frame = try collector.capture(arena.allocator(), false);
    try serialize(writer, .{
        .devices = collector.devices.items,
        .samples = frame.samples,
        .processes = frame.processes,
        .host = frame.host,
    });
}

/// Answers every request on `address` with the latest values, until canceled.
pub fn serve(collector: *Collector, address: std.Io.net.IpAddress, format: Format) void {
    const io = collector.io;
    var server = address.listen(io, .{ .reuse_address = true }) catch |err| {
        log.err("{t} server cannot listen on {f}: {t}", .{ format, address, err });
        return;
    };
    defer server.deinit(io);

    var connections: std.Io.Group = .init;
    defer connections.cancel(io);
    while (true) {
        const stream = server.accept(io) catch return;
        connections.concurrent(io, answer, .{ collector, stream, format }) catch {
            stream.close(io);
            return;
        };
    }
}

fn answer(collector: *Collector, stream: std.Io.net.Stream, format: Format) void {
    const io = collector.io;
    defer stream.close(io);
    var read_buffer: [8192]u8 = undefined;
    var write_buffer: [4096]u8 = undefined;
    var reader = stream.reader(io, &read_buffer);
    var writer = stream.writer(io, &write_buffer);
    var http: std.http.Server = .init(&reader.interface, &writer.interface);

    // The body is written to memory first, so a slow client never holds a lock.
    var body: std.Io.Writer.Allocating = .init(collector.gpa);
    defer body.deinit();

    while (true) {
        var request = http.receiveHead() catch return;
        if (request.head.method != .GET) {
            request.respond("Method Not Allowed\n", .{ .status = .method_not_allowed }) catch return;
            continue;
        }

        body.clearRetainingCapacity();
        write(&body.writer, collector, switch (format) {
            .json => formats.json.write,
            .prometheus => formats.prometheus.write,
        }) catch return;

        request.respond(body.written(), .{ .extra_headers = switch (format) {
            .json => &.{
                .{ .name = "content-type", .value = "application/json" },
                .{ .name = "access-control-allow-origin", .value = "*" },
            },
            .prometheus => &.{.{ .name = "content-type", .value = "text/plain; version=0.0.4" }},
        } }) catch return;
    }
}
