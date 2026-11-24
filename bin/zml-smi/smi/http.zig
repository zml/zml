const std = @import("std");
const smi = @import("smi.zig");

const log = std.log.scoped(.@"zml-smi/http");

/// Serves the collector over HTTP, as JSON or as Prometheus text.
pub const Server = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    collector: *smi.Collector,
    format: Format,

    pub const Format = enum { json, prometheus };

    pub fn serve(self: Server, listener: *std.Io.net.Server) void {
        const io = self.io;
        var connections: std.Io.Group = .init;
        defer connections.cancel(io);

        while (true) {
            const stream = listener.accept(io) catch |err| switch (err) {
                error.Canceled, error.SocketNotListening => return,
                else => continue,
            };

            connections.concurrent(io, answer, .{ self, stream }) catch {
                stream.close(io);
                return;
            };
        }
    }

    fn answer(self: Server, stream: std.Io.net.Stream) void {
        const io = self.io;
        defer stream.close(io);

        var read_buffer: [8192]u8 = undefined;
        var write_buffer: [4096]u8 = undefined;
        var reader = stream.reader(io, &read_buffer);
        var writer = stream.writer(io, &write_buffer);
        var http: std.http.Server = .init(&reader.interface, &writer.interface);

        var request = http.receiveHead() catch return;
        // Closing the connection skips the body, which std asserts has a length.
        if (request.head.method != .GET) {
            request.respond("Method Not Allowed\n", .{ .status = .method_not_allowed, .keep_alive = false }) catch {};
            return;
        }

        // The body is written to memory first, so a slow client never holds a lock.
        var body: std.Io.Writer.Allocating = .init(self.gpa);
        defer body.deinit();

        const written = switch (self.format) {
            .json => smi.json.write(&body.writer, self.collector),
            .prometheus => smi.prometheus.write(&body.writer, self.collector),
        };
        written catch return;

        request.respond(body.written(), .{
            .keep_alive = false,
            .extra_headers = switch (self.format) {
                .json => &.{
                    .{ .name = "content-type", .value = "application/json" },
                    .{ .name = "access-control-allow-origin", .value = "*" },
                },
                .prometheus => &.{.{ .name = "content-type", .value = "text/plain" }},
            },
        }) catch {};
    }
};

/// A remote zml-smi read through its JSON API, as a platform.
pub const Client = struct {
    platform: smi.Platform,
    io: std.Io,
    url: []const u8,
    rows: ?[]const smi.ProcessInfo = null,

    const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes };

    pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8) !Client {
        if (!std.mem.startsWith(u8, host, "http://") and !std.mem.startsWith(u8, host, "https://")) {
            log.err("{s}: a remote host starts with http:// or https://", .{host});
            return error.InvalidUrl;
        }
        const url = try std.fmt.allocPrint(arena, "{s}/metrics", .{host});

        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const response = try fetch(arena, scratch.allocator(), io, url);

        const devices = try arena.alloc(smi.Device, response.devices.len);
        for (devices, response.devices) |*device, row| {
            device.* = row.device;
            device.remote = true;
        }
        return .{ .platform = .{ .devices = devices, .vtable = &vtable }, .io = io, .url = url };
    }

    fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
        const self: *Client = @alignCast(@fieldParentPtr("platform", platform));

        self.rows = null;
        const response = fetch(scratch, scratch, self.io, self.url) catch return;

        // TODO: read the new devices once the collector supports a device list that changes.
        if (response.devices.len != samples.len) std.debug.panic("{s}: the devices changed since zml-smi started. Please restart.", .{self.url});
        for (samples, response.devices) |*sample, row| {
            sample.* = row.sample;
        }

        self.rows = response.processes;
    }

    fn processes(platform: *smi.Platform, _: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
        const self: *Client = @alignCast(@fieldParentPtr("platform", platform));

        for (self.rows orelse return error.Unavailable) |process| {
            if (process.device_idx >= self.platform.devices.len) continue;
            var row = process;
            row.remote = true;
            row.username = try out.dupe(row.username);
            row.comm = try out.dupe(row.comm);
            try out.append(row);
        }
    }

    /// The HTTP client lives in `scratch`, the body and the response in `allocator`.
    fn fetch(allocator: std.mem.Allocator, scratch: std.mem.Allocator, io: std.Io, url: []const u8) !smi.json.Response([]const smi.json.Row) {
        var client: std.http.Client = .{ .allocator = scratch, .io = io };
        defer client.deinit();

        var body: std.Io.Writer.Allocating = .init(allocator);
        const result = try client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer });
        if (result.status != .ok) return error.HttpError;

        return std.json.parseFromSliceLeaky(smi.json.Response([]const smi.json.Row), allocator, body.written(), .{ .ignore_unknown_fields = true });
    }
};
