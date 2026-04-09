const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const HostInfo = smi_info.host_info.HostInfo;
const exposition = @import("exposition.zig");

pub const Server = struct {
    io: std.Io,
    devices: []const *DeviceInfo,
    host: *const HostInfo,
    tcp: std.Io.net.Server,
    connection_group: std.Io.Group = .init,

    pub fn init(io: std.Io, listen: []const u8, devices: []const *DeviceInfo, host: *const HostInfo) !Server {
        const address = parseListenAddress(listen) orelse return error.InvalidAddress;
        var tcp = try address.listen(io, .{ .reuse_address = true });
        errdefer tcp.deinit(io);

        return .{
            .io = io,
            .devices = devices,
            .host = host,
            .tcp = tcp,
        };
    }

    pub fn run(io: std.Io, listen: []const u8, devices: []const *DeviceInfo, host_info: *const HostInfo) void {
        var self = init(io, listen, devices, host_info) catch |err| {
            std.log.err("prometheus server failed to start: {s}", .{@errorName(err)});
            return;
        };
        defer self.deinit();
        self.serve();
    }

    pub fn deinit(self: *Server) void {
        self.connection_group.await(self.io) catch {};
        self.tcp.deinit(self.io);
    }

    pub fn serve(self: *Server) void {
        while (true) {
            const stream = self.tcp.accept(self.io) catch break;
            self.connection_group.concurrent(self.io, Server.onConnection, .{ self, stream }) catch break;
        }
    }

    fn onConnection(self: *Server, stream: std.Io.net.Stream) std.Io.Cancelable!void {
        defer stream.close(self.io);
        self.handleConn(stream) catch {};
    }

    fn handleConn(self: *Server, stream: std.Io.net.Stream) !void {
        var read_buf: [8192]u8 = undefined;
        var reader = stream.reader(self.io, &read_buf);

        var write_buf: [4096]u8 = undefined;
        var writer = stream.writer(self.io, &write_buf);

        var http_server: std.http.Server = .init(&reader.interface, &writer.interface);

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };

            if (request.head.method != .GET) {
                try request.respond("Method Not Allowed\n", .{ .status = .method_not_allowed });
                continue;
            }

            if (std.mem.eql(u8, request.head.target, "/metrics")) {
                var body_buf: [4096]u8 = undefined;
                var body_writer = try request.respondStreaming(&body_buf, .{
                    .respond_options = .{
                        .status = .ok,
                        .extra_headers = &.{
                            .{ .name = "content-type", .value = "text/plain" },
                        },
                    },
                });

                try exposition.write(&body_writer.writer, self.devices, self.host);
                try body_writer.end();
            } else if (std.mem.eql(u8, request.head.target, "/")) {
                try request.respond("OK\n", .{ .status = .ok });
            } else {
                try request.respond("Not Found\n", .{ .status = .not_found });
            }
        }
    }

    fn parseListenAddress(listen: []const u8) ?std.Io.net.IpAddress {
        var ttk = std.mem.tokenizeScalar(u8, listen, ':');

        const host_str = ttk.next() orelse return null;
        const port_str = ttk.next() orelse return null;
        const port = std.fmt.parseInt(u16, port_str, 10) catch return null;

        const ip_bytes: [4]u8 = if (std.mem.eql(u8, host_str, "localhost"))
            .{ 127, 0, 0, 1 }
        else
            (std.Io.net.IpAddress.parseIp4(host_str, 0) catch return null).ip4.bytes;

        return .{ .ip4 = .{ .bytes = ip_bytes, .port = port } };
    }
};
