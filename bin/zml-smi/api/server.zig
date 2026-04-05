const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const json = @import("zml-smi/json");

pub const Server = struct {
    io: std.Io,
    devices: []const *DeviceInfo,
    tcp: std.Io.net.Server,
    connection_group: std.Io.Group = .init,

    pub fn init(io: std.Io, port: u16, devices: []const *DeviceInfo) !Server {
        const address: std.Io.net.IpAddress = .{ .ip4 = .{
            .bytes = .{ 0, 0, 0, 0 },
            .port = port,
        } };
        var tcp = try address.listen(io, .{ .reuse_address = true });
        errdefer tcp.deinit(io);

        return .{
            .io = io,
            .devices = devices,
            .tcp = tcp,
        };
    }

    pub fn deinit(self: *Server) void {
        self.connection_group.await(self.io) catch {};
        self.tcp.deinit(self.io);
    }

    pub fn serve(self: *Server) !void {
        while (true) {
            const stream = self.tcp.accept(self.io) catch break;
            try self.connection_group.concurrent(self.io, Server.onConnection, .{ self, stream });
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

            var body_buf: [4096]u8 = undefined;
            var body_writer = try request.respondStreaming(&body_buf, .{
                .respond_options = .{
                    .status = .ok,
                    .extra_headers = &.{
                        .{ .name = "content-type", .value = "application/json" },
                        .{ .name = "access-control-allow-origin", .value = "*" },
                    },
                },
            });

            try json.write(&body_writer.writer, self.devices);
            try body_writer.end();
        }
    }
};
