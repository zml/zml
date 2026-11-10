const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessInfo = smi_info.process_info.ProcessInfo;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(ProcessInfo));
const json = @import("zml-smi/json");

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    devices: []const *DeviceInfo,
    processes: *const ProcessDoubleBuffer,
    tcp: std.Io.net.Server,
    connection_group: std.Io.Group = .init,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, port: u16, devices: []const *DeviceInfo, processes: *const ProcessDoubleBuffer) !Server {
        const address: std.Io.net.IpAddress = .{ .ip4 = .{
            .bytes = .{ 0, 0, 0, 0 },
            .port = port,
        } };
        var tcp = try address.listen(io, .{ .reuse_address = true });
        errdefer tcp.deinit(io);

        return .{
            .allocator = allocator,
            .io = io,
            .devices = devices,
            .processes = processes,
            .tcp = tcp,
        };
    }

    pub fn run(allocator: std.mem.Allocator, io: std.Io, port: u16, devices: []const *DeviceInfo, processes: *const ProcessDoubleBuffer) void {
        var self = init(allocator, io, port, devices, processes) catch |err| {
            std.log.err("api server failed to start: {s}", .{@errorName(err)});
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

        var body: std.Io.Writer.Allocating = .init(self.allocator);
        defer body.deinit();

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };

            // The processes are only valid until they are polled again, so the body
            // is written out before it is sent: a slow client cannot outlive them.
            body.clearRetainingCapacity();
            try json.write(&body.writer, self.devices, self.processes.front().items);

            try request.respond(body.written(), .{
                .status = .ok,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "application/json" },
                    .{ .name = "access-control-allow-origin", .value = "*" },
                },
            });
        }
    }
};
