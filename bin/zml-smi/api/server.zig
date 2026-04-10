const std = @import("std");
const builtin = @import("builtin");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessInfo = smi_info.process_info.ProcessInfo;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(ProcessInfo));
const json = @import("zml-smi/json");
const ProcessEnricher = if (builtin.os.tag == .macos)
    @import("zml-smi/platforms/macos").process.ProcessEnricher
else
    @import("zml-smi/platforms/linux").process.ProcessEnricher;

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    devices: []const *DeviceInfo,
    process_lists: []const *const ProcessDoubleBuffer,
    enricher: *ProcessEnricher,
    tcp: std.Io.net.Server,
    connection_group: std.Io.Group = .init,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, port: u16, devices: []const *DeviceInfo, process_lists: []const *const ProcessDoubleBuffer, enricher: *ProcessEnricher) !Server {
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
            .process_lists = process_lists,
            .enricher = enricher,
            .tcp = tcp,
        };
    }

    pub fn run(allocator: std.mem.Allocator, io: std.Io, port: u16, devices: []const *DeviceInfo, process_lists: []const *const ProcessDoubleBuffer, enricher: *ProcessEnricher) void {
        var self = init(allocator, io, port, devices, process_lists, enricher) catch |err| {
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

        var procs: std.ArrayList(ProcessInfo) = .empty;
        defer procs.deinit(self.allocator);

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };

            procs.clearRetainingCapacity();
            for (self.process_lists) |pl| {
                procs.appendSlice(self.allocator, pl.front().items) catch {};
            }
            self.enricher.enrich(self.io, procs.items);

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

            try json.write(&body_writer.writer, self.devices, procs.items);
            try body_writer.end();
        }
    }
};
