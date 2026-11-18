const std = @import("std");

const stdx = @import("stdx");
const Collector = @import("zml-smi-2/collector").Collector;
const formats = @import("zml-smi-2/formats");
const Host = @import("zml-smi-2/host").Host;
const contract = @import("zml-smi-2/contract");
const output = @import("output.zig");
const platforms = @import("platforms.zig");
const Remote = @import("zml-smi-2/remote").Remote;
const source = @import("zml-smi-2/source");
const tui = @import("zml-smi-2/tui");

pub const std_options: std.Options = .{
    .log_scope_levels = &.{
        .{ .scope = .vaxis, .level = .warn },
    },
    // A /proc file can vanish between its open and its read, which std reports
    // as an unexpected errno, with a stack trace in debug builds.
    .unexpected_error_tracing = false,
};

const log = std.log;

const CliArgs = struct {
    top: bool = false,
    csv: bool = false,
    json: bool = false,
    api: bool = false,
    api_port: u16 = 9090,
    remotes: ?[]const u8 = null,
    prometheus_listen: ?[]const u8 = null,
    tui_refresh_rate: u16 = 100,
    poll_interval: u16 = 500,

    pub const help =
        \\ zml-smi-2 [--top] [--csv] [--json] [--api] [--api-port PORT]
        \\           [--remotes HOST,...] [--prometheus-listen HOST:PORT]
        \\         [--tui-refresh-rate MS] [--poll-interval MS]
        \\
        \\ --top               Interactive TUI mode
        \\ --csv               Output device metrics as CSV
        \\ --json              Output device metrics as JSON
        \\ --api               Serve a JSON snapshot at /metrics
        \\ --api-port PORT     JSON server port (default: 9090)
        \\ --remotes HOST,...  Collect snapshots from other zml-smi-2 instances
        \\ --prometheus-listen Expose metrics as Prometheus endpoint on HOST:PORT
        \\ --tui-refresh-rate  TUI refresh rate in ms (default: 100)
        \\ --poll-interval     Device polling interval in ms (default: 500)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    const interval: std.Io.Duration = .fromMilliseconds(args.poll_interval);

    const serving = args.prometheus_listen != null or args.api;
    var host_data: contract.HostData = .{};
    var host: Host = .init(arena, gpa, io, &host_data);
    defer host.deinit();
    var collector: Collector = .init(arena, gpa, io, .from(&host), host_data, .{
        .poll_interval_ms = args.poll_interval,
        .history = args.top and !serving,
    });
    defer collector.deinit();

    inline for (platforms.native_drivers) |Driver| {
        if (platforms.available(Driver, io)) {
            try addSource(&collector, Driver, @typeName(Driver), .{});
        }
    }

    if (args.remotes) |hosts| {
        var it = std.mem.splitScalar(u8, hosts, ',');
        while (it.next()) |peer| try addSource(&collector, Remote, peer, .{ peer, interval });
    }

    if (serving or args.top) {
        try collector.start();
    } else {
        collector.round();
        if (collector.two_readings) {
            try io.sleep(interval, .awake);
            collector.round();
        }
    }

    if (serving) {
        var servers: std.Io.Group = .init;
        defer servers.cancel(io);
        if (args.prometheus_listen) |listen| {
            log.info("serving prometheus metrics on {s}", .{listen});
            try servers.concurrent(io, output.serve, .{ &collector, try .parseLiteral(listen), .prometheus });
        }
        if (args.api) {
            log.info("serving api metrics on port {d}", .{args.api_port});
            const address: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = args.api_port } };
            try servers.concurrent(io, output.serve, .{ &collector, address, .json });
        }
        try servers.await(io);
        return;
    }

    if (args.csv or args.json) {
        var buffer: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writer(io, &buffer);
        try output.write(&stdout.interface, &collector, if (args.csv) formats.csv.write else formats.json.write);
        return stdout.interface.flush();
    }

    var state: tui.data.SystemState = .{
        .provider = .{ .context = &collector, .capture = captureTui },
        .devices = try platforms.devices(arena, collector.devices.items),
        .logos = platforms.logos,
        .tui_refresh_rate = args.tui_refresh_rate,
    };

    if (args.top) {
        try tui.top.run(gpa, io, init.environ_map, &state);
    } else {
        try tui.print.run(arena, io, init.environ_map, &state);
    }
}

fn addSource(collector: *Collector, comptime Backend: type, label: []const u8, args: anytype) !void {
    openSource(collector, Backend, args) catch |err| {
        if (err == error.OutOfMemory) return err;
        log.warn("{s} skipped: {t}", .{ label, err });
    };
}

fn openSource(collector: *Collector, comptime Backend: type, args: anytype) !void {
    const backend = try collector.arena.create(Backend);
    backend.* = try @call(.auto, Backend.open, .{ collector.arena, collector.gpa, collector.io } ++ args);
    const reader = source.Source.from(backend);
    errdefer reader.close(reader.context);
    try collector.add(reader);
}

fn captureTui(context: *anyopaque, arena: std.mem.Allocator) std.mem.Allocator.Error!contract.Frame {
    const collector: *Collector = @ptrCast(@alignCast(context));
    return collector.capture(arena, true);
}
