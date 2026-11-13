const std = @import("std");

const stdx = @import("stdx");
const smi = @import("zml-smi");
const tui = @import("zml-smi/tui");

const platforms = @import("platforms.zig");

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
        \\ zml-smi [--top] [--csv] [--json] [--prometheus-listen HOST:PORT]
        \\         [--tui-refresh-rate MS] [--poll-interval MS]
        \\
        \\ --top               Interactive TUI mode
        \\ --csv               Output device metrics as CSV
        \\ --json              Output device metrics as JSON
        \\ --prometheus-listen Expose metrics as Prometheus endpoint on HOST:PORT
        \\ --tui-refresh-rate  TUI refresh rate in ms (default: 100)
        \\ --poll-interval     Device polling interval in ms (default: 500)
        \\
    ;

    //  [--api] [--port PORT] [--remotes HOST,...]
    //  \\ --api               Expose device metrics as HTTP JSON endpoint
    //  \\ --api-port          API server port (default: 9090)
    //  \\ --remotes           Add devices from remote hosts (comma-separated URLs)
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    const interval: std.Io.Duration = .fromMilliseconds(args.poll_interval);

    var collector: smi.Collector = .init(arena, gpa, io, .{ .poll_interval_ms = args.poll_interval });
    defer collector.deinit();

    inline for (platforms.all) |Platform| {
        if (Platform.detect(io)) {
            const platform = try arena.create(Platform);
            if (Platform.init(arena, gpa, io)) |value| {
                platform.* = value;
                try collector.add(&platform.platform);
            } else |err| log.warn("{s} skipped: {t}", .{ @typeName(Platform), err });
        }
    }

    if (args.remotes) |hosts| {
        var it = std.mem.splitScalar(u8, hosts, ',');
        while (it.next()) |host| {
            const remote = try arena.create(smi.Remote);
            if (smi.Remote.open(arena, gpa, io, host, interval)) |value| {
                remote.* = value;
                try collector.add(&remote.platform);
            } else |err| log.warn("{s} skipped: {t}", .{ host, err });
        }
    }

    const serving = args.prometheus_listen != null or args.api;
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
            try servers.concurrent(io, smi.api.serve, .{ gpa, io, try .parseLiteral(listen), &collector, .prometheus });
        }
        if (args.api) {
            log.info("serving api metrics on port {d}", .{args.api_port});
            const address: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = args.api_port } };
            try servers.concurrent(io, smi.api.serve, .{ gpa, io, address, &collector, .json });
        }
        try servers.await(io);
        return;
    }

    if (args.csv or args.json) {
        var buffer: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writer(io, &buffer);
        try if (args.csv) smi.csv.write(&stdout.interface, &collector) else smi.json.write(&stdout.interface, &collector);
        return stdout.interface.flush();
    }

    var state: tui.data.SystemState = try .init(.{
        .collector = &collector,
        .tui_refresh_rate = args.tui_refresh_rate,
        .gpa = gpa,
        .arena = arena,
        .io = io,
    });
    defer state.deinit(arena);

    if (args.top) {
        try tui.top.run(gpa, io, init.environ_map, &state);
    } else {
        try tui.print.run(arena, io, init.environ_map, &state);
    }
}
