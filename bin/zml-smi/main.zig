const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

pub const std_options: std.Options = .{
    .log_scope_levels = &.{
        .{ .scope = .vaxis, .level = .warn },
    },
};

const Collector = @import("zml-smi/collector").Collector;
const smi_info = @import("zml-smi/info");
const HostInfo = smi_info.host_info.HostInfo;
const platform = if (builtin.os.tag == .macos)
    @import("zml-smi/platforms/macos")
else
    @import("zml-smi/platforms/linux");
const ProcessEnricher = platform.process.ProcessEnricher;
const host = platform.metrics;
const c = @import("c");
const csv = @import("zml-smi/csv");
const json = @import("zml-smi/json");
const api = @import("zml-smi/api");
const prometheus = @import("zml-smi/prometheus");
const smi_tui = @import("zml-smi/tui");
const tui = smi_tui.top;
const static_print = smi_tui.print;
const data = smi_tui.data;

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

const device_backends = switch (builtin.os.tag) {
    .macos => .{},
    .linux => switch (builtin.cpu.arch) {
        .x86_64 => .{
            .{ .cuda, @import("zml-smi/platforms/nvml") },
            .{ .rocm, @import("zml-smi/platforms/amdsmi") },
            .{ .neuron, @import("zml-smi/platforms/neuron") },
            .{ .tpu, @import("zml-smi/platforms/tpu") },
        },
        .aarch64 => .{
            .{ .cuda, @import("zml-smi/platforms/nvml") },
        },
        else => unreachable,
    },
    else => unreachable,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const targets = detect(io);

    var collector: Collector = .init(arena, gpa, io, .{
        .poll_interval_ms = args.poll_interval,
        .poll_only = !args.top and args.prometheus_listen == null and !args.api,
    });
    defer collector.deinit();

    var host_info: HostInfo = .{ .values = .{ .{}, .{} } };
    try host.init(&collector, &host_info);

    var enricher: ProcessEnricher = try .init(gpa, io);
    defer enricher.deinit();

    if (args.api) {
        std.log.info("serving api metrics on port {d}", .{args.api_port});
    }

    if (args.prometheus_listen) |listen| {
        std.log.info("serving prometheus metrics on {s}", .{listen});
    }

    inline for (device_backends) |entry| {
        const target, const backend = entry;
        if (targets.contains(target)) {
            backend.start(&collector) catch |err| {
                std.log.err("{s} skipped: {s}", .{ @tagName(target), @errorName(err) });
            };
        }
    }

    var api_group: std.Io.Group = .init;

    if (args.remotes) |hosts| {
        try api.addRemotes(&collector, hosts);
    }

    var state = try data.SystemState.init(.{
        .devices = collector.device_infos.items,
        .host = &host_info,
        .targets = targets,
        .tui_refresh_rate = args.tui_refresh_rate,
        .process_lists = collector.process_lists.items,
        .enricher = &enricher,
        .gpa = gpa,
        .arena = arena,
        .io = io,
    });
    defer state.deinit(arena);

    if (args.prometheus_listen) |listen| {
        try api_group.concurrent(io, prometheus.Server.run, .{ io, listen, collector.device_infos.items, &host_info });
    }

    if (args.api) {
        try api_group.concurrent(io, api.Server.run, .{ gpa, io, args.api_port, collector.device_infos.items, collector.process_lists.items, &enricher });
    }

    if (args.prometheus_listen != null or args.api) {
        try api_group.await(io);
    } else if (args.csv) {
        var csv_buf: [4096]u8 = undefined;
        var csv_writer = std.Io.File.stdout().writer(io, &csv_buf);

        try csv.write(&csv_writer.interface, collector.device_infos.items);
        try csv_writer.flush();
    } else if (args.json) {
        var procs: std.ArrayList(smi_info.process_info.ProcessInfo) = .empty;
        defer procs.deinit(gpa);
        for (collector.process_lists.items) |pl| {
            try procs.appendSlice(gpa, pl.front().items);
        }
        enricher.enrich(io, procs.items);

        var json_buf: [4096]u8 = undefined;
        var json_writer = std.Io.File.stdout().writer(io, &json_buf);

        try json.write(&json_writer.interface, collector.device_infos.items, procs.items);
        try json_writer.flush();
    } else if (args.top) {
        try tui.run(gpa, io, &state);
    } else {
        try static_print.run(arena, io, &state);
    }
}

fn detect(io: std.Io) smi_info.Targets {
    if (comptime builtin.os.tag == .macos) {
        return .{};
    }

    var targets: smi_info.Targets = .{};

    switch (builtin.os.tag) {
        .macos => {},
        .linux => {
            switch (builtin.cpu.arch) {
                .x86_64 => {
                    if (hasDevice(io, "/dev/nvidiactl")) {
                        targets.insert(.cuda);
                    }

                    if (hasDevice(io, "/dev/kfd")) {
                        targets.insert(.rocm);
                    }

                    if (hasDevice(io, c.NEURON_DEVICE_PREFIX ++ "0")) {
                        targets.insert(.neuron);
                    }

                    if (hasDevice(io, "/dev/accel0") or hasDevice(io, "/dev/vfio/0")) {
                        targets.insert(.tpu);
                    }
                },
                .aarch64 => {
                    if (hasDevice(io, "/dev/nvidiactl")) {
                        targets.insert(.cuda);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }

    return targets;
}

fn hasDevice(io: std.Io, path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    return true;
}
