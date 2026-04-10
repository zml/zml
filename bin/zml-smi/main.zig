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
const linux = @import("zml-smi/platforms/linux");
const ProcessEnricher = linux.process.ProcessEnricher;
const host = linux.metrics;
const c = @import("c");
const csv = @import("zml-smi/csv");
const smi_tui = @import("zml-smi/tui");
const tui = smi_tui.top;
const static_print = smi_tui.print;
const data = smi_tui.data;

const CliArgs = struct {
    top: bool = false,
    csv: bool = false,
    tui_refresh_rate: u16 = 100,
    poll_interval: u16 = 500,

    pub const help =
        \\ zml-smi [--top] [--csv] [--tui-refresh-rate MS] [--poll-interval MS]
        \\
        \\ --top               Interactive TUI mode
        \\ --csv               Output device metrics as CSV
        \\ --tui-refresh-rate  TUI refresh rate in ms (default: 100)
        \\ --poll-interval     Device polling interval in ms (default: 500)
        \\
    ;
};

const device_backends = if (builtin.os.tag != .macos) .{
    .{ .cuda, @import("zml-smi/platforms/nvml") },
    .{ .rocm, @import("zml-smi/platforms/amdsmi") },
    .{ .neuron, @import("zml-smi/platforms/neuron") },
    .{ .tpu, @import("zml-smi/platforms/tpu") },
} else .{};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const targets = detect(io);

    var collector: Collector = .init(arena, gpa, io, .{
        .poll_interval_ms = args.poll_interval,
        .poll_only = !args.top,
    });
    defer collector.deinit();

    var host_info: HostInfo = .{ .values = .{ .{}, .{} } };
    try host.init(&collector, &host_info);

    var enricher: ProcessEnricher = try .init(gpa, io);
    defer enricher.deinit();

    inline for (device_backends) |entry| {
        const target, const backend = entry;
        if (targets.contains(target)) {
            backend.start(&collector) catch |err| {
                std.log.err("{s} skipped: {s}", .{ @tagName(target), @errorName(err) });
            };
        }
    }

    var state = try data.SystemState.init(arena, .{
        .devices = collector.device_infos.items,
        .host = &host_info,
        .targets = targets,
        .tui_refresh_rate = args.tui_refresh_rate,
        .process_lists = collector.process_lists.items,
        .enricher = &enricher,
        .io = io,
    });
    defer state.deinit(arena);

    if (args.csv) {
        var csv_buf: [4096]u8 = undefined;
        var csv_writer = std.Io.File.stdout().writer(io, &csv_buf);

        try csv.write(&csv_writer.interface, collector.device_infos.items);
        try csv_writer.flush();
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

    return targets;
}

fn hasDevice(io: std.Io, path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    return true;
}
