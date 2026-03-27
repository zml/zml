const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

pub const std_options: std.Options = .{
    .log_scope_levels = &.{
        .{ .scope = .vaxis, .level = .warn },
    },
};

const Collector = @import("collector.zig").Collector;
const HostInfo = @import("info/host_info.zig").HostInfo;
const ProcessEnricher = @import("bindings/linux/process.zig").ProcessEnricher;
const platform = @import("platform.zig");
const Worker = @import("worker.zig").Worker;
const host = @import("bindings/linux/metrics.zig");
const tui = @import("tui/tui.zig");
const static_print = @import("tui/print.zig");
const data = @import("tui/data.zig");

const CliArgs = struct {
    top: bool = false,
    tui_refresh_rate: u16 = 100,
    poll_interval: u16 = 500,

    pub const help =
        \\ zml-smi [--top] [--tui-refresh-rate MS] [--poll-interval MS]
        \\
        \\ --top               Interactive TUI mode
        \\ --tui-refresh-rate  TUI refresh rate in ms (default: 100)
        \\ --poll-interval     Device polling interval in ms (default: 500)
        \\
    ;
};

const device_backends = if (builtin.os.tag != .macos) .{
    @import("bindings/nvml/metrics.zig"),
    @import("bindings/amdsmi/metrics.zig"),
    @import("bindings/neuron/metrics.zig"),
    @import("bindings/tpu/metrics.zig"),
} else .{};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    var w: Worker = .{ .poll_interval_ms = args.poll_interval };

    const targets = platform.detect(io);

    var collector: Collector = .{
        .arena = arena,
        .gpa = gpa,
        .worker = &w,
        .io = io,
    };
    defer collector.deinit();
    defer w.shutdown(io);

    var host_info: HostInfo = .{ .values = .{ .{}, .{} } };
    try host.init(&w, io, &host_info);

    var enricher: ProcessEnricher = try .init(gpa, io);
    defer enricher.deinit();

    inline for (device_backends) |backend| {
        if (targets.contains(backend.target)) {
            backend.start(&collector) catch {
                std.log.err("{s} skipped", .{@tagName(backend.target)});
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

    if (args.top) {
        try tui.run(gpa, io, &state);
    } else {
        try static_print.run(arena, io, &state);
    }
}
