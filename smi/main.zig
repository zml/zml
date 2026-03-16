const std = @import("std");
const builtin = @import("builtin");
const c = if (builtin.os.tag == .macos) @import("c") else void;

const stdx = @import("stdx");

pub const std_options: std.Options = .{
    .log_scope_levels = &.{
        .{ .scope = .vaxis, .level = .warn },
    },
};

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

const nvml = @import("bindings/nvml/metrics.zig");
const amdsmi = @import("bindings/amdsmi/metrics.zig");
const neuron = @import("bindings/neuron/metrics.zig");
const tpu = @import("bindings/tpu/metrics.zig");
const host = @import("bindings/linux/metrics.zig");

const device_info = @import("info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const HostInfo = @import("info/host_info.zig").HostInfo;
const ProcessEnricher = @import("bindings/linux/process.zig").ProcessEnricher;
const platform = @import("platform.zig");
const Worker = @import("worker.zig").Worker;
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    var w: Worker = .{ .poll_interval_ms = args.poll_interval };
    defer w.shutdown(io);

    const targets = platform.detect(io);

    var device_infos: std.ArrayList(*DeviceInfo) = .{};
    defer {
        for (device_infos.items) |info| arena.destroy(info);
        device_infos.deinit(arena);
    }

    var host_info: HostInfo = .{};
    try host.init(&w, io, &host_info);

    var enricher: ProcessEnricher = try .init(gpa, io);
    defer enricher.deinit();

    var state_config: data.SystemState.Config = .{
        .devices = device_infos.items,
        .host = &host_info,
        .targets = targets,
        .tui_refresh_rate = args.tui_refresh_rate,
        .process_lists = &.{},
        .enricher = &enricher,
        .io = io,
    };

    if (comptime builtin.os.tag == .macos) {
        try runSmi(io, gpa, arena, args.top, state_config);

        return;
    }

    var cuda: nvml.Backend = .{};
    var rocm: amdsmi.Backend = .{};
    var aws_neuron: neuron.Backend = .{};
    var google_tpu: tpu.Backend = .{};
    defer cuda.deinit(gpa);
    defer rocm.deinit(gpa);
    defer aws_neuron.deinit(gpa);
    defer google_tpu.deinit(gpa);

    if (targets.contains(.rocm)) {
        rocm.start(&w, io, arena, &device_infos, gpa) catch {
            std.log.err("rocm skipped", .{});
        };
    }

    if (targets.contains(.cuda)) {
        cuda.start(&w, io, arena, &device_infos, gpa) catch {
            std.log.err("nvml skipped", .{});
        };
    }

    if (targets.contains(.neuron)) {
        aws_neuron.start(&w, io, arena, &device_infos, gpa) catch {
            std.log.err("neuron skipped", .{});
        };
    }

    if (targets.contains(.tpu)) {
        google_tpu.start(&w, io, arena, &device_infos, gpa) catch {
            std.log.err("tpu skipped", .{});
        };
    }

    var proc_ptrs = [_]*data.ProcessShadowList{
        &cuda.processes, &rocm.processes, &aws_neuron.processes, &google_tpu.processes,
    };

    state_config.process_lists = &proc_ptrs;
    state_config.devices = device_infos.items;

    try runSmi(io, gpa, arena, args.top, state_config);
}

fn runSmi(io: std.Io, gpa: std.mem.Allocator, arena: std.mem.Allocator, top: bool, state_config: data.SystemState.Config) !void {
    var state = try data.SystemState.init(arena, state_config);
    defer state.deinit(arena);

    if (top) {
        try tui.run(gpa, io, &state);
    } else {
        try static_print.run(arena, io, &state);
    }
}
