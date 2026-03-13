const std = @import("std");
const c = @import("c");
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
        \\ zml-smi [--top] [--sample-interval MS] [--poll-interval MS]
        \\
        \\ --top               Interactive TUI mode
        \\ --tui-refresh-rate  TUI refresh rate in ms (default: 100)
        \\ --poll-interval     Device polling interval in ms (default: 100)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    var w: Worker = .{ .poll_interval_ms = args.poll_interval };
    defer w.shutdown(io);

    const targets = platform.detect(io);

    var device_infos: std.ArrayList(*DeviceInfo) = .{};
    defer {
        for (device_infos.items) |info| allocator.destroy(info);
        device_infos.deinit(allocator);
    }
    var host_info: HostInfo = .{};
    try host.init(&w, io, &host_info);

    var enricher: ProcessEnricher = try .init(gpa, io);
    defer enricher.deinit();

    var cuda: nvml.Backend = .{};
    var rocm: amdsmi.Backend = .{};
    var aws_neuron: neuron.Backend = .{};
    var google_tpu: tpu.Backend = .{};
    defer cuda.deinit(gpa);
    defer rocm.deinit(gpa);
    defer aws_neuron.deinit(gpa);
    defer google_tpu.deinit(gpa);

    if (@hasDecl(c, "ZML_RUNTIME_ROCM") and targets.contains(.rocm)) {
        rocm.start(&w, io, allocator, &device_infos, gpa) catch {
            std.log.err("rocm smi init failed; skipped", .{});
        };
    }

    if (@hasDecl(c, "ZML_RUNTIME_CUDA") and targets.contains(.cuda)) {
        cuda.start(&w, io, allocator, &device_infos, gpa) catch {
            std.log.err("nvml smi init failed; skipped", .{});
        };
    }

    if (@hasDecl(c, "ZML_RUNTIME_NEURON") and targets.contains(.neuron)) {
        aws_neuron.start(&w, io, allocator, &device_infos, gpa) catch {
            std.log.err("neuron smi init failed; skipped", .{});
        };
    }

    if (@hasDecl(c, "ZML_RUNTIME_TPU") and targets.contains(.tpu)) {
        google_tpu.start(&w, io, allocator, &device_infos, gpa) catch {
            std.log.err("tpu smi init failed; skipped", .{});
        };
    }

    var proc_ptrs = [_]*std.ArrayList(data.pi.ProcessInfo){
        &cuda.processes, &rocm.processes, &aws_neuron.processes, &google_tpu.processes,
    };

    var state = try data.SystemState.init(allocator, .{
        .devices = device_infos.items,
        .host = &host_info,
        .targets = targets,
        .tui_refresh_rate = args.tui_refresh_rate,
        .process_lists = &proc_ptrs,
        .enricher = &enricher,
        .io = io,
    });
    defer state.deinit(allocator);

    if (args.top) {
        try tui.run(gpa, io, &state);
    } else {
        try static_print.run(allocator, io, &state);
    }
}
