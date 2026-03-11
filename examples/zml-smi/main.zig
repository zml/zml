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
const nvml_process = @import("bindings/nvml/process.zig");
const amdsmi = @import("bindings/amdsmi/metrics.zig");
const amdsmi_process = @import("bindings/amdsmi/process.zig");
const neuron = @import("bindings/neuron/metrics.zig");
const neuron_process = @import("bindings/neuron/process.zig");
const tpu = @import("bindings/tpu/metrics.zig");
const tpu_process = @import("bindings/tpu/process.zig");
const host = @import("bindings/linux/metrics.zig");
const process = @import("bindings/linux/process.zig");

const device_info = @import("info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const HostInfo = @import("info/host_info.zig").HostInfo;
const platform = @import("platform.zig");
const worker = @import("worker.zig");
const tui = @import("tui/tui.zig");
const static_print = @import("tui/print.zig");
const data = @import("tui/data.zig");

fn setupRocmEnv(rocm_data_dir: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("ROCM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{rocm_data_dir}), 1);
}

const CliArgs = struct {
    top: bool = false,
    sample_interval: u16 = 100,
    poll_interval: u16 = 10,

    pub const help =
        \\ zml-smi [--top] [--sample-interval MS] [--poll-interval MS]
        \\
        \\ --top              Interactive TUI mode
        \\ --sample-interval  TUI refresh rate in ms (default: 100)
        \\ --poll-interval    Device polling interval in ms (default: 10)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    // const allocator = init.gpa;
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    worker.poll_interval_ms = args.poll_interval;
    const targets = platform.detect(io);

    var device_infos: std.ArrayList(*DeviceInfo) = .{};
    defer {
        for (device_infos.items) |info| allocator.destroy(info);
        device_infos.deinit(allocator);
    }
    var host_info: HostInfo = .{};
    var scanner: process.ProcessScanner = .{};

    try host.init(io, &host_info);

    if (@hasDecl(c, "ZML_RUNTIME_ROCM") and targets.contains(.rocm)) {
        const r = try bazel.runfiles(bazel_builtin.current_repository);
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const sandbox_path = try r.rlocation("libpjrt_rocm/sandbox", &path_buf) orelse {
            return error.FileNotFound;
        };

        try setupRocmEnv(sandbox_path);
        try amdsmi.init(io, allocator, &device_infos);
        scanner.addEnrichFn(amdsmi_process.enrichProcesses);
    }

    if (@hasDecl(c, "ZML_RUNTIME_CUDA") and targets.contains(.cuda)) {
        try nvml.init(io, allocator, &device_infos);
        scanner.addEnrichFn(nvml_process.enrichProcesses);
    }

    if (@hasDecl(c, "ZML_RUNTIME_NEURON") and targets.contains(.neuron)) {
        try neuron.init(io, allocator, &device_infos);
        scanner.addEnrichFn(neuron_process.enrichProcesses);
    }

    if (@hasDecl(c, "ZML_RUNTIME_TPU") and targets.contains(.tpu)) {
        const tpu_start = device_infos.items.len;
        try tpu.init(io, allocator, &device_infos);
        const tpu_infos = device_infos.items[tpu_start..];
        if (tpu_infos.len > 0) {
            try tpu_process.init(io, tpu.detected_devices_per_chip, tpu_infos);
            scanner.addEnrichFn(tpu_process.enrichProcesses);
        }
    }

    try process.init(io, &scanner);

    var state = try data.SystemState.init(allocator, device_infos.items, &host_info, targets, args.sample_interval);
    state.process_scanner = &scanner;
    defer state.deinit(allocator);

    defer worker.shutdown(io);

    if (args.top) {
        try tui.run(allocator, io, &state);
    } else {
        try static_print.run(allocator, io, &state);
    }
}
