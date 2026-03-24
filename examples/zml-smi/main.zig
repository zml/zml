const std = @import("std");
const zml = @import("zml");
const c = @import("c");
const stdx = @import("stdx");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

const nvml = @import("bindings/nvml/metrics.zig");
const amdsmi = @import("bindings/amdsmi/metrics.zig");
const neuron = @import("bindings/neuron/metrics.zig");
const tpu = @import("bindings/tpu/metrics.zig");
const host = @import("bindings/linux/metrics.zig");

const worker = @import("worker.zig");
const display = @import("display.zig");
const DeviceInfo = @import("device_info.zig").DeviceInfo;
const HostInfo = @import("host_info.zig").HostInfo;

pub const Target = enum { cuda, rocm, neuron, tpu };

fn hasDevice(io: std.Io, path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    return true;
}

fn detect(io: std.Io) ?Target {
    if (hasDevice(io, "/dev/nvidiactl")) return .cuda;
    if (hasDevice(io, "/dev/kfd")) return .rocm;
    if (hasDevice(io, "/dev/neuron0")) return .neuron;
    if (hasDevice(io, "/dev/accel0") or hasDevice(io, "/dev/vfio/0")) return .tpu;
    return null;
}

fn setupRocmEnv(rocm_data_dir: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("ROCM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{rocm_data_dir}), 1); // must be zero terminated
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buffer);

    const target = detect(io);

    var device_infos: std.ArrayList(DeviceInfo) = .{};
    var host_info: HostInfo = .{};
    var signal: worker.Signal = .{};

    try host.init(io, &host_info, &signal);

    if (target) |t| {
        const r = try bazel.runfiles(bazel_builtin.current_repository);
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        if (@hasDecl(c, "ZML_RUNTIME_ROCM")) {
            const sandbox_path = try r.rlocation("libpjrt_rocm/sandbox", &path_buf) orelse {
                return error.FileNotFound;
            };

            try setupRocmEnv(sandbox_path);
        }

        switch (t) {
            .cuda => try nvml.init(io, allocator, &device_infos, &signal),
            .rocm => try amdsmi.init(io, allocator, &device_infos, &signal),
            .neuron => try neuron.init(io, allocator, &device_infos, &signal), // if multiple platforms in addition to neuron need to be supported, the monitor logics needs to change otherwise it will write to every device
            .tpu => try tpu.init(io, allocator, &device_infos, &signal),
        }
    }

    while (true) {
        signal.wait(io);
        try display.draw(&writer.interface, target, &host_info, device_infos.items);
        try writer.flush();
    }
}
