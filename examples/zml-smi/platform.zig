const std = @import("std");

pub const Target = @import("info/device_info.zig").Target;

fn hasDevice(io: std.Io, path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;

    return true;
}

pub const Targets = std.EnumSet(Target);

pub fn detect(io: std.Io) Targets {
    var targets: Targets = .{};

    if (hasDevice(io, "/dev/nvidiactl")) targets.insert(.cuda);
    if (hasDevice(io, "/dev/kfd")) targets.insert(.rocm);
    if (hasDevice(io, "/dev/neuron0")) targets.insert(.neuron);
    if (hasDevice(io, "/dev/accel0") or hasDevice(io, "/dev/vfio/0")) targets.insert(.tpu);

    return targets;
}
