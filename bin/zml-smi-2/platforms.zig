//! Native readers available on this OS and architecture.
const builtin = @import("builtin");
const std = @import("std");

pub const all = switch (builtin.os.tag) {
    .linux => switch (builtin.cpu.arch) {
        .x86_64 => .{
            @import("zml-smi-2/drivers/nvml"),
            @import("zml-smi-2/drivers/amdsmi"),
            @import("zml-smi-2/drivers/oneapi"),
            @import("zml-smi-2/drivers/neuron"),
            @import("zml-smi-2/drivers/tpu"),
            @import("zml-smi-2/drivers/furiosa"),
        },
        .aarch64 => .{
            @import("zml-smi-2/drivers/nvml"),
        },
        else => .{},
    },
    .macos => switch (builtin.cpu.arch) {
        .aarch64 => .{
            @import("zml-smi-2/drivers/metal"),
        },
        else => .{},
    },
    else => .{},
};

pub fn available(comptime Backend: type, io: std.Io) bool {
    if (!@hasDecl(Backend, "detection_paths")) return true;
    for (Backend.detection_paths) |path| {
        std.Io.Dir.cwd().access(io, path, .{}) catch continue;
        return true;
    }
    return false;
}
