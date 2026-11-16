//! Native readers available on this OS and architecture.
const builtin = @import("builtin");

pub const all = switch (builtin.os.tag) {
    .linux => switch (builtin.cpu.arch) {
        .x86_64 => .{
            @import("zml-smi/platforms/nvml"),
            @import("zml-smi/platforms/amdsmi"),
            @import("zml-smi/platforms/oneapi"),
            @import("zml-smi/platforms/neuron"),
            @import("zml-smi/platforms/tpu"),
            @import("zml-smi/platforms/furiosa"),
        },
        .aarch64 => .{
            @import("zml-smi/platforms/nvml"),
        },
        else => .{},
    },
    .macos => switch (builtin.cpu.arch) {
        .aarch64 => .{
            @import("zml-smi/platforms/metal"),
        },
        else => .{},
    },
    else => .{},
};
