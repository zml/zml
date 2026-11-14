//! The platforms of this OS and architecture. Each one is a file whose struct holds an `smi.Platform`.
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
    else => .{},
};
