const std = @import("std");

const cpu = @import("platforms/cpu");
const cuda = @import("platforms/cuda");
const metal = @import("platforms/metal");
const neuron = @import("platforms/neuron");
const pjrt = @import("pjrt");
const rocm = @import("platforms/rocm");
const oneapi = @import("platforms/oneapi");
const tpu = @import("platforms/tpu");

const platforms = @This();

pub const Platform = enum {
    cpu,
    cuda,
    rocm,
    tpu,
    neuron,
    oneapi,
    metal,

    pub fn selectFirstAcceleratorEnabled() error{Unavailable}!Platform {
        inline for (@typeInfo(Platform).@"enum".fields) |field| {
            const target: Platform = @enumFromInt(field.value);
            if (comptime (target != .cpu and target.isEnabled())) return target;
        }

        if (!comptime Platform.cpu.isEnabled()) @compileError("No platform was enabled, use --@zml//platforms:cuda=true to eg enable Cuda");
        return .cpu;
    }

    pub fn isEnabled(target: Platform) bool {
        return switch (target) {
            inline else => |tag| @field(platforms, @tagName(tag)).isEnabled(),
        };
    }

    pub fn load(target: Platform, allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
        return switch (target) {
            inline else => |tag| @field(platforms, @tagName(tag)).load(allocator, io),
        };
    }
};

pub const load = Platform.load;
pub const isEnabled = Platform.isEnabled;
