const std = @import("std");

const cpu = @import("platforms/cpu");
const cuda = @import("platforms/cuda");
const metal = @import("platforms/metal");
const neuron = @import("platforms/neuron");
const pjrt = @import("pjrt");
const rocm = @import("platforms/rocm");
const oneapi = @import("platforms/oneapi");
const tpu = @import("platforms/tpu");

pub const Platform = enum {
    cpu,
    cuda,
    rocm,
    tpu,
    neuron,
    oneapi,
    metal,
};

pub fn load(allocator: std.mem.Allocator, io: std.Io, tag: Platform) !*const pjrt.Api {
    return switch (tag) {
        .cpu => try cpu.load(allocator, io),
        .cuda => try cuda.load(allocator, io),
        .rocm => try rocm.load(allocator, io),
        .tpu => try tpu.load(allocator, io),
        .neuron => try neuron.load(allocator, io),
        .oneapi => try oneapi.load(allocator, io),
        .metal => try metal.load(allocator, io),
    };
}

pub fn isEnabled(tag: Platform) bool {
    return switch (tag) {
        .cpu => cpu.isEnabled(),
        .cuda => cuda.isEnabled(),
        .rocm => rocm.isEnabled(),
        .tpu => tpu.isEnabled(),
        .neuron => neuron.isEnabled(),
        .oneapi => oneapi.isEnabled(),
        .metal => metal.isEnabled(),
    };
}
