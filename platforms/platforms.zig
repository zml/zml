const std = @import("std");

const cpu = @import("platforms/cpu");
const cuda = @import("platforms/cuda");
const neuron = @import("platforms/neuron");
const pjrt = @import("pjrt");
const rocm = @import("platforms/rocm");
const tpu = @import("platforms/tpu");

pub const Platform = enum {
    cpu,
    cuda,
    rocm,
    tpu,
    neuron,
};

pub fn load(tag: Platform, io: std.Io) !*const pjrt.Api {
    return switch (tag) {
        .cpu => try cpu.load(io),
        .cuda => try cuda.load(io),
        .rocm => try rocm.load(io),
        .tpu => try tpu.load(io),
        .neuron => try neuron.load(io),
    };
}

pub fn isEnabled(tag: Platform) bool {
    return switch (tag) {
        .cpu => cpu.isEnabled(),
        .cuda => cuda.isEnabled(),
        .rocm => rocm.isEnabled(),
        .tpu => tpu.isEnabled(),
        .neuron => neuron.isEnabled(),
    };
}
