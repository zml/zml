const pjrt = @import("pjrt");
const cpu = @import("runtimes/cpu");
const cuda = @import("runtimes/cuda");
const rocm = @import("runtimes/rocm");
const tpu = @import("runtimes/tpu");
const neuron = @import("runtimes/neuron");

pub const Platform = enum {
    cpu,
    cuda,
    rocm,
    tpu,
    neuron,
};

pub fn load(tag: Platform) !*const pjrt.Api {
    return switch (tag) {
        .cpu => try cpu.load(),
        .cuda => try cuda.load(),
        .rocm => try rocm.load(),
        .tpu => try tpu.load(),
        .neuron => try neuron.load(),
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
