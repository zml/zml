const std = @import("std");
const smi = @import("zml-smi");

const Neuron = @This();

platform: smi.Platform,

pub fn detect(_: std.Io) bool {
    return false;
}

pub fn init(_: std.mem.Allocator, _: std.mem.Allocator, _: std.Io) !Neuron {
    return error.Unsupported;
}
