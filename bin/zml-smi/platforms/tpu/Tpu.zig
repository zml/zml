const std = @import("std");
const smi = @import("zml-smi");

const Tpu = @This();

platform: smi.Platform,

pub fn detect(_: std.Io) bool {
    return false;
}

pub fn init(_: std.mem.Allocator, _: std.mem.Allocator, _: std.Io) !Tpu {
    return error.Unsupported;
}
