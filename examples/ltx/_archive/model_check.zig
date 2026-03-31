const std = @import("std");
const model = @import("model.zig");

pub fn main(_: std.process.Init) !void {
    // Minimal compile-time/runtime smoke check for the LTX model scaffold.
    const cfg: model.Config = .{ .num_transformer_blocks = 1 };
    _ = cfg;
}
