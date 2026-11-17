const std = @import("std");

const model = @import("model.zig");

pub fn main(init: std.process.Init) !void {
    std.debug.print("hello", .{});
    _ = init;
}
