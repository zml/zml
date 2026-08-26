const std = @import("std");

const core = @import("tests/core.zig");
const model = @import("tests/model.zig");
const runtime = @import("tests/runtime.zig");
const vae = @import("tests/vae.zig");
const vision = @import("tests/vision.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try core.run(allocator);
    try model.run(allocator);
    try vision.run(allocator);
    try vae.run(allocator);
    try runtime.run(allocator);

    std.debug.print("minimax_h3 tests: all passed\n", .{});
}
