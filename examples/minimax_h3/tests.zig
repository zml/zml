const std = @import("std");

const core = @import("tests/config.zig");
const draft = @import("tests/draft.zig");
const recipe = @import("tests/recipe.zig");

// =============================================================================
// tests.zig — package test entry
// =============================================================================

pub const std_options: std.Options = .{
    .log_level = .info,
};

test "minimax_h3" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try core.run(allocator);
    try draft.run(allocator);
    try recipe.run(allocator);
}
