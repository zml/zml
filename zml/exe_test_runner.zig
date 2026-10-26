//! Run the existing execution tests without requiring unrelated backend features.
const std = @import("std");
const builtin = @import("builtin");

pub fn main(init: std.process.Init.Minimal) !void {
    var count: usize = 0;
    for (builtin.test_functions) |test_fn| {
        if (!std.mem.startsWith(u8, test_fn.name, "exe.test.")) continue;
        std.testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        defer if (std.testing.allocator_instance.deinit() != 0) {
            @panic("execution test leaked memory");
        };
        std.testing.io_instance = .init(std.testing.allocator, .{
            .argv0 = .init(init.args),
            .environ = init.environ,
        });
        defer std.testing.io_instance.deinit();
        std.testing.environ = init.environ;
        std.testing.log_level = .warn;
        std.debug.print("{s}...", .{test_fn.name});
        try test_fn.func();
        std.debug.print("OK\n", .{});
        count += 1;
    }
    if (count == 0) return error.NoExecutionTestsFound;
    std.debug.print("All {d} execution tests passed.\n", .{count});
}
