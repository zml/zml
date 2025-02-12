//! Test runner for unit test based on https://github.com/ziglang/zig/blob/master/lib/compiler/test_runner.zig with async
const asynk = @import("async");
const builtin = @import("builtin");
const std = @import("std");

const io = std.io;
const testing = std.testing;
const assert = std.debug.assert;

// note: std_options.log_level does not respect testing.log_level
// ref: https://github.com/ziglang/zig/issues/5738
const log_level: std.log.Level = .warn;

pub const std_options: std.Options = .{
    .log_level = log_level,
};

var log_err_count: usize = 0;
var fba_buffer: [8192]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&fba_buffer);

pub fn main() anyerror!void {
    testing.log_level = log_level;
    try asynk.AsyncThread.main(testing.allocator, asyncMain);
}

pub fn asyncMain() !void {
    const test_fn_list: []const std.builtin.TestFn = builtin.test_functions;
    var ok_count: usize = 0;
    var skip_count: usize = 0;
    var fail_count: usize = 0;
    const root_node = std.Progress.start(.{
        .root_name = "Test",
        .estimated_total_items = test_fn_list.len,
    });
    const have_tty = std.io.getStdErr().isTty();

    var args = std.process.args();
    // Skip executable path
    _ = args.next().?;

    const identifier_query = if (args.next()) |arg| blk: {
        std.debug.print("Only tests with identifiers that includes `{s}` will be run\n", .{arg});
        break :blk arg;
    } else blk: {
        break :blk "";
    };

    var leaks: usize = 0;

    for (test_fn_list, 0..) |test_fn, i| {
        if (std.mem.indexOf(u8, test_fn.name, identifier_query) == null) {
            continue;
        }

        testing.allocator_instance = .{};
        defer {
            if (testing.allocator_instance.deinit() == .leak) {
                leaks += 1;
            }
        }

        const test_node = root_node.start(test_fn.name, 0);
        if (!have_tty) {
            std.debug.print("{d}/{d} {s}...", .{ i + 1, test_fn_list.len, test_fn.name });
        }
        if (test_fn.func()) |_| {
            ok_count += 1;
            test_node.end();
            if (!have_tty) std.debug.print("OK\n", .{});
        } else |err| switch (err) {
            error.SkipZigTest => {
                skip_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...SKIP\n", .{ i + 1, test_fn_list.len, test_fn.name });
                } else {
                    std.debug.print("SKIP\n", .{});
                }
                test_node.end();
            },
            else => {
                fail_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...FAIL ({s})\n", .{
                        i + 1, test_fn_list.len, test_fn.name, @errorName(err),
                    });
                } else {
                    std.debug.print("FAIL ({s})\n", .{@errorName(err)});
                }
                if (@errorReturnTrace()) |trace| {
                    std.debug.dumpStackTrace(trace.*);
                }
                test_node.end();
            },
        }
    }
    root_node.end();
    if (ok_count == test_fn_list.len) {
        std.debug.print("All {d} tests passed.\n", .{ok_count});
    } else {
        std.debug.print("{d} passed; {d} skipped; {d} failed.\n", .{ ok_count, skip_count, fail_count });
    }
    if (log_err_count != 0) {
        std.debug.print("{d} errors were logged.\n", .{log_err_count});
    }
    if (leaks != 0) {
        std.debug.print("{d} tests leaked memory.\n", .{leaks});
    }
    if (leaks != 0 or log_err_count != 0 or fail_count != 0) {
        std.process.exit(1);
    }

    // Explicit exit 0 to exit from async main thread and main properly
    std.process.exit(0);
}
