//! Ends a process that opened the cards in each way a ZML test or program
//! ends (exit_paths.zig), and checks after each that the cards open and
//! compute again. A client never destroyed leaves the cards' Ethernet cores in
//! fabric mode, and every later open times out until the cards are reset.
//! Bazel tag tt_hardware: run under the card lock. A failure here leaves the
//! cards wedged: reset them before anything else uses them.

const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");

const smoke = @import("smoke.zig");

const Case = struct {
    way: []const u8,
    term: std.process.Child.Term,
};

const cases = [_]Case{
    .{ .way = "return", .term = .{ .exited = 0 } },
    .{ .way = "exit", .term = .{ .exited = 3 } },
    .{ .way = "exit_thread", .term = .{ .exited = 4 } },
    .{ .way = "panic", .term = .{ .signal = .ABRT } },
    .{ .way = "sigterm", .term = .{ .signal = .TERM } },
    .{ .way = "sigint", .term = .{ .signal = .INT } },
    .{ .way = "sighup", .term = .{ .signal = .HUP } },
};

fn helperPath(buf: []u8) ![]const u8 {
    const r = try zml.bazel.runfiles(bazel_builtin.current_repository);
    return try r.rlocation("zml/platforms/tenstorrent/exit_paths", buf) orelse error.FileNotFound;
}

test "the cards stay usable however a process that opened them ends" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const helper = try helperPath(&path_buf);
    for (cases) |case| {
        errdefer std.log.err("after a process that opened the cards and ended by {s}", .{case.way});
        var arg_buf: [32]u8 = undefined;
        var child = try std.process.spawn(io, .{ .argv = &.{ helper, try std.fmt.bufPrint(&arg_buf, "--way={s}", .{case.way}) } });
        const term = try child.wait(io);
        if (!std.meta.eql(case.term, term)) {
            std.log.err("the helper {f}, expected: {f}", .{ term, case.term });
            return error.TestUnexpectedResult;
        }

        // Wedged cards fail here, after about 20 s per Ethernet core.
        const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{});
        defer platform.deinit(allocator, io);
        try smoke.expectMatchesCpu(allocator, io, platform);
    }
}
