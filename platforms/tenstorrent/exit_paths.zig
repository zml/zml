//! Helper of exit_paths_hw_test: opens the cards as `zml.testing.env()` does
//! (`Platform.auto`, never deinit'ed), runs y = x * 2 + 1 on them, then ends
//! the way `--way` names. However it ends, the cards must stay usable.

const std = @import("std");

const zml = @import("zml");

const smoke = @import("smoke.zig");

pub const Way = enum {
    /// Returns from main.
    @"return",
    /// `std.process.exit(3)`, as a failed test run ends.
    exit,
    /// A Zig panic: `abort`, then SIGABRT.
    panic,
    /// What Bazel sends a test that times out.
    sigterm,
    /// Ctrl-C.
    sigint,
    /// A closed terminal or ssh session.
    sighup,
    /// `std.process.exit(4)` from another thread, whose thread-locals are not
    /// the main thread's: the plugin's exit handler closes the cards there too.
    exit_thread,
};

const Args = struct {
    way: Way,

    pub const help =
        \\ Use exit_paths --way=<return|exit|panic|sigterm|sigint|sighup|exit_thread>
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    // The default actions, whatever the parent left (nohup ignores SIGHUP):
    // the case is a process that leaves these signals alone.
    const default: std.posix.Sigaction = .{ .handler = .{ .handler = std.posix.SIG.DFL }, .mask = std.posix.sigemptyset(), .flags = 0 };
    for ([_]std.posix.SIG{ .ABRT, .TERM, .INT, .HUP }) |sig| std.posix.sigaction(sig, &default, null);

    const platform = try zml.Platform.auto(std.heap.c_allocator, io, .{});
    if (platform.target != .tenstorrent) return error.NoCards;
    const x = try smoke.input(allocator, platform.devices.len);
    defer allocator.free(x);
    const y = try smoke.run(allocator, io, platform, x);
    allocator.free(y);
    std.log.info("exit_paths: y = x * 2 + 1 ran on {d} cards; ending by {s}", .{ platform.devices.len, @tagName(args.way) });

    const signal: std.posix.SIG = switch (args.way) {
        .@"return" => return,
        .exit => std.process.exit(3),
        .panic => @panic("exit_paths: a panic with the cards open"),
        .sigterm => .TERM,
        .sigint => .INT,
        .sighup => .HUP,
        .exit_thread => {
            const thread = try std.Thread.spawn(.{}, exitFromThread, .{});
            thread.join();
            unreachable;
        },
    };
    try std.posix.kill(std.c.getpid(), signal);
    // The signal ends the process; this only waits for it.
    while (true) try io.sleep(.fromSeconds(1), .awake);
}

fn exitFromThread() void {
    std.process.exit(4);
}
