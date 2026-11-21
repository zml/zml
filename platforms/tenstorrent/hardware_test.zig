//! Runs on the cards of this host (Bazel tag tt_hardware). The JIT cache
//! defaults to $XDG_CACHE_HOME/zml/tenstorrent or $HOME/.cache/zml/tenstorrent.

const std = @import("std");

const zml = @import("zml");

const smoke = @import("smoke.zig");

test "y = x * 2 + 1 on every card, then on one card, matches the CPU" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const first_board: u64 = all: {
        const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{});
        defer platform.deinit(allocator, io);
        std.debug.print("{f}\n", .{platform.fmtVerbose()});
        // The default view serves every card: each node of the kernel driver is
        // one P150 chip (two on a P300), so fewer devices means a card was dropped.
        const device_nodes = try countDeviceNodes(io);
        if (platform.devices.len < device_nodes) {
            std.log.err("the default view serves {d} devices for {d} device nodes", .{ platform.devices.len, device_nodes });
            return error.TestUnexpectedResult;
        }
        try smoke.expectMatchesCpu(allocator, io, platform);
        try smoke.expectLoaderRoundTrip(allocator, io, platform);
        try smoke.expectLoaderRoundTripIn(allocator, io, platform, .host_pinned);
        const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, platform.devices[0]);
        break :all info.board_id;
    };

    var chips_buf: [32]u8 = undefined;
    const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{
        .tenstorrent = .{ .visible_chips = try std.fmt.bufPrint(&chips_buf, "{d}", .{first_board}) },
    });
    defer platform.deinit(allocator, io);
    try std.testing.expectEqual(1, platform.devices.len);
    const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, platform.devices[0]);
    try std.testing.expectEqual(first_board, info.board_id);
    try smoke.expectMatchesCpu(allocator, io, platform);
    try smoke.expectLoaderRoundTrip(allocator, io, platform);
}

fn countDeviceNodes(io: std.Io) !usize {
    var dir = try std.Io.Dir.openDirAbsolute(io, "/dev/tenstorrent", .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    var count: usize = 0;
    // The numbered character devices only: the driver also adds a by-id/ directory.
    while (try it.next(io)) |entry| {
        if (entry.kind != .character_device) continue;
        _ = std.fmt.parseInt(u32, entry.name, 10) catch continue;
        count += 1;
    }
    return count;
}
