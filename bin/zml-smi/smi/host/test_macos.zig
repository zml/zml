const std = @import("std");
const model = @import("zml-smi/model");

const Host = @import("macos.zig").Host;
const c = std.c;

test "enrich" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var data: model.HostData = .{};
    var host: Host = .init(arena.allocator(), std.testing.allocator, io, &data);
    defer host.deinit();

    const self: u32 = @intCast(c.getpid());
    var rows = [_]model.ProcessInfo{ .{ .pid = self }, .{ .pid = 1 } };
    host.enrich(arena.allocator(), &rows);
    try std.testing.expectEqual(c.getuid(), rows[0].uid);
    try std.testing.expect(rows[0].rss_kib > 0);
    try std.testing.expectEqualStrings("root", rows[1].username);
    try std.testing.expectEqualStrings("/sbin/launchd", rows[1].comm);

    // CPU share needs a second reading.
    const start: std.Io.Timestamp = .now(io, .awake);
    while (start.untilNow(io, .awake).toMilliseconds() < 20) {}
    host.enrich(arena.allocator(), &rows);
    try std.testing.expect(rows[0].cpu_percent > 0);
}
