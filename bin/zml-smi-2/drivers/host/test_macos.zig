const std = @import("std");
const contract = @import("zml-smi-2/contract");

const Host = @import("zml-smi-2/host").Host;
const c = std.c;

test "enrich" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var data: contract.HostData = .{};
    var host: Host = .init(arena.allocator(), std.testing.allocator, io, &data);
    defer host.deinit();

    const self: u32 = @intCast(c.getpid());
    var rows = [_]contract.ProcessInfo{ .{ .pid = self }, .{ .pid = 1 } };
    host.read(arena.allocator(), &rows, &data);
    try std.testing.expectEqual(c.getuid(), rows[0].uid);
    try std.testing.expect(rows[0].rss_kib > 0);
    try std.testing.expectEqualStrings("root", rows[1].username);
    try std.testing.expectEqualStrings("/sbin/launchd", rows[1].comm);

    // CPU share needs a second reading.
    const start: std.Io.Timestamp = .now(io, .awake);
    while (start.untilNow(io, .awake).toMilliseconds() < 20) {}
    host.read(arena.allocator(), &rows, &data);
    try std.testing.expect(rows[0].cpu_percent > 0);
}
