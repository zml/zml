const std = @import("std");
const proc = @import("zml-smi/native").proc;

const FdHolders = proc.FdHolders;

test FdHolders {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "100/fd");
    try tmp.dir.createDirPath(io, "200/fd");
    try tmp.dir.symLink(io, "/dev/accel3", "100/fd/5", .{});
    try tmp.dir.symLink(io, "/dev/null", "100/fd/0", .{});
    try tmp.dir.symLink(io, "/dev/vfio/1", "200/fd/9", .{});

    var holders: FdHolders = proc.testing.initAt(io, try tmp.dir.openDir(io, ".", .{ .iterate = true }), &.{ "/dev/accel", "/dev/vfio/" });
    defer holders.deinit();

    var count: usize = 0;
    while (holders.next()) |holder| : (count += 1) {
        switch (holder.pid) {
            100 => {
                try std.testing.expectEqual(5, holder.fd);
                try std.testing.expectEqualStrings("3", holder.rest);
            },
            200 => try std.testing.expectEqualStrings("1", holder.rest),
            else => return error.TestUnexpectedResult,
        }
    }
    try std.testing.expectEqual(2, count);
}
