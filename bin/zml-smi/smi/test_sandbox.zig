const std = @import("std");
const path = @import("sandbox.zig").path;

test path {
    const io = std.testing.io;
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    try std.Io.Dir.cwd().access(io, path(io, &buffer) orelse return error.NoSandbox, .{});
}
