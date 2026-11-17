const std = @import("std");
const path = @import("zml-smi-2/native").testing.sandbox.path;

test path {
    const io = std.testing.io;
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    try std.Io.Dir.cwd().access(io, path(io, &buffer) orelse return error.NoSandbox, .{});
}
