const std = @import("std");

const Value = @import("value.zig").Value;

pub fn allTrue(values: []const Value, func: fn (v: Value) bool) bool {
    for (values) |v| {
        if (!func(v)) return false;
    }
    return true;
}

pub fn isBadFilename(filename: []const u8) bool {
    if (filename.len == 0 or filename[0] == '/')
        return true;

    var it = std.mem.splitScalar(u8, filename, '/');
    while (it.next()) |part| {
        if (std.mem.eql(u8, part, ".."))
            return true;
    }

    return false;
}
