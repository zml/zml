const std = @import("std");
const c = @import("c");

pub const ZigAllocator = @import("zig_allocator.zig").ZigAllocator;
pub const ZigSlice = @import("zig_slice.zig").ZigSlice;

pub fn as_path(path: []const u8) [std.fs.max_path_bytes:0]u8 {
    var result: [std.fs.max_path_bytes:0]u8 = undefined;
    @memcpy(result[0..path.len], path);
    result[path.len] = 0;
    return result;
}
