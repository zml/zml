const std = @import("std");

pub const path = struct {
    pub fn bufJoin(buf: []u8, paths: []const []const u8) ![]u8 {
        var fa: std.heap.FixedBufferAllocator = .init(buf);
        return try std.fs.path.join(fa.allocator(), paths);
    }

    pub fn bufJoinZ(buf: []u8, paths: []const []const u8) ![:0]u8 {
        var fa: std.heap.FixedBufferAllocator = .init(buf);
        return try std.fs.path.joinZ(fa.allocator(), paths);
    }
};
