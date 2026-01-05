const std = @import("std");

pub const Dir = struct {
    pub const path = struct {
        pub fn bufJoin(buf: []u8, paths: []const []const u8) ![]u8 {
            var fa: std.heap.FixedBufferAllocator = .init(buf);
            return try std.Io.Dir.path.join(fa.allocator(), paths);
        }

        pub fn bufJoinZ(buf: []u8, paths: []const []const u8) ![:0]u8 {
            var fa: std.heap.FixedBufferAllocator = .init(buf);
            return try std.Io.Dir.path.joinZ(fa.allocator(), paths);
        }
    };

    pub fn readFileAlloc(dir: std.Io.Dir, io: std.Io, sub_path: []const u8, gpa: std.mem.Allocator, limit: std.Io.Limit) ![]u8 {
        const stat = try std.Io.Dir.statFile(dir, io, sub_path, .{});
        const buffer = try gpa.alloc(u8, limit.min(.limited64(stat.size)));
        errdefer gpa.free(buffer);
        _ = try std.Io.Dir.readFile(.cwd(), io, sub_path, buffer);
        return buffer;
    }
};
