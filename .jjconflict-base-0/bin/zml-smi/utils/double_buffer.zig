const std = @import("std");

pub fn DoubleBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Value = T;

        values: [2]T,
        current: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

        pub fn front(self: *const Self) *const T {
            return &self.values[self.current.load(.acquire)];
        }

        pub fn back(self: *Self) *T {
            return &self.values[1 - self.current.load(.acquire)];
        }

        pub fn swap(self: *Self) void {
            self.current.store(1 - self.current.load(.acquire), .release);
        }

        pub fn jsonStringify(self: *const Self, jw: *std.json.Stringify) !void {
            try jw.write(self.front().*);
        }

        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !Self {
            const val = try std.json.innerParseFromValue(T, allocator, source, options);
            return .{ .values = .{ val, val } };
        }
    };
}
