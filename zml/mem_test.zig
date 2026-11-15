const std = @import("std");
const Allocator = std.mem.Allocator;

const zml = @import("zml");
const DynamicBufferPool = zml.mem.DynamicBufferPool;

test "DynamicBufferPool reuses and frees buffers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: DynamicBufferPool = .init(2, 256);
    defer pool.deinit(allocator);

    const first_ptr = blk: {
        const first = try pool.get(allocator, io);
        defer pool.put(io, first);
        const second = try pool.get(allocator, io);
        defer pool.put(io, second);
        try std.testing.expectEqual(@as(usize, 256), first.len);
        try std.testing.expect(first.ptr != second.ptr);
        @memset(first, 0xa5);
        @memset(second, 0x5a);
        break :blk first.ptr;
    };
    const reused = try pool.get(allocator, io);
    defer pool.put(io, reused);
    try std.testing.expectEqual(first_ptr, reused.ptr);
}

test "DynamicBufferPool cleans up allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            var pool: DynamicBufferPool = .init(2, 256);
            defer pool.deinit(allocator);
            const first = pool.get(allocator, std.testing.io) catch |err| {
                try std.testing.expectEqual(@as(usize, 0), pool.in_flight.load(.acquire));
                return err;
            };
            defer pool.put(std.testing.io, first);
            const second = pool.get(allocator, std.testing.io) catch |err| {
                try std.testing.expectEqual(@as(usize, 1), pool.in_flight.load(.acquire));
                return err;
            };
            defer pool.put(std.testing.io, second);
        }
    }.run, .{});
}
