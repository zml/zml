const std = @import("std");
const DoubleBuffer = @import("double_buffer.zig").DoubleBuffer;

test "swap publishes the back" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var buffer: DoubleBuffer(u32) = .{ .values = .{
        .{ .value = 1, .arena = .init(gpa) },
        .{ .value = 2, .arena = .init(gpa) },
    } };
    buffer.back().value = 3;
    const before = buffer.lock(io).value;
    buffer.publishLocked();
    buffer.unlock(io);
    try std.testing.expectEqual(1, before);
    defer buffer.unlock(io);
    try std.testing.expectEqual(3, buffer.lock(io).value);
}
