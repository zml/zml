const std = @import("std");
const testing = std.testing;
const zml = @import("zml.zig");
const Buffer = zml.Buffer;
const Shape = zml.Shape;
const DataType = zml.DataType;

test "Buffer.fromSlice basic usage" {
    const platform = try zml.testing.env();
    defer platform.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var buf = try Buffer.fromSlice(platform, .{ 2, 3 }, &data);
    defer buf.deinit();

    try testing.expectEqual(Shape.init(.{ 2, 3 }, .f32), buf.shape());

    const result = try buf.getValue([6]f32);
    try testing.expectEqualSlices(f32, &data, &result);
}

test "Buffer.constant creation" {
    const platform = try zml.testing.env();
    defer platform.deinit();

    // Test scalar constant
    {
        var buf = try Buffer.constant(platform, Shape.init(.{}, .f32), 42.0);
        defer buf.deinit();

        const val = try buf.getValue(f32);
        try testing.expectEqual(@as(f32, 42.0), val);
    }

    // Test multi-dimensional constant
    {
        var buf = try Buffer.constant(platform, Shape.init(.{ 2, 3 }, .i32), 7);
        defer buf.deinit();

        const result = try buf.getValue([6]i32);
        try testing.expectEqualSlices(i32, &[_]i32{7} ** 6, &result);
    }
}

test "Buffer.fromArray" {
    const platform = try zml.testing.env();
    defer platform.deinit();

    const arr = [2][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var buf = try Buffer.fromArray(platform, arr);
    defer buf.deinit();

    try testing.expectEqual(Shape.init(.{ 2, 3 }, .u8), buf.shape());

    const result = try buf.getValue([6]u8);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5, 6 }, &result);
}

test "Buffer dtype and shape operations" {
    const platform = try zml.testing.env();
    defer platform.deinit();

    var buf = try Buffer.constant(platform, Shape.init(.{ 2, 3, 4 }, .f32), 1.0);
    defer buf.deinit();

    try testing.expectEqual(DataType.f32, buf.dtype());
    try testing.expectEqual(@as(u4, 3), buf.rank());
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 3, 4 }, buf.dims());
}
