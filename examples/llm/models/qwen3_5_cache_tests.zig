const std = @import("std");

const zml = @import("zml");
const qwen3_5 = @import("qwen3_5/model.zig");

fn validPrefixConvState(input: zml.Tensor, valid_len: zml.Tensor) zml.Tensor {
    return qwen3_5.GatedDeltaNet.buildUpdatedConvStateFromPrefix(input, 3, valid_len);
}

test "Qwen3.5 convolution cache uses only the valid prefix" {
    const platform = zml.testing.env();
    const input: zml.Tensor = .init(.{ .b = 1, .s = 5, .mix = 1 }, .f32);
    const valid_len: zml.Tensor = .init(.{}, .u32);
    var exe = try platform.compileFn(
        std.testing.allocator,
        std.testing.io,
        validPrefixConvState,
        .{ input, valid_len },
        .{},
    );
    defer exe.deinit();

    var input_buffer: zml.Buffer = try .fromBytes(
        std.testing.io,
        platform,
        input.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[5]f32{ 10, 20, 30, 777, 888 }),
    );
    defer input_buffer.deinit();

    const expected = [_][3]f32{
        .{ 0, 0, 0 },
        .{ 0, 0, 10 },
        .{ 0, 10, 20 },
        .{ 10, 20, 30 },
    };
    for (expected, 0..) |values, length| {
        var valid_len_buffer = try zml.Buffer.scalar(std.testing.io, platform, @as(u32, @intCast(length)), .u32);
        defer valid_len_buffer.deinit();
        var output = try zml.testing.autoCall(
            std.testing.allocator,
            std.testing.io,
            &exe,
            validPrefixConvState,
            .{ input_buffer, valid_len_buffer },
        );
        defer output.deinit();
        try zml.testing.expectClose(
            std.testing.io,
            zml.Slice.initConst(output.shape(), std.mem.sliceAsBytes(&values)),
            output,
            .{ .absolute_tolerance = 0, .relative_tolerance = 0, .minimum_close_fraction = 1 },
        );
    }
}
