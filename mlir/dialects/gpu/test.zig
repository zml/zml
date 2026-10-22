const std = @import("std");

const gpu = @import("mlir/dialects/gpu");
const mlir = @import("mlir");

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

test {
    std.testing.refAllDecls(gpu);
}

test "gpu enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    inline for (@typeInfo(gpu.Dim).@"enum".field_names) |name| {
        const value: gpu.Dim = @field(gpu.Dim, name);
        const attr = try gpu.DimensionAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try std.testing.expect(attr.attribute().isA(gpu.DimensionAttr) != null);
        try expectPrints("#gpu.dim<" ++ name ++ ">", attr);
    }
    inline for (@typeInfo(gpu.ShuffleMode).@"enum".field_names) |name| {
        const value: gpu.ShuffleMode = @field(gpu.ShuffleMode, name);
        const attr = try gpu.ShuffleModeAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try expectPrints("#gpu.shuffle_mode<" ++ name ++ ">", attr);
    }
}
