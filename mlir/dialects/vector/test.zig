const std = @import("std");

const mlir = @import("mlir");
const vector = @import("mlir/dialects/vector");

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

test {
    std.testing.refAllDecls(vector);
}

test "vector enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    inline for (@typeInfo(vector.CombiningKind).@"enum".field_names) |name| {
        const value: vector.CombiningKind = @field(vector.CombiningKind, name);
        const attr = try vector.CombiningKindAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try std.testing.expect(value.attribute(ctx).isA(vector.CombiningKindAttr) != null);
        try expectPrints("#vector.kind<" ++ name ++ ">", attr);
    }
    inline for (@typeInfo(vector.IteratorType).@"enum".field_names) |name| {
        const value: vector.IteratorType = @field(vector.IteratorType, name);
        const attr = try vector.IteratorTypeAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try expectPrints("#vector.iterator_type<" ++ name ++ ">", attr);
    }
}
