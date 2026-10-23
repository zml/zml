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
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(vector);
    inline for (comptime std.meta.declarations(vector)) |decl| {
        const T = @field(vector, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "vector enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    inline for (@typeInfo(vector.CombiningKind).@"enum".field_names) |name| {
        const value: vector.CombiningKind = @field(vector.CombiningKind, name);
        const attr = try vector.CombiningKindAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.value());
        try std.testing.expect(value.attribute(ctx).isA(vector.CombiningKindAttr) != null);
        try expectPrints("#vector.kind<" ++ name ++ ">", attr);
    }
    inline for (@typeInfo(vector.IteratorType).@"enum".field_names) |name| {
        const value: vector.IteratorType = @field(vector.IteratorType, name);
        const attr = try vector.IteratorTypeAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.value());
        try expectPrints("#vector.iterator_type<" ++ name ++ ">", attr);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    vector.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (vector.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
