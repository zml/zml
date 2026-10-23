const std = @import("std");

const arith = @import("mlir/dialects/arith");
const mlir = @import("mlir");

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(arith);
    inline for (comptime std.meta.declarations(arith)) |decl| {
        const T = @field(arith, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "arith fastmath attribute is built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const cases = [_]struct { arith.FastMathFlags, []const u8 }{
        .{ .none, "#arith.fastmath<none>" },
        .{ .fast, "#arith.fastmath<fast>" },
        .{ .{ .reassoc = true }, "#arith.fastmath<reassoc>" },
        .{ .{ .nnan = true, .ninf = true, .afn = true }, "#arith.fastmath<nnan,ninf,afn>" },
        .{ .{ .nsz = true, .arcp = true, .contract = true }, "#arith.fastmath<nsz,arcp,contract>" },
    };
    for (cases) |case| {
        const attr = try arith.FastMathAttr.get(ctx, .{ .value = case[0] });
        try std.testing.expectEqual(case[0], attr.value());
        try std.testing.expect(attr.attribute().isA(arith.FastMathAttr) != null);
        try expectPrints(case[1], attr);
    }
    try std.testing.expectError(error.InvalidMlir, arith.FastMathAttr.get(ctx, .{ .value = .{ ._padding = 1 } }));
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    arith.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (arith.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
