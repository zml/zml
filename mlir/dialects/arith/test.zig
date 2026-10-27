const std = @import("std");

const arith = @import("mlir/dialects/arith");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "arith.addf",
    "arith.addi",
    "arith.addui_extended",
    "arith.andi",
    "arith.bitcast",
    "arith.ceildivsi",
    "arith.ceildivui",
    "arith.cmpf",
    "arith.cmpi",
    "arith.constant",
    "arith.convertf",
    "arith.divf",
    "arith.divsi",
    "arith.divui",
    "arith.extf",
    "arith.extsi",
    "arith.extui",
    "arith.floordivsi",
    "arith.flush_denormals",
    "arith.fptosi",
    "arith.fptoui",
    "arith.index_cast",
    "arith.index_castui",
    "arith.maximumf",
    "arith.maxnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minimumf",
    "arith.minnumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.mulsi_extended",
    "arith.mului_extended",
    "arith.negf",
    "arith.ori",
    "arith.remf",
    "arith.remsi",
    "arith.remui",
    "arith.scaling_truncf",
    "arith.select",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.sitofp",
    "arith.subf",
    "arith.subi",
    "arith.truncf",
    "arith.trunci",
    "arith.uitofp",
    "arith.xori",
};

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

test "fastmath names the attribute of a float operation" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{mlir.Attribute.dict(ctx, &.{try arith.fastmath(ctx, .fast)})});
    try std.testing.expectEqualStrings("{fastmath = #arith.fastmath<fast>}", w.buffered());
}

test "overflowFlags names the attribute of an integer operation" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{mlir.Attribute.dict(ctx, &.{try arith.overflowFlags(ctx, .{ .nsw = true })})});
    try std.testing.expectEqualStrings("{overflowFlags = #arith.overflow<nsw>}", w.buffered());
}

test "arith overflow attribute is built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const cases = [_]struct { arith.IntegerOverflowFlags, []const u8 }{
        .{ .none, "#arith.overflow<none>" },
        .{ .{ .nsw = true }, "#arith.overflow<nsw>" },
        .{ .{ .nsw = true, .nuw = true }, "#arith.overflow<nsw, nuw>" },
    };
    for (cases) |case| {
        const attr = try arith.IntegerOverflowAttr.get(ctx, .{ .value = case[0] });
        try std.testing.expectEqual(case[0], attr.value());
        try std.testing.expect(attr.attribute().isA(arith.IntegerOverflowAttr) != null);
        try expectPrints(case[1], attr);
    }
    try std.testing.expectError(error.InvalidMlir, arith.IntegerOverflowAttr.get(ctx, .{ .value = .{ ._padding = 1 } }));
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    arith.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
