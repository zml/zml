const std = @import("std");

const math = @import("mlir/dialects/math");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "math.absf",
    "math.absi",
    "math.acos",
    "math.acosh",
    "math.asin",
    "math.asinh",
    "math.atan",
    "math.atan2",
    "math.atanh",
    "math.cbrt",
    "math.ceil",
    "math.clampf",
    "math.copysign",
    "math.cos",
    "math.cosh",
    "math.ctlz",
    "math.ctpop",
    "math.cttz",
    "math.erf",
    "math.erfc",
    "math.exp",
    "math.exp2",
    "math.expm1",
    "math.floor",
    "math.fma",
    "math.fpowi",
    "math.ipowi",
    "math.isfinite",
    "math.isinf",
    "math.isnan",
    "math.isnormal",
    "math.log",
    "math.log10",
    "math.log1p",
    "math.log2",
    "math.powf",
    "math.round",
    "math.roundeven",
    "math.rsqrt",
    "math.sin",
    "math.sincos",
    "math.sinh",
    "math.sqrt",
    "math.tan",
    "math.tanh",
    "math.trunc",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(math);
    inline for (comptime std.meta.declarations(math)) |decl| {
        const T = @field(math, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    math.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
