const std = @import("std");

const mlir = @import("mlir");
const stablehlo = @import("mlir/dialects/stablehlo");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(stablehlo);
    inline for (comptime std.meta.declarations(stablehlo)) |decl| {
        const T = @field(stablehlo, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    stablehlo.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (stablehlo.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
