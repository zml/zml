const std = @import("std");

const mlir = @import("mlir");
const ub = @import("mlir/dialects/ub");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(ub);
    inline for (comptime std.meta.declarations(ub)) |decl| {
        const T = @field(ub, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "registerDialect registers ub" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    ub.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    try std.testing.expect(ctx.isRegisteredOperation("ub.poison"));
}
