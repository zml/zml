const std = @import("std");

const mlir = @import("mlir");
const ub = @import("mlir/dialects/ub");

test {
    std.testing.refAllDecls(ub);
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
