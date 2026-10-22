const std = @import("std");

const fly_rocdl = @import("mlir/dialects/fly_rocdl");
const mlir = @import("mlir");

test {
    std.testing.refAllDecls(fly_rocdl);
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly_rocdl.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (fly_rocdl.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
