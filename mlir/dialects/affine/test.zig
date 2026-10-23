const std = @import("std");

const affine = @import("mlir/dialects/affine");
const mlir = @import("mlir");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(affine);
    inline for (comptime std.meta.declarations(affine)) |decl| {
        const T = @field(affine, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    affine.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (affine.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
