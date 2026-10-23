const std = @import("std");

const func = @import("mlir/dialects/func");
const mlir = @import("mlir");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(func);
    inline for (comptime std.meta.declarations(func)) |decl| {
        const T = @field(func, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "func dialect and extensions register" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    func.registerDialect(registry);
    func.registerExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    try std.testing.expect(ctx.isRegisteredOperation("func.func"));
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    func.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (func.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
