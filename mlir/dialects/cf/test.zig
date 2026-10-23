const std = @import("std");

const cf = @import("mlir/dialects/cf");
const mlir = @import("mlir");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(cf);
    inline for (comptime std.meta.declarations(cf)) |decl| {
        const T = @field(cf, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    cf.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (cf.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
