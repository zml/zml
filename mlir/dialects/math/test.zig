const std = @import("std");

const math = @import("mlir/dialects/math");
const mlir = @import("mlir");

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
    for (math.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
