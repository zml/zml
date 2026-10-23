const std = @import("std");

const llvm = @import("mlir/dialects/llvm");
const mlir = @import("mlir");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(llvm);
    inline for (comptime std.meta.declarations(llvm)) |decl| {
        const T = @field(llvm, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (llvm.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
