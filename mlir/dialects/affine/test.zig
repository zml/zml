const std = @import("std");

const affine = @import("mlir/dialects/affine");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "affine.apply",
    "affine.delinearize_index",
    "affine.for",
    "affine.if",
    "affine.linearize_index",
    "affine.load",
    "affine.max",
    "affine.min",
    "affine.parallel",
    "affine.store",
    "affine.vector_load",
    "affine.vector_store",
    "affine.yield",
};

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
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
