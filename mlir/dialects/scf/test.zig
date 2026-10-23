const std = @import("std");

const mlir = @import("mlir");
const scf = @import("mlir/dialects/scf");

const operation_names: []const []const u8 = &.{
    "scf.condition",
    "scf.execute_region",
    "scf.for",
    "scf.forall",
    "scf.forall.in_parallel",
    "scf.if",
    "scf.index_switch",
    "scf.parallel",
    "scf.reduce",
    "scf.reduce.return",
    "scf.while",
    "scf.yield",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(scf);
    inline for (comptime std.meta.declarations(scf)) |decl| {
        const T = @field(scf, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    scf.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
