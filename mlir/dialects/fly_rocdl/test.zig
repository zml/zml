const std = @import("std");

const fly_rocdl = @import("mlir/dialects/fly_rocdl");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "fly_rocdl.get_buffer_rsrc",
    "fly_rocdl.make_tiled_tdm_load_atom",
    "fly_rocdl.make_tiled_tdm_store_atom",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(fly_rocdl);
    inline for (comptime std.meta.declarations(fly_rocdl)) |decl| {
        const T = @field(fly_rocdl, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly_rocdl.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
