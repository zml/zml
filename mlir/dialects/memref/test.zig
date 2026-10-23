const std = @import("std");

const memref = @import("mlir/dialects/memref");
const mlir = @import("mlir");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(memref);
    inline for (comptime std.meta.declarations(memref)) |decl| {
        const T = @field(memref, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    memref.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (memref.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "atomic_rmw kinds print as their IR spelling" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    memref.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc = mlir.Location.unknown(ctx);
    const block = mlir.Block.init(&.{ .float(ctx, .f32), try mlir.Type.parse(ctx, "memref<4xf32>"), .int(ctx, .i32), try mlir.Type.parse(ctx, "memref<4xi32>"), .index(ctx) }, &.{ loc, loc, loc, loc, loc });
    defer block.deinit();
    inline for (comptime std.meta.tags(memref.AtomicRMWKind)) |kind| {
        const float = switch (kind) {
            .addf, .assign, .maximumf, .maxnumf, .minimumf, .minnumf, .mulf => true,
            else => false,
        };
        const value = block.argument(if (float) 0 else 2);
        const ref = block.argument(if (float) 1 else 3);
        const op = memref.atomic_rmw(ctx, kind, value, ref, &.{block.argument(4)}, loc);
        defer op.deinit();
        var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer text.deinit();
        try text.writer.print("{f}", .{op});
        try std.testing.expect(std.mem.indexOf(u8, text.written(), "memref.atomic_rmw " ++ @tagName(kind) ++ " ") != null);
    }
}
