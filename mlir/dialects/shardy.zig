//! Bindings for OpenXLA shardy dialect
//!
//! - dialect overview: https:openxla.org/shardy/sdy_dialect
//! - C API: https://github.com/openxla/shardy/blob/main/shardy/integrations/c/attributes.h
const std = @import("std");

const mlir = @import("mlir");

pub const Sharding = opaque {
    // TODO(codex) get inspiration from mlir/mlir.zig and mlir/dialects/stablehlo/stablehlo.zig to bind
    // openxla.org/shardy/sdy_dialect#tensorshardingattr
    // You will probably need to add other helper structs.

    pub fn init(ctx: *mlir.Context) *Sharding {
        // TODO(codex): fill me up
        _ = ctx;
        return undefined;
    }

    pub fn asAttr(sharding: *const Sharding) *const mlir.Attribute {
        return @ptrCast(sharding);
    }
};

test {
    const registry: *mlir.DialectRegistry = try .init();
    registry.registerDialect("sdy");
    mlir.registerFuncExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth });

    var strategy: Strategy = .init;
    strategy.addBinding(.batch, .link_x);

    {
        defer w.clearRetainingCapacity();
        // TODO(codex): properly init
        // Tentative syntax: = .init(ctx, "folded_mesh", &.{}, &.{.link_x, .link_y});
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        // TODO(codex): properly init
        // Tentative syntax: = .init(ctx, "suggested_mesh", &.{ .link_z, .link_x }, &.{.link_y});
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@suggested_mesh, [{\"link_z\"}, {\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@strategy_fold, [{\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const Sharding = .init(ctx);
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@3d_mesh, [{\"link_x\"}, {\"link_y\"}, {\"link_z\"}]>", w.written());
    }
}
