const std = @import("std");

const func = @import("mlir/dialects/func");
const mlir = @import("mlir");
const shardy = @import("mlir/dialects/shardy");

const ManualAxesAttr = shardy.ManualAxesAttr;
const TensorShardingAttr = shardy.TensorShardingAttr;
const TensorShardingPerValueAttr = shardy.TensorShardingPerValueAttr;

test {
    std.testing.refAllDecls(shardy);
}

test TensorShardingAttr {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    shardy.registerDialect(registry);
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{.replicated(ctx)},
            .replicated_axes = &.{ .named(ctx, "link_x"), .named(ctx, "link_y") },
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "suggested_mesh",
            .dimensions = &.{
                .closed(ctx, &.{.named(ctx, "link_z")}),
                .closed(ctx, &.{.named(ctx, "link_x")}),
            },
            .replicated_axes = &.{.named(ctx, "link_y")},
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@suggested_mesh, [{\"link_z\"}, {\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "fold_mesh",
            .dimensions = &.{ .closed(ctx, &.{.named(ctx, "link_x")}), .replicated(ctx) },
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{.closed(ctx, &.{ .named(ctx, "link_x"), .named(ctx, "link_y") })},
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "strategy_fold",
            .dimensions = &.{.closed(ctx, &.{.named(ctx, "link_x")})},
            .replicated_axes = &.{.named(ctx, "link_y")},
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@strategy_fold, [{\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "mix_mesh",
            .dimensions = &.{ .open(ctx, &.{}), .replicated(ctx) },
            .replicated_axes = &.{ .named(ctx, "link_x"), .named(ctx, "link_y") },
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttr = .get(ctx, .{
            .mesh = "3d_mesh",
            .dimensions = &.{
                .closed(ctx, &.{.named(ctx, "link_x")}),
                .closed(ctx, &.{.named(ctx, "link_y")}),
                .closed(ctx, &.{.named(ctx, "link_z")}),
            },
        });
        try sharding.attribute().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@\"3d_mesh\", [{\"link_x\"}, {\"link_y\"}, {\"link_z\"}]>", w.written());
    }
}

test TensorShardingPerValueAttr {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    shardy.registerDialect(registry);
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const shardings: [2]*const TensorShardingAttr = .{
        .get(ctx, .{
            .mesh = "mesh",
            .dimensions = &.{.closed(ctx, &.{.named(ctx, "link_x")})},
        }),
        .get(ctx, .{
            .mesh = "mesh",
            .dimensions = &.{.replicated(ctx)},
            .replicated_axes = &.{.named(ctx, "link_y")},
        }),
    };

    const per_value = TensorShardingPerValueAttr.get(ctx, &shardings);

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    try per_value.attribute().format(&w.writer);
    try std.testing.expectEqualSlices(
        u8,
        "#sdy.sharding_per_value<[<@mesh, [{\"link_x\"}]>, <@mesh, [{}], replicated={\"link_y\"}>]>",
        w.written(),
    );
}

test ManualAxesAttr {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    shardy.registerDialect(registry);
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const axes: [2]*const mlir.StringAttribute = .{
        .init(ctx, "link_x"),
        .init(ctx, "link_y"),
    };

    const manual_axes: *const ManualAxesAttr = .get(ctx, &axes);

    try std.testing.expectEqual(2, manual_axes.numAxes());
    try std.testing.expectEqualStrings("link_x", manual_axes.axis(0));
    try std.testing.expectEqualStrings("link_y", manual_axes.axis(1));

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    try manual_axes.attribute().format(&w.writer);
    try std.testing.expectEqualSlices(u8, "#sdy<manual_axes{\"link_x\", \"link_y\"}>", w.written());
}
