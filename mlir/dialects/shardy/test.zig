const std = @import("std");

const func = @import("mlir/dialects/func");
const mlir = @import("mlir");
const shardy = @import("mlir/dialects/shardy");

const ManualAxesAttribute = shardy.ManualAxesAttribute;
const TensorShardingAttribute = shardy.TensorShardingAttribute;
const TensorShardingPerValueAttribute = shardy.TensorShardingPerValueAttribute;

test {
    std.testing.refAllDecls(shardy);
}

test TensorShardingAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{.replicated(ctx)},
            .replicated_axes = &.{ .named(ctx, "link_x"), .named(ctx, "link_y") },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "suggested_mesh",
            .dimensions = &.{
                .closed(ctx, &.{.named(ctx, "link_z")}),
                .closed(ctx, &.{.named(ctx, "link_x")}),
            },
            .replicated_axes = &.{.named(ctx, "link_y")},
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@suggested_mesh, [{\"link_z\"}, {\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "fold_mesh",
            .dimensions = &.{ .closed(ctx, &.{.named(ctx, "link_x")}), .replicated(ctx) },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{.closed(ctx, &.{ .named(ctx, "link_x"), .named(ctx, "link_y") })},
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "strategy_fold",
            .dimensions = &.{.closed(ctx, &.{.named(ctx, "link_x")})},
            .replicated_axes = &.{.named(ctx, "link_y")},
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@strategy_fold, [{\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "mix_mesh",
            .dimensions = &.{ .open(ctx, &.{}), .replicated(ctx) },
            .replicated_axes = &.{ .named(ctx, "link_x"), .named(ctx, "link_y") },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding: *const TensorShardingAttribute = .init(ctx, .{
            .mesh = "3d_mesh",
            .dimensions = &.{
                .closed(ctx, &.{.named(ctx, "link_x")}),
                .closed(ctx, &.{.named(ctx, "link_y")}),
                .closed(ctx, &.{.named(ctx, "link_z")}),
            },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@\"3d_mesh\", [{\"link_x\"}, {\"link_y\"}, {\"link_z\"}]>", w.written());
    }
}

test TensorShardingPerValueAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const shardings: [2]*const TensorShardingAttribute = .{
        .init(ctx, .{
            .mesh = "mesh",
            .dimensions = &.{.closed(ctx, &.{.named(ctx, "link_x")})},
        }),
        .init(ctx, .{
            .mesh = "mesh",
            .dimensions = &.{.replicated(ctx)},
            .replicated_axes = &.{.named(ctx, "link_y")},
        }),
    };

    const per_value = TensorShardingPerValueAttribute.init(ctx, &shardings);

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    try per_value.asAttr().format(&w.writer);
    try std.testing.expectEqualSlices(
        u8,
        "#sdy.sharding_per_value<[<@mesh, [{\"link_x\"}]>, <@mesh, [{}], replicated={\"link_y\"}>]>",
        w.written(),
    );
}

test ManualAxesAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    func.registerExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const axes: [2]*const mlir.StringAttribute = .{
        .init(ctx, "link_x"),
        .init(ctx, "link_y"),
    };

    const manual_axes: *const ManualAxesAttribute = .init(ctx, &axes);

    try std.testing.expectEqual(2, manual_axes.numAxes());
    try std.testing.expectEqualStrings("link_x", manual_axes.axis(0));
    try std.testing.expectEqualStrings("link_y", manual_axes.axis(1));

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    try manual_axes.asAttr().format(&w.writer);
    try std.testing.expectEqualSlices(u8, "#sdy<manual_axes{\"link_x\", \"link_y\"}>", w.written());
}
