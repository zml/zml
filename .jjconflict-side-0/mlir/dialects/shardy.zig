//! Bindings for OpenXLA shardy dialect
//!
//! - dialect overview: https://openxla.org/shardy/sdy_dialect
//! - C API: https://github.com/openxla/shardy/blob/main/shardy/integrations/c/attributes.h
const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub fn dialectHandle() *const mlir.DialectHandle {
    return @ptrCast(c.mlirGetDialectHandle__sdy__().ptr);
}

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    dialectHandle().insertDialect(registry);
}

pub const MeshAxisAttribute = opaque {
    const M = mlir.Methods(MeshAxisAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAMeshAxisAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(ctx: *mlir.Context, axis_name: []const u8, axis_size: i64) *const MeshAxisAttribute {
        return @ptrCast(c.sdyMeshAxisAttrGet(ctx.ptr(), mlir.stringRef(axis_name), axis_size).ptr);
    }

    pub fn asAttr(self: *const MeshAxisAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn name(self: *const MeshAxisAttribute) []const u8 {
        return mlir.string(c.sdyMeshAxisAttrGetName(self.ptr()));
    }

    pub fn size(self: *const MeshAxisAttribute) i64 {
        return c.sdyMeshAxisAttrGetSize(self.ptr());
    }
};

pub const MeshAttribute = opaque {
    const M = mlir.Methods(MeshAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAMeshAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const Mesh = struct {
        axes: []const *const MeshAxisAttribute,
        device_ids: []const i64 = &.{},
    };

    pub fn init(ctx: *mlir.Context, mesh: Mesh) *const MeshAttribute {
        const axes: []const c.struct_MlirAttribute = @ptrCast(mesh.axes);
        return @ptrCast(c.sdyMeshAttrGet(
            ctx.ptr(),
            @intCast(axes.len),
            axes.ptr,
            @intCast(mesh.device_ids.len),
            mesh.device_ids.ptr,
        ).ptr);
    }

    pub fn asAttr(self: *const MeshAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub const SubAxisInfoAttribute = opaque {
    const M = mlir.Methods(SubAxisInfoAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsASubAxisInfoAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(ctx: *mlir.Context, pre_size: i64, sub_axis_size: i64) *const SubAxisInfoAttribute {
        return @ptrCast(c.sdySubAxisInfoAttrGet(ctx.ptr(), pre_size, sub_axis_size).ptr);
    }

    pub fn asAttr(self: *const SubAxisInfoAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn preSize(self: *const SubAxisInfoAttribute) i64 {
        return c.sdySubAxisInfoAttrGetPreSize(self.ptr());
    }

    pub fn size(self: *const SubAxisInfoAttribute) i64 {
        return c.sdySubAxisInfoAttrGetSize(self.ptr());
    }
};

pub const AxisRefAttribute = opaque {
    const M = mlir.Methods(AxisRefAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAnAxisRefAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn named(ctx: *mlir.Context, axis_name: []const u8) *const AxisRefAttribute {
        return @ptrCast(c.sdyAxisRefAttrGet(ctx.ptr(), mlir.stringRef(axis_name), .{ .ptr = null }).ptr);
    }

    pub fn subAxis(ctx: *mlir.Context, axis_name: []const u8, pre_size: i64, size: i64) *const AxisRefAttribute {
        const sub_axis: *const SubAxisInfoAttribute = .init(ctx, pre_size, size);
        return @ptrCast(c.sdyAxisRefAttrGet(ctx.ptr(), mlir.stringRef(axis_name), sub_axis).ptr);
    }

    pub fn asAttr(self: *const AxisRefAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn name(self: *const AxisRefAttribute) []const u8 {
        return mlir.string(c.sdyAxisRefAttrGetName(self.ptr()));
    }

    pub fn subAxisInfo(self: *const AxisRefAttribute) ?*const SubAxisInfoAttribute {
        const attr = c.sdyAxisRefAttrGetSubAxisInfo(self.ptr());
        return if (attr.ptr == null) null else @ptrCast(attr.ptr);
    }
};

pub const DimensionShardingAttribute = opaque {
    const M = mlir.Methods(DimensionShardingAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsADimensionShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(
        ctx: *mlir.Context,
        axes: []const *const AxisRefAttribute,
        is_closed: bool,
        priority_: ?i64,
    ) *const DimensionShardingAttribute {
        const attrs: []const c.struct_MlirAttribute = @ptrCast(axes);
        return @ptrCast(c.sdyDimensionShardingAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr, is_closed, priority_ orelse -1).ptr);
    }

    pub fn closed(ctx: *mlir.Context, axes: []const *const AxisRefAttribute) *const DimensionShardingAttribute {
        return init(ctx, axes, true, null);
    }

    pub fn open(ctx: *mlir.Context, axes: []const *const AxisRefAttribute) *const DimensionShardingAttribute {
        return init(ctx, axes, false, null);
    }

    pub fn replicated(ctx: *mlir.Context) *const DimensionShardingAttribute {
        return closed(ctx, &.{});
    }

    pub fn asAttr(self: *const DimensionShardingAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn numAxes(self: *const DimensionShardingAttribute) usize {
        return @intCast(c.sdyDimensionShardingAttrGetAxesSize(self.ptr()));
    }

    pub fn axis(self: *const DimensionShardingAttribute, index: usize) *const AxisRefAttribute {
        return @ptrCast(c.sdyDimensionShardingAttrGetAxesElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn isClosed(self: *const DimensionShardingAttribute) bool {
        return c.sdyDimensionShardingAttrGetIsClosed(self.ptr());
    }

    pub fn priority(self: *const DimensionShardingAttribute) ?i64 {
        const value = c.sdyDimensionShardingAttrGetPriority(self.ptr());
        return if (value == -1) null else value;
    }
};

pub const Sharding = struct {
    mesh: []const u8,
    dimensions: []const *const DimensionShardingAttribute,
    replicated_axes: []const *const AxisRefAttribute = &.{},
    unreduced_axes: []const *const AxisRefAttribute = &.{},
};

pub const TensorShardingAttribute = opaque {
    const M = mlir.Methods(TensorShardingAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsATensorShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(ctx: *mlir.Context, args: Sharding) *const TensorShardingAttribute {
        comptime std.debug.assert(@sizeOf(c.struct_MlirAttribute) == @sizeOf(*const mlir.Attribute));
        const dimensions: []const c.struct_MlirAttribute = @ptrCast(args.dimensions);
        const replicated: []const c.struct_MlirAttribute = @ptrCast(args.replicated_axes);
        const unreduced: []const c.struct_MlirAttribute = @ptrCast(args.unreduced_axes);
        return @ptrCast(c.sdyTensorShardingAttrGet(
            ctx.ptr(),
            mlir.Attribute.flatSymbolRef(ctx, args.mesh).ptr(),
            @intCast(dimensions.len),
            dimensions.ptr,
            @intCast(replicated.len),
            replicated.ptr,
            @intCast(unreduced.len),
            unreduced.ptr,
        ).ptr);
    }

    pub fn asAttr(self: *const TensorShardingAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn meshOrRef(self: *const TensorShardingAttribute) *const mlir.Attribute {
        return @ptrCast(c.sdyTensorShardingAttrGetMeshOrRef(self.ptr()).ptr);
    }

    pub fn numDimensions(self: *const TensorShardingAttribute) usize {
        return @intCast(c.sdyTensorShardingAttrGetDimShardingsSize(self.ptr()));
    }

    pub fn dimension(self: *const TensorShardingAttribute, index: usize) *const DimensionShardingAttribute {
        return @ptrCast(c.sdyTensorShardingAttrGetDimShardingsElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numReplicatedAxes(self: *const TensorShardingAttribute) usize {
        return @intCast(c.sdyTensorShardingAttrGetReplicatedAxesSize(self.ptr()));
    }

    pub fn replicatedAxis(self: *const TensorShardingAttribute, index: usize) *const AxisRefAttribute {
        return @ptrCast(c.sdyTensorShardingAttrGetReplicatedAxesElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numUnreducedAxes(self: *const TensorShardingAttribute) usize {
        return @intCast(c.sdyTensorShardingAttrGetUnreducedAxesSize(self.ptr()));
    }

    pub fn unreducedAxis(self: *const TensorShardingAttribute, index: usize) *const AxisRefAttribute {
        return @ptrCast(c.sdyTensorShardingAttrGetUnreducedAxesElem(self.ptr(), @intCast(index)).ptr);
    }
};

test TensorShardingAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    mlir.registerFuncExtensions(registry);

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

pub const TensorShardingPerValueAttribute = opaque {
    const M = mlir.Methods(TensorShardingPerValueAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsATensorShardingPerValueAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(ctx: *mlir.Context, shardings: []const *const TensorShardingAttribute) *const TensorShardingPerValueAttribute {
        const attrs: []const c.struct_MlirAttribute = @ptrCast(shardings);
        return @ptrCast(c.sdyTensorShardingPerValueAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr).ptr);
    }

    pub fn asAttr(self: *const TensorShardingPerValueAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

test TensorShardingPerValueAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    mlir.registerFuncExtensions(registry);

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

pub const ManualAxesAttribute = opaque {
    const M = mlir.Methods(ManualAxesAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAManualAxesAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn init(ctx: *mlir.Context, axes: []const *const mlir.StringAttribute) *const ManualAxesAttribute {
        comptime std.debug.assert(@sizeOf(c.struct_MlirAttribute) == @sizeOf(*const mlir.StringAttribute));
        const attrs: []const c.struct_MlirAttribute = @ptrCast(axes);
        return @ptrCast(c.sdyManualAxesAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr).ptr);
    }

    pub fn asAttr(self: *const ManualAxesAttribute) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn numAxes(self: *const ManualAxesAttribute) usize {
        return @intCast(c.sdyManualAxesAttrGetAxesSize(self.ptr()));
    }

    pub fn axis(self: *const ManualAxesAttribute, index: usize) []const u8 {
        return mlir.string(c.sdyManualAxesAttrGetAxesElem(self.ptr(), @intCast(index)));
    }
};

test ManualAxesAttribute {
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    mlir.registerFuncExtensions(registry);

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
