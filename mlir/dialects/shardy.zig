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

fn allocNativeAttrs(allocator: std.mem.Allocator, attrs: anytype) ![]c.MlirAttribute {
    const native_attrs = try allocator.alloc(c.MlirAttribute, attrs.len);
    errdefer allocator.free(native_attrs);

    for (attrs, 0..) |attr, i| {
        native_attrs[i] = attr.ptr();
    }

    return native_attrs;
}

pub const MeshAxisAttribute = opaque {
    const M = mlir.Methods(MeshAxisAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAMeshAxisAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, axis_name: []const u8, axis_size: i64) *const MeshAxisAttribute {
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

    pub const GetArgs = struct {
        axes: []const *const MeshAxisAttribute,
        device_ids: []const i64 = &.{},
    };

    pub fn get(allocator: std.mem.Allocator, ctx: *mlir.Context, args: GetArgs) !*const MeshAttribute {
        const native_axes = try allocNativeAttrs(allocator, args.axes);
        defer allocator.free(native_axes);

        return @ptrCast(c.sdyMeshAttrGet(
            ctx.ptr(),
            @intCast(native_axes.len),
            native_axes.ptr,
            @intCast(args.device_ids.len),
            args.device_ids.ptr,
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

    pub fn get(ctx: *mlir.Context, pre_size: i64, sub_axis_size: i64) *const SubAxisInfoAttribute {
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

pub const Axis = union(enum) {
    named: []const u8,
    sub_axis: struct {
        name: []const u8,
        pre_size: i64,
        size: i64,
    },

    pub const link_x = .{ .named = "link_x" };
    pub const link_y = .{ .named = "link_y" };
    pub const link_z = .{ .named = "link_z" };

    pub fn named_(name: []const u8) Axis {
        return .{ .named = name };
    }

    pub fn subAxis(name: []const u8, pre_size: i64, size: i64) Axis {
        return .{ .sub_axis = .{
            .name = name,
            .pre_size = pre_size,
            .size = size,
        } };
    }

    fn toAttr(self: Axis, ctx: *mlir.Context) *const AxisRefAttribute {
        return switch (self) {
            .named => |name| AxisRefAttribute.get(ctx, .{ .name = name }),
            .sub_axis => |sub_axis| AxisRefAttribute.get(ctx, .{
                .name = sub_axis.name,
                .sub_axis_info = SubAxisInfoAttribute.get(ctx, sub_axis.pre_size, sub_axis.size),
            }),
        };
    }
};

pub const AxisRefAttribute = opaque {
    const M = mlir.Methods(AxisRefAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAnAxisRefAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const GetArgs = struct {
        name: []const u8,
        sub_axis_info: ?*const SubAxisInfoAttribute = null,
    };

    pub fn get(ctx: *mlir.Context, args: GetArgs) *const AxisRefAttribute {
        const sub_axis_info = if (args.sub_axis_info) |info| info.ptr() else c.MlirAttribute{ .ptr = null };
        return @ptrCast(c.sdyAxisRefAttrGet(
            ctx.ptr(),
            mlir.stringRef(args.name),
            sub_axis_info,
        ).ptr);
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

pub const Dimension = struct {
    axes: []const Axis,
    is_closed: bool,
    priority: ?i64,

    pub fn closed(axes: []const Axis) Dimension {
        return .{ .axes = axes, .is_closed = true, .priority = null };
    }

    pub fn open(axes: []const Axis) Dimension {
        return .{ .axes = axes, .is_closed = false, .priority = null };
    }

    pub const replicated: Dimension = .{ .axes = &.{}, .is_closed = true, .priority = null };
};

pub const DimensionShardingAttribute = opaque {
    const M = mlir.Methods(DimensionShardingAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsADimensionShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const GetArgs = struct {
        axes: []const *const AxisRefAttribute = &.{},
        is_closed: bool = true,
        priority: ?i64 = null,
    };

    pub fn get(allocator: std.mem.Allocator, ctx: *mlir.Context, args: GetArgs) !*const DimensionShardingAttribute {
        const native_axes = try allocNativeAttrs(allocator, args.axes);
        defer allocator.free(native_axes);

        return @ptrCast(c.sdyDimensionShardingAttrGet(
            ctx.ptr(),
            @intCast(native_axes.len),
            native_axes.ptr,
            args.is_closed,
            args.priority orelse -1,
        ).ptr);
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

pub const TensorShardingAttribute = opaque {
    const M = mlir.Methods(TensorShardingAttribute, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsATensorShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const GetArgs = struct {
        mesh_or_ref: *const mlir.Attribute,
        dim_shardings: []const *const DimensionShardingAttribute,
        replicated_axes: []const *const AxisRefAttribute = &.{},
        unreduced_axes: []const *const AxisRefAttribute = &.{},
    };

    pub const InitArgs = struct {
        mesh: []const u8,
        dimensions: []const Dimension,
        replicated_axes: []const Axis = &.{},
        unreduced_axes: []const Axis = &.{},
    };

    pub fn get(allocator: std.mem.Allocator, ctx: *mlir.Context, args: GetArgs) !*const TensorShardingAttribute {
        const native_dim_shardings = try allocNativeAttrs(allocator, args.dim_shardings);
        defer allocator.free(native_dim_shardings);

        const native_replicated_axes = try allocNativeAttrs(allocator, args.replicated_axes);
        defer allocator.free(native_replicated_axes);

        const native_unreduced_axes = try allocNativeAttrs(allocator, args.unreduced_axes);
        defer allocator.free(native_unreduced_axes);

        return @ptrCast(c.sdyTensorShardingAttrGet(
            ctx.ptr(),
            args.mesh_or_ref.ptr(),
            @intCast(native_dim_shardings.len),
            native_dim_shardings.ptr,
            @intCast(native_replicated_axes.len),
            native_replicated_axes.ptr,
            @intCast(native_unreduced_axes.len),
            native_unreduced_axes.ptr,
        ).ptr);
    }

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, args: InitArgs) !*const TensorShardingAttribute {
        const dim_shardings = try allocator.alloc(*const DimensionShardingAttribute, args.dimensions.len);
        defer allocator.free(dim_shardings);

        for (args.dimensions, 0..) |dim_spec, i| {
            const axis_attrs = try allocator.alloc(*const AxisRefAttribute, dim_spec.axes.len);
            defer allocator.free(axis_attrs);

            for (dim_spec.axes, 0..) |axis, j| {
                axis_attrs[j] = axis.toAttr(ctx);
            }

            dim_shardings[i] = try DimensionShardingAttribute.get(allocator, ctx, .{
                .axes = axis_attrs,
                .is_closed = dim_spec.is_closed,
                .priority = dim_spec.priority,
            });
        }

        const replicated_axes = try allocator.alloc(*const AxisRefAttribute, args.replicated_axes.len);
        defer allocator.free(replicated_axes);
        for (args.replicated_axes, 0..) |axis, i| {
            replicated_axes[i] = axis.toAttr(ctx);
        }

        const unreduced_axes = try allocator.alloc(*const AxisRefAttribute, args.unreduced_axes.len);
        defer allocator.free(unreduced_axes);
        for (args.unreduced_axes, 0..) |axis, i| {
            unreduced_axes[i] = axis.toAttr(ctx);
        }

        return get(allocator, ctx, .{
            .mesh_or_ref = mlir.flatSymbolRefAttribute(ctx, args.mesh),
            .dim_shardings = dim_shardings,
            .replicated_axes = replicated_axes,
            .unreduced_axes = unreduced_axes,
        });
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

test {
    const Sharding = TensorShardingAttribute;

    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registerDialect(registry);
    mlir.registerFuncExtensions(registry);

    var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();

    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{.replicated},
            .replicated_axes = &.{ Axis.named_("link_x"), Axis.named_("link_y") },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "suggested_mesh",
            .dimensions = &.{
                Dimension.closed(&.{Axis.named_("link_z")}),
                Dimension.closed(&.{Axis.named_("link_x")}),
            },
            .replicated_axes = &.{Axis.named_("link_y")},
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@suggested_mesh, [{\"link_z\"}, {\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "fold_mesh",
            .dimensions = &.{
                Dimension.closed(&.{Axis.named_("link_x")}),
                .replicated,
            },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "folded_mesh",
            .dimensions = &.{
                Dimension.closed(&.{ Axis.named_("link_x"), Axis.named_("link_y") }),
            },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "strategy_fold",
            .dimensions = &.{
                Dimension.closed(&.{Axis.named_("link_x")}),
            },
            .replicated_axes = &.{Axis.named_("link_y")},
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@strategy_fold, [{\"link_x\"}], replicated={\"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "mix_mesh",
            .dimensions = &.{
                Dimension.open(&.{}),
                .replicated,
            },
            .replicated_axes = &.{ Axis.named_("link_x"), Axis.named_("link_y") },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>", w.written());
    }
    {
        defer w.clearRetainingCapacity();
        const sharding = try Sharding.init(std.testing.allocator, ctx, .{
            .mesh = "3d_mesh",
            .dimensions = &.{
                Dimension.closed(&.{Axis.named_("link_x")}),
                Dimension.closed(&.{Axis.named_("link_y")}),
                Dimension.closed(&.{Axis.named_("link_z")}),
            },
        });
        try sharding.asAttr().format(&w.writer);
        try std.testing.expectEqualSlices(u8, "#sdy.sharding<@\"3d_mesh\", [{\"link_x\"}, {\"link_y\"}, {\"link_z\"}]>", w.written());
    }
}
