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

pub const attr = struct {
    pub fn mesh(ctx: *mlir.Context, mesh_: MeshAttribute.Mesh) *const mlir.Attribute {
        return MeshAttribute.init(ctx, mesh_).asAttr();
    }

    pub fn shardingsPerValue(ctx: *mlir.Context, shardings: []const *const TensorShardingAttribute) *const mlir.Attribute {
        return TensorShardingPerValueAttribute.init(ctx, shardings).asAttr();
    }
};

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
        const attr_ = c.sdyAxisRefAttrGetSubAxisInfo(self.ptr());
        return if (attr_.ptr == null) null else @ptrCast(attr_.ptr);
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
        opts: struct {
            priority_: ?i64 = null,
        },
    ) *const DimensionShardingAttribute {
        const attrs: []const c.struct_MlirAttribute = @ptrCast(axes);
        return @ptrCast(c.sdyDimensionShardingAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr, is_closed, opts.priority_ orelse -1).ptr);
    }

    pub fn closed(ctx: *mlir.Context, axes: []const *const AxisRefAttribute) *const DimensionShardingAttribute {
        return init(ctx, axes, true, .{});
    }

    pub fn open(ctx: *mlir.Context, axes: []const *const AxisRefAttribute) *const DimensionShardingAttribute {
        return init(ctx, axes, false, .{});
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
            0,
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
