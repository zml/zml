//! Bindings for OpenXLA shardy dialect
//!
//! - dialect overview: https://openxla.org/shardy/sdy_dialect
//! - C API: https://github.com/openxla/shardy/blob/main/shardy/integrations/c/attributes.h
const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("sdy");
}

/// `sdy.sharding`, the sharding of a function argument or result.
pub fn shardingNamedAttr(ctx: *mlir.Context, sharding: *const TensorShardingAttr) mlir.NamedAttribute {
    return .named(ctx, "sdy.sharding", sharding.attribute());
}

/// `sdy.mesh`. Not verified: it only verifies inside a module.
pub fn mesh(ctx: *mlir.Context, sym_name: []const u8, mesh_: *const MeshAttr, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "sdy.mesh", .{
        .attributes = &.{
            .named(ctx, "sym_name", .string(ctx, sym_name)),
            .named(ctx, "mesh", mesh_.attribute()),
        },
        .verify = false,
        .location = location,
    });
}

pub fn sharding_constraint(ctx: *mlir.Context, input: *const mlir.Value, sharding: *const TensorShardingAttr, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "sdy.sharding_constraint", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{input.type_()} },
        .attributes = &.{
            .named(ctx, "sharding", sharding.attribute()),
        },
        .location = location,
    });
}

pub fn manual_computation(
    ctx: *mlir.Context,
    tensors: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    body_block: *mlir.Block,
    in_shardings: *const TensorShardingPerValueAttr,
    out_shardings: *const TensorShardingPerValueAttr,
    manual_axes: *const ManualAxesAttr,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "sdy.manual_computation", .{
        .operands = .{ .flat = tensors },
        .results = .{ .flat = result_types },
        .blocks = &.{body_block},
        .attributes = &.{
            .named(ctx, "in_shardings", in_shardings.attribute()),
            .named(ctx, "out_shardings", out_shardings.attribute()),
            .named(ctx, "manual_axes", manual_axes.attribute()),
        },
        .location = location,
    });
}

/// `sdy.return`. Not verified: a terminator only verifies inside its parent.
pub fn return_(ctx: *mlir.Context, results: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "sdy.return", .{
        .operands = .{ .flat = results },
        .verify = false,
        .location = location,
    });
}

pub const MeshAxisAttr = opaque {
    const M = mlir.Methods(MeshAxisAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAMeshAxisAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        name: []const u8,
        size: i64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const MeshAxisAttr {
        return @ptrCast(c.sdyMeshAxisAttrGet(ctx.ptr(), mlir.stringRef(args.name), args.size).ptr);
    }

    pub fn attribute(self: *const MeshAxisAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn name(self: *const MeshAxisAttr) []const u8 {
        return mlir.string(c.sdyMeshAxisAttrGetName(self.ptr()));
    }

    pub fn size(self: *const MeshAxisAttr) i64 {
        return c.sdyMeshAxisAttrGetSize(self.ptr());
    }
};

pub const MeshAttr = opaque {
    const M = mlir.Methods(MeshAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAMeshAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        axes: []const *const MeshAxisAttr,
        device_ids: []const i64 = &.{},
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const MeshAttr {
        const axes: []const c.struct_MlirAttribute = @ptrCast(args.axes);
        return @ptrCast(c.sdyMeshAttrGet(
            ctx.ptr(),
            @intCast(axes.len),
            axes.ptr,
            @intCast(args.device_ids.len),
            args.device_ids.ptr,
        ).ptr);
    }

    pub fn attribute(self: *const MeshAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub const SubAxisInfoAttr = opaque {
    const M = mlir.Methods(SubAxisInfoAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsASubAxisInfoAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        pre_size: i64,
        size: i64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const SubAxisInfoAttr {
        return @ptrCast(c.sdySubAxisInfoAttrGet(ctx.ptr(), args.pre_size, args.size).ptr);
    }

    pub fn attribute(self: *const SubAxisInfoAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn preSize(self: *const SubAxisInfoAttr) i64 {
        return c.sdySubAxisInfoAttrGetPreSize(self.ptr());
    }

    pub fn size(self: *const SubAxisInfoAttr) i64 {
        return c.sdySubAxisInfoAttrGetSize(self.ptr());
    }
};

pub const AxisRefAttr = opaque {
    const M = mlir.Methods(AxisRefAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAnAxisRefAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn named(ctx: *mlir.Context, axis_name: []const u8) *const AxisRefAttr {
        return @ptrCast(c.sdyAxisRefAttrGet(ctx.ptr(), mlir.stringRef(axis_name), .{ .ptr = null }).ptr);
    }

    pub fn subAxis(ctx: *mlir.Context, axis_name: []const u8, pre_size: i64, size: i64) *const AxisRefAttr {
        const sub_axis: *const SubAxisInfoAttr = .get(ctx, .{ .pre_size = pre_size, .size = size });
        return @ptrCast(c.sdyAxisRefAttrGet(ctx.ptr(), mlir.stringRef(axis_name), sub_axis.ptr()).ptr);
    }

    pub fn attribute(self: *const AxisRefAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn name(self: *const AxisRefAttr) []const u8 {
        return mlir.string(c.sdyAxisRefAttrGetName(self.ptr()));
    }

    pub fn subAxisInfo(self: *const AxisRefAttr) ?*const SubAxisInfoAttr {
        const attr_ = c.sdyAxisRefAttrGetSubAxisInfo(self.ptr());
        return if (attr_.ptr == null) null else @ptrCast(attr_.ptr);
    }
};

pub const DimensionShardingAttr = opaque {
    const M = mlir.Methods(DimensionShardingAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsADimensionShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        axes: []const *const AxisRefAttr,
        is_closed: bool,
        priority: ?i64 = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const DimensionShardingAttr {
        const attrs: []const c.struct_MlirAttribute = @ptrCast(args.axes);
        return @ptrCast(c.sdyDimensionShardingAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr, args.is_closed, args.priority orelse -1).ptr);
    }

    pub fn closed(ctx: *mlir.Context, axes: []const *const AxisRefAttr) *const DimensionShardingAttr {
        return get(ctx, .{ .axes = axes, .is_closed = true });
    }

    pub fn open(ctx: *mlir.Context, axes: []const *const AxisRefAttr) *const DimensionShardingAttr {
        return get(ctx, .{ .axes = axes, .is_closed = false });
    }

    pub fn replicated(ctx: *mlir.Context) *const DimensionShardingAttr {
        return closed(ctx, &.{});
    }

    pub fn attribute(self: *const DimensionShardingAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn numAxes(self: *const DimensionShardingAttr) usize {
        return @intCast(c.sdyDimensionShardingAttrGetAxesSize(self.ptr()));
    }

    pub fn axis(self: *const DimensionShardingAttr, index: usize) *const AxisRefAttr {
        return @ptrCast(c.sdyDimensionShardingAttrGetAxesElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn isClosed(self: *const DimensionShardingAttr) bool {
        return c.sdyDimensionShardingAttrGetIsClosed(self.ptr());
    }

    pub fn priority(self: *const DimensionShardingAttr) ?i64 {
        const value = c.sdyDimensionShardingAttrGetPriority(self.ptr());
        return if (value == -1) null else value;
    }
};

pub const TensorShardingAttr = opaque {
    const M = mlir.Methods(TensorShardingAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsATensorShardingAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        mesh: []const u8,
        dimensions: []const *const DimensionShardingAttr,
        replicated_axes: []const *const AxisRefAttr = &.{},
        unreduced_axes: []const *const AxisRefAttr = &.{},
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const TensorShardingAttr {
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

    pub fn attribute(self: *const TensorShardingAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn meshOrRef(self: *const TensorShardingAttr) *const mlir.Attribute {
        return @ptrCast(c.sdyTensorShardingAttrGetMeshOrRef(self.ptr()).ptr);
    }

    pub fn numDimensions(self: *const TensorShardingAttr) usize {
        return @intCast(c.sdyTensorShardingAttrGetDimShardingsSize(self.ptr()));
    }

    pub fn dimension(self: *const TensorShardingAttr, index: usize) *const DimensionShardingAttr {
        return @ptrCast(c.sdyTensorShardingAttrGetDimShardingsElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numReplicatedAxes(self: *const TensorShardingAttr) usize {
        return @intCast(c.sdyTensorShardingAttrGetReplicatedAxesSize(self.ptr()));
    }

    pub fn replicatedAxis(self: *const TensorShardingAttr, index: usize) *const AxisRefAttr {
        return @ptrCast(c.sdyTensorShardingAttrGetReplicatedAxesElem(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numUnreducedAxes(self: *const TensorShardingAttr) usize {
        return @intCast(c.sdyTensorShardingAttrGetUnreducedAxesSize(self.ptr()));
    }

    pub fn unreducedAxis(self: *const TensorShardingAttr, index: usize) *const AxisRefAttr {
        return @ptrCast(c.sdyTensorShardingAttrGetUnreducedAxesElem(self.ptr(), @intCast(index)).ptr);
    }
};

pub const TensorShardingPerValueAttr = opaque {
    const M = mlir.Methods(TensorShardingPerValueAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsATensorShardingPerValueAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, shardings: []const *const TensorShardingAttr) *const TensorShardingPerValueAttr {
        const attrs: []const c.struct_MlirAttribute = @ptrCast(shardings);
        return @ptrCast(c.sdyTensorShardingPerValueAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr).ptr);
    }

    pub fn attribute(self: *const TensorShardingPerValueAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub const ManualAxesAttr = opaque {
    const M = mlir.Methods(ManualAxesAttr, c.MlirAttribute);

    pub const isAFn = c.sdyAttributeIsAManualAxesAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, axes: []const *const mlir.StringAttribute) *const ManualAxesAttr {
        comptime std.debug.assert(@sizeOf(c.struct_MlirAttribute) == @sizeOf(*const mlir.StringAttribute));
        const attrs: []const c.struct_MlirAttribute = @ptrCast(axes);
        return @ptrCast(c.sdyManualAxesAttrGet(ctx.ptr(), @intCast(attrs.len), attrs.ptr).ptr);
    }

    pub fn attribute(self: *const ManualAxesAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn numAxes(self: *const ManualAxesAttr) usize {
        return @intCast(c.sdyManualAxesAttrGetAxesSize(self.ptr()));
    }

    pub fn axis(self: *const ManualAxesAttr, index: usize) []const u8 {
        return mlir.string(c.sdyManualAxesAttrGetAxesElem(self.ptr(), @intCast(index)));
    }
};
