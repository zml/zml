const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

/// `#fly.int`, the unified static/dynamic integer leaf.
pub const IntAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyInt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub fn getStatic(ctx: *mlir.Context, value: i32) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetStatic(ctx.ptr(), value).ptr orelse return error.InvalidMlir);
    }

    pub fn getDynamic(ctx: *mlir.Context, width: i32, divisibility: i32) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetDynamic(ctx.ptr(), width, divisibility).ptr orelse return error.InvalidMlir);
    }

    pub fn getNone(ctx: *mlir.Context) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetNone(ctx.ptr()).ptr orelse return error.InvalidMlir);
    }
};

/// `#fly.int_tuple`: a leaf, a basis element, or a nest of those.
pub const IntTupleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        value: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntTupleAttrGet(ctx.ptr(), args.value.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getStatic(ctx: *mlir.Context, value: i32) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getStatic(ctx, value)).attribute() });
    }

    pub fn getDynamic(ctx: *mlir.Context, width: i32, divisibility: i32) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getDynamic(ctx, width, divisibility)).attribute() });
    }

    pub fn getNone(ctx: *mlir.Context) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getNone(ctx)).attribute() });
    }

    pub fn getBasis(ctx: *mlir.Context, value: *const IntAttr, modes: []const i32) mlir.Error!*const Self {
        const result = c.mlirFlyIntTupleAttrGetBasis(ctx.ptr(), value.ptr(), @intCast(modes.len), modes.ptr);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getTuple(ctx: *mlir.Context, elements: []const *const Self) mlir.Error!*const Self {
        const result = c.mlirFlyIntTupleAttrGetTuple(ctx.ptr(), @intCast(elements.len), @ptrCast(elements.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `#fly.layout`, a `shape:stride` pair.
pub const LayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        shape: *const IntTupleAttr,
        stride: *const IntTupleAttr,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyLayoutAttrGet(ctx.ptr(), args.shape.ptr(), args.stride.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `#fly.tile`: a mode, or a `[a|b]` nest of modes.
pub const TileAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        value: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyTileAttrGet(ctx.ptr(), args.value.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getModes(ctx: *mlir.Context, modes: []const *const mlir.Attribute) mlir.Error!*const Self {
        const result = c.mlirFlyTileAttrGetModes(ctx.ptr(), @intCast(modes.len), @ptrCast(modes.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `S<mask,base,shift>`.
pub const SwizzleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlySwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        mask: i32,
        base: i32,
        shift: i32,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlySwizzleAttrGet(ctx.ptr(), args.mask, args.base, args.shift);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `align<n>`, in bytes.
pub const AlignAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyAlign;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        alignment: i32,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyAlignAttrGet(ctx.ptr(), args.alignment).ptr orelse return error.InvalidMlir);
    }
};

pub const AddressSpace = enum(i32) { generic = 0, global = 1, shared = 2, register = 3 };

/// `global`, `shared`, ... A pointer or memref may also carry a foreign
/// address-space attribute, such as `#fly_rocdl.buffer_desc`.
pub const AddressSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyAddressSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub const InitArgs = struct {
        addressSpace: AddressSpace,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyAddressSpaceAttrGet(ctx.ptr(), @intFromEnum(args.addressSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

pub const MmaOperand = enum(i32) { a = 0, b = 1, c = 2, d = 3 };

pub fn mmaOperandAttr(ctx: *mlir.Context, operand: MmaOperand) mlir.Error!*const mlir.Attribute {
    const result = c.mlirFlyMmaOperandAttrGet(ctx.ptr(), @intFromEnum(operand));
    return @ptrCast(result.ptr orelse return error.InvalidMlir);
}

pub const GemmTraversalOrder = enum(i32) {
    kmn = 0,
    knm = 1,
    mkn = 2,
    mnk = 3,
    nkm = 4,
    nmk = 5,
    kmn_serpentine = 6,
    knm_serpentine = 7,
    mkn_serpentine = 8,
    mnk_serpentine = 9,
    nkm_serpentine = 10,
    nmk_serpentine = 11,
};

pub fn gemmTraversalOrderAttr(ctx: *mlir.Context, order: GemmTraversalOrder) mlir.Error!*const mlir.Attribute {
    const result = c.mlirFlyGemmTraversalOrderAttrGet(ctx.ptr(), @intFromEnum(order));
    return @ptrCast(result.ptr orelse return error.InvalidMlir);
}
