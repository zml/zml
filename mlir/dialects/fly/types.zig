const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

const attributes = @import("attributes.zig");

fn optAttr(value: anytype) c.MlirAttribute {
    return if (value) |v| v.ptr() else .{ .ptr = null };
}

/// `!fly.int_tuple`.
pub const IntTupleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        attr: *const attributes.IntTupleAttr,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntTupleTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getAttr(self: *const Self) *const attributes.IntTupleAttr {
        return @ptrCast(c.mlirFlyIntTupleTypeGetAttr(self.ptr()).ptr.?);
    }

    /// 1 for a leaf.
    pub fn getNumElements(self: *const Self) usize {
        return @intCast(c.mlirFlyIntTupleTypeGetNumElements(self.ptr()));
    }

    pub fn getElement(self: *const Self, pos: usize) *const Self {
        return @ptrCast(c.mlirFlyIntTupleTypeGetElement(self.ptr(), @intCast(pos)).ptr.?);
    }

    pub fn isLeaf(self: *const Self) bool {
        return c.mlirFlyIntTupleTypeIsLeaf(self.ptr());
    }

    pub fn isStatic(self: *const Self) bool {
        return c.mlirFlyIntTupleTypeIsStatic(self.ptr());
    }

    pub const Leaf = union(enum) { static: i64, dynamic, none, basis };

    /// Asserts the tuple is a leaf.
    pub fn getLeaf(self: *const Self) Leaf {
        if (c.mlirFlyIntTupleTypeIsStaticLeaf(self.ptr())) {
            return .{ .static = c.mlirFlyIntTupleTypeGetStaticValue(self.ptr()) };
        }
        if (c.mlirFlyIntTupleTypeIsNoneLeaf(self.ptr())) return .none;
        if (c.mlirFlyIntTupleTypeIsBasisLeaf(self.ptr())) return .basis;
        return .dynamic;
    }
};

/// `!fly.layout`.
pub const LayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        attr: *const attributes.LayoutAttr,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyLayoutTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getAttr(self: *const Self) *const attributes.LayoutAttr {
        return @ptrCast(c.mlirFlyLayoutTypeGetAttr(self.ptr()).ptr.?);
    }

    pub fn getShape(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyLayoutTypeGetShape(self.ptr()).ptr.?);
    }

    pub fn getStride(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyLayoutTypeGetStride(self.ptr()).ptr.?);
    }
};

/// `!fly.composed_layout`.
pub const ComposedLayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyComposedLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyComposedLayoutTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!fly.tile`.
pub const TileType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        attr: *const attributes.TileAttr,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyTileTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getAttr(self: *const Self) *const attributes.TileAttr {
        return @ptrCast(c.mlirFlyTileTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!fly.swizzle`.
pub const SwizzleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlySwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        attr: *const attributes.SwizzleAttr,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlySwizzleTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }
};

/// `!fly.ptr`.
pub const PointerType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyPointer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        elemTy: *const mlir.Type,
        addressSpace: *const mlir.Attribute,
        alignment: ?*const attributes.AlignAttr = null,
        swizzle: ?*const attributes.SwizzleAttr = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyPointerTypeGet(ctx.ptr(), args.elemTy.ptr(), args.addressSpace.ptr(), optAttr(args.alignment), optAttr(args.swizzle));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getElemTy(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirFlyPointerTypeGetElemTy(self.ptr()).ptr.?);
    }

    pub fn getAddressSpace(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyPointerTypeGetAddressSpace(self.ptr()).ptr.?);
    }

    pub fn getAlignment(self: *const Self) *const attributes.AlignAttr {
        return @ptrCast(c.mlirFlyPointerTypeGetAlignment(self.ptr()).ptr.?);
    }

    pub fn getSwizzle(self: *const Self) *const attributes.SwizzleAttr {
        return @ptrCast(c.mlirFlyPointerTypeGetSwizzle(self.ptr()).ptr.?);
    }
};

/// `!fly.memref`.
pub const MemRefType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        elemTy: *const mlir.Type,
        addressSpace: *const mlir.Attribute,
        layout: *const mlir.Attribute,
        alignment: ?*const attributes.AlignAttr = null,
        swizzle: ?*const attributes.SwizzleAttr = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyMemRefTypeGet(ctx.ptr(), args.elemTy.ptr(), args.addressSpace.ptr(), args.layout.ptr(), optAttr(args.alignment), optAttr(args.swizzle));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getElemTy(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirFlyMemRefTypeGetElemTy(self.ptr()).ptr.?);
    }

    pub fn getAddressSpace(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyMemRefTypeGetAddressSpace(self.ptr()).ptr.?);
    }

    pub fn getLayout(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyMemRefTypeGetLayout(self.ptr()).ptr.?);
    }
};

/// `!fly.coord_tensor`.
pub const CoordTensorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCoordTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        base: *const attributes.IntTupleAttr,
        layout: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCoordTensorTypeGet(ctx.ptr(), args.base.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!fly.copy_atom`.
pub const CopyAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCopyAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        copyOp: *const mlir.Type,
        valBits: i32,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCopyAtomTypeGet(ctx.ptr(), args.copyOp.ptr(), args.valBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getThrLayout(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrLayout(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutSrc(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutSrc(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutDst(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutDst(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutRef(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutRef(self.ptr()).ptr.?);
    }
};

/// `!fly.mma_atom`.
pub const MmaAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        mmaOp: *const mlir.Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyMmaAtomTypeGet(ctx.ptr(), args.mmaOp.ptr()).ptr orelse return error.InvalidMlir);
    }

    pub fn getThrLayout(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrLayout(self.ptr()).ptr.?);
    }

    pub fn getShapeMNK(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetShapeMNK(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutA(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutA(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutB(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutB(self.ptr()).ptr.?);
    }

    pub fn getThrValLayoutC(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutC(self.ptr()).ptr.?);
    }
};

/// `!fly.tiled_copy`.
pub const TiledCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTiledCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub fn getTiledThrValLayoutSrc(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(self.ptr()).ptr.?);
    }

    pub fn getTiledThrValLayoutDst(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(self.ptr()).ptr.?);
    }
};

/// `!fly.tiled_mma`.
pub const TiledMmaType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTiledMma;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub fn getTileSizeMNK(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTileSizeMNK(self.ptr()).ptr.?);
    }

    pub fn getThrLayoutVMNK(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetThrLayoutVMNK(self.ptr()).ptr.?);
    }

    pub fn getTiledThrValLayoutA(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutA(self.ptr()).ptr.?);
    }

    pub fn getTiledThrValLayoutB(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutB(self.ptr()).ptr.?);
    }

    pub fn getTiledThrValLayoutC(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutC(self.ptr()).ptr.?);
    }
};

/// `!fly.universal_copy`.
pub const CopyOpUniversalCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCopyOpUniversalCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        bitSize: i32,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCopyOpUniversalCopyTypeGet(ctx.ptr(), args.bitSize);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};
