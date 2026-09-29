const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");
const cute = @This();

pub const dialect_namespace = "cute";
pub const nvgpu = @import("cute_nvgpu.zig");

pub const AttributeKind = enum { int_tuple, coord, shape, stride, layout, tile, composed_layout, swizzle };

pub const CuteType = opaque {
    const M = mlir.Methods(CuteType, c.MlirType);

    pub const isAFn = c.mlirTypeIsACuteType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;

    pub fn format(self: *const CuteType, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const t: *const mlir.Type = @ptrCast(self);
        return t.format(writer);
    }
};

/// Registers both dialects in the existing MLIR registry.
pub fn registerDialects(registry: *mlir.DialectRegistry) void {
    mlir.DialectHandle.fromString(dialect_namespace).insertDialect(registry);
    mlir.DialectHandle.fromString(nvgpu.dialect_namespace).insertDialect(registry);
}

pub const AlgebraKind = AttributeKind;

/// Constructs the algebra attribute and type through their C APIs.
/// Existing attributes can be passed directly to LayoutType.get, etc.
pub fn algebraType(ctx: *mlir.Context, kind: AlgebraKind, expression: []const u8) !*const mlir.Type {
    return switch (kind) {
        inline else => |k| blk: {
            const T = switch (k) {
                .int_tuple => IntTupleType,
                .coord => CoordType,
                .shape => ShapeType,
                .stride => StrideType,
                .layout => LayoutType,
                .tile => TileType,
                .composed_layout => ComposedLayoutType,
                .swizzle => SwizzleType,
            };
            const attr = try algebraAttribute(ctx, @field(AttributeKind, @tagName(k)), expression);
            break :blk (try T.get(ctx, .{ .attr = attr })).type_();
        },
    };
}

pub fn algebraAttribute(ctx: *mlir.Context, kind: AttributeKind, expression: []const u8) !*const mlir.Attribute {
    return switch (kind) {
        inline else => |k| blk: {
            const A = switch (k) {
                .int_tuple => IntTupleAttr,
                .coord => CoordAttr,
                .shape => ShapeAttr,
                .stride => StrideAttr,
                .layout => LayoutAttr,
                .tile => TileAttr,
                .composed_layout => ComposedLayoutAttr,
                .swizzle => SwizzleAttr,
            };
            break :blk (try A.get(ctx, expression)).attribute();
        },
    };
}

/// The address space of a pointer or memref.
pub const MemorySpace = AddressSpace;

pub fn pointerType(ctx: *mlir.Context, element_type: ?*const mlir.Type, space: MemorySpace, alignment: u64) !*const mlir.Type {
    return (try PtrType.get(ctx, .{
        .valueType = element_type,
        .addressSpace = space,
        .alignment = alignment,
    })).type_();
}

pub fn constrainedIntType(ctx: *mlir.Context, width: enum { i32, i64 }, divisible_by: u64) !*const mlir.Type {
    return (try ConstrainedIntType.get(ctx, .{
        .divisibility = std.math.cast(i64, divisible_by) orelse return error.InvalidMlir,
        .width = switch (width) {
            .i32 => 32,
            .i64 => 64,
        },
        .isPow2 = false,
    })).type_();
}

/// Constructors take raw CuTe algebra expressions. Getter strings are owned by
/// the MLIR context and remain valid until that context is destroyed.
pub const IntTupleAttr = opaque {
    const M = mlir.Methods(IntTupleAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const IntTupleAttr {
        const result = c.mlirCuteIntTupleAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const IntTupleAttr) []const u8 {
        return mlir.string(c.mlirCuteIntTupleAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const IntTupleAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const IntTupleAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const CoordAttr = opaque {
    const M = mlir.Methods(CoordAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteCoord;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const CoordAttr {
        const result = c.mlirCuteCoordAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const CoordAttr) []const u8 {
        return mlir.string(c.mlirCuteCoordAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const CoordAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const CoordAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const ShapeAttr = opaque {
    const M = mlir.Methods(ShapeAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const ShapeAttr {
        const result = c.mlirCuteShapeAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const ShapeAttr) []const u8 {
        return mlir.string(c.mlirCuteShapeAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const ShapeAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const ShapeAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const StrideAttr = opaque {
    const M = mlir.Methods(StrideAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteStride;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const StrideAttr {
        const result = c.mlirCuteStrideAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const StrideAttr) []const u8 {
        return mlir.string(c.mlirCuteStrideAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const StrideAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const StrideAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const LayoutAttr = opaque {
    const M = mlir.Methods(LayoutAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const LayoutAttr {
        const result = c.mlirCuteLayoutAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const LayoutAttr) []const u8 {
        return mlir.string(c.mlirCuteLayoutAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const LayoutAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const LayoutAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const TileAttr = opaque {
    const M = mlir.Methods(TileAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const TileAttr {
        const result = c.mlirCuteTileAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const TileAttr) []const u8 {
        return mlir.string(c.mlirCuteTileAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const TileAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const TileAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const ComposedLayoutAttr = opaque {
    const M = mlir.Methods(ComposedLayoutAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteComposedLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const ComposedLayoutAttr {
        const result = c.mlirCuteComposedLayoutAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const ComposedLayoutAttr) []const u8 {
        return mlir.string(c.mlirCuteComposedLayoutAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const ComposedLayoutAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const ComposedLayoutAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

pub const SwizzleAttr = opaque {
    const M = mlir.Methods(SwizzleAttr, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsACuteSwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, value: []const u8) mlir.Error!*const SwizzleAttr {
        const result = c.mlirCuteSwizzleAttrGet(ctx.ptr(), mlir.stringRef(value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    pub fn getValue(self: *const SwizzleAttr) []const u8 {
        return mlir.string(c.mlirCuteSwizzleAttrGetValue(self.ptr()));
    }

    pub fn attribute(self: *const SwizzleAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn format(self: *const SwizzleAttr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
};

// Enums, generated from the dialect's .td files; values are the compiler's.

/// Address spaces for CuTe memrefs and pointers.
pub const AddressSpace = enum(u32) {
    generic = 0,
    gmem = 1,
    smem = 3,
    rmem = 5,
    tmem = 6,
    dsmem = 7,
    cmem = 4,
};

/// Copy operand ID.
pub const CopyOperand = enum(u32) {
    S = 0,
    D = 1,
};

/// Major mode for MMA operations.
pub const MajorMode = enum(u32) {
    k = 0,
    mn = 1,
};

/// MMA operand ID.
pub const MmaOperand = enum(u32) {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    SFA = 4,
    SFB = 5,
    E = 6,
};

/// Op for cute reduce operations.
pub const ReductionOp = enum(u32) {
    ADD = 0,
    MUL = 1,
    MIN = 2,
    MAX = 3,
};

// Types and attributes, generated from the dialect's .td files.

/// `!cute.arith_tuple_iter`: Iterator over an arithmetic tuple.
pub const ArithTupleIteratorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteArithTupleIterator;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arith_tuple_iter";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        arithTuple: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteArithTupleIteratorTypeGet(ctx.ptr(), args.arithTuple.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getArithTuple(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteArithTupleIteratorTypeGetArithTuple(self.ptr()).ptr.?);
    }
};

/// `!cute.composed_layout`: CuTe composed layout: A ∘ offset ∘ B.
pub const ComposedLayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteComposedLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "composed_layout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteComposedLayoutTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteComposedLayoutTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.i<N>`: Integer of any width with known divisibility and power of two.
pub const ConstrainedIntType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteConstrainedInt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        divisibility: i64,
        width: c_uint,
        isPow2: bool,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteConstrainedIntTypeGet(ctx.ptr(), args.divisibility, args.width, args.isPow2);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getDivisibility(self: *const Self) i64 {
        return c.mlirCuteConstrainedIntTypeGetDivisibility(self.ptr());
    }
    pub fn getWidth(self: *const Self) c_uint {
        return c.mlirCuteConstrainedIntTypeGetWidth(self.ptr());
    }
    pub fn getIsPow2(self: *const Self) bool {
        return c.mlirCuteConstrainedIntTypeGetIsPow2(self.ptr());
    }
};

/// `!cute.coord_tensor`: Coordinate tensor: an arithmetic tuple with a layout.
pub const CoordTensorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteCoordTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "coord_tensor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        arithTuple: *const mlir.Type,
        layout: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteCoordTensorTypeGet(ctx.ptr(), args.arithTuple.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getArithTuple(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteCoordTensorTypeGetArithTuple(self.ptr()).ptr.?);
    }
    pub fn getLayout(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteCoordTensorTypeGetLayout(self.ptr()).ptr.?);
    }
};

/// `!cute.coord`: Scalar integer, underscore wildcard, or recursive tuple of coordinates.
pub const CoordType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteCoord;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "coord";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteCoordTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCoordTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.fast_divmod_divisor`: Precomputed divisor of the fast divmod operations.
pub const FastDivmodDivisorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteFastDivmodDivisor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "fast_divmod_divisor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        width: c_uint,
        isPow2: bool,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteFastDivmodDivisorTypeGet(ctx.ptr(), args.width, args.isPow2);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getWidth(self: *const Self) c_uint {
        return c.mlirCuteFastDivmodDivisorTypeGetWidth(self.ptr());
    }
    pub fn getIsPow2(self: *const Self) bool {
        return c.mlirCuteFastDivmodDivisorTypeGetIsPow2(self.ptr());
    }
};

/// `!cute.int_tuple`: Scalar integer or recursive tuple of integers.
pub const IntTupleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "int_tuple";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteIntTupleTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteIntTupleTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.layout`: CuTe layout: a shape/stride pair.
pub const LayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "layout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteLayoutTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteLayoutTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.memref`: Tensor view: a pointer with a layout.
pub const MemRefType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "memref";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        ptr: *const mlir.Type,
        layout: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteMemRefTypeGet(ctx.ptr(), args.ptr.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPtr(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteMemRefTypeGetPtr(self.ptr()).ptr.?);
    }
    pub fn getLayout(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteMemRefTypeGetLayout(self.ptr()).ptr.?);
    }
};

/// `!cute.ptr`: Pointer with address space, alignment, swizzle and bit layout.
pub const PtrType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACutePtr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ptr";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valueType: ?*const mlir.Type = null,
        addressSpace: AddressSpace,
        alignment: u64,
        swizzle: ?*const mlir.Attribute = null,
        bitlayout: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCutePtrTypeGet(ctx.ptr(), if (args.valueType) |v| v.ptr() else c.MlirType{ .ptr = null }, @intFromEnum(args.addressSpace), args.alignment, if (args.swizzle) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, if (args.bitlayout) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValueType(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCutePtrTypeGetValueType(self.ptr()).ptr);
    }
    pub fn getAddressSpace(self: *const Self) AddressSpace {
        return @enumFromInt(c.mlirCutePtrTypeGetAddressSpace(self.ptr()));
    }
    pub fn getAlignment(self: *const Self) u64 {
        return c.mlirCutePtrTypeGetAlignment(self.ptr());
    }
    pub fn getSwizzle(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCutePtrTypeGetSwizzle(self.ptr()).ptr);
    }
    pub fn getBitlayout(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCutePtrTypeGetBitlayout(self.ptr()).ptr);
    }
};

/// `!cute.shape`: Scalar integer or recursive tuple of shape extents.
pub const ShapeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteShapeTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteShapeTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.sparse_elem`: Logical sparse element stored in a physical type.
pub const SparseElemType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteSparseElem;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sparse_elem";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        numLogical: c_int,
        physicalType: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteSparseElemTypeGet(ctx.ptr(), args.numLogical, args.physicalType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getNumLogical(self: *const Self) c_int {
        return c.mlirCuteSparseElemTypeGetNumLogical(self.ptr());
    }
    pub fn getPhysicalType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteSparseElemTypeGetPhysicalType(self.ptr()).ptr.?);
    }
};

/// `!cute.stride`: Scalar or recursive tuple of stride elements.
pub const StrideType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteStride;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "stride";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteStrideTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteStrideTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.swizzle`: CuTe swizzle: S<num_bits, num_base, num_shift>.
pub const SwizzleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteSwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "swizzle";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteSwizzleTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSwizzleTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.tile`: CuTe tile: a recursive sequence of layouts and underscores.
pub const TileType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tile";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        attr: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteTileTypeGet(ctx.ptr(), args.attr.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTileTypeGetAttr(self.ptr()).ptr.?);
    }
};

/// `!cute.tiled_copy`: Copy atom tiled over threads and values.
pub const TiledCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTiledCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tiled_copy";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        copyAtom: *const mlir.Type,
        layoutCopyTv: *const mlir.Attribute,
        tilerMn: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteTiledCopyTypeGet(ctx.ptr(), args.copyAtom.ptr(), args.layoutCopyTv.ptr(), args.tilerMn.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getCopyAtom(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteTiledCopyTypeGetCopyAtom(self.ptr()).ptr.?);
    }
    pub fn getLayoutCopyTv(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledCopyTypeGetLayoutCopyTv(self.ptr()).ptr.?);
    }
    pub fn getTilerMn(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledCopyTypeGetTilerMn(self.ptr()).ptr.?);
    }
    pub const getAtom = getCopyAtom;
};

/// `!cute.tiled_copy_v2`: V2 copy atom tiled over threads and values.
pub const TiledCopyV2Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTiledCopyV2;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tiled_copy_v2";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        copyAtom: *const mlir.Type,
        layoutCopyTv: *const mlir.Attribute,
        tilerMn: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteTiledCopyV2TypeGet(ctx.ptr(), args.copyAtom.ptr(), args.layoutCopyTv.ptr(), args.tilerMn.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getCopyAtom(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteTiledCopyV2TypeGetCopyAtom(self.ptr()).ptr.?);
    }
    pub fn getLayoutCopyTv(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledCopyV2TypeGetLayoutCopyTv(self.ptr()).ptr.?);
    }
    pub fn getTilerMn(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledCopyV2TypeGetTilerMn(self.ptr()).ptr.?);
    }
};

/// `!cute.tiled_mma`: MMA atom tiled over MNK.
pub const TiledMmaType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTiledMma;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tiled_mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        mmaAtom: *const mlir.Type,
        atomLayoutMNK: *const mlir.Attribute,
        permutationMNK: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteTiledMmaTypeGet(ctx.ptr(), args.mmaAtom.ptr(), args.atomLayoutMNK.ptr(), if (args.permutationMNK) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMmaAtom(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteTiledMmaTypeGetMmaAtom(self.ptr()).ptr.?);
    }
    pub fn getAtomLayoutMNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledMmaTypeGetAtomLayoutMNK(self.ptr()).ptr.?);
    }
    pub fn getPermutationMNK(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteTiledMmaTypeGetPermutationMNK(self.ptr()).ptr);
    }
    pub const getAtom = getMmaAtom;
    pub const getAtomLayoutMnk = getAtomLayoutMNK;
    pub const getPermutationMnk = getPermutationMNK;
};

/// `!cute.tuple`: Non-empty tuple of types.
pub const TupleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tuple";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        types: []const *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteTupleTypeGet(ctx.ptr(), @intCast(args.types.len), @ptrCast(args.types.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getNumTypes(self: *const Self) usize {
        return @intCast(c.mlirCuteTupleTypeGetNumTypes(self.ptr()));
    }
    pub fn getType(self: *const Self, pos: usize) *const mlir.Type {
        return @ptrCast(c.mlirCuteTupleTypeGetType(self.ptr(), @intCast(pos)).ptr.?);
    }
};

/// `#cute.bitlayout`: Bit layout of a pointer's elements within a storage chunk.
pub const BitLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteBitLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "bitlayout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        layout: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteBitLayoutAttrGet(ctx.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getLayout(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteBitLayoutAttrGetLayout(self.ptr()).ptr.?);
    }
};

/// `#cute.copy_atom`: Thread and value layouts of a copy atom.
pub const CopyAtomAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteCopyAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "copy_atom";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        thrId: *const mlir.Attribute,
        layoutSrc: *const mlir.Attribute,
        layoutDst: *const mlir.Attribute,
        layoutRef: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteCopyAtomAttrGet(ctx.ptr(), args.thrId.ptr(), args.layoutSrc.ptr(), args.layoutDst.ptr(), args.layoutRef.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomAttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutSrc(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomAttrGetLayoutSrc(self.ptr()).ptr.?);
    }
    pub fn getLayoutDst(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomAttrGetLayoutDst(self.ptr()).ptr.?);
    }
    pub fn getLayoutRef(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomAttrGetLayoutRef(self.ptr()).ptr.?);
    }
};

/// `#cute.copy_atom_v2`: Thread and value layouts and fragments of a V2 copy atom.
pub const CopyAtomV2Attr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteCopyAtomV2;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "copy_atom_v2";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        thrId: *const mlir.Attribute,
        layoutSrcTV: *const mlir.Attribute,
        layoutDstTV: *const mlir.Attribute,
        frgSrc: *const mlir.Attribute,
        frgDst: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteCopyAtomV2AttrGet(ctx.ptr(), args.thrId.ptr(), args.layoutSrcTV.ptr(), args.layoutDstTV.ptr(), args.frgSrc.ptr(), args.frgDst.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomV2AttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutSrcTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomV2AttrGetLayoutSrcTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutDstTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomV2AttrGetLayoutDstTV(self.ptr()).ptr.?);
    }
    pub fn getFrgSrc(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomV2AttrGetFrgSrc(self.ptr()).ptr.?);
    }
    pub fn getFrgDst(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteCopyAtomV2AttrGetFrgDst(self.ptr()).ptr.?);
    }
};

/// `#cute.mma_atom`: Shapes, thread-value layouts and fragments of an MMA atom.
pub const MmaAtomAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_atom";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMNK: *const mlir.Attribute,
        thrId: *const mlir.Attribute,
        layoutATV: *const mlir.Attribute,
        layoutBTV: *const mlir.Attribute,
        layoutCTV: *const mlir.Attribute,
        shapeAMK: *const mlir.Attribute,
        shapeBNK: *const mlir.Attribute,
        shapeCMN: *const mlir.Attribute,
        frgA: *const mlir.Attribute,
        frgB: *const mlir.Attribute,
        frgC: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteMmaAtomAttrGet(ctx.ptr(), args.shapeMNK.ptr(), args.thrId.ptr(), args.layoutATV.ptr(), args.layoutBTV.ptr(), args.layoutCTV.ptr(), args.shapeAMK.ptr(), args.shapeBNK.ptr(), args.shapeCMN.ptr(), args.frgA.ptr(), args.frgB.ptr(), args.frgC.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetShapeMNK(self.ptr()).ptr.?);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetLayoutATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetLayoutBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutCTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetLayoutCTV(self.ptr()).ptr.?);
    }
    pub fn getShapeAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetShapeAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetShapeBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeCMN(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetShapeCMN(self.ptr()).ptr.?);
    }
    pub fn getFrgA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetFrgA(self.ptr()).ptr.?);
    }
    pub fn getFrgB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetFrgB(self.ptr()).ptr.?);
    }
    pub fn getFrgC(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMmaAtomAttrGetFrgC(self.ptr()).ptr.?);
    }
};

/// `#cute.mx_mma_atom`: MMA atom attribute with scale-factor operands.
pub const MxMmaAtomAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteMxMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mx_mma_atom";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMNK: *const mlir.Attribute,
        thrId: *const mlir.Attribute,
        layoutATV: *const mlir.Attribute,
        layoutSFATV: *const mlir.Attribute,
        layoutBTV: *const mlir.Attribute,
        layoutSFBTV: *const mlir.Attribute,
        layoutCTV: *const mlir.Attribute,
        shapeAMK: *const mlir.Attribute,
        shapeSFAMK: *const mlir.Attribute,
        shapeBNK: *const mlir.Attribute,
        shapeSFBNK: *const mlir.Attribute,
        shapeCMN: *const mlir.Attribute,
        frgA: *const mlir.Attribute,
        frgSFA: *const mlir.Attribute,
        frgB: *const mlir.Attribute,
        frgSFB: *const mlir.Attribute,
        frgC: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteMxMmaAtomAttrGet(ctx.ptr(), args.shapeMNK.ptr(), args.thrId.ptr(), args.layoutATV.ptr(), args.layoutSFATV.ptr(), args.layoutBTV.ptr(), args.layoutSFBTV.ptr(), args.layoutCTV.ptr(), args.shapeAMK.ptr(), args.shapeSFAMK.ptr(), args.shapeBNK.ptr(), args.shapeSFBNK.ptr(), args.shapeCMN.ptr(), args.frgA.ptr(), args.frgSFA.ptr(), args.frgB.ptr(), args.frgSFB.ptr(), args.frgC.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeMNK(self.ptr()).ptr.?);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetLayoutATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutSFATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetLayoutSFATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetLayoutBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutSFBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetLayoutSFBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutCTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetLayoutCTV(self.ptr()).ptr.?);
    }
    pub fn getShapeAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeSFAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeSFAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeSFBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeSFBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeCMN(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetShapeCMN(self.ptr()).ptr.?);
    }
    pub fn getFrgA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetFrgA(self.ptr()).ptr.?);
    }
    pub fn getFrgSFA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetFrgSFA(self.ptr()).ptr.?);
    }
    pub fn getFrgB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetFrgB(self.ptr()).ptr.?);
    }
    pub fn getFrgSFB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetFrgSFB(self.ptr()).ptr.?);
    }
    pub fn getFrgC(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteMxMmaAtomAttrGetFrgC(self.ptr()).ptr.?);
    }
};

/// `#cute.reduction_op`: Op for cute reduce operations.
pub const ReductionOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteReductionOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "reduction_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteReductionOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionOp {
        return @enumFromInt(c.mlirCuteReductionOpAttrGetValue(self.ptr()));
    }
};

/// `#cute.sparse_mma_atom`: MMA atom attribute with a sparsity-metadata operand.
pub const SparseMmaAtomAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteSparseMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sparse_mma_atom";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMNK: *const mlir.Attribute,
        thrId: *const mlir.Attribute,
        layoutATV: *const mlir.Attribute,
        layoutBTV: *const mlir.Attribute,
        layoutCTV: *const mlir.Attribute,
        layoutETV: *const mlir.Attribute,
        shapeAMK: *const mlir.Attribute,
        shapeBNK: *const mlir.Attribute,
        shapeCMN: *const mlir.Attribute,
        shapeEMK: *const mlir.Attribute,
        frgA: *const mlir.Attribute,
        frgB: *const mlir.Attribute,
        frgC: *const mlir.Attribute,
        frgE: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteSparseMmaAtomAttrGet(ctx.ptr(), args.shapeMNK.ptr(), args.thrId.ptr(), args.layoutATV.ptr(), args.layoutBTV.ptr(), args.layoutCTV.ptr(), args.layoutETV.ptr(), args.shapeAMK.ptr(), args.shapeBNK.ptr(), args.shapeCMN.ptr(), args.shapeEMK.ptr(), args.frgA.ptr(), args.frgB.ptr(), args.frgC.ptr(), args.frgE.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetShapeMNK(self.ptr()).ptr.?);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetLayoutATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetLayoutBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutCTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetLayoutCTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutETV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetLayoutETV(self.ptr()).ptr.?);
    }
    pub fn getShapeAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetShapeAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetShapeBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeCMN(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetShapeCMN(self.ptr()).ptr.?);
    }
    pub fn getShapeEMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetShapeEMK(self.ptr()).ptr.?);
    }
    pub fn getFrgA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetFrgA(self.ptr()).ptr.?);
    }
    pub fn getFrgB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetFrgB(self.ptr()).ptr.?);
    }
    pub fn getFrgC(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetFrgC(self.ptr()).ptr.?);
    }
    pub fn getFrgE(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMmaAtomAttrGetFrgE(self.ptr()).ptr.?);
    }
};

/// `#cute.sparse_mx_mma_atom`: MMA atom attribute with scale-factor and metadata operands.
pub const SparseMxMmaAtomAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteSparseMxMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sparse_mx_mma_atom";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMNK: *const mlir.Attribute,
        thrId: *const mlir.Attribute,
        layoutATV: *const mlir.Attribute,
        layoutSFATV: *const mlir.Attribute,
        layoutBTV: *const mlir.Attribute,
        layoutSFBTV: *const mlir.Attribute,
        layoutCTV: *const mlir.Attribute,
        layoutETV: *const mlir.Attribute,
        shapeAMK: *const mlir.Attribute,
        shapeSFAMK: *const mlir.Attribute,
        shapeBNK: *const mlir.Attribute,
        shapeSFBNK: *const mlir.Attribute,
        shapeCMN: *const mlir.Attribute,
        shapeEMK: *const mlir.Attribute,
        frgA: *const mlir.Attribute,
        frgSFA: *const mlir.Attribute,
        frgB: *const mlir.Attribute,
        frgSFB: *const mlir.Attribute,
        frgC: *const mlir.Attribute,
        frgE: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteSparseMxMmaAtomAttrGet(ctx.ptr(), args.shapeMNK.ptr(), args.thrId.ptr(), args.layoutATV.ptr(), args.layoutSFATV.ptr(), args.layoutBTV.ptr(), args.layoutSFBTV.ptr(), args.layoutCTV.ptr(), args.layoutETV.ptr(), args.shapeAMK.ptr(), args.shapeSFAMK.ptr(), args.shapeBNK.ptr(), args.shapeSFBNK.ptr(), args.shapeCMN.ptr(), args.shapeEMK.ptr(), args.frgA.ptr(), args.frgSFA.ptr(), args.frgB.ptr(), args.frgSFB.ptr(), args.frgC.ptr(), args.frgE.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeMNK(self.ptr()).ptr.?);
    }
    pub fn getThrId(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetThrId(self.ptr()).ptr.?);
    }
    pub fn getLayoutATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutSFATV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutSFATV(self.ptr()).ptr.?);
    }
    pub fn getLayoutBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutSFBTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutSFBTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutCTV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutCTV(self.ptr()).ptr.?);
    }
    pub fn getLayoutETV(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetLayoutETV(self.ptr()).ptr.?);
    }
    pub fn getShapeAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeSFAMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeSFAMK(self.ptr()).ptr.?);
    }
    pub fn getShapeBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeSFBNK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeSFBNK(self.ptr()).ptr.?);
    }
    pub fn getShapeCMN(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeCMN(self.ptr()).ptr.?);
    }
    pub fn getShapeEMK(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetShapeEMK(self.ptr()).ptr.?);
    }
    pub fn getFrgA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgA(self.ptr()).ptr.?);
    }
    pub fn getFrgSFA(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgSFA(self.ptr()).ptr.?);
    }
    pub fn getFrgB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgB(self.ptr()).ptr.?);
    }
    pub fn getFrgSFB(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgSFB(self.ptr()).ptr.?);
    }
    pub fn getFrgC(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgC(self.ptr()).ptr.?);
    }
    pub fn getFrgE(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteSparseMxMmaAtomAttrGetFrgE(self.ptr()).ptr.?);
    }
};

// Operation builders, generated from the operations' .td.
/// `cute.add_offset`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_offset(ctx: *mlir.Context, src: *const mlir.Value, offset: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.add_offset", .{
        .operands = .{ .flat = &.{ src, offset } },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.append_to_rank`. Result types are explicit; attributes use mlir.Attribute.
pub fn append_to_rank(ctx: *mlir.Context, input: *const mlir.Value, element: *const mlir.Value, result_type: *const mlir.Type, rank: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.append_to_rank", .{
        .operands = .{ .flat = &.{ input, element } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "rank", rank),
        },
        .location = location,
    });
}

/// `cute.apply_swizzle`. Result types are explicit; attributes use mlir.Attribute.
pub fn apply_swizzle(ctx: *mlir.Context, ptr: *const mlir.Value, swizzled_ptr_type: *const mlir.Type, swizzle: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (swizzle) |value| attributes.appendAssumeCapacity(.named(ctx, "swizzle", value));
    return mlir.Operation.make(ctx, "cute.apply_swizzle", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{swizzled_ptr_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.assume`. Result types are explicit; attributes use mlir.Attribute.
pub fn assume(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.assume", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.blocked_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn blocked_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.blocked_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.ceil_div`. Result types are explicit; attributes use mlir.Attribute.
pub fn ceil_div(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.ceil_div", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.coalesce`. Result types are explicit; attributes use mlir.Attribute.
pub fn coalesce(ctx: *mlir.Context, input: *const mlir.Value, target_profile: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.coalesce", .{
        .operands = .{ .variadic = &.{
            &.{input},
            if (target_profile) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.complement`. Result types are explicit; attributes use mlir.Attribute.
pub fn complement(ctx: *mlir.Context, input: *const mlir.Value, cotarget: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.complement", .{
        .operands = .{ .flat = &.{ input, cotarget } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `cute.composed_get_inner`. Result types are explicit; attributes use mlir.Attribute.
pub fn composed_get_inner(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.composed_get_inner", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.composed_get_offset`. Result types are explicit; attributes use mlir.Attribute.
pub fn composed_get_offset(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.composed_get_offset", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.composed_get_outer`. Result types are explicit; attributes use mlir.Attribute.
pub fn composed_get_outer(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.composed_get_outer", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.composition`. Result types are explicit; attributes use mlir.Attribute.
pub fn composition(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.composition", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `cute.copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy(ctx: *mlir.Context, copy_atom: *const mlir.Value, src: []const *const mlir.Value, dst: []const *const mlir.Value, pred: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.copy", .{
        .operands = .{ .variadic = &.{
            &.{copy_atom},
            src,
            dst,
            if (pred) |value| &.{value} else &.{},
        } },
        .location = location,
    });
}

/// `cute.copy.make_fragment`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy_make_fragment(ctx: *mlir.Context, atom: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.copy.make_fragment", .{
        .operands = .{ .flat = &.{ atom, input } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .location = location,
    });
}

/// `cute.copy_atom_call`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy_atom_call(ctx: *mlir.Context, atom: *const mlir.Value, src: []const *const mlir.Value, dst: []const *const mlir.Value, pred: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.copy_atom_call", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            src,
            dst,
            if (pred) |value| &.{value} else &.{},
        } },
        .location = location,
    });
}

/// `cute.cosize`. Result types are explicit; attributes use mlir.Attribute.
pub fn cosize(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "cute.cosize", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.crd2idx`. Result types are explicit; attributes use mlir.Attribute.
pub fn crd2idx(ctx: *mlir.Context, coord: *const mlir.Value, layout: *const mlir.Value, index_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.crd2idx", .{
        .operands = .{ .flat = &.{ coord, layout } },
        .results = .{ .flat = &.{index_type} },
        .location = location,
    });
}

/// `cute.deref_arith_tuple_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn deref_arith_tuple_iter(ctx: *mlir.Context, iter: *const mlir.Value, arith_tuple_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.deref_arith_tuple_iter", .{
        .operands = .{ .flat = &.{iter} },
        .results = .{ .flat = &.{arith_tuple_type} },
        .location = location,
    });
}

/// `cute.deref_desc_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn deref_desc_iter(ctx: *mlir.Context, iter: *const mlir.Value, value__type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.deref_desc_iter", .{
        .operands = .{ .flat = &.{iter} },
        .results = .{ .flat = &.{value__type} },
        .location = location,
    });
}

/// `cute.derefine`. Result types are explicit; attributes use mlir.Attribute.
pub fn derefine(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.derefine", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.dice`. Result types are explicit; attributes use mlir.Attribute.
pub fn dice(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, coord: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.dice", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "coord", coord),
        },
        .location = location,
    });
}

/// `cute.elem_less`. Result types are explicit; attributes use mlir.Attribute.
pub fn elem_less(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.elem_less", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.equal`. Result types are explicit; attributes use mlir.Attribute.
pub fn equal(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.equal", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.fast_divmod.compute`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_compute(ctx: *mlir.Context, dividend: *const mlir.Value, divisor: *const mlir.Value, quotient_type: *const mlir.Type, remainder_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.compute", .{
        .operands = .{ .flat = &.{ dividend, divisor } },
        .results = .{ .flat = &.{ quotient_type, remainder_type } },
        .location = location,
    });
}

/// `cute.fast_divmod.create_divisor`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_create_divisor(ctx: *mlir.Context, divisor: *const mlir.Value, fast_divmod_divisor_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.create_divisor", .{
        .operands = .{ .flat = &.{divisor} },
        .results = .{ .flat = &.{fast_divmod_divisor_type} },
        .location = location,
    });
}

/// `cute.fast_divmod.divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_divide(ctx: *mlir.Context, dividend: *const mlir.Value, divisor: *const mlir.Value, quotient_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.divide", .{
        .operands = .{ .flat = &.{ dividend, divisor } },
        .results = .{ .flat = &.{quotient_type} },
        .location = location,
    });
}

/// `cute.fast_divmod.get_aux`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_get_aux(ctx: *mlir.Context, divisor: *const mlir.Value, multiplier_type: *const mlir.Type, shift1_type: *const mlir.Type, shift2_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.get_aux", .{
        .operands = .{ .flat = &.{divisor} },
        .results = .{ .flat = &.{ multiplier_type, shift1_type, shift2_type } },
        .location = location,
    });
}

/// `cute.fast_divmod.get_divisor`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_get_divisor(ctx: *mlir.Context, fast_divmod_divisor: *const mlir.Value, divisor_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.get_divisor", .{
        .operands = .{ .flat = &.{fast_divmod_divisor} },
        .results = .{ .flat = &.{divisor_type} },
        .location = location,
    });
}

/// `cute.fast_divmod.make_divisor`. Result types are explicit; attributes use mlir.Attribute.
pub fn fast_divmod_make_divisor(ctx: *mlir.Context, divisor: *const mlir.Value, multiplier: *const mlir.Value, sh1: *const mlir.Value, sh2: *const mlir.Value, fast_divmod_divisor_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.fast_divmod.make_divisor", .{
        .operands = .{ .flat = &.{ divisor, multiplier, sh1, sh2 } },
        .results = .{ .flat = &.{fast_divmod_divisor_type} },
        .location = location,
    });
}

/// `cute.filter`. Result types are explicit; attributes use mlir.Attribute.
pub fn filter(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.filter", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.filter_zeros`. Result types are explicit; attributes use mlir.Attribute.
pub fn filter_zeros(ctx: *mlir.Context, input: *const mlir.Value, target_profile: ?*const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.filter_zeros", .{
        .operands = .{ .variadic = &.{
            &.{input},
            if (target_profile) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `cute.flat_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn flat_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.flat_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.flat_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn flat_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.flat_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.gemm`. Result types are explicit; attributes use mlir.Attribute.
pub fn gemm(ctx: *mlir.Context, mma_atom: *const mlir.Value, d: *const mlir.Value, a: []const *const mlir.Value, b: []const *const mlir.Value, c_: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.gemm", .{
        .operands = .{ .variadic = &.{
            &.{mma_atom},
            &.{d},
            a,
            b,
            &.{c_},
        } },
        .location = location,
    });
}

/// `cute.get`. Result types are explicit; attributes use mlir.Attribute.
pub fn get(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "cute.get", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.get_flat_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_flat_coord(ctx: *mlir.Context, index: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_flat_coord", .{
        .operands = .{ .flat = &.{ index, input } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.get_hier_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_hier_coord(ctx: *mlir.Context, index: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_hier_coord", .{
        .operands = .{ .flat = &.{ index, input } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.get_integral_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_integral_coord(ctx: *mlir.Context, index: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_integral_coord", .{
        .operands = .{ .flat = &.{ index, input } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.get_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_iter(ctx: *mlir.Context, source: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_iter", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute.get_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_layout(ctx: *mlir.Context, input: *const mlir.Value, layout_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_layout", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{layout_type} },
        .location = location,
    });
}

/// `cute.get_layouts_from_tile`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_layouts_from_tile(ctx: *mlir.Context, tile: *const mlir.Value, layouts_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_layouts_from_tile", .{
        .operands = .{ .flat = &.{tile} },
        .results = .{ .variadic = &.{
            layouts_types,
        } },
        .location = location,
    });
}

/// `cute.get_leaves`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_leaves(ctx: *mlir.Context, input: *const mlir.Value, results__types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_leaves", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .variadic = &.{
            results__types,
        } },
        .location = location,
    });
}

/// `cute.get_scalars`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_scalars(ctx: *mlir.Context, cute_value: *const mlir.Value, scalars_types: []const *const mlir.Type, only_dynamic: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (only_dynamic) |value| attributes.appendAssumeCapacity(.named(ctx, "only_dynamic", value));
    return mlir.Operation.make(ctx, "cute.get_scalars", .{
        .operands = .{ .flat = &.{cute_value} },
        .results = .{ .variadic = &.{
            scalars_types,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.get_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_shape(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_shape", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.get_stride`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_stride(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_stride", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.group_modes`. Result types are explicit; attributes use mlir.Attribute.
pub fn group_modes(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, begin: *const mlir.Attribute, end: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.group_modes", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "begin", begin),
            .named(ctx, "end", end),
        },
        .location = location,
    });
}

/// `cute.idx2crd`. Result types are explicit; attributes use mlir.Attribute.
pub fn idx2crd(ctx: *mlir.Context, index: *const mlir.Value, shape: *const mlir.Value, coord_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.idx2crd", .{
        .operands = .{ .flat = &.{ index, shape } },
        .results = .{ .flat = &.{coord_type} },
        .location = location,
    });
}

/// `cute.increment_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn increment_coord(ctx: *mlir.Context, coord: *const mlir.Value, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.increment_coord", .{
        .operands = .{ .flat = &.{ coord, shape } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.inttoptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn inttoptr(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.inttoptr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.kernel_smem_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn kernel_smem_size(ctx: *mlir.Context, size_type: *const mlir.Type, kernel_name: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.kernel_smem_size", .{
        .results = .{ .flat = &.{size_type} },
        .attributes = &.{
            .named(ctx, "kernel_name", kernel_name),
        },
        .location = location,
    });
}

/// `cute.left_inverse`. Result types are explicit; attributes use mlir.Attribute.
pub fn left_inverse(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.left_inverse", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.load_scaled_index`. Result types are explicit; attributes use mlir.Attribute.
pub fn load_scaled_index(ctx: *mlir.Context, ptr_a: *const mlir.Value, ptr_b: *const mlir.Value, index: *const mlir.Value, stride: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.load_scaled_index", .{
        .operands = .{ .flat = &.{ ptr_a, ptr_b, index, stride } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.local_partition`. Result types are explicit; attributes use mlir.Attribute.
pub fn local_partition(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, index: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.local_partition", .{
        .operands = .{ .flat = &.{ input, tiler, index } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.local_tile`. Result types are explicit; attributes use mlir.Attribute.
pub fn local_tile(ctx: *mlir.Context, input: *const mlir.Value, tile: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, proj: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (proj) |value| attributes.appendAssumeCapacity(.named(ctx, "proj", value));
    return mlir.Operation.make(ctx, "cute.local_tile", .{
        .operands = .{ .flat = &.{ input, tile, coord } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.logical_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn logical_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.logical_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.logical_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn logical_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.logical_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_arith_tuple_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_arith_tuple_iter(ctx: *mlir.Context, value_: ?*const mlir.Value, iter_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_arith_tuple_iter", .{
        .operands = .{ .variadic = &.{
            if (value_) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{iter_type} },
        .location = location,
    });
}

/// `cute.make_atom`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_atom(ctx: *mlir.Context, values: []const *const mlir.Value, atom_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_atom", .{
        .operands = .{ .variadic = &.{
            values,
        } },
        .results = .{ .flat = &.{atom_type} },
        .location = location,
    });
}

/// `cute.make_composed_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_composed_layout(ctx: *mlir.Context, inner: *const mlir.Value, offset: *const mlir.Value, outer: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_composed_layout", .{
        .operands = .{ .flat = &.{ inner, offset, outer } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_coord(ctx: *mlir.Context, dynamicElements: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_coord", .{
        .operands = .{ .variadic = &.{
            dynamicElements,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_desc_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_desc_iter(ctx: *mlir.Context, value_: *const mlir.Value, iter_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_desc_iter", .{
        .operands = .{ .flat = &.{value_} },
        .results = .{ .flat = &.{iter_type} },
        .location = location,
    });
}

/// `cute.make_fragment_like`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_fragment_like(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_fragment_like", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.make_identity_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_identity_layout(ctx: *mlir.Context, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_identity_layout", .{
        .operands = .{ .flat = &.{shape} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_identity_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_identity_tensor(ctx: *mlir.Context, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_identity_tensor", .{
        .operands = .{ .flat = &.{shape} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_int_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_int_tuple(ctx: *mlir.Context, dynamicElements: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_int_tuple", .{
        .operands = .{ .variadic = &.{
            dynamicElements,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_layout(ctx: *mlir.Context, shape: ?*const mlir.Value, stride: ?*const mlir.Value, layout_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_layout", .{
        .operands = .{ .variadic = &.{
            if (shape) |value| &.{value} else &.{},
            if (stride) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{layout_type} },
        .location = location,
    });
}

/// `cute.make_layout_like`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_layout_like(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_layout_like", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.make_ordered_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_ordered_layout(ctx: *mlir.Context, shape: *const mlir.Value, order: *const mlir.Value, layout_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_ordered_layout", .{
        .operands = .{ .flat = &.{ shape, order } },
        .results = .{ .flat = &.{layout_type} },
        .location = location,
    });
}

/// `cute.make_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_shape(ctx: *mlir.Context, dynamicElements: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_shape", .{
        .operands = .{ .variadic = &.{
            dynamicElements,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_sparse_elem`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_sparse_elem(ctx: *mlir.Context, physical_storage: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_sparse_elem", .{
        .operands = .{ .flat = &.{physical_storage} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_stride`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_stride(ctx: *mlir.Context, dynamicElements: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_stride", .{
        .operands = .{ .variadic = &.{
            dynamicElements,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tile`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tile(ctx: *mlir.Context, dynamicElements: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tile", .{
        .operands = .{ .variadic = &.{
            dynamicElements,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tiled_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tiled_copy(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tiled_copy", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tiled_copy_v2`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tiled_copy_v2(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tiled_copy_v2", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tiled_mma`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tiled_mma(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tiled_mma", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tuple(ctx: *mlir.Context, values: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tuple", .{
        .operands = .{ .variadic = &.{
            values,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_view(ctx: *mlir.Context, iter: *const mlir.Value, layout: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_view", .{
        .operands = .{ .variadic = &.{
            &.{iter},
            if (layout) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.memref.alloc_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_alloc_smem(ctx: *mlir.Context, memref_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.alloc_smem", .{
        .results = .{ .flat = &.{memref_type} },
        .location = location,
    });
}

/// `cute.memref.alloca`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_alloca(ctx: *mlir.Context, layout: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.alloca", .{
        .operands = .{ .variadic = &.{
            if (layout) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.memref.load`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_load(ctx: *mlir.Context, src: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.load", .{
        .operands = .{ .flat = &.{ src, coord } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.memref.load_vec`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_load_vec(ctx: *mlir.Context, src: *const mlir.Value, mask: ?*const mlir.Value, pass_thru: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.load_vec", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (mask) |value| &.{value} else &.{},
            if (pass_thru) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.memref.store`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_store(ctx: *mlir.Context, dst: *const mlir.Value, coord: *const mlir.Value, value_: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.store", .{
        .operands = .{ .flat = &.{ dst, coord, value_ } },
        .location = location,
    });
}

/// `cute.memref.store_vec`. Result types are explicit; attributes use mlir.Attribute.
pub fn memref_store_vec(ctx: *mlir.Context, value_: *const mlir.Value, dst: *const mlir.Value, mask: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.memref.store_vec", .{
        .operands = .{ .variadic = &.{
            &.{value_},
            &.{dst},
            if (mask) |value| &.{value} else &.{},
        } },
        .location = location,
    });
}

/// `cute.mma.make_fragment`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_make_fragment(ctx: *mlir.Context, atom: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.mma.make_fragment", .{
        .operands = .{ .flat = &.{ atom, input } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .location = location,
    });
}

/// `cute.mma_atom_call`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_atom_call(ctx: *mlir.Context, atom: *const mlir.Value, d: *const mlir.Value, a: []const *const mlir.Value, b: []const *const mlir.Value, c_: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.mma_atom_call", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            &.{d},
            a,
            b,
            &.{c_},
        } },
        .location = location,
    });
}

/// `cute.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch(ctx: *mlir.Context, prefetch_atom: *const mlir.Value, src: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.prefetch", .{
        .operands = .{ .variadic = &.{
            &.{prefetch_atom},
            src,
        } },
        .location = location,
    });
}

/// `cute.prefetch_atom_call`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch_atom_call(ctx: *mlir.Context, atom: *const mlir.Value, src: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.prefetch_atom_call", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            src,
        } },
        .location = location,
    });
}

/// `cute.prepend_to_rank`. Result types are explicit; attributes use mlir.Attribute.
pub fn prepend_to_rank(ctx: *mlir.Context, input: *const mlir.Value, element: *const mlir.Value, result_type: *const mlir.Type, rank: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.prepend_to_rank", .{
        .operands = .{ .flat = &.{ input, element } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "rank", rank),
        },
        .location = location,
    });
}

/// `cute.print`. Result types are explicit; attributes use mlir.Attribute.
pub fn print(ctx: *mlir.Context, values: []const *const mlir.Value, fmt: ?*const mlir.Attribute, operand_signed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (fmt) |value| attributes.appendAssumeCapacity(.named(ctx, "fmt", value));
    if (operand_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "operand_signed", value));
    return mlir.Operation.make(ctx, "cute.print", .{
        .operands = .{ .variadic = &.{
            values,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.print_tma_desc_im2col`. Result types are explicit; attributes use mlir.Attribute.
pub fn print_tma_desc_im2col(ctx: *mlir.Context, input: *const mlir.Value, fd: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.print_tma_desc_im2col", .{
        .operands = .{ .flat = &.{ input, fd } },
        .location = location,
    });
}

/// `cute.print_tma_desc_tiled`. Result types are explicit; attributes use mlir.Attribute.
pub fn print_tma_desc_tiled(ctx: *mlir.Context, input: *const mlir.Value, fd: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.print_tma_desc_tiled", .{
        .operands = .{ .flat = &.{ input, fd } },
        .location = location,
    });
}

/// `cute.print_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn print_view(ctx: *mlir.Context, src: *const mlir.Value, coord: ?*const mlir.Value, verbose: ?*const mlir.Attribute, is_signed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (verbose) |value| attributes.appendAssumeCapacity(.named(ctx, "verbose", value));
    if (is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    return mlir.Operation.make(ctx, "cute.print_view", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (coord) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.ptr.load`. Result types are explicit; attributes use mlir.Attribute.
pub fn ptr_load(ctx: *mlir.Context, ptr: *const mlir.Value, value__type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.ptr.load", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{value__type} },
        .location = location,
    });
}

/// `cute.ptr.store`. Result types are explicit; attributes use mlir.Attribute.
pub fn ptr_store(ctx: *mlir.Context, value_: *const mlir.Value, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.ptr.store", .{
        .operands = .{ .flat = &.{ value_, ptr } },
        .location = location,
    });
}

/// `cute.ptrtoint`. Result types are explicit; attributes use mlir.Attribute.
pub fn ptrtoint(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.ptrtoint", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.raked_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn raked_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.raked_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.random_delay`. Result types are explicit; attributes use mlir.Attribute.
pub fn random_delay(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.random_delay", .{
        .location = location,
    });
}

/// `cute.recast_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn recast_iter(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.recast_iter", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.recast_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn recast_layout(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, new_type_bits: *const mlir.Attribute, old_type_bits: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.recast_layout", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = &.{
            .named(ctx, "new_type_bits", new_type_bits),
            .named(ctx, "old_type_bits", old_type_bits),
        },
        .location = location,
    });
}

/// `cute.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn reduce(ctx: *mlir.Context, input: *const mlir.Value, shape: *const mlir.Value, init_val: *const mlir.Value, predication: ?*const mlir.Value, output_type: *const mlir.Type, reduced_shape_type: ?*const mlir.Type, reduction_op: *const mlir.Attribute, reduction_profile: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "reduction_op", reduction_op));
    if (reduction_profile) |value| attributes.appendAssumeCapacity(.named(ctx, "reduction_profile", value));
    return mlir.Operation.make(ctx, "cute.reduce", .{
        .operands = .{ .variadic = &.{
            &.{input},
            &.{shape},
            &.{init_val},
            if (predication) |value| &.{value} else &.{},
        } },
        .results = .{ .variadic = &.{
            &.{output_type},
            if (reduced_shape_type) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.right_inverse`. Result types are explicit; attributes use mlir.Attribute.
pub fn right_inverse(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.right_inverse", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.select`. Result types are explicit; attributes use mlir.Attribute.
pub fn select(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.select", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `cute.shape_div`. Result types are explicit; attributes use mlir.Attribute.
pub fn shape_div(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.shape_div", .{
        .operands = .{ .flat = &.{ a, b } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.size`. Result types are explicit; attributes use mlir.Attribute.
pub fn size(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "cute.size", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.slice`. Result types are explicit; attributes use mlir.Attribute.
pub fn slice(ctx: *mlir.Context, input: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.slice", .{
        .operands = .{ .flat = &.{ input, coord } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.smem_partition_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn smem_partition_size(ctx: *mlir.Context, size_type: *const mlir.Type, partition_id: *const mlir.Attribute, cumulative: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "partition_id", partition_id));
    if (cumulative) |value| attributes.appendAssumeCapacity(.named(ctx, "cumulative", value));
    return mlir.Operation.make(ctx, "cute.smem_partition_size", .{
        .results = .{ .flat = &.{size_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.static`. Result types are explicit; attributes use mlir.Attribute.
pub fn static(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.static", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.stencil_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn stencil_divide(ctx: *mlir.Context, input: *const mlir.Value, stencil: *const mlir.Value, padding_upper: ?*const mlir.Value, padding_lower: ?*const mlir.Value, traversal_stride: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.stencil_divide", .{
        .operands = .{ .variadic = &.{
            &.{input},
            &.{stencil},
            if (padding_upper) |value| &.{value} else &.{},
            if (padding_lower) |value| &.{value} else &.{},
            if (traversal_stride) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.symbolic`. Result types are explicit; attributes use mlir.Attribute.
pub fn symbolic(ctx: *mlir.Context, symbolic_type: *const mlir.Type, name: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.symbolic", .{
        .results = .{ .flat = &.{symbolic_type} },
        .attributes = &.{
            .named(ctx, "name", name),
        },
        .location = location,
    });
}

/// `cute.tile_to_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn tile_to_shape(ctx: *mlir.Context, block: *const mlir.Value, trg_shape: *const mlir.Value, ord_shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tile_to_shape", .{
        .operands = .{ .flat = &.{ block, trg_shape, ord_shape } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tiled.copy.partition_D`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_copy_partition_D(ctx: *mlir.Context, tiled_copy: *const mlir.Value, input: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled.copy.partition_D", .{
        .operands = .{ .flat = &.{ tiled_copy, input, coord } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tiled.copy.partition_S`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_copy_partition_S(ctx: *mlir.Context, tiled_copy: *const mlir.Value, input: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled.copy.partition_S", .{
        .operands = .{ .flat = &.{ tiled_copy, input, coord } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tiled.copy.retile`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_copy_retile(ctx: *mlir.Context, tiled_copy: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled.copy.retile", .{
        .operands = .{ .flat = &.{ tiled_copy, input } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tiled.mma.partition`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_mma_partition(ctx: *mlir.Context, tiled_mma: *const mlir.Value, input: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled.mma.partition", .{
        .operands = .{ .flat = &.{ tiled_mma, input, coord } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .location = location,
    });
}

/// `cute.tiled.mma.partition_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_mma_partition_shape(ctx: *mlir.Context, tiled_mma: *const mlir.Value, input: *const mlir.Value, result_type: *const mlir.Type, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled.mma.partition_shape", .{
        .operands = .{ .flat = &.{ tiled_mma, input } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .location = location,
    });
}

/// `cute.tiled_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tiled_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn tiled_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.to_int_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn to_int_tuple(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.to_int_tuple", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute.tuple.product`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_product(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple.product", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple.product_each`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_product_each(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple.product_each", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_add`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_add(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_add", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_div`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_div", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_mod`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_mod(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_mod", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_mul`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_mul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_mul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_sub`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_sub(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_sub", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.unpack_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn unpack_tuple(ctx: *mlir.Context, tuple: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.unpack_tuple", .{
        .operands = .{ .flat = &.{tuple} },
        .results = .{ .variadic = &.{
            result_types,
        } },
        .location = location,
    });
}

/// `cute.zipped_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn zipped_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.zipped_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.zipped_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn zipped_product(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.zipped_product", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub const operation_names: []const []const u8 = &.{
    "cute.add_offset",
    "cute.append_to_rank",
    "cute.apply_swizzle",
    "cute.assume",
    "cute.blocked_product",
    "cute.ceil_div",
    "cute.coalesce",
    "cute.complement",
    "cute.composed_get_inner",
    "cute.composed_get_offset",
    "cute.composed_get_outer",
    "cute.composition",
    "cute.copy",
    "cute.copy.make_fragment",
    "cute.copy_atom_call",
    "cute.cosize",
    "cute.crd2idx",
    "cute.deref_arith_tuple_iter",
    "cute.deref_desc_iter",
    "cute.derefine",
    "cute.dice",
    "cute.elem_less",
    "cute.equal",
    "cute.fast_divmod.compute",
    "cute.fast_divmod.create_divisor",
    "cute.fast_divmod.divide",
    "cute.fast_divmod.get_aux",
    "cute.fast_divmod.get_divisor",
    "cute.fast_divmod.make_divisor",
    "cute.filter",
    "cute.filter_zeros",
    "cute.flat_divide",
    "cute.flat_product",
    "cute.gemm",
    "cute.get",
    "cute.get_flat_coord",
    "cute.get_hier_coord",
    "cute.get_integral_coord",
    "cute.get_iter",
    "cute.get_layout",
    "cute.get_layouts_from_tile",
    "cute.get_leaves",
    "cute.get_scalars",
    "cute.get_shape",
    "cute.get_stride",
    "cute.group_modes",
    "cute.idx2crd",
    "cute.increment_coord",
    "cute.inttoptr",
    "cute.kernel_smem_size",
    "cute.left_inverse",
    "cute.load_scaled_index",
    "cute.local_partition",
    "cute.local_tile",
    "cute.logical_divide",
    "cute.logical_product",
    "cute.make_arith_tuple_iter",
    "cute.make_atom",
    "cute.make_composed_layout",
    "cute.make_coord",
    "cute.make_desc_iter",
    "cute.make_fragment_like",
    "cute.make_identity_layout",
    "cute.make_identity_tensor",
    "cute.make_int_tuple",
    "cute.make_layout",
    "cute.make_layout_like",
    "cute.make_ordered_layout",
    "cute.make_shape",
    "cute.make_sparse_elem",
    "cute.make_stride",
    "cute.make_tile",
    "cute.make_tiled_copy",
    "cute.make_tiled_copy_v2",
    "cute.make_tiled_mma",
    "cute.make_tuple",
    "cute.make_view",
    "cute.memref.alloc_smem",
    "cute.memref.alloca",
    "cute.memref.load",
    "cute.memref.load_vec",
    "cute.memref.store",
    "cute.memref.store_vec",
    "cute.mma.make_fragment",
    "cute.mma_atom_call",
    "cute.prefetch",
    "cute.prefetch_atom_call",
    "cute.prepend_to_rank",
    "cute.print",
    "cute.print_tma_desc_im2col",
    "cute.print_tma_desc_tiled",
    "cute.print_view",
    "cute.ptr.load",
    "cute.ptr.store",
    "cute.ptrtoint",
    "cute.raked_product",
    "cute.random_delay",
    "cute.recast_iter",
    "cute.recast_layout",
    "cute.reduce",
    "cute.right_inverse",
    "cute.select",
    "cute.shape_div",
    "cute.size",
    "cute.slice",
    "cute.smem_partition_size",
    "cute.static",
    "cute.stencil_divide",
    "cute.symbolic",
    "cute.tile_to_shape",
    "cute.tiled.copy.partition_D",
    "cute.tiled.copy.partition_S",
    "cute.tiled.copy.retile",
    "cute.tiled.mma.partition",
    "cute.tiled.mma.partition_shape",
    "cute.tiled_divide",
    "cute.tiled_product",
    "cute.to_int_tuple",
    "cute.tuple.product",
    "cute.tuple.product_each",
    "cute.tuple_add",
    "cute.tuple_div",
    "cute.tuple_mod",
    "cute.tuple_mul",
    "cute.tuple_sub",
    "cute.unpack_tuple",
    "cute.zipped_divide",
    "cute.zipped_product",
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(nvgpu);
    inline for (comptime std.meta.declarations(@This())) |decl| {
        const T = @field(@This(), decl.name);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
    inline for (comptime std.meta.declarations(nvgpu)) |decl| {
        const T = @field(nvgpu, decl.name);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (cute.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    for (nvgpu.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "generated types and attributes rebuild from their getters" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (.{
        .{ ArithTupleIteratorType, "!cute.arith_tuple_iter<\"(?,?{div=16})\">" },
        .{ ComposedLayoutType, "!cute.composed_layout<\"(4,5):(1,4) o 2 o (2,3):(1,2)\">" },
        .{ ConstrainedIntType, "!cute.i16<pow2, divby 8>" },
        .{ CoordTensorType, "!cute.coord_tensor<\"(?,?,?)\", \"((512,1),1):((1/2@0,0),0)\">" },
        .{ CoordType, "!cute.coord<\"1\">" },
        .{ FastDivmodDivisorType, "!cute.fast_divmod_divisor<32>" },
        .{ IntTupleType, "!cute.int_tuple<\"1\">" },
        .{ LayoutType, "!cute.layout<\"(2,3):(1,2)\">" },
        .{ MemRefType, "!cute.memref<bf16, smem, align<1024>, S<2,4,3>, \"((8,16),(32,1),(1,3)):((32,256),(1,0),(0,4096))\">" },
        .{ PtrType, "!cute.ptr<bf16, smem, align<1024>, S<2,4,3>>" },
        .{ ShapeType, "!cute.shape<\"(2,3)\">" },
        .{ SparseElemType, "!cute.sparse_elem<2, vector<1xf16>>" },
        .{ StrideType, "!cute.stride<\"1\">" },
        .{ SwizzleType, "!cute.swizzle<\"S<0,4,3>\">" },
        .{ TileType, "!cute.tile<\"[(2,3):(1,2)]\">" },
        .{ TiledCopyType, "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32>, layout_copy_tv = <\"(32,4):(4,1)\">, tiler_mn = <\"[4:1;32:1]\">>" },
        .{ TiledCopyV2Type, "!cute.tiled_copy_v2<!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>, layout_copy_tv = <\"(1,8):(0,1)\">, tiler_mn = <\"[8:1]\">>" },
        .{ TiledMmaType, "!cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <\"(16,16,1):(16,1,0)\">, permutation_MNK = <\"[(16,4):(4,1);(16,4):(4,1);_]\">>" },
        .{ TupleType, "!cute.tuple<i32, f32>" },
        .{ BitLayoutAttr, "#cute.bitlayout<<\"(8,4):(1,8)\">>" },
        .{ CopyAtomAttr, "#cute.copy_atom<thr_id = <\"1:0\">, layout_tv = (<\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">)>" },
        .{ CopyAtomV2Attr, "#cute.copy_atom_v2<thr_id = <\"1:0\">, layout_tv = (<\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">), frg = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ MmaAtomAttr, "#cute.mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ MxMmaAtomAttr, "#cute.mx_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">, <\"1\">, <\"2\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ SparseMmaAtomAttr, "#cute.sparse_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">, <\"1\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ SparseMxMmaAtomAttr, "#cute.sparse_mx_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"2:0\">, <\"3:0\">, <\"4:0\">, <\"5:0\">, <\"6:0\">), shapes = (<\"1\">, <\"2\">, <\"3\">, <\"4\">, <\"5\">, <\"6\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ nvgpu.CopyAtomBulkCopyG2SType, "!cute_nvgpu.atom.bulk_copy_g2s<f32 copy_bits = 128 mcast = true>" },
        .{ nvgpu.CopyAtomBulkCopyS2GType, "!cute_nvgpu.atom.bulk_copy_s2g<f32 copy_bits = 128 mask = true>" },
        .{ nvgpu.CopyAtomBulkCopyS2SType, "!cute_nvgpu.atom.bulk_copy_s2s<f32 copy_bits = 128>" },
        .{ nvgpu.CopyAtomDsmemStoreType, "!cute_nvgpu.atom.dsmem_store<f32, 64 b>" },
        .{ nvgpu.CopyAtomG2RType, "!cute_nvgpu.atom.g2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu l2_prefetch_size = size_128b l1_cache_evict_priority = evict_last shared_space = cluster>" },
        .{ nvgpu.CopyAtomIm2ColTmaLoadType, "!cute_nvgpu.atom.im2col_tma_load<f16, copy_bits = 131072, num_cta = 1, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomIm2ColTmaStoreType, "!cute_nvgpu.atom.im2col_tma_store<f16, copy_bits = 65536, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomLdsmType, "!cute_nvgpu.atom.ldsm<val_type = f16, mode = <\"(8,8)\">, sz_pattern = u16, num_matrices = 4, t>" },
        .{ nvgpu.CopyAtomNonExec2DGather4TmaLoadType, "!cute_nvgpu.atom.non_exec_2d_gather4_tma_load<sm_100, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExec2DScatter4TmaStoreType, "!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(4,64):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecIm2ColTmaLoadType, "!cute_nvgpu.atom.non_exec_im2col_tma_load<sm_90, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecIm2ColTmaStoreType, "!cute_nvgpu.atom.non_exec_im2col_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaLoadType, "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaReduceType, "!cute_nvgpu.atom.non_exec_tiled_tma_reduce<ADD, f32, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F32_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaStoreType, "!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ nvgpu.CopyAtomR2GType, "!cute_nvgpu.atom.r2g<f16 copy_bits = 128 l1_cache_evict_priority = no_allocate>" },
        .{ nvgpu.CopyAtomR2SType, "!cute_nvgpu.atom.r2s<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ nvgpu.CopyAtomS2RType, "!cute_nvgpu.atom.s2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ nvgpu.CopyAtomSIMTAsyncCopyType, "!cute_nvgpu.atom.simt_async_copy<bf16, cache = global, 128 b>" },
        .{ nvgpu.CopyAtomSIMTMultimemLdReduceType, "!cute_nvgpu.atom.simt_multimem_ld_reduce<bf16, 64 b, MIN, <F32>, acquire, gpu>" },
        .{ nvgpu.CopyAtomSIMTMultimemRedType, "!cute_nvgpu.atom.simt_multimem_red<f16, 64 b, MAX, release, gpu>" },
        .{ nvgpu.CopyAtomSIMTMultimemStType, "!cute_nvgpu.atom.simt_multimem_st<f16, 64 b, release, gpu>" },
        .{ nvgpu.CopyAtomSIMTSyncCopyType, "!cute_nvgpu.atom.universal_copy<f32, 128 b, src_space = gmem, dst_space = smem>" },
        .{ nvgpu.CopyAtomSM100CopyS2TType, "!cute_nvgpu.atom.s2t_copy<f8E4M3FN, 32 DP, 128 bit, 1 cta, x4>" },
        .{ nvgpu.CopyAtomSM100S2TCopyV2Type, "!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>" },
        .{ nvgpu.CopyAtomSM100TmemLoadType, "!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x4, pack16b>" },
        .{ nvgpu.CopyAtomSM100TmemStoreType, "!cute_nvgpu.atom.tmem_store<f32, 32 DP, 32 bit, x4, expand16b>" },
        .{ nvgpu.CopyAtomSM107TmemLoadSPCompressType, "!cute_nvgpu.atom.tmem_load_spcompress<f32, 32 DP, 32 bit, x4, maxabs, red, nan>" },
        .{ nvgpu.CopyAtomSM10xTmemLoadRedType, "!cute_nvgpu.atom.tmem_load_red<f32, 16 DP, 32 bit, x4, maxabs, nan, half_split_off=3 : i64>" },
        .{ nvgpu.CopyAtomStsmType, "!cute_nvgpu.atom.stsm<f16, mode = <\"(8,8)\">, num_matrices = 4, n>" },
        .{ nvgpu.CopyAtomTmaLoadType, "!cute_nvgpu.atom.tma_load<bf16, copy_bits = 1024, mode = tiled, num_cta = 2, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,1):(1@0,1@1)\">>" },
        .{ nvgpu.CopyAtomTmaReduceType, "!cute_nvgpu.atom.tma_reduce<f32, copy_bits = 131072, mode = tiled, kind = ADD, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomTmaStoreType, "!cute_nvgpu.atom.tma_store<bf16, copy_bits = 16384, mode = tiled, g_stride = <\"()\"> tma_gbasis = <\"(64,16):(1@0,1@1)\">>" },
        .{ nvgpu.MmaAtomSM100UMMABlockScaledSparseType, "!cute_nvgpu.sm100.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32>" },
        .{ nvgpu.MmaAtomSM100UMMABlockScaledType, "!cute_nvgpu.sm100.mma_bs<128x128x96, num_cta = 1, ab_major = (k, k), elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 16, arch_promote = sm_103>" },
        .{ nvgpu.MmaAtomSM100UMMASparseType, "!cute_nvgpu.sm100.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0>" },
        .{ nvgpu.MmaAtomSM100UMMAType, "!cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0>" },
        .{ nvgpu.MmaAtomSM107UMMABlockScaledSparseType, "!cute_nvgpu.sm107.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, discard)>" },
        .{ nvgpu.MmaAtomSM107UMMABlockScaledType, "!cute_nvgpu.sm107.mma_bs<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, use), int_overflow = <satfinite>>" },
        .{ nvgpu.MmaAtomSM107UMMASparseType, "!cute_nvgpu.sm107.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0, ab_collector_op = (discard, discard)>" },
        .{ nvgpu.MmaAtomSM107UMMAType, "!cute_nvgpu.sm107.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0, ab_collector_op = (lastuse, fill)>" },
        .{ nvgpu.MmaAtomSM120BlockScaledType, "!cute_nvgpu.SM120.mma_bs<16x8x64, vec_size = 32, elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, use_sf_layout_TV = true>" },
        .{ nvgpu.MmaAtomSM80SparseType, "!cute_nvgpu.sm80.sparse_mma<16x8x64, elem_type = (i8, i8, i32), md_format = tid, int_overflow = <satfinite> >" },
        .{ nvgpu.MmaAtomSM80Type, "!cute_nvgpu.sm80.mma<16x8x32, elem_type = (i8, i8, i32), int_overflow = <wrapped>, binary_op = <none> >" },
        .{ nvgpu.MmaAtomSM89Type, "!cute_nvgpu.sm89.mma<16x8x32, elem_type = (f8E4M3FN, f8E4M3FN, f32) >" },
        .{ nvgpu.MmaAtomSM90Type, "!cute_nvgpu.sm90.mma<64x128x32, ab_major = (k, k), elem_type = (i8, i8, i32), frag_kind = ss, int_overflow = <satfinite>, a_neg, b_neg>" },
        .{ nvgpu.SmemDescCircularSM103Type, "!cute_nvgpu.sm103.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ nvgpu.SmemDescCircularSM107Type, "!cute_nvgpu.sm107.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ nvgpu.SmemDescSM107Type, "!cute_nvgpu.sm107.smem_desc" },
        .{ nvgpu.SmemDescType, "!cute_nvgpu.smem_desc" },
        .{ nvgpu.SmemDescViewType, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(2,3):(1,2)\">" },
        .{ nvgpu.TmaDescriptorIm2ColType, "!cute_nvgpu.tma_descriptor_im2col" },
        .{ nvgpu.TmaDescriptorTiledType, "!cute_nvgpu.tma_descriptor_tiled" },
        .{ nvgpu.UniversalFmaAtomType, "!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >" },
        .{ nvgpu.WorkIdResponseType, "!cute_nvgpu.workid_response" },
        .{ nvgpu.NotImplementedFrgAttr, "#cute_nvgpu.not_implemented_frg<>" },
        .{ nvgpu.RmemFrgAttr, "#cute_nvgpu.rmem_frg<value_type = f32, operand = A, derive_elem_type = true>" },
        .{ nvgpu.SM100CircularSmemFrgAttr, "#cute_nvgpu.arch.sm100.circular_smem_frg<major = k>" },
        .{ nvgpu.SM100SmemFrgAttr, "#cute_nvgpu.arch.sm100.smem_frg<major = k>" },
        .{ nvgpu.SM100TmemEFrgAttr, "#cute_nvgpu.arch.sm100.tmem_e_frg<a_type = f16, e_type = i8>" },
        .{ nvgpu.SM100TmemFrgAttr, "#cute_nvgpu.arch.sm100.tmem_frg<data_type = f32, storage_type = f32, cta_group = 1, tmem_alloc_mode = Interleaved>" },
        .{ nvgpu.SM100TmemSfFrgAttr, "#cute_nvgpu.arch.sm100.tmem_sf_frg<sf_type = f8E8M0FNU, sf_vec_size = 32, cta_group = 1, is_sfa = true, tmem_alloc_mode = Interleaved>" },
        .{ nvgpu.SM107SmemFrgAttr, "#cute_nvgpu.arch.sm107.smem_frg<major = k>" },
        .{ nvgpu.SM90SmemFrgAttr, "#cute_nvgpu.arch.sm90.smem_frg<major = k>" },
    }) |example| {
        const T = example[0];
        const is_type = @hasDecl(T, "type_");
        const parsed = if (is_type) try mlir.Type.parse(ctx, example[1]) else try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(T) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuild(T, ctx, value)).eql(value));
        var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer text.deinit();
        try text.writer.print("{f}", .{value});
        try std.testing.expect(parsed.eql(if (is_type) try mlir.Type.parse(ctx, text.written()) else try mlir.Attribute.parse(ctx, text.written())));
    }
    // Every case of every enum attribute.
    inline for (.{ cute, nvgpu }) |namespace| {
        inline for (comptime std.meta.declarations(namespace)) |decl| {
            const T = @field(namespace, decl.name);
            if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque" and @hasDecl(T, "InitArgs")) {
                const fields = @typeInfo(T.InitArgs).@"struct".fields;
                if (fields.len == 1 and comptime std.mem.eql(u8, fields[0].name, "value")) {
                    inline for (@typeInfo(fields[0].type).@"enum".fields) |case| {
                        const attr = try T.get(ctx, .{ .value = @enumFromInt(case.value) });
                        try std.testing.expectEqual(case.value, @intFromEnum(attr.getValue()));
                        try std.testing.expect((try rebuild(T, ctx, attr)).eql(attr));
                    }
                }
            }
        }
    }
    try std.testing.expect((try nvgpu.SmemDescType.get(ctx, .{})).type_().isA(nvgpu.NVGPUType) != null);
}

fn rebuild(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var elements: [8]*const mlir.Type = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".fields) |field| {
        const name = [_]u8{comptime std.ascii.toUpper(field.name[0])} ++ field.name[1..];
        if (field.type == []const *const mlir.Type) {
            const count = @field(T, "getNum" ++ name)(value);
            for (elements[0..count], 0..) |*element, i| element.* = @field(T, "get" ++ name[0 .. name.len - 1])(value, i);
            @field(args, field.name) = elements[0..count];
        } else {
            @field(args, field.name) = @field(T, "get" ++ name)(value);
        }
    }
    return T.get(ctx, args);
}
