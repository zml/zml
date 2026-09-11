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

pub const MemorySpace = enum { generic, gmem, cmem, smem, rmem, tmem, dsmem };

pub fn pointerType(ctx: *mlir.Context, element_type: ?*const mlir.Type, space: MemorySpace, alignment: u64) !*const mlir.Type {
    return (try PtrType.get(ctx, .{
        .valueType = element_type,
        .memorySpace = .string(ctx, @tagName(space)),
        .alignment = alignment,
    })).type_();
}

pub fn constrainedIntType(ctx: *mlir.Context, width: enum { i32, i64 }, divisible_by: u64) !*const mlir.Type {
    return switch (width) {
        .i32 => (try ConstrainedInt32Type.get(ctx, .{ .divisibleBy = divisible_by })).type_(),
        .i64 => (try ConstrainedInt64Type.get(ctx, .{ .divisibleBy = divisible_by })).type_(),
    };
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

/// `!cute.arith_tuple_iter`.
pub const ArithTupleIteratorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteArithTupleIterator;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.composed_layout`.
pub const ComposedLayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteComposedLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.i32`.
pub const ConstrainedInt32Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteConstrainedInt32;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        divisibleBy: u64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteConstrainedInt32TypeGet(ctx.ptr(), args.divisibleBy);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getDivisibleBy(self: *const Self) u64 {
        return c.mlirCuteConstrainedInt32TypeGetDivisibleBy(self.ptr());
    }
};

/// `!cute.i64`.
pub const ConstrainedInt64Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteConstrainedInt64;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        divisibleBy: u64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteConstrainedInt64TypeGet(ctx.ptr(), args.divisibleBy);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getDivisibleBy(self: *const Self) u64 {
        return c.mlirCuteConstrainedInt64TypeGetDivisibleBy(self.ptr());
    }
};

/// `!cute.coord_tensor`.
pub const CoordTensorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteCoordTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.coord`.
pub const CoordType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteCoord;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.fast_divmod_divisor`.
pub const FastDivmodDivisorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteFastDivmodDivisor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.int_tuple`.
pub const IntTupleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.layout`.
pub const LayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.memref`.
pub const MemRefType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.ptr`.
pub const PtrType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACutePtr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valueType: ?*const mlir.Type = null,
        memorySpace: *const mlir.Attribute,
        alignment: u64,
        swizzle: ?*const mlir.Attribute = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCutePtrTypeGet(ctx.ptr(), if (args.valueType) |value| value.ptr() else c.MlirType{ .ptr = null }, args.memorySpace.ptr(), args.alignment, if (args.swizzle) |value| value.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValueType(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCutePtrTypeGetValueType(self.ptr()).ptr);
    }
    pub fn getMemorySpace(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCutePtrTypeGetMemorySpace(self.ptr()).ptr.?);
    }
    pub fn getAlignment(self: *const Self) u64 {
        return c.mlirCutePtrTypeGetAlignment(self.ptr());
    }
    pub fn getSwizzle(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCutePtrTypeGetSwizzle(self.ptr()).ptr);
    }
};

/// `!cute.shape`.
pub const ShapeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.sparse_elem`.
pub const SparseElemType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteSparseElem;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.stride`.
pub const StrideType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteStride;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.swizzle`.
pub const SwizzleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteSwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

/// `!cute.tile`.
pub const TileType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
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

// Builders from the pinned OSS CuteOps.td.
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

/// `cute.blocked_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn blocked_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.blocked_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
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
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    if (target_profile) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.coalesce", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.complement`. Result types are explicit; attributes use mlir.Attribute.
pub fn complement(ctx: *mlir.Context, input: *const mlir.Value, cotarget: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    if (cotarget) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.complement", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
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
pub fn composition(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.composition", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
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
pub fn crd2idx(ctx: *mlir.Context, coord: *const mlir.Value, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.crd2idx", .{
        .operands = .{ .flat = &.{ coord, shape } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.dice`. Result types are explicit; attributes use mlir.Attribute.
pub fn dice(ctx: *mlir.Context, input: *const mlir.Value, coord: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.dice", .{
        .operands = .{ .flat = &.{ input, coord } },
        .results = .{ .flat = &.{result_type} },
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

/// `cute.flat_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn flat_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.flat_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.flat_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn flat_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.flat_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.flatten`. Result types are explicit; attributes use mlir.Attribute.
pub fn flatten(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.flatten", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.get_layouts_from_tile`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_layouts_from_tile(ctx: *mlir.Context, tile: *const mlir.Value, layouts_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_layouts_from_tile", .{
        .operands = .{ .flat = &.{tile} },
        .results = .{ .variadic = &.{layouts_types} },
        .location = location,
    });
}

/// `cute.get_leaves`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_leaves(ctx: *mlir.Context, input: *const mlir.Value, results_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.get_leaves", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .variadic = &.{results_types} },
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

/// `cute.get_scalars`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_scalars(ctx: *mlir.Context, cute_value: *const mlir.Value, scalars_types: []const *const mlir.Type, only_dynamic: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (only_dynamic) |value| attributes.appendAssumeCapacity(.named(ctx, "only_dynamic", value));
    return mlir.Operation.make(ctx, "cute.get_scalars", .{
        .operands = .{ .flat = &.{cute_value} },
        .results = .{ .variadic = &.{scalars_types} },
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
pub fn idx2crd(ctx: *mlir.Context, index: *const mlir.Value, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.idx2crd", .{
        .operands = .{ .flat = &.{ index, shape } },
        .results = .{ .flat = &.{result_type} },
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

/// `cute.layout_eval`. Result types are explicit; attributes use mlir.Attribute.
pub fn layout_eval(ctx: *mlir.Context, coord: *const mlir.Value, layout: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.layout_eval", .{
        .operands = .{ .flat = &.{ coord, layout } },
        .results = .{ .flat = &.{result_type} },
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

/// `cute.logical_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn logical_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.logical_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.logical_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn logical_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.logical_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
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
pub fn make_coord(ctx: *mlir.Context, operands_: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_coord", .{
        .operands = .{ .variadic = &.{operands_} },
        .results = .{ .flat = &.{result_type} },
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

/// `cute.make_int_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_int_tuple(ctx: *mlir.Context, operands_: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_int_tuple", .{
        .operands = .{ .variadic = &.{operands_} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_layout_like`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_layout_like(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_layout_like", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_layout(ctx: *mlir.Context, shape: *const mlir.Value, stride: *const mlir.Value, layout_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_layout", .{
        .operands = .{ .flat = &.{ shape, stride } },
        .results = .{ .flat = &.{layout_type} },
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
pub fn make_shape(ctx: *mlir.Context, operands_: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_shape", .{
        .operands = .{ .variadic = &.{operands_} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_stride`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_stride(ctx: *mlir.Context, operands_: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_stride", .{
        .operands = .{ .variadic = &.{operands_} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_tile`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tile(ctx: *mlir.Context, operands_: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_tile", .{
        .operands = .{ .variadic = &.{operands_} },
        .results = .{ .flat = &.{result_type} },
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
pub fn print(ctx: *mlir.Context, value_: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.print", .{
        .operands = .{ .flat = &.{value_} },
        .location = location,
    });
}

/// `cute.raked_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn raked_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.raked_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
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

/// `cute.static`. Result types are explicit; attributes use mlir.Attribute.
pub fn static(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.static", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tile_to_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn tile_to_shape(ctx: *mlir.Context, input: *const mlir.Value, shape: *const mlir.Value, order: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    operands.appendSliceAssumeCapacity(&.{shape});
    if (order) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.tile_to_shape", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
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
pub fn tiled_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tiled_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.to_coord`. Result types are explicit; attributes use mlir.Attribute.
pub fn to_coord(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.to_coord", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.to_int_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn to_int_tuple(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.to_int_tuple", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.to_shape`. Result types are explicit; attributes use mlir.Attribute.
pub fn to_shape(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.to_shape", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.to_stride`. Result types are explicit; attributes use mlir.Attribute.
pub fn to_stride(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.to_stride", .{
        .operands = .{ .flat = &.{src} },
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

/// `cute.tuple_product_each`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_product_each(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_product_each", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn tuple_product(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple_product", .{
        .operands = .{ .flat = &.{input} },
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

/// `cute.zipped_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn zipped_divide(ctx: *mlir.Context, input: *const mlir.Value, tiler: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.zipped_divide", .{
        .operands = .{ .flat = &.{ input, tiler } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.zipped_product`. Result types are explicit; attributes use mlir.Attribute.
pub fn zipped_product(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.zipped_product", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Builders from CuteOpsPrivate.td.
/// `cute.add_offset`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_offset(ctx: *mlir.Context, src: *const mlir.Value, offset: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.add_offset", .{
        .operands = .{ .flat = &.{ src, offset } },
        .results = .{ .flat = &.{dst_type} },
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

/// `cute.deref_arith_tuple_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn deref_arith_tuple_iter(ctx: *mlir.Context, iter: *const mlir.Value, arith_tuple_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.deref_arith_tuple_iter", .{
        .operands = .{ .flat = &.{iter} },
        .results = .{ .flat = &.{arith_tuple_type} },
        .location = location,
    });
}

/// `cute.deref_desc_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn deref_desc_iter(ctx: *mlir.Context, iter: *const mlir.Value, value_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.deref_desc_iter", .{
        .operands = .{ .flat = &.{iter} },
        .results = .{ .flat = &.{value_type} },
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
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    if (target_profile) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.filter_zeros", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{res_type} },
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

/// `cute.make_arith_tuple_iter`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_arith_tuple_iter(ctx: *mlir.Context, value_: ?*const mlir.Value, iter_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_arith_tuple_iter", .{
        .operands = .{ .flat = if (value_) |value| &.{value} else &.{} },
        .results = .{ .flat = &.{iter_type} },
        .location = location,
    });
}

/// `cute.make_atom`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_atom(ctx: *mlir.Context, values: []const *const mlir.Value, atom_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_atom", .{
        .operands = .{ .variadic = &.{values} },
        .results = .{ .flat = &.{atom_type} },
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

/// `cute.make_identity_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_identity_tensor(ctx: *mlir.Context, shape: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.make_identity_tensor", .{
        .operands = .{ .flat = &.{shape} },
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
        .operands = .{ .variadic = &.{values} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.make_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_view(ctx: *mlir.Context, iter: *const mlir.Value, layout: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{iter});
    if (layout) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.make_view", .{
        .operands = .{ .flat = operands.constSlice() },
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
        .operands = .{ .flat = if (layout) |value| &.{value} else &.{} },
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
    var operands: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands.appendSliceAssumeCapacity(&.{src});
    if (mask) |value| operands.appendSliceAssumeCapacity(&.{value});
    if (pass_thru) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.memref.load_vec", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{
                1,
                if (mask != null) 1 else 0,
                if (pass_thru != null) 1 else 0,
            })),
        },
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
    var operands: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands.appendSliceAssumeCapacity(&.{value_});
    operands.appendSliceAssumeCapacity(&.{dst});
    if (mask) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.memref.store_vec", .{
        .operands = .{ .flat = operands.constSlice() },
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
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{src});
    if (coord) |value| operands.appendSliceAssumeCapacity(&.{value});
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (verbose) |value| attributes.appendAssumeCapacity(.named(ctx, "verbose", value));
    if (is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    return mlir.Operation.make(ctx, "cute.print_view", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute.ptr.load`. Result types are explicit; attributes use mlir.Attribute.
pub fn ptr_load(ctx: *mlir.Context, ptr: *const mlir.Value, value_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.ptr.load", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{value_type} },
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

/// `cute.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn reduce(ctx: *mlir.Context, input: *const mlir.Value, shape: *const mlir.Value, init_val: *const mlir.Value, predication: ?*const mlir.Value, output_type: *const mlir.Type, reduced_shape_type: ?*const mlir.Type, reduction_op: *const mlir.Attribute, reduction_profile: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 4) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    operands.appendSliceAssumeCapacity(&.{shape});
    operands.appendSliceAssumeCapacity(&.{init_val});
    if (predication) |value| operands.appendSliceAssumeCapacity(&.{value});
    var results: stdx.BoundedArray(*const mlir.Type, 2) = .empty;
    results.appendSliceAssumeCapacity(&.{output_type});
    if (reduced_shape_type) |value| results.appendSliceAssumeCapacity(&.{value});
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "reduction_op", reduction_op));
    if (reduction_profile) |value| attributes.appendAssumeCapacity(.named(ctx, "reduction_profile", value));
    return mlir.Operation.make(ctx, "cute.reduce", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
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

/// `cute.stencil_divide`. Result types are explicit; attributes use mlir.Attribute.
pub fn stencil_divide(ctx: *mlir.Context, input: *const mlir.Value, stencil: *const mlir.Value, padding_upper: ?*const mlir.Value, padding_lower: ?*const mlir.Value, traversal_stride: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands.appendSliceAssumeCapacity(&.{input});
    operands.appendSliceAssumeCapacity(&.{stencil});
    if (padding_upper) |value| operands.appendSliceAssumeCapacity(&.{value});
    if (padding_lower) |value| operands.appendSliceAssumeCapacity(&.{value});
    if (traversal_stride) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute.stencil_divide", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{
                1,
                1,
                if (padding_upper != null) 1 else 0,
                if (padding_lower != null) 1 else 0,
                if (traversal_stride != null) 1 else 0,
            })),
        },
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

/// `cute.tuple.product_each`. Result types are explicit; attributes use mlir.Attribute.
pub fn @"tuple.product_each"(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple.product_each", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.tuple.product`. Result types are explicit; attributes use mlir.Attribute.
pub fn @"tuple.product"(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.tuple.product", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute.unpack_tuple`. Result types are explicit; attributes use mlir.Attribute.
pub fn unpack_tuple(ctx: *mlir.Context, tuple: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute.unpack_tuple", .{
        .operands = .{ .flat = &.{tuple} },
        .results = .{ .variadic = &.{result_types} },
        .location = location,
    });
}

pub const operation_names: []const []const u8 = &.{
    "cute.append_to_rank",
    "cute.blocked_product",
    "cute.ceil_div",
    "cute.coalesce",
    "cute.complement",
    "cute.composed_get_inner",
    "cute.composed_get_offset",
    "cute.composed_get_outer",
    "cute.composition",
    "cute.cosize",
    "cute.crd2idx",
    "cute.dice",
    "cute.elem_less",
    "cute.equal",
    "cute.flat_divide",
    "cute.flat_product",
    "cute.flatten",
    "cute.get_layouts_from_tile",
    "cute.get_leaves",
    "cute.get",
    "cute.get_scalars",
    "cute.get_shape",
    "cute.get_stride",
    "cute.group_modes",
    "cute.idx2crd",
    "cute.increment_coord",
    "cute.layout_eval",
    "cute.left_inverse",
    "cute.logical_divide",
    "cute.logical_product",
    "cute.make_composed_layout",
    "cute.make_coord",
    "cute.make_identity_layout",
    "cute.make_int_tuple",
    "cute.make_layout_like",
    "cute.make_layout",
    "cute.make_ordered_layout",
    "cute.make_shape",
    "cute.make_stride",
    "cute.make_tile",
    "cute.prepend_to_rank",
    "cute.print",
    "cute.raked_product",
    "cute.recast_layout",
    "cute.right_inverse",
    "cute.select",
    "cute.shape_div",
    "cute.size",
    "cute.slice",
    "cute.static",
    "cute.tile_to_shape",
    "cute.tiled_divide",
    "cute.tiled_product",
    "cute.to_coord",
    "cute.to_int_tuple",
    "cute.to_shape",
    "cute.to_stride",
    "cute.tuple_add",
    "cute.tuple_product_each",
    "cute.tuple_product",
    "cute.tuple_sub",
    "cute.zipped_divide",
    "cute.zipped_product",
    "cute.add_offset",
    "cute.apply_swizzle",
    "cute.assume",
    "cute.copy_atom_call",
    "cute.copy.make_fragment",
    "cute.copy",
    "cute.deref_arith_tuple_iter",
    "cute.deref_desc_iter",
    "cute.derefine",
    "cute.fast_divmod.compute",
    "cute.fast_divmod.create_divisor",
    "cute.fast_divmod.divide",
    "cute.fast_divmod.get_aux",
    "cute.fast_divmod.get_divisor",
    "cute.fast_divmod.make_divisor",
    "cute.filter",
    "cute.filter_zeros",
    "cute.gemm",
    "cute.get_flat_coord",
    "cute.get_hier_coord",
    "cute.get_integral_coord",
    "cute.get_iter",
    "cute.get_layout",
    "cute.inttoptr",
    "cute.kernel_smem_size",
    "cute.load_scaled_index",
    "cute.local_partition",
    "cute.local_tile",
    "cute.make_arith_tuple_iter",
    "cute.make_atom",
    "cute.make_desc_iter",
    "cute.make_fragment_like",
    "cute.make_identity_tensor",
    "cute.make_sparse_elem",
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
    "cute.mma_atom_call",
    "cute.mma.make_fragment",
    "cute.prefetch_atom_call",
    "cute.prefetch",
    "cute.print_tma_desc_im2col",
    "cute.print_tma_desc_tiled",
    "cute.print_view",
    "cute.ptr.load",
    "cute.ptr.store",
    "cute.ptrtoint",
    "cute.random_delay",
    "cute.recast_iter",
    "cute.reduce",
    "cute.smem_partition_size",
    "cute.stencil_divide",
    "cute.symbolic",
    "cute.tiled.copy.partition_D",
    "cute.tiled.copy.partition_S",
    "cute.tiled.copy.retile",
    "cute.tiled.mma.partition",
    "cute.tiled.mma.partition_shape",
    "cute.tuple_div",
    "cute.tuple_mod",
    "cute.tuple_mul",
    "cute.tuple.product_each",
    "cute.tuple.product",
    "cute.unpack_tuple",
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

test "public and private types through the C API" {
    const ctx = try testContext();
    defer ctx.deinit();
    const tuple = try algebraType(ctx, .int_tuple, "(16,32)");
    try std.testing.expect(tuple.eql(try mlir.Type.parse(ctx, "!cute.int_tuple<\"(16,32)\">")));
    try std.testing.expect(tuple.isA(CuteType) != null);
    const f32_type = mlir.Type.float(ctx, .f32);
    try std.testing.expect((try pointerType(ctx, f32_type, .gmem, 16)).eql(try mlir.Type.parse(ctx, "!cute.ptr<f32, gmem, align<16>>")));
    try std.testing.expect((try constrainedIntType(ctx, .i32, 8)).eql(try mlir.Type.parse(ctx, "!cute.i32<divby 8>")));
    try std.testing.expect((try algebraAttribute(ctx, .layout, "(16,32):(32,1)")).eql(try mlir.Attribute.parse(ctx, "#cute.layout<\"(16,32):(32,1)\">")));
    try std.testing.expect(mlir.Type.int(ctx, .i32).isA(CuteType) == null);
    try std.testing.expect((try nvgpu.SmemDescType.get(ctx, .{})).isA(nvgpu.NVGPUType) != null);
    const atom = try nvgpu.CopyAtomSIMTSyncCopyType.get(ctx, .{ .payload = .string(ctx, "<f32>") });
    try std.testing.expect(atom.type_().eql(try mlir.Type.parse(ctx, "!cute_nvgpu.atom.universal_copy<f32>")));
}

test "typed constructors and accessors use native type storage" {
    const ctx = try testContext();
    defer ctx.deinit();
    const f32_type = mlir.Type.float(ctx, .f32);
    const space = mlir.Attribute.string(ctx, "smem");
    const swizzle = try algebraAttribute(ctx, .swizzle, "S<3,4,3>");
    const pointer = try PtrType.get(ctx, .{
        .valueType = f32_type,
        .memorySpace = space,
        .alignment = 16,
        .swizzle = swizzle,
    });
    try std.testing.expect(pointer.getValueType().?.eql(f32_type));
    try std.testing.expect(pointer.getMemorySpace().eql(space));
    try std.testing.expectEqual(@as(u64, 16), pointer.getAlignment());
    try std.testing.expect(pointer.getSwizzle().?.eql(swizzle));
    try std.testing.expect(pointer.type_().isA(PtrType) != null);
    try std.testing.expect(f32_type.isA(PtrType) == null);
    const untyped_pointer = try PtrType.get(ctx, .{ .memorySpace = space, .alignment = 0 });
    try std.testing.expect(untyped_pointer.getValueType() == null);
    try std.testing.expect(untyped_pointer.getSwizzle() == null);

    const layout_attr = try algebraAttribute(ctx, .layout, "4:1");
    const layout = try LayoutType.get(ctx, .{ .attr = layout_attr });
    try std.testing.expect(layout.getAttr().eql(layout_attr));
    const memref = try MemRefType.get(ctx, .{ .ptr = pointer.type_(), .layout = layout.type_() });
    try std.testing.expect(memref.getPtr().eql(pointer.type_()));
    try std.testing.expect(memref.getLayout().eql(layout.type_()));
    const tuple = try algebraType(ctx, .int_tuple, "(4,8)");
    const iterator = try ArithTupleIteratorType.get(ctx, .{ .arithTuple = tuple });
    try std.testing.expect(iterator.getArithTuple().eql(tuple));
    const coords = try CoordTensorType.get(ctx, .{ .arithTuple = tuple, .layout = layout.type_() });
    try std.testing.expect(coords.getArithTuple().eql(tuple));
    try std.testing.expect(coords.getLayout().eql(layout.type_()));

    const i32_type = try ConstrainedInt32Type.get(ctx, .{ .divisibleBy = 8 });
    const i64_type = try ConstrainedInt64Type.get(ctx, .{ .divisibleBy = 16 });
    try std.testing.expectEqual(@as(u64, 8), i32_type.getDivisibleBy());
    try std.testing.expectEqual(@as(u64, 16), i64_type.getDivisibleBy());
    const divisor = try FastDivmodDivisorType.get(ctx, .{ .width = 64, .isPow2 = true });
    try std.testing.expectEqual(@as(c_uint, 64), divisor.getWidth());
    try std.testing.expect(divisor.getIsPow2());
    const sparse = try SparseElemType.get(ctx, .{ .numLogical = 2, .physicalType = f32_type });
    try std.testing.expectEqual(@as(c_int, 2), sparse.getNumLogical());
    try std.testing.expect(sparse.getPhysicalType().eql(f32_type));

    const payload = mlir.Attribute.string(ctx, "<f32>");
    const atom = try nvgpu.CopyAtomSIMTSyncCopyType.get(ctx, .{ .payload = payload });
    try std.testing.expect(atom.getPayload().eql(payload));
    try std.testing.expect(atom.type_().isA(nvgpu.CopyAtomSIMTSyncCopyType) != null);
    inline for (.{ nvgpu.SmemDescType, nvgpu.TmaDescriptorTiledType, nvgpu.TmaDescriptorIm2ColType, nvgpu.WorkIdResponseType }) |T| {
        try std.testing.expect((try T.get(ctx, .{})).type_().isA(T) != null);
    }
}

test "explicit algebra attribute bindings and type constructors" {
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (.{
        .{ IntTupleAttr, IntTupleType, "(16,32)" },
        .{ CoordAttr, CoordType, "(_,?)" },
        .{ ShapeAttr, ShapeType, "(16,32)" },
        .{ StrideAttr, StrideType, "(32,1)" },
        .{ LayoutAttr, LayoutType, "(16,32):(32,1)" },
        .{ TileAttr, TileType, "[(2,3):(1,2);_]" },
        .{ ComposedLayoutAttr, ComposedLayoutType, "S<3,5,4> o 2 o (2,3):(1,2)" },
        .{ SwizzleAttr, SwizzleType, "S<3,4,3>" },
    }) |example| {
        const A = example[0];
        const T = example[1];
        const attr = try A.get(ctx, example[2]);
        try std.testing.expect(attr.attribute().isA(A) != null);
        try std.testing.expect(mlir.Attribute.string(ctx, example[2]).isA(A) == null);
        try std.testing.expectError(error.InvalidMlir, A.get(ctx, "invalid"));
        const value = attr.getValue();
        try std.testing.expect(attr.eql(try A.get(ctx, value)));
        const type_ = try T.get(ctx, .{ .attr = attr.attribute() });
        try std.testing.expect(type_.getAttr().eql(attr.attribute()));

        var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer text.deinit();
        try text.writer.print("{f}", .{attr});
        try std.testing.expect(attr.attribute().eql(try mlir.Attribute.parse(ctx, text.written())));
        text.clearRetainingCapacity();
        try text.writer.print("{f}", .{type_});
        try std.testing.expect(type_.type_().eql(try mlir.Type.parse(ctx, text.written())));
        // The getter's view survives other attribute creation in this context.
        try std.testing.expectEqualStrings(value, attr.getValue());
    }
}

test "typed constructors reject invalid parameters" {
    const ctx = try testContext();
    defer ctx.deinit();
    const f32_type = mlir.Type.float(ctx, .f32);
    const space = mlir.Attribute.string(ctx, "smem");
    try std.testing.expectError(error.InvalidMlir, LayoutType.get(ctx, .{ .attr = space }));
    try std.testing.expectError(error.InvalidMlir, ArithTupleIteratorType.get(ctx, .{ .arithTuple = f32_type }));
    try std.testing.expectError(error.InvalidMlir, ConstrainedInt32Type.get(ctx, .{ .divisibleBy = 0 }));
    try std.testing.expectError(error.InvalidMlir, FastDivmodDivisorType.get(ctx, .{ .width = 8, .isPow2 = false }));
    try std.testing.expectError(error.InvalidMlir, SparseElemType.get(ctx, .{ .numLogical = 0, .physicalType = f32_type }));
    try std.testing.expectError(error.InvalidMlir, PtrType.get(ctx, .{ .memorySpace = space, .alignment = 3 }));
    try std.testing.expectError(error.InvalidMlir, PtrType.get(ctx, .{ .memorySpace = space, .alignment = 16, .swizzle = space }));
    const pointer = try pointerType(ctx, f32_type, .smem, 16);
    try std.testing.expectError(error.InvalidMlir, MemRefType.get(ctx, .{ .ptr = pointer, .layout = f32_type }));
    try std.testing.expectError(error.InvalidMlir, nvgpu.CopyAtomSIMTSyncCopyType.get(ctx, .{ .payload = .unit(ctx) }));

    const other_ctx = try testContext();
    defer other_ctx.deinit();
    try std.testing.expectError(error.InvalidMlir, PtrType.get(other_ctx, .{ .valueType = f32_type, .memorySpace = .string(other_ctx, "smem"), .alignment = 16 }));
    try std.testing.expectError(error.InvalidMlir, PtrType.get(other_ctx, .{ .memorySpace = space, .alignment = 16 }));
}

test "all NVGPU type constructors preserve their storage" {
    const ctx = try testContext();
    defer ctx.deinit();
    const payload = mlir.Attribute.string(ctx, "<f32>");
    inline for (comptime std.meta.declarations(nvgpu)) |decl| {
        const T = @field(nvgpu, decl.name);
        if (@TypeOf(T) == type) {
            if (@hasDecl(T, "InitArgs")) {
                const has_payload = @hasField(T.InitArgs, "payload");
                const type_ = try T.get(ctx, if (has_payload) .{ .payload = payload } else .{});
                try std.testing.expect(type_.type_().isA(T) != null);
                try std.testing.expect(type_.type_().isA(nvgpu.NVGPUType) != null);
                if (has_payload) try std.testing.expect(type_.getPayload().eql(payload));
                var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
                defer text.deinit();
                try text.writer.print("{f}", .{type_});
                try std.testing.expect(type_.type_().eql(try mlir.Type.parse(ctx, text.written())));
            }
        }
    }
}

test "dotted private tuple builders remain distinct from public builders" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const tuple_type = try algebraType(ctx, .int_tuple, "(2,(3,4))");
    const tuple = cute.static(ctx, tuple_type, loc);
    defer tuple.deinit();

    inline for (.{
        .{ cute.tuple_product, "cute.tuple_product", "24" },
        .{ cute.@"tuple.product", "cute.tuple.product", "24" },
        .{ cute.tuple_product_each, "cute.tuple_product_each", "(2,12)" },
        .{ cute.@"tuple.product_each", "cute.tuple.product_each", "(2,12)" },
    }) |example| {
        const op = example[0](ctx, tuple.result(0), try algebraType(ctx, .int_tuple, example[2]), loc);
        defer op.deinit();
        try std.testing.expectEqualStrings(example[1], op.name());
        try std.testing.expect(op.verify());
    }
}

test "NVGPU builders preserve optional operand segments and attributes" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const gmem = try pointerType(ctx, .float(ctx, .f32), .gmem, 16);
    const dsmem = try pointerType(ctx, .float(ctx, .f32), .dsmem, 16);
    const barrier = try pointerType(ctx, .int(ctx, .i64), .dsmem, 8);
    const block = mlir.Block.init(
        &.{ gmem, dsmem, barrier, .int(ctx, .i16), .int(ctx, .i64) },
        &.{ loc, loc, loc, loc, loc },
    );
    defer block.deinit();
    for ([_]bool{ false, true }) |with_mask| {
        for ([_]bool{ false, true }) |with_cache| {
            const byte_count = mlir.Attribute.int(ctx, .i32, 128);
            const op = nvgpu.arch_copy_SM90_bulk_copy_g2s(ctx, block.argument(0), block.argument(1), block.argument(2), if (with_mask) block.argument(3) else null, if (with_cache) block.argument(4) else null, byte_count, loc);
            defer op.deinit();
            try std.testing.expect(op.verify());
            try std.testing.expect(op.attributeByName("size").?.eql(byte_count));
            try std.testing.expectEqual(@as(usize, 3) + @intFromBool(with_mask) + @intFromBool(with_cache), op.numOperands());
            try std.testing.expect(op.attributeByName("operandSegmentSizes").?.eql(.denseArray(
                ctx,
                .i32,
                &.{ 1, 1, 1, @intFromBool(with_mask), @intFromBool(with_cache) },
            )));
            if (with_cache) try std.testing.expect(op.operand(op.numOperands() - 1).eql(block.argument(4)));
        }
    }
}

test "build public, private and NVGPU operations and round-trip bytecode" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const module = mlir.Module.init(loc);
    defer module.deinit();
    const body = module.body();

    const shape_type = try algebraType(ctx, .shape, "4");
    const stride_type = try algebraType(ctx, .stride, "1");
    const layout_type = try algebraType(ctx, .layout, "4:1");
    const shape = cute.static(ctx, shape_type, loc).appendTo(body);
    const stride = cute.static(ctx, stride_type, loc).appendTo(body);
    const constructed_shape = cute.make_shape(ctx, &.{}, shape_type, loc).appendTo(body);
    try std.testing.expect(constructed_shape.attributeByName("operandSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{0})));
    const leaves = cute.get_leaves(ctx, constructed_shape.result(0), &.{shape_type}, loc).appendTo(body);
    try std.testing.expectEqual(@as(usize, 1), leaves.numResults());
    try std.testing.expect(leaves.attributeByName("resultSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{1})));
    const layout = cute.make_layout(ctx, shape.result(0), stride.result(0), layout_type, loc).appendTo(body);
    try std.testing.expect(layout.result(0).type_().eql(layout_type));

    const memref_type = (try MemRefType.get(ctx, .{
        .ptr = try pointerType(ctx, .float(ctx, .f32), .smem, 16),
        .layout = layout_type,
    })).type_();
    const alloc = cute.memref_alloc_smem(ctx, memref_type, loc).appendTo(body);
    const assumed = cute.assume(ctx, alloc.result(0), memref_type, loc).appendTo(body);
    const atom_type = (try nvgpu.CopyAtomSIMTSyncCopyType.get(ctx, .{ .payload = .string(ctx, "<f32>") })).type_();
    const atom = cute.make_atom(ctx, &.{}, atom_type, loc).appendTo(body);
    const copy_op = cute.copy(ctx, atom.result(0), &.{ alloc.result(0), assumed.result(0) }, &.{assumed.result(0)}, null, loc).appendTo(body);
    const segments = mlir.Attribute.denseArray(ctx, .i32, &.{ 1, 2, 1, 0 });
    try std.testing.expectEqual(@as(usize, 4), copy_op.numOperands());
    try std.testing.expect(copy_op.attributeByName("operandSegmentSizes").?.eql(segments));

    const predicate = mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{.int(ctx, .i1)} },
        .attributes = &.{.named(ctx, "value", .boolean(ctx, true))},
        .location = loc,
    }).appendTo(body);
    const predicated_copy = cute.copy(ctx, atom.result(0), &.{alloc.result(0)}, &.{assumed.result(0)}, predicate.result(0), loc).appendTo(body);
    try std.testing.expect(predicated_copy.attributeByName("operandSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{ 1, 1, 1, 1 })));

    // Operand groups have no fixed per-group capacity in the Zig builder.
    const many_sources = [_]*const mlir.Value{alloc.result(0)} ** 256;
    const large_copy = cute.copy(ctx, atom.result(0), &many_sources, &.{assumed.result(0)}, null, loc).appendTo(body);
    try std.testing.expectEqual(@as(usize, 258), large_copy.numOperands());
    try std.testing.expect(large_copy.attributeByName("operandSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{ 1, 256, 1, 0 })));

    const smem_size = nvgpu.arch_get_dyn_smem_size(ctx, .int(ctx, .i32), loc).appendTo(body);
    try std.testing.expectEqualStrings("cute_nvgpu.arch.get_dyn_smem_size", smem_size.name());
    const pointer = try pointerType(ctx, .float(ctx, .f32), .smem, 16);
    _ = nvgpu.arch_alloc_smem(ctx, pointer, .int(ctx, .i32, 16), loc).appendTo(body);
    try std.testing.expect(module.operation().verify());

    var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    try text.writer.print("{f}", .{module.operation()});
    const parsed = try mlir.Module.parse(ctx, text.written());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());

    var bytes: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer bytes.deinit();
    try module.operation().writeBytecode(null, &bytes.writer);
    try std.testing.expect(std.mem.startsWith(u8, bytes.written(), "ML\xefR"));
    const reader_ctx = try testContext();
    defer reader_ctx.deinit();
    const round_trip = try mlir.Operation.parse(reader_ctx, bytes.written(), "cute-zig.mlirbc");
    defer round_trip.deinit();
    try std.testing.expect(round_trip.verify());
    var round_trip_text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer round_trip_text.deinit();
    try round_trip_text.writer.print("{f}", .{round_trip});
    try std.testing.expectEqualStrings(text.written(), round_trip_text.written());
}

test "SM120 block-scaled MMA preserves fragment groups and optional selectors" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const i32_type = mlir.Type.int(ctx, .i32);
    const f32_type = mlir.Type.float(ctx, .f32);
    const arg_types = [_]*const mlir.Type{i32_type} ** 6 ++
        [_]*const mlir.Type{f32_type} ** 4 ++
        [_]*const mlir.Type{.int(ctx, .i8)} ** 2 ++
        [_]*const mlir.Type{.int(ctx, .i16)} ** 2;
    const block = mlir.Block.init(&arg_types, &([_]*const mlir.Location{loc} ** 14));
    defer block.deinit();
    const shape = try algebraAttribute(ctx, .shape, "(16,8,64)");
    const fp4 = mlir.Attribute.typeAttr(try mlir.Type.parse(ctx, "f4E2M1FN"));
    const scale_type = mlir.Attribute.typeAttr(try mlir.Type.parse(ctx, "f8E8M0FNU"));
    for ([_]bool{ false, true }) |with_a| {
        for ([_]bool{ false, true }) |with_b| {
            const op = nvgpu.arch_mma_SM120_block_scaled(
                ctx,
                &.{ block.argument(0), block.argument(1), block.argument(2), block.argument(3) },
                &.{ block.argument(4), block.argument(5) },
                &.{ block.argument(6), block.argument(7), block.argument(8), block.argument(9) },
                block.argument(10),
                block.argument(11),
                if (with_a) block.argument(12) else null,
                if (with_b) block.argument(13) else null,
                &([_]*const mlir.Type{f32_type} ** 4),
                shape,
                .int(ctx, .i32, 32),
                null,
                null,
                fp4,
                fp4,
                scale_type,
                loc,
            );
            defer op.deinit();
            try std.testing.expect(op.verify());
            try std.testing.expectEqual(@as(usize, 12) + @intFromBool(with_a) + @intFromBool(with_b), op.numOperands());
            for (0..12) |i| try std.testing.expect(op.operand(i).eql(block.argument(i)));
            if (with_a) try std.testing.expect(op.operand(12).eql(block.argument(12)));
            if (with_b) try std.testing.expect(op.operand(op.numOperands() - 1).eql(block.argument(13)));
            try std.testing.expectEqual(@as(usize, 4), op.numResults());
            try std.testing.expect(op.attributeByName("operandSegmentSizes").?.eql(.denseArray(
                ctx,
                .i32,
                &.{ 4, 2, 4, 1, 1, @intFromBool(with_a), @intFromBool(with_b) },
            )));
        }
    }
}

test "private verifier rejects inconsistent operand segments" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const atom_type = (try nvgpu.CopyAtomSIMTSyncCopyType.get(ctx, .{ .payload = .string(ctx, "<f32>") })).type_();
    const atom = cute.make_atom(ctx, &.{}, atom_type, loc);
    defer atom.deinit();
    const copy_op = cute.copy(ctx, atom.result(0), &.{}, &.{}, null, loc);
    defer copy_op.deinit();
    try std.testing.expect(copy_op.verify());
    copy_op.setAttributeByName("operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 0, 0 }));
    try std.testing.expect(!copy_op.verify());
}
