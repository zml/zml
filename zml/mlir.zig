const mlir = @This();

const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const dtype = @import("dtype.zig");

const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/mlir");

pub usingnamespace @import("mlir");

pub const ext = struct {
    pub fn mlirType(ctx: mlir.Context, sh: Shape) mlir.Type {
        return mlir.RankedTensorType.init(sh.dims(), mlir.ext.Type.fromDType(ctx, sh.dtype())).asType();
    }

    pub fn denseElementAttrType(dt: dtype.DataType) ?mlir.DenseElementsAttributeTypes {
        return switch (dt) {
            .bool => .bool,
            .i8 => .i8,
            .i16 => .i16,
            .i32 => .i32,
            .i64 => .i64,
            .u8 => .u8,
            .u16 => .u16,
            .u32 => .u32,
            .u64 => .u64,
            .bf16 => .bf16,
            .f16 => .f16,
            .f32 => .f32,
            .f64 => .f64,
            else => null,
        };
    }

    pub fn denseElementsAttr(dt: dtype.DataType, _: usize, bytes: []const u8, ranked_type: mlir.RankedTensorType) mlir.Attribute {
        const ranked_type_ = ranked_type.asType();
        return switch (dt) {
            .bool => mlir.DenseElementsAttribute(.bool).init(ranked_type_, bytes).asAttr(),
            .i8 => mlir.DenseElementsAttribute(.i8).init(ranked_type_, bytes).asAttr(),
            .i16 => mlir.DenseElementsAttribute(.i16).init(ranked_type_, bytes).asAttr(),
            .i32 => mlir.DenseElementsAttribute(.i32).init(ranked_type_, bytes).asAttr(),
            .i64 => mlir.DenseElementsAttribute(.i64).init(ranked_type_, bytes).asAttr(),
            .u8 => mlir.DenseElementsAttribute(.u8).init(ranked_type_, bytes).asAttr(),
            .u16 => mlir.DenseElementsAttribute(.u16).init(ranked_type_, bytes).asAttr(),
            .u32 => mlir.DenseElementsAttribute(.u32).init(ranked_type_, bytes).asAttr(),
            .u64 => mlir.DenseElementsAttribute(.u64).init(ranked_type_, bytes).asAttr(),
            .bf16 => mlir.DenseElementsAttribute(.bf16).init(ranked_type_, bytes).asAttr(),
            .f16 => mlir.DenseElementsAttribute(.f16).init(ranked_type_, bytes).asAttr(),
            .f32 => mlir.DenseElementsAttribute(.f32).init(ranked_type_, bytes).asAttr(),
            .f64 => mlir.DenseElementsAttribute(.f64).init(ranked_type_, bytes).asAttr(),
            inline else => |tag| @panic("Unsupported data type: " ++ @tagName(tag)),
        };
    }

    pub const RankedTensorType = struct {
        pub fn fromShape(ctx: mlir.Context, sh: Shape) mlir.RankedTensorType {
            return mlir.RankedTensorType.init(sh.dims(), mlir.ext.Type.fromDType(ctx, sh.dtype()));
        }
    };

    pub const Type = struct {
        pub fn fromDType(ctx: mlir.Context, dt: dtype.DataType) mlir.Type {
            return switch (dt) {
                .bool => mlir.IntegerType(.i1).init(ctx).asType(),
                .f8e4m3b11fnuz => mlir.FloatType(.f8e4m3b11fnuz).init(ctx).asType(),
                .f8e4m3fn => mlir.FloatType(.f8e4m3fn).init(ctx).asType(),
                .f8e4m3fnuz => mlir.FloatType(.f8e4m3fnuz).init(ctx).asType(),
                .f8e5m2 => mlir.FloatType(.f8e5m2).init(ctx).asType(),
                .f8e5m2fnuz => mlir.FloatType(.f8e5m2fnuz).init(ctx).asType(),
                .bf16 => mlir.FloatType(.bf16).init(ctx).asType(),
                .f16 => mlir.FloatType(.f16).init(ctx).asType(),
                .f32 => mlir.FloatType(.f32).init(ctx).asType(),
                .f64 => mlir.FloatType(.f64).init(ctx).asType(),
                .i4 => mlir.IntegerType(.i4).init(ctx).asType(),
                .i8 => mlir.IntegerType(.i8).init(ctx).asType(),
                .i16 => mlir.IntegerType(.i16).init(ctx).asType(),
                .i32 => mlir.IntegerType(.i32).init(ctx).asType(),
                .i64 => mlir.IntegerType(.i64).init(ctx).asType(),
                .u4 => mlir.IntegerType(.u4).init(ctx).asType(),
                .u8 => mlir.IntegerType(.u8).init(ctx).asType(),
                .u16 => mlir.IntegerType(.u16).init(ctx).asType(),
                .u32 => mlir.IntegerType(.u32).init(ctx).asType(),
                .u64 => mlir.IntegerType(.u64).init(ctx).asType(),
                .c64 => mlir.ComplexType(.c64).init(ctx).asType(),
                .c128 => mlir.ComplexType(.c128).init(ctx).asType(),
            };
        }

        pub fn toDType(mlir_type: mlir.Type) dtype.DataType {
            const mapping = .{
                .{ .bool, mlir.IntegerType(.i1) },

                .{ .f8e4m3b11fnuz, mlir.FloatType(.f8e4m3b11fnuz) },
                .{ .f8e4m3fn, mlir.FloatType(.f8e4m3fn) },
                .{ .f8e4m3fnuz, mlir.FloatType(.f8e4m3fnuz) },
                .{ .f8e5m2, mlir.FloatType(.f8e5m2) },
                .{ .f8e5m2fnuz, mlir.FloatType(.f8e5m2fnuz) },
                .{ .bf16, mlir.FloatType(.bf16) },
                .{ .f16, mlir.FloatType(.f16) },
                .{ .f32, mlir.FloatType(.f32) },
                .{ .f64, mlir.FloatType(.f64) },

                .{ .i4, mlir.IntegerType(.i4) },
                .{ .i8, mlir.IntegerType(.i8) },
                .{ .i16, mlir.IntegerType(.i16) },
                .{ .i32, mlir.IntegerType(.i32) },
                .{ .i64, mlir.IntegerType(.i64) },

                .{ .u4, mlir.IntegerType(.u4) },
                .{ .u8, mlir.IntegerType(.u8) },
                .{ .u16, mlir.IntegerType(.u16) },
                .{ .u32, mlir.IntegerType(.u32) },
                .{ .u64, mlir.IntegerType(.u64) },

                .{ .c64, mlir.ComplexType(.c64) },
                .{ .c128, mlir.ComplexType(.c128) },
            };

            inline for (mapping) |entry| {
                const dt, const mlirT = entry;
                if (mlirT.Methods.is_a_fn.?(mlir_type._inner)) {
                    return dt;
                }
            }

            stdx.debug.panic("Could not convert mlir.Type to DataType: {}", .{mlir_type});
        }
    };

    pub const Attribute = struct {
        pub fn fromData(data: dtype.Data, ctx: mlir.Context) mlir.Attribute {
            switch (data) {
                .bool => |val| {
                    return mlir.IntegerAttribute(.i1).init(ctx, @intFromBool(val)).asAttr();
                },
                inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz => |val, tag| {
                    const float_type = @field(mlir.FloatTypes, @tagName(tag));
                    const float_attr = mlir.FloatAttribute(float_type).init(ctx, val.toF32());
                    return float_attr.asAttr();
                },
                inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |val, tag| {
                    const int_type = @field(mlir.IntegerTypes, @tagName(tag));
                    const int_attr = mlir.IntegerAttribute(int_type).init(ctx, @intCast(val));
                    return int_attr.asAttr();
                },
                inline else => |_, tag| stdx.debug.panic("Unsupported data type: {any}", .{tag}),
            }
        }
    };

    pub const DenseElementsAttribute = struct {
        pub fn fromData(data: dtype.Data, result_type: mlir.Type) mlir.Attribute {
            return switch (data.dtype()) {
                .bool => mlir.DenseElementsAttribute(.bool).init(result_type, data.constSlice()).asAttr(),
                .i8 => mlir.DenseElementsAttribute(.i8).init(result_type, data.constSlice()).asAttr(),
                .i16 => mlir.DenseElementsAttribute(.i16).init(result_type, data.constSlice()).asAttr(),
                .i32 => mlir.DenseElementsAttribute(.i32).init(result_type, data.constSlice()).asAttr(),
                .i64 => mlir.DenseElementsAttribute(.i64).init(result_type, data.constSlice()).asAttr(),
                .u8 => mlir.DenseElementsAttribute(.u8).init(result_type, data.constSlice()).asAttr(),
                .u16 => mlir.DenseElementsAttribute(.u16).init(result_type, data.constSlice()).asAttr(),
                .u32 => mlir.DenseElementsAttribute(.u32).init(result_type, data.constSlice()).asAttr(),
                .u64 => mlir.DenseElementsAttribute(.u64).init(result_type, data.constSlice()).asAttr(),
                .bf16 => mlir.DenseElementsAttribute(.bf16).init(result_type, data.constSlice()).asAttr(),
                .f16 => mlir.DenseElementsAttribute(.f16).init(result_type, data.constSlice()).asAttr(),
                .f32 => mlir.DenseElementsAttribute(.f32).init(result_type, data.constSlice()).asAttr(),
                .f64 => mlir.DenseElementsAttribute(.f64).init(result_type, data.constSlice()).asAttr(),
                inline else => |tag| stdx.debug.panic("Unsupported data type: {any}", .{tag}),
            };
        }
    };
};
