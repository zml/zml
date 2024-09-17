const mlir = @This();

const builtin = @import("builtin");
const std = @import("std");

const dtype = @import("dtype.zig");
const meta = @import("meta.zig");

const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.zml_mlir);

pub usingnamespace @import("mlir");

pub const ext = struct {
    pub fn mlirType(ctx: mlir.Context, sh: Shape) mlir.Type {
        return mlir.RankedTensorType.init(sh.dims(), mlir.ext.Type.fromDType(ctx, sh.dtype())).as(mlir.Type).?;
    }

    pub fn denseElementAttrType(dt: dtype.DataType) mlir.DenseElementsAttributeTypes {
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
            inline else => |tag| @panic("Unsupported data type: " ++ @tagName(tag)),
        };
    }

    pub fn denseElementsAttr(dt: dtype.DataType, _: usize, bytes: []const u8, ranked_type: mlir.RankedTensorType) mlir.Attribute {
        const ranked_type_ = ranked_type.as(mlir.Type).?;
        return switch (dt) {
            .bool => mlir.DenseIntOrFPElementsAttribute(.bool).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .i8 => mlir.DenseIntOrFPElementsAttribute(.i8).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .i16 => mlir.DenseIntOrFPElementsAttribute(.i16).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .i32 => mlir.DenseIntOrFPElementsAttribute(.i32).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .i64 => mlir.DenseIntOrFPElementsAttribute(.i64).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .u8 => mlir.DenseIntOrFPElementsAttribute(.u8).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .u16 => mlir.DenseIntOrFPElementsAttribute(.u16).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .u32 => mlir.DenseIntOrFPElementsAttribute(.u32).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .u64 => mlir.DenseIntOrFPElementsAttribute(.u64).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .bf16 => mlir.DenseIntOrFPElementsAttribute(.bf16).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .f16 => mlir.DenseIntOrFPElementsAttribute(.f16).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .f32 => mlir.DenseIntOrFPElementsAttribute(.f32).init(ranked_type_, bytes).as(mlir.Attribute).?,
            .f64 => mlir.DenseIntOrFPElementsAttribute(.f64).init(ranked_type_, bytes).as(mlir.Attribute).?,
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
                .bool => mlir.IntegerType(.i1).init(ctx).as(mlir.Type).?,
                .f8e4m3b11fnuz => mlir.FloatType(.f8e4m3b11fnuz).init(ctx).as(mlir.Type).?,
                .f8e4m3fn => mlir.FloatType(.f8e4m3fn).init(ctx).as(mlir.Type).?,
                .f8e4m3fnuz => mlir.FloatType(.f8e4m3fnuz).init(ctx).as(mlir.Type).?,
                .f8e5m2 => mlir.FloatType(.f8e5m2).init(ctx).as(mlir.Type).?,
                .f8e5m2fnuz => mlir.FloatType(.f8e5m2fnuz).init(ctx).as(mlir.Type).?,
                .bf16 => mlir.FloatType(.bf16).init(ctx).as(mlir.Type).?,
                .f16 => mlir.FloatType(.f16).init(ctx).as(mlir.Type).?,
                .f32 => mlir.FloatType(.f32).init(ctx).as(mlir.Type).?,
                .f64 => mlir.FloatType(.f64).init(ctx).as(mlir.Type).?,
                .i4 => mlir.IntegerType(.i4).init(ctx).as(mlir.Type).?,
                .i8 => mlir.IntegerType(.i8).init(ctx).as(mlir.Type).?,
                .i16 => mlir.IntegerType(.i16).init(ctx).as(mlir.Type).?,
                .i32 => mlir.IntegerType(.i32).init(ctx).as(mlir.Type).?,
                .i64 => mlir.IntegerType(.i64).init(ctx).as(mlir.Type).?,
                .u4 => mlir.IntegerType(.u4).init(ctx).as(mlir.Type).?,
                .u8 => mlir.IntegerType(.u8).init(ctx).as(mlir.Type).?,
                .u16 => mlir.IntegerType(.u16).init(ctx).as(mlir.Type).?,
                .u32 => mlir.IntegerType(.u32).init(ctx).as(mlir.Type).?,
                .u64 => mlir.IntegerType(.u64).init(ctx).as(mlir.Type).?,
                .c64 => mlir.ComplexType(.c64).init(ctx).as(mlir.Type).?,
                .c128 => mlir.ComplexType(.c128).init(ctx).as(mlir.Type).?,
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
                if (mlir_type.as(mlirT)) |_| {
                    return dt;
                }
            }

            meta.panic("Could not convert mlir.Type to DataType: {}", .{mlir_type});
        }
    };

    pub const Attribute = struct {
        pub fn fromData(data: dtype.Data, ctx: mlir.Context) mlir.Attribute {
            switch (data) {
                .bool => |val| {
                    return mlir.IntegerAttribute(.i1).init(ctx, @intFromBool(val)).as(mlir.Attribute).?;
                },
                inline .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz => |val, tag| {
                    const float_type = @field(mlir.FloatTypes, @tagName(tag));
                    const float_attr = mlir.FloatAttribute(float_type).init(ctx, val.toF32());
                    return float_attr.as(mlir.Attribute).?;
                },
                inline .i4, .i8, .i16, .i32, .i64, .u4, .u8, .u16, .u32, .u64 => |val, tag| {
                    const int_type = @field(mlir.IntegerTypes, @tagName(tag));
                    const int_attr = mlir.IntegerAttribute(int_type).init(ctx, @intCast(val));
                    return int_attr.as(mlir.Attribute).?;
                },
                inline else => |_, tag| meta.panic("Unsupported data type: {any}", .{tag}),
            }
        }
    };

    pub const DenseIntOrFPElementsAttribute = struct {
        pub fn fromData(data: dtype.Data, result_type: mlir.Type) mlir.Attribute {
            return switch (data.dataType()) {
                .bool => mlir.DenseIntOrFPElementsAttribute(.bool).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .i8 => mlir.DenseIntOrFPElementsAttribute(.i8).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .i16 => mlir.DenseIntOrFPElementsAttribute(.i16).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .i32 => mlir.DenseIntOrFPElementsAttribute(.i32).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .i64 => mlir.DenseIntOrFPElementsAttribute(.i64).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .u8 => mlir.DenseIntOrFPElementsAttribute(.u8).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .u16 => mlir.DenseIntOrFPElementsAttribute(.u16).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .u32 => mlir.DenseIntOrFPElementsAttribute(.u32).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .u64 => mlir.DenseIntOrFPElementsAttribute(.u64).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .bf16 => mlir.DenseIntOrFPElementsAttribute(.bf16).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .f16 => mlir.DenseIntOrFPElementsAttribute(.f16).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .f32 => mlir.DenseIntOrFPElementsAttribute(.f32).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                .f64 => mlir.DenseIntOrFPElementsAttribute(.f64).init(result_type, data.constSlice()).as(mlir.Attribute).?,
                inline else => |tag| meta.panic("Unsupported data type: {any}", .{tag}),
            };
        }
    };
};
