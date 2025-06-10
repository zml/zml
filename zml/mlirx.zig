const std = @import("std");

const mlir = @import("mlir");

const dtype = @import("dtype.zig");
const Shape = @import("shape.zig").Shape;

const mlirx = @This();

/// Returns the mlir.Type corresponding to a given zml.Shape.
pub fn tensorType(ctx: mlir.Context, sh: Shape) mlir.Type {
    return .tensor(sh.dims(), mlirx.Type.fromDType(ctx, sh.dtype()));
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

pub const Type = struct {
    pub fn fromDType(ctx: mlir.Context, dt: dtype.DataType) mlir.Type {
        return switch (dt) {
            .bool => .int(ctx, .i1),
            .f8e4m3b11fnuz => .float(ctx, .f8e4m3b11fnuz),
            .f8e4m3fn => .float(ctx, .f8e4m3fn),
            .f8e4m3fnuz => .float(ctx, .f8e4m3fnuz),
            .f8e5m2 => .float(ctx, .f8e5m2),
            .f8e5m2fnuz => .float(ctx, .f8e5m2fnuz),
            .bf16 => .float(ctx, .bf16),
            .f16 => .float(ctx, .f16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .i4 => .int(ctx, .i4),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .u4 => .int(ctx, .u4),
            .u8 => .int(ctx, .u8),
            .u16 => .int(ctx, .u16),
            .u32 => .int(ctx, .u32),
            .u64 => .int(ctx, .u64),
            .c64 => .complex(ctx, .c64),
            .c128 => .complex(ctx, .c128),
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
            if (mlirT.is_a_fn(mlir_type._inner)) {
                return dt;
            }
        }

        std.debug.panic("Could not convert mlir.Type to DataType: {}", .{mlir_type});
    }
};
