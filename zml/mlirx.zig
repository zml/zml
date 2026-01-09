const std = @import("std");

const mlir = @import("mlir");

const dtype = @import("dtype.zig");
const Shape = @import("shape.zig").Shape;

const mlirx = @This();

pub const Type = struct {
    pub fn fromDType(ctx: *mlir.Context, dt: dtype.DataType) *const mlir.Type {
        return switch (dt) {
            .bool => mlir.integerType(ctx, .i1),
            .f4e2m1 => mlir.floatType(ctx, .f4e2m1fn),
            .f8e3m4 => mlir.floatType(ctx, .f8e3m4),
            .f8e4m3 => mlir.floatType(ctx, .f8e4m3),
            .f8e4m3fn => mlir.floatType(ctx, .f8e4m3fn),
            .f8e4m3fnuz => mlir.floatType(ctx, .f8e4m3fnuz),
            .f8e4m3b11fnuz => mlir.floatType(ctx, .f8e4m3b11fnuz),
            .f8e5m2 => mlir.floatType(ctx, .f8e5m2),
            .f8e5m2fnuz => mlir.floatType(ctx, .f8e5m2fnuz),
            .f8e8m0 => mlir.floatType(ctx, .f8e8m0fnu),
            .bf16 => mlir.floatType(ctx, .bf16),
            .f16 => mlir.floatType(ctx, .f16),
            .f32 => mlir.floatType(ctx, .f32),
            .f64 => mlir.floatType(ctx, .f64),
            .i2 => mlir.integerType(ctx, .i2),
            .i4 => mlir.integerType(ctx, .i4),
            .i8 => mlir.integerType(ctx, .i8),
            .i16 => mlir.integerType(ctx, .i16),
            .i32 => mlir.integerType(ctx, .i32),
            .i64 => mlir.integerType(ctx, .i64),
            .u2 => mlir.integerType(ctx, .u2),
            .u4 => mlir.integerType(ctx, .u4),
            .u8 => mlir.integerType(ctx, .u8),
            .u16 => mlir.integerType(ctx, .u16),
            .u32 => mlir.integerType(ctx, .u32),
            .u64 => mlir.integerType(ctx, .u64),
            .c64 => mlir.complexType(ctx, .c64),
            .c128 => mlir.complexType(ctx, .c128),
        };
    }

    // TODO(Corentin): Maybe remove that as it now requires a mlir context.
    // It's weird to have to provide a mlir context to get a zml.DataType from a mlir.Type.
    pub fn toDType(ctx: *mlir.Context, mlir_type: *const mlir.Type) dtype.DataType {
        const mapping = .{
            .{ .bool, mlir.integerType(ctx, .i1) },

            .{ .f8e4m3b11fnuz, mlir.floatType(ctx, .f8e4m3b11fnuz) },
            .{ .f8e4m3fn, mlir.floatType(ctx, .f8e4m3fn) },
            .{ .f8e4m3fnuz, mlir.floatType(ctx, .f8e4m3fnuz) },
            .{ .f8e3m4, mlir.floatType(ctx, .f8e3m4) },
            .{ .f8e5m2, mlir.floatType(ctx, .f8e5m2) },
            .{ .f8e5m2fnuz, mlir.floatType(ctx, .f8e5m2fnuz) },
            .{ .f8e8m0, mlir.floatType(ctx, .f8e8m0fnu) },
            .{ .f4e2m1, mlir.floatType(ctx, .f4e2m1fn) },
            .{ .bf16, mlir.floatType(ctx, .bf16) },
            .{ .f16, mlir.floatType(ctx, .f16) },
            .{ .f32, mlir.floatType(ctx, .f32) },
            .{ .f64, mlir.floatType(ctx, .f64) },

            .{ .i4, mlir.integerType(ctx, .i4) },
            .{ .i8, mlir.integerType(ctx, .i8) },
            .{ .i16, mlir.integerType(ctx, .i16) },
            .{ .i32, mlir.integerType(ctx, .i32) },
            .{ .i64, mlir.integerType(ctx, .i64) },

            .{ .u4, mlir.integerType(ctx, .u4) },
            .{ .u8, mlir.integerType(ctx, .u8) },
            .{ .u16, mlir.integerType(ctx, .u16) },
            .{ .u32, mlir.integerType(ctx, .u32) },
            .{ .u64, mlir.integerType(ctx, .u64) },

            .{ .c64, mlir.complexType(ctx, .c64) },
            .{ .c128, mlir.complexType(ctx, .c128) },
        };

        inline for (mapping) |entry| {
            const dt, const mlirT = entry;
            if (mlirT.eql(mlir_type)) {
                return dt;
            }
        }

        std.debug.panic("Could not convert mlir.Type to DataType: {f}", .{mlir_type});
    }
};
