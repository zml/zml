const std = @import("std");

const mlir = @import("mlir");

const dtype = @import("dtype.zig");
const Shape = @import("shape.zig").Shape;

const mlirx = @This();

pub const Type = struct {
    pub fn rankedTensor(ctx: *mlir.Context, shape: Shape) *const mlir.Type {
        return .rankedTensor(shape.dims(), mlirx.Type.fromDType(ctx, shape.dtype()));
    }

    pub fn fromDType(ctx: *mlir.Context, dt: dtype.DataType) *const mlir.Type {
        return switch (dt) {
            .bool => .int(ctx, .i1),
            .f4e2m1 => .float(ctx, .f4e2m1fn),
            .f8e3m4 => .float(ctx, .f8e3m4),
            .f8e4m3 => .float(ctx, .f8e4m3),
            .f8e4m3fn => .float(ctx, .f8e4m3fn),
            .f8e4m3fnuz => .float(ctx, .f8e4m3fnuz),
            .f8e4m3b11fnuz => .float(ctx, .f8e4m3b11fnuz),
            .f8e5m2 => .float(ctx, .f8e5m2),
            .f8e5m2fnuz => .float(ctx, .f8e5m2fnuz),
            .f8e8m0 => .float(ctx, .f8e8m0fnu),
            .bf16 => .float(ctx, .bf16),
            .f16 => .float(ctx, .f16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .i2 => .int(ctx, .i2),
            .i4 => .int(ctx, .i4),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .u2 => .int(ctx, .u2),
            .u4 => .int(ctx, .u4),
            .u8 => .int(ctx, .u8),
            .u16 => .int(ctx, .u16),
            .u32 => .int(ctx, .u32),
            .u64 => .int(ctx, .u64),
            .c64 => .complex(ctx, .c64),
            .c128 => .complex(ctx, .c128),
        };
    }
};
