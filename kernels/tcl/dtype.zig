const std = @import("std");

const mlir = @import("mlir");

/// TCL element types, named as in TCL (`[A]/f8_e4`).
pub const DType = enum {
    bool,
    i4,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f16,
    bf16,
    f32,
    f64,
    f4_e2,
    f8_e4,
    f8_e5,

    pub fn toMlir(self: DType, ctx: *mlir.Context) *const mlir.Type {
        return switch (self) {
            .bool => .int(ctx, .i1),
            .i4 => .int(ctx, .i4),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .u8 => .int(ctx, .u8),
            .u16 => .int(ctx, .u16),
            .u32 => .int(ctx, .u32),
            .u64 => .int(ctx, .u64),
            .f16 => .float(ctx, .f16),
            .bf16 => .float(ctx, .bf16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .f4_e2 => .float(ctx, .f4e2m1fn),
            .f8_e4 => .float(ctx, .f8e4m3fn),
            .f8_e5 => .float(ctx, .f8e5m2),
        };
    }

    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f16, .bf16, .f32, .f64, .f4_e2, .f8_e4, .f8_e5 => true,
            else => false,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
