const std = @import("std");
const mlir = @import("mlir");

pub const DType = enum {
    i1,
    i8,
    i16,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,
    f8e4m3fn,
    f8e5m2,

    pub fn toMlir(self: DType, ctx: *mlir.Context) *const mlir.Type {
        return switch (self) {
            .i1 => .int(ctx, .i1),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .f16 => .float(ctx, .f16),
            .bf16 => .float(ctx, .bf16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .f8e4m3fn => .float(ctx, .f8e4m3fn),
            .f8e5m2 => .float(ctx, .f8e5m2),
        };
    }
};

pub fn isFloatDtype(dt: DType) bool {
    return switch (dt) {
        .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => true,
        else => false,
    };
}

pub fn dtypeBitwidth(dt: DType) usize {
    return switch (dt) {
        .i1 => 1,
        .i8, .f8e4m3fn, .f8e5m2 => 8,
        .i16, .f16, .bf16 => 16,
        .i32, .f32 => 32,
        .i64, .f64 => 64,
    };
}

pub fn intBitwidth(dt: DType) u32 {
    return switch (dt) {
        .i1 => 1,
        .i8 => 8,
        .i16 => 16,
        .i32 => 32,
        .i64 => 64,
        else => std.debug.panic("intBitwidth: not an integer DType: {s}", .{@tagName(dt)}),
    };
}

pub fn mlirElemToDType(ctx: *mlir.Context, elem: *const mlir.Type) DType {
    inline for (std.meta.fields(DType)) |f| {
        const dt = @field(DType, f.name);
        if (elem.eql(dt.toMlir(ctx))) return dt;
    }
    @panic("element type not a recognized Mosaic-TPU DType");
}

test {
    std.testing.refAllDecls(@This());
}
