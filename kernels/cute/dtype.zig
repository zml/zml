const std = @import("std");
const mlir = @import("mlir");

/// Element types a `!cute.ptr` can point at.
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

pub fn mlirToDType(ctx: *mlir.Context, t: *const mlir.Type) DType {
    inline for (std.meta.fields(DType)) |f| {
        const dt = @field(DType, f.name);
        if (t.eql(dt.toMlir(ctx))) return dt;
    }
    std.debug.panic("type {f} is not a CuTe DType", .{t});
}

test {
    std.testing.refAllDecls(@This());
}
