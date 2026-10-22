// Generated from LLVM ROCDL TableGen records; see REGENERATING.md.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const mlir = @import("mlir");
const llvm = @import("mlir/dialects/llvm");

/// ROCDL address spaces use LLVM pointer types.
pub const AddressSpace = enum(u32) {
    generic = 0,
    global = 1,
    shared = 3,
    constant = 4,
    private = 5,
    buffer_fat_pointer = 7,
    buffer_resource = 8,
    buffer_strided_pointer = 9,
    barrier = 15,
};

pub fn pointerType(ctx: *mlir.Context, space: AddressSpace) *const mlir.Type {
    return llvm.pointerType(ctx, space);
}

/// `vector<16xbf16>`, as used by ROCDL operations.
pub fn v16bf16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{16}, .float(ctx, .bf16));
}

/// `vector<16xf16>`, as used by ROCDL operations.
pub fn v16f16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{16}, .float(ctx, .f16));
}

/// `vector<16xf32>`, as used by ROCDL operations.
pub fn v16f32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{16}, .float(ctx, .f32));
}

/// `vector<16xi32>`, as used by ROCDL operations.
pub fn v16i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{16}, .int(ctx, .i32));
}

/// `vector<2xbf16>`, as used by ROCDL operations.
pub fn v2bf16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{2}, .float(ctx, .bf16));
}

/// `vector<2xf16>`, as used by ROCDL operations.
pub fn v2f16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{2}, .float(ctx, .f16));
}

/// `vector<2xf32>`, as used by ROCDL operations.
pub fn v2f32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{2}, .float(ctx, .f32));
}

/// `vector<2xi16>`, as used by ROCDL operations.
pub fn v2i16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{2}, .int(ctx, .i16));
}

/// `vector<2xi32>`, as used by ROCDL operations.
pub fn v2i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{2}, .int(ctx, .i32));
}

/// `vector<32xbf16>`, as used by ROCDL operations.
pub fn v32bf16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{32}, .float(ctx, .bf16));
}

/// `vector<32xf16>`, as used by ROCDL operations.
pub fn v32f16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{32}, .float(ctx, .f16));
}

/// `vector<32xf32>`, as used by ROCDL operations.
pub fn v32f32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{32}, .float(ctx, .f32));
}

/// `vector<32xi32>`, as used by ROCDL operations.
pub fn v32i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{32}, .int(ctx, .i32));
}

/// `vector<3xi32>`, as used by ROCDL operations.
pub fn v3i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{3}, .int(ctx, .i32));
}

/// `vector<4xf16>`, as used by ROCDL operations.
pub fn v4f16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{4}, .float(ctx, .f16));
}

/// `vector<4xf32>`, as used by ROCDL operations.
pub fn v4f32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{4}, .float(ctx, .f32));
}

/// `vector<4xf64>`, as used by ROCDL operations.
pub fn v4f64(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{4}, .float(ctx, .f64));
}

/// `vector<4xi16>`, as used by ROCDL operations.
pub fn v4i16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{4}, .int(ctx, .i16));
}

/// `vector<4xi32>`, as used by ROCDL operations.
pub fn v4i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{4}, .int(ctx, .i32));
}

/// `vector<6xi32>`, as used by ROCDL operations.
pub fn v6i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{6}, .int(ctx, .i32));
}

/// `vector<8xbf16>`, as used by ROCDL operations.
pub fn v8bf16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{8}, .float(ctx, .bf16));
}

/// `vector<8xf16>`, as used by ROCDL operations.
pub fn v8f16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{8}, .float(ctx, .f16));
}

/// `vector<8xf32>`, as used by ROCDL operations.
pub fn v8f32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{8}, .float(ctx, .f32));
}

/// `vector<8xi16>`, as used by ROCDL operations.
pub fn v8i16(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{8}, .int(ctx, .i16));
}

/// `vector<8xi32>`, as used by ROCDL operations.
pub fn v8i32(ctx: *mlir.Context) *const mlir.Type {
    return .vector(&.{8}, .int(ctx, .i32));
}
