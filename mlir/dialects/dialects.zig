const std = @import("std");

pub const cuda_tile = @import("mlir/dialects/cuda_tile");
pub const fly = @import("mlir/dialects/fly");
pub const mosaic_tpu = @import("mlir/dialects/mosaic_tpu");
pub const stablehlo = @import("mlir/dialects/stablehlo");
pub const ttir = @import("mlir/dialects/ttir");

pub const affine = @import("affine.zig");
pub const arith = @import("arith.zig");
pub const cf = @import("cf.zig");
pub const func = @import("func.zig");
pub const gpu = @import("gpu.zig");
pub const math = @import("math.zig");
pub const memref = @import("memref.zig");
pub const scf = @import("scf.zig");
pub const shardy = @import("shardy.zig");
pub const vector = @import("vector.zig");

test {
    std.testing.refAllDecls(@This());
}
