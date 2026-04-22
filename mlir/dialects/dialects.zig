const std = @import("std");

pub const stablehlo = @import("mlir/dialects/stablehlo");
pub const ttir = @import("mlir/dialects/ttir");

pub const arith = @import("arith.zig");
pub const func = @import("func.zig");
pub const math = @import("math.zig");
pub const scf = @import("scf.zig");

test {
    std.testing.refAllDecls(@This());
}
