const std = @import("std");

pub const stablehlo = @import("mlir/dialects/stablehlo");

pub const func = @import("func.zig");

// pub const arith = @import("arith.zig");
// pub const math = @import("math.zig");
// pub const scf = @import("scf.zig");
// pub const tensor = @import("tensor.zig");
test {
    std.testing.refAllDecls(@This());
}
