const std = @import("std");

pub const stablehlo = @import("mlir/dialects/stablehlo");

pub const func = @import("func.zig");

test {
    std.testing.refAllDecls(@This());
}
