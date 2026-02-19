const std = @import("std");

pub const stablehlo = @import("mlir/dialects/stablehlo");
pub const shardy = @import("mlir/dialects/shardy");
pub const func = @import("func.zig");

test {
    std.testing.refAllDecls(@This());
}
