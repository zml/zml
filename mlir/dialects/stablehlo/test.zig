const std = @import("std");

const stablehlo = @import("mlir/dialects/stablehlo");

test {
    std.testing.refAllDecls(stablehlo);
}
