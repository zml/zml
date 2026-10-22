const std = @import("std");

const affine = @import("mlir/dialects/affine");

test {
    std.testing.refAllDecls(affine);
}
