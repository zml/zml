const std = @import("std");

const fly_rocdl = @import("mlir/dialects/fly_rocdl");

test {
    std.testing.refAllDecls(fly_rocdl);
}
