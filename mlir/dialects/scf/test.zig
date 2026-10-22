const std = @import("std");

const scf = @import("mlir/dialects/scf");

test {
    std.testing.refAllDecls(scf);
}
