const std = @import("std");

const cf = @import("mlir/dialects/cf");

test {
    std.testing.refAllDecls(cf);
}
