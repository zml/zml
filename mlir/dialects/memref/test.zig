const std = @import("std");

const memref = @import("mlir/dialects/memref");

test {
    std.testing.refAllDecls(memref);
}
