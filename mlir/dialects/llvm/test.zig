const std = @import("std");

const llvm = @import("mlir/dialects/llvm");

test {
    std.testing.refAllDecls(llvm);
}
