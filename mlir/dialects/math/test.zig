const std = @import("std");

const math = @import("mlir/dialects/math");

test {
    std.testing.refAllDecls(math);
}
