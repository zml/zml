pub const fp8 = @import("quant/fp8.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
