pub const profiler = @import("profiler.zig");
pub const tracer = @import("tracer.zig");

test {
    @import("std").testing.refAllDecls(@This());
}

comptime {
    // Forces zig to compile `@zml//bazel` module even when users don't depend on it even indirectly.
    // Otherwise exported functions declared in `bazel` aren't compiled.
    @import("std").testing.refAllDecls(@import("bazel"));
}
