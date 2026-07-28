const std = @import("std");

const c = @import("c");
const pjrt = @import("pjrt");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDAZ");
}

pub fn load(_: std.mem.Allocator, _: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    return error.Unavailable;
}
