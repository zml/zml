const std = @import("std");
const builtin = @import("builtin");
const dynlib = @import("zml-smi/native").dynlib;

const open = dynlib.open;
const openBundled = dynlib.openBundled;

const Libc = struct {
    getpid: *const @TypeOf(std.c.getpid),
    clock_gettime: Checked(std.c.clock_gettime),
    close: Checked(std.c.close),

    const Checked = dynlib.CheckedBy(checkLibc);

    fn checkLibc(status: c_int) error{Failed}!void {
        if (status != 0) return error.Failed;
    }
};

const libc_names: []const [:0]const u8 = switch (builtin.os.tag) {
    .macos => &.{"/usr/lib/libSystem.B.dylib"},
    else => &.{ "libc.so.6", "libc.so" },
};

test open {
    const libc = try open(Libc, libc_names);
    try std.testing.expectEqual(std.c.getpid(), libc.getpid());
    try std.testing.expect((try libc.clock_gettime.get(.{.MONOTONIC})).sec > 0);
    try std.testing.expectError(error.Failed, libc.close.call(.{-1}));
    try std.testing.expectError(error.LibraryNotFound, open(Libc, &.{"libmissing.so"}));
}

test openBundled {
    try std.testing.expectError(error.LibraryNotFound, openBundled(Libc, std.testing.io, "libmissing.so"));
}
