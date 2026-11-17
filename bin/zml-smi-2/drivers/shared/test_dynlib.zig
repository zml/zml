const std = @import("std");
const builtin = @import("builtin");
const dynlib = @import("zml-smi-2/native").dynlib;

const open = dynlib.open;
const openBundled = dynlib.openBundled;

const Libc = struct {
    getpid: *const @TypeOf(std.c.getpid),
    clock_gettime: dynlib.Checked(checkLibc, @TypeOf(std.c.clock_gettime)),
    close: dynlib.Checked(checkLibc, @TypeOf(std.c.close)),

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

test "one scalar symbol supplies several metrics with independent results" {
    const Driver = struct {
        pub fn measure(handle: u32, selector: i32, result: *i32) callconv(.c) c_int {
            if (selector == 2) return -1;
            result.* = @as(i32, @intCast(handle)) + selector;
            return 0;
        }
    };
    const bindings = .{ .measure = .{
        .zero = .{@as(i32, -4)},
        .unsupported = .{@as(i32, 2)},
        .value = .{@as(i32, 3)},
        .negative = .{@as(i32, -5)},
    } };
    const Api = dynlib.Bindings(Driver, Libc.checkLibc, std.meta.fieldNames(@TypeOf(bindings)));
    const Probe = struct {
        api: *const Api,
        handle: u32,

        const scalars = dynlib.Scalars(@This(), bindings).read;
    };
    const api: Api = .{ .measure = .{ .function = Driver.measure } };
    const probe: Probe = .{ .api = &api, .handle = 4 };
    const readings = probe.scalars();
    try std.testing.expectEqual(@as(?u64, 0), readings.zero);
    try std.testing.expectEqual(@as(?u64, null), readings.unsupported);
    try std.testing.expectEqual(@as(?u64, 7), readings.value);
    try std.testing.expectEqual(@as(?u64, null), readings.negative);
}
