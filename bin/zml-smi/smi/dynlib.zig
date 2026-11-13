const std = @import("std");
const builtin = @import("builtin");
const sandbox = @import("sandbox.zig");

const log = std.log.scoped(.@"zml-smi/dynlib");

const flags: std.c.RTLD = .{ .LAZY = true, .GLOBAL = true, .NODELETE = true };

pub fn open(comptime Api: type, names: []const [:0]const u8) !Api {
    for (names) |name| {
        const handle = std.c.dlopen(name, flags) orelse continue;
        return resolve(Api, handle, name);
    }
    log.warn("{s}", .{lastError()});
    return error.LibraryNotFound;
}

pub fn openBundled(comptime Api: type, io: std.Io, name: []const u8) !Api {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const directory = sandbox.path(io, &buffer) orelse {
        log.warn("{s}: no sandbox directory", .{name});
        return error.LibraryNotFound;
    };
    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buffer, "{s}/lib/{s}", .{ directory, name }, 0);
    const handle = std.c.dlopen(path, flags) orelse {
        log.warn("{s}", .{lastError()});
        return error.LibraryNotFound;
    };
    return resolve(Api, handle, name);
}

fn lastError() []const u8 {
    return std.mem.span(std.c.dlerror() orelse return "no library to open");
}

fn resolve(comptime Api: type, handle: *anyopaque, library: []const u8) !Api {
    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, Field| {
        const symbol = std.c.dlsym(handle, name) orelse {
            log.warn("{s} has no symbol {s}", .{ library, name });
            return error.MissingSymbol;
        };
        @field(api, name) = if (@typeInfo(Field) == .@"struct") .{ .function = @ptrCast(@alignCast(symbol)) } else @ptrCast(@alignCast(symbol));
    }
    return api;
}

pub fn Checked(comptime function: anytype, comptime check: anytype) type {
    const Function = @TypeOf(function);
    const Error = @typeInfo(@typeInfo(@TypeOf(check)).@"fn".return_type.?).error_union.error_set;

    return struct {
        function: *const Function,

        const params = @typeInfo(Function).@"fn".param_types;
        const Result = @typeInfo(params[params.len - 1].?).pointer.child;
        const Inputs = inputs: {
            var types: [params.len - 1]type = undefined;
            for (&types, params[0 .. params.len - 1]) |*T, param| T.* = param.?;
            const final = types;
            break :inputs @Tuple(&final);
        };

        pub fn call(self: @This(), args: std.meta.ArgsTuple(Function)) Error!void {
            return check(@call(.auto, self.function, args));
        }

        pub fn get(self: @This(), args: Inputs) Error!Result {
            var result: Result = std.mem.zeroes(Result);
            try check(@call(.auto, self.function, args ++ .{&result}));
            return result;
        }
    };
}

const Libc = struct {
    getpid: *const @TypeOf(std.c.getpid),
    clock_gettime: Checked(std.c.clock_gettime, checkLibc),
    close: Checked(std.c.close, checkLibc),

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
