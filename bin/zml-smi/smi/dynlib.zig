const std = @import("std");
const sandbox = @import("sandbox.zig");

const log = std.log.scoped(.@"zml-smi/dynlib");

const flags: std.c.RTLD = .{ .LAZY = true, .GLOBAL = true, .NODELETE = true };

/// Opens the first system library that loads and resolves every field of `Api` by its name.
/// A field is a `Checked`, a function pointer, or a pointer to a data symbol.
pub fn open(comptime Api: type, names: []const [:0]const u8) !Api {
    const library, const handle = for (names) |name| {
        if (std.c.dlopen(name, flags)) |handle| break .{ name, handle };
    } else {
        const message = std.c.dlerror() orelse "no library to open";
        log.warn("{s}", .{std.mem.span(message)});
        return error.LibraryNotFound;
    };
    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, Field| {
        const symbol = std.c.dlsym(handle, name) orelse {
            log.warn("{s} has no symbol {s}", .{ library, name });
            return error.MissingSymbol;
        };
        @field(api, name) = if (@typeInfo(Field) == .@"struct")
            .{ .function = @ptrCast(@alignCast(symbol)) }
        else
            @ptrCast(@alignCast(symbol));
    }
    return api;
}

/// Opens a library shipped with zml-smi, from its sandbox directory.
pub fn openBundled(comptime Api: type, io: std.Io, name: []const u8) !Api {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const directory = sandbox.path(io, &buffer) orelse {
        log.warn("{s}: no sandbox directory", .{name});
        return error.LibraryNotFound;
    };
    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buffer, "{s}/lib/{s}", .{ directory, name }, 0);
    return open(Api, &.{path});
}

/// A C function that reports failure through its return value, turned into an error by `check`.
pub fn Checked(comptime function: anytype, comptime check: anytype) type {
    const Function = if (@TypeOf(function) == type) function else @TypeOf(function);
    const Error = @typeInfo(@typeInfo(@TypeOf(check)).@"fn".return_type.?).error_union.error_set;

    return struct {
        function: *const Function,

        const Args = std.meta.ArgsTuple(Function);
        const params = @typeInfo(Args).@"struct".field_types;
        const Result = @typeInfo(params[params.len - 1]).pointer.child;
        const Inputs = @Tuple(params[0 .. params.len - 1]);

        pub fn call(self: @This(), args: Args) Error!void {
            return check(@call(.auto, self.function, args));
        }

        /// For a function whose only result is its last parameter: returns that result.
        pub fn get(self: @This(), args: Inputs) Error!Result {
            var result: Result = std.mem.zeroes(Result);
            try self.call(args ++ .{&result});
            return result;
        }
    };
}
