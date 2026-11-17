const std = @import("std");
const sandbox = @import("sandbox.zig");

const log = std.log.scoped(.@"zml-smi-2/dynlib");

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

/// Opens a library from the bundled library directory.
pub fn openBundled(comptime Api: type, io: std.Io, name: []const u8) !Api {
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const directory = sandbox.path(io, &buffer) orelse {
        log.warn("{s}: no sandbox directory", .{name});
        return error.LibraryNotFound;
    };
    const suffix = try std.fmt.bufPrintSentinel(buffer[directory.len..], "/lib/{s}", .{name}, 0);
    return open(Api, &.{buffer[0 .. directory.len + suffix.len :0]});
}

/// Specializes a native scalar binding as a receiver-only probe method.
pub fn Scalar(comptime Probe: type, comptime spec: anytype) fn (*const Probe) ?u64 {
    return struct {
        fn read(probe: *const Probe) ?u64 {
            const getter = @field(probe.api.*, @tagName(spec[0]));
            const value = getter.get(.{probe.handle} ++ spec[1]) catch return null;
            return std.math.cast(u64, value);
        }
    }.read;
}

/// Uses translated C signatures for a library's required symbol table.
pub fn Bindings(comptime C: type, comptime check: anytype, comptime names: []const [:0]const u8) type {
    const Checked = CheckedBy(check);
    var types: [names.len]type = undefined;
    for (names, &types) |name, *Field| Field.* = Checked(@TypeOf(@field(C, name)));
    return @Struct(.auto, null, names, &types, &@splat(.{}));
}

/// Binds a library's status check to wrappers for its C functions.
pub fn CheckedBy(comptime check: anytype) fn (comptime anytype) type {
    const checker = @typeInfo(@TypeOf(check)).@"fn";
    const Error = @typeInfo(checker.return_type.?).error_union.error_set;
    return struct {
        fn Checked(comptime function: anytype) type {
            const Function = if (@TypeOf(function) == type) function else @TypeOf(function);
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
    }.Checked;
}
