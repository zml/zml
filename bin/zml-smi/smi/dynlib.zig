const std = @import("std");

const log = std.log.scoped(.@"zml-smi/dynlib");

pub fn open(comptime Api: type, paths: []const [:0]const u8) !Api {
    for (paths) |path| {
        if (std.c.dlopen(path, .{ .LAZY = true, .GLOBAL = true, .NODELETE = true })) |handle| {
            return lookupStruct(handle, Api);
        }
    }

    if (std.c.dlerror()) |err| log.warn("dlopen: {s}", .{err});
    return error.LibraryNotFound;
}

fn lookupStruct(handle: *anyopaque, comptime Api: type) !Api {
    var api: Api = undefined;

    inline for (comptime std.meta.fieldNames(Api)) |name| {
        const symbol = std.c.dlsym(handle, name) orelse {
            log.warn("no symbol {s}", .{name});
            return error.SymbolResolutionFailed;
        };

        @field(api, name) = @ptrCast(@alignCast(symbol));
    }

    return api;
}
