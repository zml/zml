const std = @import("std");

const log = std.log.scoped(.@"zml-smi/dynlib");

pub fn lookupStruct(handle: *anyopaque, comptime Api: type) !Api {
    var api: Api = undefined;
    inline for (comptime std.meta.fieldNames(Api), comptime std.meta.fieldTypes(Api)) |name, Field| {
        const symbol = std.c.dlsym(handle, name) orelse {
            log.warn("no symbol {s}", .{name});
            return error.SymbolResolutionFailed;
        };
        @field(api, name) = if (@typeInfo(Field) == .@"struct") .{ .function = @ptrCast(@alignCast(symbol)) } else @ptrCast(@alignCast(symbol));
    }
    return api;
}

pub fn Checked(comptime Function: type, comptime check: anytype) type {
    return struct {
        function: *const Function,

        pub fn call(self: @This(), args: std.meta.ArgsTuple(Function)) @typeInfo(@TypeOf(check)).@"fn".return_type.? {
            return check(@call(.auto, self.function, args));
        }
    };
}
