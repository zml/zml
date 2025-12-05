const std = @import("std");

const compileError = @import("debug.zig").compileError;

pub fn ArgsTuple(comptime funcT: anytype, comptime ArgsT: ?type) type {
    const params = @typeInfo(funcT).@"fn".params;
    if (params.len == 0) {
        return @TypeOf(.{});
    }

    if (@typeInfo(funcT).@"fn".is_generic == false) {
        return std.meta.ArgsTuple(funcT);
    }

    const args = std.meta.fields(ArgsT orelse @compileError("generic function requires an explicit ArgsTuple"));
    var tuple_types: [params.len]type = undefined;
    if (params.len != args.len) {
        compileError("function {} expected {} args, got {}", .{ funcT, params.len, args.len });
    }
    inline for (params, args, 0..) |param, arg, i| {
        if (param.type == null) {
            tuple_types[i] = arg.type;
            continue;
        }
        tuple_types[i] = param.type.?;
    }

    return @Tuple(tuple_types);
}

pub const Signature = struct {
    Func: type,
    FuncT: type,
    ArgsT: type,
    ReturnT: type,
    ReturnPayloadT: type,
    ReturnErrorSet: ?type,
};

pub fn FnSignature(comptime func: anytype, comptime argsT_: ?type) Signature {
    const argsT = ArgsTuple(@TypeOf(func), argsT_);
    const return_type = @TypeOf(@call(.auto, func, @as(argsT, undefined)));
    return Signature{
        .Func = struct {
            pub const Value = func;
        },
        .FuncT = @TypeOf(func),
        .ArgsT = argsT,
        .ReturnT = return_type,
        .ReturnPayloadT = switch (@typeInfo(return_type)) {
            .error_union => |u| u.payload,
            else => return_type,
        },
        .ReturnErrorSet = switch (@typeInfo(return_type)) {
            .error_union => |u| u.error_set,
            else => null,
        },
    };
}
