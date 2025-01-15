const std = @import("std");

const compileError = @import("debug.zig").compileError;

pub fn ArgsTuple(comptime funcT: anytype, comptime ArgsT: ?type) type {
    const params = @typeInfo(funcT).Fn.params;
    if (params.len == 0) {
        return @TypeOf(.{});
    }

    if (@typeInfo(funcT).Fn.is_generic == false) {
        return std.meta.ArgsTuple(funcT);
    }

    const args = std.meta.fields(ArgsT orelse @compileError("generic function requires an explicit ArgsTuple"));
    var tuple_fields: [params.len]std.builtin.Type.StructField = undefined;
    if (params.len != args.len) {
        compileError("function {} expected {} args, got {}", .{ funcT, params.len, args.len });
    }
    inline for (params, args, 0..) |param, arg, i| {
        if (param.type == null) {
            tuple_fields[i] = arg;
            continue;
        }
        const T = param.type.?;
        var num_buf: [8]u8 = undefined;
        tuple_fields[i] = .{
            .name = blk: {
                const s = std.fmt.formatIntBuf(&num_buf, i, 10, .lower, .{});
                num_buf[s] = 0;
                break :blk num_buf[0..s :0];
            },
            .type = T,
            .default_value = null,
            .is_comptime = false,
            .alignment = if (@sizeOf(T) > 0) @alignOf(T) else 0,
        };
    }

    return @Type(.{
        .Struct = .{
            .is_tuple = true,
            .layout = .auto,
            .decls = &.{},
            .fields = &tuple_fields,
        },
    });
}

pub const Signature = struct {
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
        .FuncT = @TypeOf(func),
        .ArgsT = argsT,
        .ReturnT = return_type,
        .ReturnPayloadT = switch (@typeInfo(return_type)) {
            .ErrorUnion => |u| u.payload,
            else => return_type,
        },
        .ReturnErrorSet = switch (@typeInfo(return_type)) {
            .ErrorUnion => |u| u.error_set,
            else => null,
        },
    };
}
