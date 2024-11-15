const std = @import("std");

const compileError = @import("meta.zig").compileError;

pub fn ArgsTuple(comptime funcT: anytype, comptime argsT: ?type) type {
    const params = @typeInfo(funcT).Fn.params;
    if (params.len == 0) {
        return @TypeOf(.{});
    }

    if (@typeInfo(funcT).Fn.is_generic == false) {
        return std.meta.ArgsTuple(funcT);
    }

    const args = std.meta.fields(argsT orelse compileError("generic function requires an explicit ArgsTuple", .{}));
    var tuple_fields: [params.len]std.builtin.Type.StructField = undefined;
    inline for (params, args, 0..) |param, arg, i| {
        if (param.type == null) {
            tuple_fields[i] = arg;
            continue;
        }
        const T = param.type.?;
        var num_buf: [32]u8 = undefined;
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

pub fn FnSignature(comptime func: anytype, comptime argsT: ?type) type {
    return FnSignatureX(func, ArgsTuple(@TypeOf(func), argsT));
}

fn FnSignatureX(comptime func: anytype, comptime argsT: type) type {
    return struct {
        pub const FuncT = @TypeOf(func);
        pub const ArgsT = argsT;
        pub const ReturnT = @TypeOf(@call(.auto, func, @as(ArgsT, undefined)));
        pub const ReturnPayloadT = switch (@typeInfo(ReturnT)) {
            .ErrorUnion => |u| u.payload,
            else => ReturnT,
        };
        pub const ReturnErrorSet: ?type = switch (@typeInfo(ReturnT)) {
            .ErrorUnion => |u| u.error_set,
            else => null,
        };
    };
}
