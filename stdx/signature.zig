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

    const args = std.meta.fields(ArgsT orelse compileError("generic function requires an explicit ArgsTuple", .{}));
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

pub fn FnSignature(comptime func: anytype, comptime ArgsT: ?type) type {
    const n_params = switch (@typeInfo(@TypeOf(func))) {
        .Fn => |fn_info| fn_info.params.len,
        else => compileError("FnSignature expects a function as first argument got: {}", .{@TypeOf(func)}),
    };
    if (ArgsT != null) {
        const n_args = switch (@typeInfo(ArgsT.?)) {
            .Struct => |struct_info| struct_info.fields.len,
            else => compileError("function {} need to be called with a tuple of args", .{@TypeOf(func)}),
        };
        if (n_params != n_args) {
            compileError("function {} expected {} args, got {}", .{ @TypeOf(func), n_params, n_args });
        }
    }
    return FnSignatureX(func, ArgsTuple(@TypeOf(func), ArgsT));
}

// TODO: I think this should return a struct instead of returing at type
// this gives a better error stacktrace because here the error is delayed to when the fields are read.
fn FnSignatureX(comptime func: anytype, comptime ArgsT_: type) type {
    return struct {
        pub const FuncT = @TypeOf(func);
        pub const ArgsT = ArgsT_;
        pub const ReturnT = @TypeOf(@call(.auto, func, @as(ArgsT_, undefined)));
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
