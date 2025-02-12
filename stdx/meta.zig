const std = @import("std");
const debug = @import("debug.zig");

const compileError = debug.compileError;

pub const FnSignature = @import("signature.zig").FnSignature;
pub const Signature = @import("signature.zig").Signature;

pub fn isStruct(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => true,
        else => false,
    };
}

pub fn isTuple(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| info.is_tuple,
        else => false,
    };
}

pub fn isStructOf(comptime T: type, comptime Elem: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            inline for (info.fields) |field| {
                if (field.type != Elem) {
                    break :blk false;
                }
            }
            break :blk true;
        },
        else => false,
    };
}

pub fn isStructOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            inline for (info.fields) |field| {
                if (f(field.type) == false) {
                    break :blk false;
                }
            }
            break :blk true;
        },
        else => false,
    };
}

pub fn isTupleOf(comptime T: type, comptime Elem: type) bool {
    return isTuple(T) and isStructOf(T, Elem);
}

pub fn isTupleOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return isTuple(T) and isStructOfAny(T, f);
}

pub fn isSliceOf(comptime T: type, comptime Elem: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .slice => info.child == Elem,
            .one => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .array => |arr_info| arr_info.child == Elem,
                else => false,
            },
            else => false,
        },
        else => false,
    };
}

pub fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

pub fn isSliceOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return switch (@typeInfo(T)) {
        .pointer => |info| info.size == .slice and f(info.child),
        else => false,
    };
}

pub fn DeclEnum(comptime T: type) type {
    const field_infos = std.meta.declarations(T);
    if (field_infos.len == 0) {
        compileError("Struct {} has no declarations", .{T});
    }
    return std.meta.DeclEnum(UnwrapPtr(T));
}

pub fn UnwrapPtr(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .one => info.child,
            else => T,
        },
        else => T,
    };
}

pub fn asSlice(comptime T: type) type {
    const err_msg = "Type " ++ @typeName(T) ++ " can't be interpreted as a slice";
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .slice => info.child,
            .one => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .array => |arr_info| arr_info.child,
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    };
}

pub fn TupleRange(comptime T: type, comptime start: ?usize, comptime end: ?usize) type {
    return TupleRangeX(T, start orelse 0, end orelse std.meta.fields(T).len);
}

pub fn TupleRangeX(comptime T: type, comptime start: usize, comptime end: usize) type {
    const fields = std.meta.fields(T);
    var new_fields: [end - start]std.builtin.Type.StructField = undefined;
    inline for (start..end, 0..) |i, j| {
        var new_field = fields[i];
        var num_buf: [32]u8 = undefined;
        new_field.name = blk: {
            const s = std.fmt.formatIntBuf(&num_buf, j, 10, .lower, .{});
            num_buf[s] = 0;
            break :blk num_buf[0..s :0];
        };
        new_fields[j] = new_field;
    }
    return @Type(.{
        .@"struct" = .{
            .is_tuple = true,
            .layout = .auto,
            .decls = &.{},
            .fields = &new_fields,
        },
    });
}

pub fn FnParam(comptime func: anytype, comptime n: comptime_int) type {
    return @typeInfo(@TypeOf(func)).@"fn".params[n].type orelse @compileError("anytype is not supported");
}

pub fn FnArgs(comptime func: anytype) type {
    debug.assertComptime(!@typeInfo(@TypeOf(func)).@"fn".is_generic, "FnArgs expects non generic function, got: {}", .{@TypeOf(func)});
    return FnSignature(func, null).ArgsT;
}

pub fn FnArgsWithHint(comptime func: anytype, ArgsT: type) type {
    debug.assertComptime(@typeInfo(@TypeOf(func)).@"fn".is_generic, "FnArgsWithHint expects a generic function, got: {}", .{@TypeOf(func)});
    return FnSignature(func, ArgsT).ArgsT;
}

pub fn FnResult(comptime func: anytype) type {
    return @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
}

pub fn Head(Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.fields.len == 0) @compileError("Can't tail empty tuple");
            return struct_info.fields[0].type;
        },
        else => @compileError("Head works on tuple type"),
    };
}

pub fn Tail(Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.fields.len == 0) @compileError("Can't tail empty tuple");
            var types: [struct_info.fields.len - 1]type = undefined;
            for (struct_info.fields[1..], 0..) |field, i| types[i] = field.type;
            return std.meta.Tuple(&types);
        },
        else => @compileError("Tail works on tuple type"),
    };
}
