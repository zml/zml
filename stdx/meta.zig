const std = @import("std");

const debug = @import("debug.zig");
const compileError = debug.compileError;

pub const Field = struct {
    name: [:0]const u8,
    type: type,
    attrs: std.builtin.Type.Struct.FieldAttributes = .{},
    value: comptime_int = 0,
};

pub fn fields(comptime T: type) [std.meta.fieldNames(T).len]Field {
    const info = @typeInfo(T);
    var result: [std.meta.fieldNames(T).len]Field = undefined;
    switch (info) {
        .@"struct" => |struct_info| {
            for (
                struct_info.field_names,
                struct_info.field_types,
                struct_info.field_attrs,
                &result,
            ) |name, FieldType, attrs, *field| {
                field.* = .{ .name = name, .type = FieldType, .attrs = attrs };
            }
        },
        .@"union" => |union_info| {
            for (
                union_info.field_names,
                union_info.field_types,
                union_info.field_attrs,
                &result,
            ) |name, FieldType, attrs, *field| {
                field.* = .{
                    .name = name,
                    .type = FieldType,
                    .attrs = .{ .@"align" = attrs.@"align" },
                };
            }
        },
        .@"enum" => |enum_info| {
            for (enum_info.field_names, enum_info.field_values, &result) |name, value, *field| {
                field.* = .{ .name = name, .type = void, .value = value };
            }
        },
        .error_set => |error_set_info| {
            for (error_set_info.error_names.?, &result) |name, *field| {
                field.* = .{ .name = name, .type = void };
            }
        },
        else => @compileError("Expected struct, union, enum, or error set type, found '" ++ @typeName(T) ++ "'"),
    }
    return result;
}

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
            inline for (info.field_types) |FieldType| {
                if (FieldType != Elem) {
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
            inline for (info.field_types) |FieldType| {
                if (f(FieldType) == false) {
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
        .pointer => |info| switch (info.size) {
            .one => info.child == @TypeOf(.{}),
            .slice => f(info.child),
            else => false,
        },
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
    const all_field_types = std.meta.fieldTypes(T);
    const start_ = start orelse 0;
    const end_ = end orelse all_field_types.len;

    if (start_ == end_) {
        return @Tuple(&.{});
    }

    var field_types: [end_ - start_]type = undefined;
    inline for (start_..end_, 0..) |i, j| {
        field_types[j] = all_field_types[i];
    }
    return @Tuple(&field_types);
}

pub fn FnParam(comptime func: anytype, comptime n: comptime_int) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".param_types;
    if (n >= params.len) {
        @compileError("param doesn't exist in func");
    }
    return params[n] orelse @compileError("anytype is not supported");
}

pub fn FnReturn(comptime func: anytype) type {
    return @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
}

pub fn FnReturnPayload(comptime func: anytype) type {
    const rt = @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
    return switch (@typeInfo(rt)) {
        .error_union => |u| u.payload,
        else => rt,
    };
}

pub fn FnReturnErrorSet(comptime func: anytype) ?type {
    const rt = @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
    return switch (@typeInfo(rt)) {
        .error_union => |u| u.error_set,
        else => null,
    };
}

pub fn Head(comptime Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.field_types.len == 0) @compileError("Can't tail empty tuple");
            return struct_info.field_types[0];
        },
        else => @compileError("Head works on tuple type"),
    };
}

pub fn Tail(comptime Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.field_types.len == 0) @compileError("Can't tail empty tuple");
            var types: [struct_info.field_types.len - 1]type = undefined;
            for (struct_info.field_types[1..], &types) |FieldType, *type_| {
                type_.* = FieldType;
            }
            return @Tuple(&types);
        },
        else => @compileError("Tail works on tuple type"),
    };
}
