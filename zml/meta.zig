const std = @import("std");

const testing = std.testing;

/// Computes floating point value division between two integers.
pub fn divFloat(T: type, numerator: anytype, denominator: anytype) T {
    return toFloat(T, numerator) / toFloat(T, denominator);
}

fn toFloat(T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .Float => @floatCast(x),
        else => @floatFromInt(x),
    };
}

pub fn guard(check: bool, src: std.builtin.SourceLocation) void {
    assert(check, "Invalid inputs {s}@{s}:{d}", .{ src.file, src.fn_name, src.line });
}

pub inline fn internalAssert(check: bool, comptime msg: []const u8, args: anytype) void {
    assert(check, "ZML internal error: " ++ msg, args);
}

pub fn assert(check: bool, comptime msg: []const u8, args: anytype) void {
    if (!check) panic(msg, args);
}

pub fn panic(comptime msg: []const u8, args: anytype) noreturn {
    std.log.err(msg, args);
    @panic(msg);
}

pub fn compileLog(comptime msg: []const u8, comptime args: anytype) void {
    @compileLog(std.fmt.comptimePrint(msg, args));
}

pub fn compileError(comptime msg: []const u8, comptime args: anytype) noreturn {
    @compileError(std.fmt.comptimePrint(msg, args));
}

pub fn assertComptime(comptime check: bool, comptime msg: []const u8, comptime args: anytype) void {
    if (check == false) {
        compileError(msg, args);
    }
}

pub fn isStruct(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => true,
        else => false,
    };
}

pub fn isTuple(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => |info| info.is_tuple,
        else => false,
    };
}

pub fn isStructOf(comptime T: type, comptime Elem: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => |info| blk: {
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
        .Struct => |info| blk: {
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
        .Pointer => |info| switch (info.size) {
            .Slice => info.child == Elem,
            .One => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .Array => |arr_info| arr_info.child == Elem,
                else => false,
            },
            else => false,
        },
        else => false,
    };
}

pub fn asSlice(comptime T: type) type {
    const err_msg = "Type " ++ @typeName(T) ++ " can't be interpreted as a slice";
    return switch (@typeInfo(T)) {
        .Pointer => |info| switch (info.size) {
            .Slice => info.child,
            .One => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .Array => |arr_info| arr_info.child,
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    };
}

pub fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Int, .ComptimeInt => true,
        else => false,
    };
}

pub fn isSliceOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |info| info.size == .Slice and f(info.child),
        else => false,
    };
}

pub fn DeclEnum(comptime T: type) type {
    return std.meta.DeclEnum(UnwrapPtr(T));
}

pub fn UnwrapPtr(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .Pointer => |info| switch (info.size) {
            .One => info.child,
            else => T,
        },
        else => T,
    };
}

pub fn FnParam(func: anytype, n: comptime_int) type {
    return @typeInfo(@TypeOf(func)).Fn.params[n].type orelse @compileError("anytype not supported in callbacks");
}

pub fn FnParams(func: anytype) type {
    return std.meta.ArgsTuple(@TypeOf(func));
}

pub fn FnResult(func: anytype) type {
    return @typeInfo(@TypeOf(func)).Fn.return_type.?;
}

pub fn FnResultPayload(func: anytype) type {
    const return_type = @typeInfo(@TypeOf(func)).Fn.return_type.?;
    const payload_type = switch (@typeInfo(return_type)) {
        .ErrorUnion => |u| u.payload,
        else => return_type,
    };
    return payload_type;
}

pub fn FnResultErrorSet(func: anytype) ?type {
    const return_type = @typeInfo(@TypeOf(func)).Fn.return_type.?;
    const error_set = switch (@typeInfo(return_type)) {
        .ErrorUnion => |u| u.error_set,
        else => null,
    };
    return error_set;
}

pub fn Signature(comptime func: anytype, comptime argsT: ?type) type {
    return struct {
        pub const FuncT = if (@TypeOf(func) == type) func else @TypeOf(func);
        pub const ArgsT = blk: {
            if (@typeInfo(FuncT).Fn.params.len == 0) {
                break :blk @TypeOf(.{});
            }
            break :blk argsT orelse std.meta.ArgsTuple(FuncT);
        };
        pub const ReturnT = @TypeOf(@call(.auto, func, @as(ArgsT, undefined)));
        pub const ReturnPayloadT = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.payload,
                else => ReturnT,
            };
        };
        pub const ReturnErrorSet: ?type = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.error_set,
                else => null,
            };
        };
    };
}

pub fn MapType(From: type, To: type) type {
    return struct {
        pub fn map(T: type) type {
            switch (T) {
                To => return To,
                ?To => return ?To,
                From => return To,
                *From => return *To,
                ?From => return ?To,
                else => {},
            }

            return switch (@typeInfo(T)) {
                .Struct => |struct_infos| {
                    const fields = struct_infos.fields;
                    var same: bool = true;
                    var struct_fields: [fields.len]std.builtin.Type.StructField = undefined;
                    for (struct_fields[0..], fields) |*struct_field, field| {
                        if (!field.is_comptime) {
                            const R = map(field.type);
                            if (R == field.type) {
                                struct_field.* = field;
                            } else {
                                struct_field.* = .{
                                    .name = field.name,
                                    .type = R,
                                    .default_value = null,
                                    .is_comptime = field.is_comptime,
                                    .alignment = @alignOf(R),
                                };
                                same = false;
                                // Handle the case `field: ?Tensor = null`
                                // Generic handling of default value is complicated,
                                // it would require to call the callback at comptime.
                                if (R == ?To) {
                                    struct_field.default_value = &@as(R, null);
                                }
                            }
                        } else {
                            struct_field.* = field;
                        }
                    }
                    if (same) return T;
                    return @Type(.{ .Struct = .{
                        .layout = .auto,
                        .fields = struct_fields[0..],
                        .decls = &.{},
                        .is_tuple = struct_infos.is_tuple,
                    } });
                },
                .Array => |arr_info| [arr_info.len]map(arr_info.child),
                .Pointer => |ptr_info| switch (ptr_info.size) {
                    .Slice => if (ptr_info.is_const)
                        []const map(ptr_info.child)
                    else
                        []map(ptr_info.child),
                    .One => *map(ptr_info.child),
                    else => T,
                },
                .Optional => |opt_info| ?map(opt_info.child),
                else => T,
            };
        }
    };
}

/// Given a callback: `fn(Ctx, From) To`, recursively visits the given `from` struct
/// and calls the callback when it finds a `From` element, and writes it to the `to` struct.
/// The `to` parameter must be passed with mutable pointer, and tensor data need to be mutable if callback needs it.
/// `mapAlloc` tries as much as possible to respect the conversions made by Zig itself.
/// For example it can convert from a comptime array to a runtime slice.
/// `mapAlloc` can allocate new slices to write the result if the result struct requires it.
/// The caller is owning said allocations, using an `ArenaAllocator` might help tracking them.
// TODO: handle tuple to slice conversion
pub fn mapAlloc(comptime cb: anytype, allocator: std.mem.Allocator, ctx: FnParam(cb, 0), from: anytype, to: anytype) !void {
    // const Ctx = FnParam(cb, 0);
    const From = FnParam(cb, 1);
    const FromStruct = @TypeOf(from);

    const type_info_to_ptr = @typeInfo(@TypeOf(to));
    if (type_info_to_ptr != .Pointer) {
        @compileError("convertType is expecting a mutable `to` argument but received: " ++ @typeName(@TypeOf(to)));
    }
    const ToStruct = type_info_to_ptr.Pointer.child;
    const type_info_to = @typeInfo(ToStruct);

    if (FromStruct == From) {
        // Special case for converting from shape to tensor:
        // If the target type is Shape, skip tensor conversion.
        // A general `to.* = from` assignment causes a Zig error in this scenario.
        // (see below)
        if (ToStruct == @import("shape.zig").Shape and FromStruct == ToStruct) { // FromStruct) {
            to.* = from;
        } else {
            to.* = @call(.auto, cb, .{ ctx, from });
        }
        return;
    }

    // This is generally due to a user error, but let this fn compile,
    // and the user will have a Zig error.
    if (FromStruct == ToStruct) {
        to.* = from;
        return;
    }

    // Don't go into Shape objects because of the weird tag.
    // TODO: we could not error on pointers to basic types like u8
    if (FromStruct == @import("shape.zig").Shape) {
        to.* = from;
        return;
    }
    switch (type_info_to) {
        .Struct => |info| inline for (info.fields) |field| {
            // if (field.is_comptime) continue;
            const field_type_info = @typeInfo(field.type);
            // If the field is already a pointer, we recurse with it directly, otherwise, we recurse with a pointer to the field.
            switch (field_type_info) {
                // .Pointer => try convertType(From, To, allocator, @field(from, field.name), @field(to, field.name), Ctx, ctx, cb),
                .Array, .Optional, .Union, .Struct, .Pointer => if (@hasField(FromStruct, field.name)) {
                    try mapAlloc(
                        cb,
                        allocator,
                        ctx,
                        @field(from, field.name),
                        &@field(to, field.name),
                    );
                } else if (field.default_value) |_| {
                    @field(to, field.name) = null;
                } else {
                    compileError("Mapping {} to {} failed. Missing field {s}", .{ FromStruct, ToStruct, field.name });
                },
                else => @field(to, field.name) = @field(from, field.name),
            }
        },
        .Array => for (from, to) |f, *t| {
            try mapAlloc(cb, allocator, ctx, f, t);
        },
        .Pointer => |ptr_info| switch (ptr_info.size) {
            .One => switch (type_info_to_ptr.Pointer.size) {
                // pointer to array -> slice promotion
                .Slice => {
                    to.* = try allocator.alloc(type_info_to_ptr.Pointer.child, from.len);
                    for (from, to.*) |f, *t| {
                        try mapAlloc(cb, allocator, ctx, f, t);
                    }
                },
                else => try mapAlloc(cb, allocator, ctx, from.*, to.*),
            },
            .Slice => {
                const items = try allocator.alloc(@typeInfo(ToStruct).Pointer.child, from.len);
                for (from, items) |f, *t| {
                    try mapAlloc(cb, allocator, ctx, f, t);
                }
                to.* = items;
            },
            else => @compileError("zml.meta.mapAlloc doesn't support: " ++ @typeName(FromStruct)),
        },
        .Optional => if (from) |f| {
            to.* = @as(@typeInfo(type_info_to_ptr.Pointer.child).Optional.child, undefined);
            try mapAlloc(cb, allocator, ctx, f, &(to.*.?));
        } else {
            to.* = null;
        },
        .Int, .Float => to.* = from,
        else => @compileError("zml.meta.mapAlloc doesn't support: " ++ @typeName(FromStruct)),
    }
}

test mapAlloc {
    const B = struct { b: u8 };
    const A = struct {
        a: u8,
        pub fn convert(_: void, a: @This()) B {
            return .{ .b = a.a };
        }
    };

    const AA = struct {
        field: A,
        array: [2]A,
        slice: []const A,
        other: u8,
        // We want to allow conversion from comptime to runtime, because Zig type inference works like this.
        comptime static_val: u8 = 8,
        comptime static_slice: [2]A = .{ .{ .a = 11 }, .{ .a = 12 } },
    };
    const BB = struct {
        field: B,
        array: [2]B,
        slice: []const B,
        other: u8,
        static_val: u8,
        static_slice: []B,
    };

    const aa: AA = .{
        .field = .{ .a = 4 },
        .array = .{ .{ .a = 5 }, .{ .a = 6 } },
        .other = 7,
        .slice = &.{ .{ .a = 9 }, .{ .a = 10 } },
    };
    var bb: BB = undefined;

    try mapAlloc(A.convert, testing.allocator, {}, aa, &bb);
    defer testing.allocator.free(bb.slice);
    defer testing.allocator.free(bb.static_slice);

    try testing.expectEqual(4, bb.field.b);
    try testing.expectEqual(5, bb.array[0].b);
    try testing.expectEqual(6, bb.array[1].b);
    try testing.expectEqual(7, bb.other);
    try testing.expectEqual(8, bb.static_val);
    try testing.expectEqual(9, bb.slice[0].b);
    try testing.expectEqual(10, bb.slice[1].b);
    try testing.expectEqual(11, bb.static_slice[0].b);
    try testing.expectEqual(12, bb.static_slice[1].b);
}

/// Recursively visit the given struct and calls the callback for each K found.
/// The `v` parameter must me a pointer, and tensor data need to be mutable if callbacks needs it.
pub fn visit(comptime cb: anytype, ctx: FnParam(cb, 0), v: anytype) void {
    const T = @TypeOf(v);
    const type_info_v = @typeInfo(T);
    const K = switch (@typeInfo(FnParam(cb, 1))) {
        .Pointer => |info| info.child,
        else => @compileError("zml.meta.visit is expecting a pointer value as second parameter in callback to use but found " ++ @typeName(FnParam(cb, 1))),
    };

    if (type_info_v != .Pointer) {
        const Callback = @TypeOf(cb);
        @compileError("zml.meta.visit is expecting a pointer input to go with following callback signature: " ++ @typeName(Callback) ++ " but received: " ++ @typeName(T));
    }

    const ptr_info = type_info_v.Pointer;
    // This is important, because with trivial types like void,
    // Zig sometimes decide to call `visit` at comptime, but can't do
    // the pointer wrangling logic at comptime.
    // So we detect early this case and return.
    if (@sizeOf(ptr_info.child) == 0) return;

    switch (ptr_info.size) {
        // If we have a single pointer, two cases:
        // * It's a pointer to K, in which case we call the callback.
        // * It's a pointer to something else, in which case, we explore and recurse if needed.
        .One => if (ptr_info.child == K) {
            cb(ctx, v);
        } else if (ptr_info.child == ?K) {
            if (v.*) |*val| cb(ctx, val);
        } else switch (@typeInfo(ptr_info.child)) {
            .Struct => |s| inline for (s.fields) |field_info| {
                if (field_info.is_comptime) continue;
                const field_type_info = @typeInfo(field_info.type);
                // If the field is already a pointer, we recurse with it directly, otherwise, we recurse with a pointer to the field.
                switch (field_type_info) {
                    .Pointer => visit(cb, ctx, @field(v, field_info.name)),
                    .Array, .Optional, .Union, .Struct => visit(cb, ctx, &@field(v, field_info.name)),
                    else => {},
                }
            },
            .Array => |_| for (v) |*elem| visit(cb, ctx, elem),
            .Optional => if (v.* != null) visit(cb, ctx, &v.*.?),
            .Union => switch (v.*) {
                inline else => |*v_field| visit(cb, ctx, v_field),
            },
            else => {},
        },
        // If we have a slice, two cases also:
        // * It's a slice of K, in which case we call the callback for each element of the slice.
        // * It's a slice to something else, in which case, for each element we explore and recurse if needed.
        .Slice => {
            for (v) |*v_elem| {
                if (ptr_info.child == K) {
                    cb(ctx, v_elem);
                } else switch (@typeInfo(ptr_info.child)) {
                    .Struct => |s| inline for (s.fields) |field_info| {
                        const field_type_info = @typeInfo(field_info.type);
                        // If the field is already a pointer, we recurse with it directly, otherwise, we recurse with a pointer to the field.
                        if (field_type_info == .Pointer) {
                            visit(cb, ctx, @field(v_elem, field_info.name));
                        } else {
                            visit(cb, ctx, &@field(v_elem, field_info.name));
                        }
                    },
                    .Array => |_| for (v) |*elem| visit(cb, ctx, elem),
                    .Optional => if (v.* != null) visit(cb, ctx, &v.*.?),
                    .Union => switch (v_elem.*) {
                        inline else => |*v_field| visit(cb, ctx, v_field),
                    },
                    else => {},
                }
            }
        },
        else => @compileError("Only single pointer and slice are supported. Received " ++ @typeName(T)),
    }
}

test visit {
    const Attr = struct { data: usize };
    const OtherAttr = struct { other: []const u8 };
    const NestedAttr = struct { nested: Attr };
    const NestedAttrOptional = struct { nested: ?Attr };
    const SimpleStruct = struct { prop: Attr };
    const MultipleTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: ?Attr };
    const NestedTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: NestedAttr, prop4: NestedAttrOptional };

    const LocalContext = struct {
        result: usize,
    };

    {
        var context: LocalContext = .{ .result = 0 };
        const container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visit((struct {
            fn cb(ctx: *LocalContext, attr: *const Attr) void {
                ctx.result += attr.data;
            }
        }).cb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visit((struct {
            fn cb(ctx: *LocalContext, attr: *Attr) void {
                ctx.result += attr.data;
            }
        }).cb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = null };

        visit((struct {
            fn cb(ctx: *LocalContext, attr: *Attr) void {
                ctx.result += attr.data;
            }
        }).cb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        const container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = .{ .data = 2 } };

        visit((struct {
            fn cb(ctx: *LocalContext, attr: *const Attr) void {
                ctx.result += attr.data;
            }
        }).cb, &context, &container);

        try std.testing.expectEqual(3, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        const container: NestedTypesStruct = .{
            .prop1 = .{ .data = 1 },
            .prop2 = .{ .other = "hello" },
            .prop3 = .{ .nested = .{ .data = 2 } },
            .prop4 = .{ .nested = .{ .data = 3 } },
        };

        visit((struct {
            fn cb(ctx: *LocalContext, attr: *const Attr) void {
                ctx.result += attr.data;
            }
        }).cb, &context, &container);

        try std.testing.expectEqual(6, context.result);
    }
}

/// Given a `fn([]const T, Args) T` and a slice of values,
/// will combine all values in one value.
/// Only T elements of values will be looked at.
/// This only works for simple types, in particular `zip` doesn't follow pointers.
/// Which means that zip only allocate temp memory, and nothing need to be freed after the call.
pub fn zip(func: anytype, allocator: std.mem.Allocator, values: anytype, args: anytype) error{OutOfMemory}!asSlice(@TypeOf(values)) {
    const sliceT = @typeInfo(FnParam(func, 0));
    assertComptime(sliceT == .Pointer and sliceT.Pointer.size == .Slice and sliceT.Pointer.child == FnResult(func), "zip requires a `fn([]const T, Args) T`, received: {}", .{@TypeOf(func)});

    const T = sliceT.Pointer.child;
    const V = asSlice(@TypeOf(values));
    if (V == T) {
        return @call(.auto, func, .{values} ++ args);
    }
    // const fn_args

    return switch (@typeInfo(V)) {
        .Pointer => @compileError("zip only accept by value arguments. Received: " ++ @typeName(V)),
        .Struct => |struct_info| {
            var out: V = values[0];
            inline for (struct_info.fields) |f| {
                if (f.is_comptime) continue;
                if (@typeInfo(f.type) == .Pointer) {
                    @compileError("zip doesn't follow pointers and don't accept struct containing them. Received: " ++ @typeName(V));
                }
                var fields = try allocator.alloc(f.type, values.len);
                defer allocator.free(fields);
                for (values, 0..) |val, i| {
                    fields[i] = @field(val, f.name);
                }
                @field(out, f.name) = try zip(func, allocator, fields, args);
            }
            return out;
        },
        .Array => |arr_info| {
            if (@typeInfo(arr_info.child) == .Pointer) {
                @compileError("zip doesn't follow pointers and don't accept struct containing them. Received: " ++ @typeName(V));
            }
            var out: V = undefined;
            var slice = try allocator.alloc(arr_info.child, values.len);
            defer allocator.free(slice);
            for (&out, 0..) |*o, j| {
                for (values, 0..) |val, i| {
                    slice[i] = val[j];
                }
                o.* = try zip(func, allocator, slice, args);
            }
            return out;
        },
        .Union, .Optional => @compileError("zip doesn't yet support " ++ @typeName(V)),
        else => values[0],
    };
}

test zip {
    const A = struct { a: u8, b: [2]u8 };
    const a0: A = .{ .a = 1, .b = .{ 2, 3 } };
    const a1: A = .{ .a = 4, .b = .{ 5, 6 } };

    const Sum = struct {
        pub fn call(x: []const u8) u8 {
            var res: u8 = 0;
            for (x) |xx| res += xx;
            return res;
        }
    };
    const a_sum: A = try zip(Sum.call, testing.allocator, &[_]A{ a0, a1 }, .{});
    try testing.expectEqual(A{ .a = 5, .b = .{ 7, 9 } }, a_sum);
}
