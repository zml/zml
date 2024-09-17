//! Public helpers to manipulate tensors or shapes.
const std = @import("std");

const meta = @import("meta.zig");
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const EnumLiteral = @TypeOf(.enum_literal);
const log = std.log.scoped(.zml_tensor);

const ShapeError = error{ DimMismatch, NotFound };
const NOT_SET: i64 = 0;
const DIM_MISMATCH: i64 = -1;

/// Collect the given dimensions inside a struct containing tagged tensors.
pub fn collectDims(
    comptime dims: anytype,
    v: anytype,
    comptime mode: enum { strict, allow_extra_dims, ignore_errors },
) ShapeError!ShapeStruct(dims) {
    const LocalContext = struct {
        res: ShapeStruct(dims),
        mode: @TypeOf(mode),
    };
    var context = LocalContext{
        .res = std.mem.zeroes(ShapeStruct(dims)),
        .mode = mode,
    };

    meta.visit((struct {
        fn cb(ctx: *LocalContext, shape: *const Shape) void {
            inline for (dims) |a| {
                if (shape.hasTag(a)) |axis| {
                    const dim = shape.dim(axis);

                    const expected_dim = &@field(ctx.res, @tagName(a));
                    if (expected_dim.* == NOT_SET) {
                        expected_dim.* = dim;
                    } else if (expected_dim.* == DIM_MISMATCH) {
                        // this axis has already been reported as invalid.
                    } else if (dim != expected_dim.*) {
                        if (mode != .ignore_errors) {
                            log.warn("Dim mismatch ! Axis {0s}={1d} but received a new tensor where {0s}={2d}", .{ @tagName(a), expected_dim.*, dim });
                        }
                        expected_dim.* = DIM_MISMATCH;
                    }
                }
                // TODO: strict mode:
                // else if (mode == .strict) {
                //     @compileError("Found unexpected axis " ++ @tagName(a) ++ " when collecting " ++ @typeName(ShapeStruct(dims)));
                // }
            }
        }
    }).cb, &context, v);

    if (context.mode != .ignore_errors) {
        inline for (shapeToDims(context.res), dims) |dim, dim_tag| {
            if (dim == NOT_SET) {
                log.warn("Axis not found: {s}", .{@tagName(dim_tag)});
                return error.NotFound;
            }
            if (dim == DIM_MISMATCH) return error.DimMismatch;
        }
    }
    return context.res;
}

fn shapeToDims(shape: anytype) [@divExact(@sizeOf(@TypeOf(shape)), @sizeOf(i64))]i64 {
    return @bitCast(shape);
}

test collectDims {
    const zml = @import("zml.zig");

    const Model = struct {
        x: Shape,
        y: Shape,
        bias: Shape,
    };

    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{5}, .f32).withTags(.{.d}),
        };
        try zml.testing.expectEqual(collectDims(.{ .b, .d }, &model, .strict), .{ .b = 2, .d = 5 });
    }
    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 3, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{5}, .f32).withTags(.{.d}),
        };
        try std.testing.expectEqual(
            collectDims(.{ .b, .d }, &model, .strict),
            error.DimMismatch,
        );
        try zml.testing.expectEqual(collectDims(.{ .b, .d }, &model, .ignore_errors), .{ .b = -1, .d = 5 });
    }
    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{5}, .f32).withTags(.{.d}),
        };
        try std.testing.expectEqual(collectDims(.{ .b, .d, .c }, &model, .strict), error.NotFound);
        try zml.testing.expectEqual(collectDims(.{ .b, .d, .c }, &model, .ignore_errors), .{ .b = 2, .d = 5, .c = 0 });
    }
    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{7}, .f32).withTags(.{.d}),
        };
        try std.testing.expectEqual(collectDims(.{ .b, .d }, &model, .strict), error.DimMismatch);
        try zml.testing.expectEqual(collectDims(.{ .b, .d }, &model, .ignore_errors), .{ .b = 2, .d = -1 });
    }
}

fn ShapeStruct(comptime dims: anytype) type {
    const rank = dims.len;
    @setEvalBranchQuota(rank + 5);
    var struct_fields: [rank]std.builtin.Type.StructField = undefined;
    const default: i64 = NOT_SET;
    for (&struct_fields, dims) |*struct_field, axis| {
        struct_field.* = .{
            .name = @tagName(axis),
            .type = i64,
            .default_value = &default,
            .is_comptime = false,
            .alignment = @alignOf(i64),
        };
    }
    return @Type(.{ .Struct = .{
        .layout = .@"extern",
        .fields = &struct_fields,
        .decls = &.{},
        .is_tuple = false,
    } });
}

/// Return a new struct with all tensors replaced by the output of the given function.
pub fn mapTensors(func: anytype, v: anytype, args: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    const type_info = @typeInfo(T);
    if (T == Tensor) return @call(.auto, func, .{v} ++ args);

    return switch (type_info) {
        .Pointer => @compileError("mapTensors only accept by value arguments. Received: " ++ @typeName(T)),
        .Struct => |struct_info| {
            var copy: T = v;
            inline for (struct_info.fields) |feeld| {
                if (feeld.is_comptime) continue;
                if (@typeInfo(feeld.type) == .Pointer) {
                    @compileError("mapTensors doesn't follow pointers and don't accept struct containing them. Received: " ++ @typeName(T));
                }
                @field(copy, feeld.name) = mapTensors(func, @field(v, feeld.name), args);
            }
            return copy;
        },
        .Array => {
            var res: T = undefined;
            for (v, &res) |item, *r| {
                r.* = mapTensors(func, item, args);
            }
            return res;
        },
        .Union, .Optional => @compileError("mapTensors doesn't yet support " ++ @typeName(T)),
        else => v,
    };
}
