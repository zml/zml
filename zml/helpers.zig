//! Public helpers to manipulate tensors or shapes.
const std = @import("std");

const meta = @import("meta.zig");
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const EnumLiteral = @TypeOf(.enum_literal);
const log = std.log.scoped(.@"zml/tensor");

test {
    std.testing.refAllDecls(@This());
}

const ShapeError = error{ DimMismatch, NotFound };
const NOT_SET: i64 = -2;
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
        .res = undefined,
        .mode = mode,
    };
    @memset(std.mem.bytesAsSlice(i64, std.mem.asBytes(&context.res)), NOT_SET);

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
        try zml.testing.expectEqual(collectDims(.{ .b, .d }, &model, .ignore_errors), .{ .b = DIM_MISMATCH, .d = 5 });
    }
    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{5}, .f32).withTags(.{.d}),
        };
        try std.testing.expectEqual(collectDims(.{ .b, .d, .c }, &model, .strict), error.NotFound);
        try zml.testing.expectEqual(collectDims(.{ .b, .d, .c }, &model, .ignore_errors), .{ .b = 2, .d = 5, .c = NOT_SET });
    }
    {
        var model: Model = .{
            .x = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .y = Shape.init(.{ 2, 5 }, .f32).withTags(.{ .b, .d }),
            .bias = Shape.init(.{7}, .f32).withTags(.{.d}),
        };
        try std.testing.expectEqual(collectDims(.{ .b, .d }, &model, .strict), error.DimMismatch);
        try zml.testing.expectEqual(collectDims(.{ .b, .d }, &model, .ignore_errors), .{ .b = 2, .d = DIM_MISMATCH });
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
            .default_value_ptr = &default,
            .is_comptime = false,
            .alignment = @alignOf(i64),
        };
    }
    return @Type(.{ .@"struct" = .{
        .layout = .@"extern",
        .fields = &struct_fields,
        .decls = &.{},
        .is_tuple = false,
    } });
}
