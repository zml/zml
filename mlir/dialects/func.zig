const std = @import("std");

const mlir = @import("mlir");

pub fn func(
    ctx: mlir.Context,
    args: struct {
        sym_name: []const u8,
        args: []const mlir.Type,
        arg_attrs: []const mlir.Attribute = &.{},
        results: []const mlir.Type,
        res_attrs: []const mlir.Attribute = &.{},
        block: mlir.Block,
        location: mlir.Location,
    },
) mlir.Operation {
    var attrs_tuple_buffer = std.BoundedArray(mlir.AttrTuple, 4){};
    attrs_tuple_buffer.appendAssumeCapacity(.{ "sym_name", .string(ctx, args.sym_name) });
    attrs_tuple_buffer.appendAssumeCapacity(.{ "function_type", .type_(.function(ctx, args.args, args.results)) });
    if (args.arg_attrs.len > 0) {
        attrs_tuple_buffer.appendAssumeCapacity(.{ "arg_attrs", .array(ctx, args.arg_attrs) });
    }

    if (args.res_attrs.len > 0) {
        attrs_tuple_buffer.appendAssumeCapacity(.{ "res_attrs", .array(ctx, args.res_attrs) });
    }

    return mlir.Operation.make(ctx, "func.func", .{
        .blocks = &.{args.block},
        .attributes = attrs_tuple_buffer.constSlice(),
        .location = args.location,
    });
}

pub fn call(ctx: mlir.Context, name: [:0]const u8, values: []const mlir.Value, results: []const mlir.Type, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "func.call", .{
        .variadic_operands = &.{values},
        .results = results,
        .verify = true,
        .attributes = &.{.{ "callee", .symbol(ctx, name) }},
        .location = loc,
    });
}

pub fn return_(ctx: mlir.Context, values: []const mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = values,
        .verify = false,
        .location = loc,
    });
}
