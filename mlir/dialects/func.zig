const std = @import("std");
const mlir = @import("mlir");

pub fn func(
    ctx: mlir.Context,
    args: struct {
        sym_name: [:0]const u8,
        args: []const mlir.Type,
        arg_attrs: []const mlir.Attribute = &.{},
        results: []const mlir.Type,
        block: mlir.Block,
        location: mlir.Location,
    },
) mlir.Operation {
    const AttrTuple = struct { [:0]const u8, mlir.Attribute };
    var attrs_tuple_buffer = std.BoundedArray(AttrTuple, 3){};
    attrs_tuple_buffer.appendAssumeCapacity(.{ "sym_name", mlir.StringAttribute.init(ctx, args.sym_name).as(mlir.Attribute).? });
    attrs_tuple_buffer.appendAssumeCapacity(.{ "function_type", mlir.TypeAttribute.init((mlir.FunctionType.init(ctx, args.args, args.results) catch unreachable).as(mlir.Type).?).as(mlir.Attribute).? });
    if (args.arg_attrs.len > 0) {
        attrs_tuple_buffer.appendAssumeCapacity(.{ "arg_attrs", mlir.ArrayAttribute.init(ctx, args.arg_attrs).as(mlir.Attribute).? });
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
        .attributes = &.{.{ "callee", mlir.FlatSymbolRefAttribute.init(ctx, name).as(mlir.Attribute).? }},
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
