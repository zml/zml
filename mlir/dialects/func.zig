const std = @import("std");

const mlir = @import("mlir2");
const stdx = @import("stdx");

const FuncOpArgs = struct {
    name: []const u8,
    args: []const *const mlir.Type,
    args_attributes: ?[]const *const mlir.Attribute = null,
    results: []const *const mlir.Type,
    results_attributes: ?[]const *const mlir.Attribute = null,
    block: *mlir.Block,
    location: *const mlir.Location,
};

pub fn func(ctx: *mlir.Context, args: FuncOpArgs) *mlir.Operation {
    var attrs_tuple_buffer = stdx.BoundedArray(mlir.Operation.MakeArgs.AttrTuple, 4){};
    attrs_tuple_buffer.appendAssumeCapacity(.{ "sym_name", mlir.stringAttribute(ctx, args.name) });
    attrs_tuple_buffer.appendAssumeCapacity(.{ "function_type", mlir.typeAttribute(mlir.functionType(ctx, args.args, args.results)) });
    if (args.args_attributes) |args_attributes| {
        attrs_tuple_buffer.appendAssumeCapacity(.{ "arg_attrs", mlir.arrayAttribute(ctx, args_attributes) });
    }
    if (args.results_attributes) |results_attributes| {
        attrs_tuple_buffer.appendAssumeCapacity(.{ "res_attrs", mlir.arrayAttribute(ctx, results_attributes) });
    }

    return mlir.Operation.make(ctx, "func.func", .{
        .blocks = &.{args.block},
        .attributes = attrs_tuple_buffer.constSlice(),
        .location = args.location,
    });
}

pub fn call(ctx: *mlir.Context, name: []const u8, values: []const *const mlir.Value, results: []const *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.call", .{
        .operands = .{ .variadic = &.{values} },
        .results = .{ .flat = results },
        .attributes = &.{
            .{ "callee", mlir.flatSymbolRefAttribute(ctx, name) },
        },
        .location = loc,
    });
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, loc: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = .{ .flat = values },
        .location = loc,
        .verify = false,
    });
}
