const std = @import("std");
const mlir = @import("mlir");

pub fn empty(ctx: mlir.Context, args: struct {
    result: mlir.Type,
    location: mlir.Location,
}) mlir.Operation {
    return mlir.Operation.make(ctx, "tensor.empty", .{
        .results = &.{args.result},
        .location = args.location,
    });
}

pub fn splat(ctx: mlir.Context, args: struct {
    value: mlir.Value,
    result: mlir.Type,
    location: mlir.Location,
}) mlir.Operation {
    return mlir.Operation.make(ctx, "tensor.splat", .{
        .operands = &.{args.value},
        .results = &.{args.result},
        .location = args.location,
    });
}

pub fn cast(ctx: mlir.Context, args: struct {
    source: mlir.Value,
    dest: mlir.Type,
    location: mlir.Location,
}) mlir.Operation {
    return mlir.Operation.make(ctx, "tensor.cast", .{
        .operands = &.{args.source},
        .results = &.{args.dest},
        .location = args.location,
    });
}
