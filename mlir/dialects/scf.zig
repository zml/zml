const std = @import("std");
const mlir = @import("mlir");

pub fn ForBody(ExtraArgs: type) type {
    return fn (mlir.Context, mlir.Block, ExtraArgs) mlir.Operation;
}

pub const ForRange = struct {
    start: mlir.Value,
    end: mlir.Value,
    step: mlir.Value,
};

pub fn @"for"(
    ExtraArgs: type,
    ctx: mlir.Context,
    range: ForRange,
    init_values: []const mlir.Value,
    body: ForBody(ExtraArgs),
    extra_args: ExtraArgs,
    loc: mlir.Location,
) mlir.Operation {
    const n_args = init_values.len;
    var init_types_buf: [32]mlir.Type = undefined;
    var locs_buf: [32]mlir.Location = undefined;

    // The first block argument is the for loop induction variable,
    // followed then by all the loop-carried variables.
    const init_types = init_types_buf[0 .. n_args + 1];
    const locs = locs_buf[0 .. n_args + 1];
    init_types[0] = range.start.getType();
    locs[0] = loc;
    for (1.., init_values) |i, val| {
        init_types[i] = val.getType();
        locs[i] = loc;
    }

    const block = mlir.Block.init(init_types, locs) catch unreachable;
    const yield_op = @call(.auto, body, .{ ctx, block, extra_args });
    std.debug.assert(std.mem.eql(u8, "scf.yield", yield_op.name().str()));
    block.appendOperationRecursive(yield_op, .open);

    const for_op = mlir.Operation.make(ctx, "scf.for", .{
        .variadic_operands = &.{ &.{ range.start, range.end, range.step }, init_values },
        .results = init_types[1..],
        .blocks = &.{block},
        .location = loc,
        .verify = false,
    });
    return for_op;
}

pub fn yield(ctx: mlir.Context, res: []const mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "scf.yield", .{
        .variadic_operands = &.{res},
        .results = &.{},
        .location = loc,
        .verify = false,
    });
}
