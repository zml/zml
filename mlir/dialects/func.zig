const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const FuncOpArgs = struct {
    pub const Visibility = enum {
        public,
        private,
    };

    name: []const u8,
    args: ?[]const *const mlir.Type = null,
    args_attributes: ?[]const *const mlir.Attribute = null,
    results: ?[]const *const mlir.Type = null,
    results_attributes: ?[]const *const mlir.Attribute = null,
    block: *mlir.Block,
    location: *const mlir.Location,
    no_inline: bool = false,
    visibility: Visibility = .public,
};

pub fn func(ctx: *mlir.Context, args: FuncOpArgs) *mlir.Operation {
    var args_buffer: stdx.BoundedArray(*const mlir.Type, 1024) = .{};
    var results_buffer: stdx.BoundedArray(*const mlir.Type, 32) = .{};

    var attr_tuples_buffer: stdx.BoundedArray(mlir.NamedAttribute, 16) = .{};
    attr_tuples_buffer.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", mlir.stringAttribute(ctx, args.name)),
        .named(ctx, "sym_visibility", mlir.stringAttribute(ctx, @tagName(args.visibility))),
        .named(ctx, "function_type", mlir.typeAttribute(mlir.functionType(
            ctx,
            args.args orelse args: {
                for (0..args.block.numArguments()) |i| {
                    args_buffer.appendAssumeCapacity(args.block.argument(i).type_());
                }
                break :args args_buffer.constSlice();
            },
            args.results orelse results: {
                const terminator = args.block.terminator() orelse {
                    @panic("block has no terminator");
                };
                for (0..terminator.numOperands()) |i| {
                    results_buffer.appendAssumeCapacity(terminator.operand(i).type_());
                }
                break :results results_buffer.constSlice();
            },
        ))),
    });

    if (args.args_attributes) |args_attributes| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "arg_attrs", mlir.arrayAttribute(ctx, args_attributes)));
    }
    if (args.results_attributes) |results_attributes| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "res_attrs", mlir.arrayAttribute(ctx, results_attributes)));
    }
    if (args.no_inline) {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "no_inline", mlir.unitAttribute(ctx)));
    }

    return mlir.Operation.make(ctx, "func.func", .{
        .blocks = &.{args.block},
        .attributes = attr_tuples_buffer.constSlice(),
        .location = args.location,
    });
}

pub fn call(ctx: *mlir.Context, name: []const u8, values: []const *const mlir.Value, results: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.call", .{
        .operands = .{ .variadic = &.{values} },
        .results = .{ .flat = results },
        .attributes = &.{
            .named(ctx, "callee", mlir.flatSymbolRefAttribute(ctx, name)),
        },
        .location = location,
    });
}

pub fn return_(ctx: *mlir.Context, values: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = .{ .flat = &.{values} },
        .location = location,
        .verify = false,
    });
}

pub fn returns(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = .{ .variadic = &.{values} },
        .verify = false,
        .location = location,
    });
}
