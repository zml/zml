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
    visibility: ?Visibility = .public,
    verify: bool = true,
    extra_attributes: []const mlir.NamedAttribute = &.{},
};

pub fn func(ctx: *mlir.Context, args: FuncOpArgs) *mlir.Operation {
    var args_buffer: stdx.BoundedArray(*const mlir.Type, 2048) = .empty;
    var results_buffer: stdx.BoundedArray(*const mlir.Type, 32) = .empty;

    var attr_tuples_buffer: stdx.BoundedArray(mlir.NamedAttribute, 16) = .empty;
    attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "sym_name", .string(ctx, args.name)));
    if (args.visibility) |v| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "sym_visibility", .string(ctx, @tagName(v))));
    }
    attr_tuples_buffer.appendSliceAssumeCapacity(&.{
        .named(ctx, "function_type", .typeAttr(.function(
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
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "arg_attrs", .array(ctx, args_attributes)));
    }
    if (args.results_attributes) |results_attributes| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "res_attrs", .array(ctx, results_attributes)));
    }
    if (args.no_inline) {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "no_inline", .unit(ctx)));
    }
    for (args.extra_attributes) |a| {
        attr_tuples_buffer.appendAssumeCapacity(a);
    }

    return mlir.Operation.make(ctx, "func.func", .{
        .blocks = &.{args.block},
        .attributes = attr_tuples_buffer.constSlice(),
        .location = args.location,
        .verify = args.verify,
    });
}

pub fn call(ctx: *mlir.Context, name: []const u8, values: []const *const mlir.Value, results: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.call", .{
        .operands = .{ .variadic = &.{values} },
        .results = .{ .flat = results },
        .attributes = &.{
            .named(ctx, "callee", .flatSymbolRef(ctx, name)),
        },
        .location = location,
    });
}

pub fn return_(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = .{ .flat = &.{value} },
        .location = location,
        .verify = false,
    });
}

pub fn returns(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "func.return", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}
