const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

pub const Dim = enum { x, y, z };

pub fn module(ctx: *mlir.Context, name: []const u8, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.module", .{
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "sym_name", .string(ctx, name)),
        },
        .location = location,
        .verify = false,
    });
}

pub const FuncArgs = struct {
    name: []const u8,
    args: []const *const mlir.Type,
    block: *mlir.Block,
    location: *const mlir.Location,
    kernel: bool = true,
    extra_attributes: []const mlir.NamedAttribute = &.{},
};

/// `gpu.func @name(args) kernel { ... gpu.return }`.
pub fn func(ctx: *mlir.Context, args: FuncArgs) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 16) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", .string(ctx, args.name)),
        .named(ctx, "function_type", .typeAttr(.function(ctx, args.args, &.{}))),
    });
    if (args.kernel) attrs.appendAssumeCapacity(.named(ctx, "gpu.kernel", .unit(ctx)));
    attrs.appendSliceAssumeCapacity(args.extra_attributes);
    return mlir.Operation.make(ctx, "gpu.func", .{
        .blocks = &.{args.block},
        .attributes = attrs.constSlice(),
        .location = args.location,
        .verify = false,
    });
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.return", .{
        .operands = .{ .flat = values },
        .location = location,
        .verify = false,
    });
}

fn dimAttr(ctx: *mlir.Context, dim: Dim) *const mlir.Attribute {
    return switch (dim) {
        inline else => |d| mlir.Attribute.parse(ctx, "#gpu<dim " ++ @tagName(d) ++ ">") catch unreachable,
    };
}

fn idOp(ctx: *mlir.Context, comptime name: []const u8, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, name, .{
        .results = .{ .flat = &.{.index(ctx)} },
        .attributes = &.{.named(ctx, "dimension", dimAttr(ctx, dim))},
        .location = location,
    });
}

pub fn thread_id(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.thread_id", dim, location);
}

pub fn block_id(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.block_id", dim, location);
}

pub fn block_dim(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.block_dim", dim, location);
}

pub fn grid_dim(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.grid_dim", dim, location);
}

pub fn barrier(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.barrier", .{ .location = location });
}

test {
    std.testing.refAllDecls(@This());
}
