//! Tenstorrent TT-MLIR tensor IR. `mlir/dialects/ttir` is Triton's dialect.
const std = @import("std");
const mlir = @import("mlir");

pub const dialects_needed = [_][]const u8{ "func", "arith", "ttir" };

pub const Binary = enum { add, subtract, multiply, divide, maximum, minimum, gt, ge, lt, le, eq, ne };

pub fn binary(ctx: *mlir.Context, op: Binary, lhs: *const mlir.Value, rhs: *const mlir.Value, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return switch (op) {
        inline else => |kind| .make(ctx, "ttir." ++ @tagName(kind), .{
            .operands = .{ .flat = &.{ lhs, rhs } },
            .results = .{ .flat = &.{result} },
            .location = loc,
        }),
    };
}

pub fn reshape(ctx: *mlir.Context, input: *const mlir.Value, shape: []const i32, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.reshape", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result} },
        .attributes = &.{.named(ctx, "shape", .intArray(ctx, i32, shape))},
        .location = loc,
    });
}

pub fn matmul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result: *const mlir.Type, transpose_lhs: bool, transpose_rhs: bool, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.matmul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result} },
        .attributes = &.{
            .named(ctx, "transpose_a", .boolean(ctx, transpose_lhs)),
            .named(ctx, "transpose_b", .boolean(ctx, transpose_rhs)),
        },
        .location = loc,
    });
}

pub fn sum(ctx: *mlir.Context, input: *const mlir.Value, axes: []const i32, keep_dim: bool, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.sum", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result} },
        .attributes = &.{
            .named(ctx, "dim_arg", .intArray(ctx, i32, axes)),
            .named(ctx, "keep_dim", .boolean(ctx, keep_dim)),
        },
        .location = loc,
    });
}

pub fn permute(ctx: *mlir.Context, input: *const mlir.Value, order: []const i64, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.permute", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result} },
        .attributes = &.{.named(ctx, "permutation", .denseArray(ctx, .i64, order))},
        .location = loc,
    });
}

pub fn sliceStatic(ctx: *mlir.Context, input: *const mlir.Value, begins: []const i32, ends: []const i32, steps: []const i32, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.slice_static", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result} },
        .attributes = &.{
            .named(ctx, "begins", .intArray(ctx, i32, begins)),
            .named(ctx, "ends", .intArray(ctx, i32, ends)),
            .named(ctx, "step", .intArray(ctx, i32, steps)),
        },
        .location = loc,
    });
}

pub fn concat(ctx: *mlir.Context, inputs: []const *const mlir.Value, axis: i32, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.concat", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = &.{result} },
        .attributes = &.{.named(ctx, "dim", .int(ctx, .i32, axis))},
        .location = loc,
    });
}

pub fn where(ctx: *mlir.Context, predicate: *const mlir.Value, on_true: *const mlir.Value, on_false: *const mlir.Value, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.where", .{
        .operands = .{ .flat = &.{ predicate, on_true, on_false } },
        .results = .{ .flat = &.{result} },
        .location = loc,
    });
}

test "registered TTIR verifies tensor operations" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |name| mlir.DialectHandle.fromString(name).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    try std.testing.expect(ctx.isRegisteredOperation("ttir.matmul"));
    const module = try mlir.Module.parse(ctx,
        \\module {
        \\  func.func @kernel(%a: tensor<2x32x32xf32>, %b: tensor<2x32x32xf32>) -> tensor<2x32x32xf32> {
        \\    %0 = "ttir.add"(%a, %b) : (tensor<2x32x32xf32>, tensor<2x32x32xf32>) -> tensor<2x32x32xf32>
        \\    return %0 : tensor<2x32x32xf32>
        \\  }
        \\}
    );
    defer module.deinit();
    try std.testing.expect(module.operation().verify());
}

test {
    std.testing.refAllDecls(@This());
}
