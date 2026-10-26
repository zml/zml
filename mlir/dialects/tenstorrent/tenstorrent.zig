//! Tenstorrent TT-MLIR tensor IR. `mlir/dialects/ttir` is Triton's dialect.
const std = @import("std");
const mlir = @import("mlir");

pub const dialects_needed = [_][]const u8{ "func", "arith", "ttir" };

pub const Binary = enum { add, subtract, multiply, divide, maximum, minimum, gt, ge, lt, le, eq, ne };

pub fn binary(ctx: *mlir.Context, op: Binary, lhs: *const mlir.Value, rhs: *const mlir.Value, result: *const mlir.Type, loc: *const mlir.Location) *mlir.Operation {
    return switch (op) {
        inline else => |kind| .make(ctx, "ttir." ++ (if (kind == .divide) "div" else @tagName(kind)), .{
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
        .attributes = &.{.named(ctx, "dim", .int(ctx, .si32, axis))},
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

pub const AttentionOptions = struct {
    is_causal: bool = true,
    scale: ?f32 = null,
};

/// FlashAttention-2: q=[B,Hq,Sq,D], k/v=[B,Hkv,Sk,D].
/// A mask broadcasts over batch/heads and requires is_causal=false.
pub fn attention(ctx: *mlir.Context, q: *const mlir.Value, k: *const mlir.Value, v: *const mlir.Value, mask: ?*const mlir.Value, options: AttentionOptions, loc: *const mlir.Location) *mlir.Operation {
    const operands = [_]*const mlir.Value{ q, k, v, mask orelse q };
    const attributes = [_]mlir.NamedAttribute{
        .named(ctx, "is_causal", .boolean(ctx, options.is_causal)),
        .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 1, if (mask != null) 1 else 0, 0 })),
        .named(ctx, "scale", .float(ctx, .f32, options.scale orelse 1)),
    };
    return .make(ctx, "ttir.scaled_dot_product_attention", .{
        .operands = .{ .flat = operands[0..if (mask != null) 4 else 3] },
        .results = .{ .flat = &.{q.type_()} },
        .attributes = attributes[0..if (options.scale != null) 3 else 2],
        .location = loc,
    });
}

/// Causal decode: q=[1,B,Hq,D], k/v=[B,Hkv,S,D], position=[B].
pub fn attentionDecode(ctx: *mlir.Context, q: *const mlir.Value, k: *const mlir.Value, v: *const mlir.Value, position: *const mlir.Value, loc: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "ttir.scaled_dot_product_attention_decode", .{
        .operands = .{ .flat = &.{ q, k, v, position } },
        .results = .{ .flat = &.{q.type_()} },
        .attributes = &.{
            .named(ctx, "is_causal", .boolean(ctx, true)),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 1, 0, 1, 0 })),
        },
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

test "every binary wrapper names a registered TTIR operation" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |name| mlir.DialectHandle.fromString(name).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc = mlir.Location.unknown(ctx);
    const input = mlir.Type.rankedTensor(&.{32}, mlir.Type.float(ctx, .f32));
    const predicate = mlir.Type.rankedTensor(&.{32}, mlir.Type.int(ctx, .i1));
    const block = mlir.Block.init(&.{ input, input }, &.{ loc, loc });
    defer block.deinit();
    inline for (std.meta.tags(Binary)) |kind| {
        const result = switch (kind) {
            .gt, .ge, .lt, .le, .eq, .ne => predicate,
            else => input,
        };
        const op = binary(ctx, kind, block.argument(0), block.argument(1), result, loc).appendTo(block);
        try std.testing.expect(op.verify());
    }
}
