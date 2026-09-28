const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const mlirx = @import("../mlirx.zig");
const zml = @import("../zml.zig");

/// Causal FlashAttention on shard-local heads. The payload is typed MLIR
/// bytecode; the plugin verifies and inlines it before native TCL lowering.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const Inputs = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        offset: zml.Tensor,
        unsigned_offset: bool,

        fn body(self: @This(), output: zml.Shape) zml.Tensor {
            const grouped = self.q.splitAxis(.h, .{ .h = self.k.dim(.h), .hq = .auto });
            const batched = self.q.shape().hasTags(.{.b});
            const qc = if (batched) grouped.transpose(.{ .q, .b, .h, .hq, .hd }).merge(.{ .h = .{ .b, .h } }) else grouped.transpose(.{ .q, .h, .hq, .hd });
            const kc = if (batched) self.k.transpose(.{ .k, .b, .h, .hd }).merge(.{ .h = .{ .b, .h } }) else self.k.transpose(.{ .k, .h, .hd });
            const vc = if (batched) self.v.transpose(.{ .k, .b, .h, .hd }).merge(.{ .h = .{ .b, .h } }) else self.v.transpose(.{ .k, .h, .hd });
            const offset = if (batched) self.offset.broad(zml.Shape.init(.{ .b = self.q.dim(.b), .h = self.k.dim(.h) }, .i32)).merge(.{ .h = .{ .b, .h } }) else self.offset;
            var result = kernel(qc, kc, vc, offset, self.unsigned_offset);
            if (batched) result = result.splitAxis(.h, .{ .b = self.q.dim(.b), .h = self.k.dim(.h) });
            return result.transpose(grouped.shape()).merge(.{ .h = .{ .h, .hq } }).transpose(output);
        }
    };
    return zml.ops.manualComputation(Inputs.body, .{
        .q = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated }),
        .k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated }),
        .v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated }),
        .offset = token_index.convert(.i32),
        .unsigned_offset = token_index.dtype() == .u32,
    }, q.shape().withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated }));
}

fn kernel(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, offset: zml.Tensor, unsigned_offset: bool) zml.Tensor {
    const compiler = zml.Compiler.current();
    const ctx = compiler.mlir_ctx;
    const loc = compiler.location;
    const previous = ctx.allowUnregisteredDialects();
    ctx.setAllowUnregisteredDialects(true);
    defer ctx.setAllowUnregisteredDialects(previous);
    const module = mlir.Module.init(loc);
    defer module.deinit();
    const tensors = [_]zml.Tensor{ q, k, v, offset };
    var types: [4]*const mlir.Type = undefined;
    for (tensors, 0..) |t, i| {
        types[i] = mlirx.Type.rankedTensor(ctx, t.shape());
    }
    const block = mlir.Block.init(&types, &.{ loc, loc, loc, loc });
    const op = mlir.Operation.make(ctx, "tcl.flash_attention", .{
        .operands = .{ .flat = &.{ block.argument(0), block.argument(1), block.argument(2), block.argument(3) } },
        .results = .{ .flat = &.{types[0]} },
        .location = loc,
        .attributes = &.{.named(ctx, "unsigned_offset", .boolean(ctx, unsigned_offset))},
        .verify = false, // The plugin owns the TCL dialect and verifies the payload.
    }).appendTo(block);
    _ = dialects.func.return_(ctx, op.result(0), loc).appendTo(block);
    _ = dialects.func.func(ctx, .{ .name = "main", .block = block, .location = loc }).appendTo(module.body());
    return zml.ops.furiosaTcl(.{ q, k, v, offset }, q.shape(), module);
}
