const std = @import("std");

const mlir = @import("mlir");
const tt_ops = @import("platforms/tt/ops");

const Compiler = @import("../Compiler.zig");
const mlirx = @import("../mlirx.zig");
const zml = @import("../zml.zig");

/// SDPA via the TTNN-native `tt.scaled_dot_product_attention[_decode]`
/// custom_calls. tt-mlir handles the scale (`1/sqrt(hd)`) and the causal mask.
///
/// Decode (`q.dim(.q) == 1`) emits flash-decode with `token_index` as the
/// per-batch `cur_pos`. Prefill emits causal flash-attention and assumes the
/// prompt starts at position 0.
///
/// Layouts are lowered to `[B,H,S,D]` (decode `q` is `[1,B,H,D]`).
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const compiler = Compiler.current();
    const mlir_ctx = compiler.mlir_ctx;

    const is_decode = q.dim(.q) == 1;
    const had_b = q.shape().hasTag(.b) != null;

    const q4 = if (is_decode)
        (if (had_b) q.transpose(.{ .q, .b, .h, .hd }) else q.insertAxes(.h, .{.b}))
    else
        (if (had_b) q.transpose(.{ .b, .h, .q, .hd }) else q.insertAxes(0, .{.b}).transpose(.{ .b, .h, .q, .hd }));
    const k4 = if (had_b) k.transpose(.{ .b, .h, .k, .hd }) else k.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });
    const v4 = if (had_b) v.transpose(.{ .b, .h, .k, .hd }) else v.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });

    // tt-mlir requires `cur_pos.batch == query.batch`, so a scalar position is
    // broadcast over the batch.
    const cur_pos: ?*const mlir.Value = if (!is_decode) null else cp_blk: {
        const cp = if (had_b and token_index.shape().hasTag(.b) != null)
            token_index
        else cp_inner: {
            const with_b = token_index.insertAxes(.last, .{.b});
            break :cp_inner if (had_b)
                with_b.broad(zml.Shape.init(.{ .b = q.dim(.b) }, with_b.dtype()))
            else
                with_b;
        };
        break :cp_blk cp.value();
    };

    const out_type = mlirx.Type.rankedTensor(mlir_ctx, q4.shape());
    const value = tt_ops.scaledDotProductAttention(mlir_ctx, compiler.currentScope().block, q4.value(), k4.value(), v4.value(), cur_pos, out_type, is_decode);
    const result = zml.Tensor._result(q4.shape(), value);
    return if (had_b)
        result.transpose(.{ .b, .q, .h, .hd })
    else if (is_decode)
        result.squeeze(.b)
    else
        result.transpose(.{ .q, .b, .h, .hd }).squeeze(.b);
}
