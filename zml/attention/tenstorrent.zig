const std = @import("std");

const mlir = @import("mlir");
const tt_ops = @import("platforms/tt/ops");

const zml = @import("../zml.zig");
const mlirx = @import("../mlirx.zig");
const CompilationContext = @import("../module.zig").CompilationContext;

/// Backend `.tenstorrent`. Both siblings are empty — tt-mlir's SDPA op handles
/// workspace, scale (`1/sqrt(D)`), and mask generation internally.
pub const Parameters = struct {
    pub const InitOptions = struct {};
    pub fn init(_: InitOptions) Parameters {
        return .{};
    }
};

pub const Metadata = struct {
    pub const InitOptions = struct {};
    pub fn init(_: InitOptions) Metadata {
        return .{};
    }
};

/// SDPA via `tt.scaled_dot_product_attention[_decode]` custom_call. Branches
/// on `q.dim(.q) == 1`: decode emits Flash-Decode, prefill emits Flash-Attn-2.
/// Layouts: q,k,v lowered to [B,H,S,D] (decode q is [1,B,H,D]); decode passes
/// `cur_pos[B]` as a 4th operand.
pub fn attention(
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    token_index: zml.Tensor,
    _: Metadata,
    _: Parameters,
) zml.Tensor {
    const ctx = CompilationContext.current();
    const mlir_ctx = ctx.mlir_ctx;

    const is_decode = q.dim(.q) == 1;
    const had_b = q.shape().hasTag(.b) != null;

    const q4 = if (is_decode)
        (if (had_b) q.transpose(.{ .q, .b, .h, .hd }) else q.insertAxes(.h, .{.b}))
    else
        (if (had_b) q.transpose(.{ .b, .h, .q, .hd }) else q.insertAxes(0, .{.b}).transpose(.{ .b, .h, .q, .hd }));
    const k4 = if (had_b) k.transpose(.{ .b, .h, .k, .hd }) else k.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });
    const v4 = if (had_b) v.transpose(.{ .b, .h, .k, .hd }) else v.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });

    // Decode passes `cur_pos` as a 4th operand; ttir requires
    // `cur_pos.batch == query.batch` so broadcast scalar→batch when batched.
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
    const value = tt_ops.scaledDotProductAttention(mlir_ctx, ctx.currentScope().block, q4.value(), k4.value(), v4.value(), cur_pos, out_type, is_decode);
    const result = zml.Tensor._result(q4.shape(), value);
    return if (had_b)
        result.transpose(.{ .b, .q, .h, .hd })
    else if (is_decode)
        result.squeeze(.b)
    else
        result.transpose(.{ .q, .b, .h, .hd }).squeeze(.b);
}

/// Scatter-based KV-cache update on the `.k` axis.
/// TT-FIX: explicit 1-D position index — a scalar would make `scatterSlices`
/// degrade to `dynamic_update_slice`, which tt-mlir's `CacheFillUpdatePattern`
/// does not match. As a real `stablehlo.scatter`, it fuses into the native
/// `ttir.update_cache` (n==1, decode) or `ttir.fill_cache` (n>1, prefill).
/// `pos` must trace back to a rank-1 function input.
pub fn updateKvCache(cache: zml.Tensor, new: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    const updates = new.convert(cache.dtype()).transpose(cache.shape());
    const n = updates.dim(.k);
    const positions = zml.Tensor.arange(.{ .end = n }, pos.dtype())
        .withTags(.{.k})
        .add(pos.broad(zml.Shape.init(.{ .k = n }, pos.dtype())));
    return cache.scatterSlices(
        .{ .k = positions },
        updates,
        .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
}
