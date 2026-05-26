//! TT-specific Tensor-level wrappers over `platforms/tt/ops`. Depends on
//! zml, so zml/ can't import this — call sites in models switch on
//! `platform.target` and invoke these for the TT path.

const std = @import("std");

const mlir = @import("mlir");
const tt_ops = @import("platforms/tt/ops");
const zml = @import("zml");

const Shape = zml.Shape;
const Tensor = zml.Tensor;

/// RMSNorm with affine weight via `tenstorrent.rms_norm` custom_call.
/// Math-decomposed RMSNorm trips `sdy.reshard` "duplicate axis ref" at
/// sharding propagation; the single op sidesteps it.
pub fn rmsNormFused(x: Tensor, weight: Tensor, eps: f32) Tensor {
    const ctx = zml.module.CompilationContext.current();
    const out_type = zml.mlir.Type.rankedTensor(ctx.mlir_ctx, x.shape());
    const result = tt_ops.rmsNormFused(ctx.mlir_ctx, ctx.currentScope().block, x.value(), weight.value(), out_type, x.dim(-1), eps);
    return Tensor._result(x.shape(), result);
}

/// `tt.sampling` — temperature scale + multinomial sample on pre-top-k'd
/// candidates in one op (kernel handles RNG via the `seed` attr). Inputs
/// `{values, indices}` tagged `{.b, .topk}` (rank-2); `k`/`p`/`temp` are
/// per-batch. Broken at `trace + opt>=1` (tt-xla #4570); workaround is
/// `ZML_TT_OPTIMIZATION_LEVEL=0`.
pub fn sampling(values: Tensor, indices: Tensor, opts: zml.nn.SamplingStrategy) Tensor {
    const ctx = zml.module.CompilationContext.current();
    const vals = values.transpose(.{ .b, .topk }).convert(.bf16);
    const idxs = indices.transpose(.{ .b, .topk });
    const B = vals.dim(.b);
    const k = Tensor.scalar(opts.topk, .u32).broad(Shape.init(.{ .b = B }, .u32));
    const p = Tensor.scalar(@as(f32, 1.0), .bf16).broad(Shape.init(.{ .b = B }, .bf16));
    const t = Tensor.scalar(opts.temperature, .bf16).broad(Shape.init(.{ .b = B }, .bf16));
    const out_shape = Shape.init(.{ .b = B }, .i32);
    const out_type = zml.mlir.Type.rankedTensor(ctx.mlir_ctx, out_shape);
    const result = tt_ops.sampling(ctx.mlir_ctx, ctx.currentScope().block, vals.value(), idxs.value(), k.value(), p.value(), t.value(), out_type, "0");
    return Tensor._result(out_shape, result);
}
