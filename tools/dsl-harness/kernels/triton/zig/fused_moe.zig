//! Registration of `fused_moe_kernel` for the harness.
//!
//! Re-exports the production Kernel from `zml.moe.triton_kernels`
//! (`FusedMoe`). One default sweep matching the bf16 / no-quant / no-bias
//! path that `zml.moe.fusedExpertsImpl` actually launches (mirrors the
//! `withConfig(...)` in `examples/triton_emitter/kernels_zig.zig`).
//!
//! Known divergence vs. the Python reference (`moe.py:fused_moe_kernel`,
//! vLLM-derived): the Python source loads every `stride_*_ptr` (incl. the
//! quant ones) unconditionally, while this DSL port only loads the strides
//! the bf16 path uses. The harness diff will surface that.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub const Kernel = zml.moe.triton_kernels.FusedMoe.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .a_dtype = .bf16,
        .b_dtype = .bf16,
        .c_dtype = .bf16,
        .a_scale_dtype = null,
        .b_scale_dtype = null,
        .b_bias_dtype = null,
        .topk_weights_dtype = null,
        .block_size_m = 64,
        .block_size_n = 64,
        .block_size_k = 32,
        .group_size_m = 4,
        .top_k = 2,
        .naive_block_assignment = false,
        .mul_routed_weight = true,
        .compute_type = .bf16,
        .use_fp8_w8a8 = false,
        .use_int8_w8a8 = false,
        .use_int8_w8a16 = false,
        .per_channel_quant = false,
        .has_bias = false,
    } },
};

// =============================================================================
// XLA-pipeline driver. 24 inputs + 1 output (`c`) → 25 `tt.func` params.
// Shapes are synthetic placeholders; XLA only runs the codegen pipeline.
// =============================================================================

const M: i64 = 256;
const N: i64 = 1024;
const K: i64 = 1024;
const E: i64 = 8;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    a: Tensor, b: Tensor, b_bias: Tensor, a_scale: Tensor, b_scale: Tensor,
    topk_weights: Tensor, sorted_token_ids: Tensor, expert_ids: Tensor, num_tokens_post_padded: Tensor,
    n_ptr: Tensor, k_ptr: Tensor, em_ptr: Tensor, num_valid_tokens: Tensor,
    stride_am: Tensor, stride_be: Tensor, stride_bn: Tensor, stride_cm: Tensor,
    stride_asm: Tensor, stride_ask: Tensor, stride_bse: Tensor, stride_bsk: Tensor, stride_bsn: Tensor,
    stride_bbe: Tensor, stride_bbn: Tensor,
    _: Tensor,
) Tensor {
    return ops.triton(
        .{
            a, b, b_bias, a_scale, b_scale,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            n_ptr, k_ptr, em_ptr, num_valid_tokens,
            stride_am, stride_be, stride_bn, stride_cm,
            stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn,
            stride_bbe, stride_bbn,
        },
        .{Shape.init(.{ M, N }, .bf16)},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 16, 1, 1 },
            .num_warps = 4,
            .num_stages = 2,
        },
    )[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    const p = struct {
        fn t() Tensor {
            return Tensor.init(.{1}, .i64);
        }
    }.t;
    return .{
        Tensor.init(.{ M, K }, .bf16), // a_ptr
        Tensor.init(.{ E, N, K }, .bf16), // b_ptr
        Tensor.init(.{ E, N }, .bf16), // b_bias_ptr
        Tensor.init(.{1}, .f32), // a_scale_ptr
        Tensor.init(.{1}, .f32), // b_scale_ptr
        Tensor.init(.{M}, .f32), // topk_weights_ptr
        Tensor.init(.{M}, .i32), // sorted_token_ids_ptr
        Tensor.init(.{ E * 4 }, .i32), // expert_ids_ptr
        Tensor.init(.{1}, .i32), // num_tokens_post_padded_ptr
        p(), p(), p(), p(), // N_ptr, K_ptr, EM_ptr, num_valid_tokens_ptr
        p(), p(), p(), p(), // stride_am_ptr, stride_be_ptr, stride_bn_ptr, stride_cm_ptr
        p(), p(), p(), p(), p(), // stride_asm_ptr, stride_ask_ptr, stride_bse_ptr, stride_bsk_ptr, stride_bsn_ptr
        p(), p(), // stride_bbe_ptr, stride_bbn_ptr
        Tensor.init(.{ M, N }, .bf16), // c_ptr placeholder
    };
}
