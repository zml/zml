//! ROCm Flash Attention wrapper.
//!
//! Bridges the `attention.zig` full-sequence attention API to the Zig-native
//! Triton flash attention kernel (`flash_attn_fwd.zig`) via `zml.kernel.triton`.
//! The kernel targets MI300X (8 XCDs).

const std = @import("std");
const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const MhaFwd = @import("triton_kernels/flash_attn_fwd.zig").MhaFwd;

pub const Parameters = struct {
    is_causal: bool = false,

    pub const InitOptions = struct {};

    pub fn init(_: InitOptions) Parameters {
        return .{};
    }
};

/// Full-sequence attention via the ROCm Triton flash attention kernel.
///
/// Expects q/k/v with tags .{.b, .q/.k, .h, .hd} and bf16 dtype.
/// Returns output with the same shape as q.
pub fn fullSequenceAttention(q: Tensor, k: Tensor, v: Tensor, params: Parameters) Tensor {
    _ = params; // is_causal=false is baked into the kernel config

    const batch: i32 = @intCast(q.dim(.b));
    const seqlen_q: i32 = @intCast(q.dim(.q));
    const seqlen_k: i32 = @intCast(k.dim(.k));
    const num_q_heads: i32 = @intCast(q.dim(.h));
    const num_k_heads: i32 = @intCast(k.dim(.h));
    const head_dim: i32 = @intCast(q.dim(.hd));

    // Padded head dimension (next power of 2).
    const head_dim_pow2: i32 = blk: {
        var p: i32 = 1;
        while (p < head_dim) p <<= 1;
        break :blk p;
    };

    const BLOCK_M: i32 = 128;

    const sm_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const cfg: MhaFwd.Config = .{
        .dtype = .bf16,
        .NUM_Q_HEADS = num_q_heads,
        .NUM_K_HEADS = num_k_heads,
        .BLOCK_M = BLOCK_M,
        .BLOCK_N = 64,
        .BLOCK_DMODEL = head_dim,
        .BLOCK_DMODEL_POW2 = head_dim_pow2,
        .SEQLEN_Q = seqlen_q,
        .SEQLEN_K = seqlen_k,
        .sm_scale = sm_scale,
        .BATCH = batch,
        .NUM_XCD = 8,
        .PRELOAD_V = true,
        .IS_CAUSAL = false,
        .VARLEN = false,
        .IS_FP8 = false,
        .ENABLE_SINK = false,
        .SLIDING_WINDOW = 0,
        .HAS_PE = false,
        .USE_INT64_STRIDES = false,
    };

    // Flatten q/k/v to 1-D buffers (the kernel addresses via strides).
    const q_flat = q.reshape(.{@as(i64, batch) * num_q_heads * seqlen_q * head_dim}).withTags(.{.flat});
    const k_flat = k.reshape(.{@as(i64, batch) * num_k_heads * seqlen_k * head_dim}).withTags(.{.flat});
    const v_flat = v.reshape(.{@as(i64, batch) * num_k_heads * seqlen_k * head_dim}).withTags(.{.flat});

    // Dummy/unused inputs (not FP8, no ALiBi, no sink).
    const descale_q = Tensor.init(.{1}, .f32);
    const descale_k = Tensor.init(.{1}, .f32);
    const descale_v = Tensor.init(.{1}, .f32);
    const alibi_slopes = Tensor.init(.{num_q_heads}, .f32);
    const sink = Tensor.init(.{1}, .f32);

    // softmax_lse: written by kernel but not consumed downstream.
    const softmax_lse = Tensor.init(.{@as(i64, seqlen_q) * num_q_heads}, .f32);

    // Strides for BSHD layout: q[b, h, s, d]
    const stride_qz = Tensor.scalar(@as(i32, num_q_heads * seqlen_q * head_dim), .i32);
    const stride_qh = Tensor.scalar(@as(i32, seqlen_q * head_dim), .i32);
    const stride_qm = Tensor.scalar(head_dim, .i32);
    const stride_qk = Tensor.scalar(@as(i32, 1), .i32);

    const stride_kz = Tensor.scalar(@as(i32, num_k_heads * seqlen_k * head_dim), .i32);
    const stride_kh = Tensor.scalar(@as(i32, seqlen_k * head_dim), .i32);
    const stride_kn = Tensor.scalar(head_dim, .i32);
    const stride_kk = Tensor.scalar(@as(i32, 1), .i32);

    const stride_vz = Tensor.scalar(@as(i32, num_k_heads * seqlen_k * head_dim), .i32);
    const stride_vh = Tensor.scalar(@as(i32, seqlen_k * head_dim), .i32);
    const stride_vn = Tensor.scalar(head_dim, .i32);
    const stride_vk = Tensor.scalar(@as(i32, 1), .i32);

    // descale strides (unused, one element each).
    const stride_dqz = Tensor.scalar(@as(i32, 1), .i32);
    const stride_dkz = Tensor.scalar(@as(i32, 1), .i32);
    const stride_dvz = Tensor.scalar(@as(i32, 1), .i32);

    // Output strides (same layout as Q).
    const stride_oz = stride_qz;
    const stride_oh = stride_qh;
    const stride_om = stride_qm;
    const stride_on = stride_qk;

    // ALiBi strides (unused).
    const stride_az = Tensor.scalar(@as(i32, num_q_heads), .i32);
    const stride_ah = Tensor.scalar(@as(i32, 1), .i32);

    // LSE strides: [batch, heads, seqlen_q]
    const stride_lz = Tensor.scalar(@as(i32, num_q_heads * seqlen_q), .i32);
    const stride_lh = Tensor.scalar(seqlen_q, .i32);
    const stride_lm = Tensor.scalar(@as(i32, 1), .i32);

    // Runtime scalars.
    const sm_scale_tensor = Tensor.scalar(sm_scale, .f32);
    const cu_seqlens_q = Tensor.init(.{@as(i64, batch) + 1}, .i32);
    const cu_seqlens_k = Tensor.init(.{@as(i64, batch) + 1}, .i32);

    // Grid: one program per (batch * heads * ceil(seqlen_q / BLOCK_M)).
    const grid_x: i32 = batch * num_q_heads * @divTrunc(seqlen_q + BLOCK_M - 1, BLOCK_M);

    const out_buf_size: i64 = @as(i64, batch) * num_q_heads * seqlen_q * head_dim;

    const results = MhaFwd.Kernel.call(
        .{
            .q_ptr = q_flat,
            .k_ptr = k_flat,
            .v_ptr = v_flat,
            .descale_q_ptr = descale_q,
            .descale_k_ptr = descale_k,
            .descale_v_ptr = descale_v,
            .alibi_slopes_ptr = alibi_slopes,
            .softmax_lse_ptr = softmax_lse,
            .sink_ptr = sink,
            .stride_qz_in = stride_qz,
            .stride_qh_in = stride_qh,
            .stride_qm_in = stride_qm,
            .stride_qk_in = stride_qk,
            .stride_kz_in = stride_kz,
            .stride_kh_in = stride_kh,
            .stride_kn_in = stride_kn,
            .stride_kk_in = stride_kk,
            .stride_vz_in = stride_vz,
            .stride_vh_in = stride_vh,
            .stride_vn_in = stride_vn,
            .stride_vk_in = stride_vk,
            .stride_descale_q_z_in = stride_dqz,
            .stride_descale_k_z_in = stride_dkz,
            .stride_descale_v_z_in = stride_dvz,
            .stride_oz_in = stride_oz,
            .stride_oh_in = stride_oh,
            .stride_om_in = stride_om,
            .stride_on_in = stride_on,
            .stride_alibi_z_in = stride_az,
            .stride_alibi_h_in = stride_ah,
            .stride_lse_z_in = stride_lz,
            .stride_lse_h_in = stride_lh,
            .stride_lse_m_in = stride_lm,
            .sm_scale = sm_scale_tensor,
            .cu_seqlens_q = cu_seqlens_q,
            .cu_seqlens_k = cu_seqlens_k,
        },
        .{ .out = Shape.init(.{out_buf_size}, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ grid_x, 1, 1 },
            .num_warps = 4,
            .num_stages = 2,
        },
    );

    // Reshape flat output back to [batch, heads, seqlen_q, head_dim] then tag.
    return results.out
        .reshape(.{ @as(i64, batch), @as(i64, num_q_heads), @as(i64, seqlen_q), @as(i64, head_dim) })
        .withTags(.{ .b, .h, .q, .hd });
}
