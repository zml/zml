//! Registration of `_attn_fwd` (AMD flash-attention forward) for the harness.
//!
//! Re-exports the production Kernel from
//! `zml/attention/triton_kernels/attn_fwd.zig` and declares sweeps covering
//! the main dispatch axes: head-dim (64/128), GQA group sizes (1/4/8),
//! and block tiling (BLOCK_M=64/128, BLOCK_N=32/64).

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;

// =============================================================================
// Re-export the production Kernel + declare sweeps.
// =============================================================================

pub const Kernel = zml.attention.flash_attn_fwd.FlashAttnFwd.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    // Default: Llama-style, 32 Q heads / 8 KV heads, head_dim=128
    .{ .label = "default", .cfg = .{} },
    // GQA group=1 (MHA), head_dim=128
    .{ .label = "mha_h128", .cfg = .{
        .NUM_Q_HEADS = 32,
        .NUM_K_HEADS = 32,
        .BLOCK_DMODEL = 128,
        .BLOCK_DMODEL_POW2 = 128,
    } },
    // GQA group=4, head_dim=64 (smaller models)
    .{ .label = "gqa4_h64", .cfg = .{
        .NUM_Q_HEADS = 32,
        .NUM_K_HEADS = 8,
        .BLOCK_DMODEL = 64,
        .BLOCK_DMODEL_POW2 = 64,
    } },
    // GQA group=8, head_dim=128 (e.g. Llama-3 70B)
    .{ .label = "gqa8_h128", .cfg = .{
        .NUM_Q_HEADS = 64,
        .NUM_K_HEADS = 8,
        .BLOCK_DMODEL = 128,
        .BLOCK_DMODEL_POW2 = 128,
    } },
    // Smaller tile: BLOCK_M=64, BLOCK_N=32, head_dim=128
    .{ .label = "tile_64x32_h128", .cfg = .{
        .BLOCK_M = 64,
        .BLOCK_N = 32,
        .BLOCK_DMODEL = 128,
        .BLOCK_DMODEL_POW2 = 128,
    } },
    // MHA head_dim=64, smaller tile
    .{ .label = "mha_h64_tile64x32", .cfg = .{
        .NUM_Q_HEADS = 16,
        .NUM_K_HEADS = 16,
        .BLOCK_M = 64,
        .BLOCK_N = 32,
        .BLOCK_DMODEL = 64,
        .BLOCK_DMODEL_POW2 = 64,
    } },
};

// =============================================================================
// XLA-pipeline driver
// =============================================================================

const MAX_BATCH: i64 = 1;
const MAX_NUM_Q_HEADS: i64 = 64;
const MAX_NUM_K_HEADS: i64 = 32;
const MAX_SEQLEN: i64 = 512;
const MAX_HEAD_DIM: i64 = 128;
const MAX_Q_BUF: i64 = MAX_BATCH * MAX_NUM_Q_HEADS * MAX_SEQLEN * MAX_HEAD_DIM;
const MAX_KV_BUF: i64 = MAX_BATCH * MAX_NUM_K_HEADS * MAX_SEQLEN * MAX_HEAD_DIM;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    descale_q: Tensor,
    descale_k: Tensor,
    descale_v: Tensor,
    alibi_slopes: Tensor,
    s_dmask: Tensor,
    dropout_mask: Tensor,
    softmax_lse: Tensor,
    sink: Tensor,
    // strides (i32 scalars)
    stride_qz: Tensor,
    stride_qh: Tensor,
    stride_qm: Tensor,
    stride_qk: Tensor,
    stride_kz: Tensor,
    stride_kh: Tensor,
    stride_kn: Tensor,
    stride_kk: Tensor,
    stride_vz: Tensor,
    stride_vh: Tensor,
    stride_vn: Tensor,
    stride_vk: Tensor,
    stride_descale_q_z: Tensor,
    stride_descale_k_z: Tensor,
    stride_descale_v_z: Tensor,
    stride_oz: Tensor,
    stride_oh: Tensor,
    stride_om: Tensor,
    stride_on: Tensor,
    stride_alibi_z: Tensor,
    stride_alibi_h: Tensor,
    stride_sd_z: Tensor,
    stride_sd_h: Tensor,
    stride_sd_m: Tensor,
    stride_sd_n: Tensor,
    stride_lse_z: Tensor,
    stride_lse_h: Tensor,
    stride_lse_m: Tensor,
    // other scalars
    sm_scale: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    dropout_p: Tensor,
    philox_seed: Tensor,
    philox_offset_base: Tensor,
    seqlen_q_t: Tensor,
    seqlen_k_t: Tensor,
    batch_t: Tensor,
) Tensor {
    return ops.triton(.{
        q,                  k,                  v,
        descale_q,          descale_k,          descale_v,
        alibi_slopes,       s_dmask,            dropout_mask,
        softmax_lse,        sink,               stride_qz,
        stride_qh,          stride_qm,          stride_qk,
        stride_kz,          stride_kh,          stride_kn,
        stride_kk,          stride_vz,          stride_vh,
        stride_vn,          stride_vk,          stride_descale_q_z,
        stride_descale_k_z, stride_descale_v_z, stride_oz,
        stride_oh,          stride_om,          stride_on,
        stride_alibi_z,     stride_alibi_h,     stride_sd_z,
        stride_sd_h,        stride_sd_m,        stride_sd_n,
        stride_lse_z,       stride_lse_h,       stride_lse_m,
        sm_scale,           cu_seqlens_q,       cu_seqlens_k,
        dropout_p,          philox_seed,        philox_offset_base,
        seqlen_q_t,         seqlen_k_t,         batch_t,
    }, .{q.shape()}, .{
        .name = Kernel.name,
        .ir = active_ttir,
        .grid = .{ 1 * 32 * 4, 1, 1 },
        .num_warps = 4,
        .num_stages = 1,
    })[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    const t = struct {
        fn scalar(comptime dt: zml.DataType) Tensor {
            return Tensor.init(.{}, dt);
        }
        fn buf(comptime dt: zml.DataType, n: i64) Tensor {
            return Tensor.init(.{n}, dt);
        }
    };
    return .{
        t.buf(.bf16, MAX_Q_BUF), // q_ptr
        t.buf(.bf16, MAX_KV_BUF), // k_ptr
        t.buf(.bf16, MAX_KV_BUF), // v_ptr
        t.buf(.f32, MAX_BATCH), // descale_q_ptr
        t.buf(.f32, MAX_BATCH), // descale_k_ptr
        t.buf(.f32, MAX_BATCH), // descale_v_ptr
        t.buf(.f32, MAX_NUM_Q_HEADS), // alibi_slopes_ptr
        t.buf(.f32, MAX_Q_BUF), // s_dmask_ptr
        t.buf(.f32, MAX_Q_BUF), // dropout_mask_ptr
        t.buf(.f32, MAX_NUM_Q_HEADS * MAX_SEQLEN), // softmax_lse_ptr
        t.buf(.f32, MAX_Q_BUF), // sink_ptr
        // strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // q strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // k strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // v strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // descale strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // o strides
        t.scalar(.i32), t.scalar(.i32), // alibi strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // sd strides
        t.scalar(.i32), t.scalar(.i32), t.scalar(.i32), // lse strides
        // other scalars
        t.scalar(.f32), // sm_scale
        t.buf(.i32, 2), // cu_seqlens_q
        t.buf(.i32, 2), // cu_seqlens_k
        t.scalar(.f32), // dropout_p
        t.scalar(.i64), // philox_seed
        t.scalar(.i32), // philox_offset_base
        t.scalar(.i32), // SEQLEN_Q
        t.scalar(.i32), // SEQLEN_K
        t.scalar(.i32), // BATCH
    };
}
