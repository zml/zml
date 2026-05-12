//! Registration of `_attn_fwd` (AMD flash-attention forward) for the harness.
//!
//! Re-exports the production Kernel from
//! `zml/attention/triton_kernels/flash_attn_fwd.zig` (no fork, no local copy)
//! and declares one sweep: the default non-causal, non-varlen, non-fp8
//! config with bf16, BLOCK_M=128, BLOCK_N=64, head_dim=128, batch=1,
//! MI300X 8-XCD layout.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

// =============================================================================
// Re-export the production Kernel + declare sweeps.
// =============================================================================

pub const Kernel = zml.attention.triton_mha_kernels.MhaFwd.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{
        .label = "default",
        .cfg = .{
            .dtype = .bf16,
            .NUM_Q_HEADS = 32,
            .NUM_K_HEADS = 8,
            .BLOCK_M = 128,
            .BLOCK_N = 64,
            .BLOCK_DMODEL = 128,
            .BLOCK_DMODEL_POW2 = 128,
            .SEQLEN_Q = 1024,
            .SEQLEN_K = 1024,
            .sm_scale = 0.08838834764831843, // 1/sqrt(128)
            .BATCH = 1,
            .NUM_XCD = 8,
            .PRELOAD_V = true,
            .IS_CAUSAL = false,
            .VARLEN = false,
            .IS_FP8 = false,
            .ENABLE_SINK = false,
            .SLIDING_WINDOW = 0,
            .HAS_PE = false,
            .USE_INT64_STRIDES = false,
        },
    },
};

// =============================================================================
// XLA-pipeline driver: synthetic Tensor signature + dummy shapes.
// XLA never launches the kernel — it only runs the codegen pipeline — so the
// runtime values don't have to be correct, only the dtypes/ranks.
// =============================================================================

const BATCH: i64 = 1;
const SEQLEN_Q: i64 = 1024;
const SEQLEN_K: i64 = 1024;
const NUM_Q_HEADS: i64 = 32;
const NUM_K_HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;

// Flat buffer sizes for Q/K/V/O tensors: BATCH * HEADS * SEQ * DIM
const Q_BUF: i64 = BATCH * NUM_Q_HEADS * SEQLEN_Q * HEAD_DIM;
const K_BUF: i64 = BATCH * NUM_K_HEADS * SEQLEN_K * HEAD_DIM;
const V_BUF: i64 = BATCH * NUM_K_HEADS * SEQLEN_K * HEAD_DIM;
const O_BUF: i64 = BATCH * NUM_Q_HEADS * SEQLEN_Q * HEAD_DIM;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    // Tensor pointers
    q: Tensor,
    k: Tensor,
    v: Tensor,
    descale_q: Tensor,
    descale_k: Tensor,
    descale_v: Tensor,
    alibi_slopes: Tensor,
    softmax_lse: Tensor,
    sink: Tensor,
    // Strides — q
    stride_qz: Tensor,
    stride_qh: Tensor,
    stride_qm: Tensor,
    stride_qk: Tensor,
    // Strides — k
    stride_kz: Tensor,
    stride_kh: Tensor,
    stride_kn: Tensor,
    stride_kk: Tensor,
    // Strides — v
    stride_vz: Tensor,
    stride_vh: Tensor,
    stride_vn: Tensor,
    stride_vk: Tensor,
    // Strides — descale
    stride_dqz: Tensor,
    stride_dkz: Tensor,
    stride_dvz: Tensor,
    // Strides — output
    stride_oz: Tensor,
    stride_oh: Tensor,
    stride_om: Tensor,
    stride_on: Tensor,
    // Strides — alibi
    stride_az: Tensor,
    stride_ah: Tensor,
    // Strides — lse
    stride_lz: Tensor,
    stride_lh: Tensor,
    stride_lm: Tensor,
    // Runtime scalars
    sm_scale: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    // Output placeholder
    _: Tensor,
) Tensor {
    return ops.triton(
        .{
            q,            k,            v,
            descale_q,    descale_k,    descale_v,
            alibi_slopes, softmax_lse,  sink,
            stride_qz,    stride_qh,    stride_qm,
            stride_qk,    stride_kz,    stride_kh,
            stride_kn,    stride_kk,    stride_vz,
            stride_vh,    stride_vn,    stride_vk,
            stride_dqz,   stride_dkz,   stride_dvz,
            stride_oz,    stride_oh,    stride_om,
            stride_on,    stride_az,    stride_ah,
            stride_lz,    stride_lh,    stride_lm,
            sm_scale,     cu_seqlens_q, cu_seqlens_k,
        },
        .{Shape.init(.{O_BUF}, .bf16)},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ @as(i32, @intCast(BATCH * NUM_Q_HEADS * @divTrunc(SEQLEN_Q + 128 - 1, 128))), 1, 1 },
            .num_warps = 4,
            .num_stages = 2,
        },
    )[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    const ten1d = struct {
        fn t(comptime dt: zml.DataType, n: i64) Tensor {
            return Tensor.init(.{n}, dt);
        }
    }.t;
    const s = struct {
        fn t() Tensor {
            return Tensor.init(.{}, .i32);
        }
    }.t;
    return .{
        // q, k, v
        ten1d(.bf16, Q_BUF),      ten1d(.bf16, K_BUF),                 ten1d(.bf16, V_BUF),
        // descale_q, descale_k, descale_v
        ten1d(.f32, 1),           ten1d(.f32, 1),                      ten1d(.f32, 1),
        // alibi_slopes, softmax_lse, sink
        ten1d(.f32, NUM_Q_HEADS), ten1d(.f32, SEQLEN_Q * NUM_Q_HEADS), ten1d(.f32, 1),
        // strides — q (4)
        s(),                      s(),                                 s(),
        s(),
        // strides — k (4)
                             s(),                                 s(),
        s(),                      s(),
        // strides — v (4)
                                        s(),
        s(),                      s(),                                 s(),
        // strides — descale (3)
        s(),                      s(),                                 s(),
        // strides — output (4)
        s(),                      s(),                                 s(),
        s(),
        // strides — alibi (2)
                             s(),                                 s(),
        // strides — lse (3)
        s(),                      s(),                                 s(),
        // sm_scale, cu_seqlens_q, cu_seqlens_k
        Tensor.init(.{}, .f32),   ten1d(.i32, BATCH + 1),              ten1d(.i32, BATCH + 1),
        // output placeholder
        ten1d(.bf16, O_BUF),
    };
}
