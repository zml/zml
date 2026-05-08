//! Registration of `KernelUnifiedAttention3dPtr` for the harness.
//!
//! Re-exports the production Kernel from
//! `zml/attention/triton_kernels/unified_attention.zig`. Sweeps mirror
//! `examples/triton_emitter/kernels_zig.zig` lines 107-125 (default) +
//! 225-265 (4 fuzzer variants over segment counts {16, 32, 64} and
//! head_size {128, 256}).

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub const Kernel = zml.attention.triton_kernels.KernelUnifiedAttention3dPtr.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 16,              .block_m = 16,
        .num_segments_per_seq = 4,
    } },
    .{ .label = "pre_h128_g4_seg16", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 16,
    } },
    .{ .label = "pre_h128_g8_seg32", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 64,      .num_queries_per_kv = 8,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 2,               .block_m = 16,
        .num_segments_per_seq = 32,
    } },
    .{ .label = "dec_h128_g4_seg64", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 64,
        .all_decode = true,
    } },
    .{ .label = "pre_h256_seg16", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 16,
    } },
};

// =============================================================================
// XLA-pipeline driver. Synthetic shapes oversized to span every sweep
// (max num_query_heads=64, head_size_padded=256, num_segments_per_seq=128).
// =============================================================================

const MAX_NUM_TOKENS: i64 = 64;
const MAX_NUM_QUERY_HEADS: i64 = 64;
const MAX_HEAD_SIZE_PADDED: i64 = 256;
const MAX_NUM_BLOCKS: i64 = 64;
const MAX_BLOCK_SIZE: i64 = 16;
const MAX_NUM_SEGMENTS: i64 = 128;
const MAX_NUM_KV_HEADS: i64 = 16;
const MAX_Q_BUF: i64 = MAX_NUM_TOKENS * MAX_NUM_QUERY_HEADS * MAX_HEAD_SIZE_PADDED;
const MAX_KV_BUF: i64 = MAX_NUM_BLOCKS * MAX_NUM_KV_HEADS * MAX_BLOCK_SIZE * MAX_HEAD_SIZE_PADDED;
const MAX_SEGM_BASE: i64 = MAX_NUM_TOKENS * MAX_NUM_QUERY_HEADS * MAX_NUM_SEGMENTS;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    query: Tensor, key_cache: Tensor, value_cache: Tensor, sink: Tensor,
    block_tables: Tensor, seq_lens: Tensor, alibi: Tensor, qq_bias: Tensor,
    scale: Tensor, k_scale: Tensor, v_scale: Tensor, softcap: Tensor,
    bt_stride: Tensor, q_s0: Tensor, q_s1: Tensor, qqb_s0: Tensor,
    k_s0: Tensor, k_s1: Tensor, k_s2: Tensor, v_s0: Tensor, v_s1: Tensor, v_s2: Tensor,
    qsl: Tensor, num_seqs: Tensor, _: Tensor, _: Tensor, _: Tensor,
) Tensor {
    // Stub output shape — XLA only runs the codegen pipeline, not actual
    // launch, so the shape just has to be consistent with the kernel.
    const segm = Shape.init(.{64 * 32 * 4}, .f32);
    return ops.triton(
        .{
            query, key_cache, value_cache, sink, block_tables, seq_lens, alibi, qq_bias,
            scale, k_scale, v_scale, softcap,
            bt_stride, q_s0, q_s1, qqb_s0,
            k_s0, k_s1, k_s2, v_s0, v_s1, v_s2,
            qsl, num_seqs,
        },
        .{segm},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 1, 8, 4 },
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
    return .{
        // query, key_cache, value_cache.
        ten1d(.bf16, MAX_Q_BUF), ten1d(.bf16, MAX_KV_BUF), ten1d(.bf16, MAX_KV_BUF),
        // sink, block_tables, seq_lens, alibi, qq_bias.
        ten1d(.f32, MAX_NUM_QUERY_HEADS), ten1d(.i32, MAX_NUM_BLOCKS), ten1d(.i32, 1),
        ten1d(.f32, MAX_NUM_QUERY_HEADS), ten1d(.f32, 1),
        // scale, k_scale, v_scale, softcap.
        ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1),
        // bt_stride, q_s0..q_s1, qqb_s0.
        ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
        // k_s0..k_s2, v_s0..v_s2.
        ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
        // qsl, num_seqs, segm_output, segm_max, segm_expsum (last 3 are output placeholders).
        ten1d(.i32, 2), ten1d(.i32, 1),
        ten1d(.f32, MAX_SEGM_BASE * MAX_HEAD_SIZE_PADDED), ten1d(.f32, MAX_SEGM_BASE), ten1d(.f32, MAX_SEGM_BASE),
    };
}
