//! Registration of `KernelUnifiedAttention2dPtr` for the harness.
//!
//! Re-exports the production Kernel from
//! `zml/attention/triton_kernels/unified_attention.zig` (no fork, no
//! local copy) and declares 8 sweeps: the default config that production
//! callers (`pagedAttention2d`) typically launch, plus 7 fuzzer variants
//! covering the dispatch axes that `monorepo/llmd/select2dConfig`
//! actually triggers (decode vs prefill, GQA group sizes 1/4/8, head-dim
//! 64/128/256, sliding window on/off, long-prefill block_m=128).
//!
//! Sweeps mirror `examples/triton_emitter/kernels_zig.zig`'s
//! `withConfig(...)` + `variantOf(...)` table for `KernelUnifiedAttention2dPtr`.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;

// =============================================================================
// Re-export the production Kernel + declare sweeps.
// =============================================================================

pub const Kernel = zml.attention.triton_kernels.KernelUnifiedAttention2dPtr.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 16,              .block_m = 16,
        .use_fp8 = false,
    } },
    .{ .label = "dec_h128_g4", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = true,
    } },
    .{ .label = "pre_h128_g4", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    } },
    .{ .label = "pre_h128_g8", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 64,      .num_queries_per_kv = 8,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 2,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    } },
    .{ .label = "pre_h128_g4_long", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 32,              .block_m = 128,
        .use_fp8 = false,           .all_decode = false,
    } },
    .{ .label = "dec_h256_swa", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 4096,
        .block_q = 16,              .block_m = 64,
        .use_fp8 = false,           .all_decode = true,
    } },
    .{ .label = "pre_h256_swa", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 4096,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    } },
    .{ .label = "dec_h64_g1", .cfg = .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 16,      .num_queries_per_kv = 1,
        .block_size = 16,           .tile_size = 16,
        .head_size = 64,            .head_size_padded = 64,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 16,              .block_m = 16,
        .use_fp8 = false,           .all_decode = true,
    } },
};

// =============================================================================
// XLA-pipeline driver: synthetic Tensor signature + dummy shapes that span
// every sweep above (oversized: max num_query_heads=64, head_size_padded=256).
// XLA never launches the kernel — it only runs the codegen pipeline — so the
// runtime values don't have to be correct, only the dtypes/ranks.
// Adapted from `examples/triton_emitter/dump_via_xla.zig:KernelUnifiedAttention2dPtr`.
// =============================================================================

const MAX_NUM_TOKENS: i64 = 64;
const MAX_NUM_QUERY_HEADS: i64 = 64;
const MAX_HEAD_SIZE_PADDED: i64 = 256;
const MAX_NUM_BLOCKS: i64 = 64;
const MAX_BLOCK_SIZE: i64 = 16;
const MAX_Q_BUF: i64 = MAX_NUM_TOKENS * MAX_NUM_QUERY_HEADS * MAX_HEAD_SIZE_PADDED;
// num_kv_heads = num_query_heads / num_queries_per_kv. Worst case is
// max=64 / min=1 = 64; bound MAX_KV_HEADS at 16 (matches monorepo's
// realistic call space — none of our sweeps go higher).
const MAX_NUM_KV_HEADS: i64 = 16;
const MAX_KV_BUF: i64 = MAX_NUM_BLOCKS * MAX_NUM_KV_HEADS * MAX_BLOCK_SIZE * MAX_HEAD_SIZE_PADDED;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    query: Tensor, key_cache: Tensor, value_cache: Tensor, sink: Tensor,
    block_tables: Tensor, seq_lens: Tensor, alibi: Tensor, qq_bias: Tensor,
    scale: Tensor, k_scale: Tensor, v_scale: Tensor, out_scale: Tensor, softcap: Tensor,
    bt_stride: Tensor, q_s0: Tensor, q_s1: Tensor, o_s0: Tensor, o_s1: Tensor, qqb_s0: Tensor,
    k_s0: Tensor, k_s1: Tensor, k_s2: Tensor, v_s0: Tensor, v_s1: Tensor, v_s2: Tensor,
    qsl: Tensor, num_seqs: Tensor, _: Tensor,
) Tensor {
    return ops.triton(
        .{
            query, key_cache, value_cache, sink, block_tables, seq_lens, alibi, qq_bias,
            scale, k_scale, v_scale, out_scale, softcap,
            bt_stride, q_s0, q_s1, o_s0, o_s1, qqb_s0,
            k_s0, k_s1, k_s2, v_s0, v_s1, v_s2,
            qsl, num_seqs,
        },
        .{query.shape()},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 8, 1, 1 },
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
        // query, key_cache, value_cache
        ten1d(.bf16, MAX_Q_BUF), ten1d(.bf16, MAX_KV_BUF), ten1d(.bf16, MAX_KV_BUF),
        // sink, block_tables, seq_lens, alibi, qq_bias
        ten1d(.f32, MAX_NUM_QUERY_HEADS), ten1d(.i32, MAX_NUM_BLOCKS), ten1d(.i32, 1), ten1d(.f32, MAX_NUM_QUERY_HEADS), ten1d(.f32, 1),
        // scale, k_scale, v_scale, out_scale, softcap
        ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1),
        // bt_stride, q_s0..q_s1, o_s0..o_s1, qqb_s0
        ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
        // k_s0..k_s2, v_s0..v_s2
        ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
        // qsl, num_seqs, output_ptr placeholder
        ten1d(.i32, 2), ten1d(.i32, 1), ten1d(.bf16, MAX_Q_BUF),
    };
}
