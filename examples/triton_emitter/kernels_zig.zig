//! Aggregator for the Zig-side Triton kernels in this example.
//!
//! Simple kernels (vector_add, softmax, etc.) live in `kernels_zig/*.zig`
//! locally — they're standalone demonstrations of the DSL. MoE kernels are
//! the **production** declarations from `zml.moe.triton_kernels` (one
//! source of truth for both production model code and this dump tool).
//!
//! The dump driver iterates `KERNELS` and calls `K.emit(allocator, ctx, cfg)`
//! for each. Comptime-known dump configs are inlined as `withConfig(K, .{...})`.

const std = @import("std");

const mlir = @import("mlir");

const zml = @import("zml");
const moe = zml.moe.triton_kernels;
const attn = zml.attention.triton_kernels;

const VectorAdd = @import("kernels_zig/vector_add.zig").VectorAdd;
const VectorExpFwd = @import("kernels_zig/vector_exp.zig").VectorExpFwd;
const VectorExpBwd = @import("kernels_zig/vector_exp.zig").VectorExpBwd;
const LowMemDropout = @import("kernels_zig/low_mem_dropout.zig").LowMemDropout;
const SumScalar = @import("kernels_zig/sum_scalar.zig").SumScalar;
const Softmax = @import("kernels_zig/softmax.zig").Softmax;
const FusedRecurrentGatedDeltaRule = @import("kernels_zig/gdn.zig").FusedRecurrentGatedDeltaRule;

pub const Entry = struct {
    name: []const u8,
    emit: *const fn (std.mem.Allocator, *mlir.Context) anyerror![:0]const u8,
};

/// Lift a kernel-wrapper type (`struct { Cfg, Kernel = tri.Kernel(...), run }`)
/// with a default-constructible `Cfg` to the dump driver's `Entry` shape.
fn defaults(comptime W: type) Entry {
    const E = struct {
        fn emit(allocator: std.mem.Allocator, ctx: *mlir.Context) anyerror![:0]const u8 {
            return W.Kernel.emit(allocator, ctx, .{});
        }
    };
    return .{ .name = W.Kernel.name, .emit = E.emit };
}

/// Same, but with an explicit `Cfg` value (for kernels whose required
/// fields have no defaults).
fn withConfig(comptime W: type, comptime cfg: W.Kernel.Config) Entry {
    const E = struct {
        fn emit(allocator: std.mem.Allocator, ctx: *mlir.Context) anyerror![:0]const u8 {
            return W.Kernel.emit(allocator, ctx, cfg);
        }
    };
    return .{ .name = W.Kernel.name, .emit = E.emit };
}

/// Like `withConfig`, but suffixes the **filename** with `__<label>` while
/// keeping the kernel's inner `tt.func` symbol identical (= `W.Kernel.name`).
/// Used by the unified-attention fuzzer to register multiple Config2D /
/// Config3D / ConfigReduce variants under the same Python source kernel —
/// each variant's TTIR ends up in `<W.Kernel.name>__<label>.ttir` but the
/// body still says `tt.func @<W.Kernel.name>`, so `compare_ir.py` pairs the
/// per-variant Zig and Python files by stem and the XLA pipeline finds the
/// same symbol inside both. The label string is what the Python sweep also
/// uses.
fn variantOf(comptime W: type, comptime label: []const u8, comptime cfg: W.Kernel.Config) Entry {
    const E = struct {
        fn emit(allocator: std.mem.Allocator, ctx: *mlir.Context) anyerror![:0]const u8 {
            return W.Kernel.emit(allocator, ctx, cfg);
        }
    };
    const variant_name = std.fmt.comptimePrint("{s}__{s}", .{ W.Kernel.name, label });
    return .{ .name = variant_name, .emit = E.emit };
}

pub const KERNELS: []const Entry = &.{
    defaults(VectorAdd),
    defaults(VectorExpFwd),
    defaults(VectorExpBwd),
    defaults(LowMemDropout),
    defaults(SumScalar),
    defaults(Softmax),
    // Output dtype is e5m2 (not e4m3fn) — Python's frontend rejects e4m3fn on
    // this Triton+GPU combo ("fp8e4nv not supported in this architecture").
    // The supported set on this build is {fp8e4b15, fp8e5}; e5m2 / e5 is the
    // straight fp8 type, so both sides use it for an apples-to-apples diff.
    withConfig(moe.PerTokenGroupQuantFp8, .{
        .input_dtype = .bf16,
        .output_dtype = .f8e5m2,
        .scale_dtype = .bf16,
        .block = 128,
        .fp8_min = -57344.0,
        .fp8_max = 57344.0,
        .use_ue8m0 = false,
    }),
    withConfig(moe.FusedMoe, .{
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
    }),
    withConfig(moe.MoeAlignBlockSize, .{
        .numel = 1024,
        .num_experts = 8,
        .padded_num_experts = 8,
        .max_num_tokens_padded = 2048,
        .max_num_m_blocks = 32,
        .block_size_m = 64,
        .hist_block = 64,
    }),
    withConfig(moe.CountAndSortExpertTokens, .{
        .numel = 1024,
        .num_experts = 8,
        .sort_block_size = 256,
    }),
    // Unified-attention kernels — match what `pagedAttention2d` /
    // `pagedAttention3d` typically launch in production model code.
    withConfig(attn.KernelUnifiedAttention2dPtr, .{
        .q_dtype = .bf16,
        .kv_dtype = .bf16,
        .o_dtype = .bf16,
        .scale_dtype = .f32,
        .num_query_heads = 32,
        .num_queries_per_kv = 4,
        .block_size = 16,
        .tile_size = 64,
        .head_size = 128,
        .head_size_padded = 128,
        .use_alibi_slopes = false,
        .use_qq_bias = false,
        .use_softcap = false,
        .use_sinks = false,
        .sliding_window = 0,
        .block_q = 16,
        .block_m = 16,
        .use_fp8 = false,
    }),
    withConfig(attn.KernelUnifiedAttention3dPtr, .{
        .q_dtype = .bf16,
        .kv_dtype = .bf16,
        .scale_dtype = .f32,
        .num_query_heads = 32,
        .num_queries_per_kv = 4,
        .block_size = 16,
        .tile_size = 64,
        .head_size = 128,
        .head_size_padded = 128,
        .use_alibi_slopes = false,
        .use_qq_bias = false,
        .use_softcap = false,
        .use_sinks = false,
        .sliding_window = 0,
        .block_q = 16,
        .block_m = 16,
        .num_segments_per_seq = 4,
    }),
    withConfig(attn.ReduceSegmentsPtr, .{
        .o_dtype = .bf16,
        .scale_dtype = .f32,
        .num_query_heads = 32,
        .tile_size = 64,
        .head_size = 128,
        .head_size_padded = 128,
        .block_q = 16,
        .num_segments_per_seq = 4,
        .use_fp8 = false,
    }),
    //
    // Unified-attention fuzzer — additional config variants drawn from
    // monorepo/llmd's call space (`zml/attention/triton_attention.zig:select2dConfig` /
    // `select3dConfig`). Each variant's filename suffix matches the label
    // emitted by `dump_python_ir.py`'s `_UNIFIED_ATTENTION_VARIANTS`. The
    // inner `tt.func` symbol stays equal to the production kernel name so
    // both sides flow through the standard XLA pipeline.
    //
    // 2D variants — covers the dispatch axes that monorepo actually triggers:
    //   - decode vs prefill (`all_decode`, `tile_size`)
    //   - GQA group sizes 1/4/8 (`num_queries_per_kv`)
    //   - head_size 64/128/256 (`head_size`, `head_size_padded`)
    //   - sliding window 0/4096 (gemma3 path)
    //   - long-prefill `block_m=128` path (max_seqlen_q ≥ 256)
    //
    variantOf(attn.KernelUnifiedAttention2dPtr, "dec_h128_g4", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = true,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "pre_h128_g4", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "pre_h128_g8", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 64,      .num_queries_per_kv = 8,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 2,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "pre_h128_g4_long", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 32,              .block_m = 128,
        .use_fp8 = false,           .all_decode = false,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "dec_h256_swa", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 4096,
        .block_q = 16,              .block_m = 64,
        .use_fp8 = false,           .all_decode = true,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "pre_h256_swa", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 4096,
        .block_q = 4,               .block_m = 16,
        .use_fp8 = false,           .all_decode = false,
    }),
    variantOf(attn.KernelUnifiedAttention2dPtr, "dec_h64_g1", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,         .o_dtype = .bf16,
        .num_query_heads = 16,      .num_queries_per_kv = 1,
        .block_size = 16,           .tile_size = 16,
        .head_size = 64,            .head_size_padded = 64,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 16,              .block_m = 16,
        .use_fp8 = false,           .all_decode = true,
    }),

    // 3D variants — `tile_size = block_size` always; segment count varies
    // with `target_num_prgms / num_2d_prgms` (we sample 16, 32, 64).
    variantOf(attn.KernelUnifiedAttention3dPtr, "pre_h128_g4_seg16", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 16,
    }),
    variantOf(attn.KernelUnifiedAttention3dPtr, "pre_h128_g8_seg32", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 64,      .num_queries_per_kv = 8,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 2,               .block_m = 16,
        .num_segments_per_seq = 32,
    }),
    variantOf(attn.KernelUnifiedAttention3dPtr, "dec_h128_g4_seg64", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 64,
        .all_decode = true,
    }),
    variantOf(attn.KernelUnifiedAttention3dPtr, "pre_h256_seg16", .{
        .q_dtype = .bf16,           .kv_dtype = .bf16,
        .num_query_heads = 32,      .num_queries_per_kv = 4,
        .block_size = 16,           .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .use_alibi_slopes = false,  .use_qq_bias = false,      .use_softcap = false,
        .use_sinks = false,         .sliding_window = 0,
        .block_q = 4,               .block_m = 16,
        .num_segments_per_seq = 16,
    }),

    // Reduce variants — match each 3D head/segment combination.
    variantOf(attn.ReduceSegmentsPtr, "h128_qh32_seg16", .{
        .o_dtype = .bf16,
        .num_query_heads = 32,      .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .block_q = 4,               .num_segments_per_seq = 16,
        .use_fp8 = false,
    }),
    variantOf(attn.ReduceSegmentsPtr, "h128_qh64_seg32", .{
        .o_dtype = .bf16,
        .num_query_heads = 64,      .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .block_q = 2,               .num_segments_per_seq = 32,
        .use_fp8 = false,
    }),
    variantOf(attn.ReduceSegmentsPtr, "h256_qh32_seg16", .{
        .o_dtype = .bf16,
        .num_query_heads = 32,      .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .block_q = 4,               .num_segments_per_seq = 16,
        .use_fp8 = false,
    }),

    // Gated Delta Net recurrent forward — config matches the monorepo's
    // `TritonGatedDeltaNetHelper.forward`. num_tokens=64, num_qk_heads=4,
    // num_v_heads=16, key_dim=32, value_dim=64, num_sequences=2 → BK=32, BV=8.
    withConfig(FusedRecurrentGatedDeltaRule, .{
        .scale = 1.0 / @sqrt(@as(f32, 32.0)),
        .T = 64,
        .H = 4,
        .HV = 16,
        .K = 32,
        .V = 64,
        .BK = 32,
        .BV = 8,
        .USE_G = true,
        .USE_GK = false,
        .USE_GV = false,
        .USE_QK_L2NORM_IN_KERNEL = true,
        .IS_BETA_HEADWISE = true,
        .USE_INITIAL_STATE = true,
        .STORE_FINAL_STATE = true,
        .USE_EXP2 = false,
        .TRANSPOSE_STATE = false,
        .IS_VARLEN = true,
    }),
};
