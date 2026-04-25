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

pub const Entry = struct {
    name: []const u8,
    emit: *const fn (std.mem.Allocator, *mlir.Context) anyerror![:0]const u8,
};

/// Lift a `zml.Kernel(...)` type with a default-constructible `Config` to
/// the dump driver's `Entry` shape.
fn defaults(comptime K: type) Entry {
    const E = struct {
        fn emit(allocator: std.mem.Allocator, ctx: *mlir.Context) anyerror![:0]const u8 {
            return K.emit(allocator, ctx, .{});
        }
    };
    return .{ .name = K.name, .emit = E.emit };
}

/// Same, but with an explicit `Config` value (for kernels whose required
/// fields have no defaults).
fn withConfig(comptime K: type, comptime cfg: K.Config) Entry {
    const E = struct {
        fn emit(allocator: std.mem.Allocator, ctx: *mlir.Context) anyerror![:0]const u8 {
            return K.emit(allocator, ctx, cfg);
        }
    };
    return .{ .name = K.name, .emit = E.emit };
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
};
