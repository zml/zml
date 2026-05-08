//! Registration of `ReduceSegmentsPtr` — the second-pass kernel that
//! follows `kernel_unified_attention_3d_ptr` and reduces per-segment
//! partial outputs into the final attention result.
//!
//! Sweeps mirror `examples/triton_emitter/kernels_zig.zig` lines 126-136
//! (default) + 268-288 (3 fuzzer variants matching each 3d head/segment
//! combination).

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub const Kernel = zml.attention.triton_kernels.ReduceSegmentsPtr.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .o_dtype = .bf16,
        .num_query_heads = 32,      .tile_size = 64,
        .head_size = 128,           .head_size_padded = 128,
        .block_q = 16,              .num_segments_per_seq = 4,
        .use_fp8 = false,
    } },
    .{ .label = "h128_qh32_seg16", .cfg = .{
        .o_dtype = .bf16,
        .num_query_heads = 32,      .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .block_q = 4,               .num_segments_per_seq = 16,
        .use_fp8 = false,
    } },
    .{ .label = "h128_qh64_seg32", .cfg = .{
        .o_dtype = .bf16,
        .num_query_heads = 64,      .tile_size = 16,
        .head_size = 128,           .head_size_padded = 128,
        .block_q = 2,               .num_segments_per_seq = 32,
        .use_fp8 = false,
    } },
    .{ .label = "h256_qh32_seg16", .cfg = .{
        .o_dtype = .bf16,
        .num_query_heads = 32,      .tile_size = 16,
        .head_size = 256,           .head_size_padded = 256,
        .block_q = 4,               .num_segments_per_seq = 16,
        .use_fp8 = false,
    } },
};

const MAX_NUM_TOKENS: i64 = 64;
const MAX_NUM_QUERY_HEADS: i64 = 64;
const MAX_HEAD_SIZE_PADDED: i64 = 256;
const MAX_NUM_SEGMENTS: i64 = 128;
const MAX_Q_BUF: i64 = MAX_NUM_TOKENS * MAX_NUM_QUERY_HEADS * MAX_HEAD_SIZE_PADDED;
const MAX_SEGM_BASE: i64 = MAX_NUM_TOKENS * MAX_NUM_QUERY_HEADS * MAX_NUM_SEGMENTS;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    segm_out: Tensor, segm_max: Tensor, segm_expsum: Tensor,
    seq_lens: Tensor, num_seqs: Tensor, out_scale_inv: Tensor,
    o_s0: Tensor, o_s1: Tensor, bt_stride: Tensor, qsl: Tensor, _: Tensor,
) Tensor {
    return ops.triton(
        .{
            segm_out, segm_max, segm_expsum, seq_lens, num_seqs, out_scale_inv,
            o_s0, o_s1, bt_stride, qsl,
        },
        .{Shape.init(.{MAX_Q_BUF}, .bf16)},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 64, 32, 1 },
            .num_warps = 4,
            .num_stages = 1,
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
        // segm_output, segm_max, segm_expsum (inputs from the 3d pass).
        ten1d(.f32, MAX_SEGM_BASE * MAX_HEAD_SIZE_PADDED), ten1d(.f32, MAX_SEGM_BASE), ten1d(.f32, MAX_SEGM_BASE),
        // seq_lens, num_seqs, out_scale_inv.
        ten1d(.i32, 1), ten1d(.i32, 1), ten1d(.f32, 1),
        // output_stride_0/1, block_table_stride, query_start_len, output (placeholder).
        ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i32, 2),
        ten1d(.bf16, MAX_Q_BUF),
    };
}
