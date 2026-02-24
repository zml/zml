const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;

const test_cfg = struct {
    const token_count = 8;
    const batch_size = 8;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_size = 128;
    const block_size = 16;
    const num_blocks = 4096;
    const max_input_len = 1;
    const max_seq_len = 8192;
    const scale = 0.08838834765;
    const k_scale = 1.0;
    const v_scale = 1.0;
};

pub fn wrappedUnifiedAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    block_table: Tensor,
) Tensor {
    const q_strides = q.shape().withDtype(.u8).computeStrides();
    const k_strides = k.shape().withDtype(.u8).computeStrides();
    const v_strides = v.shape().withDtype(.u8).computeStrides();
    const bt_strides = block_table.shape().withDtype(.u8).computeStrides();

    // These are part of the torch-side API but are not explicit TTIR operands.
    const max_seqlen_q = test_cfg.max_input_len;
    const max_seqlen_k = test_cfg.max_seq_len;
    const causal = true;
    const window_size = [2]i32{ -1, -1 };
    const q_descale: ?f32 = null;
    _ = .{ max_seqlen_q, max_seqlen_k, causal, window_size, q_descale };

    const num_seqs = Tensor.scalar(block_table.dim(0), .i32);
    const grid: [3]i32 = .{ @intCast(test_cfg.batch_size), @intCast(test_cfg.num_kv_heads), 1 };

    return zml.ops.triton(.{
        out,
        q,
        k,
        v,
        block_table,
        seqused_k,
        Tensor.scalar(test_cfg.scale, .f32), // scale_ptr
        Tensor.scalar(test_cfg.k_scale, .f32), // k_scale_ptr
        Tensor.scalar(test_cfg.v_scale, .f32), // v_scale_ptr
        Tensor.scalar(1.0, .f32), // out_scale_ptr
        Tensor.scalar(0.0, .f32), // softcap_ptr
        Tensor.scalar(bt_strides.get(0), .i64), // block_table_stride_ptr
        Tensor.scalar(q_strides.get(0), .i64), // query_stride_0_ptr
        Tensor.scalar(q_strides.get(1), .i64), // query_stride_1_ptr
        Tensor.scalar(q_strides.get(0), .i64), // output_stride_0_ptr
        Tensor.scalar(q_strides.get(1), .i64), // output_stride_1_ptr
        Tensor.scalar(0, .i64), // qq_bias_stride_0_ptr
        Tensor.scalar(k_strides.get(0), .i64), // stride_k_cache_0_ptr
        Tensor.scalar(k_strides.get(1), .i64), // stride_k_cache_1_ptr
        Tensor.scalar(k_strides.get(2), .i64), // stride_k_cache_2_ptr
        Tensor.scalar(v_strides.get(0), .i64), // stride_v_cache_0_ptr
        Tensor.scalar(v_strides.get(1), .i64), // stride_v_cache_1_ptr
        Tensor.scalar(v_strides.get(2), .i64), // stride_v_cache_2_ptr
        cu_seqlens_q,
        num_seqs,
    }, .{out.shape()}, .{
        .name = "wrapped_kernel_unified_attention_2d",
        .ir = @embedFile("2d_unified_attention.ttir"),
        .grid = grid,
        .num_stages = 1,
        .num_warps = 4,
        .debug = true,
        .output_operand_aliases = &.{0},
    })[0];
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    if (platform.target != .cuda and platform.target != .rocm) {
        std.debug.print("ttir_test is compile-only for CUDA/ROCm targets, got {s}\n", .{@tagName(platform.target)});
        return;
    }

    const query_shape = zml.Shape.init(.{
        test_cfg.token_count,
        test_cfg.num_heads,
        test_cfg.head_size,
    }, .bf16);
    const out_shape = query_shape;
    const key_cache_shape = zml.Shape.init(.{
        test_cfg.num_blocks,
        test_cfg.block_size,
        test_cfg.num_kv_heads,
        test_cfg.head_size,
    }, .bf16);
    const value_cache_shape = zml.Shape.init(.{
        test_cfg.num_blocks,
        test_cfg.block_size,
        test_cfg.num_kv_heads,
        test_cfg.head_size,
    }, .bf16);
    const block_tables_shape = zml.Shape.init(.{
        test_cfg.batch_size,
        @divFloor(test_cfg.max_seq_len, test_cfg.block_size),
    }, .i32);
    const context_seq_lens_shape = zml.Shape.init(.{test_cfg.batch_size}, .i32);
    const start_loc_shape = zml.Shape.init(.{test_cfg.batch_size + 1}, .i32);

    const exe = try zml.compileFn(
        allocator,
        wrappedUnifiedAttention,
        .{
            query_shape,
            key_cache_shape,
            value_cache_shape,
            out_shape,
            start_loc_shape,
            context_seq_lens_shape,
            block_tables_shape,
        },
        platform,
    );
    defer exe.deinit();

    std.debug.print("Compiled TTIR custom op successfully.\n", .{});
}
