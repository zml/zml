const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;

pub fn unifiedAttention2d(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    query_start_len: Tensor,
) Tensor {
    const q_strides = query.shape().withDtype(.u8).computeStrides();
    const k_strides = key_cache.shape().withDtype(.u8).computeStrides();
    const v_strides = value_cache.shape().withDtype(.u8).computeStrides();
    const bt_strides = block_tables.shape().withDtype(.u8).computeStrides();

    const output = Tensor.constant(query.shape(), query.dtype().zero());
    const num_seqs = Tensor.scalar(seq_lens.dim(0), .i32);

    const grid: [3]i32 = .{
        @intCast(@divFloor(query.dim(0) + 15, 16)),
        @intCast(query.dim(1)),
        1,
    };

    return zml.ops.triton(.{
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        Tensor.scalar(1.0, .f32), // scale_ptr
        Tensor.scalar(1.0, .f32), // k_scale_ptr
        Tensor.scalar(1.0, .f32), // v_scale_ptr
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
        query_start_len,
        num_seqs,
    }, .{output.shape()}, .{
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

    const query_shape = zml.Shape.init(.{ 128, 4, 128 }, .bf16);
    const key_cache_shape = zml.Shape.init(.{ 128, 4, 128 }, .bf16);
    const value_cache_shape = zml.Shape.init(.{ 128, 4, 128 }, .bf16);
    const block_tables_shape = zml.Shape.init(.{ 8, 32 }, .i32);
    const seq_lens_shape = zml.Shape.init(.{8}, .i32);
    const query_start_len_shape = zml.Shape.init(.{9}, .i32);

    const exe = try zml.compileFn(
        allocator,
        unifiedAttention2d,
        .{
            query_shape,
            key_cache_shape,
            value_cache_shape,
            block_tables_shape,
            seq_lens_shape,
            query_start_len_shape,
        },
        platform,
    );
    defer exe.deinit();

    std.debug.print("Compiled TTIR custom op successfully.\n", .{});
}
