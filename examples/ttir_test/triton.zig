const std = @import("std");
const async = @import("async");
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
    const target = zml.module.CompilationContext.current().target();
    const num_warps: i32 = switch (target) {
        .rocm => 2,
        .cuda => 4,
        else => 4,
    };

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
        .num_warps = num_warps,
        .debug = true,
        .output_operand_aliases = &.{0},
    })[0];
}

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);
    if (platform.target != .cuda and platform.target != .rocm) {
        std.debug.print("ttir_test requires CUDA/ROCm to run, got {s}\n", .{@tagName(platform.target)});
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
    const value_cache_shape = key_cache_shape;
    const block_table_width = @divFloor(test_cfg.max_seq_len, test_cfg.block_size);
    const block_tables_shape = zml.Shape.init(.{
        test_cfg.batch_size,
        block_table_width,
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

    // Dummy buffers equivalent to the torch setup.
    var query = try zml.Buffer.constant(platform, query_shape, 0, .{});
    defer query.deinit();
    var key_cache = try zml.Buffer.constant(platform, key_cache_shape, 0, .{});
    defer key_cache.deinit();
    var value_cache = try zml.Buffer.constant(platform, value_cache_shape, 0, .{});
    defer value_cache.deinit();
    var out = try zml.Buffer.constant(platform, out_shape, 0, .{});
    defer out.deinit();

    const start_loc_h = try allocator.alloc(i32, test_cfg.batch_size + 1);
    defer allocator.free(start_loc_h);
    for (start_loc_h, 0..) |*x, i| x.* = @intCast(i);
    var start_loc = try zml.Buffer.fromSlice(platform, .{test_cfg.batch_size + 1}, start_loc_h);
    defer start_loc.deinit();

    const context_seq_lens_h = try allocator.alloc(i32, test_cfg.batch_size);
    defer allocator.free(context_seq_lens_h);
    @memset(context_seq_lens_h, 1);
    var context_seq_lens = try zml.Buffer.fromSlice(platform, .{test_cfg.batch_size}, context_seq_lens_h);
    defer context_seq_lens.deinit();

    var block_tables_h = try allocator.alloc(i32, test_cfg.batch_size * block_table_width);
    defer allocator.free(block_tables_h);
    @memset(block_tables_h, 0);
    for (0..test_cfg.batch_size) |b| {
        block_tables_h[b * block_table_width] = @intCast(b);
    }
    var block_tables = try zml.Buffer.fromSlice(platform, .{ test_cfg.batch_size, block_table_width }, block_tables_h);
    defer block_tables.deinit();

    var result: zml.Buffer = exe.call(.{
        query,
        key_cache,
        value_cache,
        out,
        start_loc,
        context_seq_lens,
        block_tables,
    });
    defer result.deinit();

    var host_out = try result.toHostAlloc(allocator);
    defer host_out.deinit(allocator);

    const out_bytes = host_out.bytes();
    const n = @min(out_bytes.len / 2, 8);
    std.debug.print("Output shape: {f}\n", .{result.shape()});
    std.debug.print("Output sample (first {d} bf16 lanes as u16):", .{n});
    for (0..n) |i| {
        const lane = @as(u16, out_bytes[i * 2]) | (@as(u16, out_bytes[i * 2 + 1]) << 8);
        std.debug.print(" 0x{x:0>4}", .{lane});
    }
    std.debug.print("\n", .{});
}
