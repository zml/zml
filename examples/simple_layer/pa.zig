const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

pub const std_options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn forward(
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    block_tables: zml.Tensor,
    seq_lens: zml.Tensor,
) zml.Tensor {
    const output_shape = zml.Shape.init(.{ 256, 32, 128 }, .bf16);
    const strides_o = output_shape.computeStrides();
    const strides_q = q.shape().computeStrides();
    const strides_k = k.shape().computeStrides();
    const strides_bt = block_tables.shape().computeStrides();
    //  Arguments to kernel:
    // 'out_ptr': '*bf16',
    // 'q_ptr': '*bf16',
    // 'k_cache_ptr': '*bf16',
    // 'v_cache_ptr': '*bf16',
    // 'blk_tables_ptr': '*i32',
    // 'seq_lens_ptr': '*i64',
    // 'alibi_slopes': '*i8',
    // 'scale': 'fp32',
    // 'k_scale': 'fp32',
    // 'v_scale': 'fp32',
    // 'stride_o_s': 'i32',
    // 'stride_o_nh': 'i32',
    // 'stride_o_hs': 'i32',
    // 'stride_q_s': 'i32',
    // 'stride_q_nh': 'i32',
    // 'stride_q_hs': 'i32',
    // 'stride_k_b': 'i32',
    // 'stride_k_nh': 'i32',
    // 'stride_k_kb': 'i32',
    // 'stride_k_hs': 'i32',
    // 'stride_bt_s': 'i32',
    // 'stride_bt_nb': 'i32'

    const y = zml.ops.triton(.{
        q,
        k,
        v,
        seq_lens,
        block_tables,
        zml.Tensor.constant(.{8 * 4}, zml.Data.init(.i8, 0)), // check actually it's none
        zml.Tensor.scalar(0.08838834765, .f32),
        // zml.Tensor.scalar(1.0, .f32),
        // zml.Tensor.scalar(1.0, .f32),
        zml.Tensor.scalar(strides_o.get(0), .i32),
        zml.Tensor.scalar(strides_o.get(1), .i32),
        // zml.Tensor.scalar(strides_o.get(2), .i32),
        zml.Tensor.scalar(strides_q.get(0), .i32),
        zml.Tensor.scalar(strides_q.get(1), .i32),
        zml.Tensor.scalar(strides_q.get(2), .i32),
        zml.Tensor.scalar(strides_k.get(0), .i32),
        zml.Tensor.scalar(strides_k.get(1), .i32),
        zml.Tensor.scalar(strides_k.get(2), .i32),
        zml.Tensor.scalar(strides_k.get(3), .i32),
        zml.Tensor.scalar(strides_bt.get(0), .i32),
        // zml.Tensor.scalar(strides_bt.get(1), .i32),
    }, output_shape, .{
        .name = "_paged_attn_decode_v1_w_dot_kernel",
        .ir = @embedFile("pa_kernel.mlir"),
        .grid = .{ 256, 8, 1 },
        .num_stages = 3,
        .num_warps = 4,
        .debug = false,
    });

    return y;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    _ = arena; // autofix

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const query = zml.Shape.init(.{ 256, 8 * 4, 128 }, .bf16);
    const key_cache = zml.Shape.init(.{ 16384, 8, 16, 128 }, .bf16);
    const value_cache = zml.Shape.init(.{ 16384, 8, 16, 128 }, .bf16);
    const seq_lens = zml.Shape.init(.{256}, .i64);
    const block_tables = zml.Shape.init(.{ 256, 128 }, .i32);

    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, forward, .{ query, key_cache, value_cache, block_tables, seq_lens }, platform });
    const compiled = try compilation.awaitt();
    _ = compiled; // autofix
    std.debug.print("Compilation finished\n", .{});

    // var model_weights = try zml.aio.loadModelBuffers(PagedAttention, model_shapes, buffer_store, arena, platform);
    // defer zml.aio.unloadBuffers(&model_weights); // for good practice

    // var executable = compiled.prepare(model_weights);
    // defer executable.deinit();

    // var input_buffer = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(input_shape, &input));
    // defer input_buffer.deinit();

    // call our executable module
    // var result: zml.Buffer = executable.call(.{input_buffer});
    // defer result.deinit();
}
