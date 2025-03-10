const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

pub const std_options: std.Options = .{
    .log_level = .info,
    // .logFn = asynk.logFn(std.log.defaultLog),
    // .log_scope_levels = .{
    //     .@"zml/async" = .warn,
    // },
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

    // _paged_attn_decode_v1_w_dot_kernel.kd
    // _paged_attn_decode_v1_w_dot_kernel_tt_load_only__1.kd

    const y = zml.ops.triton(.{
        q,
        k,
        v,
        block_tables,
        seq_lens,
        // zml.Tensor.empty(.{8 * 4}, .i8), // check actually it's none

        zml.Tensor.scalar(0.08838834765, .f32),
        zml.Tensor.scalar(1.0, .f32),
        zml.Tensor.scalar(1.0, .f32),

        zml.Tensor.scalar(@divExact(strides_o.get(0), 2), .i32),
        zml.Tensor.scalar(@divExact(strides_o.get(1), 2), .i32),
        // zml.Tensor.scalar(strides_o.get(2), .i32), stride_o_hs
        zml.Tensor.scalar(@divExact(strides_q.get(0), 2), .i32),
        zml.Tensor.scalar(@divExact(strides_q.get(1), 2), .i32),
        // zml.Tensor.scalar(strides_q.get(2), .i32),
        zml.Tensor.scalar(@divExact(strides_k.get(0), 2), .i32),
        zml.Tensor.scalar(@divExact(strides_k.get(1), 2), .i32),
        zml.Tensor.scalar(@divExact(strides_k.get(2), 2), .i32),
        // zml.Tensor.scalar(strides_k.get(3), .i32),
        zml.Tensor.scalar(@divExact(strides_bt.get(0), 4), .i32),
        // zml.Tensor.scalar(strides_bt.get(1), .i32),
    }, output_shape, .{
        .name = "_paged_attn_decode_v1_w_dot_kernel_tt_load_only",
        .ir = @embedFile("pa_kernel.mlir"),
        .grid = .{ 256, 8, 1 },
        .num_stages = 1,
        .num_warps = 4,
        .debug = true,
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

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .sharding_enabled = false,
    });

    const max_seq_len = 2048;
    const query = zml.Shape.init(.{ 256, 8 * 4, 128 }, .bf16);
    const key_cache = zml.Shape.init(.{ 16384, 8, 16, 128 }, .bf16);
    const value_cache = zml.Shape.init(.{ 16384, 8, 16, 128 }, .bf16);
    const seq_lens = zml.Shape.init(.{256}, .i64);
    const block_tables = zml.Shape.init(.{ 256, max_seq_len / 16 }, .i32);

    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, forward, .{ query, key_cache, value_cache, block_tables, seq_lens }, platform });
    const compiled = try compilation.awaitt();
    defer compiled.deinit();
    std.debug.print("Compilation finished\n", .{});

    const seqlen_host = try allocator.alloc(i64, 256);
    defer allocator.free(seqlen_host);
    @memset(seqlen_host, 2048);
    const page_table_host = try allocator.alloc(i32, 256 * (max_seq_len / 16));
    defer allocator.free(page_table_host);
    for (0..256) |i| {
        for (0..(max_seq_len / 16)) |j| {
            page_table_host[i * (max_seq_len / 16) + j] = @intCast(j);
        }
    }

    for (seqlen_host, 0..) |v, i| {
        std.debug.print("seq_lens[{}] = {}\n", .{ i, v });
    }

    {
        var q_buffer = try createRandomBuffer(allocator, platform, query, random);
        defer q_buffer.deinit();
        var k_buffer = try createRandomBuffer(allocator, platform, key_cache, random);
        defer k_buffer.deinit();
        var v_buffer = try createRandomBuffer(allocator, platform, value_cache, random);
        defer v_buffer.deinit();
        var bt_buffer = try zml.Buffer.fromBytes(platform, block_tables, std.mem.sliceAsBytes(page_table_host));
        defer bt_buffer.deinit();
        var sq_buffer = try zml.Buffer.fromBytes(platform, seq_lens, std.mem.sliceAsBytes(seqlen_host));
        defer sq_buffer.deinit();

        // var profiler = platform.getProfiler(null);
        // defer profiler.deinit();

        // profiler.start();
        var result: zml.Buffer = compiled.call(.{ q_buffer, k_buffer, v_buffer, bt_buffer, sq_buffer });
        defer result.deinit();

        var result2: zml.Buffer = compiled.call(.{ q_buffer, k_buffer, v_buffer, bt_buffer, sq_buffer });
        defer result2.deinit();

        var r = try result.toHostAlloc(allocator);
        // defer r.deinit(allocator);
        // profiler.stop();
        // try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), "profile_file.json");
        std.debug.print("Result: {}\n", .{r.shape()});
    }
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(data);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f64);
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = if (ZigType == f64)
                        value
                    else if (ZigType == f32)
                        @floatCast(value)
                    else if (ZigType == f16)
                        @floatCast(value)
                    else
                        @bitCast(random.int(std.meta.Int(.unsigned, @bitSizeOf(ZigType))));
                },
                .complex => unreachable,
            }
        },
    }

    var host_buffer = zml.HostBuffer.fromBytes(shape, data);
    errdefer host_buffer.deinit(allocator);
    return zml.Buffer.from(platform, host_buffer);
}
