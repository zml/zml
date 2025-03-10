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
    const o_shape = zml.Shape.init(.{ 256, 32, 128 }, .bf16);
    const l_shape = zml.Shape.init(.{ 8, 8, 16 }, .f32);
    const m_shape = zml.Shape.init(.{ 8, 8, 16 }, .f32);

    // _paged_attn_decode_v1_w_dot_kernel.kd
    // _paged_attn_decode_v1_w_dot_kernel_tt_load_only__1.kd

    const y = zml.ops.triton(.{
        q,
        k,
        v,
        block_tables,
        seq_lens,
    }, o_shape, l_shape, m_shape, .{
        .name = "paged_attention_block_h_16_pages_per_compute_block_8_batched",
        .ir = @embedFile("pa_kernel.mlir"),
        .grid = .{ 8, 8, 256 },
        .num_stages = 2,
        .num_warps = 8,
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
    _ = random; // autofix

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .sharding_enabled = false,
    });

    // q, k, v, block_tables, lengths, out_o, out_l, out_m
    // q: (256, 32, 128) bfloat16
    // k_pages: (8, 32768, 16, 128) bfloat16
    // v_pages: (8, 32768, 16, 128) bfloat16
    // block_tables: (256, 128) int32
    // seq_lens: (256,) int32
    // o: (256, 32, 128) bfloat16
    const max_seq_len = 2048;
    const query = zml.Shape.init(.{ 256, 32, 128 }, .bf16);
    const key_cache = zml.Shape.init(.{ 8, 32768, 16, 128 }, .bf16);
    const value_cache = zml.Shape.init(.{ 8, 32768, 16, 128 }, .bf16);
    const seq_lens = zml.Shape.init(.{256}, .i32);
    const block_tables = zml.Shape.init(.{ 256, 128 }, .i32);

    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, forward, .{ query, key_cache, value_cache, block_tables, seq_lens }, platform });
    const compiled = try compilation.awaitt();
    defer compiled.deinit();
    std.debug.print("Compilation finished\n", .{});

    const seqlen_host = try allocator.alloc(i32, 256);
    defer allocator.free(seqlen_host);
    @memset(seqlen_host, 2048);
    const page_table_host = try allocator.alloc(i32, 256 * (max_seq_len / 16));
    defer allocator.free(page_table_host);
    for (0..256) |i| {
        for (0..(max_seq_len / 16)) |j| {
            page_table_host[i * (max_seq_len / 16) + j] = 0;
        }
    }

    {
        const rng_q = try zml.compileFn(allocator, zml.Tensor.Rng.normal, .{ query, .{ .mean = 0, .stddev = 1 } }, platform);
        defer rng_q.deinit();
        const rng_k = try zml.compileFn(allocator, zml.Tensor.Rng.normal, .{ key_cache, .{ .mean = 0, .stddev = 1 } }, platform);
        defer rng_k.deinit();
        const rng_v = try zml.compileFn(allocator, zml.Tensor.Rng.normal, .{ value_cache, .{ .mean = 0, .stddev = 1 } }, platform);
        defer rng_v.deinit();

        const q_buffer = rng_q.call(undefined);
        const k_buffer = rng_k.call(undefined);
        const v_buffer = rng_k.call(undefined);

        var bt_buffer = try zml.Buffer.fromBytes(platform, block_tables, std.mem.sliceAsBytes(page_table_host));
        defer bt_buffer.deinit();
        var sq_buffer = try zml.Buffer.fromBytes(platform, seq_lens, std.mem.sliceAsBytes(seqlen_host));
        defer sq_buffer.deinit();

        // var profiler = platform.getProfiler(null);
        // defer profiler.deinit();

        // profiler.start();
        var result: zml.Buffer = compiled.call(.{ q_buffer, k_buffer, v_buffer, bt_buffer, sq_buffer });
        defer result.deinit();

        var r = try result.toHostAlloc(allocator);
        defer r.deinit(allocator);
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
