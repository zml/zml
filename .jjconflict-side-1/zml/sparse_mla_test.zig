const std = @import("std");

const zml = @import("zml");

const paged = zml.attention.paged_attention;
const KvCache = paged.KvCache;
const Mla = paged.Mla;
const Parameters = paged.Parameters;

fn replicated(tensor: zml.Tensor) zml.Tensor {
    return .fromShape(tensor.shape().withReplicatedPartitioning());
}

test "ROCm GLM sparse MLA decode geometry" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm or !paged.Backend.triton.isAvailable(platform)) return error.SkipZigTest;
    const query_count = if (std.c.getenv("ZML_SPARSE_MLA_QUERIES")) |value|
        try std.fmt.parseInt(usize, std.mem.span(value), 10)
    else
        1;

    var parameters = Parameters.init(.fromBackend(.{
        .backend = .triton,
        .is_prefill = false,
        .batch_size = @intCast(query_count),
        .seq_len = 2048,
        .max_num_pages = 128,
        .max_token_count = @intCast(query_count),
        .num_heads = 16,
        .num_kv_heads = 1,
        .head_dim = 576,
        .max_seqlen_q = 1,
    }));
    parameters.triton.block_table = replicated(parameters.triton.block_table);
    parameters.triton.seq_lens = replicated(parameters.triton.seq_lens);
    parameters.triton.query_start_len = replicated(parameters.triton.query_start_len);

    const q_shape = zml.Shape.init(.{ .q = query_count, .h = 16, .hd = 576 }, .bf16).withReplicatedPartitioning();
    const kv_shape = zml.Shape.init(.{ .page = 128, .k_chunk = 16, .hkv = 1, .hd = 576 }, .bf16).withReplicatedPartitioning();
    const sink_shape = zml.Shape.init(.{ .h = 16 }, .f32).withReplicatedPartitioning();
    const topk_shape = zml.Shape.init(.{ .q = query_count, .topk = 2048 }, .i32).withReplicatedPartitioning();
    const tokens_pos_shape = zml.Shape.init(.{ .q = query_count }, .i32).withReplicatedPartitioning();

    const q = zml.Tensor.fromShape(q_shape);
    const kv: KvCache = .{ .latent = zml.Tensor.fromShape(kv_shape) };
    const sink = zml.Tensor.fromShape(sink_shape);
    const topk = zml.Tensor.fromShape(topk_shape);
    const tokens_pos = zml.Tensor.fromShape(tokens_pos_shape);

    const q_data = try allocator.alloc(zml.floats.BFloat16, q_shape.count());
    defer allocator.free(q_data);
    @memset(q_data, zml.floats.BFloat16.fromF32(0));
    const kv_data = try allocator.alloc(zml.floats.BFloat16, kv_shape.count());
    defer allocator.free(kv_data);
    @memset(kv_data, zml.floats.BFloat16.fromF32(1));
    const sink_data = try allocator.alloc(f32, sink_shape.count());
    defer allocator.free(sink_data);
    @memset(sink_data, -std.math.inf(f32));
    const topk_data = try allocator.alloc(i32, topk_shape.count());
    defer allocator.free(topk_data);
    for (topk_data, 0..) |*position, i| position.* = @intCast(i % 2048);
    const tokens_pos_data = try allocator.alloc(i32, query_count);
    defer allocator.free(tokens_pos_data);
    @memset(tokens_pos_data, 2047);
    const block_table = try allocator.alloc(i32, query_count * 128);
    defer allocator.free(block_table);
    for (block_table, 0..) |*physical_page, i| physical_page.* = @intCast(i % 128);
    const seq_lens = try allocator.alloc(i32, query_count);
    defer allocator.free(seq_lens);
    @memset(seq_lens, 2048);
    const query_start_len = try allocator.alloc(i32, query_count + 1);
    defer allocator.free(query_start_len);
    for (query_start_len, 0..) |*query_start, i| query_start.* = @intCast(i);

    var parameters_d: zml.Bufferized(Parameters) = .{ .triton = .{
        .block_table = try .fromBytes(io, platform, parameters.triton.block_table.shape(), .replicated, std.mem.sliceAsBytes(block_table)),
        .seq_lens = try .fromBytes(io, platform, parameters.triton.seq_lens.shape(), .replicated, std.mem.sliceAsBytes(seq_lens)),
        .query_start_len = try .fromBytes(io, platform, parameters.triton.query_start_len.shape(), .replicated, std.mem.sliceAsBytes(query_start_len)),
    } };
    defer zml.Buffer.deinitAll(Parameters, &parameters_d);
    var q_d = try zml.Buffer.fromBytes(io, platform, q_shape, .replicated, std.mem.sliceAsBytes(q_data));
    defer q_d.deinit();
    var kv_d: zml.Bufferized(KvCache) = .{ .latent = try .fromBytes(io, platform, kv_shape, .replicated, std.mem.sliceAsBytes(kv_data)) };
    defer zml.Buffer.deinitAll(KvCache, &kv_d);
    var sink_d = try zml.Buffer.fromBytes(io, platform, sink_shape, .replicated, std.mem.sliceAsBytes(sink_data));
    defer sink_d.deinit();
    var topk_d = try zml.Buffer.fromBytes(io, platform, topk_shape, .replicated, std.mem.sliceAsBytes(topk_data));
    defer topk_d.deinit();
    var tokens_pos_d = try zml.Buffer.fromBytes(io, platform, tokens_pos_shape, .replicated, std.mem.sliceAsBytes(tokens_pos_data));
    defer tokens_pos_d.deinit();

    const num_splits = if (std.c.getenv("ZML_SPARSE_MLA_SPLITS")) |value|
        try std.fmt.parseInt(u8, std.mem.span(value), 10)
    else
        null;
    const iterations = if (std.c.getenv("ZML_SPARSE_MLA_ITERATIONS")) |value|
        try std.fmt.parseInt(usize, std.mem.span(value), 10)
    else
        1;

    var exe = try platform.compileFn(
        allocator,
        io,
        Mla.pagedSparseAttention,
        .{ parameters, q, kv, sink, topk, tokens_pos, .{ .rope_rank = 64, .num_kv_splits = num_splits } },
        .{},
    );
    defer exe.deinit();

    for (0..iterations) |_| {
        var output = try zml.testing.autoCall(
            allocator,
            io,
            &exe,
            Mla.pagedSparseAttention,
            .{ parameters_d, q_d, kv_d, sink_d, topk_d, tokens_pos_d },
        );
        if (iterations == 1) {
            const expected_data = try allocator.alloc(zml.floats.BFloat16, q_shape.count());
            defer allocator.free(expected_data);
            @memset(expected_data, zml.floats.BFloat16.fromF32(1));
            const expected = zml.Slice.init(q_shape, std.mem.sliceAsBytes(expected_data));
            try zml.testing.expectClose(io, expected, output, .{ .absolute_tolerance = 0.01, .relative_tolerance = 0.01 });
        }
        output.deinit();
    }
}
