const std = @import("std");

const zml = @import("../../../zml.zig");
const cute = @import("cute.zig");
const paged = @import("../paged_attention.zig");

const Case = struct {
    head_dim: usize = 512,
    full_batch: bool = false,
    active_slots: ?usize = null,
    group: usize = 6,
    heads: usize = 2,
    page_size: usize = 16,
    decode: bool = false,
    single_queries: bool = false,
    q_scale: f32 = 1,
    shard: enum { none, kv, group } = .none,
    causal: bool = true,
    window: ?u32 = null,
    sink: bool = false,
    query_count: usize = 65,
    length: usize = 137,
};

test "CuTe paged attention numerical prefill and decode" {
    try check(.bf16, .{ .group = 3, .decode = true, .length = 257 });
    try check(.bf16, .{});
}

test "CuTe paged attention general GQA and masks" {
    for ([_]usize{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64 }) |group| {
        try check(.bf16, .{ .group = group, .heads = if (group % 2 == 0) 1 else 3, .decode = true, .length = 129, .sink = true });
        try check(.bf16, .{ .group = group, .query_count = 17, .length = 67, .window = 17, .causal = false });
    }
    try check(.f16, .{ .group = 6, .decode = true, .length = 4093, .page_size = 32, .sink = true });
    try check(.f16, .{ .group = 3, .query_count = 67, .length = 137, .page_size = 64, .window = 17, .sink = true });
    try check(.bf16, .{ .group = 8, .decode = true, .length = 4093, .window = 1024 });
    try check(.bf16, .{ .group = 8, .single_queries = true, .query_count = 1, .length = 257, .sink = true });
    try check(.bf16, .{ .group = 8, .decode = true, .length = 257, .sink = true });
    try check(.bf16, .{ .group = 4, .query_count = 17, .length = 67, .q_scale = 64, .sink = true });
}

fn asF32(comptime dt: zml.DataType, value: dt.toZigType()) f32 {
    return if (dt == .bf16) value.toF32() else @floatCast(value);
}

fn check(comptime dt: zml.DataType, c: Case) !void {
    const platform = zml.testing.env();
    if (!cute.isAvailable(platform)) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const T = dt.toZigType();
    const initial_qcounts = [_]usize{ if (c.decode or c.single_queries) 1 else c.query_count, if (c.decode or c.single_queries) 1 else 3, 1, 0 };
    const initial_lengths = [_]i32{ @intCast(c.length), if (c.decode or c.single_queries) 17 else 3, 1, 0 };
    const batch: usize = if (c.full_batch) 256 else 4;
    const qcounts = try allocator.alloc(usize, batch);
    defer allocator.free(qcounts);
    const lengths = try allocator.alloc(i32, batch);
    defer allocator.free(lengths);
    if (c.full_batch) {
        @memset(qcounts, 1);
        @memset(lengths, @intCast(c.length));
        if (c.active_slots) |active| {
            @memset(qcounts[active..], 0);
            @memset(lengths[active..], 0);
        }
    } else {
        @memcpy(qcounts, &initial_qcounts);
        @memcpy(lengths, &initial_lengths);
    }
    const starts = try allocator.alloc(i32, batch + 1);
    defer allocator.free(starts);
    starts[0] = 0;
    for (qcounts, 0..) |count, i| starts[i + 1] = starts[i] + @as(i32, @intCast(count));
    const total: usize = @intCast(starts[batch]);
    const capacity = total + 7;
    const pages = @divTrunc(@max(c.length, 17) + c.page_size - 1, c.page_size) + 3;
    var q: zml.Tensor = .fromShape(.init(.{ .b = capacity, .hkv = c.heads, .hg = c.group, .hd = c.head_dim }, dt));
    var k: zml.Tensor = .fromShape(.init(.{ .page = pages, .k_chunk = c.page_size, .hkv = c.heads, .hd = c.head_dim }, dt));
    const mesh = platform.meshes.get("model").?;
    switch (c.shard) {
        .none => {},
        .kv => {
            q = q.withPartitioning(mesh, .{ .hkv = .model });
            k = k.withPartitioning(mesh, .{ .hkv = .model });
        },
        .group => q = q.withPartitioning(mesh, .{ .hg = .model }),
    }
    const v: zml.Tensor = .fromShape(k.shape());
    const q_host = try allocator.alloc(T, q.count());
    defer allocator.free(q_host);
    const k_host = try allocator.alloc(T, k.count());
    defer allocator.free(k_host);
    const v_host = try allocator.alloc(T, v.count());
    defer allocator.free(v_host);
    var rng = std.Random.DefaultPrng.init(928173);
    for ([_][]T{ q_host, k_host, v_host }) |buffer| {
        for (buffer) |*x| x.* = @field(zml.DataType.Value.init(dt, (rng.random().float(f32) - 0.5) * 2), @tagName(dt));
    }
    if (c.q_scale != 1) for (q_host) |*x| {
        x.* = @field(zml.DataType.Value.init(dt, asF32(dt, x.*) * c.q_scale), @tagName(dt));
    };
    var q_d = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(q_host));
    defer q_d.deinit();
    var k_d = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(k_host));
    defer k_d.deinit();
    var v_d = try zml.Buffer.fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(v_host));
    defer v_d.deinit();
    const table = try allocator.alloc(i32, batch * pages);
    defer allocator.free(table);
    for (0..batch) |s| for (0..pages) |j| {
        table[s * pages + j] = if (j * c.page_size < lengths[s]) @intCast((j * 13 + s * 7) % pages) else -1;
    };
    const p: cute.Parameters = .init(.{ .batch_size = batch, .max_num_pages = pages, .max_seqlen_q = if (c.decode) 1 else c.query_count, .is_prefill = !c.decode });
    var pd: zml.Bufferized(cute.Parameters) = .{
        .block_table = try .fromBytes(io, platform, p.block_table.shape(), std.mem.sliceAsBytes(table)),
        .seq_lens = undefined,
        .query_start_len = undefined,
    };
    defer pd.block_table.deinit();
    pd.seq_lens = try .fromBytes(io, platform, p.seq_lens.shape(), std.mem.sliceAsBytes(lengths));
    defer pd.seq_lens.deinit();
    pd.query_start_len = try .fromBytes(io, platform, p.query_start_len.shape(), std.mem.sliceAsBytes(starts));
    defer pd.query_start_len.deinit();
    const sink_host = try allocator.alloc(f32, c.heads * c.group);
    defer allocator.free(sink_host);
    for (sink_host, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) - 1.5;
    var sink: zml.Tensor = .fromShape(.init(.{ .hkv = c.heads, .hg = c.group }, .f32));
    switch (c.shard) {
        .none => {},
        .kv => sink = sink.withPartitioning(mesh, .{ .hkv = .model }),
        .group => sink = sink.withPartitioning(mesh, .{ .hg = .model }),
    }
    var sink_d = try zml.Buffer.fromBytes(io, platform, sink.shape(), std.mem.sliceAsBytes(sink_host));
    defer sink_d.deinit();
    const opts: paged.AttentionOptions = .{ .mask = .{ .standard = .{ .causal = c.causal, .sliding_window = c.window } }, .scale = 0.071, .sink = if (c.sink) sink else null };
    const exe = try platform.compileFn(allocator, io, cute.pagedAttention, .{ p, q, k, v, opts }, .{ .meshes = &.{ platform.replicated_mesh, mesh } });
    defer exe.deinit();
    var actual_d = try exe.eval(allocator, io, .{ pd, q_d, k_d, v_d, .{ .sink = if (c.sink) sink_d else null } });
    defer actual_d.deinit();
    const actual = try actual_d.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    const output = actual.items(T);
    const scores = try allocator.alloc(f64, @max(c.length, 17));
    defer allocator.free(scores);
    // Independent scalar reference: gather physical pages and normalize over
    // the permitted logical keys. It does not reuse either GPU backend's math.
    for (0..batch) |s| for (0..qcounts[s]) |qi| for (0..c.heads) |head| for (0..c.group) |g| {
        const token = @as(usize, @intCast(starts[s])) + qi;
        const pos = @as(usize, @intCast(lengths[s])) - qcounts[s] + qi;
        const qoff = ((token * c.heads + head) * c.group + g) * c.head_dim;
        const len: usize = @intCast(lengths[s]);
        var maximum: f64 = if (c.sink) sink_host[head * c.group + g] else -std.math.inf(f64);
        for (0..len) |j| {
            const visible = (!c.causal or j <= pos) and (c.window == null or @as(i64, @intCast(j)) > @as(i64, @intCast(pos)) - c.window.?);
            scores[j] = -std.math.inf(f64);
            if (!visible) continue;
            const physical: usize = @intCast(table[s * pages + j / c.page_size]);
            const off = ((physical * c.page_size + j % c.page_size) * c.heads + head) * c.head_dim;
            var dot: f64 = 0;
            for (0..c.head_dim) |d| dot += @as(f64, asF32(dt, q_host[qoff + d])) * asF32(dt, k_host[off + d]);
            scores[j] = dot * 0.071;
            maximum = @max(maximum, scores[j]);
        }
        var sum: f64 = if (c.sink) @exp(@as(f64, sink_host[head * c.group + g]) - maximum) else 0;
        for (scores[0..len]) |*score| {
            score.* = @exp(score.* - maximum);
            sum += score.*;
        }
        var expected_storage: [512]f64 = @splat(0);
        const expected = expected_storage[0..c.head_dim];
        for (0..len) |j| {
            const physical: usize = @intCast(table[s * pages + j / c.page_size]);
            const off = ((physical * c.page_size + j % c.page_size) * c.heads + head) * c.head_dim;
            for (0..c.head_dim) |d| expected[d] += scores[j] * asF32(dt, v_host[off + d]) / sum;
        }
        for (expected, 0..) |want, d| {
            const got = asF32(dt, output[qoff + d]);
            if (!std.math.isFinite(got) or @abs(got - want) > 0.003 + 0.015 * @abs(want)) {
                std.log.err("CuTe mismatch decode={} G={} seq={} q={} head={}/{} d={}: want={} got={}", .{ c.decode, c.group, s, qi, head, g, d, want, got });
                return error.TestUnexpectedResult;
            }
        }
    };
}

test "CuTe paged attention respects head sharding" {
    const platform = zml.testing.env();
    if (platform.devices.len != 2) return error.SkipZigTest;
    try check(.bf16, .{ .group = 6, .heads = 4, .decode = true, .sink = true, .shard = .kv });
    try check(.bf16, .{ .group = 6, .heads = 4, .query_count = 17, .sink = true, .shard = .kv });
    try check(.bf16, .{ .group = 6, .heads = 1, .decode = true, .sink = true, .shard = .group });
    try check(.bf16, .{ .group = 6, .heads = 1, .query_count = 17, .sink = true, .shard = .group });
}

test "CuTe D256 sliding attention" {
    try check(.bf16, .{ .head_dim = 256, .group = 2, .heads = 8, .decode = true, .length = 129, .window = 65, .full_batch = true });
    for ([_]usize{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64 }) |group| {
        try check(.bf16, .{ .head_dim = 256, .group = group, .decode = true, .length = 129, .window = 65, .sink = true });
    }
    try check(.bf16, .{ .head_dim = 256, .group = 2, .decode = true, .length = 4093, .window = 1024 });
    try check(.f16, .{ .head_dim = 256, .group = 3, .decode = true, .length = 257, .window = 1, .page_size = 32 });
    try check(.bf16, .{ .head_dim = 256, .group = 3, .query_count = 67, .length = 137, .window = 17 });
    try check(.f16, .{ .head_dim = 256, .group = 6, .query_count = 17, .length = 67, .window = 17, .causal = false, .sink = true });
    const platform = zml.testing.env();
    if (platform.devices.len == 2) {
        try check(.bf16, .{ .head_dim = 256, .group = 2, .heads = 4, .decode = true, .length = 257, .window = 65, .shard = .kv });
        try check(.bf16, .{ .head_dim = 256, .group = 6, .heads = 1, .decode = true, .length = 257, .window = 65, .shard = .group });
    }
}

test "CuTe dual stage sliding decode" {
    try check(.bf16, .{ .head_dim = 256, .group = 1, .heads = 1, .decode = true, .length = 33025, .window = 33025, .page_size = 64 });
    // Full occupancy still partitions a long window, including a partial
    // first/last tile and a shorter last partition. The sink belongs to merge.
    try check(.bf16, .{ .head_dim = 256, .group = 2, .heads = 8, .decode = true, .length = 1105, .window = 1024, .sink = true, .full_batch = true });
    try check(.f16, .{ .head_dim = 256, .group = 3, .decode = true, .length = 1027, .window = 513, .page_size = 32, .sink = true });
    try check(.bf16, .{ .head_dim = 256, .group = 12, .decode = true, .length = 769, .window = 512, .q_scale = 64, .sink = true });
    const platform = zml.testing.env();
    if (platform.devices.len == 2) {
        try check(.bf16, .{ .head_dim = 256, .group = 2, .heads = 4, .decode = true, .length = 1105, .window = 1024, .shard = .kv });
        try check(.bf16, .{ .head_dim = 256, .group = 6, .heads = 1, .decode = true, .length = 1105, .window = 1024, .sink = true, .shard = .group });
    }
}

test "CuTe full batch global decode direct output" {
    try check(.bf16, .{ .head_dim = 512, .full_batch = true, .group = 8, .heads = 2, .decode = true, .length = 129, .sink = true });
    try check(.f16, .{ .head_dim = 512, .full_batch = true, .group = 3, .heads = 2, .decode = true, .length = 65 });
}

test "CuTe persistent sliding decode masks and sharding" {
    // Odd GQA, inactive slots and unequal numbers of partitions per worker.
    try check(.bf16, .{ .head_dim = 256, .full_batch = true, .active_slots = 19, .group = 3, .heads = 2, .decode = true, .length = 1105, .window = 513, .sink = true, .q_scale = 64 });
    try check(.f16, .{ .head_dim = 256, .full_batch = true, .group = 4, .heads = 2, .decode = true, .length = 257, .window = 129, .page_size = 32, .sink = true });
    const platform = zml.testing.env();
    if (platform.devices.len == 2) {
        try check(.bf16, .{ .head_dim = 256, .full_batch = true, .group = 2, .heads = 4, .decode = true, .length = 1027, .window = 1024, .shard = .kv });
    }
}

test "CuTe large tile non-split sliding decode" {
    // Physical pages remain 16 tokens; exercise partial tiles, window wrapping,
    // inactive slots, GQA tails, sinks, and both supported input formats.
    try check(.bf16, .{ .head_dim = 256, .full_batch = true, .heads = 8, .group = 2, .decode = true, .length = 129, .window = 1024 });
    try check(.bf16, .{ .head_dim = 256, .full_batch = true, .heads = 3, .group = 3, .decode = true, .length = 1025, .window = 1024, .sink = true, .active_slots = 251 });
    try check(.f16, .{ .head_dim = 256, .full_batch = true, .heads = 4, .group = 4, .decode = true, .length = 1430, .window = 1024, .sink = true });
    try check(.bf16, .{ .head_dim = 256, .full_batch = true, .heads = 8, .group = 2, .decode = true, .length = 1023, .window = 1024, .shard = .kv });
}
