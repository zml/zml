const std = @import("std");

const zml = @import("../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const Value = cute.Value;

pub const Config = struct {
    batch: i64,
    query_length: i64,
    key_length: i64,
    query_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    dtype: cute.DType,
    splits: i64 = 1,
};

/// Planning uses local (post-sharding) dimensions. XLA owns the compiled CUDA
/// function, so this is an estimated resource budget, not an occupancy query.
pub const DecodeResources = struct {
    multiprocessors: i64,
    shared_bytes: i64,
};

pub fn decodeSplits(q: zml.Shape, k: zml.Shape, resources: DecodeResources) i64 {
    const query_length = q.dim(.q);
    const key_length = k.dim(.k);
    const num_heads = q.dim(.h);
    const num_kv_heads = k.dim(.h);
    const head_dim = q.dim(.hd);
    const batch = if (q.hasTag(.b) != null) q.dim(.b) else 1;
    const dtype = q.dtype();

    if (query_length != 1 or key_length <= 64) return 1;
    if (resources.multiprocessors <= 0 or resources.shared_bytes <= 0) return 1;
    const group = @divExact(num_heads, num_kv_heads);
    const blocks_per_sequence = num_kv_heads * @divTrunc(group + 3, 4);
    const base_blocks = batch * blocks_per_sequence;
    var padded_dim: i64 = 32;
    while (padded_dim < head_dim) padded_dim *= 2;
    const bytes_per_element: i64 = if (dtype == .f32) 4 else 2;
    const shared_per_block = 2 * tile_keys * padded_dim * bytes_per_element;
    const blocks_per_sm = @max(1, @min(12, @divTrunc(resources.shared_bytes, shared_per_block)));
    // Multiple scheduling waves keep the serial tile loop supplied with work.
    // The measured tradeoff and distinction from CUDA occupancy are in README.
    const grid_budget = 4 * resources.multiprocessors * blocks_per_sm;
    if (base_blocks >= grid_budget) return 1;
    // Round up a partial wave: flooring can leave SMs idle and add a whole
    // extra KV tile to every block (e.g. 31 partitions versus 32).
    const split_budget = @divTrunc(grid_budget + base_blocks - 1, base_blocks);
    // One tile for short caches, two otherwise. FlashInfer's larger chunk
    // floor is tuned for a different per-block KV processing rate.
    // With static dense shapes, capacity bounds the scratch space and grid;
    // each invocation balances its live prefix over these partitions on device.
    const min_chunk: i64 = if (key_length <= 256) tile_keys else 2 * tile_keys;
    const chunk = @max(min_chunk, @divTrunc(key_length + split_budget - 1, split_budget));
    return @divTrunc(key_length + chunk - 1, chunk);
}

test "CuTe decode planner respects workload and resource limits" {
    const resources: DecodeResources = .{ .multiprocessors = 170, .shared_bytes = 101376 };
    var q = zml.Shape.init(.{ .b = 1, .q = 1, .h = 16, .hd = 128 }, .bf16);
    var k = zml.Shape.init(.{ .b = 1, .k = 2048, .h = 4, .hd = 128 }, .bf16);
    try std.testing.expectEqual(64, decodeSplits(q, k, resources));
    k = k.setDim(.k, 64);
    try std.testing.expectEqual(1, decodeSplits(q, k, resources));
    k = k.setDim(.k, 65);
    try std.testing.expectEqual(5, decodeSplits(q, k, resources));
    k = k.setDim(.k, 2048);
    var previous: i64 = 64;
    for ([_]i64{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 }) |batch| {
        q = q.setDim(.b, batch);
        k = k.setDim(.b, batch);
        const splits = decodeSplits(q, k, resources);
        try std.testing.expect(splits <= previous);
        try std.testing.expect(splits >= 1);
        try std.testing.expect(splits == 1 or (splits - 1) * batch * 4 < 8160);
        previous = splits;
    }
    try std.testing.expectEqual(1, previous);
    q = q.setDim(.b, 16);
    k = k.setDim(.b, 16);
    q = q.setDim(.hd, 512);
    k = k.setDim(.hd, 512);
    q = q.withDtype(.f32);
    k = k.withDtype(.f32);
    try std.testing.expectEqual(11, decodeSplits(q, k, resources));
    q = q.setDim(.q, 7);
    try std.testing.expectEqual(1, decodeSplits(q, k, resources));
    q = q.setDim(.q, 1);
    try std.testing.expectEqual(1, decodeSplits(q, k, .{ .multiprocessors = 0, .shared_bytes = 0 }));
    // More query-head groups already create more independent CTAs.
    q = q.setDim(.b, 1);
    k = k.setDim(.b, 1);
    q = q.setDim(.hd, 128);
    k = k.setDim(.hd, 128);
    q = q.withDtype(.bf16);
    k = k.withDtype(.bf16);
    q = q.setDim(.h, 1024);
    try std.testing.expectEqual(32, decodeSplits(q, k, resources));
}

pub const block: [3]i32 = .{ 128, 1, 1 };
const tile_keys = 16;

pub const Kernel = cute.Kernel(Config, .{
    .name = "flash_attention_cute",
    .inputs = &.{ "q", "k", "v", "positions" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *B, cfg: Config) cute.FinishError!void {
    std.debug.assert(cfg.head_dim > 0 and cfg.head_dim <= 512);
    inline for (.{ 1, 2, 4, 8, 16 }) |fragments| {
        if (cfg.head_dim <= fragments * 32) return runSized(b, cfg, fragments);
    }
    unreachable;
}

fn tupleOf(array: anytype) @Tuple(&@as([array.len]type, @splat(Value))) {
    var result: @Tuple(&@as([array.len]type, @splat(Value))) = undefined;
    inline for (0..array.len) |i| result[i] = array[i];
    return result;
}

/// Four warps own four (query position, query head) rows sharing one KV head.
/// Only a KV tile lives in shared memory; maxima, sums, and outputs stay in
/// registers. All lanes participate in shuffles, including padded rows/features.
fn runSized(b: *B, cfg: Config, comptime fragments: usize) cute.FinishError!void {
    const padded_dim = fragments * 32;
    const split = cfg.splits > 1;
    const out_shape = if (split) &[_]i64{ cfg.batch, cfg.splits, cfg.query_length, cfg.query_heads, cfg.head_dim + 2 } else &[_]i64{ cfg.batch, cfg.query_length, cfg.query_heads, cfg.head_dim };
    const a = try b.declareArgs(.{
        .q = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.query_length, cfg.query_heads, cfg.head_dim } } },
        .k = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.key_length, cfg.kv_heads, cfg.head_dim } } },
        .v = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.key_length, cfg.kv_heads, cfg.head_dim } } },
        .positions = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.batch} } },
        .out = .{ .tensor = .{ .dtype = if (split) .f32 else cfg.dtype, .shape = out_shape } },
    });
    const tid = b.threadIdx().x;
    const lane = tid.rem(32);
    const group = @divExact(cfg.query_heads, cfg.kv_heads);
    const batch = b.blockIdx().y.div(cfg.kv_heads);
    const kv_head = b.blockIdx().y.rem(cfg.kv_heads);
    const first_row = b.blockIdx().x.mul(4);
    const row = first_row.add(tid.div(32));
    // Padded warps compute a valid dummy row and suppress their final stores.
    const query = row.div(group).minimum(cfg.query_length - 1);
    const head = kv_head.mul(group).add(row.rem(group));
    const start = a.positions.get(.{batch});
    const position = start.add(query);
    const causal_end = start.add(first_row.add(3).div(group).minimum(cfg.query_length - 1)).add(1);
    // Partition the live cache, rather than its capacity, so decode uses all
    // blocks even when only a prefix of the cache is populated.
    const chunk = causal_end.add(cfg.splits - 1).div(cfg.splits);
    const begin = if (split) b.blockIdx().z.mul(chunk) else b.cst(.i32, 0);
    const end = if (split) begin.add(chunk).minimum(causal_end) else causal_end;
    const zero = b.cst(.f32, 0);
    const neg_inf = b.cst(.f32, -std.math.inf(f32));

    var queries: [fragments]Value = undefined;
    inline for (0..fragments) |f| {
        const d = lane.add(f * 32);
        var valid = b.openIfElse(d.lt(cfg.head_dim), .{cute.DType.f32.toMlir(b.ctx)});
        valid.yieldThen(.{a.q.get(.{ batch, query, head, d }).to(.f32)});
        valid.yieldElse(.{zero});
        queries[f] = valid.results[0];
    }

    const layout = b.makeLayout(&.{ tile_keys, padded_dim }, .{ .stride = &.{ padded_dim, 1 } });
    const keys = b.makeTensor(b.allocSmem(cfg.dtype, tile_keys * padded_dim, 16), layout);
    const values = b.makeTensor(b.allocSmem(cfg.dtype, tile_keys * padded_dim, 16), layout);
    var initial: [fragments + 2]Value = @splat(zero);
    initial[0] = neg_inf;
    var tiles = b.openFor(begin, end, tile_keys, tupleOf(initial));

    // Zero-fill partial tiles and feature padding before the CTA-wide barrier.
    var copy = b.openFor(tid, tile_keys * padded_dim, 128, .{});
    const tile_key = copy.iv.div(padded_dim);
    const d = copy.iv.rem(padded_dim);
    const key = tiles.iv.add(tile_key);
    var valid_load = b.openIfElse(key.lt(end).bitAnd(d.lt(cfg.head_dim)), .{ cfg.dtype.toMlir(b.ctx), cfg.dtype.toMlir(b.ctx) });
    valid_load.yieldThen(.{ a.k.get(.{ batch, key, kv_head, d }), a.v.get(.{ batch, key, kv_head, d }) });
    valid_load.yieldElse(.{ zero.to(cfg.dtype), zero.to(cfg.dtype) });
    keys.set(.{ tile_key, d }, valid_load.results[0]);
    values.set(.{ tile_key, d }, valid_load.results[1]);
    copy.close();
    b.syncThreads();

    // Each lane keeps one score; dot products reduce across the feature lanes.
    var scores = b.openFor(b.cst(.i32, 0), tile_keys, 1, .{neg_inf});
    var dot = zero;
    inline for (0..fragments) |f| {
        dot = dot.add(queries[f].mul(keys.get(.{ scores.iv, lane.add(f * 32) }).to(.f32)));
    }
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| dot = dot.add(b.shuffleXor(dot, offset));
    const scale = std.math.log2e / std.math.sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const score = b.select(tiles.iv.add(scores.iv).le(position).bitAnd(tiles.iv.add(scores.iv).lt(end)), dot.mul(scale), neg_inf);
    scores.yield(.{b.select(lane.eq(scores.iv), score, scores.carried[0])});
    var maximum = scores.results[0];
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    maximum = maximum.maximum(tiles.carried[0]);
    const rescale = b.exp2Approx(tiles.carried[0].sub(maximum));
    const probability = b.exp2Approx(scores.results[0].sub(maximum));
    var sum = probability;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| sum = sum.add(b.shuffleXor(sum, offset));
    sum = sum.add(tiles.carried[1].mul(rescale));

    var accumulators: [fragments]Value = undefined;
    inline for (0..fragments) |f| accumulators[f] = tiles.carried[f + 2].mul(rescale);
    var weighted = b.openFor(b.cst(.i32, 0), tile_keys, 1, tupleOf(accumulators));
    const weight = b.shuffleIdx(probability, weighted.iv);
    inline for (0..fragments) |f| {
        accumulators[f] = weighted.carried[f].add(weight.mul(values.get(.{ weighted.iv, lane.add(f * 32) }).to(.f32)));
    }
    weighted.yield(tupleOf(accumulators));
    // No warp may overwrite the tile before every warp has consumed it.
    b.syncThreads();
    var next: [fragments + 2]Value = undefined;
    next[0] = maximum;
    next[1] = sum;
    inline for (0..fragments) |f| next[f + 2] = weighted.results[f];
    tiles.yield(tupleOf(next));

    inline for (0..fragments) |f| {
        const col = lane.add(f * 32);
        var valid_store = b.openIf(row.lt(cfg.query_length * group).bitAnd(col.lt(cfg.head_dim)));
        if (split) {
            a.out.set(.{ batch, b.blockIdx().z, query, head, col }, tiles.results[f + 2]);
        } else {
            a.out.set(.{ batch, query, head, col }, tiles.results[f + 2].div(tiles.results[1]));
        }
        valid_store.close();
    }
    if (split) {
        var stats = b.openIf(row.lt(cfg.query_length * group).bitAnd(lane.eq(0)));
        a.out.set(.{ batch, b.blockIdx().z, query, head, cfg.head_dim }, tiles.results[0]);
        a.out.set(.{ batch, b.blockIdx().z, query, head, cfg.head_dim + 1 }, tiles.results[1]);
        stats.close();
    }
}

/// Merge unnormalized partial attention states in FP32. Empty partitions have
/// m=-inf, l=0, u=0 and contribute zero, including at the first decode token.
pub const Merge = cute.Kernel(Config, .{
    .name = "flash_attention_cute_merge",
    .inputs = &.{"partials"},
    .outputs = &.{"out"},
    .run = merge,
});

fn merge(b: *B, cfg: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.batch, cfg.splits, cfg.query_heads, cfg.head_dim + 2 } } },
        .out = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.query_heads, cfg.head_dim } } },
    });
    const batch = b.blockIdx().y;
    const head = b.blockIdx().x;
    const lane = b.threadIdx().x.rem(32);
    const zero = b.cst(.f32, 0);
    const neg_inf = b.cst(.f32, -std.math.inf(f32));
    // Every warp reduces the partition statistics, then independently handles
    // a contiguous range of output features.
    var maxima = b.openFor(lane, cfg.splits, 32, .{neg_inf});
    maxima.yield(.{maxima.carried[0].maximum(a.partials.get(.{ batch, maxima.iv, head, cfg.head_dim }))});
    var maximum = maxima.results[0];
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    const weights = b.makeTensor(b.allocSmem(.f32, cfg.splits, 16), b.makeLayout(&.{cfg.splits}, .{}));
    var stats = b.openFor(b.threadIdx().x, cfg.splits, 128, .{});
    weights.set(.{stats.iv}, b.exp2Approx(a.partials.get(.{ batch, stats.iv, head, cfg.head_dim }).sub(maximum)));
    stats.close();
    b.syncThreads();
    var features = b.openFor(b.threadIdx().x, cfg.head_dim, 128, .{});
    var sum = b.openFor(b.cst(.i32, 0), cfg.splits, 1, .{ zero, zero });
    const weight = weights.get(.{sum.iv});
    sum.yield(.{
        sum.carried[0].add(weight.mul(a.partials.get(.{ batch, sum.iv, head, features.iv }))),
        sum.carried[1].add(weight.mul(a.partials.get(.{ batch, sum.iv, head, cfg.head_dim + 1 }))),
    });
    a.out.set(.{ batch, head, features.iv }, sum.results[0].div(sum.results[1]));
    features.close();
}

test "CuTe attention emits tiled online softmax through head dimension 512" {
    for ([_]i64{ 64, 80, 128, 256, 512 }) |head_dim| {
        const ir = try Kernel.emit(std.testing.allocator, .{
            .batch = 2,
            .query_length = 7,
            .key_length = 37,
            .query_heads = 6,
            .kv_heads = 2,
            .head_dim = head_dim,
            .dtype = .bf16,
        }, block);
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "scf.for") != null);
    }
}
