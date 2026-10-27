const std = @import("std");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;
const sm103 = @import("sm103.zig");

pub const Architecture = enum { sm100, sm103 };

pub const Config = struct {
    architecture: Architecture,
    head_dim: i64,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages_per_seq: i64,
    page_size: i64,
    max_query: i64,
    dtype: cute.DType,
    decode: bool,
    causal: bool,
    window: i64,
    scale: f32,
    sink: bool,
    splits: i64 = 1,

    pub fn rows(c: Config) i64 {
        return if (c.decode) (if (c.group <= 8) 8 else 16) else 64;
    }
    pub fn keys(c: Config) i64 {
        return if (c.decode) 128 else 64;
    }
    pub fn queryTiles(c: Config) i64 {
        return @divTrunc(c.max_query * c.group + c.rows() - 1, c.rows());
    }
    fn paddedDim(c: Config) i64 {
        return @max(128, c.head_dim);
    }
    fn features(c: Config) i64 {
        return @divExact(c.paddedDim(), 128);
    }
    fn buffers(c: Config) i64 {
        return if (c.decode) 2 else 3;
    }
    fn kvBytes(c: Config) i64 {
        return c.keys() * 128 * 2;
    }
    fn qBase(c: Config) i64 {
        return c.buffers() * c.kvBytes();
    }
    fn pBase(c: Config) i64 {
        return c.qBase() + c.rows() * c.paddedDim() * 2;
    }
    fn barriers(c: Config) i64 {
        return c.pBase() + c.rows() * c.keys() * 2;
    }
    fn completion(c: Config) i64 {
        return c.barriers() + c.buffers() * 16;
    }
    fn holder(c: Config) i64 {
        return c.completion() + 16;
    }
    fn reductions(c: Config) i64 {
        return c.holder() + 16;
    }
    fn sharedBytes(c: Config) i64 {
        return c.reductions() + 4 * 16 * 4;
    }
    fn scoreColumn(c: Config) i64 {
        return if (c.decode) c.features() * c.rows() else @min(c.paddedDim(), 448);
    }
    fn columns(c: Config) u32 {
        return std.math.ceilPowerOfTwoAssert(u32, @intCast(@max(32, c.scoreColumn() + (if (c.decode) c.rows() else c.keys()))));
    }
};

pub const block: [3]i32 = .{ 256, 1, 1 };
pub const Attention = cute.Kernel(Config, .{
    .name = "paged_attention_cute",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink" },
    .outputs = &.{"out"},
    .run = attention,
});

fn loadInt(b: *B, pointer: V, index: V) V {
    return b.loadWords(1, pointer.add(index.to(.i64).mul(4)), .global, .{})[0];
}

// Each 64-feature plane uses a 128-byte XOR swizzle. The descriptor and
// vector copies use this same layout for Q, K, V and probabilities.
fn swizzle(row: V, col: V, rows: i64) V {
    return col.div(64).mul(rows * 128).add(row.mul(128)).add(col.rem(64).div(8).bitXor(row.rem(8)).mul(16)).add(col.rem(8).mul(2));
}

fn descriptor(b: *B, address: V, leading_bytes: i64) V {
    const bits: u64 = (64 << 32) | (1 << 46) | (@as(u64, 2) << 61) | (@as(u64, @intCast(@divExact(leading_bytes, 16))) << 16);
    return address.to(.i64).shrLogical(4).bitAnd(0x3fff).bitOr(b.constant(.i64, @as(i64, @bitCast(bits))));
}

fn instruction(c: Config, qk: bool) u32 {
    const format: u32 = if (c.dtype == .bf16) 1 else 0;
    const m: u32 = if (c.decode) 128 else 64;
    const n: u32 = @intCast(if (c.decode) c.rows() else if (qk) c.keys() else 128);
    return (1 << 4) | (format << 7) | (format << 10) |
        (@as(u32, @intFromBool(c.decode and !qk)) << 15) |
        (@as(u32, @intFromBool(!c.decode and !qk)) << 16) | ((n >> 3) << 17) | ((m >> 4) << 24);
}

const Range = struct { first: V, count: V };

// Used by both attention and merge: every output row must merge exactly the
// partitions produced for its query tile, including causal/window boundaries.
fn keyRange(b: *B, c: Config, first_row: V, qlen: V, klen: V) Range {
    const first_pos = klen.sub(qlen).add(first_row.div(c.group));
    const last_pos = klen.sub(qlen).add(first_row.add(c.rows() - 1).div(c.group).minimum(qlen.sub(1)));
    const begin = if (c.window > 0) first_pos.sub(c.window - 1).maximum(0) else b.constant(.i32, 0);
    const end = if (c.causal) last_pos.add(1).minimum(klen) else klen;
    const first = begin.div(c.keys());
    return .{ .first = first, .count = end.add(c.keys() - 1).div(c.keys()).sub(first).maximum(0) };
}

fn liveSplits(c: Config, tiles: V) V {
    return tiles.add(1).div(2).maximum(1).minimum(c.splits);
}

const Tile = struct {
    b: *B,
    c: Config,
    q: V,
    k: V,
    v: V,
    table: V,
    sink: V,
    out: V,
    shared: V,
    tmem: V,
    tid: V,
    lane: V,
    warp: V,
    seq: V,
    head: V,
    first_row: V,
    start: V,
    qlen: V,
    klen: V,
    first_key_tile: V,
    tiles: V,
    split: V,

    fn ready(t: Tile, id: V) V {
        return t.shared.add(t.c.barriers()).add(id.rem(t.c.buffers()).mul(8));
    }
    fn released(t: Tile, id: V) V {
        return t.ready(id).add(t.c.buffers() * 8);
    }
    fn phase(t: Tile, id: V) V {
        return id.div(t.c.buffers()).rem(2);
    }
    fn valid(t: Tile, row: V, key: V) V {
        const pos = t.klen.sub(t.qlen).add(row.div(t.c.group));
        var result = row.lt(t.qlen.mul(t.c.group)).bitAnd(key.lt(t.klen));
        if (t.c.causal) result = result.bitAnd(key.le(pos));
        if (t.c.window > 0) result = result.bitAnd(key.gt(pos.sub(t.c.window)));
        return result;
    }
    fn putProbability(t: Tile, row: V, key: V, value: V) void {
        t.b.storeHalf(t.shared.add(t.c.pBase()).add(swizzle(row, key, t.c.rows())), .shared, value.to(t.c.dtype).bitCast(.i16).to(.i32));
    }
    fn put(t: Tile, row: V, dim: V, value: V) void {
        const token = t.start.add(row.div(t.c.group));
        var index = token.to(.i64).mul(t.c.kv_heads).add(t.head.to(.i64)).mul(t.c.group).add(row.rem(t.c.group).to(.i64));
        if (t.c.splits > 1) {
            index = t.split.to(.i64).mul(t.c.tokens * t.c.kv_heads * t.c.group).add(index).mul(t.c.head_dim + 2).add(dim.to(.i64));
            t.b.storeWords(1, t.out.add(index.mul(4)), .global, .{value.bitCast(.i32)});
        } else {
            index = index.mul(t.c.head_dim).add(dim.to(.i64));
            t.b.storeHalf(t.out.add(index.mul(2)), .global, value.to(t.c.dtype).bitCast(.i16).to(.i32));
        }
    }
    fn norm(t: Tile, row: V, maximum: V, denominator: V) V {
        const b = t.b;
        if (t.c.splits > 1) return b.constant(.f32, 1);
        if (!t.c.sink) return b.select(denominator.gt(0), b.rcpApprox(denominator), b.constant(.f32, 0));
        const sink = loadInt(b, t.sink, t.head.mul(t.c.group).add(row.rem(t.c.group))).bitCast(.f32).mul(std.math.log2e);
        const reference = maximum.maximum(sink).maximum(-std.math.floatMax(f32));
        const weight = b.exp2Approx(maximum.sub(reference));
        return weight.div(denominator.mul(weight).add(b.exp2Approx(sink.sub(reference))));
    }
};

fn attention(b: *B, c: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .q = .{ .ptr = c.dtype },
        .k = .{ .ptr = c.dtype },
        .v = .{ .ptr = c.dtype },
        .table = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .starts = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .out = .{ .ptr = if (c.splits > 1) cute.DType.f32 else c.dtype },
    });
    std.debug.assert(c.sharedBytes() <= 227 * 1024);
    b.setFunctionAttribute("cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())}));
    const seq = b.blockIdx().x.div(c.queryTiles());
    const first_row = b.blockIdx().x.rem(c.queryTiles()).mul(c.rows());
    const starts = b.ptrToInt(a.starts, .i64);
    const start = loadInt(b, starts, seq);
    const qlen = loadInt(b, starts, seq.add(1)).sub(start);
    const klen = loadInt(b, b.ptrToInt(a.lengths, .i64), seq);
    const range = keyRange(b, c, first_row, qlen, klen);
    const splits = liveSplits(c, range.count);
    const split = b.blockIdx().z;
    var active = b.openIf(first_row.lt(qlen.mul(c.group)).bitAnd(split.lt(splits)));
    {
        const tid = b.threadIdx().x;
        const lo = range.count.mul(split).div(splits);
        const hi = range.count.mul(split.add(1)).div(splits);
        const shared = b.ptrToInt(b.allocSmem(.i8, @intCast(c.sharedBytes()), 1024), .i32);
        var t: Tile = .{
            .b = b,
            .c = c,
            .q = b.ptrToInt(a.q, .i64),
            .k = b.ptrToInt(a.k, .i64),
            .v = b.ptrToInt(a.v, .i64),
            .table = b.ptrToInt(a.table, .i64),
            .sink = b.ptrToInt(a.sink, .i64),
            .out = b.ptrToInt(a.out, .i64),
            .shared = shared,
            .tmem = undefined,
            .tid = tid,
            .lane = tid.rem(32),
            .warp = tid.div(32),
            .seq = seq,
            .head = b.blockIdx().y,
            .first_row = first_row,
            .start = start,
            .qlen = qlen,
            .klen = klen,
            .first_key_tile = range.first.add(lo),
            .tiles = hi.sub(lo),
            .split = split,
        };
        var leader = b.openIf(tid.eq(0));
        {
            for (0..@as(usize, @intCast(c.buffers()))) |i| {
                b.mbarrierInit(shared.add(c.barriers() + @as(i64, @intCast(i * 8))), 128);
                b.mbarrierInit(shared.add(c.barriers() + c.buffers() * 8 + @as(i64, @intCast(i * 8))), 1);
            }
            b.mbarrierInit(shared.add(c.completion()), 1);
            b.fenceMbarrierInit();
            leader.close();
        }
        var alloc = b.openIf(tid.lt(32));
        {
            b.tmemAllocAt(shared.add(c.holder()), c.columns());
            b.relinquishTmemAllocPermit(.{});
            alloc.close();
        }
        var queries = b.openFor(tid, c.rows() * @divExact(c.paddedDim(), 8), 256, .{});
        {
            const row = queries.iv.div(@divExact(c.paddedDim(), 8));
            const col = queries.iv.rem(@divExact(c.paddedDim(), 8)).mul(8);
            const packed_row = first_row.add(row);
            const valid = packed_row.lt(qlen.mul(c.group)).bitAnd(col.lt(c.head_dim));
            const token = start.add(b.select(valid, packed_row.div(c.group), b.constant(.i32, 0)));
            const offset = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(packed_row.rem(c.group).to(.i64)).mul(c.head_dim).add(col.to(.i64));
            b.cpAsync16(shared.add(c.qBase()).add(swizzle(row, col, c.rows())), t.q.add(offset.mul(2)), valid);
            queries.close();
        }
        b.cpAsyncWaitAll();
        b.syncThreads();
        t.tmem = b.loadWords(1, shared.add(c.holder()), .shared, .{})[0];
        b.tcgen05Fence(.after);
        var producer = b.openIf(tid.ge(128));
        {
            produce(t);
            producer.close();
        }
        var consumer = b.openIf(tid.lt(128));
        {
            if (c.decode) {
                if (c.group <= 2) decode(t, 2) else if (c.group <= 4) decode(t, 4) else if (c.group <= 8) decode(t, 8) else decode(t, 16);
            } else prefill(t);
            consumer.close();
        }
        b.tcgen05Fence(.before);
        b.syncThreads();
        var free = b.openIf(tid.lt(32));
        {
            b.tmemDeallocAt(t.tmem, c.columns());
            free.close();
        }
        active.close();
    }
}

fn produce(t: Tile) void {
    const b = t.b;
    const c = t.c;
    const row0 = t.tid.sub(128).div(16);
    const col = t.tid.rem(16).mul(8);
    var tiles = b.openFor(0, t.tiles, 1, .{});
    {
        // Paged offsets are shared by K/V and every 128-feature stage.
        var offsets: [16]V = undefined;
        var valid: [16]V = undefined;
        for (0..@as(usize, @intCast(@divExact(c.keys(), 8)))) |i| {
            const key = t.first_key_tile.add(tiles.iv).mul(c.keys()).add(row0).add(i * 8);
            valid[i] = key.lt(t.klen).bitAnd(col.lt(c.head_dim));
            const safe = b.select(key.lt(t.klen), key, b.constant(.i32, 0));
            const page = loadInt(b, t.table, t.seq.mul(c.pages_per_seq).add(safe.div(c.page_size)));
            offsets[i] = page.to(.i64).mul(c.page_size).add(safe.rem(c.page_size).to(.i64)).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.head_dim).add(col.to(.i64)).mul(2);
        }
        for (0..@as(usize, @intCast(c.features() * 2))) |stage| {
            const id = tiles.iv.mul(c.features() * 2).add(stage);
            var reuse = b.openIf(id.ge(c.buffers()));
            {
                b.mbarrierWait(t.released(id), t.phase(id).bitXor(1));
                reuse.close();
            }
            const source = if (stage < c.features()) t.k else t.v;
            const feature_offset: i64 = @mod(@as(i64, @intCast(stage)), c.features()) * 256;
            for (0..@as(usize, @intCast(@divExact(c.keys(), 8)))) |i| {
                const row = row0.add(i * 8);
                b.cpAsync16(t.shared.add(id.rem(c.buffers()).mul(c.kvBytes())).add(swizzle(row, col, c.keys())), source.add(offsets[i]).add(feature_offset), valid[i]);
            }
            b.cpAsyncMbarrierArrive(t.ready(id), true);
        }
        tiles.close();
    }
}

fn multiply(t: Tile, tile: V, comptime qk: bool) void {
    const b = t.b;
    const c = t.c;
    var leader = b.openIf(t.tid.eq(0));
    {
        for (0..@as(usize, @intCast(c.features()))) |part| {
            const id = tile.mul(c.features() * 2).add(@as(i64, @intCast(part)) + if (qk) @as(i64, 0) else c.features());
            b.mbarrierWait(t.ready(id), t.phase(id));
            b.tcgen05Fence(.after);
            const kv = descriptor(b, t.shared.add(id.rem(c.buffers()).mul(c.kvBytes())), if (qk) 16 else c.keys() * 128);
            const qp = descriptor(b, t.shared.add(if (qk) c.qBase() else c.pBase()), 16);
            const steps: usize = @intCast(if (qk) 8 else @divExact(c.keys(), 16));
            for (0..steps) |step| {
                const kv_offset: i64 = if (qk) @as(i64, @intCast(step / 4)) * c.keys() * 128 + @as(i64, @intCast(step % 4)) * 32 else @as(i64, @intCast(step)) * 2048;
                const qp_offset: i64 = (if (qk) @as(i64, @intCast(part * 2 + step / 4)) else @as(i64, @intCast(step / 4))) * c.rows() * 128 + @as(i64, @intCast(step % 4)) * 32;
                const left = if (c.decode) kv.add(@divExact(kv_offset, 16)) else qp.add(@divExact(qp_offset, 16));
                const right = if (c.decode) qp.add(@divExact(qp_offset, 16)) else kv.add(@divExact(kv_offset, 16));
                const column: i64 = if (qk) c.scoreColumn() else @as(i64, @intCast(part)) * (if (c.decode) c.rows() else 128);
                const accumulate = if (qk) b.constant(.i32, @intFromBool(part > 0 or step > 0)).ne(0) else tile.gt(0).bitOr(b.constant(.i32, @intFromBool(step > 0)).ne(0));
                b.tcgen05MmaF16(t.tmem.add(column), left, right, b.constant(.i32, @as(i32, @bitCast(instruction(c, qk)))), accumulate);
            }
            b.tcgen05Commit(t.released(id));
        }
        b.tcgen05Commit(t.shared.add(c.completion()));
        leader.close();
    }
    b.mbarrierWait(t.shared.add(c.completion()), b.constant(.i32, if (qk) 0 else 1));
    b.tcgen05Fence(.after);
    // A fast warp must not wrap completion parity before every consumer has
    // observed the preceding PV completion.
    if (!qk) b.barrierSync(1, 128);
}

fn publishProbabilities(t: Tile) void {
    t.b.fenceProxyAsyncShared();
    t.b.tcgen05Fence(.before);
    t.b.barrierSync(1, 128);
    t.b.tcgen05Fence(.after);
}

fn reduceRow(comptime n: usize, values: [n]V, comptime maximum: bool) V {
    var tree = values;
    var width: usize = n;
    while (width > 1) {
        width /= 2;
        for (0..width) |i| tree[i] = if (maximum) tree[i * 2].maximum(tree[i * 2 + 1]) else tree[i * 2].add(tree[i * 2 + 1]);
    }
    return tree[0];
}

fn reduceHeads(t: Tile, comptime heads: usize, values: [heads]V, comptime maximum: bool) [heads]V {
    const b = t.b;
    for (values, 0..) |value, h| {
        var reduced = value;
        inline for (.{ 16, 8, 4, 2, 1 }) |step| reduced = if (maximum) reduced.maximum(b.shuffleXor(reduced, step)) else reduced.add(b.shuffleXor(reduced, step));
        var leader = b.openIf(t.lane.eq(0));
        {
            b.storeWords(1, t.shared.add(t.c.reductions()).add(t.warp.mul(64)).add(h * 4), .shared, .{reduced.bitCast(.i32)});
            leader.close();
        }
    }
    b.barrierSync(1, 128);
    var result: [heads]V = undefined;
    for (0..heads) |h| {
        var value = b.loadWords(1, t.shared.add(t.c.reductions() + @as(i64, @intCast(h * 4))), .shared, .{})[0].bitCast(.f32);
        for (1..4) |w| {
            const next = b.loadWords(1, t.shared.add(t.c.reductions() + @as(i64, @intCast(w * 64 + h * 4))), .shared, .{})[0].bitCast(.f32);
            value = if (maximum) value.maximum(next) else value.add(next);
        }
        result[h] = value;
    }
    b.barrierSync(1, 128);
    return result;
}

fn decode(t: Tile, comptime heads: usize) void {
    const b = t.b;
    const c = t.c;
    const address = t.tmem.add(t.warp.mul(32).shl(16));
    const key_row = t.warp.mul(32).add(t.lane);
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));
    var initial: [2 * heads]V = undefined;
    for (0..heads) |h| {
        initial[h] = neg_inf;
        initial[heads + h] = zero;
    }
    var tiles = b.openFor(0, t.tiles, 1, initial);
    {
        multiply(t, tiles.iv, true);
        const raw = b.tmemLoad32x32b(heads, address.add(c.scoreColumn()));
        b.fenceTmemLoad();
        const key = t.first_key_tile.add(tiles.iv).mul(c.keys()).add(key_row);
        var scores: [heads]V = undefined;
        for (0..heads) |h| scores[h] = b.select(t.valid(t.first_row.add(h), key), raw[h].bitCast(.f32).mul(c.scale * std.math.log2e), neg_inf);
        const maxima = reduceHeads(t, heads, scores, true);
        var next: [2 * heads]V = undefined;
        var alpha: [heads]V = undefined;
        var probabilities: [heads]V = undefined;
        for (0..heads) |h| {
            next[h] = tiles.carried[h].maximum(maxima[h]);
            const reference = next[h].maximum(-std.math.floatMax(f32));
            alpha[h] = b.exp2Approx(tiles.carried[h].sub(reference));
            probabilities[h] = b.exp2Approx(scores[h].sub(reference));
        }
        const sums = reduceHeads(t, heads, probabilities, false);
        for (0..heads) |h| {
            next[heads + h] = tiles.carried[heads + h].mul(alpha[h]).add(sums[h]);
            t.putProbability(b.constant(.i32, h), key_row, probabilities[h]);
        }
        for (heads..@as(usize, @intCast(c.rows()))) |h| t.putProbability(b.constant(.i32, h), key_row, zero);
        var rescale = b.openIf(tiles.iv.gt(0));
        {
            for (0..@as(usize, @intCast(c.features()))) |part| {
                const addr = address.add(@as(i64, @intCast(part)) * c.rows());
                var out = b.tmemLoad32x32b(heads, addr);
                b.fenceTmemLoad();
                for (0..heads) |h| out[h] = out[h].bitCast(.f32).mul(alpha[h]).bitCast(.i32);
                b.tmemStore32x32b(heads, addr, out);
            }
            b.fenceTmemStore();
            rescale.close();
        }
        publishProbabilities(t);
        multiply(t, tiles.iv, false);
        tiles.yield(next);
    }
    for (0..@as(usize, @intCast(c.features()))) |part| {
        const out = b.tmemLoad32x32b(heads, address.add(@as(i64, @intCast(part)) * c.rows()));
        b.fenceTmemLoad();
        const dim = key_row.add(part * 128);
        for (0..heads) |h| {
            const row = t.first_row.add(h);
            var valid = b.openIf(row.lt(t.qlen.mul(c.group)));
            {
                var feature = b.openIf(dim.lt(c.head_dim));
                {
                    const value = b.select(tiles.results[heads + h].gt(0), out[h].bitCast(.f32).mul(t.norm(row, tiles.results[h], tiles.results[heads + h])), zero);
                    t.put(row, dim, value);
                    feature.close();
                }
                if (c.splits > 1 and part == 0) {
                    var stats = b.openIf(t.tid.eq(0));
                    {
                        t.put(row, b.constant(.i32, c.head_dim), tiles.results[h]);
                        t.put(row, b.constant(.i32, c.head_dim + 1), tiles.results[heads + h]);
                        stats.close();
                    }
                }
                valid.close();
            }
        }
    }
}

fn prefill(t: Tile) void {
    const b = t.b;
    const c = t.c;
    const address = t.tmem.add(t.warp.mul(32).shl(16));
    const local_row = t.warp.mul(16).add(t.lane);
    const row = t.first_row.add(local_row);
    const owns = t.lane.lt(16);
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));
    var tiles = b.openFor(0, t.tiles, 1, .{ neg_inf, zero });
    {
        // D512 fills TMEM with O. Preserve its last 64 columns in registers
        // while QK reuses them for scores, avoiding duplicate QK/softmax CTAs.
        var saved: [64]V = undefined;
        if (c.head_dim == 512) {
            var preserve = b.openIfElse(tiles.iv.gt(0), @as([64]V, @splat(zero)));
            {
                const tail = b.tmemLoad32x32b(64, address.add(c.scoreColumn()));
                b.fenceTmemLoad();
                var words: [64]V = undefined;
                for (tail, 0..) |value, i| words[i] = value.bitCast(.f32);
                preserve.yieldThen(words);
            }
            {
                preserve.yieldElse(@as([64]V, @splat(zero)));
            }
            @memcpy(&saved, &preserve.results);
            b.barrierSync(1, 128);
        }
        multiply(t, tiles.iv, true);
        var scores: [64]V = undefined;
        var hardware_max = neg_inf;
        if (c.architecture == .sm103) {
            const loaded = sm103.loadScores(b, address.add(c.scoreColumn()));
            scores = loaded.values;
            hardware_max = loaded.maximum;
        } else {
            const loaded = b.tmemLoad32x32b(64, address.add(c.scoreColumn()));
            b.fenceTmemLoad();
            for (loaded, 0..) |value, i| scores[i] = value.bitCast(.f32);
        }
        const key0 = t.first_key_tile.add(tiles.iv).mul(c.keys());
        // Hardware maxima describe raw scores. Only fully visible tiles may
        // use them; page tails and mask boundaries reduce post-mask values.
        const full = owns.bitAnd(t.valid(row, key0)).bitAnd(t.valid(row, key0.add(63)));
        for (0..64) |i| {
            scores[i] = b.select(owns.bitAnd(t.valid(row, key0.add(i))), scores[i].mul(c.scale * std.math.log2e), neg_inf);
        }
        const tile_max = if (c.architecture == .sm103 and c.scale > 0) blk: {
            var choose = b.openIfElse(full, .{zero});
            {
                choose.yieldThen(.{hardware_max.mul(c.scale * std.math.log2e)});
            }
            {
                choose.yieldElse(.{reduceRow(64, scores, true)});
            }
            break :blk choose.results[0];
        } else reduceRow(64, scores, true);
        const maximum = tiles.carried[0].maximum(tile_max);
        const reference = maximum.maximum(-std.math.floatMax(f32));
        const alpha = b.exp2Approx(tiles.carried[0].sub(reference));
        var probabilities: [64]V = undefined;
        for (0..64) |i| {
            probabilities[i] = b.exp2Approx(scores[i].sub(reference));
        }
        const denominator = tiles.carried[1].mul(alpha).add(reduceRow(64, probabilities, false));
        var write = b.openIf(owns);
        {
            for (0..64) |i| t.putProbability(local_row, b.constant(.i32, i), probabilities[i]);
            write.close();
        }
        var rescale = b.openIf(tiles.iv.gt(0));
        {
            for (0..@as(usize, @intCast(@divExact(if (c.head_dim == 512) c.scoreColumn() else c.paddedDim(), 32)))) |chunk| {
                const addr = address.add(chunk * 32);
                var out = b.tmemLoad32x32b(32, addr);
                b.fenceTmemLoad();
                for (&out) |*value| value.* = value.bitCast(.f32).mul(alpha).bitCast(.i32);
                b.tmemStore32x32b(32, addr, out);
            }
            if (c.head_dim == 512) {
                for (&saved) |*value| value.* = value.mul(alpha).bitCast(.i32);
                b.tmemStore32x32b(64, address.add(c.scoreColumn()), saved);
            }
            b.fenceTmemStore();
            rescale.close();
        }
        publishProbabilities(t);
        multiply(t, tiles.iv, false);
        tiles.yield(.{ maximum, denominator });
    }
    const norm = t.norm(row, tiles.results[0], tiles.results[1]);
    for (0..@as(usize, @intCast(@divExact(c.paddedDim(), 32)))) |chunk| {
        const out = b.tmemLoad32x32b(32, address.add(chunk * 32));
        b.fenceTmemLoad();
        var valid = b.openIf(owns.bitAnd(row.lt(t.qlen.mul(c.group))));
        {
            for (0..32) |i| {
                const dim = chunk * 32 + i;
                if (dim < c.head_dim) t.put(row, b.constant(.i32, dim), b.select(tiles.results[1].gt(0), out[i].bitCast(.f32).mul(norm), zero));
            }
            if (c.splits > 1 and chunk == 0) {
                t.put(row, b.constant(.i32, c.head_dim), tiles.results[0]);
                t.put(row, b.constant(.i32, c.head_dim + 1), tiles.results[1]);
            }
            valid.close();
        }
    }
}

pub const Merge = cute.Kernel(Config, .{
    .name = "paged_attention_cute_merge",
    .inputs = &.{ "partials", "lengths", "starts", "sink" },
    .outputs = &.{"out"},
    .run = merge,
});

fn merge(b: *B, c: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 } } },
        .lengths = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch} } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{if (c.sink) c.kv_heads * c.group else 1} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    var active = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        var find = b.openFor(0, c.batch, 1, .{b.constant(.i32, 0)});
        {
            const owns = token.ge(a.starts.get(.{find.iv})).bitAnd(token.lt(a.starts.get(.{find.iv.add(1)})));
            find.yield(.{b.select(owns, find.iv, find.carried[0])});
        }
        const seq = find.results[0];
        const start = a.starts.get(.{seq});
        const qlen = a.starts.get(.{seq.add(1)}).sub(start);
        const row = token.sub(start).mul(c.group).add(head.rem(c.group));
        const range = keyRange(b, c, row.div(c.rows()).mul(c.rows()), qlen, a.lengths.get(.{seq}));
        const splits = liveSplits(c, range.count);
        const zero = b.constant(.f32, 0);
        const sink = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        var maximum = b.openFor(0, splits, 1, .{sink});
        {
            maximum.yield(.{maximum.carried[0].maximum(a.partials.get(.{ maximum.iv, token, head, c.head_dim }))});
        }
        const reference = maximum.results[0].maximum(-std.math.floatMax(f32));
        var dims = b.openFor(b.threadIdx().x, c.head_dim, 128, .{});
        {
            var sum = b.openFor(0, splits, 1, .{ zero, if (c.sink) b.exp2Approx(sink.sub(reference)) else zero });
            {
                const weight = b.exp2Approx(a.partials.get(.{ sum.iv, token, head, c.head_dim }).sub(reference));
                sum.yield(.{
                    sum.carried[0].add(weight.mul(a.partials.get(.{ sum.iv, token, head, dims.iv }))),
                    sum.carried[1].add(weight.mul(a.partials.get(.{ sum.iv, token, head, c.head_dim + 1 }))),
                });
            }
            const norm = b.select(sum.results[1].gt(0), b.rcpApprox(sum.results[1]), zero);
            a.out.set(.{ token, head, dims.iv }, sum.results[0].mul(norm).to(c.dtype));
            dims.close();
        }
        active.close();
    }
}

test "SM10x layouts and instruction feature gates" {
    for ([_]i64{ 32, 64, 128, 256, 512 }) |dim| for ([_]bool{ true, false }) |decode_only| for ([_]Architecture{ .sm100, .sm103 }) |architecture| {
        const c: Config = .{ .architecture = architecture, .head_dim = dim, .tokens = 80, .batch = 4, .kv_heads = 2, .group = 5, .pages_per_seq = 256, .page_size = 16, .max_query = if (decode_only) 1 else 17, .dtype = .bf16, .decode = decode_only, .causal = true, .window = 0, .scale = 0.125, .sink = true, .splits = 4 };
        try std.testing.expect(c.columns() >= 32 and c.columns() <= 512);
        try std.testing.expect(c.sharedBytes() <= 227 * 1024);
        const ir = try Attention.emit(std.testing.allocator, c, block);
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "tcgen05.mma") != null);
        try std.testing.expectEqual(architecture == .sm103 and !decode_only, std.mem.indexOf(u8, ir, "tcgen05.ld.red") != null);
        const merge_ir = try Merge.emit(std.testing.allocator, c, .{ 128, 1, 1 });
        defer std.testing.allocator.free(merge_ir);
    };
}
