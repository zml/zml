const std = @import("std");

const zml = @import("../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

pub const Config = struct {
    head_dim: i64 = 512,
    // Logical key tile; independent of physical cache page size.
    kv_tile: i64 = 64,
    load_buffers: i64 = 2,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages_per_seq: i64,
    page_size: i64,
    max_query: i64,
    dtype: cute.DType,
    all_decode: bool,
    causal: bool,
    window: i64,
    scale: f32,
    sink: bool,
    sms: i64,
    splits: i64,
    // A positive bound partitions sliding decode by work per CTA, even when
    // the batch already provides enough CTAs to fill the device.
    tiles_per_split: i64 = 0,
    // Fixed workers per sequence/head tile; zero launches each split separately.
    split_stride: i64 = 0,

    pub fn headsInX(c: Config) bool {
        return c.split_stride > 0 or (c.all_decode and c.kv_tile > 64);
    }
    pub fn rows(c: Config) i64 {
        return if (c.all_decode) decodeRows(c.group) else 64;
    }
    pub fn queryTiles(c: Config) i64 {
        return @divTrunc(c.max_query * c.group + c.rows() - 1, c.rows());
    }
    pub fn outputTiles(c: Config) i64 {
        return if (c.all_decode) 1 else @divExact(c.head_dim, 256);
    }
};

// Use legal MMA widths, with masked tails for arbitrary GQA ratios.
pub fn decodeRows(group: i64) i64 {
    return if (group <= 8) 8 else 16;
}

pub const block: [3]i32 = .{ 256, 1, 1 };
pub const Kernel = cute.Kernel(Config, .{
    .name = "paged_attention_cute",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink" },
    .outputs = &.{"out"},
    .run = run,
});

// KV buffers hold kv_tile keys by 128 features. The load ring streams D/128
// K stages followed by D/128 V stages while Q stays resident. Prefill retains
// 64-key tiles and 256 output features per CTA.
fn kvBytes(c: Config) i64 {
    return c.kv_tile * 128 * 2;
}
fn qBase(c: Config) i32 {
    return @intCast(c.load_buffers * kvBytes(c));
}
fn pBase(c: Config) i32 {
    return qBase(c) + @as(i32, @intCast(c.rows())) * @as(i32, @intCast(c.head_dim)) * 2;
}
fn barBase(c: Config) i32 {
    return pBase(c) + @as(i32, @intCast(c.rows() * c.kv_tile)) * 2;
}
fn holder(c: Config) i32 {
    return barBase(c) + @as(i32, @intCast(c.load_buffers * 16)) + 16;
}
fn redBase(c: Config) i32 {
    return holder(c) + 16;
}
fn sharedBytes(c: Config) i32 {
    return redBase(c) + 4 * 16 * 4;
}
fn scoreCol(c: Config) i32 {
    return if (c.all_decode) @intCast(@divExact(c.head_dim, 128) * c.rows()) else 256;
}
fn tmemCols(c: Config) u32 {
    return if (c.all_decode) @intCast(std.math.ceilPowerOfTwoAssert(u32, @intCast((@divExact(c.head_dim, 128) + @divTrunc(c.kv_tile + 127, 128)) * c.rows()))) else 512;
}

const Ctx = struct {
    b: *B,
    c: Config,
    q: V,
    k: V,
    v: V,
    table: V,
    sink: V,
    out: V,
    s: V,
    tmem: V,
    tid: V,
    warp: V,
    lane: V,
    seq: V,
    head: V,
    first_row: V,
    start: V,
    qlen: V,
    klen: V,
    first_tile: V,
    tiles: V,
    split: V,
    output_half: V,
};

fn loadI32(b: *B, ptr: V, index: V) V {
    return b.loadWords(1, ptr.add(index.to(.i64).mul(4)), .global, .{})[0];
}

// All-decode metadata gives the live request count without host readback. The
// fixed grid/scratch bound is independent of it and safe for graph replay.
fn liveSplits(b: *B, c: Config, active: V, tiles: V) V {
    if (c.splits == 1) return b.cst(.i32, 1);
    if (c.tiles_per_split > 0)
        return tiles.add(c.tiles_per_split - 1).div(c.tiles_per_split).maximum(1).minimum(c.splits);
    const work = active.maximum(1).mul(c.kv_heads * @divTrunc(c.group + c.rows() - 1, c.rows()));
    const waves: i64 = if (c.kv_tile > 64 and c.window > 0) 4 else 2;
    const budget = work.add(c.sms * waves - 1).div(work).maximum(1).minimum(c.splits);
    return budget.minimum(tiles.add(1).div(2).maximum(1));
}

fn run(b: *B, c: Config) cute.FinishError!void {
    std.debug.assert(c.kv_tile == 64 or c.kv_tile == 128 or c.kv_tile == 256);
    std.debug.assert(c.all_decode or c.kv_tile == 64);
    std.debug.assert(c.load_buffers == 2 or c.load_buffers == 3);
    // Persistent independent partitions restart stage numbering each time.
    std.debug.assert(c.split_stride == 0 or c.load_buffers == 2);
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
    b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin}"));
    const starts = b.ptrToInt(a.starts, .i64);
    const seq = b.blockIdx().x.div(c.queryTiles() * (if (c.headsInX()) c.kv_heads else 1));
    const first_row = b.blockIdx().x.rem(c.queryTiles()).mul(c.rows());
    const start = loadI32(b, starts, seq);
    const qlen = loadI32(b, starts, seq.add(1)).sub(start);
    const klen = loadI32(b, b.ptrToInt(a.lengths, .i64), seq);
    const first_pos = klen.sub(qlen).add(first_row.div(c.group));
    const last_pos = klen.sub(qlen).add(first_row.add(c.rows() - 1).div(c.group).minimum(qlen.sub(1)));
    const begin = if (c.window > 0) first_pos.sub(c.window - 1).maximum(0) else b.cst(.i32, 0);
    const end = if (c.causal) last_pos.add(1) else klen;
    const first_tile = begin.div(c.kv_tile);
    const total_tiles = end.add(c.kv_tile - 1).div(c.kv_tile).sub(first_tile).maximum(0);
    const splits = liveSplits(b, c, loadI32(b, starts, b.cst(.i32, c.batch)), total_tiles);
    const split = b.blockIdx().z;
    var active = b.openIf(first_row.lt(qlen.mul(c.group)).bitAnd(split.lt(splits)));
    {
        const lo = if (c.tiles_per_split > 0) split.mul(c.tiles_per_split) else total_tiles.mul(split).div(splits);
        const hi = if (c.tiles_per_split > 0) lo.add(c.tiles_per_split).minimum(total_tiles) else total_tiles.mul(split.add(1)).div(splits);
        const tid = b.threadIdx().x;
        const s = b.ptrToInt(b.allocSmem(.i8, @intCast(sharedBytes(c)), 1024), .i32);
        var ctx: Ctx = .{
            .b = b,
            .c = c,
            .q = b.ptrToInt(a.q, .i64),
            .k = b.ptrToInt(a.k, .i64),
            .v = b.ptrToInt(a.v, .i64),
            .table = b.ptrToInt(a.table, .i64),
            .sink = b.ptrToInt(a.sink, .i64),
            .out = b.ptrToInt(a.out, .i64),
            .s = s,
            .tmem = undefined,
            .tid = tid,
            .warp = tid.div(32),
            .lane = tid.rem(32),
            .seq = seq,
            .head = if (c.headsInX()) b.blockIdx().x.div(c.queryTiles()).rem(c.kv_heads) else b.blockIdx().y.div(c.outputTiles()),
            .output_half = b.blockIdx().y.rem(c.outputTiles()),
            .first_row = first_row,
            .start = start,
            .qlen = qlen,
            .klen = klen,
            .first_tile = first_tile.add(lo),
            .tiles = if (c.tiles_per_split == 1) b.cst(.i32, 1) else hi.sub(lo),
            .split = split,
        };
        var init = b.openIf(tid.eq(0));
        {
            for (0..@as(usize, @intCast(c.load_buffers))) |i| {
                b.mbarrierInit(s.add(barBase(c) + @as(i32, @intCast(i * 8))), 128);
                b.mbarrierInit(s.add(barBase(c) + @as(i32, @intCast(c.load_buffers * 8)) + @as(i32, @intCast(i * 8))), 1);
            }
            b.mbarrierInit(s.add(barBase(c) + @as(i32, @intCast(c.load_buffers * 16))), 1);
            b.fenceMbarrierInit();
            init.yieldThen(.{});
        }
        var alloc = b.openIf(tid.lt(32));
        {
            b.tmemAllocAt(s.add(holder(c)), tmemCols(c));
            b.relinquishTmemAllocPermit(.{});
            alloc.yieldThen(.{});
        }
        // Queries are packed_row by token, KV head, and group head. Mask the final
        // query tile without ever forming an out-of-range source load.
        var copy = b.openFor(tid, c.rows() * @divExact(c.head_dim, 8), 256, .{});
        {
            const row = copy.iv.div(@divExact(c.head_dim, 8));
            const col = copy.iv.rem(@divExact(c.head_dim, 8)).mul(8);
            const packed_row = first_row.add(row);
            const valid = packed_row.lt(qlen.mul(c.group));
            const token = start.add(b.select(valid, packed_row.div(c.group), b.cst(.i32, 0)));
            const qi = token.to(.i64).mul(c.kv_heads).add(ctx.head.to(.i64)).mul(c.group).add(packed_row.rem(c.group).to(.i64)).mul(c.head_dim).add(col.to(.i64));
            b.cpAsync16(s.add(qBase(c)).add(swizzle(row, col, c.rows())), ctx.q.add(qi.mul(2)), valid);
            copy.yield(.{});
        }
        b.cpAsyncWaitAll();
        b.syncThreads();
        ctx.tmem = b.loadWords(1, s.add(holder(c)), .shared, .{})[0];
        b.tcgen05Fence(.after);
        if (c.split_stride > 0) {
            // A fixed grid reuses Q, shared memory and TMEM across independent
            // split partitions. Runtime lengths bound this device loop only.
            // D256 has four load stages: both two-stage rings and the QK/PV
            // completion barrier return to parity zero after every partition.
            var partitions = b.openFor(split, splits, c.split_stride, .{});
            {
                ctx.first_tile = first_tile.add(partitions.iv);
                ctx.split = partitions.iv;
                var producer = b.openIf(tid.ge(128));
                {
                    loadStages(ctx);
                    producer.yieldThen(.{});
                }
                var consumer = b.openIf(tid.lt(128));
                {
                    decode(ctx);
                    consumer.yieldThen(.{});
                }
                b.tcgen05Fence(.before);
                b.syncThreads();
                b.tcgen05Fence(.after);
                partitions.yield(.{});
            }
        } else {
            var producer = b.openIf(tid.ge(128));
            {
                loadStages(ctx);
                producer.yieldThen(.{});
            }
            var consumer = b.openIf(tid.lt(128));
            {
                if (c.all_decode) decode(ctx) else prefill(ctx);
                consumer.yieldThen(.{});
            }
            b.tcgen05Fence(.before);
            b.syncThreads();
        }
        var free = b.openIf(tid.lt(32));
        {
            b.tmemDeallocAt(ctx.tmem, tmemCols(c));
            free.yieldThen(.{});
        }
        active.yieldThen(.{});
    }
}

fn ready(c: Ctx, id: V) V {
    return c.s.add(barBase(c.c)).add(id.rem(c.c.load_buffers).mul(8));
}
fn released(c: Ctx, id: V) V {
    return ready(c, id).add(c.c.load_buffers * 8);
}
fn phase(c: Ctx, id: V) V {
    return id.div(c.c.load_buffers).rem(2);
}

fn loadStages(c: Ctx) void {
    if (c.c.all_decode) return loadDecodeStages(c);
    const b = c.b;
    // Prefill computes half the output features per CTA; decode computes all.
    const k_stages = @divExact(c.c.head_dim, 128);
    const stages = k_stages + if (c.c.all_decode) k_stages else 2;
    var work = b.openFor(0, c.tiles.mul(stages), 1, .{});
    {
        const id = work.iv;
        const part = id.rem(stages);
        var reuse = b.openIf(id.ge(c.c.load_buffers));
        {
            b.mbarrierWait(released(c, id), phase(c, id).bitXor(1));
            reuse.yieldThen(.{});
        }
        const is_k = part.lt(k_stages);
        const d0 = b.select(is_k, part.mul(128), part.sub(k_stages).mul(128).add(c.output_half.mul(256)));
        const source = b.select(is_k, c.k, c.v);
        const key0 = c.first_tile.add(id.div(stages)).mul(64);
        var vec = b.openFor(c.tid.sub(128), 64 * 16, 128, .{});
        {
            const row = vec.iv.div(16);
            const col = vec.iv.rem(16).mul(8);
            const key = key0.add(row);
            const valid = key.lt(c.klen);
            // Do not read unused page table entries (which may contain -1).
            const safe_key = b.select(valid, key, b.cst(.i32, 0));
            const page = loadI32(b, c.table, c.seq.mul(c.c.pages_per_seq).add(safe_key.div(c.c.page_size)));
            const offset = page.to(.i64).mul(c.c.page_size).add(safe_key.rem(c.c.page_size).to(.i64)).mul(c.c.kv_heads).add(c.head.to(.i64)).mul(c.c.head_dim).add(d0.add(col).to(.i64));
            b.cpAsync16(c.s.add(id.rem(c.c.load_buffers).mul(kvBytes(c.c))).add(swizzle(row, col, 64)), source.add(offset.mul(2)), valid);
            vec.yield(.{});
        }
        b.cpAsyncMbarrierArrive(ready(c, id), true);
        work.yield(.{});
    }
}

// Reuse the paged addresses across all K/V feature stages of a decode tile.
fn loadDecodeStages(c: Ctx) void {
    const b = c.b;
    const k_stages: usize = @intCast(@divExact(c.c.head_dim, 128));
    const stages = 2 * k_stages;
    const tid = c.tid.sub(128);
    const row0 = tid.div(16);
    const col = tid.rem(16).mul(8);
    var tiles = b.openFor(0, c.tiles, 1, .{});
    {
        const key0 = c.first_tile.add(tiles.iv).mul(c.c.kv_tile).add(row0);
        var offsets: [32]V = undefined;
        var masks: [32]V = undefined;
        for (0..@as(usize, @intCast(@divExact(c.c.kv_tile, 8)))) |i| {
            const key = key0.add(@as(i32, @intCast(i * 8)));
            masks[i] = key.lt(c.klen);
            const safe_key = b.select(masks[i], key, b.cst(.i32, 0));
            const page = loadI32(b, c.table, c.seq.mul(c.c.pages_per_seq).add(safe_key.div(c.c.page_size)));
            offsets[i] = page.to(.i64).mul(c.c.page_size).add(safe_key.rem(c.c.page_size).to(.i64)).mul(c.c.kv_heads).add(c.head.to(.i64)).mul(c.c.head_dim).add(col.to(.i64)).mul(2);
        }
        for (0..stages) |part| {
            const id = tiles.iv.mul(stages).add(@as(i32, @intCast(part)));
            var reuse = b.openIf(id.ge(c.c.load_buffers));
            {
                b.mbarrierWait(released(c, id), phase(c, id).bitXor(1));
                reuse.yieldThen(.{});
            }
            const source = if (part < k_stages) c.k else c.v;
            const d0: i32 = @intCast((part % k_stages) * 128 * 2);
            for (0..@as(usize, @intCast(@divExact(c.c.kv_tile, 8)))) |i| {
                const row = row0.add(@as(i32, @intCast(i * 8)));
                b.cpAsync16(c.s.add(id.rem(c.c.load_buffers).mul(kvBytes(c.c))).add(swizzle(row, col, c.c.kv_tile)), source.add(offsets[i]).add(d0), masks[i]);
            }
            b.cpAsyncMbarrierArrive(ready(c, id), true);
        }
        tiles.yield(.{});
    }
}

fn mma(c: Ctx, tile: V, comptime qk: bool) void {
    const b = c.b;
    const k_stages = @divExact(c.c.head_dim, 128);
    const stages = k_stages + if (c.c.all_decode) k_stages else 2;
    const count: usize = if (qk or c.c.all_decode) @intCast(k_stages) else 2;
    var leader = b.openIf(c.tid.eq(0));
    {
        for (0..count) |part| {
            const id = tile.mul(stages).add(@as(i32, @intCast(part + if (qk) @as(usize, 0) else @as(usize, @intCast(k_stages)))));
            b.mbarrierWait(ready(c, id), phase(c, id));
            b.tcgen05Fence(.after);
            const kv = c.s.add(id.rem(c.c.load_buffers).mul(kvBytes(c.c)));
            const kd = descriptor(b, kv, if (qk) 16 else @intCast(c.c.kv_tile * 128));
            const qd = descriptor(b, c.s.add(if (qk) qBase(c.c) else pBase(c.c)), 16);
            const score_slabs: usize = if (qk and c.c.all_decode) @intCast(@divTrunc(c.c.kv_tile + 127, 128)) else 1;
            const steps: usize = if (qk) 8 else @intCast(@divExact(c.c.kv_tile, 16));
            for (0..score_slabs) |slab| {
                for (0..steps) |step| {
                    const step_off: i64 = @as(i64, @intCast(step / 4)) * c.c.kv_tile * 128 + @as(i64, @intCast(step % 4)) * 32;
                    const kvd = kd.add(if (qk) @divExact(step_off + @as(i64, @intCast(slab)) * 128 * 128, 16) else @as(i64, @intCast(step * 2048 / 16)));
                    const other_off: i64 = if (qk) @as(i64, @intCast(part * 2 + step / 4)) * c.c.rows() * 128 + @as(i64, @intCast(step % 4)) * 32 else @as(i64, @intCast(step / 4)) * c.c.rows() * 128 + @as(i64, @intCast(step % 4)) * 32;
                    const other = qd.add(@divExact(other_off, 16));
                    const swapped = c.c.all_decode;
                    const ad = if (swapped) kvd else other;
                    const bd = if (swapped) other else kvd;
                    const m: u32 = if (swapped) (if (qk) @intCast(@min(128, c.c.kv_tile)) else 128) else 64;
                    const n: u32 = if (swapped) @intCast(c.c.rows()) else if (qk) 64 else 128;
                    const instruction = instrDesc(c.c.dtype, m, n, swapped and !qk, !swapped and !qk);
                    const dst = c.tmem.add(if (qk) scoreCol(c.c) + @as(i32, @intCast(slab)) * @as(i32, @intCast(c.c.rows())) else @as(i32, @intCast(part * (if (swapped) @as(usize, @intCast(c.c.rows())) else 128))));
                    const acc = if (qk) b.cst(.i32, @intFromBool(part > 0 or step > 0)).ne(0) else tile.gt(0).bitOr(b.cst(.i32, @intFromBool(step > 0)).ne(0));
                    b.tcgen05MmaF16(dst, ad, bd, b.cst(.i32, @as(i32, @bitCast(instruction))), acc);
                }
            }
            b.tcgen05Commit(released(c, id));
        }
        b.tcgen05Commit(c.s.add(barBase(c.c) + @as(i32, @intCast(c.c.load_buffers * 16))));
        leader.yieldThen(.{});
    }
    // Two completions per tile: QK parity 0 and PV parity 1.
    b.mbarrierWait(c.s.add(barBase(c.c) + @as(i32, @intCast(c.c.load_buffers * 16))), b.cst(.i32, if (qk) 0 else 1));
    b.tcgen05Fence(.after);
    // All consumers must observe PV completion before the issuing warp can
    // complete the next QK and wrap this shared barrier's parity again.
    // Single-tile splits rendezvous at the outer full-CTA barrier instead.
    if (!qk and c.c.tiles_per_split != 1) b.barrierSync(1, 128);
}

fn validScore(c: Ctx, row: V, key: V) V {
    const pos = c.klen.sub(c.qlen).add(row.div(c.c.group));
    var valid = row.lt(c.qlen.mul(c.c.group)).bitAnd(key.lt(c.klen));
    if (c.c.causal) valid = valid.bitAnd(key.le(pos));
    if (c.c.window > 0) valid = valid.bitAnd(key.gt(pos.sub(c.c.window)));
    return valid;
}

fn storeP(c: Ctx, row: V, col: V, value: V) void {
    c.b.storeHalf(c.s.add(pBase(c.c)).add(swizzle(row, col, c.c.rows())), .shared, value.to(c.c.dtype).bitCast(.i16).to(.i32));
}

fn consumerSync(c: Ctx) void {
    c.b.fenceProxyAsyncShared();
    c.b.tcgen05Fence(.before);
    c.b.barrierSync(1, 128);
    c.b.tcgen05Fence(.after);
}

fn prefill(c: Ctx) void {
    const b = c.b;
    const row = c.first_row.add(c.warp.mul(16)).add(c.lane);
    const owns = c.lane.lt(16);
    const lane_tmem = c.tmem.add(c.warp.mul(32).shl(16));
    const zero = b.cst(.f32, 0);
    const neg_inf = b.cst(.f32, -std.math.inf(f32));
    var loop = b.openFor(0, c.tiles, 1, .{ neg_inf, zero });
    {
        mma(c, loop.iv, true);
        const raw = b.tmemLoad32x32b(64, lane_tmem.add(scoreCol(c.c)));
        b.fenceTmemLoad();
        var scores: [64]V = undefined;
        var maximum = loop.carried[0];
        for (0..64) |j| {
            const key = c.first_tile.add(loop.iv).mul(64).add(@as(i32, @intCast(j)));
            scores[j] = b.select(validScore(c, row, key).bitAnd(owns), raw[j].bitCast(.f32).mul(c.c.scale * std.math.log2e), neg_inf);
            maximum = maximum.maximum(scores[j]);
        }
        const safe_max = b.select(maximum.eq(neg_inf), zero, maximum);
        const alpha = b.exp2Approx(loop.carried[0].sub(safe_max));
        var sum = loop.carried[1].mul(alpha);
        var probabilities: [64]V = undefined;
        for (0..64) |j| {
            probabilities[j] = b.exp2Approx(scores[j].sub(safe_max));
            sum = sum.add(probabilities[j]);
        }
        var write = b.openIf(owns);
        {
            for (0..64) |j| storeP(c, c.warp.mul(16).add(c.lane), b.cst(.i32, @as(i32, @intCast(j))), probabilities[j]);
            write.yieldThen(.{});
        }
        var rescale = b.openIf(loop.iv.gt(0));
        {
            for (0..8) |chunk| {
                const addr = lane_tmem.add(@as(i32, @intCast(chunk * 32)));
                var out = b.tmemLoad32x32b(32, addr);
                b.fenceTmemLoad();
                for (&out) |*v| v.* = v.bitCast(.f32).mul(alpha).bitCast(.i32);
                b.tmemStore32x32b(32, addr, out);
            }
            b.fenceTmemStore();
            rescale.yieldThen(.{});
        }
        consumerSync(c);
        mma(c, loop.iv, false);
        loop.yield(.{ maximum, sum });
    }
    const norm = normalization(c, row, loop.results[0], loop.results[1]);
    for (0..8) |chunk| {
        const out = b.tmemLoad32x32b(32, lane_tmem.add(@as(i32, @intCast(chunk * 32))));
        b.fenceTmemLoad();
        var valid = b.openIf(owns.bitAnd(row.lt(c.qlen.mul(c.c.group))));
        {
            for (0..32) |j| {
                const dim = c.output_half.mul(256).add(@as(i32, @intCast(chunk * 32 + j)));
                storeOutput(c, row, dim, out[j].bitCast(.f32).mul(norm));
            }
            valid.yieldThen(.{});
        }
    }
}

fn tuple(comptime n: usize, a: [n]V) @Tuple(&@as([n]type, @splat(V))) {
    var t: @Tuple(&@as([n]type, @splat(V))) = undefined;
    inline for (0..n) |i| t[i] = a[i];
    return t;
}

fn reduceHeads(c: Ctx, comptime heads: usize, values: [heads]V, comptime max: bool) [heads]V {
    const b = c.b;
    for (values, 0..) |v, h| {
        var r = v;
        inline for (.{ 16, 8, 4, 2, 1 }) |step| r = if (max) r.maximum(b.shuffleXor(r, step)) else r.add(b.shuffleXor(r, step));
        var leader = b.openIf(c.lane.eq(0));
        {
            b.storeWords(1, c.s.add(redBase(c.c)).add(c.warp.mul(64)).add(@as(i32, @intCast(h * 4))), .shared, .{r.bitCast(.i32)});
            leader.yieldThen(.{});
        }
    }
    b.barrierSync(1, 128);
    var result: [heads]V = undefined;
    for (0..heads) |h| {
        var value = b.loadWords(1, c.s.add(redBase(c.c) + @as(i32, @intCast(h * 4))), .shared, .{})[0].bitCast(.f32);
        for (1..4) |w| {
            const next = b.loadWords(1, c.s.add(redBase(c.c) + @as(i32, @intCast(w * 64 + h * 4))), .shared, .{})[0].bitCast(.f32);
            value = if (max) value.maximum(next) else value.add(next);
        }
        result[h] = value;
    }
    b.barrierSync(1, 128);
    return result;
}

fn decode(c: Ctx) void {
    if (c.c.kv_tile == 256) decodeTile(c, 2) else decodeTile(c, 1);
}

fn decodeTile(c: Ctx, comptime slabs: usize) void {
    // MMA still uses its legal 8/16-row tile. Only process the statically
    // occupied head columns in the scalar softmax and output paths.
    if (c.c.group <= 2) decodeSized(c, 2, slabs) else if (c.c.group <= 4) decodeSized(c, 4, slabs) else if (c.c.rows() == 8) decodeSized(c, 8, slabs) else decodeSized(c, 16, slabs);
}

fn decodeSized(c: Ctx, comptime heads: usize, comptime slabs: usize) void {
    const b = c.b;
    const lane_tmem = c.tmem.add(c.warp.mul(32).shl(16));
    const key_row = c.warp.mul(@as(i32, if (c.c.kv_tile == 64) 16 else 32)).add(c.lane);
    const owns = if (c.c.kv_tile == 64) c.lane.lt(16) else b.cst(.i32, 1).ne(0);
    const zero = b.cst(.f32, 0);
    const neg_inf = b.cst(.f32, -std.math.inf(f32));
    var initial: [2 * heads]V = undefined;
    for (0..heads) |h| {
        initial[h] = neg_inf;
        initial[h + heads] = zero;
    }
    var loop = b.openFor(0, c.tiles, 1, tuple(2 * heads, initial));
    {
        mma(c, loop.iv, true);
        var scores: [slabs][heads]V = undefined;
        var local_max: [heads]V = @splat(neg_inf);
        for (0..slabs) |slab| {
            const raw = b.tmemLoad32x32b(heads, lane_tmem.add(scoreCol(c.c) + @as(i32, @intCast(slab)) * @as(i32, @intCast(c.c.rows()))));
            b.fenceTmemLoad();
            const key = c.first_tile.add(loop.iv).mul(c.c.kv_tile).add(key_row).add(@as(i32, @intCast(slab * 128)));
            for (0..heads) |h| {
                scores[slab][h] = b.select(owns.bitAnd(validScore(c, c.first_row.add(@as(i32, @intCast(h))), key)), raw[h].bitCast(.f32).mul(c.c.scale * std.math.log2e), neg_inf);
                local_max[h] = local_max[h].maximum(scores[slab][h]);
            }
        }
        const tile_max = reduceHeads(c, heads, local_max, true);
        var next: [2 * heads]V = undefined;
        var alpha: [heads]V = undefined;
        var probabilities: [slabs][heads]V = undefined;
        var local_sum: [heads]V = @splat(zero);
        for (0..heads) |h| {
            next[h] = loop.carried[h].maximum(tile_max[h]);
            const safe_max = b.select(next[h].eq(neg_inf), zero, next[h]);
            alpha[h] = b.exp2Approx(loop.carried[h].sub(safe_max));
            for (0..slabs) |slab| {
                probabilities[slab][h] = b.exp2Approx(scores[slab][h].sub(safe_max));
                local_sum[h] = local_sum[h].add(probabilities[slab][h]);
            }
        }
        const sums = reduceHeads(c, heads, local_sum, false);
        for (0..heads) |h| next[h + heads] = loop.carried[h + heads].mul(alpha[h]).add(sums[h]);
        var write = b.openIf(owns);
        {
            for (0..slabs) |slab| {
                const key_col = key_row.add(@as(i32, @intCast(slab * 128)));
                for (0..heads) |h| storeP(c, b.cst(.i32, @as(i32, @intCast(h))), key_col, probabilities[slab][h]);
                for (heads..@as(usize, @intCast(c.c.rows()))) |h| storeP(c, b.cst(.i32, @as(i32, @intCast(h))), key_col, zero);
            }
            write.yieldThen(.{});
        }
        if (c.c.tiles_per_split != 1) {
            var rescale = b.openIf(loop.iv.gt(0));
            {
                for (0..@as(usize, @intCast(@divExact(c.c.head_dim, 128)))) |chunk| {
                    const addr = lane_tmem.add(@as(i32, @intCast(chunk * @as(usize, @intCast(c.c.rows())))));
                    var out = b.tmemLoad32x32b(heads, addr);
                    b.fenceTmemLoad();
                    for (0..heads) |h| out[h] = out[h].bitCast(.f32).mul(alpha[h]).bitCast(.i32);
                    b.tmemStore32x32b(heads, addr, out);
                }
                b.fenceTmemStore();
                rescale.yieldThen(.{});
            }
        }
        consumerSync(c);
        mma(c, loop.iv, false);
        loop.yield(tuple(2 * heads, next));
    }
    var norms: [heads]V = undefined;
    for (0..heads) |h| norms[h] = if (c.c.splits > 1) b.cst(.f32, 1) else normalization(c, c.first_row.add(@as(i32, @intCast(h))), loop.results[h], loop.results[h + heads]);
    for (0..@as(usize, @intCast(@divExact(c.c.head_dim, 128)))) |chunk| {
        const out = b.tmemLoad32x32b(heads, lane_tmem.add(@as(i32, @intCast(chunk * @as(usize, @intCast(c.c.rows()))))));
        b.fenceTmemLoad();
        const dim = c.warp.mul(32).add(c.lane).add(@as(i32, @intCast(chunk * 128)));
        for (0..heads) |h| {
            const row = c.first_row.add(@as(i32, @intCast(h)));
            var valid = b.openIf(row.lt(c.qlen.mul(c.c.group)));
            {
                storeOutput(c, row, dim, out[h].bitCast(.f32).mul(norms[h]));
                if (c.c.splits > 1 and chunk == 0) {
                    var stats = b.openIf(c.tid.eq(0));
                    {
                        storeOutput(c, row, b.cst(.i32, c.c.head_dim), loop.results[h]);
                        storeOutput(c, row, b.cst(.i32, c.c.head_dim + 1), loop.results[h + heads]);
                        stats.yieldThen(.{});
                    }
                }
                valid.yieldThen(.{});
            }
        }
    }
}

fn normalization(c: Ctx, row: V, m: V, l: V) V {
    const b = c.b;
    if (!c.c.sink) return b.select(l.gt(0), b.rcpApprox(l), b.cst(.f32, 0));
    const h = c.head.mul(c.c.group).add(row.rem(c.c.group));
    const sink = loadI32(b, c.sink, h).bitCast(.f32).mul(std.math.log2e);
    const maximum = m.maximum(sink);
    const weight = b.exp2Approx(m.sub(maximum));
    return weight.div(l.mul(weight).add(b.exp2Approx(sink.sub(maximum))));
}

fn storeOutput(c: Ctx, row: V, dim: V, value: V) void {
    const token = c.start.add(row.div(c.c.group));
    var offset = token.to(.i64).mul(c.c.kv_heads).add(c.head.to(.i64)).mul(c.c.group).add(row.rem(c.c.group).to(.i64));
    if (c.c.splits > 1) {
        offset = c.split.to(.i64).mul(c.c.tokens * c.c.kv_heads * c.c.group).add(offset).mul(c.c.head_dim + 2).add(dim.to(.i64));
        c.b.storeWords(1, c.out.add(offset.mul(4)), .global, .{value.bitCast(.i32)});
    } else {
        offset = offset.mul(c.c.head_dim).add(dim.to(.i64));
        c.b.storeHalf(c.out.add(offset.mul(2)), .global, value.to(c.c.dtype).bitCast(.i16).to(.i32));
    }
}

fn descriptor(b: *B, address: V, lbo: u64) V {
    const flags: u64 = (64 << 32) | (1 << 46) | (@as(u64, 2) << 61) | ((lbo / 16) << 16);
    return address.to(.i64).shrLogical(4).bitAnd(0x3fff).bitOr(b.cst(.i64, @as(i64, @bitCast(flags))));
}
fn swizzle(row: V, col: V, rows: i64) V {
    return col.div(64).mul(rows * 128).add(row.mul(128)).add(col.rem(64).div(8).bitXor(row.rem(8)).mul(16)).add(col.rem(8).mul(2));
}
fn instrDesc(dtype: cute.DType, m: u32, n: u32, a_mn: bool, b_mn: bool) u32 {
    const format: u32 = if (dtype == .bf16) 1 else 0;
    return (1 << 4) | (format << 7) | (format << 10) | (@as(u32, @intFromBool(a_mn)) << 15) | (@as(u32, @intFromBool(b_mn)) << 16) | ((n >> 3) << 17) | ((m >> 4) << 24);
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
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const seq = b.blockIdx().y;
    const token = a.starts.get(.{seq});
    var valid = b.openIf(token.lt(a.starts.get(.{seq.add(1)})));
    {
        const head = b.blockIdx().x;
        const len = a.lengths.get(.{seq});
        const begin = if (c.window > 0) len.sub(c.window).maximum(0) else b.cst(.i32, 0);
        const tiles = len.add(c.kv_tile - 1).div(c.kv_tile).sub(begin.div(c.kv_tile));
        const splits = liveSplits(b, c, a.starts.get(.{c.batch}), tiles);
        const neg_inf = b.cst(.f32, -std.math.inf(f32));
        var maximum = b.openFor(0, splits, 1, .{if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else neg_inf});
        {
            maximum.yield(.{maximum.carried[0].maximum(a.partials.get(.{ maximum.iv, token, head, c.head_dim }))});
        }
        const m = maximum.results[0];
        var dims = b.openFor(b.threadIdx().x, c.head_dim, b.blockDim().x, .{});
        {
            const zero = b.cst(.f32, 0);
            var sum = b.openFor(0, splits, 1, .{ zero, if (c.sink) b.exp2Approx(a.sink.get(.{head}).mul(std.math.log2e).sub(m)) else zero });
            {
                const weight = b.exp2Approx(a.partials.get(.{ sum.iv, token, head, c.head_dim }).sub(m));
                sum.yield(.{ sum.carried[0].add(weight.mul(a.partials.get(.{ sum.iv, token, head, dims.iv }))), sum.carried[1].add(weight.mul(a.partials.get(.{ sum.iv, token, head, c.head_dim + 1 }))) });
            }
            a.out.set(.{ token, head, dims.iv }, sum.results[0].div(sum.results[1]).to(c.dtype));
            dims.yield(.{});
        }
        valid.yieldThen(.{});
    }
}

test "CuTe paged attention emits prefill and decode tensor cores" {
    for ([_]i64{ 256, 512 }) |head_dim| for ([_]bool{ false, true }) |all_decode| {
        const cfg: Config = .{ .head_dim = head_dim, .tokens = 80, .batch = 4, .kv_heads = 2, .group = 6, .pages_per_seq = 128, .page_size = 16, .max_query = if (all_decode) 1 else 65, .dtype = .bf16, .all_decode = all_decode, .causal = true, .window = 0, .scale = 0.044194174, .sink = true, .sms = 148, .splits = if (all_decode) 64 else 1 };
        const ir = try Kernel.emit(std.testing.allocator, cfg, block);
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "tcgen05.mma") != null);
        if (all_decode) {
            const merge_ir = try Merge.emit(std.testing.allocator, cfg, .{ 128, 1, 1 });
            defer std.testing.allocator.free(merge_ir);
        }
    };
}
