const std = @import("std");
const zml = @import("../zml.zig");
const cut = zml.kernel.cuda_tile;
const triton = @import("triton_attention.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

pub const Options = triton.paged.Options;
pub const Parameters = triton.paged.Parameters;

pub const Config = struct {
    batch: i64,
    // Query storage capacity can differ from the number of metadata slots.
    query_tokens: i64,
    heads: i64,
    pages: i64,
    max_pages: i64,
    page_size: i64 = 64,
    splits: i64,
    scale: f32,
    // Cache layout in logical [page, KV head, token, dimension] order.
    strides: [4]i64,
};

// Cache allocation granularity is independent of the 64-token MMA tile.
// Large divisible layouts retain tile loads; small or irregular pages use masked
// row-addressed loads without switching attention implementations.
const KVTile = struct { k: cut.Value, v: cut.Value };
const KVLoader = struct {
    b: *cut.Builder,
    c: Config,
    k: cut.Value,
    v: cut.Value,
    v_ptr: cut.Value,
    table: cut.Value,
    table_ptr: cut.Value,
    tiled: bool,
    token_major: bool,

    fn init(b: *cut.Builder, c: Config, k: cut.Value, v: cut.Value, table: cut.Value, prefill: bool) KVLoader {
        const width: i64 = @intCast(std.math.gcd(@as(u64, @intCast(c.page_size)), 64));
        // Four tiny transfers per operand are expensive on SM103. Pointer
        // loads win for 16-token pieces and for 32-token pieces at larger batches.
        const small_pages = width == 16 or (width == 32 and c.batch >= 32);
        const tiled = width >= 16 and !small_pages and (!prefill or @mod(c.page_size, 64) == 0);
        const token_major = c.strides[1] < c.strides[2];
        const shape = if (token_major) [_]i64{ c.pages, c.page_size, c.heads, 128 } else [_]i64{ c.pages, c.heads, c.page_size, 128 };
        const strides = if (token_major) [_]i64{ c.strides[0], c.strides[2], c.strides[1], c.strides[3] } else c.strides;
        const tile = if (token_major) [_]i64{ 1, width, 1, 128 } else [_]i64{ 1, 1, width, 128 };
        return .{
            .b = b,
            .c = c,
            .tiled = tiled,
            .token_major = token_major,
            .table_ptr = table,
            .v_ptr = v,
            .k = if (tiled) b.partitionView(b.tensorView(k, &shape, &strides), &tile, .{ .padding = null }) else k,
            .v = if (tiled) b.partitionView(b.tensorView(v, &shape, &strides), &tile, .{ .padding = null }) else v,
            .table = b.partitionView(b.tensorView(table, &.{ c.batch, c.max_pages }, &.{ c.max_pages, 1 }), &.{ 1, 1 }, .{ .padding = null }),
        };
    }

    fn load(self: KVLoader, seq: cut.Value, head: cut.Value, block: cut.Value, len: cut.Value, mask_tail: bool) KVTile {
        const b = self.b;
        const c = self.c;
        const zero = b.cst(.i32, 0);
        if (self.tiled) {
            const width: i64 = @intCast(std.math.gcd(@as(u64, @intCast(c.page_size)), 64));
            const count = @divExact(64, width);
            var pieces: [4]KVTile = undefined;
            var i: i64 = 0;
            while (i < count) : (i += 1) {
                const start = block.mul(64).add(b.cst(.i32, i * width));
                const page = (if (c.page_size < 64 and @mod(64, c.page_size) == 0) block.mul(b.cst(.i32, count)).add(b.cst(.i32, i)).minimum(len.sub(1).div(b.cst(.i32, c.page_size))) else if (c.page_size == 64) block else start.div(b.cst(.i32, c.page_size))).to(.i32);
                const safe_page = if (@mod(64, c.page_size) != 0 and @mod(c.page_size, 64) != 0) page.minimum(len.sub(1).div(b.cst(.i32, c.page_size))) else page;
                const physical = b.load(self.table, &.{ seq.to(.i32), safe_page }).reshape(&.{});
                const within = (if (c.page_size == width) zero else start.rem(b.cst(.i32, c.page_size)).div(b.cst(.i32, width))).to(.i32);
                const index = if (self.token_major) [_]cut.Value{ physical, within, head, zero } else [_]cut.Value{ physical, head, within, zero };
                const k = b.loadOpts(self.k, &index, .{ .hints = &.{.{ .allow_tma = true, .latency = 5 }} }).tile.reshape(&.{ width, 128 });
                const v = if (mask_tail) blk: {
                    // Express the valid extent in the view so padding is
                    // handled by the load, not by converting an MMA operand.
                    const valid_rows = len.sub(safe_page.mul(c.page_size)).minimum(c.page_size);
                    const base = self.v_ptr.offset(physical.to(.i64).mul(c.strides[0]).add(head.to(.i64).mul(c.strides[1])));
                    const view = b.partitionView(b.tensorViewDyn(base, &.{ .{ .dynamic = valid_rows }, .{ .static = 128 } }, &.{ .{ .static = c.strides[2] }, .{ .static = c.strides[3] } }), &.{ width, 128 }, .{});
                    break :blk b.loadOpts(view, &.{ within, zero }, .{ .hints = &.{.{ .allow_tma = true, .latency = 5 }} }).tile;
                } else b.loadOpts(self.v, &index, .{ .hints = &.{.{ .allow_tma = true, .latency = 5 }} }).tile.reshape(&.{ width, 128 });
                pieces[@intCast(i)] = .{ .k = k, .v = v };
            }
            var n: usize = @intCast(count);
            while (n > 1) : (n /= 2) {
                for (0..n / 2) |j| pieces[j] = .{ .k = b.cat(pieces[j * 2].k, pieces[j * 2 + 1].k, 0), .v = b.cat(pieces[j * 2].v, pieces[j * 2 + 1].v, 0) };
            }
            return pieces[0];
        }
        const token = b.iota(64, .i32).add(b.assumeDivBy(block.mul(64), 64));
        // Clamp only the unused tail's page-table lookup, not valid tokens.
        const page = token.minimum(len.sub(1)).div(c.page_size);
        const raw_physical = b.loadPtr(self.table_ptr.offset(seq.mul(c.max_pages).add(page)));
        const physical = if (@mod(64, c.page_size) == 0) b.assumeSameElements(raw_physical, &.{c.page_size}) else raw_physical;
        const offset = physical.to(.i64).mul(c.strides[0]).add(head.to(.i64).mul(c.strides[1])).add(token.rem(c.page_size).to(.i64).mul(c.strides[2]));
        const address = offset.reshape(&.{ 64, 1 }).add(b.iota(128, .i64).mul(c.strides[3]).reshape(&.{ 1, 128 }));
        const mask = token.lt(len).reshape(&.{ 64, 1 }).broadcastTo(&.{ 64, 128 });
        return .{
            .k = b.loadPtrOpts(self.k.offset(address), .{ .mask = mask, .padding = b.zeros(&.{ 64, 128 }, .bf16) }).tile,
            .v = b.loadPtrOpts(self.v.offset(address), .{ .mask = mask, .padding = b.zeros(&.{ 64, 128 }, .bf16) }).tile,
        };
    }
};

// Plan on device: metadata capacity is not the number of active requests.
// Four rows contain inclusive extra-decode-work offsets, prefill work offsets,
// decode splits and prefill splits. No host readback or graph update is needed.
pub const Plan = cut.Kernel(Config, .{
    .name = "unified_attention_plan",
    .inputs = &.{ "lengths", "starts" },
    .outputs = &.{"plan"},
    .run = emitPlan,
});

fn emitPlan(b: *cut.Builder, c: Config) cut.FinishError!void {
    const a = try b.declareArgs(.{ .lengths = .{ .ptr = .i32 }, .starts = .{ .ptr = .i32 }, .plan = .{ .ptr = .i32 } });
    const width: i64 = @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(c.batch)));
    const seq = b.iota(width, .i32);
    const valid = seq.lt(c.batch);
    const start = b.loadPtrOpts(a.starts.offset(seq), .{ .mask = valid, .padding = b.zeros(&.{width}, .i32) }).tile;
    const end = b.loadPtrOpts(a.starts.offset(seq.add(1)), .{ .mask = valid, .padding = b.zeros(&.{width}, .i32) }).tile;
    const len = b.loadPtrOpts(a.lengths.offset(seq), .{ .mask = valid, .padding = b.zeros(&.{width}, .i32) }).tile;
    const decode = end.sub(start).eq(1);
    const pages = b.where(decode, len.cdiv(64), b.zeros(&.{width}, .i32));
    const active = decode.to(.i32).sum(0);
    // Two waves of head CTAs on a 152-SM GB300, or one work item per
    // active sequence for bandwidth-bound batches. A split processes >=512
    // tokens unless it is the final partial segment.
    const target = active.maximum(@divTrunc(304 + c.heads - 1, c.heads));
    const chunk = pages.to(.i64).sum(0).cdiv(target.to(.i64)).maximum(8).to(.i32);
    const splits = pages.cdiv(chunk).minimum(32);
    const prefill = b.where(end.sub(start).gt(1), end.sub(start).cdiv(32), b.zeros(&.{width}, .i32));
    const prefill_splits = b.where(prefill.gt(0), b.cst(.i32, @divTrunc(304 + c.heads - 1, c.heads)).cdiv(prefill.sum(0).maximum(1)).minimum(4).maximum(1), b.zeros(&.{width}, .i32));
    _ = b.storePtrOpts(a.plan.offset(seq), b.cumsum(splits.sub(1).maximum(0), 0), .{ .mask = valid });
    _ = b.storePtrOpts(a.plan.offset(seq.add(c.batch)), b.cumsum(prefill.mul(prefill_splits), 0), .{ .mask = valid });
    _ = b.storePtrOpts(a.plan.offset(seq.add(2 * c.batch)), splits, .{ .mask = valid });
    _ = b.storePtrOpts(a.plan.offset(seq.add(3 * c.batch)), prefill_splits, .{ .mask = valid });
}

pub fn decodeWorkCapacity(batch: i64, heads: i64) i64 {
    return @min(batch * 32, batch + @max(batch, @divTrunc(304 + heads - 1, heads)));
}

pub fn prefillWorkCapacity(batch: i64, query_tokens: i64, heads: i64) i64 {
    const chunks = batch + @divTrunc(query_tokens, 32);
    return @min(chunks * 4, chunks + @divTrunc(304 + heads - 1, heads));
}

// First inclusive prefix strictly greater than work. Empty metadata slots
// have repeated prefixes and are skipped by construction.
fn workSequence(b: *cut.Builder, prefix: cut.Value, work: cut.Value, batch: i64) cut.Value {
    // A vector reduction in this conditional path stalled SM103 mixed
    // batches with short decode requests. Scalar search also handles repeated
    // prefixes from inactive slots without a collective in the branch.
    var search = b.openFor(0, std.math.log2_int_ceil(u64, @intCast(batch)) + 1, 1, .{ b.cst(.i32, 0), b.cst(.i32, batch - 1) });
    const mid = search.carried[0].add(search.carried[1]).div(2);
    const right = b.loadPtr(prefix.offset(mid)).le(work);
    search.yield(.{ b.where(right, mid.add(1), search.carried[0]), b.where(right, search.carried[1], mid) });
    return b.assumeBounded(search.results[0], 0, batch - 1);
}

fn previousWork(b: *cut.Builder, prefix: cut.Value, seq: cut.Value) cut.Value {
    return b.loadPtrOpts(prefix.offset(seq.sub(1).maximum(0)), .{ .mask = seq.gt(0), .padding = b.cst(.i32, 0) }).tile;
}

pub const ScheduledDecode = cut.Kernel(Config, .{
    .name = "unified_attention_decode",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "plan" },
    .outputs = &.{ "out", "maxima", "sums", "acc" },
    .run = emitScheduledDecode,
});

fn emitScheduledDecode(b: *cut.Builder, c: Config) cut.FinishError!void {
    return emitDecodeImpl(b, c, true);
}

pub fn splitCount(batch: i64, max_len: i64) i64 {
    if (batch > 16 or max_len < 2048) return 1;
    const cap: i64 = if (batch == 1 and max_len >= 32768) 32 else if (batch <= 2) 16 else if (batch <= 8) 8 else 4;
    var n: i64 = 1;
    while (n < cap and n * 512 < max_len) n *= 2;
    return n;
}

pub const Decode = cut.Kernel(Config, .{
    .name = "paged_decode_bf16_h128_gqa4",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts" },
    .outputs = &.{ "out", "maxima", "sums", "acc" },
    .run = emitDecode,
});

fn emitDecode(b: *cut.Builder, c: Config) cut.FinishError!void {
    return emitDecodeImpl(b, c, false);
}

fn emitDecodeImpl(b: *cut.Builder, c: Config, comptime scheduled: bool) cut.FinishError!void {
    const query_tokens = c.query_tokens;
    const rows: i64 = 16;
    const args = if (scheduled) .{
        .q = .{ .ptr = .bf16 },
        .k = .{ .ptr = .bf16 },
        .v = .{ .ptr = .bf16 },
        .table = .{ .ptr = .i32 },
        .lengths = .{ .ptr = .i32 },
        .starts = .{ .ptr = .i32 },
        .plan = .{ .ptr = .i32 },
        .out = .{ .ptr = .bf16 },
        .maxima = .{ .ptr = .f32 },
        .sums = .{ .ptr = .f32 },
        .acc = .{ .ptr = .f32 },
    } else .{
        .q = .{ .ptr = .bf16 },
        .k = .{ .ptr = .bf16 },
        .v = .{ .ptr = .bf16 },
        .table = .{ .ptr = .i32 },
        .lengths = .{ .ptr = .i32 },
        .starts = .{ .ptr = .i32 },
        .out = .{ .ptr = .bf16 },
        .maxima = .{ .ptr = .f32 },
        .sums = .{ .ptr = .f32 },
        .acc = .{ .ptr = .f32 },
    };
    const a = try b.declareArgsOpts(args, .{ .hints = &.{.{ .arch = .sm_103, .occupancy = 4, .num_worker_warps_per_cta = 4 }} });
    const id = b.tileBlockId().x;
    const ns = b.cst(.i32, c.splits);
    const nh = b.cst(.i32, c.heads);
    var work_active: @TypeOf(b.openIf(id.eq(0))) = if (scheduled) b.openIf(id.div(nh).lt(b.loadPtr(a.plan.offset(b.cst(.i32, c.batch - 1))).add(c.batch))) else undefined;
    const head = if (scheduled) id.rem(nh) else id.div(ns).rem(nh);
    const mapping = if (scheduled) blk: {
        const work = id.div(nh);
        // The common unsplit work maps directly to its metadata slot.
        // Only extra KV chunks pay for a prefix search.
        var branch = b.openIfElse(work.lt(c.batch), .{ b.tileTy(&.{}, .i32), b.tileTy(&.{}, .i32) });
        branch.yieldThen(.{ work, b.cst(.i32, 0) });
        const extra = work.sub(c.batch).to(.i32);
        const seq_extra = workSequence(b, a.plan, extra, c.batch);
        branch.yieldElse(.{ seq_extra, extra.sub(previousWork(b, a.plan, seq_extra)).add(1) });
        break :blk branch.results;
    } else [_]cut.Value{ id.div(ns.mul(nh)), id.rem(ns) };
    const seq = mapping[0];
    const split = mapping[1];
    const split_count = if (scheduled) b.loadPtr(a.plan.offset(seq.add(2 * c.batch))) else ns;
    var has_decode: @TypeOf(work_active) = if (scheduled) b.openIf(split_count.gt(0)) else undefined;
    const zero = b.cst(.i32, 0);
    const qidx = b.loadPtr(a.starts.offset(seq));
    const qend = b.loadPtr(a.starts.offset(seq.add(1)));
    const lengths = b.partitionView(b.tensorView(a.lengths, &.{c.batch}, &.{1}), &.{1}, .{ .padding = null });
    const len = b.where(qidx.lt(qend), b.load(lengths, &.{seq}).reshape(&.{}), zero);
    const qview = b.partitionView(b.tensorView(a.q, &.{ query_tokens, c.heads * 4, 128 }, &.{ c.heads * 512, 128, 1 }), &.{ 1, 4, 128 }, .{});
    const q4 = b.load(qview, &.{ qidx, head, zero }).reshape(&.{ 4, 128 });
    const q8 = b.cat(q4, b.zeros(&.{ 4, 128 }, .bf16), 0);
    const q = b.cat(q8, b.zeros(&.{ 8, 128 }, .bf16), 0);
    const loader = KVLoader.init(b, c, a.k, a.v, a.table, false);
    const blocks = @divTrunc(c.max_pages * c.page_size + 63, 64);
    const pages_per_split = @divTrunc(blocks + c.splits - 1, c.splits);
    const factor = @as(u64, 1) << @intCast(@ctz(@as(u64, @intCast(pages_per_split))));
    const chunk = if (scheduled) len.cdiv(64).cdiv(split_count) else b.cst(.i32, pages_per_split);
    const first = if (scheduled) split.mul(chunk) else b.assumeDivBy(split.mul(chunk), factor);
    const last = first.add(chunk).minimum(len.add(63).div(64));
    // Keep the full-page pipeline free of V masking/conversion. Select once
    // per CTA, not once per iteration; the other path handles arbitrary tails.
    var aligned = b.openIfElse(if (loader.tiled) len.rem(64).eq(0) else b.cst(.i1, 1), .{ b.tileTy(&.{ 16, 128 }, .f32), b.tileTy(&.{rows}, .f32), b.tileTy(&.{rows}, .f32) });
    inline for (.{ false, true }) |mask_tail| {
        var loop = b.openFor(first, last, 1, .{ b.zeros(&.{ 16, 128 }, .f32), b.full(&.{rows}, @as(f64, -1.0e20), .f32), b.zeros(&.{rows}, .f32) });
        const page = loop.iv;
        const loaded = loader.load(seq, head, page, len, mask_tail);
        const k = loaded.k;
        const v = loaded.v;
        const full_scores = b.mmaf(q, k.permute(&.{ 1, 0 }), b.zeros(&.{ 16, 64 }, .f32));
        const scores = full_scores.mul(c.scale);
        const start = b.assumeDivBy(page.mul(64), 64);
        var tail = b.openIfElse(start.add(64).gt(len), .{b.tileTy(&.{ rows, 64 }, .f32)});
        const valid = b.iota(64, .i32).add(start).lt(len).reshape(&.{ 1, 64 });
        tail.yieldThen(.{b.where(valid, scores, b.full(&.{ rows, 64 }, @as(f64, -1.0e20), .f32))});
        tail.yieldElse(.{scores});
        const s = tail.results[0];
        const m = loop.carried[1].maximum(s.max(1));
        const alpha = b.exp2(loop.carried[1].sub(m).mul(1.4426950408889634));
        const p = b.exp2(s.sub(m.reshape(&.{ rows, 1 })).mul(1.4426950408889634));
        // Match the Python frontend's contraction explicitly. Separate mul/add
        // increases live registers and changes tileiras' pipeline allocation.
        const sum = b.fma(loop.carried[2], alpha, p.sum(1));
        const scaled_acc = loop.carried[0].mul(alpha.reshape(&.{ 16, 1 }));
        const acc = b.mmaf(p.to(.bf16), v, scaled_acc);
        loop.yield(.{ acc, m, sum });
        if (!mask_tail) aligned.yieldThen(.{ loop.results[0], loop.results[1], loop.results[2] }) else aligned.yieldElse(.{ loop.results[0], loop.results[1], loop.results[2] });
    }
    const m4 = b.extract(aligned.results[1], &.{zero}, &.{4});
    const sum4 = b.extract(aligned.results[2], &.{zero}, &.{4});
    const acc4 = b.extract(aligned.results[0], &.{ zero, zero }, &.{ 4, 128 });
    var active = b.openIf(qidx.lt(qend));
    var direct: @TypeOf(work_active) = if (scheduled) b.openIf(split_count.eq(1)) else undefined;
    if (scheduled or c.splits == 1) {
        const outview = b.partitionView(b.tensorView(a.out, &.{ query_tokens, c.heads, 4, 128 }, &.{ c.heads * 512, 512, 128, 1 }), &.{ 1, 1, 4, 128 }, .{});
        const out = if (scheduled)
            acc4.mul(b.full(&.{4}, @as(f64, 1), .f32).div(sum4.maximum(1e-20)).reshape(&.{ 4, 1 })).to(.bf16)
        else
            acc4.div(sum4.maximum(1.0e-20).reshape(&.{ 4, 1 })).to(.bf16);
        _ = b.store(out.reshape(&.{ 1, 1, 4, 128 }), outview, &.{ qidx, head, zero, zero });
    }
    if (scheduled) direct.yieldThen(.{});
    var partial: @TypeOf(work_active) = if (scheduled) b.openIf(split_count.gt(1)) else undefined;
    if (scheduled or c.splits > 1) {
        const base = seq.mul(c.heads).add(head).mul(c.splits).add(split).to(.i32);
        const stat_shape = [_]i64{ c.batch * c.heads * c.splits, 4 };
        const mv = b.partitionView(b.tensorView(a.maxima, &stat_shape, &.{ 4, 1 }), &.{ 1, 4 }, .{});
        const sv = b.partitionView(b.tensorView(a.sums, &stat_shape, &.{ 4, 1 }), &.{ 1, 4 }, .{});
        const av = b.partitionView(b.tensorView(a.acc, &.{ c.batch * c.heads * c.splits, 4, 128 }, &.{ 512, 128, 1 }), &.{ 1, 4, 128 }, .{});
        _ = b.store(m4.reshape(&.{ 1, 4 }), mv, &.{ base, zero });
        _ = b.store(sum4.reshape(&.{ 1, 4 }), sv, &.{ base, zero });
        _ = b.store(acc4.reshape(&.{ 1, 4, 128 }), av, &.{ base, zero, zero });
    }
    if (scheduled) partial.yieldThen(.{});
    active.yieldThen(.{});
    if (scheduled) {
        has_decode.yieldThen(.{});
        work_active.yieldThen(.{});
    }
}

pub const Reduce = cut.Kernel(Config, .{
    .name = "paged_decode_reduce_bf16_h128_gqa4",
    .inputs = &.{ "maxima", "sums", "acc", "starts" },
    .outputs = &.{"out"},
    .run = emitReduce,
});

pub const ScheduledReduce = cut.Kernel(Config, .{
    .name = "unified_attention_decode_reduce",
    .inputs = &.{ "maxima", "sums", "acc", "starts", "plan", "initial" },
    .outputs = &.{"out"},
    .run = emitScheduledReduce,
});

fn emitScheduledReduce(b: *cut.Builder, c: Config) cut.FinishError!void {
    return emitReduceImpl(b, c, true);
}

fn emitReduce(b: *cut.Builder, c: Config) cut.FinishError!void {
    return emitReduceImpl(b, c, false);
}

fn emitReduceImpl(b: *cut.Builder, c: Config, comptime scheduled: bool) cut.FinishError!void {
    const query_tokens = c.query_tokens;
    const a = if (scheduled)
        try b.declareArgs(.{ .maxima = .{ .ptr = .f32 }, .sums = .{ .ptr = .f32 }, .acc = .{ .ptr = .f32 }, .starts = .{ .ptr = .i32 }, .plan = .{ .ptr = .i32 }, .initial = .{ .ptr = .bf16 }, .out = .{ .ptr = .bf16 } })
    else
        try b.declareArgs(.{ .maxima = .{ .ptr = .f32 }, .sums = .{ .ptr = .f32 }, .acc = .{ .ptr = .f32 }, .starts = .{ .ptr = .i32 }, .out = .{ .ptr = .bf16 } });
    const id = b.tileBlockId().x;
    const seq = id.div(c.heads).to(.i32);
    const head = id.rem(c.heads).to(.i32);
    const qi = b.loadPtr(a.starts.offset(seq));
    const split_count = if (scheduled) b.loadPtr(a.plan.offset(seq.add(2 * c.batch))) else b.cst(.i32, c.splits);
    var active = b.openIf(if (scheduled) split_count.gt(1) else qi.lt(b.loadPtr(a.starts.offset(seq.add(1)))));
    const zero = b.cst(.i32, 0);
    const shape = [_]i64{ c.batch * c.heads, c.splits, 4 };
    const strides = [_]i64{ c.splits * 4, 4, 1 };
    const mv = b.partitionView(b.tensorView(a.maxima, &shape, &strides), &.{ 1, c.splits, 4 }, .{});
    const sv = b.partitionView(b.tensorView(a.sums, &shape, &strides), &.{ 1, c.splits, 4 }, .{});
    const av = b.partitionView(b.tensorView(a.acc, &.{ c.batch * c.heads, c.splits, 4, 128 }, &.{ c.splits * 512, 512, 128, 1 }), &.{ 1, c.splits, 4, 128 }, .{});
    const valid = b.iota(c.splits, .i32).lt(split_count).reshape(&.{ c.splits, 1 });
    const m = b.where(valid, b.load(mv, &.{ id, zero, zero }).reshape(&.{ c.splits, 4 }), b.full(&.{ c.splits, 4 }, @as(f64, -1e20), .f32));
    const sums = b.where(valid, b.load(sv, &.{ id, zero, zero }).reshape(&.{ c.splits, 4 }), b.zeros(&.{ c.splits, 4 }, .f32));
    const weights = b.exp2(m.sub(m.max(0).reshape(&.{ 1, 4 })).mul(1.4426950408889634));
    const denom = sums.mul(weights).sum(0).maximum(1.0e-20);
    const acc = b.where(valid.reshape(&.{ c.splits, 1, 1 }), b.load(av, &.{ id, zero, zero, zero }).reshape(&.{ c.splits, 4, 128 }), b.zeros(&.{ c.splits, 4, 128 }, .f32));
    const merged = acc.mul(weights.reshape(&.{ c.splits, 4, 1 })).sum(0);
    const out = if (scheduled)
        merged.mul(b.full(&.{4}, @as(f64, 1), .f32).div(denom).reshape(&.{ 4, 1 })).to(.bf16)
    else
        merged.div(denom.reshape(&.{ 4, 1 })).to(.bf16);
    const ov = b.partitionView(b.tensorView(a.out, &.{ query_tokens, c.heads, 4, 128 }, &.{ c.heads * 512, 512, 128, 1 }), &.{ 1, 1, 4, 128 }, .{});
    _ = b.store(out.reshape(&.{ 1, 1, 4, 128 }), ov, &.{ qi, head, zero, zero });
    active.yieldThen(.{});
}

pub const Prefill = cut.Kernel(Config, .{
    .name = "unified_attention_prefill_bf16_h128_gqa4",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "plan", "initial" },
    .outputs = &.{ "out", "maxima", "sums", "acc" },
    .run = emitPrefill,
});

fn emitPrefill(b: *cut.Builder, c: Config) cut.FinishError!void {
    // Eight workers amortize short token-major gather loads, but regress long
    // loops, sparse queries and tiled loads. These are allocation bounds, not
    // assumptions about runtime sequence lengths.
    const short_gather = c.page_size == 16 and c.max_pages <= 32 and
        c.query_tokens >= c.batch * 64 and c.strides[1] < c.strides[2];
    const a = try b.declareArgsOpts(.{
        .q = .{ .ptr = .bf16 },
        .k = .{ .ptr = .bf16 },
        .v = .{ .ptr = .bf16 },
        .table = .{ .ptr = .i32 },
        .lengths = .{ .ptr = .i32 },
        .starts = .{ .ptr = .i32 },
        .plan = .{ .ptr = .i32 },
        .initial = .{ .ptr = .bf16 },
        .out = .{ .ptr = .bf16 },
        .maxima = .{ .ptr = .f32 },
        .sums = .{ .ptr = .f32 },
        .acc = .{ .ptr = .f32 },
    }, .{ .hints = &.{.{ .arch = .sm_103, .occupancy = 2, .num_worker_warps_per_cta = if (short_gather) 8 else 4 }} });
    const id = b.tileBlockId().x;
    const work = id.div(c.heads).to(.i32);
    const head = id.rem(c.heads).to(.i32);
    const prefix = a.plan.offset(b.cst(.i32, c.batch));
    var active = b.openIf(work.lt(b.loadPtr(prefix.offset(b.cst(.i32, c.batch - 1)))));
    const seq = workSequence(b, prefix, work, c.batch);
    const ns = b.loadPtr(a.plan.offset(seq.add(3 * c.batch)));
    const local = work.sub(previousWork(b, prefix, seq));
    const split = local.rem(ns);
    const first = b.assumeDivBy(local.div(ns).mul(32), 32);
    const begin = b.loadPtr(a.starts.offset(seq));
    const end = b.loadPtr(a.starts.offset(seq.add(1)));
    const len = b.loadPtr(a.lengths.offset(seq));
    const row = b.iota(128, .i32);
    const qi = row.div(4).add(first).add(begin);
    const offsets = qi.mul(c.heads * 512).add(head.mul(512)).add(row.rem(4).mul(128)).reshape(&.{ 128, 1 }).add(b.iota(128, .i32).reshape(&.{ 1, 128 }));
    const mask = qi.lt(end).reshape(&.{ 128, 1 }).broadcastTo(&.{ 128, 128 });
    const q = b.loadPtrOpts(a.q.offset(offsets), .{ .mask = mask, .padding = b.zeros(&.{ 128, 128 }, .bf16) }).tile;
    // Concatenating small tiles is particularly expensive for the larger
    // prefill MMA. Keep tiled loads only when one page contains a whole tile.
    const loader = KVLoader.init(b, c, a.k, a.v, a.table, true);
    const context = len.sub(end.sub(begin));
    const stop = len.minimum(context.add(first).add(32)).cdiv(64);
    const chunk = stop.cdiv(ns);
    const begin_page = split.mul(chunk);
    var loop = b.openFor(begin_page, begin_page.add(chunk).minimum(stop), 1, .{ b.zeros(&.{ 128, 128 }, .f32), b.full(&.{128}, @as(f64, -1e20), .f32), b.zeros(&.{128}, .f32) });
    const loaded = loader.load(seq, head, loop.iv, len, true);
    const k = loaded.k;
    const v = loaded.v;
    const scores = b.mmaf(q, k.permute(&.{ 1, 0 }), b.zeros(&.{ 128, 64 }, .f32)).mul(c.scale * 1.4426950408889634);
    const token = b.iota(64, .i32).add(b.assumeDivBy(loop.iv.mul(64), 64)).reshape(&.{ 1, 64 });
    const causal = token.le(context.add(first).add(row.div(4)).reshape(&.{ 128, 1 }));
    const valid = b.andi(causal, token.lt(len).broadcastTo(&.{ 128, 64 }));
    const s = b.where(valid, scores, b.full(&.{ 128, 64 }, @as(f64, -1e20), .f32));
    const m = loop.carried[1].maximum(s.max(1));
    const alpha = b.exp2Opts(loop.carried[1].sub(m), true);
    // A later split can be entirely masked for an early query row. Its
    // contribution must be exactly neutral, including the -1e20/-1e20 case.
    const p = b.where(valid, b.exp2Opts(s.sub(m.reshape(&.{ 128, 1 })), true), b.zeros(&.{ 128, 64 }, .f32));
    const sum = b.fma(loop.carried[2], alpha, p.sum(1));
    const acc = b.mmaf(p.to(.bf16), v, loop.carried[0].mul(alpha.reshape(&.{ 128, 1 })));
    loop.yield(.{ acc, m, sum });
    var direct = b.openIf(ns.eq(1));
    const inv = b.full(&.{128}, @as(f64, 1), .f32).div(loop.results[2].maximum(1e-20));
    _ = b.storePtrOpts(a.out.offset(offsets), loop.results[0].mul(inv.reshape(&.{ 128, 1 })).to(.bf16), .{ .mask = mask });
    direct.yieldThen(.{});
    var partial = b.openIf(ns.gt(1));
    const stat = qi.mul(c.heads).add(head).mul(4).add(split).mul(4).add(row.rem(4));
    _ = b.storePtrOpts(a.maxima.offset(stat), loop.results[1], .{ .mask = qi.lt(end) });
    _ = b.storePtrOpts(a.sums.offset(stat), loop.results[2], .{ .mask = qi.lt(end) });
    const acc_offset = stat.mul(128).reshape(&.{ 128, 1 }).add(b.iota(128, .i32).reshape(&.{ 1, 128 }));
    _ = b.storePtrOpts(a.acc.offset(acc_offset), loop.results[0], .{ .mask = mask });
    partial.yieldThen(.{});
    active.yieldThen(.{});
}

pub const PrefillReduce = cut.Kernel(Config, .{
    .name = "unified_attention_prefill_reduce",
    .inputs = &.{ "maxima", "sums", "acc", "starts", "plan", "initial" },
    .outputs = &.{"out"},
    .run = emitPrefillReduce,
});

fn emitPrefillReduce(b: *cut.Builder, c: Config) cut.FinishError!void {
    const a = try b.declareArgsOpts(.{ .maxima = .{ .ptr = .f32 }, .sums = .{ .ptr = .f32 }, .acc = .{ .ptr = .f32 }, .starts = .{ .ptr = .i32 }, .plan = .{ .ptr = .i32 }, .initial = .{ .ptr = .bf16 }, .out = .{ .ptr = .bf16 } }, .{ .hints = &.{.{ .arch = .sm_103, .occupancy = 4 }} });
    const id = b.tileBlockId().x;
    const qi = id.div(c.heads).to(.i32);
    var active = b.openIf(qi.lt(b.loadPtr(a.starts.offset(b.cst(.i32, c.batch)))));
    // Packed decode followed by a prefill tail and uniform prefill both have
    // cheap likely owners. Validate the guess before using it; arbitrary
    // ordering, ragged lengths and inactive slots retain the general lookup.
    const guess = (if (c.query_tokens <= c.batch * 2) qi.minimum(c.batch - 1) else qi.mul(c.batch).div(c.query_tokens).minimum(c.batch - 1)).to(.i32);
    const guessed_begin = b.loadPtr(a.starts.offset(guess));
    const guessed_end = b.loadPtr(a.starts.offset(guess.add(1)));
    var owner = b.openIfElse(b.andi(qi.ge(guessed_begin), qi.lt(guessed_end)), .{b.tileTy(&.{}, .i32)});
    owner.yieldThen(.{guess});
    owner.yieldElse(.{workSequence(b, a.starts.offset(b.cst(.i32, 1)), qi, c.batch)});
    const seq = owner.results[0];
    const ns = b.loadPtr(a.plan.offset(seq.add(c.batch * 3)));
    var split = b.openIf(ns.gt(1));
    const zero = b.cst(.i32, 0);
    const stat_shape = [_]i64{ c.query_tokens * c.heads, 4, 4 };
    const mv = b.partitionView(b.tensorView(a.maxima, &stat_shape, &.{ 16, 4, 1 }), &.{ 1, 1, 4 }, .{});
    const sv = b.partitionView(b.tensorView(a.sums, &stat_shape, &.{ 16, 4, 1 }), &.{ 1, 1, 4 }, .{});
    const av = b.partitionView(b.tensorView(a.acc, &.{ c.query_tokens * c.heads, 4, 4, 128 }, &.{ 2048, 512, 128, 1 }), &.{ 1, 1, 4, 128 }, .{});
    // Keep one four-row partial live at a time. Materializing all splits
    // caused a 255-register merge and poor occupancy on GB300.
    var merge = b.openFor(0, ns, 1, .{ b.full(&.{4}, @as(f64, -1e20), .f32), b.zeros(&.{4}, .f32), b.zeros(&.{ 4, 128 }, .f32) });
    const part_m = b.load(mv, &.{ id, merge.iv, zero }).reshape(&.{4});
    const part_s = b.load(sv, &.{ id, merge.iv, zero }).reshape(&.{4});
    const part_a = b.load(av, &.{ id, merge.iv, zero, zero }).reshape(&.{ 4, 128 });
    const m = merge.carried[0].maximum(part_m);
    const old_scale = b.exp2(merge.carried[0].sub(m));
    const part_scale = b.exp2(part_m.sub(m));
    merge.yield(.{ m, b.fma(merge.carried[1], old_scale, part_s.mul(part_scale)), b.fma(merge.carried[2], old_scale.reshape(&.{ 4, 1 }).broadcastTo(&.{ 4, 128 }), part_a.mul(part_scale.reshape(&.{ 4, 1 }))) });
    const denom = merge.results[1].maximum(1e-20);
    const inv = b.full(&.{4}, @as(f64, 1), .f32).div(denom);
    const out = merge.results[2].mul(inv.reshape(&.{ 4, 1 })).to(.bf16);
    const ov = b.partitionView(b.tensorView(a.out, &.{ c.query_tokens * c.heads, 4, 128 }, &.{ 512, 128, 1 }), &.{ 1, 4, 128 }, .{});
    _ = b.store(out.reshape(&.{ 1, 4, 128 }), ov, &.{ id, zero, zero });
    split.yieldThen(.{});
    active.yieldThen(.{});
}

pub fn pagedAttention(params: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    const cc = zml.platform.cuda.computeCapability(zml.Compiler.current().platform);
    if (cc == null or !cc.?.eql(.{ .major = 10, .minor = 3 }) or
        q.dtype() != .bf16 or k.dtype() != .bf16 or v.dtype() != .bf16 or
        q.dim(.hg) != 4 or q.dim(.hd) != 128 or k.dim(.k_chunk) <= 0 or
        opts.sliding_window >= 0 or
        q.axis(.b) != 0 or q.axis(.hkv) != 1 or q.axis(.hg) != 2 or q.axis(.hd) != 3 or
        !k.shape().eql(v.shape()))
        return triton.paged.pagedAttention(params, q, k.transpose(.{ .page, .k_chunk, .hkv, .hd }), v.transpose(.{ .page, .k_chunk, .hkv, .hd }), opts);
    const Context = struct {
        params: Parameters,
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        opts: AttentionOptions,

        fn body(self: @This(), _: zml.Shape) zml.Tensor {
            return pagedAttentionLocal(self.params, self.q, self.k, self.v, self.opts);
        }
    };
    return zml.ops.manualComputation(Context.body, Context{ .params = params, .q = q, .k = k, .v = v, .opts = opts }, q.shape());
}

fn useDirectDecode(max_pages: i64, page_size: i64, is_prefill: bool) bool {
    return !is_prefill and max_pages <= @divTrunc(512, page_size);
}

fn pagedAttentionLocal(params: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    const strides = k.shape().computeElementStrides();
    const batch = params.block_table.dim(.b);
    const heads = q.dim(.hkv);
    const splits: i64 = 32;
    const cfg: Config = .{ .batch = batch, .query_tokens = q.dim(.b), .heads = heads, .pages = k.dim(.page), .max_pages = params.block_table.dim(.p), .page_size = k.dim(.k_chunk), .splits = splits, .scale = opts.scale orelse 0.08838834764831845, .strides = .{ strides.get(k.axis(.page)), strides.get(k.axis(.hkv)), strides.get(k.axis(.k_chunk)), strides.get(k.axis(.hd)) } };
    // At most eight KV compute tiles fit in this cache capacity. The device
    // planner would always select one split, so neither planning nor merging
    // is needed. This uses the allocation bound, not an assumed runtime length.
    if (useDirectDecode(cfg.max_pages, cfg.page_size, params.options_.is_prefill)) {
        var direct_cfg = cfg;
        direct_cfg.splits = 1;
        return Decode.call(.{ .q = q, .k = k, .v = v, .table = params.block_table, .lengths = params.seq_lens, .starts = params.query_start_len }, .{
            .out = q.shape(),
            .maxima = .init(.{1}, .f32),
            .sums = .init(.{1}, .f32),
            .acc = .init(.{1}, .f32),
        }, .{ .cfg = direct_cfg, .grid = .{ @intCast(batch * heads), 1, 1 } }).out;
    }
    const plan = Plan.call(.{ .lengths = params.seq_lens, .starts = params.query_start_len }, .{ .plan = .init(.{ 4, batch }, .i32) }, .{ .cfg = cfg, .grid = .{ 1, 1, 1 } }).plan;
    const parts = ScheduledDecode.call(.{ .q = q, .k = k, .v = v, .table = params.block_table, .lengths = params.seq_lens, .starts = params.query_start_len, .plan = plan }, .{
        .out = q.shape(),
        .maxima = .init(.{ batch, heads, splits, 4 }, .f32),
        .sums = .init(.{ batch, heads, splits, 4 }, .f32),
        .acc = .init(.{ batch, heads, splits, 4, 128 }, .f32),
    }, .{ .cfg = cfg, .grid = .{ @intCast(decodeWorkCapacity(batch, heads) * heads), 1, 1 } });
    const decoded = ScheduledReduce.call(.{ .maxima = parts.maxima, .sums = parts.sums, .acc = parts.acc, .starts = params.query_start_len, .plan = plan, .initial = parts.out }, .{ .out = q.shape() }, .{ .cfg = cfg, .grid = .{ @intCast(batch * heads), 1, 1 }, .output_operand_aliases = .{ .out = .initial } }).out;
    if (!params.options_.is_prefill) return decoded;
    const prefilled = Prefill.call(.{ .q = q, .k = k, .v = v, .table = params.block_table, .lengths = params.seq_lens, .starts = params.query_start_len, .plan = plan, .initial = decoded }, .{
        .out = q.shape(),
        .maxima = .init(.{ q.dim(.b), heads, 4, 4 }, .f32),
        .sums = .init(.{ q.dim(.b), heads, 4, 4 }, .f32),
        .acc = .init(.{ q.dim(.b), heads, 4, 4, 128 }, .f32),
    }, .{ .cfg = cfg, .grid = .{ @intCast(prefillWorkCapacity(batch, q.dim(.b), heads) * heads), 1, 1 }, .output_operand_aliases = .{ .out = .initial } });
    return PrefillReduce.call(.{ .maxima = prefilled.maxima, .sums = prefilled.sums, .acc = prefilled.acc, .starts = params.query_start_len, .plan = plan, .initial = prefilled.out }, .{ .out = q.shape() }, .{ .cfg = cfg, .grid = .{ @intCast(q.dim(.b) * heads), 1, 1 }, .output_operand_aliases = .{ .out = .initial } }).out;
}

test "cutile decode and reduction IR" {
    const cfg: Config = .{ .batch = 8, .query_tokens = 8, .heads = 8, .pages = 512, .max_pages = 64, .splits = 8, .scale = 0.08838835, .strides = .{ 65536, 8192, 128, 1 } };
    const ir = try Decode.emit(std.testing.allocator, cfg);
    defer std.testing.allocator.free(ir);
    const reduce_ir = try Reduce.emit(std.testing.allocator, cfg);
    defer std.testing.allocator.free(reduce_ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "allow_tma") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "assume") != null);
    var direct = cfg;
    direct.max_pages = 3;
    direct.query_tokens = 3;
    direct.splits = 1;
    direct.strides = .{ 65536, 128, 1024, 1 };
    const direct_ir = try Decode.emit(std.testing.allocator, direct);
    defer std.testing.allocator.free(direct_ir);
    inline for (.{ Plan, ScheduledDecode, ScheduledReduce, Prefill, PrefillReduce }) |K| {
        const unified_ir = try K.emit(std.testing.allocator, cfg);
        defer std.testing.allocator.free(unified_ir);
    }
}

test "cutile split policy preserves large batches" {
    try std.testing.expectEqual(@as(i64, 1), splitCount(1, 129));
    try std.testing.expectEqual(@as(i64, 16), splitCount(1, 8192));
    try std.testing.expectEqual(@as(i64, 32), splitCount(1, 32768));
    try std.testing.expectEqual(@as(i64, 8), splitCount(8, 8192));
    try std.testing.expectEqual(@as(i64, 4), splitCount(16, 8192));
    try std.testing.expectEqual(@as(i64, 1), splitCount(64, 65536));
    try std.testing.expectEqual(@as(i64, 1), splitCount(256, 65536));
}

test "cutile direct decode requires a bounded cache and decode-only graph" {
    for ([_]i64{ 1, 16, 32, 64, 128, 256, 512 }) |page_size| {
        const pages = @divTrunc(512, page_size);
        try std.testing.expect(useDirectDecode(pages, page_size, false));
        try std.testing.expect(!useDirectDecode(pages + 1, page_size, false));
        try std.testing.expect(!useDirectDecode(pages, page_size, true));
    }
    try std.testing.expect(!useDirectDecode(1, 1024, false));
}

test "cutile cache pages are independent of MMA tiles" {
    for ([_]i64{ 1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128, 192, 256 }) |page_size| {
        for ([_]bool{ false, true }) |token_major| {
            const cfg: Config = .{
                .batch = 4,
                .query_tokens = 130,
                .heads = 8,
                .pages = 520,
                .max_pages = 130,
                .page_size = page_size,
                .splits = 32,
                .scale = 0.08838835,
                .strides = if (token_major) .{ page_size * 1024, 128, 1024, 1 } else .{ page_size * 1024, page_size * 128, 128, 1 },
            };
            inline for (.{ ScheduledDecode, Prefill }) |K| {
                const ir = try K.emit(std.testing.allocator, cfg);
                defer std.testing.allocator.free(ir);
                try std.testing.expect(std.mem.indexOf(u8, ir, "mmaf") != null);
            }
        }
    }
}

test "cutile prefill work capacity covers split query chunks" {
    for ([_]i64{ 1, 3, 8, 64, 256 }) |batch| {
        for ([_]i64{ 2, 31, 32, 33, 128, 256 }) |qlen| {
            const chunks = batch * @divTrunc(qlen + 31, 32);
            const splits = @min(4, @max(1, @divTrunc(38 + chunks - 1, chunks)));
            try std.testing.expect(chunks * splits <= prefillWorkCapacity(batch, batch * qlen, 8));
        }
    }
}

test "cutile adaptive work capacity covers ragged and inactive requests" {
    for ([_]usize{ 1, 3, 8, 17, 64, 256 }) |batch| {
        for ([_]usize{ 8, 64, 513, 2000 }) |max_pages| {
            var pages: [256]usize = @splat(0);
            var active: usize = 0;
            var total: usize = 0;
            for (pages[0..batch], 0..) |*p, i| {
                // Alternate inactive, prefill, short decode and long decode.
                p.* = if (i % 4 < 2) 0 else 1 + (i * 137) % max_pages;
                if (i == batch - 1) p.* = max_pages;
                active += @intFromBool(p.* > 0);
                total += p.*;
            }
            const target = @max(active, 38);
            const chunk = @max(8, (total + target - 1) / target);
            var work: usize = batch;
            for (pages[0..batch]) |p| work += @min(32, (p + chunk - 1) / chunk) -| 1;
            try std.testing.expect(work <= decodeWorkCapacity(@intCast(batch), 8));
        }
    }
    // The trace-like 255 short + one long case gets eight independent
    // chunks for the outlier instead of eight long-lived head CTAs total.
    const chunk = @max(8, (255 * 8 + 64 + 255) / 256);
    try std.testing.expectEqual(@as(usize, 8), (64 + chunk - 1) / chunk);
    try std.testing.expectEqual(@as(usize, 1), (8 + chunk - 1) / chunk);
}
