//! Paged attention for Hopper with warpgroup MMA. Two math warpgroups each own 64 of the 128 packed query rows of a CTA;
//! K and V tiles of 128 keys are double-buffered in 128-byte-swizzled shared memory with cp.async, S = Q K^T and O += P V run on
//! wgmma with P kept in registers, and the online softmax runs on the accumulator fragments.
const std = @import("std");

const builder = @import("dsl/cute/builder");
const zml = @import("zml/core");
const cute = zml.kernel.cute;
const ptx = @import("ptx.zig");

const B = cute.Builder;
const V = cute.Value;

pub const block_rows: i64 = 128;
pub const block_keys: i64 = 128;
pub const thread_count: i64 = 384;
const producer_threads: i64 = 128;
const stages: i64 = 3;
const producer_registers: u32 = 40;
const consumer_registers: u32 = 232;

pub const Config = struct {
    head_dim: i64,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages_per_seq: i64,
    page_size: i64,
    max_query: i64,
    dtype: cute.DType,
    causal: bool,
    window: i64,
    scale: f32,
    sink: bool,
    splits: i64,

    pub fn queryTiles(c: Config) i64 {
        return @divTrunc(c.max_query * c.group + block_rows - 1, block_rows);
    }
    fn tileBytes(c: Config) i64 {
        return block_keys * c.head_dim * 2;
    }
    fn keyBase(c: Config) i64 {
        return block_rows * c.head_dim * 2;
    }
    fn valueBase(c: Config) i64 {
        return c.keyBase() + stages * c.tileBytes();
    }
    fn barrierBase(c: Config) i64 {
        return c.valueBase() + stages * c.tileBytes();
    }
    fn sharedBytes(c: Config) i64 {
        return c.barrierBase() + 8 * (1 + 2 * stages);
    }
};

pub const Kernel = cute.Kernel(Config, .{
    .name = "paged_attention_cute_sm90",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *B, c: Config) cute.FinishError!void {
    switch (c.head_dim) {
        inline 64, 128 => |d| switch (c.dtype) {
            inline .bf16, .f16 => |dt| try runTile(b, c, d, dt),
            else => unreachable,
        },
        else => unreachable,
    }
}

fn loadI32(b: *B, ptr: V, index: V) V {
    return b.loadWords(1, ptr.add(index.to(.i64).mul(4)), .global, .{})[0];
}

// Offset in a tile stored as 64-column slabs of 128-byte rows, whose 16-byte chunks are XORed by the row mod 8 (the 128-byte
// swizzle wgmma expects). `rows` is the slab height.
fn tileOffset(row: V, chunk: V, rows: i64) V {
    const slab = chunk.div(8);
    const within = chunk.rem(8).bitXor(row.rem(8));
    return slab.mul(rows * 128).add(row.mul(128)).add(within.mul(16));
}

const Rows = [2]V;

const TileContext = struct {
    b: *B,
    c: Config,
    q: V,
    k: V,
    v: V,
    table: V,
    shared: V,
    tid: V,
    seq: V,
    head: V,
    first_row: V,
    start: V,
    qlen: V,
    klen: V,

    // Pages hold 16 rows, so the page table is read once per page and shared by K and V instead of once per 16-byte copy.
    fn loadKeyValue(t: TileContext, comptime head_dim: i64, key_base: V, stage: V) void {
        const b = t.b;
        const c = t.c;
        const chunks = @divExact(head_dim, 8);
        const rows_per_pass = @divExact(producer_threads, chunks);
        const passes = @divExact(block_keys, rows_per_pass);
        const tile_pages = @divExact(block_keys, 16);
        const chunk = t.tid.rem(chunks);
        const first_row = t.tid.div(chunks);
        var pages: [tile_pages]V = undefined;
        for (&pages, 0..) |*page, index| {
            const first_key = key_base.add(index * 16);
            const logical_page = b.select(first_key.lt(t.klen), first_key.div(16), b.constant(.i32, 0));
            page.* = loadI32(b, t.table, t.seq.mul(c.pages_per_seq).add(logical_page));
        }
        const head_offset = t.head.to(.i64).mul(c.head_dim).add(chunk.mul(8).to(.i64));
        const slab_offset = chunk.div(8).mul(block_keys * 128);
        const swizzled = chunk.rem(8).bitXor(first_row.rem(8)).mul(16);
        const key_destination = t.shared.add(c.keyBase()).add(stage.mul(c.tileBytes())).add(slab_offset).add(swizzled);
        const value_destination = t.shared.add(c.valueBase()).add(stage.mul(c.tileBytes())).add(slab_offset).add(swizzled);
        inline for (0..@as(usize, @intCast(passes))) |pass| {
            const row = first_row.add(pass * rows_per_pass);
            const valid = key_base.add(row).lt(t.klen);
            const page = pages[(pass * @as(usize, @intCast(rows_per_pass))) / 16];
            const element = page.to(.i64).mul(16 * c.kv_heads * c.head_dim).add(row.rem(16).to(.i64).mul(c.kv_heads * c.head_dim)).add(head_offset);
            b.cpAsync16(key_destination.add(row.mul(128)), t.k.add(element.mul(2)), valid);
            b.cpAsync16(value_destination.add(row.mul(128)), t.v.add(element.mul(2)), valid);
        }
    }

    fn loadRows(t: TileContext, source: V, key_base: V, destination: V, comptime is_query: bool) void {
        const b = t.b;
        const c = t.c;
        const chunks = @divExact(c.head_dim, 8);
        const row_count = if (is_query) block_rows else block_keys;
        var copy = b.openFor(t.tid, row_count * chunks, producer_threads, .{});
        {
            const row = copy.iv.div(chunks);
            const chunk = copy.iv.rem(chunks);
            const column = chunk.mul(8);
            const target = destination.add(tileOffset(row, chunk, row_count));
            if (is_query) {
                const packed_row = t.first_row.add(row);
                const valid = packed_row.lt(t.qlen.mul(c.group));
                const token = t.start.add(b.select(valid, packed_row.div(c.group), b.constant(.i32, 0)));
                const offset = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(packed_row.rem(c.group).to(.i64)).mul(c.head_dim).add(column.to(.i64));
                b.cpAsync16(target, source.add(offset.mul(2)), valid);
            } else {
                const key = key_base.add(row);
                const valid = key.lt(t.klen);
                const safe_key = b.select(valid, key, b.constant(.i32, 0));
                const page = loadI32(b, t.table, t.seq.mul(c.pages_per_seq).add(safe_key.div(c.page_size)));
                const offset = page.to(.i64).mul(c.page_size).add(safe_key.rem(c.page_size).to(.i64)).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.head_dim).add(column.to(.i64));
                b.cpAsync16(target, source.add(offset.mul(2)), valid);
            }
            copy.close();
        }
    }
};

fn fullBarrier(t: TileContext, stage: V) V {
    return t.shared.add(t.c.barrierBase() + 8).add(stage.mul(8));
}

fn emptyBarrier(t: TileContext, stage: V) V {
    return t.shared.add(t.c.barrierBase() + 8 + 8 * stages).add(stage.mul(8));
}

fn rowOf(index: usize) usize {
    return (index % 4) / 2;
}

fn quadMax(b: *B, value: V) V {
    const pair = b.fmax(value, b.shuffleXor(value, 1), .{ .ftz = true });
    return b.fmax(pair, b.shuffleXor(pair, 2), .{ .ftz = true });
}

fn quadSum(b: *B, value: V) V {
    const far = value.add(b.shuffleXor(value, 2));
    return far.add(b.shuffleXor(far, 1));
}

fn Mainloop(comptime d: usize, comptime dt: cute.DType) type {
    return struct {
        const Self = @This();
        const accumulator_count = d / 2;
        const score_count = block_keys / 2;

        b: *B,
        c: Config,
        t: TileContext,
        query_descriptor: ptx.MatrixDescriptor,
        first_row: V,
        row_in_tile: V,
        column0: V,
        lo: V,
        scale_log2: V,
        zero: V,
        neg_inf: V,
        release_leader: V,

        const SoftmaxState = struct { maximum: [2]V, sum: [2]V, scale: [2]V };

        fn stageOf(self: Self, block: V) V {
            return block.sub(self.lo).rem(stages);
        }

        fn waitFull(self: Self, block: V) void {
            const step = block.sub(self.lo);
            self.b.mbarrierWait(fullBarrier(self.t, self.stageOf(block)), step.div(stages).bitAnd(1), .{});
            ptx.asyncProxyFence(self.b);
        }

        fn release(self: Self, block: V) void {
            var leader = self.b.openIf(self.release_leader);
            {
                builder.mbarrierArrive(self.b, emptyBarrier(self.t, self.stageOf(block)));
                leader.close();
            }
        }

        // S = Q K^T of one block, issued without waiting.
        fn issueScores(self: Self, scores: *[score_count]V, block: V) void {
            const b = self.b;
            const key_descriptor = ptx.matrixDescriptor(b, self.t.shared.add(self.c.keyBase()).add(self.stageOf(block).mul(self.c.tileBytes())), 16, 1024);
            ptx.fenceOperands(b, .f32, scores);
            ptx.wgmmaFence(b);
            inline for (0..d / 16) |k_step| {
                const slab: i32 = @intCast(k_step / 4);
                const within_slab: i32 = @intCast((k_step % 4) * 32);
                const query_offset: i32 = (slab * @as(i32, block_rows) * 128 + within_slab) >> 4;
                const key_offset: i32 = (slab * @as(i32, block_keys) * 128 + within_slab) >> 4;
                ptx.wgmmaSharedShared(b, dt, block_keys, scores, self.query_descriptor.advance(query_offset), key_descriptor.advance(key_offset), k_step > 0);
            }
            ptx.wgmmaCommit(b);
        }

        // O += P V of one block with P in registers, issued without waiting.
        fn issueValues(self: Self, output: *[accumulator_count]V, words: *[score_count / 2]V, block: V) void {
            const b = self.b;
            const value_descriptor = ptx.matrixDescriptor(b, self.t.shared.add(self.c.valueBase()).add(self.stageOf(block).mul(self.c.tileBytes())), block_keys * 128, 1024);
            ptx.fenceOperands(b, .f32, output);
            ptx.fenceOperands(b, .i32, words);
            ptx.wgmmaFence(b);
            inline for (0..block_keys / 16) |k_step| {
                const value_offset: i32 = @intCast((k_step * 16 * 128) >> 4);
                ptx.wgmmaRegistersShared(b, dt, d, output, words[4 * k_step ..][0..4].*, value_descriptor.advance(value_offset));
            }
            ptx.wgmmaCommit(b);
        }

        fn rescale(self: Self, output: *[accumulator_count]V, scale: [2]V) void {
            _ = self;
            for (output, 0..) |*value, index| {
                value.* = value.*.mul(scale[rowOf(index)]);
            }
        }

        fn packProbabilities(self: Self, scores: *const [score_count]V) [score_count / 2]V {
            var words: [score_count / 2]V = undefined;
            for (&words, 0..) |*word, index| {
                word.* = ptx.packHalves(self.b, dt, scores[2 * index], scores[2 * index + 1]);
            }
            return words;
        }

        // Masks the scores of `block`, folds them into the running maximum and sum, and replaces them by probabilities.
        fn softmax(self: Self, block: V, scores: *[score_count]V, previous_maximum: [2]V, previous_sum: [2]V) SoftmaxState {
            const b = self.b;
            const c = self.c;
            const t = self.t;
            // A column offset (relative to this thread's first key) is visible when lower <= offset <= upper, one compare per bound.
            var upper: [2]V = undefined;
            var lower: [2]V = undefined;
            const thread_key = block.mul(block_keys).add(self.column0);
            for (0..2) |h| {
                const row = self.first_row.add(self.row_in_tile).add(h * 8);
                const position = t.klen.sub(t.qlen).add(row.div(c.group));
                var last_key = t.klen.sub(1);
                if (c.causal) {
                    last_key = last_key.minimum(position);
                }
                upper[h] = b.select(row.lt(t.qlen.mul(c.group)), last_key.sub(thread_key), b.constant(.i32, -1));
                lower[h] = position.sub(c.window - 1).sub(thread_key);
            }
            var thread_max: [2]V = .{ self.neg_inf, self.neg_inf };
            for (scores, 0..) |*score, index| {
                const h = rowOf(index);
                const column_offset: i32 = @intCast((index / 4) * 8 + index % 2);
                var valid = upper[h].ge(column_offset);
                if (c.window > 0) {
                    valid = valid.bitAnd(lower[h].le(column_offset));
                }
                score.* = b.select(valid, score.*, self.neg_inf);
                thread_max[h] = b.fmax(thread_max[h], score.*, .{ .ftz = true });
            }
            var state: SoftmaxState = undefined;
            var offset: [2]V = undefined;
            for (0..2) |h| {
                const maximum = b.fmax(quadMax(b, thread_max[h]), previous_maximum[h], .{ .ftz = true });
                const finite = b.select(maximum.eq(self.neg_inf), self.zero, maximum);
                state.scale[h] = b.exp2Approx(previous_maximum[h].sub(finite).mul(self.scale_log2));
                offset[h] = finite.mul(self.scale_log2).neg();
                state.maximum[h] = maximum;
                state.sum[h] = previous_sum[h].mul(state.scale[h]);
            }
            for (scores, 0..) |*score, index| {
                score.* = b.exp2Approx(b.fma(score.*, self.scale_log2, offset[rowOf(index)]));
                state.sum[rowOf(index)] = state.sum[rowOf(index)].add(score.*);
            }
            return state;
        }
    };
}

fn runTile(b: *B, c: Config, comptime d: usize, comptime dt: cute.DType) cute.FinishError!void {
    std.debug.assert(c.sharedBytes() <= 227 * 1024);
    const accumulator_count = d / 2;
    const score_count = block_keys / 2;
    const a = try b.declareArgs(.{
        .q = .{ .ptr = dt },
        .k = .{ .ptr = dt },
        .v = .{ .ptr = dt },
        .table = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .starts = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .out = .{ .ptr = if (c.splits > 1) cute.DType.f32 else dt },
    });
    b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())})));
    const seq = b.blockIdx().x.div(c.queryTiles());
    const first_row = b.blockIdx().x.rem(c.queryTiles()).mul(block_rows);
    const starts = b.ptrToInt(a.starts, .i64);
    const start = loadI32(b, starts, seq);
    const qlen = loadI32(b, starts, seq.add(1)).sub(start);
    var active = b.openIf(first_row.lt(qlen.mul(c.group)));
    {
        const tid = b.threadIdx().x;
        const t: TileContext = .{
            .b = b,
            .c = c,
            .q = b.ptrToInt(a.q, .i64),
            .k = b.ptrToInt(a.k, .i64),
            .v = b.ptrToInt(a.v, .i64),
            .table = b.ptrToInt(a.table, .i64),
            .shared = b.ptrToInt(b.allocSmem(.i8, c.sharedBytes(), 1024), .i32),
            .tid = tid,
            .seq = seq,
            .head = b.blockIdx().y,
            .first_row = first_row,
            .start = start,
            .qlen = qlen,
            .klen = loadI32(b, b.ptrToInt(a.lengths, .i64), seq),
        };
        const split = b.blockIdx().z;
        const sink_pointer = b.ptrToInt(a.sink, .i64);
        const out_pointer = b.ptrToInt(a.out, .i64);

        const first_pos = t.klen.sub(qlen).add(first_row.div(c.group));
        const last_pos = t.klen.sub(qlen).add(first_row.add(block_rows - 1).div(c.group).minimum(qlen.sub(1)));
        const begin = if (c.window > 0) first_pos.sub(c.window - 1).maximum(0) else b.constant(.i32, 0);
        const end = if (c.causal) last_pos.add(1).minimum(t.klen) else t.klen;
        const first_tile = begin.div(block_keys);
        const tiles = end.add(block_keys - 1).div(block_keys).sub(first_tile).maximum(0);
        const lo = first_tile.add(tiles.mul(split).div(c.splits));
        const hi = first_tile.add(tiles.mul(split.add(1)).div(c.splits));

        const warpgroup = tid.div(128).sub(1);
        const lane = tid.rem(32);
        const row_in_tile = warpgroup.mul(64).add(tid.rem(128).div(32).mul(16)).add(lane.div(4));
        const column0 = lane.rem(4).mul(2);
        const zero = b.constant(.f32, 0);
        const neg_inf = b.constant(.f32, -std.math.inf(f32));
        const scale_log2 = b.constant(.f32, c.scale * std.math.log2e);

        const query_barrier = t.shared.add(c.barrierBase());
        var initializer = b.openIf(tid.eq(0));
        {
            b.mbarrierInit(query_barrier, producer_threads);
            inline for (0..stages) |stage_index| {
                b.mbarrierInit(fullBarrier(t, b.constant(.i32, stage_index)), producer_threads);
                b.mbarrierInit(emptyBarrier(t, b.constant(.i32, stage_index)), 2);
            }
            b.fenceMbarrierInit();
            initializer.close();
        }
        b.syncThreads();

        var producer = b.openIf(tid.lt(producer_threads));
        {
            b.setMaxRegisterDecrease(producer_registers);
            t.loadRows(t.q, b.constant(.i32, 0), t.shared, true);
            b.cpAsyncMbarrierArrive(query_barrier, true);
            var loads = b.openFor(lo, hi, 1, .{});
            {
                const step = loads.iv.sub(lo);
                const stage = step.rem(stages);
                b.mbarrierWait(emptyBarrier(t, stage), step.div(stages).bitAnd(1).bitXor(1), .{});
                if (c.page_size == 16) {
                    t.loadKeyValue(d, loads.iv.mul(block_keys), stage);
                } else {
                    t.loadRows(t.k, loads.iv.mul(block_keys), t.shared.add(c.keyBase()).add(stage.mul(c.tileBytes())), false);
                    t.loadRows(t.v, loads.iv.mul(block_keys), t.shared.add(c.valueBase()).add(stage.mul(c.tileBytes())), false);
                }
                b.cpAsyncMbarrierArrive(fullBarrier(t, stage), true);
                loads.close();
            }
            producer.close();
        }

        var consumer = b.openIf(tid.ge(producer_threads));
        {
        b.setMaxRegisterIncrease(consumer_registers);
        const query_descriptor = ptx.matrixDescriptor(b, t.shared.add(warpgroup.mul(64 * 128)), 16, 1024);
        b.mbarrierWait(query_barrier, b.constant(.i32, 0), .{});
        ptx.asyncProxyFence(b);

        const mainloop: Mainloop(d, dt) = .{
            .b = b,
            .c = c,
            .t = t,
            .query_descriptor = query_descriptor,
            .first_row = first_row,
            .row_in_tile = row_in_tile,
            .column0 = column0,
            .lo = lo,
            .scale_log2 = scale_log2,
            .zero = zero,
            .neg_inf = neg_inf,
            .release_leader = tid.rem(128).eq(0),
        };
        const zero_output: [accumulator_count]V = @splat(zero);
        const zero_packed = ptx.packVector(b, &zero_output);
        var nonempty = b.openIfElse(hi.gt(lo), .{ zero_packed.type_(), zero.type_(), zero.type_(), zero.type_(), zero.type_() });
        {
            var scores: [score_count]V = @splat(zero);
            mainloop.waitFull(lo);
            mainloop.issueScores(&scores, lo);
            ptx.wgmmaWait(b, 0);
            ptx.fenceOperands(b, .f32, &scores);
            const first = mainloop.softmax(lo, &scores, .{ neg_inf, neg_inf }, .{ zero, zero });
            var first_words = mainloop.packProbabilities(&scores);
            const carried: [8]V = .{ zero_packed, ptx.packVector(b, &first_words), first.maximum[0], first.maximum[1], first.sum[0], first.sum[1], first.scale[0], first.scale[1] };
            var loop = b.openFor(lo.add(1), hi, 1, carried);
            {
                const block = loop.iv;
                var output = ptx.unpackVector(b, accumulator_count, loop.carried[0], .f32);
                var previous_words = ptx.unpackVector(b, score_count / 2, loop.carried[1], .i32);
                var block_scores: [score_count]V = @splat(zero);
                mainloop.waitFull(block);
                mainloop.issueScores(&block_scores, block);
                mainloop.rescale(&output, loop.carried[6..8].*);
                mainloop.issueValues(&output, &previous_words, block.sub(1));
                ptx.wgmmaWait(b, 1);
                ptx.fenceOperands(b, .f32, &block_scores);
                const state = mainloop.softmax(block, &block_scores, loop.carried[2..4].*, loop.carried[4..6].*);
                ptx.wgmmaWait(b, 0);
                ptx.fenceOperands(b, .f32, &output);
                mainloop.release(block.sub(1));
                var words = mainloop.packProbabilities(&block_scores);
                loop.yield(.{ ptx.packVector(b, &output), ptx.packVector(b, &words), state.maximum[0], state.maximum[1], state.sum[0], state.sum[1], state.scale[0], state.scale[1] });
            }
            var output = ptx.unpackVector(b, accumulator_count, loop.results[0], .f32);
            var last_words = ptx.unpackVector(b, score_count / 2, loop.results[1], .i32);
            mainloop.rescale(&output, loop.results[6..8].*);
            mainloop.issueValues(&output, &last_words, hi.sub(1));
            ptx.wgmmaWait(b, 0);
            ptx.fenceOperands(b, .f32, &output);
            mainloop.release(hi.sub(1));
            nonempty.yieldThen(.{ ptx.packVector(b, &output), loop.results[2], loop.results[3], loop.results[4], loop.results[5] });
        }
        {
            nonempty.yieldElse(.{ zero_packed, neg_inf, neg_inf, zero, zero });
        }

        const final_output = ptx.unpackVector(b, accumulator_count, nonempty.results[0], .f32);
        for (0..2) |h| {
            const row = first_row.add(row_in_tile).add(h * 8);
            var valid = b.openIf(row.lt(qlen.mul(c.group)));
            {
                const m = nonempty.results[1 + h].mul(scale_log2);
                const l = quadSum(b, nonempty.results[3 + h]);
                var norm = b.select(l.gt(0), b.rcpApprox(l), zero);
                if (c.splits == 1 and c.sink) {
                    const sink = loadI32(b, sink_pointer, t.head.mul(c.group).add(row.rem(c.group))).bitCast(.f32).mul(std.math.log2e);
                    const maximum = m.maximum(sink);
                    const weight = b.exp2Approx(m.sub(maximum));
                    norm = weight.div(l.mul(weight).add(b.exp2Approx(sink.sub(maximum))));
                }
                const token = start.add(row.div(c.group));
                var base = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(row.rem(c.group).to(.i64));
                if (c.splits > 1) {
                    base = split.to(.i64).mul(c.tokens * c.kv_heads * c.group).add(base).mul(c.head_dim + 2);
                } else {
                    base = base.mul(c.head_dim);
                }
                for (0..d / 8) |group_index| {
                    const column = column0.add(group_index * 8);
                    const first = final_output[group_index * 4 + h * 2];
                    const second = final_output[group_index * 4 + h * 2 + 1];
                    if (c.splits > 1) {
                        b.storeWords(2, out_pointer.add(base.add(column.to(.i64)).mul(4)), .global, .{ first.bitCast(.i32), second.bitCast(.i32) });
                    } else {
                        b.storeWords(1, out_pointer.add(base.add(column.to(.i64)).mul(2)), .global, .{b.packHalf2(first.mul(norm), second.mul(norm), dt)});
                    }
                }
                if (c.splits > 1) {
                    var leader = b.openIf(lane.rem(4).eq(0));
                    {
                        b.storeWords(1, out_pointer.add(base.add(c.head_dim).mul(4)), .global, .{m.bitCast(.i32)});
                        b.storeWords(1, out_pointer.add(base.add(c.head_dim + 1).mul(4)), .global, .{l.bitCast(.i32)});
                        leader.close();
                    }
                }
                valid.close();
            }
        }
        consumer.close();
        }
        active.close();
    }
}