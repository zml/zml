const std = @import("std");
const zml = @import("../../zml.zig");
const sparse_mla = @import("../sparse_mla.zig");
const fly = zml.kernel.fly;
const Tensor = zml.Tensor;
const Value = fly.Value;

pub const Config = struct {
    queries: i64,
    heads: i64,
    pages: i64,
    page_size: i64,
    topk: i64,
    splits: i64,
    has_sink: bool,
    all_decode: bool,
};

const main_inputs: []const [:0]const u8 = &.{ "query", "kv_cache", "sink", "indices", "active", "scale" };
pub const Main2D = fly.Kernel(Config, .{
    .name = "sparse_mla_2d",
    .inputs = main_inputs,
    .outputs = &.{"output"},
    .run = runMain2D,
});
pub const Main3D = fly.Kernel(Config, .{
    .name = "sparse_mla_3d",
    .inputs = main_inputs,
    .outputs = &.{ "partial_output", "partial_lse" },
    .run = runMain3D,
});

pub const ReduceConfig = struct {
    queries: i64,
    heads: i64,
    input_splits: i64,
    output_splits: i64,
    has_sink: bool,
    all_decode: bool,
};
const reduce_inputs: []const [:0]const u8 = &.{ "input", "input_lse", "sink", "active" };
pub const Reduce = fly.Kernel(ReduceConfig, .{
    .name = "sparse_mla_reduce",
    .inputs = reduce_inputs,
    .outputs = &.{"output"},
    .run = runReduce,
});
pub const ReducePartials = fly.Kernel(ReduceConfig, .{
    .name = "sparse_mla_reduce_partials",
    .inputs = reduce_inputs,
    .outputs = &.{ "partial_output", "partial_lse" },
    .run = runReducePartials,
});

/// Shapes are local to a manual computation. Unsupported contracts fall back
/// before transposes or native launches enter the graph.
pub fn pagedAttention(q: Tensor, kv_cache: Tensor, sink: ?Tensor, indices: Tensor, active: Tensor, opts: sparse_mla.Options, cu_count: usize) ?Tensor {
    if (q.rank() != 3 or q.dtype() != .bf16 or q.dim(.hd) != 512 or
        q.dim(.q) <= 0 or q.dim(.h) <= 0 or opts.value_rank != 512 or
        kv_cache.rank() != 4 or kv_cache.dtype() != .bf16 or kv_cache.dim(.hd) != 512 or
        kv_cache.dim(.hkv) != 1 or kv_cache.dim(.page) <= 0 or kv_cache.dim(.k_chunk) <= 0 or
        indices.rank() != 2 or indices.dtype() != .i32 or indices.dim(.q) != q.dim(.q) or indices.dim(.topk) <= 0 or
        active.count() != 1 or active.dtype() != .i32) return null;
    if (sink) |value| {
        if (value.count() != @as(usize, @intCast(q.dim(.h)))) return null;
    }
    const config = sparse_mla.launchConfig(opts, @intCast(indices.dim(.topk)), cu_count);
    if (config.block_m != 16 or config.tile_size == 0 or config.tile_size > 16 or
        q.count() > std.math.maxInt(i32) / config.num_splits or indices.count() > std.math.maxInt(i32) or
        kv_cache.count() / 4 > std.math.maxInt(i32)) return null;
    return call(
        q.transpose(.{ .q, .h, .hd }),
        kv_cache.transpose(.{ .page, .k_chunk, .hkv, .hd }).squeeze(.hkv),
        sink,
        indices,
        active,
        opts,
        config,
    ).transpose(q.shape().tags());
}

/// Consumes local, row-major [query, head, 512] and [page, slot, 512] tensors.
/// Values span the full 512 columns; `config` supplies split/reduction settings.
pub fn call(q: Tensor, kv_cache: Tensor, sink: ?Tensor, indices: Tensor, active: Tensor, opts: sparse_mla.Options, config: sparse_mla.Config) Tensor {
    std.debug.assert(q.rank() == 3 and q.dim(2) == 512 and q.dtype() == .bf16);
    std.debug.assert(opts.value_rank == 512);
    std.debug.assert(kv_cache.rank() == 3 and kv_cache.dim(2) == 512 and kv_cache.dtype() == .bf16);
    std.debug.assert(indices.rank() == 2 and indices.dim(0) == q.dim(0) and indices.dtype() == .i32);
    std.debug.assert(active.count() == 1 and active.dtype() == .i32);
    std.debug.assert(config.block_m == 16 and config.tile_size > 0 and config.tile_size <= 16);
    std.debug.assert(q.count() <= std.math.maxInt(i32) and indices.count() <= std.math.maxInt(i32));
    std.debug.assert(config.num_splits > 0 and q.count() <= std.math.maxInt(i32) / config.num_splits);
    // Each i64 view element contains four adjacent BF16 values without copying.
    // View counts are bounded; the kernel forms byte addresses in i64.
    std.debug.assert(kv_cache.count() / 4 <= std.math.maxInt(i32));
    const cfg: Config = .{
        .queries = q.dim(0),
        .heads = q.dim(1),
        .pages = kv_cache.dim(0),
        .page_size = kv_cache.dim(1),
        .topk = indices.dim(1),
        .splits = @intCast(config.num_splits),
        .has_sink = sink != null,
        .all_decode = opts.all_decode,
    };
    const sink_ = if (sink) |value| value.convert(.f32).reshape(.{cfg.heads}) else Tensor.scalar(0, .f32).reshape(.{1});
    const inputs: Main3D.Inputs = .{
        .query = q.reshape(.{ cfg.queries, cfg.heads, 128, 4 }).bitCast(.i64),
        .kv_cache = kv_cache.reshape(.{ cfg.pages, cfg.page_size, 128, 4 }).bitCast(.i64),
        .sink = sink_,
        .indices = indices.reshape(.{ cfg.queries, cfg.topk }),
        .active = active.reshape(.{1}),
        .scale = Tensor.scalar(opts.scale orelse @as(f32, 1.0 / @sqrt(512.0)), .f32).reshape(.{1}),
    };
    const head_blocks = std.math.divCeil(i64, cfg.heads, 16) catch unreachable;
    const grid: [3]i32 = .{ @intCast(cfg.queries * head_blocks), @intCast(cfg.splits), 1 };
    if (cfg.splits == 1) {
        return Main2D.call(inputs, .{ .output = q.shape() }, .{ .cfg = cfg, .grid = grid, .threads = 256 }).output;
    }
    const partials = Main3D.call(inputs, .{
        .partial_output = zml.Shape.init(.{ cfg.queries, cfg.heads, cfg.splits, 512 }, .f32),
        .partial_lse = zml.Shape.init(.{ cfg.queries, cfg.heads, cfg.splits }, .f32),
    }, .{ .cfg = cfg, .grid = grid, .threads = 256 });
    var partial_output = partials.partial_output;
    var partial_lse = partials.partial_lse;
    var reduce_splits = cfg.splits;
    if (cfg.splits > config.grouped_reduce_threshold) {
        const output_splits = @divExact(cfg.splits, @as(i64, @intCast(config.splits_per_group)));
        const grouped = ReducePartials.call(.{
            .input = partial_output,
            .input_lse = partial_lse,
            .sink = Tensor.scalar(0, .f32).reshape(.{1}),
            .active = active.reshape(.{1}),
        }, .{
            .partial_output = zml.Shape.init(.{ cfg.queries, cfg.heads, output_splits, 512 }, .f32),
            .partial_lse = zml.Shape.init(.{ cfg.queries, cfg.heads, output_splits }, .f32),
        }, .{
            .cfg = .{ .queries = cfg.queries, .heads = cfg.heads, .input_splits = cfg.splits, .output_splits = output_splits, .has_sink = false, .all_decode = cfg.all_decode },
            .grid = .{ @intCast(cfg.queries), @intCast(cfg.heads * output_splits), 1 },
            .threads = 64,
        });
        partial_output = grouped.partial_output;
        partial_lse = grouped.partial_lse;
        reduce_splits = output_splits;
    }
    return Reduce.call(.{
        .input = partial_output,
        .input_lse = partial_lse,
        .sink = sink_,
        .active = active.reshape(.{1}),
    }, .{ .output = q.shape() }, .{
        .cfg = .{ .queries = cfg.queries, .heads = cfg.heads, .input_splits = reduce_splits, .output_splits = 1, .has_sink = cfg.has_sink, .all_decode = cfg.all_decode },
        .grid = .{ @intCast(cfg.queries), @intCast(cfg.heads), 1 },
        .threads = 64,
    }).output;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    return sparse_mla.Backend.auto(platform, .bf16) == .fly;
}

fn runMain2D(b: *fly.Builder, cfg: Config) fly.FinishError!void {
    std.debug.assert(cfg.splits == 1);
    try runMain(b, cfg, false, Main2D.args(b));
}

fn runMain3D(b: *fly.Builder, cfg: Config) fly.FinishError!void {
    try runMain(b, cfg, true, Main3D.args(b));
}

fn loadVector(b: *fly.Builder, ptr: Value, offset: Value, comptime dtype: fly.DType, comptime count: i64) Value {
    const atom = b.copyAtom(.{ .universal = @intCast(count * @as(i64, dtype.bitWidth())) }, dtype);
    return b.copyAtomLoad(atom, ptr.addOffset(offset).view(fly.L(count, 1)));
}

fn storeVector(b: *fly.Builder, ptr: Value, offset: Value, value: Value, comptime dtype: fly.DType, comptime count: i64) void {
    const atom = b.copyAtom(.{ .universal = @intCast(count * @as(i64, dtype.bitWidth())) }, dtype);
    b.copyAtomStore(atom, value, ptr.addOffset(offset).view(fly.L(count, 1)));
}

// One CTA owns one query, sixteen heads and one KV split. The four waves
// partition the 512 contraction dimensions; their score sums meet in LDS.
fn runMain(b: *fly.Builder, cfg: Config, comptime three_d: bool, a: anytype) fly.FinishError!void {
    const head_blocks = std.math.divCeil(i64, cfg.heads, 16) catch unreachable;
    const tiles = std.math.divCeil(i64, cfg.topk, 16) catch unreachable;
    const tiles_per_split = std.math.divCeil(i64, tiles, cfg.splits) catch unreachable;
    std.debug.assert(cfg.queries > 0 and cfg.heads > 0 and cfg.topk > 0 and cfg.page_size > 0);
    std.debug.assert(cfg.splits > 0 and cfg.splits <= tiles);
    const zero = b.constant(.f32, 0.0);
    const one = b.constant(.f32, 1.0);
    const ninf = b.constant(.f32, -std.math.inf(f32));
    const zero4 = zero.splat(4);
    const one4 = one.splat(4);
    const ninf4 = ninf.splat(4);
    const zero_bf16 = b.constant(.i16, 0).splat(4);
    const zero_words = b.constant(.i64, 0).splat(2);
    const block = b.blockId(.x).to(.i64);
    const split = b.blockId(.y).to(.i64);
    const qi = block.divUnsigned(head_blocks);
    const head_base = block.remUnsigned(head_blocks).mul(16);
    const active = b.constant(.i1, @intFromBool(cfg.all_decode)).bitOr(qi.cmp(.lt, a.active.at(0).to(.i64)));
    const output = if (three_d) a.partial_output else a.output;
    // Inactive queries leave their preallocated output rows untouched.
    var enabled = b.openIf(active);
    {
        const tid = b.threadId(.x).to(.i64);
        const lane = tid.remUnsigned(64);
        const wave = tid.divUnsigned(64);
        const mn = lane.remUnsigned(16);
        const kg4 = lane.divUnsigned(16).mul(4);
        const wave128 = wave.mul(128);
        const query_head = head_base.add(mn);
        const query_word = wave.mul(32).add(lane.divUnsigned(16));
        const valid_query_head = query_head.cmpUnsigned(.lt, cfg.heads);
        var q: [8]Value = undefined;
        for (0..8) |i| {
            var valid = b.openIfElse(valid_query_head, .{zero_bf16.type_()});
            valid.yieldThen(.{a.query.at(.{ qi, query_head, query_word.add(i * 4) }).splat(1).bitcast(.i16)});
            valid.yieldElse(.{zero_bf16});
            q[i] = valid.results[0];
        }
        const tile_start = split.mul(tiles).divUnsigned(cfg.splits);
        const tile_end = split.add(1).mul(tiles).divUnsigned(cfg.splits);
        const loader_key = lane.divUnsigned(4);
        const loader_dimension = wave128.add(lane.remUnsigned(4).mul(32));
        const scale = a.scale.at(0).splat(4);
        const cache_pointer = a.kv_cache.emitIter();
        const shared = b.sharedArray(.i16, 10496, 16);
        const shared_ptr = shared.emitIter();
        const atom = b.mmaAtom((fly.rocdl.MmaOpCDNA3MFMAType.get(b.ctx, .{
            .m = 16,
            .n = 16,
            .k = 16,
            .elemTyA = .float(b.ctx, .bf16),
            .elemTyB = .float(b.ctx, .bf16),
            .elemTyAcc = .float(b.ctx, .f32),
        }) catch return error.InvalidMlir).type_());
        var loop = b.openFor(b.constant(.i64, 0), tiles_per_split, 1, .{
            zero4, zero4, zero4, zero4, zero4, zero4, zero4, zero4, ninf4, zero4,
        });
        {
            const tile_index = tile_start.add(loop.iv);
            const load_pos = tile_index.mul(16).add(loader_key);
            const valid_index = tile_index.cmpUnsigned(.lt, tile_end).bitAnd(load_pos.cmpUnsigned(.lt, cfg.topk));
            const invalid_index = b.constant(.i32, -1);
            var index = b.openIfElse(valid_index, .{invalid_index.type_()});
            index.yieldThen(.{a.indices.at(.{ qi, load_pos })});
            index.yieldElse(.{invalid_index});
            const position = index.results[0];
            const load_valid = position.cmp(.ge, 0);
            // Form the word offset in i64 before multiplying cache dimensions.
            const page = position.to(.i64).divUnsigned(cfg.page_size);
            const slot = position.to(.i64).remUnsigned(cfg.page_size);
            const cache_base = page.mul(cfg.page_size).add(slot).mul(128).add(loader_dimension.divUnsigned(4));
            for (0..4) |i| {
                var valid = b.openIfElse(load_valid, .{zero_words.type_()});
                valid.yieldThen(.{loadVector(b, cache_pointer, cache_base.add(i * 2), .i64, 2)});
                valid.yieldElse(.{zero_words});
                const bits = valid.results[0].bitcast(.i16);
                const column = loader_dimension.add(i * 8).divUnsigned(8).bitXor(loader_key).mul(8);
                storeVector(b, shared_ptr, loader_key.mul(512).add(column), bits, .i16, 8);
            }
            b.barrier();

            var qk = zero4;
            for (0..8) |i| {
                const dimension = wave128.add(kg4).add(i * 16);
                const column = dimension.divUnsigned(8).bitXor(mn).mul(8).add(dimension.remUnsigned(8));
                const k = loadVector(b, shared_ptr, mn.mul(512).add(column), .i16, 4);
                qk = b.mmaAtomCall(atom, q[i], k, qk);
            }
            const score_lane = lane.mul(8);
            storeVector(b, shared_ptr, wave.mul(512).add(score_lane).add(8192), qk.bitcast(.i16), .i16, 8);
            b.barrier();
            var score_parts: [4]Value = undefined;
            for (0..4) |w| score_parts[w] = loadVector(b, shared_ptr, score_lane.add(8192 + w * 512), .i16, 8).bitcast(.f32);
            const score = score_parts[0].add(score_parts[1]).add(score_parts[2].add(score_parts[3])).mul(scale);
            const key_valid = load_valid.to(.i32).shuffle(.idx, mn.mul(4).to(.i32), 64).cmp(.ne, 0);
            var new_m = zero4;
            var new_l = zero4;
            var alpha = zero4;
            var probability: [4]Value = undefined;
            for (0..4) |i| {
                const row = kg4.add(i);
                const keep = head_base.add(row).cmpUnsigned(.lt, cfg.heads).bitAnd(key_valid);
                const s = keep.select(score.extract(@intCast(i)), ninf);
                var max = s;
                inline for (.{ 1, 2, 4, 8 }) |shift| max = max.maxNum(max.shuffleXor(shift, 16));
                const old_m = loop.carried[8].extract(@intCast(i));
                const old_l = loop.carried[9].extract(@intCast(i));
                const next_max = old_m.maxNum(max);
                const m = next_max.cmpf(.gt, ninf).select(next_max, zero);
                const prob = b.exp(s.sub(m));
                var sum = prob;
                inline for (.{ 1, 2, 4, 8 }) |shift| sum = sum.add(sum.shuffleXor(shift, 16));
                const rescale = b.exp(old_m.sub(m));
                new_m = new_m.insert(@intCast(i), m);
                new_l = new_l.insert(@intCast(i), old_l.mul(rescale).add(sum));
                alpha = alpha.insert(@intCast(i), rescale);
                probability[i] = prob;
            }
            var write_probability = b.openIf(wave.cmp(.eq, 0));
            for (0..4) |i| {
                const offset = kg4.add(i).mul(16).add(mn).add(10240);
                shared.set(offset, probability[i].to(.bf16).bitcast(.i16));
            }
            write_probability.yieldThen(.{});
            b.barrier();

            const p = loadVector(b, shared_ptr, mn.mul(16).add(kg4).add(10240), .i16, 4);
            var next_acc: [8]Value = undefined;
            for (0..8) |n| {
                const dimension = wave128.add(mn).add(n * 16);
                var v = zero_bf16;
                for (0..4) |k| {
                    const row = kg4.add(k);
                    const column = dimension.divUnsigned(8).bitXor(row).mul(8).add(dimension.remUnsigned(8));
                    v = v.insert(@intCast(k), shared.at(row.mul(512).add(column)));
                }
                next_acc[n] = b.mmaAtomCall(atom, p, v, loop.carried[n].mul(alpha));
            }
            // All waves finish consuming K and P before the next tile reuses LDS.
            b.barrier();
            loop.yield(.{ next_acc[0], next_acc[1], next_acc[2], next_acc[3], next_acc[4], next_acc[5], next_acc[6], next_acc[7], new_m, new_l });
        }
        var denominator = loop.results[9];
        var sink_alpha = one4;
        if (!three_d and cfg.has_sink) {
            var sink = zero4;
            for (0..4) |i| {
                const head = head_base.add(kg4).add(i);
                var valid = b.openIfElse(head.cmpUnsigned(.lt, cfg.heads), .{ninf.type_()});
                valid.yieldThen(.{a.sink.at(head)});
                valid.yieldElse(.{ninf});
                sink = sink.insert(@intCast(i), valid.results[0]);
            }
            const max = loop.results[8].maxNum(sink);
            const delta = loop.results[8].sub(max);
            const center = sink.sub(max);
            var sink_probability = zero4;
            sink_alpha = zero4;
            for (0..4) |i| {
                sink_alpha = sink_alpha.insert(@intCast(i), b.exp(delta.extract(@intCast(i))));
                sink_probability = sink_probability.insert(@intCast(i), b.exp(center.extract(@intCast(i))));
            }
            denominator = denominator.mul(sink_alpha).add(sink_probability);
        }
        const has_values = denominator.cmpf(.gt, zero4);
        const safe_denominator = has_values.select(denominator, one4);
        const inverse = one4.div(safe_denominator);
        for (0..8) |n| {
            const acc = if (three_d) loop.results[n] else loop.results[n].mul(sink_alpha);
            const normalized = has_values.select(acc.mul(inverse), zero4);
            const col = wave128.add(mn).add(n * 16);
            for (0..4) |i| {
                const head = head_base.add(kg4).add(i);
                var valid = b.openIf(head.cmpUnsigned(.lt, cfg.heads));
                const value = normalized.extract(@intCast(i));
                if (three_d) {
                    output.set(.{ qi, head, split, col }, value);
                } else {
                    output.set(.{ qi, head, col }, value.to(.bf16));
                }
                valid.yieldThen(.{});
            }
        }
        if (three_d) {
            // Keep logarithms scalar: the published ROCm lowering rejects vector log.
            var log_denominator = zero4;
            for (0..4) |i| log_denominator = log_denominator.insert(@intCast(i), b.log(safe_denominator.extract(@intCast(i))));
            const lse = has_values.select(loop.results[8].add(log_denominator), ninf4);
            var writer = b.openIf(wave.cmp(.eq, 0).bitAnd(mn.cmp(.eq, 0)));
            for (0..4) |i| {
                const head = head_base.add(kg4).add(i);
                var valid = b.openIf(head.cmpUnsigned(.lt, cfg.heads));
                a.partial_lse.set(.{ qi, head, split }, lse.extract(@intCast(i)));
                valid.yieldThen(.{});
            }
            writer.yieldThen(.{});
        }
        enabled.yieldThen(.{});
    }
}

fn runReduce(b: *fly.Builder, cfg: ReduceConfig) fly.FinishError!void {
    std.debug.assert(cfg.output_splits == 1);
    runReduction(b, cfg, false, Reduce.args(b));
}

fn runReducePartials(b: *fly.Builder, cfg: ReduceConfig) fly.FinishError!void {
    std.debug.assert(!cfg.has_sink);
    runReduction(b, cfg, true, ReducePartials.args(b));
}

fn runReduction(b: *fly.Builder, cfg: ReduceConfig, comptime grouped: bool, a: anytype) void {
    const group_splits = @divExact(cfg.input_splits, cfg.output_splits);
    std.debug.assert(group_splits > 0 and group_splits <= 64);
    const qi = b.blockId(.x).to(.i64);
    const head_group = b.blockId(.y).to(.i64);
    const head = head_group.divUnsigned(cfg.output_splits);
    const group = head_group.remUnsigned(cfg.output_splits);
    const group_start = group.mul(group_splits);
    const active = b.constant(.i1, @intFromBool(cfg.all_decode)).bitOr(qi.cmp(.lt, a.active.at(0).to(.i64)));
    const zero = b.constant(.f32, 0.0);
    const one = b.constant(.f32, 1.0);
    const ninf = b.constant(.f32, -std.math.inf(f32));
    const zero4 = zero.splat(4);
    var enabled = b.openIf(active);
    {
        const lane = b.threadId(.x).to(.i64);
        var valid_split = b.openIfElse(lane.cmpUnsigned(.lt, group_splits), .{ninf.type_()});
        valid_split.yieldThen(.{a.input_lse.at(.{ qi, head, group_start.add(lane) })});
        valid_split.yieldElse(.{ninf});
        const lse = valid_split.results[0];
        var max = lse;
        inline for (.{ 32, 16, 8, 4, 2, 1 }) |shift| max = max.maxNum(max.shuffleXor(shift, 64));
        const sink = if (!grouped and cfg.has_sink) a.sink.at(head) else ninf;
        const overall_max = max.maxNum(sink);
        const has_values = overall_max.cmpf(.gt, ninf);
        const safe_max = has_values.select(overall_max, zero);
        const weight = b.exp(lse.sub(safe_max));
        var sum = weight;
        inline for (.{ 32, 16, 8, 4, 2, 1 }) |shift| sum = sum.add(sum.shuffleXor(shift, 64));
        const sink_weight = if (!grouped and cfg.has_sink) b.exp(sink.sub(safe_max)) else zero;
        const denominator = sum.add(sink_weight);
        const safe_denominator = denominator.cmpf(.gt, zero).select(denominator, one);
        const normalizer = one.div(safe_denominator).splat(4);
        const group_flat = qi.mul(cfg.heads).add(head).mul(cfg.input_splits).add(group_start);
        const dim0 = lane.mul(8);
        var loop = b.openFor(b.constant(.i64, 0), group_splits, 1, .{ zero4, zero4 });
        {
            const weight4 = weight.shuffle(.idx, loop.iv.to(.i32), 64).splat(4);
            const base = group_flat.add(loop.iv).mul(512).add(dim0);
            const part0 = loadVector(b, a.input.emitIter(), base, .f32, 4);
            const part1 = loadVector(b, a.input.emitIter(), base.add(4), .f32, 4);
            loop.yield(.{ loop.carried[0].add(part0.mul(weight4)), loop.carried[1].add(part1.mul(weight4)) });
        }
        const output = if (grouped) a.partial_output else a.output;
        const row = qi.mul(cfg.heads).add(head);
        const base = if (grouped) row.mul(cfg.output_splits).add(group).mul(512) else row.mul(512);
        for (0..2) |i| {
            const value = has_values.select(loop.results[i].mul(normalizer), zero4);
            const offset = base.add(dim0).add(i * 4);
            if (grouped) {
                storeVector(b, output.emitIter(), offset, value, .f32, 4);
            } else {
                storeVector(b, output.emitIter(), offset, value.to(.bf16), .bf16, 4);
            }
        }
        if (grouped) {
            var writer = b.openIf(lane.cmp(.eq, 0));
            const lse_value = has_values.select(safe_max.add(b.log(safe_denominator)), ninf);
            a.partial_lse.set(.{ qi, head, group }, lse_value);
            writer.yieldThen(.{});
        }
        enabled.yieldThen(.{});
    }
}
