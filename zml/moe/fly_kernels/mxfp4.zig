const std = @import("std");

const zml = @import("../../zml.zig");
const Tensor = zml.Tensor;
const fly = zml.kernel.fly;
const Value = fly.Value;
const Builder = fly.Builder;

pub const matrixInstruction: @import("platforms").capabilities.matmul.Instruction = .mfma_f32_16x16x16bf16_1k;

/// The caller owns route localization, activation and reduction.
/// `expert_ids` contains local bank indices, including -1 for remote experts.
/// Sorted routes must contain a real prefix followed by padding per expert.
/// `block_m` is their alignment; native tiles contain 16 or 32 rows.
pub fn call(
    input: Tensor,
    weight: Tensor,
    scales: Tensor,
    sorted_ids: Tensor,
    expert_ids: Tensor,
    padded_count: Tensor,
    output: zml.Shape,
    topk: usize,
    naive: bool,
    block_m: usize,
) Tensor {
    std.debug.assert(input.rank() == 2 and input.dtype() == .bf16 and weight.rank() == 3 and
        (weight.dtype() == .u8 or weight.dtype() == .i8) and scales.rank() == 3 and scales.dtype() == .u8 and
        sorted_ids.rank() == 1 and sorted_ids.dtype() == .i32 and expert_ids.rank() == 1 and expert_ids.dtype() == .i32 and
        padded_count.count() == 1 and padded_count.dtype() == .i32 and output.rank() >= 2 and output.dtype() == .bf16 and
        topk > 0 and block_m > 0 and block_m % 16 == 0);
    const n = weight.dim(1);
    const k = input.dim(1);
    std.debug.assert((n == 4608 and k == 5120) or (n == 5120 and k == 2304));
    std.debug.assert(input.dim(0) > 0 and weight.dim(0) > 0 and weight.dim(2) == @divExact(k, 2) and
        scales.dim(0) == weight.dim(0) and scales.dim(1) == n and scales.dim(2) == @divExact(k, 32) and
        output.dim(-1) == n);
    // Keep packed-word counts and route/output indices within signed i32.
    std.debug.assert(weight.count() / 4 <= std.math.maxInt(i32) and output.count() <= std.math.maxInt(i32) and
        sorted_ids.count() <= std.math.maxInt(i32));
    const routes: i64 = @intCast(output.count() / @as(usize, @intCast(n)));
    std.debug.assert(topk <= routes and @mod(@as(usize, @intCast(routes)), topk) == 0 and
        input.dim(0) == @divTrunc(routes, @as(i64, @intCast(topk))));
    const capacity: i64 = @intCast(sorted_ids.count());
    const blocks: i64 = @intCast(expert_ids.count());
    if (naive) {
        std.debug.assert(blocks == routes);
    } else {
        std.debug.assert(capacity > 0 and block_m <= capacity and @mod(capacity, @as(i64, @intCast(block_m))) == 0 and
            blocks == @divExact(capacity, @as(i64, @intCast(block_m))));
    }
    if (naive and routes <= 6) return Gemv.call(input, weight, scales, expert_ids, output, topk);
    return callMfma(input, weight, scales, sorted_ids, expert_ids, padded_count, output, topk, naive, block_m);
}

fn callMfma(input: Tensor, weight: Tensor, scales: Tensor, sorted_ids: Tensor, expert_ids: Tensor, padded_count: Tensor, output: zml.Shape, topk: usize, naive: bool, block_m: usize) Tensor {
    const n = weight.dim(1);
    const k = input.dim(1);
    const routes = input.dim(0) * @as(i64, @intCast(topk));
    const native_m: i64 = if (!naive and block_m % 32 == 0) 32 else 16;
    const cfg: Config = .{
        .tokens = input.dim(0),
        .experts = weight.dim(0),
        .n = n,
        .k = k,
        .ka = @divExact(k, 4),
        .kb = @divExact(k, 8),
        .ks = @divExact(k, 32),
        .topk = @as(i64, @intCast(topk)),
        .routes = routes,
        .capacity = @as(i64, @intCast(sorted_ids.count())),
        .blocks = @as(i64, @intCast(expert_ids.count())),
        .subtiles = @divExact(@as(i64, @intCast(block_m)), native_m),
        .native_m = native_m,
        .wide = @as(i64, @intFromBool(native_m == 32)),
        .stage_elements = native_m * 128,
        .shared_elements = native_m * 256,
        .naive = @as(i64, @intFromBool(naive)),
    };
    const a = input.reshape(.{ input.dim(0), @divExact(k, 4), 4 }).bitCast(.i64);
    const b = weight.reshape(.{ weight.dim(0), n, @divExact(k, 8), 4 }).bitCast(.i32);
    return Kernel.call(.{ .a = a, .b = b, .scales = scales, .sorted = sorted_ids, .experts = expert_ids, .padded = padded_count.reshape(.{1}) }, .{
        .output = zml.Shape.init(.{ routes, n }, .bf16),
    }, .{
        .cfg = cfg,
        .threads = 256,
        .grid = .{ @intCast(@divExact(n, 64)), @intCast(if (naive) routes else @divExact(cfg.capacity, native_m)), 1 },
    }).output.reshape(output);
}

const Config = struct {
    tokens: i64,
    experts: i64,
    n: i64,
    k: i64,
    ka: i64,
    kb: i64,
    ks: i64,
    topk: i64,
    routes: i64,
    capacity: i64,
    blocks: i64,
    subtiles: i64,
    native_m: i64,
    wide: i64,
    stage_elements: i64,
    shared_elements: i64,
    naive: i64,
};

const Kernel = zml.kernel.fly.Kernel(Config, .{
    .name = "mxfp4_raw_mfma",
    .inputs = &.{ "a", "b", "scales", "sorted", "experts", "padded" },
    .outputs = &.{"output"},
    .run = run,
});

fn run(b: *Builder, cfg: Config) fly.FinishError!void {
    const args = Kernel.args(b);
    const zero = b.constant(.i32, 0);
    const fzero = b.constant(.f32, 0.0);
    const zeros = fzero.splat(4);
    const block_n = b.blockId(.x).to(.i64);
    const block_m = b.blockId(.y).to(.i64);
    const route_start = block_m.mul(cfg.native_m);
    const has_rows = if (cfg.naive != 0) b.constant(.i1, 1) else blk: {
        const padded = args.padded.at(b.constant(.i64, 0)).to(.i64);
        var has_block = b.openIfElse(route_start.cmpUnsigned(.lt, padded), .{b.constant(.i1, 0).type_()});
        has_block.yieldThen(.{args.sorted.at(route_start).cmp(.ne, cfg.routes)});
        has_block.yieldElse(.{b.constant(.i1, 0)});
        break :blk has_block.results[0];
    };
    var block = b.openIf(has_rows);
    const expert_block = if (cfg.naive != 0) block_m else block_m.divUnsigned(cfg.subtiles);
    const eid = args.experts.at(expert_block);
    const valid_expert = eid.cmp(.ge, 0).bitAnd(eid.cmp(.lt, cfg.experts));
    const thread = b.threadId(.x).to(.i64);
    const lane = thread.remUnsigned(64);
    const wave = thread.divUnsigned(64);
    const mn_lane = lane.remUnsigned(16);
    const k_lane = lane.divUnsigned(16);
    const out_n = block_n.mul(64).add(wave.mul(16)).add(mn_lane);
    // Each expert has a contiguous real prefix before route-count padding.
    const second_active = if (cfg.wide != 0) blk: {
        const first = args.sorted.at(route_start.add(16));
        break :blk first.cmp(.ge, 0).bitAnd(first.cmp(.lt, cfg.routes));
    } else b.constant(.i1, 0);

    var sums = b.openIfElse(valid_expert, .{ zeros.type_(), zeros.type_() });
    const expert = eid.to(.i64);
    const load_row = thread.divUnsigned(16);
    const load_chunk = thread.remUnsigned(16);
    const a_route = routeForRow(b, args.sorted, block_m, route_start, load_row, cfg.naive != 0);
    const a_valid = a_route.cmp(.ge, 0).bitAnd(a_route.cmp(.lt, cfg.routes));
    const a_base = a_route.to(.i64).divUnsigned(cfg.topk).mul(cfg.ka);
    var second_route = b.openIfElse(second_active, .{zero.type_()});
    second_route.yieldThen(.{args.sorted.at(route_start.add(load_row).add(16))});
    second_route.yieldElse(.{b.constant(.i32, -1)});
    const a_route_g1 = second_route.results[0];
    const a_valid_g1 = a_route_g1.cmp(.ge, 0).bitAnd(a_route_g1.cmp(.lt, cfg.routes));
    const a_base_g1 = a_route_g1.to(.i64).divUnsigned(cfg.topk).mul(cfg.ka);

    // One allocation, two disjoint ping-pong regions; never two identical leaves.
    const shared = b.sharedArray(.i16, cfg.shared_elements, 16);
    const load_a = b.copyAtom(.{ .universal = 128 }, .i64);
    const load_b = b.copyAtom(.{ .universal = 128 }, .i32);
    const copy_shared = b.copyAtom(.{ .universal = 128 }, .i16);
    const loader: TileLoader = .{
        .a = args.a,
        .weight = args.b,
        .scales = args.scales,
        .shared = shared,
        .load_a = load_a,
        .load_b = load_b,
        .copy_shared = copy_shared,
        .a_base = a_base,
        .a_base_g1 = a_base_g1,
        .a_valid = a_valid,
        .a_valid_g1 = a_valid_g1,
        .second_active = second_active,
        .a_col = load_chunk.mul(2),
        .a_store = load_row.mul(128).add(load_chunk.bitXor(load_row).mul(8)),
        .b_base = expert.mul(cfg.n * cfg.kb).add(out_n.mul(cfg.kb)),
        .expert = expert,
        .out_n = out_n,
        .k_lane = k_lane,
    };
    const initial = loader.load(b, b.constant(.i64, 0), b.constant(.i64, 0));
    b.barrier();
    const atom = b.mmaAtom((fly.mmaAtomType(b.ctx, matrixInstruction) catch return error.InvalidMlir));
    var k = b.openFor(b.constant(.i64, 0), cfg.k, 128, .{ zeros, zeros, initial.words, initial.scale });
    const read_base = k.iv.divUnsigned(128).bitAnd(1).mul(cfg.stage_elements);
    const write_base = k.iv.divUnsigned(128).bitAnd(1).bitXor(1).mul(cfg.stage_elements);
    var a_fragments: [4]Value = undefined;
    var a_fragments_g1: [4]Value = undefined;
    inline for (0..4) |u| {
        const offset = read_base.add(mn_lane.mul(128)).add(k_lane.mul(4).add(u).bitXor(mn_lane).mul(8));
        a_fragments[u] = b.copyAtomLoad(copy_shared, vectorView(shared, offset, 8));
    }
    inline for (0..4) |u| {
        const offset = read_base.add(mn_lane.mul(128)).add(k_lane.mul(4).add(u).bitXor(mn_lane).mul(8)).add(2048);
        const azero = b.constant(.i16, 0).splat(8);
        var second = b.openIfElse(second_active, .{azero.type_()});
        second.yieldThen(.{b.copyAtomLoad(copy_shared, vectorView(shared, offset, 8))});
        second.yieldElse(.{azero});
        a_fragments_g1[u] = second.results[0];
    }
    // Prefetch the other LDS region only after all current A fragments are read.
    const next_k = k.iv.add(128);
    var next = b.openIfElse(next_k.cmpUnsigned(.lt, cfg.k), .{ initial.words.type_(), initial.scale.type_() });
    const next_tile = loader.load(b, next_k, write_base);
    next.yieldThen(.{ next_tile.words, next_tile.scale });
    next.yieldElse(.{ b.constant(.i32, 0).splat(4), fzero });
    var acc = k.carried[0];
    var acc_g1 = k.carried[1];
    inline for (0..4) |u| {
        const decoded = decodeWeight(b, k.carried[2].extract(u), k.carried[3]);
        const a0 = a_fragments[u].shuffleVector(a_fragments[u], &.{ 0, 1, 2, 3 }).bitcast(.bf16);
        const a1 = a_fragments[u].shuffleVector(a_fragments[u], &.{ 4, 5, 6, 7 }).bitcast(.bf16);
        const w0 = decoded.shuffleVector(decoded, &.{ 0, 1, 2, 3 });
        const w1 = decoded.shuffleVector(decoded, &.{ 4, 5, 6, 7 });
        acc = b.mmaAtomCall(atom, a0, w0, acc);
        acc = b.mmaAtomCall(atom, a1, w1, acc);
        var second = b.openIfElse(second_active, .{zeros.type_()});
        const a0_g1 = a_fragments_g1[u].shuffleVector(a_fragments_g1[u], &.{ 0, 1, 2, 3 }).bitcast(.bf16);
        const a1_g1 = a_fragments_g1[u].shuffleVector(a_fragments_g1[u], &.{ 4, 5, 6, 7 }).bitcast(.bf16);
        const first_acc = b.mmaAtomCall(atom, a0_g1, w0, acc_g1);
        second.yieldThen(.{b.mmaAtomCall(atom, a1_g1, w1, first_acc)});
        second.yieldElse(.{acc_g1});
        acc_g1 = second.results[0];
    }
    // All current reads finish before a later iteration reuses this LDS region.
    b.barrier();
    k.yield(.{ acc, acc_g1, next.results[0], next.results[1] });
    sums.yieldThen(.{ k.results[0], k.results[1] });
    sums.yieldElse(.{ zeros, zeros });

    inline for (0..4) |i| {
        storeRow(b, args.output, args.sorted, block_m, route_start, k_lane.mul(4).add(i), out_n, sums.results[0].extract(i), cfg);
    }
    var store_second = b.openIf(second_active);
    inline for (0..4) |i| {
        storeRow(b, args.output, args.sorted, block_m, route_start, k_lane.mul(4).add(i + 16), out_n, sums.results[1].extract(i), cfg);
    }
    store_second.yieldThen(.{});
    block.yieldThen(.{});
}

fn routeForRow(b: *Builder, sorted: Value, block: Value, start: Value, row: Value, naive: bool) Value {
    return if (naive) row.cmp(.eq, 0).select(block.to(.i32), b.constant(.i32, -1)) else sorted.at(start.add(row));
}

fn storeRow(b: *Builder, output: Value, sorted: Value, block: Value, start: Value, row: Value, col: Value, value: Value, cfg: Config) void {
    const route = routeForRow(b, sorted, block, start, row, cfg.naive != 0);
    var valid = b.openIf(route.cmp(.ge, 0).bitAnd(route.cmp(.lt, cfg.routes)));
    output.set(.{ route.to(.i64), col }, value.to(.bf16));
    valid.yieldThen(.{});
}

fn vectorView(tensor: Value, offset: Value, comptime n: i64) Value {
    return tensor.emitIter().addOffset(offset).view(fly.L(n, 1));
}

const TileLoader = struct {
    a: Value,
    weight: Value,
    scales: Value,
    shared: Value,
    load_a: Value,
    load_b: Value,
    copy_shared: Value,
    a_base: Value,
    a_base_g1: Value,
    a_valid: Value,
    a_valid_g1: Value,
    second_active: Value,
    a_col: Value,
    a_store: Value,
    b_base: Value,
    expert: Value,
    out_n: Value,
    k_lane: Value,

    fn load(self: TileLoader, b: *Builder, k: Value, write_base: Value) struct { words: Value, scale: Value } {
        self.storeActivation(b, self.a_base, self.a_valid, k, self.a_store.add(write_base));
        var second = b.openIf(self.second_active);
        self.storeActivation(b, self.a_base_g1, self.a_valid_g1, k, self.a_store.add(write_base).add(2048));
        second.yieldThen(.{});
        const offset = self.b_base.add(k.divUnsigned(8)).add(self.k_lane.mul(4));
        const words = b.copyAtomLoad(self.load_b, vectorView(self.weight, offset, 4));
        const byte = self.scales.at(.{ self.expert, self.out_n, k.divUnsigned(32).add(self.k_lane) });
        const scale = byte.toUnsigned(.i32);
        const finite = scale.cmp(.eq, 0).select(b.constant(.i32, 0x00400000), scale.shl(23));
        return .{ .words = words, .scale = scale.cmp(.eq, 255).select(b.constant(.i32, 0x7fc00000), finite).bitcast(.f32) };
    }

    fn storeActivation(self: TileLoader, b: *Builder, base: Value, valid: Value, k: Value, destination: Value) void {
        const zeros = b.constant(.i64, 0).splat(2);
        var guarded = b.openIfElse(valid, .{zeros.type_()});
        const offset = base.add(k.divUnsigned(4)).add(self.a_col);
        guarded.yieldThen(.{b.copyAtomLoad(self.load_a, vectorView(self.a, offset, 2))});
        guarded.yieldElse(.{zeros});
        b.copyAtomStore(self.copy_shared, guarded.results[0].bitcast(.i16), vectorView(self.shared, destination, 8));
    }
};

fn decodeWeight(b: *Builder, word: Value, scale: Value) Value {
    const lo = b.constant(.i32, 0x44403800);
    const hi = b.constant(.i32, 0x54504c48);
    const even = b.perm(hi, lo, word.bitAnd(0x07070707));
    const odd = b.perm(hi, lo, word.shrU(4).bitAnd(0x07070707));
    const ev0 = b.cvtPkF32Fp8(even, false);
    const od0 = b.cvtPkF32Fp8(odd, false);
    const ev1 = b.cvtPkF32Fp8(even, true);
    const od1 = b.cvtPkF32Fp8(odd, true);
    const first = ev0.shuffleVector(od0, &.{ 0, 2, 1, 3 });
    const last = ev1.shuffleVector(od1, &.{ 0, 2, 1, 3 });
    const magnitude = first.shuffleVector(last, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    var shifts = b.constant(.i32, 0).splat(8);
    inline for (0..8) |i| shifts = shifts.insert(i, b.constant(.i32, i * 4));
    const signs = word.splat(8).shrU(shifts).bitAnd(8).shl(28);
    // FNUZ has no -0: restore FP4's sign in F32 before scaling and BF16 rounding.
    return magnitude.bitcast(.i32).bitOr(signs).bitcast(.f32).mul(scale).to(.bf16);
}

const Gemv = struct {
    // Packed-word GEMV derived from the supplied moe-mi300x/fly_word.py algorithm.
    // Several rows share a CTA, and the shorter down projection uses subwaves to
    // expose independent outputs without paying for a full wave reduction per row.
    const Config = struct { rows: usize, lanes: usize, unroll: usize };

    /// One local expert ID per original route; negative/out-of-bank IDs write zero.
    fn call(x: Tensor, weight: Tensor, scales: Tensor, ids: Tensor, output: zml.Shape, topk: usize) Tensor {
        const cfg: Gemv.Config = if (weight.dim(1) == 4608)
            .{ .rows = 4, .lanes = 64, .unroll = 1 }
        else
            .{ .rows = 4, .lanes = 32, .unroll = 5 };
        const rows = cfg.rows;
        const lanes = cfg.lanes;
        const unroll = cfg.unroll;
        const n = weight.dim(1);
        const k = x.dim(1);
        const routes: i64 = @intCast(ids.count());
        std.debug.assert(x.dtype() == .bf16 and (weight.dtype() == .u8 or weight.dtype() == .i8));
        std.debug.assert(@mod(n, 8) == 0 and @mod(k, 32) == 0);
        std.debug.assert(topk > 0 and routes == x.dim(0) * @as(i64, @intCast(topk)));
        std.debug.assert(weight.dim(2) * 2 == k);
        std.debug.assert(scales.dim(0) == weight.dim(0) and scales.dim(1) == n and scales.dim(2) == @divExact(k, 32));
        const tile_rows = rows * (64 / lanes);
        std.debug.assert(@mod(n, @as(i64, @intCast(tile_rows * 8))) == 0);
        const fields: EmitConfig = .{ .unroll = unroll, .rows = tile_rows, .lanes = lanes, .stride = unroll * lanes, .ntiles = @divExact(n, @as(i64, @intCast(tile_rows))), .tokens = x.dim(0), .experts = weight.dim(0), .n = n, .k_pairs = @divExact(k, 2), .k_words = @divExact(k, 8), .k_scales = @divExact(k, 32), .routes = routes, .grid = @divExact(n, @as(i64, @intCast(tile_rows))) * routes, .topk = @as(i64, @intCast(topk)) };
        const a = x.reshape(.{ .token = x.dim(0), .in = @divExact(k, 2), .pair = 2 }).bitCast(.i32);
        const b = weight.reshape(.{ .expert = weight.dim(0), .out = n, .in = @divExact(weight.dim(2), 4), .word = 4 }).bitCast(.i32);
        const matrix_output = zml.Shape.init(.{ .token = routes, .out = n }, output.dtype());
        return Gemv.Kernel.call(.{ .a = a, .b = b, .scales = scales.bitCast(.u8), .ids = ids.reshape(.{routes}) }, .{
            .output = matrix_output,
        }, .{
            .cfg = fields,
            .threads = @intCast(rows * 64),
            .grid = .{ @intCast(fields.grid), 1, 1 },
        }).output.reshape(output);
    }

    const EmitConfig = struct {
        unroll: usize,
        rows: usize,
        lanes: usize,
        stride: usize,
        ntiles: i64,
        tokens: i64,
        experts: i64,
        n: i64,
        k_pairs: i64,
        k_words: i64,
        k_scales: i64,
        routes: i64,
        grid: i64,
        topk: i64,
    };

    const Kernel = zml.kernel.fly.Kernel(EmitConfig, .{
        .name = "mxfp4_gemv",
        .inputs = &.{ "a", "b", "scales", "ids" },
        .outputs = &.{"output"},
        .run = Gemv.run,
    });

    fn run(b: *Builder, cfg: EmitConfig) fly.FinishError!void {
        const args = Gemv.Kernel.args(b);
        const block = b.blockId(.x).to(.i64);
        const thread = b.threadId(.x).to(.i64);
        const lane = thread.remUnsigned(cfg.lanes);
        const rowgroup = thread.divUnsigned(cfg.lanes);
        // Preserve the eight-XCD workgroup permutation and original subwave layout.
        const pid = block.remUnsigned(8).mul(@divExact(cfg.grid, 8)).add(block.divUnsigned(8));
        const row = pid.remUnsigned(cfg.ntiles).mul(cfg.rows).add(rowgroup);
        const route = pid.divUnsigned(cfg.ntiles);
        const expert32 = args.ids.at(route);
        const expert = expert32.to(.i64);
        const token = route.divUnsigned(cfg.topk);
        const local = expert32.cmp(.ge, 0).bitAnd(expert32.cmp(.lt, cfg.experts));
        const inrow = row.cmpUnsigned(.lt, cfg.n);
        const fzero = b.constant(.f32, 0.0);
        var sum = b.openIfElse(local.bitAnd(inrow), .{fzero.type_()});
        var k = b.openFor(lane, cfg.k_words, cfg.stride, .{fzero});
        var carried = k.carried[0];
        for (0..cfg.unroll) |u| {
            const pos = k.iv.add(u * cfg.lanes);
            var inbounds = b.openIfElse(pos.cmpUnsigned(.lt, cfg.k_words), .{fzero.type_()});
            const word = args.b.at(.{ expert, row, pos });
            const scale8 = args.scales.at(.{ expert, row, pos.divUnsigned(4) });
            const scale = decodeScale(b, scale8);
            const low = word.bitAnd(0x07070707);
            const high = word.shrU(4).bitAnd(0x07070707);
            const even_bits = b.perm(b.constant(.i32, 0xd4d0ccc8), b.constant(.i32, 0xc4c0b800), low);
            const odd_bits = b.perm(b.constant(.i32, 0xd4d0ccc8), b.constant(.i32, 0xc4c0b800), high);
            const even = even_bits.bitAnd(word.shl(4).bitOr(0x7f7f7f7f));
            const odd = odd_bits.bitAnd(word.bitOr(0x7f7f7f7f));
            // Retain the scalar decoder and its FP32 scale -> BF16 -> FP32 boundary.
            const weights = [_]Value{
                b.cvtPkF32Fp8(even, false).mul(scale).to(.bf16).to(.f32),
                b.cvtPkF32Fp8(odd, false).mul(scale).to(.bf16).to(.f32),
                b.cvtPkF32Fp8(even, true).mul(scale).to(.bf16).to(.f32),
                b.cvtPkF32Fp8(odd, true).mul(scale).to(.bf16).to(.f32),
            };
            var partial = fzero;
            inline for (0..4) |pair| {
                const x = args.a.at(.{ token, pos.mul(4).add(pair) });
                const xlo = x.to(.i16).bitcast(.bf16).to(.f32);
                const xhi = x.shrU(16).to(.i16).bitcast(.bf16).to(.f32);
                const wlo = weights[(pair / 2) * 2].extract(pair % 2);
                const whi = weights[(pair / 2) * 2 + 1].extract(pair % 2);
                // Odd then even FMA, followed by one add to the K-loop accumulator.
                partial = b.fma(xlo, wlo, b.fma(xhi, whi, partial));
            }
            inbounds.yieldThen(.{partial.add(carried)});
            inbounds.yieldElse(.{carried});
            carried = inbounds.results[0];
        }
        k.yield(.{carried});
        var reduced = k.results[0];
        var shift: i32 = 1;
        while (shift < @as(i32, @intCast(cfg.lanes))) : (shift *= 2) {
            reduced = reduced.add(reduced.shuffleXor(shift, 64));
        }
        sum.yieldThen(.{reduced});
        sum.yieldElse(.{fzero});
        var store = b.openIf(lane.cmp(.eq, 0).bitAnd(inrow));
        args.output.set(.{ route, row }, sum.results[0].to(.bf16));
        store.yieldThen(.{});
    }

    fn decodeScale(b: *Builder, byte: Value) Value {
        const scale = byte.toUnsigned(.i32);
        const finite = scale.cmp(.eq, 0).select(b.constant(.i32, 0x00400000), scale.shl(23));
        return scale.cmp(.eq, 255).select(b.constant(.i32, 0x7fc00000), finite).bitcast(.f32);
    }
};
