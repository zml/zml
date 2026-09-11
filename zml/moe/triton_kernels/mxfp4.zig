//! FP4 x FP8 MoE emitted through the ZML Triton DSL.
//! BF16 activations use per-32 FP8 scaling; weights retain per-32 E8M0 scales.
//! Gate/up weight and scale rows are interleaved gate-first.
const std = @import("std");
const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;
const B = tri.Builder;
const V = tri.Value;

pub const Config = struct {
    tokens: i64,
    hidden: i64,
    intermediate: i64,
    experts: i64,
    global_experts: i64,
    topk: i64,
    expert_offset: i64 = 0,
    // Swiglu clamping. Set to 0 to disable.
    swiglu_limit: f32,
    /// Scale SwiGLU before FP8 quantization, or scale the down result before reduction.
    routing_weight_placement: zml.moe.triton.RoutingWeightPlacement = .before_down,

    pub fn validate(c: Config) !void {
        if (c.tokens <= 0 or
            c.experts <= 0 or
            c.topk <= 0 or c.topk > c.global_experts or
            c.expert_offset < 0 or c.expert_offset + c.experts > c.global_experts)
        {
            return error.InvalidRoutingShape;
        }

        if (c.hidden < 128 or @mod(c.hidden, 128) != 0 or
            c.intermediate < 128 or @mod(c.intermediate, 128) != 0)
        {
            return error.InvalidHiddenShape;
        }

        if (!std.math.isFinite(c.swiglu_limit) or c.swiglu_limit < 0) {
            return error.InvalidScale;
        }
    }
};

pub const Inputs = struct {
    x: zml.Tensor,
    ids: zml.Tensor,
    scales: zml.Tensor,
    w1: zml.Tensor,
    s1: zml.Tensor,
    w2: zml.Tensor,
    s2: zml.Tensor,
};

pub fn validateInputs(c: Config, a: Inputs) !void {
    if (a.s1.rank() != 3 or a.s2.rank() != 3) return error.InvalidInputShape;
    if (a.s1.dim(2) < @divTrunc(c.hidden, 32) or
        a.s2.dim(2) < @divTrunc(c.intermediate, 32)) return error.InvalidInputShape;
    const shapes = .{
        .x = zml.Shape.init(.{ c.tokens, c.hidden }, .bf16),
        .ids = zml.Shape.init(.{ c.tokens, c.topk }, .i32),
        .scales = zml.Shape.init(.{ c.tokens, c.topk }, .f32),
        .w1 = zml.Shape.init(.{ c.experts, 2 * c.intermediate, @divTrunc(c.hidden, 2) }, .u8),
        .s1 = zml.Shape.init(.{ c.experts, 2 * c.intermediate, a.s1.dim(2) }, .f8e8m0),
        .w2 = zml.Shape.init(.{ c.experts, c.hidden, @divTrunc(c.intermediate, 2) }, .u8),
        .s2 = zml.Shape.init(.{ c.experts, c.hidden, a.s2.dim(2) }, .f8e8m0),
    };
    inline for (std.meta.fields(Inputs)) |field| {
        if (!@field(a, field.name).shape().eql(@field(shapes, field.name))) return error.InvalidInputShape;
    }
}

const bn = 16;
const bk = 512;

/// Computes an upper bound on the number of expert tiles needed.
/// When there is only one token, reserve a single tile per selected expert.
fn tiles(c: Config) i64 {
    if (c.tokens == 1) return c.topk;
    return @min(c.tokens * c.topk, @divTrunc(c.tokens * c.topk + c.experts * (bn - 1) + bn - 1, bn));
}

/// Returns the next power of two.
fn pow2(x: i64) i64 {
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(x)) catch unreachable);
}

fn quantRows(c: Config) i64 {
    return if (c.tokens >= 16) 16 else 1;
}

fn combineBlock(c: Config) i64 {
    // Keep at least 512 CTAs before increasing per-CTA reduction work.
    const elements = c.tokens * c.hidden;
    if (c.tokens > 1 and elements >= 512 * 2048) return 2048;
    if (c.tokens > 1 and elements >= 512 * 1024) return 1024;
    return 256;
}

fn ar(b: *B, n: i64) V {
    return b.arange(0, n, .i32);
}

fn ci(b: *B, n: i64) V {
    return b.liftAs(n, .i32);
}

fn ld(b: *B, p: V, mask: V, dt: tri.DType, other: i64) V {
    return b.loadOpts(p, .{ .mask = mask, .other = if (p.isTensor()) b.full(p.shape().constSlice(), other, dt) else b.liftAs(other, dt) });
}

// Scale rows can be tightly packed or padded. Specialize the GEMM on their
// physical width without imposing an alignment requirement on the tensor.
const GemmConfig = struct {
    moe: Config,
    scale_stride: i64,
};

const Quant = tri.Kernel(Config, .{ .name = "mxfp4_triton_quant", .inputs = &.{"x"}, .outputs = &.{ "q", "s" }, .run = quantInput });
const Route = tri.Kernel(Config, .{ .name = "mxfp4_triton_route", .inputs = &.{"ids"}, .outputs = &.{ "counts", "pos" }, .run = route });
const Schedule = tri.Kernel(Config, .{ .name = "mxfp4_triton_schedule", .inputs = &.{ "ids", "counts", "pos" }, .outputs = &.{ "map", "perm", "sched" }, .run = schedule });
const GateUp = tri.Kernel(GemmConfig, .{ .name = "mxfp4_triton_up", .inputs = &.{ "q", "s", "w", "ws", "ids", "rw", "map", "sched" }, .outputs = &.{ "mid", "ms" }, .run = gateUp });
const Down = tri.Kernel(GemmConfig, .{ .name = "mxfp4_triton_down", .inputs = &.{ "mid", "ms", "w", "ws", "ids", "sched" }, .outputs = &.{"d"}, .run = down });
const Combine = tri.Kernel(Config, .{ .name = "mxfp4_triton_combine", .inputs = &.{ "d", "ids", "perm", "rw" }, .outputs = &.{"y"}, .run = combine });

pub fn forward(c: Config, a: Inputs) zml.Tensor {
    c.validate() catch @panic("Invalid MXFP4 MoE configuration");
    validateInputs(c, a) catch @panic("Invalid MXFP4 tensor layout");

    const tile_count = tiles(c);

    // Quantize activations
    const q = Quant.call(
        .{ .x = a.x },
        .{
            .q = .init(.{ c.tokens, c.hidden }, .f8e4m3fn),
            .s = .init(.{ c.tokens, @divTrunc(c.hidden, 32) }, .u8),
        },
        .{
            .cfg = c,
            .grid = .{ @intCast(@divTrunc(c.tokens + quantRows(c) - 1, quantRows(c))), @intCast(@divTrunc(c.hidden, 128)), 1 },
            .num_warps = 4,
            .num_stages = 1,
        },
    );

    const dummy = zml.Tensor.scalar(0, .i32);
    var map = dummy;
    var perm = dummy;
    var sched = dummy;

    if (c.tokens != 1) {
        const r = Route.call(
            .{ .ids = a.ids },
            .{ .counts = .init(.{c.experts}, .i32), .pos = .init(.{c.tokens * c.topk}, .i32) },
            .{ .cfg = c, .grid = .{ @intCast(c.experts), 1, 1 }, .num_warps = 4, .num_stages = 1 },
        );

        const s = Schedule.call(
            .{ .ids = a.ids, .counts = r.counts, .pos = r.pos },
            .{ .map = .init(.{tile_count * bn}, .i32), .perm = .init(.{c.tokens * c.topk}, .i32), .sched = .init(.{2 * tile_count + 1}, .i32) },
            .{
                .cfg = c,
                .grid = .{ @intCast(@divTrunc(c.tokens * c.topk + 127, 128)), 1, 1 },
                .num_warps = 4,
                .num_stages = 1,
            },
        );

        map = s.map;
        perm = s.perm;
        sched = s.sched;
    }

    // Each narrow tile produces 64 activations: two complete scale groups.
    const up_columns: i64 = if (c.tokens <= 8) 64 else 128;
    const m = GateUp.call(
        .{ .q = q.q, .s = q.s, .w = a.w1, .ws = a.s1.bitCast(.u8), .ids = a.ids, .rw = a.scales, .map = map, .sched = sched },
        .{ .mid = .init(.{ tile_count * bn, c.intermediate }, .f8e4m3fn), .ms = .init(.{ tile_count * bn, @divTrunc(c.intermediate, 32) }, .u8) },
        .{
            .cfg = .{ .moe = c, .scale_stride = a.s1.dim(2) },
            .grid = .{ @intCast(@divTrunc(c.intermediate, up_columns)), @intCast(tile_count), 1 },
            .num_warps = if (c.tokens <= 8) 4 else 8,
            .num_stages = 2,
            .global_scratch_memory_size = @intCast(128 * @divTrunc(c.intermediate, up_columns) * tile_count),
        },
    );

    const d = Down.call(
        .{ .mid = m.mid, .ms = m.ms, .w = a.w2, .ws = a.s2.bitCast(.u8), .ids = a.ids, .sched = sched },
        .{ .d = .init(.{ tile_count * bn, c.hidden }, .bf16) },
        .{
            .cfg = .{ .moe = c, .scale_stride = a.s2.dim(2) },
            .grid = .{ @intCast(@divTrunc(c.hidden, 128)), @intCast(tile_count), 1 },
            .num_warps = if (c.tokens <= 8) 4 else 8,
            .num_stages = 2,
            .global_scratch_memory_size = @intCast(128 * @divTrunc(c.hidden, 128) * tile_count),
        },
    );

    return Combine.call(
        .{ .d = d.d, .ids = a.ids, .perm = perm, .rw = a.scales },
        .{ .y = a.x.shape() },
        .{
            .cfg = c,
            .grid = .{ @intCast(c.tokens), @intCast(@divTrunc(c.hidden + combineBlock(c) - 1, combineBlock(c))), 1 },
            .num_warps = 4,
            .num_stages = 1,
        },
    ).y;
}

fn quant32(b: *B, x: V) struct { q: V, s: V } {
    const rows = x.shape().constSlice()[0];
    const cols = x.shape().constSlice()[1];
    const groups = @divExact(cols, 32);
    const blocks = b.reshape(x, &.{ rows * groups, 32 });
    const sf = b.exp2(b.ceil(b.log2(b.maxOpts(b.absf(blocks), .{ .axis = 1 }).maximum(1e-4).mul(1.0 / 448.0))));
    const q = blocks.div(sf.expandDims(1)).maximum(-448.0).minimum(448.0).to(.f8e4m3fn);
    const bits = b.shrui(b.bitcast(sf, .i32), b.full(sf.shape().constSlice(), 23, .i32)).bitAnd(255).to(.i8);
    return .{ .q = b.reshape(q, &.{ rows, cols }), .s = b.reshape(bits, &.{ rows, groups }) };
}

fn quantInput(b: *B, c: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{ .x = .{ .ptr = .bf16 }, .q = .{ .ptr = .f8e4m3fn }, .s = .{ .ptr = .i8 } });
    const rows = b.programId(.x).mul(quantRows(c)).add(ar(b, quantRows(c)));
    const block = b.programId(.y);
    const cols = block.mul(128).add(ar(b, 128));
    const offsets = rows.expandDims(1).mul(c.hidden).add(cols.expandDims(0));
    const mask = rows.lt(c.tokens).expandDims(1);
    const x = ld(b, a.x.addPtr(offsets), mask, .bf16, 0).to(.f32);
    const q = quant32(b, x);
    b.storeOpts(a.q.addPtr(offsets), q.q, .{ .mask = mask });
    const scale_offsets = rows.expandDims(1).mul(@divTrunc(c.hidden, 32)).add(block.mul(4).add(ar(b, 4)).expandDims(0));
    b.storeOpts(a.s.addPtr(scale_offsets), q.s, .{ .mask = mask });
}

fn route(b: *B, c: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{ .ids = .{ .ptr = .i32 }, .counts = .{ .ptr = .i32 }, .pos = .{ .ptr = .i32 } });
    const e = b.programId(.x);
    const p = ar(b, pow2(c.tokens * c.topk));
    const ids = ld(b, a.ids.addPtr(p), p.lt(c.tokens * c.topk), .i32, -1);
    const valid = p.lt(c.tokens * c.topk).bitAnd(ids.eq(e.add(c.expert_offset)));
    b.storeOpts(a.pos.addPtr(p), valid.to(.i32).cumsum().sub(1), .{ .mask = valid });
    b.store(a.counts.addPtr(e), valid.to(.i32).sum());
}

fn schedule(b: *B, c: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{ .ids = .{ .ptr = .i32 }, .counts = .{ .ptr = .i32 }, .pos = .{ .ptr = .i32 }, .map = .{ .ptr = .i32 }, .perm = .{ .ptr = .i32 }, .sched = .{ .ptr = .i32 } });
    const nt = tiles(c);
    const p = b.programId(.x).mul(128).add(ar(b, 128));
    const e = ld(b, a.ids.addPtr(p), p.lt(c.tokens * c.topk), .i32, -1).sub(ci(b, c.expert_offset));
    const es = ar(b, pow2(c.experts));
    const counts = ld(b, a.counts.addPtr(es), es.lt(c.experts), .i32, 0);
    const ts = counts.cdiv(bn);
    const starts = ts.cumsum().sub(ts);
    var first = b.openIf(b.programId(.x).eq(0));
    b.store(a.sched.addPtr(2 * nt), ts.sum());
    first.yieldThen(.{});
    const live = p.lt(c.tokens * c.topk).bitAnd(e.ge(0)).bitAnd(e.lt(c.experts));
    const safe = e.maximum(0).minimum(pow2(c.experts) - 1);
    const pos = ld(b, a.pos.addPtr(p), live, .i32, 0);
    const start = starts.gather(safe, 0);
    const tile = start.add(pos.div(bn));
    const row = pos.rem(bn);
    b.storeOpts(a.perm.addPtr(p), b.select(live, tile.mul(bn).add(row), b.full(&.{128}, -1, .i32)), .{ .mask = p.lt(c.tokens * c.topk) });
    b.storeOpts(a.map.addPtr(tile.mul(bn).add(row)), p, .{ .mask = live });
    b.storeOpts(a.sched.addPtr(tile), e, .{ .mask = live.bitAnd(row.eq(0)) });
    b.storeOpts(a.sched.addPtr(nt).addPtr(tile), counts.gather(safe, 0).sub(pos).minimum(bn), .{ .mask = live.bitAnd(row.eq(0)) });
}

fn gateUp(b: *B, c: GemmConfig) tri.FinishError!void {
    return gemm(b, c.moe, .gate_up, c.scale_stride);
}

fn down(b: *B, c: GemmConfig) tri.FinishError!void {
    return gemm(b, c.moe, .down, c.scale_stride);
}

const Projection = enum { gate_up, down };

fn gemm(b: *B, c: Config, comptime projection: Projection, scale_stride: i64) tri.FinishError!void {
    const is_up = projection == .gate_up;
    const a = if (is_up) try b.declareArgs(.{ .q = .{ .ptr = .f8e4m3fn }, .s = .{ .ptr = .i8 }, .w = .{ .ptr = .i8 }, .ws = .{ .ptr = .i8 }, .ids = .{ .ptr = .i32 }, .rw = .{ .ptr = .f32 }, .map = .{ .ptr = .i32 }, .sched = .{ .ptr = .i32 }, .mid = .{ .ptr = .f8e4m3fn }, .ms = .{ .ptr = .i8 } }) else try b.declareArgs(.{ .q = .{ .ptr = .f8e4m3fn }, .s = .{ .ptr = .i8 }, .w = .{ .ptr = .i8 }, .ws = .{ .ptr = .i8 }, .ids = .{ .ptr = .i32 }, .sched = .{ .ptr = .i32 }, .d = .{ .ptr = .bf16 } });
    const block = b.programId(.x);
    const tile = b.programId(.y);
    const n = ar(b, bn);
    const nt = tiles(c);
    const expert = if (c.tokens == 1) b.load(a.ids.addPtr(tile)).sub(ci(b, c.expert_offset)) else ld(b, a.sched.addPtr(tile), tile.lt(b.load(a.sched.addPtr(2 * nt))), .i32, 0);
    const valid = if (c.tokens == 1) expert.ge(0).bitAnd(expert.lt(c.experts)) else tile.lt(b.load(a.sched.addPtr(2 * nt)));
    var live = b.openIf(valid);
    const count = if (c.tokens == 1) ci(b, 1) else b.load(a.sched.addPtr(nt).addPtr(tile));
    const pair = if (c.tokens == 1) b.full(&.{bn}, 0, .i32).add(tile) else if (is_up) ld(b, a.map.addPtr(tile.mul(bn).add(n)), n.lt(count), .i32, 0) else n;
    const width: i64 = if (is_up and c.tokens > 8) 256 else 128;
    const ksize = if (is_up) c.hidden else c.intermediate;
    const rows = if (is_up) 2 * c.intermediate else c.hidden;
    const m = block.mul(width).add(ar(b, width));
    const k = ar(b, bk);
    const ks = ar(b, bk / 32);
    const row = if (is_up) pair.div(c.topk) else tile.mul(bn).add(n);
    const wd = b.makeTensorDescriptor(a.w.addPtr(expert.to(.i64).mul(@divTrunc(rows * ksize, 2))), &.{ ci(b, rows), ci(b, @divTrunc(ksize, 2)) }, &.{ b.liftAs(@divTrunc(ksize, 2), .i64), b.liftAs(1, .i64) }, .pad_zero, &.{ width, bk / 2 }, .i8);
    var loop = b.openFor(0, @divTrunc(ksize + bk - 1, bk), 1, .{b.zeros(&.{ width, bn }, .f32)});
    const sk = loop.iv.mul(bk / 32).add(ks);
    const kk = loop.iv.mul(bk).add(k);
    const w = b.descriptorLoad(wd, &.{ block.mul(ci(b, width)), loop.iv.mul(bk / 2) }, &.{ width, bk / 2 }, .i8);
    const scale_offset = m.expandDims(1).mul(scale_stride).add(sk.expandDims(0));
    const ws = ld(b, a.ws.addPtr(expert.to(.i64).mul(rows * scale_stride)).addPtr(scale_offset), sk.expandDims(0).lt(@divTrunc(ksize, 32)), .i8, 127);
    var mask = kk.expandDims(0).lt(ksize);
    var smask = sk.expandDims(0).lt(@divTrunc(ksize, 32));
    // The small-batch epilogue leaves padded rows unwritten.
    if (is_up or c.tokens <= 8) {
        mask = mask.bitAnd(n.lt(count).expandDims(1));
        smask = smask.bitAnd(n.lt(count).expandDims(1));
    }
    const x = ld(b, a.q.addPtr(row.expandDims(1).mul(ksize).add(kk.expandDims(0))), mask, .f8e4m3fn, 0);
    const xs = ld(b, a.s.addPtr(row.expandDims(1).mul(@divTrunc(ksize, 32)).add(sk.expandDims(0))), smask, .i8, 127);
    loop.yield(.{b.dotScaled(w, b.permute(x, &.{ 1, 0 }), loop.carried[0], ws, xs, .e2m1, .e4m3)});
    const epilogue_n: i64 = if (c.tokens == 1) 1 else bn;
    const out_n = ar(b, epilogue_n);
    const acc = if (c.tokens == 1) live_row: {
        // The remaining MMA columns are padding during single-token decode.
        // Extract column zero before activation/reduction and output stores.
        var v = loop.results[0];
        var cols: i64 = bn;
        while (cols > 1) : (cols = @divTrunc(cols, 2)) {
            v = b.split(b.reshape(v, &.{ width, @divTrunc(cols, 2), 2 }))[0];
        }
        break :live_row b.reshape(v, &.{ 1, width }).to(.bf16);
    } else b.permute(loop.results[0], &.{ 1, 0 }).to(.bf16);
    if (is_up) {
        const columns = @divExact(width, 2);
        const halves = b.split(b.reshape(acc.to(.f32), &.{ epilogue_n, columns, 2 }));
        var g = halves[0];
        var u = halves[1];
        if (c.swiglu_limit > 0) {
            g = g.minimum(c.swiglu_limit);
            u = u.minimum(c.swiglu_limit).maximum(-c.swiglu_limit);
        }
        const ex = b.externElementwise(&.{g.mul(-1.0)}, &.{ epilogue_n, columns }, .f32, "libdevice", "", "__nv_expf");
        var mid = u.mul(b.divRn(g, ex.add(1.0)));
        if (c.routing_weight_placement == .before_down) {
            const rw = if (c.tokens == 1) b.load(a.rw.addPtr(tile)) else ld(b, a.rw.addPtr(pair), n.lt(count), .f32, 0).expandDims(1);
            mid = mid.mul(rw);
        }
        // Compute scales and quantize directly from the FP32 SwiGLU result.
        const q = quant32(b, mid);
        const col = block.mul(columns).add(ar(b, columns));
        b.store(a.mid.addPtr(tile.mul(bn).add(out_n).expandDims(1).mul(c.intermediate).add(col.expandDims(0))), q.q);
        const scale_cols = block.mul(@divExact(columns, 32)).add(ar(b, @divExact(columns, 32)));
        b.store(a.ms.addPtr(tile.mul(bn).add(out_n).expandDims(1).mul(@divTrunc(c.intermediate, 32)).add(scale_cols.expandDims(0))), q.s);
    } else b.store(a.d.addPtr(tile.mul(bn).add(out_n).expandDims(1).mul(c.hidden).add(m.expandDims(0))), acc);
    live.yieldThen(.{});
}

fn combine(b: *B, c: Config) tri.FinishError!void {
    return combineWithBn(b, c, bn);
}

fn combineWithBn(b: *B, c: Config, tile_n: i64) tri.FinishError!void {
    const a = try b.declareArgs(.{ .d = .{ .ptr = .bf16 }, .ids = .{ .ptr = .i32 }, .perm = .{ .ptr = .i32 }, .rw = .{ .ptr = .f32 }, .y = .{ .ptr = .bf16 } });
    const t = b.programId(.x);
    const block = combineBlock(c);
    const h = b.programId(.y).mul(block).add(ar(b, block));
    var acc = b.zeros(&.{block}, .f32);
    for (0..@intCast(c.topk)) |s| {
        const pair = t.mul(c.topk).add(@as(i64, @intCast(s)));
        const row = if (c.tokens == 1) pair.mul(tile_n) else b.load(a.perm.addPtr(pair));
        const valid = if (c.tokens == 1) v: {
            const e = b.load(a.ids.addPtr(pair)).sub(ci(b, c.expert_offset));
            break :v e.ge(0).bitAnd(e.lt(c.experts));
        } else row.ge(0);
        var value = ld(b, a.d.addPtr(row.mul(c.hidden).add(h)), valid.bitAnd(h.lt(c.hidden)), .bf16, 0).to(.f32);
        if (c.routing_weight_placement == .after_down) {
            value = value.mul(ld(b, a.rw.addPtr(pair), valid, .f32, 0));
        }
        acc = acc.add(value);
    }
    b.storeOpts(a.y.addPtr(t.mul(c.hidden).add(h)), acc.to(.bf16), .{ .mask = h.lt(c.hidden) });
}
