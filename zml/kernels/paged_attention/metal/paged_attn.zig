//! Metal paged attention kernels: fa_vec_paged and its split-K reduce for
//! decode, and paged_attention_tiled with zero_padding_rows otherwise.

const std = @import("std");

const metal = @import("zml/core").kernel.metal;

const ArgSpec = metal.ArgSpec;
const Builder = metal.Builder;
const Type = metal.Type;
const Value = metal.Value;

const flt_max = std.math.floatMax(f32);
/// A masked score: below any real one yet finite, since fast math assumes
/// no infinities. Halved so subtracting a running maximum cannot overflow.
const masked_score = -flt_max / 2.0;
const max_half = 65504.0;
const simd_width = 32;

/// fa_vec_paged: bf16 decode over paged caches. Simdgroup s takes query head
/// s % heads of the threadgroup's KV head group and position split s / heads.
pub const FaVecPaged = struct {
    pub const Cfg = struct {
        /// The key and value head dimension.
        head_dim: i32,
        /// The page size.
        block_size: i32,
        /// The in-page position stride, n_kv * head_dim elements.
        position_stride: i32,
        heads: i32,
        splits: i32,
    };

    pub const Args: metal.Struct = .{ .name = "fa_vec_paged_args", .fields = &.{
        .{ .name = "num_heads", .type = .int },
        .{ .name = "n_kv", .type = .int },
        .{ .name = "max_blocks", .type = .int },
        .{ .name = "num_seqs", .type = .int },
        .{ .name = "scale", .type = .float },
        .{ .name = "sliding_window", .type = .int },
    } };

    pub const HostArgs = extern struct {
        num_heads: i32,
        n_kv: i32,
        max_blocks: i32,
        num_seqs: i32,
        scale: f32,
        /// Below 0 for global attention.
        sliding_window: i32,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_vec_paged",
        .inputs = &.{ "q", "k_cache", "v_cache", "block_tables", "seq_lens", "cu_seqlens_q" },
        .outputs = &.{"dst"},
        .constants = &.{"args"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        switch (cfg.head_dim) {
            32, 64, 128, 256, 512 => {},
            else => std.debug.panic("fa_vec_paged: head_dim {d} unsupported (32, 64, 128, 256 or 512)", .{cfg.head_dim}),
        }
        const char_in: ArgSpec = .{ .buffer = .{ .type = .char, .const_ = true } };
        const int_in: ArgSpec = .{ .buffer = .{ .type = .int, .const_ = true } };
        const a = try b.declareArgs(.{
            .args = .{ .constant_struct = Args },
            .q = char_in,
            .k_cache = char_in,
            .v_cache = char_in,
            .block_tables = int_in,
            .seq_lens = int_in,
            .cu_seqlens_q = int_in,
            .dst = .{ .buffer = .{ .type = .char } },
            .tgpig = .{ .threadgroup_position_in_grid = .uint3 },
            .tgpg = .{ .threadgroups_per_grid = .uint3 },
            .tiisg = .{ .thread_index_in_simdgroup = .ushort },
            .sgitg = .{ .simdgroup_index_in_threadgroup = .ushort },
            .nsg = .{ .simdgroups_per_threadgroup = .ushort },
        });
        impl(b, cfg, a);
    }

    fn impl(b: *Builder, cfg: Cfg, a: anytype) void {
        const D = cfg.head_dim;
        const D4 = @divExact(D, 4);
        // A K/V row spans NL lanes of NI float4s; a simdgroup reads R rows per step.
        const NL = @min(D4, simd_width);
        const R = @divExact(simd_width, NL);
        const NI = @divExact(D4, NL);
        const C = simd_width;
        const BS = cfg.block_size;
        const PS = cfg.position_stride;
        const HG = cfg.heads;
        const NWG = cfg.splits;
        const full: metal.LoopOpts = .{ .unroll = .full };

        const num_heads = a.args.field("num_heads");
        const group = num_heads.div(a.args.field("n_kv"));
        const groups = group.div(HG);
        const r = a.tgpig.at(2).to(.int);
        const tg_head = a.tgpig.at(1).to(.int);
        const ikv = tg_head.div(groups);
        const iq2 = ikv.mul(group).add(tg_head.rem(groups).mul(HG)).add(a.sgitg.rem(HG).to(.int));
        const iwg = a.tgpig.at(0).to(.int).mul(a.nsg.div(HG).to(.int)).add(a.sgitg.div(HG).to(.int));
        const rid = r.to(.long).mul(num_heads.to(.uint)).add(iq2);
        const nrows = a.tgpg.at(2).to(.long).mul(num_heads.to(.uint));

        // Padding rows: FaVecPagedReduce zeroes them when there are splits.
        var padding = b.openIf(r.ge(a.cu_seqlens_q.load(a.args.field("num_seqs"))));
        {
            if (NWG == 1) {
                const dst4 = a.dst.to(.bfloat4).add(rid.mul(D4));
                var loop = b.openFor(a.tiisg.to(.short), D4, simd_width);
                {
                    dst4.store(loop.iv, b.constant(.float4, 0).to(.bfloat4));
                    loop.close();
                }
            }
            b.return_();
            padding.close();
        }

        const seq = sequenceOf(b, a, r);
        const kv_len = seq.kv_len;
        const kv_lo = seq.kv_lo;
        const ic0_lo = kv_lo.div(C);
        const live = if (NWG > 1) liveSplits(b, seq, NWG) else b.lift(@as(i32, 1));

        if (NWG > 1) {
            var idle = b.openIf(iwg.ge(live));
            {
                b.return_();
                idle.close();
            }
        }

        const pt = a.block_tables.add(seq.idx.to(.ulong).mul(a.args.field("max_blocks").to(.uint)));
        const k_cache = a.k_cache.add(ikv.to(.ulong).mul(D).mul(2));
        const v_cache = a.v_cache.add(ikv.to(.ulong).mul(D).mul(2));

        const lane = if (R == 1) a.tiisg else a.tiisg.rem(NL);
        const sub = a.tiisg.div(NL).to(.int);
        const q4 = a.q.add(rid.mul(D).mul(2)).to(.bfloat4).add(lane);
        const qr = b.array(.float4, @intCast(NI));
        const o = b.array(.float4, @intCast(NI));
        var init = b.openForOpts(b.constant(.short, 0), NI, 1, full);
        {
            const ii = init.iv;
            qr.store(ii, q4.load(ii.mul(NL)).to(.float4));
            o.store(ii, b.constant(.float4, 0));
            init.close();
        }

        const S = b.local(.float, 0.0);
        const M = b.local(.float, -flt_max / 2.0);

        const ic0 = b.local(.int, ic0_lo.add(iwg));
        var blocks = b.openLoop(.{});
        {
            const ic = ic0.mul(C);
            var past = b.openIf(ic.ge(kv_len));
            {
                blocks.breakLoop();
                past.close();
            }

            const s = b.local(.float, 0.0);
            var qk = b.openForOpts(b.constant(.short, 0), @divExact(C, R), 1, full);
            {
                const cc = qk.iv;
                const p = b.min(if (R == 1) ic.add(cc) else ic.add(cc.to(.int).mul(R)).add(sub), kv_len.sub(1));
                const pk4 = pagedRow(b, k_cache, pt, p, BS, PS).to(.bfloat4).add(lane);
                const d = b.local(.float, 0.0);
                var dots = b.openForOpts(b.constant(.short, 0), NI, 1, full);
                {
                    const ii = dots.iv;
                    d.assign(d.add(b.dot(pk4.load(ii.mul(NL)).to(.float4), qr.load(ii))));
                    dots.close();
                }
                if (R == 1) {
                    s.assign(b.ternary(a.tiisg.to(.short).eq(cc), b.simdSum(d.rvalue()), s.rvalue()));
                } else {
                    var x = d.rvalue();
                    var mask: i32 = 1;
                    while (mask < NL) : (mask *= 2) x = x.add(b.simdShuffleXor(x, b.constant(.ushort, mask)));
                    const t = b.simdShuffle(x, a.tiisg.rem(R).mul(NL));
                    s.assign(b.ternary(a.tiisg.div(R).to(.short).eq(cc), t, s.rvalue()));
                }
                qk.close();
            }

            const pos = ic.add(a.tiisg.to(.int));
            const in_kv = pos.lt(kv_len).bitAnd(pos.ge(kv_lo));
            const score = b.ternary(in_kv, s.mul(a.args.field("scale")), b.constant(.float, -max_half));

            const m = M.rvalue();
            M.assign(b.simdMax(b.max(M.rvalue(), score)));
            const ms = b.exp(m.sub(M));
            const vs = b.exp(score.sub(M));
            S.assign(S.mul(ms).add(b.simdSum(vs)));

            var rescale = b.openForOpts(b.constant(.short, 0), NI, 1, full);
            {
                o.store(rescale.iv, o.load(rescale.iv).mul(ms));
                rescale.close();
            }

            var pv = b.openForOpts(b.constant(.short, 0), @divExact(C, R), 1, full);
            {
                const cc = pv.iv;
                const row = cc.to(.int).mul(R).add(sub);
                const p = b.min(if (R == 1) ic.add(cc) else ic.add(row), kv_len.sub(1));
                const pv4 = pagedRow(b, v_cache, pt, p, BS, PS).to(.bfloat4).add(lane);
                const w = b.vec(.float4, .{if (R == 1) b.simdBroadcast(vs, cc.to(.ushort)) else b.simdShuffle(vs, row)});
                var dims = b.openForOpts(b.constant(.short, 0), NI, 1, full);
                {
                    const ii = dims.iv;
                    o.store(ii, o.load(ii).add(pv4.load(ii.mul(NL)).to(.float4).mul(w)));
                    dims.close();
                }
                pv.close();
            }

            blocks.next();
            ic0.assign(ic0.add(live));
            blocks.close();
        }

        var mask: i32 = NL;
        while (mask < simd_width) : (mask *= 2) {
            var sum = b.openForOpts(b.constant(.short, 0), NI, 1, full);
            {
                o.store(sum.iv, o.load(sum.iv).add(b.simdShuffleXor(o.load(sum.iv), b.constant(.ushort, mask))));
                sum.close();
            }
        }
        var first = if (R > 1) b.openIf(sub.eq(0)) else null;
        if (NWG == 1) {
            const inv_sum = b.ternary(S.eq(0.0), 0.0, b.lift(1.0).div(S));
            const dst4 = a.dst.to(.bfloat4).add(rid.mul(D4)).add(lane);
            var out = b.openForOpts(b.constant(.short, 0), NI, 1, full);
            {
                dst4.store(out.iv.mul(NL), o.load(out.iv).mul(inv_sum).to(.bfloat4));
                out.close();
            }
        } else {
            // [rows][D4][splits] float4 partials, then [rows][splits] (S, M).
            const dst4 = a.dst.to(.float4).add(rid.mul(D4).mul(NWG));
            var out = b.openForOpts(b.constant(.short, 0), NI, 1, full);
            {
                const i = lane.to(.int).add(out.iv.to(.int).mul(NL));
                dst4.store(i.mul(NWG).add(iwg), o.load(out.iv));
                out.close();
            }
            var lane0 = b.openIf(a.tiisg.eq(0));
            {
                const dst1 = a.dst.to(.float).add(nrows.mul(D).mul(NWG)).add(rid.mul(2 * NWG));
                dst1.store(b.lift(2).mul(iwg), S);
                dst1.store(b.lift(2).mul(iwg).add(1), M);
                lane0.close();
            }
        }
        if (first) |*scope| scope.close();
    }

    /// The address of position `p`'s bf16 row in `cache`, through the page
    /// table `pt`.
    fn pagedRow(b: *Builder, cache: Value, pt: Value, p: Value, BS: i32, PS: i32) Value {
        const page = pt.load(p.div(BS)).to(.uint).to(.ulong).mul(b.constant(.uint, BS * PS));
        const row = p.rem(BS).to(.uint).mul(b.constant(.uint, PS));
        return cache.add(page.add(row).mul(2));
    }
};

/// The sequence of query row `r` and its causal KV range [kv_lo, kv_len).
fn sequenceOf(b: *Builder, a: anytype, r: Value) struct { idx: Value, kv_len: Value, kv_lo: Value } {
    const first_seq = b.local(.int, 0);
    const last_seq = b.local(.int, a.args.field("num_seqs").sub(1));
    var search = b.openLoop(.{});
    {
        search.body(first_seq.lt(last_seq));
    }
    {
        const mid = first_seq.add(last_seq).add(1).div(2);
        var half = b.openIfElse(a.cu_seqlens_q.load(mid).le(r), .{});
        {
            first_seq.assign(mid);
            half.elseBranch();
        }
        {
            last_seq.assign(mid.sub(1));
            half.close();
        }
        search.close();
    }
    const idx = first_seq.rvalue();
    const q_pos = r.sub(a.cu_seqlens_q.load(idx));
    const query_len = a.cu_seqlens_q.load(idx.add(1)).sub(a.cu_seqlens_q.load(idx));
    const kv_len = a.seq_lens.load(idx).sub(query_len).add(q_pos).add(1);
    const kv_lo = b.ternary(a.args.field("sliding_window").lt(0), 0, b.max(0, kv_len.sub(a.args.field("sliding_window"))));
    return .{ .idx = idx, .kv_len = kv_len, .kv_lo = kv_lo };
}

fn liveSplits(b: *Builder, seq: anytype, splits: i32) Value {
    const chunks = seq.kv_len.add(simd_width - 1).div(simd_width).sub(seq.kv_lo.div(simd_width));
    return b.min(splits, chunks);
}

/// fa_vec_paged_reduce: merges the fa_vec_paged split partials.
pub const FaVecPagedReduce = struct {
    pub const Cfg = struct {
        splits: i32,
        /// The head dimension.
        head_dim: i32,
    };

    pub const simdgroups = 8;

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_vec_paged_reduce",
        .inputs = &.{ "htmp", "seq_lens", "cu_seqlens_q" },
        .outputs = &.{"dst"},
        .constants = &.{"args"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        const int_in: ArgSpec = .{ .buffer = .{ .type = .int, .const_ = true } };
        const a = try b.declareArgs(.{
            .args = .{ .constant_struct = FaVecPaged.Args },
            .htmp = .{ .buffer = .{ .type = .char, .const_ = true } },
            .seq_lens = int_in,
            .cu_seqlens_q = int_in,
            .dst = .{ .buffer = .{ .type = .char } },
            .tgpig = .{ .threadgroup_position_in_grid = .uint2 },
            .tgpg = .{ .threadgroups_per_grid = .uint2 },
            .tiisg = .{ .thread_index_in_simdgroup = .ushort },
            .sgitg = .{ .simdgroup_index_in_threadgroup = .ushort },
        });
        const NWG = cfg.splits;
        if (NWG > 4 * simd_width) std.debug.panic("fa_vec_paged_reduce: {d} splits, at most {d}", .{ NWG, 4 * simd_width });
        const NP = @divFloor(NWG + simd_width - 1, simd_width);
        const DV = cfg.head_dim;
        const DV4 = @divExact(DV, 4);
        const full: metal.LoopOpts = .{ .unroll = .full };

        const rid = a.tgpig.at(0).to(.ulong);
        const r = a.tgpig.at(0).div(a.args.field("num_heads").to(.uint)).to(.int);
        const i = a.tgpig.at(1).to(.int).mul(simdgroups).add(a.sgitg.to(.int));
        const dst4 = a.dst.to(.bfloat4).add(rid.mul(DV4));

        const live = b.local(.int, 0);
        var real = b.openIf(r.lt(a.cu_seqlens_q.load(a.args.field("num_seqs"))));
        {
            live.assign(liveSplits(b, sequenceOf(b, a, r), NWG));
            real.close();
        }
        var empty = b.openIf(live.le(0));
        {
            var lane0 = b.openIf(a.tiisg.eq(0));
            {
                dst4.store(i, b.constant(.float4, 0).to(.bfloat4));
                lane0.close();
            }
            b.return_();
            empty.close();
        }

        const ss = a.htmp.to(.float).add(a.tgpg.at(0).to(.ulong).mul(DV).mul(NWG)).add(rid.mul(2 * NWG));
        const Mj = b.array(.float, @intCast(NP));
        const Sj = b.array(.float, @intCast(NP));
        const mx = b.local(.float, -flt_max / 2.0);
        var load = b.openForOpts(b.constant(.short, 0), NP, 1, full);
        {
            const j = load.iv;
            const idx = a.tiisg.to(.int).add(j.to(.int).mul(simd_width));
            const ok = idx.lt(live);
            Mj.store(j, b.ternary(ok, ss.load(idx.mul(2).add(1)), b.constant(.float, -flt_max / 2.0)));
            Sj.store(j, b.ternary(ok, ss.load(idx.mul(2)), b.constant(.float, 0)));
            mx.assign(b.max(mx.rvalue(), Mj.load(j)));
            load.close();
        }
        const m = b.simdMax(mx.rvalue());
        const sum = b.local(.float, 0.0);
        var weigh = b.openForOpts(b.constant(.short, 0), NP, 1, full);
        {
            const j = weigh.iv;
            Mj.store(j, b.exp(Mj.load(j).sub(m)));
            sum.assign(sum.add(Sj.load(j).mul(Mj.load(j))));
            weigh.close();
        }
        const S = b.simdSum(sum.rvalue());
        const inv_sum = b.ternary(S.eq(0.0), 0.0, b.lift(1.0).div(S));

        const htmp4 = a.htmp.to(.float4).add(rid.mul(DV4).mul(NWG));
        const acc = b.local(.float4, b.constant(.float4, 0));
        var parts = b.openForOpts(b.constant(.short, 0), NP, 1, full);
        {
            const j = parts.iv;
            const idx = a.tiisg.to(.int).add(j.to(.int).mul(simd_width));
            var ok = b.openIf(idx.lt(live));
            {
                acc.assign(acc.add(htmp4.load(i.mul(NWG).add(idx)).mul(Mj.load(j))));
                ok.close();
            }
            parts.close();
        }
        const v = b.simdSum(acc.rvalue());
        var lane0 = b.openIf(a.tiisg.eq(0));
        {
            dst4.store(i, v.mul(inv_sum).to(.bfloat4));
            lane0.close();
        }
    }
};

/// zero_padding_rows: zeroes the output rows past the last real token.
pub const ZeroPaddingRows = struct {
    pub const Cfg = struct { dtype: Type };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "zero_padding_rows",
        .inputs = &.{ "cu_seqlens_q", "out" },
        .outputs = &.{"out"},
        .constants = &.{ "num_seqs", "row_stride" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        switch (cfg.dtype.scalar) {
            .bfloat, .half => {},
            else => std.debug.panic("zero_padding_rows: {s} unsupported (half or bfloat)", .{cfg.dtype.name()}),
        }
        const a = try b.declareArgs(.{
            .out = .{ .buffer = .{ .type = cfg.dtype } },
            .cu_seqlens_q = .{ .buffer = .{ .type = .int, .const_ = true } },
            .num_seqs = .{ .constant = .int },
            .row_stride = .{ .constant = .int },
            .tgp = .{ .threadgroup_position_in_grid = .uint3 },
            .tpt = .{ .thread_position_in_threadgroup = .uint3 },
            .ntg = .{ .threads_per_threadgroup = .uint3 },
        });
        const row = a.tgp.x().to(.int);
        b.returnIf(row.lt(a.cu_seqlens_q.load(a.num_seqs)));
        const dst = a.out.add(row.to(.ulong).mul(a.row_stride.to(.ulong)));
        var loop = b.openFor(a.tpt.x().to(.int), a.row_stride, a.ntg.x().to(.int));
        {
            dst.store(loop.iv, b.constant(cfg.dtype, 0));
            loop.close();
        }
    }
};

/// paged_attention_tiled: FlashAttention-2 over 8x8 simdgroup matrices, causal
/// or bidirectional, one threadgroup per (query head, query block).
pub const PagedAttentionTiled = struct {
    pub const Cfg = struct {
        /// half or bfloat.
        dtype: Type,
        head_size: i32,
        block_size: i32,
        bq: i32 = 32,
        tile_kv: i32 = 32,
        num_threads: i32 = 128,
        /// Bidirectional when false.
        causal: bool = true,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "paged_attention_tiled",
        .inputs = &.{ "q", "k_cache", "v_cache", "block_tables", "context_lens", "cu_seqlens_q" },
        .outputs = &.{"out"},
        .constants = &.{ "num_kv_heads", "scale", "softcapping", "max_num_blocks_per_seq", "q_stride", "kv_block_stride", "kv_head_stride", "num_seqs", "sliding_window" },
        .run = run,
    });

    /// What the MSL instantiates: half or bfloat, a head size of tileConfig
    /// with its tiles, and pages of 8, 16 or 32 positions.
    fn supported(cfg: Cfg) bool {
        const tile = tileConfig(cfg.head_size) orelse return false;
        return (cfg.dtype.eql(.half) or cfg.dtype.eql(.bfloat)) and
            (cfg.block_size == 8 or cfg.block_size == 16 or cfg.block_size == 32) and
            cfg.bq == tile[0] and cfg.tile_kv == tile[1] and cfg.num_threads == tile[2];
    }

    /// The (bq, tile_kv, num_threads) of a head size, null when unsupported.
    pub fn tileConfig(head_size: i64) ?[3]i32 {
        return switch (head_size) {
            32, 64, 96, 128 => .{ 32, 32, 128 },
            256 => .{ 16, 16, 64 },
            512 => .{ 8, 8, 32 },
            else => null,
        };
    }

    pub fn threadgroupMemory(cfg: Cfg) u64 {
        const ld = cfg.head_size + @divExact(16, @as(i32, @intCast(cfg.dtype.size())));
        return @intCast((cfg.bq + 2 * cfg.tile_kv) * ld * @as(i32, @intCast(cfg.dtype.size())));
    }

    /// Lane `lane_id`'s (column, row) in an 8x8 simdgroup_matrix fragment.
    fn fragCoord(b: *Builder, lane_id: Value) Value {
        const qid = lane_id.to(.short).div(4).to(.short);
        const fm = qid.bitAnd(4).add(lane_id.to(.short).div(2).rem(4)).to(.short);
        const fn_ = qid.bitAnd(2).mul(2).add(lane_id.to(.short).rem(2).mul(2)).to(.short);
        return b.vec(.short2, .{ fn_, fm });
    }

    /// Reduces `v` with `op` across the lanes that hold one row of an 8x8
    /// simdgroup_matrix fragment.
    fn fragRowReduce(b: *Builder, comptime op: enum { max, sum }, v: Value) Value {
        var x = v.rvalue();
        inline for (.{ 1, 8 }) |mask| {
            const y = b.simdShuffleXor(x, b.constant(.ushort, mask));
            x = switch (op) {
                .max => b.max(x, y),
                .sum => x.add(y),
            };
        }
        return x;
    }

    /// Sets lane `lane` of the vector `p[i]` to `x`.
    fn setLane(p: Value, i: Value, lane: anytype, x: Value) void {
        p.store(i, p.load(i).insert(lane, x));
    }

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        if (!supported(cfg)) std.debug.panic("paged_attention_tiled: {s} at head size {d}, block size {d} (8, 16 or 32) and tiles {d}, {d}, {d} unsupported", .{ cfg.dtype.name(), cfg.head_size, cfg.block_size, cfg.bq, cfg.tile_kv, cfg.num_threads });
        const T = cfg.dtype;
        const HEAD_SIZE = cfg.head_size;
        const BLOCK_SIZE = cfg.block_size;
        const BQ = cfg.bq;
        const TILE_KV = cfg.tile_kv;
        const NUM_THREADS = cfg.num_threads;
        const int_arg: ArgSpec = .{ .constant = .int };
        const float_arg: ArgSpec = .{ .constant = .float };
        const a = try b.declareArgs(.{
            .out = .{ .buffer = .{ .type = T } },
            .q = .{ .buffer = .{ .type = T, .const_ = true } },
            .k_cache = .{ .buffer = .{ .type = T, .const_ = true } },
            .v_cache = .{ .buffer = .{ .type = T, .const_ = true } },
            .num_kv_heads = int_arg,
            .scale = float_arg,
            .softcapping = float_arg,
            .block_tables = .{ .buffer = .{ .type = .uint, .const_ = true } },
            .context_lens = .{ .buffer = .{ .type = .uint, .const_ = true } },
            .max_num_blocks_per_seq = int_arg,
            .q_stride = int_arg,
            .kv_block_stride = int_arg,
            .kv_head_stride = int_arg,
            .cu_seqlens_q = .{ .buffer = .{ .type = .int, .const_ = true } },
            .num_seqs = int_arg,
            .sliding_window = int_arg,
            .shared_mem = .{ .threadgroup = .char },
            .tgp = .{ .threadgroup_position_in_grid = .uint3 },
            .tgpg = .{ .threadgroups_per_grid = .uint3 },
            .tpt = .{ .thread_position_in_threadgroup = .uint3 },
            .sg_idx = .{ .simdgroup_index_in_threadgroup = .uint },
            .lane = .{ .thread_index_in_simdgroup = .uint },
        });

        const NUM_SIMD_LANES = 32;
        const NUM_SG = @divExact(NUM_THREADS, NUM_SIMD_LANES);
        const TD = @divExact(HEAD_SIZE, 8);
        const TK = @divExact(TILE_KV, 8);
        if (@divExact(BQ, NUM_SG) != 8) @panic("paged_attention_tiled: ROWS_PER_SG must be 8");
        const unroll: metal.LoopOpts = .{ .unroll = .enable };

        const thread_idx = a.tpt.x().to(.int);
        const head_idx = a.tgp.x().to(.int);
        const q_block_global_idx = a.tgp.y().to(.int);
        const num_heads = a.tgpg.x().to(.int);
        const num_queries_per_kv = num_heads.div(a.num_kv_heads);
        const kv_head_idx = head_idx.div(num_queries_per_kv);
        const kv_token_stride = a.num_kv_heads.mul(a.kv_head_stride);

        const first_seq = b.local(.int, 0);
        const last_seq = b.local(.int, a.num_seqs);
        var search = b.openLoop(.{});
        {
            search.body(first_seq.lt(last_seq));
        }
        {
            const mid = first_seq.add(last_seq).add(1).div(2);
            var half = b.openIfElse(a.cu_seqlens_q.load(mid).div(BQ).add(mid).le(q_block_global_idx), .{});
            {
                first_seq.assign(mid);
                half.elseBranch();
            }
            {
                last_seq.assign(mid.sub(1));
                half.close();
            }
            search.close();
        }
        // The grid is an upper bound on the query blocks: the extra ones land past
        // the last sequence and clamp to it.
        const seq_idx = b.min(first_seq.rvalue(), a.num_seqs.sub(1));

        const q_seq_start = a.cu_seqlens_q.load(seq_idx);
        const cur_batch_query_len = a.cu_seqlens_q.load(seq_idx.add(1)).sub(q_seq_start);
        const q_block_start = q_seq_start.div(BQ).add(seq_idx);
        const q_block_local = q_block_global_idx.sub(q_block_start);
        const q_pos_start = q_block_local.mul(BQ);

        b.returnIf(q_pos_start.ge(cur_batch_query_len));

        const seq_len = a.context_lens.load(seq_idx).to(.int);
        const context_len = seq_len.sub(cur_batch_query_len);
        const valid_q = b.min(BQ, cur_batch_query_len.sub(q_pos_start));

        const SMEM_PAD = @divExact(16, @as(i32, @intCast(T.size())));
        const LD = HEAD_SIZE + SMEM_PAD;
        const Q_ELEMS = BQ * LD;
        const KV_ELEMS = TILE_KV * LD;

        const Q_smem = a.shared_mem.to(T);
        const K_smem = Q_smem.add(Q_ELEMS);
        const V_smem = K_smem.add(KV_ELEMS);

        const log2e: f32 = std.math.log2e;
        const scale_log2 = a.scale.mul(log2e);

        const q_base = a.q.add(q_seq_start.add(q_pos_start).mul(a.q_stride)).add(head_idx.mul(HEAD_SIZE));
        var load_q = b.openFor(thread_idx, BQ * HEAD_SIZE, NUM_THREADS);
        {
            const i = load_q.iv;
            const r = i.div(HEAD_SIZE);
            const d = i.rem(HEAD_SIZE);
            var value = b.openIfElse(r.lt(valid_q), .{T});
            {
                value.yieldThen(.{q_base.load(r.mul(a.q_stride).add(d))});
            }
            {
                value.yieldElse(.{b.constant(T, 0)});
            }
            Q_smem.store(r.mul(LD).add(d), value.results[0]);
            load_q.close();
        }

        b.threadgroupBarrier(.mem_threadgroup);

        const T2: Type = .vec(T.scalar, 2);
        const q_reg = b.array(T2, @intCast(TD));
        var load_q_reg = b.openForOpts(0, TD, 1, unroll);
        {
            const d = load_q_reg.iv;
            const tmp = b.simdgroupLoad(Q_smem.add(a.sg_idx.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
            q_reg.store(d, tmp.threadElements());
            load_q_reg.close();
        }

        const fc = fragCoord(b, a.lane.to(.ushort));
        const fm = fc.y();
        const fn_ = fc.x();

        const max_score = b.local(.float, masked_score);
        const sum_score = b.local(.float, 0.0);

        const o_reg = b.array(.float2, @intCast(TD));
        var zero_o = b.openForOpts(0, TD, 1, unroll);
        {
            o_reg.store(zero_o.iv, b.constant(.float2, 0));
            zero_o.close();
        }

        const block_table = a.block_tables.add(seq_idx.mul(a.max_num_blocks_per_seq));
        const num_kv_tiles = seq_len.add(TILE_KV).sub(1).div(TILE_KV);

        var tiles = b.openFor(0, num_kv_tiles, 1);
        {
            const tile_idx = tiles.iv;
            const tile_start = tile_idx.mul(TILE_KV);

            if (cfg.causal) {
                var past = b.openIf(tile_start.gt(context_len.add(q_pos_start).add(valid_q).sub(1)));
                {
                    tiles.breakLoop();
                    past.close();
                }
            }

            const VEC_BYTES: i32 = if (HEAD_SIZE >= 256) 16 else 8;
            const VEC = @divExact(VEC_BYTES, @as(i32, @intCast(T.size())));
            const LoadUnit: Type = if (VEC_BYTES == 16) .uint4 else .uint2;
            std.debug.assert(@rem(LD, VEC) == 0);
            var rows = b.openForOpts(0, @divExact(TILE_KV, NUM_SG), 1, unroll);
            {
                const t = a.sg_idx.mul(@divExact(TILE_KV, NUM_SG)).add(rows.iv).to(.int);
                const kv_pos = tile_start.add(t);
                var present = b.openIfElse(kv_pos.lt(seq_len), .{});
                {
                    const pb = block_table.load(kv_pos.div(BLOCK_SIZE)).to(.long);
                    const off = pb.mul(a.kv_block_stride).add(kv_pos.rem(BLOCK_SIZE).mul(kv_token_stride)).add(kv_head_idx.mul(a.kv_head_stride));
                    const k_ptr = a.k_cache.add(off);
                    const v_ptr = a.v_cache.add(off);
                    var loop = b.openForOpts(a.lane.mul(VEC).to(.int), HEAD_SIZE, NUM_SIMD_LANES * VEC, unroll);
                    {
                        const d = loop.iv;
                        const k = k_ptr.add(d).to(LoadUnit).load(0);
                        K_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, k);
                        const v = v_ptr.add(d).to(LoadUnit).load(0);
                        V_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, v);
                        loop.close();
                    }
                    present.elseBranch();
                }
                {
                    const zero = b.constant(LoadUnit, 0);
                    var loop = b.openForOpts(a.lane.mul(VEC).to(.int), HEAD_SIZE, NUM_SIMD_LANES * VEC, unroll);
                    {
                        const d = loop.iv;
                        K_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, zero);
                        V_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, zero);
                        loop.close();
                    }
                    present.close();
                }
                rows.close();
            }

            b.threadgroupBarrier(.mem_threadgroup);

            const s_reg = b.array(.float2, @intCast(TK));
            var zero_s = b.openForOpts(0, TK, 1, unroll);
            {
                s_reg.store(zero_s.iv, b.constant(.float2, 0));
                zero_s.close();
            }

            var qk_dims = b.openForOpts(0, TD, 1, unroll);
            {
                const d = qk_dims.iv;
                const q_frag = b.matrix(T.scalar, q_reg.load(d));
                var qk_keys = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = qk_keys.iv;
                    const k_frag = b.simdgroupLoad(K_smem.add(k.mul(8).mul(LD)).add(d.mul(8)), LD, .{ .transpose = true });
                    const s_frag = b.matrix(.float, s_reg.load(k));
                    s_reg.store(k, b.simdgroupMultiplyAccumulate(q_frag, k_frag, s_frag).threadElements());
                    qk_keys.close();
                }
                qk_dims.close();
            }

            const q_abs_pos = context_len.add(q_pos_start).add(a.sg_idx.mul(8)).add(fm).to(.int);
            const row_masked = a.sg_idx.mul(8).add(fm).ge(valid_q);
            const min_q_abs_pos = context_len.add(q_pos_start);
            var tile_no_mask = tile_start.add(TILE_KV).le(seq_len).bitAnd(a.softcapping.le(0.0)).bitAnd(a.sliding_window.lt(0));
            if (cfg.causal) tile_no_mask = tile_no_mask.bitAnd(tile_start.add(TILE_KV).sub(1).lt(min_q_abs_pos));

            var masking = b.openIfElse(tile_no_mask.bitAnd(row_masked.eq(false)), .{});
            {
                var scale = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = scale.iv;
                    setLane(s_reg, k, 0, s_reg.load(k).at(0).mul(scale_log2));
                    setLane(s_reg, k, 1, s_reg.load(k).at(1).mul(scale_log2));
                    scale.close();
                }
                masking.elseBranch();
            }
            {
                var keys = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = keys.iv;
                    var pair = b.openForOpts(0, 2, 1, unroll);
                    {
                        const jj = pair.iv;
                        const s = b.local(.float, s_reg.load(k).at(jj).mul(scale_log2));
                        var capped = b.openIf(a.softcapping.gt(0.0));
                        {
                            const s_orig = s.div(log2e);
                            s.assign(a.softcapping.mul(b.precise().tanh(s_orig.div(a.softcapping))).mul(log2e));
                            capped.close();
                        }
                        const kv_pos = tile_start.add(k.mul(8)).add(fn_).add(jj);
                        var masked = row_masked.bitOr(kv_pos.ge(seq_len));
                        if (cfg.causal) masked = masked.bitOr(kv_pos.gt(q_abs_pos));
                        const before_window = a.sliding_window.ge(0).bitAnd(kv_pos.lt(q_abs_pos.add(1).sub(a.sliding_window)));
                        masked = masked.bitOr(before_window);
                        setLane(s_reg, k, jj, b.ternary(masked, masked_score, s));
                        pair.close();
                    }
                    keys.close();
                }
                masking.close();
            }

            const local_max = b.local(.float, masked_score);
            var max_keys = b.openForOpts(0, TK, 1, unroll);
            {
                const k = max_keys.iv;
                local_max.assign(b.max(local_max.rvalue(), b.max(s_reg.load(k).at(0), s_reg.load(k).at(1))));
                max_keys.close();
            }
            const row_max = fragRowReduce(b, .max, local_max);

            const prev_max = max_score.rvalue();
            const grown_max = b.max(prev_max, row_max);
            const grew = grown_max.gt(prev_max);
            const was_empty = prev_max.eq(masked_score);
            const factor = b.ternary(grew, b.ternary(was_empty, 0.0, b.fast().exp2(prev_max.sub(grown_max))), 1.0);
            const new_max = b.ternary(grew.eq(false).bitAnd(was_empty), 0.0, grown_max);
            // A tile masking a whole row leaves the maximum masked: zero would
            // pin it, and later scores far below zero would underflow.
            max_score.assign(grown_max);

            const local_sum = b.local(.float, 0.0);
            var p_keys = b.openForOpts(0, TK, 1, unroll);
            {
                const k = p_keys.iv;
                var pair = b.openForOpts(0, 2, 1, unroll);
                {
                    const jj = pair.iv;
                    const s = s_reg.load(k).at(jj);
                    const p = b.ternary(s.eq(masked_score), 0.0, b.fast().exp2(s.sub(new_max)));
                    setLane(s_reg, k, jj, p);
                    local_sum.assign(local_sum.add(p));
                    pair.close();
                }
                p_keys.close();
            }
            const row_sum = fragRowReduce(b, .sum, local_sum);

            sum_score.assign(sum_score.mul(factor).add(row_sum));

            var rescale_o = b.openForOpts(0, TD, 1, unroll);
            {
                o_reg.store(rescale_o.iv, o_reg.load(rescale_o.iv).mul(factor));
                rescale_o.close();
            }

            var pv_keys = b.openForOpts(0, TK, 1, unroll);
            {
                const k = pv_keys.iv;
                const p_frag = b.matrix(.float, s_reg.load(k));
                var pv_dims = b.openForOpts(0, TD, 1, unroll);
                {
                    const d = pv_dims.iv;
                    const v_frag = b.simdgroupLoad(V_smem.add(k.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
                    const o_frag = b.matrix(.float, o_reg.load(d));
                    o_reg.store(d, b.simdgroupMultiplyAccumulate(p_frag, v_frag, o_frag).threadElements());
                    pv_dims.close();
                }
                pv_keys.close();
            }

            b.threadgroupBarrier(.mem_threadgroup);
            tiles.close();
        }

        const inv_sum = b.lift(1.0).div(sum_score.add(@as(f32, 1e-6)));
        var normalize = b.openForOpts(0, TD, 1, unroll);
        {
            o_reg.store(normalize.iv, o_reg.load(normalize.iv).mul(inv_sum));
            normalize.close();
        }

        const O_smem = Q_smem.to(.float);
        var store_o = b.openForOpts(0, TD, 1, unroll);
        {
            const d = store_o.iv;
            const o_frag = b.matrix(.float, o_reg.load(d));
            b.simdgroupStore(o_frag, O_smem.add(a.sg_idx.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
            store_o.close();
        }

        b.threadgroupBarrier(.mem_threadgroup);

        const out_base = a.out.add(q_seq_start.add(q_pos_start).mul(a.q_stride)).add(head_idx.mul(HEAD_SIZE));
        const total = valid_q.mul(HEAD_SIZE);
        var write_out = b.openFor(thread_idx, total, NUM_THREADS);
        {
            const i = write_out.iv;
            const r = i.div(HEAD_SIZE);
            const d = i.rem(HEAD_SIZE);
            const value = O_smem.load(r.mul(LD).add(d)).to(T);
            out_base.store(r.mul(a.q_stride).add(d), value);
            write_out.close();
        }
    }
};

test "the paged attention kernels emit verified modules" {
    const allocator = std.testing.allocator;
    for ([_]i32{ 32, 64, 128, 256, 512 }) |head_dim| {
        for ([_][2]i32{ .{ 1, 1 }, .{ 4, 1 }, .{ 2, 8 }, .{ 4, 128 } }) |launch| {
            allocator.free(try FaVecPaged.Kernel.emit(allocator, .{ .head_dim = head_dim, .block_size = 16, .position_stride = 2 * head_dim, .heads = launch[0], .splits = launch[1] }));
        }
        for ([_]i32{ 8, 128 }) |splits| {
            allocator.free(try FaVecPagedReduce.Kernel.emit(allocator, .{ .splits = splits, .head_dim = head_dim }));
        }
    }
    for ([_]Type{ .half, .bfloat }) |dtype| {
        allocator.free(try ZeroPaddingRows.Kernel.emit(allocator, .{ .dtype = dtype }));
        for ([_]i32{ 32, 64, 96, 128, 256, 512 }) |head_size| {
            const tile = PagedAttentionTiled.tileConfig(head_size).?;
            for ([_]bool{ true, false }) |causal| {
                allocator.free(try PagedAttentionTiled.Kernel.emit(allocator, .{ .dtype = dtype, .head_size = head_size, .block_size = 16, .bq = tile[0], .tile_kv = tile[1], .num_threads = tile[2], .causal = causal }));
            }
        }
    }
}
