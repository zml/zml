//! Metal flash attention over contiguous `[batch, keys, kv_heads, head_dim]`
//! K/V, causal from `token_index`: fa_vec and fa_vec_reduce for decode,
//! fa_tiled otherwise.

const std = @import("std");

const metal = @import("../../../kernel/kernel.zig").metal;

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

/// fa_vec: bf16 decode, one query per batch row. Simdgroup s takes query head
/// s % heads of the threadgroup's KV head group and position split s / heads.
pub const FaVec = struct {
    pub const Cfg = struct {
        head_dim: i32,
        heads: i32,
        splits: i32,
    };

    pub const Args: metal.Struct = .{ .name = "fa_vec_args", .fields = &.{
        .{ .name = "num_heads", .type = .int },
        .{ .name = "n_kv", .type = .int },
        .{ .name = "k_len", .type = .int },
        .{ .name = "scale", .type = .float },
    } };

    pub const HostArgs = extern struct {
        num_heads: i32,
        n_kv: i32,
        k_len: i32,
        scale: f32,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_vec",
        .inputs = &.{ "q", "k", "v", "token_index" },
        .outputs = &.{"dst"},
        .constants = &.{"args"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        switch (cfg.head_dim) {
            32, 64, 128, 256, 512 => {},
            else => std.debug.panic("fa_vec: head_dim {d} unsupported (32, 64, 128, 256 or 512)", .{cfg.head_dim}),
        }
        const char_in: ArgSpec = .{ .buffer = .{ .type = .char, .const_ = true } };
        const a = try b.declareArgs(.{
            .args = .{ .constant_struct = Args },
            .q = char_in,
            .k = char_in,
            .v = char_in,
            .token_index = .{ .buffer = .{ .type = .int, .const_ = true } },
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
        const HG = cfg.heads;
        const NWG = cfg.splits;
        const full: metal.LoopOpts = .{ .unroll = .full };

        const num_heads = a.args.field("num_heads");
        const n_kv = a.args.field("n_kv");
        const group = num_heads.div(n_kv);
        const groups = group.div(HG);
        const r = a.tgpig.at(2).to(.int);
        const tg_head = a.tgpig.at(1).to(.int);
        const ikv = tg_head.div(groups);
        const iq2 = ikv.mul(group).add(tg_head.rem(groups).mul(HG)).add(a.sgitg.rem(HG).to(.int));
        const iwg = a.tgpig.at(0).to(.int).mul(a.nsg.div(HG).to(.int)).add(a.sgitg.div(HG).to(.int));
        const rid = r.to(.long).mul(num_heads.to(.uint)).add(iq2);
        const nrows = a.tgpg.at(2).to(.long).mul(num_heads.to(.uint));

        const kv_len = a.token_index.load(r).add(1);
        const live = if (NWG > 1) liveSplits(b, kv_len, NWG) else b.lift(@as(i32, 1));

        if (NWG > 1) {
            var idle = b.openIf(iwg.ge(live));
            {
                b.return_();
                idle.close();
            }
        }

        const position_bytes = n_kv.to(.ulong).mul(D * 2);
        const first = r.to(.ulong).mul(a.args.field("k_len").to(.ulong)).mul(position_bytes).add(ikv.to(.ulong).mul(D * 2));
        const k0 = a.k.add(first);
        const v0 = a.v.add(first);

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

        const ic0 = b.local(.int, iwg);
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
                const pk4 = k0.add(p.to(.ulong).mul(position_bytes)).to(.bfloat4).add(lane);
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
            const score = b.ternary(pos.lt(kv_len), s.mul(a.args.field("scale")), b.constant(.float, -max_half));

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
                const pv4 = v0.add(p.to(.ulong).mul(position_bytes)).to(.bfloat4).add(lane);
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
        var first_row = if (R > 1) b.openIf(sub.eq(0)) else null;
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
        if (first_row) |*scope| scope.close();
    }
};

fn liveSplits(b: *Builder, kv_len: Value, splits: i32) Value {
    return b.min(splits, kv_len.add(simd_width - 1).div(simd_width));
}

/// fa_vec_reduce: merges the fa_vec split partials.
pub const FaVecReduce = struct {
    pub const Cfg = struct {
        splits: i32,
        head_dim: i32,
    };

    pub const simdgroups = 8;

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_vec_reduce",
        .inputs = &.{ "htmp", "token_index" },
        .outputs = &.{"dst"},
        .constants = &.{"args"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .args = .{ .constant_struct = FaVec.Args },
            .htmp = .{ .buffer = .{ .type = .char, .const_ = true } },
            .token_index = .{ .buffer = .{ .type = .int, .const_ = true } },
            .dst = .{ .buffer = .{ .type = .char } },
            .tgpig = .{ .threadgroup_position_in_grid = .uint2 },
            .tgpg = .{ .threadgroups_per_grid = .uint2 },
            .tiisg = .{ .thread_index_in_simdgroup = .ushort },
            .sgitg = .{ .simdgroup_index_in_threadgroup = .ushort },
        });
        const NWG = cfg.splits;
        if (NWG > 4 * simd_width) std.debug.panic("fa_vec_reduce: {d} splits, at most {d}", .{ NWG, 4 * simd_width });
        const NP = @divFloor(NWG + simd_width - 1, simd_width);
        const DV = cfg.head_dim;
        const DV4 = @divExact(DV, 4);
        const full: metal.LoopOpts = .{ .unroll = .full };

        const rid = a.tgpig.at(0).to(.ulong);
        const r = a.tgpig.at(0).div(a.args.field("num_heads").to(.uint)).to(.int);
        const i = a.tgpig.at(1).to(.int).mul(simdgroups).add(a.sgitg.to(.int));
        const dst4 = a.dst.to(.bfloat4).add(rid.mul(DV4));
        const live = liveSplits(b, a.token_index.load(r).add(1), NWG);

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

/// fa_tiled: causal FlashAttention-2 over simdgroup matrices. Queries past
/// `live` output zeros.
pub const FaTiled = struct {
    pub const Cfg = struct {
        dtype: Type,
        head_size: i32,
        bq: i32 = 32,
        tile_kv: i32 = 32,
        num_threads: i32 = 128,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_tiled",
        .inputs = &.{ "q", "k", "v", "token_index", "live" },
        .outputs = &.{"out"},
        .constants = &.{ "num_kv_heads", "scale", "q_len", "k_len" },
        .run = run,
    });

    fn supported(cfg: Cfg) bool {
        const tile = tileConfig(cfg.head_size) orelse return false;
        return (cfg.dtype.eql(.half) or cfg.dtype.eql(.bfloat)) and
            cfg.bq == tile[0] and cfg.tile_kv == tile[1] and cfg.num_threads == tile[2];
    }

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

    fn setLane(p: Value, i: Value, lane: anytype, x: Value) void {
        p.store(i, p.load(i).insert(lane, x));
    }

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        if (!supported(cfg)) std.debug.panic("fa_tiled: {s} at head size {d} and tiles {d}, {d}, {d} unsupported", .{ cfg.dtype.name(), cfg.head_size, cfg.bq, cfg.tile_kv, cfg.num_threads });
        const T = cfg.dtype;
        const HEAD_SIZE = cfg.head_size;
        const BQ = cfg.bq;
        const TILE_KV = cfg.tile_kv;
        const NUM_THREADS = cfg.num_threads;
        const int_arg: ArgSpec = .{ .constant = .int };
        const int_in: ArgSpec = .{ .buffer = .{ .type = .int, .const_ = true } };
        const a = try b.declareArgs(.{
            .out = .{ .buffer = .{ .type = T } },
            .q = .{ .buffer = .{ .type = T, .const_ = true } },
            .k = .{ .buffer = .{ .type = T, .const_ = true } },
            .v = .{ .buffer = .{ .type = T, .const_ = true } },
            .token_index = int_in,
            .live = int_in,
            .num_kv_heads = int_arg,
            .scale = .{ .constant = .float },
            .q_len = int_arg,
            .k_len = int_arg,
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
        if (@divExact(BQ, NUM_SG) != 8) @panic("fa_tiled: ROWS_PER_SG must be 8");
        const unroll: metal.LoopOpts = .{ .unroll = .enable };

        const thread_idx = a.tpt.x().to(.int);
        const head_idx = a.tgp.x().to(.int);
        const q_pos_start = a.tgp.y().to(.int).mul(BQ);
        const batch = a.tgp.z().to(.int);
        const num_heads = a.tgpg.x().to(.int);
        const kv_head_idx = head_idx.div(num_heads.div(a.num_kv_heads));
        const q_stride = num_heads.mul(HEAD_SIZE);
        const kv_token_stride = a.num_kv_heads.mul(HEAD_SIZE);

        const block_rows = b.min(BQ, a.q_len.sub(q_pos_start));
        const live = a.live.load(0);
        const valid_q = b.min(block_rows, b.max(0, live.sub(q_pos_start)));
        const out_base = a.out.add(batch.mul(a.q_len).add(q_pos_start).to(.long).mul(q_stride).add(head_idx.mul(HEAD_SIZE)));

        var dead = b.openIf(valid_q.le(0));
        {
            var zero = b.openFor(thread_idx, block_rows.mul(HEAD_SIZE), NUM_THREADS);
            {
                const i = zero.iv;
                out_base.store(i.div(HEAD_SIZE).mul(q_stride).add(i.rem(HEAD_SIZE)), b.constant(T, 0));
                zero.close();
            }
            b.return_();
            dead.close();
        }

        // Query i of the batch row is at position context_len + i.
        const context_len = a.token_index.load(batch);
        const seq_len = context_len.add(live);
        const kv_base = batch.to(.long).mul(a.k_len).mul(kv_token_stride).add(kv_head_idx.mul(HEAD_SIZE));

        const SMEM_PAD = @divExact(16, @as(i32, @intCast(T.size())));
        const LD = HEAD_SIZE + SMEM_PAD;
        const Q_ELEMS = BQ * LD;
        const KV_ELEMS = TILE_KV * LD;

        const Q_smem = a.shared_mem.to(T);
        const K_smem = Q_smem.add(Q_ELEMS);
        const V_smem = K_smem.add(KV_ELEMS);

        const log2e: f32 = std.math.log2e;
        const scale_log2 = a.scale.mul(log2e);

        const q_base = a.q.add(batch.mul(a.q_len).add(q_pos_start).to(.long).mul(q_stride).add(head_idx.mul(HEAD_SIZE)));
        var load_q = b.openFor(thread_idx, BQ * HEAD_SIZE, NUM_THREADS);
        {
            const i = load_q.iv;
            const r = i.div(HEAD_SIZE);
            const d = i.rem(HEAD_SIZE);
            var value = b.openIfElse(r.lt(valid_q), .{T});
            {
                value.yieldThen(.{q_base.load(r.mul(q_stride).add(d))});
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

        const num_kv_tiles = seq_len.add(TILE_KV).sub(1).div(TILE_KV);
        const last_q_pos = context_len.add(q_pos_start).add(valid_q).sub(1);

        var tiles = b.openFor(0, num_kv_tiles, 1);
        {
            const tile_idx = tiles.iv;
            const tile_start = tile_idx.mul(TILE_KV);

            var past = b.openIf(tile_start.gt(last_q_pos));
            {
                tiles.breakLoop();
                past.close();
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
                    const off = kv_base.add(kv_pos.to(.long).mul(kv_token_stride));
                    const k_ptr = a.k.add(off);
                    const v_ptr = a.v.add(off);
                    var loop = b.openForOpts(a.lane.mul(VEC).to(.int), HEAD_SIZE, NUM_SIMD_LANES * VEC, unroll);
                    {
                        const d = loop.iv;
                        K_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, k_ptr.add(d).to(LoadUnit).load(0));
                        V_smem.add(t.mul(LD).add(d)).to(LoadUnit).store(0, v_ptr.add(d).to(LoadUnit).load(0));
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
            const tile_no_mask = tile_start.add(TILE_KV).le(seq_len).bitAnd(tile_start.add(TILE_KV).sub(1).lt(context_len.add(q_pos_start)));

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
                        const s = s_reg.load(k).at(jj).mul(scale_log2);
                        const kv_pos = tile_start.add(k.mul(8)).add(fn_).add(jj);
                        const masked = row_masked.bitOr(kv_pos.ge(seq_len)).bitOr(kv_pos.gt(q_abs_pos));
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

        var write_out = b.openFor(thread_idx, block_rows.mul(HEAD_SIZE), NUM_THREADS);
        {
            const i = write_out.iv;
            const r = i.div(HEAD_SIZE);
            const d = i.rem(HEAD_SIZE);
            const value = b.ternary(r.lt(valid_q), O_smem.load(r.mul(LD).add(d)).to(T), b.constant(T, 0));
            out_base.store(r.mul(q_stride).add(d), value);
            write_out.close();
        }
    }
};

test "the flash attention kernels emit verified modules" {
    const allocator = std.testing.allocator;
    for ([_]i32{ 32, 64, 128, 256, 512 }) |head_dim| {
        for ([_][2]i32{ .{ 1, 1 }, .{ 4, 1 }, .{ 2, 8 }, .{ 4, 128 } }) |launch| {
            allocator.free(try FaVec.Kernel.emit(allocator, .{ .head_dim = head_dim, .heads = launch[0], .splits = launch[1] }));
        }
        for ([_]i32{ 8, 128 }) |splits| {
            allocator.free(try FaVecReduce.Kernel.emit(allocator, .{ .splits = splits, .head_dim = head_dim }));
        }
    }
    for ([_]Type{ .half, .bfloat }) |dtype| {
        for ([_]i32{ 32, 64, 96, 128, 256, 512 }) |head_size| {
            const tile = FaTiled.tileConfig(head_size).?;
            allocator.free(try FaTiled.Kernel.emit(allocator, .{ .dtype = dtype, .head_size = head_size, .bq = tile[0], .tile_kv = tile[1], .num_threads = tile[2] }));
        }
    }
}
