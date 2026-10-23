//! Metal paged attention kernels: fa_vec_paged and its split-K reduce for
//! decode, and paged_attention_tiled with zero_padding_rows otherwise.

const std = @import("std");

const kernel = @import("../../../kernel.zig");

const metal = kernel.metal;
const ArgSpec = metal.ArgSpec;
const Builder = metal.Builder;
const Type = metal.Type;
const Value = metal.Value;

const flt_max = std.math.floatMax(f32);
const max_half = 65504.0;
const simd_width = 32;

fn pad2(x: i32, n: i32) i32 {
    return (x + n - 1) & ~(n - 1);
}

/// fa_vec_paged: decode attention over paged bf16 caches, one threadgroup
/// of `nsg` simdgroups per (position split, query head, query row).
pub const FaVecPaged = struct {
    pub const Cfg = struct {
        /// The key and value head dimension, DK = DV.
        head_dim: i32,
        /// Simdgroups splitting the KV positions.
        nsg: i32,
        /// The page size.
        block_size: i32,
        /// The in-page position stride, n_kv * head_dim elements.
        position_stride: i32,
        /// Position-split threadgroups. Above 1 the kernel writes f32 partials
        /// for FaVecPagedReduce.
        nwg: i32,
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

    /// The threadgroup memory of a launch with `nsg` simdgroups.
    pub fn threadgroupMemory(head_dim: i32, nsg: i32) u64 {
        const padded = pad2(head_dim, 128);
        const bytes = (padded + 4 * 32 + 2 * padded) * nsg * 2;
        return @intCast(@divFloor(bytes + 15, 16) * 16);
    }

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        switch (cfg.head_dim) {
            128, 256, 512 => {},
            else => std.debug.panic("fa_vec_paged: head_dim {d} unsupported (128, 256 or 512)", .{cfg.head_dim}),
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
            .shmem_f16 = .{ .threadgroup = .half },
            .tgpig = .{ .threadgroup_position_in_grid = .uint3 },
            .tgpg = .{ .threadgroups_per_grid = .uint3 },
            .tiisg = .{ .thread_index_in_simdgroup = .ushort },
            .sgitg = .{ .simdgroup_index_in_threadgroup = .ushort },
        });
        impl(b, cfg, a);
    }

    /// The attention body.
    fn impl(b: *Builder, cfg: Cfg, a: anytype) void {
        const DK = cfg.head_dim;
        const DV = cfg.head_dim;
        const NE = 1;
        const C = 32;
        const NSG = cfg.nsg;
        const BS = cfg.block_size;
        const PS = cfg.position_stride;
        const NWG = cfg.nwg;
        const q = b.localPointer(a.q);
        const k_cache = b.localPointer(a.k_cache);
        const v_cache = b.localPointer(a.v_cache);

        const r = a.tgpig.at(2).to(.int);
        const iq2 = a.tgpig.at(1).to(.ushort);
        const iwg = a.tgpig.at(0).to(.short);

        const DK4 = @divExact(DK, 4);
        const DV4 = @divExact(DV, 4);

        const nrows_static = a.tgpg.at(2).to(.long).mul(a.args.field("num_heads").to(.uint));

        var padding = b.openIf(r.ge(a.cu_seqlens_q.load(a.args.field("num_seqs"))));
        {
            var first = b.openIf(a.sgitg.eq(0));
            {
                const rid = r.to(.long).mul(a.args.field("num_heads").to(.uint)).add(iq2);
                if (NWG == 1) {
                    const dst4 = a.dst.to(.bfloat4);
                    var loop = b.openFor(a.tiisg.to(.short), DV4, simd_width);
                    {
                        const zero = b.constant(.float4, 0).to(.bfloat4);
                        dst4.store(rid.mul(DV4).add(loop.iv), zero);
                        loop.close();
                    }
                } else {
                    const dst4 = a.dst.to(.float4);
                    const dst1 = a.dst.to(.float).add(nrows_static.mul(DV).mul(NWG));
                    var loop = b.openFor(a.tiisg.to(.short), DV4, simd_width);
                    {
                        dst4.store(rid.mul(DV4).mul(NWG).add(b.lift(NWG).mul(loop.iv)).add(iwg), b.constant(.float4, 0));
                        loop.close();
                    }
                    var lane0 = b.openIf(a.tiisg.eq(0));
                    {
                        dst1.store(rid.mul(2 * NWG).add(b.lift(2).mul(iwg)).add(0), 0.0);
                        const lowest = b.lift(-flt_max).div(2);
                        dst1.store(rid.mul(2 * NWG).add(b.lift(2).mul(iwg)).add(1), lowest);
                        lane0.close();
                    }
                }
                first.close();
            }
            b.return_();
            padding.close();
        }

        const seq_idx = b.local(.int, null);
        {
            const lo = b.local(.int, 0);
            const hi = b.local(.int, a.args.field("num_seqs").sub(1));
            var search = b.openLoop(.{});
            {
                search.body(lo.lt(hi));
            }
            {
                const mid = lo.add(hi).add(1).div(2);
                var half = b.openIfElse(a.cu_seqlens_q.load(mid).le(r), .{});
                {
                    lo.assign(mid);
                    half.elseBranch();
                }
                {
                    hi.assign(mid.sub(1));
                    half.close();
                }
                search.close();
            }
            seq_idx.assign(lo);
        }
        const q_pos = r.sub(a.cu_seqlens_q.load(seq_idx));
        const query_len = a.cu_seqlens_q.load(seq_idx.add(1)).sub(a.cu_seqlens_q.load(seq_idx));
        const kv_total = a.seq_lens.load(seq_idx);
        const kv_len = kv_total.sub(query_len).add(q_pos).add(1);
        var window = b.openIfElse(a.args.field("sliding_window").lt(0), .{.int});
        {
            window.yieldThen(.{0});
        }
        {
            window.yieldElse(.{b.max(0, kv_len.sub(a.args.field("sliding_window")))});
        }
        const kv_lo = window.results[0];

        const pt = a.block_tables.add(seq_idx.to(.ulong).mul(a.args.field("max_blocks").to(.uint)));

        const PK = pad2(DK, 128);
        const PK4 = @divExact(PK, 4);
        const PV = pad2(DV, 128);
        const PV4 = @divExact(PV, 4);
        const NW = simd_width;
        const NL = @divExact(NW, NE);
        const SH = 4 * C;

        const shmem = a.shmem_f16;
        const sq4 = shmem.add(0 * PK).to(.half4);
        const ss = shmem.add(a.sgitg.mul(SH)).add(NSG * PK).to(.float);
        const ss4 = shmem.add(a.sgitg.mul(SH)).add(NSG * PK).to(.float4);
        const sm = shmem.add(a.sgitg.mul(SH)).add(2 * C).add(NSG * PK);
        const so4 = b.localPointer(shmem.add(b.lift(2).mul(a.sgitg).mul(PV)).add(NSG * PK).add(NSG * SH).to(.float4));

        so4.assign(so4.add(a.tiisg));

        {
            const by = r.to(.ulong).mul(a.args.field("num_heads").to(.uint)).add(iq2).mul(DK).mul(2);
            q.assign(q.add(by));
            const ikv2 = iq2.div(a.args.field("num_heads").div(a.args.field("n_kv"))).to(.short);
            const k_by = ikv2.to(.ulong).mul(DK).mul(2);
            k_cache.assign(k_cache.add(k_by));
            const v_by = ikv2.to(.ulong).mul(DK).mul(2);
            v_cache.assign(v_cache.add(v_by));
        }

        const q4 = q.to(.bfloat4);
        {
            var loop = b.openFor(a.tiisg.to(.short), PK4, NW);
            {
                const i = loop.iv;
                var value = b.openIfElse(i.lt(DK4), .{.half4});
                {
                    value.yieldThen(.{q4.load(i).to(.half4)});
                }
                {
                    value.yieldElse(.{b.constant(.half4, 0)});
                }
                sq4.store(i, value.results[0]);
                loop.close();
            }
        }
        {
            var loop = b.openFor(b.constant(.short, 0), @divExact(DV4, NL), 1);
            {
                so4.store(loop.iv.mul(NL), b.constant(.float4, 0));
                loop.close();
            }
        }
        {
            var loop = b.openFor(a.tiisg.to(.short), SH / 4, NW);
            {
                ss4.store(loop.iv, b.constant(.float4, 0));
                loop.close();
            }
        }

        b.threadgroupBarrier(.mem_threadgroup);

        // When DV4/NL is a multiple of NW every lane updates the output, so the
        // `ty == 0` guard is dropped at comptime.
        const all_lanes = @rem(@divExact(DV4, NL), NW) == 0;
        const full: metal.LoopOpts = .{ .unroll = .full };
        {
            const S = b.local(.float, 0.0);
            const M = b.local(.float, b.lift(-flt_max).div(2));

            const tx = a.tiisg.rem(NL).to(.short);
            const ty = a.tiisg.div(NL).to(.short);

            const ic0_lo = kv_lo.div(C);
            const ic0 = b.local(.int, ic0_lo.add(iwg.mul(NSG)).add(a.sgitg));
            var blocks = b.openLoop(.{});
            {
                const ic = ic0.mul(C);
                var past = b.openIf(ic.ge(kv_len));
                {
                    blocks.breakLoop();
                    past.close();
                }

                var and_ = b.openAnd();
                and_.term(ic.add(a.tiisg).lt(kv_len));
                sm.store(a.tiisg, b.ternary(and_.close(ic.add(a.tiisg).ge(kv_lo)), b.constant(.half, 0), b.constant(.half, -max_half)));

                {
                    const pq4 = b.localPointer(sq4);
                    pq4.assign(pq4.add(tx));

                    const mqk = b.zeroedArray(.float, C / NE);

                    var qk = b.openForOpts(b.constant(.short, 0), C / NE, 1, full);
                    {
                        const cc = qk.iv;
                        const p = b.min(ic.add(cc), kv_len.sub(1));
                        const pk4 = b.localPointer(pagedRow(b, k_cache, pt, p, BS, PS).to(.bfloat4));
                        pk4.assign(pk4.add(ty.mul(PS).div(4).add(tx)));
                        var dots = b.openForOpts(b.constant(.short, 0), @divExact(DK4, NL), 1, full);
                        {
                            const ii = dots.iv;
                            const dot = b.dot(pk4.load(ii.mul(NL)).to(.float4), pq4.load(ii.mul(NL)).to(.float4));
                            mqk.store(cc, mqk.load(cc).add(dot));
                            dots.close();
                        }
                        mqk.store(cc, b.simdSum(mqk.load(cc)));
                        qk.close();
                    }

                    const score = b.fma(mqk.load(tx), a.args.field("scale"), sm.load(b.lift(NE).mul(tx).add(ty)).to(.float));
                    ss.store(b.lift(NE).mul(tx).add(ty), score);
                }

                b.simdgroupBarrier(.mem_threadgroup);

                {
                    const m = M.rvalue();
                    const s = ss.load(a.tiisg);

                    M.assign(b.simdMax(b.max(M.rvalue(), s)));

                    const ms = b.exp(m.sub(M));
                    const vs = b.exp(s.sub(M));

                    S.assign(S.mul(ms).add(b.simdSum(vs)));

                    ss.store(a.tiisg, vs);

                    var lanes = if (all_lanes) null else b.openIf(ty.eq(0));
                    var loop = b.openForOpts(b.constant(.short, 0), @divExact(DV4, NL), 1, full);
                    {
                        const at = loop.iv.mul(NL);
                        so4.store(at, so4.load(at).mul(ms));
                        loop.close();
                    }
                    if (lanes) |*l| l.close();
                }

                b.simdgroupBarrier(.mem_threadgroup);

                {
                    const lo = b.array(.float4, @intCast(@divExact(DV4, NL)));
                    {
                        var loop = b.openForOpts(b.constant(.short, 0), @divExact(DV4, NL), 1, full);
                        {
                            lo.store(loop.iv, b.constant(.float4, 0));
                            loop.close();
                        }
                    }

                    const sst = ss.add(ty);

                    var pv = b.openForOpts(b.constant(.short, 0), C / NE, 1, full);
                    {
                        const cc = pv.iv;
                        const p = b.min(ic.add(cc), kv_len.sub(1));
                        const pv4 = b.localPointer(pagedRow(b, v_cache, pt, p, BS, PS).to(.bfloat4));
                        pv4.assign(pv4.add(ty.mul(PS).div(4).add(tx)));
                        var loop = b.openForOpts(b.constant(.short, 0), @divExact(DV4, NL), 1, full);
                        {
                            const ii = loop.iv;
                            const weighted = pv4.load(ii.mul(NL)).to(.float4).mul(b.vec(.float4, .{sst.load(cc.mul(NE))}));
                            lo.store(ii, lo.load(ii).add(weighted));
                            loop.close();
                        }
                        pv.close();
                    }

                    var lanes = if (all_lanes) null else b.openIf(ty.eq(0));
                    var loop = b.openForOpts(b.constant(.short, 0), @divExact(DV4, NL), 1, full);
                    {
                        const ii = loop.iv;
                        const partial = lo.load(ii);
                        so4.store(ii.mul(NL), so4.load(ii.mul(NL)).add(partial));
                        loop.close();
                    }
                    if (lanes) |*l| l.close();
                }
                blocks.next();
                ic0.assign(ic0.add(NWG * NSG));
                blocks.close();
            }

            var lane0 = b.openIf(a.tiisg.eq(0));
            {
                ss.store(0, S);
                ss.store(1, M);
                lane0.close();
            }
        }

        so4.assign(so4.sub(a.tiisg));

        b.threadgroupBarrier(.mem_threadgroup);

        {
            const rr = b.local(.short, @divFloor(NSG, 2));
            var halving = b.openLoop(.{});
            {
                halving.body(rr.gt(0));
            }
            {
                {
                    var group = b.openIf(a.sgitg.lt(rr));
                    {
                        const S0 = ss.load(0);
                        const S1 = ss.load(rr.mul(@divExact(SH, 2)).add(0));

                        const M0 = ss.load(1);
                        const M1 = ss.load(rr.mul(@divExact(SH, 2)).add(1));

                        const M_ = b.max(M0, M1);

                        const ms0 = b.exp(M0.sub(M_));
                        const ms1 = b.exp(M1.sub(M_));

                        const S_ = S0.mul(ms0).add(S1.mul(ms1));

                        var lane0 = b.openIf(a.tiisg.eq(0));
                        {
                            ss.store(0, S_);
                            ss.store(1, M_);
                            lane0.close();
                        }

                        var loop = b.openFor(a.tiisg.to(.short), DV4, NW);
                        {
                            const i = loop.iv;
                            so4.store(i, so4.load(i).mul(ms0).add(so4.load(i.add(rr.mul(PV4))).mul(ms1)));
                            loop.close();
                        }
                        group.close();
                    }
                }
                b.threadgroupBarrier(.mem_threadgroup);
                rr.assign(rr.shr(1));
                halving.close();
            }
        }

        var first = b.openIf(a.sgitg.eq(0));
        {
            const rid = r.to(.long).mul(a.args.field("num_heads").to(.uint)).add(iq2);
            if (NWG == 1) {
                var norm = b.openIfElse(ss.load(0).eq(0.0), .{.float});
                {
                    norm.yieldThen(.{0.0});
                }
                {
                    norm.yieldElse(.{b.lift(1.0).div(ss.load(0))});
                }
                const S = norm.results[0];
                const dst4 = a.dst.to(.bfloat4);
                var loop = b.openFor(a.tiisg.to(.short), DV4, NW);
                {
                    const out = so4.load(loop.iv).mul(S).to(.bfloat4);
                    dst4.store(rid.mul(DV4).add(loop.iv), out);
                    loop.close();
                }
            } else {
                const dst4 = a.dst.to(.float4);
                const dst1 = a.dst.to(.float).add(nrows_static.mul(DV).mul(NWG));
                var loop = b.openFor(a.tiisg.to(.short), DV4, NW);
                {
                    const part = so4.load(loop.iv);
                    dst4.store(rid.mul(DV4).mul(NWG).add(b.lift(NWG).mul(loop.iv)).add(iwg), part);
                    loop.close();
                }
                var lane0 = b.openIf(a.tiisg.eq(0));
                {
                    const sum = ss.load(0);
                    dst1.store(rid.mul(2 * NWG).add(b.lift(2).mul(iwg)).add(0), sum);
                    const max = ss.load(1);
                    dst1.store(rid.mul(2 * NWG).add(b.lift(2).mul(iwg)).add(1), max);
                    lane0.close();
                }
            }
            first.close();
        }
    }

    /// The address of position `p`'s bf16 row in `cache`, through the page
    /// table `pt`.
    fn pagedRow(b: *Builder, cache: Value, pt: Value, p: Value, BS: i32, PS: i32) Value {
        const page = pt.load(p.div(BS)).to(.uint).to(.ulong).mul(b.constant(.uint, BS * PS));
        const row = p.rem(BS).to(.uint).mul(b.constant(.uint, PS));
        return cache.add(page.add(row).mul(2));
    }
};

/// fa_vec_paged_reduce: merges the `nwg` partials of a split-K fa_vec_paged,
/// one threadgroup per row and head.
pub const FaVecPagedReduce = struct {
    pub const Cfg = struct {
        /// The partials to merge, FaVecPaged's `nwg`.
        nwg: i32,
        /// The head dimension.
        head_dim: i32,
    };

    pub const Args: metal.Struct = .{ .name = "fa_vec_paged_reduce_args", .fields = &.{
        .{ .name = "nrows", .type = .int },
    } };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "fa_vec_paged_reduce",
        .inputs = &.{"htmp"},
        .outputs = &.{"dst"},
        .constants = &.{"args"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .args = .{ .constant_struct = Args },
            .htmp = .{ .buffer = .{ .type = .char, .const_ = true } },
            .dst = .{ .buffer = .{ .type = .char } },
            .tgpig = .{ .threadgroup_position_in_grid = .uint },
            .tiisg = .{ .thread_index_in_simdgroup = .ushort },
            .sgitg = .{ .simdgroup_index_in_threadgroup = .ushort },
        });
        const nwg = cfg.nwg;
        const DV = cfg.head_dim;
        const rid = a.tgpig.to(.ulong);
        const iwg = a.tiisg.to(.short);
        const ss = a.htmp.to(.float).add(a.args.field("nrows").to(.ulong).mul(DV).mul(nwg));

        var max = b.openIfElse(iwg.lt(nwg), .{.float});
        {
            max.yieldThen(.{ss.load(rid.mul(2 * nwg).add(b.lift(2).mul(iwg)).add(1))});
        }
        {
            max.yieldElse(.{b.lift(-flt_max).div(2)});
        }
        const M = b.local(.float, max.results[0]);
        var sum = b.openIfElse(iwg.lt(nwg), .{.float});
        {
            sum.yieldThen(.{ss.load(rid.mul(2 * nwg).add(b.lift(2).mul(iwg)).add(0))});
        }
        {
            sum.yieldElse(.{0.0});
        }
        const S = b.local(.float, sum.results[0]);

        const m = b.simdMax(M.rvalue());
        const ms = b.exp(M.sub(m));
        S.assign(b.simdSum(S.mul(ms)));
        var norm = b.openIfElse(S.eq(0.0), .{.float});
        {
            norm.yieldThen(.{0.0});
        }
        {
            norm.yieldElse(.{b.lift(1.0).div(S)});
        }
        S.assign(norm.results[0]);

        const DV4 = @divTrunc(DV, 4);
        const htmp4 = a.htmp.to(.float4).add(rid.mul(DV4).mul(nwg));
        const dst4 = a.dst.to(.bfloat4).add(rid.mul(DV4));
        var loop = b.openFor(a.sgitg.to(.int), DV4, nwg);
        {
            const i = loop.iv;
            var part_ = b.openIfElse(iwg.lt(nwg), .{.float4});
            {
                part_.yieldThen(.{htmp4.load(i.mul(nwg).add(iwg)).mul(ms)});
            }
            {
                part_.yieldElse(.{b.constant(.float4, 0)});
            }
            const part = part_.results[0];
            const v = b.simdSum(part);
            var lane0 = b.openIf(iwg.eq(0));
            {
                dst4.store(i, v.mul(S).to(.bfloat4));
                lane0.close();
            }
            loop.close();
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
        const i = b.local(.int, a.tpt.x().to(.int));
        var loop = b.openLoop(.{});
        {
            loop.body(i.lt(a.row_stride));
        }
        {
            dst.store(i, b.constant(cfg.dtype, 0));
            i.assign(i.add(a.ntg.x().to(.int)));
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
            64, 96, 128 => .{ 32, 32, 128 },
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

        const seq_idx = b.local(.int, null);
        {
            const lo = b.local(.int, 0);
            const hi = b.local(.int, a.num_seqs);
            var search = b.openLoop(.{});
            {
                search.body(lo.lt(hi));
            }
            {
                const mid = b.local(.int, lo.add(hi).add(1).div(2));
                var half = b.openIfElse(a.cu_seqlens_q.load(mid).div(BQ).add(mid).le(q_block_global_idx), .{});
                {
                    lo.assign(mid);
                    half.elseBranch();
                }
                {
                    hi.assign(mid.sub(1));
                    half.close();
                }
                search.close();
            }
            seq_idx.assign(lo);
        }
        // The grid is an upper bound on the query blocks: the extra ones land past
        // the last sequence and clamp to it.
        {
            var clamp = b.openIfElse(seq_idx.lt(a.num_seqs.sub(1)), .{.int});
            {
                clamp.yieldThen(.{seq_idx});
            }
            {
                clamp.yieldElse(.{a.num_seqs.sub(1)});
            }
            seq_idx.assign(clamp.results[0]);
        }

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
        {
            var loop = b.openFor(thread_idx, BQ * HEAD_SIZE, NUM_THREADS);
            {
                const i = loop.iv;
                const r = i.div(HEAD_SIZE);
                const d = i.rem(HEAD_SIZE);
                var value = b.openIfElse(r.lt(valid_q), .{T});
                {
                    value.yieldThen(.{q_base.load(r.mul(a.q_stride).add(d))});
                }
                {
                    value.yieldElse(.{b.constant(T, 0)});
                }
                const q_elem = value.results[0];
                Q_smem.store(r.mul(LD).add(d), q_elem);
                loop.close();
            }
        }

        b.threadgroupBarrier(.mem_threadgroup);

        const T2: Type = .vec(T.scalar, 2);
        const q_reg = b.array(T2, @intCast(TD));
        {
            var loop = b.openForOpts(0, TD, 1, unroll);
            {
                const d = loop.iv;
                const tmp = b.simdgroupLoad(Q_smem.add(a.sg_idx.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
                q_reg.store(d, tmp.threadElements());
                loop.close();
            }
        }

        const fc = fragCoord(b, a.lane.to(.ushort));
        const fm = fc.y();
        const fn_ = fc.x();

        const max_score = b.local(.float, -std.math.inf(f32));
        const sum_score = b.local(.float, 0.0);

        const o_reg = b.array(.float2, @intCast(TD));
        {
            var loop = b.openForOpts(0, TD, 1, unroll);
            {
                o_reg.store(loop.iv, b.constant(.float2, 0));
                loop.close();
            }
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
            {
                var loop = b.openForOpts(0, TK, 1, unroll);
                {
                    s_reg.store(loop.iv, b.constant(.float2, 0));
                    loop.close();
                }
            }

            {
                var dims = b.openForOpts(0, TD, 1, unroll);
                {
                    const d = dims.iv;
                    const q_frag = b.matrix(T.scalar, q_reg.load(d));
                    var keys = b.openForOpts(0, TK, 1, unroll);
                    {
                        const k = keys.iv;
                        const k_frag = b.simdgroupLoad(K_smem.add(k.mul(8).mul(LD)).add(d.mul(8)), LD, .{ .transpose = true });
                        const s_frag = b.matrix(.float, s_reg.load(k));
                        s_reg.store(k, b.simdgroupMultiplyAccumulate(q_frag, k_frag, s_frag).threadElements());
                        keys.close();
                    }
                    dims.close();
                }
            }

            const q_abs_pos = context_len.add(q_pos_start).add(a.sg_idx.mul(8)).add(fm).to(.int);
            const row_masked = b.local(.bool, a.sg_idx.mul(8).add(fm).ge(valid_q));
            const min_q_abs_pos = context_len.add(q_pos_start);
            var in_tile = b.openAnd();
            if (cfg.causal) in_tile.term(tile_start.add(TILE_KV).sub(1).lt(min_q_abs_pos));
            in_tile.term(tile_start.add(TILE_KV).le(seq_len));
            in_tile.term(a.softcapping.le(0.0));
            const tile_no_mask = b.local(.bool, in_tile.close(a.sliding_window.lt(0)));

            {
                var c = b.condition();
                c.and_(tile_no_mask);
                var masking = c.openIfElseNot(row_masked, .{});
                {
                    var loop = b.openForOpts(0, TK, 1, unroll);
                    {
                        const k = loop.iv;
                        setLane(s_reg, k, 0, s_reg.load(k).at(0).mul(scale_log2));
                        setLane(s_reg, k, 1, s_reg.load(k).at(1).mul(scale_log2));
                        loop.close();
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
                            var any = b.openOr();
                            any.term(row_masked);
                            if (cfg.causal) any.term(kv_pos.gt(q_abs_pos));
                            const masked = b.local(.bool, any.close(kv_pos.ge(seq_len)));
                            var window = b.openIf(a.sliding_window.ge(0));
                            {
                                var before = b.openOr();
                                before.term(masked);
                                masked.assign(before.close(kv_pos.lt(q_abs_pos.add(1).sub(a.sliding_window))));
                                window.close();
                            }
                            setLane(s_reg, k, jj, b.ternary(masked, -std.math.inf(f32), s));
                            pair.close();
                        }
                        keys.close();
                    }
                    masking.close();
                }
            }

            const local_max = b.local(.float, -std.math.inf(f32));
            {
                var loop = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = loop.iv;
                    local_max.assign(b.max(local_max.rvalue(), b.max(s_reg.load(k).at(0), s_reg.load(k).at(1))));
                    loop.close();
                }
            }
            const row_max = fragRowReduce(b, .max, local_max);

            const new_max = b.local(.float, b.max(max_score.rvalue(), row_max));
            const factor = b.local(.float, null);
            var grew = b.openIfElse(new_max.gt(max_score), .{});
            {
                var first = b.openIfElse(max_score.eq(-std.math.inf(f32)), .{.float});
                {
                    first.yieldThen(.{0.0});
                }
                {
                    first.yieldElse(.{b.fast().exp2(max_score.sub(new_max))});
                }
                factor.assign(first.results[0]);
                grew.elseBranch();
            }
            {
                factor.assign(1.0);
                var empty = b.openIf(max_score.eq(-std.math.inf(f32)));
                {
                    new_max.assign(0.0);
                    empty.close();
                }
                grew.close();
            }
            max_score.assign(new_max);

            const local_sum = b.local(.float, 0.0);
            {
                var keys = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = keys.iv;
                    var pair = b.openForOpts(0, 2, 1, unroll);
                    {
                        const jj = pair.iv;
                        var exp = b.openIfElse(s_reg.load(k).at(jj).eq(-std.math.inf(f32)), .{.float});
                        {
                            exp.yieldThen(.{0.0});
                        }
                        {
                            exp.yieldElse(.{b.fast().exp2(s_reg.load(k).at(jj).sub(new_max))});
                        }
                        const p = b.local(.float, exp.results[0]);
                        setLane(s_reg, k, jj, p.rvalue());
                        local_sum.assign(local_sum.add(p));
                        pair.close();
                    }
                    keys.close();
                }
            }
            const row_sum = fragRowReduce(b, .sum, local_sum);

            sum_score.assign(sum_score.mul(factor).add(row_sum));

            {
                var loop = b.openForOpts(0, TD, 1, unroll);
                {
                    o_reg.store(loop.iv, o_reg.load(loop.iv).mul(factor));
                    loop.close();
                }
            }

            {
                var keys = b.openForOpts(0, TK, 1, unroll);
                {
                    const k = keys.iv;
                    const p_frag = b.matrix(.float, s_reg.load(k));
                    var dims = b.openForOpts(0, TD, 1, unroll);
                    {
                        const d = dims.iv;
                        const v_frag = b.simdgroupLoad(V_smem.add(k.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
                        const o_frag = b.matrix(.float, o_reg.load(d));
                        o_reg.store(d, b.simdgroupMultiplyAccumulate(p_frag, v_frag, o_frag).threadElements());
                        dims.close();
                    }
                    keys.close();
                }
            }

            b.threadgroupBarrier(.mem_threadgroup);
            tiles.close();
        }

        const inv_sum = b.lift(1.0).div(sum_score.add(@as(f32, 1e-6)));
        {
            var loop = b.openForOpts(0, TD, 1, unroll);
            {
                o_reg.store(loop.iv, o_reg.load(loop.iv).mul(inv_sum));
                loop.close();
            }
        }

        const O_smem = Q_smem.to(.float);
        {
            var loop = b.openForOpts(0, TD, 1, unroll);
            {
                const d = loop.iv;
                const o_frag = b.matrix(.float, o_reg.load(d));
                b.simdgroupStore(o_frag, O_smem.add(a.sg_idx.mul(8).mul(LD)).add(d.mul(8)), LD, .{});
                loop.close();
            }
        }

        b.threadgroupBarrier(.mem_threadgroup);

        const out_base = a.out.add(q_seq_start.add(q_pos_start).mul(a.q_stride)).add(head_idx.mul(HEAD_SIZE));
        const total = valid_q.mul(HEAD_SIZE);
        var loop = b.openFor(thread_idx, total, NUM_THREADS);
        {
            const i = loop.iv;
            const r = i.div(HEAD_SIZE);
            const d = i.rem(HEAD_SIZE);
            const value = O_smem.load(r.mul(LD).add(d)).to(T);
            out_base.store(r.mul(a.q_stride).add(d), value);
            loop.close();
        }
    }
};

test "the paged attention kernels emit verified modules" {
    const allocator = std.testing.allocator;
    for ([_]i32{ 128, 256, 512 }) |head_dim| {
        for ([_]i32{ 4, 8, 16 }) |nsg| {
            allocator.free(try FaVecPaged.Kernel.emit(allocator, .{ .head_dim = head_dim, .nsg = nsg, .block_size = 16, .position_stride = 2 * head_dim, .nwg = 1 }));
        }
    }
    allocator.free(try FaVecPaged.Kernel.emit(allocator, .{ .head_dim = 512, .nsg = 8, .block_size = 16, .position_stride = 1024, .nwg = 4 }));
    allocator.free(try FaVecPagedReduce.Kernel.emit(allocator, .{ .nwg = 4, .head_dim = 512 }));
    for ([_]Type{ .half, .bfloat }) |dtype| {
        allocator.free(try ZeroPaddingRows.Kernel.emit(allocator, .{ .dtype = dtype }));
        for ([_]i32{ 64, 96, 128, 256, 512 }) |head_size| {
            const tile = PagedAttentionTiled.tileConfig(head_size).?;
            for ([_]bool{ true, false }) |causal| {
                allocator.free(try PagedAttentionTiled.Kernel.emit(allocator, .{ .dtype = dtype, .head_size = head_size, .block_size = 16, .bq = tile[0], .tile_kv = tile[1], .num_threads = tile[2], .causal = causal }));
            }
        }
    }
}
