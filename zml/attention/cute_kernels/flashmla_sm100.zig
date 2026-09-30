//! Sparse MLA decode / prefill for DeepSeek-V4.1 on SM100: persistent CTAs,
//! each processing query tokens t = cta, cta + ctas, ... (one token per
//! cluster when a token's key blocks are split across a cluster).
//!
//! Swapped-operand formulation (the 16 real heads are the MMA N dimension):
//!   S^T [64 keys, 16 heads] = K_blk [64, 512] . Q^T         tcgen05 M=64  N=16, K-major A/B
//!   O^T [512, 16]          += V_blk^T [512, 64] . P^T      tcgen05 M=128 N=16, MN-major A
//! K and V are the same dequantized BF16 tile (SW128 K-major, 64 KiB), read
//! through two descriptors. Nothing is materialized in global memory and there
//! is no split-KV reduction (a cluster merges its partials through DSMEM).
//!
//! Warp roles (384 threads):
//!   0-3   softmax (lazy rescale, one barrier vote per block), O rescale (TMEM), epilogue
//!   4     MMA issue (one elected lane)
//!   5-6   loaders: cp.async of the raw rows and scales of each key block (zero-fill
//!         for invalid rows)
//!   7     front: per token, validated indices and block counts (one token ahead), Q
//!   8-11  FP8/FP4 -> BF16 dequantization into the tile, in 128-column parts
//! Key blocks are numbered across the CTA's tokens: block g uses buffer g & 1
//! with parity (g >> 1) & 1. Per-token state (indices, Q, O in TMEM) is
//! double-buffered by token parity so consecutive tokens overlap.
const std = @import("std");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;
const ptx = @import("ptx.zig");

pub const Cfg = struct {
    queries: i64,
    window_capacity: i64,
    compressed_capacity: i64,
    window_slots: i64,
    compressed_slots: i64,
    scale: f64,
    /// CTAs per query token (a thread-block cluster): each takes a contiguous
    /// range of the token's key blocks and the partial results are merged
    /// through distributed shared memory, inside the kernel. Power of two.
    /// Clusters of more than one CTA require `ctas == queries`.
    cluster: i64 = 1,
    /// Persistent CTAs (clusters): token t is processed by CTA t % ctas.
    ctas: i64 = 0,

    fn gridCtas(c: Cfg) i64 {
        return if (c.ctas > 0) @min(c.ctas, c.queries) else c.queries;
    }
};

pub const Program = cute.Program(Cfg, .{
    .name = "flashmla_sm100",
    .inputs = &.{ "q", "wv", "ws", "cv", "cs", "wi", "ci", "lengths", "sink", "active" },
    .outputs = &.{"out"},
    .run = program,
});

const kernel_name = "flashmla_sm100_kernel";
const H = 16;
const D = 512;
const BK = 64;
const threads = 384;
const log2e = 1.4426950408889634;
/// Lazy rescale: the running max only moves when a block exceeds it by this
/// many log2 units, so P stays below 2^32 (BF16 P, F32 l and O keep their
/// relative precision) and O is rarely rescaled.
const rescale_threshold = 32.0;

// Shared memory plan (bytes).
const smem = struct {
    const tile = [2]i32{ 0, 65536 };
    /// Q of token parity kb (SW128 K-major, the B operand of QK).
    const q = [2]i32{ 131072, 147456 };
    /// Raw (quantized) rows: a ring of slots of 64 rows x 288 bytes (see `rawBase`).
    /// An FP4 block takes one slot (256 data bytes, then the 32 scale bytes of each
    /// row); an FP8 block takes two (row bytes 0-255, then 256-511; its 16 scale
    /// bytes follow the second slot's data).
    const raw_slot = 64 * 288;
    const scale_off = 256;
    const p = [2]i32{ 221184, 223232 };
    const red = [2]i32{ 225280, 225536 };
    const red_sum = [2]i32{ 225792, 226048 };
    const bars = 226304;
    const holder = bars + 8 * 40;
    /// Per token parity: physical row per candidate (or -1), window [128] then
    /// compressed [512], followed by the token's block counts (window, compressed).
    const idx = holder + 16;
    const idx_bytes = 640 * 4 + 16;
    const counts = 640 * 4;
    /// Sink logits (f32 [16], prefetched at start).
    const sink = idx + 2 * idx_bytes;
    const total = sink + H * 4;
    /// Output staging, bf16 [16 heads][512] (reuses Q after the CTA's last QK).
    const ostage = q[0];
    /// Cluster merge (after every block of the cluster is done), in the raw ring:
    /// per-head max and sum of this CTA, merge weights [16 CTAs][16 heads], and the
    /// inbox of (m, l) per head from every rank [16 ranks][32] f32.
    const stats = 163840;
    const weights = stats + 2 * H * 4;
    const stats_in = weights + 16 * H * 4;
    /// Cluster merge: O^T [512][16] f32 of this CTA (reuses tile 0 after the last PV),
    /// and the inbox of the dims this rank owns, [ranks][512 / C][16] f32 (tile 0, upper half).
    const obuf = 0;
    const inbox = 32768;
};

const Bar = enum(i32) {
    raw_full = 0, // [4]
    raw_empty = 4, // [4]
    tile_full = 8,
    s_full = 10,
    p_full = 12,
    pv_done = 14,
    idx_ready = 16,
    idx_free = 18,
    o_free = 20,
    q_free = 22,
    q_ready = 24,
};

/// Arrivals on idx_free: softmax warps (4), MMA (1), loaders (2), dequant warps (4).
const idx_consumers = 11;

fn bar(b: *B, base: V, comptime which: Bar, buf: anytype) V {
    return base.add(smem.bars + 8 * @intFromEnum(which)).add(b.lift(buf).mul(8));
}

// TMEM columns: O^T of token parity ob in 64 * ob (4 chunks of 128 dims x 16
// heads); S^T of buffer `buf` in 128 + 64 * buf: four partial sums over 128
// latent dims each, 16 columns apart (independent accumulators so the QK MMAs pipeline).
const tmem_cols = 256;
const tmem_s = 128;
const qk_parts = 4;

const Format = enum {
    fp8,
    fp4,
    fn rowBytes(f: Format) i32 {
        return if (f == .fp8) 512 else 256;
    }
    fn scaleBytes(f: Format) i32 {
        return if (f == .fp8) 16 else 32;
    }
};

const Globals = struct {
    q: V,
    wv: V,
    ws: V,
    cv: V,
    cs: V,
    wi: V,
    ci: V,
    lengths: V,
    sink: V,
    active: V,
    out: V,
};

const Ctx = struct {
    b: *B,
    cfg: Cfg,
    g: Globals,
    s: V, // shared base
    tid: V,
    warp: V,
    lane: V,
    taddr: V,
    rank: V, // CTA rank in the token's cluster
    cta: V, // cluster index: first token
    ntok: V, // tokens of this CTA
    // Per token (see `forToken`).
    k: V = undefined, // token ordinal within the CTA
    t: V = undefined, // query token
    kb: V = undefined, // token parity (index buffer, O buffer)
    ib: V = undefined, // shared address of the token's index buffer
    nwb: V = undefined, // window blocks
    ncb: V = undefined, // compressed blocks
    b0: V = undefined, // this CTA's key blocks [b0, b1) of the token
    b1: V = undefined,
    n: V = undefined, // b1 - b0; local block j is block b0 + j of the token
    nwl: V = undefined, // local window (FP8) blocks: the first nwl local blocks
    g0: V = undefined, // CTA-wide number of the token's first local block
    slot0: V = undefined, // CTA-wide raw-slot use of the token's first local block
};

/// The per-token context of token ordinal k; `counts` waits for the token's
/// index buffer and loads the block counts.
fn forToken(c0: Ctx, k: V, g0: V, slot0: V, comptime counts: bool) Ctx {
    const b = c0.b;
    var c = c0;
    c.k = k;
    c.t = c0.cta.add(k.mul(@as(i32, @intCast(c0.cfg.gridCtas()))));
    c.kb = k.bitAnd(1);
    c.ib = c0.s.add(smem.idx).add(c.kb.mul(smem.idx_bytes));
    c.g0 = g0;
    c.slot0 = slot0;
    if (counts) {
        ptx.mbarWait(b, bar(b, c.s, .idx_ready, c.kb), parity(k));
        const nc = ptx.ldSharedV2(b, c.ib.add(smem.counts));
        c.nwb = nc[0];
        c.ncb = nc[1];
        const nb = c.nwb.add(c.ncb);
        const log_c: i32 = @intCast(std.math.log2_int(u64, @intCast(c0.cfg.cluster)));
        c.b0 = c0.rank.mul(nb).shrLogical(log_c);
        c.b1 = c0.rank.add(1).mul(nb).shrLogical(log_c);
        c.n = c.b1.sub(c.b0);
        c.nwl = c.b1.minimum(c.nwb).sub(c.b0).maximum(0);
    }
    return c;
}

/// This warp is done with the token's index buffer.
fn releaseIndices(c: Ctx) void {
    const b = c.b;
    ptx.syncWarp(b);
    var l0 = b.openIf(c.lane.eq(0));
    ptx.mbarArrive(b, bar(b, c.s, .idx_free, c.kb));
    l0.yieldThen(.{});
}

/// First raw-slot use of local block j: FP8 (window) blocks come first and use two.
fn slotUse(c: Ctx, j: V) V {
    return c.slot0.add(j).add(j.minimum(c.nwl));
}

/// With one token per CTA the second Q buffer is unused: the ring starts there and
/// has 4 slots (two FP8 blocks in flight); otherwise 3 slots after it.
fn oneTokenPerCta(cfg: Cfg) bool {
    return cfg.gridCtas() == cfg.queries;
}

fn rawBase(cfg: Cfg) i32 {
    return if (oneTokenPerCta(cfg)) smem.q[1] else smem.q[1] + 16384;
}

fn rawSlots(cfg: Cfg) i32 {
    return if (oneTokenPerCta(cfg)) 4 else 3;
}

fn slotAddr(c: Ctx, use: V) V {
    return c.s.add(rawBase(c.cfg)).add(use.rem(rawSlots(c.cfg)).mul(smem.raw_slot));
}

fn scaleAddr(c: Ctx, use: V) V {
    return slotAddr(c, use).add(smem.scale_off);
}

fn slotParity(c: Ctx, use: V) V {
    return use.div(rawSlots(c.cfg)).bitAnd(1);
}

fn parity(j: V) V {
    return j.shrLogical(1).bitAnd(1);
}

fn tupleOf(comptime n: usize, arr: [n]V) std.meta.Tuple(&([_]type{V} ** n)) {
    var t: std.meta.Tuple(&([_]type{V} ** n)) = undefined;
    inline for (0..n) |i| t[i] = arr[i];
    return t;
}

fn program(b: *B, cfg: Cfg) cute.FinishError!void {
    b.beginFunction(kernel_name, .cuda_kernel);
    try kernel(b, cfg);
    b.endFunction(.{ threads, 1, 1 });

    b.beginFunction(Program.name, .host);
    const h = try b.declareArgs(.{
        .q = .{ .ptr = cute.DType.bf16 },
        .wv = .{ .ptr = cute.DType.i8 },
        .ws = .{ .ptr = cute.DType.i8 },
        .cv = .{ .ptr = cute.DType.i8 },
        .cs = .{ .ptr = cute.DType.i8 },
        .wi = .{ .ptr = cute.DType.i32 },
        .ci = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .active = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = cute.DType.bf16 },
    });
    launch(b, cfg, .{ h.q, h.wv, h.ws, h.cv, h.cs, h.wi, h.ci, h.lengths, h.sink, h.active, h.out });
    b.endFunction(null);
}

fn launch(b: *B, cfg: Cfg, args: anytype) void {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, @as(i32, @intCast(cfg.gridCtas() * cfg.cluster))), one, one },
        .block = .{ b.cst(.i32, threads), one, one },
        .dynamic_smem = b.kernelSmemSize(kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ b.cst(.i32, @as(i32, @intCast(cfg.cluster))), one, one },
        .use_pdl = true,
    });
    const status = b.launchEx(kernel_name, config, args);
    b.returnHostStatus(b.cudaResultStatus(status));
}

fn kernel(b: *B, cfg: Cfg) cute.FinishError!void {
    const g = try globalsOf(b, try b.declareArgs(.{
        .q = .{ .ptr = cute.DType.bf16 },
        .wv = .{ .ptr = cute.DType.i8 },
        .ws = .{ .ptr = cute.DType.i8 },
        .cv = .{ .ptr = cute.DType.i8 },
        .cs = .{ .ptr = cute.DType.i8 },
        .wi = .{ .ptr = cute.DType.i32 },
        .ci = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .active = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = cute.DType.bf16 },
    }));
    try kernelBody(b, cfg, g);
}

fn globalsOf(b: *B, a: anytype) !Globals {
    b.setFunctionAttribute("nvvm.minctasm", .int(b.ctx, .i32, 1));
    b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}"));

    return .{
        .q = ptx.globalAddress(b, a.q),
        .wv = ptx.globalAddress(b, a.wv),
        .ws = ptx.globalAddress(b, a.ws),
        .cv = ptx.globalAddress(b, a.cv),
        .cs = ptx.globalAddress(b, a.cs),
        .wi = ptx.globalAddress(b, a.wi),
        .ci = ptx.globalAddress(b, a.ci),
        .lengths = ptx.globalAddress(b, a.lengths),
        .sink = ptx.globalAddress(b, a.sink),
        .active = ptx.globalAddress(b, a.active),
        .out = ptx.globalAddress(b, a.out),
    };
}

fn kernelBody(b: *B, cfg: Cfg, g: Globals) cute.FinishError!void {
    const tid = b.threadIdx().x;
    const warp = b.makeWarpUniform(tid.shrLogical(5));
    const lane = tid.bitAnd(31);
    std.debug.assert(std.math.isPowerOfTwo(cfg.cluster) and cfg.cluster <= 16);
    std.debug.assert(cfg.cluster == 1 or cfg.gridCtas() == cfg.queries);
    const log_c: i32 = @intCast(std.math.log2_int(u64, @intCast(cfg.cluster)));
    const cta = b.blockIdx().x.shrLogical(log_c);
    const stride: i32 = @intCast(cfg.gridCtas());
    const queries: i32 = @intCast(cfg.queries);
    const rank = if (cfg.cluster > 1) ptx.clusterCtaRank(b) else b.cst(.i32, 0);
    const s = ptx.sharedStorage(b, smem.total);

    // Inputs come from the previous kernels of the layer.
    b.waitForDependency();
    const c0: Ctx = .{
        .b = b,
        .cfg = cfg,
        .g = g,
        .s = s,
        .tid = tid,
        .warp = warp,
        .lane = lane,
        .taddr = undefined,
        .rank = rank,
        .cta = cta,
        .ntok = cta.sub(queries).neg().add(stride - 1).div(stride),
    };
    // The front warp issues the first token's loads right away, overlapping the setup
    // (other warps load harmless in-bounds addresses and discard them).
    const first = frontLoad(c0, b.select(warp.eq(7), cta, b.cst(.i32, 0)));
    {
        var w7 = b.openIf(warp.eq(7));
        issueQ(c0, cta, b.cst(.i32, 0));
        w7.yieldThen(.{});
    }
    {
        var t0 = b.openIf(tid.eq(0));
        for (0..@intCast(rawSlots(cfg))) |i| {
            ptx.mbarInit(b, bar(b, s, .raw_full, @as(i32, @intCast(i))), 64);
            ptx.mbarInit(b, bar(b, s, .raw_empty, @as(i32, @intCast(i))), 128);
        }
        for (0..2) |i| {
            ptx.mbarInit(b, bar(b, s, .tile_full, @as(i32, @intCast(i))), 128);
            ptx.mbarInit(b, bar(b, s, .s_full, @as(i32, @intCast(i))), 1);
            ptx.mbarInit(b, bar(b, s, .p_full, @as(i32, @intCast(i))), 128);
            ptx.mbarInit(b, bar(b, s, .pv_done, @as(i32, @intCast(i))), 1);
            ptx.mbarInit(b, bar(b, s, .idx_ready, @as(i32, @intCast(i))), 32);
            ptx.mbarInit(b, bar(b, s, .idx_free, @as(i32, @intCast(i))), idx_consumers);
            ptx.mbarInit(b, bar(b, s, .o_free, @as(i32, @intCast(i))), 128);
            ptx.mbarInit(b, bar(b, s, .q_ready, @as(i32, @intCast(i))), 32);
            ptx.mbarInit(b, bar(b, s, .q_free, @as(i32, @intCast(i))), 1);
        }
        ptx.fenceMbarInit(b);
        t0.yieldThen(.{});
    }
    {
        var w0 = b.openIf(warp.eq(0));
        ptx.tmemAlloc(b, s.add(smem.holder), tmem_cols);
        ptx.tmemRelinquish(b);
        w0.yieldThen(.{});
    }

    ptx.tcgenFenceBefore(b);
    ptx.barSync(b, 0, threads);
    ptx.tcgenFenceAfter(b);
    var c = c0;
    c.taddr = ptx.ldSharedU32(b, s.add(smem.holder));
    {
        var role = b.openIf(warp.lt(4));
        softmaxRole(c);
        role.yieldThen(.{});
    }
    {
        var role = b.openIf(warp.eq(4));
        mmaRole(c);
        role.yieldThen(.{});
    }
    {
        var role = b.openIf(warp.eq(5).bitOr(warp.eq(6)));
        loaderRole(c);
        role.yieldThen(.{});
    }
    {
        var role = b.openIf(warp.eq(7));
        frontRole(c, first);
        role.yieldThen(.{});
    }
    {
        var role = b.openIf(warp.ge(8));
        dequantRole(c);
        role.yieldThen(.{});
    }

    ptx.tcgenFenceBefore(b);
    ptx.barSync(b, 0, threads);
    {
        var w0 = b.openIf(warp.eq(0));
        ptx.tcgenFenceAfter(b);
        ptx.tmemDealloc(b, c.taddr, tmem_cols);
        w0.yieldThen(.{});
    }
    if (cfg.cluster > 1) {
        // One token per cluster.
        var cm = c;
        cm.t = cta;
        clusterMerge(cm);
    }
    b.launchDependents();
}

// ---- front (warp 7): indices, lengths and Q of each token ---------------------------------

const FrontRegs = struct {
    active: V,
    lens: [2]V,
    w: [4]V, // this lane's 4 window indices
    c: [16]V, // this lane's 16 compressed indices
};

/// Issue the global loads of token t's lengths and indices (lane l holds
/// candidates [4 l, 4 l + 4) of the window and [16 l, 16 l + 16) of the compressed stream).
fn frontLoad(c: Ctx, t: V) FrontRegs {
    const b = c.b;
    var r: FrontRegs = undefined;
    r.active = ptx.ldGlobalU32(b, c.g.active);
    r.lens = ptx.ldGlobalV2(b, c.g.lengths.add(t.to(.i64).mul(8)));
    const wcap: i32 = @intCast(c.cfg.window_capacity);
    const ccap: i32 = @intCast(c.cfg.compressed_capacity);
    r.w = ptx.ldGlobalV4(b, c.g.wi.add(t.mul(wcap).add(c.lane.mul(@divExact(wcap, 32))).to(.i64).mul(4)));
    for (&r.c) |*x| x.* = b.cst(.i32, -1);
    if (ccap > 0) {
        const per_lane = @divExact(ccap, 32);
        const src = c.g.ci.add(t.mul(ccap).add(c.lane.mul(per_lane)).to(.i64).mul(4));
        for (0..@intCast(@divExact(per_lane, 4))) |v| {
            const y = ptx.ldGlobalV4(b, src.add(@as(i64, @intCast(v * 16))));
            for (0..4) |i| r.c[v * 4 + i] = y[i];
        }
    }
    return r;
}

/// Validate token t's candidates (invalid ones become -1), store them and the
/// block counts in index buffer `kb`, and publish them.
fn frontStore(c: Ctx, t: V, kb: V, r: FrontRegs) void {
    const b = c.b;
    const zero = b.cst(.i32, 0);
    // Padded (inactive) query rows get no key blocks: their output is zero.
    const is_active = t.lt(r.active);
    const wlen = b.select(is_active, r.lens[0].minimum(@as(i32, @intCast(c.cfg.window_capacity))).maximum(0), zero);
    const clen = if (c.cfg.compressed_capacity > 0) b.select(is_active, r.lens[1].minimum(@as(i32, @intCast(c.cfg.compressed_capacity))).maximum(0), zero) else zero;
    const ib = c.s.add(smem.idx).add(kb.mul(smem.idx_bytes));
    inline for (.{ Format.fp8, Format.fp4 }) |f| {
        const capacity: i32 = @intCast(if (f == .fp8) c.cfg.window_capacity else c.cfg.compressed_capacity);
        if (capacity > 0) {
            const slots: i32 = @intCast(if (f == .fp8) c.cfg.window_slots else c.cfg.compressed_slots);
            const length = if (f == .fp8) wlen else clen;
            const per_lane = @divExact(capacity, 32); // 4 (window) or 16 (compressed)
            for (0..@intCast(per_lane)) |i| {
                const pos = c.lane.mul(per_lane).add(@as(i32, @intCast(i)));
                const ix = if (f == .fp8) r.w[i] else r.c[i];
                const ok = pos.lt(length).bitAnd(ix.ge(0)).bitAnd(ix.lt(slots));
                ptx.stSharedU32(b, ib.add(idxBase(f)).add(pos.mul(4)), b.select(ok, ix, b.cst(.i32, -1)));
            }
        }
    }
    {
        var l0 = b.openIf(c.lane.eq(0));
        ptx.stSharedV2(b, ib.add(smem.counts), .{ wlen.add(BK - 1).shrLogical(6), clen.add(BK - 1).shrLogical(6) });
        l0.yieldThen(.{});
    }
    ptx.mbarArrive(b, bar(b, c.s, .idx_ready, kb));
}

/// Q [16, 512] BF16 of token t -> SW128 K-major shared tile (the B operand of QK).
fn issueQ(c: Ctx, t: V, kb: V) void {
    const b = c.b;
    const q_row = c.g.q.add(t.to(.i64).mul(H * D * 2));
    const q_smem = c.s.add(b.select(kb.eq(0), b.cst(.i32, smem.q[0]), b.cst(.i32, smem.q[1])));
    const policy = ptx.evictFirstPolicy(b);
    for (0..32) |it| {
        const chunk = c.lane.add(@as(i32, @intCast(it * 32)));
        const row = chunk.shrLogical(6);
        const col = chunk.bitAnd(63).mul(8);
        ptx.cpAsync16(b, q_smem.add(ptx.sw128Offset(b, row, col, H)), q_row.add(row.mul(D).add(col).mul(2).to(.i64)), policy);
    }
}

fn frontRole(c0: Ctx, first: FrontRegs) void {
    const b = c0.b;
    frontStore(c0, c0.cta, b.cst(.i32, 0), first);
    var loop = b.openFor(0, c0.ntok, 1, .{});
    const k = loop.iv;
    const c = forToken(c0, k, undefined, undefined, false);
    {
        // Q of token k (token 0: issued at start) once every QK of token k - 2 has completed.
        var later = b.openIf(k.gt(0));
        {
            var reuse = b.openIf(k.ge(2));
            ptx.mbarWait(b, bar(b, c.s, .q_free, c.kb), parity(k).bitXor(1));
            reuse.yieldThen(.{});
        }
        issueQ(c, c.t, c.kb);
        later.yieldThen(.{});
    }
    // The MMA reads Q through the async proxy.
    ptx.cpAsyncWaitAll(b);
    ptx.fenceProxyAsync(b);
    ptx.mbarArrive(b, bar(b, c.s, .q_ready, c.kb));
    {
        var more = b.openIf(k.add(1).lt(c.ntok));
        const next_t = c.t.add(@as(i32, @intCast(c.cfg.gridCtas())));
        // Warm L2 with the next token's Q (16 KiB: 4 lines per lane).
        const q_next = c.g.q.add(next_t.to(.i64).mul(H * D * 2));
        for (0..4) |i| ptx.prefetchL2(b, q_next.add(c.lane.add(@as(i32, @intCast(i * 32))).mul(128).to(.i64)));
        const r = frontLoad(c, next_t);
        const nkb = k.add(1).bitAnd(1);
        {
            // The buffer's previous token (k - 1) must be released by every consumer.
            var reuse = b.openIf(k.ge(1));
            ptx.mbarWait(b, bar(b, c.s, .idx_free, nkb), parity(k.add(1)).bitXor(1));
            reuse.yieldThen(.{});
        }
        frontStore(c, next_t, nkb, r);
        more.yieldThen(.{});
    }
    loop.yield(.{});
}

// ---- loaders (warps 5-6) ------------------------------------------------------------

fn loaderRole(c0: Ctx) void {
    const b = c0.b;
    const policy = ptx.evictFirstPolicy(b);
    var loop = b.openFor(0, c0.ntok, 1, .{b.cst(.i32, 0)});
    const c = forToken(c0, loop.iv, undefined, loop.carried[0], true);
    {
        var blocks = b.openFor(c.b0.minimum(c.nwb), c.b1.minimum(c.nwb), 1, .{});
        loadBlock(c, .fp8, blocks.iv.sub(c.b0), blocks.iv, policy);
        blocks.yield(.{});
    }
    if (c.cfg.compressed_capacity > 0) {
        var blocks = b.openFor(c.b0.maximum(c.nwb), c.b1, 1, .{});
        loadBlock(c, .fp4, blocks.iv.sub(c.b0), blocks.iv.sub(c.nwb), policy);
        blocks.yield(.{});
    }
    releaseIndices(c);
    loop.yield(.{slotUse(c, c.n)});
}

fn idxBase(comptime f: Format) i32 {
    return if (f == .fp8) 0 else 128 * 4;
}

fn loadBlock(c: Ctx, comptime f: Format, j: V, local: V, policy: V) void {
    const b = c.b;
    const use_a = slotUse(c, j);
    const use_b = use_a.add(1); // second slot of an FP8 block
    const slot_a = use_a.rem(rawSlots(c.cfg));
    const slot_b = use_b.rem(rawSlots(c.cfg));
    for (0..@as(usize, if (f == .fp8) 2 else 1)) |k| {
        const use = if (k == 0) use_a else use_b;
        var reuse = b.openIf(use.ge(rawSlots(c.cfg)));
        ptx.mbarWait(b, bar(b, c.s, .raw_empty, use.rem(rawSlots(c.cfg))), slotParity(c, use).bitXor(1));
        reuse.yieldThen(.{});
    }
    const values = if (f == .fp8) c.g.wv else c.g.cv;
    const scales = if (f == .fp8) c.g.ws else c.g.cs;
    // FP8 rows are split across two slots: lanes 0-15 copy bytes 0-255 into slot a,
    // lanes 16-31 bytes 256-511 into slot b (both with the 288-byte FP4 row stride).
    const raw = if (f == .fp8) b.select(c.lane.lt(16), slotAddr(c, use_a), slotAddr(c, use_b)) else slotAddr(c, use_a);
    const sc = scaleAddr(c, if (f == .fp8) use_b else use_a);
    const idx = c.ib.add(idxBase(f)).add(local.mul(BK * 4));
    // Row data in 16-byte chunks. FP4 (16 chunks per row): lanes 0-15 copy rows
    // 0-31 and lanes 16-31 rows 32-63, two rows per warp instruction. FP8 (32
    // chunks per row): one row per instruction. Invalid rows (-1) are zero-filled.
    const chunks_per_row = @divExact(f.rowBytes(), 16);
    const chunk16 = c.lane.bitAnd(chunks_per_row - 1).mul(16);
    const chunk_off = chunk16.to(.i64);
    const dst_chunk = c.lane.bitAnd(15).mul(16); // within a 288-byte slot row
    // Two loader warps: warp 5 takes the first half of this lane's rows, warp 6 the second.
    // Four rows per iteration of a runtime loop (small code).
    const half_rows: i32 = comptime if (f == .fp4) 16 else 32;
    const second = c.warp.eq(6).to(.i32).bitAnd(1);
    const first_row = (if (f == .fp4) c.lane.shrLogical(4).mul(32) else b.cst(.i32, 0)).add(second.mul(half_rows));
    const src_base = values.add(chunk_off);
    const dst_base = raw.add(first_row.mul(288)).add(dst_chunk);
    const idx_base = idx.add(first_row.mul(4));
    {
        var rows = b.openFor(0, ptx.hiddenConst(b, half_rows), 4, .{});
        const r = rows.iv;
        const ixs = ptx.ldSharedBatch(b, 1, 4, idx_base.add(r.mul(4)), .{0});
        for (0..4) |k| {
            ptx.cpAsync16Row(b, dst_base.add(r.add(@as(i32, @intCast(k))).mul(288)), src_base, ixs[k], @intCast(f.rowBytes()), b.cst(.i64, 0), policy);
        }
        rows.yield(.{});
    }
    // Scales: 16 (FP8) or 32 (FP4) bytes per row, one 16-byte chunk per lane and step.
    const scale_chunks = comptime @divExact(f.scaleBytes(), 16);
    const scale_steps = comptime @divExact(BK * scale_chunks, 32);
    const sidx_offsets = comptime blk: {
        var o: [scale_steps]i32 = undefined;
        for (0..scale_steps) |k| o[k] = @as(i32, @intCast(k)) * @divExact(32, scale_chunks) * 4;
        break :blk o;
    };
    const scale_row0 = c.lane.shrLogical(std.math.log2_int(u32, @intCast(scale_chunks)));
    const part16 = c.lane.bitAnd(scale_chunks - 1).mul(16);
    const srow = ptx.ldSharedBatch(b, scale_steps, 1, idx.add(scale_row0.mul(4)), sidx_offsets);
    const second_warp = c.warp.eq(6);
    for (0..@intCast(scale_steps)) |k| {
        // Warp 5 copies the even scale steps, warp 6 the odd ones.
        var mine = b.openIf(if (k % 2 == 1) second_warp else c.warp.eq(5));
        const row = scale_row0.add(@as(i32, @intCast(k)) * @divExact(32, scale_chunks));
        ptx.cpAsync16Row(b, sc.add(row.mul(288)).add(part16), scales, srow[k], @intCast(f.scaleBytes()), part16.to(.i64), policy);
        mine.yieldThen(.{});
    }
    ptx.cpAsyncMbarArrive(b, bar(b, c.s, .raw_full, slot_a));
    if (f == .fp8) ptx.cpAsyncMbarArrive(b, bar(b, c.s, .raw_full, slot_b));
}

// ---- dequant (warps 8-11) ------------------------------------------------------------

fn dequantRole(c0: Ctx) void {
    const b = c0.b;
    const tid = c0.tid.sub(256);
    var loop = b.openFor(0, c0.ntok, 1, .{ b.cst(.i32, 0), b.cst(.i32, 0) });
    const c = forToken(c0, loop.iv, loop.carried[0], loop.carried[1], true);
    releaseIndices(c);
    {
        var blocks = b.openFor(0, c.nwl, 1, .{});
        dequantBlock(c, .fp8, blocks.iv, tid);
        blocks.yield(.{});
    }
    if (c.cfg.compressed_capacity > 0) {
        var blocks = b.openFor(c.nwl, c.n, 1, .{});
        dequantBlock(c, .fp4, blocks.iv, tid);
        blocks.yield(.{});
    }
    loop.yield(.{ c.g0.add(c.n), slotUse(c, c.n) });
}

/// Thread (g = tid / 8, i = tid % 8) converts 8 consecutive values per step:
/// rows g + 16 * r (r < 4), columns 64 * step + 8 * i. The 8 threads of a row
/// write one 128-byte swizzle row, so each STS.128 is conflict free. The tile is
/// produced in four parts of 128 columns (steps 2p, 2p + 1): an FP8 block's first
/// raw slot (columns 0-255) is released after part 1, so the next block's copies
/// can start while parts 2-3 are converted.
fn dequantBlock(c: Ctx, comptime f: Format, j: V, tid: V) void {
    const b = c.b;
    const gj = c.g0.add(j);
    const buf = gj.bitAnd(1);
    const par = parity(gj);
    const use_a = slotUse(c, j);
    const use_b = use_a.add(1);
    ptx.mbarWait(b, bar(b, c.s, .raw_full, use_a.rem(rawSlots(c.cfg))), slotParity(c, use_a));
    if (f == .fp8) ptx.mbarWait(b, bar(b, c.s, .raw_full, use_b.rem(rawSlots(c.cfg))), slotParity(c, use_b));
    {
        var reuse = b.openIf(gj.ge(2));
        ptx.mbarWait(b, bar(b, c.s, .pv_done, buf), par.bitXor(1));
        reuse.yieldThen(.{});
    }
    const g8 = tid.shrLogical(3);
    const r8 = g8.bitAnd(7);
    const in8 = tid.bitAnd(7);
    const swz = in8.bitXor(r8).shl(4);
    const raw = slotAddr(c, use_a).add(g8.mul(288));
    const raw_b = slotAddr(c, use_b).add(g8.mul(288));
    // FP8 scales follow the second slot's data (the first slot is released early).
    const sc = scaleAddr(c, if (f == .fp8) use_b else use_a).add(g8.mul(288));
    const tile = c.s.add(b.select(buf.eq(0), b.cst(.i32, smem.tile[0]), b.cst(.i32, smem.tile[1])));
    const dst = tile.add(g8.shrLogical(3).mul(1024)).add(r8.mul(128)).add(swz);
    // Raw words of part p for the thread's 4 rows (row stride 16 * 288 bytes):
    // FP8: 2 x 2 words per row, then one scale word per row;
    // FP4: 2 words per row, then 2 scale words per row.
    const nw = if (f == .fp8) 20 else 16;
    const loadPart = struct {
        fn f_(cc: Ctx, comptime ff: Format, raw_: V, raw_b_: V, sc_: V, in8_: V, p: V) [nw]V {
            const bb = cc.b;
            var out: [nw]V = undefined;
            if (ff == .fp8) {
                // Steps 2p, 2p + 1: bytes [128 (p & 1), + 128) of the first (p < 2) or second slot.
                const src = bb.select(p.lt(2), raw_, raw_b_).add(in8_.mul(8)).add(p.bitAnd(1).mul(128));
                const d = ptx.ldSharedBatch(bb, 8, 2, src, .{ 0, 64, 4608, 4672, 9216, 9280, 13824, 13888 });
                for (0..16) |i| out[i] = d[i];
                const sw = ptx.ldSharedBatch(bb, 4, 1, sc_.add(p.mul(4)), .{ 0, 4608, 9216, 13824 });
                for (0..4) |i| out[16 + i] = sw[i];
            } else {
                const d = ptx.ldSharedBatch(bb, 8, 1, raw_.add(in8_.mul(4)).add(p.mul(64)), .{ 0, 32, 4608, 4640, 9216, 9248, 13824, 13856 });
                for (0..8) |i| out[i] = d[i];
                const sw = ptx.ldSharedBatch(bb, 4, 2, sc_.add(p.mul(8)), .{ 0, 4608, 9216, 13824 });
                for (0..8) |i| out[8 + i] = sw[i];
            }
            return out;
        }
    }.f_;
    var parts = b.openFor(0, ptx.hiddenConst(b, 4), 1, tupleOf(nw, loadPart(c, f, raw, raw_b, sc, in8, b.cst(.i32, 0))));
    const p = parts.iv;
    var cur: [nw]V = undefined;
    for (0..nw) |i| cur[i] = parts.carried[i];
    // The last iteration re-loads its own part (harmless) to keep the loop uniform.
    const next = loadPart(c, f, raw, raw_b, sc, in8, p.add(1).minimum(3));
    const dst_p = dst.add(p.mul(2 * BK * 128));
    // Invalid rows arrive zero-filled (raw bytes and scales), so they dequantize to 0.
    for (0..4) |r| {
        for (0..2) |k| {
            const out = if (f == .fp8) blk: {
                // Scale index 2 * step + (i >= 4) = 4 p + 2 k + (i >> 2): byte 2 k + (i >> 2) of word p.
                const e = cur[16 + r].shrLogical(in8.shrLogical(2).add(@as(i32, @intCast(2 * k))).mul(8)).bitAnd(0xFF);
                const scale = e.shl(7).bitOr(e.shl(23));
                break :blk ptx.fp8x8ToBf16x2x4Scaled(b, cur[r * 4 + k * 2], cur[r * 4 + k * 2 + 1], scale);
            } else blk: {
                // Scale index 4 * step + i / 2: byte i / 2 of the step's word.
                const e = cur[8 + r * 2 + k].shrLogical(in8.shrLogical(1).mul(8)).bitAnd(0xFF);
                const scale = ptx.e4m3x2ToBf16x2(b, e.bitOr(e.shl(8)));
                break :blk ptx.fp4x8ToBf16x2x4Scaled(b, cur[r * 2 + k], scale);
            };
            ptx.stSharedV4(b, dst_p.add(@as(i32, @intCast(r * 2048 + k * BK * 128))), out);
        }
    }
    if (f == .fp8) {
        // Parts 0-1 were the first slot's last reads (part 2's loads read the second slot).
        var first_done = b.openIf(p.eq(1));
        ptx.mbarArrive(b, bar(b, c.s, .raw_empty, use_a.rem(rawSlots(c.cfg))));
        first_done.yieldThen(.{});
    }
    parts.yield(tupleOf(nw, next));
    ptx.fenceProxyAsync(b);
    ptx.mbarArrive(b, bar(b, c.s, .tile_full, buf));
    ptx.mbarArrive(b, bar(b, c.s, .raw_empty, (if (f == .fp8) use_b else use_a).rem(rawSlots(c.cfg))));
}

// ---- MMA (warp 4) ------------------------------------------------------------------------

fn mmaRole(c0: Ctx) void {
    const b = c0.b;
    var leader = b.openIf(ptx.electOne(b).ne(0));
    // Base descriptors; an MMA adds a constant (byte offset >> 4) to the start address field.
    const q_descs = [2]V{ ptx.smemDescSw128(b, c0.s.add(smem.q[0]), 16, 1024), ptx.smemDescSw128(b, c0.s.add(smem.q[1]), 16, 1024) };
    const k_desc = [2]V{ ptx.smemDescSw128(b, c0.s.add(smem.tile[0]), 16, 1024), ptx.smemDescSw128(b, c0.s.add(smem.tile[1]), 16, 1024) };
    const v_desc = [2]V{ ptx.smemDescSw128(b, c0.s.add(smem.tile[0]), 8192, 1024), ptx.smemDescSw128(b, c0.s.add(smem.tile[1]), 8192, 1024) };
    const p_desc = [2]V{ ptx.smemDescSw128(b, c0.s.add(smem.p[0]), 16, 1024), ptx.smemDescSw128(b, c0.s.add(smem.p[1]), 16, 1024) };
    const qk_idesc = b.cst(.i32, @as(i32, @bitCast(ptx.instrDescBf16(64, H, .k, .k))));
    const pv_idesc = b.cst(.i32, @as(i32, @bitCast(ptx.instrDescBf16(128, H, .mn, .k))));
    const i32t = cute.DType.i32.toMlir(b.ctx);

    var tokens = b.openFor(0, c0.ntok, 1, .{b.cst(.i32, 0)});
    const c = forToken(c0, tokens.iv, tokens.carried[0], undefined, true);
    // The MMA thread only needs the block count; the elected lane releases for the warp.
    ptx.mbarArrive(b, bar(b, c.s, .idx_free, c.kb));
    ptx.mbarWait(b, bar(b, c.s, .q_ready, c.kb), parity(c.k));
    const q_desc = b.select(c.kb.eq(0), q_descs[0], q_descs[1]);
    {
        // O^T buffer kb: its previous token's epilogue must have read it.
        var reuse = b.openIf(c.k.ge(2));
        ptx.mbarWait(b, bar(b, c.s, .o_free, c.kb), parity(c.k).bitXor(1));
        reuse.yieldThen(.{});
    }
    ptx.tcgenFenceAfter(b);
    {
        // No QK: Q is free right away.
        var none = b.openIf(c.n.eq(0));
        ptx.tcgenCommit(b, bar(b, c.s, .q_free, c.kb));
        none.yieldThen(.{});
    }
    const o_taddr = c.taddr.add(c.kb.mul(4 * H));
    // Iteration i issues QK(i + 1) and PV(i) in whichever order their inputs
    // become ready; i = -1 only issues QK(0).
    var loop = b.openFor(-1, c.n, 1, .{});
    const i = loop.iv;
    const next = i.add(1);
    var poll = b.openWhile(.{ next.lt(c.n).to(.i32).bitAnd(1), i.ge(0).to(.i32).bitAnd(1) }, .{ i32t, i32t });
    poll.yieldBefore(poll.before_carried[0].bitOr(poll.before_carried[1]).ne(0), .{ poll.before_carried[0], poll.before_carried[1] });
    const qk_pending = poll.after_carried[0];
    const pv_pending = poll.after_carried[1];
    const gn = c.g0.add(next);
    const nbuf = gn.bitAnd(1);
    const qk_go = qk_pending.ne(0).bitAnd(ptx.mbarTryWait(b, bar(b, c.s, .tile_full, nbuf), parity(gn)).ne(0));
    {
        var go = b.openIf(qk_go);
        ptx.tcgenFenceAfter(b);
        const kd = b.select(nbuf.eq(0), k_desc[0], k_desc[1]);
        const d = c.taddr.add(tmem_s).add(nbuf.mul(qk_parts * H));
        // K step s of part p covers latent dims 128 p + 16 s: consecutive MMAs go to
        // different accumulators.
        for (0..8) |step| {
            for (0..qk_parts) |part| {
                const ks = part * 8 + step;
                const kb: i64 = @intCast(ks / 4);
                const kin: i64 = @intCast(ks % 4);
                const adesc = kd.add(@divExact(kb * 8192 + kin * 32, 16));
                const bdesc = q_desc.add(@divExact(kb * 2048 + kin * 32, 16));
                ptx.mmaSS(b, d.add(@as(i32, @intCast(part * H))), adesc, bdesc, qk_idesc, b.cst(.i32, @intFromBool(step > 0)));
            }
        }
        ptx.tcgenCommit(b, bar(b, c.s, .s_full, nbuf));
        {
            // The token's last QK: Q may be replaced once it completes.
            var last = b.openIf(next.eq(c.n.sub(1)));
            ptx.tcgenCommit(b, bar(b, c.s, .q_free, c.kb));
            last.yieldThen(.{});
        }
        go.yieldThen(.{});
    }
    const gi = c.g0.add(i);
    const buf = gi.bitAnd(1);
    const pv_go = pv_pending.ne(0).bitAnd(ptx.mbarTryWait(b, bar(b, c.s, .p_full, buf), parity(gi)).ne(0));
    {
        var go = b.openIf(pv_go);
        ptx.tcgenFenceAfter(b);
        const vd = b.select(buf.eq(0), v_desc[0], v_desc[1]);
        const pd = b.select(buf.eq(0), p_desc[0], p_desc[1]);
        const first = b.select(i.eq(0), b.cst(.i32, 0), b.cst(.i32, 1));
        for (0..4) |ks| {
            for (0..4) |chunk| {
                const adesc = vd.add(@as(i64, @intCast(@divExact(2 * chunk * 8192 + ks * 2048, 16))));
                const bdesc = pd.add(@as(i64, @intCast(@divExact(ks * 32, 16))));
                const acc = if (ks > 0) b.cst(.i32, 1) else first;
                ptx.mmaSS(b, o_taddr.add(@as(i32, @intCast(chunk * H))), adesc, bdesc, pv_idesc, acc);
            }
        }
        ptx.tcgenCommit(b, bar(b, c.s, .pv_done, buf));
        go.yieldThen(.{});
    }
    poll.yieldAfter(.{ b.select(qk_go, b.cst(.i32, 0), qk_pending), b.select(pv_go, b.cst(.i32, 0), pv_pending) });
    loop.yield(.{});
    tokens.yield(.{c.g0.add(c.n)});
    leader.yieldThen(.{});
}

// ---- softmax / epilogue (warps 0-3) ------------------------------------------------------

fn softmaxRole(c0: Ctx) void {
    const b = c0.b;
    const scale_log2: f32 = @floatCast(c0.cfg.scale * log2e);
    const lane_tmem = c0.taddr.add(c0.warp.mul(32).shl(16));
    const row = c0.warp.mul(16).add(c0.lane); // S^T row held by this lane (lanes 0-15)
    const has_row = c0.lane.lt(16);

    {
        // Prefetch the sink logits for the epilogue.
        var h16 = b.openIf(c0.tid.lt(H));
        ptx.stSharedF32(b, c0.s.add(smem.sink).add(c0.tid.mul(4)), ptx.val(b, .f32, "ld.global.nc.f32 $0, [$1];", "=f,l", &.{c0.g.sink.add(c0.tid.mul(4).to(.i64))}, true));
        h16.yieldThen(.{});
    }
    var tokens = b.openFor(0, c0.ntok, 1, .{b.cst(.i32, 0)});
    const c = forToken(c0, tokens.iv, tokens.carried[0], undefined, true);
    const o_tmem = lane_tmem.add(c.kb.mul(4 * H));
    var init: [2 * H]V = undefined;
    for (0..H) |h| {
        // The running max only guards exp2 against overflow: it need not be exact,
        // and starting at 0 lets typical first blocks skip the cross-warp max. Logits
        // far below it just give small P (BF16 / F32 range), as with a stale max.
        init[h] = b.cst(.f32, 0);
        init[H + h] = b.cst(.f32, 0);
    }
    var loop = b.openFor(0, c.n, 1, tupleOf(2 * H, init));
    const j = loop.iv; // local block; token block is b0 + j
    var m: [H]V = undefined;
    var l: [H]V = undefined;
    for (0..H) |h| {
        m[h] = loop.carried[h];
        l[h] = loop.carried[H + h];
    }
    const gj = c.g0.add(j);
    const buf = gj.bitAnd(1);
    const par = parity(gj);
    ptx.mbarWait(b, bar(b, c.s, .s_full, buf), par);
    ptx.tcgenFenceAfter(b);
    const sparts = ptx.tmemLoad32x32b(b, qk_parts * H, lane_tmem.add(tmem_s).add(buf.mul(qk_parts * H)));
    ptx.tmemWaitLoad(b);
    var sv: [H]V = undefined;
    for (0..H) |h| {
        var acc = sparts[h].bitCast(.f32);
        for (1..qk_parts) |p_| acc = acc.add(sparts[p_ * H + h].bitCast(.f32));
        sv[h] = acc.bitCast(.i32);
    }
    // Candidate index of this row: window blocks first, then compressed blocks.
    const tb = c.b0.add(j);
    const is_window = tb.lt(c.nwb);
    const cand = b.select(is_window, b.cst(.i32, idxBase(.fp8)).add(tb.mul(BK * 4)), b.cst(.i32, idxBase(.fp4)).add(tb.sub(c.nwb).mul(BK * 4)));
    const row_ix = ptx.ldSharedU32(b, c.ib.add(cand).add(row.bitAnd(BK - 1).mul(4)));
    const valid = has_row.bitAnd(row_ix.ge(0));

    var sc: [H]V = undefined;
    var exceed = b.cst(.i32, 0);
    for (0..H) |h| {
        sc[h] = b.select(valid, sv[h].bitCast(.f32).mul(scale_log2), b.cst(.f32, -std.math.inf(f32)));
        exceed = exceed.bitOr(sc[h].gt(m[h].add(rescale_threshold)).to(.i32));
    }
    // Usually no score exceeds the running max by the threshold: one barrier vote
    // then skips the block max entirely (m stays, alpha = 1).
    const f32t = cute.DType.f32.toMlir(b.ctx);
    const i32t = cute.DType.i32.toMlir(b.ctx);
    var upd = b.openIfElse(ptx.barRedOr(b, 1, 128, exceed).ne(0), .{f32t} ** (2 * H) ++ .{i32t});
    {
        // Block max per head: warp redux, then across the 4 warps through shared memory.
        const red = c.s.add(b.select(buf.eq(0), b.cst(.i32, smem.red[0]), b.cst(.i32, smem.red[1])));
        for (0..H) |h| {
            const wm = ptx.warpMaxF32(b, sc[h]);
            var l0 = b.openIf(c.lane.eq(0));
            ptx.stSharedF32(b, red.add(c.warp.mul(H * 4)).add(@as(i32, @intCast(h * 4))), wm);
            l0.yieldThen(.{});
        }
        ptx.barSync(b, 1, 128);
        const bms = blockReduce(c, red, .max);
        var out: [2 * H + 1]V = undefined;
        var changed = b.cst(.i32, 0);
        for (0..H) |h| {
            const bm = bms[h];
            const move = bm.gt(m[h].add(rescale_threshold));
            out[h] = b.select(move, bm, m[h]);
            out[H + h] = ptx.exp2(b, m[h].sub(out[h]));
            changed = changed.bitOr(move.to(.i32));
        }
        out[2 * H] = changed;
        upd.yieldThen(tupleOf(2 * H + 1, out));
    }
    {
        var out: [2 * H + 1]V = undefined;
        for (0..H) |h| {
            out[h] = m[h];
            out[H + h] = b.cst(.f32, 1);
        }
        out[2 * H] = b.cst(.i32, 0);
        upd.yieldElse(tupleOf(2 * H + 1, out));
    }
    var new_m: [H]V = undefined;
    var alpha: [H]V = undefined;
    for (0..H) |h| {
        new_m[h] = upd.results[h];
        alpha[h] = upd.results[H + h];
    }
    const changed = upd.results[2 * H];
    // P^T [16 heads, 64 keys] BF16, SW128 K-major (one 128-byte row per head).
    const p = c.s.add(b.select(buf.eq(0), b.cst(.i32, smem.p[0]), b.cst(.i32, smem.p[1])));
    var next_l: [H]V = undefined;
    for (0..H) |h| {
        const pr = b.select(valid, ptx.exp2(b, sc[h].sub(new_m[h])), b.cst(.f32, 0));
        next_l[h] = l[h].mul(alpha[h]).add(pr);
        const off = row.shrLogical(3).bitXor(@as(i32, @intCast(h & 7))).shl(4).add(row.bitAnd(7).shl(1));
        ptx.stSharedBf16If(b, p.add(@as(i32, @intCast(h * 128))).add(off), pr, has_row.to(.i32));
    }
    // Rescale O^T (TMEM) once the previous PV has landed.
    {
        var rescale = b.openIf(j.gt(0).bitAnd(changed.ne(0)));
        const prev = gj.sub(1);
        ptx.mbarWait(b, bar(b, c.s, .pv_done, prev.bitAnd(1)), parity(prev));
        ptx.tcgenFenceAfter(b);
        for (0..4) |chunk| {
            const addr = o_tmem.add(@as(i32, @intCast(chunk * H)));
            const o = ptx.tmemLoad32x32b(b, H, addr);
            ptx.tmemWaitLoad(b);
            var scaled: [H]V = undefined;
            for (0..H) |h| scaled[h] = o[h].bitCast(.f32).mul(alpha[h]).bitCast(.i32);
            ptx.tmemStore32x32b(b, H, addr, scaled);
        }
        ptx.tmemWaitStore(b);
        rescale.yieldThen(.{});
    }
    ptx.fenceProxyAsync(b);
    ptx.tcgenFenceBefore(b);
    ptx.mbarArrive(b, bar(b, c.s, .p_full, buf));
    var carry: [2 * H]V = undefined;
    for (0..H) |h| {
        carry[h] = new_m[h];
        carry[H + h] = next_l[h];
    }
    loop.yield(tupleOf(2 * H, carry));
    releaseIndices(c);
    epilogue(c, o_tmem, loop.results[0..H].*, loop.results[H .. 2 * H].*);
    tokens.yield(.{c.g0.add(c.n)});
}

/// Per head, the reduction over the 4 softmax warps of `buf` [4 warps][16 heads] f32.
fn blockReduce(c: Ctx, buf: V, comptime op: ptx.ReduceOp) [H]V {
    const b = c.b;
    var r: [H]V = undefined;
    for (0..4) |w| {
        for (0..4) |q4| {
            const v = ptx.ldSharedV4(b, buf.add(@as(i32, @intCast((w * H + q4 * 4) * 4))));
            for (0..4) |i| {
                const x = v[i].bitCast(.f32);
                const h = q4 * 4 + i;
                r[h] = if (w == 0) x else if (op == .max) r[h].maximum(x) else r[h].add(x);
            }
        }
    }
    return r;
}

/// O^T / (l + exp(sink - m)) of the token (or this CTA's partial result, for a cluster merge).
fn epilogue(c: Ctx, o_tmem: V, m: [H]V, l: [H]V) void {
    const b = c.b;
    const lane_tmem = c.taddr.add(c.warp.mul(32).shl(16));
    // The sums overlap the token's last PV.
    const red_sum = c.s.add(b.select(c.kb.eq(0), b.cst(.i32, smem.red_sum[0]), b.cst(.i32, smem.red_sum[1])));
    {
        // Lanes 16-31 hold no key rows (l = 0); lane h (< 16) gets head h's sum.
        for (0..H) |h| {
            const lsum = ptx.warpSumF32(b, l[h]);
            var l0 = b.openIf(c.lane.eq(0));
            ptx.stSharedF32(b, red_sum.add(c.warp.mul(H * 4)).add(@as(i32, @intCast(h * 4))), lsum);
            l0.yieldThen(.{});
        }
    }
    ptx.barSync(b, 1, 128);
    const total = blockReduce(c, red_sum, .sum);
    const has_blocks = c.n.gt(0);
    {
        var any = b.openIf(has_blocks);
        const last = c.g0.add(c.n).sub(1);
        ptx.mbarWait(b, bar(b, c.s, .pv_done, last.bitAnd(1)), parity(last));
        any.yieldThen(.{});
    }
    ptx.tcgenFenceAfter(b);
    if (c.cfg.cluster > 1) {
        // Partial result for the cluster merge: (m, l) per head and unnormalized O^T.
        {
            var t0 = b.openIf(c.tid.eq(0));
            for (0..H) |h| {
                ptx.stSharedF32(b, c.s.add(smem.stats + @as(i32, @intCast(h * 4))), m[h]);
                ptx.stSharedF32(b, c.s.add(smem.stats + @as(i32, @intCast((H + h) * 4))), total[h]);
            }
            t0.yieldThen(.{});
        }
        // Normalized by this CTA's own sum (a convex combination of cache rows: bounded,
        // so it travels as FP16); the owner re-weights with exp2(m - M) * l / L.
        var inv_l: [H]V = undefined;
        for (0..H) |h| inv_l[h] = b.select(total[h].gt(0).bitAnd(has_blocks), ptx.rcp(b, total[h]), b.cst(.f32, 0));
        for (0..4) |chunk| {
            const o = ptx.tmemLoad32x32b(b, H, o_tmem.add(@as(i32, @intCast(chunk * H))));
            ptx.tmemWaitLoad(b);
            const dim = c.warp.mul(32).add(c.lane).add(@as(i32, @intCast(chunk * 128)));
            const dst = c.s.add(smem.obuf).add(dim.mul(H * 4));
            for (0..4) |q4| {
                var v: [4]V = undefined;
                // A rank without key blocks never wrote its O^T (TMEM holds stale data, possibly
                // NaN/Inf): select, do not multiply by 0.
                for (0..4) |i| v[i] = b.select(has_blocks, o[q4 * 4 + i].bitCast(.f32).mul(inv_l[q4 * 4 + i]), b.cst(.f32, 0)).bitCast(.i32);
                ptx.stSharedV4(b, dst.add(@as(i32, @intCast(q4 * 16))), v);
            }
        }
        _ = lane_tmem;
    } else {
        var inv: [H]V = undefined;
        for (0..H) |h| {
            const sink = ptx.ldSharedF32(b, c.s.add(smem.sink + @as(i32, @intCast(h * 4))));
            const denom = total[h].add(ptx.exp2(b, sink.mul(@as(f32, log2e)).sub(m[h])));
            // denom underflows only if every logit and the sink are ~2^126 below the running max.
            inv[h] = b.select(has_blocks.bitAnd(denom.gt(0)), ptx.rcp(b, denom), b.cst(.f32, 0));
        }
        // The CTA's last token is staged in shared memory (Q is no longer needed) and
        // written with 16-byte stores; earlier tokens store straight to global memory
        // (a warp writes 64 contiguous bytes per head).
        const last_token = c.k.eq(c.ntok.sub(1));
        const out_row = c.g.out.add(c.t.to(.i64).mul(H * D * 2));
        {
            var direct = b.openIf(last_token.eq(false));
            var os: [4][H]V = undefined;
            for (0..4) |chunk| os[chunk] = ptx.tmemLoad32x32b(b, H, o_tmem.add(@as(i32, @intCast(chunk * H))));
            ptx.tmemWaitLoad(b);
            // Lane pairs (d, d + 1) swap half their heads: the even lane stores heads 0-7,
            // the odd lane heads 8-15, each as bf16 pairs of dims (d, d + 1).
            const odd = c.lane.bitAnd(1).ne(0);
            const d_even = c.warp.mul(32).add(c.lane.bitAnd(30));
            const h0 = b.select(odd, b.cst(.i32, 8), b.cst(.i32, 0));
            const base = out_row.add(h0.mul(D).add(d_even).mul(2).to(.i64));
            for (0..4) |chunk| {
                for (0..8) |i| {
                    const lo = b.select(has_blocks, os[chunk][i].bitCast(.f32).mul(inv[i]), b.cst(.f32, 0));
                    const hi = b.select(has_blocks, os[chunk][8 + i].bitCast(.f32).mul(inv[8 + i]), b.cst(.f32, 0));
                    const send = b.select(odd, lo, hi);
                    const keep = b.select(odd, hi, lo);
                    const recv = b.shuffleXor(send, 1).bitCast(.f32);
                    // Even lane: (own dim, partner's dim); odd lane: (partner's dim, own dim).
                    const pair = ptx.packBf16x2(b, b.select(odd, recv, keep), b.select(odd, keep, recv));
                    ptx.stGlobalU32(b, base.add(@as(i64, @intCast((i * D + chunk * 128) * 2))), pair);
                }
            }
            direct.yieldThen(.{});
        }
        {
            var staged = b.openIf(last_token);
            // The staging buffer is Q buffer 0: a token without key blocks never waited for
            // its Q copy (cp.async into that buffer), which must land before it is overwritten.
            ptx.mbarWait(b, bar(b, c.s, .q_ready, c.kb), parity(c.k));
            var os: [4][H]V = undefined;
            for (0..4) |chunk| os[chunk] = ptx.tmemLoad32x32b(b, H, o_tmem.add(@as(i32, @intCast(chunk * H))));
            ptx.tmemWaitLoad(b);
            for (0..4) |chunk| {
                const o = os[chunk];
                const dim = c.warp.mul(32).add(c.lane).add(@as(i32, @intCast(chunk * 128)));
                for (0..H) |h| {
                    const v = b.select(has_blocks, o[h].bitCast(.f32).mul(inv[h]), b.cst(.f32, 0));
                    ptx.stSharedB16(b, c.s.add(smem.ostage).add(dim.add(@as(i32, @intCast(h * D))).mul(2)), ptx.f32ToBf16Bits(b, v));
                }
            }
            staged.yieldThen(.{});
        }
        // O^T buffer kb may take the token after next.
        ptx.tcgenFenceBefore(b);
        ptx.mbarArrive(b, bar(b, c.s, .o_free, c.kb));
        {
            var staged = b.openIf(last_token);
            ptx.barSync(b, 1, 128);
            for (0..8) |it| {
                const off = c.tid.add(@as(i32, @intCast(it * 128))).mul(16);
                ptx.stGlobalV4(b, out_row.add(off.to(.i64)), ptx.ldSharedV4(b, c.s.add(smem.ostage).add(off)));
            }
            staged.yieldThen(.{});
        }
    }
}

// ---- cluster merge (all 384 threads, cluster > 1) ----------------------------------------------

/// Combine the cluster's partial results. Rank r owns output dims
/// [r * 512 / C, (r + 1) * 512 / C): after every rank has finished its blocks,
/// each pushes its (m, l) and the owners' slices of its O^T into their shared
/// memory (remote stores do not stall), then every owner combines locally.
/// No global partials and no second kernel.
fn clusterMerge(c: Ctx) void {
    const b = c.b;
    const C: i32 = @intCast(c.cfg.cluster);
    const dims = @divExact(D, C);
    // Every rank is done with its tiles (the inbox reuses tile 0).
    ptx.clusterSync(b);
    {
        // (m, l) of head h to every rank's inbox, slot `rank`.
        var heads_ = b.openIf(c.tid.lt(2 * H));
        const v = ptx.ldSharedF32(b, c.s.add(smem.stats).add(c.tid.mul(4)));
        const dst = c.s.add(smem.stats_in).add(c.rank.mul(2 * H * 4)).add(c.tid.mul(4));
        for (0..@intCast(C)) |p| ptx.stClusterF32(b, ptx.mapaShared(b, dst, b.cst(.i32, @as(i32, @intCast(p)))), v);
        heads_.yieldThen(.{});
    }
    {
        // O^T rows of peer p's dims to its inbox slot `rank`: 16-byte chunks.
        const chunks = D * @divExact(H, 4);
        var loop = b.openFor(c.tid, chunks, threads, .{});
        const chunk = loop.iv;
        const d = chunk.shrLogical(2);
        const q4 = chunk.bitAnd(3);
        const v = ptx.ldSharedV4(b, c.s.add(smem.obuf).add(d.mul(H * 4)).add(q4.mul(16)));
        const owner = d.div(dims);
        // Normalized partials travel as FP16.
        const packed_ = [2]V{ ptx.packF16x2(b, v[0].bitCast(.f32), v[1].bitCast(.f32)), ptx.packF16x2(b, v[2].bitCast(.f32), v[3].bitCast(.f32)) };
        const dst = c.s.add(smem.inbox).add(c.rank.mul(dims * H * 2)).add(d.rem(dims).mul(H * 2)).add(q4.mul(8));
        ptx.stClusterV2(b, ptx.mapaShared(b, dst, owner), packed_);
        loop.yield(.{});
    }
    ptx.clusterSync(b);
    // Per-head merge weights exp2(m_p - M) / L, with L = sum_p exp2(m_p - M) l_p + exp2(sink - M).
    {
        var heads_ = b.openIf(c.tid.lt(H));
        const h = c.tid;
        var ms: [16]V = undefined;
        var ls: [16]V = undefined;
        var mx = b.cst(.f32, @as(f32, -1e30));
        for (0..@intCast(C)) |p| {
            const slot = c.s.add(smem.stats_in + @as(i32, @intCast(p * 2 * H * 4))).add(h.mul(4));
            ms[p] = ptx.ldSharedF32(b, slot);
            ls[p] = ptx.ldSharedF32(b, slot.add(H * 4));
            mx = mx.maximum(ms[p]);
        }
        const sink = ptx.ldSharedF32(b, c.s.add(smem.sink).add(h.mul(4)));
        var denom = ptx.exp2(b, sink.mul(@as(f32, log2e)).sub(mx));
        var w: [16]V = undefined;
        for (0..@intCast(C)) |p| {
            w[p] = ptx.exp2(b, ms[p].sub(mx));
            denom = denom.add(w[p].mul(ls[p]));
        }
        const inv = ptx.rcp(b, denom);
        // Partials are O_p / l_p: weight exp2(m_p - M) * l_p / L.
        for (0..@intCast(C)) |p| ptx.stSharedF32(b, c.s.add(smem.weights + @as(i32, @intCast(p * H * 4))).add(h.mul(4)), w[p].mul(ls[p]).mul(inv));
        heads_.yieldThen(.{});
    }
    ptx.barSync(b, 0, threads);
    // Output: this rank's dims x 16 heads, 4 heads (one 16-byte load per rank) per item.
    const items = dims * @divExact(H, 4);
    const out_row = c.g.out.add(c.t.to(.i64).mul(H * D * 2));
    {
        var loop = b.openFor(c.tid, items, threads, .{});
        const item = loop.iv;
        // Consecutive threads take consecutive dims so the output stores coalesce.
        const dl = item.rem(dims);
        const hq = item.div(dims); // head quad
        var acc = [4]V{ b.cst(.f32, 0), b.cst(.f32, 0), b.cst(.f32, 0), b.cst(.f32, 0) };
        for (0..@intCast(C)) |p| {
            const o2 = ptx.ldSharedV2(b, c.s.add(smem.inbox + @as(i32, @intCast(p)) * dims * H * 2).add(dl.mul(H * 2)).add(hq.mul(8)));
            const lo = ptx.unpackF16x2(b, o2[0]);
            const hi = ptx.unpackF16x2(b, o2[1]);
            const o = [4]V{ lo[0], lo[1], hi[0], hi[1] };
            const wv = ptx.ldSharedV4(b, c.s.add(smem.weights + @as(i32, @intCast(p * H * 4))).add(hq.mul(16)));
            for (0..4) |i| acc[i] = acc[i].add(o[i].mul(wv[i].bitCast(.f32)));
        }
        for (0..4) |i| {
            const head = hq.mul(4).add(@as(i32, @intCast(i)));
            ptx.stSharedB16(b, c.s.add(smem.ostage).add(head.mul(dims).add(dl).mul(2)), ptx.f32ToBf16Bits(b, acc[i]));
        }
        loop.yield(.{});
    }
    ptx.barSync(b, 0, threads);
    {
        // This rank's slice of every head row: dims / 8 16-byte chunks per head.
        const per_head = @divExact(dims, 8);
        var loop = b.openFor(c.tid, H * per_head, threads, .{});
        const h = loop.iv.div(per_head);
        const k = loop.iv.rem(per_head);
        const v = ptx.ldSharedV4(b, c.s.add(smem.ostage).add(h.mul(dims).add(k.mul(8)).mul(2)));
        ptx.stGlobalV4(b, out_row.add(h.mul(D).add(c.rank.mul(dims)).add(k.mul(8)).to(.i64).mul(2)), v);
        loop.yield(.{});
    }
}
