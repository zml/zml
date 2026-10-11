//! The device kernels of tt-metal's chunked Gated Delta Rule
//! (`chunk_gated_delta_rule/device/kernels/`, tt-metal d04395ed) in the
//! TTKernel DSL, for the configuration the port runs: token-major queries,
//! keys and values, normalized beforehand (`QK_FLAT`, `V_FLAT`, no
//! `QK_NORM`).
//!
//! A kernel's circular buffers are its first compile-time arguments, in the
//! order of its `*_cbs` list, followed by the accessor arguments of its
//! tensors; tile counts are built in.

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;

/// Circular buffer indices shared by both phases (`pcb` of the factory).
pub const cb = struct {
    pub const q = 0;
    pub const k = 1;
    pub const v = 2;
    pub const g = 3;
    pub const beta = 4;
    pub const eye = 5;
    pub const tril = 6;
    pub const ones = 7;
    pub const s = 8;
    pub const decay = 9;
    pub const decay_exp = 10;
    /// `decayfac` in the prep; the scan reads `dl` into it.
    pub const decayfac = 11;
    pub const lmask = 12;
    pub const tinv = 13;
    pub const vbeta = 14;
    pub const kbeta = 15;
    pub const out = 16;
    /// The scan's `v_beta`; the prep loads the quadrant masks into it.
    pub const u = 17;
    pub const w = 18;
    pub const qdecay = 19;
    pub const intra = 20;
    pub const s2 = 21;
    /// The scan's `v_new`; the prep's `dl`.
    pub const vnew = 22;
    pub const ointer = 23;
    pub const kdec_t = 24;
    pub const supd = 25;
    pub const stmp = 26;
    pub const final_s = 27;
    pub const scr1 = 28;
    pub const scr2 = 29;
    pub const scr3 = 30;
    pub const s3 = 31;
};

/// A kernel's circular buffers, by `cb` name.
const Cbs = struct {
    q: Cb = undefined,
    k: Cb = undefined,
    v: Cb = undefined,
    g: Cb = undefined,
    beta: Cb = undefined,
    eye: Cb = undefined,
    tril: Cb = undefined,
    ones: Cb = undefined,
    s: Cb = undefined,
    decay: Cb = undefined,
    decay_exp: Cb = undefined,
    decayfac: Cb = undefined,
    lmask: Cb = undefined,
    tinv: Cb = undefined,
    vbeta: Cb = undefined,
    kbeta: Cb = undefined,
    out: Cb = undefined,
    u: Cb = undefined,
    w: Cb = undefined,
    qdecay: Cb = undefined,
    intra: Cb = undefined,
    s2: Cb = undefined,
    vnew: Cb = undefined,
    ointer: Cb = undefined,
    kdec_t: Cb = undefined,
    supd: Cb = undefined,
    stmp: Cb = undefined,
    final_s: Cb = undefined,
    scr1: Cb = undefined,
    scr2: Cb = undefined,
    scr3: Cb = undefined,
    s3: Cb = undefined,
};

/// The circular buffers `names`, compile-time arguments `0..names.len`.
/// Their type only types the argument: sizes and formats are the program's.
fn declare(k: *Builder, comptime names: []const []const u8) Cbs {
    var out: Cbs = .{};
    inline for (names, 0..) |n, i| @field(out, n) = k.circularBufferOf(i, .f32, 1);
    return out;
}

/// The indices of the circular buffers `names`, the program side of
/// `declare`.
pub fn cbIds(comptime names: []const []const u8) [names.len]u32 {
    var out: [names.len]u32 = undefined;
    inline for (names, 0..) |n, i| out[i] = @field(cb, n);
    return out;
}

/// The tensor accessors of `addresses`, their arguments following the
/// `first_ct` compile-time arguments.
fn accessors(comptime n: usize, k: *Builder, first_ct: usize, addresses: [n]Value, tile_bytes: [n]Value) [n]Value {
    var out: [n]Value = undefined;
    var args = k.tensorAccessorArgs(@intCast(first_ct), 0);
    for (&out, addresses, tile_bytes, 0..) |*a, addr, tb, i| {
        if (i > 0) args = k.nextTensorAccessorArgs(args);
        a.* = k.tensorAccessor(args, addr, tb);
    }
    return out;
}

fn runtimeArgs(comptime n: usize, k: *Builder, first: u32) [n]Value {
    var out: [n]Value = undefined;
    for (&out, 0..) |*v, i| v.* = k.runtimeArg(first + @as(u32, @intCast(i)));
    return out;
}

// =============================================================================
// Prep (phase A)
// =============================================================================

pub const PrepConfig = struct {
    /// Tiles per chunk, key dim, value dim.
    ct: u32,
    kt: u32,
    vt: u32,
};

pub const prep_reader_cbs = [_][]const u8{ "q", "k", "v", "g", "beta", "eye", "tril", "ones", "u" };

/// `reader_chunk_gdn_prep.cpp`: the constants once, then per work item
/// (head, chunk) its queries, keys, values, gates. Runtime args: first
/// work item, work items, the addresses of `q`, `k`, `v`, `g`, `beta`,
/// `eye`, `tril`, `ones`, `masks`, then chunks, value heads, key heads.
pub fn prepReader(k: *Builder, c: PrepConfig) void {
    const b = declare(k, &prep_reader_cbs);
    const wi_start, const wi_count = runtimeArgs(2, k, 0);
    const addrs = runtimeArgs(9, k, 2);
    const nc, const hv_n, const hk_n = runtimeArgs(3, k, 11);
    const tb_io = k.op(.get_tile_size, .{b.q});
    const tb_f = k.op(.get_tile_size, .{b.g});
    const q_acc, const k_acc, const v_acc, const g_acc, const b_acc, const eye_acc, const tril_acc, const ones_acc, const mask_acc =
        accessors(9, k, prep_reader_cbs.len, addrs, .{ tb_io, tb_io, tb_io, tb_f, tb_f, tb_f, tb_f, tb_f, tb_f });
    const cc = c.ct * c.ct;

    const zero = k.int(0);
    llk.readTiles(k, eye_acc, b.eye, zero, cc, tb_f);
    llk.readTiles(k, tril_acc, b.tril, zero, cc, tb_f);
    llk.readTiles(k, ones_acc, b.ones, zero, cc, tb_f);
    // Quadrant masks: top-left, bottom-right, bottom-left.
    llk.readTiles(k, mask_acc, b.u, zero, 3, tb_f);

    var items, const i = llk.range(k, wi_count);
    {
        const hc = wi_start.add(i);
        const bh = hc.div(nc);
        const chunk = hc.rem(nc);
        const hv = bh.rem(hv_n);
        const batch = bh.div(hv_n);
        // Token-major `[b, t, heads * dim]` tiles: this chunk's rows, this
        // head's columns. A value head reads key head `hv / (hv_n / hk_n)`.
        const hk = hv.div(hv_n.div(hk_n));
        const qk_stride = hk_n.mul(c.kt);
        const qk_base = batch.mul(nc).add(chunk).mul(c.ct).mul(qk_stride).add(hk.mul(c.kt));
        llk.readBlock(k, q_acc, b.q, qk_base, c.ct, c.kt, qk_stride, tb_io);
        llk.readBlock(k, k_acc, b.k, qk_base, c.ct, c.kt, qk_stride, tb_io);
        const v_stride = hv_n.mul(c.vt);
        const v_base = batch.mul(nc).add(chunk).mul(c.ct).mul(v_stride).add(hv.mul(c.vt));
        llk.readBlock(k, v_acc, b.v, v_base, c.ct, c.vt, v_stride, tb_io);
        llk.readTiles(k, g_acc, b.g, hc.mul(c.ct), c.ct, tb_f);
        llk.readTiles(k, b_acc, b.beta, hc.mul(c.ct), c.ct, tb_f);
    }
    items.yield(.{});
}

pub const prep_writer_cbs = [_][]const u8{ "vbeta", "tinv", "w", "intra", "qdecay", "kdec_t", "vnew" };

/// `writer_chunk_gdn_prep.cpp`: per work item, the seven outputs in the
/// compute's push order. Runtime args: first work item, work items, the
/// addresses of `v_beta`, `kd`, `q_decay`, `intra`, `k_dec_t`, `dl`, `t_inv`.
pub fn prepWriter(k: *Builder, c: PrepConfig) void {
    const b = declare(k, &prep_writer_cbs);
    const wi_start, const wi_count = runtimeArgs(2, k, 0);
    const tb = k.op(.get_tile_size, .{b.vbeta});
    const vb_acc, const kd_acc, const qd_acc, const it_acc, const kc_acc, const dl_acc, const ti_acc =
        accessors(7, k, prep_writer_cbs.len, runtimeArgs(7, k, 2), @splat(tb));
    const cc = c.ct * c.ct;
    const ck = c.ct * c.kt;
    const cv = c.ct * c.vt;
    const kc = c.kt * c.ct;

    var items, const i = llk.range(k, wi_count);
    {
        const hc = wi_start.add(i);
        llk.writeTiles(k, vb_acc, b.vbeta, hc.mul(cv), cv, tb);
        llk.writeTiles(k, ti_acc, b.tinv, hc.mul(cc), cc, tb);
        llk.writeTiles(k, kd_acc, b.w, hc.mul(ck), ck, tb);
        llk.writeTiles(k, it_acc, b.intra, hc.mul(cc), cc, tb);
        llk.writeTiles(k, qd_acc, b.qdecay, hc.mul(ck), ck, tb);
        llk.writeTiles(k, kc_acc, b.kdec_t, hc.mul(kc), kc, tb);
        llk.writeTiles(k, dl_acc, b.vnew, hc, 1, tb);
    }
    items.yield(.{});
}

pub const prep_compute_cbs = [_][]const u8{
    "q",        "k",     "v",    "g",     "beta",   "eye",    "tril",    "ones",   "u",  "decay", "decay_exp",
    "decayfac", "lmask", "tinv", "vbeta", "kbeta",  "qdecay", "intra",   "kdec_t", "w",  "vnew",  "scr1",
    "scr2",     "scr3",  "supd", "stmp",  "ointer", "s",      "final_s", "s2",     "s3",
};

/// `(I - nq)^-1` for a strictly lower 16x16 block `nq` alone in its
/// quadrant: 15 terms of Horner's scheme (`invert16`).
fn invert16(k: *Builder, b: Cbs, nq: Cb, out: Cb, tmp: Cb) void {
    llk.ew(k, b.eye, nq, out, 1, .add);
    llk.wait(k, out, 1);
    var loop = k.openFor(2, 16, 1, .{});
    {
        llk.mm(k, nq, out, tmp, 1, 1, 1, false);
        llk.wait(k, tmp, 1);
        llk.pop(k, out, 1);
        llk.ew(k, b.eye, tmp, out, 1, .add);
        llk.wait(k, out, 1);
        llk.pop(k, tmp, 1);
    }
    loop.yield(.{});
}

/// `out[0] = (I - negN)^-1` of the strictly lower tile `src[tile]`, by
/// 16x16 quadrants as FLA's `solve_tril` (`invert_block`).
fn invertBlock(k: *Builder, b: Cbs, src: Cb, tile: u32, out: Cb, tmp_n: Cb, tmp_t: Cb) void {
    // Private scratch, never drained by the writer.
    const qa, const qb, const qc, const qd = .{ b.s, b.final_s, b.s2, b.s3 };
    llk.copyTile(k, src, tile, tmp_n);
    llk.wait(k, tmp_n, 1);
    // Top-left inverse.
    llk.ewt(k, tmp_n, 0, b.u, 0, qa, .mul);
    llk.wait(k, qa, 1);
    invert16(k, b, qa, qb, tmp_t);
    llk.wait(k, qb, 1);
    llk.pop(k, qa, 1);
    // Bottom-right inverse.
    llk.ewt(k, tmp_n, 0, b.u, 1, qa, .mul);
    llk.wait(k, qa, 1);
    invert16(k, b, qa, qc, tmp_t);
    llk.wait(k, qc, 1);
    llk.pop(k, qa, 1);
    // Bottom-left: inv11 @ n10 @ inv00, exactly.
    llk.ewt(k, tmp_n, 0, b.u, 2, qa, .mul);
    llk.wait(k, qa, 1);
    llk.pop(k, tmp_n, 1);
    llk.mm(k, qc, qa, tmp_t, 1, 1, 1, false);
    llk.wait(k, tmp_t, 1);
    llk.pop(k, qa, 1);
    llk.mm(k, tmp_t, qb, qa, 1, 1, 1, false);
    llk.wait(k, qa, 1);
    llk.pop(k, tmp_t, 1);
    // The three quadrants together.
    llk.ewt(k, qb, 0, b.u, 0, qd, .mul);
    llk.wait(k, qd, 1);
    llk.pop(k, qb, 1);
    llk.ewt(k, qc, 0, b.u, 1, qb, .mul);
    llk.wait(k, qb, 1);
    llk.pop(k, qc, 1);
    llk.ewt(k, qd, 0, qb, 0, qc, .add);
    llk.wait(k, qc, 1);
    llk.pop(k, qd, 1);
    llk.pop(k, qb, 1);
    llk.ewt(k, qc, 0, qa, 0, out, .add);
    llk.pop(k, qc, 1);
    llk.pop(k, qa, 1);
}

/// `chunk_gdn_prep.cpp`: per work item, the state-independent quantities
/// of a chunk. Runtime arg: work items.
pub fn prepCompute(k: *Builder, c: PrepConfig) void {
    const b = declare(k, &prep_compute_cbs);
    const items_n = k.runtimeArg(0);
    const ct, const kt, const vt = .{ c.ct, c.kt, c.vt };
    const cc = ct * ct;
    const ck = ct * kt;
    const cv = ct * vt;
    const C = ct * 32;

    k.op(.compute_kernel_hw_startup, .{ b.q, b.k, b.u });
    llk.wait(k, b.eye, cc);
    llk.wait(k, b.tril, cc);
    llk.wait(k, b.ones, cc);
    llk.wait(k, b.u, 3);

    var items, _ = llk.range(k, items_n);
    {
        llk.wait(k, b.q, ck);
        llk.wait(k, b.k, ck);
        llk.wait(k, b.v, cv);
        llk.wait(k, b.g, ct);
        llk.wait(k, b.beta, ct);

        // v_beta, k_beta
        llk.bcast(k, b.v, b.beta, b.vbeta, ct, vt, .mul, .cols);
        llk.wait(k, b.vbeta, cv);
        llk.bcast(k, b.k, b.beta, b.kbeta, ct, kt, .mul, .cols);
        llk.wait(k, b.kbeta, ck);
        llk.pop(k, b.beta, ct);
        llk.pop(k, b.v, cv);

        // decay = tril @ g, its exponential, as a row
        llk.mm(k, b.tril, b.g, b.decay, ct, ct, 1, false);
        llk.wait(k, b.decay, ct);
        llk.exp(k, b.decay, b.decay_exp, ct);
        llk.wait(k, b.decay_exp, ct);
        llk.transposeBlock(k, b.decay, b.scr1, ct, 1);
        llk.wait(k, b.scr1, ct);

        // l_mask = tril(exp(decay_i - decay_j))
        llk.bcast(k, b.ones, b.decay, b.scr2, ct, ct, .mul, .cols);
        llk.wait(k, b.scr2, cc);
        llk.bcast(k, b.scr2, b.scr1, b.scr3, ct, ct, .sub, .rows);
        llk.wait(k, b.scr3, cc);
        llk.pop(k, b.scr1, ct);
        llk.pop(k, b.scr2, cc);
        llk.ew(k, b.scr3, b.tril, b.scr2, cc, .mul);
        llk.wait(k, b.scr2, cc);
        llk.pop(k, b.scr3, cc);
        llk.exp(k, b.scr2, b.scr3, cc);
        llk.wait(k, b.scr3, cc);
        llk.pop(k, b.scr2, cc);
        llk.ew(k, b.scr3, b.tril, b.lmask, cc, .mul);
        llk.wait(k, b.lmask, cc);
        llk.pop(k, b.scr3, cc);

        // decayfac = exp(g_sum - decay)
        llk.mm(k, b.ones, b.g, b.scr1, ct, ct, 1, false);
        llk.wait(k, b.scr1, ct);
        llk.pop(k, b.g, ct);
        llk.ew(k, b.scr1, b.decay, b.scr2, ct, .sub);
        llk.wait(k, b.scr2, ct);
        llk.pop(k, b.scr1, ct);
        llk.pop(k, b.decay, ct);
        llk.exp(k, b.scr2, b.decayfac, ct);
        llk.wait(k, b.decayfac, ct);
        llk.pop(k, b.scr2, ct);

        // -N = diag(kk) - kk, kk = (k_beta @ k^T) * l_mask; t_inv = (I + N)^-1
        llk.mm(k, b.kbeta, b.k, b.scr1, ct, kt, ct, true);
        llk.wait(k, b.scr1, cc);
        llk.ew(k, b.scr1, b.lmask, b.scr2, cc, .mul);
        llk.wait(k, b.scr2, cc);
        llk.pop(k, b.scr1, cc);
        llk.ew(k, b.scr2, b.eye, b.scr1, cc, .mul);
        llk.wait(k, b.scr1, cc);
        llk.ew(k, b.scr1, b.scr2, b.scr3, cc, .sub);
        llk.wait(k, b.scr3, cc);
        llk.pop(k, b.scr1, cc);
        llk.pop(k, b.scr2, cc);
        switch (ct) {
            1 => {
                invertBlock(k, b, b.scr3, 0, b.tinv, b.scr1, b.scr2);
                llk.wait(k, b.tinv, cc);
                llk.pop(k, b.scr3, cc);
            },
            2 => {
                // 2x2 tiles: the diagonal inverses, then the off-diagonal
                // inv22 @ -N21 @ inv11.
                invertBlock(k, b, b.scr3, 0, b.supd, b.scr1, b.scr2);
                invertBlock(k, b, b.scr3, 3, b.stmp, b.scr1, b.scr2);
                llk.copyTile(k, b.scr3, 2, b.scr1);
                llk.wait(k, b.scr1, 1);
                llk.mm(k, b.scr1, b.supd, b.scr2, 1, 1, 1, false);
                llk.wait(k, b.scr2, 1);
                llk.pop(k, b.scr1, 1);
                llk.mm(k, b.stmp, b.scr2, b.ointer, 1, 1, 1, false);
                llk.wait(k, b.ointer, 1);
                llk.pop(k, b.scr2, 1);
                llk.pop(k, b.scr3, cc);
                // eye's tile 1 is the zero block.
                llk.assemble(k, &.{ .{ .cb = b.supd, .tile = 0 }, .{ .cb = b.eye, .tile = 1 }, .{ .cb = b.ointer, .tile = 0 }, .{ .cb = b.stmp, .tile = 0 } }, b.tinv);
                llk.wait(k, b.tinv, cc);
                llk.pop(k, b.supd, 1);
                llk.pop(k, b.stmp, 1);
                llk.pop(k, b.ointer, 1);
            },
            else => {
                // Horner's scheme over the whole chunk.
                llk.ew(k, b.eye, b.scr3, b.tinv, cc, .add);
                llk.wait(k, b.tinv, cc);
                var horner = k.openFor(2, C, 1, .{});
                {
                    llk.mm(k, b.scr3, b.tinv, b.scr1, ct, ct, ct, false);
                    llk.wait(k, b.scr1, cc);
                    llk.pop(k, b.tinv, cc);
                    llk.ew(k, b.eye, b.scr1, b.tinv, cc, .add);
                    llk.wait(k, b.tinv, cc);
                    llk.pop(k, b.scr1, cc);
                }
                horner.yield(.{});
                llk.pop(k, b.scr3, cc);
            },
        }

        // kd = k_beta * decay_exp
        llk.bcast(k, b.kbeta, b.decay_exp, b.w, ct, kt, .mul, .cols);
        llk.wait(k, b.w, ck);
        llk.pop(k, b.kbeta, ck);

        // intra = (q @ k^T) * l_mask; q_decay; k_dec_t
        llk.mm(k, b.q, b.k, b.scr1, ct, kt, ct, true);
        llk.wait(k, b.scr1, cc);
        llk.ew(k, b.scr1, b.lmask, b.intra, cc, .mul);
        llk.wait(k, b.intra, cc);
        llk.pop(k, b.scr1, cc);
        llk.pop(k, b.lmask, cc);
        llk.bcast(k, b.q, b.decay_exp, b.qdecay, ct, kt, .mul, .cols);
        llk.wait(k, b.qdecay, ck);
        llk.pop(k, b.q, ck);
        llk.bcast(k, b.k, b.decayfac, b.scr1, ct, kt, .mul, .cols);
        llk.wait(k, b.scr1, ck);
        llk.pop(k, b.k, ck);
        llk.transposeBlock(k, b.scr1, b.kdec_t, ct, kt);
        llk.pop(k, b.scr1, ck);

        // dl = exp(g_sum) = decayfac * decay_exp, in its first element
        llk.ew(k, b.decayfac, b.decay_exp, b.vnew, 1, .mul);
        llk.wait(k, b.vnew, 1);
        llk.pop(k, b.decayfac, ct);
        llk.pop(k, b.decay_exp, ct);
    }
    items.yield(.{});
}

// =============================================================================
// Scan (phase B)
// =============================================================================

pub const ScanConfig = struct {
    /// Tiles per chunk, key dim, this core's value block, the full value dim.
    ct: u32,
    kt: u32,
    vt: u32,
    vt_full: u32,
};

pub const scan_reader_cbs = [_][]const u8{ "s", "u", "w", "qdecay", "intra", "kdec_t", "decayfac", "tinv" };

/// `reader_chunk_gdn_scan.cpp`: the initial state's value block once, then
/// per chunk the seven prep outputs. Runtime args: head, value block,
/// chunks, then the addresses of `v_beta`, `kd`, `q_decay`, `intra`,
/// `k_dec_t`, `dl`, `t_inv` and the initial state.
pub fn scanReader(k: *Builder, c: ScanConfig) void {
    const b = declare(k, &scan_reader_cbs);
    const h, const vb, const nc = runtimeArgs(3, k, 0);
    const tb = k.op(.get_tile_size, .{b.u});
    const vb_acc, const kd_acc, const qd_acc, const it_acc, const kc_acc, const dl_acc, const ti_acc, const s0_acc =
        accessors(8, k, scan_reader_cbs.len, runtimeArgs(8, k, 3), @splat(tb));
    const cc = c.ct * c.ct;
    const ck = c.ct * c.kt;
    const kc = c.kt * c.ct;

    // This core's value block: `vt` of each row's `vt_full` tiles.
    const col = vb.mul(c.vt);
    llk.readBlock(k, s0_acc, b.s, h.mul(c.kt * c.vt_full).add(col), c.kt, c.vt, c.vt_full, tb);
    var chunks, const ci = llk.range(k, nc);
    {
        const hc = h.mul(nc).add(ci);
        llk.readBlock(k, vb_acc, b.u, hc.mul(c.ct * c.vt_full).add(col), c.ct, c.vt, c.vt_full, tb);
        llk.readTiles(k, kd_acc, b.w, hc.mul(ck), ck, tb);
        llk.readTiles(k, qd_acc, b.qdecay, hc.mul(ck), ck, tb);
        llk.readTiles(k, it_acc, b.intra, hc.mul(cc), cc, tb);
        llk.readTiles(k, kc_acc, b.kdec_t, hc.mul(kc), kc, tb);
        llk.readTiles(k, dl_acc, b.decayfac, hc, 1, tb);
        llk.readTiles(k, ti_acc, b.tinv, hc.mul(cc), cc, tb);
    }
    chunks.yield(.{});
}

pub const scan_writer_cbs = [_][]const u8{ "out", "final_s" };

/// `writer_chunk_gdn_scan.cpp`: this core's value block of every chunk's
/// output, then of the final state. Runtime args: head, value block,
/// chunks, the addresses of the output and the final state.
pub fn scanWriter(k: *Builder, c: ScanConfig) void {
    const b = declare(k, &scan_writer_cbs);
    const h, const vb, const nc, const o_addr, const fs_addr = runtimeArgs(5, k, 0);
    const tb_o = k.op(.get_tile_size, .{b.out});
    const tb_fs = k.op(.get_tile_size, .{b.final_s});
    const o_acc, const fs_acc = accessors(2, k, scan_writer_cbs.len, .{ o_addr, fs_addr }, .{ tb_o, tb_fs });
    const col = vb.mul(c.vt);

    var chunks, const ci = llk.range(k, nc);
    llk.writeBlock(k, o_acc, b.out, h.mul(nc).add(ci).mul(c.ct * c.vt_full).add(col), c.ct, c.vt, c.vt_full, tb_o);
    chunks.yield(.{});
    llk.writeBlock(k, fs_acc, b.final_s, h.mul(c.kt * c.vt_full).add(col), c.kt, c.vt, c.vt_full, tb_fs);
}

pub const scan_compute_cbs = [_][]const u8{ "w", "u", "out", "s", "s2", "s3", "final_s", "scr1", "ointer", "tinv", "vnew", "qdecay", "intra", "kdec_t", "supd", "decayfac", "stmp" };

/// `chunk_gdn_scan.cpp`: the recurrence over this core's value block, the
/// state carried on-core. Runtime arg: chunks.
pub fn scanCompute(k: *Builder, c: ScanConfig) void {
    const b = declare(k, &scan_compute_cbs);
    const nc = k.runtimeArg(0);
    const ct, const kt, const vt = .{ c.ct, c.kt, c.vt };
    const cc = ct * ct;
    const ck = ct * kt;
    const cv = ct * vt;
    const kv = kt * vt;
    const kc = kt * ct;
    // The scan's names for the prep outputs it reads.
    const cb_kd, const cb_vbeta, const cb_dl = .{ b.w, b.u, b.decayfac };

    k.op(.compute_kernel_hw_startup, .{ cb_kd, cb_vbeta, b.out });
    var chunks, const ci = llk.range(k, nc);
    {
        // The state in single-producer buffers: the reader's initial state
        // for chunk 0, then a ping-pong of compute-produced ones.
        const odd = ci.bitAnd(1).ne(0);
        const cur_s = k.selectCb(ci.eq(0), b.s, k.selectCb(odd, b.s2, b.s3));
        const nxt_s = k.selectCb(odd, b.s3, b.s2);
        const dst = k.selectCb(ci.eq(nc.sub(1)), b.final_s, nxt_s);

        // v_new = t_inv @ (v_beta - kd @ S)
        llk.wait(k, cb_kd, ck);
        llk.wait(k, cur_s, kv);
        llk.mm(k, cb_kd, cur_s, b.scr1, ct, kt, vt, false);
        llk.wait(k, b.scr1, cv);
        llk.pop(k, cb_kd, ck);
        llk.wait(k, cb_vbeta, cv);
        llk.ew(k, cb_vbeta, b.scr1, b.ointer, cv, .sub);
        llk.wait(k, b.ointer, cv);
        llk.pop(k, cb_vbeta, cv);
        llk.pop(k, b.scr1, cv);
        llk.wait(k, b.tinv, cc);
        llk.mm(k, b.tinv, b.ointer, b.vnew, ct, ct, vt, false);
        llk.wait(k, b.vnew, cv);
        llk.pop(k, b.tinv, cc);
        llk.pop(k, b.ointer, cv);
        // o = q_decay @ S + intra @ v_new
        llk.wait(k, b.qdecay, ck);
        llk.mm(k, b.qdecay, cur_s, b.ointer, ct, kt, vt, false);
        llk.wait(k, b.ointer, cv);
        llk.pop(k, b.qdecay, ck);
        llk.wait(k, b.intra, cc);
        llk.mm(k, b.intra, b.vnew, b.scr1, ct, ct, vt, false);
        llk.wait(k, b.scr1, cv);
        llk.pop(k, b.intra, cc);
        llk.ew(k, b.ointer, b.scr1, b.out, cv, .add);
        llk.pop(k, b.ointer, cv);
        llk.pop(k, b.scr1, cv);
        // S = S * dl + k_dec_t @ v_new
        llk.wait(k, b.kdec_t, kc);
        llk.mm(k, b.kdec_t, b.vnew, b.supd, kt, ct, vt, false);
        llk.wait(k, b.supd, kv);
        llk.pop(k, b.kdec_t, kc);
        llk.pop(k, b.vnew, cv);
        llk.wait(k, cb_dl, 1);
        llk.bcast(k, cur_s, cb_dl, b.stmp, 1, kv, .mul, .scalar);
        llk.wait(k, b.stmp, kv);
        llk.pop(k, cb_dl, 1);
        llk.pop(k, cur_s, kv);
        llk.ew(k, b.stmp, b.supd, dst, kv, .add);
        llk.pop(k, b.stmp, kv);
        llk.pop(k, b.supd, kv);
    }
    chunks.yield(.{});
}
