//! SM100 PTX primitives for hand-scheduled CuTe kernels: inline assembly over
//! 32-bit shared-memory addresses and 64-bit global addresses, for the pieces
//! the CuTe builder does not wrap yet (mbarriers, cp.async with zero-fill,
//! tcgen05 MMA / TMEM traffic, vector loads, FP8/FP4 conversions, cluster
//! shared memory).
//!
//! Conventions: shared-memory addresses are `i32` values in the shared window,
//! global addresses are `i64`, TMEM addresses are `i32` (lane << 16 | column).
const std = @import("std");

const mlir = @import("mlir");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;
const DType = cute.DType;

fn inlineAsm(b: *B, text: []const u8, constraints: []const u8, operands: []const V, result: ?*const mlir.Type, side_effects: bool) ?V {
    const alloc = b.arena.allocator();
    const inner = alloc.alloc(*const mlir.Value, operands.len) catch @panic("OOM");
    for (operands, inner) |o, *x| x.* = o.inner;
    var attrs: std.ArrayList(mlir.NamedAttribute) = .empty;
    attrs.append(alloc, .named(b.ctx, "asm_string", .string(b.ctx, text))) catch @panic("OOM");
    attrs.append(alloc, .named(b.ctx, "constraints", .string(b.ctx, constraints))) catch @panic("OOM");
    if (side_effects) attrs.append(alloc, .named(b.ctx, "has_side_effects", .unit(b.ctx))) catch @panic("OOM");
    const op = mlir.Operation.make(b.ctx, "llvm.inline_asm", .{
        .operands = .{ .flat = inner },
        .results = .{ .flat = if (result) |r| &.{r} else &.{} },
        .attributes = attrs.items,
        .location = b.loc(),
    });
    if (result == null) {
        b.emitVoid(op);
        return null;
    }
    return b.emit(op);
}

/// Side-effecting instruction(s) without results.
pub fn exec(b: *B, text: []const u8, constraints: []const u8, operands: []const V) void {
    _ = inlineAsm(b, text, constraints, operands, null, true);
}

/// One result. `volatile_` keeps it from being hoisted/CSE'd (loads, waits).
pub fn val(b: *B, dtype: DType, text: []const u8, constraints: []const u8, operands: []const V, volatile_: bool) V {
    return inlineAsm(b, text, constraints, operands, dtype.toMlir(b.ctx), volatile_).?;
}

/// `n` results of the same dtype (returned through an LLVM struct).
pub fn vals(b: *B, comptime n: usize, dtype: DType, text: []const u8, constraints: []const u8, operands: []const V, volatile_: bool) [n]V {
    @setEvalBranchQuota(100_000);
    const alloc = b.arena.allocator();
    var ty: std.Io.Writer.Allocating = .init(alloc);
    ty.writer.writeAll("!llvm.struct<(") catch @panic("OOM");
    for (0..n) |i| ty.writer.print("{s}{s}", .{ if (i == 0) "" else ", ", @tagName(dtype) }) catch @panic("OOM");
    ty.writer.writeAll(")>") catch @panic("OOM");
    const struct_ty = b.parseType(ty.written());
    const s = inlineAsm(b, text, constraints, operands, struct_ty, volatile_).?;
    var out: [n]V = undefined;
    for (0..n) |i| {
        out[i] = b.emit(mlir.Operation.make(b.ctx, "llvm.extractvalue", .{
            .operands = .{ .flat = &.{s.inner} },
            .results = .{ .flat = &.{dtype.toMlir(b.ctx)} },
            .attributes = &.{.named(b.ctx, "position", .denseArray(b.ctx, .i64, &.{@intCast(i)}))},
            .location = b.loc(),
        }));
    }
    return out;
}

/// Constraint string "=r,=r,...,<tail>" for `n` outputs.
pub fn outs(comptime n: usize, comptime letter: []const u8, comptime tail: []const u8) []const u8 {
    @setEvalBranchQuota(100_000);
    comptime var s: []const u8 = "";
    inline for (0..n) |i| s = s ++ (if (i == 0) "" else ",") ++ "=" ++ letter;
    return s ++ (if (tail.len > 0) "," else "") ++ tail;
}

/// "{$0, $1, ..., $(n-1)}" register list starting at operand `first`.
pub fn regList(comptime first: usize, comptime n: usize) []const u8 {
    @setEvalBranchQuota(100_000);
    comptime var s: []const u8 = "{";
    inline for (0..n) |i| s = s ++ std.fmt.comptimePrint("{s}${d}", .{ if (i == 0) "" else ", ", first + i });
    return s ++ "}";
}

// ---- addresses --------------------------------------------------------------

/// Allocate `bytes` of compiler-visible dynamic shared memory (1024-aligned) and
/// return its base address in the shared window.
pub fn sharedStorage(b: *B, comptime bytes: i64) V {
    const storage = b.allocSmemStorage(.i8, 1024, b.layoutType(b.layoutSpec(.{bytes}, .{1})), 0, &.{std.fmt.comptimePrint("storage:{d}:0", .{bytes})});
    const base = b.getIterTyped(storage, b.ptrTy(.i8, .smem, 1024) catch @panic("bad smem ptr")).value();
    return ptrToInt(b, base, .i32);
}

pub fn ptrToInt(b: *B, ptr: V, dtype: DType) V {
    return b.ptrToInt(ptr, dtype);
}

pub fn globalAddress(b: *B, ptr: V) V {
    return ptrToInt(b, ptr, .i64);
}

// ---- barriers, fences ---------------------------------------------------------

pub fn barSync(b: *B, id: i32, threads: i32) void {
    exec(b, std.fmt.allocPrint(b.arena.allocator(), "bar.sync {d}, {d};", .{ id, threads }) catch @panic("OOM"), "", &.{});
}

pub fn mbarInit(b: *B, bar: V, count: i32) void {
    exec(b, "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r", &.{ bar, b.cst(.i32, count) });
}

pub fn fenceMbarInit(b: *B) void {
    exec(b, "fence.mbarrier_init.release.cluster;", "", &.{});
}

pub fn mbarArrive(b: *B, bar: V) void {
    exec(b, "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", &.{bar});
}

/// Spin until the phase with the given parity (0/1) has completed.
pub fn mbarWait(b: *B, bar: V, parity: V) void {
    exec(b,
        \\{ .reg .pred p; WAIT_${:uid}: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 0x989680; @!p bra WAIT_${:uid}; }
    , "r,r", &.{ bar, parity });
}

pub fn fenceProxyAsync(b: *B) void {
    exec(b, "fence.proxy.async.shared::cta;", "", &.{});
}

// ---- L2 cache policy ------------------------------------------------------------

pub fn evictFirstPolicy(b: *B) V {
    return val(b, .i64, "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", &.{}, false);
}

// ---- global / shared scalar and vector access ----------------------------------

pub fn ldGlobalU32(b: *B, addr: V) V {
    return val(b, .i32, "ld.global.nc.u32 $0, [$1];", "=r,l", &.{addr}, true);
}

pub fn ldGlobalV4(b: *B, addr: V) [4]V {
    return vals(b, 4, .i32, "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", &.{addr}, true);
}

pub fn ldGlobalV2(b: *B, addr: V) [2]V {
    return vals(b, 2, .i32, "ld.global.nc.v2.u32 {$0, $1}, [$2];", "=r,=r,l", &.{addr}, true);
}

pub fn stGlobalU32(b: *B, addr: V, v: V) void {
    exec(b, "st.global.u32 [$0], $1;", "l,r", &.{ addr, v });
}

pub fn stGlobalV4(b: *B, addr: V, v: [4]V) void {
    exec(b, "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", &.{ addr, v[0], v[1], v[2], v[3] });
}

pub fn ldSharedU32(b: *B, addr: V) V {
    return val(b, .i32, "ld.shared.u32 $0, [$1];", "=r,r", &.{addr}, true);
}

pub fn ldSharedF32(b: *B, addr: V) V {
    return val(b, .f32, "ld.shared.f32 $0, [$1];", "=f,r", &.{addr}, true);
}

pub fn ldSharedV4(b: *B, addr: V) [4]V {
    return vals(b, 4, .i32, "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", &.{addr}, true);
}

pub fn ldSharedV2(b: *B, addr: V) [2]V {
    return vals(b, 2, .i32, "ld.shared.v2.u32 {$0, $1}, [$2];", "=r,=r,r", &.{addr}, true);
}

pub fn stSharedU32(b: *B, addr: V, v: V) void {
    exec(b, "st.shared.u32 [$0], $1;", "r,r", &.{ addr, v });
}

pub fn stSharedF32(b: *B, addr: V, v: V) void {
    exec(b, "st.shared.f32 [$0], $1;", "r,f", &.{ addr, v });
}

pub fn stSharedV4(b: *B, addr: V, v: [4]V) void {
    exec(b, "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", &.{ addr, v[0], v[1], v[2], v[3] });
}

pub fn stSharedV2(b: *B, addr: V, v: [2]V) void {
    exec(b, "st.shared.v2.u32 [$0], {$1, $2};", "r,r,r", &.{ addr, v[0], v[1] });
}

// ---- thread / warp ---------------------------------------------------------------

pub fn electOne(b: *B) V {
    return val(b, .i32, "{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 $0, 1, 0, p; }", "=r", &.{}, true);
}

// ---- tcgen05 ---------------------------------------------------------------------

/// Warp-wide TMEM allocation; the base address is written to `holder` (smem).
pub fn tmemAlloc(b: *B, holder: V, comptime columns: u32) void {
    exec(b, std.fmt.comptimePrint("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {d};", .{columns}), "r", &.{holder});
}

pub fn tmemRelinquish(b: *B) void {
    exec(b, "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "", &.{});
}

pub fn tmemDealloc(b: *B, taddr: V, comptime columns: u32) void {
    exec(b, std.fmt.comptimePrint("tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {d};", .{columns}), "r", &.{taddr});
}

pub fn tcgenFenceBefore(b: *B) void {
    exec(b, "tcgen05.fence::before_thread_sync;", "", &.{});
}

pub fn tcgenFenceAfter(b: *B) void {
    exec(b, "tcgen05.fence::after_thread_sync;", "", &.{});
}

/// Arrive on `bar` once all previously issued MMAs of this thread complete.
pub fn tcgenCommit(b: *B, bar: V) void {
    exec(b, "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r", &.{bar});
}

/// D (tmem) = A (smem desc) * B (smem desc) [+ D]. `accumulate` is an i32 (0/1).
pub fn mmaSS(b: *B, d: V, adesc: V, bdesc: V, idesc: V, accumulate: V) void {
    exec(b, "{ .reg .pred p; setp.ne.b32 p, $4, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }", "r,l,l,r,r", &.{ d, adesc, bdesc, idesc, accumulate });
}

/// 32 lanes x 32 bits x n columns: thread i of the warp gets row (warp_base + i).
pub fn tmemLoad32x32b(b: *B, comptime n: usize, taddr: V) [n]V {
    @setEvalBranchQuota(100_000);
    const text = comptime std.fmt.comptimePrint("tcgen05.ld.sync.aligned.32x32b.x{d}.b32 {s}, [${d}];", .{ n, regList(0, n), n });
    return vals(b, n, .i32, text, comptime outs(n, "r", "r"), &.{taddr}, true);
}

pub fn tmemStore32x32b(b: *B, comptime n: usize, taddr: V, v: [n]V) void {
    const text = comptime std.fmt.comptimePrint("tcgen05.st.sync.aligned.32x32b.x{d}.b32 [$0], {s};", .{ n, regList(1, n) });
    var ops: [n + 1]V = undefined;
    ops[0] = taddr;
    for (0..n) |i| ops[i + 1] = v[i];
    const constraints = comptime blk: {
        var s: []const u8 = "r";
        for (0..n) |_| s = s ++ ",r";
        break :blk s;
    };
    exec(b, text, constraints, &ops);
}

pub fn tmemWaitLoad(b: *B) void {
    exec(b, "tcgen05.wait::ld.sync.aligned;", "", &.{});
}

pub fn tmemWaitStore(b: *B) void {
    exec(b, "tcgen05.wait::st.sync.aligned;", "", &.{});
}

// ---- UMMA descriptors -------------------------------------------------------------

pub const Major = enum(u1) { k = 0, mn = 1 };

/// kind::f16 instruction descriptor: BF16 A/B, F32 accumulator.
pub fn instrDescBf16(comptime m: u32, comptime n: u32, comptime a_major: Major, comptime b_major: Major) u32 {
    std.debug.assert((m == 64 or m == 128) and n % 8 == 0 and n >= 8 and n <= 256);
    return (1 << 4) | // c_format = F32
        (1 << 7) | // a_format = BF16
        (1 << 10) | // b_format = BF16
        (@as(u32, @intFromEnum(a_major)) << 15) |
        (@as(u32, @intFromEnum(b_major)) << 16) |
        ((n >> 3) << 17) |
        ((m >> 4) << 24);
}

/// SM100 shared-memory matrix descriptor with 128-byte swizzle. `lbo`/`sbo` in bytes.
/// For K-major operands `lbo` is ignored (pass 16); `sbo` is the 8-row group stride.
/// For MN-major operands `lbo` is the stride between 64-element MN atoms and `sbo`
/// the stride between 8-row K groups.
pub fn smemDescSw128(b: *B, addr: V, comptime lbo: u32, comptime sbo: u32) V {
    const hi: u64 = (@as(u64, sbo >> 4) << 32) | (@as(u64, 1) << 46) | (@as(u64, 2) << 61);
    const lo: u64 = @as(u64, lbo >> 4) << 16;
    const start = addr.to(.i64).shrLogical(4).bitAnd(0x3FFF);
    return start.bitOr(b.cst(.i64, @as(i64, @bitCast(hi | lo))));
}

/// Byte offset of element (row, col) of a bf16 tile stored as SW128 K-major
/// atoms (8 rows x 64 columns, 1024 bytes), atoms tiled along rows first:
/// `rows` rows per 64-column slab, slabs `rows * 128` bytes apart.
pub fn sw128Offset(b: *B, row: V, col: V, comptime rows: i32) V {
    _ = b;
    const slab = col.shrLogical(6).mul(rows * 128);
    const chunk = col.bitAnd(63).shrLogical(3).bitXor(row.bitAnd(7));
    return slab.add(row.mul(128)).add(chunk.shl(4)).add(col.bitAnd(7).shl(1));
}

// ---- warp reductions and conversions -------------------------------------------------

/// Warp-wide maximum of an f32 (any sign) with one `redux.sync`: floats are
/// mapped to order-preserving signed integers.
pub fn warpMaxF32(b: *B, v: V) V {
    return val(b, .f32,
        \\{ .reg .s32 k, m, t; .reg .pred n; mov.b32 k, $1; shr.s32 t, k, 31; and.b32 t, t, 0x7FFFFFFF; xor.b32 k, k, t;
        \\  redux.sync.max.s32 m, k, 0xffffffff; shr.s32 t, m, 31; and.b32 t, t, 0x7FFFFFFF; xor.b32 m, m, t; mov.b32 $0, m; }
    , "=f,f", &.{v}, true);
}

pub fn warpSumF32(b: *B, v: V) V {
    var x = v;
    inline for (.{ 16, 8, 4, 2, 1 }) |o| x = x.add(b.shuffleXor(x, o));
    return x;
}

pub fn syncWarp(b: *B) void {
    exec(b, "bar.warp.sync 0xffffffff;", "", &.{});
}

/// Two FP8 e4m3 (low 16 bits of `pair`) -> bf16x2.
pub fn e4m3x2ToBf16x2(b: *B, pair: V) V {
    return val(b, .i32, "{ .reg .b16 h; cvt.u16.u32 h, $1; cvt.rn.bf16x2.e4m3x2 $0, h; }", "=r,r", &.{pair}, false);
}

/// Eight packed FP8 e4m3 (two words) -> four bf16x2, each multiplied by the bf16x2 `scale`.
pub fn fp8x8ToBf16x2x4Scaled(b: *B, lo: V, hi: V, scale: V) [4]V {
    return vals(b, 4, .i32,
        \\{ .reg .b16 h0, h1, h2, h3; .reg .b32 c0, c1, c2, c3;
        \\  mov.b32 {h0, h1}, $4; mov.b32 {h2, h3}, $5;
        \\  cvt.rn.bf16x2.e4m3x2 c0, h0; cvt.rn.bf16x2.e4m3x2 c1, h1; cvt.rn.bf16x2.e4m3x2 c2, h2; cvt.rn.bf16x2.e4m3x2 c3, h3;
        \\  mul.rn.bf16x2 $0, c0, $6; mul.rn.bf16x2 $1, c1, $6; mul.rn.bf16x2 $2, c2, $6; mul.rn.bf16x2 $3, c3, $6; }
    , "=r,=r,=r,=r,r,r,r", &.{ lo, hi, scale }, false);
}

/// Eight packed FP4 e2m1 (one word, even element in the low nibble) -> four
/// bf16x2, each multiplied by the bf16x2 `scale`.
pub fn fp4x8ToBf16x2x4Scaled(b: *B, word: V, scale: V) [4]V {
    return vals(b, 4, .i32,
        \\{ .reg .b8 b0, b1, b2, b3; .reg .b32 c0, c1, c2, c3;
        \\  mov.b32 {b0, b1, b2, b3}, $4;
        \\  cvt.rn.bf16x2.e2m1x2 c0, b0; cvt.rn.bf16x2.e2m1x2 c1, b1; cvt.rn.bf16x2.e2m1x2 c2, b2; cvt.rn.bf16x2.e2m1x2 c3, b3;
        \\  mul.rn.bf16x2 $0, c0, $5; mul.rn.bf16x2 $1, c1, $5; mul.rn.bf16x2 $2, c2, $5; mul.rn.bf16x2 $3, c3, $5; }
    , "=r,=r,=r,=r,r,r", &.{ word, scale }, false);
}

/// f32 -> bf16 bits (round to nearest even), in the low 16 bits of an i32.
pub fn f32ToBf16Bits(b: *B, v: V) V {
    return val(b, .i32, "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }", "=r,f", &.{v}, false);
}

/// Two f32 -> packed bf16x2 (`lo` in the low half).
pub fn packBf16x2(b: *B, lo: V, hi: V) V {
    return val(b, .i32, "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", &.{ lo, hi }, false);
}

pub fn stSharedB16(b: *B, addr: V, v: V) void {
    exec(b, "{ .reg .b16 h; cvt.u16.u32 h, $1; st.shared.u16 [$0], h; }", "r,r", &.{ addr, v });
}

pub fn exp2(b: *B, v: V) V {
    return b.unaryF32("ex2.approx.ftz.f32", v);
}

pub fn rcp(b: *B, v: V) V {
    return b.unaryF32("rcp.approx.ftz.f32", v);
}

/// Test (without blocking) whether the phase with the given parity has completed; i32 0/1.
pub fn mbarTryWait(b: *B, bar_: V, parity_: V) V {
    return val(b, .i32, "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [$1], $2; selp.u32 $0, 1, 0, p; }", "=r,r,r", &.{ bar_, parity_ }, true);
}

/// `n` loads of `words` 32-bit words each (n * words results), from `base + offsets[i]`
/// in one statement so their latencies overlap.
pub fn ldSharedBatch(b: *B, comptime n: usize, comptime words: usize, base: V, comptime offsets: [n]i32) [n * words]V {
    comptime var text: []const u8 = "{";
    inline for (0..n) |i| {
        const regs = comptime regList(i * words, words);
        const op = switch (words) {
            1 => "ld.shared.u32",
            2 => "ld.shared.v2.u32",
            4 => "ld.shared.v4.u32",
            else => @compileError("unsupported width"),
        };
        text = text ++ std.fmt.comptimePrint(" {s} {s}, [${d}+{d}];", .{ op, if (words == 1) std.fmt.comptimePrint("${d}", .{i}) else regs, n * words, offsets[i] });
    }
    text = text ++ " }";
    return vals(b, n * words, .i32, text, comptime outs(n * words, "r", "r"), &.{base}, true);
}

/// 16-byte global -> shared async copy (LDGSTS, L2 only) with an L2 cache policy.
pub fn cpAsync16(b: *B, dst: V, src: V, policy: V) void {
    exec(b, "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;", "r,l,l", &.{ dst, src, policy });
}

/// Arrive on `bar` once this thread's prior cp.async copies have landed
/// (the barrier's expected count includes this arrival).
pub fn cpAsyncMbarArrive(b: *B, bar_: V) void {
    exec(b, "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];", "r", &.{bar_});
}

/// 16-byte cp.async of `base + row * row_bytes + offset`; when `row < 0` it
/// zero-fills the destination without reading memory (src-size 0).
pub fn cpAsync16Row(b: *B, dst: V, base: V, row: V, comptime row_bytes: u32, offset: V, policy: V) void {
    exec(b, std.fmt.comptimePrint(
        \\{{ .reg .pred p; .reg .s32 n, r; .reg .s64 w; .reg .u64 a; setp.ge.s32 p, $2, 0; selp.s32 n, 16, 0, p; selp.s32 r, $2, 0, p;
        \\  cvt.s64.s32 w, r; mad.lo.s64 a, w, {d}, $1; add.s64 a, a, $3;
        \\  cp.async.cg.shared.global.L2::cache_hint [$0], [a], 16, n, $4; }}
    , .{row_bytes}), "r,l,r,l,l", &.{ dst, base, row, offset, policy });
}

// ---- clusters / distributed shared memory ---------------------------------------------

pub fn clusterCtaRank(b: *B) V {
    return val(b, .i32, "mov.u32 $0, %cluster_ctarank;", "=r", &.{}, false);
}

/// Address of the same shared-memory location in CTA `rank` of the cluster.
pub fn mapaShared(b: *B, addr: V, rank: V) V {
    return val(b, .i32, "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", &.{ addr, rank }, false);
}

/// Full cluster barrier (all non-exited threads of every CTA), release/acquire.
pub fn clusterSync(b: *B) void {
    exec(b, "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;", "", &.{});
}

/// Predicated 16-bit shared store of the bf16 rounding of `v` (f32) when `pred` (i32) != 0.
pub fn stSharedBf16If(b: *B, addr: V, v: V, pred: V) void {
    exec(b, "{ .reg .pred p; .reg .b16 h; setp.ne.b32 p, $2, 0; cvt.rn.bf16.f32 h, $1; @p st.shared.u16 [$0], h; }", "r,f,r", &.{ addr, v, pred });
}

/// An i32 constant the compiler cannot see through, e.g. a loop bound that
/// must keep a loop rolled (small code, fewer instruction-cache misses).
pub fn hiddenConst(b: *B, comptime value: i32) V {
    return val(b, .i32, std.fmt.comptimePrint("mov.u32 $0, {d};", .{value}), "=r", &.{}, true);
}

pub fn stClusterF32(b: *B, addr: V, v: V) void {
    exec(b, "st.shared::cluster.f32 [$0], $1;", "r,f", &.{ addr, v });
}

pub fn stClusterV2(b: *B, addr: V, v: [2]V) void {
    exec(b, "st.shared::cluster.v2.u32 [$0], {$1, $2};", "r,r,r", &.{ addr, v[0], v[1] });
}

/// Two f32 -> packed f16x2 (`lo` in the low half).
pub fn packF16x2(b: *B, lo: V, hi: V) V {
    return val(b, .i32, "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f", &.{ lo, hi }, false);
}

/// Packed f16x2 -> two f32.
pub fn unpackF16x2(b: *B, v: V) [2]V {
    return vals(b, 2, .f32, "{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.f16 $0, l; cvt.f32.f16 $1, h; }", "=f,=f,r", &.{v}, false);
}

/// Wait until all of this thread's cp.async copies have landed.
pub fn cpAsyncWaitAll(b: *B) void {
    exec(b, "cp.async.wait_all;", "", &.{});
}

/// Prefetch the global line holding `addr` into L2.
pub fn prefetchL2(b: *B, addr: V) void {
    exec(b, "prefetch.global.L2::evict_last [$0];", "l", &.{addr});
}

/// bar.red.or over `threads` threads of named barrier `id`: i32 1 if any `pred` (i32) != 0.
pub fn barRedOr(b: *B, comptime id: i32, comptime threads: i32, pred: V) V {
    return val(b, .i32, std.fmt.comptimePrint("{{ .reg .pred p, q; setp.ne.b32 q, $1, 0; bar.red.or.pred p, {d}, {d}, q; selp.u32 $0, 1, 0, p; }}", .{ id, threads }), "=r,r", &.{pred}, true);
}

/// Reduction of `blockReduce`-style helpers.
pub const ReduceOp = enum { max, sum };
