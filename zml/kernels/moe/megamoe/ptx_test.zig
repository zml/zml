//! Emission test of `ptx.zig`: one `cuda.kernel` calls every primitive; the module must verify
//! and carry DeepGEMM's PTX strings. (Execution is covered by `gpu_test.zig`.)
const std = @import("std");

const mlir = @import("mlir");
const cute = @import("kernels/cute/builder");

const ptx = @import("ptx.zig");
const layout = @import("layout.zig");

const B = cute.Builder;
const V = cute.Value;

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (cute.dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("missing {s}\n", .{needle});
        return error.TestExpectedContains;
    }
}

fn maxOp(b: *B, x: V, y: V) V {
    return ptx.fmaxf(b, x, y);
}

const Cond = struct {
    ptr: V,
    pub fn pending(self: Cond, b: *B) V {
        return ptx.ldAcq(b, self.ptr).ne(0);
    }
};

/// Calls every primitive of `ptx.zig` once (`g`: an i64 global address, `f`: an f32), for the
/// emission test here and the ptxas compile test in `gpu_test.zig`.
/// Groups of `emitEveryPrimitive` (all by default).
pub const Groups = struct {
    /// Primitives of the `shared` group (bit i: the i-th call).
    shared_mask: u16 = 0xFFF,
    special: bool = true,
    memory: bool = true,
    shared: bool = true,
    mbarrier: bool = true,
    tma: bool = true,
    tcgen05: bool = true,
    collectives: bool = true,
    numerics: bool = true,
    control: bool = true,
};

/// Calls every primitive of `ptx.zig` once (`g`: an i64 global address, `s`: an i32 shared
/// address, `f`: an f32), for the emission test here and the ptxas compile test in `gpu_test.zig`.
pub fn emitEveryPrimitive(bp: *B, g: V, s: V, f: V, groups: Groups) void {
    const u = bp.threadIdx().x;
    const lane = ptx.getLaneIdx(bp);

    if (groups.special) {
        _ = ptx.getSmIdx(bp);
        _ = ptx.getGridIdx(bp);
        _ = ptx.blockRankInCluster(bp);
        _ = ptx.clock64(bp);
        _ = ptx.canonicalWarpIdxSync(bp);
        _ = ptx.electOneSync(bp);
        ptx.syncAligned(bp, 128, 0);
        ptx.syncUnaligned(bp, 384, 1);
        ptx.syncWarp(bp);
        ptx.clusterSyncWithRelaxedArrive(bp);
        ptx.clusterSync(bp);
        ptx.gridDependencySynchronize(bp);
        ptx.warpgroupRegDealloc(bp, 96);
        ptx.warpgroupRegAlloc(bp, 168);
    }

    if (groups.memory) {
        ptx.fenceAcqRelCta(bp);
        _ = ptx.ldVolatileU64(bp, g);
        _ = ptx.ldVolatileU32(bp, g);
        _ = ptx.ldAcq(bp, g);
        _ = ptx.ldAcqSysU64(bp, g);
        _ = ptx.ldAcqSysS32(bp, g);
        _ = ptx.ldAcqGpuU64(bp, g);
        _ = ptx.atomicAddU32(bp, g, 1);
        _ = ptx.atomicAddU64(bp, g, g);
        _ = ptx.atomicAddSys(bp, g, g);
        _ = ptx.atomicAddRel(bp, g, u);
        ptx.stRelSys(bp, g, g);
        ptx.redAddS32(bp, g, -1);
        ptx.redAddU32(bp, g, u);
        ptx.redXorRel(bp, g, g);
        ptx.redAddRel(bp, g, 1);
        ptx.redAddRelSys(bp, g, 1);
        _ = ptx.atomicAddBlockShared(bp, s, 1);
        ptx.stGlobal(bp, 4, g, .{ u, u, u, u });
        _ = ptx.ldGlobal(bp, 2, g);
        ptx.stGlobalU64(bp, g, g);
        _ = ptx.ldGlobalU64(bp, g);
        ptx.stGlobalU8(bp, g, u);
    }

    if (groups.shared) {
        const m = groups.shared_mask;
        if (m & 1 != 0) _ = ptx.ldSharedU32(bp, s);
        if (m & 2 != 0) _ = ptx.ldSharedF32x2(bp, s);
        if (m & 4 != 0) _ = ptx.ldSharedF32x4(bp, s);
        if (m & 8 != 0) _ = ptx.ldSharedU32x4(bp, s);
        if (m & 16 != 0) ptx.stSharedU32x4(bp, s, .{ u, u, u, u });
        if (m & 32 != 0) ptx.stSharedU32(bp, s, u);
        if (m & 64 != 0) ptx.stSharedF32x2(bp, s, .{ f, f });
        if (m & 128 != 0) ptx.stSharedBulk(bp, s, 2048);
        if (m & 256 != 0) _ = ptx.mapaShared(bp, s, lane);
        if (m & 512 != 0) ptx.stAsyncCluster(bp, s, &.{ u, u, u, u, u, u, u, u }, lane, s);
        if (m & 1024 != 0) ptx.stmatrixU8x4T(bp, u, s);
        if (m & 2048 != 0) ptx.stmatrixU32x4T(bp, .{ u, u, u, u }, s);
    }

    if (groups.mbarrier) {
        ptx.mbarrierInit(bp, s, 4);
        ptx.fenceBarrierInit(bp);
        ptx.mbarrierWait(bp, s, u.bitAnd(1));
        _ = ptx.mbarrierWaitAndFlipPhase(bp, s, u.bitAnd(1));
        ptx.mbarrierArrive(bp, s);
        ptx.mbarrierArriveCluster(bp, s, bp.cst(.i32, 0));
        ptx.mbarrierArriveAndExpectTx(bp, s, 44032);
        ptx.mbarrierArriveAndExpectTxCluster(bp, s, 32, lane, bp.cst(.i32, 1));
        ptx.mbarrierArriveAndSetTx(bp, s, 5120);
    }

    if (groups.tma) {
        ptx.prefetchTmaDescriptor(bp, g);
        ptx.tma2SmLoad2d(bp, g, s, ptx.CacheHint.evict_normal, s, u, u);
        ptx.tma2SmLoad4d(bp, g, s, ptx.CacheHint.evict_normal, s, .{ u, u, u, u });
        ptx.tmaCopy2Sm(bp, 128, 96, 128, 1, g, s, s, u, &.{u});
        ptx.tmaCopy2Sm(bp, 256, 1, 0, 4, g, s, s, u, &.{u});
        ptx.tmaCopy2Sm(bp, 128, 128, 128, 1, g, s, s, u, &.{ u, u, u });
        ptx.tmaLoad1d(bp, s, g, s, 5120, ptx.CacheHint.evict_first);
        ptx.tmaStore1d(bp, g, s, 5120, ptx.CacheHint.evict_normal);
        ptx.tmaStore2d(bp, g, s, u, u);
        ptx.tmaStoreFence(bp);
        ptx.fenceViewAsyncShared(bp);
        ptx.tmaStoreArrive(bp);
        ptx.tmaStoreWait(bp, 1);
    }

    if (groups.tcgen05) {
        ptx.tmemAllocate2Sm(bp, 512, s);
        ptx.tcgen05BeforeThreadSync(bp);
        ptx.tcgen05AfterThreadSync(bp);
        const t = ptx.tmemLoad16dp256b1x(bp, u);
        ptx.fenceViewAsyncTmemLoad(bp);
        _ = ptx.fenceViewAsyncTmemLoadValues(bp, 4, t);
        const sf_desc = ptx.makeSfDesc(bp, s);
        ptx.utccp4x32dp128bit2cta(bp, ptx.replaceSmemDescAddr(bp, sf_desc, s), u);
        const k: layout.Kernel = undefined;
        _ = k;
        const idesc = ptx.updateInstrDescWithUmmaN(bp, bp.cst(.i32, @as(i32, @bitCast(@as(u32, 0x10b00280)))), u);
        const a_desc = ptx.makeUmmaDescKSw128(bp, s);
        const lo = ptx.advanceUmmaDescLo(bp, ptx.descLo(bp, a_desc), 96 * 128, 64);
        ptx.mmaMxf8f6f42x1SmSs(bp, ptx.withDescLo(bp, a_desc, lo), a_desc, u, u, ptx.makeRuntimeInstrDescWithSfId(bp, idesc, 2, 2), u, u);
        ptx.ummaArriveMulticast2x1Sm(bp, s, 0b11);
        ptx.tmemFree2Sm(bp, bp.cst(.i32, 0), 512);
    }

    if (groups.collectives) {
        _ = ptx.shflSync(bp, f, 3);
        _ = ptx.exchange64(bp, g, lane);
        _ = ptx.shflXorSync(bp, u, 4);
        _ = ptx.shflUpSync(bp, u, 2);
        _ = ptx.reduceAddSync(bp, u);
        _ = ptx.reduceMinSync(bp, u);
        const mask = ptx.ballotSync(bp, u.gt(5));
        _ = ptx.allSync(bp, u.gt(7));
        _ = ptx.popc(bp, mask);
        _ = ptx.ffs(bp, mask);
        _ = ptx.fns(bp, mask, 0, u);
        _ = ptx.warpReduce(bp, 4, true, f, maxOp);
        _ = ptx.warpInclusiveSum(bp, u, lane);
    }

    if (groups.numerics) {
        const gate = ptx.float22bfloat162Rn(bp, f, f);
        const clamp = ptx.bf16x2Splat(bp, 10.0);
        const neg_clamp = ptx.bf16x2Splat(bp, -10.0);
        const gated = ptx.hmax2Bf16(bp, ptx.hmin2Bf16(bp, gate, clamp), neg_clamp);
        const gf = ptx.bfloat1622float2(bp, gated);
        const e = [2]V{ ptx.expfFastNeg(bp, gf[0]), ptx.expfFast(bp, gf[1]) };
        const den = ptx.fadd2Rn(bp, .{ bp.cst(.f32, 1.0), bp.cst(.f32, 1.0) }, e);
        const act = ptx.fmul2Rn(bp, gf, .{ ptx.fastRcp(bp, den[0]), ptx.fastRcp(bp, den[1]) });
        const amax = ptx.fmaxf(bp, ptx.fabsf(bp, act[0]), ptx.fabsf(bp, act[1]));
        const sf_exp = ptx.getUe8m0SfExp(bp, amax);
        const sf_inv = ptx.getUe8m0SfInv(bp, sf_exp);
        _ = ptx.fp8x4E4m3(bp, act[0], act[1], sf_inv, amax);
        _ = ptx.castIntoBf16AndPack(bp, act[0], act[1]);
        _ = ptx.accumulate(bp, act, gate);
    }

    if (groups.control) {
        _ = ptx.divU(bp, u, 7);
        _ = ptx.remU(bp, u, 7);
        _ = ptx.minU(bp, u, 7);
        _ = ptx.maxU(bp, u, 7);
        _ = ptx.ceilDivU(bp, u, 7);
        _ = ptx.ltU(bp, u, 7);
        _ = ptx.geU(bp, u, 7);
        ptx.waitUntil(bp, 1 << 40, Cond{ .ptr = g });
        ptx.gridSync(bp, 152, g, bp.blockIdx().x, u, ptx.AlignedScope{ .num_threads = 128, .barrier_idx = 0 }, 0);
        ptx.trap(bp);
    }
}

test "every MegaMoE PTX primitive emits and verifies" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try B.openProgram(std.testing.allocator, ctx, "megamoe_ptx_emission");
    defer b.deinit();
    b.beginFunction("megamoe_ptx_emission_kernel", .cuda_kernel);
    const a = try b.declareArgs(.{
        .g = .{ .mlir_type = .{ .type_ = cute.DType.i64.toMlir(ctx) } },
        .f = .{ .mlir_type = .{ .type_ = cute.DType.f32.toMlir(ctx) } },
    });
    emitEveryPrimitive(&b, a.g, b.cst(.i32, 1024), a.f, .{});
    b.endFunction(.{ 512, 1, 1 });

    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    for ([_][]const u8{
        "ld.acquire.gpu.global.b32 $0, [$1];",
        "red.release.gpu.global.xor.b64 [$0], $1;",
        "st.release.sys.global.u64 [$0], $1;",
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$3, $4}], [$2], $5;",
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint",
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3}], [$1];",
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$5], [$6], p;",
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;",
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $1;",
        "tcgen05.ld.sync.aligned.16x256b.x1.b32{$0, $1, $2, $3},[$4];",
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};",
        "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [$0], {$1, $2, $3, $4};",
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [$0], {$1, $2, $3, $4}, [$5];",
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, $2;",
        "setmaxnreg.dec.sync.aligned.u32 96;",
        "add.rn.f32.bf16 $0, $1, $0;",
        "mul.rn.f32x2",
        "ex2.approx.f32",
        "rcp.approx.ftz.f32 $0, $1;",
        "cvt.rn.satfinite.e4m3x2.f32",
        "llvm.extractvalue",
    }) |needle| try expectContains(ir, needle);
}
