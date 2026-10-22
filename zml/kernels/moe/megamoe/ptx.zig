//! The PTX primitives of DeepGEMM's MegaMoE kernel (`sm100_fp8_fp4_mega_moe_impl` and the
//! headers it uses: `ptx/{ld_st,tma,utils,tcgen05}.cuh`, `common/{math,tma_copy,utils}.cuh`,
//! `comm/barrier.cuh`, `mma/sm100.cuh`, `scheduler/mega_moe.cuh`, and the CUTLASS/CuTe
//! primitives it calls: `cutlass/arch/{barrier,reg_reconfig}.h`, `cute/arch/{cluster_sm90,
//! copy_sm90_desc,copy_sm90_tma,copy_sm100_tma,copy_sm100,tmem_allocator_sm100}.hpp`, as
//! vendored by vLLM), for the Zig CuTe builder.
//!
//! One function per primitive, named in camelCase after DeepGEMM / CUTLASS
//! (`ptx::ld_acq` -> `ldAcq`, `cute::elect_one_sync` -> `electOneSync`, ...). Each emits
//! DeepGEMM's exact PTX string through `llvm.inline_asm`; the text is quoted verbatim with
//! LLVM's operand syntax (`%N` -> `$N`, `%%` -> `%`). The CUDA intrinsics DeepGEMM uses
//! (`__shfl_sync`, `__float22bfloat162_rn`, `__expf`, ...) emit the PTX nvcc 13.0 generates for
//! them at `-O3` for `sm_100a` (`gt/intrin.cu`). Inline-asm volatility follows DeepGEMM
//! (`asm volatile` -> `has_side_effects`); every acquire, release, fence and barrier also
//! carries a `~{memory}` clobber (compiler ordering only; the PTX is unchanged).
//!
//! Operands are raw integers: shared-memory addresses `i32` (the `shared::cta` window),
//! global addresses `i64` (generic or global), tensor-memory addresses `i32`
//! (`lane << 16 | column`), tensor-map descriptor addresses `i64` (generic). Unsigned C++
//! values are carried in signless `i32`/`i64`; `f32` stays `f32`. Packed pairs (bf16x2,
//! fp8x4) are `i32` words. Every function takes the builder first.
const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const cute = @import("kernels/cute/builder");

const B = cute.Builder;
const V = cute.Value;
const DType = cute.DType;
const llvm = dialects.llvm;
const arith = dialects.arith;

// ==================== inline assembly ====================

/// `asm volatile` (or plain `asm` with `side_effects = false`) with no result.
pub fn asmVoid(b: *B, text: []const u8, args: []const V, constraints: []const u8) void {
    const inner = b.arena.allocator().alloc(*const mlir.Value, args.len) catch @panic("OOM");
    for (args, inner) |a, *x| x.* = a.inner;
    b.emitVoid(llvm.inline_asm(b.ctx, text, inner, null, .{ .constraints = constraints, .has_side_effects = true }, b.loc()));
}

/// Inline assembly with one result of `dtype`.
pub fn asmValue(b: *B, text: []const u8, args: []const V, dtype: DType, constraints: []const u8, side_effects: bool) V {
    const inner = b.arena.allocator().alloc(*const mlir.Value, args.len) catch @panic("OOM");
    for (args, inner) |a, *x| x.* = a.inner;
    return b.emit(llvm.inline_asm(b.ctx, text, inner, dtype.toMlir(b.ctx), .{ .constraints = constraints, .has_side_effects = side_effects }, b.loc()));
}

/// Inline assembly with `n` results of `dtype` (an `!llvm.struct` result split with
/// `llvm.extractvalue`, how LLVM represents multi-output asm).
pub fn asmValues(b: *B, comptime n: usize, text: []const u8, args: []const V, dtype: DType, constraints: []const u8, side_effects: bool) [n]V {
    const inner = b.arena.allocator().alloc(*const mlir.Value, args.len) catch @panic("OOM");
    for (args, inner) |a, *x| x.* = a.inner;
    const elem = dtype.toMlir(b.ctx);
    const struct_type = b.parseType(std.fmt.allocPrint(b.arena.allocator(), "!llvm.struct<({s})>", .{structFields(b, n, elem)}) catch @panic("OOM"));
    const packed_ = b.emit(llvm.inline_asm(b.ctx, text, inner, struct_type, .{ .constraints = constraints, .has_side_effects = side_effects }, b.loc()));
    var out: [n]V = undefined;
    for (&out, 0..) |*o, i| {
        o.* = b.emit(mlir.Operation.make(b.ctx, "llvm.extractvalue", .{
            .operands = .{ .flat = &.{packed_.inner} },
            .results = .{ .flat = &.{elem} },
            .attributes = &.{.named(b.ctx, "position", .denseArray(b.ctx, .i64, &[_]i64{@intCast(i)}))},
            .location = b.loc(),
        }));
    }
    return out;
}

fn structFields(b: *B, n: usize, elem: *const mlir.Type) []const u8 {
    var w: std.Io.Writer.Allocating = .init(b.arena.allocator());
    for (0..n) |i| w.writer.print("{s}{f}", .{ if (i == 0) "" else ", ", elem }) catch @panic("OOM");
    return w.written();
}

/// `cvta`-style widening of an `i32` shared-memory address to the 64-bit operand
/// `__cvta_generic_to_shared` gives DeepGEMM's `"l"` constraints.
pub fn zext64(b: *B, v: V) V {
    if (v.dtype() == .i64) return v;
    return b.emit(arith.extui(b.ctx, v.inner, DType.i64.toMlir(b.ctx), b.loc()));
}

/// Zero-extension to `dt` (C++ unsigned widening; `Value.to` sign-extends).
pub fn zext(b: *B, v: V, dt: DType) V {
    if (v.dtype() == dt) return v;
    return b.emit(arith.extui(b.ctx, v.inner, dt.toMlir(b.ctx), b.loc()));
}

/// `static_cast<uint32_t>(v)` of a 64-bit value (low word).
pub fn lo32(b: *B, v: V) V {
    return if (v.dtype() == .i32) v else b.emit(arith.trunci(b.ctx, v.inner, DType.i32.toMlir(b.ctx), b.loc()));
}

/// `static_cast<uint32_t>(v >> 32)` of a 64-bit value (high word).
pub fn hi32(b: *B, v: V) V {
    return lo32(b, v.shrLogical(32));
}

fn i16Of(b: *B, v: V) V {
    return if (v.dtype() == .i16) v else b.emit(arith.trunci(b.ctx, v.inner, DType.i16.toMlir(b.ctx), b.loc()));
}

fn cst32(b: *B, v: anytype) V {
    return b.cst(.i32, @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(@as(i64, v))))))));
}

fn cst64(b: *B, v: u64) V {
    return b.cst(.i64, @as(i64, @bitCast(v)));
}

/// A C++ `uint32_t` constant as an `i32` (bit pattern kept; `Value` literals above
/// `0x7FFFFFFF` would otherwise widen the other operand to `i64`).
pub fn constU32(b: *B, v: u32) V {
    return b.cst(.i32, @as(i32, @bitCast(v)));
}

/// A C++ `uint64_t` constant as an `i64`.
pub fn constU64(b: *B, v: u64) V {
    return cst64(b, v);
}

// ==================== unsigned integer arithmetic (C++ uint32_t / uint64_t) ====================

fn binop(b: *B, comptime f: anytype, x: V, y: anytype) V {
    const l, const r = b.coerce(x, y);
    return b.emit(f(b.ctx, l.inner, r.inner, b.loc()));
}

pub fn divU(b: *B, x: V, y: anytype) V {
    return binop(b, arith.divui, x, y);
}
pub fn remU(b: *B, x: V, y: anytype) V {
    return binop(b, arith.remui, x, y);
}
pub fn minU(b: *B, x: V, y: anytype) V {
    return binop(b, arith.minui, x, y);
}
pub fn maxU(b: *B, x: V, y: anytype) V {
    return binop(b, arith.maxui, x, y);
}
pub fn ceilDivU(b: *B, x: V, y: anytype) V {
    return binop(b, arith.ceildivui, x, y);
}
fn cmpU(b: *B, pred: arith.CmpIPredicate, x: V, y: anytype) V {
    const l, const r = b.coerce(x, y);
    return b.emit(arith.cmpi(b.ctx, pred, l.inner, r.inner, b.loc()));
}
pub fn ltU(b: *B, x: V, y: anytype) V {
    return cmpU(b, .ult, x, y);
}
pub fn leU(b: *B, x: V, y: anytype) V {
    return cmpU(b, .ule, x, y);
}
pub fn gtU(b: *B, x: V, y: anytype) V {
    return cmpU(b, .ugt, x, y);
}
pub fn geU(b: *B, x: V, y: anytype) V {
    return cmpU(b, .uge, x, y);
}

// ==================== special registers, identity (ptx/utils.cuh, cute, cutlass) ====================

/// `ptx::get_sm_idx`: `%smid` (plain `asm`).
pub fn getSmIdx(b: *B) V {
    return asmValue(b, "mov.u32 $0, %smid;", &.{}, .i32, "=r", false);
}

/// `ptx::get_lane_idx`: `%laneid` (plain `asm`).
pub fn getLaneIdx(b: *B) V {
    return asmValue(b, "mov.u32 $0, %laneid;", &.{}, .i32, "=r", false);
}

/// `ptx::get_grid_idx`: the 64-bit `%gridid`, unique per launch.
pub fn getGridIdx(b: *B) V {
    return asmValue(b, "mov.u64 $0, %gridid;", &.{}, .i64, "=l", true);
}

/// `cute::block_rank_in_cluster`.
pub fn blockRankInCluster(b: *B) V {
    return asmValue(b, "mov.u32 $0, %cluster_ctarank;\n", &.{}, .i32, "=r", true);
}

/// `clock64()` (`mov.u64 %rd, %clock64;`).
pub fn clock64(b: *B) V {
    return asmValue(b, "mov.u64 $0, %clock64;", &.{}, .i64, "=l", true);
}

/// `cutlass::canonical_warp_idx_sync`: `__shfl_sync(0xffffffff, threadIdx.x / 32, 0)`.
pub fn canonicalWarpIdxSync(b: *B) V {
    return shflSync(b, divU(b, b.threadIdx().x, 32), b.cst(.i32, 0));
}

/// `cute::elect_one_sync`: 1 in exactly one active lane, 0 elsewhere (i1). The C++ asm's
/// `"+r"(pred)` input 0 is the `mov.u32` here.
pub fn electOneSync(b: *B) V {
    const pred = asmValue(b,
        \\{
        \\.reg .b32 %rx;
        \\.reg .pred %px;
        \\     mov.u32 $0, 0;
        \\     elect.sync %rx|%px, $1;
        \\@%px mov.s32 $0, 1;
        \\}
    , &.{cst32(b, 0xFFFFFFFF)}, .i32, "=r,r", true);
    return pred.ne(0);
}

/// `asm volatile("trap;")` (`DG_TRAP_ONLY_DEVICE_ASSERT`, and this port's timeout).
pub fn trap(b: *B) void {
    asmVoid(b, "trap;", &.{}, "~{memory}");
}

// ==================== CTA / cluster / grid synchronization ====================

/// `ptx::sync_aligned(num_threads, barrier_idx)`: `bar.sync id, n`.
pub fn syncAligned(b: *B, num_threads: anytype, barrier_idx: anytype) void {
    asmVoid(b, "bar.sync $0, $1;", &.{ b.liftAs(barrier_idx, .i32), b.liftAs(num_threads, .i32) }, "r,r,~{memory}");
}

/// `ptx::sync_unaligned(num_threads, barrier_idx)`: `barrier.sync id, n` (threads of a warp may
/// reach it at different program points).
pub fn syncUnaligned(b: *B, num_threads: anytype, barrier_idx: anytype) void {
    asmVoid(b, "barrier.sync $0, $1;", &.{ b.liftAs(barrier_idx, .i32), b.liftAs(num_threads, .i32) }, "r,r,~{memory}");
}

/// `__syncwarp()` (`bar.warp.sync -1;`).
pub fn syncWarp(b: *B) void {
    asmVoid(b, "bar.warp.sync -1;", &.{}, "~{memory}");
}

/// `cute::cluster_arrive_relaxed`.
pub fn clusterArriveRelaxed(b: *B) void {
    asmVoid(b, "barrier.cluster.arrive.relaxed.aligned;\n", &.{}, "~{memory}");
}

/// `cute::cluster_arrive`.
pub fn clusterArrive(b: *B) void {
    asmVoid(b, "barrier.cluster.arrive.aligned;\n", &.{}, "~{memory}");
}

/// `cute::cluster_wait`.
pub fn clusterWait(b: *B) void {
    asmVoid(b, "barrier.cluster.wait.aligned;\n", &.{}, "~{memory}");
}

/// `cute::cluster_sync`.
pub fn clusterSync(b: *B) void {
    clusterArrive(b);
    clusterWait(b);
}

/// `comm::cluster_sync_with_relaxed_arrive`.
pub fn clusterSyncWithRelaxedArrive(b: *B) void {
    clusterArriveRelaxed(b);
    clusterWait(b);
}

/// `cudaGridDependencySynchronize()` (`griddepcontrol.wait;`, PDL).
pub fn gridDependencySynchronize(b: *B) void {
    asmVoid(b, "griddepcontrol.wait;", &.{}, "~{memory}");
}

/// `cutlass::arch::warpgroup_reg_alloc<RegCount>`.
pub fn warpgroupRegAlloc(b: *B, reg_count: u32) void {
    asmVoid(b, std.fmt.allocPrint(b.arena.allocator(), "setmaxnreg.inc.sync.aligned.u32 {d};\n", .{reg_count}) catch @panic("OOM"), &.{}, "");
}

/// `cutlass::arch::warpgroup_reg_dealloc<RegCount>`.
pub fn warpgroupRegDealloc(b: *B, reg_count: u32) void {
    asmVoid(b, std.fmt.allocPrint(b.arena.allocator(), "setmaxnreg.dec.sync.aligned.u32 {d};\n", .{reg_count}) catch @panic("OOM"), &.{}, "");
}

// ==================== memory ordering and atomics (ptx/ld_st.cuh) ====================

/// `ptx::fence_acq_rel_cta`.
pub fn fenceAcqRelCta(b: *B) void {
    asmVoid(b, "fence.acq_rel.cta;", &.{}, "~{memory}");
}

/// `ptx::ld_volatile(const uint64_t*)`.
pub fn ldVolatileU64(b: *B, ptr: V) V {
    return asmValue(b, "ld.volatile.global.b64 $0, [$1];", &.{ptr}, .i64, "=l,l", true);
}

/// `ptx::ld_volatile(const uint32_t*)`.
pub fn ldVolatileU32(b: *B, ptr: V) V {
    return asmValue(b, "ld.volatile.global.b32 $0, [$1];", &.{ptr}, .i32, "=r,l", true);
}

/// `ptx::ld_acq(const uint32_t*)`: acquire at GPU scope.
pub fn ldAcq(b: *B, ptr: V) V {
    return asmValue(b, "ld.acquire.gpu.global.b32 $0, [$1];", &.{ptr}, .i32, "=r,l,~{memory}", true);
}

/// `ptx::ld_acq_sys(const uint64_t*)`.
pub fn ldAcqSysU64(b: *B, ptr: V) V {
    return asmValue(b, "ld.acquire.sys.global.b64 $0, [$1];", &.{ptr}, .i64, "=l,l,~{memory}", true);
}

/// `ptx::ld_acq_sys(const int*)` (no clobber in DeepGEMM).
pub fn ldAcqSysS32(b: *B, ptr: V) V {
    return asmValue(b, "ld.acquire.sys.global.s32 $0, [$1];", &.{ptr}, .i32, "=r,l,~{memory}", true);
}

/// `ptx::ld_acq_sys(const uint32_t*)` (no clobber in DeepGEMM).
pub fn ldAcqSysU32(b: *B, ptr: V) V {
    return asmValue(b, "ld.acquire.sys.global.u32 $0, [$1];", &.{ptr}, .i32, "=r,l,~{memory}", true);
}

/// `ptx::ld_acq_gpu(const uint64_t*)`.
pub fn ldAcqGpuU64(b: *B, ptr: V) V {
    return asmValue(b, "ld.acquire.gpu.global.u64 $0, [$1];", &.{ptr}, .i64, "=l,l,~{memory}", true);
}

/// `ptx::atomic_add(const uint32_t*, uint32_t)`: relaxed, returns the old value.
pub fn atomicAddU32(b: *B, ptr: V, value: anytype) V {
    return asmValue(b, "atom.global.add.u32 $0, [$1], $2;", &.{ ptr, b.liftAs(value, .i32) }, .i32, "=r,l,r", true);
}

/// `ptx::atomic_add(const uint64_t*, uint64_t)`.
pub fn atomicAddU64(b: *B, ptr: V, value: V) V {
    return asmValue(b, "atom.global.add.u64 $0, [$1], $2;", &.{ ptr, value }, .i64, "=l,l,l", true);
}

/// `ptx::atomic_add_sys(const uint64_t*, uint64_t)`.
pub fn atomicAddSys(b: *B, ptr: V, value: V) V {
    return asmValue(b, "atom.sys.global.add.u64 $0, [$1], $2;", &.{ ptr, value }, .i64, "=l,l,l", true);
}

/// `ptx::atomic_add_rel(const uint32_t*, uint32_t)`: release at GPU scope.
pub fn atomicAddRel(b: *B, ptr: V, value: anytype) V {
    return asmValue(b, "atom.release.gpu.global.add.u32 $0, [$1], $2;", &.{ ptr, b.liftAs(value, .i32) }, .i32, "=r,l,r,~{memory}", true);
}

/// `ptx::st_rel_sys(const uint64_t*, uint64_t)`.
pub fn stRelSys(b: *B, ptr: V, value: V) void {
    asmVoid(b, "st.release.sys.global.u64 [$0], $1;", &.{ ptr, value }, "l,l,~{memory}");
}

/// `ptx::red_add(const int*, int)`.
pub fn redAddS32(b: *B, ptr: V, value: anytype) void {
    asmVoid(b, "red.gpu.global.add.s32 [$0], $1;", &.{ ptr, b.liftAs(value, .i32) }, "l,r");
}

/// `ptx::red_add(const uint32_t*, uint32_t)`.
pub fn redAddU32(b: *B, ptr: V, value: anytype) void {
    asmVoid(b, "red.gpu.global.add.u32 [$0], $1;", &.{ ptr, b.liftAs(value, .i32) }, "l,r");
}

/// `ptx::red_xor_rel(const uint64_t*, uint64_t)`.
pub fn redXorRel(b: *B, ptr: V, value: V) void {
    asmVoid(b, "red.release.gpu.global.xor.b64 [$0], $1;", &.{ ptr, value }, "l,l,~{memory}");
}

/// `ptx::red_add_rel(const uint32_t*, uint32_t)` (no clobber in DeepGEMM).
pub fn redAddRel(b: *B, ptr: V, value: anytype) void {
    asmVoid(b, "red.release.gpu.global.add.u32 [$0], $1;", &.{ ptr, b.liftAs(value, .i32) }, "l,r,~{memory}");
}

/// `ptx::red_add_rel_sys(const int*, int)` (no clobber in DeepGEMM).
pub fn redAddRelSys(b: *B, ptr: V, value: anytype) void {
    asmVoid(b, "red.release.sys.global.add.s32 [$0], $1;", &.{ ptr, b.liftAs(value, .i32) }, "l,r,~{memory}");
}

/// `atomicAdd_block(uint32_t* smem, v)` on shared memory: `atom.shared.cta.add.u32`, returns
/// the old value.
pub fn atomicAddBlockShared(b: *B, smem: V, value: anytype) V {
    return asmValue(b, "atom.shared.cta.add.u32 $0, [$1], $2;", &.{ smem, b.liftAs(value, .i32) }, .i32, "=r,r,r,~{memory}", true);
}

/// Plain global stores/loads of DeepGEMM's `*ptr = v` / `v = *ptr` (`st.global`/`ld.global`
/// through an `i64` address; `n` 32-bit words, `n` in {1, 2, 4}).
pub fn stGlobal(b: *B, comptime n: usize, ptr: V, words: [n]V) void {
    b.storeWords(n, ptr, .global, words);
}
pub fn ldGlobal(b: *B, comptime n: usize, ptr: V) [n]V {
    return b.loadWords(n, ptr, .global, .{});
}
/// 64-bit plain global store (`st.global.u64`).
pub fn stGlobalU64(b: *B, ptr: V, value: V) void {
    b.emitVoid(llvm.store(b.ctx, value.inner, b.addressPtr(ptr, .global).inner, .{ .alignment = 8 }, b.loc()));
}
/// 64-bit plain global load (`ld.global.u64`).
pub fn ldGlobalU64(b: *B, ptr: V) V {
    return b.emit(llvm.load(b.ctx, b.addressPtr(ptr, .global).inner, DType.i64.toMlir(b.ctx), .{ .alignment = 8 }, b.loc()));
}
/// Generic-address forms of `stGlobal` / `ldGlobal` / `stGlobalU64`: DeepGEMM's accesses
/// through `SymBuffer::map` with `kNumRanks > 1` (the mapped pointer is an integer cast, so
/// nvcc emits generic `st`/`ld`, `ST.E`/`LD.E` in SASS). Use them through `device.Sym`.
pub fn stGeneric(b: *B, comptime n: usize, ptr: V, words: [n]V) void {
    b.storeWords(n, ptr, .generic, words);
}
pub fn ldGeneric(b: *B, comptime n: usize, ptr: V) [n]V {
    return b.loadWords(n, ptr, .generic, .{});
}
pub fn stGenericU64(b: *B, ptr: V, value: V) void {
    b.emitVoid(llvm.store(b.ctx, value.inner, b.addressPtr(ptr, .generic).inner, .{ .alignment = 8 }, b.loc()));
}
/// Byte load (`v = *(const uint8_t*)ptr`), zero-extended to `i32`.
pub fn ldGlobalU8(b: *B, ptr: V) V {
    const byte = b.emit(llvm.load(b.ctx, b.addressPtr(ptr, .global).inner, DType.i8.toMlir(b.ctx), .{ .alignment = 1 }, b.loc()));
    return zext(b, byte, .i32);
}
/// Byte store (`sf_base_ptr[addr] = uint8_t(v)`).
pub fn stGlobalU8(b: *B, ptr: V, value: V) void {
    const byte = if (value.dtype() == .i8) value else b.emit(arith.trunci(b.ctx, value.inner, DType.i8.toMlir(b.ctx), b.loc()));
    b.emitVoid(llvm.store(b.ctx, byte.inner, b.addressPtr(ptr, .global).inner, .{ .alignment = 1 }, b.loc()));
}

// ==================== shared memory (ptx/ld_st.cuh) ====================

/// `ptx::ld_shared(const uint32_t*)`.
pub fn ldSharedU32(b: *B, smem: V) V {
    return asmValue(b, "ld.shared.u32 $0, [$1];", &.{zext64(b, smem)}, .i32, "=r,l", true);
}

/// `ptx::ld_shared(const float2*)`.
pub fn ldSharedF32x2(b: *B, smem: V) [2]V {
    return asmValues(b, 2, "ld.shared.v2.f32 {$0, $1}, [$2];", &.{zext64(b, smem)}, .f32, "=f,=f,l", true);
}

/// `ptx::ld_shared(const float4*)`.
pub fn ldSharedF32x4(b: *B, smem: V) [4]V {
    return asmValues(b, 4, "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", &.{zext64(b, smem)}, .f32, "=f,=f,=f,=f,l", true);
}

/// `ptx::ld_shared(const uint4*)`.
pub fn ldSharedU32x4(b: *B, smem: V) [4]V {
    return asmValues(b, 4, "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];", &.{zext64(b, smem)}, .i32, "=r,=r,=r,=r,l", true);
}

/// `ptx::st_shared(const void*, x, y, z, w)`.
pub fn stSharedU32x4(b: *B, smem: V, w: [4]V) void {
    asmVoid(b, "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", &.{ zext64(b, smem), w[0], w[1], w[2], w[3] }, "l,r,r,r,r");
}

/// `ptx::st_shared(const uint32_t*, uint32_t)`.
pub fn stSharedU32(b: *B, smem: V, value: V) void {
    asmVoid(b, "st.shared.u32 [$0], $1;", &.{ zext64(b, smem), value }, "l,r");
}

/// `ptx::st_shared(const float2*, float2)`.
pub fn stSharedF32x2(b: *B, smem: V, v: [2]V) void {
    asmVoid(b, "st.shared.v2.f32 [$0], {$1, $2};", &.{ zext64(b, smem), v[0], v[1] }, "l,f,f");
}

/// `ptx::st_shared_bulk(smem, num_bytes)`: zero `num_bytes` (a multiple of 8) of shared memory.
pub fn stSharedBulk(b: *B, smem: V, num_bytes: u32) void {
    std.debug.assert(num_bytes % 8 == 0);
    asmVoid(b, "st.bulk.weak.shared::cta [$0], $1, 0;", &.{ zext64(b, smem), b.cst(.i64, num_bytes) }, "l,l,~{memory}");
}

/// `ptx::mapa_shared(ptr, dst_cta_idx)`: this CTA's shared address in cluster CTA `cta`.
pub fn mapaShared(b: *B, smem: V, dst_cta_idx: V) V {
    return asmValue(b, "mapa.shared::cluster.u32 $0, $1, $2;", &.{ smem, dst_cta_idx }, .i32, "=r,r,r", true);
}

/// `ptx::st_async_cluster(dst, src, dst_cta_idx, barrier)` for `words.len / 4` 16-byte
/// chunks: `st.async` into CTA `dst_cta_idx`'s shared memory, completing the bytes on its
/// mbarrier at `barrier` (both local addresses, mapped with `mapa`).
pub fn stAsyncCluster(b: *B, dst: V, words: []const V, dst_cta_idx: V, barrier: V) void {
    std.debug.assert(words.len % 4 == 0);
    const mapped_barrier = mapaShared(b, barrier, dst_cta_idx);
    const mapped_dst = mapaShared(b, dst, dst_cta_idx);
    var i: usize = 0;
    while (i < words.len) : (i += 4) {
        asmVoid(
            b,
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [$0], {$1, $2, $3, $4}, [$5];",
            &.{ mapped_dst.add(@as(i32, @intCast(i * 4))), words[i], words[i + 1], words[i + 2], words[i + 3], mapped_barrier },
            "r,r,r,r,r,r,~{memory}",
        );
    }
}

/// `ptx::SM100_U8x4_STSM_T<T>::copy(src, smem)`: `stmatrix.m16n8.x1.trans.b8`.
pub fn stmatrixU8x4T(b: *B, src: V, smem: V) void {
    asmVoid(b, "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};\n", &.{ zext64(b, smem), src }, "l,r");
}

/// `ptx::SM90_U32x4_STSM_T<T>::copy(s0, s1, s2, s3, smem)`: `stmatrix.x4.m8n8.trans.b16`.
pub fn stmatrixU32x4T(b: *B, src: [4]V, smem: V) void {
    asmVoid(b, "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [$0], {$1, $2, $3, $4};\n", &.{ zext64(b, smem), src[0], src[1], src[2], src[3] }, "l,r,r,r,r");
}

/// The kernel's dynamic shared memory as one compiler-visible allocation of `bytes`
/// (1024-aligned, like DeepGEMM's `extern __shared__ __align__(1024) uint8_t smem_buffer[]`):
/// its `shared::cta` base address (i32). Size the launch with `Builder.kernelSmemSize`.
pub fn sharedStorageBase(b: *B, bytes: u32) V {
    const field = std.fmt.allocPrint(b.arena.allocator(), "storage:{d}:0", .{bytes}) catch @panic("OOM");
    const fields = b.arena.allocator().dupe([]const u8, &.{field}) catch @panic("OOM");
    const storage = b.allocSmemStorage(.i8, 1024, b.layoutType(b.layoutSpec(.{@as(i64, bytes)}, .{1})), 0, fields);
    const base = b.getIterTyped(storage, b.ptrTy(.i8, .smem, 1024) catch @panic("bad smem pointer")).value();
    return b.ptrToInt(base, .i32);
}

// ==================== mbarriers (cutlass/arch/barrier.h, ptx/tma.cuh) ====================

/// `ClusterBarrier::init(arrive_count)`.
pub fn mbarrierInit(b: *B, bar: V, arrive_count: u32) void {
    asmVoid(b, "{\n\tmbarrier.init.shared::cta.b64 [$1], $0; \n}", &.{ cst32(b, arrive_count), bar }, "r,r,~{memory}");
}

/// `cutlass::arch::fence_barrier_init`.
pub fn fenceBarrierInit(b: *B) void {
    asmVoid(b, "{\n\tfence.mbarrier_init.release.cluster; \n}", &.{}, "~{memory}");
}

/// `ClusterBarrier::wait(phase)`: block until phase `phase` (i32 parity) completes, retrying
/// `mbarrier.try_wait` with the 0x989680-tick suspend hint.
pub fn mbarrierWait(b: *B, bar: V, phase: V) void {
    asmVoid(
        b,
        "{\n\t.reg .pred       P1; \n\tLAB_WAIT: \n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, $2; \n\t@P1 bra DONE; \n\tbra     LAB_WAIT; \n\tDONE: \n\t}",
        &.{ bar, phase, cst32(b, 0x989680) },
        "r,r,r,~{memory}",
    );
}

/// `ptx::mbarrier_wait_and_flip_phase(ptr, phase)`: the same wait, then `phase ^ 1`.
pub fn mbarrierWaitAndFlipPhase(b: *B, bar: V, phase: V) V {
    asmVoid(
        b,
        "{\n\t.reg .pred       P1; \n\tLAB_WAIT: \n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, $2; \n\t@P1 bra DONE; \n\tbra     LAB_WAIT; \n\tDONE: \n\t}",
        &.{ bar, phase, cst32(b, 0x989680) },
        "r,r,r,~{memory}",
    );
    return phase.bitXor(1);
}

/// `ClusterBarrier::arrive()` on this CTA's barrier.
pub fn mbarrierArrive(b: *B, bar: V) void {
    asmVoid(b, "{\n\tmbarrier.arrive.shared::cta.b64 _, [$0];\n\t}", &.{bar}, "r,~{memory}");
}

/// `ClusterBarrier::arrive(cta_id, pred = true)`: arrive on the barrier at the same offset of
/// cluster CTA `cta_id`.
pub fn mbarrierArriveCluster(b: *B, bar: V, cta_id: V) void {
    asmVoid(
        b,
        "{\n\t.reg .b32 remAddr32;\n\tmapa.shared::cluster.u32  remAddr32, $0, $1;\n\tmbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t}",
        &.{ bar, cta_id },
        "r,r,~{memory}",
    );
}

/// `ClusterTransactionBarrier::arrive_and_expect_tx(bytes)`.
pub fn mbarrierArriveAndExpectTx(b: *B, bar: V, transaction_bytes: anytype) void {
    asmVoid(b, "{\n\tmbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $0; \n\t}", &.{ b.liftAs(transaction_bytes, .i32), bar }, "r,r,~{memory}");
}

/// `ClusterTransactionBarrier::arrive_and_expect_tx(bytes, cta_id, pred)`: on cluster CTA
/// `cta_id`'s barrier, only when `pred == 1`.
pub fn mbarrierArriveAndExpectTxCluster(b: *B, bar: V, transaction_bytes: anytype, cta_id: V, pred: V) void {
    asmVoid(
        b,
        "{\n\t.reg .pred p;\n\t.reg .b32 remAddr32;\n\tsetp.eq.u32 p, $2, 1;\n\t@p mapa.shared::cluster.u32  remAddr32, $0, $1;\n\t@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], $3;\n\t}",
        &.{ bar, cta_id, pred, b.liftAs(transaction_bytes, .i32) },
        "r,r,r,r,~{memory}",
    );
}

/// `ptx::mbarrier_arrive_and_set_tx(ptr, num_bytes)` (DeepGEMM's own spelling).
pub fn mbarrierArriveAndSetTx(b: *B, bar: V, num_bytes: anytype) void {
    asmVoid(b, "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $0; \n\t", &.{ b.liftAs(num_bytes, .i32), bar }, "r,r,~{memory}");
}

// ==================== TMA and bulk copies ====================

/// `cute::TMA::CacheHintSm100`.
pub const CacheHint = struct {
    pub const evict_normal: u64 = 0x1000000000000000;
    pub const evict_first: u64 = 0x12F0000000000000;
    pub const evict_last: u64 = 0x14F0000000000000;
};

/// `cute::Sm100MmaPeerBitMask`.
pub const sm100_mma_peer_bit_mask: u32 = 0xFEFFFFFF;

/// `cute::prefetch_tma_descriptor(desc)` (generic descriptor address).
pub fn prefetchTmaDescriptor(b: *B, desc: V) void {
    asmVoid(b, "prefetch.tensormap [$0];", &.{desc}, "l,~{memory}");
}

/// `SM100_TMA_2SM_LOAD_2D::copy(desc, mbar, cache_hint, smem, c0, c1)`: issued by both CTAs of
/// the pair; the barrier address is masked so the bytes complete on the leader's barrier.
pub fn tma2SmLoad2d(b: *B, desc: V, mbar: V, cache_hint: u64, smem: V, c0: V, c1: V) void {
    asmVoid(
        b,
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$3, $4}], [$2], $5;",
        &.{ smem, desc, mbar.bitAnd(cst32(b, sm100_mma_peer_bit_mask)), c0, c1, cst64(b, cache_hint) },
        "r,l,r,r,r,l,~{memory}",
    );
}

/// `SM100_TMA_2SM_LOAD_4D::copy(desc, mbar, cache_hint, smem, c0, c1, c2, c3)`.
pub fn tma2SmLoad4d(b: *B, desc: V, mbar: V, cache_hint: u64, smem: V, c: [4]V) void {
    asmVoid(
        b,
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$3, $4, $5, $6}], [$2], $7;",
        &.{ smem, desc, mbar.bitAnd(cst32(b, sm100_mma_peer_bit_mask)), c[0], c[1], c[2], c[3], cst64(b, cache_hint) },
        "r,l,r,r,r,r,r,l,~{memory}",
    );
}

/// `tma::copy_nd<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, kNumDimensions>(desc, bar,
/// smem, num_tma_multicast = 2, inner_idx, outer...)` on SM100: one 2-CTA load per swizzle
/// atom along the inner dimension. `elem_bytes` is `sizeof(dtype_t)` (1 for FP8 and the
/// unpacked FP4 type, 4 for the int32 SF).
pub fn tmaCopy2Sm(b: *B, block_inner: u32, block_outer: u32, swizzle_mode: u32, elem_bytes: u32, desc: V, bar: V, smem: V, inner_idx: V, outer: []const V) void {
    const atom: u32 = if (swizzle_mode == 0) block_inner else swizzle_mode / elem_bytes;
    std.debug.assert(block_inner % atom == 0);
    for (0..block_inner / atom) |i| {
        const dst = smem.add(@as(i32, @intCast(i * block_outer * atom * elem_bytes)));
        const inner = inner_idx.add(@as(i32, @intCast(i * atom)));
        switch (outer.len) {
            1 => tma2SmLoad2d(b, desc, bar, CacheHint.evict_normal, dst, inner, outer[0]),
            3 => tma2SmLoad4d(b, desc, bar, CacheHint.evict_normal, dst, .{ inner, outer[0], outer[1], outer[2] }),
            else => @panic("tmaCopy2Sm: MegaMoE uses 2-D and 4-D maps"),
        }
    }
}

/// `ptx::tma_load_1d(dst, src, mbar, num_bytes, hint = EVICT_FIRST)`: non-tensor bulk copy of
/// `num_bytes` from a global (possibly peer) address into this CTA's shared memory.
pub fn tmaLoad1d(b: *B, dst: V, src: V, mbar: V, num_bytes: anytype, cache_hint: u64) void {
    asmVoid(
        b,
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;\n",
        &.{ dst, src, b.liftAs(num_bytes, .i32), mbar, cst64(b, cache_hint) },
        "r,l,r,r,l,~{memory}",
    );
}

/// `ptx::tma_store_1d(dst, src, num_bytes, hint = EVICT_NORMAL)`.
pub fn tmaStore1d(b: *B, dst: V, src: V, num_bytes: anytype, cache_hint: u64) void {
    asmVoid(
        b,
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], $2, $3;\n",
        &.{ dst, src, b.liftAs(num_bytes, .i32), cst64(b, cache_hint) },
        "l,r,r,l,~{memory}",
    );
}

/// `cute::SM90_TMA_STORE_2D::copy(desc, smem, c0, c1)`.
pub fn tmaStore2d(b: *B, desc: V, smem: V, c0: V, c1: V) void {
    asmVoid(b, "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3}], [$1];", &.{ desc, smem, c0, c1 }, "l,r,r,r,~{memory}");
}

/// `cute::tma_store_fence`.
pub fn tmaStoreFence(b: *B) void {
    asmVoid(b, "fence.proxy.async.shared::cta;", &.{}, "~{memory}");
}

/// `cutlass::arch::fence_view_async_shared`.
pub fn fenceViewAsyncShared(b: *B) void {
    asmVoid(b, "{\n\tfence.proxy.async.shared::cta; \n}", &.{}, "~{memory}");
}

/// `cute::tma_store_arrive`.
pub fn tmaStoreArrive(b: *B) void {
    asmVoid(b, "cp.async.bulk.commit_group;", &.{}, "~{memory}");
}

/// `ptx::tma_store_wait<N>` (no `.read`).
pub fn tmaStoreWait(b: *B, comptime num_remaining_waits: u32) void {
    asmVoid(b, std.fmt.comptimePrint("cp.async.bulk.wait_group {d};", .{num_remaining_waits}), &.{}, "~{memory}");
}

// ==================== tcgen05: TMEM, UMMA (ptx/tcgen05.cuh, cute) ====================

/// `cute::TMEM::Allocator2Sm::allocate(num_columns, dst)`: write the base column to shared `dst`.
pub fn tmemAllocate2Sm(b: *B, num_columns: u32, dst: V) void {
    asmVoid(b, "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [$0], $1;", &.{ dst, cst32(b, num_columns) }, "r,r,~{memory}");
}

/// `cute::TMEM::Allocator2Sm::free(tmem_ptr, num_columns)`.
pub fn tmemFree2Sm(b: *B, tmem_ptr: V, num_columns: u32) void {
    asmVoid(b, "{\n\ttcgen05.dealloc.cta_group::2.sync.aligned.b32  $0, $1; \n\t}", &.{ tmem_ptr, cst32(b, num_columns) }, "r,r,~{memory}");
}

/// `ptx::tcgen05_before_thread_sync`.
pub fn tcgen05BeforeThreadSync(b: *B) void {
    asmVoid(b, "tcgen05.fence::before_thread_sync;", &.{}, "~{memory}");
}

/// `ptx::tcgen05_after_thread_sync`.
pub fn tcgen05AfterThreadSync(b: *B) void {
    asmVoid(b, "tcgen05.fence::after_thread_sync;", &.{}, "~{memory}");
}

/// `cutlass::arch::fence_view_async_tmem_load` (`tcgen05.wait::ld`).
pub fn fenceViewAsyncTmemLoad(b: *B) void {
    asmVoid(b, "{\n\ttcgen05.wait::ld.sync.aligned; \n}", &.{}, "~{memory}");
}

/// `fence_view_async_tmem_load` with the values of earlier TMEM loads threaded through it, so
/// LLVM cannot move their uses above the wait (the PTX is the same single instruction).
pub fn fenceViewAsyncTmemLoadValues(b: *B, comptime n: usize, values: [n]V) [n]V {
    comptime var constraints: []const u8 = "";
    inline for (0..n) |i| constraints = constraints ++ (if (i == 0) "" else ",") ++ "=r";
    inline for (0..n) |i| constraints = constraints ++ std.fmt.comptimePrint(",{d}", .{i});
    constraints = constraints ++ ",~{memory}";
    return asmValues(b, n, "{\n\ttcgen05.wait::ld.sync.aligned; \n}", &values, .i32, constraints, true);
}

/// `fenceViewAsyncTmemLoadValues` for a run-time number of values (the epilogue's store blocks
/// hold `STORE_BLOCK_M / ATOM_M * 8` words): the same single `tcgen05.wait::ld`, every value
/// tied through it.
pub fn fenceViewAsyncTmemLoadSlice(b: *B, values: []const V) []V {
    const n = values.len;
    const a = b.arena.allocator();
    var w: std.Io.Writer.Allocating = .init(a);
    for (0..n) |i| w.writer.print("{s}=r", .{if (i == 0) "" else ","}) catch @panic("OOM");
    for (0..n) |i| w.writer.print(",{d}", .{i}) catch @panic("OOM");
    w.writer.writeAll(",~{memory}") catch @panic("OOM");
    const inner = a.alloc(*const mlir.Value, n) catch @panic("OOM");
    for (values, inner) |v, *x| x.* = v.inner;
    const elem = DType.i32.toMlir(b.ctx);
    const struct_type = b.parseType(std.fmt.allocPrint(a, "!llvm.struct<({s})>", .{structFields(b, n, elem)}) catch @panic("OOM"));
    const packed_ = b.emit(llvm.inline_asm(b.ctx, "{\n\ttcgen05.wait::ld.sync.aligned; \n}", inner, struct_type, .{ .constraints = w.written(), .has_side_effects = true }, b.loc()));
    const out = a.alloc(V, n) catch @panic("OOM");
    for (out, 0..) |*o, i| {
        o.* = b.emit(mlir.Operation.make(b.ctx, "llvm.extractvalue", .{
            .operands = .{ .flat = &.{packed_.inner} },
            .results = .{ .flat = &.{elem} },
            .attributes = &.{.named(b.ctx, "position", .denseArray(b.ctx, .i64, &[_]i64{@intCast(i)}))},
            .location = b.loc(),
        }));
    }
    return out;
}

/// `cute::SM100_TMEM_LOAD_16dp256b1x::copy(src, d0, d1, d2, d3)`: lane t of the warp gets
/// rows `base + t/4` (d0, d1) and `base + 8 + t/4` (d2, d3), columns `2(t%4)`, `2(t%4)+1`.
pub fn tmemLoad16dp256b1x(b: *B, taddr: V) [4]V {
    return asmValues(b, 4, "tcgen05.ld.sync.aligned.16x256b.x1.b32{$0, $1, $2, $3},[$4];\n", &.{taddr}, .i32, "=r,=r,=r,=r,r", true);
}

/// `cute::SM100_UTCCP_4x32dp128bit_2cta::copy(src_desc, dst_taddr)`.
pub fn utccp4x32dp128bit2cta(b: *B, src_desc: V, dst_taddr: V) void {
    asmVoid(b, "tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $1;", &.{ dst_taddr, src_desc }, "r,l");
}

/// `ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(desc_a, desc_b, tmem_c, scale_c, desc, tmem_sfa,
/// tmem_sfb)`: `desc` is the 64-bit runtime descriptor (`instr << 32`), `scale_c` (i32)
/// nonzero accumulates.
pub fn mmaMxf8f6f42x1SmSs(b: *B, desc_a: V, desc_b: V, tmem_c: V, scale_c: V, desc: V, tmem_sfa: V, tmem_sfb: V) void {
    const idesc = b.emit(arith.trunci(b.ctx, desc.shrLogical(32).inner, DType.i32.toMlir(b.ctx), b.loc()));
    asmVoid(
        b,
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$5], [$6], p; \n\t}\n",
        &.{ tmem_c, desc_a, desc_b, idesc, scale_c, tmem_sfa, tmem_sfb },
        "r,l,l,r,r,r,r",
    );
}

/// `cutlass::arch::umma_arrive_multicast_2x1SM(barrier, cta_mask)`: inside `elect_one_sync`,
/// `tcgen05.commit` arriving on `barrier` in every CTA of `cta_mask`.
pub fn ummaArriveMulticast2x1Sm(b: *B, bar: V, cta_mask: u16) void {
    var elected = b.openIf(electOneSync(b));
    {
        asmVoid(
            b,
            "{\n\ttcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1; \n\t}",
            &.{ bar, b.cst(.i16, @as(i16, @bitCast(cta_mask))) },
            "r,h,~{memory}",
        );
        elected.yieldThen(.{});
    }
}

// ==================== UMMA descriptors (mma/sm100.cuh, cute/arch/mma_sm100_desc.hpp) ====================

/// `mma::sm100::make_smem_desc(layout, smem_ptr, sbo, lbo)` with a run-time start address:
/// `{lo, hi}` words (version 1, base offset 0, LBO mode 0).
pub fn makeSmemDesc(b: *B, layout: SmemLayout, smem: V, stride_byte_offset: u32, leading_byte_offset: u32) V {
    const hi: u64 = (@as(u64, stride_byte_offset >> 4) & 0x3FFF) | (1 << 14) | (@as(u64, @backingInt(layout)) << 29);
    const lo_const: u64 = (@as(u64, leading_byte_offset >> 4) & 0x3FFF) << 16;
    const start = zext(b, smem.shrLogical(4).bitAnd(0x3FFF), .i64);
    return start.bitOr(cst64(b, (hi << 32) | lo_const));
}

/// `cute::UMMA::LayoutType`.
pub const SmemLayout = enum(u32) { swizzle_none = 0, swizzle_128b_base32b = 1, swizzle_128b = 2, swizzle_64b = 4, swizzle_32b = 6 };

/// `mma::sm100::make_sf_desc(smem)`: SWIZZLE_NONE, SBO 8 x 16 B, LBO 0.
pub fn makeSfDesc(b: *B, smem: V) V {
    return makeSmemDesc(b, .swizzle_none, smem, 8 * 16, 0);
}

/// `make_umma_desc<K, BLOCK_MN, 128, 128>(smem, 0, 0)` for 1-byte elements: SW128, SBO
/// 8 x 128 B, LBO 0.
pub fn makeUmmaDescKSw128(b: *B, smem: V) V {
    return makeSmemDesc(b, .swizzle_128b, smem, 8 * 128, 0);
}

/// `mma::sm100::replace_smem_desc_addr(desc, smem)`: a descriptor's start address field.
pub fn replaceSmemDescAddr(b: *B, desc: V, smem: V) V {
    const start = zext(b, smem.shrLogical(4).bitAnd(0x3FFF), .i64);
    return desc.bitAnd(cst64(b, ~@as(u64, 0x3FFF))).bitOr(start);
}

/// The low word of a descriptor (`desc.lo`, i32).
pub fn descLo(b: *B, desc: V) V {
    return b.emit(arith.trunci(b.ctx, desc.inner, DType.i32.toMlir(b.ctx), b.loc()));
}

/// `desc.lo = lo` (i32), keeping the high word.
pub fn withDescLo(b: *B, desc: V, lo: V) V {
    return desc.bitAnd(cst64(b, 0xFFFFFFFF00000000)).bitOr(zext(b, lo, .i64));
}

/// `mma::sm100::advance_umma_desc_lo<K, BLOCK_MN, 128, dtype>(base, offset, k_idx)` for
/// 1-byte, unpacked elements: `base + ((offset + k_idx) >> 4)` with `offset` and `k_idx` in
/// bytes (K-major stride 1).
pub fn advanceUmmaDescLo(b: *B, base: V, offset: anytype, k_idx: anytype) V {
    return base.add(b.liftAs(offset, .i32).add(k_idx).shrLogical(4));
}

/// `make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id)`: the 64-bit runtime descriptor
/// (`desc << 32`) for UMMA_K slice `sf_id` of a 128-wide k-block. `instr` is the 32-bit
/// descriptor (i32) after `update_instr_desc_with_umma_n`.
pub fn makeRuntimeInstrDescWithSfId(b: *B, instr: V, sfa_id: u32, sfb_id: u32) V {
    const cleared = instr.bitAnd(cst32(b, ~@as(u32, (3 << 29) | (3 << 4))));
    const with_ids = cleared.bitOr(cst32(b, (sfa_id << 29) | (sfb_id << 4)));
    return zext(b, with_ids, .i64).shl(32);
}

/// `update_instr_desc_with_umma_n(desc, umma_n)`: `n_dim_ = umma_n >> 3` (i32 descriptor).
pub fn updateInstrDescWithUmmaN(b: *B, instr: V, umma_n: V) V {
    const cleared = instr.bitAnd(cst32(b, ~@as(u32, 0x3F << 17)));
    return cleared.bitOr(umma_n.shrLogical(3).bitAnd(0x3F).shl(17));
}

// ==================== warp collectives (CUDA intrinsics, nvcc's PTX) ====================

fn bits32(v: V) V {
    return if (v.dtype() == .i32) v else v.bitCast(.i32);
}

fn fromBits32(v: V, dt: DType) V {
    return if (dt == .i32) v else v.bitCast(dt);
}

/// `__shfl_sync(0xffffffff, v, src_lane)` / `ptx::exchange` (32-bit).
pub fn shflSync(b: *B, v: V, src_lane: anytype) V {
    const r = asmValue(b, "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", &.{ bits32(v), b.liftAs(src_lane, .i32) }, .i32, "=r,r,r", true);
    return fromBits32(r, v.dtype());
}

/// `ptx::exchange(T, src_lane)` for 64-bit values: two 32-bit shuffles (low word first).
pub fn exchange64(b: *B, v: V, src_lane: anytype) V {
    const lo = b.emit(arith.trunci(b.ctx, v.inner, DType.i32.toMlir(b.ctx), b.loc()));
    const hi = b.emit(arith.trunci(b.ctx, v.shrLogical(32).inner, DType.i32.toMlir(b.ctx), b.loc()));
    const lo2 = shflSync(b, lo, src_lane);
    const hi2 = shflSync(b, hi, src_lane);
    return zext(b, lo2, .i64).bitOr(zext(b, hi2, .i64).shl(32));
}

/// `__shfl_xor_sync(0xffffffff, v, lane_mask)`.
pub fn shflXorSync(b: *B, v: V, lane_mask: anytype) V {
    const r = asmValue(b, "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;", &.{ bits32(v), b.liftAs(lane_mask, .i32) }, .i32, "=r,r,r", true);
    return fromBits32(r, v.dtype());
}

/// `__shfl_up_sync(0xffffffff, v, delta)`.
pub fn shflUpSync(b: *B, v: V, delta: anytype) V {
    const r = asmValue(b, "shfl.sync.up.b32 $0, $1, $2, 0, -1;", &.{ bits32(v), b.liftAs(delta, .i32) }, .i32, "=r,r,r", true);
    return fromBits32(r, v.dtype());
}

/// `__reduce_add_sync(0xffffffff, v)` (u32).
pub fn reduceAddSync(b: *B, v: V) V {
    return asmValue(b, "redux.sync.add.u32 $0, $1, $2;", &.{ v, cst32(b, 0xFFFFFFFF) }, .i32, "=r,r,r", true);
}

/// `__reduce_min_sync(0xffffffff, v)` (u32).
pub fn reduceMinSync(b: *B, v: V) V {
    return asmValue(b, "redux.sync.min.u32 $0, $1, $2;", &.{ v, cst32(b, 0xFFFFFFFF) }, .i32, "=r,r,r", true);
}

/// `__ballot_sync(0xffffffff, pred)` (`pred` i1).
pub fn ballotSync(b: *B, pred: V) V {
    return asmValue(b, "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2; }", &.{ pred.to(.i32), cst32(b, 0xFFFFFFFF) }, .i32, "=r,r,r", true);
}

/// `__all_sync(0xffffffff, pred)` (`pred` i1; returns i1).
pub fn allSync(b: *B, pred: V) V {
    const r = asmValue(b, "{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.all.pred q, p, $2; selp.u32 $0, 1, 0, q; }", &.{ pred.to(.i32), cst32(b, 0xFFFFFFFF) }, .i32, "=r,r,r", true);
    return r.ne(0);
}

/// `__popc(v)`.
pub fn popc(b: *B, v: V) V {
    return asmValue(b, "popc.b32 $0, $1;", &.{v}, .i32, "=r,r", false);
}

/// `__ffs(v)`: 1-based index of the lowest set bit, 0 for 0 (nvcc: `brev` + `bfind.shiftamt`).
pub fn ffs(b: *B, v: V) V {
    return asmValue(b, "{ .reg .b32 r; brev.b32 r, $1; bfind.shiftamt.u32 $0, r; }", &.{v}, .i32, "=r,r", false).add(1);
}

/// `__fns(mask, base, offset)`: position of the `offset`-th set bit of `mask` from `base`.
pub fn fns(b: *B, mask: V, base: anytype, offset: V) V {
    return asmValue(b, "fns.b32 $0, $1, $2, $3;", &.{ mask, b.liftAs(base, .i32), offset }, .i32, "=r,r,r,r", false);
}

/// `math::warp_reduce<kNumLanesPerGroup, kIntergroupReduce>(v, op)` for `op(a, b)`.
pub fn warpReduce(b: *B, comptime num_lanes_per_group: u32, comptime intergroup: bool, v: V, comptime op: fn (*B, V, V) V) V {
    var value = v;
    if (intergroup) {
        inline for (.{ 1, 2, 4, 8, 16 }) |m| {
            if (num_lanes_per_group <= m) value = op(b, value, shflXorSync(b, value, m));
        }
    } else {
        inline for (.{ 16, 8, 4, 2, 1 }) |m| {
            if (num_lanes_per_group >= 2 * m) value = op(b, value, shflXorSync(b, value, m));
        }
    }
    return value;
}

/// `math::warp_inclusive_sum(value, lane_idx)` (u32).
pub fn warpInclusiveSum(b: *B, v: V, lane_idx: V) V {
    var value = v;
    inline for (.{ 1, 2, 4, 8, 16 }) |offset| {
        const synced = shflUpSync(b, value, offset);
        value = b.select(geU(b, lane_idx, offset), value.add(synced), value);
    }
    return value;
}

// ==================== numerics (common/math.cuh, CUDA intrinsics) ====================

/// `fmaxf(a, b)` (`max.f32`, NaN-ignoring; also `math::ReduceMax<float>`).
pub fn fmaxf(b: *B, x: V, y: V) V {
    return asmValue(b, "max.f32 $0, $1, $2;", &.{ x, y }, .f32, "=f,f,f", false);
}

/// `fminf(a, b)` (`min.f32`, NaN-ignoring; `kClampAfterSilu`).
pub fn fminf(b: *B, x: V, y: V) V {
    return asmValue(b, "min.f32 $0, $1, $2;", &.{ x, y }, .f32, "=f,f,f", false);
}

/// `fabsf(x)` (`abs.f32`).
pub fn fabsf(b: *B, x: V) V {
    return asmValue(b, "abs.f32 $0, $1;", &.{x}, .f32, "=f,f", false);
}

/// `math::fast_rcp(x)`.
pub fn fastRcp(b: *B, x: V) V {
    return asmValue(b, "rcp.approx.ftz.f32 $0, $1;", &.{x}, .f32, "=f,f", true);
}

/// `__expf(x)` without `--use_fast_math`: `ex2.approx.f32(x * log2(e))` (non-FTZ).
pub fn expfFast(b: *B, x: V) V {
    return asmValue(b, "{ .reg .f32 t; mul.f32 t, $1, 0f3FB8AA3B; ex2.approx.f32 $0, t; }", &.{x}, .f32, "=f,f", false);
}

/// `__expf(-x)`: nvcc folds the negation into the constant (`mul.f32 t, x, -log2(e)`).
pub fn expfFastNeg(b: *B, x: V) V {
    return asmValue(b, "{ .reg .f32 t; mul.f32 t, $1, 0fBFB8AA3B; ex2.approx.f32 $0, t; }", &.{x}, .f32, "=f,f", false);
}

/// `__float22bfloat162_rn({lo, hi})`: bf16x2 word, `lo` in the low half.
pub fn float22bfloat162Rn(b: *B, lo: V, hi: V) V {
    return asmValue(b, "{ cvt.rn.bf16x2.f32 $0, $2, $1;}\n", &.{ lo, hi }, .i32, "=r,f,f", false);
}

/// `math::cast_into_bf16_and_pack(x, y)` (f32 inputs).
pub fn castIntoBf16AndPack(b: *B, x: V, y: V) V {
    return float22bfloat162Rn(b, fromBits32(x, .f32), fromBits32(y, .f32));
}

/// `__hmin2(a, b)` on bf16x2 words.
pub fn hmin2Bf16(b: *B, x: V, y: V) V {
    return asmValue(b, "{ min.bf16x2 $0,$1,$2;\n}", &.{ x, y }, .i32, "=r,r,r", false);
}

/// `__hmax2(a, b)` on bf16x2 words.
pub fn hmax2Bf16(b: *B, x: V, y: V) V {
    return asmValue(b, "{ max.bf16x2 $0,$1,$2;\n}", &.{ x, y }, .i32, "=r,r,r", false);
}

/// A bf16x2 word with both halves `cvt.rn.bf16.f32(x)` (nvcc's `{x, x}` constant).
pub fn bf16x2Splat(b: *B, comptime x: f32) V {
    const h: u32 = comptime roundBf16(x);
    return cst32(b, (h << 16) | h);
}

/// `bf16x2Splat` for a constant known when the kernel is emitted (a `Config` value).
pub fn bf16x2Const(b: *B, x: f32) V {
    const h: u32 = roundBf16(x);
    return cst32(b, (h << 16) | h);
}

/// f32 -> bf16 bits, round to nearest even (host/comptime).
pub fn roundBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    std.debug.assert(!std.math.isNan(x));
    const rounding = 0x7FFF + ((u >> 16) & 1);
    return @truncate((u + rounding) >> 16);
}

/// `__bfloat1622float2(word)`: `{low, high}` f32 (exact).
pub fn bfloat1622float2(b: *B, word: V) [2]V {
    return asmValues(b, 2, "{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.bf16 $0, l; cvt.f32.bf16 $1, h; }", &.{word}, .f32, "=f,=f,r", false);
}

fn packF32x2Asm(comptime op: []const u8) []const u8 {
    return "{ .reg .b64 a, b, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; " ++ op ++ " d, a, b; mov.b64 {$0, $1}, d; }";
}

/// `__fadd2_rn(a, b)` (`add.rn.f32x2`, the FADD2 nvcc emits).
pub fn fadd2Rn(b: *B, x: [2]V, y: [2]V) [2]V {
    return asmValues(b, 2, packF32x2Asm("add.rn.f32x2"), &.{ x[0], x[1], y[0], y[1] }, .f32, "=f,=f,f,f,f,f", false);
}

/// `__fmul2_rn(a, b)` (`mul.rn.f32x2`).
pub fn fmul2Rn(b: *B, x: [2]V, y: [2]V) [2]V {
    return asmValues(b, 2, packF32x2Asm("mul.rn.f32x2"), &.{ x[0], x[1], y[0], y[1] }, .f32, "=f,=f,f,f,f,f", false);
}

/// `math::get_ue8m0_sf_exp<e4m3>(amax)` (f32 amax -> u32 biased exponent):
/// `max((bits + 0x7FFFFF - 0x600000) >> 23, 105 + 8) - 8`.
pub fn getUe8m0SfExp(b: *B, amax: V) V {
    const bits = amax.bitCast(.i32);
    const rounded_exp = bits.add(@as(i32, 0x7FFFFF - 0x600000)).shrLogical(23);
    return maxU(b, rounded_exp, @as(i32, 105 + 8)).sub(8);
}

/// `math::get_ue8m0_sf_inv<float>(sf_exp)`: `as_float((254 - sf_exp) << 23)`.
pub fn getUe8m0SfInv(b: *B, sf_exp: V) V {
    return b.cst(.i32, 254).sub(sf_exp).shl(23).bitCast(.f32);
}

/// `__nv_fp8x4_e4m3(make_float4(x, y, z, w)).__x`: two `cvt.rn.satfinite.e4m3x2.f32`, `x` in
/// the lowest byte.
pub fn fp8x4E4m3(b: *B, x: V, y: V, z: V, w: V) V {
    return asmValue(
        b,
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
        &.{ x, y, z, w },
        .i32,
        "=r,f,f,f,f",
        false,
    );
}

/// `ptx::accumulate(float2& a, nv_bfloat162 b)`: `a.x += bf16(b.x)`, `a.y += bf16(b.y)` with
/// `add.rn.f32.bf16` (one rounding, the bf16 widened exactly).
pub fn accumulate(b: *B, acc: [2]V, bf16x2: V) [2]V {
    const lo = i16Of(b, bf16x2);
    const hi = i16Of(b, bf16x2.shrLogical(16));
    return .{
        asmValue(b, "add.rn.f32.bf16 $0, $1, $0;\n", &.{ lo, acc[0] }, .f32, "=f,h,0", false),
        asmValue(b, "add.rn.f32.bf16 $0, $1, $0;\n", &.{ hi, acc[1] }, .f32, "=f,h,0", false),
    };
}

// ==================== comm/barrier.cuh ====================

/// `comm::grid_sync<kNumSMs, kGridSyncIndex>`: `sync_scope` (a named barrier of the calling
/// role), then thread 0 adds to the counter (`0x80000000 - (kNumSMs - 1)` on SM 0, 1 elsewhere)
/// and spins until bit 31 flips; then `sync_scope` again. `timeout_cycles` (0: none) replaces
/// DeepGEMM's printf + assert with `trap` (debug-only deviation).
pub fn gridSync(b: *B, num_sms: u32, count_ptr: V, sm_idx: V, thread_idx: V, sync_scope: anytype, timeout_cycles: i64) void {
    sync_scope.run(b);
    var leader = b.openIf(thread_idx.eq(0));
    {
        const add = b.select(sm_idx.eq(0), cst32(b, 0x80000000 - (num_sms - 1)), b.cst(.i32, 1));
        const old_value = atomicAddRel(b, count_ptr, add);
        waitUntil(b, timeout_cycles, struct {
            ptr: V,
            old: V,
            pub fn pending(self: @This(), bb: *B) V {
                return ldAcq(bb, self.ptr).bitXor(self.old).bitAnd(cst32(bb, 0x80000000)).eq(0);
            }
        }{ .ptr = count_ptr, .old = old_value });
        leader.yieldThen(.{});
    }
    sync_scope.run(b);
}

/// `comm::wait_until(pred, ...)`: spin while `cond.pending(b)` (i1) holds; with
/// `timeout_cycles > 0`, `trap` after that many `clock64` cycles.
pub fn waitUntil(b: *B, timeout_cycles: i64, cond: anytype) void {
    const start = if (timeout_cycles > 0) clock64(b) else undefined;
    var loop = b.openWhile(.{}, .{});
    {
        const pending = cond.pending(b);
        loop.yieldBefore(pending, .{});
    }
    {
        if (timeout_cycles > 0) {
            var late = b.openIf(clock64(b).sub(start).ge(timeout_cycles));
            trap(b);
            late.yieldThen(.{});
        }
        loop.yieldAfter(.{});
    }
}

/// A `sync_scope` callback of `gridSync`: `bar.sync id, n`.
pub const AlignedScope = struct {
    num_threads: u32,
    barrier_idx: u32,
    pub fn run(self: AlignedScope, b: *B) void {
        syncAligned(b, @as(i32, @intCast(self.num_threads)), @as(i32, @intCast(self.barrier_idx)));
    }
};

test {
    std.testing.refAllDecls(@This());
}
