//! GPU micro-tests of `ptx.zig`, `tma.zig` and the builder's raw TMA maps against reference
//! micro-kernels compiled from DeepGEMM's own device functions (`gt/numerics_ref.cu`,
//! `gt/tma_ref.cu`; their `.py` drivers write the inputs and DeepGEMM's outputs to
//! `$MEGAMOE_REF_DIR`). Every comparison is bit for bit.
//!
//!   bazel test --config=remote --config=release --@zml//platforms:cuda=true \
//!     --test_strategy=standalone --test_env=MEGAMOE_REF_DIR=<dir> --test_env=CUDA_VISIBLE_DEVICES=0 \
//!     //zml/kernels/moe/megamoe:gpu_test
//!
//! Also: MEGAMOE_RUN=1 (one-rank protocol runs on random inputs, %smid probe),
//! MEGAMOE_DUMPS=<dump dir>[,...] (the whole kernel on single-rank DeepGEMM dumps: `y` bit for
//! bit), MEGAMOE_DUMP_LAUNCHES=<n> (checked back-to-back launches per dump on one workspace
//! that is never re-zeroed, default 2: `y`, dispatch outputs, cleanup and barrier counters after
//! each), MEGAMOE_DUMP_ITERS=<n> (extra unchecked launches per dump, for profiler timings),
//! MEGAMOE_API_DUMPS=<dump dir>[,...] (`zml.moe.megamoe` with `Workspace.symmetric` on one rank,
//! staged and bf16 inputs: `y` bit for bit; needs the symmetric-workspace XLA plugin).
const std = @import("std");

const zml = @import("zml");
const lib = @import("zml/kernels/moe/megamoe/lib");

const ptx = lib.ptx;
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

const n_threads = 128 * 64;
const block = 128;

// ==================== L1 epilogue numerics ====================

const L1Cfg = struct { clamp: bool };

const L1Program = cute.Program(L1Cfg, .{
    .name = "megamoe_l1_numerics",
    .inputs = &.{ "raw", "w" },
    .outputs = &.{ "out", "act" },
    .run = l1Program,
});

fn f32Of(v: V) V {
    return v.bitCast(.f32);
}

fn maxOp(b: *B, x: V, y: V) V {
    return ptx.fmaxf(b, x, y);
}

/// One thread of DeepGEMM's L1 epilogue for one atom: SwiGLU of two (gate, up) pairs of two
/// tokens, routing weight, amax over the 8 lanes sharing `lane % 4`, UE8M0 scale, FP8 cast.
fn l1Kernel(b: *B, cfg: L1Cfg) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .raw = .{ .ptr = cute.DType.i32 },
        .w = .{ .ptr = cute.DType.f32 },
        .out = .{ .ptr = cute.DType.i32 },
        .act = .{ .ptr = cute.DType.i32 },
    });
    const t = b.blockIdx().x.mul(block).add(b.threadIdx().x).to(.i64);
    const raw_addr = b.ptrToInt(a.raw, .i64).add(t.mul(32));
    const lo4 = ptx.ldGlobal(b, 4, raw_addr);
    const hi4 = ptx.ldGlobal(b, 4, raw_addr.add(16));
    const raw = [8]V{ lo4[0], lo4[1], lo4[2], lo4[3], hi4[0], hi4[1], hi4[2], hi4[3] };
    const w2 = ptx.ldGlobal(b, 2, b.ptrToInt(a.w, .i64).add(t.mul(8)));
    const weights = [2]V{ f32Of(w2[0]), f32Of(w2[1]) };

    var act: [2][2]V = undefined;
    for (0..2) |k| {
        var bf16_gate = ptx.float22bfloat162Rn(b, f32Of(raw[4 * k]), f32Of(raw[4 * k + 1]));
        var bf16_up = ptx.float22bfloat162Rn(b, f32Of(raw[4 * k + 2]), f32Of(raw[4 * k + 3]));
        if (cfg.clamp) {
            bf16_gate = ptx.hmin2Bf16(b, bf16_gate, ptx.bf16x2Splat(b, 10.0));
            bf16_up = ptx.hmax2Bf16(b, bf16_up, ptx.bf16x2Splat(b, -10.0));
            bf16_up = ptx.hmin2Bf16(b, bf16_up, ptx.bf16x2Splat(b, 10.0));
        }
        const gate = ptx.bfloat1622float2(b, bf16_gate);
        const neg_gate_exp = [2]V{ ptx.expfFastNeg(b, gate[0]), ptx.expfFastNeg(b, gate[1]) };
        const denom = ptx.fadd2Rn(b, .{ b.cst(.f32, 1.0), b.cst(.f32, 1.0) }, neg_gate_exp);
        const gated = ptx.fmul2Rn(b, gate, .{ ptx.fastRcp(b, denom[0]), ptx.fastRcp(b, denom[1]) });
        const up = ptx.bfloat1622float2(b, bf16_up);
        act[k] = ptx.fmul2Rn(b, ptx.fmul2Rn(b, gated, up), weights);
    }

    var amax: [2]V = undefined;
    for (0..2) |c| {
        var thread_local = b.cst(.f32, 0.0);
        for (0..2) |k| thread_local = ptx.fmaxf(b, thread_local, ptx.fabsf(b, act[k][c]));
        amax[c] = ptx.warpReduce(b, 4, true, thread_local, maxOp);
    }
    const sf_exp = [2]V{ ptx.getUe8m0SfExp(b, amax[0]), ptx.getUe8m0SfExp(b, amax[1]) };
    const sf_inv = [2]V{ ptx.getUe8m0SfInv(b, sf_exp[0]), ptx.getUe8m0SfInv(b, sf_exp[1]) };
    const upper = ptx.fmul2Rn(b, act[0], sf_inv);
    const lower = ptx.fmul2Rn(b, act[1], sf_inv);
    const fp8x4 = ptx.fp8x4E4m3(b, upper[0], upper[1], lower[0], lower[1]);

    ptx.stGlobal(b, 4, b.ptrToInt(a.out, .i64).add(t.mul(16)), .{ fp8x4, sf_exp[0], sf_exp[1], amax[0].bitCast(.i32) });
    ptx.stGlobal(b, 4, b.ptrToInt(a.act, .i64).add(t.mul(16)), .{ act[0][0].bitCast(.i32), act[0][1].bitCast(.i32), act[1][0].bitCast(.i32), act[1][1].bitCast(.i32) });
}

fn launch1d(b: *B, comptime kernel_name: []const u8, args: anytype) void {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, n_threads / block), one, one },
        .block = .{ b.cst(.i32, block), one, one },
        .dynamic_smem = b.cst(.i64, 0),
        .stream = b.cudaStream(),
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx(kernel_name, config, args)));
}

fn l1Program(b: *B, cfg: L1Cfg) cute.FinishError!void {
    b.beginFunction("megamoe_l1_numerics_kernel", .cuda_kernel);
    try l1Kernel(b, cfg);
    b.endFunction(.{ block, 1, 1 });
    b.beginFunction(L1Program.name, .host);
    const h = try b.declareArgs(.{
        .raw = .{ .ptr = cute.DType.i32 },
        .w = .{ .ptr = cute.DType.f32 },
        .out = .{ .ptr = cute.DType.i32 },
        .act = .{ .ptr = cute.DType.i32 },
    });
    launch1d(b, "megamoe_l1_numerics_kernel", .{ h.raw, h.w, h.out, h.act });
    b.endFunction(null);
}

// ==================== L2 epilogue packing and combine ====================

const L2Program = cute.Program(void, .{
    .name = "megamoe_l2_numerics",
    .inputs = &.{ "raw", "slots", "num_slots" },
    .outputs = &.{ "packed_bf16", "combined" },
    .run = l2Program,
});

/// `cast_into_bf16_and_pack` of 4 pairs per thread; and the combine's `accumulate` over up to
/// 7 bf16x2 slots in order, then `__float22bfloat162_rn`.
fn l2Kernel(b: *B) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .raw = .{ .ptr = cute.DType.i32 },
        .slots = .{ .ptr = cute.DType.i32 },
        .num_slots = .{ .ptr = cute.DType.i32 },
        .packed_ = .{ .ptr = cute.DType.i32 },
        .combined = .{ .ptr = cute.DType.i32 },
    });
    const t = b.blockIdx().x.mul(block).add(b.threadIdx().x).to(.i64);
    const raw_addr = b.ptrToInt(a.raw, .i64).add(t.mul(32));
    const lo4 = ptx.ldGlobal(b, 4, raw_addr);
    const hi4 = ptx.ldGlobal(b, 4, raw_addr.add(16));
    const raw = [8]V{ lo4[0], lo4[1], lo4[2], lo4[3], hi4[0], hi4[1], hi4[2], hi4[3] };
    var packed_: [4]V = undefined;
    for (0..4) |q| packed_[q] = ptx.castIntoBf16AndPack(b, f32Of(raw[2 * q]), f32Of(raw[2 * q + 1]));
    ptx.stGlobal(b, 4, b.ptrToInt(a.packed_, .i64).add(t.mul(16)), packed_);

    const n = ptx.ldGlobal(b, 1, b.ptrToInt(a.num_slots, .i64).add(t.mul(4)))[0];
    const slots_addr = b.ptrToInt(a.slots, .i64).add(t.mul(28));
    var acc = [2]V{ b.cst(.f32, 0.0), b.cst(.f32, 0.0) };
    for (0..7) |s| {
        const word = ptx.ldGlobal(b, 1, slots_addr.add(@as(i64, @intCast(4 * s))))[0];
        const next = ptx.accumulate(b, acc, word);
        const present = b.cst(.i32, @as(i32, @intCast(s))).lt(n);
        acc = .{ b.select(present, next[0], acc[0]), b.select(present, next[1], acc[1]) };
    }
    ptx.stGlobal(b, 1, b.ptrToInt(a.combined, .i64).add(t.mul(4)), .{ptx.float22bfloat162Rn(b, acc[0], acc[1])});
}

fn l2Program(b: *B, _: void) cute.FinishError!void {
    b.beginFunction("megamoe_l2_numerics_kernel", .cuda_kernel);
    try l2Kernel(b);
    b.endFunction(.{ block, 1, 1 });
    b.beginFunction(L2Program.name, .host);
    const h = try b.declareArgs(.{
        .raw = .{ .ptr = cute.DType.i32 },
        .slots = .{ .ptr = cute.DType.i32 },
        .num_slots = .{ .ptr = cute.DType.i32 },
        .packed_ = .{ .ptr = cute.DType.i32 },
        .combined = .{ .ptr = cute.DType.i32 },
    });
    launch1d(b, "megamoe_l2_numerics_kernel", .{ h.raw, h.slots, h.num_slots, h.packed_, h.combined });
    b.endFunction(null);
}

// ==================== host side ====================

fn refDir() ![]const u8 {
    const dir = std.c.getenv("MEGAMOE_REF_DIR") orelse return error.SkipZigTest;
    return std.mem.span(dir);
}

fn readRef(allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ try refDir(), name });
    defer allocator.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 26));
}

fn expectBits(what: []const u8, expected: []const u8, actual: []const u8) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    const e = std.mem.bytesAsSlice(u32, expected);
    const a = std.mem.bytesAsSlice(u32, actual);
    var mismatches: usize = 0;
    for (e, a, 0..) |x, y, i| {
        if (x != y) {
            if (mismatches < 8) std.debug.print("{s}[{d}]: expected 0x{x:0>8}, got 0x{x:0>8}\n", .{ what, i, x, y });
            mismatches += 1;
        }
    }
    if (mismatches != 0) {
        std.debug.print("{s}: {d} of {d} words differ\n", .{ what, mismatches, e.len });
        return error.TestExpectedEqual;
    }
}

fn l1Call(raw: zml.Tensor, w: zml.Tensor, cfg: L1Cfg) [2]zml.Tensor {
    const r = L1Program.call(.{ .raw = raw, .w = w }, .{
        .out = .init(.{n_threads * 4}, .i32),
        .act = .init(.{n_threads * 4}, .i32),
    }, .{ .cfg = cfg });
    return .{ r.out, r.act };
}

fn l2Call(raw: zml.Tensor, slots: zml.Tensor, num_slots: zml.Tensor) [2]zml.Tensor {
    const r = L2Program.call(.{ .raw = raw, .slots = slots, .num_slots = num_slots }, .{
        .packed_bf16 = .init(.{n_threads * 4}, .i32),
        .combined = .init(.{n_threads}, .i32),
    }, .{ .cfg = {} });
    return .{ r.packed_bf16, r.combined };
}

test "L1 epilogue numerics match DeepGEMM bit for bit" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const raw_bytes = try readRef(allocator, io, "l1_raw.bin");
    defer allocator.free(raw_bytes);
    const w_bytes = try readRef(allocator, io, "l1_w.bin");
    defer allocator.free(w_bytes);
    var raw = try zml.Buffer.fromBytes(io, platform, .init(.{n_threads * 8}, .i32), raw_bytes);
    defer raw.deinit();
    var w = try zml.Buffer.fromBytes(io, platform, .init(.{n_threads * 2}, .f32), w_bytes);
    defer w.deinit();
    for ([_]bool{ true, false }) |clamp| {
        var exe = try platform.compileFn(allocator, io, l1Call, .{ zml.Tensor.init(.{n_threads * 8}, .i32), zml.Tensor.init(.{n_threads * 2}, .f32), L1Cfg{ .clamp = clamp } }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var output: [2]zml.Buffer = undefined;
        runner.run(io, .{ raw, w }, &output, .{ .wait = true });
        defer for (&output) |*o| o.deinit();
        const out = try output[0].toSliceAlloc(allocator, io);
        defer out.free(allocator);
        const act = try output[1].toSliceAlloc(allocator, io);
        defer act.free(allocator);
        const suffix = if (clamp) "cl1" else "cl0";
        const ref_out = try readRef(allocator, io, if (clamp) "l1_out_fm1_cl1.bin" else "l1_out_fm1_cl0.bin");
        defer allocator.free(ref_out);
        const ref_act = try readRef(allocator, io, if (clamp) "l1_act_fm1_cl1.bin" else "l1_act_fm1_cl0.bin");
        defer allocator.free(ref_act);
        std.debug.print("L1 epilogue, clamp {s}\n", .{suffix});
        try expectBits("activation", ref_act, act.constData());
        try expectBits("fp8x4/sf/amax", ref_out, out.constData());
    }
}

test "L2 bf16 packing and combine accumulation match DeepGEMM bit for bit" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const raw_bytes = try readRef(allocator, io, "l2_raw.bin");
    defer allocator.free(raw_bytes);
    const slots_bytes = try readRef(allocator, io, "comb_slots.bin");
    defer allocator.free(slots_bytes);
    const n_bytes = try readRef(allocator, io, "comb_n.bin");
    defer allocator.free(n_bytes);
    var raw = try zml.Buffer.fromBytes(io, platform, .init(.{n_threads * 8}, .i32), raw_bytes);
    defer raw.deinit();
    var slots = try zml.Buffer.fromBytes(io, platform, .init(.{n_threads * 7}, .i32), slots_bytes);
    defer slots.deinit();
    var num_slots = try zml.Buffer.fromBytes(io, platform, .init(.{n_threads}, .i32), n_bytes);
    defer num_slots.deinit();
    var exe = try platform.compileFn(allocator, io, l2Call, .{ zml.Tensor.init(.{n_threads * 8}, .i32), zml.Tensor.init(.{n_threads * 7}, .i32), zml.Tensor.init(.{n_threads}, .i32) }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [2]zml.Buffer = undefined;
    runner.run(io, .{ raw, slots, num_slots }, &output, .{ .wait = true });
    defer for (&output) |*o| o.deinit();
    const packed_ = try output[0].toSliceAlloc(allocator, io);
    defer packed_.free(allocator);
    const combined = try output[1].toSliceAlloc(allocator, io);
    defer combined.free(allocator);
    const ref_packed = try readRef(allocator, io, "l2_out.bin");
    defer allocator.free(ref_packed);
    const ref_combined = try readRef(allocator, io, "comb_out.bin");
    defer allocator.free(ref_combined);
    try expectBits("l2 bf16x2", ref_packed, packed_.constData());
    try expectBits("combine", ref_combined, combined.constData());
}

// ==================== TMA descriptors ====================

const L = lib.layout;

/// The 18 tensor maps of vLLM's DSv4.1 instantiation (EP2, T 9600, BLOCK_M 192, FP4 routed and
/// FP8 shared weights in DeepGEMM's default layout).
fn dsv41Maps() L.TensorMaps {
    const shape: L.Shape = .{ .num_ranks = 2, .num_experts = 384, .num_topk = 6, .hidden = 5120, .intermediate_hidden = 2304, .num_shared_experts = 1, .num_max_tokens_per_rank = L.alignTokens(8192) };
    const symm: L.SymmBufferLayout = .init(shape, 152);
    const kernel: L.Kernel = .init(shape, L.config(shape, 4096, symm), 152, .fp4_unpacked);
    const weights: L.DefaultWeights = .init(shape, .fp4_unpacked);
    return .init(kernel, symm.buffer, weights.tensors(true));
}

fn mapSpec(which: usize) L.TensorMapSpec {
    const maps = dsv41Maps();
    inline for (L.TensorMaps.names, 0..) |name, i| {
        if (i == which) return @field(maps, name);
    }
    unreachable;
}

const DescCfg = struct { which: u8 };

const DescProgram = cute.Program(DescCfg, .{
    .name = "megamoe_tma_desc",
    .inputs = &.{"buf"},
    .outputs = &.{"out"},
    .run = descProgram,
});

fn descProgram(b: *B, cfg: DescCfg) cute.FinishError!void {
    const map: lib.tma.Map = .init(mapSpec(cfg.which));
    b.beginFunction("megamoe_tma_desc_kernel", .cuda_kernel);
    {
        const a = try b.declareArgs(.{
            .desc = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(map.raw()), .grid_constant = true } },
            .out = .{ .ptr = cute.DType.i32 },
            .buf = .{ .ptr = cute.DType.i8 },
        });
        var first = b.openIf(b.threadIdx().x.eq(0));
        {
            const desc = b.rawTmaDescAddr(a.desc, map.raw());
            const out = b.ptrToInt(a.out, .i64);
            for (0..8) |i| {
                const words = b.loadWords(4, desc.add(@as(i64, @intCast(16 * i))), .generic, .{});
                ptx.stGlobal(b, 4, out.add(@as(i64, @intCast(16 * i))), words);
            }
            const base = b.ptrToInt(a.buf, .i64);
            ptx.stGlobalU64(b, out.add(128), base);
            first.yieldThen(.{});
        }
    }
    b.endFunction(.{ 32, 1, 1 });

    b.beginFunction(DescProgram.name, .host);
    const h = try b.declareArgs(.{
        .buf = .{ .ptr = cute.DType.i8 },
        .out = .{ .ptr = cute.DType.i32 },
    });
    const atom = b.makeRawTmaAtom(h.buf, map.raw());
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, 32), one, one },
        .dynamic_smem = b.cst(.i64, 0),
        .stream = b.cudaStream(),
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_tma_desc_kernel", config, .{ atom.atom, h.out, h.buf })));
    b.endFunction(null);
}

fn descCall(buf: zml.Tensor, cfg: DescCfg) zml.Tensor {
    return DescProgram.call(.{ .buf = buf }, .{ .out = .init(.{36}, .i32) }, .{ .cfg = cfg }).out;
}

const CUtensorMap = extern struct { opaque_: [16]u64 align(64) };
const EncodeTiled = *const fn (*CUtensorMap, u32, u32, ?*anyopaque, [*]const u64, [*]const u64, [*]const u32, [*]const u32, u32, u32, u32, u32) callconv(.c) c_int;

/// `cuTensorMapEncodeTiled` with DeepGEMM's parameters (`spec`) at `base + spec.base_offset`.
fn encodeReference(spec: L.TensorMapSpec, base: u64) !CUtensorMap {
    const handle = std.c.dlopen("libcuda.so.1", .{ .LAZY = true }) orelse return error.NoLibcuda;
    const sym = std.c.dlsym(handle, "cuTensorMapEncodeTiled") orelse return error.NoEncode;
    const encode: EncodeTiled = @ptrCast(@alignCast(sym));
    var map: CUtensorMap = undefined;
    @memset(&map.opaque_, 0);
    const status = encode(
        &map,
        @backingInt(spec.data_type),
        spec.rank,
        @ptrFromInt(base + spec.base_offset),
        &spec.global_dims,
        &spec.global_strides,
        &spec.box_dims,
        &spec.element_strides,
        @backingInt(spec.interleave),
        @backingInt(spec.swizzle),
        @backingInt(spec.l2_promotion),
        @backingInt(spec.oob_fill),
    );
    if (status != 0) {
        std.debug.print("cuTensorMapEncodeTiled failed: {d}\n", .{status});
        return error.EncodeFailed;
    }
    return map;
}

/// What CuTe's inline descriptor encoder writes differently from DeepGEMM's
/// `cuTensorMapEncodeTiled` call, in the first 64 bytes (the hardware fields):
/// L2 promotion (qword 1 bits 17-18) is 128B instead of DeepGEMM's 256B, and qword 1 bit 1 is
/// set (no `cuTensorMapEncodeTiled` parameter sets it; CuTe-built descriptors run correctly in
/// zml's CuTe GEMMs). The driver also fills bytes 64-79 (box bytes and smem alignment) that CuTe
/// leaves uninitialized and the hardware does not read.
fn cuteQword1(deepgemm: u64) u64 {
    return (deepgemm & ~(@as(u64, 3) << 17)) | (@as(u64, @backingInt(L.TensorMapL2Promotion.@"128b")) << 17) | 2;
}

test "TMA descriptors built by the CuTe host match DeepGEMM's cuTensorMapEncodeTiled" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const buf_shape: zml.Shape = .init(.{1 << 16}, .i8);
    const zeros = try allocator.alloc(u8, 1 << 16);
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var buf = try zml.Buffer.fromBytes(io, platform, buf_shape, zeros);
    defer buf.deinit();
    var failures: usize = 0;
    inline for (L.TensorMaps.names, 0..) |name, which| {
        var exe = try platform.compileFn(allocator, io, descCall, .{ zml.Tensor.fromShape(buf_shape), DescCfg{ .which = which } }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var output: [1]zml.Buffer = undefined;
        runner.run(io, .{buf}, &output, .{ .wait = true });
        defer output[0].deinit();
        const out = try output[0].toSliceAlloc(allocator, io);
        defer out.free(allocator);
        const bytes = out.constData();
        const base = std.mem.readInt(u64, bytes[128..136], .little);
        // Every map is built over `buf` here; the reference uses the same address.
        var spec = mapSpec(which);
        spec.base_offset = 0;
        const ref = try encodeReference(spec, base);
        var mismatch = false;
        for (0..8) |q| {
            const r = ref.opaque_[q];
            const c = std.mem.readInt(u64, bytes[q * 8 ..][0..8], .little);
            const expected = if (q == 1) cuteQword1(r) else r;
            if (c != expected) {
                mismatch = true;
                std.debug.print("  {s} qword {d}: deepgemm 0x{x:0>16}  cute 0x{x:0>16}\n", .{ name, q, r, c });
            }
        }
        std.debug.print("{s:<22} rank {d} {s}\n", .{ name, spec.rank, if (mismatch) "DIFFERENT" else "same fields (L2 128B vs 256B, bit 1)" });
        if (mismatch) failures += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

// ==================== register <-> shared / tensor memory maps ====================

const MapsProgram = cute.Program(void, .{
    .name = "megamoe_maps",
    .inputs = &.{ "l1_vals", "l2_vals" },
    .outputs = &.{ "l1_out", "l2_out", "tmem_out" },
    .run = mapsProgram,
});

const maps_smem_bytes = 2048 + 8192 + 1024;

/// One epilogue warpgroup: the L1 `stmatrix.m16n8.x1.trans.b8` and L2 `stmatrix.x4.m8n8.trans.b16`
/// stores with DeepGEMM's swizzled addresses (4 atoms), and `tcgen05.ld.16x256b.x1` of a
/// TMEM pattern (lane L, column C = L * 65536 + C) at columns 0 and 8, each twice (+16 lanes).
fn mapsKernel(b: *B) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .l1_vals = .{ .ptr = cute.DType.i32 },
        .l2_vals = .{ .ptr = cute.DType.i32 },
        .l1_out = .{ .ptr = cute.DType.i32 },
        .l2_out = .{ .ptr = cute.DType.i32 },
        .tmem_out = .{ .ptr = cute.DType.i32 },
    });
    const smem = ptx.sharedStorageBase(b, maps_smem_bytes);
    const l1_smem = smem;
    const l2_smem = smem.add(2048);
    const holder = smem.add(2048 + 8192);
    const tid = b.threadIdx().x;
    const lane = ptx.remU(b, tid, 32);
    const warp = ptx.divU(b, tid, 32);

    var fill = b.openFor(0, (2048 + 8192) / 4 / block, 1, .{});
    ptx.stSharedU32(b, smem.add(fill.iv.mul(block).add(tid).mul(4)), cst(b, 0xdeadbeef));
    fill.yield(.{});
    ptx.syncAligned(b, block, 0);

    const l1_vals = b.ptrToInt(a.l1_vals, .i64);
    const l2_vals = b.ptrToInt(a.l2_vals, .i64);
    for (0..4) |i| {
        // L1: row = lane, col = warp_idx_in_wg, 64B swizzle.
        const row = lane;
        const col = warp;
        const addr = l1_smem.add(@as(i32, @intCast(i * 8 * 64))).add(row.mul(64)).add(col.bitXor(ptx.divU(b, row, 2)).mul(16));
        const v = ptx.ldGlobal(b, 1, l1_vals.add(tid.to(.i64).mul(16)).add(@as(i64, @intCast(4 * i))))[0];
        ptx.stmatrixU8x4T(b, v, addr);
    }
    for (0..4) |i| {
        // L2: row = lane % 8, col = (warp % 2) * 4 + lane / 8, 128B swizzle.
        const row = ptx.remU(b, lane, 8);
        const col = ptx.remU(b, warp, 2).mul(4).add(ptx.divU(b, lane, 8));
        const addr = l2_smem.add(ptx.divU(b, warp, 2).mul(32 * 128)).add(@as(i32, @intCast(i * 8 * 128))).add(row.mul(128)).add(col.bitXor(row).mul(16));
        const w = ptx.ldGlobal(b, 4, l2_vals.add(tid.to(.i64).mul(64)).add(@as(i64, @intCast(16 * i))));
        ptx.stmatrixU32x4T(b, w, addr);
    }
    ptx.syncAligned(b, block, 0);
    var copy1 = b.openFor(0, 2048 / 4 / block, 1, .{});
    {
        const j = copy1.iv.mul(block).add(tid);
        ptx.stGlobal(b, 1, b.ptrToInt(a.l1_out, .i64).add(j.to(.i64).mul(4)), .{ptx.ldSharedU32(b, l1_smem.add(j.mul(4)))});
        copy1.yield(.{});
    }
    var copy2 = b.openFor(0, 8192 / 4 / block, 1, .{});
    {
        const j = copy2.iv.mul(block).add(tid);
        ptx.stGlobal(b, 1, b.ptrToInt(a.l2_out, .i64).add(j.to(.i64).mul(4)), .{ptx.ldSharedU32(b, l2_smem.add(j.mul(4)))});
        copy2.yield(.{});
    }

    // TMEM (1-CTA allocation for the test).
    var alloc = b.openIf(warp.eq(0));
    b.tmemAllocAt(holder, 64);
    alloc.yieldThen(.{});
    ptx.syncAligned(b, block, 0);
    ptx.tcgen05AfterThreadSync(b);
    const base = ptx.ldSharedU32(b, holder);
    for (0..8) |c| {
        var words: [8]V = undefined;
        const tmem_lane = warp.mul(32).add(lane);
        for (0..8) |k| words[k] = tmem_lane.mul(65536).add(@as(i32, @intCast(8 * c + k)));
        b.tmemStore32x32b(8, base.add(warp.mul(32).shl(16)).add(@as(i32, @intCast(8 * c))), words);
    }
    b.fenceTmemStore();
    ptx.tcgen05BeforeThreadSync(b);
    ptx.syncAligned(b, block, 0);
    ptx.tcgen05AfterThreadSync(b);
    var values: [16]V = undefined;
    for (0..2) |j| {
        const taddr = base.add(@as(i32, @intCast(8 * j)));
        const lo = ptx.tmemLoad16dp256b1x(b, taddr);
        const hi = ptx.tmemLoad16dp256b1x(b, taddr.bitOr(0x00100000));
        for (0..4) |k| {
            values[8 * j + k] = lo[k];
            values[8 * j + 4 + k] = hi[k];
        }
    }
    values = ptx.fenceViewAsyncTmemLoadValues(b, 16, values);
    const tmem_out = b.ptrToInt(a.tmem_out, .i64).add(tid.to(.i64).mul(64));
    for (0..4) |q| ptx.stGlobal(b, 4, tmem_out.add(@as(i64, @intCast(16 * q))), values[4 * q ..][0..4].*);
    ptx.tcgen05BeforeThreadSync(b);
    ptx.syncAligned(b, block, 0);
    var free = b.openIf(warp.eq(0));
    b.tmemDeallocAt(base, 64);
    free.yieldThen(.{});
}

fn cst(b: *B, v: u32) V {
    return b.cst(.i32, @as(i32, @bitCast(v)));
}

fn mapsProgram(b: *B, _: void) cute.FinishError!void {
    b.beginFunction("megamoe_maps_kernel", .cuda_kernel);
    try mapsKernel(b);
    b.endFunction(.{ block, 1, 1 });
    b.beginFunction(MapsProgram.name, .host);
    const h = try b.declareArgs(.{
        .l1_vals = .{ .ptr = cute.DType.i32 },
        .l2_vals = .{ .ptr = cute.DType.i32 },
        .l1_out = .{ .ptr = cute.DType.i32 },
        .l2_out = .{ .ptr = cute.DType.i32 },
        .tmem_out = .{ .ptr = cute.DType.i32 },
    });
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, block), one, one },
        .dynamic_smem = b.kernelSmemSize("megamoe_maps_kernel"),
        .stream = b.cudaStream(),
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_maps_kernel", config, .{ h.l1_vals, h.l2_vals, h.l1_out, h.l2_out, h.tmem_out })));
    b.endFunction(null);
}

fn mapsCall(l1_vals: zml.Tensor, l2_vals: zml.Tensor) [3]zml.Tensor {
    const r = MapsProgram.call(.{ .l1_vals = l1_vals, .l2_vals = l2_vals }, .{
        .l1_out = .init(.{2048 / 4}, .i32),
        .l2_out = .init(.{8192 / 4}, .i32),
        .tmem_out = .init(.{block * 16}, .i32),
    }, .{ .cfg = {} });
    return .{ r.l1_out, r.l2_out, r.tmem_out };
}

test "stmatrix and tcgen05.ld register maps match DeepGEMM's (CUTLASS) copies" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const l1_bytes = try readRef(allocator, io, "stsm_l1_vals.bin");
    defer allocator.free(l1_bytes);
    const l2_bytes = try readRef(allocator, io, "stsm_l2_vals.bin");
    defer allocator.free(l2_bytes);
    var l1 = try zml.Buffer.fromBytes(io, platform, .init(.{block * 4}, .i32), l1_bytes);
    defer l1.deinit();
    var l2 = try zml.Buffer.fromBytes(io, platform, .init(.{block * 16}, .i32), l2_bytes);
    defer l2.deinit();
    var exe = try platform.compileFn(allocator, io, mapsCall, .{ zml.Tensor.init(.{block * 4}, .i32), zml.Tensor.init(.{block * 16}, .i32) }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [3]zml.Buffer = undefined;
    runner.run(io, .{ l1, l2 }, &output, .{ .wait = true });
    defer for (&output) |*o| o.deinit();
    inline for (.{ "stsm_l1_out.bin", "stsm_l2_out.bin", "tmem_ld_out.bin" }, 0..) |name, i| {
        const got = try output[i].toSliceAlloc(allocator, io);
        defer got.free(allocator);
        const ref = try readRef(allocator, io, name);
        defer allocator.free(ref);
        try expectBits(name, ref, got.constData());
    }
}

// ==================== experiment: explicit cuTensorMapEncodeTiled in the host function ====================
// `cuda.tensor_map_encode_tiled` (the CuTe DSL's `create_tensor_map_tiled`) would give DeepGEMM's
// exact descriptor, L2 promotion included. This checks whether XLA's CuTe host program runs it.

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

fn hostArray(b: *B, comptime T: type, values: []const T) V {
    const ctx = b.ctx;
    const elem: *const mlir.Type = if (T == u64) .int(ctx, .i64) else .int(ctx, .i32);
    const ptr_type = dialects.llvm.pointerType(ctx, dialects.llvm.NVPTXAddressSpace.generic);
    const array = b.emit(mlir.Operation.make(ctx, "llvm.alloca", .{
        .operands = .{ .flat = &.{b.cst(.i32, @as(i32, @intCast(@max(values.len, 1)))).inner} },
        .results = .{ .flat = &.{ptr_type} },
        .attributes = &.{ .named(ctx, "elem_type", .typeAttr(elem)), .named(ctx, "alignment", .int(ctx, .i64, 8)) },
        .location = b.loc(),
    }));
    for (values, 0..) |v, i| {
        const slot = b.emit(mlir.Operation.make(ctx, "llvm.getelementptr", .{
            .operands = .{ .flat = &.{array.inner} },
            .results = .{ .flat = &.{ptr_type} },
            .attributes = &.{ .named(ctx, "elem_type", .typeAttr(elem)), .named(ctx, "rawConstantIndices", .denseArray(ctx, .i32, &[_]i32{@intCast(i)})) },
            .location = b.loc(),
        }));
        const value = if (T == u64) b.cst(.i64, @as(i64, @intCast(v))) else b.cst(.i32, @as(i32, @intCast(v)));
        b.emitVoid(dialects.llvm.store(ctx, value.inner, slot.inner, .{ .alignment = @sizeOf(T) }, b.loc()));
    }
    return array;
}

fn tensorMapType(b: *B) *const mlir.Type {
    return (dialects.cute.cuda.TensorMapType.get(b.ctx, .{}) catch unreachable).type_();
}

fn encodeTiled(b: *B, spec: L.TensorMapSpec, base: V) V {
    b.ctx.setAllowUnregisteredDialects(true);
    const addr = b.emit(dialects.cuda.cast(b.ctx, b.parseType("!ptr.ptr<#ptr.generic_space>"), b.ptrToInt(base, .i64).add(@as(i64, @intCast(spec.base_offset))).inner, b.loc()));
    const results = b.emitResults(2, dialects.cute.cuda.tensor_map_encode_tiled(
        b.ctx,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.data_type)))).inner,
        b.cst(.i32, @as(i32, @intCast(spec.rank))).inner,
        addr.inner,
        hostArray(b, u64, spec.global_dims[0..spec.rank]).inner,
        hostArray(b, u64, spec.global_strides[0 .. spec.rank - 1]).inner,
        hostArray(b, u32, spec.box_dims[0..spec.rank]).inner,
        hostArray(b, u32, spec.element_strides[0..spec.rank]).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.interleave)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.swizzle)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.l2_promotion)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.oob_fill)))).inner,
        dialects.cuda.resultType(b.ctx),
        tensorMapType(b),
        b.loc(),
    ));
    return results[1];
}

const ExplicitProgram = cute.Program(DescCfg, .{
    .name = "megamoe_tma_desc_explicit",
    .inputs = &.{"buf"},
    .outputs = &.{"out"},
    .run = explicitProgram,
});

fn explicitProgram(b: *B, cfg: DescCfg) cute.FinishError!void {
    var spec = mapSpec(cfg.which);
    spec.base_offset = 0;
    b.beginFunction("megamoe_tma_desc_explicit_kernel", .cuda_kernel);
    {
        const a = try b.declareArgs(.{
            .desc = .{ .mlir_type = .{ .type_ = tensorMapType(b), .grid_constant = true } },
            .out = .{ .ptr = cute.DType.i32 },
            .buf = .{ .ptr = cute.DType.i8 },
        });
        // The CuTe DSL marks `GridConstant[TensorMap]` with `cuda.grid_constant`.
        const empty: *const mlir.Attribute = .dict(b.ctx, &.{});
        b.setFunctionAttribute("arg_attrs", .array(b.ctx, &.{ .dict(b.ctx, &.{.named(b.ctx, "cuda.grid_constant", .unit(b.ctx))}), empty, empty }));
        var first = b.openIf(b.threadIdx().x.eq(0));
        {
            const llvm_ptr = b.emit(dialects.cuda.cast(b.ctx, dialects.llvm.pointerType(b.ctx, dialects.llvm.NVPTXAddressSpace.generic), a.desc.inner, b.loc()));
            const desc = b.emit(mlir.Operation.make(b.ctx, "llvm.ptrtoint", .{
                .operands = .{ .flat = &.{llvm_ptr.inner} },
                .results = .{ .flat = &.{cute.DType.i64.toMlir(b.ctx)} },
                .location = b.loc(),
            }));
            const out = b.ptrToInt(a.out, .i64);
            for (0..8) |i| ptx.stGlobal(b, 4, out.add(@as(i64, @intCast(16 * i))), b.loadWords(4, desc.add(@as(i64, @intCast(16 * i))), .generic, .{}));
            ptx.stGlobalU64(b, out.add(128), b.ptrToInt(a.buf, .i64));
            first.yieldThen(.{});
        }
    }
    b.endFunction(.{ 32, 1, 1 });

    b.beginFunction(ExplicitProgram.name, .host);
    const h = try b.declareArgs(.{
        .buf = .{ .ptr = cute.DType.i8 },
        .out = .{ .ptr = cute.DType.i32 },
    });
    const map = encodeTiled(b, spec, h.buf);
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, 32), one, one },
        .dynamic_smem = b.cst(.i64, 0),
        .stream = b.cudaStream(),
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_tma_desc_explicit_kernel", config, .{ map, h.out, h.buf })));
    b.endFunction(null);
}

fn explicitCall(buf: zml.Tensor, cfg: DescCfg) zml.Tensor {
    return ExplicitProgram.call(.{ .buf = buf }, .{ .out = .init(.{36}, .i32) }, .{ .cfg = cfg }).out;
}

test "experiment: explicit cuda.tensor_map_encode_tiled in the CuTe host function" {
    if (std.c.getenv("MEGAMOE_EXPLICIT_TMAP") == null) return error.SkipZigTest;
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const buf_shape: zml.Shape = .init(.{1 << 16}, .i8);
    const zeros = try allocator.alloc(u8, 1 << 16);
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var buf = try zml.Buffer.fromBytes(io, platform, buf_shape, zeros);
    defer buf.deinit();
    var exe = try platform.compileFn(allocator, io, explicitCall, .{ zml.Tensor.fromShape(buf_shape), DescCfg{ .which = 2 } }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [1]zml.Buffer = undefined;
    runner.run(io, .{buf}, &output, .{ .wait = true });
    defer output[0].deinit();
    const out = try output[0].toSliceAlloc(allocator, io);
    defer out.free(allocator);
    const bytes = out.constData();
    var spec = mapSpec(2);
    spec.base_offset = 0;
    const ref = try encodeReference(spec, std.mem.readInt(u64, bytes[128..136], .little));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&ref.opaque_), bytes[0..128]);
}

// ==================== 2-CTA TMA copies (DeepGEMM's TMA warps) ====================

const tma_smem_a = 0;
const tma_smem_b = 16384;
const tma_smem_sf = 32768;
const tma_smem_bulk = 33792;
const tma_smem_bar = 34816;
const tma_smem_bytes = 35840;

/// The small maps of `gt/tma_ref.cu`: acts [192][256] FP8 (box 128x96, SW128), SF int32 [2][256]
/// (box 256x1), packed FP4 weights [2][256][128] as 4-D (256, 128, 2, 2) (box 128x128, SW128,
/// unpacked), and a 64B-swizzled [32][128] store map (box 64x32).
fn tmaTestMaps() [4]lib.tma.Map {
    return .{
        .init(L.tma2d(.symm_buffer, 0, .fp8, 256, 192, 128, 96, 256, 128, true)),
        .init(L.tma2d(.symm_buffer, 0, .int32, 256, 2, 256, 1, 256, 0, true)),
        .init(L.tma4d(.l1_weights, .packed_fp4, .{ 256, 128, 2, 2 }, .{ 128, 128, 1, 1 }, .{ 128, 128 * 128, 256 * 128 }, 128, true)),
        .init(L.tma2d(.symm_buffer, 0, .fp8, 128, 32, 64, 32, 128, 64, false)),
    };
}

const TmaProgram = cute.Program(void, .{
    .name = "megamoe_tma_copies",
    .inputs = &.{ "acts", "sf", "weights", "bulk_src" },
    .outputs = &.{ "smem_out", "store_out", "bulk_out" },
    .run = tmaProgram,
});

fn tmaKernel(b: *B) cute.FinishError!void {
    const maps = tmaTestMaps();
    const a = try b.declareArgs(.{
        .map_a = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(maps[0].raw()), .grid_constant = true } },
        .map_sf = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(maps[1].raw()), .grid_constant = true } },
        .map_b = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(maps[2].raw()), .grid_constant = true } },
        .map_out = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(maps[3].raw()), .grid_constant = true } },
        .bulk_src = .{ .ptr = cute.DType.i8 },
        .smem_out = .{ .ptr = cute.DType.i32 },
        .bulk_out = .{ .ptr = cute.DType.i8 },
    });
    b.setFunctionAttribute("nvvm.minctasm", .int(b.ctx, .i32, 1));
    const smem = ptx.sharedStorageBase(b, tma_smem_bytes);
    const bar = smem.add(tma_smem_bar);
    const bulk_bar = smem.add(tma_smem_bar + 8);
    const tid = b.threadIdx().x;
    const cta = ptx.blockRankInCluster(b);
    const is_leader = cta.eq(0);

    var fill = b.openFor(0, tma_smem_bar / 4 / block, 1, .{});
    ptx.stSharedU32(b, smem.add(fill.iv.mul(block).add(tid).mul(4)), cst(b, 0xdeadbeef));
    fill.yield(.{});
    var init = b.openIf(tid.eq(0));
    ptx.mbarrierInit(b, bar, 2);
    ptx.mbarrierInit(b, bulk_bar, 1);
    ptx.fenceBarrierInit(b);
    init.yieldThen(.{});
    ptx.syncAligned(b, block, 0);
    ptx.clusterSync(b);

    var warp0 = b.openIf(ptx.divU(b, tid, 32).eq(0));
    {
        var elected = b.openIf(ptx.electOneSync(b));
        {
            const desc_a = b.rawTmaDescAddr(a.map_a, maps[0].raw());
            const desc_sf = b.rawTmaDescAddr(a.map_sf, maps[1].raw());
            const desc_b = b.rawTmaDescAddr(a.map_b, maps[2].raw());
            const m_idx = b.select(is_leader, b.cst(.i32, 0), b.cst(.i32, 96));
            ptx.tmaCopy2Sm(b, 128, 96, 128, 1, desc_a, bar, smem.add(tma_smem_a), b.cst(.i32, 128), &.{m_idx});
            ptx.tmaCopy2Sm(b, 256, 1, 0, 4, desc_sf, bar, smem.add(tma_smem_sf), b.cst(.i32, 0), &.{b.cst(.i32, 1)});
            ptx.tmaCopy2Sm(b, 128, 128, 128, 1, desc_b, bar, smem.add(tma_smem_b), b.cst(.i32, 128), &.{ b.cst(.i32, 0), cta, b.cst(.i32, 1) });
            var leader = b.openIf(is_leader);
            ptx.mbarrierArriveAndExpectTx(b, bar, 2 * 96 * 128 + 2 * 1024 + 2 * 128 * 128 / 2);
            leader.yieldThen(.{});
            var peer = b.openIf(is_leader.eq(false));
            ptx.mbarrierArriveCluster(b, bar, b.cst(.i32, 0));
            peer.yieldThen(.{});
            const bulk_src = b.ptrToInt(a.bulk_src, .i64).add(cta.to(.i64).mul(1024));
            ptx.tmaLoad1d(b, smem.add(tma_smem_bulk), bulk_src, bulk_bar, 1024, ptx.CacheHint.evict_first);
            ptx.mbarrierArriveAndSetTx(b, bulk_bar, 1024);
            _ = ptx.mbarrierWaitAndFlipPhase(b, bulk_bar, b.cst(.i32, 0));
            ptx.tmaStore1d(b, b.ptrToInt(a.bulk_out, .i64).add(cta.to(.i64).mul(1024)), smem.add(tma_smem_bulk), 1024, ptx.CacheHint.evict_normal);
            ptx.tmaStoreArrive(b);
            ptx.tmaStoreWait(b, 0);
            elected.yieldThen(.{});
        }
        ptx.syncWarp(b);
        warp0.yieldThen(.{});
    }
    var wait = b.openIf(is_leader);
    ptx.mbarrierWait(b, bar, b.cst(.i32, 0));
    wait.yieldThen(.{});
    ptx.clusterSync(b);
    var store = b.openIf(tid.eq(0));
    {
        const desc_out = b.rawTmaDescAddr(a.map_out, maps[3].raw());
        ptx.tmaStoreFence(b);
        ptx.tmaStore2d(b, desc_out, smem.add(tma_smem_a), cta.mul(64), b.cst(.i32, 0));
        ptx.tmaStoreArrive(b);
        ptx.tmaStoreWait(b, 0);
        store.yieldThen(.{});
    }
    ptx.syncAligned(b, block, 0);
    var copy = b.openFor(0, tma_smem_bar / 4 / block, 1, .{});
    {
        const j = copy.iv.mul(block).add(tid);
        const dst = b.ptrToInt(a.smem_out, .i64).add(cta.to(.i64).mul(tma_smem_bar)).add(j.to(.i64).mul(4));
        ptx.stGlobal(b, 1, dst, .{ptx.ldSharedU32(b, smem.add(j.mul(4)))});
        copy.yield(.{});
    }
}

fn tmaProgram(b: *B, _: void) cute.FinishError!void {
    b.beginFunction("megamoe_tma_copies_kernel", .cuda_kernel);
    try tmaKernel(b);
    b.endFunction(.{ block, 1, 1 });
    b.beginFunction(TmaProgram.name, .host);
    const h = try b.declareArgs(.{
        .acts = .{ .ptr = cute.DType.i8 },
        .sf = .{ .ptr = cute.DType.i32 },
        .weights = .{ .ptr = cute.DType.i8 },
        .bulk_src = .{ .ptr = cute.DType.i8 },
        .smem_out = .{ .ptr = cute.DType.i32 },
        .store_out = .{ .ptr = cute.DType.i8 },
        .bulk_out = .{ .ptr = cute.DType.i8 },
    });
    const maps = tmaTestMaps();
    const map_a = b.makeRawTmaAtom(h.acts, maps[0].raw());
    const map_sf = b.makeRawTmaAtom(h.sf, maps[1].raw());
    const map_b = b.makeRawTmaAtom(h.weights, maps[2].raw());
    const map_out = b.makeRawTmaAtom(h.store_out, maps[3].raw());
    const one = b.cst(.i32, 1);
    const two = b.cst(.i32, 2);
    const config = b.makeLaunchConfig(.{
        .grid = .{ two, one, one },
        .block = .{ b.cst(.i32, block), one, one },
        .dynamic_smem = b.kernelSmemSize("megamoe_tma_copies_kernel"),
        .stream = b.cudaStream(),
        .cluster = .{ two, one, one },
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_tma_copies_kernel", config, .{ map_a.atom, map_sf.atom, map_b.atom, map_out.atom, h.bulk_src, h.smem_out, h.bulk_out })));
    b.endFunction(null);
}

fn tmaCall(acts: zml.Tensor, sf: zml.Tensor, weights: zml.Tensor, bulk_src: zml.Tensor) [3]zml.Tensor {
    const r = TmaProgram.call(.{ .acts = acts, .sf = sf, .weights = weights, .bulk_src = bulk_src }, .{
        .smem_out = .init(.{2 * tma_smem_bar / 4}, .i32),
        .store_out = .init(.{32 * 128}, .i8),
        .bulk_out = .init(.{2048}, .i8),
    }, .{ .cfg = {} });
    return .{ r.smem_out, r.store_out, r.bulk_out };
}

test "2-CTA TMA loads, TMA store and bulk copies match DeepGEMM's on a cluster" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const shapes = [_]zml.Shape{ .init(.{192 * 256}, .i8), .init(.{2 * 256}, .i32), .init(.{2 * 256 * 128}, .i8), .init(.{2048}, .i8) };
    const names = [_][]const u8{ "tma_a.bin", "tma_sf.bin", "tma_w.bin", "tma_bulk_src.bin" };
    var inputs: [4]zml.Buffer = undefined;
    for (&inputs, shapes, names) |*input, shape, name| {
        const bytes = try readRef(allocator, io, name);
        defer allocator.free(bytes);
        input.* = try zml.Buffer.fromBytes(io, platform, shape, bytes);
    }
    defer for (&inputs) |*input| input.deinit();
    var exe = try platform.compileFn(allocator, io, tmaCall, .{ zml.Tensor.fromShape(shapes[0]), zml.Tensor.fromShape(shapes[1]), zml.Tensor.fromShape(shapes[2]), zml.Tensor.fromShape(shapes[3]) }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [3]zml.Buffer = undefined;
    runner.run(io, .{ inputs[0], inputs[1], inputs[2], inputs[3] }, &output, .{ .wait = true });
    defer for (&output) |*o| o.deinit();
    inline for (.{ "tma_smem_out.bin", "tma_store_out.bin", "tma_bulk_out.bin" }, 0..) |name, i| {
        const got = try output[i].toSliceAlloc(allocator, io);
        defer got.free(allocator);
        const ref = try readRef(allocator, io, name);
        defer allocator.free(ref);
        try expectBits(name, ref, got.constData());
    }
}

// ==================== every primitive through the CuTe compiler and ptxas ====================

const Groups = lib.ptx_test.Groups;

const CompileProgram = cute.Program(Groups, .{
    .name = "megamoe_ptx_compile",
    .inputs = &.{"buf"},
    .outputs = &.{"out"},
    .run = compileProgram,
});

/// Every `ptx` primitive behind a branch no thread takes: compiling the program runs the CuTe
/// compiler and ptxas over all of DeepGEMM's PTX strings (inline asm is only checked there).
fn compileProgram(b: *B, groups: Groups) cute.FinishError!void {
    b.beginFunction("megamoe_ptx_compile_kernel", .cuda_kernel);
    {
        const a = try b.declareArgs(.{
            .buf = .{ .ptr = cute.DType.i8 },
            .out = .{ .ptr = cute.DType.i32 },
            .f = .{ .mlir_type = .{ .type_ = cute.DType.f32.toMlir(b.ctx) } },
        });
        // ptxas aborts with an internal error on `stmatrix` to a shared address outside any
        // shared allocation, so the primitives get a real one.
        const smem = ptx.sharedStorageBase(b, 4096);
        var never = b.openIf(b.blockIdx().x.eq(12345));
        lib.ptx_test.emitEveryPrimitive(b, b.ptrToInt(a.buf, .i64), smem.add(1024), a.f, groups);
        never.yieldThen(.{});
        var first = b.openIf(b.threadIdx().x.eq(0));
        ptx.stGlobal(b, 1, b.ptrToInt(a.out, .i64), .{b.cst(.i32, 42)});
        first.yieldThen(.{});
    }
    b.endFunction(.{ 512, 1, 1 });
    b.beginFunction(CompileProgram.name, .host);
    const h = try b.declareArgs(.{
        .buf = .{ .ptr = cute.DType.i8 },
        .out = .{ .ptr = cute.DType.i32 },
    });
    const one = b.cst(.i32, 1);
    const two = b.cst(.i32, 2);
    const config = b.makeLaunchConfig(.{
        .grid = .{ two, one, one },
        .block = .{ b.cst(.i32, 512), one, one },
        .dynamic_smem = b.kernelSmemSize("megamoe_ptx_compile_kernel"),
        .stream = b.cudaStream(),
        .cluster = .{ two, one, one },
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_ptx_compile_kernel", config, .{ h.buf, h.out, b.cst(.f32, 1.0) })));
    b.endFunction(null);
}

fn compileCall(buf: zml.Tensor, groups: Groups) zml.Tensor {
    return CompileProgram.call(.{ .buf = buf }, .{ .out = .init(.{1}, .i32) }, .{ .cfg = groups }).out;
}

fn compileAndRun(groups: Groups) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const shape: zml.Shape = .init(.{256}, .i8);
    var exe = try platform.compileFn(allocator, io, compileCall, .{ zml.Tensor.fromShape(shape), groups }, .{});
    defer exe.deinit();
    const zeros: [256]u8 = @splat(0);
    var buf = try zml.Buffer.fromBytes(io, platform, shape, &zeros);
    defer buf.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [1]zml.Buffer = undefined;
    runner.run(io, .{buf}, &output, .{ .wait = true });
    defer output[0].deinit();
    const out = try output[0].toSliceAlloc(allocator, io);
    defer out.free(allocator);
    try std.testing.expectEqual(@as(i32, 42), out.constItems(i32)[0]);
}

test "every PTX primitive compiles with the CuTe compiler and ptxas" {
    var failures: usize = 0;
    inline for (@typeInfo(Groups).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "shared_mask")) continue;
        var only: Groups = undefined;
        inline for (@typeInfo(Groups).@"struct".field_names) |other| {
            if (comptime std.mem.eql(u8, other, "shared_mask")) continue;
            @field(only, other) = comptime std.mem.eql(u8, name, other);
        }
        only.shared_mask = 0xFFF;
        if (compileAndRun(only)) {
            std.debug.print("group {s}: ok\n", .{name});
        } else |err| {
            std.debug.print("group {s}: {s}\n", .{ name, @errorName(err) });
            failures += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
    try compileAndRun(.{});
}

// ==================== the MegaMoE kernel through the CuTe compiler and ptxas ====================

const KernelInstance = lib.kernel_test.Instance;

/// Which instantiation `megamoeProgram` emits.
const MegaMoECfg = struct { which: u8 };

fn megamoeInstance(which: u8) KernelInstance {
    return switch (which) {
        0 => lib.kernel_test.vllm_dsv41,
        1 => lib.kernel_test.single_rank,
        else => unreachable,
    };
}

const MegaMoEProgram = cute.Program(MegaMoECfg, .{
    .name = "megamoe_compile",
    .inputs = &.{ "workspace", "l1_w", "l1_sf", "l2_w", "l2_sf", "shared_l1_w", "shared_l1_sf", "shared_l2_w", "shared_l2_sf", "sm_locality_domains" },
    .outputs = &.{"y"},
    .run = megamoeProgram,
});

/// The MegaMoE kernel and its host launch (`program.emitLaunch`). For the two-rank instance the
/// host passes the same buffer for both ranks: this program is only compiled, never run.
fn megamoeProgram(b: *B, mcfg: MegaMoECfg) cute.FinishError!void {
    const inst = megamoeInstance(mcfg.which);
    const cfg = inst.config();
    try lib.kernel.emit(b, cfg, "megamoe_kernel");
    b.beginFunction(MegaMoEProgram.name, .host);
    const h = try b.declareArgs(.{
        .workspace = .{ .ptr = cute.DType.i8 },
        .l1_w = .{ .ptr = cute.DType.i8 },
        .l1_sf = .{ .ptr = cute.DType.i8 },
        .l2_w = .{ .ptr = cute.DType.i8 },
        .l2_sf = .{ .ptr = cute.DType.i8 },
        .shared_l1_w = .{ .ptr = cute.DType.i8 },
        .shared_l1_sf = .{ .ptr = cute.DType.i8 },
        .shared_l2_w = .{ .ptr = cute.DType.i8 },
        .shared_l2_sf = .{ .ptr = cute.DType.i8 },
        .sm_locality_domains = .{ .ptr = cute.DType.i8 },
        .y = .{ .ptr = cute.DType.bf16 },
    });
    const symmetric: lib.program.Symmetric = if (inst.num_ranks == 1)
        .{ .single = h.workspace }
    else
        .{ .ranks = .{ .buffers = &.{ h.workspace, h.workspace }, .rank = b.cst(.i32, 0) } };
    const status = lib.program.emitLaunch(b, cfg, "megamoe_kernel", .{
        .y = h.y,
        .symmetric = symmetric,
        .weights = .{ h.l1_w, h.l1_sf, h.l2_w, h.l2_sf, h.shared_l1_w, h.shared_l1_sf, h.shared_l2_w, h.shared_l2_sf },
        .sm_locality_domains = h.sm_locality_domains,
        .num_tokens = b.cst(.i32, @as(i32, @intCast(inst.num_tokens))),
    });
    b.returnHostStatus(status);
    b.endFunction(null);
}

/// Input shapes (flat bytes; only the pointers matter to the host function).
fn megamoeShapes(inst: KernelInstance) [10]zml.Shape {
    const cfg = inst.config();
    const k = cfg.kernel;
    const s = k.shape;
    const symm: L.SymmBufferLayout = .init(s, inst.num_sms);
    const E: u64 = k.numExpertsPerRank();
    const H: u64 = s.hidden;
    const I: u64 = s.intermediate_hidden;
    const sI: u64 = s.sharedIntermediateHidden();
    const pack: u64 = if (k.weight_type == .fp4_unpacked) 2 else 1;
    const bytes = [10]u64{
        symm.numBytes(),
        E * 2 * I * H / pack,
        E * 2 * I * (H / 128) * 4,
        E * H * I / pack,
        E * H * (I / 128) * 4,
        @max(1, 2 * sI * H),
        @max(1, 2 * sI * (H / 128) * 4),
        @max(1, H * sI),
        @max(1, H * (sI / 128) * 4),
        inst.num_sms,
    };
    var shapes: [10]zml.Shape = undefined;
    for (&shapes, bytes) |*sh, n| sh.* = .init(.{@as(i64, @intCast(n))}, .i8);
    return shapes;
}

fn megamoeCall(workspace: zml.Tensor, l1_w: zml.Tensor, l1_sf: zml.Tensor, l2_w: zml.Tensor, l2_sf: zml.Tensor, shared_l1_w: zml.Tensor, shared_l1_sf: zml.Tensor, shared_l2_w: zml.Tensor, shared_l2_sf: zml.Tensor, sm_locality_domains: zml.Tensor, mcfg: MegaMoECfg) zml.Tensor {
    const inst = megamoeInstance(mcfg.which);
    return MegaMoEProgram.call(.{
        .workspace = workspace,
        .l1_w = l1_w,
        .l1_sf = l1_sf,
        .l2_w = l2_w,
        .l2_sf = l2_sf,
        .shared_l1_w = shared_l1_w,
        .shared_l1_sf = shared_l1_sf,
        .shared_l2_w = shared_l2_w,
        .shared_l2_sf = shared_l2_sf,
        .sm_locality_domains = sm_locality_domains,
    }, .{ .y = .init(.{ @as(i64, inst.num_tokens), @as(i64, inst.hidden) }, .bf16) }, .{ .cfg = mcfg }).y;
}

test "MegaMoE kernel compiles with the CuTe compiler and ptxas (not run)" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for ([_]u8{ 0, 1 }) |which| {
        const inst = megamoeInstance(which);
        if (std.c.getenv("MEGAMOE_DUMP_IR")) |dir| {
            const ir = try MegaMoEProgram.emit(allocator, .{ .which = which });
            defer allocator.free(ir);
            const path = try std.fmt.allocPrint(allocator, "{s}/megamoe_{d}.mlir", .{ std.mem.span(dir), which });
            defer allocator.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = ir });
        }
        const shapes = megamoeShapes(inst);
        var args: [10]zml.Tensor = undefined;
        for (&args, shapes) |*a, s| a.* = .fromShape(s);
        // MEGAMOE_XLA_DUMP=<dir>: XLA's dumps, incl. the CuTe module and its fatbin.
        const dump: ?[]const u8 = if (std.c.getenv("MEGAMOE_XLA_DUMP")) |d| try std.fmt.allocPrint(allocator, "{s}/instance{d}", .{ std.mem.span(d), which }) else null;
        defer if (dump) |d| allocator.free(d);
        var exe = try platform.compileFn(allocator, io, megamoeCall, .{ args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], MegaMoECfg{ .which = which } }, .{ .xla_dump_to = dump });
        exe.deinit();
        std.debug.print("MegaMoE instance {d} (ranks {d}, BLOCK_M {d}): compiled\n", .{ which, inst.num_ranks, inst.config().kernel.cfg.block_m });
    }
}

// ==================== one-rank run: protocol with random inputs ====================
// MEGAMOE_RUN=1 runs the kernel on one rank with staged random inputs and uninitialized weights
// (garbage numerics, so `y` is not checked here; the DeepGEMM dumps below check it): the
// dispatch's outputs in the symmetric buffer (L1 ring tokens, SF, weights, source metadata), the
// workspace's self-cleaning and the grid / NVLink barrier counters. Two launches run back to
// back on the same buffer.

const RunCfg = struct { num_tokens: u32 };

fn runInstance(num_tokens: u32) KernelInstance {
    // About 15 s at 2 GHz before a `wait_until` traps.
    return .{ .num_ranks = 1, .num_tokens = num_tokens, .timeout_cycles = 30_000_000_000 };
}

/// SM locality table size: covers any `%smid` (DeepGEMM indexes it by `%smid`).
const run_sm_table = 256;

const RunProgram = cute.Program(RunCfg, .{
    .name = "megamoe_run",
    .inputs = &.{ "workspace", "l1_w", "l1_sf", "l2_w", "l2_sf", "shared_l1_w", "shared_l1_sf", "shared_l2_w", "shared_l2_sf", "sm_locality_domains" },
    .outputs = &.{ "y", "workspace_out" },
    .run = runProgram,
});

fn runProgram(b: *B, rcfg: RunCfg) cute.FinishError!void {
    const inst = runInstance(rcfg.num_tokens);
    const cfg = inst.config();
    try lib.kernel.emit(b, cfg, "megamoe_kernel");
    b.beginFunction(RunProgram.name, .host);
    const h = try b.declareArgs(.{
        .workspace = .{ .ptr = cute.DType.i8 },
        .l1_w = .{ .ptr = cute.DType.i8 },
        .l1_sf = .{ .ptr = cute.DType.i8 },
        .l2_w = .{ .ptr = cute.DType.i8 },
        .l2_sf = .{ .ptr = cute.DType.i8 },
        .shared_l1_w = .{ .ptr = cute.DType.i8 },
        .shared_l1_sf = .{ .ptr = cute.DType.i8 },
        .shared_l2_w = .{ .ptr = cute.DType.i8 },
        .shared_l2_sf = .{ .ptr = cute.DType.i8 },
        .sm_locality_domains = .{ .ptr = cute.DType.i8 },
        .y = .{ .ptr = cute.DType.bf16 },
        .workspace_out = .{ .ptr = cute.DType.i8 },
    });
    const status = lib.program.emitLaunch(b, cfg, "megamoe_kernel", .{
        .y = h.y,
        .symmetric = .{ .single = h.workspace_out },
        .weights = .{ h.l1_w, h.l1_sf, h.l2_w, h.l2_sf, h.shared_l1_w, h.shared_l1_sf, h.shared_l2_w, h.shared_l2_sf },
        .sm_locality_domains = h.sm_locality_domains,
        .num_tokens = b.cst(.i32, @as(i32, @intCast(inst.num_tokens))),
    });
    b.returnHostStatus(status);
    b.endFunction(null);
}

fn runCall(workspace: zml.Tensor, l1_w: zml.Tensor, l1_sf: zml.Tensor, l2_w: zml.Tensor, l2_sf: zml.Tensor, shared_l1_w: zml.Tensor, shared_l1_sf: zml.Tensor, shared_l2_w: zml.Tensor, shared_l2_sf: zml.Tensor, sm_locality_domains: zml.Tensor, rcfg: RunCfg) [2]zml.Tensor {
    const inst = runInstance(rcfg.num_tokens);
    const r = RunProgram.call(.{
        .workspace = workspace,
        .l1_w = l1_w,
        .l1_sf = l1_sf,
        .l2_w = l2_w,
        .l2_sf = l2_sf,
        .shared_l1_w = shared_l1_w,
        .shared_l1_sf = shared_l1_sf,
        .shared_l2_w = shared_l2_w,
        .shared_l2_sf = shared_l2_sf,
        .sm_locality_domains = sm_locality_domains,
    }, .{
        .y = .init(.{ @as(i64, inst.num_tokens), @as(i64, inst.hidden) }, .bf16),
        .workspace_out = workspace.shape(),
    }, .{ .cfg = rcfg, .output_operand_aliases = .{ .workspace_out = .workspace } });
    return .{ r.y, r.workspace_out };
}

/// The staged inputs of one rank and the routing they imply.
const RunData = struct {
    T: u32,
    H: u32,
    K: u32,
    E: u32,
    x: []u8,
    x_sf: []u32,
    topk_idx: []i64,
    topk_w: []f32,

    fn init(allocator: std.mem.Allocator, inst: KernelInstance) !RunData {
        const T = inst.num_tokens;
        const H = inst.hidden;
        const K = inst.num_topk;
        const E = inst.num_experts;
        var prng: std.Random.DefaultPrng = .init(0x5eed_0000 + T);
        const rnd = prng.random();
        const d: RunData = .{
            .T = T,
            .H = H,
            .K = K,
            .E = E,
            .x = try allocator.alloc(u8, @as(usize, T) * H),
            .x_sf = try allocator.alloc(u32, @as(usize, T) * (H / 128)),
            .topk_idx = try allocator.alloc(i64, @as(usize, T) * K),
            .topk_w = try allocator.alloc(f32, @as(usize, T) * K),
        };
        rnd.bytes(d.x);
        for (d.x_sf) |*v| v.* = rnd.int(u32);
        for (d.topk_w) |*v| v.* = rnd.float(f32);
        for (0..T) |t| {
            // Distinct experts per token; every 7th token has its last slot unrouted (-1).
            var k: usize = 0;
            while (k < K) {
                const e: i64 = rnd.intRangeLessThan(i64, 0, @as(i64, E));
                var dup = false;
                for (0..k) |j| dup = dup or d.topk_idx[t * K + j] == e;
                if (dup) continue;
                d.topk_idx[t * K + k] = e;
                k += 1;
            }
            if (t % 7 == 3) d.topk_idx[t * K + K - 1] = -1;
        }
        return d;
    }

    fn deinit(self: RunData, allocator: std.mem.Allocator) void {
        allocator.free(self.x);
        allocator.free(self.x_sf);
        allocator.free(self.topk_idx);
        allocator.free(self.topk_w);
    }
};

fn readAt(comptime T: type, bytes: []const u8, offset: u64) T {
    return std.mem.readInt(T, bytes[@intCast(offset)..][0..@sizeOf(T)], .little);
}

/// Checks the symmetric buffer after a launch: dispatch outputs (every routed token-topk exactly
/// once in its expert's pool slots, with its token row, SF and weight in the L1 ring when the pool
/// fits the ring) and the workspace cleanup / barrier counters.
fn checkRun(allocator: std.mem.Allocator, inst: KernelInstance, d: RunData, ws: []const u8, launches: u32) !void {
    const cfg = inst.config();
    const k = cfg.kernel;
    const s = k.shape;
    const symm: L.SymmBufferLayout = .init(s, inst.num_sms);
    const buf = symm.buffer;
    const BM = k.cfg.block_m;
    const E = s.num_experts; // one rank: every expert is local
    const sig = L.Signals;

    // Expected routing: per expert, the multiset of token-topk indices.
    const counts = try allocator.alloc(u32, E);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (d.topk_idx) |e| if (e >= 0) {
        counts[@intCast(e)] += 1;
    };
    var pool_blocks: u32 = 0;
    var mismatches: usize = 0;
    const seen = try allocator.alloc(bool, d.topk_idx.len);
    defer allocator.free(seen);
    @memset(seen, false);
    var total_pool_blocks: u32 = 0;
    for (counts) |n| total_pool_blocks += L.ceilDiv(n, BM);
    const no_reuse = total_pool_blocks <= k.numRingBlocks();
    const num_sf = s.hidden / 128;
    for (0..E) |e| {
        for (0..counts[e]) |i_| {
            const i: u32 = @intCast(i_);
            const p = pool_blocks * BM + i;
            const meta = buf.workspace.tokenSrcMetadataOffset(p);
            const rank = readAt(u32, ws, meta);
            const token = readAt(u32, ws, meta + 4);
            const topk = readAt(u32, ws, meta + 8);
            const tk = @as(usize, token) * d.K + topk;
            if (rank != 0 or token >= d.T or topk >= d.K or d.topk_idx[tk] != @as(i64, @intCast(e)) or seen[tk]) {
                if (mismatches < 8) std.debug.print("expert {d} slot {d}: metadata ({d}, {d}, {d})\n", .{ e, i, rank, token, topk });
                mismatches += 1;
                continue;
            }
            seen[tk] = true;
            if (!no_reuse) continue;
            // L1 ring: token row, weight, SF (UTCCP-transposed).
            const ring_token = p % k.cfg.num_ring_tokens;
            const row = ws[@intCast(buf.l1_token.itemOffset(0, ring_token))..][0..d.H];
            if (!std.mem.eql(u8, row, d.x[@as(usize, token) * d.H ..][0..d.H])) {
                if (mismatches < 8) std.debug.print("expert {d} slot {d}: token row differs\n", .{ e, i });
                mismatches += 1;
            }
            const w = readAt(u32, ws, buf.l1_topk_weights.itemOffset(0, ring_token));
            if (w != @as(u32, @bitCast(d.topk_w[tk]))) {
                if (mismatches < 8) std.debug.print("expert {d} slot {d}: weight differs\n", .{ e, i });
                mismatches += 1;
            }
            const ring_block = (p / BM) % k.numRingBlocks();
            const sf_token = ring_block * k.cfg.sf_block_m + k.transformSfTokenIdx(i % BM);
            for (0..num_sf) |j| {
                const got = readAt(u32, ws, buf.l1_sf.offset + (@as(u64, j) * k.cfg.num_sf_ring_tokens + sf_token) * 4);
                if (got != d.x_sf[@as(usize, token) * num_sf + j]) {
                    if (mismatches < 8) std.debug.print("expert {d} slot {d}: SF {d} differs\n", .{ e, i, j });
                    mismatches += 1;
                    break;
                }
            }
        }
        pool_blocks += L.ceilDiv(counts[e], BM);
    }
    for (d.topk_idx, seen, 0..) |e, was_seen, tk| {
        if (e >= 0 and !was_seen) {
            if (mismatches < 8) std.debug.print("token-topk {d} (expert {d}) not dispatched\n", .{ tk, e });
            mismatches += 1;
        }
    }

    // Workspace cleanup and barrier counters.
    var dirty: usize = 0;
    const zero_u64 = [_][]const u8{ "expert_send_count", "expert_recv_count", "expert_recv_count_sum" };
    inline for (zero_u64) |name| {
        const n = if (comptime std.mem.eql(u8, name, "expert_recv_count_sum")) sig.max_experts_per_rank else sig.max_experts;
        for (0..n) |i| if (readAt(u64, ws, sig.offset(name) + 8 * i) != 0) {
            dirty += 1;
        };
    }
    inline for (.{ "l1_task_count", "l2_task_count", "shared_l1_task_count", "shared_l2_task_count" }) |name| {
        for (0..L.device_locality_domains) |i| if (readAt(u32, ws, sig.offset(name) + 4 * i) != 0) {
            dirty += 1;
        };
    }
    for (0..k.numRingBlocks()) |rb| {
        if (readAt(u32, ws, sig.offset("l1_full_count") + 4 * rb) != 0) dirty += 1;
        if (readAt(u32, ws, sig.offset("l1_empty_count") + 4 * rb) != 0) dirty += 1;
        if (readAt(u64, ws, sig.offset("l2_full_mask") + 8 * rb) != 0) dirty += 1;
        if (readAt(u32, ws, sig.offset("l2_empty_count") + 4 * rb) != 0) dirty += 1;
    }
    for (0..buf.workspace.num_shared_l2_pool_blocks) |i| if (readAt(u32, ws, sig.offset("shared_l2_full_count") + 4 * i) != 0) {
        dirty += 1;
    };
    if (dirty != 0) std.debug.print("{d} workspace signals not cleaned\n", .{dirty});

    // Grid syncs: counter 0 is used 3 times per launch (dispatch, both NVLink barriers), counter 1
    // once (epilogue); each sync adds 0x80000000. NVLink barrier: 2 per launch.
    const gs0 = readAt(u32, ws, sig.offset("grid_sync_count"));
    const gs1 = readAt(u32, ws, sig.offset("grid_sync_count") + 4);
    const nvl = readAt(u32, ws, sig.offset("nvl_barrier_counter"));
    const peer = readAt(u64, ws, sig.offset("peer_grid_idx"));
    const ready = readAt(u64, ws, sig.offset("combine_ready_grid_idx"));
    std.debug.print("grid syncs 0x{x} 0x{x}, NVLink barriers {d}, peer grid idx {d}, combine ready {d}, pool blocks {d} (ring {d}), {d} metadata/ring mismatches\n", .{ gs0, gs1, nvl, peer, ready, total_pool_blocks, k.numRingBlocks(), mismatches });
    const expected_gs0: u32 = @truncate(@as(u64, 3 * launches) * 0x80000000);
    const expected_gs1: u32 = @truncate(@as(u64, launches) * 0x80000000);
    try std.testing.expectEqual(expected_gs0, gs0);
    try std.testing.expectEqual(expected_gs1, gs1);
    try std.testing.expectEqual(2 * launches, nvl);
    try std.testing.expect(peer != 0 and peer == ready);
    try std.testing.expectEqual(@as(usize, 0), dirty);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

fn runOnce(num_tokens: u32) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const inst = runInstance(num_tokens);
    const cfg = inst.config();
    const s = cfg.kernel.shape;
    const symm: L.SymmBufferLayout = .init(s, inst.num_sms);
    const buf = symm.buffer;
    std.debug.print("MegaMoE run: T {d}, BLOCK_M {d}, buffer {d} bytes\n", .{ num_tokens, cfg.kernel.cfg.block_m, symm.numBytes() });

    // Stage the inputs into a zeroed symmetric buffer.
    const d = try RunData.init(allocator, inst);
    defer d.deinit(allocator);
    const ws_host = try allocator.alloc(u8, @intCast(symm.numBytes()));
    defer allocator.free(ws_host);
    @memset(ws_host, 0);
    @memcpy(ws_host[@intCast(buf.input_token.offset)..][0..d.x.len], d.x);
    @memcpy(ws_host[@intCast(buf.input_sf.offset)..][0 .. d.x_sf.len * 4], std.mem.sliceAsBytes(d.x_sf));
    @memcpy(ws_host[@intCast(buf.input_topk_idx.offset)..][0 .. d.topk_idx.len * 8], std.mem.sliceAsBytes(d.topk_idx));
    @memcpy(ws_host[@intCast(buf.input_topk_weights.offset)..][0 .. d.topk_w.len * 4], std.mem.sliceAsBytes(d.topk_w));

    const shapes = megamoeShapes(inst);
    var inputs: [10]zml.Buffer = undefined;
    inputs[0] = try zml.Buffer.fromBytes(io, platform, shapes[0], ws_host);
    for (1..9) |i| inputs[i] = try zml.Buffer.uninitialized(io, platform, shapes[i], .{});
    var table: [run_sm_table]u8 = undefined;
    for (&table, 0..) |*t, i| t.* = L.evenSmLocalityDomain(@intCast(i));
    inputs[9] = try zml.Buffer.fromBytes(io, platform, .init(.{run_sm_table}, .i8), &table);
    defer for (inputs[1..]) |*x| x.deinit();

    var args: [10]zml.Tensor = undefined;
    for (&args, inputs) |*a, x| a.* = .fromShape(x.shape());
    var exe = try platform.compileFn(allocator, io, runCall, .{ args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], RunCfg{ .num_tokens = num_tokens } }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var ws = inputs[0];
    for (1..3) |launch| {
        var output: [2]zml.Buffer = undefined;
        runner.run(io, .{ ws, inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9] }, &output, .{ .wait = true });
        output[0].deinit();
        ws.deinit();
        ws = output[1];
        const got = try ws.toSliceAlloc(allocator, io);
        defer got.free(allocator);
        std.debug.print("launch {d}: ", .{launch});
        try checkRun(allocator, inst, d, got.constData(), @intCast(launch));
    }
    ws.deinit();
}

test "MegaMoE one-rank run: dispatch, scheduler, TMA and MMA protocol, workspace cleanup" {
    if (std.c.getenv("MEGAMOE_RUN") == null) return error.SkipZigTest;
    // BLOCK_M 16 (the pool fits the ring: ring contents checked), then BLOCK_M 192 (ring reuse).
    try runOnce(512);
    try runOnce(8192);
}

// ==================== single-rank DeepGEMM dumps: bit-exact y ====================
// MEGAMOE_DUMPS=<dir>[,<dir>...] (reference-track dumps `S/dumps/<name>` with one rank): stage
// the dump's kernel inputs (x, x_sf, topk_idx, topk_weights and the shared expert's L1 SF pages)
// into a zeroed symmetric buffer, load its weights, run the kernel MEGAMOE_DUMP_LAUNCHES times
// (default 2) back to back without re-zeroing and compare `y` with DeepGEMM's bit for bit after
// each launch; the dispatch outputs, the workspace cleanup and the barrier counters are checked
// as in the one-rank run above.

/// A safetensors file: its JSON header, tensors read by name with positional reads.
const Safetensors = struct {
    file: std.Io.File,
    data_start: u64,
    header_bytes: []u8,
    header: std.json.Parsed(std.json.Value),

    fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Safetensors {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        var len_buf: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &len_buf, 0) != 8) return error.ShortRead;
        const n = std.mem.readInt(u64, &len_buf, .little);
        const header_bytes = try allocator.alloc(u8, @intCast(n));
        errdefer allocator.free(header_bytes);
        if (try file.readPositionalAll(io, header_bytes, 8) != n) return error.ShortRead;
        const header = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        return .{ .file = file, .data_start = 8 + n, .header_bytes = header_bytes, .header = header };
    }

    fn has(self: *const Safetensors, name: []const u8) bool {
        return self.header.value.object.get(name) != null;
    }

    /// The tensor's bytes (its memory image as saved).
    fn read(self: *const Safetensors, allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
        const entry = self.header.value.object.get(name) orelse {
            std.debug.print("tensor {s} missing\n", .{name});
            return error.MissingTensor;
        };
        const offsets = entry.object.get("data_offsets").?.array.items;
        const start: u64 = @intCast(offsets[0].integer);
        const end: u64 = @intCast(offsets[1].integer);
        const buf = try allocator.alloc(u8, @intCast(end - start));
        errdefer allocator.free(buf);
        if (try self.file.readPositionalAll(io, buf, self.data_start + start) != buf.len) return error.ShortRead;
        return buf;
    }

    fn deinit(self: *Safetensors, allocator: std.mem.Allocator, io: std.Io) void {
        self.header.deinit();
        allocator.free(self.header_bytes);
        self.file.close(io);
    }
};

const DumpProgram = cute.Program(KernelInstance, .{
    .name = "megamoe_dump",
    .inputs = &.{ "workspace", "l1_w", "l1_sf", "l2_w", "l2_sf", "shared_l1_w", "shared_l1_sf", "shared_l2_w", "shared_l2_sf", "sm_locality_domains" },
    .outputs = &.{ "y", "workspace_out" },
    .run = dumpProgram,
});

fn dumpProgram(b: *B, inst: KernelInstance) cute.FinishError!void {
    const cfg = inst.config();
    try lib.kernel.emit(b, cfg, "megamoe_kernel");
    b.beginFunction(DumpProgram.name, .host);
    const h = try b.declareArgs(.{
        .workspace = .{ .ptr = cute.DType.i8 },
        .l1_w = .{ .ptr = cute.DType.i8 },
        .l1_sf = .{ .ptr = cute.DType.i8 },
        .l2_w = .{ .ptr = cute.DType.i8 },
        .l2_sf = .{ .ptr = cute.DType.i8 },
        .shared_l1_w = .{ .ptr = cute.DType.i8 },
        .shared_l1_sf = .{ .ptr = cute.DType.i8 },
        .shared_l2_w = .{ .ptr = cute.DType.i8 },
        .shared_l2_sf = .{ .ptr = cute.DType.i8 },
        .sm_locality_domains = .{ .ptr = cute.DType.i8 },
        .y = .{ .ptr = cute.DType.bf16 },
        .workspace_out = .{ .ptr = cute.DType.i8 },
    });
    const status = lib.program.emitLaunch(b, cfg, "megamoe_kernel", .{
        .y = h.y,
        .symmetric = .{ .single = h.workspace_out },
        .weights = .{ h.l1_w, h.l1_sf, h.l2_w, h.l2_sf, h.shared_l1_w, h.shared_l1_sf, h.shared_l2_w, h.shared_l2_sf },
        .sm_locality_domains = h.sm_locality_domains,
        .num_tokens = b.cst(.i32, @as(i32, @intCast(inst.num_tokens))),
    });
    b.returnHostStatus(status);
    b.endFunction(null);
}

fn dumpCall(workspace: zml.Tensor, l1_w: zml.Tensor, l1_sf: zml.Tensor, l2_w: zml.Tensor, l2_sf: zml.Tensor, shared_l1_w: zml.Tensor, shared_l1_sf: zml.Tensor, shared_l2_w: zml.Tensor, shared_l2_sf: zml.Tensor, sm_locality_domains: zml.Tensor, inst: KernelInstance) [2]zml.Tensor {
    const r = DumpProgram.call(.{
        .workspace = workspace,
        .l1_w = l1_w,
        .l1_sf = l1_sf,
        .l2_w = l2_w,
        .l2_sf = l2_sf,
        .shared_l1_w = shared_l1_w,
        .shared_l1_sf = shared_l1_sf,
        .shared_l2_w = shared_l2_w,
        .shared_l2_sf = shared_l2_sf,
        .sm_locality_domains = sm_locality_domains,
    }, .{
        .y = .init(.{ @as(i64, inst.num_tokens), @as(i64, inst.hidden) }, .bf16),
        .workspace_out = workspace.shape(),
    }, .{ .cfg = inst, .output_operand_aliases = .{ .workspace_out = .workspace } });
    return .{ r.y, r.workspace_out };
}

fn jsonInt(obj: std.json.Value, name: []const u8) u32 {
    return @intCast(obj.object.get(name).?.integer);
}

fn jsonF32(obj: std.json.Value, name: []const u8) f32 {
    return switch (obj.object.get(name).?) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

/// `bytes` reinterpreted as `[]T` in a fresh, aligned allocation.
fn copyAs(comptime T: type, allocator: std.mem.Allocator, bytes: []const u8) ![]T {
    std.debug.assert(bytes.len % @sizeOf(T) == 0);
    const out = try allocator.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// Compares `y` (bf16 bits) with DeepGEMM's.
fn compareY(expected: []const u8, actual: []const u8, H: u32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    const e = std.mem.bytesAsSlice(u16, expected);
    const a = std.mem.bytesAsSlice(u16, actual);
    var mismatches: usize = 0;
    var tokens_bad: usize = 0;
    var last_bad_token: usize = std.math.maxInt(usize);
    for (e, a, 0..) |x, y, i| {
        if (x != y) {
            if (mismatches < 8) std.debug.print("y[{d}, {d}]: expected 0x{x:0>4}, got 0x{x:0>4}\n", .{ i / H, i % H, x, y });
            mismatches += 1;
            if (i / H != last_bad_token) {
                tokens_bad += 1;
                last_bad_token = i / H;
            }
        }
    }
    std.debug.print("y: {d} of {d} elements differ ({d} tokens)\n", .{ mismatches, e.len, tokens_bad });
    if (mismatches != 0) return error.TestExpectedEqual;
}

fn runDump(dir: []const u8) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const meta_path = try std.fs.path.join(allocator, &.{ dir, "rank0.json" });
    defer allocator.free(meta_path);
    const meta_bytes = try std.Io.Dir.cwd().readFileAlloc(io, meta_path, allocator, .limited(1 << 24));
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSlice(std.json.Value, allocator, meta_bytes, .{});
    defer meta.deinit();
    const m = meta.value;
    std.debug.assert(jsonInt(m, "num_ranks") == 1);

    const inst: KernelInstance = .{
        .num_ranks = 1,
        .num_experts = jsonInt(m, "num_experts"),
        .num_topk = jsonInt(m, "num_topk"),
        .hidden = jsonInt(m, "hidden"),
        .intermediate_hidden = jsonInt(m, "intermediate_hidden"),
        .num_shared_experts = jsonInt(m, "num_shared_experts"),
        .num_max_tokens_per_rank = jsonInt(m, "num_max_tokens_per_rank_requested"),
        .num_tokens = jsonInt(m, "num_tokens"),
        .num_sms = jsonInt(m, "num_sms"),
        .weight_type = .fp4_unpacked,
        .activation_clamp = jsonF32(m, "activation_clamp"),
        // About 15 s at 2 GHz before a `wait_until` traps.
        .timeout_cycles = 30_000_000_000,
    };
    std.debug.assert(jsonF32(m, "activation_alpha") == 1.0 and jsonF32(m, "activation_beta") == 0.0);
    const cfg = inst.config();
    const s = cfg.kernel.shape;
    std.debug.print("MegaMoE dump {s}: T {d}, H {d}, I {d}, E {d}, topk {d}, shared {d}, BLOCK_M {d} (DeepGEMM {d})\n", .{ dir, inst.num_tokens, inst.hidden, inst.intermediate_hidden, inst.num_experts, inst.num_topk, inst.num_shared_experts, cfg.kernel.cfg.block_m, jsonInt(m, "block_m") });
    try std.testing.expectEqual(jsonInt(m, "block_m"), cfg.kernel.cfg.block_m);
    const symm: L.SymmBufferLayout = .init(s, inst.num_sms);
    const buf = symm.buffer;

    // Kernel inputs, staged into a zeroed symmetric buffer.
    const rank_path = try std.fs.path.join(allocator, &.{ dir, "rank0.safetensors" });
    defer allocator.free(rank_path);
    var rank_file = try Safetensors.open(allocator, io, rank_path);
    defer rank_file.deinit(allocator, io);
    const x = try rank_file.read(allocator, io, "x");
    defer allocator.free(x);
    const x_sf = try rank_file.read(allocator, io, "x_sf");
    defer allocator.free(x_sf);
    const topk_idx = try rank_file.read(allocator, io, "topk_idx");
    defer allocator.free(topk_idx);
    const topk_w = try rank_file.read(allocator, io, "topk_weights");
    defer allocator.free(topk_w);
    const y_ref = try rank_file.read(allocator, io, "y");
    defer allocator.free(y_ref);
    const table = try rank_file.read(allocator, io, "sm_locality_domains");
    defer allocator.free(table);
    try std.testing.expectEqual(@as(usize, inst.num_sms), table.len);

    const ws_host = try allocator.alloc(u8, @intCast(symm.numBytes()));
    defer allocator.free(ws_host);
    @memset(ws_host, 0);
    @memcpy(ws_host[@intCast(buf.input_token.offset)..][0..x.len], x);
    @memcpy(ws_host[@intCast(buf.input_sf.offset)..][0..x_sf.len], x_sf);
    @memcpy(ws_host[@intCast(buf.input_topk_idx.offset)..][0..topk_idx.len], topk_idx);
    @memcpy(ws_host[@intCast(buf.input_topk_weights.offset)..][0..topk_w.len], topk_w);
    if (inst.num_shared_experts > 0) {
        const shared_sf = try rank_file.read(allocator, io, "shared_l1_acts_sf");
        defer allocator.free(shared_sf);
        try std.testing.expectEqual(buf.shared_l1_sf.numBytes(), shared_sf.len);
        @memcpy(ws_host[@intCast(buf.shared_l1_sf.offset)..][0..shared_sf.len], shared_sf);
    }

    // Weights (DeepGEMM's post-transform layout, memory order).
    const shapes = megamoeShapes(inst);
    var inputs: [10]zml.Buffer = undefined;
    inputs[0] = try zml.Buffer.fromBytes(io, platform, shapes[0], ws_host);
    {
        const weights_rel = m.object.get("weights_file").?.string;
        const weights_path = try std.fs.path.join(allocator, &.{ dir, weights_rel });
        defer allocator.free(weights_path);
        var weights_file = try Safetensors.open(allocator, io, weights_path);
        defer weights_file.deinit(allocator, io);
        var shared_file: ?Safetensors = null;
        defer if (shared_file) |*f| f.deinit(allocator, io);
        if (inst.num_shared_experts > 0) {
            const shared_rel = m.object.get("shared_weights_file").?.string;
            const shared_path = try std.fs.path.join(allocator, &.{ dir, shared_rel });
            defer allocator.free(shared_path);
            shared_file = try Safetensors.open(allocator, io, shared_path);
        }
        const names = [8][]const u8{ "l1_w", "l1_sf", "l2_w", "l2_sf", "shared_l1_w", "shared_l1_sf", "shared_l2_w", "shared_l2_sf" };
        for (names, 1..) |name, i| {
            if (i >= 5 and inst.num_shared_experts == 0) {
                inputs[i] = try zml.Buffer.uninitialized(io, platform, shapes[i], .{});
                continue;
            }
            const file = if (i >= 5) &shared_file.? else &weights_file;
            const bytes = try file.read(allocator, io, name);
            defer allocator.free(bytes);
            try std.testing.expectEqual(@as(usize, @intCast(shapes[i].dim(0))), bytes.len);
            inputs[i] = try zml.Buffer.fromBytes(io, platform, shapes[i], bytes);
        }
    }
    inputs[9] = try zml.Buffer.fromBytes(io, platform, .init(.{@as(i64, inst.num_sms)}, .i8), table);
    defer for (inputs[1..]) |*b_| b_.deinit();

    var args: [10]zml.Tensor = undefined;
    for (&args, inputs) |*a, b_| a.* = .fromShape(b_.shape());
    var exe = try platform.compileFn(allocator, io, dumpCall, .{ args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], inst }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    // The dispatch checks of the one-rank run, on the dump's inputs.
    const d: RunData = .{
        .T = inst.num_tokens,
        .H = inst.hidden,
        .K = inst.num_topk,
        .E = inst.num_experts,
        .x = x,
        .x_sf = try copyAs(u32, allocator, x_sf),
        .topk_idx = try copyAs(i64, allocator, topk_idx),
        .topk_w = try copyAs(f32, allocator, topk_w),
    };
    defer {
        allocator.free(d.x_sf);
        allocator.free(d.topk_idx);
        allocator.free(d.topk_w);
    }

    const launches: u32 = if (std.c.getenv("MEGAMOE_DUMP_LAUNCHES")) |n| try std.fmt.parseInt(u32, std.mem.span(n), 10) else 2;
    var ws = inputs[0];
    var failures: usize = 0;
    for (1..launches + 1) |launch| {
        var output: [2]zml.Buffer = undefined;
        runner.run(io, .{ ws, inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9] }, &output, .{ .wait = true });
        ws.deinit();
        ws = output[1];
        const y = try output[0].toSliceAlloc(allocator, io);
        defer y.free(allocator);
        output[0].deinit();
        std.debug.print("launch {d}: ", .{launch});
        compareY(y_ref, y.constData(), inst.hidden) catch {
            failures += 1;
        };
        const got = try ws.toSliceAlloc(allocator, io);
        defer got.free(allocator);
        try checkRun(allocator, inst, d, got.constData(), @intCast(launch));
    }
    // MEGAMOE_DUMP_ITERS=<n>: n more launches (for a profiler's kernel timings).
    if (std.c.getenv("MEGAMOE_DUMP_ITERS")) |iters_str| {
        const iters = try std.fmt.parseInt(u32, std.mem.span(iters_str), 10);
        for (0..iters) |_| {
            var output: [2]zml.Buffer = undefined;
            runner.run(io, .{ ws, inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9] }, &output, .{ .wait = true });
            output[0].deinit();
            ws.deinit();
            ws = output[1];
        }
        std.debug.print("{d} timing launches done\n", .{iters});
    }
    ws.deinit();
    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "MegaMoE single-rank DeepGEMM dumps: y bit for bit" {
    const dirs = std.c.getenv("MEGAMOE_DUMPS") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(dirs), ',');
    var failures: usize = 0;
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        runDump(dir) catch |err| {
            std.debug.print("dump {s}: {s}\n", .{ dir, @errorName(err) });
            failures += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

// ==================== %smid probe ====================

const SmidProgram = cute.Program(void, .{
    .name = "megamoe_smid",
    .inputs = &.{},
    .outputs = &.{"smid"},
    .run = smidProgram,
});

fn smidProgram(b: *B, _: void) cute.FinishError!void {
    b.beginFunction("megamoe_smid_kernel", .cuda_kernel);
    {
        const a = try b.declareArgs(.{ .out = .{ .ptr = cute.DType.i32 } });
        b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}"));
        var first = b.openIf(b.threadIdx().x.eq(0));
        ptx.stGlobal(b, 1, b.ptrToInt(a.out, .i64).add(b.blockIdx().x.to(.i64).mul(4)), .{ptx.getSmIdx(b)});
        first.yieldThen(.{});
    }
    b.endFunction(.{ 32, 1, 1 });
    b.beginFunction(SmidProgram.name, .host);
    const h = try b.declareArgs(.{ .out = .{ .ptr = cute.DType.i32 } });
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, 152), one, one },
        .block = .{ b.cst(.i32, 32), one, one },
        .dynamic_smem = b.cst(.i64, 200 * 1024),
        .stream = b.cudaStream(),
        .cluster = .{ b.cst(.i32, 2), one, one },
    });
    b.returnHostStatus(b.cudaResultStatus(b.launchEx("megamoe_smid_kernel", config, .{h.out})));
    b.endFunction(null);
}

fn smidCall() zml.Tensor {
    return SmidProgram.call(.{}, .{ .smid = .init(.{152}, .i32) }, .{ .cfg = {} }).smid;
}

test "%smid of a 152-CTA cluster launch (sm_locality_domains is indexed by it)" {
    if (std.c.getenv("MEGAMOE_RUN") == null) return error.SkipZigTest;
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var exe = try platform.compileFn(allocator, io, smidCall, .{}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: [1]zml.Buffer = undefined;
    runner.run(io, .{}, &output, .{ .wait = true });
    defer output[0].deinit();
    const got = try output[0].toSliceAlloc(allocator, io);
    defer got.free(allocator);
    const ids = got.constItems(i32);
    var max: i32 = 0;
    for (ids) |id| max = @max(max, id);
    std.debug.print("%smid over 152 CTAs: max {d}\n", .{max});
}

// ==================== the op API with an XLA symmetric workspace (one rank) ====================
//
// `zml.moe.megamoe.forwardStaged` / `forward` with `Workspace.symmetric` on a single-rank dump:
// the Program's host function takes XLA's appended workspace pointer and rank (needs the XLA
// plugin with symmetric workspaces). MEGAMOE_API_DUMPS=<dir>[,...]; each dump runs both input
// kinds, two calls each, `y` bit for bit. (The `Workspace.operand` path is `check.zig`'s.)

const megamoe = zml.moe.megamoe;

fn apiStagedCall(x: zml.Tensor, x_sf: zml.Tensor, topk_idx: zml.Tensor, topk_weights: zml.Tensor, w: [8]zml.Tensor, sm_locality_domains: zml.Tensor, problem: megamoe.Problem) zml.Tensor {
    return megamoe.forwardStaged(problem, .{ .x = x, .x_sf = x_sf, .topk_idx = topk_idx, .topk_weights = topk_weights }, apiWeights(problem, w), sm_locality_domains, .{ .symmetric = .{ .name = "megamoe_gpu_test" } }).y;
}

fn apiBf16Call(x: zml.Tensor, topk_idx: zml.Tensor, topk_weights: zml.Tensor, w: [8]zml.Tensor, sm_locality_domains: zml.Tensor, problem: megamoe.Problem) zml.Tensor {
    return megamoe.forward(problem, x, topk_idx.convert(.i32), topk_weights, apiWeights(problem, w), sm_locality_domains, .{ .symmetric = .{ .name = "megamoe_gpu_test" } }).y;
}

fn apiWeights(problem: megamoe.Problem, w: [8]zml.Tensor) megamoe.Weights {
    const shared = problem.num_shared_experts > 0;
    return .{
        .l1_w = w[0],
        .l1_sf = w[1],
        .l2_w = w[2],
        .l2_sf = w[3],
        .shared_l1_w = if (shared) w[4] else null,
        .shared_l1_sf = if (shared) w[5] else null,
        .shared_l2_w = if (shared) w[6] else null,
        .shared_l2_sf = if (shared) w[7] else null,
    };
}

fn runApiDump(dir: []const u8) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const meta_path = try std.fs.path.join(allocator, &.{ dir, "rank0.json" });
    defer allocator.free(meta_path);
    const meta_bytes = try std.Io.Dir.cwd().readFileAlloc(io, meta_path, allocator, .limited(1 << 24));
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSlice(std.json.Value, allocator, meta_bytes, .{});
    defer meta.deinit();
    const m = meta.value;
    std.debug.assert(jsonInt(m, "num_ranks") == 1);
    const problem: megamoe.Problem = .{
        .num_ranks = 1,
        .num_experts = jsonInt(m, "num_experts"),
        .num_topk = jsonInt(m, "num_topk"),
        .hidden = jsonInt(m, "hidden"),
        .intermediate = jsonInt(m, "intermediate_hidden"),
        .num_shared_experts = jsonInt(m, "num_shared_experts"),
        .num_max_tokens_per_rank = jsonInt(m, "num_max_tokens_per_rank_requested"),
        .activation_clamp = jsonF32(m, "activation_clamp"),
    };
    const T: i64 = jsonInt(m, "num_tokens");
    const H: i64 = problem.hidden;
    const I: i64 = problem.intermediate;
    const K: i64 = problem.num_topk;
    const E: i64 = problem.numExpertsPerRank();
    const sI: i64 = I * problem.num_shared_experts;
    const num_sms: i64 = jsonInt(m, "num_sms");
    std.debug.print("MegaMoE API (symmetric workspace) dump {s}: T {d}\n", .{ dir, T });

    const rank_path = try std.fs.path.join(allocator, &.{ dir, "rank0.safetensors" });
    defer allocator.free(rank_path);
    var rank_file = try Safetensors.open(allocator, io, rank_path);
    defer rank_file.deinit(allocator, io);
    const y_ref = try rank_file.read(allocator, io, "y");
    defer allocator.free(y_ref);

    var bufs: std.ArrayList(zml.Buffer) = .empty;
    defer {
        for (bufs.items) |*b_| b_.deinit();
        bufs.deinit(allocator);
    }
    const load = struct {
        fn f(list: *std.ArrayList(zml.Buffer), file: *const Safetensors, name: []const u8, shape: zml.Shape) !zml.Buffer {
            const a = std.testing.allocator;
            const bytes = try file.read(a, std.testing.io, name);
            defer a.free(bytes);
            try std.testing.expectEqual(shape.byteSize(), bytes.len);
            const buf = try zml.Buffer.fromBytes(std.testing.io, zml.testing.env(), shape, bytes);
            try list.append(a, buf);
            return buf;
        }
    }.f;

    const x = try load(&bufs, &rank_file, "x", .init(.{ T, H }, .u8));
    const x_sf = try load(&bufs, &rank_file, "x_sf", .init(.{ T, @divExact(H, 128) }, .i32));
    const topk_idx = try load(&bufs, &rank_file, "topk_idx", .init(.{ T, K }, .i64));
    const topk_w = try load(&bufs, &rank_file, "topk_weights", .init(.{ T, K }, .f32));
    const x_bf16 = try load(&bufs, &rank_file, "x_bf16", .init(.{ T, H }, .bf16));
    const table = try load(&bufs, &rank_file, "sm_locality_domains", .init(.{num_sms}, .u8));

    var w: [8]zml.Buffer = undefined;
    {
        const weights_path = try std.fs.path.join(allocator, &.{ dir, m.object.get("weights_file").?.string });
        defer allocator.free(weights_path);
        var weights_file = try Safetensors.open(allocator, io, weights_path);
        defer weights_file.deinit(allocator, io);
        w[0] = try load(&bufs, &weights_file, "l1_w", .init(.{ E, 2 * I, @divExact(H, 2) }, .u8));
        w[1] = try load(&bufs, &weights_file, "l1_sf", .init(.{ E, @divExact(H, 128), 2 * I }, .i32));
        w[2] = try load(&bufs, &weights_file, "l2_w", .init(.{ E, H, @divExact(I, 2) }, .u8));
        w[3] = try load(&bufs, &weights_file, "l2_sf", .init(.{ E, @divExact(I, 128), H }, .i32));
        if (problem.num_shared_experts > 0) {
            const shared_path = try std.fs.path.join(allocator, &.{ dir, m.object.get("shared_weights_file").?.string });
            defer allocator.free(shared_path);
            var shared_file = try Safetensors.open(allocator, io, shared_path);
            defer shared_file.deinit(allocator, io);
            w[4] = try load(&bufs, &shared_file, "shared_l1_w", .init(.{ 2 * sI, H }, .u8));
            w[5] = try load(&bufs, &shared_file, "shared_l1_sf", .init(.{ @divExact(H, 128), 2 * sI }, .i32));
            w[6] = try load(&bufs, &shared_file, "shared_l2_w", .init(.{ H, sI }, .u8));
            w[7] = try load(&bufs, &shared_file, "shared_l2_sf", .init(.{ @divExact(sI, 128), H }, .i32));
        } else {
            for (w[4..], w[0..4]) |*s_, r| s_.* = r;
        }
    }
    var wt: [8]zml.Tensor = undefined;
    for (&wt, w) |*t, b_| t.* = .fromShape(b_.shape());

    var failures: usize = 0;
    for ([_]bool{ false, true }) |bf16| {
        var exe = if (bf16)
            try platform.compileFn(allocator, io, apiBf16Call, .{ zml.Tensor.fromShape(x_bf16.shape()), zml.Tensor.fromShape(topk_idx.shape()), zml.Tensor.fromShape(topk_w.shape()), wt, zml.Tensor.fromShape(table.shape()), problem }, .{})
        else
            try platform.compileFn(allocator, io, apiStagedCall, .{ zml.Tensor.fromShape(x.shape()), zml.Tensor.fromShape(x_sf.shape()), zml.Tensor.fromShape(topk_idx.shape()), zml.Tensor.fromShape(topk_w.shape()), wt, zml.Tensor.fromShape(table.shape()), problem }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        for (1..3) |call_idx| {
            var output: [1]zml.Buffer = undefined;
            if (bf16)
                runner.run(io, .{ x_bf16, topk_idx, topk_w, w, table }, &output, .{ .wait = true })
            else
                runner.run(io, .{ x, x_sf, topk_idx, topk_w, w, table }, &output, .{ .wait = true });
            defer output[0].deinit();
            const y = try output[0].toSliceAlloc(allocator, io);
            defer y.free(allocator);
            std.debug.print("{s} call {d}: ", .{ if (bf16) "forward (bf16)" else "forwardStaged", call_idx });
            compareY(y_ref, y.constData(), problem.hidden) catch {
                failures += 1;
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "megamoe API with Workspace.symmetric on one rank: y bit for bit" {
    const dirs = std.c.getenv("MEGAMOE_API_DUMPS") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(dirs), ',');
    var failures: usize = 0;
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        runApiDump(dir) catch |err| {
            std.debug.print("dump {s}: {s}\n", .{ dir, @errorName(err) });
            failures += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}
