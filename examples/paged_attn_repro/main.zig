const std = @import("std");

const zml = @import("zml");
const tri = zml.kernel.triton;

const log = std.log.scoped(.paged_attn_repro);

pub const std_options: std.Options = .{ .log_level = .info };

// Minimal reproducer for the oneAPI / Intel-XPU prefill-NaN bug.
//
// It exercises the SAME Triton `unified_attention` paged-attention kernel that
// llmd's Llama uses (zml.attention.paged_attention with the .triton backend,
// which `Backend.auto` selects for BOTH .cuda and .oneapi). So:
//   - on 4090 (CUDA, known good)   -> finite, sane output  (the ORACLE)
//   - on b70  (oneAPI, broken)     -> NaN/garbage           (the BUG)
//
// A bf16 matmul sanity check runs first to split "is plain bf16 DPAS GEMM
// broken" from "is the attention kernel broken".
//
// Build + run (CUDA oracle, on 4090):
//   bazel run //examples/paged_attn_repro --config=release \
//     --@zml//platforms:cpu=false --@zml//platforms:cuda=true
// Build + run (oneAPI, on b70 — mirror the llmd override flags):
//   bazel run //examples/paged_attn_repro --config=sycl_hermetic \
//     --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
//     --override_repository=zml++oneapi_packages+libpjrt_oneapi=/home/raph/libpjrt_oneapi/libpjrt_oneapi \
//     --override_repository=zml++non_module_deps+xla=/home/raph/Git-Repos/xla_kevin-oneapi-2026 \
//     --override_repository=zml++xla+triton=/home/raph/Git-Repos/intel-xpu-backend-for-triton

// Llama-3.1-8B attention shape, one fresh prefill chunk.
const num_tokens: i64 = 256; // compiled prefill bucket (model pads 42 real tokens to 256)
const real_len: i64 = 42; // ACTUAL sequence length (rest of the 256 query rows are padding)
const num_kv_heads: i64 = 8;
const num_q_groups: i64 = 4; // num_heads / num_kv_heads = 32 / 8
const head_dim: i64 = 128;
const block_size: i64 = 16; // KV-cache page size (.k_chunk); <=512 -> 2D kernel
const num_pages: i64 = 4; // total pages in the cache
const max_pages_per_seq: i64 = 3; // 3*16=48 >= real_len=42

const AttnRepro = struct {
    pub fn forward(
        _: AttnRepro,
        q_f32: zml.Tensor,
        k_cache_f32: zml.Tensor,
        v_cache_f32: zml.Tensor,
        block_table: zml.Tensor,
        seq_lens: zml.Tensor,
        query_start_len: zml.Tensor,
    ) zml.Tensor {
        // The model runs attention in bf16; convert here so the kernel sees bf16.
        const q = q_f32.convert(.bf16);
        const k_cache = k_cache_f32.convert(.bf16);
        const v_cache = v_cache_f32.convert(.bf16);

        // Construct the triton paged-attention parameters directly so we always
        // take the .triton kernel path regardless of platform.
        const parameters: zml.attention.paged_attention.Parameters = .{ .triton = .{
            .block_table = block_table,
            .seq_lens = seq_lens,
            .query_start_len = query_start_len,
            .options_ = .{
                .batch_size = 1,
                .max_num_pages = @intCast(max_pages_per_seq),
                .max_seqlen_q = @intCast(num_tokens),
                .is_prefill = true,
            },
        } };

        // The .triton path ignores the positional k/v args (it reads K/V from
        // the paged cache passed as kv_cache); pass q as a placeholder.
        const out = zml.attention.paged_attention.pagedAttention(
            parameters,
            q,
            q,
            q,
            .{ .split = .{ .k = k_cache, .v = v_cache } },
            .{ .is_causal = true },
        );
        return out.convert(.f32);
    }
};

const MatmulRepro = struct {
    pub fn forward(_: MatmulRepro, a_f32: zml.Tensor, b_f32: zml.Tensor) zml.Tensor {
        const a = a_f32.convert(.bf16);
        const b = b_f32.convert(.bf16);
        return a.dot(b, .k).convert(.f32);
    }
};

// Minimal isolation of the cross-warp reduce (the SLM sub-group-transpose path):
// a bare hand-written Triton kernel that sums a [N] vector to a scalar. Run with
// num_warps=2 so the N-axis reduction spans 2 warps -> within-warp shuffle then
// cross-warp combine through shared memory (the suspected-broken path). With an
// all-ones input the result MUST be N (=64); on the B70 it comes back 0.
const MiniReduce = struct {
    pub const Cfg = struct { N: i32 = 64 };
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "mini_cross_warp_reduce",
        .inputs = &.{ "in_ptr", "out_ptr" },
        .outputs = &.{},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        const a = try b.declareArgs(.{
            .in_ptr = .{ .ptr = .f32 },
            .out_ptr = .{ .ptr = .f32 },
        });
        const offsets = b.arange(0, cfg.N, .i32);
        const mask = offsets.lt(cfg.N);
        const x = b.loadOpts(a.in_ptr.addPtr(offsets), .{ .mask = mask, .other = mask.to(.f32) });
        const total = b.sum(x); // full reduce over [N] -> scalar (cross-warp w/ num_warps=2)
        const out_ptrs = a.out_ptr.addPtr(b.arange(0, 1, .i32));
        const true_mask = b.full(&.{1}, 1, .i1);
        b.storeOpts(out_ptrs, total.splatTo(&.{1}), .{ .mask = true_mask });
    }
};

const MiniReduceMod = struct {
    pub fn forward(_: MiniReduceMod, in: zml.Tensor) zml.Tensor {
        const cc = zml.module.CompilationContext.current();
        const ir = MiniReduce.Kernel.emit(cc.allocator, .{}) catch @panic("emit failed");
        defer cc.allocator.free(ir);
        const out_shape = zml.Shape.init(.{ .o = 1 }, .f32);
        return zml.ops.triton(.{in}, .{out_shape}, .{
            .name = "mini_cross_warp_reduce",
            .ir = ir,
            .grid = .{ 1, 1, 1 },
            .num_warps = 2, // 2 warps -> forces the cross-warp SLM combine (the bug)
            .num_stages = 1,
        })[0];
    }
};

// Minimal isolation of a SCALAR pointer load (`b.load(ptr)`, no offset) — the
// exact op the unified_attention kernel uses to read its runtime strides
// (output_stride_*, query_stride_*), which on the B70 all come back 0. Pass a
// 0-D i64 = 12345; the kernel loads it and stores it. Expect 12345; 0 -> the
// scalar/i64 arg load is broken.
const MiniScalarLoad = struct {
    pub const Cfg = struct {};
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "mini_scalar_load",
        .inputs = &.{
            "s0", "s1", "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
            "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "out_ptr",
        },
        .outputs = &.{},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        _ = cfg;
        const a = try b.declareArgs(.{
            // s0 is a RUNTIME buffer (passed via PJRT execute), s1..s8 are CONSTANTS
            // -- mirrors the attention mixing runtime data buffers + constant strides.
            .s0 = .{ .ptr = .i64 },  .s1 = .{ .ptr = .i64 },  .s2 = .{ .ptr = .i64 },  .s3 = .{ .ptr = .i64 },
            .s4 = .{ .ptr = .i64 },  .s5 = .{ .ptr = .i64 },  .s6 = .{ .ptr = .i64 },  .s7 = .{ .ptr = .i64 },
            .s8 = .{ .ptr = .i64 },  .s9 = .{ .ptr = .i64 },  .s10 = .{ .ptr = .i64 }, .s11 = .{ .ptr = .i64 },
            .s12 = .{ .ptr = .i64 }, .s13 = .{ .ptr = .i64 }, .s14 = .{ .ptr = .i64 }, .s15 = .{ .ptr = .i64 },
            .out_ptr = .{ .ptr = .f32 },
        });
        // Load the CONSTANT args s8..s15 (after a runtime buffer at s0 + constants).
        // value 2^(i-8); sum=255 if the runtime+constant interleave is fine.
        const s = b.load(a.s8).add(b.load(a.s9)).add(b.load(a.s10)).add(b.load(a.s11))
            .add(b.load(a.s12)).add(b.load(a.s13)).add(b.load(a.s14)).add(b.load(a.s15));
        const out_ptrs = a.out_ptr.addPtr(b.arange(0, 1, .i32));
        const true_mask = b.full(&.{1}, 1, .i1);
        b.storeOpts(out_ptrs, s.to(.f32).splatTo(&.{1}), .{ .mask = true_mask });
    }
};
const MiniScalarLoadMod = struct {
    pub fn forward(_: MiniScalarLoadMod, scalar_in: zml.Tensor) zml.Tensor {
        const cc = zml.module.CompilationContext.current();
        const ir = MiniScalarLoad.Kernel.emit(cc.allocator, .{}) catch @panic("emit failed");
        defer cc.allocator.free(ir);
        const out_shape = zml.Shape.init(.{ .o = 1 }, .f32);
        // s0 = RUNTIME buffer (scalar_in), s1..s7 = filler constants, s8..s15 =
        // loaded constants 2^0..2^7. Mirrors runtime data buffers + constant
        // strides. sum=255 if the runtime+constant interleave is fine.
        const f = zml.Tensor.constant(zml.DataType.i64.constant(99));
        return zml.ops.triton(.{
            scalar_in, f, f, f, f, f, f, f,
            zml.Tensor.constant(zml.DataType.i64.constant(1)),
            zml.Tensor.constant(zml.DataType.i64.constant(2)),
            zml.Tensor.constant(zml.DataType.i64.constant(4)),
            zml.Tensor.constant(zml.DataType.i64.constant(8)),
            zml.Tensor.constant(zml.DataType.i64.constant(16)),
            zml.Tensor.constant(zml.DataType.i64.constant(32)),
            zml.Tensor.constant(zml.DataType.i64.constant(64)),
            zml.Tensor.constant(zml.DataType.i64.constant(128)),
        }, .{out_shape}, .{
            .name = "mini_scalar_load",
            .ir = ir,
            .grid = .{ 1, 1, 1 },
            .num_warps = 1,
            .num_stages = 1,
        })[0];
    }
};

// Minimal isolation of the DPAS-reduce that OptimizeReductionLocality rewrites:
// a hand-written kernel doing acc = dot(A[128,128], B[128,64]) then
// m = max(acc, axis=1) -> [128]. The dot yields a #mma (DPAS) tensor with
// repCluster=[4,2] (matching the attention score), so the rowmax reduce hits
// the exact pass + transpose + register-reduce path. Compared per-row to a CPU
// reference: a wrong row -> the reduce miscompiles in isolation.
const DotReduce = struct {
    pub const M: i64 = 128;
    pub const K: i64 = 128;
    pub const N: i64 = 64;
    pub const NITER: i64 = 4;
    pub const BW: i64 = N * NITER; // B is [K, BW]; iter i uses cols [i*N, (i+1)*N)
    pub const Cfg = struct {};
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "dot_reduce_repro",
        .inputs = &.{ "a_ptr", "b_ptr", "m_ptr" },
        .outputs = &.{},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        _ = cfg;
        const a = try b.declareArgs(.{
            .a_ptr = .{ .ptr = .bf16 },
            .b_ptr = .{ .ptr = .bf16 },
            .m_ptr = .{ .ptr = .f32 },
        });
        const om = b.arange(0, M, .i32); // [M]
        const ok = b.arange(0, K, .i32); // [K]
        const on = b.arange(0, N, .i32); // [N]
        const om1 = b.expandDims(om, 1); // [M,1]
        const ok0 = b.expandDims(ok, 0); // [1,K]
        const ok1 = b.expandDims(ok, 1); // [K,1]
        const on0 = b.expandDims(on, 0); // [1,N]

        _ = NITER;
        // A[M,K] = a_ptr + m*K + k
        const ap = b.broadcast(om1.mul(K), ok0);
        const a_tile = b.load(a.a_ptr.addPtr(ap[0].add(ap[1])));
        // B[K,N] = b_ptr + k*BW + n (first N cols of the BW-wide buffer)
        const bp = b.broadcast(ok1.mul(BW), on0);
        const b_tile = b.load(a.b_ptr.addPtr(bp[0].add(bp[1])));

        const acc = b.dot(a_tile, b_tile, b.full(&.{ M, N }, 0.0, .f32)); // [M,N] f32
        // SUM reduce (AddF) over axis 1 -> [M]. This is the VECTORIZED reduce
        // path (packAlongAxis in reduceWithinThreads), unlike the non-vectorized
        // max. The attention's softmax denominator uses this path.
        const s = b.sumOpts(acc, .{ .axis = 1 }); // [M]
        b.store(a.m_ptr.addPtr(om), s);
    }
};

const DotReduceMod = struct {
    pub fn forward(_: DotReduceMod, a_f32: zml.Tensor, b_f32: zml.Tensor) zml.Tensor {
        const cc = zml.module.CompilationContext.current();
        const ir = DotReduce.Kernel.emit(cc.allocator, .{}) catch @panic("emit failed");
        defer cc.allocator.free(ir);
        const a = a_f32.convert(.bf16);
        const b = b_f32.convert(.bf16);
        const m_shape = zml.Shape.init(.{ .m = DotReduce.M }, .f32);
        return zml.ops.triton(.{ a, b }, .{m_shape}, .{
            .name = "dot_reduce_repro",
            .ir = ir,
            .grid = .{ 1, 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
        })[0];
    }
};

/// Allocate a host f32 buffer of `shape`, filled with bounded varied values
/// (base + step*(i%23)). step=0 gives a constant `base`.
/// Upload an explicit f32 host array as a device buffer of `shape`.
fn bufFromData(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, data: []const f32) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    @memcpy(slice.items(f32), data);
    return zml.Buffer.fromSlice(io, platform, slice, .replicated);
}

fn f32Buf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, base: f32, step: f32) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    for (slice.items(f32), 0..) |*e, i| {
        e.* = base + step * @as(f32, @floatFromInt(i % 23));
    }
    return zml.Buffer.fromSlice(io, platform, slice, .replicated);
}

/// Allocate a host i32 buffer of `shape`, filled by cycling through `values`.
fn i32Buf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, values: []const i32) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    for (slice.items(i32), 0..) |*e, i| {
        e.* = values[i % values.len];
    }
    return zml.Buffer.fromSlice(io, platform, slice, .replicated);
}

fn i64Buf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, values: []const i64) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    for (slice.items(i64), 0..) |*e, i| {
        e.* = values[i % values.len];
    }
    return zml.Buffer.fromSlice(io, platform, slice, .replicated);
}

/// Round an f32 to bf16 precision (round-to-nearest-even) and back to f32, so
/// the CPU reference matches the kernel's bf16 dot inputs.
fn toBf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const rounded: u32 = (bits +% 0x7FFF +% ((bits >> 16) & 1)) & 0xFFFF0000;
    return @bitCast(rounded);
}

fn report(label: []const u8, items: []const f32, expect: ?f32) void {
    var nan: usize = 0;
    var inf: usize = 0;
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    var mismatch: usize = 0;
    for (items) |v| {
        if (std.math.isNan(v)) {
            nan += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf += 1;
            continue;
        }
        mn = @min(mn, v);
        mx = @max(mx, v);
        if (expect) |e| {
            if (@abs(v - e) > 0.5) mismatch += 1;
        }
    }
    const preview = items[0..@min(items.len, 12)];
    log.info("[{s}] n={d} nan={d} inf={d} min={d:.4} max={d:.4} mismatch={d} preview={any}", .{ label, items.len, nan, inf, mn, mx, mismatch, preview });
    // [oneAPI-probe] sample one element per token row (stride = hkv*hg*hd) so we can
    // read a per-row probe value (e.g. the causal bound rhs) for tokens 0..47.
    {
        const rs: usize = @intCast(num_kv_heads * num_q_groups * head_dim);
        if (items.len >= rs * 4) {
            var samples: [48]f32 = undefined;
            var ns: usize = 0;
            while (ns < 48 and (ns + 1) * rs <= items.len) : (ns += 1) samples[ns] = items[ns * rs];
            log.info("[{s}] per-token row[t]: {any}", .{ label, samples[0..ns] });
        }
    }
    if (nan > 0 or inf > 0) {
        // [oneAPI-probe] print the distinct token rows (stride = hkv*hg*hd) that hold
        // non-finite values, to reveal the pattern of fully-masked (L=0) query rows.
        const row_stride: usize = @intCast(num_kv_heads * num_q_groups * head_dim);
        var rows_buf: [40]usize = undefined;
        var nrows: usize = 0;
        var last_row: usize = std.math.maxInt(usize);
        for (items, 0..) |v, i| {
            if (std.math.isNan(v) or std.math.isInf(v)) {
                const row = i / row_stride;
                if (row != last_row and nrows < rows_buf.len) {
                    rows_buf[nrows] = row;
                    nrows += 1;
                    last_row = row;
                }
            }
        }
        log.err("[{s}] non-finite token rows (first {d}): {any}", .{ label, nrows, rows_buf[0..nrows] });
        log.err("[{s}] *** NON-FINITE OUTPUT: {d} NaN, {d} Inf -- bug reproduced ***", .{ label, nan, inf });
    } else if (expect != null and mismatch > 0) {
        log.err("[{s}] *** {d} values differ from expected -- bug reproduced ***", .{ label, mismatch });
    } else {
        log.info("[{s}] OK (finite{s})", .{ label, if (expect != null) ", matches expected" else "" });
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform target: {s}", .{@tagName(platform.target)});

    // ---- 1) bf16 matmul (DPAS) sanity: ones[m,k] . ones[k,n] => all == k ----
    {
        const m: i64 = 64;
        const k: i64 = 128;
        const n: i64 = 64;
        const a_shape = zml.Shape.init(.{ .m = m, .k = k }, .f32);
        const b_shape = zml.Shape.init(.{ .k = k, .n = n }, .f32);

        var exe = try platform.compile(allocator, io, MatmulRepro{}, .forward, .{ zml.Tensor.fromShape(a_shape), zml.Tensor.fromShape(b_shape) }, .{});
        defer exe.deinit();

        var a_buf = try f32Buf(allocator, io, platform, a_shape, 1.0, 0.0);
        defer a_buf.deinit();
        var b_buf = try f32Buf(allocator, io, platform, b_shape, 1.0, 0.0);
        defer b_buf.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ a_buf, b_buf });
        exe.call(args, &results);
        var out = results.get(zml.Buffer);
        defer out.deinit();

        const s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);
        report("bf16 matmul ones.ones (expect all 128)", s.items(f32), 128.0);
    }

    // ---- 1b) minimal cross-warp reduce — isolates the SLM sub-group-transpose ----
    {
        const n: i64 = 64;
        const in_shape = zml.Shape.init(.{ .n = n }, .f32);

        var exe = try platform.compile(allocator, io, MiniReduceMod{}, .forward, .{zml.Tensor.fromShape(in_shape)}, .{});
        defer exe.deinit();

        var in_buf = try f32Buf(allocator, io, platform, in_shape, 1.0, 0.0); // all ones
        defer in_buf.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{in_buf});
        exe.call(args, &results);
        var out = results.get(zml.Buffer);
        defer out.deinit();

        const s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);
        report("mini cross-warp reduce sum(ones[64]) (expect 64)", s.items(f32), 64.0);
    }

    // ---- 1c) minimal scalar load — isolates the runtime-stride load (b.load) ----
    {
        const scalar_shape = zml.Shape.init(.{}, .i64);
        var exe = try platform.compile(allocator, io, MiniScalarLoadMod{}, .forward, .{zml.Tensor.fromShape(scalar_shape)}, .{});
        defer exe.deinit();

        var scalar_buf = try i64Buf(allocator, io, platform, scalar_shape, &.{12345});
        defer scalar_buf.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{scalar_buf});
        exe.call(args, &results);
        var out = results.get(zml.Buffer);
        defer out.deinit();

        const s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);
        report("mini scalar load runtime s0 + const s8..15 (expect 255)", s.items(f32), 255.0);
    }

    // ---- 1d) minimal DPAS dot+reduce — isolates the OptimizeReductionLocality reduce ----
    {
        const aM = DotReduce.M;
        const aK = DotReduce.K;
        const aN = DotReduce.N;
        const aBW = DotReduce.BW;
        const aNITER = DotReduce.NITER;
        const a_shape = zml.Shape.init(.{ .m = aM, .k = aK }, .f32);
        const b_shape = zml.Shape.init(.{ .k = aK, .n = aBW }, .f32);

        var exe = try platform.compile(allocator, io, DotReduceMod{}, .forward, .{ zml.Tensor.fromShape(a_shape), zml.Tensor.fromShape(b_shape) }, .{});
        defer exe.deinit();

        // EXACT integer data, distinct per row: A[m,k] = m, B = 1.
        // score[m,n] = sum_k m = K*m; rowsum = N*K*m = 8192*m (exact in f32, no
        // bf16 rounding since m<=127 and 1 are exact in bf16). A row-mixing /
        // wrong-set reduce bug -> grossly wrong (8192*other_row); precision ->
        // exact. Distinct per row so wrong-set is detectable (unlike all-ones).
        const a_host = try allocator.alloc(f32, @intCast(aM * aK));
        defer allocator.free(a_host);
        const b_host = try allocator.alloc(f32, @intCast(aK * aBW));
        defer allocator.free(b_host);
        const kKf: usize = @intCast(aK);
        for (a_host, 0..) |*e, i| e.* = @floatFromInt(i / kKf); // a[m,k] = m
        for (b_host) |*e| e.* = 1.0;

        var a_buf = try bufFromData(allocator, io, platform, a_shape, a_host);
        defer a_buf.deinit();
        var b_buf = try bufFromData(allocator, io, platform, b_shape, b_host);
        defer b_buf.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ a_buf, b_buf });
        exe.call(args, &results);
        var out = results.get(zml.Buffer);
        defer out.deinit();
        const s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);

        // CPU reference: A[m,k]=m, B=1 -> rowsum = N*K*m = 8192*m (exact).
        _ = aNITER;
        const kK: usize = @intCast(aK);
        const kN: usize = @intCast(aN);
        var ref: [128]f32 = undefined;
        for (0..@intCast(aM)) |m| {
            ref[m] = @floatFromInt(m * kN * kK); // 8192*m
        }

        const items = s.items(f32);
        var mismatch: usize = 0;
        var first_bad: [16]usize = undefined;
        var nbad: usize = 0;
        for (items, 0..) |v, i| {
            const tol = 0.05 * @abs(ref[i]) + 0.5; // relative: sum accumulates f32 rounding
            if (std.math.isNan(v) or std.math.isInf(v) or @abs(v - ref[i]) > tol) {
                mismatch += 1;
                if (nbad < first_bad.len) {
                    first_bad[nbad] = i;
                    nbad += 1;
                }
            }
        }
        if (mismatch == 0) {
            log.info("[dot+reduce rowmax] OK ({d} rows match CPU ref)", .{items.len});
        } else {
            log.err("[dot+reduce rowmax] *** {d} rows differ from CPU ref; first bad rows: {any} ***", .{ mismatch, first_bad[0..nbad] });
            log.err("[dot+reduce rowmax] kernel[0..16]={any}", .{items[0..@min(items.len, 16)]});
            log.err("[dot+reduce rowmax] ref[0..16]   ={any}", .{ref[0..@min(items.len, 16)]});
        }
    }

    // ---- 2) paged attention (the unified_attention Triton kernel) ----
    {
        const q_shape = zml.Shape.init(.{ .b = num_tokens, .hkv = num_kv_heads, .hg = num_q_groups, .hd = head_dim }, .f32);
        const kv_shape = zml.Shape.init(.{ .page = num_pages, .k_chunk = block_size, .hkv = num_kv_heads, .hd = head_dim }, .f32);
        const block_table_shape = zml.Shape.init(.{ .b = 1, .p = max_pages_per_seq }, .i32);
        const seq_lens_shape = zml.Shape.init(.{ .b = 1 }, .i32);
        const query_start_len_shape = zml.Shape.init(.{ .b = 2 }, .i32);

        var exe = try platform.compile(allocator, io, AttnRepro{}, .forward, .{
            zml.Tensor.fromShape(q_shape),
            zml.Tensor.fromShape(kv_shape),
            zml.Tensor.fromShape(kv_shape),
            zml.Tensor.fromShape(block_table_shape),
            zml.Tensor.fromShape(seq_lens_shape),
            zml.Tensor.fromShape(query_start_len_shape),
        }, .{});
        defer exe.deinit();

        // Bounded, varied, finite inputs — identical on every backend.
        var q_buf = try f32Buf(allocator, io, platform, q_shape, -0.1, 0.01);
        defer q_buf.deinit();
        var k_buf = try f32Buf(allocator, io, platform, kv_shape, -0.08, 0.008);
        defer k_buf.deinit();
        var v_buf = try f32Buf(allocator, io, platform, kv_shape, 0.0, 0.006); // V ramp (matches CUDA oracle ramp ~0..0.13)
        defer v_buf.deinit();
        // seq 0 -> page 0; context+query length 16; cumulative query offsets [0,16].
        var block_table_buf = try i32Buf(allocator, io, platform, block_table_shape, &.{ 0, 1, 2 });
        defer block_table_buf.deinit();
        var seq_lens_buf = try i32Buf(allocator, io, platform, seq_lens_shape, &.{@intCast(real_len)});
        defer seq_lens_buf.deinit();
        var query_start_len_buf = try i32Buf(allocator, io, platform, query_start_len_shape, &.{ 0, @intCast(real_len) });
        defer query_start_len_buf.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ q_buf, k_buf, v_buf, block_table_buf, seq_lens_buf, query_start_len_buf });
        exe.call(args, &results);
        var out = results.get(zml.Buffer);
        defer out.deinit();

        const s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);
        report("paged_attention prefill output", s.items(f32), null);
    }
}
