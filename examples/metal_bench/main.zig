//! The Metal microbenchmark harness — sibling to examples/metal_ops. Where
//! metal_ops checks correctness vs the CPU oracle, this measures throughput of
//! what the AIR emitter can run today (elementwise / fusion / transpose /
//! reduce) over largish tensors, to find hot spots. It grows one bench at a
//! time as the emitter gains coverage, exactly like the ops suite.
//!
//! These kernels are memory-bound, so the headline metric is **effective
//! bandwidth** (GB/s = bytes moved / wall time). On an M-series chip compare it
//! against the unified-memory peak (M4 Max ≈ 410 GB/s) to gauge efficiency.
//! Per-call latency (ms) is also reported. NOTE: matmul/conv aren't emitted yet,
//! so this is per-kernel throughput, not end-to-end model latency.
//!
//! Each run appends one JSON line to a history file (default in metal-xla-docs)
//! so we keep a record of how performance moves over time. Pass --label to tag
//! a run (e.g. the xla plugin commit).
//!
//! Run:  bazel run //examples/metal_bench --//platforms:metal=true -- --iters=100
//! CPU reference (the correctness oracle, for a sanity contrast):
//!      bazel run //examples/metal_bench --//platforms:metal=true -- --device=cpu

const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

// --- the ops under bench (a representative slice, one per kernel path) ----
fn add(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.add(b);
}
fn div(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.div(b);
}
fn fma3(a: zml.Tensor, b: zml.Tensor, c: zml.Tensor) zml.Tensor {
    return a.add(b).mul(c); // (a+b)*c — one fused kernel (E5)
}
fn exp_(a: zml.Tensor) zml.Tensor {
    return a.exp();
}
fn tanh_(a: zml.Tensor) zml.Tensor {
    return a.tanh();
}
fn transpose2d(a: zml.Tensor) zml.Tensor {
    return a.transpose(.{ .j, .i }); // [i,j] -> [j,i], indexed-copy kernel (E3)
}
fn sumCols(a: zml.Tensor) zml.Tensor {
    return a.sum(.j); // [i,j] -> [i], serial-per-thread reduction (E4)
}
fn sumAll(a: zml.Tensor) zml.Tensor {
    return a.sum(.n); // [n] -> scalar, num_out=1 (small-output reduction stress)
}
fn affine(x: zml.Tensor, scale: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    const xs = x.shape();
    return x.mul(scale.broad(xs)).add(bias.broad(xs)); // x*scale+bias, [d]->[r,d] bcast in fusion (E5.2)
}
fn matmul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k); // [m,k]·[k,n] -> [m,n], naive one-thread-per-output kernel
}

const BenchResult = struct {
    name: []const u8,
    elems: usize, // work-proportional element count (output, or input for reduce)
    bytes: usize, // bytes moved per call (the bandwidth numerator)
    iters: usize,
    avg_ns: u64, // mean per-call latency
    gbps: f64, // effective bandwidth = bytes*iters / total_ns
};

fn makeResult(name: []const u8, elems: usize, bytes: usize, iters: usize, total_ns: u64) BenchResult {
    return .{
        .name = name,
        .elems = elems,
        .bytes = bytes,
        .iters = iters,
        .avg_ns = total_ns / @as(u64, @intCast(iters)),
        // bytes/ns == GB/s (the 1e9 factors cancel).
        .gbps = @as(f64, @floatFromInt(bytes * iters)) / @as(f64, @floatFromInt(total_ns)),
    };
}

/// Fill with varied strictly-positive values (valid for div / log / sqrt).
fn fillPositive(s: []f32) void {
    for (s, 0..) |*e, i| e.* = 0.5 + @as(f32, @floatFromInt(i % 17)) * 0.0625;
}

/// warmup (untimed) then `iters` call+await pairs; returns total measured ns.
/// We await every call so each iteration is one full kernel launch+finish — the
/// honest per-kernel cost, no cross-call overlap.
fn timeCalls(io: std.Io, exe: anytype, exe_args: anytype, exe_results: anytype, iters: usize, warmup: usize) !u64 {
    var w: usize = 0;
    while (w < warmup) : (w += 1) {
        exe.call(exe_args, exe_results);
        var r = exe_results.get(zml.Buffer);
        _ = try r.await(io);
        r.deinit();
    }
    const start: std.Io.Timestamp = .now(io, .awake);
    var k: usize = 0;
    while (k < iters) : (k += 1) {
        exe.call(exe_args, exe_results);
        var r = exe_results.get(zml.Buffer);
        _ = try r.await(io);
        r.deinit();
    }
    const elapsed = start.untilNow(io, .awake);
    return @intCast(elapsed.toNanoseconds());
}

fn benchUnary(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    n: usize,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const shape = zml.Shape.init(.{ .n = n }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    const a_data = try allocator.alloc(f32, n);
    defer allocator.free(a_data);
    fillPositive(a_data);
    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, n, 2 * n * @sizeOf(f32), iters, total_ns); // 1 in + 1 out
}

fn benchBinary(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    n: usize,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const shape = zml.Shape.init(.{ .n = n }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t }, .{});
    defer exe.deinit();

    const a_data = try allocator.alloc(f32, n);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, n);
    defer allocator.free(b_data);
    fillPositive(a_data);
    fillPositive(b_data);
    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf });

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, n, 3 * n * @sizeOf(f32), iters, total_ns); // 2 in + 1 out
}

fn benchTernary(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    n: usize,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const shape = zml.Shape.init(.{ .n = n }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    const c_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t, c_t }, .{});
    defer exe.deinit();

    const a_data = try allocator.alloc(f32, n);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, n);
    defer allocator.free(b_data);
    const c_data = try allocator.alloc(f32, n);
    defer allocator.free(c_data);
    fillPositive(a_data);
    fillPositive(b_data);
    fillPositive(c_data);
    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();
    var c_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(c_data));
    defer c_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf, c_buf });

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, n, 4 * n * @sizeOf(f32), iters, total_ns); // 3 in + 1 out
}

/// Affine `x*scale + bias` over x=[rows,cols] with scale/bias=[cols] broadcast
/// inside the fusion. Bytes ≈ read x + write out (the [cols] vectors are tiny).
fn benchAffine(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    rows: usize,
    cols: usize,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const xs = zml.Shape.init(.{ .b = rows, .d = cols }, .f32);
    const ds = zml.Shape.init(.{ .d = cols }, .f32);
    const xt: zml.Tensor = .fromShape(xs);
    const st: zml.Tensor = .fromShape(ds);
    const bt: zml.Tensor = .fromShape(ds);
    var exe = try platform.compileFn(allocator, io, affine, .{ xt, st, bt }, .{});
    defer exe.deinit();

    const x_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(x_data);
    const d_data = try allocator.alloc(f32, cols);
    defer allocator.free(d_data);
    fillPositive(x_data);
    fillPositive(d_data);
    var xb = try zml.Buffer.fromBytes(io, platform, xs, .replicated, std.mem.sliceAsBytes(x_data));
    defer xb.deinit();
    var sb = try zml.Buffer.fromBytes(io, platform, ds, .replicated, std.mem.sliceAsBytes(d_data));
    defer sb.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, ds, .replicated, std.mem.sliceAsBytes(d_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ xb, sb, bb });

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, rows * cols, 2 * rows * cols * @sizeOf(f32), iters, total_ns);
}

/// A unary shape-transform (transpose / reduce): caller supplies the input
/// shape, the element count, and the bytes-moved model (they differ per op).
fn benchShaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_elems: usize,
    bytes: usize,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const a_t: zml.Tensor = .fromShape(in_shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    const a_data = try allocator.alloc(f32, in_elems);
    defer allocator.free(a_data);
    fillPositive(a_data);
    var a_buf = try zml.Buffer.fromBytes(io, platform, in_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, in_elems, bytes, iters, total_ns);
}

/// Fill `buf` (raw bytes for `dt`) with varied strictly-positive values. Manual
/// byte writes keep it alignment-safe for any dtype (no bytesAsSlice).
fn fillDtypeBytes(buf: []u8, dt: zml.DataType) void {
    const sz: usize = dt.sizeOf();
    var i: usize = 0;
    while (i * sz < buf.len) : (i += 1) {
        const x: f32 = 0.5 + @as(f32, @floatFromInt(i % 17)) * 0.0625;
        switch (dt) {
            .f32 => {
                const b: [4]u8 = @bitCast(x);
                @memcpy(buf[i * 4 ..][0..4], &b);
            },
            .f16 => {
                const b: [2]u8 = @bitCast(@as(f16, @floatCast(x)));
                @memcpy(buf[i * 2 ..][0..2], &b);
            },
            .bf16 => {
                const b: [2]u8 = @bitCast(zml.floats.BFloat16.fromF32(x));
                @memcpy(buf[i * 2 ..][0..2], &b);
            },
            else => unreachable,
        }
    }
}

/// Matmul a:[m,k]·op(b) -> [m,n], dtype-generic. NN feeds b as [k,n]; NT (the
/// y=x·Wᵀ linear) feeds b as [n,k] — `a.dot(b,.k)` contracts by tag either way.
/// Compute-bound, so we store the FLOP count (2·M·N·K) in `bytes`; the shared
/// `gbps` formula reads out as **GFLOP/s** for these rows.
fn benchMatmulImpl(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    m: usize,
    k: usize,
    n: usize,
    dt: zml.DataType,
    nt: bool,
    iters: usize,
    warmup: usize,
) !BenchResult {
    const sz: usize = dt.sizeOf();
    const a_shape = zml.Shape.init(.{ .m = m, .k = k }, dt);
    const b_shape = if (nt) zml.Shape.init(.{ .n = n, .k = k }, dt) else zml.Shape.init(.{ .k = k, .n = n }, dt);
    const a_t: zml.Tensor = .fromShape(a_shape);
    const b_t: zml.Tensor = .fromShape(b_shape);
    var exe = try platform.compileFn(allocator, io, matmul, .{ a_t, b_t }, .{});
    defer exe.deinit();

    const a_data = try allocator.alloc(u8, m * k * sz);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(u8, k * n * sz);
    defer allocator.free(b_data);
    fillDtypeBytes(a_data, dt);
    fillDtypeBytes(b_data, dt);
    var a_buf = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, a_data);
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, b_data);
    defer b_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf });

    const total_ns = try timeCalls(io, &exe, exe_args, &exe_results, iters, warmup);
    return makeResult(name, m * n, 2 * m * k * n, iters, total_ns);
}

/// As benchMatmulImpl, but a backend that can't compile/run this shape (e.g.
/// metalBLAS routing a gemv/M==1 to an unwired kernel → loud Unimplemented)
/// yields a sentinel row (gbps=0, iters=0) instead of aborting the whole run.
fn benchMatmulG(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    name: []const u8,
    m: usize,
    k: usize,
    n: usize,
    dt: zml.DataType,
    nt: bool,
    iters: usize,
    warmup: usize,
) BenchResult {
    return benchMatmulImpl(allocator, io, platform, name, m, k, n, dt, nt, iters, warmup) catch |e| {
        log.warn("matmul {s}: unsupported on this backend ({s})", .{ name, @errorName(e) });
        return .{ .name = name, .elems = m * n, .bytes = 2 * m * k * n, .iters = 0, .avg_ns = 0, .gbps = 0 };
    };
}

fn printTable(rows: []const BenchResult) void {
    log.info("{s:<13} {s:>12} {s:>11} {s:>10} {s:>9}", .{ "op", "elems", "bytes/call", "avg", "GB/s" });
    for (rows) |r| {
        const ms = @as(f64, @floatFromInt(r.avg_ns)) / 1.0e6;
        const mb = @as(f64, @floatFromInt(r.bytes)) / (1024.0 * 1024.0);
        log.info("{s:<13} {d:>12} {d:>8.1} MB {d:>7.3} ms {d:>9.1}", .{ r.name, r.elems, mb, ms, r.gbps });
    }
}

/// Append one JSON line (one run) to the history file. Append-only so the file
/// is a chronological record; one object per run with a results[] array.
fn appendHistory(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    label: []const u8,
    platform_name: []const u8,
    size: usize,
    rows: []const BenchResult,
) !void {
    // Read existing history (absent on the first run → start empty). The file
    // is small, so read-modify-write keeps it simple (no seek/append dance).
    var prior: []u8 = &.{};
    var prior_owned = false;
    if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited)) |buf| {
        prior = buf;
        prior_owned = true;
    } else |e| switch (e) {
        error.FileNotFound => {},
        else => return e,
    }
    defer if (prior_owned) allocator.free(prior);

    var results_json: std.ArrayList(u8) = .empty;
    defer results_json.deinit(allocator);
    for (rows, 0..) |r, i| {
        const sep = if (i == 0) "" else ",";
        const piece = try std.fmt.allocPrint(
            allocator,
            "{s}{{\"name\":\"{s}\",\"elems\":{d},\"bytes\":{d},\"iters\":{d},\"avg_ns\":{d},\"gbps\":{d:.2}}}",
            .{ sep, r.name, r.elems, r.bytes, r.iters, r.avg_ns, r.gbps },
        );
        defer allocator.free(piece);
        try results_json.appendSlice(allocator, piece);
    }
    const line = try std.fmt.allocPrint(
        allocator,
        "{{\"ts\":{d},\"label\":\"{s}\",\"platform\":\"{s}\",\"size\":{d},\"results\":[{s}]}}\n",
        .{ std.Io.Timestamp.now(io, .real).toSeconds(), label, platform_name, size, results_json.items },
    );
    defer allocator.free(line);

    const full = try std.mem.concat(allocator, u8, &.{ prior, line });
    defer allocator.free(full);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = full });
    log.info("📈 appended 1 run to {s}", .{path});
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ metal_bench --device=metal|cpu --size=4194304 --iters=100 --warmup=10 --rows=4096 --cols=4096 --mm=1024 --label=<tag> --out=<abs path>
        ;
        device: []const u8 = "metal", // "metal" (default) or "cpu" (oracle, for a contrast)
        size: usize = 1 << 22, // elementwise length (4_194_304 f32 = 16 MB/buffer)
        iters: usize = 100,
        warmup: usize = 10,
        rows: usize = 4096, // 2D shape for transpose / reduce
        cols: usize = 4096,
        mm: usize = 1024, // square matmul size (M=N=K)
        label: []const u8 = "", // free tag, e.g. the xla plugin commit
        out: []const u8 = "/Users/raph/Documents/Git-Repos/metal-xla-docs/bench/history.jsonl",
    };

    const allocator = init.gpa;
    const io = init.io;
    const cli: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const on_cpu = std.mem.eql(u8, cli.device, "cpu");
    const platform: *zml.Platform = if (on_cpu)
        try .init(allocator, io, .cpu, .{})
    else
        try .init(allocator, io, .metal, .{});
    defer platform.deinit(allocator, io);
    log.info("microbench on '{s}': {f}", .{ cli.device, platform.fmtVerbose() });
    log.info(
        "size={d} elems ({d:.1} MB/f32 buf)  2D={d}x{d}  iters={d}  warmup={d}",
        .{ cli.size, @as(f64, @floatFromInt(cli.size * @sizeOf(f32))) / (1024.0 * 1024.0), cli.rows, cli.cols, cli.iters, cli.warmup },
    );

    var results: std.ArrayList(BenchResult) = .empty;
    defer results.deinit(allocator);

    const n = cli.size;
    try results.append(allocator, try benchBinary(allocator, io, platform, "add", add, n, cli.iters, cli.warmup));
    try results.append(allocator, try benchBinary(allocator, io, platform, "div", div, n, cli.iters, cli.warmup));
    try results.append(allocator, try benchTernary(allocator, io, platform, "fma(a+b)*c", fma3, n, cli.iters, cli.warmup));
    try results.append(allocator, try benchAffine(allocator, io, platform, "affine_bcast", cli.rows, cli.cols, cli.iters, cli.warmup));
    try results.append(allocator, try benchUnary(allocator, io, platform, "exp", exp_, n, cli.iters, cli.warmup));
    try results.append(allocator, try benchUnary(allocator, io, platform, "tanh", tanh_, n, cli.iters, cli.warmup));

    const rc = cli.rows * cli.cols;
    const shape2d = zml.Shape.init(.{ .i = cli.rows, .j = cli.cols }, .f32);
    // transpose moves the whole tensor twice (read + write).
    try results.append(allocator, try benchShaped(allocator, io, platform, "transpose", transpose2d, shape2d, rc, 2 * rc * @sizeOf(f32), cli.iters, cli.warmup));
    // sum_cols reads the whole tensor, writes one element per row.
    try results.append(allocator, try benchShaped(allocator, io, platform, "sum_cols", sumCols, shape2d, rc, (rc + cli.rows) * @sizeOf(f32), cli.iters, cli.warmup));
    // sum_all reduces the whole vector to one scalar (num_out=1): the
    // small-output reduction the tree reduction targets (serial-per-thread runs
    // this on a single thread).
    {
        const vshape = zml.Shape.init(.{ .n = n }, .f32);
        try results.append(allocator, try benchShaped(allocator, io, platform, "sum_all", sumAll, vshape, n, (n + 1) * @sizeOf(f32), cli.iters, cli.warmup));
    }

    // Matmul. gbps column reads as GFLOP/s (bytes = 2·M·N·K); backend = XLA_METAL_MATMUL.
    // Backend-AWARE: a backend that can't do a shape returns a loud Unimplemented, which ZML's
    // async compile surfaces as a PANIC (not a catchable error), so we must not submit it. Gates:
    //   bf16  → MPSGraph (default) + metalBLAS only (legacy MPSMatrix aborts on bf16; naive=f32).
    //   M==1  → any-shape backends (MPSGraph/MPSMatrix/naive); metalBLAS routes M=1 to its unwired
    //           gemv kernel, so it's skipped there. Square NN runs on every backend (→ m5_tensor).
    const be: []const u8 = blk: {
        const v = std.c.getenv("XLA_METAL_MATMUL") orelse break :blk "";
        break :blk std.mem.span(v);
    };
    const is_mm = std.mem.eql(u8, be, "mpsmatrix");
    const is_mb = std.mem.eql(u8, be, "metalblas");
    const is_naive = std.mem.eql(u8, be, "naive");
    const is_graph = !is_mm and !is_mb and !is_naive; // default = MPSGraph
    const can_bf16 = is_graph or is_mb;
    const m1_ok = is_graph or is_mm or is_naive; // M==1 gemv (metalBLAS gemv unwired)
    const it = cli.iters;
    try results.append(allocator, benchMatmulG(allocator, io, platform, "mm512_f32", 512, 512, 512, .f32, false, it, cli.warmup));
    try results.append(allocator, benchMatmulG(allocator, io, platform, "mm2048_f32", 2048, 2048, 2048, .f32, false, it, cli.warmup));
    if (can_bf16) try results.append(allocator, benchMatmulG(allocator, io, platform, "mm512_bf16", 512, 512, 512, .bf16, false, it, cli.warmup));
    if (can_bf16) try results.append(allocator, benchMatmulG(allocator, io, platform, "mm2048_bf16", 2048, 2048, 2048, .bf16, false, it, cli.warmup));
    if (m1_ok) try results.append(allocator, benchMatmulG(allocator, io, platform, "dec_qkv_f32", 1, 4096, 4096, .f32, false, it, cli.warmup));
    if (m1_ok and can_bf16) try results.append(allocator, benchMatmulG(allocator, io, platform, "dec_qkv_bf16", 1, 4096, 4096, .bf16, false, it, cli.warmup));
    if (m1_ok and can_bf16) try results.append(allocator, benchMatmulG(allocator, io, platform, "dec_up_bf16", 1, 4096, 11008, .bf16, false, it, cli.warmup));
    if (m1_ok and can_bf16) try results.append(allocator, benchMatmulG(allocator, io, platform, "dec_down_bf16", 1, 11008, 4096, .bf16, false, it, cli.warmup));

    log.info("", .{});
    printTable(results.items);

    if (cli.out.len > 0) {
        appendHistory(allocator, io, cli.out, cli.label, cli.device, n, results.items) catch |e| {
            log.warn("could not write history to {s}: {s}", .{ cli.out, @errorName(e) });
        };
    }
}
