//! The tiny growing Metal ops example. Starts at the f32 elementwise family and
//! grows one op at a time as the Metal AIR emitter gains coverage (then
//! reductions, …) — this is the regression suite and dev driver, NOT an
//! add-specific demo. See metal-xla-docs/PLAN.md.
//!
//! Each op runs on the CPU backend (the correctness oracle) and on Metal (via
//! our XLA fork's PJRT plugin + the new AIR-native emitter); the results must
//! match within a per-op tolerance (exact ops ~0; fast-math transcendentals
//! looser).
//!
//! Run: bazel run //examples/metal_ops --//platforms:metal=true

const std = @import("std");
const log = std.log;

const zml = @import("zml");

// --- the ops under test (elementwise f32) --------------------------------
fn add(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.add(b);
}
fn sub(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.sub(b);
}
fn mul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.mul(b);
}
fn div(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.div(b);
}
fn maximum(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.maximum(b);
}
fn minimum(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.minimum(b);
}
fn fma3(a: zml.Tensor, b: zml.Tensor, c: zml.Tensor) zml.Tensor {
    return a.add(b).mul(c); // (a+b)*c — should fuse into ONE kFusion kernel
}
// Broadcast INSIDE a fusion (E5.2): x*scale + bias with scale,bias [d] broadcast
// to x's [b,d]. Fuses to one kFusion containing two broadcasts (the bias/scale
// pattern). Exercises the index-remap in the fused elemental emitter.
fn affineBcast(x: zml.Tensor, scale: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    const xs = x.shape();
    return x.mul(scale.broad(xs)).add(bias.broad(xs));
}
// Matmul: contract the shared .k axis. With a={.m,.k} and b={.k,.n} this is a
// plain row-major NN matmul; with b={.n,.k} it's the y = x·Wᵀ linear (rhs
// contracts its inner dim) — same forward fn, the input tags pick NN vs NT.
fn matmul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}
fn negate(a: zml.Tensor) zml.Tensor {
    return a.negate();
}
fn abs(a: zml.Tensor) zml.Tensor {
    return a.abs();
}
fn exp(a: zml.Tensor) zml.Tensor {
    return a.exp();
}
fn log_(a: zml.Tensor) zml.Tensor {
    return a.log();
}
fn sqrt(a: zml.Tensor) zml.Tensor {
    return a.sqrt();
}
fn rsqrt(a: zml.Tensor) zml.Tensor {
    return a.rsqrt();
}
fn tanh(a: zml.Tensor) zml.Tensor {
    return a.tanh();
}

fn runBinaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    b_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t }, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn runTernaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    b_data: []const f32,
    c_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    const c_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t, c_t }, .{});
    defer exe.deinit();

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
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn runUnaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

// Shape transforms (E3, via the indexing-map / indexed-copy kernel).
fn transpose23(a: zml.Tensor) zml.Tensor {
    return a.transpose(.{ .j, .i }); // [2,3] -> [3,2]
}
fn bcast(a: zml.Tensor) zml.Tensor {
    return a.broad(zml.Shape.init(.{ .i = 4, .j = 3 }, .f32)); // [3] -> [4,3]
}
fn reshape23(a: zml.Tensor) zml.Tensor {
    return a.reshape(.{ .i = 2, .j = 3 }); // [6] -> [2,3]
}
fn sumj(a: zml.Tensor) zml.Tensor {
    return a.sum(.j); // [2,3] -> [2] (reduce over axis 1)
}
// Transpose / reshape INSIDE a fusion (E5.2): a data-movement op feeding an
// elementwise op must fuse into one kFusion (negate is exact, keeps the check
// tight). If XLA keeps them separate these hit Unimplemented and fail loudly.
fn transposeNeg(a: zml.Tensor) zml.Tensor {
    return a.transpose(.{ .j, .i }).negate(); // [2,3]->[3,2] then negate
}
fn reshapeNeg(a: zml.Tensor) zml.Tensor {
    return a.reshape(.{ .i = 3, .j = 2 }).negate(); // [2,3]->[3,2] (flat) then negate
}
fn sumAll(a: zml.Tensor) zml.Tensor {
    return a.sum(.n); // [n] -> scalar (num_out=1: the tree-reduction case)
}
// Multi-axis reduce (one kReduce over several axes): zml.sum is single-axis, so
// drop to ops.reduce. Only contiguous reduced axes are supported (they merge to
// one extent/stride). sumJK reduces the trailing [j,k]; sumAll3 reduces all.
fn sumAxes(a: zml.Tensor, axes: []const i64) zml.Tensor {
    return zml.ops.reduce(.{a}, .{zml.Tensor.constant(a.dtype().zero())}, axes, struct {
        pub fn acc(args: zml.ops.ReduceArgs) struct { zml.Tensor } {
            return .{args.right.add(args.left.convert(args.right.dtype()))};
        }
    }.acc, .{})[0];
}
fn sumJK(a: zml.Tensor) zml.Tensor {
    return sumAxes(a, &.{ 1, 2 }); // [i,j,k] -> [i]
}
fn sumAll3(a: zml.Tensor) zml.Tensor {
    return sumAxes(a, &.{ 0, 1, 2 }); // [i,j,k] -> scalar
}
const iota24: [24]f32 = blk: {
    var a: [24]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt(i);
    break :blk a;
};
// Reduce-window via cumulativeSum (a stride-1, [N-1,0]-padded, single-axis
// window): 1D prefix sum, and 2D over the inner axis (exercises the kept-axis
// base offset).
fn cumsum1(a: zml.Tensor) zml.Tensor {
    return a.cumulativeSum(.n); // [n] -> [n]
}
fn cumsum2(a: zml.Tensor) zml.Tensor {
    return a.cumulativeSum(.j); // [i,j] -> [i,j], prefix sum along j
}

// Inputs large enough (extent >= 256) to exercise the threadgroup tree
// reduction rather than the serial-per-thread kernel.
const vec1024: [1024]f32 = blk: {
    @setEvalBranchQuota(4000);
    var a: [1024]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt(i % 13);
    break :blk a;
};
const mat3x512: [1536]f32 = blk: {
    @setEvalBranchQuota(4000);
    var a: [1536]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt((i % 11) + 1);
    break :blk a;
};

/// Run a unary shape-transform `func` (input `in_shape`) on `platform`,
/// writing the flattened result into `out`.
fn runUnaryShaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const f32,
    out: []f32,
) !void {
    const a_t: zml.Tensor = .fromShape(in_shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, in_shape, .replicated, std.mem.sliceAsBytes(in_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkShaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const f32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runUnaryShaped(allocator, io, cpu, func, in_shape, in_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runUnaryShaped(allocator, io, metal, func, in_shape, in_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

/// Pass if every lane is within abs floor 1e-4 OR relative tolerance.
fn compare(name: []const u8, cpu_out: []const f32, metal_out: []const f32, rel_tol: f32) usize {
    var max_abs: f32 = 0;
    var worst_rel: f32 = 0;
    for (cpu_out, metal_out) |c, m| {
        const e = @abs(m - c);
        if (e > max_abs) max_abs = e;
        const rel = e / (@abs(c) + 1e-4);
        if (rel > worst_rel) worst_rel = rel;
    }
    if (max_abs <= 1e-4 or worst_rel <= rel_tol) {
        log.info("{s:>6}  OK     max_abs={e} max_rel={e}", .{ name, max_abs, worst_rel });
        return 0;
    }
    log.err("{s:>6}  FAIL   max_abs={e} max_rel={e}", .{ name, max_abs, worst_rel });
    log.err("        cpu  ={any}", .{cpu_out});
    log.err("        metal={any}", .{metal_out});
    return 1;
}

fn checkBinary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    b: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runBinaryOn(allocator, io, cpu, func, a, b, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runBinaryOn(allocator, io, metal, func, a, b, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

fn checkTernary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    b: []const f32,
    c: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runTernaryOn(allocator, io, cpu, func, a, b, c, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runTernaryOn(allocator, io, metal, func, a, b, c, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

/// Run a `func(x, scale, bias)` where x has `x_shape` and scale/bias share the
/// smaller `d_shape` (broadcast inside the fusion). Writes the flat result.
fn runAffineBcast(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    x_shape: zml.Shape,
    x_data: []const f32,
    d_shape: zml.Shape,
    scale_data: []const f32,
    bias_data: []const f32,
    out: []f32,
) !void {
    const xt: zml.Tensor = .fromShape(x_shape);
    const st: zml.Tensor = .fromShape(d_shape);
    const bt: zml.Tensor = .fromShape(d_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ xt, st, bt }, .{});
    defer exe.deinit();

    var xb = try zml.Buffer.fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_data));
    defer xb.deinit();
    var sb = try zml.Buffer.fromBytes(io, platform, d_shape, .replicated, std.mem.sliceAsBytes(scale_data));
    defer sb.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, d_shape, .replicated, std.mem.sliceAsBytes(bias_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ xb, sb, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkAffineBcast(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    x_shape: zml.Shape,
    x_data: []const f32,
    d_shape: zml.Shape,
    scale_data: []const f32,
    bias_data: []const f32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runAffineBcast(allocator, io, cpu, func, x_shape, x_data, d_shape, scale_data, bias_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runAffineBcast(allocator, io, metal, func, x_shape, x_data, d_shape, scale_data, bias_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

// Two differently-shaped inputs (a:[M,K], b:[K,N] or [N,K]) → out:[M,N].
fn runMatmul(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    out: []f32,
) !void {
    const at: zml.Tensor = .fromShape(a_shape);
    const bt: zml.Tensor = .fromShape(b_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ at, bt }, .{});
    defer exe.deinit();

    var ab = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ ab, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkMatmul(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [256]f32 = undefined;
    var mo: [256]f32 = undefined;
    runMatmul(allocator, io, cpu, func, a_shape, a_data, b_shape, b_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmul(allocator, io, metal, func, a_shape, a_data, b_shape, b_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], rel_tol);
}

// dtype-generic matmul (f16 = native f16, bf16 = zml.floats.BFloat16). Same as
// runMatmul but typed; the output is converted to f32 by the caller for compare.
fn runMatmulT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const T,
    b_shape: zml.Shape,
    b_data: []const T,
    out: []T,
) !void {
    const at: zml.Tensor = .fromShape(a_shape);
    const bt: zml.Tensor = .fromShape(b_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ at, bt }, .{});
    defer exe.deinit();

    var ab = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ ab, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(T));
}

fn toF32(comptime T: type, v: T) f32 {
    return if (T == f32) v else if (T == f16) @floatCast(v) else v.toF32();
}

fn checkMatmulT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const T,
    b_shape: zml.Shape,
    b_data: []const T,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [16]T = undefined;
    var mo: [16]T = undefined;
    runMatmulT(T, allocator, io, cpu, func, a_shape, a_data, b_shape, b_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmulT(T, allocator, io, metal, func, a_shape, a_data, b_shape, b_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    var cof: [16]f32 = undefined;
    var mof: [16]f32 = undefined;
    for (0..n_out) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    return compare(name, cof[0..n_out], mof[0..n_out], rel_tol);
}

fn dtOf(comptime T: type) zml.DataType {
    return if (T == f32) .f32 else if (T == f16) .f16 else .bf16;
}

fn fillFrac(comptime T: type, s: []T, seed: usize) void {
    for (s, 0..) |*e, i| {
        const x: f32 = 0.5 - @as(f32, @floatFromInt((i * seed + 7) % 97)) / 97.0;
        e.* = if (T == f32) x else if (T == f16) @floatCast(x) else zml.floats.BFloat16.fromF32(x);
    }
}

// Heap-backed, dtype+layout-generic matmul check for medium shapes (the [16]
// buffers of checkMatmulT can't hold them). transpose_b picks NN (b={k,n}) vs
// NT (b={n,k}, the y=x·Wᵀ linear). Fractional values stress tiling/accumulation.
fn checkMM(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    M: i64,
    K: i64,
    N: i64,
    transpose_b: bool,
    rel_tol: f32,
) usize {
    const dt = dtOf(T);
    const a_shape = zml.Shape.init(.{ .m = M, .k = K }, dt);
    const b_shape = if (transpose_b)
        zml.Shape.init(.{ .n = N, .k = K }, dt)
    else
        zml.Shape.init(.{ .k = K, .n = N }, dt);
    const mk: usize = @intCast(M * K);
    const kn: usize = @intCast(K * N);
    const mn: usize = @intCast(M * N);
    const a = allocator.alloc(T, mk) catch return 1;
    defer allocator.free(a);
    const b = allocator.alloc(T, kn) catch return 1;
    defer allocator.free(b);
    fillFrac(T, a, 31);
    fillFrac(T, b, 17);
    const co = allocator.alloc(T, mn) catch return 1;
    defer allocator.free(co);
    const mo = allocator.alloc(T, mn) catch return 1;
    defer allocator.free(mo);
    runMatmulT(T, allocator, io, cpu, matmul, a_shape, a, b_shape, b, co) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmulT(T, allocator, io, metal, matmul, a_shape, a, b_shape, b, mo) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    const cof = allocator.alloc(f32, mn) catch return 1;
    defer allocator.free(cof);
    const mof = allocator.alloc(f32, mn) catch return 1;
    defer allocator.free(mof);
    for (0..mn) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    return compare(name, cof, mof, rel_tol);
}

fn checkUnary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runUnaryOn(allocator, io, cpu, func, a, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runUnaryOn(allocator, io, metal, func, a, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cpu: *zml.Platform = try .init(allocator, io, .cpu, .{});
    defer cpu.deinit(allocator, io);
    const metal: *zml.Platform = try .init(allocator, io, .metal, .{});
    defer metal.deinit(allocator, io);
    log.info("CPU oracle vs Metal: {f}", .{metal.fmtVerbose()});

    // Binary inputs: negatives + positives; no zeros in b (for div).
    const a_bin = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
    const b_bin = [_]f32{ 10, 20, -30, 40, -50, 60, 70, 80 };
    // Unary inputs: strictly positive so log/sqrt/rsqrt are valid.
    const a_un = [_]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };

    const exact = 1e-6; // a single fast-math FP op is bit-identical to plain
    const transc = 1e-2; // fast-math exp/log/sqrt/rsqrt/tanh approximations

    var failures: usize = 0;
    inline for (.{
        .{ "add", add, exact }, .{ "sub", sub, exact },
        .{ "mul", mul, exact }, .{ "div", div, 1e-3 },
        .{ "max", maximum, exact }, .{ "min", minimum, exact },
    }) |c| {
        failures += checkBinary(allocator, io, cpu, metal, c[0], c[1], &a_bin, &b_bin, c[2]);
    }

    // Multi-op fusion (E5): (a+b)*c must compile into ONE kFusion kernel.
    {
        const c_bin = [_]f32{ 2, 2, 2, 2, 2, 2, 2, 2 };
        failures += checkTernary(allocator, io, cpu, metal, "fma", fma3, &a_bin, &b_bin, &c_bin, exact);
    }

    // Broadcast inside a fusion (E5.2): x*scale + bias, scale/bias [d]->[2,3].
    // = [[11,202,3003],[41,502,6003]]. A kFusion with two broadcasts inside.
    failures += checkAffineBcast(allocator, io, cpu, metal, "affine", affineBcast,
        zml.Shape.init(.{ .b = 2, .d = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .d = 3 }, .f32), &[_]f32{ 10, 100, 1000 }, &[_]f32{ 1, 2, 3 }, 6);
    inline for (.{
        .{ "neg", negate, exact }, .{ "abs", abs, exact },
        .{ "exp", exp, transc }, .{ "log", log_, transc },
        .{ "sqrt", sqrt, transc }, .{ "rsqrt", rsqrt, transc },
        .{ "tanh", tanh, transc },
    }) |c| {
        failures += checkUnary(allocator, io, cpu, metal, c[0], c[1], &a_un, c[2]);
    }

    // Shape transforms (E3): pure data movement, exact vs CPU.
    failures += checkShaped(allocator, io, cpu, metal, "transp", transpose23,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "bcast", bcast,
        zml.Shape.init(.{ .j = 3 }, .f32), &[_]f32{ 10, 20, 30 }, 12);
    failures += checkShaped(allocator, io, cpu, metal, "reshape", reshape23,
        zml.Shape.init(.{ .n = 6 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);

    // Transpose / reshape inside a fusion (E5.2 finish): negate∘transpose and
    // negate∘reshape, [2,3]->[3,2], must each fuse into one kFusion.
    failures += checkShaped(allocator, io, cpu, metal, "tneg", transposeNeg,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "rneg", reshapeNeg,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);

    // Reduction (E4): sum over axis 1, [2,3] -> [2] = [3, 12].
    failures += checkShaped(allocator, io, cpu, metal, "sum", sumj,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 2);

    // Multi-axis reduce (contiguous): [2,3,4] reduce {j,k} -> [2] = [66, 210];
    // reduce-all {i,j,k} -> [] = 276. Both merge to one extent/stride.
    failures += checkShaped(allocator, io, cpu, metal, "sumjk", sumJK,
        zml.Shape.init(.{ .i = 2, .j = 3, .k = 4 }, .f32), &iota24, 2);
    failures += checkShaped(allocator, io, cpu, metal, "sumall3", sumAll3,
        zml.Shape.init(.{ .i = 2, .j = 3, .k = 4 }, .f32), &iota24, 1);

    // Reduce-window (E4+): cumulative sum. 1D [6]->[0,1,3,6,10,15]; 2D prefix
    // sum along j (the windowed axis), i kept (tests the base offset).
    failures += checkShaped(allocator, io, cpu, metal, "cumsum", cumsum1,
        zml.Shape.init(.{ .n = 6 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "cumsum2", cumsum2,
        zml.Shape.init(.{ .i = 2, .j = 4 }, .f32), &[_]f32{ 0, 1, 2, 3, 10, 20, 30, 40 }, 8);
    // Reduce-to-scalar via the TREE kernel (E4 harden): [1024] -> [], num_out=1,
    // extent=1024 (>=256) so this is one threadgroup cooperatively reducing
    // (base=0, exercises the grid-stride accumulate + tree). Serial-per-thread
    // would run it on a single thread.
    failures += checkShaped(allocator, io, cpu, metal, "sumall", sumAll,
        zml.Shape.init(.{ .n = 1024 }, .f32), &vec1024, 1);
    // Tree reduction with num_out>1: [3,512] -> [3]. base != 0, so this also
    // exercises the tree kernel's output-index delinearization.
    failures += checkShaped(allocator, io, cpu, metal, "sumtree", sumj,
        zml.Shape.init(.{ .i = 3, .j = 512 }, .f32), &mat3x512, 3);

    // Two-level reduction (extent >= 65536, tiny num_out): pass 1 runs many
    // threadgroups per output into partials, pass 2 reduces them. sumbig is
    // reduce-to-scalar (base=0); sum2dbig has num_out=4 (base!=0, exercises the
    // pass-1 base delinearization). Runtime-allocated (too big for comptime).
    {
        const big = try allocator.alloc(f32, 65536);
        defer allocator.free(big);
        for (big, 0..) |*e, i| e.* = @floatFromInt(i % 13);
        failures += checkShaped(allocator, io, cpu, metal, "sumbig", sumAll,
            zml.Shape.init(.{ .n = 65536 }, .f32), big, 1);

        const big2d = try allocator.alloc(f32, 4 * 65536);
        defer allocator.free(big2d);
        for (big2d, 0..) |*e, i| e.* = @floatFromInt((i % 13) + 1);
        failures += checkShaped(allocator, io, cpu, metal, "sum2dbig", sumj,
            zml.Shape.init(.{ .i = 4, .j = 65536 }, .f32), big2d, 4);
    }

    // Matmul. Backend mirrors XLA_METAL_MATMUL: "" (MPS default) | "naive" |
    // "metalblas". Tiny NN/NT (N<32) route, under metalBLAS, to its m5/gemv
    // kernels (not wired into the plugin yet → loud Unimplemented), so skip them
    // there; mmMed (96x48, K=64) is m5_tensor-eligible and runs on every backend.
    const mm_backend: []const u8 = blk: {
        const v = std.c.getenv("XLA_METAL_MATMUL") orelse break :blk "";
        break :blk std.mem.span(v);
    };
    const mm_is_metalblas = std.mem.eql(u8, mm_backend, "metalblas");
    const mm_is_naive = std.mem.eql(u8, mm_backend, "naive");

    // NN: [2,3]·[3,2] = [[22,28],[49,64]]. NT (the y = x·Wᵀ linear, rhs contracts
    // its inner dim): [2,3]·[2,3]ᵀ with W=[[1,0,1],[0,1,0]] = [[4,2],[10,5]].
    // Small integers → exact in f32.
    if (!mm_is_metalblas) {
        failures += checkMatmul(allocator, io, cpu, metal, "mmNN", matmul,
            zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
            zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 4, exact);
        failures += checkMatmul(allocator, io, cpu, metal, "mmNT", matmul,
            zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
            zml.Shape.init(.{ .n = 2, .k = 3 }, .f32), &[_]f32{ 1, 0, 1, 0, 1, 0 }, 4, exact);
    }
    // Medium NN ([96,64]·[64,48]) and NT (the y=x·Wᵀ linear: [96,64]·[48,64]ᵀ),
    // fractional values. NN runs on every backend (metalBLAS → m5_tensor); NT
    // under metalBLAS would route to m5_gemm, which isn't wired yet (it produced
    // wrong output — simdgroup-builtin issue), so NT is MPS/naive only.
    failures += checkMM(f32, allocator, io, cpu, metal, "mmMed", 96, 64, 48, false, 1e-4);
    if (!mm_is_metalblas)
        failures += checkMM(f32, allocator, io, cpu, metal, "mmNTmed", 96, 64, 48, true, 1e-4);

    // f16 — MPS default only (naive is f32-only; metalBLAS's tiny path isn't wired).
    if (!mm_is_naive and !mm_is_metalblas) {
        failures += checkMatmulT(f16, allocator, io, cpu, metal, "mmF16", matmul, zml.Shape.init(.{ .m = 2, .k = 3 }, .f16), &[_]f16{ 1, 2, 3, 4, 5, 6 }, zml.Shape.init(.{ .k = 3, .n = 2 }, .f16), &[_]f16{ 1, 2, 3, 4, 5, 6 }, 4, 1e-2);
    }

    // bf16 — metalBLAS only (MPS aborts on bf16; naive is f32-only). NN→m5_tensor,
    // NT→m5_gemm. bf16 is metalBLAS's edge over MPS, so this is the key coverage.
    if (mm_is_metalblas) {
        // bf16 NN-square via metalBLAS m5_tensor (MPS can't do bf16; naive is
        // f32-only). bf16 NT and other shapes route to m5_gemm/gemv, not yet wired.
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16NN", 96, 64, 48, false, 3e-2);
    }

    if (failures != 0) {
        log.err("❌ {d} op(s) mismatched Metal vs CPU", .{failures});
        return error.MetalMismatch;
    }
    log.info("✅ PASS: all Metal ops match the CPU oracle (elementwise + transpose/bcast/reshape + sum-reduce)", .{});
}
