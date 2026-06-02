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
    inline for (.{
        .{ "neg", negate, exact }, .{ "abs", abs, exact },
        .{ "exp", exp, transc }, .{ "log", log_, transc },
        .{ "sqrt", sqrt, transc }, .{ "rsqrt", rsqrt, transc },
        .{ "tanh", tanh, transc },
    }) |c| {
        failures += checkUnary(allocator, io, cpu, metal, c[0], c[1], &a_un, c[2]);
    }

    if (failures != 0) {
        log.err("❌ {d} op(s) mismatched Metal vs CPU", .{failures});
        return error.MetalMismatch;
    }
    log.info("✅ PASS: all 13 elementwise ops match the CPU oracle", .{});
}
