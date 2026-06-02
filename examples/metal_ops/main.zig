//! The tiny growing Metal ops example. Starts at `c = a + b` (E1) and grows one
//! op at a time as the Metal AIR emitter gains coverage (elementwise, then
//! reductions, …) — this is the regression suite and dev driver, NOT an
//! add-specific demo. See metal-xla-docs/PLAN.md.
//!
//! Each op runs on the CPU backend (the correctness oracle) and on Metal (via
//! our XLA fork's PJRT plugin + the new AIR-native emitter); the results must
//! match.
//!
//! Run: bazel run //examples/metal_ops --//platforms:metal=true

const std = @import("std");
const log = std.log;

const zml = @import("zml");

fn add(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.add(b);
}

/// Compile + run `add` on `target`, writing the result into `out`.
fn runOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    target: zml.Target,
    a_data: []const f32,
    b_data: []const f32,
    out: []f32,
) !void {
    const platform: *zml.Platform = try .init(allocator, io, target, .{});
    defer platform.deinit(allocator, io);
    log.info("  [{s}] {f}", .{ @tagName(target), platform.fmtVerbose() });

    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);

    var exe = try platform.compileFn(allocator, io, add, .{ a_t, b_t }, .{});
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const a = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    const n = a.len;

    var cpu_out: [n]f32 = undefined;
    var metal_out: [n]f32 = undefined;

    log.info("Running add on CPU (oracle)...", .{});
    try runOn(allocator, io, .cpu, &a, &b, &cpu_out);

    log.info("Running add on Metal...", .{});
    try runOn(allocator, io, .metal, &a, &b, &metal_out);

    var max_abs_err: f32 = 0.0;
    for (0..n) |i| {
        const e = @abs(metal_out[i] - cpu_out[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    log.info("a        = {any}", .{a});
    log.info("b        = {any}", .{b});
    log.info("cpu  a+b = {any}", .{cpu_out});
    log.info("metal a+b= {any}", .{metal_out});
    log.info("max abs err = {d}", .{max_abs_err});

    if (max_abs_err > 1e-5) {
        log.err("❌ MISMATCH: Metal add does not match CPU oracle", .{});
        return error.MetalMismatch;
    }
    log.info("✅ PASS: Metal add matches CPU oracle", .{});
}
