const std = @import("std");
const log = std.log.scoped(.mxfp8);

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const MX_BLOCK_SIZE: i64 = 32;
const FP8_E4M3FN_MAX: f32 = 448.0;
const LN2: f32 = 0.69314718055994530942;
const E8M0_MIN_SCALE: f32 = 5.877471754111438e-39;

const Quantized = struct {
    values: zml.Tensor,
    scales: zml.Tensor,
};

const BenchStats = struct {
    elapsed_ns: u64,
    iterations: usize,
};

const Output = struct {
    reference: zml.Tensor,
    mxfp8: zml.Tensor,
    a_values: zml.Tensor,
    a_scales: zml.Tensor,
    b_values: zml.Tensor,
    b_scales: zml.Tensor,
};

fn quantizeMxfp8K(x: zml.Tensor) Quantized {
    stdx.debug.assert(x.dtype() == .bf16, "quantizeMxfp8K expects bf16 input, got {}", .{x.dtype()});
    stdx.debug.assert(x.axis(.k) == x.rank() - 1, "quantizeMxfp8K expects .k to be the minor axis, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(.k), MX_BLOCK_SIZE) == 0, "K dimension must be divisible by {}, got {}", .{ MX_BLOCK_SIZE, x.dim(.k) });

    const block_count = @divExact(x.dim(.k), MX_BLOCK_SIZE);
    const blocked = x.convert(.f32).splitAxis(.k, .{ .mx_block = block_count, .mx = MX_BLOCK_SIZE });
    const absmax = blocked.abs().max(.mx);
    const scale_raw = absmax.divByConst(FP8_E4M3FN_MAX)
        .maximum(zml.Tensor.scalar(E8M0_MIN_SCALE, .f32));
    const scale_pow2 = scale_raw.log().divByConst(LN2).ceil().scale(LN2).exp();
    const scales_blocked = scale_pow2.convert(.f8e8m0);
    const scale_f32 = scales_blocked.convert(.f32);

    const values = blocked.div(scale_f32.broad(blocked.shape()))
        .maximum(zml.Tensor.scalar(-FP8_E4M3FN_MAX, .f32))
        .minimum(zml.Tensor.scalar(FP8_E4M3FN_MAX, .f32))
        .convert(.f8e4m3fn)
        .merge(.{ .k = .{ .mx_block, .mx } });

    return .{ .values = values, .scales = scales_blocked.squeeze(.mx) };
}

fn blockScaledDot(a: Quantized, b: Quantized) zml.Tensor {
    return zml.ops.scaled_dot(a.values, b.values, a.scales, b.scales, .k);
}

fn bf16Dot(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}

fn mxfp8Dot(a_values: zml.Tensor, b_values: zml.Tensor, a_scales: zml.Tensor, b_scales: zml.Tensor) zml.Tensor {
    return zml.ops.scaled_dot(a_values, b_values, a_scales, b_scales, .k);
}

fn run(a: zml.Tensor, b: zml.Tensor) Output {
    const a_mx = quantizeMxfp8K(a);
    const b_mx = quantizeMxfp8K(b.transpose(.{ 1, 0 }));
    return .{
        .reference = a.dot(b, .k),
        .mxfp8 = blockScaledDot(a_mx, b_mx),
        .a_values = a_mx.values,
        .a_scales = a_mx.scales,
        .b_values = b_mx.values,
        .b_scales = b_mx.scales,
    };
}

const CliArgs = struct {
    m: i64 = 64,
    n: i64 = 64,
    k: i64 = 128,
    seed: u64 = 0,
    xla_dump_to: ?[]const u8 = null,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    benchmark: bool = false,
    profile: bool = false,
    profile_repository_path: []const u8 = "/tmp/xprof",
    profile_session_prefix: []const u8 = "mxfp8",
    warmups: usize = 5,
    iterations: usize = 20,
    absolute_tolerance: f32 = 0.25,
    relative_tolerance: f32 = 0.20,
    minimum_close_fraction: f32 = 0.99,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    if (@mod(cli_args.k, MX_BLOCK_SIZE) != 0) {
        log.err("--k must be divisible by {}, got {}", .{ MX_BLOCK_SIZE, cli_args.k });
        return error.InvalidK;
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const a_shape = zml.Shape.init(.{ .m = cli_args.m, .k = cli_args.k }, .bf16);
    const b_shape = zml.Shape.init(.{ .k = cli_args.k, .n = cli_args.n }, .bf16);
    const a: zml.Tensor = .fromShape(a_shape);
    const b: zml.Tensor = .fromShape(b_shape);

    const xla_dump_hlo_pass_re: ?[]const u8 = if (cli_args.xla_dump_to != null)
        cli_args.xla_dump_hlo_pass_re orelse ".*"
    else
        cli_args.xla_dump_hlo_pass_re;

    var exe = blk: {
        log.info("Compiling MXFP8 StableHLO example...", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Compiled MXFP8 example [{f}]", .{start.untilNow(io, .awake)});
        break :blk try platform.compileFn(allocator, io, run, .{ a, b }, .{
            .program_name = "mxfp8",
            .partitioner = .gspmd,
            .xla_dump_to = cli_args.xla_dump_to,
            .xla_dump_hlo_pass_re = xla_dump_hlo_pass_re,
        });
    };
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(cli_args.seed);
    const random = rng.random();

    var a_buffer = try createRandomBf16Buffer(allocator, io, platform, a_shape, random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBf16Buffer(allocator, io, platform, b_shape, random);
    defer b_buffer.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ a_buffer, b_buffer });
    exe.callOpts(io, args, &results, .{ .wait = true });

    var output = results.get(zml.Bufferized(Output));
    defer deinitOutput(&output);

    log.info("A MXFP8 values: {f}; scales: {f}", .{ output.a_values.shape(), output.a_scales.shape() });
    log.info("B MXFP8 values: {f}; scales: {f}", .{ output.b_values.shape(), output.b_scales.shape() });
    if (cli_args.xla_dump_to) |dump_to| {
        log.info("HLO dumped to {s}", .{dump_to});
    }

    try zml.testing.expectClose(io, output.reference, output.mxfp8, .{
        .absolute_tolerance = cli_args.absolute_tolerance,
        .relative_tolerance = cli_args.relative_tolerance,
        .minimum_close_fraction = cli_args.minimum_close_fraction,
    });

    log.info("MXFP8 block-scaled dot matches bf16 reference for {d}x{d}x{d}", .{ cli_args.m, cli_args.k, cli_args.n });

    if (cli_args.benchmark or cli_args.profile) {
        try benchmarkDots(
            allocator,
            io,
            platform,
            cli_args,
            xla_dump_hlo_pass_re,
            a,
            b,
            a_buffer,
            b_buffer,
            output.a_values,
            output.b_values,
            output.a_scales,
            output.b_scales,
        );
    }
}

fn benchmarkDots(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    cli_args: CliArgs,
    xla_dump_hlo_pass_re: ?[]const u8,
    a: zml.Tensor,
    b: zml.Tensor,
    a_buffer: zml.Buffer,
    b_buffer: zml.Buffer,
    a_values_buffer: zml.Buffer,
    b_values_buffer: zml.Buffer,
    a_scales_buffer: zml.Buffer,
    b_scales_buffer: zml.Buffer,
) !void {
    const block_count = @divExact(cli_args.k, MX_BLOCK_SIZE);
    const a_values: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = cli_args.m, .k = cli_args.k }, .f8e4m3fn));
    const b_values: zml.Tensor = .fromShape(zml.Shape.init(.{ .n = cli_args.n, .k = cli_args.k }, .f8e4m3fn));
    const a_scales: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = cli_args.m, .mx_block = block_count }, .f8e8m0));
    const b_scales: zml.Tensor = .fromShape(zml.Shape.init(.{ .n = cli_args.n, .mx_block = block_count }, .f8e8m0));

    var bf16_exe = blk: {
        log.info("Compiling bf16 dot benchmark...", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Compiled bf16 dot benchmark [{f}]", .{start.untilNow(io, .awake)});
        break :blk try platform.compileFn(allocator, io, bf16Dot, .{ a, b }, compileOptions(
            "mxfp8_bf16_dot",
            cli_args.xla_dump_to,
            xla_dump_hlo_pass_re,
        ));
    };
    defer bf16_exe.deinit();

    var mxfp8_exe = blk: {
        log.info("Compiling MXFP8 scaled-dot benchmark...", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Compiled MXFP8 scaled-dot benchmark [{f}]", .{start.untilNow(io, .awake)});
        break :blk try platform.compileFn(allocator, io, mxfp8Dot, .{ a_values, b_values, a_scales, b_scales }, compileOptions(
            "mxfp8_scaled_dot",
            cli_args.xla_dump_to,
            xla_dump_hlo_pass_re,
        ));
    };
    defer mxfp8_exe.deinit();

    var bf16_args = try bf16_exe.args(allocator);
    defer bf16_args.deinit(allocator);
    var bf16_results = try bf16_exe.results(allocator);
    defer bf16_results.deinit(allocator);
    bf16_args.set(.{ a_buffer, b_buffer });

    var mxfp8_args = try mxfp8_exe.args(allocator);
    defer mxfp8_args.deinit(allocator);
    var mxfp8_results = try mxfp8_exe.results(allocator);
    defer mxfp8_results.deinit(allocator);
    mxfp8_args.set(.{ a_values_buffer, b_values_buffer, a_scales_buffer, b_scales_buffer });

    log.info("Warming benchmarks: {d} iterations", .{cli_args.warmups});
    for (0..cli_args.warmups) |_| {
        callAndDrop(io, &bf16_exe, bf16_args, &bf16_results);
        callAndDrop(io, &mxfp8_exe, mxfp8_args, &mxfp8_results);
    }

    const bf16_stats = try benchmarkExecutable(
        allocator,
        io,
        platform,
        "bf16-dot",
        &bf16_exe,
        bf16_args,
        &bf16_results,
        cli_args,
    );
    const mxfp8_stats = try benchmarkExecutable(
        allocator,
        io,
        platform,
        "mxfp8-scaled-dot",
        &mxfp8_exe,
        mxfp8_args,
        &mxfp8_results,
        cli_args,
    );

    logBenchStats("bf16 dot", bf16_stats, cli_args);
    logBenchStats("MXFP8 scaled-dot", mxfp8_stats, cli_args);

    const speedup = @as(f64, @floatFromInt(bf16_stats.elapsed_ns)) / @as(f64, @floatFromInt(mxfp8_stats.elapsed_ns));
    if (mxfp8_stats.elapsed_ns < bf16_stats.elapsed_ns) {
        log.info("MXFP8 scaled-dot is {d:.3}x faster than bf16 dot", .{speedup});
    } else {
        log.warn("MXFP8 scaled-dot was not faster than bf16 dot in this run: {d:.3}x", .{speedup});
    }
}

fn compileOptions(program_name: []const u8, xla_dump_to: ?[]const u8, xla_dump_hlo_pass_re: ?[]const u8) zml.CompilationOptions {
    return .{
        .program_name = program_name,
        .partitioner = .gspmd,
        .xla_dump_to = xla_dump_to,
        .xla_dump_hlo_pass_re = xla_dump_hlo_pass_re,
    };
}

fn benchmarkExecutable(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    label: []const u8,
    exe: *const zml.Exe,
    args: zml.Exe.Arguments,
    results: *zml.Exe.Results,
    cli_args: CliArgs,
) !BenchStats {
    var profiler: ?zml.Platform.Profiler = null;
    defer if (profiler) |*p| p.deinit();

    if (cli_args.profile) {
        const session_id = try std.fmt.allocPrint(allocator, "{s}-{s}", .{ cli_args.profile_session_prefix, label });
        defer allocator.free(session_id);
        profiler = try platform.profiler(allocator, io, .{
            .repository_path = cli_args.profile_repository_path,
            .session_id = session_id,
            .device_type = .gpu,
        });
        try profiler.?.start();
    }

    const span_name = try zml.tracer.formatSpanName(allocator, "mxfp8.benchmark", .{
        .kernel = label,
        .iterations = cli_args.iterations,
    });
    defer allocator.free(span_name);
    var span = zml.tracer.Span.start(span_name);
    defer span.end();

    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..cli_args.iterations) |_| {
        callAndDrop(io, exe, args, results);
    }
    const elapsed_ns: u64 = @intCast(start.untilNow(io, .awake).toNanoseconds());

    if (profiler) |*p| {
        _ = try p.stop();
    }

    return .{
        .elapsed_ns = elapsed_ns,
        .iterations = cli_args.iterations,
    };
}

fn callAndDrop(io: std.Io, exe: *const zml.Exe, args: zml.Exe.Arguments, results: *zml.Exe.Results) void {
    exe.callOpts(io, args, results, .{ .wait = true });
    var output = results.get(zml.Buffer);
    output.deinit();
}

fn logBenchStats(label: []const u8, stats: BenchStats, cli_args: CliArgs) void {
    const elapsed_s = @as(f64, @floatFromInt(stats.elapsed_ns)) / std.time.ns_per_s;
    const per_iter_ms = elapsed_s * 1000.0 / @as(f64, @floatFromInt(stats.iterations));
    const flops = 2.0 *
        @as(f64, @floatFromInt(cli_args.m)) *
        @as(f64, @floatFromInt(cli_args.n)) *
        @as(f64, @floatFromInt(cli_args.k)) *
        @as(f64, @floatFromInt(stats.iterations));
    const tflops = flops / elapsed_s / 1_000_000_000_000.0;

    log.info("{s}: {d} iterations, {d:.3} ms/iter, {d:.3} TFLOP/s", .{
        label,
        stats.iterations,
        per_iter_ms,
        tflops,
    });
}

fn createRandomBf16Buffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    random: std.Random,
) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(zml.floats.BFloat16)) |*item| {
        const val = (random.float(f32) - 0.5) * 2.0;
        item.* = .fromF32(val);
    }

    return zml.Buffer.fromSlice(io, platform, slice, .replicated);
}

fn deinitOutput(output: *zml.Bufferized(Output)) void {
    output.reference.deinit();
    output.mxfp8.deinit();
    output.a_values.deinit();
    output.a_scales.deinit();
    output.b_values.deinit();
    output.b_scales.deinit();
}
