const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.benchmark_profiler);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const saxpy_dispatch_calls: usize = 5;

const Mode = enum {
    saxpy,
    matmul,
    both,
};

const CliArgs = struct {
    pub const help =
        \\ benchmark_profiler [options]
        \\
        \\ Profiles host dispatch overhead (`exe.call`) and heavy device compute with xprof.
        \\
        \\ Options:
        \\   --mode=<saxpy|matmul|both>  Which benchmark to run (default: both)
        \\   --dtype=<dtype>             Tensor dtype for both benchmarks (default: f32)
        \\   --saxpySize=<n>             SAXPY vector length (default: 4096)
        \\   --saxpyAlpha=<a>            SAXPY scalar coefficient (default: 1.5)
        \\   --matmulM=<m>               MATMUL left rows (default: 4096)
        \\   --matmulK=<k>               Shared MATMUL axis (default: 4096)
        \\   --matmulN=<n>               MATMUL right cols (default: 4096)
        \\   --matmulCalls=<n>           MATMUL profiled call count (default: 1)
        \\   --xprofDir=<path>           xprof repository path (default: /tmp/xprof)
        \\   --sessionId=<name>          Base xprof session id (default: benchmark-profiler)
        \\
        \\ Examples:
        \\   benchmark_profiler --mode=saxpy --sessionId=dispatch
        \\   benchmark_profiler --mode=matmul --dtype=f16 --matmulM=8192 --matmulK=8192 --matmulN=8192 --sessionId=compute
        \\   benchmark_profiler --mode=both
        \\
    ;

    mode: Mode = .both,
    dtype: zml.DataType = .f32,

    saxpySize: usize = 4096,
    saxpyAlpha: f32 = 1.5,

    matmulM: usize = 4096,
    matmulK: usize = 4096,
    matmulN: usize = 4096,
    matmulCalls: usize = 1,

    xprofDir: []const u8 = "/tmp/xprof",
    sessionId: []const u8 = "benchmark-profiler",
};

fn validateArgs(args: CliArgs) !void {
    if (!args.dtype.isFloat()) {
        log.err("Only floating-point dtypes are supported, got {s}", .{@tagName(args.dtype)});
        return error.InvalidDType;
    }
    if (args.xprofDir.len == 0) return error.InvalidXprofDir;
    if (args.sessionId.len == 0) return error.InvalidSessionId;

    switch (args.mode) {
        .saxpy => {
            if (args.saxpySize == 0) return error.InvalidSaxpySize;
        },
        .matmul => {
            if (args.matmulM == 0 or args.matmulK == 0 or args.matmulN == 0) return error.InvalidMatmulShape;
            if (args.matmulCalls == 0) return error.InvalidMatmulCalls;
        },
        .both => {
            if (args.saxpySize == 0) return error.InvalidSaxpySize;
            if (args.matmulM == 0 or args.matmulK == 0 or args.matmulN == 0) return error.InvalidMatmulShape;
            if (args.matmulCalls == 0) return error.InvalidMatmulCalls;
        },
    }
}

fn saxpy(a: zml.Tensor, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
    return x.mul(a).add(y);
}

fn matmul(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.dot(rhs, .k);
}

fn withSessionSuffix(allocator: std.mem.Allocator, session_id: []const u8, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}-{s}", .{ session_id, suffix });
}

fn createProfiler(
    platform: *const zml.Platform,
    allocator: std.mem.Allocator,
    io: std.Io,
    xprof_dir: []const u8,
    session_id: []const u8,
) !zml.Platform.Profiler {
    var profiler_options: zml.Platform.ProfilerOptions = .defaults;
    profiler_options.repository_path = xprof_dir;
    profiler_options.session_id = session_id;
    profiler_options.host_tracer_level = 3;
    profiler_options.device_tracer_level = 3;
    return platform.profiler(allocator, io, profiler_options);
}

fn createFilledBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.sharding.Sharding,
    fill_value: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |dtype| switch (comptime dtype.class()) {
            .float => {
                const ZigType = dtype.toZigType();
                const converted_value: ZigType = switch (ZigType) {
                    f16, f32, f64 => @floatCast(fill_value),
                    inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(fill_value) else unreachable,
                };
                for (slice.items(ZigType)) |*item| item.* = converted_value;
            },
            else => return error.InvalidDType,
        },
    }

    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}

fn runCallAndAwait(io: std.Io, exe: *const zml.Exe, exe_args: zml.Exe.Arguments, exe_results: *zml.Exe.Results) !void {
    exe.call(exe_args, exe_results);
    var out = exe_results.get(zml.Buffer);
    defer out.deinit();
    _ = try out.await(io);
}

fn runCalls(io: std.Io, exe: *const zml.Exe, exe_args: zml.Exe.Arguments, exe_results: *zml.Exe.Results, call_count: usize) !void {
    for (0..call_count) |_| {
        try runCallAndAwait(io, exe, exe_args, exe_results);
    }
}

fn runSaxpyProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    args: CliArgs,
) !void {
    const alpha_shape = zml.Shape.init(.{}, args.dtype);
    const vector_shape = zml.Shape.init(.{ .n = args.saxpySize }, args.dtype);

    const alpha: zml.Tensor = .fromShape(alpha_shape);
    const x: zml.Tensor = .fromShape(vector_shape);
    const y: zml.Tensor = .fromShape(vector_shape);

    log.info("Compiling SAXPY executable...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        saxpy,
        .{ alpha, x, y },
        .{
            .shardings = &.{sharding},
            .program_name = "benchmark_profiler_saxpy",
        },
    );
    defer exe.deinit();

    var alpha_buffer = try createFilledBuffer(allocator, io, platform, alpha_shape, sharding, args.saxpyAlpha);
    defer alpha_buffer.deinit();
    var x_buffer = try createFilledBuffer(allocator, io, platform, vector_shape, sharding, 1.0);
    defer x_buffer.deinit();
    var y_buffer = try createFilledBuffer(allocator, io, platform, vector_shape, sharding, 2.0);
    defer y_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ alpha_buffer, x_buffer, y_buffer });

    log.info("Running SAXPY warmup...", .{});
    try runCalls(io, &exe, exe_args, &exe_results, 1);

    const session_id = try withSessionSuffix(allocator, args.sessionId, "saxpy");
    defer allocator.free(session_id);

    var profiler = try createProfiler(platform, allocator, io, args.xprofDir, session_id);
    defer profiler.deinit();

    try profiler.start();
    defer {
        if ((profiler.stop() catch unreachable)) |profile| {
            log.info("SAXPY profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
        } else {
            log.warn("SAXPY profiler extension returned no trace artifacts on this platform.", .{});
        }
    }

    log.info("Profiling SAXPY dispatch with {d} sequential exe.call invocations", .{saxpy_dispatch_calls});
    const started_at: std.Io.Timestamp = .now(io, .awake);
    try runCalls(io, &exe, exe_args, &exe_results, saxpy_dispatch_calls);
    log.info("SAXPY profiled window elapsed: {f}", .{started_at.untilNow(io, .awake)});
}

fn runMatmulProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    args: CliArgs,
) !void {
    const lhs_shape = zml.Shape.init(.{ .m = args.matmulM, .k = args.matmulK }, args.dtype);
    const rhs_shape = zml.Shape.init(.{ .k = args.matmulK, .n = args.matmulN }, args.dtype);

    const lhs: zml.Tensor = .fromShape(lhs_shape);
    const rhs: zml.Tensor = .fromShape(rhs_shape);

    log.info("Compiling MATMUL executable...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        matmul,
        .{ lhs, rhs },
        .{
            .shardings = &.{sharding},
            .program_name = "benchmark_profiler_matmul",
        },
    );
    defer exe.deinit();

    var lhs_buffer = try createFilledBuffer(allocator, io, platform, lhs_shape, sharding, 1.0);
    defer lhs_buffer.deinit();
    var rhs_buffer = try createFilledBuffer(allocator, io, platform, rhs_shape, sharding, 0.5);
    defer rhs_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ lhs_buffer, rhs_buffer });

    log.info("Running MATMUL warmup...", .{});
    try runCalls(io, &exe, exe_args, &exe_results, 1);

    const session_id = try withSessionSuffix(allocator, args.sessionId, "matmul");
    defer allocator.free(session_id);

    var profiler = try createProfiler(platform, allocator, io, args.xprofDir, session_id);
    defer profiler.deinit();

    try profiler.start();
    defer {
        if ((profiler.stop() catch unreachable)) |profile| {
            log.info("MATMUL profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
        } else {
            log.warn("MATMUL profiler extension returned no trace artifacts on this platform.", .{});
        }
    }

    log.info(
        "Profiling MATMUL compute with {d} exe.call(s) on {d}x{d} * {d}x{d}",
        .{ args.matmulCalls, args.matmulM, args.matmulK, args.matmulK, args.matmulN },
    );
    const started_at: std.Io.Timestamp = .now(io, .awake);
    try runCalls(io, &exe, exe_args, &exe_results, args.matmulCalls);
    log.info("MATMUL profiled window elapsed: {f}", .{started_at.untilNow(io, .awake)});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    try validateArgs(args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    switch (args.mode) {
        .saxpy => try runSaxpyProfile(allocator, io, platform, replicated_sharding, args),
        .matmul => try runMatmulProfile(allocator, io, platform, replicated_sharding, args),
        .both => {
            try runSaxpyProfile(allocator, io, platform, replicated_sharding, args);
            try runMatmulProfile(allocator, io, platform, replicated_sharding, args);
        },
    }

    log.info("Open traces with: bazel run //tools/xprof:xprof -- {s}", .{args.xprofDir});
}
