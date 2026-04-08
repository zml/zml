const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.benchmark_profiler);
const max_pprof_frequency_hz: u32 = 4000;

const TracyZoneCtx = extern struct {
    id: u32,
    active: i32,
};

extern fn ProfilerStart(name: [*:0]const u8) c_int;
extern fn ProfilerFlush() void;
extern fn ProfilerStop() void;
extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern fn zml_tracy_set_thread_name(name: [*:0]const u8) void;
extern fn zml_tracy_zone_begin(
    name: [*:0]const u8,
    function: [*:0]const u8,
    file: [*:0]const u8,
    line: u32,
    color: u32,
) TracyZoneCtx;
extern fn zml_tracy_zone_end(ctx: TracyZoneCtx) void;
extern fn zml_tracy_zone_text(ctx: TracyZoneCtx, text: [*]const u8, size: usize) void;
extern fn zml_tracy_zone_value(ctx: TracyZoneCtx, value: u64) void;
extern fn zml_tracy_frame_mark() void;
extern fn zml_tracy_frame_mark_named(name: [*:0]const u8) void;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const saxpy_dispatch_calls: usize = 5;

const Mode = enum {
    saxpy,
    matmul,
    both,
};

const TracyMode = enum {
    on,
    off,
};

const MatmulReinjectSide = enum {
    lhs,
    rhs,
};

const TracyScope = struct {
    enabled: bool = false,
    ctx: TracyZoneCtx = .{ .id = 0, .active = 0 },

    inline fn end(self: TracyScope) void {
        if (!self.enabled) return;
        zml_tracy_zone_end(self.ctx);
    }

    inline fn text(self: TracyScope, text_: []const u8) void {
        if (!self.enabled or text_.len == 0) return;
        zml_tracy_zone_text(self.ctx, text_.ptr, text_.len);
    }

    inline fn textFmt(self: TracyScope, comptime fmt: []const u8, args: anytype) void {
        if (!self.enabled) return;

        var buf: [160]u8 = undefined;
        const text_ = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.text(text_);
    }

    inline fn value(self: TracyScope, value_: u64) void {
        if (!self.enabled) return;
        zml_tracy_zone_value(self.ctx, value_);
    }
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
        \\   --matmulCalls=<n>           MATMUL profiled pipeline depth (default: 1)
        \\   --matmulPipelines=<m>       MATMUL profiled pipeline repeats (default: 1)
        \\   --tracy=<on|off>            enable Tracy client zones and frame marks (default: on)
        \\   --xprofDir=<path>           xprof repository path (default: /tmp/xprof)
        \\   --pprofDir=<path>           pprof output directory, empty disables (default: /tmp/pprof)
        \\   --pprofFrequency=<hz>       pprof samples per second, capped by gperftools at 4000 (default: 4000)
        \\   --pprofDurationMs=<ms>      repeated sampling window for pprof capture (default: 250)
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
    matmulPipelines: usize = 1,

    tracy: TracyMode = .on,
    xprofDir: []const u8 = "/tmp/xprof",
    pprofDir: []const u8 = "/tmp/pprof",
    pprofFrequency: u32 = max_pprof_frequency_hz,
    pprofDurationMs: u32 = 250,
    sessionId: []const u8 = "benchmark-profiler",
};

fn validateArgs(args: CliArgs) !void {
    if (!args.dtype.isFloat()) {
        log.err("Only floating-point dtypes are supported, got {s}", .{@tagName(args.dtype)});
        return error.InvalidDType;
    }
    if (args.xprofDir.len == 0) return error.InvalidXprofDir;
    if (args.pprofDir.len != 0 and args.pprofFrequency == 0) return error.InvalidPprofFrequency;
    if (args.pprofDir.len != 0 and args.pprofDurationMs == 0) return error.InvalidPprofDuration;
    if (args.sessionId.len == 0) return error.InvalidSessionId;

    switch (args.mode) {
        .saxpy => {
            if (args.saxpySize == 0) return error.InvalidSaxpySize;
        },
        .matmul => {
            if (args.matmulM == 0 or args.matmulK == 0 or args.matmulN == 0) return error.InvalidMatmulShape;
            if (args.matmulCalls == 0) return error.InvalidMatmulCalls;
            if (args.matmulPipelines == 0) return error.InvalidMatmulPipelines;
            if (args.matmulK != args.matmulN and args.matmulM != args.matmulK) return error.InvalidMatmulPipelineShape;
            _ = try matmulTotalCalls(args);
        },
        .both => {
            if (args.saxpySize == 0) return error.InvalidSaxpySize;
            if (args.matmulM == 0 or args.matmulK == 0 or args.matmulN == 0) return error.InvalidMatmulShape;
            if (args.matmulCalls == 0) return error.InvalidMatmulCalls;
            if (args.matmulPipelines == 0) return error.InvalidMatmulPipelines;
            if (args.matmulK != args.matmulN and args.matmulM != args.matmulK) return error.InvalidMatmulPipelineShape;
            _ = try matmulTotalCalls(args);
        },
    }
}

fn matmulTotalCalls(args: CliArgs) !usize {
    return std.math.mul(usize, args.matmulCalls, args.matmulPipelines) catch error.InvalidMatmulTotalCalls;
}

fn saxpy(a: zml.Tensor, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
    return x.mul(a).add(y);
}

fn matmul(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.dot(rhs, .k);
}

inline fn tracyEnabled(args: CliArgs) bool {
    return args.tracy == .on;
}

inline fn tracyZoneNamed(comptime name: [:0]const u8, enabled: bool) TracyScope {
    if (!enabled) return .{};
    const src = @src();
    return .{
        .enabled = true,
        .ctx = zml_tracy_zone_begin(name.ptr, src.fn_name.ptr, src.file.ptr, src.line, 0),
    };
}

inline fn tracyFrameMark(enabled: bool) void {
    if (!enabled) return;
    zml_tracy_frame_mark();
}

inline fn tracyFrameMarkNamed(comptime name: [:0]const u8, enabled: bool) void {
    if (!enabled) return;
    zml_tracy_frame_mark_named(name.ptr);
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

const PprofProfile = struct {
    allocator: std.mem.Allocator,
    file_path: ?[:0]u8 = null,
    frequency_hz: u32 = max_pprof_frequency_hz,
    started: bool = false,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        pprof_dir: []const u8,
        session_id: []const u8,
        frequency_hz: u32,
    ) !PprofProfile {
        var profile: PprofProfile = .{
            .allocator = allocator,
            .frequency_hz = frequency_hz,
        };
        if (pprof_dir.len == 0) return profile;

        try std.Io.Dir.createDirPath(.cwd(), io, pprof_dir);

        const basename = try std.fmt.allocPrint(allocator, "{s}.prof", .{session_id});
        defer allocator.free(basename);

        profile.file_path = try std.Io.Dir.path.joinZ(allocator, &.{ pprof_dir, basename });
        return profile;
    }

    fn deinit(self: *PprofProfile) void {
        self.stop();
        if (self.file_path) |file_path| self.allocator.free(file_path);
        self.file_path = null;
    }

    fn start(self: *PprofProfile) !void {
        const file_path = self.file_path orelse return;
        const frequency = try std.fmt.allocPrint(self.allocator, "{d}", .{@min(self.frequency_hz, max_pprof_frequency_hz)});
        defer self.allocator.free(frequency);
        const frequency_z = try self.allocator.dupeZ(u8, frequency);
        defer self.allocator.free(frequency_z);
        if (setenv("CPUPROFILE_FREQUENCY", frequency_z.ptr, 1) != 0) return error.PprofSetFrequencyFailed;
        if (ProfilerStart(file_path.ptr) == 0) return error.PprofStartFailed;
        self.started = true;
    }

    fn stop(self: *PprofProfile) void {
        if (!self.started) return;
        ProfilerFlush();
        ProfilerStop();
        self.started = false;
    }

    fn outputPath(self: *const PprofProfile) ?[:0]const u8 {
        return self.file_path;
    }
};

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

fn runCallAndAwait(
    io: std.Io,
    exe: *const zml.Exe,
    exe_args: zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    tracy_enabled: bool,
) !void {
    const tracy_zone = tracyZoneNamed("dispatch call", tracy_enabled);
    defer tracy_zone.end();
    {
        const call_zone = tracyZoneNamed("exe.call", tracy_enabled);
        defer call_zone.end();
        exe.call(exe_args, exe_results);
    }
    var out = exe_results.get(zml.Buffer);
    defer out.deinit();
    {
        const await_zone = tracyZoneNamed("buffer.await", tracy_enabled);
        defer await_zone.end();
        _ = try out.await(io);
    }
}

fn runCalls(
    io: std.Io,
    exe: *const zml.Exe,
    exe_args: zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    call_count: usize,
    comptime tracy_zone_name: ?[:0]const u8,
    tracy_enabled: bool,
) !void {
    const tracy_zone: TracyScope = if (tracy_zone_name) |name| tracyZoneNamed(name, tracy_enabled) else .{};
    defer tracy_zone.end();
    tracy_zone.value(call_count);

    for (0..call_count) |_| {
        try runCallAndAwait(io, exe, exe_args, exe_results, tracy_enabled);
    }
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn selectMatmulReinjectSide(args: CliArgs) MatmulReinjectSide {
    if (args.matmulK == args.matmulN) return .lhs;
    return .rhs;
}

fn runMatmulPipelineCalls(
    io: std.Io,
    exe: *const zml.Exe,
    exe_args: *zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    lhs_buffer: *zml.Buffer,
    rhs_buffer: *zml.Buffer,
    out_slice: zml.Slice,
    call_count: usize,
    comptime tracy_zone_name: ?[:0]const u8,
    tracy_enabled: bool,
) !void {
    const tracy_zone: TracyScope = if (tracy_zone_name) |name| tracyZoneNamed(name, tracy_enabled) else .{};
    defer tracy_zone.end();
    tracy_zone.value(call_count);

    var out = lhs_buffer.*;
    for (0..call_count) |call_index| {
        const call_zone = tracyZoneNamed("matmul pipeline call", tracy_enabled);
        defer call_zone.end();
        call_zone.value(call_index + 1);

        exe_args.set(.{ out, rhs_buffer.* });
        {
            const queue_zone = tracyZoneNamed("exe.call", tracy_enabled);
            defer queue_zone.end();
            exe.call(exe_args.*, exe_results);
        }
        out = exe_results.get(zml.Buffer);
        {
            const copy_zone = tracyZoneNamed("buffer.toSlice", tracy_enabled);
            defer copy_zone.end();
            try out.toSlice(io, out_slice);
        }
        // _ = try out.await(io);
    }
}

fn runSaxpyProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    args: CliArgs,
) !void {
    const tracy_enabled = tracyEnabled(args);
    const tracy_zone = tracyZoneNamed("runSaxpyProfile", tracy_enabled);
    defer tracy_zone.end();
    tracy_zone.textFmt("dtype={s} n={d}", .{ @tagName(args.dtype), args.saxpySize });

    const alpha_shape = zml.Shape.init(.{}, args.dtype);
    const vector_shape = zml.Shape.init(.{ .n = args.saxpySize }, args.dtype);

    const alpha: zml.Tensor = .fromShape(alpha_shape);
    const x: zml.Tensor = .fromShape(vector_shape);
    const y: zml.Tensor = .fromShape(vector_shape);

    log.info("Compiling SAXPY executable...", .{});
    var exe = blk: {
        const compile_zone = tracyZoneNamed("saxpy compile", tracy_enabled);
        defer compile_zone.end();
        break :blk try platform.compileFn(
            allocator,
            io,
            saxpy,
            .{ alpha, x, y },
            .{
                .shardings = &.{sharding},
                .program_name = "benchmark_profiler_saxpy",
            },
        );
    };
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
    try runCalls(io, &exe, exe_args, &exe_results, 1, "saxpy warmup dispatch batch", tracy_enabled);
    tracyFrameMarkNamed("saxpy warmup", tracy_enabled);

    const session_id = try withSessionSuffix(allocator, args.sessionId, "saxpy");
    defer allocator.free(session_id);

    var profiler = try createProfiler(platform, allocator, io, args.xprofDir, session_id);
    defer profiler.deinit();
    var pprof = try PprofProfile.init(allocator, io, args.pprofDir, session_id, args.pprofFrequency);
    defer pprof.deinit();
    var xprof_stopped = false;

    try profiler.start();
    errdefer {
        if (!xprof_stopped) {
            if ((profiler.stop() catch unreachable)) |profile| {
                log.info("SAXPY profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
            } else {
                log.warn("SAXPY profiler extension returned no trace artifacts on this platform.", .{});
            }
        }
    }

    log.info("Profiling SAXPY dispatch with {d} sequential exe.call invocations", .{saxpy_dispatch_calls});
    const started_at: std.Io.Timestamp = .now(io, .awake);
    {
        const profile_zone = tracyZoneNamed("saxpy profile window", tracy_enabled);
        defer profile_zone.end();
        profile_zone.textFmt("dispatch_calls={d}", .{saxpy_dispatch_calls});
        profile_zone.value(saxpy_dispatch_calls);
        try runCalls(io, &exe, exe_args, &exe_results, saxpy_dispatch_calls, "saxpy dispatch batch", tracy_enabled);
    }
    tracyFrameMarkNamed("saxpy profile window", tracy_enabled);
    log.info("SAXPY profiled window elapsed: {f}", .{started_at.untilNow(io, .awake)});
    if ((try profiler.stop())) |profile| {
        log.info("SAXPY profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
    } else {
        log.warn("SAXPY profiler extension returned no trace artifacts on this platform.", .{});
    }
    xprof_stopped = true;

    if (pprof.outputPath() != null) {
        log.info(
            "Capturing SAXPY pprof CPU samples over repeated dispatch windows for ~{d}ms at up to {d} Hz",
            .{ args.pprofDurationMs, @min(args.pprofFrequency, max_pprof_frequency_hz) },
        );
        try pprof.start();
        errdefer pprof.stop();

        const pprof_started_at: std.Io.Timestamp = .now(io, .awake);
        const target_ns = @as(u64, args.pprofDurationMs) * std.time.ns_per_ms;
        while (pprof_started_at.untilNow(io, .awake).toNanoseconds() < target_ns) {
            try runCalls(io, &exe, exe_args, &exe_results, saxpy_dispatch_calls, "saxpy pprof dispatch batch", tracy_enabled);
            tracyFrameMark(tracy_enabled);
        }
        pprof.stop();
        if (pprof.outputPath()) |file_path| {
            log.info("SAXPY pprof CPU profile dumped: {s}", .{file_path});
        }
    }
}

fn runMatmulProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    args: CliArgs,
) !void {
    const tracy_enabled = tracyEnabled(args);
    const tracy_zone = tracyZoneNamed("runMatmulProfile", tracy_enabled);
    defer tracy_zone.end();
    tracy_zone.textFmt(
        "dtype={s} lhs={d}x{d} rhs={d}x{d}",
        .{ @tagName(args.dtype), args.matmulM, args.matmulK, args.matmulK, args.matmulN },
    );

    const total_calls = try matmulTotalCalls(args);

    const lhs_shape = zml.Shape.init(.{ .m = args.matmulM, .k = args.matmulK }, args.dtype);
    const rhs_shape = zml.Shape.init(.{ .k = args.matmulK, .n = args.matmulN }, args.dtype);

    const lhs: zml.Tensor = .fromShape(lhs_shape);
    const rhs: zml.Tensor = .fromShape(rhs_shape);

    log.info("Compiling MATMUL executable...", .{});
    var exe = blk: {
        const compile_zone = tracyZoneNamed("matmul compile", tracy_enabled);
        defer compile_zone.end();
        break :blk try platform.compileFn(
            allocator,
            io,
            matmul,
            .{ lhs, rhs },
            .{
                .shardings = &.{sharding},
                .program_name = "benchmark_profiler_matmul",
            },
        );
    };
    defer exe.deinit();

    var lhs_buffer = try createFilledBuffer(allocator, io, platform, lhs_shape, sharding, 1.0);
    defer lhs_buffer.deinit();
    var rhs_buffer = try createFilledBuffer(allocator, io, platform, rhs_shape, sharding, 0.5);
    defer rhs_buffer.deinit();
    const out_shape = zml.Shape.init(.{ .m = args.matmulM, .n = args.matmulN }, args.dtype);
    var zml_allocator = zml.mem.DmaAllocator.init(allocator, &platform.devices[0]);
    const out_slice = try zml.Slice.alloc(zml_allocator.allocator(), out_shape);
    defer out_slice.free(zml_allocator.allocator());

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    // log.info("Running MATMUL warmup...", .{});
    // try runMatmulPipelineCalls(
    //     io,
    //     &exe,
    //     &exe_args,
    //     &exe_results,
    //     &lhs_buffer,
    //     &rhs_buffer,
    //     reinject_side,
    //     1,
    // );

    const session_id = try withSessionSuffix(allocator, args.sessionId, "matmul");
    defer allocator.free(session_id);

    var profiler = try createProfiler(platform, allocator, io, args.xprofDir, session_id);
    defer profiler.deinit();
    var pprof = try PprofProfile.init(allocator, io, args.pprofDir, session_id, args.pprofFrequency);
    defer pprof.deinit();
    var xprof_stopped = false;

    try profiler.start();
    errdefer {
        if (!xprof_stopped) {
            if ((profiler.stop() catch unreachable)) |profile| {
                log.info("MATMUL profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
            } else {
                log.warn("MATMUL profiler extension returned no trace artifacts on this platform.", .{});
            }
        }
    }

    log.info(
        "Profiling MATMUL compute pipeline with {d} call(s)/pipeline x {d} pipeline(s) = {d} total exe.call(s) on {d}x{d} * {d}x{d}",
        .{
            args.matmulCalls,
            args.matmulPipelines,
            total_calls,
            args.matmulM,
            args.matmulK,
            args.matmulK,
            args.matmulN,
        },
    );
    const started_at: std.Io.Timestamp = .now(io, .awake);
    {
        const profile_zone = tracyZoneNamed("matmul profile window", tracy_enabled);
        defer profile_zone.end();
        profile_zone.textFmt("pipelines={d} calls_per_pipeline={d}", .{ args.matmulPipelines, args.matmulCalls });
        profile_zone.value(total_calls);
        for (0..args.matmulPipelines) |pipeline_index| {
            const pipeline_zone = tracyZoneNamed("matmul profile pipeline", tracy_enabled);
            defer pipeline_zone.end();
            pipeline_zone.value(pipeline_index + 1);
            try runMatmulPipelineCalls(
                io,
                &exe,
                &exe_args,
                &exe_results,
                &lhs_buffer,
                &rhs_buffer,
                out_slice,
                args.matmulCalls,
                "matmul profile dispatch batch",
                tracy_enabled,
            );
            tracyFrameMark(tracy_enabled);
        }
    }
    tracyFrameMarkNamed("matmul profile window", tracy_enabled);
    log.info("MATMUL profiled window elapsed: {f}", .{started_at.untilNow(io, .awake)});
    if ((try profiler.stop())) |profile| {
        log.info("MATMUL profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
    } else {
        log.warn("MATMUL profiler extension returned no trace artifacts on this platform.", .{});
    }
    xprof_stopped = true;

    if (pprof.outputPath() != null) {
        log.info(
            "Capturing MATMUL pprof CPU samples over repeated pipelines for ~{d}ms at up to {d} Hz",
            .{ args.pprofDurationMs, @min(args.pprofFrequency, max_pprof_frequency_hz) },
        );
        try pprof.start();
        errdefer pprof.stop();

        const pprof_started_at: std.Io.Timestamp = .now(io, .awake);
        const target_ns = @as(u64, args.pprofDurationMs) * std.time.ns_per_ms;
        while (pprof_started_at.untilNow(io, .awake).toNanoseconds() < target_ns) {
            for (0..args.matmulPipelines) |pipeline_index| {
                const pipeline_zone = tracyZoneNamed("matmul pprof pipeline", tracy_enabled);
                defer pipeline_zone.end();
                pipeline_zone.value(pipeline_index + 1);
                try runMatmulPipelineCalls(
                    io,
                    &exe,
                    &exe_args,
                    &exe_results,
                    &lhs_buffer,
                    &rhs_buffer,
                    out_slice,
                    args.matmulCalls,
                    "matmul pprof dispatch batch",
                    tracy_enabled,
                );
                tracyFrameMark(tracy_enabled);
            }
        }
        pprof.stop();
        if (pprof.outputPath()) |file_path| {
            log.info("MATMUL pprof CPU profile dumped: {s}", .{file_path});
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    try validateArgs(args);
    if (tracyEnabled(args)) {
        zml_tracy_set_thread_name("benchmark_profiler");
    }

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
    if (args.pprofDir.len != 0) {
        switch (args.mode) {
            .saxpy => log.info(
                "Open CPU profile with: bazel run //tools/pprof:pprof -- -http=: {s}/{s}-saxpy.prof",
                .{ args.pprofDir, args.sessionId },
            ),
            .matmul => log.info(
                "Open CPU profile with: bazel run //tools/pprof:pprof -- -http=: {s}/{s}-matmul.prof",
                .{ args.pprofDir, args.sessionId },
            ),
            .both => {
                log.info(
                    "Open SAXPY CPU profile with: bazel run //tools/pprof:pprof -- -http=: {s}/{s}-saxpy.prof",
                    .{ args.pprofDir, args.sessionId },
                );
                log.info(
                    "Open MATMUL CPU profile with: bazel run //tools/pprof:pprof -- -http=: {s}/{s}-matmul.prof",
                    .{ args.pprofDir, args.sessionId },
                );
            },
        }
    }
    if (tracyEnabled(args)) {
        log.info("Monitor live traces with: bazel run //tools/tracy:tracy-profiler --", .{});
    }
}
