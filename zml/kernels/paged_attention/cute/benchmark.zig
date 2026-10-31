const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");
const reporting = @import("benchmark_report.zig");
const Inputs = fixtures.Inputs;

const Args = struct {
    pub const help =
        \\Usage: benchmark [--mode=benchmark|profile] [--case=INDEX] [--list-cases]
        \\                 [--warmup=5] [--iterations=101] [--output=PREFIX]
        \\
        \\benchmark: repeat unprofiled synchronized invocations; write PREFIX.csv and
        \\           PREFIX.json (including chronological latency samples).
        \\profile:   requires --case; compile/upload/warm up, then run exactly once
        \\           inside NVTX range TSL@paged_attention.profile. Use //tools:ncu.
        \\
        \\--list-cases lists zero-based indices without initializing CUDA.
        \\--output defaults to profiling-data/paged_attention_benchmark.
        \\--iterations and --output apply only to benchmark mode.
        \\
    ;

    mode: enum { benchmark, profile } = .benchmark,
    case: ?usize = null,
    list_cases: bool = false,
    warmup: usize = 5,
    iterations: usize = 101,
    output: []const u8 = "profiling-data/paged_attention_benchmark",
};

pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.list_cases) {
        for (cases.benchmarks, 0..) |c, index| printCase(c, index);
        return;
    }
    if (args.case) |index| {
        if (index >= cases.benchmarks.len) zml.stdx.flags.fatal("--case must be in [0, {d})", .{cases.benchmarks.len});
    }
    if (args.mode == .profile and args.case == null) zml.stdx.flags.fatal("--mode=profile requires --case=INDEX; use --list-cases", .{});
    if (args.warmup == 0) zml.stdx.flags.fatal("--warmup must be positive", .{});
    if (args.mode == .benchmark and args.iterations == 0) zml.stdx.flags.fatal("--iterations must be positive", .{});
    if (args.mode == .benchmark and args.output.len == 0) zml.stdx.flags.fatal("--output must be a nonempty file prefix", .{});

    const platform = try zml.Platform.auto(init.gpa, init.io, .{});
    defer platform.deinit(init.gpa, init.io);

    if (platform.target != .cuda) return error.CudaRequired;

    const mesh = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));

    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnknownComputeCapability;
    std.debug.print("PAGED_ATTENTION dtype=bf16 page_size=16 seed=42 cc={d}.{d}\n{f}\n", .{
        cc.major,
        cc.minor,
        platform.fmtVerbose(),
    });

    if (args.mode == .benchmark) {
        std.debug.print("timing=synchronized_invocation_us devices={d} hkv=per_device cache=reused_inputs_no_flush (includes runtime, output allocation, attention, merge and wait; excludes compile, upload and output destruction)\n", .{platform.devices.len});
    } else {
        std.debug.print("mode=profile nvtx=TSL@paged_attention.profile invocations=1 (latency is not measured under profiler instrumentation)\n", .{});
    }
    var results: std.ArrayList(reporting.Result) = .empty;
    defer {
        for (results.items) |r| init.gpa.free(r.samples_us);
        results.deinit(init.gpa);
    }
    var failures: usize = 0;
    for (cases.benchmarks, 0..) |c, index| {
        if (args.case) |selected| if (index != selected) continue;
        printCase(c, index);
        const result = runCase(init.gpa, init.io, platform, mesh, c, index, args) catch |err| {
            failures += 1;
            std.debug.print("FAIL error={s}\n", .{@errorName(err)});
            continue;
        };
        if (result) |r| {
            errdefer init.gpa.free(r.samples_us);
            try results.append(init.gpa, r);
        }
    }

    if (failures != 0) return error.BenchmarkFailed;
    if (args.mode == .benchmark) {
        // Bazel runs binaries in their runfiles directory. Keep relative
        // exports in the workspace when invoked through `bazel run`.
        const workspace = std.c.getenv("BUILD_WORKSPACE_DIRECTORY");
        const prefix = if (!std.fs.path.isAbsolute(args.output) and workspace != null)
            try std.Io.Dir.path.join(init.gpa, &.{ std.mem.span(workspace.?), args.output })
        else
            try init.gpa.dupe(u8, args.output);
        defer init.gpa.free(prefix);
        try (reporting.Report{
            .devices = platform.devices.len,
            .compute_capability_major = @intCast(cc.major),
            .compute_capability_minor = @intCast(cc.minor),
            .warmup = args.warmup,
            .results = results.items,
        }).write(init.gpa, init.io, prefix);
    }
}

fn printCase(c: cases.Case, index: usize) void {
    std.debug.print("CASE index={d} name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} mask={s} backend=cute\n", .{
        index, c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, @tagName(c.mask),
    });
}

fn runCase(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, c: cases.Case, index: usize, args: Args) !?reporting.Result {
    // Keep number of head per GPU constant.
    var global_case = c;
    global_case.kv_heads *= platform.devices.len;
    var inputs = Inputs.init(global_case);
    inputs.q = inputs.q.withPartitioning(mesh, .{ .hkv = .model });
    inputs.k = inputs.k.withPartitioning(mesh, .{ .hkv = .model });
    inputs.v = inputs.v.withPartitioning(mesh, .{ .hkv = .model });
    inputs.sinks = inputs.sinks.withPartitioning(mesh, .{ ._0 = .model });

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    const exe = try platform.compileFn(a, io, fixtures.forward, .{ inputs, global_case }, .{
        .program_name = "paged_attention_harness",
        .meshes = &.{mesh},
    });
    defer exe.deinit();
    const compile_us = elapsedUs(compile_start, io);

    var runner = try exe.runner(a);
    defer runner.deinit(a);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const fixture = try fixtures.Fixture.init(arena.allocator(), global_case);

    var buffers = try fixtures.upload(io, platform, inputs, fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);

    var output: zml.Buffer = undefined;

    for (0..args.warmup) |_| {
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        output.deinit();
    }

    if (args.mode == .profile) {
        // XLA may launch on worker threads, so NCU must use process-scoped
        // NVTX filtering. Waiting keeps every launch inside the capture range.
        var range = zml.tracer.span("paged_attention.profile", .{});
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        range.end();
        output.deinit();
        std.debug.print("PROFILE_COMPLETE case_index={d} compile_ms={d:.3} warmup={d} invocations=1\n", .{ index, compile_us / 1000, args.warmup });
        return null;
    }

    const samples = try a.alloc(f64, args.iterations);
    errdefer a.free(samples);
    for (samples) |*sample| {
        const call_start: std.Io.Timestamp = .now(io, .awake);
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        sample.* = elapsedUs(call_start, io);
        output.deinit();
    }

    const stats = try reporting.Stats.init(a, samples);

    std.debug.print("RESULT name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} backend=cute compile_ms={d:.3} n={d} median_us={d:.3} p10_us={d:.3} p90_us={d:.3}\n", .{
        c.name,
        c.head_dim,
        c.batch,
        c.kv_heads,
        c.group,
        c.prefill,
        if (c.prefill > 0) c.chunk else 1,
        c.kv_len,
        compile_us / 1000,
        samples.len,
        stats.median_us,
        stats.p10_us,
        stats.p90_us,
    });
    return .{ .case_index = index, .case = c, .compile_ms = compile_us / 1000, .latency = stats, .samples_us = samples };
}

fn elapsedUs(start: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1000;
}
