const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");
const Inputs = fixtures.Inputs;

pub fn main(init: std.process.Init) !void {
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

    std.debug.print("timing=synchronized_invocation_us devices={d} hkv=per_device (KV heads sharded across all visible GPUs; includes runtime, output allocation, attention, merge and wait; excludes compile, upload and output destruction)\n", .{platform.devices.len});
    var failures: usize = 0;
    for (cases.benchmarks) |c| {
        std.debug.print("CASE name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} mask={s} backend=cute\n", .{
            c.name,
            c.head_dim,
            c.batch,
            c.kv_heads,
            c.group,
            c.prefill,
            if (c.prefill > 0) c.chunk else 1,
            c.kv_len,
            @tagName(c.mask),
        });

        runBenchmark(init.gpa, init.io, platform, mesh, c) catch |err| {
            failures += 1;
            std.debug.print("FAIL error={s}\n", .{@errorName(err)});
        };
    }

    if (failures != 0) return error.BenchmarkFailed;
}

fn runBenchmark(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, c: cases.Case) !void {
    // Keep number of head per GPU constant.
    var global_case = c;
    global_case.kv_heads *= platform.devices.len;
    var inputs = Inputs.init(global_case);
    inputs.q = inputs.q.withPartitioning(mesh, .{ .hkv = .model });
    inputs.k = inputs.k.withPartitioning(mesh, .{ .hkv = .model });
    inputs.v = inputs.v.withPartitioning(mesh, .{ .hkv = .model });
    inputs.sinks = inputs.sinks.withPartitioning(mesh, .{ ._0 = .model });

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    const exe = try platform.compileFn(a, io, fixtures.forward, .{ inputs, global_case, zml.paged_attention.Backend.cute }, .{
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

    // Warmup
    for (0..5) |_| {
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        output.deinit();
    }

    var samples: [101]f64 = undefined;
    var sample_count: usize = 0;
    var accumulated_us: f64 = 0;
    while (sample_count < samples.len and (sample_count < 21 or accumulated_us < 100_000)) {
        const call_start: std.Io.Timestamp = .now(io, .awake);
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        samples[sample_count] = elapsedUs(call_start, io);
        accumulated_us += samples[sample_count];
        sample_count += 1;
        output.deinit();
    }

    std.mem.sort(f64, samples[0..sample_count], {}, std.sort.asc(f64));
    const median = if (sample_count % 2 == 0)
        (samples[sample_count / 2 - 1] + samples[sample_count / 2]) / 2
    else
        samples[sample_count / 2];

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
        sample_count,
        median,
        samples[sample_count / 10],
        samples[(sample_count - 1) * 9 / 10],
    });
}

fn elapsedUs(start: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1000;
}
