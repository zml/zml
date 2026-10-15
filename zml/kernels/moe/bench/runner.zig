//! Compile a graph once, prepare weights per seed, validate, then sample completed API calls.
const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const workload = @import("workload.zig");
const results = @import("results.zig");
const Record = results.Record;
const Comparison = results.Comparison;

pub const Compiled = struct {
    setup: zml.Exe,
    timed: zml.Exe,
    reference: ?zml.Exe,
    template: workload.Generated,
    compile_ns: i64,

    pub fn deinit(c: *const Compiled) void {
        if (c.reference) |ref| ref.deinit();
        c.timed.deinit();
        c.setup.deinit();
    }
};

pub fn compile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, experts_mesh: *const zml.Mesh, w: cases.Workload, check: bool) !Compiled {
    const start: std.Io.Timestamp = .now(io, .awake);
    const c = w.case;
    var global_case = c;
    global_case.experts = w.global_experts;
    const setup = try platform.compileFn(allocator, io, workload.generate, .{
        zml.Tensor.init(.{}, .u32), global_case, w.data, check, w.isSharded(),
    }, .{ .meshes = &.{experts_mesh} });
    errdefer setup.deinit();
    const template = workload.withOutputShapes(workload.templates(w, check), setup.output_shapes);
    const timed = try platform.compileFn(allocator, io, zml.moe.forwardMoe, .{
        zml.Tensor.init(.{ .t = c.tokens, .d = c.width }, c.activation_dtype),
        zml.Tensor.init(.{ .t = c.tokens, .topk = c.topk }, .i32),
        zml.Tensor.init(.{ .t = c.tokens, .topk = c.topk }, .f32),
        template.prepared.gate_up,
        template.prepared.down,
        c.backend,
        c.options(),
    }, .{ .meshes = &.{experts_mesh} });
    errdefer timed.deinit();
    // One reference row bounds the gathered expert matrices independently of prefill size.
    const reference = if (template.canonical) |canonical| try platform.compileFn(allocator, io, zml.moe.stablehlo.reference, .{
        zml.Tensor.init(.{ .t = 1, .d = c.width }, c.activation_dtype),
        zml.Tensor.init(.{ .t = 1, .topk = c.topk }, .i32),
        zml.Tensor.init(.{ .t = 1, .topk = c.topk }, .f32),
        canonical.gate_up,
        canonical.down,
        c.referenceOptions(platform.target),
    }, .{ .meshes = &.{experts_mesh} }) else null;
    return .{ .setup = setup, .timed = timed, .reference = reference, .template = template, .compile_ns = @intCast(start.untilNow(io, .awake).toNanoseconds()) };
}

pub const Prepared = struct {
    seed: u64,
    buffers: zml.Bufferized(workload.Generated),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, compiled: *const Compiled, w: cases.Workload) !Prepared {
        const seed: u32 = if (w.data == .fixture) 0 else @truncate(w.seed ^ (w.seed >> 32));
        var seed_buffer = try zml.Buffer.fromBytes(io, platform, .init(.{}, .u32), std.mem.asBytes(&seed));
        defer seed_buffer.deinit();
        var buffers = try zml.mem.bufferize(allocator, workload.Generated, &compiled.template);
        var setup_runner = try compiled.setup.runner(allocator);
        defer setup_runner.deinit(allocator);
        setup_runner.run(io, .{seed_buffer}, .{&buffers}, .{ .wait = true });
        return .{ .seed = w.seed, .buffers = buffers };
    }

    pub fn deinit(p: *Prepared) void {
        zml.Buffer.deinitAll(workload.Generated, &p.buffers);
    }
};

pub fn runCase(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, args: cases.Args, compiled: *const Compiled, prepared: *const Prepared, record: *Record) !void {
    const w = record.workload;
    const c = w.case;
    const routes = try workload.generateRoutes(allocator, w);
    defer routes.deinit(allocator);
    const host = try workload.makeInput(allocator, w);
    defer host.free(allocator);
    var x = try zml.Buffer.fromSlice(io, platform, host);
    defer x.deinit();
    var ids = try zml.Buffer.fromBytes(io, platform, .init(.{ .t = c.tokens, .topk = c.topk }, .i32), std.mem.sliceAsBytes(routes.ids));
    defer ids.deinit();
    var weights = try zml.Buffer.fromBytes(io, platform, .init(.{ .t = c.tokens, .topk = c.topk }, .f32), std.mem.sliceAsBytes(routes.weights));
    defer weights.deinit();
    var runner = try compiled.timed.runner(allocator);
    defer runner.deinit(allocator);
    const inputs = .{ x, ids, weights, prepared.buffers.prepared };
    var output: zml.Buffer = undefined;
    runner.run(io, inputs, .{&output}, .{ .wait = true });
    defer output.deinit();
    const host_output = try output.toSliceAlloc(allocator, io);
    defer host_output.free(allocator);
    var nonfinite: usize = 0;
    var sum_abs: f64 = 0;
    for (0..host_output.shape.count()) |i| {
        const v = valueAt(c.activation_dtype, host_output.constData(), i);
        if (std.math.isFinite(v)) sum_abs += @abs(v) else nonfinite += 1;
    }
    const projections: f64 = if (c.activation == .relu or c.activation == .silu or c.activation == .gelu) 2 else 3;
    const matrix_elements = projections * @as(f64, @floatFromInt(c.width)) * @as(f64, @floatFromInt(c.intermediate));
    var stored_bytes: usize = 0;
    const CountBytes = struct {
        fn visit(total: *usize, buffer: *const zml.Buffer) void {
            total.* += buffer.shape().byteSize();
        }
    };
    zml.meta.visit(CountBytes.visit, &stored_bytes, &prepared.buffers.prepared);
    var output_mismatch = false;
    if (args.compare) |path| {
        const metadata = try readDumpMetadata(allocator, io, path);
        defer metadata.deinit();
        try metadata.value.validate(allocator, w, host_output.constData().len);
        record.dump_source = try copySource(allocator, metadata.value.source);
        const bytes = try zml.stdx.Io.Dir.readFileAlloc(.cwd(), io, path, allocator, .limited(host_output.constData().len + 1));
        defer allocator.free(bytes);
        record.dump_comparison = .{};
        try compareValues(c, host_output.constData(), bytes, &record.dump_comparison.?);
        output_mismatch = record.dump_comparison.?.outside_tolerance != 0;
    }
    if (args.dump) |path| {
        const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
        defer file.close(io);
        try file.writePositionalAll(io, host_output.constData(), 0);
        try writeDumpMetadata(allocator, io, path, w, host_output.constData().len);
    }
    if (output_mismatch) return error.OutputMismatch;
    if (compiled.reference) |*ref| {
        record.reference_check = .{};
        try checkReference(allocator, io, platform, w, ref, prepared.buffers.canonical.?, host, routes, host_output, &record.reference_check.?);
    }
    if (nonfinite != 0) return error.NonFiniteOutput;
    const timing = try measure(allocator, io, &runner, inputs, args);
    const wall_us = timing.median_us;
    record.statistics = .{
        .timing = timing,
        .local_routes = routes.local_routes,
        .touched_experts = routes.touched,
        .maximum_routes = routes.maximum,
        .padded_rows = routes.padded_rows,
        .implied_tflops = 2 * matrix_elements * @as(f64, @floatFromInt(routes.local_routes)) / wall_us / 1e6,
        .implied_weight_gbs = @as(f64, @floatFromInt(stored_bytes)) * @as(f64, @floatFromInt(routes.touched)) / @as(f64, @floatFromInt(w.global_experts)) / wall_us / 1e3,
        .output_mean_absolute = sum_abs / @as(f64, @floatFromInt(host_output.shape.count())),
        .nonfinite = nonfinite,
    };
    record.status = if (compiled.reference != null or args.compare != null) .passed else .measured;
}

fn measureBatch(io: std.Io, runner: *zml.Exe.Runner, inputs: anytype, iterations: u32) f64 {
    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..iterations) |_| {
        var output: zml.Buffer = undefined;
        // Measure completed public-API calls, without relying on backend queue ordering.
        runner.run(io, inputs, .{&output}, .{ .wait = true });
        output.deinit();
    }
    return @floatFromInt(start.untilNow(io, .awake).toNanoseconds());
}

fn measure(allocator: std.mem.Allocator, io: std.Io, runner: *zml.Exe.Runner, inputs: anytype, args: cases.Args) !results.Timing {
    if (args.warmup != 0) _ = measureBatch(io, runner, inputs, args.warmup);
    const target_ns: f64 = @as(f64, @floatFromInt(args.sample_ms)) * 1e6;
    const samples = try allocator.alloc(f64, args.samples);
    errdefer allocator.free(samples);
    var iterations = args.iters;
    var accepted: usize = 0;
    while (accepted < samples.len) {
        const elapsed = measureBatch(io, runner, inputs, iterations);
        if (elapsed < target_ns) {
            const next = @as(f64, @floatFromInt(iterations)) * @min(10, 1.2 * target_ns / @max(elapsed, 1));
            if (iterations == std.math.maxInt(u32) or next > std.math.maxInt(u32)) return error.TooManyTimingIterations;
            iterations = @max(iterations + 1, @as(u32, @intFromFloat(@ceil(next))));
            // A warmed kernel can get faster during sampling. Restart so every
            // retained sample meets the duration and uses the same iteration count.
            accepted = 0;
            continue;
        }
        samples[accepted] = elapsed / @as(f64, @floatFromInt(iterations)) / 1000;
        accepted += 1;
    }
    return results.Timing.summarize(allocator, samples, iterations);
}

const DumpMetadata = struct {
    schema_version: u32 = 1,
    workload_id: []const u8,
    dtype: zml.DataType,
    shape: [2]i64,
    byte_count: usize,
    source: results.Source,

    fn validate(metadata: DumpMetadata, allocator: std.mem.Allocator, w: cases.Workload, bytes: usize) !void {
        if (metadata.schema_version != 1) return error.UnsupportedDumpVersion;
        const id = try w.id(allocator);
        defer allocator.free(id);
        if (!std.mem.eql(u8, metadata.workload_id, id) or metadata.dtype != w.case.activation_dtype or
            metadata.shape[0] != w.case.tokens or metadata.shape[1] != w.case.width or metadata.byte_count != bytes)
            return error.DumpWorkloadMismatch;
    }
};

fn writeDumpMetadata(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: cases.Workload, bytes: usize) !void {
    const metadata_path = try std.fmt.allocPrint(allocator, "{s}.json", .{path});
    defer allocator.free(metadata_path);
    const id = try w.id(allocator);
    defer allocator.free(id);
    const metadata: DumpMetadata = .{
        .workload_id = id,
        .dtype = w.case.activation_dtype,
        .shape = .{ w.case.tokens, w.case.width },
        .byte_count = bytes,
        .source = results.source(),
    };
    const file = try std.Io.Dir.createFile(.cwd(), io, metadata_path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try results.writeJsonLine(&writer.interface, metadata);
}

fn readDumpMetadata(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(DumpMetadata) {
    const metadata_path = try std.fmt.allocPrint(allocator, "{s}.json", .{path});
    defer allocator.free(metadata_path);
    const bytes = try zml.stdx.Io.Dir.readFileAlloc(.cwd(), io, metadata_path, allocator, .limited(64 * 1024));
    defer allocator.free(bytes);
    return std.json.parseFromSlice(DumpMetadata, allocator, bytes, .{ .allocate = .alloc_always });
}

fn copySource(allocator: std.mem.Allocator, source: results.Source) !results.Source {
    const revision = try allocator.dupe(u8, source.revision);
    errdefer allocator.free(revision);
    const hash = try allocator.dupe(u8, source.source_hash);
    errdefer allocator.free(hash);
    const zig_version = try allocator.dupe(u8, source.zig_version);
    return .{ .revision = revision, .dirty = source.dirty, .source_hash = hash, .zig_version = zig_version, .optimization = source.optimization };
}

fn valueAt(dtype: zml.DataType, bytes: []const u8, index: usize) f64 {
    return switch (dtype) {
        .bf16 => std.mem.bytesAsSlice(zml.floats.BFloat16, bytes)[index].toF32(),
        .f16 => std.mem.bytesAsSlice(f16, bytes)[index],
        .f32 => std.mem.bytesAsSlice(f32, bytes)[index],
        else => unreachable,
    };
}

fn compareValues(c: cases.Case, actual: []const u8, expected: []const u8, result: *Comparison) !void {
    if (actual.len != expected.len) return error.OutputSizeMismatch;
    const bytes = c.activation_dtype.sizeOf();
    const tolerance = c.tolerance();
    for (0..actual.len / bytes) |i| {
        const a = valueAt(c.activation_dtype, actual, i);
        const e = valueAt(c.activation_dtype, expected, i);
        result.elements += 1;
        result.exact += @intFromBool(std.mem.eql(u8, actual[i * bytes ..][0..bytes], expected[i * bytes ..][0..bytes]));
        if (!std.math.isFinite(a) or !std.math.isFinite(e)) {
            result.outside_tolerance += 1;
            continue;
        }
        const diff = @abs(a - e);
        result.max_absolute_error = @max(result.max_absolute_error, diff);
        result.sum_squared_error += diff * diff;
        result.sum_squared_reference += e * e;
        if (diff > tolerance.absolute_tolerance + tolerance.relative_tolerance * @max(@abs(a), @abs(e))) result.outside_tolerance += 1;
    }
}

fn checkReference(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, w: cases.Workload, ref: *const zml.Exe, canonical: zml.Bufferized(workload.Experts), host: zml.Slice, routes: workload.Routes, output: zml.Slice, comparison: *Comparison) !void {
    const c = w.case;
    const row_bytes: usize = @intCast(c.width * c.activation_dtype.sizeOf());
    const k: usize = @intCast(c.topk);
    var runner = try ref.runner(allocator);
    defer runner.deinit(allocator);
    for (0..@intCast(c.tokens)) |t| {
        var x = try zml.Buffer.fromBytes(io, platform, .init(.{ .t = 1, .d = c.width }, c.activation_dtype), host.constData()[t * row_bytes ..][0..row_bytes]);
        defer x.deinit();
        var ids = try zml.Buffer.fromBytes(io, platform, .init(.{ .t = 1, .topk = c.topk }, .i32), std.mem.sliceAsBytes(routes.ids[t * k ..][0..k]));
        defer ids.deinit();
        var weights = try zml.Buffer.fromBytes(io, platform, .init(.{ .t = 1, .topk = c.topk }, .f32), std.mem.sliceAsBytes(routes.weights[t * k ..][0..k]));
        defer weights.deinit();
        var expected: zml.Buffer = undefined;
        runner.run(io, .{ x, ids, weights, canonical }, .{&expected}, .{ .wait = true });
        defer expected.deinit();
        const expected_host = try expected.toSliceAlloc(allocator, io);
        defer expected_host.free(allocator);
        try compareValues(c, output.constData()[t * row_bytes ..][0..row_bytes], expected_host.constData(), comparison);
    }
    if (comparison.outside_tolerance != 0) return error.ReferenceMismatch;
}

pub fn unavailable(w: cases.Workload, platform: *const zml.Platform, experts_mesh: *const zml.Mesh) ?[]const u8 {
    if (w.unsupported(platform.target)) |reason| return reason;
    if (w.isSharded() and @divExact(w.global_experts, w.case.experts) != experts_mesh.numPartitionsForLogicalAxis(.experts)) return "global-experts/local-experts must match the device mesh partition count";
    if (!w.case.backend.isAvailable(platform)) return "backend unavailable on this machine";
    if (w.case.backend == .triton and w.case.scheme == .mxfp4 and platform.target == .cuda and !zml.moe.triton.isAvailable(platform)) return "CUDA Triton MXFP4 requires a supported Blackwell GPU";
    if (w.case.backend == .flashinfer_cutlass and w.case.scheme == .nvfp4 and !zml.moe.cutlass_flashinfer.isNvfp4Supported(platform)) return "NVFP4 unavailable on this architecture";
    return null;
}

test "comparison uses the compliance tolerance and rejects nonfinite values" {
    const c: cases.Case = .{ .backend = .stablehlo, .activation_dtype = .f32 };
    var comparison: Comparison = .{};
    try compareValues(c, std.mem.sliceAsBytes(&[_]f32{ 1, 1.03, 2 }), std.mem.sliceAsBytes(&[_]f32{ 1, 1, 1 }), &comparison);
    try std.testing.expectEqual(@as(usize, 1), comparison.exact);
    try std.testing.expectEqual(@as(usize, 1), comparison.outside_tolerance);
    try compareValues(c, std.mem.sliceAsBytes(&[_]f32{std.math.nan(f32)}), std.mem.sliceAsBytes(&[_]f32{0}), &comparison);
    try std.testing.expectEqual(@as(usize, 2), comparison.outside_tolerance);
    try std.testing.expectError(error.OutputSizeMismatch, compareValues(c, &.{0}, &.{}, &comparison));
}

test "benchmark executes the public MoE API with sharded experts" {
    if (zml.testing.env().target != .cpu) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 2 } });
    defer platform.deinit(allocator, io);
    const experts_mesh = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth }));
    try std.testing.expectEqual(@as(i64, 2), experts_mesh.numPartitionsForLogicalAxis(.experts));
    inline for (.{ @as(?zml.Quantization.Scheme, null), @as(?zml.Quantization.Scheme, .fp8_per_channel) }) |scheme| {
        inline for (.{ zml.moe.RoutingWeightPlacement.before_down, zml.moe.RoutingWeightPlacement.after_down }) |placement| {
            const w: cases.Workload = .{
                .case = .{
                    .backend = .stablehlo,
                    .scheme = scheme,
                    .activation = .{ .swiglu = .{ .limit = 0.25 } },
                    .tokens = 2,
                    .topk = 3,
                    .experts = 2,
                    .bias = true,
                    .placement = placement,
                },
                .routing = .balanced,
                .global_experts = 4,
                .active_experts = 0,
                .sigma = 0.6,
                .seed = 1,
                .data = if (scheme == null) .fixture else .random,
            };
            try std.testing.expect(unavailable(w, platform, experts_mesh) == null);
            var mismatched = w;
            mismatched.global_experts = 6;
            try std.testing.expect(unavailable(mismatched, platform, experts_mesh) != null);
            const compiled = try compile(allocator, io, platform, experts_mesh, w, true);
            defer compiled.deinit();
            try std.testing.expectEqual(@as(i64, 4), compiled.timed.input_shapes[3].dim(.expert));
            try std.testing.expect(compiled.timed.input_shapes[3].partition(.expert).isSharded());
            if (compiled.template.prepared.gate_up.quantization) |q| {
                try std.testing.expect(q.scales.shape().partition(0).isSharded());
                try std.testing.expect(!q.scales.shape().partition(2).isSharded());
            }
            var prepared = try Prepared.init(allocator, io, platform, &compiled, w);
            defer prepared.deinit();
            var record: Record = .{ .id = "test", .graph_id = "test", .workload = w, .status = .passed };
            try runCase(allocator, io, platform, .{ .warmup = 0, .iters = 1, .samples = 3, .sample_ms = 1 }, &compiled, &prepared, &record);
            defer allocator.free(record.statistics.?.timing.samples_us);
            try std.testing.expectEqual(@as(usize, 0), record.reference_check.?.outside_tolerance);
            const timing = record.statistics.?.timing;
            for (timing.samples_us) |sample| try std.testing.expect(sample * @as(f64, @floatFromInt(timing.iterations_per_sample)) >= 1000);
        }
    }
}

test "dump metadata rejects different inputs with the same output byte count" {
    const allocator = std.testing.allocator;
    var w: cases.Workload = .{
        .case = .{ .backend = .stablehlo, .tokens = 2, .width = 128 },
        .routing = .uniform,
        .global_experts = 8,
        .active_experts = 0,
        .sigma = 0.6,
        .seed = 1,
        .data = .random,
    };
    const id = try w.id(allocator);
    defer allocator.free(id);
    var metadata: DumpMetadata = .{
        .workload_id = id,
        .dtype = .bf16,
        .shape = .{ 2, 128 },
        .byte_count = 512,
        .source = results.source(),
    };
    try metadata.validate(allocator, w, 512);
    w.seed = 2;
    try std.testing.expectError(error.DumpWorkloadMismatch, metadata.validate(allocator, w, 512));
    w.seed = 1;
    metadata.dtype = .f16;
    try std.testing.expectError(error.DumpWorkloadMismatch, metadata.validate(allocator, w, 512));
    metadata.dtype = .bf16;
    metadata.shape = .{ 1, 256 };
    try std.testing.expectError(error.DumpWorkloadMismatch, metadata.validate(allocator, w, 512));
    metadata.shape = .{ 2, 128 };
    metadata.source.revision = "baseline-revision";
    try metadata.validate(allocator, w, 512);
    metadata.schema_version = 2;
    try std.testing.expectError(error.UnsupportedDumpVersion, metadata.validate(allocator, w, 512));
}
