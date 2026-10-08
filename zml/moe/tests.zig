//! Shared public-entry-point tests. Hardware absence is the only runtime skip:
//! supported cases are explicit, and errors in those cases must fail the test.
const std = @import("std");
const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const reference = @import("stablehlo.zig");
const Placement = @import("fused_experts.zig").RoutingWeightPlacement;

const Case = struct {
    backend: zml.moe.Backend,
    activation: zml.moe.Activation = .{ .swiglu = .{} },
    scheme: ?zml.Quantization.Scheme = null,
    activation_dtype: zml.DataType = .bf16,
    quantize_input: bool = false,
    placement: Placement = .after_down,
    bias: bool = false,
    tokens: i64 = 1,
    topk: i64 = 1,
    batch: i64 = 1,
    width: i64 = 128,
    intermediate: i64 = 128,
    experts: i64 = 8,

    pub fn compile(c: Case, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Exe {
        std.debug.assert(c.backend.isAvailable(platform));
        const x: Tensor = .init(.{ .b = c.batch, .s = c.tokens, .d = c.width }, c.activation_dtype);
        return platform.compileFn(allocator, io, forward, .{ x, c }, .{});
    }

    pub fn check(c: Case, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, exe: *const zml.Exe) !void {
        const x: Tensor = .init(.{ .b = c.batch, .s = c.tokens, .d = c.width }, c.activation_dtype);

        var host = try zml.Slice.alloc(allocator, x.shape());
        defer host.free(allocator);
        switch (c.activation_dtype) {
            inline .bf16, .f16, .f32 => |dtype| {
                for (host.items(dtype.toZigType()), 0..) |*value, i| {
                    const f = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + i / 128) % 23)) - 11)) / 32;
                    value.* = if (dtype == .bf16) .fromF32(f) else @floatCast(f);
                }
            },
            else => unreachable,
        }

        var input = try zml.Buffer.fromSlice(io, platform, host);
        defer input.deinit();

        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var output: zml.Bufferized(Outputs) = undefined;
        runner.run(io, .{input}, .{&output}, .{ .wait = true });
        defer zml.Buffer.deinitAll(Outputs, &output);

        // BF16 intermediate rounding and different GEMM reduction orders. Every
        // element must pass; near-zero values use the absolute bound.
        try zml.testing.expectClose(io, output.expected, output.actual, .{
            // The NVFP4 matrix observed a maximum absolute error of 0.07519531.
            // Allow a small margin; this is an empirical comparison budget.
            .absolute_tolerance = if (c.scheme == .nvfp4) 0.078125 else 0.015625,
            .relative_tolerance = 0.02,
            .minimum_close_fraction = 1,
        });
    }
};

/// Generate a reproducible Tensor with a simple pattern.
///
/// Generate as many elements as the shape requires, with this formula:
/// value[i] = (((i * multiplier + 3) % 17) - 8) / divisor
///
/// Then reshape to the requested shape and convert to f32.
fn generateDeterministicTensor(shape: Shape, multiplier: i64, divisor: f64) Tensor {
    return Tensor.arange(.{ .end = @intCast(shape.count()) }, .i32)
        .mul(Tensor.scalar(multiplier, .i32)).addConstant(3)
        .remainder(Tensor.scalar(17, .i32)).addConstant(-8)
        .convert(.f32).scale(1 / divisor).reshape(shape.withDtype(.f32));
}

/// Select exactly representable FP4 values with distinct even/odd sequences.
fn generateFp4Tensor(shape: Shape, multiplier: i64) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 16 }, .f32), std.mem.sliceAsBytes(&[_]f32{
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    }));
    const index = Tensor.arange(.{ .end = @intCast(shape.count()) }, .i32);
    const pair = index.div(Tensor.scalar(2, .i32));
    // Prime periods distinguish experts, output rows, and gate/up halves.
    const even = pair.mul(Tensor.scalar(multiplier, .i32)).addConstant(3)
        .remainder(Tensor.scalar(17, .i32)).remainder(Tensor.scalar(16, .i32));
    const odd = pair.mul(Tensor.scalar(3, .i32)).addConstant(7)
        .remainder(Tensor.scalar(19, .i32)).remainder(Tensor.scalar(16, .i32));
    const indices = index.remainder(Tensor.scalar(2, .i32)).cmp(.EQ, Tensor.scalar(0, .i32)).select(even, odd);
    return values.gather(.{ .value = indices }, .{}).convert(.f4e2m1).reshape(shape.withDtype(.f4e2m1));
}

/// Create a Linear fixture with a reproducible pattern, optionally quantized and biased.
fn linear(shape: Shape, scheme: ?zml.Quantization.Scheme, bias: bool, multiplier: i64) zml.nn.Linear {
    const fixture_bias = if (bias) generateDeterministicTensor(shape.remove(2), 3, 32).convert(shape.dtype()) else null;

    const s = scheme orelse return .{
        .weight = generateDeterministicTensor(shape, multiplier, 16).convert(shape.dtype()),
        .tag = shape.tag(2),
        .bias = fixture_bias,
    };

    const weight = switch (s) {
        .mxfp4, .nvfp4 => generateFp4Tensor(shape, multiplier),
        else => generateDeterministicTensor(shape, multiplier, 16).convert(.f8e4m3fn),
    };

    const scale_shape: Shape = switch (s) {
        .mxfp4, .mxfp8 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 32) }, .f8e8m0),
        .nvfp4 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 16) }, .f8e4m3fn),
        .fp8_per_tensor => .init(.{ shape.dim(0), 1, 1 }, .f32),
        .fp8_per_channel => .init(.{ shape.dim(0), shape.dim(1), 1 }, .f32),
        .fp8_block128 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 128), @divExact(shape.dim(2), 128) }, .f32),
        .fp8_block32 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 32), @divExact(shape.dim(2), 32) }, .f32),
    };

    const scales = Tensor.arange(.{ .end = @intCast(scale_shape.count()) }, .i32)
        .remainder(Tensor.scalar(2, .i32)).convert(.f32).addConstant(1).scale(1.0 / 16.0)
        .reshape(scale_shape.withDtype(.f32)).convert(scale_shape.dtype());

    const input_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = Tensor.scalar(1, .f32), .operation = .multiply } else null;
    const global_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = Tensor.scalar(1, .f32).broad(.init(.{ .expert = shape.dim(0) }, .f32)), .operation = .multiply } else null;

    return .{
        .weight = weight,
        .tag = shape.tag(2),
        .bias = fixture_bias,
        .quantization = .{
            .scheme = s,
            .scales = scales,
            .input_scale = input_scale,
            .global_scale = global_scale,
        },
    };
}

const ExpertFixture = struct {
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
};

/// Canonical weights use logical dimensions, concatenated gate/up, and plain scales.
fn makeExpertFixture(c: Case) ExpertFixture {
    const projection_width: i64 = switch (c.activation) {
        .silu, .gelu, .relu => c.intermediate,
        else => 2 * c.intermediate,
    };
    return .{
        .gate_up = linear(.init(.{ .expert = c.experts, .dout = projection_width, .d = c.width }, c.activation_dtype), c.scheme, c.bias, 7),
        .down = linear(.init(.{ .expert = c.experts, .d = c.width, .dout = c.intermediate }, c.activation_dtype), c.scheme, c.bias, 11),
    };
}

/// Interleave the gate/up halves of the output axis: [E, 2, N/2, K] -> [E, N/2, 2, K].
fn interleave(tensor: Tensor) Tensor {
    const tags = tensor.shape().tags();
    return tensor.withTags(.{ .expert, .out, .in })
        .splitAxis(.out, .{ .projection = 2, .mid = @divExact(tensor.dim(1), 2) })
        .transpose(.{ .expert, .mid, .projection, .in })
        .merge(.{ .out = .{ .mid, .projection } }).withTags(tags);
}

/// Blackwell block scales: [E, N/128, 4, 32, K/4, 4] ->
/// [E, N/128, K/4, 32, 4, 4], retaining the declared [E, N, K] shape.
fn swizzleScales(scales: Tensor) Tensor {
    return scales.reshape(.{ scales.dim(0), @divExact(scales.dim(1), 128), 4, 32, @divExact(scales.dim(2), 4), 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 }).reshape(scales.shape());
}

fn packFp4Bytes(weight: Tensor) Tensor {
    std.debug.assert(weight.dtype() == .f4e2m1);
    return weight.reshape(weight.shape().setDim(2, @divExact(weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8);
}

fn prepareGateUp(tensor: Tensor, layout: zml.moe.ExpertsLayout) Tensor {
    const ordered = switch (layout.gate_up_order) {
        .gate_up => tensor,
        .up_gate => blk: {
            const mid = @divExact(tensor.dim(1), 2);
            break :blk Tensor.concatenate(&.{ tensor.slice(1, .{ .start = mid }), tensor.slice(1, .{ .end = mid }) }, 1);
        },
    };
    return switch (layout.gate_up) {
        .concatenated => ordered,
        .interleaved => interleave(ordered),
    };
}

/// Adapt a copy of both projections to the backend's storage and layout contract.
fn prepareForBackend(canonical: ExpertFixture, backend: zml.moe.Backend) ExpertFixture {
    var prepared = canonical;
    const scheme = canonical.gate_up.quantizationScheme();
    const layout = backend.expertsLayout(scheme) catch |err| zml.stdx.debug.panic("fixture layout: {}", .{err});
    // Plain activations have one projection, so gate/up ordering does not apply.
    const gated = canonical.gate_up.weight.dim(1) == 2 * canonical.down.weight.dim(2);
    if (gated) {
        prepared.gate_up.weight = prepareGateUp(canonical.gate_up.weight, layout);
        if (canonical.gate_up.bias) |bias| prepared.gate_up.bias = prepareGateUp(bias.appendAxes(.{.dummy}), layout).squeeze(.dummy);
        if (prepared.gate_up.quantization) |*q| q.scales = prepareGateUp(q.scales, layout);
    }

    for ([_]*zml.nn.Linear{ &prepared.gate_up, &prepared.down }) |projection| {
        if (layout.packing == .swizzled_scales or layout.packing == .flashinfer_nvfp4) {
            const q = &projection.quantization.?;
            q.scales = swizzleScales(q.scales);
            q.swizzled_scales = true;
        }
        if (backend == .cute_mxfp4 or backend == .triton_mxfp4) {
            projection.weight = packFp4Bytes(projection.weight);
            projection.quantization.?.scales = projection.quantization.?.scales.bitCast(.u8);
        }
        if (backend == .mosaic_tpu) projection.weight = projection.weight.transpose(.{ 0, 2, 1 });
        if (backend == .metal) {
            if (projection.quantization) |*q| {
                if (q.scheme != .nvfp4) q.scales = q.scales.convert(.bf16);
            }
        }
    }
    if (backend == .metal and scheme == .nvfp4) {
        const global = &prepared.gate_up.quantization.?.global_scale.?;
        global.value = global.value.appendAxes(.{.proj}).repeat1d(.proj, 2);
    }
    return prepared;
}

const Outputs = struct { actual: Tensor, expected: Tensor };

/// Forward a fixture through the MoE implementation under test, and also through the StableHLO
/// reference implementation. Return both outputs for comparison.
fn forward(input: Tensor, c: Case) Outputs {
    const canonical = makeExpertFixture(c);
    const prepared = prepareForBackend(canonical, c.backend);

    const route_shape = Shape.init(.{ .b = c.batch, .s = c.tokens, .topk = c.topk }, .i32);
    const route = Tensor.arange(.{ .end = @intCast(route_shape.count()) }, .i32).reshape(route_shape);

    // Only six of eight experts receive routes, with nonuniform counts.
    const ids = route.mul(Tensor.scalar(5, .i32)).addConstant(2).remainder(Tensor.scalar(@min(c.experts, 6), .i32));

    const weights = route.remainder(Tensor.scalar(3, .i32)).convert(.f32).addConstant(1).scale(0.25);
    const options: zml.moe.Options = .{
        .activation = c.activation,
        .quantize_input = c.quantize_input,
        .routing_weight_placement = c.placement,
    };

    const actual = zml.moe.forwardMoe(input, ids, weights, prepared.gate_up, prepared.down, c.backend, options);
    const expected = reference.reference(input, ids, weights, canonical.gate_up, canonical.down, .{
        .activation = c.activation,
        .quantize_input = c.quantize_input,
        .routing_weight_placement = c.placement,
        .input_quantization = switch (c.backend) {
            .metal => .none,
            .cute_mxfp4 => .mxfp8_bf16,
            .triton_mxfp4 => .mxfp8,
            else => .automatic,
        },
    });

    return .{ .actual = actual.reshape(input.shape()), .expected = expected };
}

const gated_activations = [_]zml.moe.Activation{
    .{ .swiglu = .{} },
    .{ .swiglu = .{ .limit = 0.25 } },
    .{ .swiglu = .{ .bias = 1 } },
    .{ .swiglu = .{ .limit = 0.25, .bias = 1 } },
    .{ .swiglu_step = .{} },
    .{ .swiglu_step = .{ .limit = 0.25 } },
    .geglu,
    .geglu_tanh,
};
const all_activations = gated_activations ++ [_]zml.moe.Activation{ .relu, .silu, .gelu };
const clipped_swiglu = [_]zml.moe.Activation{
    .{ .swiglu = .{ .limit = 0.25 } },
    .{ .swiglu = .{ .limit = 7 } },
};

const FixtureShape = struct {
    batch: i64 = 1,
    tokens: i64 = 1,
    topk: i64 = 1,
    width: i64 = 128,
    intermediate: i64 = 128,
    experts: i64 = 8,
};

// Every option combination runs on decode, prefill, and batched prefill.
// Shapes and floating-point parameter values are representative finite sets.
const fixture_shapes = [_]FixtureShape{
    .{},
    .{ .tokens = 17, .topk = 2 },
    .{ .batch = 2, .tokens = 17, .topk = 3 },
};
const specialized_shapes = [_]FixtureShape{
    .{ .width = 5120, .intermediate = 2304, .experts = 2 },
    .{ .width = 5120, .intermediate = 2304, .experts = 2, .tokens = 17, .topk = 2 },
    .{ .width = 5120, .intermediate = 2304, .experts = 2, .batch = 2, .tokens = 17, .topk = 2 },
};

const Format = struct {
    scheme: ?zml.Quantization.Scheme = null,
    activation_dtype: zml.DataType = .bf16,
};

const Matrix = struct {
    formats: []const Format = &.{.{}},
    activations: []const zml.moe.Activation,
    quantize_inputs: []const bool = &.{false},
    placements: []const Placement = &.{.after_down},
    biases: []const bool = &.{false},
    shapes: []const FixtureShape = &fixture_shapes,

    pub fn count(self: Matrix) usize {
        return self.formats.len * self.activations.len * self.quantize_inputs.len *
            self.placements.len * self.biases.len * self.shapes.len;
    }

    /// Shapes vary fastest, matching the nested loops in the backend tests.
    pub fn caseAt(self: Matrix, backend: zml.moe.Backend, index: usize) Case {
        std.debug.assert(index < self.count());
        var remaining = index;
        const shape = self.shapes[remaining % self.shapes.len];
        remaining /= self.shapes.len;
        const bias = self.biases[remaining % self.biases.len];
        remaining /= self.biases.len;
        const placement = self.placements[remaining % self.placements.len];
        remaining /= self.placements.len;
        const quantize_input = self.quantize_inputs[remaining % self.quantize_inputs.len];
        remaining /= self.quantize_inputs.len;
        const activation = self.activations[remaining % self.activations.len];
        remaining /= self.activations.len;
        const format = self.formats[remaining];
        return .{
            .backend = backend,
            .activation = activation,
            .scheme = format.scheme,
            .activation_dtype = format.activation_dtype,
            .quantize_input = quantize_input,
            .placement = placement,
            .bias = bias,
            .batch = shape.batch,
            .tokens = shape.tokens,
            .topk = shape.topk,
            .width = shape.width,
            .intermediate = shape.intermediate,
            .experts = shape.experts,
        };
    }
};

const Options = struct {
    producers: usize = 16,
    consumers: usize = 1,
    case_queue_capacity: usize = 16,
    executable_queue_capacity: usize = 16,
};

const CaseJob = struct {
    index: usize,
    case: Case,
};

const ExecutableJob = struct {
    job: CaseJob,
    exe: zml.Exe,
};

const Result = struct {
    case: Case,
    outcome: union(enum) {
        passed,
        skipped,
        compilation_failed: anyerror,
        check_failed: anyerror,
    },
};

const Pipeline = struct {
    platform: *const zml.Platform,

    fn generateCases(self: *const Pipeline, io: std.Io, backend: zml.moe.Backend, matrix: Matrix, case_queue: *std.Io.Queue(CaseJob), results: []?Result) std.Io.Cancelable!void {
        defer case_queue.close(io);
        const available = backend.isAvailable(self.platform);

        var index: usize = 0;
        for (0..matrix.count()) |local_index| {
            const c = matrix.caseAt(backend, local_index);
            if (available) {
                case_queue.putOne(io, .{ .index = index, .case = c }) catch |err| return switch (err) {
                    error.Closed => {},
                    error.Canceled => error.Canceled,
                };
            } else {
                results[index] = .{ .case = c, .outcome = .skipped };
            }
            index += 1;
        }
    }

    fn compileExecutablesWorker(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, case_queue: *std.Io.Queue(CaseJob), executable_queue: *std.Io.Queue(ExecutableJob), results: []?Result) std.Io.Cancelable!void {
        while (true) {
            const job = case_queue.getOne(io) catch |err| return switch (err) {
                error.Closed => {},
                error.Canceled => error.Canceled,
            };
            var exe = job.case.compile(allocator, io, self.platform) catch |err| {
                if (err == error.Canceled) return error.Canceled;
                results[job.index] = .{ .case = job.case, .outcome = .{ .compilation_failed = err } };
                continue;
            };
            // Ownership transfers only on a successful enqueue.
            executable_queue.putOne(io, .{ .job = job, .exe = exe }) catch |err| {
                exe.deinit();
                return switch (err) {
                    error.Closed => {},
                    error.Canceled => error.Canceled,
                };
            };
        }
    }

    fn runTestCaseWorker(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, executable_queue: *std.Io.Queue(ExecutableJob), results: []?Result) std.Io.Cancelable!void {
        while (true) {
            var compiled = executable_queue.getOne(io) catch |err| return switch (err) {
                error.Closed => {},
                error.Canceled => error.Canceled,
            };
            defer compiled.exe.deinit();
            const job = compiled.job;
            job.case.check(allocator, io, self.platform, &compiled.exe) catch |err| {
                if (err == error.Canceled) return error.Canceled;
                results[job.index] = .{ .case = job.case, .outcome = .{ .check_failed = err } };
                continue;
            };
            results[job.index] = .{ .case = job.case, .outcome = .passed };
        }
    }

    fn run(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, backend: zml.moe.Backend, matrix: Matrix, options: Options) !void {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const cases_buffer = try arena.allocator().alloc(CaseJob, options.case_queue_capacity);
        const executables_buffer = try arena.allocator().alloc(ExecutableJob, options.executable_queue_capacity);
        const results = try arena.allocator().alloc(?Result, matrix.count());
        @memset(results, null);

        var case_queue: std.Io.Queue(CaseJob) = .init(cases_buffer);
        var executable_queue: std.Io.Queue(ExecutableJob) = .init(executables_buffer);

        var enumerator: std.Io.Group = .init;
        var producers: std.Io.Group = .init;
        var consumers: std.Io.Group = .init;
        defer {
            case_queue.close(io);
            executable_queue.close(io);
            enumerator.cancel(io);
            producers.cancel(io);
            consumers.cancel(io);
            // On partial startup or cancellation, queued executables still own resources.
            while (executable_queue.getOneUncancelable(io)) |compiled| {
                compiled.exe.deinit();
            } else |_| {}
        }

        for (0..options.consumers) |_| try consumers.concurrent(io, runTestCaseWorker, .{ self, io, allocator, &executable_queue, results });
        for (0..options.producers) |_| try producers.concurrent(io, compileExecutablesWorker, .{ self, io, allocator, &case_queue, &executable_queue, results });
        try enumerator.concurrent(io, generateCases, .{ self, io, backend, matrix, &case_queue, results });

        try enumerator.await(io);
        try producers.await(io);
        // Closed queues drain before getOne returns error.Closed.
        executable_queue.close(io);
        try consumers.await(io);

        try report(results);
    }

    fn report(results: []const ?Result) !void {
        var failed: usize = 0;
        for (results, 0..) |result, index| {
            const r = result orelse return error.MissingCaseResult;
            switch (r.outcome) {
                .passed, .skipped => {},
                inline .compilation_failed, .check_failed => |err, stage| {
                    failed += 1;
                    std.debug.print("MoE case {}/{} {s}: {any}\nError: {}\n", .{
                        index + 1, results.len, @tagName(stage), r.case, err,
                    });
                },
            }
        }
        if (failed != 0) return error.MoeComplianceFailed;
    }
};

test "Triton compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    try pipeline.run(std.testing.io, std.testing.allocator, .triton, .{
        .formats = &.{
            .{},
            .{ .scheme = .mxfp4 },
            .{ .scheme = .mxfp8 },
            .{ .scheme = .fp8_per_tensor },
            .{ .scheme = .fp8_per_channel },
            .{ .scheme = .fp8_block128 },
            .{ .scheme = .fp8_block32 },
        },
        .activations = &all_activations,
        // BF16 and MXFP4 ignore this option; exercise both accepted settings.
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
        .biases = &.{ false, true },
    }, .{});
}

test "FlashInfer compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // Only default SwiGLU, no linear bias, and routing after the down projection.
    try pipeline.run(std.testing.io, std.testing.allocator, .flashinfer_cutlass, .{
        .formats = &.{ .{}, .{ .scheme = .nvfp4 } },
        .activations = &.{.{ .swiglu = .{} }},
    }, .{});
}

test "specialized Triton MXFP4 compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // The kernel always quantizes inputs to MXFP8, independently of the option.
    try pipeline.run(std.testing.io, std.testing.allocator, .triton_mxfp4, .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &clipped_swiglu,
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
    }, .{});
}

test "specialized CuTe MXFP4 compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    try pipeline.run(std.testing.io, std.testing.allocator, .cute_mxfp4, .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &clipped_swiglu,
        .quantize_inputs = &.{ false, true },
        .placements = &.{.before_down},
        .shapes = &specialized_shapes,
    }, .{});
}

test "Fly compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // These constraints keep execution on Fly rather than its Triton fallback.
    try pipeline.run(std.testing.io, std.testing.allocator, .fly, .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &gated_activations,
        .quantize_inputs = &.{ false, true },
        .placements = &.{.before_down},
        .shapes = &specialized_shapes,
    }, .{});
}

test "Metal compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // Quantized Metal kernels require BF16 activations and do not support bias.
    try pipeline.run(std.testing.io, std.testing.allocator, .metal, .{
        .formats = &.{
            .{},
            .{ .activation_dtype = .f16 },
            .{ .activation_dtype = .f32 },
            .{ .scheme = .nvfp4 },
            .{ .scheme = .mxfp8 },
            .{ .scheme = .fp8_per_tensor },
            .{ .scheme = .fp8_per_channel },
            .{ .scheme = .fp8_block128 },
            .{ .scheme = .fp8_block32 },
        },
        .activations = &all_activations,
    }, .{});
}

test "Mosaic TPU compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // canonicalizeDown currently requires a gated projection; quantized weights
    // are not supported by gmmDType. Plain activations need a backend fix first.
    try pipeline.run(std.testing.io, std.testing.allocator, .mosaic_tpu, .{
        .formats = &.{ .{}, .{ .activation_dtype = .f16 }, .{ .activation_dtype = .f32 } },
        .activations = &gated_activations,
        .biases = &.{ false, true },
    }, .{});
}
