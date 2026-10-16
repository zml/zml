//! Shared MoE compatibility cases, canonical fixtures, and backend layout preparation.
const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const reference = zml.moe.stablehlo;
const Placement = zml.moe.RoutingWeightPlacement;

pub const Case = struct {
    backend: zml.moe.Backend,
    activation: zml.moe.Activation = .{ .swiglu = .{} },
    scheme: ?zml.Quantization.Scheme = null,
    activation_dtype: zml.DataType = .bf16,
    quantize_input: bool = false,
    placement: Placement = .after_down,
    bias: bool = false,
    tokens: i64 = 1,
    topk: i64 = 1,
    width: i64 = 128,
    intermediate: i64 = 128,
    experts: i64 = 8,

    pub fn options(c: Case) zml.moe.Options {
        return .{ .activation = c.activation, .quantize_input = c.quantize_input, .routing_weight_placement = c.placement };
    }

    pub fn referenceOptions(c: Case, target: zml.Target) reference.Options {
        return .{
            .activation = c.activation,
            .quantize_input = c.quantize_input,
            .routing_weight_placement = c.placement,
            .input_quantization = switch (c.backend) {
                .metal => .none,
                .cute => .mxfp8_bf16,
                .triton => if (c.scheme == .mxfp4 and target == .cuda) .mxfp8 else .automatic,
                else => .automatic,
            },
        };
    }

    pub fn tolerance(c: Case) zml.testing.CompareOpts {
        return .{
            .absolute_tolerance = if (c.scheme == .nvfp4) 0.078125 else 0.015625,
            .relative_tolerance = 0.02,
            .minimum_close_fraction = 1,
        };
    }

    pub fn compile(c: Case, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Exe {
        std.debug.assert(c.backend.isAvailable(platform));
        const x: Tensor = .init(.{ .t = c.tokens, .d = c.width }, c.activation_dtype);
        return platform.compileFn(allocator, io, forward, .{ x, c }, .{});
    }

    pub fn check(c: Case, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, exe: *const zml.Exe) !void {
        const x: Tensor = .init(.{ .t = c.tokens, .d = c.width }, c.activation_dtype);

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
        try zml.testing.expectClose(io, output.expected, output.actual, c.tolerance());
    }
};

/// Generate a reproducible Tensor with a simple pattern.
///
/// Generate as many elements as the shape requires, with this formula:
/// value[i] = (((i * multiplier + 3) % 17) - 8) / divisor
///
/// Then reshape to the requested shape and convert to f32.
fn generateDeterministicTensor(shape: Shape, multiplier: i64, divisor: f64, seed: Tensor) Tensor {
    return Tensor.arange(.{ .end = @intCast(shape.count()) }, .i64).add(seed.convert(.i64))
        .mul(Tensor.scalar(multiplier, .i64)).addConstant(3)
        .remainder(Tensor.scalar(17, .i64)).addConstant(-8)
        .convert(.f32).scale(1 / divisor).reshape(shape.withDtype(.f32));
}

/// Select exactly representable FP4 values with distinct even/odd sequences.
fn generateFp4Tensor(shape: Shape, multiplier: i64, seed: Tensor) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 16 }, .f32), std.mem.sliceAsBytes(&[_]f32{
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    }));
    const index = Tensor.arange(.{ .end = @intCast(shape.count()) }, .i64).add(seed.convert(.i64));
    const pair = index.div(Tensor.scalar(2, .i64));
    // Prime periods distinguish experts, output rows, and gate/up halves.
    const even = pair.mul(Tensor.scalar(multiplier, .i64)).addConstant(3)
        .remainder(Tensor.scalar(17, .i64)).remainder(Tensor.scalar(16, .i64));
    const odd = pair.mul(Tensor.scalar(3, .i64)).addConstant(7)
        .remainder(Tensor.scalar(19, .i64)).remainder(Tensor.scalar(16, .i64));
    const indices = index.remainder(Tensor.scalar(2, .i64)).cmp(.EQ, Tensor.scalar(0, .i64)).select(even, odd);
    return values.gather(.{ .value = indices }, .{}).convert(.f4e2m1).reshape(shape.withDtype(.f4e2m1));
}

/// Create a Linear fixture with a reproducible pattern, optionally quantized and biased.
fn linear(shape: Shape, scheme: ?zml.Quantization.Scheme, bias: bool, multiplier: i64, mode: FixtureMode, seed: Tensor) zml.nn.Linear {
    const fixture_bias = if (bias) fixtureTensor(shape.remove(2), 3, 32, mode, seed) else null;

    const s = scheme orelse return .{
        .weight = fixtureTensor(shape, multiplier, 16, mode, seed),
        .tag = shape.tag(2),
        .bias = fixture_bias,
    };

    const weight = switch (s) {
        .mxfp4, .nvfp4 => switch (mode) {
            .template => Tensor.fromShape(shape.withDtype(.f4e2m1)),
            .fixture => generateFp4Tensor(shape, multiplier, seed),
            .random => randomFp4(shape, seed.addConstant(multiplier)),
        },
        else => fixtureTensor(shape.withDtype(.f8e4m3fn), multiplier, 16, mode, seed),
    };

    const scale_shape: Shape = switch (s) {
        .mxfp4, .mxfp8 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 32) }, .f8e8m0),
        .nvfp4 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 16) }, .f8e4m3fn),
        .fp8_per_tensor => .init(.{ shape.dim(0), 1, 1 }, .f32),
        .fp8_per_channel => .init(.{ shape.dim(0), shape.dim(1), 1 }, .f32),
        .fp8_block128 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 128), @divExact(shape.dim(2), 128) }, .f32),
        .fp8_block32 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 32), @divExact(shape.dim(2), 32) }, .f32),
    };

    const scales = if (mode == .template) Tensor.fromShape(scale_shape) else if (mode == .random and (s == .mxfp4 or s == .nvfp4)) randomScales(scale_shape, seed.addConstant(multiplier + 17)) else (if (mode == .random) randomBits(scale_shape, seed).reshape(.{ .value = scale_shape.count() }).remainder(Tensor.scalar(2, .u32)).convert(.i64) else Tensor.arange(.{ .end = @intCast(scale_shape.count()) }, .i64).add(seed.convert(.i64)))
        .remainder(Tensor.scalar(2, .i64)).convert(.f32).addConstant(1).scale(1.0 / 16.0)
        .reshape(scale_shape.withDtype(.f32)).convert(scale_shape.dtype());

    const input_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = if (mode == .template) Tensor.init(.{}, .f32) else Tensor.scalar(1, .f32), .operation = .multiply } else null;
    const global_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = if (mode == .template) Tensor.init(.{ .expert = shape.dim(0) }, .f32) else Tensor.scalar(1, .f32).broad(.init(.{ .expert = shape.dim(0) }, .f32)), .operation = .multiply } else null;

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

pub const ExpertFixture = struct {
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
};

/// Canonical weights use logical dimensions, concatenated gate/up, and plain scales.
pub fn makeExpertFixture(c: Case) ExpertFixture {
    return expertFixture(c, .fixture, Tensor.scalar(0, .u32));
}

pub const FixtureMode = enum { template, fixture, random };

/// `template` describes canonical operands without requiring a compiler context.
pub fn expertFixture(c: Case, mode: FixtureMode, seed: Tensor) ExpertFixture {
    const projection_width: i64 = switch (c.activation) {
        .silu, .gelu, .relu => c.intermediate,
        else => 2 * c.intermediate,
    };
    return .{
        .gate_up = linear(.init(.{ .expert = c.experts, .dout = projection_width, .d = c.width }, c.activation_dtype), c.scheme, c.bias, 7, mode, seed),
        .down = linear(.init(.{ .expert = c.experts, .d = c.width, .dout = c.intermediate }, c.activation_dtype), c.scheme, c.bias, 11, mode, seed),
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
pub fn prepareForBackend(canonical: ExpertFixture, backend: zml.moe.Backend) ExpertFixture {
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
        switch (backend) {
            .cute => {
                projection.weight = packFp4Bytes(projection.weight);
                projection.quantization.?.scales = projection.quantization.?.scales.bitCast(.u8);
            },
            .triton => if (scheme == .mxfp4) {
                projection.weight = packFp4Bytes(projection.weight);
            },
            .mosaic_tpu => {
                projection.weight = projection.weight.transpose(.{ 0, 2, 1 });
            },
            .metal => if (projection.quantization) |*q| {
                if (q.scheme != .nvfp4) q.scales = q.scales.convert(.bf16);
            },
            else => {},
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

    const route_shape = Shape.init(.{ .t = c.tokens, .topk = c.topk }, .i32);
    const route = Tensor.arange(.{ .end = @intCast(route_shape.count()) }, .i32).reshape(route_shape);

    // Only six of eight experts receive routes, with nonuniform counts.
    const ids = route.mul(Tensor.scalar(5, .i32)).addConstant(2).remainder(Tensor.scalar(@min(c.experts, 6), .i32));

    const weights = route.remainder(Tensor.scalar(3, .i32)).convert(.f32).addConstant(1).scale(0.25);
    const options = c.options();

    const actual = zml.moe.forwardMoe(input, ids, weights, prepared.gate_up, prepared.down, c.backend, options);
    const expected = reference.reference(input, ids, weights, canonical.gate_up, canonical.down, c.referenceOptions(zml.Compiler.current().platform.target));

    return .{ .actual = actual, .expected = expected };
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

pub const FixtureShape = struct {
    tokens: i64 = 1,
    topk: i64 = 1,
    width: i64 = 128,
    intermediate: i64 = 128,
    experts: i64 = 8,
};

// Every option combination runs with 1, 17, and 34 flattened tokens.
// Shapes and floating-point parameter values are representative finite sets.
const fixture_shapes = [_]FixtureShape{
    .{},
    .{ .tokens = 17, .topk = 2 },
    .{ .tokens = 34, .topk = 3 },
};
const specialized_shapes = [_]FixtureShape{
    .{ .width = 5120, .intermediate = 2304, .experts = 2 },
    .{ .width = 5120, .intermediate = 2304, .experts = 2, .tokens = 17, .topk = 2 },
    .{ .width = 5120, .intermediate = 2304, .experts = 2, .tokens = 34, .topk = 2 },
};
const cute_shapes = specialized_shapes ++ [_]FixtureShape{
    // Singleton TMA expert and group dimensions, in direct and grouped routing.
    .{ .width = 5120, .intermediate = 2304, .experts = 1 },
    .{ .width = 5120, .intermediate = 2304, .experts = 1, .tokens = 2 },
};

pub const Format = struct {
    scheme: ?zml.Quantization.Scheme = null,
    activation_dtype: zml.DataType = .bf16,
};

pub const Matrix = struct {
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
            .tokens = shape.tokens,
            .topk = shape.topk,
            .width = shape.width,
            .intermediate = shape.intermediate,
            .experts = shape.experts,
        };
    }
};

pub const matrices = struct {
    pub const triton_base: Matrix = .{
        .formats = &.{
            .{},
            .{ .scheme = .mxfp8 },
            .{ .scheme = .fp8_per_tensor },
            .{ .scheme = .fp8_per_channel },
            .{ .scheme = .fp8_block128 },
            .{ .scheme = .fp8_block32 },
        },
        .activations = &all_activations,
        // BF16 ignores this option; exercise both accepted settings.
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
        .biases = &.{ false, true },
    };

    pub const triton_mxfp4_general: Matrix = .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &all_activations,
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
        .biases = &.{ false, true },
    };

    pub const triton_mxfp4_cuda_small: Matrix = .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &.{gated_activations[1]},
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
    };

    pub const triton_mxfp4_cuda: Matrix = .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &clipped_swiglu,
        .quantize_inputs = &.{ false, true },
        .placements = &.{ .before_down, .after_down },
    };

    pub const flashinfer: Matrix = .{
        .formats = &.{ .{}, .{ .scheme = .nvfp4 } },
        .activations = &.{.{ .swiglu = .{} }},
    };

    pub const cute: Matrix = .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &clipped_swiglu,
        .quantize_inputs = &.{ false, true },
        .placements = &.{.before_down},
        .shapes = &cute_shapes,
    };

    pub const fly: Matrix = .{
        .formats = &.{.{ .scheme = .mxfp4 }},
        .activations = &gated_activations,
        .quantize_inputs = &.{ false, true },
        .placements = &.{.before_down},
        .shapes = &specialized_shapes,
    };

    pub const metal: Matrix = .{
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
    };

    pub const mosaic_tpu: Matrix = .{
        .formats = &.{ .{}, .{ .activation_dtype = .f16 }, .{ .activation_dtype = .f32 } },
        .activations = &gated_activations,
        .biases = &.{ false, true },
    };

    pub const stablehlo: Matrix = .{
        .activations = &all_activations,
        .placements = &.{ .before_down, .after_down },
    };
};

pub fn matricesFor(backend: zml.moe.Backend, target: zml.platform.Target) []const Matrix {
    return switch (backend) {
        .triton => if (target == .cuda) &.{ matrices.triton_base, matrices.triton_mxfp4_cuda_small, matrices.triton_mxfp4_cuda } else &.{ matrices.triton_base, matrices.triton_mxfp4_general },
        .cute => &.{matrices.cute},
        .flashinfer_cutlass => &.{matrices.flashinfer},
        .fly => &.{matrices.fly},
        .metal => &.{matrices.metal},
        .mosaic_tpu => &.{matrices.mosaic_tpu},
        .stablehlo => &.{matrices.stablehlo},
    };
}

// Counter-based integer hashing keeps random fixtures reproducible across machines.
fn randomBits(shape: Shape, seed: Tensor) Tensor {
    var x = Tensor.arange(.{ .end = @intCast(shape.count()) }, .u32).add(seed);
    x = x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(16, .u32))).mul(Tensor.scalar(0x7feb352d, .u32));
    x = x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(15, .u32))).mul(Tensor.scalar(0x846ca68b, .u32));
    return x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(16, .u32))).reshape(shape.withDtype(.u32));
}

fn fixtureTensor(shape: Shape, multiplier: i64, divisor: f64, mode: FixtureMode, seed: Tensor) Tensor {
    return switch (mode) {
        .template => Tensor.fromShape(shape),
        .fixture => generateDeterministicTensor(shape, multiplier, divisor, seed).convert(shape.dtype()),
        .random => randomBits(shape, seed.addConstant(multiplier)).shiftRightLogical(Tensor.scalar(8, .u32))
            .convert(.f32).scale(2.0 / 16777216.0).addConstant(-1).convert(shape.dtype()),
    };
}

fn randomFp4(shape: Shape, seed: Tensor) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 16 }, .f32), std.mem.sliceAsBytes(&[_]f32{
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    }));
    const index = randomBits(shape, seed).remainder(Tensor.scalar(16, .u32)).convert(.i32).flatten();
    return values.gather(.{ .value = index }, .{}).convert(.f4e2m1).reshape(shape.withDtype(.f4e2m1));
}

fn randomScales(shape: Shape, seed: Tensor) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 4 }, .f32), std.mem.sliceAsBytes(&[_]f32{ 1.0 / 512.0, 1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0 }));
    const index = randomBits(shape, seed).remainder(Tensor.scalar(4, .u32)).convert(.i32).flatten();
    return values.gather(.{ .value = index }, .{}).convert(shape.dtype()).reshape(shape);
}

test "MoE matrix coverage remains stable" {
    try std.testing.expectEqual(@as(usize, 1584), matrices.triton_base.count());
    try std.testing.expectEqual(@as(usize, 264), matrices.triton_mxfp4_general.count());
    try std.testing.expectEqual(@as(usize, 12), matrices.triton_mxfp4_cuda_small.count());
    try std.testing.expectEqual(@as(usize, 24), matrices.triton_mxfp4_cuda.count());
    try std.testing.expectEqual(@as(usize, 6), matrices.flashinfer.count());
    try std.testing.expectEqual(@as(usize, 20), matrices.cute.count());
    try std.testing.expectEqual(@as(usize, 48), matrices.fly.count());
    try std.testing.expectEqual(@as(usize, 297), matrices.metal.count());
    try std.testing.expectEqual(@as(usize, 144), matrices.mosaic_tpu.count());
    try std.testing.expectEqual(@as(usize, 66), matrices.stablehlo.count());
    const first = matrices.cute.caseAt(.cute, 0);
    const last = matrices.cute.caseAt(.cute, matrices.cute.count() - 1);
    try std.testing.expectEqual(@as(i64, 5120), first.width);
    try std.testing.expectEqual(@as(i64, 2), last.tokens);
    try std.testing.expectEqual(@as(?f32, 7), last.activation.swiglu.limit);
}
