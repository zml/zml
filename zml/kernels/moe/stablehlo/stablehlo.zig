//! Deliberately simple MoE reference: no routed GEMMs, packing, or custom calls.
//! Weights use [expert, output, input] order with concatenated gate/up columns.
const std = @import("std");

const Activation = @import("zml/kernels/moe/api").Activation;
const GlobalOptions = @import("zml/kernels/moe/api").Options;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const nn = @import("zml/nn");
const quantization = @import("zml/quantization");
const RoutingWeightPlacement = @import("zml/kernels/moe/api").RoutingWeightPlacement;
const stdx = @import("stdx");
const zml = @import("zml/core");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

/// More precise GELU approximation than the default Zig implementation, which is
/// based on tanh. This is the same approximation used in JAX.
pub fn gelu(x: Tensor) Tensor {
    const xf = x.convert(.f32);
    const cdf = xf.scale(-1.0 / @sqrt(@as(f64, 2)))
        .erfc()
        .scale(0.5);
    return xf.mul(cdf).convert(x.dtype());
}

fn expandScales(scales: Tensor, shape: Shape) Tensor {
    if (scales.count() == 1) return scales.asScalar().convert(.f32).broad(shape.withDtype(.f32));
    stdx.debug.assert(scales.rank() == shape.rank(), "reference scale rank must match weights", .{});
    var result = scales.convert(.f32);
    for (0..shape.rank()) |axis| {
        const factor = std.math.divCeil(i64, shape.dim(axis), result.dim(axis)) catch unreachable;
        const expanded = result.shape().insert(axis + 1, .{factor});
        result = result.broadcast(expanded, Shape.range(expanded.rank(), .i64).remove(axis + 1).dims())
            .reshape(result.shape().setDim(axis, result.dim(axis) * factor))
            .slice(axis, .{ .end = shape.dim(axis) });
    }
    return result.withTags(shape.tags());
}

pub fn dequantize(linear: nn.Linear) Tensor {
    var weight = linear.weight;
    const q = linear.quantization orelse return weight.convert(.f32);
    stdx.debug.assert(!q.swizzled_scales, "StableHLO reference requires plain block scales", .{});
    if (nn.isPackedFp4(q.scheme, weight.dtype())) {
        // StableHLO bitcast expands the low nibble first along the input axis.
        weight = weight.bitCast(.f4e2m1).merge(.{ .unpacked = .{ linear.tag, .bitcast } }).renameTag(.unpacked, linear.tag);
    }
    const scales = if (q.scheme.isMx() and q.scales.dtype() == .u8) q.scales.bitCast(.f8e8m0) else q.scales;
    var result = weight.convert(.f32).mul(expandScales(scales, weight.shape()));
    if (q.global_scale) |scale| result = result.mul(scale.asMultiplier().broad(result.shape()));
    return result;
}

/// The rows of a projection, quantized or not. The GEMM reads `values`, the rows their
/// quantization represents, in FP32; its output is left to multiply by their `global_scale`,
/// a scalar or one per row.
const MaybeQuantizedInput = struct { values: Tensor, global_scale: ?Tensor = null };

// Quantize and dequantize activation groups using tensor math. NVFP4 goes through the
// quantizer the backends share, which its own tests compare with vLLM.
fn roundedInput(x: Tensor, linear: nn.Linear, opts: Options, dtype: zml.DataType) MaybeQuantizedInput {
    if (opts.input_quantization == .none) return .{ .values = x.convert(dtype).convert(.f32) };
    const q = linear.quantization orelse return .{ .values = x.convert(dtype).convert(.f32) };
    const fixed_mxfp8 = opts.input_quantization == .mxfp8 or opts.input_quantization == .mxfp8_bf16;
    const scheme = if (fixed_mxfp8) .fp8_block32 else q.scheme;
    if (scheme == .nvfp4) {
        const quantized = quantization.nvfp4.quantize(x, .in, q.nvfp4_input orelse @panic("NVFP4 experts need Quantization.nvfp4_input"));
        // Without the global scale, which multiplies the GEMM's output as in the kernels:
        // multiplied into the rows, the sum rounds differently from theirs.
        const values = quantization.nvfp4.dequantize(.{ .values = quantized.values, .scales = quantized.scales, .global_scale = null }, .in);
        return .{ .values = values, .global_scale = quantized.global_scale };
    }
    if (opts.input_quantization == .automatic and (!opts.quantize_input or scheme == .mxfp4)) return .{ .values = x.convert(dtype).convert(.f32) };
    const output_dtype = if (fixed_mxfp8) .f8e4m3fn else linear.weight.dtype();
    const group: i64 = switch (scheme) {
        .mxfp8, .fp8_block32 => 32,
        .fp8_block128 => 128,
        .fp8_per_channel, .fp8_per_tensor => x.dim(.in),
        else => unreachable,
    };
    const source = if (opts.input_quantization == .mxfp8_bf16) x.convert(.bf16) else x;
    const grouped = source.convert(.f32).splitAxis(.in, .{ .group = @divExact(x.dim(.in), group), .element = group });
    const epsilon: f32 = if (scheme == .fp8_block32) 1e-4 else 1e-10;
    const max_value: f32 = if (output_dtype == .f8e4m3fnuz) 224 else 448;
    var scale = grouped.abs().max(.element).maximum(Tensor.scalar(epsilon, .f32)).scale(1 / max_value);
    if (scheme == .mxfp8 or scheme == .fp8_block32) scale = scale.log().scale(1 / @log(@as(f32, 2))).ceil().scale(@log(@as(f32, 2))).exp();
    const broad = scale.broad(grouped.shape());
    return .{ .values = grouped.div(broad).clamp(Tensor.scalar(-max_value, .f32), Tensor.scalar(max_value, .f32))
        .convert(output_dtype).optimizationBarrier().convert(.f32).mul(broad).reshape(x.shape().withDtype(.f32)) };
}

fn project(x: Tensor, linear: nn.Linear, ids: Tensor, opts: Options, dtype: zml.DataType) Tensor {
    const weight = dequantize(linear).withTags(.{ .expert, .out, .in });
    const selected = weight.gather(.{ .expert = ids }, .{});
    const input = roundedInput(x, linear, opts, dtype);
    var result = input.values.dot(selected, .in);
    if (input.global_scale) |scale| result = result.mul(scale.broad(result.shape()));
    if (linear.bias) |bias| {
        const selected_bias = if (bias.rank() == 2)
            bias.withTags(.{ .expert, .out }).gather(.{ .expert = ids }, .{})
        else
            bias.withTags(.{.out});
        result = result.add(selected_bias.convert(.f32).broad(result.shape()));
    }
    return result;
}

pub const Options = struct {
    activation: Activation,
    quantize_input: bool = false,
    /// Model fixed backend contracts independently of optional input quantization:
    /// Metal uses weight-only quantization; specialized MXFP4 GEMMs always use MXFP8 inputs.
    /// CuTe additionally rounds the intermediate activation to BF16 before quantizing.
    input_quantization: enum { automatic, none, mxfp8, mxfp8_bf16 } = .automatic,
    routing_weight_placement: RoutingWeightPlacement = .after_down,
};

/// Negative expert IDs denote nonlocal routes and contribute zero, including bias.
pub fn reference(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: nn.Linear, down: nn.Linear, opts: Options) Tensor {
    stdx.debug.assert(topk_ids.dim(.topk) > 0, "reference top-k must be positive", .{});
    stdx.debug.assert(topk_ids.dtype() == .i32, "reference expects i32 expert ids", .{});
    stdx.debug.assert(gate_up.weight.rank() == 3 and down.weight.rank() == 3, "reference expects rank-three expert weights", .{});
    stdx.debug.assert(gate_up.weight.dim(.expert) == down.weight.dim(.expert), "reference expert counts must match", .{});
    const tokens = input.dim(.t);
    const k = topk_ids.dim(.topk);
    // Nonlocal routes use expert zero for safe gathers; discard their full
    // contribution after the down projection, including any projection bias.
    const local_routes = topk_ids.cmp(.GE, Tensor.scalar(0, .i32));
    const safe_ids = topk_ids.maximum(Tensor.scalar(0, .i32));
    const ids = safe_ids.reshape(.{ .route = tokens * k });
    const weights = topk_weights.reshape(.{ .route = tokens * k }).convert(.f32);
    const hidden = input.reshape(.{ .token = tokens, .in = input.dim(.d) })
        .insertAxes(.in, .{.topk}).broad(Shape.init(.{ .token = tokens, .topk = k, .in = input.dim(.d) }, input.dtype()))
        .merge(.{ .route = .{ .token, .topk } });
    // Keep the projection's rounding boundary when XLA fuses activation/routing
    // arithmetic; otherwise the reference can disagree with explicit BF16 GEMMs.
    const projected = project(hidden, gate_up, ids, opts, input.dtype()).convert(input.dtype()).optimizationBarrier().convert(.f32);
    var activated = opts.activation.applyActivation(projected, .concatenated);
    if (opts.routing_weight_placement == .before_down) activated = activated.mul(weights.broad(activated.shape()));
    var output = project(activated.rename(.{ .out = .in }), down, ids, opts, input.dtype());
    if (opts.routing_weight_placement == .after_down) output = output.mul(weights.broad(output.shape()));
    output = local_routes.reshape(.{ .route = tokens * k }).broad(output.shape()).select(output, Tensor.scalar(0, output.dtype()));
    return output.convert(input.dtype()).convert(.f32)
        .reshape(.{ .t = tokens, .topk = k, .d = output.dim(.out) })
        .sum(.topk).squeeze(.topk).convert(input.dtype());
}

pub fn fusedExperts(input: Tensor, routing: LocalRouting, gate_up: nn.Linear, down: nn.Linear, opts: GlobalOptions) Tensor {
    stdx.debug.assert(routing.ids.dim(.topk) > 0 and routing.ids.dim(.topk) <= routing.global_num_experts, "MoE top-k must fit the global expert count", .{});
    return reference(input, routing.localIds(), routing.weights, gate_up, down, .{
        .activation = opts.activation,
        .quantize_input = opts.quantize_input,
        .routing_weight_placement = opts.routing_weight_placement,
    });
}

test "NVFP4 experts without a calibrated scale quantize each row over its own" {
    const width = 32;
    const Local = struct {
        fn forward(x: Tensor) Tensor {
            // Identity projections: the experts return relu(x) when x survives its quantization.
            const weight: Shape = .init(.{ .expert = 1, .out = width, .in = width }, .f4e2m1);
            const projection: nn.Linear = .{
                .weight = Tensor.iota(weight, .out).cmp(.EQ, .iota(weight, .in)).convert(.f32).convert(.f4e2m1),
                .tag = Shape.toTag(.in),
                .quantization = .{
                    .scheme = .nvfp4,
                    .nvfp4_input = .{ .global_scale = .per_row, .block_scale = .best_of_four_or_six },
                    .scales = Tensor.scalar(1, .f32).broad(.init(.{ .expert = 1, .out = width, .sc = width / 16 }, .f32)).convert(.f8e4m3fn),
                },
            };
            const routes: Shape = .init(.{ .t = x.dim(.t), .topk = 1 }, .i32);
            return reference(x, .zeroes(routes), Tensor.scalar(1, .f32).broad(routes.withDtype(.f32)), projection, projection, .{
                .activation = .relu,
            });
        }
    };
    // Rows 2^22 apart, each exact in NVFP4 over its own scale: no single scale holds both.
    const pattern = [8]f32{ 6, -3, 1.5, 0.5, 1, -6, 2, 0 };
    var values: [2 * width]f32 = undefined;
    var expected: [2 * width]f32 = undefined;
    for (&values, &expected, 0..) |*value, *rectified, i| {
        value.* = std.math.ldexp(pattern[i % pattern.len], if (i < width) -12 else 10);
        rectified.* = @max(value.*, 0);
    }
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const x: Tensor = .init(.{ .t = 2, .d = width }, .f32);
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
    defer exe.deinit();
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
    defer input.deinit();
    var output = try exe.eval(allocator, io, .{input});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    try std.testing.expectEqualSlices(f32, &expected, actual.constItems(f32));
}

test "MoE compliance: capped SwiGLU" {
    const activation: Activation = .{ .capped_swiglu = .{ .gate_cap = 4, .up_cap = 25 } };
    const Local = struct {
        // The activation Triton, Fly and Metal apply between their GEMMs, and this reference.
        fn call(x: Tensor, act: Activation) Tensor {
            return Tensor.stack(&.{ act.applyActivation(x, .concatenated), act.applyActivation(x, .concatenated) }, 0, .implementation);
        }
    };
    const gates = [_]f32{ -9, -2.5, -0.5, 0, 0.75, 3, 12 };
    const ups = [_]f32{ 40, -30, 2, 7, -1.5, 0.25, -60 };
    const values = gates ++ ups;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const x: Tensor = .init(.{ .token = 1, .out = values.len }, .f32);
    var exe = try platform.compileFn(allocator, io, Local.call, .{ x, activation }, .{});
    defer exe.deinit();
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
    defer input.deinit();
    var output = try exe.eval(allocator, io, .{input});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    const results = actual.constItems(f32);
    for (0..2) |implementation| {
        for (gates, ups, 0..) |gate, up, i| {
            const expected = 4 * std.math.tanh(gate / 4) / (1 + @exp(-gate)) * 25 * std.math.tanh(up / 25);
            try std.testing.expectApproxEqAbs(expected, results[implementation * gates.len + i], 1e-4 * @max(1, @abs(expected)));
        }
    }
}
