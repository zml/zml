//! Deliberately simple MoE reference: no routed GEMMs, packing, or custom calls.
//! Weights use [expert, output, input] order with concatenated gate/up columns.
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub fn activate(x: Tensor, activation: zml.moe.Activation) Tensor {
    return switch (activation) {
        .silu => x.silu(),
        .gelu => x.gelu(),
        .relu => x.relu(),
        .swiglu, .swiglu_step, .geglu, .geglu_tanh => blk: {
            const mid = @divExact(x.dim(.out), 2);
            var gate = x.slice(.out, .{ .end = mid });
            var up = x.slice(.out, .{ .start = mid });

            switch (activation) {
                .swiglu_step => {
                    gate = gate.div(gate.negate().exp().addConstant(1));
                    if (activation.swiglu_step.limit) |limit| {
                        gate = gate.minimum(Tensor.scalar(limit, x.dtype()));
                        up = up.clamp(Tensor.scalar(-limit, x.dtype()), Tensor.scalar(limit, x.dtype()));
                    }
                    break :blk gate.mul(up);
                },
                .swiglu => {
                    if (activation.swiglu.limit) |limit| {
                        gate = gate.minimum(Tensor.scalar(limit, x.dtype()));
                        up = up.clamp(Tensor.scalar(-limit, x.dtype()), Tensor.scalar(limit, x.dtype()));
                    }
                    if (activation.swiglu.bias) |bias| up = up.addConstant(bias);
                    break :blk gate.div(gate.negate().exp().addConstant(1)).mul(up);
                },
                .geglu => break :blk gelu(gate).mul(up),
                .geglu_tanh => break :blk gate.gelu().mul(up),
                else => unreachable, // already handled above
            }
        },
    };
}

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

pub fn dequantize(linear: zml.nn.Linear) Tensor {
    var weight = linear.weight;
    const q = linear.quantization orelse return weight.convert(.f32);
    stdx.debug.assert(!q.swizzled_scales, "StableHLO reference requires plain block scales", .{});
    if (zml.nn.isPackedFp4(q.scheme, weight.dtype())) {
        // StableHLO bitcast expands the low nibble first along the input axis.
        weight = weight.bitCast(.f4e2m1).merge(.{ .unpacked = .{ linear.tag, .bitcast } }).renameTag(.unpacked, linear.tag);
    }
    const scales = if (q.scheme.isMx() and q.scales.dtype() == .u8) q.scales.bitCast(.f8e8m0) else q.scales;
    var result = weight.convert(.f32).mul(expandScales(scales, weight.shape()));
    if (q.global_scale) |scale| result = result.mul(scale.asMultiplier().broad(result.shape()));
    return result;
}

// E2M1 round-to-nearest, ties-to-even expressed independently of native
// float-to-FP4 lowering. Even encoding indices win midpoint ties.
fn roundFp4(x: Tensor) Tensor {
    const magnitude = x.abs();
    var rounded = Tensor.scalar(6, .f32).broad(x.shape());
    const thresholds = [_]f32{ 5, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25 };
    const values = [_]f32{ 4, 3, 2, 1.5, 1, 0.5, 0 };
    inline for (thresholds, values, 0..) |threshold, value, i| {
        rounded = magnitude.cmp(if (i % 2 == 0) .LE else .LT, Tensor.scalar(threshold, .f32))
            .select(Tensor.scalar(value, .f32).broad(x.shape()), rounded);
    }
    return x.cmp(.LT, Tensor.scalar(0, .f32)).select(rounded.negate(), rounded);
}

// Quantize and dequantize activation groups using tensor math, independently of
// Triton's quantizer. The return value is the represented value in FP32.
fn roundedInput(x: Tensor, linear: zml.nn.Linear, opts: Options, dtype: zml.DataType) Tensor {
    if (opts.input_quantization == .none) return x.convert(dtype).convert(.f32);
    const q = linear.quantization orelse return x.convert(dtype).convert(.f32);
    const fixed_mxfp8 = opts.input_quantization == .mxfp8 or opts.input_quantization == .mxfp8_bf16;
    const scheme = if (fixed_mxfp8) .fp8_block32 else q.scheme;
    if (scheme == .nvfp4) {
        const global = if (q.input_scale) |scale| scale.asMultiplier() else Tensor.scalar(1, .f32);
        stdx.debug.assert(global.count() == 1, "reference NVFP4 input scale must be global", .{});
        const grouped = x.convert(.bf16).convert(.f32).div(global.asScalar())
            .splitAxis(.in, .{ .group = @divExact(x.dim(.in), 16), .element = 16 });
        const scale = grouped.abs().max(.element).scale(1.0 / 6.0)
            .minimum(Tensor.scalar(448, .f32))
            .convert(.f8e4m3fn).convert(.f32).broad(grouped.shape());
        const divisor = scale.cmp(.NE, Tensor.scalar(0, .f32)).select(scale, Tensor.scalar(1, .f32).broad(scale.shape()));
        return roundFp4(grouped.mul(Tensor.scalar(1, .f32).div(divisor))).mul(scale)
            .reshape(x.shape().withDtype(.f32)).mul(global.asScalar());
    }
    if (opts.input_quantization == .automatic and (!opts.quantize_input or scheme == .mxfp4)) return x.convert(dtype).convert(.f32);
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
    return grouped.div(broad).clamp(Tensor.scalar(-max_value, .f32), Tensor.scalar(max_value, .f32))
        .convert(output_dtype).convert(.f32).mul(broad).reshape(x.shape().withDtype(.f32));
}

fn project(x: Tensor, linear: zml.nn.Linear, ids: Tensor, opts: Options, dtype: zml.DataType) Tensor {
    const weight = dequantize(linear).withTags(.{ .expert, .out, .in });
    const selected = weight.gather(.{ .expert = ids }, .{});
    var result = roundedInput(x, linear, opts, dtype).dot(selected, .in);
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
    activation: zml.moe.Activation,
    quantize_input: bool = false,
    /// Model fixed backend contracts independently of optional input quantization:
    /// Metal uses weight-only quantization; specialized MXFP4 GEMMs always use MXFP8 inputs.
    /// CuTe additionally rounds the intermediate activation to BF16 before quantizing.
    input_quantization: enum { automatic, none, mxfp8, mxfp8_bf16 } = .automatic,
    routing_weight_placement: @import("fused_experts.zig").RoutingWeightPlacement = .after_down,
};

pub fn reference(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: zml.nn.Linear, down: zml.nn.Linear, opts: Options) Tensor {
    stdx.debug.assert(input.rank() == 3 and topk_ids.rank() == 3 and topk_weights.rank() == 3, "reference expects rank-three input and routing tensors", .{});
    stdx.debug.assert(topk_ids.dtype() == .i32, "reference expects i32 expert ids", .{});
    stdx.debug.assert(gate_up.weight.rank() == 3 and down.weight.rank() == 3, "reference expects rank-three expert weights", .{});
    stdx.debug.assert(gate_up.weight.dim(.expert) == down.weight.dim(.expert), "reference expert counts must match", .{});
    const b = input.dim(.b);
    const s = input.dim(.s);
    const k = topk_ids.dim(.topk);
    stdx.debug.assert(k > 0 and k <= gate_up.weight.dim(.expert), "reference top-k must fit the expert count", .{});
    stdx.debug.assert(topk_ids.dim(.b) == b and topk_ids.dim(.s) == s and topk_weights.dim(.b) == b and topk_weights.dim(.s) == s and topk_weights.dim(.topk) == k, "reference routing dimensions must match input and ids", .{});
    const ids = topk_ids.reshape(.{ .route = b * s * k });
    const weights = topk_weights.reshape(.{ .route = b * s * k }).convert(.f32);
    const hidden = input.reshape(.{ .token = b * s, .in = input.dim(.d) })
        .insertAxes(.in, .{.topk}).broad(Shape.init(.{ .token = b * s, .topk = k, .in = input.dim(.d) }, input.dtype()))
        .merge(.{ .route = .{ .token, .topk } });
    // Keep the projection's rounding boundary when XLA fuses activation/routing
    // arithmetic; otherwise the reference can disagree with explicit BF16 GEMMs.
    const projected = project(hidden, gate_up, ids, opts, input.dtype()).convert(input.dtype()).optimizationBarrier().convert(.f32);
    var activated = activate(projected, opts.activation);
    if (opts.routing_weight_placement == .before_down) activated = activated.mul(weights.broad(activated.shape()));
    var output = project(activated.rename(.{ .out = .in }), down, ids, opts, input.dtype());
    if (opts.routing_weight_placement == .after_down) output = output.mul(weights.broad(output.shape()));
    return output.convert(input.dtype()).convert(.f32)
        .reshape(.{ .b = b, .s = s, .topk = k, .d = output.dim(.out) })
        .sum(.topk).squeeze(.topk).convert(input.dtype());
}

pub fn fusedExperts(input: Tensor, ids: Tensor, weights: Tensor, gate_up: zml.nn.Linear, down: zml.nn.Linear, opts: zml.moe.Options) Tensor {
    return reference(input, ids, weights, gate_up, down, .{
        .activation = opts.activation,
        .quantize_input = opts.quantize_input,
        .routing_weight_placement = opts.routing_weight_placement,
    });
}

test "MoE compliance: FP4 midpoint rounding" {
    const Local = struct {
        fn call(x: Tensor) Tensor {
            return roundFp4(x);
        }
    };
    const values = [_]f32{ -7, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7 };
    const expected = [_]f32{ -6, -4, -4, -2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 4, 4, 6 };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const x: Tensor = .init(.{values.len}, .f32);
    var exe = try platform.compileFn(allocator, io, Local.call, .{x}, .{});
    defer exe.deinit();
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
    defer input.deinit();
    var output = try exe.eval(allocator, io, .{input});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    try std.testing.expectEqualSlices(f32, &expected, actual.constItems(f32));
}
