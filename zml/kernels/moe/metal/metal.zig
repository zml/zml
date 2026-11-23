const std = @import("std");

const stdx = @import("stdx");

const nn = @import("zml/nn");

const zml = @import("zml/core");
const Tensor = zml.Tensor;
const Activation = @import("zml/kernels/moe/api").Activation;
const GlobalOptions = @import("zml/kernels/moe/api").Options;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const unpackedWeight = @import("zml/kernels/moe/weights").unpackedWeight;
const quantization = @import("zml/quantization");

const log = std.log.scoped(.@"zml/moe/metal");

const QuantMode = enum { none, fp8, nvfp4 };

fn quantMode(dtype: zml.DataType) QuantMode {
    return switch (dtype) {
        .bf16, .f16, .f32 => .none,
        .f8e4m3fn => .fp8,
        .u8, .f4e2m1 => .nvfp4,
        else => stdx.debug.panic("unsupported data type: {}", .{dtype}),
    };
}

fn moeGemm(
    x_rows: Tensor,
    w: Tensor,
    scale: ?Tensor,
    expert_ids: Tensor,
    out_shape: zml.Shape,
    mode: QuantMode,
) Tensor {
    const side: zml.ops.CustomCallOptions = .{ .has_side_effect = false };
    return switch (mode) {
        .none => zml.ops.customCall("__metal$moe_gemm", .{ x_rows, w, expert_ids }, .{out_shape}, .{}, side),
        .fp8 => zml.ops.customCall("__metal$moe_gemm$f8", .{ x_rows, w, scale.?, expert_ids }, .{out_shape}, .{}, side),
        .nvfp4 => zml.ops.customCall("__metal$moe_gemm$f4", .{ x_rows, w, scale.?, expert_ids }, .{out_shape}, .{}, side),
    };
}

/// Multiplies each route's output in FP32 by its expert's global scale. Gate/up scales may hold
/// one value per projection, `{.expert, .proj = 2}`, for the halves of the concatenated output.
fn applyGlobalScale(output: Tensor, global_scale: ?Tensor, expert_ids: Tensor) Tensor {
    const scale = global_scale orelse return output.convert(.f32);
    stdx.debug.assert(scale.dtype() == .f32 and (scale.rank() == 1 or (scale.rank() == 2 and scale.dim(.proj) == 2)), "metal backend expected an f32 global scale {{.expert}} or {{.expert, .proj = 2}}, got {f}", .{scale.shape()});

    var selected = scale.gather(.{ .expert = expert_ids }, .{});
    if (scale.rank() == 2) {
        const mid = @divExact(output.dim(.out), 2);
        selected = selected.appendAxes(.{.mid})
            .broad(zml.Shape.init(.{ .r = output.dim(.r), .proj = 2, .mid = mid }, .f32))
            .merge(.{ .out = .{ .proj, .mid } });
    }
    return output.convert(.f32).mul(selected.broad(output.shape().withDtype(.f32)));
}

/// BF16 rows holding exactly the NVFP4 activations the Triton backend multiplies: each row is
/// divided by a power of two, and each block of 16 gets an E4M3 scale and E2M1 values (the
/// lower-error one of amax/6 and amax/4 with `.four_or_six`). Products of E2M1 values, E4M3
/// scales and powers of two are exact in BF16. Rounding uses tensor math, which fuses better
/// than FP8 and FP4 conversions and avoids FP4 vector stores Metal cannot emit for every shape.
fn emulateNvfp4Input(x: Tensor, axis: anytype, opts: GlobalOptions) Tensor {
    // Triton rounds activations to BF16 before quantizing; the barrier keeps that rounding.
    const rows = (if (x.dtype() == .bf16) x else x.convert(.bf16).optimizationBarrier()).convert(.f32);
    const row_scale = quantization.nvfp4RowScale(rows, axis).broad(rows.shape());
    const blocks = rows.div(row_scale).splitAxis(axis, .{ .sc = -1, .blk = quantization.nvfp4_block_size });
    const six = nvfp4Candidate(blocks, 6);
    const values = switch (opts.nvfp4_input_scale) {
        .six => six.values,
        .four_or_six => blk: {
            const four = nvfp4Candidate(blocks, 4);
            // Ties keep amax/6.
            break :blk four.err.cmp(.LT, six.err).broad(blocks.shape().withDtype(.bool)).select(four.values, six.values);
        },
    };
    return values.reshape(rows.shape()).mul(row_scale).convert(.bf16);
}

/// Dequantized blocks whose largest magnitude maps to `value_max`, and each block's absolute error.
fn nvfp4Candidate(blocks: Tensor, value_max: f32) struct { values: Tensor, err: Tensor } {
    const amax = blocks.abs().max(.blk).scale(1.0 / value_max);
    const scale = quantization.roundE4m3Normal(amax.clamp(.scalar(0x1p-6, .f32), .scalar(448, .f32))).broad(blocks.shape());
    const values = quantization.roundE2m1(blocks.div(scale)).mul(scale);
    return .{ .values = values, .err = values.sub(blocks).abs().sum(.blk) };
}

/// The rows of `x` along `.r` in `order`.
fn permute(x: Tensor, order: Tensor) Tensor {
    return x.gather(.{ .r = order.withTags(.{.sorted}) }, .{}).rename(.{ .sorted = .r });
}

pub fn fusedExperts(
    hidden_states: zml.Tensor,
    routing: LocalRouting,
    gate_up_: nn.Linear,
    down_: nn.Linear,
    opts: GlobalOptions,
) zml.Tensor {
    stdx.debug.assert(gate_up_.bias == null, "metal backend doesn't support gate_up bias", .{});
    stdx.debug.assert(down_.bias == null, "metal backend doesn't support down bias", .{});

    const gate_up_scales: ?zml.Tensor = if (gate_up_.quantization) |q| q.scales else null;
    const gate_up_global_scale: ?zml.Tensor = if (gate_up_.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;

    const down_scales: ?zml.Tensor = if (down_.quantization) |q| q.scales else null;
    const down_global_scale: ?zml.Tensor = if (down_.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;

    const gate_up = unpackedWeight(gate_up_);
    const down = unpackedWeight(down_);
    stdx.debug.assert(gate_up.dtype() == down.dtype(), "metal backend expected gate_up and down data type to match, got {} and {}", .{ gate_up.dtype(), down.dtype() });
    const mode = quantMode(gate_up.dtype());
    // Metal reproduces Triton's NVFP4 activations; other weight formats keep BF16 activations.
    const quantize_input = opts.quantize_input and mode == .nvfp4;
    switch (mode) {
        .none => {
            stdx.debug.assert(gate_up_scales == null and down_scales == null, "metal backend expected gate_up and down scale to be null when the weights are not quantized", .{});
            stdx.debug.assert(gate_up_global_scale == null and down_global_scale == null, "metal backend expected gate_up and down global scale to be null when the weights are not quantized", .{});
            const is_hidden_states_valid_dtype = switch (hidden_states.dtype()) {
                .bf16, .f16, .f32 => true,
                else => false,
            };
            stdx.debug.assert(is_hidden_states_valid_dtype, "metal backend expected hidden_states data type to be bf16, f16 or f32, got {}", .{hidden_states.dtype()});
        },
        .fp8 => {
            stdx.debug.assert(gate_up_scales != null and down_scales != null, "metal backend expected gate_up and down scale to be non null", .{});
            stdx.debug.assert(gate_up_scales.?.dtype() == .bf16 and down_scales.?.dtype() == .bf16, "metal backend expected gate_up and down scale data type to be bf16, got {} and {}", .{ gate_up_scales.?.dtype(), down_scales.?.dtype() });
            stdx.debug.assert(gate_up_global_scale == null and down_global_scale == null, "metal backend expected gate_up and down global scale to be null", .{});
            stdx.debug.assert(hidden_states.dtype() == .bf16, "metal backend expected hidden_states data type to be bf16, got {}", .{hidden_states.dtype()});
        },
        .nvfp4 => {
            stdx.debug.assert(gate_up_scales != null and down_scales != null, "metal backend expected gate_up and down scale to be non null", .{});
            stdx.debug.assert(gate_up_scales.?.dtype() == .f8e4m3fn and down_scales.?.dtype() == .f8e4m3fn, "metal backend expected gate_up and down scale data type to be f8e4m3fn, got {} and {}", .{ gate_up_scales.?.dtype(), down_scales.?.dtype() });
            stdx.debug.assert(hidden_states.dtype() == .bf16, "metal backend expected hidden_states data type to be bf16, got {}", .{hidden_states.dtype()});
        },
    }

    const is_topk_weights_valid_dtype = switch (routing.weights.dtype()) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
    stdx.debug.assert(is_topk_weights_valid_dtype, "metal backend expected topk_weights data type to be bf16, f16 or f32, got {}", .{routing.weights.dtype()});
    stdx.debug.assert(routing.ids.dtype() == .i32, "metal backend expected topk_ids data type to be i32, got {}", .{routing.ids.dtype()});

    const d = hidden_states.dim(.d);
    const num_tokens = hidden_states.dim(.t);
    const topk = routing.ids.dim(.topk);
    const num_routes = num_tokens * topk;

    const act_dtype: zml.DataType = switch (mode) {
        .none => hidden_states.dtype(),
        .fp8, .nvfp4 => .bf16,
    };

    // Routes sorted by expert run each expert's rows back to back, so the per-route kernel
    // finds the expert's weights in cache: 30% faster for 16 tokens.
    const routes = routing.ids.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } }).convert(.i32);
    const route_weights = routing.weights.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } }).convert(.f32);
    const order = routes.argsort(.r, .{});
    const sorted_ids = permute(routes, order);
    // Routes without an expert (negative ids) contribute nothing. NVFP4 kernels skip them;
    // the other kernels need a valid expert, whose result the zero routing weight discards.
    const has_expert = sorted_ids.cmp(.GE, .scalar(0, .i32));
    const expert_ids = if (mode == .nvfp4) sorted_ids else sorted_ids.maximum(.scalar(0, .i32));
    const weights = has_expert.select(permute(route_weights, order), .scalar(0, .f32));
    const hidden = hidden_states.reshape(.{ .token = num_tokens, .d = d }).withTags(.{ .token, .d });
    const x_rows = hidden.gather(.{ .token = order.divByConst(topk).withTags(.{.r}) }, .{});
    // Each route quantizes a copy of its token, so the quantizer writes the GEMM input.
    const gate_up_input = if (quantize_input) emulateNvfp4Input(x_rows, .d, opts) else x_rows.convert(act_dtype);

    const gate_up_out = applyGlobalScale(moeGemm(
        gate_up_input,
        gate_up,
        gate_up_scales,
        expert_ids,
        zml.Shape.init(.{ .r = num_routes, .out = gate_up.dim(.dout) }, act_dtype),
        mode,
    ), gate_up_global_scale, expert_ids);
    var activated = opts.activation.applyActivation(gate_up_out, .concatenated);
    if (opts.routing_weight_placement == .before_down) activated = activated.mul(weights.broad(activated.shape()));
    const down_input = if (quantize_input) emulateNvfp4Input(activated, .out, opts) else activated.convert(act_dtype);

    var down_out = applyGlobalScale(moeGemm(
        down_input,
        down,
        down_scales,
        expert_ids,
        zml.Shape.init(.{ .r = num_routes, .d = down.dim(.d) }, act_dtype),
        mode,
    ), down_global_scale, expert_ids);
    if (opts.routing_weight_placement == .after_down) down_out = down_out.mul(weights.broad(down_out.shape()));
    // Back to token-major routes to sum each token's experts.
    const combined = permute(down_out, order.argsort(.r, .{})).splitAxis(.r, .{ .token = num_tokens, .topk = topk })
        .sum(.topk).squeeze(.topk).convert(act_dtype);

    return combined.reshape(.{ .t = num_tokens, .d = down.dim(.d) });
}
