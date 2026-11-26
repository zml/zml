const std = @import("std");

const Activation = @import("zml/kernels/moe/api").Activation;
const GlobalOptions = @import("zml/kernels/moe/api").Options;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const nn = @import("zml/nn");
const quantization = @import("zml/quantization");
const Quantization = quantization.Quantization;
const stdx = @import("stdx");
const unpackedWeight = @import("zml/kernels/moe/weights").unpackedWeight;
const zml = @import("zml/core");
const Tensor = zml.Tensor;

const grouped_gemv = @import("grouped_gemv.zig");

const log = std.log.scoped(.@"zml/moe/metal");

test {
    _ = grouped_gemv;
}

const QuantMode = enum { none, fp8, mxfp8, nvfp4 };

/// Metal kernels run NVFP4, MXFP8 and FP8 experts with 128x128 block scales;
/// `Backend.auto` rejects the other formats.
fn quantMode(scheme: ?Quantization.Scheme) QuantMode {
    const s = scheme orelse return .none;
    return switch (s) {
        .nvfp4 => .nvfp4,
        .mxfp8 => .mxfp8,
        .fp8_block128 => .fp8,
        .mxfp4, .fp8_per_channel, .fp8_per_tensor, .fp8_block32 => stdx.debug.panic("metal backend does not support {s} experts", .{@tagName(s)}),
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
        // Only the grouped kernel runs MXFP8.
        .mxfp8 => unreachable,
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

/// The rows of a projection, quantized or not. The kernel reads `values`, which hold quantized
/// rows times their block scales; its output is left to multiply by their `global_scale`, a
/// scalar or one per row.
const MaybeQuantizedInput = struct { values: Tensor, global_scale: ?Tensor = null };

/// The activations Triton multiplies, in BF16 for the kernels.
fn emulateInput(x: Tensor, axis: anytype, q: Quantization) MaybeQuantizedInput {
    switch (q.scheme) {
        .nvfp4 => {
            const quantized = quantization.nvfp4.quantize(x, axis, q.nvfp4_input orelse @panic("NVFP4 experts need Quantization.nvfp4_input"));
            // Without the global scale: BF16 holds the values times their block scales exactly,
            // not their product with a global scale, which multiplies the kernel's output instead.
            const values = quantization.nvfp4.dequantize(.{ .values = quantized.values, .scales = quantized.scales, .global_scale = null }, axis);
            return .{ .values = values.convert(.bf16), .global_scale = quantized.global_scale };
        },
        .mxfp8 => {
            // Triton rounds activations to BF16 before quantizing; the barrier keeps that rounding.
            const rows = (if (x.dtype() == .bf16) x else x.convert(.bf16).optimizationBarrier()).convert(.f32);
            return .{ .values = emulateMxfp8(rows, axis).convert(.bf16) };
        },
        else => stdx.debug.panic("metal backend only emulates NVFP4 and MXFP8 activations, got {s}", .{@tagName(q.scheme)}),
    }
}

/// `output` {.r, ...} times `scale`, a scalar or one per route.
fn scaleRoutes(output: Tensor, scale: ?Tensor) Tensor {
    const factor = scale orelse return output;
    return output.mul(factor.broad(output.shape()));
}

/// MXFP8 rows: blocks of 32 with a power-of-two scale and E4M3 values.
fn emulateMxfp8(rows: Tensor, axis: anytype) Tensor {
    const blocks = rows.splitAxis(axis, .{ .sc = -1, .blk = quantization.mx_block_size });
    const scale = quantization.ceilPowerOfTwo(blocks.abs().max(.blk).maximum(.scalar(1e-10, .f32)).scale(1.0 / 448.0)).broad(blocks.shape());
    // Dividing by a power of two is exact; the barrier keeps the E4M3 rounding.
    const values = blocks.div(scale).clamp(.scalar(-448, .f32), .scalar(448, .f32)).convert(.f8e4m3fn).optimizationBarrier().convert(.f32);
    return values.mul(scale).reshape(rows.shape());
}

/// Whether packed NVFP4 experts run the grouped kernel rather than the plugin's. On an M3 Max
/// the plugin's per-route kernel over sorted routes is as fast up to 64 routes and its Steel
/// GEMM is faster from 2048; in between the grouped kernel wins (16 tokens over 48 experts:
/// 0.73 vs 0.86 ms per layer, 64 tokens over 128 experts: 2.4 vs 4.3 ms).
fn groupedNvfp4(routes: i64) bool {
    return routes >= 128 and routes <= 1024;
}

/// The projection of routes sorted by expert: each expert's quantized weights are read once
/// and applied to all its routes.
fn grouped(x_rows: Tensor, linear: nn.Linear, sorted_ids: Tensor, out_shape: zml.Shape, format: grouped_gemv.Format) Tensor {
    // Four routes per pass keep the accumulators in registers; wider loads help from 16 tokens.
    const cfg: grouped_gemv.GroupedGemv.Cfg = .{ .format = format, .rows = 4, .columns = 4, .words = 2 };
    const n = linear.weight.dim(1);
    const routes = sorted_ids.dim(.r);
    const launch = grouped_gemv.GroupedGemv.launch(cfg, n, routes, 2);
    return grouped_gemv.GroupedGemv.Kernel.call(.{
        .x = x_rows,
        .w = linear.weight.bitCast(.u8),
        .scales = linear.quantization.?.scales.bitCast(.u8),
        .ids = sorted_ids,
    }, .{ .out = out_shape }, .{
        .cfg = cfg,
        .grid = launch.grid,
        .threadgroup = launch.threadgroup,
        .constants = .{
            .k = .of(@as(i32, @intCast(x_rows.dim(1)))),
            .n = .of(@as(i32, @intCast(n))),
            .routes = .of(@as(i32, @intCast(routes))),
        },
    }).out;
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
    const mode = quantMode(gate_up_.quantizationScheme());
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
        .mxfp8 => {
            stdx.debug.assert(gate_up_scales != null and down_scales != null, "metal backend expected gate_up and down scale to be non null", .{});
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
        .fp8, .mxfp8, .nvfp4 => .bf16,
    };

    // Metal emulates the NVFP4 and MXFP8 activations Triton multiplies; FP8 experts keep BF16 activations.
    const quantize_input = opts.quantize_input and switch (mode) {
        .nvfp4, .mxfp8 => true,
        .none, .fp8 => false,
    };

    // The grouped kernel reads each expert's weights once. It runs MXFP8 experts, which the
    // plugin has no kernel for, and packed NVFP4 experts where it is faster.
    const format: ?grouped_gemv.Format = switch (mode) {
        .mxfp8 => .mxfp8,
        .nvfp4 => if (gate_up_.weight.dtype() == .u8 and down_.weight.dtype() == .u8 and groupedNvfp4(num_routes)) .nvfp4 else null,
        .none, .fp8 => null,
    };

    // Routes sorted by expert run each expert's rows back to back, so the per-route kernel
    // finds the expert's weights in cache: 30% faster for 16 tokens.
    const routes = routing.ids.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } }).convert(.i32);
    const route_weights = routing.weights.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } }).convert(.f32);
    const order = routes.argsort(.r, .{});
    const sorted_ids = permute(routes, order);
    const weights = permute(route_weights, order);

    // Routes without an expert (negative ids) are masked out of the sum. The NVFP4 and grouped
    // kernels skip them; the BF16 and FP8 kernels need a valid expert.
    const has_expert = sorted_ids.cmp(.GE, .scalar(0, .i32));
    const expert_ids = switch (mode) {
        .nvfp4, .mxfp8 => sorted_ids,
        .none, .fp8 => sorted_ids.maximum(.scalar(0, .i32)),
    };

    // Tokens are quantized once, before each route copies its own.
    const hidden = hidden_states.reshape(.{ .token = num_tokens, .d = d }).withTags(.{ .token, .d });
    const tokens: MaybeQuantizedInput = if (quantize_input) emulateInput(hidden, .d, gate_up_.quantization.?) else .{ .values = hidden.convert(act_dtype) };
    const route_tokens = order.divByConst(topk).withTags(.{.r});
    const gate_up_input = tokens.values.gather(.{ .token = route_tokens }, .{});
    // A calibrated scale is a scalar; row scales follow their tokens to the routes.
    const gate_up_input_global_scale: ?Tensor = if (tokens.global_scale) |scale| (if (scale.rank() == 0) scale else scale.gather(.{ .token = route_tokens }, .{})) else null;

    const gate_up_shape = zml.Shape.init(.{ .r = num_routes, .out = gate_up.dim(.dout) }, act_dtype);
    const gate_up_out = scaleRoutes(applyGlobalScale(if (format) |f|
        grouped(gate_up_input, gate_up_, sorted_ids, gate_up_shape, f)
    else
        moeGemm(gate_up_input, gate_up, gate_up_scales, expert_ids, gate_up_shape, mode), gate_up_global_scale, expert_ids), gate_up_input_global_scale);
    var activated = opts.activation.applyActivation(gate_up_out, .concatenated);
    if (opts.routing_weight_placement == .before_down) activated = activated.mul(weights.broad(activated.shape()));
    const down_input: MaybeQuantizedInput = if (quantize_input) emulateInput(activated, .out, down_.quantization.?) else .{ .values = activated.convert(act_dtype) };

    const down_shape = zml.Shape.init(.{ .r = num_routes, .d = down.dim(.d) }, act_dtype);
    var down_out = scaleRoutes(applyGlobalScale(if (format) |f|
        grouped(down_input.values, down_, sorted_ids, down_shape, f)
    else
        moeGemm(down_input.values, down, down_scales, expert_ids, down_shape, mode), down_global_scale, expert_ids), down_input.global_scale);
    if (opts.routing_weight_placement == .after_down) down_out = down_out.mul(weights.broad(down_out.shape()));
    down_out = down_out.mask(has_expert, 0);

    // Back to token-major routes to sum each token's experts.
    const combined = permute(down_out, order.argsort(.r, .{})).splitAxis(.r, .{ .token = num_tokens, .topk = topk })
        .sum(.topk).squeeze(.topk).convert(act_dtype);

    return combined.reshape(.{ .t = num_tokens, .d = down.dim(.d) });
}
