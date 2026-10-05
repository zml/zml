const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/moe/metal");

pub const Options = struct {
    activation: zml.moe.Activation,
    global_num_experts: i64 = -1,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_global_scale: ?Tensor = null,
    w2_global_scale: ?Tensor = null,
};

fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(!opts.quantize_input, "Optional FP8 input quantization requires the Triton MoE backend", .{});
    stdx.debug.assert(opts.routing_weight_placement == .after_down, "Non-Triton MoE backends require routing weights after the down projection", .{});
}

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

fn applyGateUpGlobalScale(output: Tensor, global_scale: ?Tensor, expert_ids: Tensor) Tensor {
    const scale = global_scale orelse return output;
    stdx.debug.assert(scale.dtype() == .f32, "metal backend expected gate_up scale data type to be f32, got {}", .{scale.dtype()});
    stdx.debug.assert(scale.rank() == 2 and scale.dim(.proj) == 2, "metal backend expected gate_up scale to be a 2D tensor with a .proj dim equal to 2, got {}", .{scale.shape()});

    const mid = @divExact(output.dim(.out), 2);
    const selected = scale.gather(.{ .expert = expert_ids }, .{});
    const expanded = selected.appendAxes(.{.mid})
        .broad(zml.Shape.init(.{ .r = output.dim(.r), .proj = 2, .mid = mid }, .f32))
        .merge(.{ .out = .{ .proj, .mid } });
    return output.convert(.f32).mul(expanded).convert(output.dtype());
}

fn applyDownGlobalScale(output: Tensor, global_scale: ?Tensor, expert_ids: Tensor) Tensor {
    const scale = global_scale orelse return output;
    stdx.debug.assert(scale.dtype() == .f32, "metal backend expected down scale data type to be f32, got {}", .{scale.dtype()});
    stdx.debug.assert(scale.rank() == 1, "metal backend expected down scale to be a 1D tensor, got {}", .{scale.shape()});

    const selected = scale.gather(.{ .expert = expert_ids }, .{})
        .appendAxes(.{.d})
        .broad(output.shape().withDtype(.f32));
    return output.convert(.f32).mul(selected).convert(output.dtype());
}

pub fn fusedExperts(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    opts: zml.moe.Options,
) zml.Tensor {
    validateOptions(opts);
    stdx.debug.assert(gate_up.bias == null, "metal backend doesn't support gate_up bias", .{});
    stdx.debug.assert(down.bias == null, "metal backend doesn't support down bias", .{});

    const gate_up_scales: ?zml.Tensor = if (gate_up.quantization) |q| q.scales else null;
    const gate_up_global_scale: ?zml.Tensor = if (gate_up.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;

    const down_scales: ?zml.Tensor = if (down.quantization) |q| q.scales else null;
    const down_global_scale: ?zml.Tensor = if (down.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;

    const gate_up_weight_unpacked = zml.moe.unpackedWeight(gate_up);
    const down_weight_unpacked = zml.moe.unpackedWeight(down);

    return fusedExpertsImpl(
        input,
        gate_up_weight_unpacked,
        down_weight_unpacked,
        topk_weights,
        topk_ids,
        .{
            .activation = opts.activation,
            .global_num_experts = gate_up_weight_unpacked.dim(.expert),
            .w1_scale = gate_up_scales,
            .w2_scale = down_scales,
            .w1_global_scale = gate_up_global_scale,
            .w2_global_scale = down_global_scale,
        },
    );
}

pub fn fusedExpertsImpl(
    hidden_states: Tensor,
    gate_up: Tensor,
    down: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    opts: Options,
) Tensor {
    stdx.debug.assert(gate_up.dtype() == down.dtype(), "metal backend expected gate_up and down data type to match, got {} and {}", .{ gate_up.dtype(), down.dtype() });
    const mode = quantMode(gate_up.dtype());
    switch (mode) {
        .none => {
            stdx.debug.assert(opts.w1_scale == null and opts.w2_scale == null, "metal backend expected gate_up and down scale to be null when the weights are not quantized", .{});
            stdx.debug.assert(opts.w1_global_scale == null and opts.w2_global_scale == null, "metal backend expected gate_up and down global scale to be null when the weights are not quantized", .{});
            const is_hidden_states_valid_dtype = switch (hidden_states.dtype()) {
                .bf16, .f16, .f32 => true,
                else => false,
            };
            stdx.debug.assert(is_hidden_states_valid_dtype, "metal backend expected hidden_states data type to be bf16, f16 or f32, got {}", .{hidden_states.dtype()});
        },
        .fp8 => {
            stdx.debug.assert(opts.w1_scale != null and opts.w2_scale != null, "metal backend expected gate_up and down scale to be non null", .{});
            stdx.debug.assert(opts.w1_scale.?.dtype() == .bf16 and opts.w2_scale.?.dtype() == .bf16, "metal backend expected gate_up and down scale data type to be bf16, got {} and {}", .{ opts.w1_scale.?.dtype(), opts.w2_scale.?.dtype() });
            stdx.debug.assert(opts.w1_global_scale == null and opts.w2_global_scale == null, "metal backend expected gate_up and down global scale to be null", .{});
            stdx.debug.assert(hidden_states.dtype() == .bf16, "metal backend expected hidden_states data type to be bf16, got {}", .{hidden_states.dtype()});
        },
        .nvfp4 => {
            stdx.debug.assert(opts.w1_scale != null and opts.w2_scale != null, "metal backend expected gate_up and down scale to be non null", .{});
            stdx.debug.assert(opts.w1_scale.?.dtype() == .f8e4m3fn and opts.w2_scale.?.dtype() == .f8e4m3fn, "metal backend expected gate_up and down scale data type to be f8e4m3fn, got {} and {}", .{ opts.w1_scale.?.dtype(), opts.w2_scale.?.dtype() });
            stdx.debug.assert(hidden_states.dtype() == .bf16, "metal backend expected hidden_states data type to be bf16, got {}", .{hidden_states.dtype()});
        },
    }

    const is_topk_weights_valid_dtype = switch (topk_weights.dtype()) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
    stdx.debug.assert(is_topk_weights_valid_dtype, "metal backend expected topk_weights data type to be bf16, f16 or f32, got {}", .{topk_weights.dtype()});
    stdx.debug.assert(topk_ids.dtype() == .i32, "metal backend expected topk_ids data type to be i32, got {}", .{topk_ids.dtype()});

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const d = hidden_states.dim(.d);
    const num_tokens = b * s;
    const topk = topk_ids.dim(.topk);
    const num_routes = num_tokens * topk;

    const act_dtype: zml.DataType = switch (mode) {
        .none => hidden_states.dtype(),
        .fp8, .nvfp4 => .bf16,
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .d = d }).withTags(.{ .token, .d });
    const x_rows = hidden.insertAxes(.d, .{.topk})
        .broad(zml.Shape.init(.{ .token = num_tokens, .topk = topk, .d = d }, act_dtype))
        .merge(.{ .r = .{ .token, .topk } });
    const expert_ids = topk_ids.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } }).convert(.i32);

    const gate_up_out = applyGateUpGlobalScale(moeGemm(
        x_rows,
        gate_up,
        opts.w1_scale,
        expert_ids,
        zml.Shape.init(.{ .r = num_routes, .out = gate_up.dim(.dout) }, act_dtype),
        mode,
    ), opts.w1_global_scale, expert_ids);
    const activated = zml.moe.applyActivation(gate_up_out, opts.activation, .concatenated).convert(act_dtype);

    const down_out = applyDownGlobalScale(moeGemm(
        activated,
        down,
        opts.w2_scale,
        expert_ids,
        zml.Shape.init(.{ .r = num_routes, .d = down.dim(.d) }, act_dtype),
        mode,
    ), opts.w2_global_scale, expert_ids);

    const weights = topk_weights.reshape(.{ .token = num_tokens, .topk = topk }).withTags(.{ .token, .topk })
        .merge(.{ .r = .{ .token, .topk } });
    const weighted = down_out.mul(weights.convert(down_out.dtype()).broad(down_out.shape()));
    const combined = weighted.splitAxis(.r, .{ .token = num_tokens, .topk = topk })
        .sum(.topk).squeeze(.topk);

    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.d) });
}
