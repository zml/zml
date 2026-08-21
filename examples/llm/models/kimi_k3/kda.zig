const std = @import("std");

const zml = @import("zml");
const kda_cache = @import("kda_cache.zig");
const primitives = @import("primitives.zig");
const recurrent_kernel = zml.attention.kda_recurrent_kernel;

pub const Cache = kda_cache.Cache;

pub const Weights = struct {
    q_weight: zml.Tensor,
    k_weight: zml.Tensor,
    v_weight: zml.Tensor,
    q_conv_weight: zml.Tensor,
    k_conv_weight: zml.Tensor,
    v_conv_weight: zml.Tensor,
    decay_a_weight: zml.Tensor,
    decay_b_weight: zml.Tensor,
    a_log: zml.Tensor,
    dt_bias: zml.Tensor,
    beta_weight: zml.Tensor,
    gate_weight: zml.Tensor,
    norm_weight: zml.Tensor,
    output_weight: zml.Tensor,
};

/// Named decode boundaries returned only by the Milestone 7 differential
/// executable. Production integration will consume `projection_output` and
/// `cache` without transferring these diagnostics to the host.
pub const DecodeResult = struct {
    q_proj: zml.Tensor,
    k_proj: zml.Tensor,
    v_proj: zml.Tensor,
    q_conv: zml.Tensor,
    k_conv: zml.Tensor,
    v_conv: zml.Tensor,
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    q_norm: zml.Tensor,
    k_norm: zml.Tensor,
    raw_decay: zml.Tensor,
    log_alpha: zml.Tensor,
    alpha: zml.Tensor,
    raw_beta: zml.Tensor,
    beta: zml.Tensor,
    prediction: zml.Tensor,
    error_value: zml.Tensor,
    recurrent_output: zml.Tensor,
    output_gate: zml.Tensor,
    norm_gated: zml.Tensor,
    projection_output: zml.Tensor,
    cache: Cache,
};

fn linear(input: zml.Tensor, weight: zml.Tensor) zml.Tensor {
    return input.dot(weight, .d);
}

const ConvResult = struct {
    output: zml.Tensor,
    cache: zml.Tensor,
};

const ConvSequenceResult = struct {
    output: zml.Tensor,
    cache: zml.Tensor,
};

/// Exact single-token `ShortConvolution.step`: shift the full kernel window,
/// append the projected token, correlate per channel, then apply SiLU.
fn convStep(projected: zml.Tensor, previous: zml.Tensor, weight: zml.Tensor) ConvResult {
    const current = projected.rename(.{ .out = .channel }).reshape(.{
        .b = projected.dim(.b),
        .channel = projected.dim(.out),
        .kernel = 1,
    });
    const retained = previous.slice1d(.kernel, .{ .start = 1, .end = previous.dim(.kernel) });
    const cache = zml.Tensor.concatenate(&.{ retained, current }, .kernel);
    const output = cache.convert(.f32)
        .mul(weight.convert(.f32).broad(cache.shape()))
        .sum(.kernel).squeeze(.kernel).silu().convert(projected.dtype());
    return .{ .output = output, .cache = cache };
}

/// Readable causal sequence convolution with a full Moonshot decode window.
/// Concatenating the prior width-W window and slicing outputs at W makes the
/// first new token consume cache[1..W] plus itself, exactly like `step`.
fn convSequence(projected: zml.Tensor, previous: zml.Tensor, weight: zml.Tensor) ConvSequenceResult {
    const projected_sequence = projected.rename(.{ .out = .channel });
    const history = previous.transpose(.{ .b, .kernel, .channel }).rename(.{ .kernel = .s });
    const input = zml.Tensor.concatenate(&.{ history, projected_sequence }, .s);
    const kernel = weight.reshape(.{
        .channel = weight.dim(.channel),
        .one = 1,
        .kernel = weight.dim(.kernel),
    });
    const mixed = primitives.causalDepthwiseConv1d(
        input.convert(.f32).rename(.{ .b = .batch, .s = .sequence }),
        kernel.convert(.f32),
    ).silu().convert(projected.dtype()).rename(.{ .batch = .b, .sequence = .s });
    const output = mixed.slice1d(.s, .{
        .start = previous.dim(.kernel),
        .end = input.dim(.s),
    });
    const final_history = input.slice1d(.s, .{
        .start = input.dim(.s) - previous.dim(.kernel),
        .end = input.dim(.s),
    }).rename(.{ .s = .kernel }).transpose(.{ .b, .channel, .kernel });
    return .{ .output = output, .cache = final_history };
}

pub const CompactResult = struct {
    output: zml.Tensor,
    cache: Cache,
};

pub fn decodeCompact(hidden: zml.Tensor, weights: Weights, cache: Cache) CompactResult {
    return decodeOptimized(hidden, weights, cache);
}

const Scan = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    alpha: zml.Tensor,
    beta: zml.Tensor,

    pub const State = struct {
        recurrent: zml.Tensor,
        outputs: zml.Tensor,
        step: zml.Tensor,
    };

    fn sliceStep(input: zml.Tensor, step: zml.Tensor) zml.Tensor {
        return input.dynamicSlice(.{ .s = zml.Tensor.DynSlice{ .start = step, .len = 1 } }).squeeze(.s);
    }

    pub fn cond(scan: Scan, state: State) zml.Tensor {
        return state.step.cmp(.LT, .scalar(scan.q.dim(.s), .i32));
    }

    pub fn body(scan: Scan, state: State) State {
        const q = sliceStep(scan.q, state.step);
        const k = sliceStep(scan.k, state.step);
        const v = sliceStep(scan.v, state.step);
        const alpha = sliceStep(scan.alpha, state.step);
        const beta = sliceStep(scan.beta, state.step);
        const decayed = state.recurrent.mul(alpha.broad(state.recurrent.shape()));
        const prediction = decayed.dot(k, .k);
        const error_value = v.sub(prediction).mul(beta.broad(prediction.shape()));
        const recurrent = decayed.add(
            error_value.broad(state.recurrent.shape()).mul(k.broad(state.recurrent.shape())),
        );
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q.dim(.k))));
        const output = recurrent.dot(q, .k).scale(scale);
        return .{
            .recurrent = recurrent,
            .outputs = state.outputs.dynamicUpdateSlice(.{ .s = state.step }, output),
            .step = state.step.addConstant(1),
        };
    }
};

pub const RecurrentResult = struct {
    output: zml.Tensor,
    state: zml.Tensor,
};

pub fn recurrentOptimized(
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    alpha: zml.Tensor,
    beta: zml.Tensor,
    state: zml.Tensor,
) RecurrentResult {
    std.debug.assert(q.dtype() == .f32 and k.dtype() == .f32 and v.dtype() == .f32);
    std.debug.assert(alpha.dtype() == .f32 and beta.dtype() == .f32 and state.dtype() == .f32);
    std.debug.assert(q.shape().hasTags(.{ .b, .s, .h, .k }));
    std.debug.assert(v.shape().hasTags(.{ .b, .s, .h, .v }));
    std.debug.assert(state.shape().hasTags(.{ .b, .h, .v, .k }));

    const batch = q.dim(.b);
    const sequence = q.dim(.s);
    const heads = q.dim(.h);
    const key_dim = q.dim(.k);
    const value_dim = v.dim(.v);
    const block_v: i64 = 32;
    const block_k: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(key_dim)));
    const value_tiles = std.math.divCeil(i64, value_dim, block_v) catch unreachable;
    const results = recurrent_kernel.Kernel.call(
        .{
            .q_ptr = q,
            .k_ptr = k,
            .v_ptr = v,
            .alpha_ptr = alpha,
            .beta_ptr = beta,
            .state_ptr = state,
        },
        .{
            .recurrent_output = v.shape().withDtype(.f32),
            .state_output = state.shape().withDtype(.f32),
        },
        .{
            .cfg = .{
                .batch = @intCast(batch),
                .sequence = @intCast(sequence),
                .heads = @intCast(heads),
                .value_dim = @intCast(value_dim),
                .key_dim = @intCast(key_dim),
                .block_v = @intCast(block_v),
                .block_k = @intCast(block_k),
                .input_dtype = .f32,
            },
            .grid = .{ @intCast(batch * heads * value_tiles), 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
            .output_operand_aliases = .{ .state_output = .state_ptr },
        },
    );
    return .{ .output = results.recurrent_output, .state = results.state_output };
}

/// Sequential StableHLO recurrence retained as the independent oracle.
pub fn recurrentReference(
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    alpha: zml.Tensor,
    beta: zml.Tensor,
    state: zml.Tensor,
) RecurrentResult {
    const initial: Scan.State = .{
        .recurrent = state.convert(.f32),
        .outputs = zml.Tensor.zeroes(v.shape().withDtype(.f32)),
        .step = .scalar(0, .i32),
    };
    const final = zml.ops.@"while"(Scan, .{ .q = q, .k = k, .v = v, .alpha = alpha, .beta = beta }, initial);
    return .{ .output = final.outputs, .state = final.recurrent };
}

fn prefillImpl(hidden: zml.Tensor, weights: Weights, cache: Cache, comptime optimized: bool) CompactResult {
    const q_proj = linear(hidden, weights.q_weight);
    const k_proj = linear(hidden, weights.k_weight);
    const v_proj = linear(hidden, weights.v_weight);
    const q_conv = convSequence(q_proj, cache.q_conv, weights.q_conv_weight);
    const k_conv = convSequence(k_proj, cache.k_conv, weights.k_conv_weight);
    const v_conv = convSequence(v_proj, cache.v_conv, weights.v_conv_weight);
    const heads = weights.dt_bias.dim(.h);
    const head_dim = weights.norm_weight.dim(.v);
    const q = primitives.normalizeL2(
        q_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim }),
        1e-6,
    ).convert(.f32);
    const k = primitives.normalizeL2(
        k_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim }),
        1e-6,
    ).convert(.f32);
    const v = v_conv.output.rename(.{ .channel = .mix })
        .splitAxis(.mix, .{ .h = heads, .v = head_dim }).convert(.f32);
    const decay_rank = linear(hidden, weights.decay_a_weight).rename(.{ .out = .rank });
    const raw_decay = decay_rank.dot(weights.decay_b_weight, .rank)
        .rename(.{ .channel = .mix })
        .splitAxis(.mix, .{ .h = heads, .k = head_dim })
        .convert(.f32);
    const decay_input = raw_decay.add(weights.dt_bias.convert(.f32).broad(raw_decay.shape()));
    const decay_rate = weights.a_log.slice1d(.h, .{ .start = 0, .end = heads }).convert(.f32).exp()
        .reshape(.{ .b = 1, .s = 1, .h = heads, .k = 1 })
        .broad(raw_decay.shape());
    const alpha = decay_rate.mul(decay_input).sigmoid().scale(-5.0).exp();
    const beta = linear(hidden, weights.beta_weight).rename(.{ .out = .h }).convert(.f32).sigmoid();
    const recurrence: RecurrentResult = if (optimized) recurrentOptimized(q, k, v, alpha, beta, cache.recurrent_state.convert(.f32)) else reference: {
        const initial_state: Scan.State = .{
            .recurrent = cache.recurrent_state.convert(.f32),
            .outputs = zml.Tensor.zeroes(v.shape()),
            .step = .scalar(0, .i32),
        };
        const final = zml.ops.@"while"(Scan, .{ .q = q, .k = k, .v = v, .alpha = alpha, .beta = beta }, initial_state);
        break :reference .{ .output = final.outputs, .state = final.recurrent };
    };
    const output_gate = linear(hidden, weights.gate_weight)
        .rename(.{ .out = .mix })
        .splitAxis(.mix, .{ .h = heads, .v = head_dim })
        .convert(.f32);
    const variance = recurrence.output.powByConst(2).mean(.v);
    const norm_gated = recurrence.output
        .mul(variance.addConstant(1e-5).rsqrt().broad(recurrence.output.shape()))
        .mul(weights.norm_weight.convert(.f32).broad(recurrence.output.shape()))
        .mul(output_gate.sigmoid());
    const output = norm_gated.merge(.{ .out = .{ .h, .v } })
        .convert(hidden.dtype()).dot(weights.output_weight, .out);
    return .{
        .output = output,
        .cache = .{
            .q_conv = q_conv.cache,
            .k_conv = k_conv.cache,
            .v_conv = v_conv.cache,
            .recurrent_state = recurrence.state,
        },
    };
}
/// Production KDA prefill. CUDA lowers the complete channel-wise recurrent
/// scan to one fused Triton kernel while retaining the compact FP32 state.
pub fn prefill(hidden: zml.Tensor, weights: Weights, cache: Cache) CompactResult {
    return prefillImpl(hidden, weights, cache, true);
}

/// Sequential StableHLO oracle retained for differential tests.
pub fn prefillReference(hidden: zml.Tensor, weights: Weights, cache: Cache) CompactResult {
    return prefillImpl(hidden, weights, cache, false);
}

/// Readable Kimi K3 fused-recurrent decode equation.
///
/// Sensitive decay, recurrence, and gated-normalization operations stay in
/// FP32. This is a correctness path; Milestone 18 owns fused optimization.
pub fn decode(hidden: zml.Tensor, weights: Weights, cache: Cache) DecodeResult {
    const q_proj = linear(hidden, weights.q_weight);
    const k_proj = linear(hidden, weights.k_weight);
    const v_proj = linear(hidden, weights.v_weight);
    const q_conv = convStep(q_proj, cache.q_conv, weights.q_conv_weight);
    const k_conv = convStep(k_proj, cache.k_conv, weights.k_conv_weight);
    const v_conv = convStep(v_proj, cache.v_conv, weights.v_conv_weight);

    const heads = weights.dt_bias.dim(.h);
    const head_dim = weights.norm_weight.dim(.v);
    const q = q_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim });
    const k = k_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim });
    const v = v_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .v = head_dim });
    const q_norm = primitives.normalizeL2(q, 1e-6).convert(.f32);
    const k_norm = primitives.normalizeL2(k, 1e-6).convert(.f32);

    const decay_rank = linear(hidden, weights.decay_a_weight).rename(.{ .out = .rank });
    const raw_decay = decay_rank.dot(weights.decay_b_weight, .rank)
        .rename(.{ .channel = .mix })
        .splitAxis(.mix, .{ .h = heads, .k = head_dim })
        .convert(.f32);
    const decay_input = raw_decay.add(weights.dt_bias.convert(.f32).broad(raw_decay.shape()));
    const decay_rate = weights.a_log.slice1d(.h, .{ .start = 0, .end = heads }).convert(.f32).exp()
        .reshape(.{ .b = 1, .h = heads, .k = 1 })
        .broad(raw_decay.shape());
    const log_alpha = decay_rate.mul(decay_input).sigmoid().scale(-5.0);
    const alpha = log_alpha.exp();
    const raw_beta = linear(hidden, weights.beta_weight).rename(.{ .out = .h }).convert(.f32);
    const beta = raw_beta.sigmoid();

    const state_f32 = cache.recurrent_state.convert(.f32);
    const decayed_state = state_f32.mul(alpha.broad(state_f32.shape()));
    const prediction = decayed_state.dot(k_norm, .k);
    const error_value = v.convert(.f32).sub(prediction).mul(beta.broad(prediction.shape()));
    const correction = error_value.broad(state_f32.shape()).mul(k_norm.broad(state_f32.shape()));
    const recurrent_state = decayed_state.add(correction);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    const recurrent_output = recurrent_state.dot(q_norm, .k).scale(scale);

    const output_gate = linear(hidden, weights.gate_weight)
        .rename(.{ .out = .mix })
        .splitAxis(.mix, .{ .h = heads, .v = head_dim })
        .convert(.f32);
    const variance = recurrent_output.powByConst(2).mean(.v);
    const normalized = recurrent_output.mul(
        variance.addConstant(1e-5).rsqrt().broad(recurrent_output.shape()),
    );
    const norm_gated = normalized
        .mul(weights.norm_weight.convert(.f32).broad(normalized.shape()))
        .mul(output_gate.sigmoid());
    const flattened = norm_gated.merge(.{ .out = .{ .h, .v } }).convert(hidden.dtype());
    const projection_output = flattened.dot(weights.output_weight, .out);

    return .{
        .q_proj = q_proj,
        .k_proj = k_proj,
        .v_proj = v_proj,
        .q_conv = q_conv.output,
        .k_conv = k_conv.output,
        .v_conv = v_conv.output,
        .q = q,
        .k = k,
        .v = v,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .raw_decay = raw_decay,
        .log_alpha = log_alpha,
        .alpha = alpha,
        .raw_beta = raw_beta,
        .beta = beta,
        .prediction = prediction,
        .error_value = error_value,
        .recurrent_output = recurrent_output,
        .output_gate = output_gate,
        .norm_gated = norm_gated,
        .projection_output = projection_output,
        .cache = .{
            .q_conv = q_conv.cache,
            .k_conv = k_conv.cache,
            .v_conv = v_conv.cache,
            .recurrent_state = recurrent_state,
        },
    };
}

/// Readable single-token oracle retained for differential tests.
pub fn decodeCompactReference(hidden: zml.Tensor, weights: Weights, cache: Cache) CompactResult {
    const result = decode(hidden, weights, cache);
    return .{ .output = result.projection_output, .cache = result.cache };
}

fn decodeOptimized(hidden: zml.Tensor, weights: Weights, cache: Cache) CompactResult {
    const q_proj = linear(hidden, weights.q_weight);
    const k_proj = linear(hidden, weights.k_weight);
    const v_proj = linear(hidden, weights.v_weight);
    const q_conv = convStep(q_proj, cache.q_conv, weights.q_conv_weight);
    const k_conv = convStep(k_proj, cache.k_conv, weights.k_conv_weight);
    const v_conv = convStep(v_proj, cache.v_conv, weights.v_conv_weight);

    const batch = hidden.dim(.b);
    const heads = weights.dt_bias.dim(.h);
    const head_dim = weights.norm_weight.dim(.v);
    const q = q_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim });
    const k = k_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .k = head_dim });
    const v = v_conv.output.rename(.{ .channel = .mix }).splitAxis(.mix, .{ .h = heads, .v = head_dim }).convert(.f32);
    const q_norm = primitives.normalizeL2(q, 1e-6).convert(.f32);
    const k_norm = primitives.normalizeL2(k, 1e-6).convert(.f32);

    const decay_rank = linear(hidden, weights.decay_a_weight).rename(.{ .out = .rank });
    const raw_decay = decay_rank.dot(weights.decay_b_weight, .rank)
        .rename(.{ .channel = .mix })
        .splitAxis(.mix, .{ .h = heads, .k = head_dim })
        .convert(.f32);
    const decay_input = raw_decay.add(weights.dt_bias.convert(.f32).broad(raw_decay.shape()));
    const decay_rate = weights.a_log.slice1d(.h, .{ .start = 0, .end = heads }).convert(.f32).exp()
        .reshape(.{ .b = 1, .h = heads, .k = 1 })
        .broad(raw_decay.shape());
    const alpha = decay_rate.mul(decay_input).sigmoid().scale(-5.0).exp();
    const beta = linear(hidden, weights.beta_weight).rename(.{ .out = .h }).convert(.f32).sigmoid();

    const recurrence = recurrentOptimized(
        q_norm.reshape(.{ .b = batch, .s = 1, .h = heads, .k = head_dim }),
        k_norm.reshape(.{ .b = batch, .s = 1, .h = heads, .k = head_dim }),
        v.reshape(.{ .b = batch, .s = 1, .h = heads, .v = head_dim }),
        alpha.reshape(.{ .b = batch, .s = 1, .h = heads, .k = head_dim }),
        beta.reshape(.{ .b = batch, .s = 1, .h = heads }),
        cache.recurrent_state.convert(.f32),
    );
    const recurrent_output = recurrence.output.squeeze(.s);
    const output_gate = linear(hidden, weights.gate_weight)
        .rename(.{ .out = .mix })
        .splitAxis(.mix, .{ .h = heads, .v = head_dim })
        .convert(.f32);
    const variance = recurrent_output.powByConst(2).mean(.v);
    const normalized = recurrent_output.mul(
        variance.addConstant(1e-5).rsqrt().broad(recurrent_output.shape()),
    );
    const norm_gated = normalized
        .mul(weights.norm_weight.convert(.f32).broad(normalized.shape()))
        .mul(output_gate.sigmoid());
    const output = norm_gated.merge(.{ .out = .{ .h, .v } })
        .convert(hidden.dtype()).dot(weights.output_weight, .out);
    return .{
        .output = output,
        .cache = .{
            .q_conv = q_conv.cache,
            .k_conv = k_conv.cache,
            .v_conv = v_conv.cache,
            .recurrent_state = recurrence.state,
        },
    };
}
