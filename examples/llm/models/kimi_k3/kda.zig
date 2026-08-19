const std = @import("std");

const zml = @import("zml");
const kda_cache = @import("kda_cache.zig");
const primitives = @import("primitives.zig");

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
// KIMI_K3_TEMP_REMOVE_M20: diagnostic result arity exposes intermediate KDA
// activations for parity debugging and must be removed during cleanup.
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
    const output = cache.mul(weight.broad(cache.shape())).sum(.kernel).squeeze(.kernel).silu();
    return .{ .output = output, .cache = cache };
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

    const heads = weights.a_log.dim(.h);
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
    const decay_rate = weights.a_log.convert(.f32).exp()
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
    const flattened = norm_gated.merge(.{ .out = .{ .h, .v } });
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
