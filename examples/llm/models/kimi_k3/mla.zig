const std = @import("std");

const zml = @import("zml");

pub const Weights = struct {
    q_a_proj: zml.Tensor,
    q_a_norm: zml.Tensor,
    q_b_proj: zml.Tensor,
    kv_a_proj: zml.Tensor,
    kv_a_norm: zml.Tensor,
    kv_b_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    output_proj: zml.Tensor,
};

/// Expanded per-head K/V cache used as the readable Milestone 12 oracle.
/// Milestone 13 replaces this with the 512+64 latent cache algebra.
pub const ExpandedCache = struct {
    key: zml.Tensor,
    value: zml.Tensor,
};

// KIMI_K3_TEMP_REMOVE_M20: named MLA boundaries are returned to the isolated
// differential harness and removed from the production result during cleanup.
pub const Result = struct {
    q_a: zml.Tensor,
    q_norm: zml.Tensor,
    q_b: zml.Tensor,
    q_pass: zml.Tensor,
    q_extra: zml.Tensor,
    kv_a: zml.Tensor,
    compressed_kv: zml.Tensor,
    k_extra: zml.Tensor,
    kv_norm: zml.Tensor,
    kv_b: zml.Tensor,
    k_pass: zml.Tensor,
    value_new: zml.Tensor,
    query: zml.Tensor,
    key_new: zml.Tensor,
    cache: ExpandedCache,
    scores: zml.Tensor,
    masked_scores: zml.Tensor,
    probabilities: zml.Tensor,
    aggregation: zml.Tensor,
    flattened: zml.Tensor,
    gate_logits: zml.Tensor,
    gate: zml.Tensor,
    gated: zml.Tensor,
    output: zml.Tensor,
};

fn linear(input: zml.Tensor, weight: zml.Tensor, axis: anytype) zml.Tensor {
    return input.dot(weight, axis);
}

fn weightedRmsNorm(input: zml.Tensor, weight: zml.Tensor, axis: anytype) zml.Tensor {
    const normalized = zml.nn.rmsNorm(input, axis, 1e-5);
    return normalized.convert(.f32)
        .mul(weight.convert(.f32).broad(normalized.shape()))
        .convert(input.dtype());
}

fn core(hidden: zml.Tensor, weights: Weights, past: ?ExpandedCache) Result {
    const batch = hidden.dim(.b);
    const sequence = hidden.dim(.s);
    const heads: i64 = 96;

    const q_a = linear(hidden, weights.q_a_proj, .d);
    const q_norm = weightedRmsNorm(q_a, weights.q_a_norm, .rank);
    const q_b = linear(q_norm, weights.q_b_proj, .rank);
    const q_heads = q_b.splitAxis(.mix, .{ .h = heads, .hd = 192 })
        .transpose(.{ .b, .h, .s, .hd }).rename(.{ .s = .q });
    const q_pass = q_heads.slice1d(.hd, .{ .start = 0, .end = 128 });
    const q_extra = q_heads.slice1d(.hd, .{ .start = 128, .end = 192 });

    const kv_a = linear(hidden, weights.kv_a_proj, .d);
    const compressed_kv = kv_a.slice1d(.kv_mix, .{ .start = 0, .end = 512 })
        .rename(.{ .kv_mix = .kv_rank });
    const k_extra = kv_a.slice1d(.kv_mix, .{ .start = 512, .end = 576 })
        .rename(.{ .kv_mix = .extra });
    const kv_norm = weightedRmsNorm(compressed_kv, weights.kv_a_norm, .kv_rank);
    const kv_b = linear(kv_norm, weights.kv_b_proj, .kv_rank);
    const kv_heads = kv_b.splitAxis(.kv_mix, .{ .h = heads, .kv_width = 256 })
        .transpose(.{ .b, .h, .s, .kv_width }).rename(.{ .s = .k });
    const k_pass = kv_heads.slice1d(.kv_width, .{ .start = 0, .end = 128 })
        .rename(.{ .kv_width = .hd });
    const value_new = kv_heads.slice1d(.kv_width, .{ .start = 128, .end = 256 })
        .rename(.{ .kv_width = .v });
    const k_extra_heads = k_extra.reshape(.{ .b = batch, .h = 1, .k = sequence, .hd = 64 })
        .broad(zml.Shape.init(.{ .b = batch, .h = heads, .k = sequence, .hd = 64 }, hidden.dtype()));
    const query = zml.Tensor.concatenate(&.{ q_pass, q_extra }, .hd);
    const key_new = zml.Tensor.concatenate(&.{ k_pass, k_extra_heads }, .hd);
    const cache: ExpandedCache = if (past) |previous| .{
        .key = zml.Tensor.concatenate(&.{ previous.key, key_new }, .k),
        .value = zml.Tensor.concatenate(&.{ previous.value, value_new }, .k),
    } else .{ .key = key_new, .value = value_new };

    const scores = query.dot(cache.key, .hd).scale(1.0 / std.math.sqrt(192.0));
    const past_length: i64 = if (past) |previous| previous.key.dim(.k) else 0;
    const query_index = zml.Tensor.iota(scores.shape(), .q).addConstant(past_length);
    const key_index = zml.Tensor.iota(scores.shape(), .k);
    const masked_scores = key_index.cmp(.LE, query_index).select(
        scores,
        zml.Tensor.scalar(-std.math.inf(f32), scores.dtype()),
    );
    const probabilities = masked_scores.convert(.f32).softmax(.k).convert(query.dtype());
    const aggregation = probabilities.dot(cache.value, .k);
    const flattened = aggregation.transpose(.{ .b, .q, .h, .v })
        .rename(.{ .q = .s }).merge(.{ .out = .{ .h, .v } });
    const gate_logits = linear(hidden, weights.gate_proj, .d);
    const gate = gate_logits.sigmoid();
    const gated = flattened.mul(gate);
    const output = linear(gated, weights.output_proj, .out);
    return .{
        .q_a = q_a,
        .q_norm = q_norm,
        .q_b = q_b,
        .q_pass = q_pass,
        .q_extra = q_extra,
        .kv_a = kv_a,
        .compressed_kv = compressed_kv,
        .k_extra = k_extra,
        .kv_norm = kv_norm,
        .kv_b = kv_b,
        .k_pass = k_pass,
        .value_new = value_new,
        .query = query,
        .key_new = key_new,
        .cache = cache,
        .scores = scores,
        .masked_scores = masked_scores,
        .probabilities = probabilities,
        .aggregation = aggregation,
        .flattened = flattened,
        .gate_logits = gate_logits,
        .gate = gate,
        .gated = gated,
        .output = output,
    };
}

/// Readable expanded-cache causal prefill used for Gate C correctness.
pub fn prefill(hidden: zml.Tensor, weights: Weights) Result {
    return core(hidden, weights, null);
}

/// Readable expanded-cache single-step continuation used for Gate C.
pub fn decode(hidden: zml.Tensor, weights: Weights, cache: ExpandedCache) Result {
    return core(hidden, weights, cache);
}
