const zml = @import("zml");

const attn_res = @import("attn_res.zig");
const kda = @import("kda.zig");
const mla = @import("mla.zig");
const moe = @import("moe.zig");
const primitives = @import("primitives.zig");
const router = @import("router.zig");

pub const DenseMlp = struct {
    gate_weight: zml.Tensor,
    up_weight: zml.Tensor,
    down_weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) DenseMlp {
        return .{
            .gate_weight = store.createTensor("gate_proj.weight", .{ .intermediate, .d }, .{ .intermediate = .replicated, .d = .replicated }),
            .up_weight = store.createTensor("up_proj.weight", .{ .intermediate, .d }, .{ .intermediate = .replicated, .d = .replicated }),
            .down_weight = store.createTensor("down_proj.weight", .{ .d, .intermediate }, .{ .d = .replicated, .intermediate = .replicated }),
        };
    }
};

pub const KdaRawWeights = struct {
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

pub const Layer0Weights = struct {
    input_norm: zml.Tensor,
    kda_weights: KdaRawWeights,
    mlp_res_norm: zml.Tensor,
    mlp_res_projection: zml.Tensor,
    post_attention_norm: zml.Tensor,
    mlp: DenseMlp,

    pub fn init(root: zml.io.TensorStore.View) Layer0Weights {
        const store = root.withPrefix("language_model.model.layers.0");
        const attention = store.withPrefix("self_attn");
        return .{
            .input_norm = store.createTensor("input_layernorm.weight", .{.d}, .{ .d = .replicated }),
            .kda_weights = .{
                .q_weight = attention.createTensor("q_proj.weight", .{ .out, .d }, .{ .out = .replicated, .d = .replicated }),
                .k_weight = attention.createTensor("k_proj.weight", .{ .out, .d }, .{ .out = .replicated, .d = .replicated }),
                .v_weight = attention.createTensor("v_proj.weight", .{ .out, .d }, .{ .out = .replicated, .d = .replicated }),
                .q_conv_weight = attention.createTensor("q_conv1d.weight", .{ .channel, .one, .kernel }, .{ .channel = .replicated, .one = .replicated, .kernel = .replicated }),
                .k_conv_weight = attention.createTensor("k_conv1d.weight", .{ .channel, .one, .kernel }, .{ .channel = .replicated, .one = .replicated, .kernel = .replicated }),
                .v_conv_weight = attention.createTensor("v_conv1d.weight", .{ .channel, .one, .kernel }, .{ .channel = .replicated, .one = .replicated, .kernel = .replicated }),
                .decay_a_weight = attention.createTensor("f_a_proj.weight", .{ .out, .d }, .{ .out = .replicated, .d = .replicated }),
                .decay_b_weight = attention.createTensor("f_b_proj.weight", .{ .channel, .rank }, .{ .channel = .replicated, .rank = .replicated }),
                .a_log = attention.createTensor("A_log", .{.h}, .{ .h = .replicated }),
                .dt_bias = attention.createTensor("dt_bias", .{.mix}, .{ .mix = .replicated }),
                .beta_weight = attention.createTensor("b_proj.weight", .{ .h, .d }, .{ .h = .replicated, .d = .replicated }),
                .gate_weight = attention.createTensor("g_proj.weight", .{ .out, .d }, .{ .out = .replicated, .d = .replicated }),
                .norm_weight = attention.createTensor("o_norm.weight", .{.v}, .{ .v = .replicated }),
                .output_weight = attention.createTensor("o_proj.weight", .{ .d, .out }, .{ .d = .replicated, .out = .replicated }),
            },
            .mlp_res_norm = store.createTensor("mlp_res_norm.weight", .{.d}, .{ .d = .replicated }),
            .mlp_res_projection = store.createTensor("mlp_res_proj.weight", .{ .one, .d }, .{ .one = .replicated, .d = .replicated }),
            .post_attention_norm = store.createTensor("post_attention_layernorm.weight", .{.d}, .{ .d = .replicated }),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }
};

// KIMI_K3_TEMP_REMOVE_M20: major layer boundaries and selector weights are
// returned for Gate A activation parity and removed from production arity in M20.
pub const Layer0Result = struct {
    input_norm: zml.Tensor,
    kda_output: zml.Tensor,
    block_residual: zml.Tensor,
    mlp_selector_weights: zml.Tensor,
    post_attention_norm: zml.Tensor,
    mlp_gate: zml.Tensor,
    mlp_up: zml.Tensor,
    mlp_situ: zml.Tensor,
    mlp_output: zml.Tensor,
    output: zml.Tensor,
    cache: kda.Cache,
};

pub fn forwardLayer0(input: zml.Tensor, weights: Layer0Weights, cache: kda.Cache) Layer0Result {
    const input_norm = primitives.rmsNorm(input, weights.input_norm, 1e-5);
    const kda_weights: kda.Weights = .{
        .q_weight = weights.kda_weights.q_weight,
        .k_weight = weights.kda_weights.k_weight,
        .v_weight = weights.kda_weights.v_weight,
        .q_conv_weight = weights.kda_weights.q_conv_weight.squeeze(.one),
        .k_conv_weight = weights.kda_weights.k_conv_weight.squeeze(.one),
        .v_conv_weight = weights.kda_weights.v_conv_weight.squeeze(.one),
        .decay_a_weight = weights.kda_weights.decay_a_weight,
        .decay_b_weight = weights.kda_weights.decay_b_weight,
        .a_log = weights.kda_weights.a_log,
        .dt_bias = weights.kda_weights.dt_bias.splitAxis(.mix, .{ .h = 96, .k = 128 }),
        .beta_weight = weights.kda_weights.beta_weight.rename(.{ .h = .out }),
        .gate_weight = weights.kda_weights.gate_weight,
        .norm_weight = weights.kda_weights.norm_weight,
        .output_weight = weights.kda_weights.output_weight,
    };
    const attention = kda.prefill(input_norm, kda_weights, cache);
    const token_count = input.dim(.b) * input.dim(.s);
    const block_residual = input.merge(.{ .token = .{ .b, .s } }).reshape(.{
        .token = token_count,
        .source = 1,
        .d = input.dim(.d),
    });
    const prefix = attention.output.merge(.{ .token = .{ .b, .s } });
    const active = zml.Tensor.scalar(true, .bool).reshape(.{ .source = 1 });
    const selected = attn_res.select(
        prefix,
        block_residual,
        active,
        weights.mlp_res_norm,
        weights.mlp_res_projection.squeeze(.one),
        1e-5,
    );
    const selected_sequence = selected.output.reshape(.{
        .b = input.dim(.b),
        .s = input.dim(.s),
        .d = input.dim(.d),
    });
    const post_attention_norm = primitives.rmsNorm(selected_sequence, weights.post_attention_norm, 1e-5);
    const mlp_gate = post_attention_norm.dot(weights.mlp.gate_weight, .d);
    const mlp_up = post_attention_norm.dot(weights.mlp.up_weight, .d);
    const mlp_situ = primitives.situGlu(mlp_gate, mlp_up);
    const mlp_output = mlp_situ.dot(weights.mlp.down_weight, .intermediate);
    return .{
        .input_norm = input_norm,
        .kda_output = attention.output,
        .block_residual = block_residual,
        .mlp_selector_weights = selected.probabilities,
        .post_attention_norm = post_attention_norm,
        .mlp_gate = mlp_gate,
        .mlp_up = mlp_up,
        .mlp_situ = mlp_situ,
        .mlp_output = mlp_output,
        .output = attention.output.add(mlp_output),
        .cache = attention.cache,
    };
}

pub const PrefixWeights = struct {
    embedding: zml.Tensor,
    layer0: Layer0Weights,
    output_res_norm: zml.Tensor,
    output_res_projection: zml.Tensor,
    final_norm: zml.Tensor,
    lm_head: zml.Tensor,

    pub fn init(root: zml.io.TensorStore.View) PrefixWeights {
        return .{
            .embedding = root.createTensor(
                "language_model.model.embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .replicated },
            ),
            .layer0 = .init(root),
            .output_res_norm = root.createTensor(
                "language_model.model.output_attn_res_norm.weight",
                .{.d},
                .{ .d = .replicated },
            ),
            .output_res_projection = root.createTensor(
                "language_model.model.output_attn_res_proj.weight",
                .{ .one, .d },
                .{ .one = .replicated, .d = .replicated },
            ),
            .final_norm = root.createTensor(
                "language_model.model.norm.weight",
                .{.d},
                .{ .d = .replicated },
            ),
            .lm_head = root.createTensor(
                "language_model.lm_head.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .replicated },
            ),
        };
    }
};

// KIMI_K3_TEMP_REMOVE_M20: end-to-end prefix activations are returned for
// Gate A differential debugging and removed from production arity in M20.
pub const PrefixResult = struct {
    embedding: zml.Tensor,
    layer_output: zml.Tensor,
    block_residual: zml.Tensor,
    output_candidates: zml.Tensor,
    output_selector_weights: zml.Tensor,
    output_selected: zml.Tensor,
    final_norm: zml.Tensor,
    logits: zml.Tensor,
    greedy_token: zml.Tensor,
    cache: kda.Cache,
};

pub fn forwardPrefix(tokens: zml.Tensor, weights: PrefixWeights, cache: kda.Cache) PrefixResult {
    const embedding = weights.embedding.gather(.{ .voc = tokens.convert(.u32) }, .{});
    const result = forwardLayer0(embedding, weights.layer0, cache);
    const token_count = embedding.dim(.b) * embedding.dim(.s);
    const prefix = result.output.merge(.{ .token = .{ .b, .s } });
    const active = zml.Tensor.scalar(true, .bool).reshape(.{ .source = 1 });
    const selected = attn_res.select(
        prefix,
        result.block_residual,
        active,
        weights.output_res_norm,
        weights.output_res_projection.squeeze(.one),
        1e-5,
    );
    const output_selected = selected.output.reshape(.{
        .b = embedding.dim(.b),
        .s = embedding.dim(.s),
        .d = embedding.dim(.d),
    });
    const final_norm = primitives.rmsNorm(output_selected, weights.final_norm, 1e-5);
    const logits = final_norm.dot(weights.lm_head, .d);
    const greedy_token = logits.slice1d(.s, .{
        .start = logits.dim(.s) - 1,
        .end = logits.dim(.s),
    }).squeeze(.s).argMax(.voc).indices.squeeze(.voc).convert(.i64);
    const output_candidates = zml.Tensor.concatenate(&.{
        result.block_residual,
        prefix.reshape(.{ .token = token_count, .source = 1, .d = embedding.dim(.d) }),
    }, .source);
    return .{
        .embedding = embedding,
        .layer_output = result.output,
        .block_residual = result.block_residual,
        .output_candidates = output_candidates,
        .output_selector_weights = selected.probabilities,
        .output_selected = output_selected,
        .final_norm = final_norm,
        .logits = logits,
        .greedy_token = greedy_token,
        .cache = result.cache,
    };
}

// KIMI_K3_TEMP_REMOVE_M20: the expanded head result is retained while the
// multi-layer prefix is compared boundary-by-boundary with Moonshot. Production
// inference keeps only logits/token outputs after the cleanup milestone.
pub const DiagnosticHeadResult = struct {
    output_candidates: zml.Tensor,
    output_selector_weights: zml.Tensor,
    output_selected: zml.Tensor,
    final_norm: zml.Tensor,
    logits: zml.Tensor,
    greedy_token: zml.Tensor,
};

pub fn diagnosticHead(
    hidden: zml.Tensor,
    block_residual: zml.Tensor,
    output_res_norm: zml.Tensor,
    output_res_projection: zml.Tensor,
    final_norm_weight: zml.Tensor,
    lm_head: zml.Tensor,
) DiagnosticHeadResult {
    const token_count = hidden.dim(.b) * hidden.dim(.s);
    const prefix = hidden.merge(.{ .token = .{ .b, .s } });
    const active = zml.Tensor.scalar(true, .bool).reshape(.{ .source = 1 });
    const selected = attn_res.select(
        prefix,
        block_residual,
        active,
        output_res_norm,
        output_res_projection.squeeze(.one),
        1e-5,
    );
    const output_selected = selected.output.reshape(.{
        .b = hidden.dim(.b),
        .s = hidden.dim(.s),
        .d = hidden.dim(.d),
    });
    const final_norm = primitives.rmsNorm(output_selected, final_norm_weight, 1e-5);
    const logits = final_norm.dot(lm_head, .d);
    const greedy_token = logits.slice1d(.s, .{
        .start = logits.dim(.s) - 1,
        .end = logits.dim(.s),
    }).squeeze(.s).argMax(.voc).indices.squeeze(.voc).convert(.i64);
    const output_candidates = zml.Tensor.concatenate(&.{
        block_residual,
        prefix.reshape(.{ .token = token_count, .source = 1, .d = hidden.dim(.d) }),
    }, .source);
    return .{
        .output_candidates = output_candidates,
        .output_selector_weights = selected.probabilities,
        .output_selected = output_selected,
        .final_norm = final_norm,
        .logits = logits,
        .greedy_token = greedy_token,
    };
}

pub const MoeLayerWeights = struct {
    attention_res_norm: zml.Tensor,
    attention_res_projection: zml.Tensor,
    input_norm: zml.Tensor,
    mlp_res_norm: zml.Tensor,
    mlp_res_projection: zml.Tensor,
    post_attention_norm: zml.Tensor,
    moe: moe.Weights,
};

pub const KdaMoeWeights = struct {
    common: MoeLayerWeights,
    attention: kda.Weights,
};

pub const MlaMoeWeights = struct {
    common: MoeLayerWeights,
    attention: mla.Weights,
};

const SelectionResult = struct {
    output: zml.Tensor,
    probabilities: zml.Tensor,
};

fn selectSequence(
    prefix: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    norm_weight: zml.Tensor,
    projection_weight: zml.Tensor,
) SelectionResult {
    const selected = attn_res.select(
        prefix.merge(.{ .token = .{ .b, .s } }),
        block_sources,
        active_blocks,
        norm_weight,
        projection_weight.squeeze(.one),
        1e-5,
    );
    return .{
        .output = selected.output.reshape(.{
            .b = prefix.dim(.b),
            .s = prefix.dim(.s),
            .d = prefix.dim(.d),
        }),
        .probabilities = selected.probabilities,
    };
}

// KIMI_K3_TEMP_REMOVE_M20: composed layer boundaries and selector/router
// diagnostics are returned for Milestone 14 parity and reduced in cleanup.
pub const KdaMoeResult = struct {
    selected_input: zml.Tensor,
    input_selector_weights: zml.Tensor,
    input_norm: zml.Tensor,
    attention_output: zml.Tensor,
    prefix_after_attention: zml.Tensor,
    selected_mlp: zml.Tensor,
    mlp_selector_weights: zml.Tensor,
    moe_input: zml.Tensor,
    moe_result: moe.Result,
    output: zml.Tensor,
    cache: kda.Cache,
};

pub const KdaMoeBoundaryResult = struct {
    layer: KdaMoeResult,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
};

pub const MlaMoeBoundaryResult = struct {
    layer: MlaMoeResult,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
};

fn appendBoundarySource(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    block_index: zml.Tensor,
) struct { zml.Tensor, zml.Tensor } {
    const source = input.merge(.{ .token = .{ .b, .s } }).reshape(.{
        .token = input.dim(.b) * input.dim(.s),
        .source = 1,
        .d = input.dim(.d),
    });
    const enabled = zml.Tensor.scalar(true, .bool).reshape(.{ .source = 1 });
    return .{
        block_sources.dynamicUpdateSlice(.{ .source = block_index }, source),
        active_blocks.dynamicUpdateSlice(.{ .source = block_index }, enabled),
    };
}

fn finishKdaMoe(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: KdaMoeWeights,
    attention: kda.CompactResult,
    route_config: router.Config,
    selected_input: SelectionResult,
    input_norm: zml.Tensor,
) KdaMoeResult {
    const prefix_after_attention = input.add(attention.output);
    const selected_mlp = selectSequence(
        prefix_after_attention,
        block_sources,
        active_blocks,
        weights.common.mlp_res_norm,
        weights.common.mlp_res_projection,
    );
    const moe_input = primitives.rmsNorm(
        selected_mlp.output,
        weights.common.post_attention_norm,
        1e-5,
    );
    const moe_result = moe.forward(moe_input, weights.common.moe, route_config);
    return .{
        .selected_input = selected_input.output,
        .input_selector_weights = selected_input.probabilities,
        .input_norm = input_norm,
        .attention_output = attention.output,
        .prefix_after_attention = prefix_after_attention,
        .selected_mlp = selected_mlp.output,
        .mlp_selector_weights = selected_mlp.probabilities,
        .moe_input = moe_input,
        .moe_result = moe_result,
        .output = prefix_after_attention.add(moe_result.output),
        .cache = attention.cache,
    };
}

pub fn forwardKdaMoePrefill(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: KdaMoeWeights,
    cache: kda.Cache,
    route_config: router.Config,
) KdaMoeResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const input_norm = primitives.rmsNorm(
        selected_input.output,
        weights.common.input_norm,
        1e-5,
    );
    const attention = kda.prefill(input_norm, weights.attention, cache);
    return finishKdaMoe(
        input,
        block_sources,
        active_blocks,
        weights,
        attention,
        route_config,
        selected_input,
        input_norm,
    );
}

pub fn forwardKdaMoeDecode(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: KdaMoeWeights,
    cache: kda.Cache,
    route_config: router.Config,
) KdaMoeResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const input_norm = primitives.rmsNorm(
        selected_input.output,
        weights.common.input_norm,
        1e-5,
    );
    const step = kda.decodeCompact(input_norm.squeeze(.s), weights.attention, cache);
    const attention: kda.CompactResult = .{
        .output = step.output.reshape(.{
            .b = input.dim(.b),
            .s = 1,
            .d = input.dim(.d),
        }),
        .cache = step.cache,
    };
    return finishKdaMoe(
        input,
        block_sources,
        active_blocks,
        weights,
        attention,
        route_config,
        selected_input,
        input_norm,
    );
}

/// Official AttnRes block-boundary decode. Attention input selection uses the
/// old source set; MLP selection uses the source appended at this layer.
pub fn forwardKdaMoeBoundary(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    block_index: zml.Tensor,
    weights: KdaMoeWeights,
    cache: kda.Cache,
    route_config: router.Config,
) KdaMoeBoundaryResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const updated = appendBoundarySource(input, block_sources, active_blocks, block_index);
    const input_norm = primitives.rmsNorm(selected_input.output, weights.common.input_norm, 1e-5);
    const step = kda.decodeCompact(input_norm.squeeze(.s), weights.attention, cache);
    const attention_output = step.output.reshape(.{
        .b = input.dim(.b),
        .s = 1,
        .d = input.dim(.d),
    });
    const selected_mlp = selectSequence(
        attention_output,
        updated[0],
        updated[1],
        weights.common.mlp_res_norm,
        weights.common.mlp_res_projection,
    );
    const moe_input = primitives.rmsNorm(selected_mlp.output, weights.common.post_attention_norm, 1e-5);
    const moe_result = moe.forward(moe_input, weights.common.moe, route_config);
    return .{
        .layer = .{
            .selected_input = selected_input.output,
            .input_selector_weights = selected_input.probabilities,
            .input_norm = input_norm,
            .attention_output = attention_output,
            .prefix_after_attention = attention_output,
            .selected_mlp = selected_mlp.output,
            .mlp_selector_weights = selected_mlp.probabilities,
            .moe_input = moe_input,
            .moe_result = moe_result,
            .output = attention_output.add(moe_result.output),
            .cache = step.cache,
        },
        .block_sources = updated[0],
        .active_blocks = updated[1],
    };
}

// KIMI_K3_TEMP_REMOVE_M20: composed latent-MLA/MoE boundaries are exposed to
// the Milestone 14 harness and reduced to production output/cache in cleanup.
pub const MlaMoeResult = struct {
    selected_input: zml.Tensor,
    input_selector_weights: zml.Tensor,
    input_norm: zml.Tensor,
    attention_output: zml.Tensor,
    prefix_after_attention: zml.Tensor,
    selected_mlp: zml.Tensor,
    mlp_selector_weights: zml.Tensor,
    moe_input: zml.Tensor,
    moe_result: moe.Result,
    output: zml.Tensor,
    cache: mla.LatentCache,
};

fn finishMlaMoe(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: MlaMoeWeights,
    attention: mla.CompactResult,
    route_config: router.Config,
    selected_input: SelectionResult,
    input_norm: zml.Tensor,
) MlaMoeResult {
    const prefix_after_attention = input.add(attention.output);
    const selected_mlp = selectSequence(
        prefix_after_attention,
        block_sources,
        active_blocks,
        weights.common.mlp_res_norm,
        weights.common.mlp_res_projection,
    );
    const moe_input = primitives.rmsNorm(
        selected_mlp.output,
        weights.common.post_attention_norm,
        1e-5,
    );
    const moe_result = moe.forward(moe_input, weights.common.moe, route_config);
    return .{
        .selected_input = selected_input.output,
        .input_selector_weights = selected_input.probabilities,
        .input_norm = input_norm,
        .attention_output = attention.output,
        .prefix_after_attention = prefix_after_attention,
        .selected_mlp = selected_mlp.output,
        .mlp_selector_weights = selected_mlp.probabilities,
        .moe_input = moe_input,
        .moe_result = moe_result,
        .output = prefix_after_attention.add(moe_result.output),
        .cache = attention.cache,
    };
}

pub fn forwardMlaMoePrefill(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: MlaMoeWeights,
    route_config: router.Config,
) MlaMoeResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const input_norm = primitives.rmsNorm(
        selected_input.output,
        weights.common.input_norm,
        1e-5,
    );
    const attention = mla.latentPrefillCompact(input_norm, weights.attention);
    return finishMlaMoe(
        input,
        block_sources,
        active_blocks,
        weights,
        attention,
        route_config,
        selected_input,
        input_norm,
    );
}

pub fn forwardMlaMoeContinue(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: MlaMoeWeights,
    cache: mla.LatentCache,
    route_config: router.Config,
) MlaMoeResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const input_norm = primitives.rmsNorm(
        selected_input.output,
        weights.common.input_norm,
        1e-5,
    );
    const attention = mla.latentContinueCompact(input_norm, weights.attention, cache);
    return finishMlaMoe(
        input,
        block_sources,
        active_blocks,
        weights,
        attention,
        route_config,
        selected_input,
        input_norm,
    );
}

/// Fixed-capacity MLA+MoE session step. The returned cache retains the input
/// allocation shape so the same executable is valid at every token position.
pub fn forwardMlaMoeSession(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    weights: MlaMoeWeights,
    cache: mla.SessionCache,
    token_index: zml.Tensor,
    route_config: router.Config,
) MlaMoeResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const input_norm = primitives.rmsNorm(
        selected_input.output,
        weights.common.input_norm,
        1e-5,
    );
    const attention = mla.latentSessionCompact(input_norm, weights.attention, cache, token_index);
    return finishMlaMoe(
        input,
        block_sources,
        active_blocks,
        weights,
        attention,
        route_config,
        selected_input,
        input_norm,
    );
}

/// Fixed-capacity latent-MLA session step at an official AttnRes boundary.
pub fn forwardMlaMoeBoundary(
    input: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    block_index: zml.Tensor,
    weights: MlaMoeWeights,
    cache: mla.SessionCache,
    token_index: zml.Tensor,
    route_config: router.Config,
) MlaMoeBoundaryResult {
    const selected_input = selectSequence(
        input,
        block_sources,
        active_blocks,
        weights.common.attention_res_norm,
        weights.common.attention_res_projection,
    );
    const updated = appendBoundarySource(input, block_sources, active_blocks, block_index);
    const input_norm = primitives.rmsNorm(selected_input.output, weights.common.input_norm, 1e-5);
    const attention = mla.latentSessionCompact(input_norm, weights.attention, cache, token_index);
    const selected_mlp = selectSequence(
        attention.output,
        updated[0],
        updated[1],
        weights.common.mlp_res_norm,
        weights.common.mlp_res_projection,
    );
    const moe_input = primitives.rmsNorm(selected_mlp.output, weights.common.post_attention_norm, 1e-5);
    const moe_result = moe.forward(moe_input, weights.common.moe, route_config);
    return .{
        .layer = .{
            .selected_input = selected_input.output,
            .input_selector_weights = selected_input.probabilities,
            .input_norm = input_norm,
            .attention_output = attention.output,
            .prefix_after_attention = attention.output,
            .selected_mlp = selected_mlp.output,
            .mlp_selector_weights = selected_mlp.probabilities,
            .moe_input = moe_input,
            .moe_result = moe_result,
            .output = attention.output.add(moe_result.output),
            .cache = attention.cache,
        },
        .block_sources = updated[0],
        .active_blocks = updated[1],
    };
}

pub fn diagnosticSessionHead(
    hidden: zml.Tensor,
    block_residual: zml.Tensor,
    active_blocks: zml.Tensor,
    output_res_norm: zml.Tensor,
    output_res_projection: zml.Tensor,
    final_norm_weight: zml.Tensor,
    lm_head: zml.Tensor,
) DiagnosticHeadResult {
    const token_count = hidden.dim(.b) * hidden.dim(.s);
    const prefix = hidden.merge(.{ .token = .{ .b, .s } });
    const selected = attn_res.select(
        prefix,
        block_residual,
        active_blocks,
        output_res_norm,
        output_res_projection.squeeze(.one),
        1e-5,
    );
    const output_selected = selected.output.reshape(.{
        .b = hidden.dim(.b),
        .s = hidden.dim(.s),
        .d = hidden.dim(.d),
    });
    const final_norm = primitives.rmsNorm(output_selected, final_norm_weight, 1e-5);
    const logits = final_norm.dot(lm_head, .d);
    const greedy_token = logits.slice1d(.s, .{
        .start = logits.dim(.s) - 1,
        .end = logits.dim(.s),
    }).squeeze(.s).argMax(.voc).indices.squeeze(.voc).convert(.i64);
    const output_candidates = zml.Tensor.concatenate(&.{
        block_residual,
        prefix.reshape(.{ .token = token_count, .source = 1, .d = hidden.dim(.d) }),
    }, .source);
    return .{
        .output_candidates = output_candidates,
        .output_selector_weights = selected.probabilities,
        .output_selected = output_selected,
        .final_norm = final_norm,
        .logits = logits,
        .greedy_token = greedy_token,
    };
}
