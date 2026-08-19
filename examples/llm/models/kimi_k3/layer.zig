const zml = @import("zml");

const attn_res = @import("attn_res.zig");
const kda = @import("kda.zig");
const primitives = @import("primitives.zig");

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
