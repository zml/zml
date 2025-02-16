const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log;
const Tensor = zml.Tensor;

pub const ModernBertOptions = struct {
    num_attention_heads: i64,
    tie_word_embeddings: bool = false,
};

pub const ModernBertEmbeddings = struct {
    tok_embeddings: zml.nn.TokenEmbedding,
    norm: zml.nn.LayerNorm,

    pub fn forward(self: ModernBertEmbeddings, input_ids: Tensor) Tensor {
        // Perform tok_embeddings
        const hidden_states = zml.call(self.tok_embeddings, .forward, .{input_ids});

        // Perform norm
        return zml.call(self.norm, .forward, .{hidden_states});
    }
};

/// Switch out the old MLP layers for GeGLU layers, improving on the original BERT’s GeLU activation function.
///
/// The GeGLU activation function is a combination of the Gated Linear Unit (GLU) and the Gaussian Error Linear Unit (GeLU).
///
/// see: https://paperswithcode.com/method/geglu
pub const ModernBertMLP = struct {
    Wi: zml.nn.Linear,
    Wo: zml.nn.Linear,

    pub fn forward(self: ModernBertMLP, hidden_states: Tensor) Tensor {
        // Perform Wi
        const wi_output: Tensor = zml.call(self.Wi, .forward, .{hidden_states});

        // Split into input and gate tensors along the last dimension
        const input, const gate = wi_output.chunkExact(-1, 2);

        // Apply activation
        const activated_input = input.gelu().mul(gate);

        // Perform Wo
        return zml.call(self.Wo, .forward, .{activated_input});
    }
};

/// Performs multi-headed self attention on a batch of unpadded sequences.
///
/// If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
/// If Flash Attention 2 is not installed, the implementation will use SDPA,
pub const ModernBertAttention = struct {
    Wqkv: zml.nn.Linear,
    Wo: zml.nn.Linear,
    is_global_attention: bool = false,
    num_heads: i64 = undefined,

    /// sdpa_attention_forward
    pub fn forward(
        self: ModernBertAttention,
        hidden_states: Tensor,
        attention_mask: Tensor,
        sliding_window_mask: Tensor,
    ) Tensor {
        const batch_size = hidden_states.shape().dim(0);
        const seq_length = hidden_states.shape().dim(1);
        const hidden_size = hidden_states.shape().dim(2);
        const num_heads = self.num_heads;
        const head_dim = @divExact(hidden_size, num_heads);

        // Project to query, key, value - { batch_size, seq_len, 3 * num_heads * head_dim }
        var qkv: Tensor = zml.call(self.Wqkv, .forward, .{hidden_states});

        // Reshape to { batch_size, seq_len, 3, num_heads, head_dim }
        qkv = qkv.reshape(.{ batch_size, seq_length, 3, num_heads, head_dim }).withTags(.{ .b, .s, .chunk, .h, .hd });

        // Split into query, key, value tensors - each { batch_size, seq_length, num_heads, head_dim }
        var q, var k, var v = qkv.chunkExact(.chunk, 3);
        q = q.squeeze(.chunk);
        k = k.squeeze(.chunk);
        v = v.squeeze(.chunk);

        // Apply rotary position embeddings (RoPE)
        // Layer 0, 3, 6, 9, 12 ... use global RoPE
        // Layer 1, 2, 4, 5, 7, 8, 10, 11 ... use local RoPE
        const rope_opts = zml.nn.RopeOpts{
            .impl = .sequential,
            .freq_base = if (self.is_global_attention) 160_000 else 10_000,
        };

        q = zml.nn.rope(q, null, rope_opts);
        k = zml.nn.rope(k, null, rope_opts);

        // rename dimensions for sdpa
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        var mask = attention_mask;
        if (!self.is_global_attention) mask = sliding_window_mask;

        const sdqa_opts = zml.nn.SdpaOpts{
            .allow_cudnn = false,
            .attn_mask = mask,
        };

        // Scaled dot product attention
        const attn_output = zml.nn.sdpa(q, k, v, sdqa_opts);
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        // Final projection
        return zml.call(self.Wo, .forward, .{attn});
    }
};

pub const ModernBertEncoderLayer = struct {
    attn_norm: ?zml.nn.LayerNorm = null,
    attn: ModernBertAttention,
    mlp_norm: zml.nn.LayerNorm,
    mlp: ModernBertMLP,

    pub fn forward(
        self: ModernBertEncoderLayer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        sliding_window_mask: Tensor,
    ) Tensor {
        const attn_norm_output = if (self.attn_norm) |attn_norm|
            zml.call(attn_norm, .forward, .{hidden_states})
        else
            hidden_states;

        const attn_output: Tensor = zml.call(self.attn, .forward, .{
            attn_norm_output,
            attention_mask,
            sliding_window_mask,
        });

        var output = hidden_states.add(attn_output);

        const mlp_norm_output: Tensor = zml.call(self.mlp_norm, .forward, .{output});
        const mlp_output = zml.call(self.mlp, .forward, .{mlp_norm_output});
        output = output.add(mlp_output);

        return output;
    }
};

pub fn generateSlidingWindowMask(global_attention_mask: Tensor) Tensor {
    const tgt_seq_len = global_attention_mask.dim(.tgt);
    const src_seq_len = global_attention_mask.dim(.src);
    const mask_shape = zml.Shape.init(.{ .tgt = tgt_seq_len, .src = src_seq_len }, global_attention_mask.dtype());

    // Create position indices (for rows and cols)
    const rows = Tensor.iota(mask_shape, .tgt);
    const cols = Tensor.iota(mask_shape, .src);

    // Calculate distance between positions
    const distance = rows.sub(cols).abs();

    // Create sliding window mask (1 for positions within window, 0 outside)
    const local_attention = 128; // TODO: config.json: local_attention
    var window_mask = distance.cmp(.LE, Tensor.scalar(@divExact(local_attention, 2), distance.dtype()))
        .unsqueeze(0)
        .unsqueeze(0);

    // match global_attention_mask shape
    window_mask = window_mask.broadcastLeft(global_attention_mask.shape());

    // Combine with existing mask
    if (global_attention_mask.dtype().isFloat()) {
        const minus_inf = Tensor.constant(global_attention_mask.shape(), global_attention_mask.dtype().minValue());
        return Tensor.select(window_mask, global_attention_mask, minus_inf);
    } else {
        return window_mask.convert(global_attention_mask.dtype());
    }
}

pub const ModernBertModel = struct {
    embeddings: ModernBertEmbeddings,
    layers: []ModernBertEncoderLayer,
    final_norm: zml.nn.LayerNorm,
    dtype: zml.DataType = .f32, // config.json: torch_dtype

    pub fn init(self: *ModernBertModel, options: ModernBertOptions) void {
        self.final_norm.eps = 1e-5;
        for (self.layers, 0..) |*encoder_layer, layer_idx| {
            if (encoder_layer.attn_norm) |*norm| norm.eps = 1e-5;
            encoder_layer.mlp_norm.eps = 1e-5;
            encoder_layer.attn.is_global_attention = (layer_idx % 3 == 0);
            encoder_layer.attn.num_heads = options.num_attention_heads;
        }
    }

    pub fn forward(self: ModernBertModel, input_ids: Tensor, attention_mask: Tensor) Tensor {
        var hidden_states: Tensor = zml.call(self.embeddings, .forward, .{input_ids});

        // global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        const global_attention_mask = zml.nn.expandMask(attention_mask, self.dtype, null);

        const sliding_window_mask = generateSlidingWindowMask(global_attention_mask);

        // Process through all encoder layers
        for (self.layers) |encoder_layer| {
            hidden_states = zml.call(encoder_layer, .forward, .{
                hidden_states,
                global_attention_mask,
                sliding_window_mask,
            });
        }

        // Final layer normalization
        hidden_states = zml.call(self.final_norm, .forward, .{hidden_states});

        return hidden_states;
    }
};

pub const ModernBertPredictionHead = struct {
    dense: zml.nn.Linear,
    norm: zml.nn.LayerNorm,

    pub fn forward(self: ModernBertPredictionHead, hidden_states: Tensor) Tensor {
        const dense_output: Tensor = zml.call(self.dense, .forward, .{hidden_states});

        const activated_output = dense_output.gelu();

        return zml.call(self.norm, .forward, .{activated_output});
    }
};

pub const ModernBertForMaskedLM = struct {
    model: ModernBertModel,
    head: ModernBertPredictionHead,
    decoder: struct { weight: ?zml.Tensor, bias: zml.Tensor },

    pub fn init(self: *ModernBertForMaskedLM, options: ModernBertOptions) void {
        self.model.init(options);
        self.head.norm.eps = 1e-5;
        if (options.tie_word_embeddings == true) {
            self.decoder.weight = null;
        }
    }

    pub fn forward(self: ModernBertForMaskedLM, input_ids: Tensor, attention_mask: Tensor) struct { Tensor, Tensor } {
        const outputs: Tensor = zml.call(self.model, .forward, .{ input_ids, attention_mask });
        const head_outputs: Tensor = zml.call(self.head, .forward, .{outputs});

        // either use decoder or tied weights
        const decoder_weights = self.decoder.weight orelse self.model.embeddings.tok_embeddings.weight;

        const logits = head_outputs.withTags(.{ .b, .s, .d }).dot(decoder_weights.withTags(.{ .voc, .d }), .{.d});
        const biased_logits = logits.add(self.decoder.bias.withTags(.{.voc}).broad(logits.shape()));

        const probabilities = biased_logits.softmax(.voc);
        const top_k = probabilities.topK(5, .voc, .{ .descending = true });
        return .{ top_k.indices, top_k.values };
    }
};
