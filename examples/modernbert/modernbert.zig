const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log;
const Tensor = zml.Tensor;

// TODO: I will variablize this later (ModernBertOptions)
pub const hidden_size = 768;
pub const intermediate_size = 1152;
pub const num_attention_heads = 12;
pub const layer_norm_eps = 1e-05;
pub const norm_eps = 1e-05;
pub const local_attention = 128;

pub const ModernBertEmbeddings = struct {
    tok_embeddings: zml.nn.TokenEmbedding,
    norm: zml.nn.LayerNorm,
    drop: void,

    pub fn init(self: *ModernBertEmbeddings) void {
        self.norm.eps = layer_norm_eps;
    }

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
        // input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        const input = wi_output.slice1d(-1, .{ .end = intermediate_size });
        const gate = wi_output.slice1d(-1, .{ .start = intermediate_size });

        // Apply activation function to input and multiply by gate :
        // self.Wo(self.drop(self.act(input) * gate))
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

    /// sdpa_attention_forward
    pub fn forward(
        self: ModernBertAttention,
        hidden_states: Tensor, // { batch_size, seq_length, hidden_size } {1,9,768}
        attention_mask: Tensor, // {1,1,9,9}
        sliding_window_mask: Tensor, // {1,1,9,9}
    ) Tensor {
        const batch_size = hidden_states.shape().dim(0);
        const seq_length = hidden_states.shape().dim(1);
        const num_heads = 12; // config.json: num_attention_heads
        const head_dim = 64; // config.json hidden_size / num_attention_heads = 768 / 12

        if (self.is_global_attention) {
            log.info("Global attention", .{});
        } else {
            log.info("Local attention", .{});
        }

        // Project to query, key, value - { batch_size, seq_len, 3 * num_heads * head_dim }
        var qkv: Tensor = zml.call(self.Wqkv, .forward, .{hidden_states}); // Wqkv.out.0

        // Reshape to { batch_size, seq_len, 3, num_heads, head_dim }
        qkv = qkv.reshape(.{ batch_size, seq_length, 3, num_heads, head_dim }).withTags(.{ .b, .s, .fixed, .h, .hd });

        // Split into query, key, value tensors - each { batch_size, seq_length, num_heads, head_dim }
        // TODO: replace with: var q, var k, var v = qkv.chunkExact(.fixed, 2);
        var q = qkv.slice1d(.fixed, .{ .start = 0, .end = 1 }).squeeze(.fixed);
        var k = qkv.slice1d(.fixed, .{ .start = 1, .end = 2 }).squeeze(.fixed);
        var v = qkv.slice1d(.fixed, .{ .start = 2, .end = 3 }).squeeze(.fixed);

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

    pub fn init(self: *ModernBertEncoderLayer) void {
        if (self.attn_norm) |attn_norm| {
            attn_norm.eps = norm_eps;
        }
        self.mlp_norm.eps = norm_eps;
    }

    pub fn forward(
        self: ModernBertEncoderLayer,
        hidden_states: Tensor, // { batch_size, seq_length, hidden_size } {1,9,768}
        attention_mask: Tensor, // {1,1,9,9}
        sliding_window_mask: Tensor, // {1,1,9,9}
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
        log.info("attn_output : {}", .{attn_output});
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
    const target_shape = zml.Shape.init(.{ .tgt = tgt_seq_len, .src = src_seq_len }, global_attention_mask.dtype());

    // TODO: Confirm this is the correct way to do this : reshape (or insertAxes, appendAxes) instead of unsqueeze ?

    // Create position indices (for rows and cols)
    // Original py code : rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
    var rows = Tensor.arange(.{ .end = tgt_seq_len }, global_attention_mask.dtype())
        .unsqueeze(0);
    // var rows = Tensor.arange(.{ .end = tgt_seq_len }, global_attention_mask.dtype())
    //     .reshape(.{ 1, tgt_seq_len });
    log.info("rows : {}", .{rows});
    var cols = rows.transpose(.{});
    log.info("cols : {}", .{cols});

    // Calculate distance between positions
    // Original py code : distance = torch.abs(rows - rows.T)

    // broadcast both tensors to a common shape
    rows = rows.broad(target_shape);
    cols = cols.broad(target_shape);

    // now both rows and cols have shape
    const distance = rows.sub(cols).abs();
    log.info("distance : {}", .{distance});

    // Create sliding window mask (1 for positions within window, 0 outside)
    // Original py code : window_mask = (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0)
    // config.json : local_attention
    var window_mask = distance.cmp(.LE, Tensor.scalar(@divExact(local_attention, 2), distance.dtype()))
        .unsqueeze(0)
        .unsqueeze(0);

    // match global_attention_mask shape
    window_mask = window_mask.broadcastLeft(global_attention_mask.shape());

    // Combine with existing mask
    // Original py code : sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)
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

    pub fn init(self: *ModernBertModel) void {
        for (self.layers, 0..) |*encoder_layer, layer_idx| {
            encoder_layer.attn.is_global_attention = (layer_idx % 3 == 0);
        }
    }

    pub fn forward(self: ModernBertModel, input_ids: Tensor, attention_mask: Tensor) Tensor {
        var hidden_states: Tensor = zml.call(self.embeddings, .forward, .{input_ids});

        // global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        const global_attention_mask = zml.nn.expandMask(attention_mask, self.dtype, null);
        log.info("global_attention_mask : {}", .{global_attention_mask});

        const sliding_window_mask = generateSlidingWindowMask(global_attention_mask);
        log.info("sliding_window_mask : {}", .{sliding_window_mask});

        // Process through all encoder layers
        for (self.layers) |encoder_layer| {
            const layer_outputs: Tensor = zml.call(encoder_layer, .forward, .{
                hidden_states,
                global_attention_mask,
                sliding_window_mask,
            });
            hidden_states = layer_outputs;
        }

        // Final layer normalization
        hidden_states = zml.call(self.final_norm, .forward, .{hidden_states});

        return hidden_states;
    }
};
