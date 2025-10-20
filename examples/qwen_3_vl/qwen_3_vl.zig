const std = @import("std");
const testing = std.testing;

const async = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const Linear = zml.nn.Linear;
const Shape = zml.Shape;

const log = std.log.scoped(.qwen3_vl);

/// Qwen2.5-VL architecture, using huggingface transformers naming.
/// Vision-Language model with vision transformer and text model.
pub const Qwen3VL = struct {
    pub const VisionConfig = struct {
        depth: u32 = 32,
        hidden_size: u32 = 1280,
        hidden_act: []const u8 = "silu",
        intermediate_size: u32 = 3420,
        num_heads: u32 = 16,
        in_channels: u32 = 3,
        patch_size: u32 = 14,
        spatial_merge_size: u32 = 2,
        temporal_patch_size: u32 = 2,
        //tokens_per_second: u32 = 2,
        //window_size: u32 = 112,
        out_hidden_size: u32 = 2048,
        //fullatt_block_indexes: []const u32 = &[_]u32{ 7, 15, 23, 31 },
        initializer_range: f32 = 0.02,
        deepstack_visual_indexes: []const u32 = &[_]u32{ 5, 11, 17 },
    };

    pub const TextConfig = struct {
        bos_token_id: u32,
        eos_token_id: u32,
        head_dim: ?u32 = null,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        tie_word_embeddings: bool = true,
    };

    pub const Config = struct {
        //Vision config (depuis vision_config)
        vision_config: VisionConfig,
        text_config: TextConfig,
        // rope_theta: f32,
        // hf_rope_impl: bool = true,
        tie_word_embeddings: bool = true,
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: u32,
    };
    vision_transformer: VisionTransformer,
    text_model: TextModel,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config, options: Options, store: zml.aio.BufferStore) !Qwen3VL {
        return .{
            .config = config,
            .gen_opts = options.sampling_strategy orelse .{},
            .vision_transformer = try VisionTransformer.init(allocator, config, store),
            .text_model = try TextModel.init(allocator, config, store),
        };
    }
};

//========================Vision model========================

pub const VisionTransformer = struct {
    vision_patch_embed: VisionPatchEmbed,
    pos_embed: zml.nn.TokenEmbedding,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: []VisionBlock,
    patch_merger: PatchMerger,
    deepstack_patch_mergers: []PatchMerger,
    // Config values pushed down
    num_heads: u32,
    //window_size: u32,
    //fullatt_block_indexes: []u32,
    //liste de vision patch merger

    pub fn init(allocator: std.mem.Allocator, config: Qwen3VL.Config, store: zml.aio.BufferStore) !VisionTransformer {
        const blocks = try allocator.alloc(VisionBlock, config.vision_config.depth);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.visual.blocks");
        for (0.., blocks) |i, *block| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var vision_attn = try zml.aio.populateModelWithPrefix(VisionAttention, allocator, store, prefix.concat("attn"));
            vision_attn.num_heads = config.vision_config.num_heads;
            //vision_attn.window_size = config.window_size;

            const mlp = try zml.aio.populateModelWithPrefix(VisionMlp, allocator, store, prefix.concat("mlp"));
            //mlp.hidden_act = config.hidden_act;

            var norm1 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm1"));
            norm1.eps = config.text_config.rms_norm_eps;

            var norm2 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm2"));
            norm2.eps = config.text_config.rms_norm_eps;

            block.* = .{
                .attn = vision_attn,
                .mlp = mlp,
                .norm1 = norm1,
                .norm2 = norm2,
                .num_heads = config.vision_config.num_heads,
                //.window_size = config.window_size,
                //.is_full_attention = std.mem.indexOfScalar(u32, config.fullatt_block_indexes, @intCast(i)) != null, // indexof voir dans zig std
            };
        }

        prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.visual.deepstack_merger_list");
        const deepstack_patch_mergers = try allocator.alloc(PatchMerger, config.vision_config.deepstack_visual_indexes.len);
        for (0.., deepstack_patch_mergers) |i, *deepstack_patch_merger| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            const norm = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm"));
            const linear_fc1 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("linear_fc1"));
            const linear_fc2 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("linear_fc2"));
            deepstack_patch_merger.* = .{
                .norm = norm,
                .linear_fc1 = linear_fc1,
                .linear_fc2 = linear_fc2,
                .out_hidden_size = config.vision_config.out_hidden_size,
            };
        }
        log.info("patch_size: {d}", .{config.vision_config.patch_size});
        log.info("temporal_patch_size: {d}", .{config.vision_config.temporal_patch_size});
        log.info("in_channels: {d}", .{config.vision_config.in_channels});
        log.info("out_hidden_size: {d}", .{config.vision_config.out_hidden_size});

        return .{
            .pos_embed = try zml.aio.populateModelWithPrefix(zml.nn.TokenEmbedding, allocator, store, "model.visual.pos_embed"),
            .blocks = blocks,
            .patch_merger = .{
                .norm = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, "model.visual.merger.norm"),
                .linear_fc1 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, "model.visual.merger.linear_fc1"),
                .linear_fc2 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, "model.visual.merger.linear_fc2"),
                .out_hidden_size = config.vision_config.out_hidden_size,
            },
            .num_heads = config.vision_config.num_heads,
            .deepstack_patch_mergers = deepstack_patch_mergers,
            //.window_size = config.window_size,
            //.fullatt_block_indexes = config.fullatt_block_indexes,
            .vision_patch_embed = try VisionPatchEmbed.init(allocator, config.vision_config.patch_size, config.vision_config.temporal_patch_size, config.vision_config.in_channels, config.vision_config.out_hidden_size, store),
            .rotary_pos_emb = try VisionRotaryEmbedding.init(allocator, config.vision_config.out_hidden_size, 10000.0),
        };
    }
};

// input : kernel spatials dims : [time, height, width]
// output : padding : [front, back, top, bottom, left, right]
// fn getPaddingForSame(kernel_dims: [3]i64) [6]i64 {
//     for (0.., kernel_dims) |i, k| {
//         log.info("kernel_dims: {d} at index {d}", .{ k, i });
//     }
//     var res: [6]i64 = undefined;
//     for (kernel_dims, 0..) |k, out| {
//         log.info("kernel_dims: {d}", .{k});
//         const pad = @divFloor(k, 2);
//         res[2 * out] = pad;
//         res[2 * out + 1] = pad;
//     }
//     return res;
// }

fn getPaddingForSame(input_dims: [3]i64, kernel_dims: [3]i64, strides: [3]i64) [6]i64 {
    var res: [6]i64 = undefined;
    for (0..3) |i| {
        const output_size = @divFloor(input_dims[i], strides[i]);
        const total_padding = (output_size - 1) * strides[i] + kernel_dims[i] - input_dims[i];
        const pad_start = @divFloor(total_padding, 2);
        const pad_end = total_padding - pad_start;
        res[2 * i] = pad_start;
        res[2 * i + 1] = pad_end;
    }
    return res;
}

pub const Conv3d = struct {
    weight: Tensor,
    bias: ?Tensor = null,

    pub fn forward(self: Conv3d, input: Tensor) Tensor {
        //const x = if (input.rank() == 4) zml.torch.unsqueeze(input, 0) else input;
        const x = input;
        for (0.., x.dims()) |i, d| {
            log.info("x_dims: {d} at index {d}", .{ d, i });
        }
        for (0.., self.weight.dims()) |i, d| {
            log.info("hernel_dims: {d} at index {d}", .{ d, i });
        }
        var strides: [3]i64 = undefined;
        for (self.weight.dims()[2..5], 0..) |k, out| strides[out] = k;
        for (0.., strides) |i, d| {
            log.info("stride_dims: {d} at index {d}", .{ d, i });
        }
        const padding = getPaddingForSame(x.dims()[2..5].*, self.weight.dims()[2..5].*, strides);
        for (0.., padding) |i, d| {
            log.info("padding_dims: {d} at index {d}", .{ d, i });
        }
        const weight = self.weight.convert(x.dtype()); // Convertir le poids au même type que l'input
        var y = x.conv3d(weight, .{ .padding = &padding, .window_strides = &strides });
        //, .window_strides = &strides
        if (self.bias) |b| y = y.add(b.convert(y.dtype()).broadcast(y._shape, &.{1}));
        //return if (input.rank() == 3) y.squeeze(0) else y;
        return y;
    }
};

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    norm2: zml.nn.LayerNorm,
    attn: VisionAttention, // Ici window attention
    mlp: VisionMlp,

    // Config values pushed down
    num_heads: u32,
    //window_size: u32,
    //is_full_attention: bool, // based on fullatt_block_indexes

};

// Necessite une couche convolutiom=nnelle 3D

pub const VisionPatchEmbed = struct {
    // Linear layer for patch embedding (fallback temporaire)
    proj: Conv3d,

    // Config values
    patch_size: u32 = 14,
    temporal_patch_size: u32 = 2,
    in_channels: u32 = 3,
    embed_dim: u32 = 1152, // correspond à hidden_size dans la config

    pub fn init(
        allocator: std.mem.Allocator,
        patch_size: u32,
        temporal_patch_size: u32,
        in_channels: u32,
        embed_dim: u32,
        store: zml.aio.BufferStore,
    ) !VisionPatchEmbed {
        const conv3d = try zml.aio.populateModelWithPrefix(Conv3d, allocator, store, "model.visual.patch_embed.proj");

        return .{
            .proj = conv3d,
            .patch_size = patch_size,
            .temporal_patch_size = temporal_patch_size,
            .in_channels = in_channels,
            .embed_dim = embed_dim,
        };
    }

    pub fn forward(self: VisionPatchEmbed, hidden_states: Tensor) Tensor {
        // 1. Reshape pour la convolution 3D
        // hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        const reshaped = hidden_states.reshape(.{ -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size });

        log.info("resahped dim", .{});
        for (0.., reshaped.dims()) |i, d| {
            log.info("reshaped_dims: {d} at index {d}", .{ d, i });
        }
        // 2. Appliquer la convolution 3D
        const conv_output = self.proj.forward(reshaped);

        // 3. Reshape final pour avoir [batch_size, embed_dim]
        // .view(-1, self.embed_dim)

        const reshaped_output = conv_output.reshape(.{ conv_output.dim(0), conv_output.dim(1) });
        for (0.., reshaped_output.dims()) |i, d| {
            log.info("reshaped_output_dims: {d} at index {d}", .{ d, i });
        }
        return reshaped_output;
    }
};

pub const VisionRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    dim: u32,

    pub fn init(allocator: std.mem.Allocator, dim: u32, theta: f32) !VisionRotaryEmbedding {
        _ = allocator;
        return .{
            .rope_opts = zml.nn.RopeOpts{
                .layout = .sequential,
                .freq_base = theta,
                .scaling = .{ .default = {} },
            },
            .dim = dim,
        };
    }

    pub fn forward(self: VisionRotaryEmbedding) Tensor {
        const seqlen = 128;
        // Utiliser directement invFreq de ZML
        const inv_freq = zml.nn.invFreq(@intCast(32), self.rope_opts).withTags(.{.s});
        //inv_freq par rapport a une dim;
        log.info("inv_freq : {f}", .{inv_freq.shape()});
        // Créer la séquence de positions
        const seq = zml.Tensor.arange(.{ .end = seqlen }, .f32).withTags(.{.d});

        // Produit tensoriel
        return zml.Tensor.outer(seq, inv_freq);
    }
};

// Vision-specific components
pub const VisionAttention = struct {
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,

    num_heads: u32,
    //window_size: u32,
    //is_full_attention: bool,
};

pub const PatchMerger = struct {
    norm: zml.nn.LayerNorm,
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,

    // Config values
    out_hidden_size: u32,
};

pub const MlpMerger = struct { // MLP classique
    @"0": zml.nn.Linear,
    @"2": zml.nn.Linear,

    //hidden_act: []const u8,
};

pub const VisionMlp = struct { // MLP classique
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,
    //hidden_act: []const u8,
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    // Config values pushed down
    max_seq_len: u32 = 1000, // a definir en option
    num_heads: u32,
    num_kv_heads: u32,
    // rope_opts: zml.nn.RopeOpts = .{
    //     .layout = .interleaved,
    //     .freq_base = 10_000,
    // },

    pub fn init(allocator: std.mem.Allocator, config: Qwen3VL.Config, store: zml.aio.BufferStore) !TextModel {
        // const rope_opts: zml.nn.RopeOpts = .{
        //     .layout = if (config.hf_rope_impl) .sequential else .interleaved,
        //     .freq_base = config.rope_theta,
        //     //.scaling = config.rope_scaling,
        // };
        const layers = try allocator.alloc(TransformerLayer, config.text_config.num_hidden_layers);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.language_model.layers");
        for (0.., layers) |i, *layer| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var self_attn = try zml.aio.populateModelWithPrefix(SelfAttn, allocator, store, prefix.concat("self_attn"));
            self_attn.num_heads = config.text_config.num_attention_heads;
            self_attn.num_kv_heads = config.text_config.num_key_value_heads;
            // self_attn.rope_opts = rope_opts;

            const mlp = try zml.aio.populateModelWithPrefix(Mlp, allocator, store, prefix.concat("mlp"));
            //mlp.hidden_act = config.hidden_act;

            var input_layernorm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, prefix.concat("input_layernorm"));
            input_layernorm.eps = config.text_config.rms_norm_eps;

            var post_attention_layernorm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, prefix.concat("post_attention_layernorm"));
            post_attention_layernorm.eps = config.text_config.rms_norm_eps;

            layer.* = .{
                .self_attn = self_attn,
                .mlp = mlp,
                .input_layernorm = input_layernorm,
                .post_attention_layernorm = post_attention_layernorm,
                .num_heads = config.text_config.num_attention_heads,
            };
        }
        return .{
            .embed_tokens = try zml.aio.populateModelWithPrefix(zml.nn.TokenEmbedding, allocator, store, "model.language_model.embed_tokens"),
            .layers = layers,
            .norm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, "model.language_model.norm"),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            // .rope_opts = .{
            //     .layout = if (config.hf_rope_impl) .sequential else .interleaved,
            //     .freq_base = config.rope_theta,
            //     //.scaling = config.rope_scaling,
            // },
        };
    }

    pub fn forward(self: TextModel, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache, zml.Tensor.Rng } {
        const embeds = zml.call(self.embed_tokens, .forward, .{tokens});
        var hidden = embeds;
        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }
        const output = zml.call(self.norm, .forward, .{hidden});
        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,
    num_heads: u32,

    pub fn forward(self: TransformerLayer, x0: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

// Reuse Llama components
pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    // rope_opts: zml.nn.RopeOpts = undefined,

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});
        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };
        _ = pos_index; // autofix

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        // q = zml.nn.rope(q, pos_index, self.rope_opts);
        // k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ zml.call(self.o_proj, .forward, .{attn}), new_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    //hidden_act: []const u8,

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = zml.call(self.up_proj, .forward, .{x});
        var output = zml.call(self.gate_proj, .forward, .{x});
        output = output.silu().mul(proj);
        return zml.call(self.down_proj, .forward, .{output});
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

// KV Cache from Llama
pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,
};
