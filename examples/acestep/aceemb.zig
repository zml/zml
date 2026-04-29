const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const inference = @import("inference.zig");

const hz_type = .f32;

pub const AceEmb_handler = struct {
    model: AceEmb,
    params: AceEmbParams,
    config: AceEmb.Config,
    tokenizer: zml.tokenizer.Tokenizer,
    partial_embed_exe: zml.Exe,
    full_embed_exe: zml.Exe,
    model_buffers: zml.Bufferized(AceEmb),

    pub fn initFromFile(zml_handler: main.Zml_handler, full_seq_len: u32, embed_seq_len: u32) !AceEmb_handler {
        const model_path = "//Users//sboulmier//zml//examples//acestep//models//Qwen3-Embedding-0.6B//model.safetensors";
        const token_path = "//Users//sboulmier//zml//examples//acestep//models//Qwen3-Embedding-0.6B//tokenizer.json";
        const config_path = "//Users//sboulmier//zml//examples//acestep//models//Qwen3-Embedding-0.6B//config.json";

        var registry: zml.safetensors.TensorRegistry = try .fromPath(zml_handler.allocator, zml_handler.io, model_path);
        defer registry.deinit();
        
        //try main.printSafetensors(zml_handler.allocator, zml_handler.io, model_path);

        std.log.info("EMB parse config and safetensors", .{});
        const parsed_config = try parseConfig(zml_handler, config_path);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        std.log.info("EMB parsed", .{});
        
        std.log.info("EMB init model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const model: AceEmb = try .init(zml_handler.allocator, store.view(), config);
        std.log.info("EMB initialized", .{});
        
        const params: AceEmbParams = .{
            .full_embed_tokens = .init(.{ .s = full_seq_len }, .u32),
            .partial_embed_tokens = .init(.{ .s = embed_seq_len }, .u32),
            .shardings = try .init(zml_handler.platform),
        };

        std.log.info("EMB compile models", .{});
        const full_embed_exe = try compileFullModel(zml_handler, model, params);
        std.log.info("EMB compiled full model", .{});
        const partial_embed_exe = try compilePartialModel(zml_handler, model, params);
        std.log.info("EMB compiled partial model", .{});
        const tokenizer = try zml.tokenizer.Tokenizer.fromFile(zml_handler.allocator, zml_handler.io, token_path);
        
        std.log.info("EMB load buffers", .{});
        const model_buffers = try model.load(zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, &store, &params.shardings.all());
        std.log.info("EMB loaded", .{});

        return .{
            .model = model,
            .params = params,
            .config = config,
            .tokenizer = tokenizer,
            .full_embed_exe = full_embed_exe,
            .partial_embed_exe = partial_embed_exe,
            .model_buffers = model_buffers,
        };
    }

    pub fn unloadBuffers(self: *AceEmb_handler) void {
        AceEmb.unloadBuffers(&self.model_buffers);
    }

    pub fn deinit(self: *AceEmb_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.tokenizer.deinit();
        self.full_embed_exe.deinit();
        self.partial_embed_exe.deinit();
    }
};

pub const AceEmbParams = struct {
    full_embed_tokens: zml.Tensor,
    partial_embed_tokens: zml.Tensor,
    shardings: main.Shardings,
};


pub fn parseConfig(zml_handler: main.Zml_handler, path: []const u8) !std.json.Parsed(AceEmb.Config) {
    const parsed_config = blk: {
        const config_json_file = try std.Io.Dir.openFileAbsolute(zml_handler.io, path, .{});
        defer config_json_file.close(zml_handler.io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(zml_handler.io, &config_json_buffer);
        var reader = std.json.Reader.init(zml_handler.allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(AceEmb.Config, zml_handler.allocator, &reader, .{ .ignore_unknown_fields = true, });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}

pub fn compileFullModel(zml_handler: main.Zml_handler, model: AceEmb, parameters: AceEmbParams) !zml.Exe {
    const shardings_arr = parameters.shardings.all();
    const opts: zml.module.CompilationOptions = .{
        .shardings = &shardings_arr,
    };
    return zml_handler.platform.compile(
        zml_handler.allocator,
        zml_handler.io,
        model,
        .forward,
        .{ parameters.full_embed_tokens },
        opts,
    );
}

pub fn compilePartialModel(zml_handler: main.Zml_handler, model: AceEmb, parameters: AceEmbParams) !zml.Exe {
    const opts: zml.module.CompilationOptions = .{
        .shardings = &parameters.shardings.all(),
    };
    return zml_handler.platform.compile(
        zml_handler.allocator,
        zml_handler.io,
        model,
        .embed,
        .{ parameters.partial_embed_tokens },
        opts,
    );
}

pub fn embeddingLengths(zml_handler: main.Zml_handler, audio_metadata: inference.AudioMetadata) !struct { u32, u32 } {
    const token_path = "//Users//sboulmier//zml//examples//acestep//models//Qwen3-Embedding-0.6B//tokenizer.json";
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(zml_handler.allocator, zml_handler.io, token_path);
    defer tokenizer.deinit();
    
    const caption_tok = try inference.tokenizeInputCaption(zml_handler.allocator, tokenizer, audio_metadata);
    const lyrics_tok = try inference.tokenizeInputLyrics(zml_handler.allocator, tokenizer, audio_metadata);
    defer zml_handler.allocator.free(caption_tok);
    defer zml_handler.allocator.free(lyrics_tok);
    
    const l1: u32 = @intCast(caption_tok.len);
    const l2: u32 = @intCast(lyrics_tok.len);
    return .{ l1, l2 };
}


pub const AceEmb = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: u32,
        head_dim: ?u32 = null,
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        rope_theta: f32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        tie_word_embeddings: bool = true,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
        vocab_size: u32,

        pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
            _ = allocator;
            return .{
                .bos_token_id = self.bos_token_id,
                .eos_token_id = self.eos_token_id,
                .head_dim = self.head_dim,
                .hidden_size = self.hidden_size,
                .num_hidden_layers = self.num_hidden_layers,
                .num_attention_heads = self.num_attention_heads,
                .num_key_value_heads = self.num_key_value_heads,
                .rope_theta = self.rope_theta,
                .max_position_embeddings = self.max_position_embeddings,
                .rms_norm_eps = self.rms_norm_eps,
                .tie_word_embeddings = self.tie_word_embeddings,
                .rope_scaling = self.rope_scaling,
                .vocab_size = self.vocab_size,
            };
        }

        pub fn deinit(self: Config, _: std.mem.Allocator) void {
            _ = self;
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !AceEmb {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                    "embed_tokens.weight",
                    .{ .voc, .d },
                    .{ .voc = .replicated, .d = .model },
                )},
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
        };
    }

    pub fn load(
        self: *const AceEmb,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(AceEmb) {
        return zml.io.load(AceEmb, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }

    pub fn deinit(self: *const AceEmb, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceEmb)) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(self: AceEmb, tokens: zml.Tensor) zml.Tensor {
        var output = self.embed_tokens.convert(hz_type).forward(tokens).withPartialTags(.{ .d });
        for (self.layers) |layer| {
            output = layer.forward(output);
        }
        output = self.norm.forward(output);
        return output;
    }
    
    pub fn embed(self: AceEmb, tokens: zml.Tensor) zml.Tensor {
        const output = self.embed_tokens.convert(hz_type).forward(tokens).withPartialTags(.{ .d });
        return output;
    }
};

const TransformerLayer = struct {
    id: u32,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_: u32, store: zml.io.TensorStore.View, config: AceEmb.Config) !TransformerLayer {
        return .{
            .id = id_,
            .input_norm = .init(store.withPrefix("input_layernorm"), config),
            .att_layer = try .init(store.withPrefix("self_attn"), config),
            .post_att_norm = .init(store.withPrefix("post_attention_layernorm"), config),
            .mlp_layer = try .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_norm);
        AttLayer.unloadBuffers(&self.att_layer);
        RmsNorm.unloadBuffers(&self.post_att_norm);
        MlpLayer.unloadBuffers(&self.mlp_layer);
    }

    pub fn forward(self: TransformerLayer, x0: zml.Tensor) zml.Tensor {
        std.log.debug("FWD : embedding transformer layer {d}", .{self.id});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_norm.forward(x0_replicated);
        const delta0 = self.att_layer.forward(x0_normalized);

        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized)
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return x2.reuseBuffer(x0);
    }
};

const AttLayer = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: AceEmb.Config) !AttLayer {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config),
            .k_norm = .init(store.withPrefix("k_norm"), config),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    /// Full-sequence self-attention without KV cache.
    pub fn forward(self: AttLayer, x: zml.Tensor) zml.Tensor {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        const pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{ .s });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        q = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, q.dtype(), null);
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true }).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        
        const delta = self.o_proj.convert(hz_type).forward(attn_output);
        
        return delta.rename(.{ .d_out = .d }).withPartitioning(.{ .d = .replicated });
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        std.log.debug("FWD : embedding mlp", .{});
        const up_projection = input.dot(self.up_proj.convert(hz_type), .d);
        const gate_projection = input.dot(self.gate_proj.convert(hz_type), .d);
        const activation = gate_projection.silu().mul(up_projection);
        const output = activation.dot(self.down_proj.convert(hz_type), .d_out);
        return output;
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: AceEmb.Config) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{ .d_out }, null),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        return normalized.mul(self.weights.convert(hz_type).withTags(.{ .d }).broad(input.shape()));
    }
};
