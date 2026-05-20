const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const inference = @import("inference.zig");


pub const AceEmb_handler = struct {
    model: AceEmb,
    params: Params,
    config: Config,
    options: Options,
    exes: Exes,
    model_buffers: zml.Bufferized(AceEmb),

    pub fn init(zml_handler: *main.Zml_handler) !AceEmb_handler {
        zml_handler.tic(&zml_handler.timers.emb.init);
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.aceemb);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();
        
        std.log.info("EMB parse config and safetensors", .{});
        const parsed_config = try main.parseConfig(Config, zml_handler.allocator, zml_handler.io, repo);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        std.log.info("EMB parsed", .{});

        const options: Options = .{};
        
        std.log.info("EMB init model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const model: AceEmb = try .init(zml_handler.allocator, store.view(), config);
        std.log.info("EMB initialized", .{});
        
        const params: Params = .{
            .lyric_tokens = .init(.{ .s = options.seq_len }, .u32),
            .text_tokens = .init(.{ .s = options.seq_len }, .u32),
            .text_embeds = .init(.{ .s = options.seq_len, .d = config.hidden_size }, .bf16),
            .shardings = try .init(zml_handler.platform),
        };

        zml_handler.toc(&zml_handler.timers.emb.init);
        zml_handler.tic(&zml_handler.timers.emb.compile);
        
        const exes = try compileModel(zml_handler, model, params);

        zml_handler.toc(&zml_handler.timers.emb.compile);
        zml_handler.tic(&zml_handler.timers.emb.load);
        
        std.log.info("EMB load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &params.shardings.all());
        std.log.info("EMB loaded", .{});

        zml_handler.toc(&zml_handler.timers.emb.load);
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .options = options,
            .exes = exes,
            .model_buffers = model_buffers,
        };
    }
    
    pub fn loadTokenizer(zml_handler: *main.Zml_handler, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
        const tokenizer_json_file = try dir.openFile(zml_handler.io, "tokenizer.json", .{});
        defer tokenizer_json_file.close(zml_handler.io);
        
        var reader = tokenizer_json_file.reader(zml_handler.io, &.{});
        const bytes = try reader.interface.readAlloc(zml_handler.allocator, try tokenizer_json_file.length(zml_handler.io));

        defer zml_handler.allocator.free(bytes);
        return try .fromBytes(zml_handler.allocator, zml_handler.io, bytes);
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceEmb, params: Params) !Exes {
        const shardings_arr = params.shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
        std.log.info("EMB compile models", .{});

        var lyric_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceEmb, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{
                        params_.lyric_tokens }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var lyric_embed_future_awaited = false;
        errdefer if (!lyric_embed_future_awaited) if (lyric_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var text_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceEmb, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{
                        params_.text_tokens }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var text_embed_future_awaited = false;
        errdefer if (!text_embed_future_awaited) if (text_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward,
                    .{ params_.text_embeds }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], params, opts });
        var layer_future_awaited = false;
        errdefer if (!layer_future_awaited) if (layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var norm_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceEmb, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .normEmbed,
                    .{ params_.text_embeds }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var norm_future_awaited = false;
        errdefer if (!norm_future_awaited) if (norm_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        const lyric_embed_exe = try lyric_embed_future.await(zml_handler.io);
        lyric_embed_future_awaited = true;

        const text_embed_exe = try text_embed_future.await(zml_handler.io);
        text_embed_future_awaited = true;

        const layer_exe = try layer_future.await(zml_handler.io);
        layer_future_awaited = true;

        const norm_exe = try norm_future.await(zml_handler.io);
        norm_future_awaited = true;

        return .{
            .lyric_embed_exe = lyric_embed_exe,
            .lyric_embed_args = try lyric_embed_exe.args(zml_handler.allocator),
            .lyric_embed_results = try lyric_embed_exe.results(zml_handler.allocator),
            .text_embed_exe = text_embed_exe,
            .text_embed_args = try text_embed_exe.args(zml_handler.allocator),
            .text_embed_results = try text_embed_exe.results(zml_handler.allocator),
            .layer_exe = layer_exe,
            .layer_args = try layer_exe.args(zml_handler.allocator),
            .layer_results = try layer_exe.results(zml_handler.allocator),
            .norm_exe = norm_exe,
            .norm_args = try norm_exe.args(zml_handler.allocator),
            .norm_results = try norm_exe.results(zml_handler.allocator),
        };
    }
    
    pub fn unloadBuffers(self: *AceEmb_handler, allocator: std.mem.Allocator) void {
        AceEmb.unloadBuffers(&self.model_buffers, allocator);
    }

    pub fn deinit(self: *AceEmb_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.exes.deinit(allocator);
    }
};


pub const Params = struct {
    lyric_tokens: zml.Tensor,
    text_tokens: zml.Tensor,
    text_embeds: zml.Tensor,
    shardings: main.Shardings,
};

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

pub const Options = struct {
    seq_len: u32 = 2048,
};

pub const Exes = struct {
    lyric_embed_exe: zml.Exe,
    lyric_embed_args: zml.Exe.Arguments,
    lyric_embed_results: zml.Exe.Results,

    text_embed_exe: zml.Exe,
    text_embed_args: zml.Exe.Arguments,
    text_embed_results: zml.Exe.Results,

    layer_exe: zml.Exe,
    layer_args: zml.Exe.Arguments,
    layer_results: zml.Exe.Results,
    
    norm_exe: zml.Exe,
    norm_args: zml.Exe.Arguments,
    norm_results: zml.Exe.Results,

    pub fn deinit(self: Exes, allocator: std.mem.Allocator) void {
        self.lyric_embed_exe.deinit();
        self.lyric_embed_args.deinit(allocator);
        self.lyric_embed_results.deinit(allocator);
        self.text_embed_exe.deinit();
        self.text_embed_args.deinit(allocator);
        self.text_embed_results.deinit(allocator);
        self.layer_exe.deinit();
        self.layer_args.deinit(allocator);
        self.layer_results.deinit(allocator);
        self.norm_exe.deinit();
        self.norm_args.deinit(allocator);
        self.norm_results.deinit(allocator);
    }
};


pub const AceEmb = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !AceEmb {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, null) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
        };
    }

    pub fn load(self: *const AceEmb, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceEmb) {
        var progress = zml_handler.progress.start("Load Emb weights", store.registry.tensors.count());
        defer progress.end();
        return zml.io.load(AceEmb, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = &progress,
        });
    }

    pub fn deinit(self: *const AceEmb, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceEmb), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(self: AceEmb, tokens: zml.Tensor) zml.Tensor {
        var output = self.embed_tokens.forward(tokens).withPartialTags(.{ .d });
        for (self.layers) |layer| {
            output = layer.forward(output);
        }
        output = self.norm.forward(output);
        return output;
    }
    
    pub fn embedTokens(self: AceEmb, tokens: zml.Tensor) zml.Tensor {
        return self.embed_tokens.forward(tokens).withPartialTags(.{ .d });
    }

    pub fn normEmbed(self: AceEmb, embed: zml.Tensor) zml.Tensor {
        return self.norm.forward(embed).reuseBuffer(embed);
    }
};


const TransformerLayer = struct {
    id: u32,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_: u32, store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
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
        const x0_normalized = self.input_norm.forward(x0);
        const delta0 = self.att_layer.forward(x0_normalized);
        const x1 = x0.add(delta0);
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized).add(x1);
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

    pub fn init(store: zml.io.TensorStore.View, config: Config) !AttLayer {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, null), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, null), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, null), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, null), null, .d),
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

    /// Full-sequence causal self-attention without KV cache.
    pub fn forward(self: AttLayer, x: zml.Tensor) zml.Tensor {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        const pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{ .s });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const attn_mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, q.dtype(), null);
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn_output);
        return delta.rename(.{ .d_out = .d });
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, null),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, null),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, null),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        const up_projection = input.dot(self.up_proj, .d);
        const gate_projection = input.dot(self.gate_proj, .d);
        const activation = gate_projection.silu().mul(up_projection);
        const output = activation.dot(self.down_proj, .d_out);
        return output.reuseBuffer(input);
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) RmsNorm {
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
        return normalized.mul(self.weights.withTags(.{ .d }).broad(input.shape()));
    }
};
