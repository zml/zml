const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const inference = @import("inference.zig");

const dialects = @import("mlir/dialects");


pub const AceLlm_handler = struct {
    model: AceLlm,
    kv_cache: KvCache,
    config: Config,
    options: Options,
    shardings: main.Shardings,
    phase: Phase,
    tokenizer: zml.tokenizer.Tokenizer,
    exes: LlmExes,
    model_buffers: zml.Bufferized(AceLlm),
    kv_cache_buffers: zml.Bufferized(KvCache),

    pub fn init(zml_handler: *main.Zml_handler) !AceLlm_handler {
        zml_handler.tic(&zml_handler.timers.llm.init);
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.acellm);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();

        try main.printSafetensors(registry);

        std.log.info("5Hz parse config and safetensors", .{});
        const parsed_config = try main.parseConfig(Config, zml_handler.allocator, zml_handler.io, repo);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        std.log.info("5Hz parsed", .{});

        const tokenizer = try loadTokenizer(zml_handler, repo);
        const phase = try buildTokenMasks(zml_handler.allocator, tokenizer, config.vocab_size);

        std.log.info("5Hz initialize model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const view = if (store.view().hasKey("model")) store.view().withPrefix("model") else store.view();
        const model: AceLlm = try .init(zml_handler.allocator, view, config, phase);
        std.log.info("5Hz initialized", .{});

        const options: Options = .{
            .seq_len = 2048,
            .hidden_size = config.hidden_size,
            .voc_size = config.vocab_size,
            .num_hidden_layers = config.num_hidden_layers,
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        };

        const shardings: main.Shardings = try .init(zml_handler.platform);

        const kv_cache: KvCache = .init(zml.Shape.init(.{
            .layer = config.num_hidden_layers,
            .k = options.seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, .bf16));

        zml_handler.toc(&zml_handler.timers.llm.init);
        zml_handler.tic(&zml_handler.timers.llm.compile);

        const exes = try compileModel(zml_handler, model, options, shardings);
        
        zml_handler.toc(&zml_handler.timers.llm.compile);
        zml_handler.tic(&zml_handler.timers.llm.load);

        std.log.info("5Hz load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &shardings.all());
        std.log.info("5Hz model loaded", .{});

        zml_handler.toc(&zml_handler.timers.llm.load);
        
        return .{
            .model = model,
            .kv_cache = kv_cache,
            .config = config,
            .options = options,
            .shardings = shardings,
            .phase = phase,
            .tokenizer = tokenizer,
            .exes = exes,
            .model_buffers = model_buffers,
            .kv_cache_buffers = try kv_cache.initBuffer(zml_handler.io, zml_handler.platform, shardings.replicated),
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

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceLlm, options: Options, shardings: main.Shardings) !LlmExes {
        const shardings_arr = shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
        std.log.info("5Hz compile models", .{});
        
        // compile token embeddings
        
        var prefill_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.EmbedTokensParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{ params.tokens }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_embed_future_awaited = false;
        errdefer if (!prefill_embed_future_awaited) if (prefill_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.EmbedTokensParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{ params.tokens }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var decode_embed_future_awaited = false;
        errdefer if (!decode_embed_future_awaited) if (decode_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile layers pre/attn/post
        
        var prefill_layer_pre_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.PreAttnParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .preAttention, .{ params.embeds }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var prefill_layer_pre_future_awaited = false;
        errdefer if (!prefill_layer_pre_future_awaited) if (prefill_layer_pre_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_layer_pre_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.PreAttnParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .preAttention, .{ params.embeds }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var decode_layer_pre_future_awaited = false;
        errdefer if (!decode_layer_pre_future_awaited) if (decode_layer_pre_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_layer_attn_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.AttnParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .attention,
                    .{ params.x, params.index, params.kv_cache, params.layer, params.q, params.k, params.v }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var prefill_layer_attn_future_awaited = false;
        errdefer if (!prefill_layer_attn_future_awaited) if (prefill_layer_attn_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_layer_attn_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.AttnParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .attention,
                    .{ params.x, params.index, params.kv_cache, params.layer, params.q, params.k, params.v }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var decode_layer_attn_future_awaited = false;
        errdefer if (!decode_layer_attn_future_awaited) if (decode_layer_attn_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_layer_post_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.PostAttnParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .postAttention,
                    .{ params.x1, params.attn1, params.x2, params.attn2 }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var prefill_layer_post_future_awaited = false;
        errdefer if (!prefill_layer_post_future_awaited) if (prefill_layer_post_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_layer_post_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.PostAttnParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .postAttention,
                    .{ params.x1, params.attn1, params.x2, params.attn2 }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var decode_layer_post_future_awaited = false;
        errdefer if (!decode_layer_post_future_awaited) if (decode_layer_post_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile select/logits/sample embedding
        
        var prefill_select_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.SelectEmbedsParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .selectEmbed,
                    .{ params.embeds, params.pred_index }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_select_future_awaited = false;
        errdefer if (!prefill_select_future_awaited) if (prefill_select_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var logit_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.ComputeLogitsParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .computeLogits, .{
                        params.embeds }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var logit_future_awaited = false;
        errdefer if (!logit_future_awaited) if (logit_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};
        
        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.SampleParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{
                        params.logits, params.rng, false }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // wait all parallel compilations terminate
        
        const prefill_embed_exe = try prefill_embed_future.await(zml_handler.io);
        prefill_embed_future_awaited = true;

        const prefill_layer_pre_exe = try prefill_layer_pre_future.await(zml_handler.io);
        prefill_layer_pre_future_awaited = true;

        const prefill_layer_attn_exe = try prefill_layer_attn_future.await(zml_handler.io);
        prefill_layer_attn_future_awaited = true;

        const prefill_layer_post_exe = try prefill_layer_post_future.await(zml_handler.io);
        prefill_layer_post_future_awaited = true;

        const prefill_select_exe = try prefill_select_future.await(zml_handler.io);
        prefill_select_future_awaited = true;

        const decode_embed_exe = try decode_embed_future.await(zml_handler.io);
        decode_embed_future_awaited = true;

        const decode_layer_pre_exe = try decode_layer_pre_future.await(zml_handler.io);
        decode_layer_pre_future_awaited = true;

        const decode_layer_attn_exe = try decode_layer_attn_future.await(zml_handler.io);
        decode_layer_attn_future_awaited = true;

        const decode_layer_post_exe = try decode_layer_post_future.await(zml_handler.io);
        decode_layer_post_future_awaited = true;

        const logits_exe = try logit_future.await(zml_handler.io);
        logit_future_awaited = true;
        
        const sample_exe = try sample_future.await(zml_handler.io);
        sample_future_awaited = true;

        return .{
            .prefill_embed_exe = prefill_embed_exe,
            .prefill_embed_args = try prefill_embed_exe.args(zml_handler.allocator),
            .prefill_embed_results = try prefill_embed_exe.results(zml_handler.allocator),
            .prefill_layer_pre_exe = prefill_layer_pre_exe,
            .prefill_layer_pre_args = try prefill_layer_pre_exe.args(zml_handler.allocator),
            .prefill_layer_pre_results = try prefill_layer_pre_exe.results(zml_handler.allocator),
            .prefill_layer_attn_exe = prefill_layer_attn_exe,
            .prefill_layer_attn_args = try prefill_layer_attn_exe.args(zml_handler.allocator),
            .prefill_layer_attn_results = try prefill_layer_attn_exe.results(zml_handler.allocator),
            .prefill_layer_post_exe = prefill_layer_post_exe,
            .prefill_layer_post_args = try prefill_layer_post_exe.args(zml_handler.allocator),
            .prefill_layer_post_results = try prefill_layer_post_exe.results(zml_handler.allocator),
            .prefill_select_exe = prefill_select_exe,
            .prefill_select_args = try prefill_select_exe.args(zml_handler.allocator),
            .prefill_select_results = try prefill_select_exe.results(zml_handler.allocator),
            .decode_embed_exe = decode_embed_exe,
            .decode_embed_args = try decode_embed_exe.args(zml_handler.allocator),
            .decode_embed_results = try decode_embed_exe.results(zml_handler.allocator),
            .decode_layer_pre_exe = decode_layer_pre_exe,
            .decode_layer_pre_args = try decode_layer_pre_exe.args(zml_handler.allocator),
            .decode_layer_pre_results = try decode_layer_pre_exe.results(zml_handler.allocator),
            .decode_layer_attn_exe = decode_layer_attn_exe,
            .decode_layer_attn_args = try decode_layer_attn_exe.args(zml_handler.allocator),
            .decode_layer_attn_results = try decode_layer_attn_exe.results(zml_handler.allocator),
            .decode_layer_post_exe = decode_layer_post_exe,
            .decode_layer_post_args = try decode_layer_post_exe.args(zml_handler.allocator),
            .decode_layer_post_results = try decode_layer_post_exe.results(zml_handler.allocator),
            .logits_exe = logits_exe,
            .logits_args = try logits_exe.args(zml_handler.allocator),
            .logits_results = try logits_exe.results(zml_handler.allocator),
            .sample_exe = sample_exe,
            .sample_args = try sample_exe.args(zml_handler.allocator),
            .sample_results = try sample_exe.results(zml_handler.allocator),
        };
    }

    pub fn buildTokenMasks(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, voc_size: usize) !Phase {
        var tokenizer_decoder = try tokenizer.decoder();
        defer tokenizer_decoder.deinit();
        const token_slice = try allocator.alloc(u32, 1);
        defer allocator.free(token_slice);
        var text_voc_size: u32 = 0;
        for (0..voc_size) |i| {
            const id: u32 = @intCast(i);
            token_slice[0] = id;
            const chunk = try tokenizer_decoder.decode(token_slice);
            const l = chunk.len;
            if (l < 16) {
                continue;
            } else {
                if (std.mem.eql(u8, chunk[0..12], "<|audio_code")) {
                    text_voc_size = @intCast(i);
                    break;
                } else {
                    continue;
                }
            }
        }
        const codes_voc_size: u32 = 64000;
        return .{
            .text_voc_size = text_voc_size,
            .codes_voc_size = codes_voc_size,
            .phase1_voc = .{ .start = 0, .end = text_voc_size },
            .phase2_voc = .{ .start = text_voc_size, .end = text_voc_size + codes_voc_size },
        };
    }

    pub fn unloadBuffers(self: *AceLlm_handler, allocator: std.mem.Allocator) void {
        AceLlm.unloadBuffers(&self.model_buffers, allocator);
        KvCache.deinitBuffer(&self.kv_cache_buffers);
    }

    pub fn deinit(self: *AceLlm_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.tokenizer.deinit();
        self.exes.deinit(allocator);
    }

};

pub const AceCfg_handler = struct {
    llm: *AceLlm_handler,
    exes: CfgExes,
    cond_kv_cache_buffers: zml.Bufferized(KvCache),
    uncond_kv_cache_buffers: zml.Bufferized(KvCache),

    pub fn initFromLlm(zml_handler: *main.Zml_handler, acellm: *AceLlm_handler) !AceCfg_handler {
        zml_handler.tic(&zml_handler.timers.cfg.compile);

        const exes = try compileModel(zml_handler, acellm);

        zml_handler.toc(&zml_handler.timers.cfg.compile);

        return .{
            .llm = acellm,
            .exes = exes,
            .cond_kv_cache_buffers = try acellm.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, acellm.shardings.replicated),
            .uncond_kv_cache_buffers = try acellm.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, acellm.shardings.replicated),
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, llm: *AceLlm_handler) !CfgExes {
        const shardings_arr = llm.shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
        std.log.info("5Hz compile CFG", .{});

        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.SampleParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{ params.logits, params.rng, false }, opts_);
            }
        }.call, .{ zml_handler, llm.model, llm.options, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var cfg_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: AceLlm.CfgParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .cfg, .{ params.logits }, opts_);
            }
        }.call, .{ zml_handler, llm.model, llm.options, opts });
        var cfg_future_awaited = false;
        errdefer if (!cfg_future_awaited) if (cfg_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        const sample_exe = try sample_future.await(zml_handler.io);
        sample_future_awaited = true;

        const cfg_exe = try cfg_future.await(zml_handler.io);
        cfg_future_awaited = true;
        
        return .{ 
            .sample_exe = sample_exe,
            .sample_args = try sample_exe.args(zml_handler.allocator),
            .sample_results = try sample_exe.results(zml_handler.allocator),
            .cfg_exe = cfg_exe,
            .cfg_args = try cfg_exe.args(zml_handler.allocator),
            .cfg_results = try cfg_exe.results(zml_handler.allocator),
        };
    }

    pub fn unloadBuffers(self: *AceCfg_handler) void {
        KvCache.deinitBuffer(&self.cond_kv_cache_buffers);
        KvCache.deinitBuffer(&self.uncond_kv_cache_buffers);
    }

    pub fn deinit(self: *AceCfg_handler, allocator: std.mem.Allocator) void {
        self.exes.deinit(allocator);
    }

};


pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }),
    head_dim: ?u32 = null,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool = false,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    vocab_size: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .bos_token_id = self.bos_token_id,
            .eos_token_id = switch (self.eos_token_id.value) {
                .int => .{ .value = .{ .int = self.eos_token_id.value.int } },
                .ints => .{ .value = .{ .ints = try allocator.dupe(u32, self.eos_token_id.value.ints) } },
            },
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

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
        switch (self.eos_token_id.value) {
            .int => {},
            .ints => allocator.free(self.eos_token_id.value.ints),
        }
    }
};

pub const Options = struct {
    seq_len: u32,
    hidden_size: u32,
    voc_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
};

pub const Phase = struct {
    text_voc_size: u32,
    codes_voc_size: u32,
    phase1_voc: zml.Tensor.Slice,
    phase2_voc: zml.Tensor.Slice,
};

pub const LlmExes = struct {
    prefill_embed_exe: zml.Exe,
    prefill_embed_args: zml.Exe.Arguments,
    prefill_embed_results: zml.Exe.Results,
    
    prefill_layer_pre_exe: zml.Exe,
    prefill_layer_pre_args: zml.Exe.Arguments,
    prefill_layer_pre_results: zml.Exe.Results,
    prefill_layer_attn_exe: zml.Exe,
    prefill_layer_attn_args: zml.Exe.Arguments,
    prefill_layer_attn_results: zml.Exe.Results,
    prefill_layer_post_exe: zml.Exe,
    prefill_layer_post_args: zml.Exe.Arguments,
    prefill_layer_post_results: zml.Exe.Results,
    
    prefill_select_exe: zml.Exe,
    prefill_select_args: zml.Exe.Arguments,
    prefill_select_results: zml.Exe.Results,
    
    decode_embed_exe: zml.Exe,
    decode_embed_args: zml.Exe.Arguments,
    decode_embed_results: zml.Exe.Results,
    
    decode_layer_pre_exe: zml.Exe,
    decode_layer_pre_args: zml.Exe.Arguments,
    decode_layer_pre_results: zml.Exe.Results,
    decode_layer_attn_exe: zml.Exe,
    decode_layer_attn_args: zml.Exe.Arguments,
    decode_layer_attn_results: zml.Exe.Results,
    decode_layer_post_exe: zml.Exe,
    decode_layer_post_args: zml.Exe.Arguments,
    decode_layer_post_results: zml.Exe.Results,
    
    logits_exe: zml.Exe,
    logits_args: zml.Exe.Arguments,
    logits_results: zml.Exe.Results,
    sample_exe: zml.Exe,
    sample_args: zml.Exe.Arguments,
    sample_results: zml.Exe.Results,

    pub fn deinit(self: LlmExes, allocator: std.mem.Allocator) void {
        self.prefill_embed_exe.deinit();
        self.prefill_embed_args.deinit(allocator);
        self.prefill_embed_results.deinit(allocator);
        self.prefill_layer_pre_exe.deinit();
        self.prefill_layer_pre_args.deinit(allocator);
        self.prefill_layer_pre_results.deinit(allocator);
        self.prefill_layer_attn_exe.deinit();
        self.prefill_layer_attn_args.deinit(allocator);
        self.prefill_layer_attn_results.deinit(allocator);
        self.prefill_layer_post_exe.deinit();
        self.prefill_layer_post_args.deinit(allocator);
        self.prefill_layer_post_results.deinit(allocator);
        self.prefill_select_exe.deinit();
        self.prefill_select_args.deinit(allocator);
        self.prefill_select_results.deinit(allocator);
        self.decode_embed_exe.deinit();
        self.decode_embed_args.deinit(allocator);
        self.decode_embed_results.deinit(allocator);
        self.decode_layer_pre_exe.deinit();
        self.decode_layer_pre_args.deinit(allocator);
        self.decode_layer_pre_results.deinit(allocator);
        self.decode_layer_attn_exe.deinit();
        self.decode_layer_attn_args.deinit(allocator);
        self.decode_layer_attn_results.deinit(allocator);
        self.decode_layer_post_exe.deinit();
        self.decode_layer_post_args.deinit(allocator);
        self.decode_layer_post_results.deinit(allocator);
        self.logits_exe.deinit();
        self.logits_args.deinit(allocator);
        self.logits_results.deinit(allocator);
        self.sample_exe.deinit();
        self.sample_args.deinit(allocator);
        self.sample_results.deinit(allocator);
    }
};

pub const CfgExes = struct {
    sample_exe: zml.Exe,
    sample_args: zml.Exe.Arguments,
    sample_results: zml.Exe.Results,
    cfg_exe: zml.Exe,
    cfg_args: zml.Exe.Arguments,
    cfg_results: zml.Exe.Results,

    pub fn deinit(self: *CfgExes, allocator: std.mem.Allocator) void {
        self.sample_exe.deinit();
        self.sample_args.deinit(allocator);
        self.sample_results.deinit(allocator);
        self.cfg_exe.deinit();
        self.cfg_args.deinit(allocator);
        self.cfg_results.deinit(allocator);
    }
};


pub const AceLlm = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    phase: Phase,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, phase: Phase) !AceLlm {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, null) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
            .phase = phase,
        };
    }

    pub fn load(self: *const AceLlm, zml_handler: *main.Zml_handler, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceLlm) {
        var progress = zml_handler.progress.start("Load 5Hz weights", store.registry.tensors.count());
        defer progress.end();
        return zml.io.load(AceLlm, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = &progress,
        });
    }

    pub fn deinit(self: *const AceLlm, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceLlm), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }


    pub const EmbedTokensParams = struct {
        tokens: zml.Tensor,
        pub fn prefill(options: Options) EmbedTokensParams {
            return .{ .tokens = .init(.{ .b = 2, .s = options.seq_len }, .u32) };
        }
        pub fn decode(_: Options) EmbedTokensParams {
            return .{ .tokens = .init(.{ .b = 2, .s = 1 }, .u32) };
        }
    };
    
    pub fn embedTokens(self: AceLlm, tokens: zml.Tensor) zml.Tensor {
        return self.embed_tokens.forward(tokens.withPartialTags(.{ .s })).withPartialTags(.{ .d });
    }


    pub const SelectEmbedsParams = struct {
        embeds: zml.Tensor,
        pred_index: zml.Tensor,
        pub fn prefill(options: Options) SelectEmbedsParams {
            return .{
                .embeds = .init(.{ .b = 2, .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .pred_index = .init(.{ .b = 2, .s = 1 }, .u32),
            };
        }
    };
    
    pub fn selectEmbed(_: AceLlm, embeddings: zml.Tensor, pred_index: zml.Tensor) zml.Tensor {
        const e1 = embeddings.slice1d(.b, .{ .start = 0, .end = 1 }).squeeze(.b);
        const e2 = embeddings.slice1d(.b, .{ .start = 1, .end = 2 }).squeeze(.b);
        
        const p1 = pred_index.slice1d(.b, .{ .start = 0, .end = 1 }).squeeze(.b).squeeze(.s);
        const p2 = pred_index.slice1d(.b, .{ .start = 1, .end = 2 }).squeeze(.b).squeeze(.s);

        const s1 = e1.dynamicSlice1d(e1.axis(.s), .{ .start = p1, .len = 1 });
        const s2 = e2.dynamicSlice1d(e2.axis(.s), .{ .start = p2, .len = 1 });
        
        return zml.Tensor.stack(&.{ s1, s2 }, 0, .b);
    }


    pub const ComputeLogitsParams = struct {
        embeds: zml.Tensor,
        pub fn exec(options: Options) ComputeLogitsParams {
            return .{ .embeds = .init(.{ .b = 2, .s = 1, .d = options.hidden_size }, .bf16) };
        }
    };
    
    pub fn computeLogits(self: AceLlm, embed: zml.Tensor) zml.Tensor {
        const output = self.norm.forward(embed);
        // TODO: we could improve the perf by only computing the logits for the relevant voc slice
        const logits = self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(output, .d).convert(.f32);
        return logits.transpose(.{ .b, .voc, .s });
    }


    pub const CfgParams = struct {
        logits: zml.Tensor,
        pub fn exec(options: Options) CfgParams {
            return .{ .logits = .init(.{ .b = 2, .voc = options.voc_size, .s = 1 }, .f32) };
        }
    };
    
    pub fn cfg(_: AceLlm, logits: zml.Tensor) zml.Tensor {
        const uncond_logits = logits.slice1d(.b, .{ .start = 1, .end = 2 }).broad(logits.shape());
        const combined = uncond_logits.add((logits.sub(uncond_logits)).scale(2.0));
        return combined.reuseBuffer(logits);
    }

    
    pub const SampleParams = struct {
        logits: zml.Tensor,
        rng: zml.Tensor.Rng,
        pub fn exec(options: Options) SampleParams {
            return .{
                .logits = .init(.{ .b = 2, .voc = options.voc_size, .s = 1 }, .f32),
                .rng = zml.Tensor.Rng.init(),
            };
        }
    };
    
    pub fn sampleTokens(self: AceLlm, logits: zml.Tensor, rng: zml.Tensor.Rng, cfg_phase: bool) struct { zml.Tensor, zml.Tensor.Rng } {
        const phase_logits_b = logits.slice1d(.voc, if (cfg_phase) self.phase.phase2_voc else self.phase.phase1_voc);
        const phase_logits = phase_logits_b.slice1d(.b, .{ .start = 0, .end = 1 }).squeeze(.b);
        var next_token, const new_rng = sampleNucleus(phase_logits, rng);
        // in cfg, next token is a position relative to the audiocodes sub voc slice, translate it back to the full voc slice
        if (cfg_phase) next_token = next_token.addConstant(self.phase.text_voc_size);
        return .{ next_token.convert(.u32), new_rng };
    }

    pub fn sampleNucleus(logits: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const top_p = 0.9;
        const temperature = 0.85;

        const sorted = logits.sort(.voc, .{ .descending = true });
        var sorted_logits = sorted.values;
        sorted_logits = sorted_logits.scale(1.0 / temperature);

        // softmax implementation is resilient to overflows
        const probs = sorted_logits.softmax(.voc);
        const cumulative_probs = probs.cumulativeSum(.voc);
        // shift cumulative sum to the right by 1, padding with 0 at the start
        // this allows the first token that makes the sum exceed top_p to be included
        // this also makes so that the most probable token is included even if its proba exceeds top_p
        const padding: zml.Tensor.Pad = .{ .low = 1, .high = 0, .interior = 0 };
        const size_slice_voc: zml.Tensor.Slice = .{ .start = 0, .end = logits.dim(.voc) };
        const size_slice_s: zml.Tensor.Slice = .{ .start = 0, .end = logits.dim(.s) };
        const shifted_cum_probs = cumulative_probs.pad(0, .{ .voc = padding }).slice(&.{ size_slice_voc, size_slice_s });

        const top_p_tensor = zml.Tensor.scalar(top_p, .f32).broad(probs.shape());
        const is_in_top_p = shifted_cum_probs.cmp(.LE, top_p_tensor);

        const zero = zml.Tensor.scalar(0.0, .f32).broad(probs.shape());
        const filtered_probs = is_in_top_p.select(probs, zero);
        const filtered_sum = filtered_probs.sum(.voc);
        const normalized_probs = filtered_probs.div(filtered_sum.broad(filtered_probs.shape()));

        const next_rng, const uniform_sample = rng.uniform(zml.Shape.init(.{}, .f32), .{});
        const sample_threshold = uniform_sample.broad(normalized_probs.shape());
        const normalized_cdf = normalized_probs.cumulativeSum(.voc);
        // we compare the cumulative probas to the threshold: we get a tensor [0 0 ... 0 1 1 ... 1]
        // and the sampled token is the position of the first 1. argMax returns the first max value
        const sampled_token = normalized_cdf.cmp(.GE, sample_threshold).argMax(.voc).indices;
        const next_token = sorted.indices.gather(.{ .voc = sampled_token.squeeze(.voc) }, .{});
        return .{ next_token, next_rng };
    }
    
};


const TransformerLayer = struct {
    id: u8,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_: u8, store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
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


    pub const PreAttnParams = struct {
        embeds: zml.Tensor,
        pub fn prefill(options: Options) PreAttnParams {
            return .{ .embeds = .init(.{ .b = 2, .s = options.seq_len, .d = options.hidden_size }, .bf16) };
        }
        pub fn decode(options: Options) PreAttnParams {
            return .{ .embeds = .init(.{ .b = 2, .s = 1, .d = options.hidden_size }, .bf16) };
        }
    };
    
    pub fn preAttention(self: TransformerLayer, x: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        // this is batched with size 2
        const x_normalized = self.input_norm.forward(x);
        const q = self.att_layer.q_proj.forward(x_normalized);
        const k = self.att_layer.k_proj.forward(x_normalized);
        const v = self.att_layer.v_proj.forward(x_normalized);

        const s1: zml.Tensor.Slice = .{ .start = 0, .end = 1 };
        const s2: zml.Tensor.Slice = .{ .start = 1, .end = 2 };

        const x1 = x_normalized.slice1d(.b, s1).squeeze(.b);
        const q1 = q.slice1d(.b, s1).squeeze(.b);
        const k1 = k.slice1d(.b, s1).squeeze(.b);
        const v1 = v.slice1d(.b, s1).squeeze(.b);

        const x2 = x_normalized.slice1d(.b, s2).squeeze(.b);
        const q2 = q.slice1d(.b, s2).squeeze(.b);
        const k2 = k.slice1d(.b, s2).squeeze(.b);
        const v2 = v.slice1d(.b, s2).squeeze(.b);
        
        return .{ x1, q1, k1, v1, x2, q2, k2, v2 };
    }


    pub const AttnParams = struct {
        x: zml.Tensor,
        index: zml.Tensor,
        kv_cache: KvCache,
        layer: zml.Tensor,
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        pub fn prefill(options: Options) AttnParams {
            return .{
                .x = .init(.{ .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .index = .init(.{}, .u32),
                .kv_cache = .init(zml.Shape.init(.{
                    .layer = options.num_hidden_layers,
                    .k = options.seq_len,
                    .h = options.num_key_value_heads,
                    .hd = options.head_dim,
                }, .bf16)),
                .layer = .init(.{}, .u32),
                .q = .init(.{ .s = options.seq_len, .d = options.num_attention_heads * options.head_dim }, .bf16),
                .k = .init(.{ .s = options.seq_len, .d = options.num_key_value_heads * options.head_dim }, .bf16),
                .v = .init(.{ .s = options.seq_len, .d = options.num_key_value_heads * options.head_dim }, .bf16),
            };
        }
        pub fn decode(options: Options) AttnParams {
            return .{
                .x = .init(.{ .s = 1, .d = options.hidden_size }, .bf16),
                .index = .init(.{}, .u32),
                .kv_cache = .init(zml.Shape.init(.{
                    .layer = options.num_hidden_layers,
                    .k = options.seq_len,
                    .h = options.num_key_value_heads,
                    .hd = options.head_dim,
                }, .bf16)),
                .layer = .init(.{}, .u32),
                .q = .init(.{ .s = 1, .d = options.num_attention_heads * options.head_dim }, .bf16),
                .k = .init(.{ .s = 1, .d = options.num_key_value_heads * options.head_dim }, .bf16),
                .v = .init(.{ .s = 1, .d = options.num_key_value_heads * options.head_dim }, .bf16),
            };
        }
    };
    
    pub fn attention(self: TransformerLayer, x_norm: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) struct { zml.Tensor, KvCache } {
        // this is not batched
        const delta, const updated_kv_cache = self.att_layer.forward(x_norm, token_index, kv_cache, layer_index, q, k, v);
        return .{ delta, updated_kv_cache.reuseBuffer(kv_cache) };
    }


    pub const PostAttnParams = struct {
        x1: zml.Tensor,
        attn1: zml.Tensor,
        x2: zml.Tensor,
        attn2: zml.Tensor,
        pub fn prefill(options: Options) PostAttnParams {
            return .{
                .x1 = .init(.{ .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .attn1 = .init(.{ .s = options.seq_len, .d = options.num_attention_heads * options.head_dim }, .bf16),
                .x2 = .init(.{ .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .attn2 = .init(.{ .s = options.seq_len, .d = options.num_attention_heads * options.head_dim }, .bf16),
            };
        }
        pub fn decode(options: Options) PostAttnParams {
            return .{
                .x1 = .init(.{ .s = 1, .d = options.hidden_size }, .bf16),
                .attn1 = .init(.{ .s = 1, .d = options.num_attention_heads * options.head_dim }, .bf16),
                .x2 = .init(.{ .s = 1, .d = options.hidden_size }, .bf16),
                .attn2 = .init(.{ .s = 1, .d = options.num_attention_heads * options.head_dim }, .bf16),
            };
        }
    };
    
    pub fn postAttention(self: TransformerLayer, x1: zml.Tensor, attn1: zml.Tensor, x2: zml.Tensor, attn2: zml.Tensor) zml.Tensor {
        // this is batched with size 2
        var x = zml.Tensor.stack(&.{ x1, x2 }, 0, .b);
        const attn = zml.Tensor.stack(&.{ attn1, attn2 }, 0, .b);
        const delta1 = self.att_layer.o_proj.forward(attn).rename(.{ .d_out = .d });
        x = x.add(delta1);
        const delta2 = self.mlp_layer.forward(self.post_att_norm.forward(x));
        x = x.add(delta2);
        return x;
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

    pub fn forward(self: AttLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor, q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        var q = q_.splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = k_.splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = v_.splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        // [0..dim(.s)],  [0..seq_len] in prefill,  { 0 } in decode
        var pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{ .s });
        // translate to [0..seq_len] in prefill and { token_index } in decode
        pos_index = pos_index.add(token_index.broad(pos_index.shape()));
        
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(layer_index, k, v, token_index);
        k = new_kv_cache.keys(layer_index).convert(dtype);
        v = new_kv_cache.values(layer_index).convert(dtype);

        const attn_heads_output = zml.attention.attention.attention(q, k, v, token_index, .vanilla, .vanilla);
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        return .{ attn_output, new_kv_cache.reuseBuffer(kv_cache) };
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
        return output;
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


pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn keys(self: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, layer_index: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;
        return if (token_index) |idx| .{
                .k = self.k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
            } else .{
                .k = self.k.scatterSlices(.{ .layer = layer }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
            };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
        };
    }
};

pub fn print(x: zml.Tensor, n: u8) void {
    const name = &.{ n };
    const sh = x._shape;
    const nb_dims = sh.rank();
    switch (nb_dims) {
        0 => std.log.info("{s} is scalar", .{ name }),
        1 => std.log.info("{s} is dim 1 : {s} = {d}", .{ name, sh.tag(0), sh.dim(0) }),
        2 => std.log.info("{s} is dim 2 : {s} x {s} = {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.dim(0), sh.dim(1) }),
        3 => std.log.info("{s} is dim 3 : {s} x {s} x {s} = {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.dim(0), sh.dim(1), sh.dim(2) }),
        4 => std.log.info("{s} is dim 4 : {s} x {s} x {s} x {s} = {d} x {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.tag(3), sh.dim(0), sh.dim(1), sh.dim(2), sh.dim(3) }),
        else => std.log.info("{s} is rank >= 5", .{ name }),
    }
}