const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const inference = @import("inference.zig");

const dialects = @import("mlir/dialects");


pub const AceLlm_handler = struct {
    model: AceLlm,
    params: LlmParams,
    config: Config,
    options: Options,
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

        //try main.printSafetensors(registry);

        std.log.info("5Hz parse config and safetensors", .{});
        const parsed_config = try main.parseConfig(Config, zml_handler.allocator, zml_handler.io, repo);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        const acellm_options: Options = .{};
        std.log.info("5Hz parsed", .{});

        const tokenizer = try loadTokenizer(zml_handler, repo);
        const phase = try buildTokenMasks(zml_handler.allocator, tokenizer, config.vocab_size);

        std.log.info("5Hz initialize model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const view = if (store.view().hasKey("model")) store.view().withPrefix("model") else store.view();
        const model: AceLlm = try .init(zml_handler.allocator, view, config, phase);
        std.log.info("5Hz initialized", .{});

        // Specify shapes of input arguments
        const dtype = model.embed_tokens.weight.dtype();
        const params: LlmParams = .{
            .prefill_tokens = .init(.{ .s = acellm_options.seq_len }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .pred_index = .init(.{}, .u32),
            .prefill_embeds = .init(.{ .s = acellm_options.seq_len, .d = config.hidden_size }, .bf16),
            .decode_embeds = .init(.{ .s = 1, .d = config.hidden_size }, .bf16),
            .logits = .init(.{ .voc = config.vocab_size, .s = 1 }, .f32),
            .kv_cache = .init(zml.Shape.init(.{
                .layer = config.num_hidden_layers,
                .k = acellm_options.seq_len,
                .h = config.num_key_value_heads,
                .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
            }, dtype), zml_handler.io),
            .layer_index = .init(.{}, .u32),
            .rng = .init(),
            .shardings = try .init(zml_handler.platform),
            .attention_metadata = .init(.fromBackend(.cuda_fa2, @intCast(2048), @intCast(config.num_attention_heads))),
        };

        zml_handler.toc(&zml_handler.timers.llm.init);
        zml_handler.tic(&zml_handler.timers.llm.compile);

        const exes = try compileModel(zml_handler, model, params);
        
        zml_handler.toc(&zml_handler.timers.llm.compile);
        zml_handler.tic(&zml_handler.timers.llm.load);

        std.log.info("5Hz load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &params.shardings.all());
        std.log.info("5Hz model loaded", .{});

        zml_handler.toc(&zml_handler.timers.llm.load);
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .options = acellm_options,
            .phase = phase,
            .tokenizer = tokenizer,
            .exes = exes,
            .model_buffers = model_buffers,
            .kv_cache_buffers = try params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, params.shardings.replicated),
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

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceLlm, params: LlmParams) !LlmExes {
        const shardings_arr = params.shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
        std.log.info("5Hz compile models", .{});

        var prefill_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{
                        params_.prefill_tokens }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var prefill_embed_future_awaited = false;
        errdefer if (!prefill_embed_future_awaited) if (prefill_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward,
                    .{ params_.prefill_embeds, params_.token_index, params_.kv_cache, params_.layer_index, params_.attention_metadata }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], params, opts });
        var prefill_layer_future_awaited = false;
        errdefer if (!prefill_layer_future_awaited) if (prefill_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_select_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .selectEmbed,
                    .{ params_.prefill_embeds, params_.pred_index }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var prefill_select_future_awaited = false;
        errdefer if (!prefill_select_future_awaited) if (prefill_select_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{
                        params_.decode_tokens }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var decode_embed_future_awaited = false;
        errdefer if (!decode_embed_future_awaited) if (decode_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward,
                    .{ params_.decode_embeds, params_.token_index, params_.kv_cache, params_.layer_index, params_.attention_metadata }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], params, opts });
        var decode_layer_future_awaited = false;
        errdefer if (!decode_layer_future_awaited) if (decode_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var logit_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .computeLogits, .{
                        params_.decode_embeds }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var logit_future_awaited = false;
        errdefer if (!logit_future_awaited) if (logit_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};
        
        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: LlmParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{
                        params_.logits, params_.rng, false }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        const prefill_embed_exe = try prefill_embed_future.await(zml_handler.io);
        prefill_embed_future_awaited = true;

        const prefill_layer_exe = try prefill_layer_future.await(zml_handler.io);
        prefill_layer_future_awaited = true;

        const prefill_select_exe = try prefill_select_future.await(zml_handler.io);
        prefill_select_future_awaited = true;

        const decode_embed_exe = try decode_embed_future.await(zml_handler.io);
        decode_embed_future_awaited = true;

        const decode_layer_exe = try decode_layer_future.await(zml_handler.io);
        decode_layer_future_awaited = true;

        const logits_exe = try logit_future.await(zml_handler.io);
        logit_future_awaited = true;
        
        const sample_exe = try sample_future.await(zml_handler.io);
        sample_future_awaited = true;

        return .{
            .prefill_embed_exe = prefill_embed_exe,
            .prefill_embed_args = try prefill_embed_exe.args(zml_handler.allocator),
            .prefill_embed_results = try prefill_embed_exe.results(zml_handler.allocator),
            .prefill_layer_exe = prefill_layer_exe,
            .prefill_layer_args = try prefill_layer_exe.args(zml_handler.allocator),
            .prefill_layer_results = try prefill_layer_exe.results(zml_handler.allocator),
            .prefill_select_exe = prefill_select_exe,
            .prefill_select_args = try prefill_select_exe.args(zml_handler.allocator),
            .prefill_select_results = try prefill_select_exe.results(zml_handler.allocator),
            .decode_embed_exe = decode_embed_exe,
            .decode_embed_args = try decode_embed_exe.args(zml_handler.allocator),
            .decode_embed_results = try decode_embed_exe.results(zml_handler.allocator),
            .decode_layer_exe = decode_layer_exe,
            .decode_layer_args = try decode_layer_exe.args(zml_handler.allocator),
            .decode_layer_results = try decode_layer_exe.results(zml_handler.allocator),
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
    params: CfgParams,
    exes: CfgExes,
    cond_kv_cache_buffers: zml.Bufferized(KvCache),
    uncond_kv_cache_buffers: zml.Bufferized(KvCache),

    pub fn initFromLlm(zml_handler: *main.Zml_handler, acellm: *AceLlm_handler) !AceCfg_handler {
        zml_handler.tic(&zml_handler.timers.cfg.init);
        const dtype = acellm.model.embed_tokens.weight.dtype();
        const params: CfgParams = .{
            .prefill_tokens = .init(.{ .s = acellm.options.seq_len }, .u32),
            .decode_token = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .pred_index = .init(.{}, .u32),
            .layer_index = .init(.{}, .u32),
            .cond_logits = .init(.{ .voc = acellm.config.vocab_size, .s = 1 }, .f32),
            .uncond_logits = .init(.{ .voc = acellm.config.vocab_size, .s = 1 }, .f32),
            .kv_cache = .init(zml.Shape.init(.{
                .layer = acellm.config.num_hidden_layers,
                .k = acellm.options.seq_len,
                .h = acellm.config.num_key_value_heads,
                .hd = acellm.config.head_dim orelse @divExact(acellm.config.hidden_size, acellm.config.num_attention_heads),
            }, dtype), zml_handler.io),
            .rng = .init(),
            .shardings = try .init(zml_handler.platform),
        };

        zml_handler.toc(&zml_handler.timers.cfg.init);
        zml_handler.tic(&zml_handler.timers.cfg.compile);

        const exes = try compileModel(zml_handler, acellm.model, params);

        zml_handler.toc(&zml_handler.timers.cfg.compile);

        return .{
            .llm = acellm,
            .params = params,
            .exes = exes,
            .cond_kv_cache_buffers = try params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, params.shardings.replicated),
            .uncond_kv_cache_buffers = try params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, params.shardings.replicated),
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceLlm, params: CfgParams) !CfgExes {
        const shardings_arr = params.shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
        std.log.info("5Hz compile CFG", .{});

        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: CfgParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{
                        params_.cond_logits, params_.rng, true }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var cfg_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceLlm, params_: CfgParams, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .cfg, .{
                        params_.cond_logits, params_.uncond_logits }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
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


pub const LlmParams = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    pred_index: zml.Tensor,
    prefill_embeds: zml.Tensor,
    decode_embeds: zml.Tensor,
    logits: zml.Tensor,
    kv_cache: KvCache,
    layer_index: zml.Tensor,
    rng: zml.Tensor.Rng,
    shardings: main.Shardings,
    attention_metadata: zml.attention.attention.Metadata,
};

pub const CfgParams = struct {
    prefill_tokens: zml.Tensor,
    decode_token: zml.Tensor,
    token_index: zml.Tensor,
    pred_index: zml.Tensor,
    layer_index: zml.Tensor,
    cond_logits: zml.Tensor,
    uncond_logits: zml.Tensor,
    kv_cache: KvCache,
    rng: zml.Tensor.Rng,
    shardings: main.Shardings,
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
    seq_len: u32 = 2048,
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
    prefill_layer_exe: zml.Exe,
    prefill_layer_args: zml.Exe.Arguments,
    prefill_layer_results: zml.Exe.Results,
    prefill_select_exe: zml.Exe,
    prefill_select_args: zml.Exe.Arguments,
    prefill_select_results: zml.Exe.Results,
    decode_embed_exe: zml.Exe,
    decode_embed_args: zml.Exe.Arguments,
    decode_embed_results: zml.Exe.Results,
    decode_layer_exe: zml.Exe,
    decode_layer_args: zml.Exe.Arguments,
    decode_layer_results: zml.Exe.Results,

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
        self.prefill_layer_exe.deinit();
        self.prefill_layer_args.deinit(allocator);
        self.prefill_layer_results.deinit(allocator);
        self.prefill_select_exe.deinit();
        self.prefill_select_args.deinit(allocator);
        self.prefill_select_results.deinit(allocator);
        self.decode_embed_exe.deinit();
        self.decode_embed_args.deinit(allocator);
        self.decode_embed_results.deinit(allocator);
        self.decode_layer_exe.deinit();
        self.decode_layer_args.deinit(allocator);
        self.decode_layer_results.deinit(allocator);
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
        std.log.info("OK", .{});
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


    pub fn embedTokens(self: AceLlm, tokens: zml.Tensor) zml.Tensor {
        return self.embed_tokens.forward(tokens.withPartialTags(.{ .s })).withPartialTags(.{ .d });
    }

    pub fn selectEmbed(self: AceLlm, embeddings: zml.Tensor, pred_index: zml.Tensor) zml.Tensor {
        _ = self;
        return embeddings.dynamicSlice1d(embeddings.axis(.s), .{ .start = pred_index, .len = 1 });
    }

    pub fn computeLogits(self: AceLlm, embed: zml.Tensor) zml.Tensor {
        const output = self.norm.forward(embed);
        // TODO: we could improve the perf by only computing the logits for the relevant voc slice
        return self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(output, .d).convert(.f32);
    }
    
    pub fn cfg(self: AceLlm, cond_logits: zml.Tensor, uncond_logits: zml.Tensor) zml.Tensor {
        _ = self;
        const combined = uncond_logits.add((cond_logits.sub(uncond_logits)).scale(2.0));
        return combined.reuseBuffer(cond_logits);
    }
    
    pub fn sampleTokens(self: AceLlm, logits: zml.Tensor, rng: zml.Tensor.Rng, cfg_phase: bool) struct { zml.Tensor, zml.Tensor.Rng } {
        const phase_logits = logits.slice1d(.voc, if (cfg_phase) self.phase.phase2_voc else self.phase.phase1_voc);
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
    
    pub fn forward(self: TransformerLayer, x0: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor, attention_metadata: zml.attention.attention.Metadata) struct { zml.Tensor, KvCache } {
        const x0_normalized = self.input_norm.forward(x0);
        const delta0, const updated_kv_cache = self.att_layer.forward(x0_normalized, token_index, kv_cache, layer_index, attention_metadata);
        const x1 = x0.add(delta0);
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized).add(x1);
        return .{ x2.reuseBuffer(x0), updated_kv_cache.reuseBuffer(kv_cache) };
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
    attention_parameters: zml.attention.attention.Parameters = undefined,
    

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
            .attention_parameters = .init(.fromBackend(.cuda_fa2)),
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

    pub fn forward(self: AttLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor, attention_metadata: zml.attention.attention.Metadata) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        var pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{ .s });
        pos_index = pos_index.add(token_index.broad(pos_index.shape()));

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(layer_index, k, v, token_index);
        k = new_kv_cache.keys(layer_index).convert(dtype);
        v = new_kv_cache.values(layer_index).convert(dtype);

        const attn_output = zml.attention.attention.attention(q, k, v, token_index, attention_metadata, self.attention_parameters);

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn).rename(.{ .d_out = .d });
        return .{ delta, new_kv_cache };
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
    io: std.Io,

    pub fn init(kv_shape: zml.Shape, io: std.Io) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .io = io,
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
                .io = self.io,
            } else .{
                .k = self.k.scatterSlices(.{ .layer = layer }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .io = self.io,
            };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .io = self.io,
        };
    }
};
