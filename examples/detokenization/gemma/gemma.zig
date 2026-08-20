const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;
const main = @import("../main.zig");
const dialects = @import("mlir/dialects");

pub const Gemma_handler = struct {
    model: Gemma,
    kv_cache: KvCache,
    config: Config,
    generation_config: GenerationConfig,
    options: Options,
    tokenizer: zml.tokenizer.Tokenizer,
    exes: GemmaExes,
    model_buffers: zml.Bufferized(Gemma),
    kv_cache_buffers: zml.Bufferized(KvCache),
    sampling_strategy_buffers: zml.Bufferized(zml.nn.DynamicSamplingStrategy),

    pub fn init(zml_handler: *main.Zml_handler, path: []const u8) !Gemma_handler {
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, path);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();

        //try main.printSafetensors(registry);

        std.log.info("Gemma parse config and safetensors", .{});
        const parsed_config = try parseConfig(ModelConfig, zml_handler.allocator, zml_handler.io, repo);
        defer parsed_config.deinit();
        const config = try parsed_config.value.text_config.dupe(zml_handler.allocator);
        errdefer config.deinit(zml_handler.allocator);
        const generation_config = try GenerationConfig.load(zml_handler.allocator, zml_handler.io, repo, config);
        errdefer generation_config.deinit(zml_handler.allocator);
        std.log.info("Gemma parsed", .{});

        const tokenizer = try loadTokenizer(zml_handler, repo);

        std.log.info("Gemma initialize model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const model: Gemma = try .init(zml_handler.allocator, store.view(), config, generation_config);
        std.log.info("Gemma initialized", .{});

        var num_local_layers: u32 = 0;
        var num_global_layers: u32 = 0;
        for (config.layer_types) |layer_type| switch (layer_type) {
            .sliding_attention => num_local_layers += 1,
            .full_attention => num_global_layers += 1,
        };

        const options: Options = .{
            .seq_len = 2048,
            .hidden_size = config.hidden_size,
            .voc_size = config.vocab_size,
            .num_hidden_layers = config.num_hidden_layers,
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .head_dim = config.head_dim,
            .num_global_key_value_heads = config.num_global_key_value_heads,
            .global_head_dim = config.global_head_dim,
            .num_local_layers = num_local_layers,
            .num_global_layers = num_global_layers,
        };

        const kv_cache: KvCache = .init(
            zml.Shape.init(.{
                .layer = num_local_layers,
                .k = options.seq_len,
                .h = config.num_key_value_heads,
                .hd = config.head_dim,
            }, .bf16),
            zml.Shape.init(.{
                .layer = num_global_layers,
                .k = options.seq_len,
                .h = config.num_global_key_value_heads,
                .hd = config.global_head_dim,
            }, .bf16),
        );

        const exes = try compileModel(zml_handler, model, options);

        std.log.info("Gemma load buffers", .{});
        var model_buffers = try model.load(zml_handler, &store);
        errdefer Gemma.unloadBuffers(&model_buffers, zml_handler.allocator);
        std.log.info("Gemma model loaded", .{});

        var kv_cache_buffers = try kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
        errdefer KvCache.deinitBuffer(&kv_cache_buffers);
        var sampling_strategy_buffers = try generation_config.samplingStrategyBuffers(zml_handler.io, zml_handler.platform);
        errdefer zml.nn.DynamicSamplingStrategy.deinitBuffers(&sampling_strategy_buffers);

        return .{
            .model = model,
            .kv_cache = kv_cache,
            .config = config,
            .generation_config = generation_config,
            .options = options,
            .tokenizer = tokenizer,
            .exes = exes,
            .model_buffers = model_buffers,
            .kv_cache_buffers = kv_cache_buffers,
            .sampling_strategy_buffers = sampling_strategy_buffers,
        };
    }

    pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
        const file = try dir.openFile(io, "config.json", .{});
        defer file.close(io);

        var buffer: [256]u8 = undefined;
        var file_reader = file.reader(io, &buffer);
        var reader: std.json.Reader = .init(allocator, &file_reader.interface);
        defer reader.deinit();

        return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
    }

    pub fn loadTokenizer(zml_handler: *main.Zml_handler, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
        const tokenizer_json_file = try dir.openFile(zml_handler.io, "tokenizer.json", .{});
        defer tokenizer_json_file.close(zml_handler.io);
        var reader = tokenizer_json_file.reader(zml_handler.io, &.{});
        const bytes = try reader.interface.readAlloc(zml_handler.allocator, try tokenizer_json_file.length(zml_handler.io));
        defer zml_handler.allocator.free(bytes);
        return try .fromBytes(zml_handler.allocator, bytes);
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: Gemma, options: Options) !GemmaExes {
        const opts: zml.module.CompilationOptions = .{};
        std.log.info("Gemma compile models", .{});
        const global_layer_index = for (model.layers, 0..) |layer, i| {
            if (layer.att_layer.is_global) break i;
        } else return error.MissingGlobalAttentionLayer;

        // compile token embeddings

        var prefill_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Gemma, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Gemma.EmbedTokensParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{params.tokens}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_embed_future_awaited = false;
        errdefer if (!prefill_embed_future_awaited) if (prefill_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Gemma, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Gemma.EmbedTokensParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{params.tokens}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var decode_embed_future_awaited = false;
        errdefer if (!decode_embed_future_awaited) if (decode_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile layers

        var prefill_local_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var prefill_local_layer_future_awaited = false;
        errdefer if (!prefill_local_layer_future_awaited) if (prefill_local_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_local_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var decode_local_layer_future_awaited = false;
        errdefer if (!decode_local_layer_future_awaited) if (decode_local_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_global_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[global_layer_index], options, opts });
        var prefill_global_layer_future_awaited = false;
        errdefer if (!prefill_global_layer_future_awaited) if (prefill_global_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_global_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[global_layer_index], options, opts });
        var decode_global_layer_future_awaited = false;
        errdefer if (!decode_global_layer_future_awaited) if (decode_global_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile select/logits/sample embedding

        var prefill_select_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Gemma, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Gemma.SelectEmbedsParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .selectEmbed, .{ params.embeds, params.pred_index }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_select_future_awaited = false;
        errdefer if (!prefill_select_future_awaited) if (prefill_select_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var logit_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Gemma, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Gemma.ComputeLogitsParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .computeLogits, .{params.embeds}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var logit_future_awaited = false;
        errdefer if (!logit_future_awaited) if (logit_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Gemma, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Gemma.SampleParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{ params.logits, params.sampling_strategy, params.rng }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // wait all parallel compilations terminate

        const prefill_embed_exe = try prefill_embed_future.await(zml_handler.io);
        prefill_embed_future_awaited = true;

        const prefill_local_layer_exe = try prefill_local_layer_future.await(zml_handler.io);
        prefill_local_layer_future_awaited = true;

        const prefill_global_layer_exe = try prefill_global_layer_future.await(zml_handler.io);
        prefill_global_layer_future_awaited = true;

        const prefill_select_exe = try prefill_select_future.await(zml_handler.io);
        prefill_select_future_awaited = true;

        const decode_embed_exe = try decode_embed_future.await(zml_handler.io);
        decode_embed_future_awaited = true;

        const decode_local_layer_exe = try decode_local_layer_future.await(zml_handler.io);
        decode_local_layer_future_awaited = true;

        const decode_global_layer_exe = try decode_global_layer_future.await(zml_handler.io);
        decode_global_layer_future_awaited = true;

        const logits_exe = try logit_future.await(zml_handler.io);
        logit_future_awaited = true;

        const sample_exe = try sample_future.await(zml_handler.io);
        sample_future_awaited = true;

        return .{
            .prefill_embed_exe = prefill_embed_exe,
            .prefill_embed_args = try prefill_embed_exe.args(zml_handler.allocator),
            .prefill_embed_results = try prefill_embed_exe.results(zml_handler.allocator),
            .prefill_local_layer_exe = prefill_local_layer_exe,
            .prefill_local_layer_args = try prefill_local_layer_exe.args(zml_handler.allocator),
            .prefill_local_layer_results = try prefill_local_layer_exe.results(zml_handler.allocator),
            .prefill_global_layer_exe = prefill_global_layer_exe,
            .prefill_global_layer_args = try prefill_global_layer_exe.args(zml_handler.allocator),
            .prefill_global_layer_results = try prefill_global_layer_exe.results(zml_handler.allocator),
            .prefill_select_exe = prefill_select_exe,
            .prefill_select_args = try prefill_select_exe.args(zml_handler.allocator),
            .prefill_select_results = try prefill_select_exe.results(zml_handler.allocator),
            .decode_embed_exe = decode_embed_exe,
            .decode_embed_args = try decode_embed_exe.args(zml_handler.allocator),
            .decode_embed_results = try decode_embed_exe.results(zml_handler.allocator),
            .decode_local_layer_exe = decode_local_layer_exe,
            .decode_local_layer_args = try decode_local_layer_exe.args(zml_handler.allocator),
            .decode_local_layer_results = try decode_local_layer_exe.results(zml_handler.allocator),
            .decode_global_layer_exe = decode_global_layer_exe,
            .decode_global_layer_args = try decode_global_layer_exe.args(zml_handler.allocator),
            .decode_global_layer_results = try decode_global_layer_exe.results(zml_handler.allocator),
            .logits_exe = logits_exe,
            .logits_args = try logits_exe.args(zml_handler.allocator),
            .logits_results = try logits_exe.results(zml_handler.allocator),
            .sample_exe = sample_exe,
            .sample_args = try sample_exe.args(zml_handler.allocator),
            .sample_results = try sample_exe.results(zml_handler.allocator),
        };
    }

    pub fn unloadBuffers(self: *Gemma_handler, allocator: std.mem.Allocator) void {
        Gemma.unloadBuffers(&self.model_buffers, allocator);
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.nn.DynamicSamplingStrategy.deinitBuffers(&self.sampling_strategy_buffers);
    }

    pub fn resetKvCache(self: *Gemma_handler, zml_handler: *main.Zml_handler) !void {
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        self.kv_cache_buffers = try self.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
    }

    pub fn deinit(self: *Gemma_handler, allocator: std.mem.Allocator) void {
        self.unloadBuffers(allocator);
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.generation_config.deinit(allocator);
        self.tokenizer.deinit();
        self.exes.deinit(allocator);
    }
};

pub const TokenIds = stdx.json.Union(union(enum) {
    int: u32,
    ints: []u32,
});

fn dupeTokenIds(token_ids: TokenIds, allocator: std.mem.Allocator) !TokenIds {
    return switch (token_ids.value) {
        .int => .{ .value = .{ .int = token_ids.value.int } },
        .ints => .{ .value = .{ .ints = try allocator.dupe(u32, token_ids.value.ints) } },
    };
}

fn deinitTokenIds(token_ids: TokenIds, allocator: std.mem.Allocator) void {
    switch (token_ids.value) {
        .int => {},
        .ints => allocator.free(token_ids.value.ints),
    }
}

fn tokenIdsContain(token_ids: TokenIds, token_id: u32) bool {
    return switch (token_ids.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}

pub const LayerType = enum {
    sliding_attention,
    full_attention,
};

pub const RopeParameters = struct {
    full_attention: zml.nn.RopeOpts.Scaling,
    sliding_attention: zml.nn.RopeOpts.Scaling,
};

pub const ModelConfig = struct {
    text_config: Config,
};

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: TokenIds,
    pad_token_id: u32,
    head_dim: u32,
    global_head_dim: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    num_global_key_value_heads: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    attention_k_eq_v: bool,
    sliding_window: u32,
    final_logit_softcapping: ?f32 = null,
    layer_types: []LayerType,
    rope_parameters: RopeParameters,
    vocab_size: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .bos_token_id = self.bos_token_id,
            .eos_token_id = try dupeTokenIds(self.eos_token_id, allocator),
            .pad_token_id = self.pad_token_id,
            .head_dim = self.head_dim,
            .global_head_dim = self.global_head_dim,
            .hidden_size = self.hidden_size,
            .intermediate_size = self.intermediate_size,
            .num_hidden_layers = self.num_hidden_layers,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .num_global_key_value_heads = self.num_global_key_value_heads,
            .max_position_embeddings = self.max_position_embeddings,
            .rms_norm_eps = self.rms_norm_eps,
            .tie_word_embeddings = self.tie_word_embeddings,
            .attention_k_eq_v = self.attention_k_eq_v,
            .sliding_window = self.sliding_window,
            .final_logit_softcapping = self.final_logit_softcapping,
            .layer_types = try allocator.dupe(LayerType, self.layer_types),
            .rope_parameters = self.rope_parameters,
            .vocab_size = self.vocab_size,
        };
    }

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
        deinitTokenIds(self.eos_token_id, allocator);
        allocator.free(self.layer_types);
    }
};

pub const GenerationConfig = struct {
    bos_token_id: u32,
    do_sample: bool = false,
    eos_token_id: TokenIds,
    pad_token_id: u32,
    temperature: f32 = 1.0,
    top_k: u32 = 50,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    suppress_tokens: []u32 = &.{},

    pub const max_sampling_top_k: u32 = 64;

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, model_config: Config) !GenerationConfig {
        const file = dir.openFile(io, "generation_config.json", .{}) catch |err| {
            if (err == error.FileNotFound) return GenerationConfig.fromModelConfig(model_config, allocator);
            return err;
        };
        defer file.close(io);

        var buffer: [256]u8 = undefined;
        var file_reader = file.reader(io, &buffer);
        var reader: std.json.Reader = .init(allocator, &file_reader.interface);
        defer reader.deinit();

        const parsed = try std.json.parseFromTokenSource(GenerationConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return parsed.value.dupe(allocator);
    }

    pub fn fromModelConfig(model_config: Config, allocator: std.mem.Allocator) !GenerationConfig {
        return .{
            .bos_token_id = model_config.bos_token_id,
            .do_sample = false,
            .eos_token_id = try dupeTokenIds(model_config.eos_token_id, allocator),
            .pad_token_id = model_config.pad_token_id,
            .suppress_tokens = try allocator.alloc(u32, 0),
        };
    }

    pub fn dupe(self: GenerationConfig, allocator: std.mem.Allocator) !GenerationConfig {
        return .{
            .bos_token_id = self.bos_token_id,
            .do_sample = self.do_sample,
            .eos_token_id = try dupeTokenIds(self.eos_token_id, allocator),
            .pad_token_id = self.pad_token_id,
            .temperature = self.temperature,
            .top_k = self.top_k,
            .top_p = self.top_p,
            .min_p = self.min_p,
            .suppress_tokens = try allocator.dupe(u32, self.suppress_tokens),
        };
    }

    pub fn deinit(self: GenerationConfig, allocator: std.mem.Allocator) void {
        deinitTokenIds(self.eos_token_id, allocator);
        allocator.free(self.suppress_tokens);
    }

    pub fn samplingStrategy(self: GenerationConfig) zml.nn.SamplingStrategy {
        // zml.nn.SamplingStrategy currently supports top-k and temperature; top_p is parsed for config fidelity.
        if (!self.do_sample) return .{};
        return .{
            .topk = self.top_k,
            .temperature = self.temperature,
        };
    }

    pub fn samplingStrategyBuffers(self: GenerationConfig, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(zml.nn.DynamicSamplingStrategy) {
        const top_k = if (self.do_sample) @min(self.top_k, max_sampling_top_k) else 1;
        return try zml.nn.DynamicSamplingStrategy.makeBuffers(io, platform, .f32, .{
            .top_k = top_k,
            .temperature = self.temperature,
            .top_p = if (self.do_sample) self.top_p else 1.0,
            .min_p = if (self.do_sample) self.min_p else 0.0,
        });
    }

    pub fn isEosToken(self: GenerationConfig, token_id: u32) bool {
        return tokenIdsContain(self.eos_token_id, token_id);
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
    num_global_key_value_heads: u32,
    global_head_dim: u32,
    num_local_layers: u32,
    num_global_layers: u32,
};

pub const GemmaExes = struct {
    prefill_embed_exe: zml.Exe,
    prefill_embed_args: zml.Exe.Arguments,
    prefill_embed_results: zml.Exe.Results,

    prefill_local_layer_exe: zml.Exe,
    prefill_local_layer_args: zml.Exe.Arguments,
    prefill_local_layer_results: zml.Exe.Results,

    prefill_global_layer_exe: zml.Exe,
    prefill_global_layer_args: zml.Exe.Arguments,
    prefill_global_layer_results: zml.Exe.Results,

    prefill_select_exe: zml.Exe,
    prefill_select_args: zml.Exe.Arguments,
    prefill_select_results: zml.Exe.Results,

    decode_embed_exe: zml.Exe,
    decode_embed_args: zml.Exe.Arguments,
    decode_embed_results: zml.Exe.Results,

    decode_local_layer_exe: zml.Exe,
    decode_local_layer_args: zml.Exe.Arguments,
    decode_local_layer_results: zml.Exe.Results,

    decode_global_layer_exe: zml.Exe,
    decode_global_layer_args: zml.Exe.Arguments,
    decode_global_layer_results: zml.Exe.Results,

    logits_exe: zml.Exe,
    logits_args: zml.Exe.Arguments,
    logits_results: zml.Exe.Results,

    sample_exe: zml.Exe,
    sample_args: zml.Exe.Arguments,
    sample_results: zml.Exe.Results,

    pub fn deinit(self: GemmaExes, allocator: std.mem.Allocator) void {
        self.prefill_embed_exe.deinit();
        self.prefill_embed_args.deinit(allocator);
        self.prefill_embed_results.deinit(allocator);
        self.prefill_local_layer_exe.deinit();
        self.prefill_local_layer_args.deinit(allocator);
        self.prefill_local_layer_results.deinit(allocator);
        self.prefill_global_layer_exe.deinit();
        self.prefill_global_layer_args.deinit(allocator);
        self.prefill_global_layer_results.deinit(allocator);
        self.prefill_select_exe.deinit();
        self.prefill_select_args.deinit(allocator);
        self.prefill_select_results.deinit(allocator);
        self.decode_embed_exe.deinit();
        self.decode_embed_args.deinit(allocator);
        self.decode_embed_results.deinit(allocator);
        self.decode_local_layer_exe.deinit();
        self.decode_local_layer_args.deinit(allocator);
        self.decode_local_layer_results.deinit(allocator);
        self.decode_global_layer_exe.deinit();
        self.decode_global_layer_args.deinit(allocator);
        self.decode_global_layer_results.deinit(allocator);
        self.logits_exe.deinit();
        self.logits_args.deinit(allocator);
        self.logits_results.deinit(allocator);
        self.sample_exe.deinit();
        self.sample_args.deinit(allocator);
        self.sample_results.deinit(allocator);
    }
};

pub const Gemma = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    embed_scale: f32,
    final_logit_softcapping: ?f32,
    suppress_tokens: []const u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, generation_config: GenerationConfig) !Gemma {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("model.language_model.layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor("model.language_model.embed_tokens.weight", .{ .voc, .d }, .replicated) },
            .layers = layers,
            .norm = .init(store.withPrefix("model.language_model.norm"), config),
            .embed_scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size))),
            .final_logit_softcapping = config.final_logit_softcapping,
            .suppress_tokens = generation_config.suppress_tokens,
        };
    }

    pub fn load(self: *const Gemma, zml_handler: *main.Zml_handler, store: *const zml.io.TensorStore) !zml.Bufferized(Gemma) {
        var progress = zml_handler.progress.start("Load Gemma weights", store.registry.tensors.count());
        defer progress.end();

        var buffers = try zml.mem.bufferize(zml_handler.allocator, Gemma, self);
        errdefer Gemma.unloadBuffers(&buffers, zml_handler.allocator);

        var loader: zml.io.Loader = try .init(zml_handler.allocator, zml_handler.platform, .{
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
        });
        defer loader.deinit();

        loader.load(zml_handler.io, Gemma, self, &buffers, store, &.{}, .{ .progress = &progress });
        try loader.await(zml_handler.io);
        return buffers;
    }

    pub fn deinit(self: *const Gemma, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma), allocator: std.mem.Allocator) void {
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
            return .{ .tokens = .init(.{ .s = options.seq_len }, .u32) };
        }
        pub fn decode(_: Options) EmbedTokensParams {
            return .{ .tokens = .init(.{ .s = 1 }, .u32) };
        }
    };

    pub fn embedTokens(self: Gemma, tokens: zml.Tensor) zml.Tensor {
        return self.embed_tokens.forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d}).scale(self.embed_scale);
    }

    pub const SelectEmbedsParams = struct {
        embeds: zml.Tensor,
        pred_index: zml.Tensor,
        pub fn prefill(options: Options) SelectEmbedsParams {
            return .{
                .embeds = .init(.{ .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .pred_index = .init(.{}, .u32),
            };
        }
    };

    pub fn selectEmbed(_: Gemma, embeddings: zml.Tensor, pred_index: zml.Tensor) zml.Tensor {
        return embeddings.dynamicSlice1d(embeddings.axis(.s), .{ .start = pred_index, .len = 1 });
    }

    pub const ComputeLogitsParams = struct {
        embeds: zml.Tensor,
        pub fn exec(options: Options) ComputeLogitsParams {
            return .{ .embeds = .init(.{ .s = 1, .d = options.hidden_size }, .bf16) };
        }
    };

    pub fn computeLogits(self: Gemma, embed: zml.Tensor) zml.Tensor {
        const normalized_embed = self.norm.forward(embed);
        var logits = self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(normalized_embed, .d).convert(.f32);
        if (self.final_logit_softcapping) |softcap| {
            logits = logits.divByConst(softcap).tanh().scale(softcap);
        }
        const token_ids = zml.Tensor.arange(.{ .end = logits.dim(.voc) }, .u32).withTags(.{.voc});
        for (self.suppress_tokens) |token_id| {
            const suppressed = token_ids.cmp(.EQ, zml.Tensor.scalar(token_id, .u32));
            logits = suppressed.broad(logits.shape()).select(zml.Tensor.scalar(-std.math.inf(f32), .f32).broad(logits.shape()), logits);
        }
        return logits.transpose(.{ .voc, .s });
    }

    pub const SampleParams = struct {
        logits: zml.Tensor,
        sampling_strategy: zml.nn.DynamicSamplingStrategy,
        rng: zml.Tensor.Rng,
        pub fn exec(options: Options) SampleParams {
            return .{
                .logits = .init(.{ .voc = options.voc_size, .s = 1 }, .f32),
                .sampling_strategy = zml.nn.DynamicSamplingStrategy.init(.f32, GenerationConfig.max_sampling_top_k),
                .rng = zml.Tensor.Rng.init(),
            };
        }
    };

    pub fn sampleTokens(_: Gemma, logits: zml.Tensor, sampling_strategy: zml.nn.DynamicSamplingStrategy, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const next_token, const new_rng = zml.nn.sampleTokensDynamic(logits, sampling_strategy, rng);
        return .{ next_token.convert(.u32), new_rng };
    }
};

const TransformerLayer = struct {
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    mlp_layer: MlpLayer,
    layer_scalar: zml.Tensor,

    pub fn init(id_: u8, store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_norm = .init(store.withPrefix("input_layernorm"), config),
            .att_layer = try .init(store.withPrefix("self_attn"), id_, config),
            .post_att_norm = .init(store.withPrefix("post_attention_layernorm"), config),
            .pre_ff_norm = .init(store.withPrefix("pre_feedforward_layernorm"), config),
            .post_ff_norm = .init(store.withPrefix("post_feedforward_layernorm"), config),
            .mlp_layer = try .init(store.withPrefix("mlp")),
            .layer_scalar = store.createTensor("layer_scalar", .{.scalar}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_norm);
        AttLayer.unloadBuffers(&self.att_layer);
        RmsNorm.unloadBuffers(&self.post_att_norm);
        RmsNorm.unloadBuffers(&self.pre_ff_norm);
        RmsNorm.unloadBuffers(&self.post_ff_norm);
        MlpLayer.unloadBuffers(&self.mlp_layer);
        self.layer_scalar.deinit();
    }

    pub const TransformerParams = struct {
        x: zml.Tensor,
        index: zml.Tensor,
        kv_cache: KvCache,
        layer: zml.Tensor,

        pub fn prefill(options: Options) TransformerParams {
            return .{
                .x = .init(.{ .s = options.seq_len, .d = options.hidden_size }, .bf16),
                .index = .init(.{}, .u32),
                .kv_cache = initCache(options),
                .layer = .init(.{}, .u32),
            };
        }

        pub fn decode(options: Options) TransformerParams {
            return .{
                .x = .init(.{ .s = 1, .d = options.hidden_size }, .bf16),
                .index = .init(.{}, .u32),
                .kv_cache = initCache(options),
                .layer = .init(.{}, .u32),
            };
        }

        fn initCache(options: Options) KvCache {
            return .init(
                zml.Shape.init(.{
                    .layer = options.num_local_layers,
                    .k = options.seq_len,
                    .h = options.num_key_value_heads,
                    .hd = options.head_dim,
                }, .bf16),
                zml.Shape.init(.{
                    .layer = options.num_global_layers,
                    .k = options.seq_len,
                    .h = options.num_global_key_value_heads,
                    .hd = options.global_head_dim,
                }, .bf16),
            );
        }
    };

    pub fn forward(self: TransformerLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor) struct { zml.Tensor, KvCache } {
        const attn, const new_cache = self.att_layer.forward(self.input_norm.forward(x), token_index, kv_cache, layer_index);
        const x_after_attn = x.add(self.post_att_norm.forward(attn));
        const mlp = self.mlp_layer.forward(self.pre_ff_norm.forward(x_after_attn));
        const out = x_after_attn.add(self.post_ff_norm.forward(mlp)).mul(self.layer_scalar.asScalar());
        return .{ out.reuseBuffer(x), new_cache.reuseBuffer(kv_cache) };
    }
};

const AttLayer = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: ?zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNormNoScale,
    o_proj: zml.nn.Linear,
    num_heads: i64,
    num_kv_heads: i64,
    sliding_window: u32,
    is_global: bool,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, layer_id: u8, config: Config) !AttLayer {
        const is_global = config.layer_types[layer_id] == .full_attention;
        const rope_scaling = if (is_global) config.rope_parameters.full_attention else config.rope_parameters.sliding_attention;
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .v_proj = if (is_global and config.attention_k_eq_v) null else .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config),
            .k_norm = .init(store.withPrefix("k_norm"), config),
            .v_norm = .{ .eps = config.rms_norm_eps },
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(if (is_global) config.num_global_key_value_heads else config.num_key_value_heads),
            .sliding_window = @intCast(config.sliding_window),
            .is_global = is_global,
            .rope_opts = .{
                .layout = if (is_global) .real_pass_im_pass else .real_im_pass,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        if (self.v_proj) |*v_proj| v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: AttLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor) struct { zml.Tensor, KvCache } {
        const q_projection = self.q_proj.forward(x);
        const k_projection = self.k_proj.forward(x);
        const v_projection = if (self.v_proj) |v_proj| v_proj.forward(x) else k_projection;

        var q = q_projection.splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = k_projection.splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        var v = v_projection.splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        v = self.v_norm.forward(v.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        // [0..dim(.s)],  [0..seq_len] in prefill,  { 0 } in decode
        var pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s});
        // translate to [0..seq_len] in prefill and { token_index } in decode
        pos_index = pos_index.add(token_index.broad(pos_index.shape()));

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(self.is_global, layer_index, k, v, token_index);
        k = new_kv_cache.keys(self.is_global, layer_index).convert(dtype);
        v = new_kv_cache.values(self.is_global, layer_index).convert(dtype);

        const seq_len = k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(
            .{ .q = seq_len, .k = seq_len },
            q.dtype(),
            if (self.is_global) null else self.sliding_window,
        );
        attn_mask = attn_mask.gatherSlices(
            zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()),
            token_index.appendAxes(.{.coord}),
            .{},
        );
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{
            .attn_mask = attn_mask,
            .scale = zml.Tensor.scalar(1.0, q.dtype()),
        });
        const attn_output = self.o_proj.forward(attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s })).rename(.{ .d_out = .d });

        return .{ attn_output, new_kv_cache.reuseBuffer(kv_cache) };
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .replicated),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .replicated),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .replicated),
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
        const activation = gate_projection.gelu().mul(up_projection);
        const output = activation.dot(self.down_proj, .d_out);
        return output;
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{.d_out}, .replicated),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const input_f32 = input.convert(.f32);
        const variance = input_f32.powByConst(2).mean(.d);
        const normalized = input_f32.mul(variance.addConstant(self.eps).powByConst(-0.5).broad(input.shape()));
        return normalized.mul(self.weights.withTags(.{.d}).convert(.f32).broad(input.shape())).convert(input.dtype());
    }
};

const RmsNormNoScale = struct {
    eps: f32,

    pub fn forward(self: RmsNormNoScale, input: zml.Tensor) zml.Tensor {
        const input_f32 = input.convert(.f32);
        const variance = input_f32.powByConst(2).mean(.d);
        return input_f32.mul(variance.addConstant(self.eps).powByConst(-0.5).broad(input.shape())).convert(input.dtype());
    }
};

pub const KvCache = struct {
    local_k: zml.Tensor,
    local_v: zml.Tensor,
    global_k: zml.Tensor,
    global_v: zml.Tensor,

    pub fn init(local_shape: zml.Shape, global_shape: zml.Shape) KvCache {
        return .{
            .local_k = .fromShape(local_shape),
            .local_v = .fromShape(local_shape),
            .global_k = .fromShape(global_shape),
            .global_v = .fromShape(global_shape),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
        var local_k = try zml.Buffer.uninitialized(io, platform, self.local_k.shape(), sharding, .{});
        errdefer local_k.deinit();
        var local_v = try zml.Buffer.uninitialized(io, platform, self.local_v.shape(), sharding, .{});
        errdefer local_v.deinit();
        var global_k = try zml.Buffer.uninitialized(io, platform, self.global_k.shape(), sharding, .{});
        errdefer global_k.deinit();
        return .{
            .local_k = local_k,
            .local_v = local_v,
            .global_k = global_k,
            .global_v = try zml.Buffer.uninitialized(io, platform, self.global_v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.local_k.deinit();
        self.local_v.deinit();
        self.global_k.deinit();
        self.global_v.deinit();
    }

    pub fn keys(self: KvCache, is_global: bool, layer_index: zml.Tensor) zml.Tensor {
        const k = if (is_global) self.global_k else self.local_k;
        return k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, is_global: bool, layer_index: zml.Tensor) zml.Tensor {
        const v = if (is_global) self.global_v else self.local_v;
        return v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, is_global: bool, layer_index: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const old_k = if (is_global) self.global_k else self.local_k;
        const old_v = if (is_global) self.global_v else self.local_v;
        const k_shape = old_k.shape().drop(.layer);
        const v_shape = old_v.shape().drop(.layer);
        var layer = layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;
        const updated_k = if (token_index) |idx|
            old_k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(old_k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_k)
        else
            old_k.scatterSlices(.{ .layer = layer }, new_k.convert(old_k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_k);
        const updated_v = if (token_index) |idx|
            old_v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(old_v.dtype()).transpose(v_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_v)
        else
            old_v.scatterSlices(.{ .layer = layer }, new_v.convert(old_v.dtype()).transpose(v_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_v);

        var result = self;
        if (is_global) {
            result.global_k = updated_k;
            result.global_v = updated_v;
        } else {
            result.local_k = updated_k;
            result.local_v = updated_v;
        }
        return result;
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .local_k = self.local_k.reuseBuffer(other.local_k),
            .local_v = self.local_v.reuseBuffer(other.local_v),
            .global_k = self.global_k.reuseBuffer(other.global_k),
            .global_v = self.global_v.reuseBuffer(other.global_v),
        };
    }
};
