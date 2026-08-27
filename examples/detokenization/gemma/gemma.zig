const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;
const main = @import("../main.zig");
const dialects = @import("mlir/dialects");
const base = @import("base.zig");

const TokenIds = base.TokenIds;
const LayerType = base.LayerType;
const RopeParameters = base.RopeParameters;
const ModelConfig = base.ModelConfig;
const Config = base.Config;
const GenerationConfig = base.GenerationConfig;
const Options = base.Options;
const KvCache = base.KvCache;
const ActivationCache = base.ActivationCache;

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
    activation_cache: ?ActivationCache,
    activation_cache_buffers: ?zml.Bufferized(ActivationCache),
    collect_activations: bool,
    sampling_strategy_buffers: zml.Bufferized(zml.nn.DynamicSamplingStrategy),

    pub fn init(zml_handler: *main.Zml_handler, path: []const u8, collect_activations: bool) !Gemma_handler {
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
            .seq_len = 1024,
            .hidden_size = config.hidden_size,
            .intermediate_size = config.intermediate_size,
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
        const activation_cache: ?ActivationCache = if (collect_activations) .init(options) else null;

        const exes = try compileModel(zml_handler, model, options, collect_activations);

        std.log.info("Gemma load buffers", .{});
        var model_buffers = try model.load(zml_handler, &store);
        errdefer Gemma.unloadBuffers(&model_buffers, zml_handler.allocator);
        std.log.info("Gemma model loaded", .{});

        var kv_cache_buffers = try kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
        errdefer KvCache.deinitBuffer(&kv_cache_buffers);
        var activation_cache_buffers: ?zml.Bufferized(ActivationCache) = if (activation_cache) |cache|
            try cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated)
        else
            null;
        errdefer if (activation_cache_buffers) |*buffers| ActivationCache.deinitBuffer(buffers);
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
            .activation_cache = activation_cache,
            .activation_cache_buffers = activation_cache_buffers,
            .collect_activations = collect_activations,
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

    pub fn compileModel(zml_handler: *main.Zml_handler, model: Gemma, options: Options, collect_activations: bool) !GemmaExes {
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
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, collect_activations_: bool, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .prefill(options_);
                const activation_cache_: ?ActivationCache = if (collect_activations_) .init(options_) else null;
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer, activation_cache_, collect_activations_ }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, collect_activations, opts });
        var prefill_local_layer_future_awaited = false;
        errdefer if (!prefill_local_layer_future_awaited) if (prefill_local_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_local_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, collect_activations_: bool, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .decode(options_);
                const activation_cache_: ?ActivationCache = if (collect_activations_) .init(options_) else null;
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer, activation_cache_, collect_activations_ }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, collect_activations, opts });
        var decode_local_layer_future_awaited = false;
        errdefer if (!decode_local_layer_future_awaited) if (decode_local_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var prefill_global_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, collect_activations_: bool, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .prefill(options_);
                const activation_cache_: ?ActivationCache = if (collect_activations_) .init(options_) else null;
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer, activation_cache_, collect_activations_ }, opts_);
            }
        }.call, .{ zml_handler, model.layers[global_layer_index], options, collect_activations, opts });
        var prefill_global_layer_future_awaited = false;
        errdefer if (!prefill_global_layer_future_awaited) if (prefill_global_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_global_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, collect_activations_: bool, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .decode(options_);
                const activation_cache_: ?ActivationCache = if (collect_activations_) .init(options_) else null;
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer, activation_cache_, collect_activations_ }, opts_);
            }
        }.call, .{ zml_handler, model.layers[global_layer_index], options, collect_activations, opts });
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
        if (self.activation_cache_buffers) |*buffers| ActivationCache.deinitBuffer(buffers);
        zml.nn.DynamicSamplingStrategy.deinitBuffers(&self.sampling_strategy_buffers);
    }

    pub fn resetKvCache(self: *Gemma_handler, zml_handler: *main.Zml_handler) !void {
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        self.kv_cache_buffers = try self.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
    }

    pub fn resetActivationCache(self: *Gemma_handler, zml_handler: *main.Zml_handler) !void {
        const cache = self.activation_cache orelse return;
        if (self.activation_cache_buffers) |*buffers| ActivationCache.deinitBuffer(buffers);
        self.activation_cache_buffers = null;
        self.activation_cache_buffers = try cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
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

    pub fn sampleTokens(_: Gemma, logits: zml.Tensor, _: zml.nn.DynamicSamplingStrategy, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        // Keep sampling greedy for now: the dynamic sampler lowers top-k
        // to a full vocabulary sort, whose pipeline overwhelms the Metal compiler.
        const next_token = logits.argMax(.voc).indices.squeeze(.voc).convert(.u32);
        return .{ next_token, rng };
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

    pub fn forward(
        self: TransformerLayer,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        layer_index: zml.Tensor,
        activation_cache: ?ActivationCache,
        collect_activations: bool,
    ) struct { zml.Tensor, KvCache, ?ActivationCache } {
        const input_norm = self.input_norm.forward(x);
        const attn = self.att_layer.forward(input_norm, token_index, kv_cache, layer_index);
        const attention_projection = self.post_att_norm.forward(attn.output);
        const x_after_attn = x.add(attention_projection);
        const pre_ff_norm = self.pre_ff_norm.forward(x_after_attn);
        const mlp = self.mlp_layer.forward(pre_ff_norm);
        const out = x_after_attn.add(self.post_ff_norm.forward(mlp.output)).mul(self.layer_scalar.asScalar());

        const updated_activation_cache: ?ActivationCache = if (collect_activations) blk: {
            const cache = activation_cache orelse unreachable;
            const activation_index = token_index.remainder(zml.Tensor.scalar(ActivationCache.capacity, token_index.dtype()));
            const updated = cache.update(self.att_layer.is_global, layer_index, activation_index, .{
                .layer_input = x,
                .input_norm = input_norm,
                .q = attn.q,
                .k = attn.k,
                .v = attn.v,
                .attention_context = attn.context,
                .attention_output = attn.output,
                .post_attention_residual = x_after_attn,
                .pre_ff_norm = pre_ff_norm,
                .gate = mlp.gate,
                .up = mlp.up,
                .geglu = mlp.geglu,
                .post_mlp_residual = out,
            });
            break :blk updated.reuseBuffer(cache);
        } else null;

        return .{ out.reuseBuffer(x), attn.kv_cache.reuseBuffer(kv_cache), updated_activation_cache };
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

    const Output = struct {
        output: zml.Tensor,
        kv_cache: KvCache,
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        context: zml.Tensor,
    };

    pub fn forward(self: AttLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor) Output {
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

        const activation_q = q;
        const activation_k = k;
        const activation_v = v;
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

        return .{
            .output = attn_output,
            .kv_cache = new_kv_cache.reuseBuffer(kv_cache),
            .q = activation_q,
            .k = activation_k,
            .v = activation_v,
            .context = attn_heads_output,
        };
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

    const Output = struct {
        output: zml.Tensor,
        gate: zml.Tensor,
        up: zml.Tensor,
        geglu: zml.Tensor,
    };

    pub fn forward(self: MlpLayer, input: zml.Tensor) Output {
        const up_projection = input.dot(self.up_proj, .d);
        const gate_projection = input.dot(self.gate_proj, .d);
        const activation = gate_projection.gelu().mul(up_projection);
        const output = activation.dot(self.down_proj, .d_out);
        return .{
            .output = output,
            .gate = gate_projection,
            .up = up_projection,
            .geglu = activation,
        };
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
