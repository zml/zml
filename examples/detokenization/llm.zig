const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const dialects = @import("mlir/dialects");

pub const Llm_handler = struct {
    model: Llm,
    kv_cache: KvCache,
    config: Config,
    generation_config: GenerationConfig,
    options: Options,
    tokenizer: zml.tokenizer.Tokenizer,
    exes: LlmExes,
    model_buffers: zml.Bufferized(Llm),
    kv_cache_buffers: zml.Bufferized(KvCache),

    pub fn init(zml_handler: *main.Zml_handler) !Llm_handler {
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();

        //try main.printSafetensors(registry);

        std.log.info("LLM parse config and safetensors", .{});
        const parsed_config = try parseConfig(Config, zml_handler.allocator, zml_handler.io, repo);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        errdefer config.deinit(zml_handler.allocator);
        const generation_config = try GenerationConfig.load(zml_handler.allocator, zml_handler.io, repo, config);
        errdefer generation_config.deinit(zml_handler.allocator);
        std.log.info("LLM parsed", .{});

        const tokenizer = try loadTokenizer(zml_handler, repo);

        std.log.info("LLM initialize model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const model: Llm = try .init(zml_handler.allocator, store.view(), config, generation_config);
        std.log.info("LLM initialized", .{});

        const options: Options = .{
            .seq_len = 2048,
            .hidden_size = config.hidden_size,
            .voc_size = config.vocab_size,
            .num_hidden_layers = config.num_hidden_layers,
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        };

        const kv_cache: KvCache = .init(zml.Shape.init(.{
            .layer = config.num_hidden_layers,
            .k = options.seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, .bf16));

        const exes = try compileModel(zml_handler, model, options);

        std.log.info("LLM load buffers", .{});
        var model_buffers = try model.load(zml_handler, &store);
        errdefer Llm.unloadBuffers(&model_buffers, zml_handler.allocator);
        std.log.info("LLM model loaded", .{});

        var kv_cache_buffers = try kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
        errdefer KvCache.deinitBuffer(&kv_cache_buffers);

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

    pub fn compileModel(zml_handler: *main.Zml_handler, model: Llm, options: Options) !LlmExes {
        const opts: zml.module.CompilationOptions = .{};
        std.log.info("LLM compile models", .{});

        // compile token embeddings

        var prefill_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.EmbedTokensParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{params.tokens}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_embed_future_awaited = false;
        errdefer if (!prefill_embed_future_awaited) if (prefill_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.EmbedTokensParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .embedTokens, .{params.tokens}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var decode_embed_future_awaited = false;
        errdefer if (!decode_embed_future_awaited) if (decode_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile layers

        var prefill_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var prefill_layer_future_awaited = false;
        errdefer if (!prefill_layer_future_awaited) if (prefill_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var decode_layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TransformerLayer, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: TransformerLayer.TransformerParams = .decode(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ params.x, params.index, params.kv_cache, params.layer }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], options, opts });
        var decode_layer_future_awaited = false;
        errdefer if (!decode_layer_future_awaited) if (decode_layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // compile select/logits/sample/graph embedding

        var prefill_select_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.SelectEmbedsParams = .prefill(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .selectEmbed, .{ params.embeds, params.pred_index }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var prefill_select_future_awaited = false;
        errdefer if (!prefill_select_future_awaited) if (prefill_select_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var logit_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.ComputeLogitsParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .computeLogits, .{params.embeds}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var logit_future_awaited = false;
        errdefer if (!logit_future_awaited) if (logit_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var sample_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.SampleParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .sampleTokens, .{ params.logits, params.rng }, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var sample_future_awaited = false;
        errdefer if (!sample_future_awaited) if (sample_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var graph_embed_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: Llm, options_: Options, opts_: zml.module.CompilationOptions) !zml.Exe {
                const params: Llm.GraphEmbedParams = .exec(options_);
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .graphEmbed, .{params.embed}, opts_);
            }
        }.call, .{ zml_handler, model, options, opts });
        var graph_embed_future_awaited = false;
        errdefer if (!graph_embed_future_awaited) if (graph_embed_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        // wait all parallel compilations terminate

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

        const graph_embed_exe = try graph_embed_future.await(zml_handler.io);
        graph_embed_future_awaited = true;

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
            .graph_embed_exe = graph_embed_exe,
            .graph_embed_args = try graph_embed_exe.args(zml_handler.allocator),
            .graph_embed_results = try graph_embed_exe.results(zml_handler.allocator),
        };
    }

    pub fn unloadBuffers(self: *Llm_handler, allocator: std.mem.Allocator) void {
        Llm.unloadBuffers(&self.model_buffers, allocator);
        KvCache.deinitBuffer(&self.kv_cache_buffers);
    }

    pub fn resetKvCache(self: *Llm_handler, zml_handler: *main.Zml_handler) !void {
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        self.kv_cache_buffers = try self.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, .replicated);
    }

    pub fn deinit(self: *Llm_handler, allocator: std.mem.Allocator) void {
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

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: TokenIds,
    head_dim: ?u32 = null,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    vocab_size: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .bos_token_id = self.bos_token_id,
            .eos_token_id = try dupeTokenIds(self.eos_token_id, allocator),
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
        deinitTokenIds(self.eos_token_id, allocator);
    }
};

pub const GenerationConfig = struct {
    bos_token_id: u32,
    do_sample: bool = false,
    eos_token_id: TokenIds,
    pad_token_id: u32,
    temperature: f32 = 1.0,
    top_k: u32 = 1,
    top_p: f32 = 1.0,

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
            .pad_token_id = model_config.bos_token_id,
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
        };
    }

    pub fn deinit(self: GenerationConfig, allocator: std.mem.Allocator) void {
        deinitTokenIds(self.eos_token_id, allocator);
    }

    pub fn samplingStrategy(self: GenerationConfig) zml.nn.SamplingStrategy {
        // zml.nn.SamplingStrategy currently supports top-k and temperature; top_p is parsed for config fidelity.
        if (!self.do_sample) return .{};
        return .{
            .topk = self.top_k,
            .temperature = self.temperature,
        };
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

    graph_embed_exe: zml.Exe,
    graph_embed_args: zml.Exe.Arguments,
    graph_embed_results: zml.Exe.Results,

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
        self.graph_embed_exe.deinit();
        self.graph_embed_args.deinit(allocator);
        self.graph_embed_results.deinit(allocator);
    }
};

pub const Llm = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    lm_head: zml.Tensor,
    sampling_strategy: zml.nn.SamplingStrategy,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, generation_config: GenerationConfig) !Llm {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("model.layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor("model.embed_tokens.weight", .{ .voc, .d }, .replicated) },
            .layers = layers,
            .norm = .init(store.withPrefix("model.norm"), config),
            .lm_head = store.createTensor("lm_head.weight", .{ .voc, .d }, .replicated),
            .sampling_strategy = generation_config.samplingStrategy(),
        };
    }

    pub fn load(self: *const Llm, zml_handler: *main.Zml_handler, store: *const zml.io.TensorStore) !zml.Bufferized(Llm) {
        var progress = zml_handler.progress.start("Load LLM weights", store.registry.tensors.count());
        defer progress.end();
        return zml.io.load(Llm, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = &progress,
        });
    }

    pub fn deinit(self: *const Llm, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Llm), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.lm_head.deinit();
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

    pub fn embedTokens(self: Llm, tokens: zml.Tensor) zml.Tensor {
        return self.embed_tokens.forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d});
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

    pub fn selectEmbed(_: Llm, embeddings: zml.Tensor, pred_index: zml.Tensor) zml.Tensor {
        return embeddings.dynamicSlice1d(embeddings.axis(.s), .{ .start = pred_index, .len = 1 });
    }

    pub const GraphEmbedParams = struct {
        embed: zml.Tensor,
        pub fn exec(options: Options) GraphEmbedParams {
            return .{
                .embed = .init(.{ .s = 1, .d = options.hidden_size }, .bf16),
            };
        }
    };

    pub fn graphEmbed(self: Llm, embed: zml.Tensor) zml.Tensor {
        return self.norm.forward(embed).convert(.f32);
    }

    pub const ComputeLogitsParams = struct {
        embeds: zml.Tensor,
        pub fn exec(options: Options) ComputeLogitsParams {
            return .{ .embeds = .init(.{ .s = 1, .d = options.hidden_size }, .bf16) };
        }
    };

    pub fn computeLogits(self: Llm, embed: zml.Tensor) zml.Tensor {
        const normalized_embed = self.norm.forward(embed);
        const logits = self.lm_head.withTags(.{ .voc, .d }).dot(normalized_embed, .d).convert(.f32);
        return logits.transpose(.{ .voc, .s });
    }

    pub const SampleParams = struct {
        logits: zml.Tensor,
        rng: zml.Tensor.Rng,
        pub fn exec(options: Options) SampleParams {
            return .{
                .logits = .init(.{ .voc = options.voc_size, .s = 1 }, .f32),
                .rng = zml.Tensor.Rng.init(),
            };
        }
    };

    pub fn sampleTokens(self: Llm, logits: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const next_token, const new_rng = zml.nn.sampleTokens(logits, self.sampling_strategy, rng);
        return .{ next_token.convert(.u32), new_rng };
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

    pub const TransformerParams = struct {
        x: zml.Tensor,
        index: zml.Tensor,
        kv_cache: KvCache,
        layer: zml.Tensor,
        pub fn prefill(options: Options) TransformerParams {
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
            };
        }
        pub fn decode(options: Options) TransformerParams {
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
            };
        }
    };

    pub fn forward(self: TransformerLayer, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, layer_index: zml.Tensor) struct { zml.Tensor, KvCache } {
        const x_normalized = self.input_norm.forward(x);
        const q = self.att_layer.q_proj.forward(x_normalized);
        const k = self.att_layer.k_proj.forward(x_normalized);
        const v = self.att_layer.v_proj.forward(x_normalized);

        const attn, const new_cache = self.att_layer.forward(x, token_index, kv_cache, layer_index, q, k, v);

        const delta1 = self.att_layer.o_proj.forward(attn).rename(.{ .d_out = .d });
        const x_after_attn = x.add(delta1);
        const delta2 = self.mlp_layer.forward(self.post_att_norm.forward(x_after_attn));
        const out = x_after_attn.add(delta2);
        return .{ out.reuseBuffer(x), new_cache.reuseBuffer(kv_cache) };
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
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .replicated), null, .d),
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
        var pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s});
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
            .weights = store.createTensor("weight", .{.d_out}, .replicated),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        return normalized.mul(self.weights.withTags(.{.d}).broad(input.shape()));
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

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
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
