const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const qwen35 = @import("qwen3_5.zig");
const Qwen35 = qwen35.Qwen35;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const CliArgs = struct {
    model: []const u8,
    prompt: []const u8 = "Write me a long story about a cat",
    len: i64 = 2048,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    const options = Qwen35.GenOptions{
        .max_seq_len = args.len,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };

    //======================= Tensor store init ========================

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    //======================= Platform auto-init ========================

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("\n{f}", .{platform.fmtVerbose()});

    //======================= Model & cache init ========================

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var qwen_model: Qwen35 = try .init(allocator, store.view(), config, options);
    defer qwen_model.deinit(allocator);
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const model_dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const kv_cache = qwen35.KvCache.init(
        config,
        1,
        options.max_seq_len,
        model_dtype,
        .f32,
    );
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);

    //======================= Progress tracking setup ========================

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    //======================= Loading tokenizer (async) ========================

    var tokenizer_future = try io.concurrent(loadTokenizer, .{ allocator, io, repo, &progress });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    //======================= Loading weights (async) ========================

    var qwen35_buffers_future = try io.concurrent(qwen35.Qwen35.load, .{
        &qwen_model,
        allocator,
        io,
        platform,
        &store,
        &.{replicated_sharding},
        &progress,
    });
    defer blk: {
        var v = qwen35_buffers_future.cancel(io) catch break :blk;
        Qwen35.unloadBuffers(&v, allocator);
    }

    // var tokenizer = try tokenizer_future.await(io);
    // const input_token_ids = try tokenizePrompt(allocator, tokenizer, args.prompt, qwen_model);
    // defer allocator.free(input_token_ids);
    // const prefill_len: usize = @intCast(options.max_seq_len);

    //======================= Model compilation (async) ========================

    // var compile_result_future = try io.concurrent(compileModel, .{ allocator, io, platform, qwen_model, kv_cache, &progress, prefill_len });
    // defer if (compile_result_future.cancel(io)) |v| {
    //     v.prefill_exe.deinit();
    //     v.decode_exe.deinit();
    // } else |_| {};

    //======================= Awaiting futures ========================

    // const compile_result = try compile_result_future.await(io);
    const qwen35_buffers = try qwen35_buffers_future.await(io);

    progress.end();

    //======================= Running model for prompt ========================

    log.info("\n🕹️ ZML 🕹️ running model {s} on following prompt:\n{s}\n", .{ args.model, args.prompt });

    // var stdout = std.Io.File.stdout().writer(io, &.{});
    // var tokenizer_decoder = try tokenizer.decoder();
    // defer tokenizer_decoder.deinit();

    // try runAndGenerate(allocator, io, platform, &tokenizer_decoder, qwen35_buffers, &kv_cache_buffers, compile_result, input_token_ids, &stdout.interface, qwen_model, prefill_len);

    //======================= Output check (panics if fail) ========================
    try checkLayers(allocator, io, platform, qwen_model, qwen35_buffers);
}

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, qwen35_model: Qwen35, kv_cache: qwen35.KvCache, progress: *std.Progress.Node, prefill_len: usize) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});
    log.info("Compiling model for platform {any} with prefill length {d}...", .{ platform.target, prefill_len });
    const prefill_tokens = Tensor.init(.{ .b = 1, .s = prefill_len }, .u32);
    const decode_tokens = Tensor.init(.{ .b = 1, .s = 1 }, .u32);
    const token_index = Tensor.init(.{}, .u32);

    const rng = zml.Tensor.Rng.init();

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen35_model_: Qwen35,
            prefill_tokens_: Tensor,
            token_index_: Tensor,
            kv_cache_: qwen35.KvCache,
            rng_: zml.Tensor.Rng,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, qwen35_model_, .forward, .{ prefill_tokens_, token_index_, kv_cache_, rng_ }, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model, prefill_tokens, token_index, kv_cache, rng, replicated_sharding, progress });
    errdefer if (prefill_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen35_model_: Qwen35,
            decode_tokens_: Tensor,
            token_index_: Tensor,
            kv_cache_: qwen35.KvCache,
            rng_: zml.Tensor.Rng,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, qwen35_model_, .forward, .{ decode_tokens_, token_index_, kv_cache_, rng_ }, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model, decode_tokens, token_index, kv_cache, rng, replicated_sharding, progress });
    errdefer if (decode_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

//======================= Generation setup ========================

fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, qwen_model: Qwen35) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const encoded_prompt = try encoder.encode(prompt);
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, encoded_prompt.len + 1);

    try tokens.append(allocator, qwen_model.special_tokens.im_start_token_id);
    try tokens.appendSlice(allocator, encoded_prompt);
    try tokens.append(allocator, qwen_model.special_tokens.im_end_token_id);

    return tokens.toOwnedSlice(allocator);
}

fn runAndGenerate(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, tokenizer_decoder: *zml.tokenizer.Tokenizer.Decoder, qwen35_buffers: zml.Bufferized(Qwen35), kv_cache_buffers: *zml.Bufferized(qwen35.KvCache), compile_result: CompileModelResult, input_token_ids: []u32, writer: *std.Io.Writer, qwen_model: Qwen35, prefill_len: usize) !void {
    const max_seq_len: usize = @intCast(qwen_model.gen_options.max_seq_len);
    if (input_token_ids.len > max_seq_len) return error.PromptTooLong;
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = prefill_len }, .u32);
    const decode_tokens_shape = zml.Shape.init(.{ .b = 1, .s = 1 }, .u32);

    const seed: u128 = 0;
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    var generated_token_slice: zml.Slice = try .alloc(allocator, decode_tokens_shape);
    defer generated_token_slice.free(allocator);

    {
        var prefill_args = try compile_result.prefill_exe.args(allocator);
        defer prefill_args.deinit(allocator);

        var prefill_results = try compile_result.prefill_exe.results(allocator);
        defer prefill_results.deinit(allocator);

        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..input_token_ids.len], input_token_ids);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();
        var prefill_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        prefill_args.set(.{ qwen35_buffers, prefill_tokens_buffer, prefill_token_index_buffer, kv_cache_buffers, rng_buffers });
        compile_result.prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers, &rng_buffers });
        try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[input_token_ids.len - 1];
        generated_token_slice.items(u32)[0] = generated_token;
    }

    const now: std.Io.Timestamp = .now(io, .awake);
    const output_tokens_len = max_seq_len - input_token_ids.len - 1;
    var num_tokens_generated: usize = 1;

    var decode_args = try compile_result.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);

    var decode_results = try compile_result.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, replicated_sharding);
    defer current_token_buffer.deinit();

    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.*.next(generated_token)) |chunk| {
            // if (i % 50 == 0) {
            //     var buf: [32]u8 = undefined;
            //     try writer.writeAll(try std.fmt.bufPrint(&buf, "\x1b[31m{d}\x1b[0m", .{i}));
            // }
            try writer.writeAll(chunk);
            try writer.flush();
        }

        if (i == output_tokens_len) break :generation;
        if (generated_token == qwen_model.special_tokens.end_of_text_token_id) break :generation;

        var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(input_token_ids.len + i)), .u32, replicated_sharding);
        defer token_index_buffer.deinit();

        decode_args.set(.{ qwen35_buffers, current_token_buffer, token_index_buffer, kv_cache_buffers, rng_buffers });

        compile_result.decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }

    const duration = now.untilNow(io, .awake);
    std.debug.print("\n", .{});
    log.info("Generated {} tokens in {f}: {:.3} tok/s", .{
        num_tokens_generated,
        duration,
        stdx.Io.Duration.hzFloat(stdx.Io.Duration.div(duration, num_tokens_generated)),
    });
}

//======================= Load and parse setup ========================

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(qwen35.Qwen35.Config) {
    const now: std.Io.Timestamp = .now(io, .awake);
    log.info("Loading model config", .{});
    defer log.info("Loaded model config [{f}]", .{now.untilNow(io, .awake)});

    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(qwen35.Qwen35.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();

    return parsed_config;
}

//======================= Activation check setup ========================

fn checkLayers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    qwen_model: Qwen35,
    qwen35_buffers: zml.Bufferized(Qwen35),
) !void {
    var activations_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, "/home/tristan/zml/examples/qwen3_5/safetensors/Qwen3.5-0.8B.activations-bf16-with-caches.safetensors");
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    const comp_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .minimum_close_fraction = 0.999,
    };
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    const shardings = &.{replicated_sharding};

    if (activations_store.view().getShape("model.model.embed_tokens.in.0") == null) {
        try checkMixedPrefillCaches(allocator, io, platform, qwen_model, qwen35_buffers, activations_store.view(), replicated_sharding, comp_opts);
        return;
    }

    // Testing embed_tokens

    const EmbedTokensHarness = struct {
        embedTokens: zml.nn.TokenEmbedding,
        pub fn forward(self: @This(), inputIds: Tensor) Tensor {
            return self.embedTokens.weight
                .gather(.{ .voc = inputIds.withTags(.{ .b, .s }) }, .{});
        }
    };
    const embedTokensHarness: EmbedTokensHarness = .{
        .embedTokens = qwen_model.text_model.embed_tokens,
    };
    const embedTokensHarnessBuffers: zml.Bufferized(EmbedTokensHarness) = .{
        .embedTokens = qwen35_buffers.text_model.embed_tokens,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        embedTokensHarness,
        .forward,
        activations_store.view(),
        "model.model.embed_tokens",
        embedTokensHarnessBuffers,
        shardings,
        comp_opts,
    );

    // Testing each layer

    const LayerHarness = struct {
        layer: qwen35.TransformerLayer,
        pub fn forward(
            self: @This(),
            x: Tensor,
            cos: Tensor,
            sin: Tensor,
            position_ids: Tensor,
            cache_position: Tensor,
        ) struct { Tensor, Tensor, Tensor } {
            _ = cos;
            _ = sin;
            _ = position_ids;
            const token_index = cachePositionToTokenIndex(cache_position);
            const tagged_x = x.withTags(.{ .b, .s, .d });
            const kv_cache = emptyLayerKvCache(self.layer, tagged_x);
            const output, const updated_cache = self.layer.forward(x.withTags(.{ .b, .s, .d }), token_index, kv_cache.atLayer(0));
            return switch (self.layer.attn) {
                .self_attn => .{
                    output,
                    flatKeyCache(updated_cache.self_attn),
                    flatValueCache(updated_cache.self_attn),
                },
                .linear_attn => .{
                    output,
                    flatConvState(updated_cache.gated_delta_net),
                    flatRecurrentState(updated_cache.gated_delta_net),
                },
            };
        }
    };

    const InputLayerNormHarness = struct {
        inputLayerNorm: qwen35.RmsNorm,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.inputLayerNorm.forward(x.withTags(.{ .b, .s, .d }));
        }
    };

    const LinearAttnHarness = struct {
        linearAttn: qwen35.GatedDeltaNet,
        pub fn forward(
            self: @This(),
            x: Tensor,
            cache_position: Tensor,
        ) struct { Tensor, Tensor, Tensor } {
            _ = cache_position;
            const cache = emptyLinearAttnCache(self.linearAttn, x.withTags(.{ .b, .s, .d }));
            const output, const updated_cache = self.linearAttn.forward(x.withTags(.{ .b, .s, .d }), cache);
            return .{
                output,
                flatConvState(updated_cache),
                flatRecurrentState(updated_cache),
            };
        }
    };

    const PostAttentionLayerNormHarness = struct {
        postAttentionLayerNorm: qwen35.RmsNorm,

        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.postAttentionLayerNorm.forward(x.withTags(.{ .b, .s, .d }));
        }
    };

    const MlpHarness = struct {
        mlp: qwen35.Mlp,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.mlp.forward(x.withTags(.{ .b, .s, .d }));
        }
    };

    const SelfAttnHarness = struct {
        selfAttn: qwen35.SelfAttn,
        pub fn forward(
            self: @This(),
            x: Tensor,
            position_ids: Tensor,
            cache_position: Tensor,
            cos: Tensor,
            sin: Tensor,
        ) struct { Tensor, Tensor, Tensor } {
            _ = position_ids;
            _ = cos;
            _ = sin;
            const token_index = cachePositionToTokenIndex(cache_position);
            const cache = emptySelfAttnCache(self.selfAttn, x.withTags(.{ .b, .s, .d }));
            const output, const updated_cache = self.selfAttn.forward(x.withTags(.{ .b, .s, .d }), token_index, cache);
            return .{
                output,
                flatKeyCache(updated_cache),
                flatValueCache(updated_cache),
            };
        }
    };

    const layers_to_test = .{ 0, 3 };
    for (qwen_model.text_model.layers, qwen35_buffers.text_model.layers, 0..) |layer, layer_buffers, layer_index| {
        const should_test = inline for (layers_to_test) |test_idx| {
            if (layer_index == test_idx) break true;
        } else false;
        if (!should_test) {
            continue;
        }
        var name_buf: [128]u8 = undefined;

        const layer_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}", .{layer_index});
        const layerHarness: LayerHarness = .{
            .layer = layer,
        };
        const layerHarnessBuffers: zml.Bufferized(LayerHarness) = .{
            .layer = layer_buffers,
        };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            layerHarness,
            .forward,
            activations_store.view(),
            layer_name,
            layerHarnessBuffers,
            shardings,
            comp_opts,
        );

        // Testing input_layernorm

        const input_layernorm_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.input_layernorm", .{layer_index});
        const inputLayerNormHarness: InputLayerNormHarness = .{ .inputLayerNorm = layer.input_layernorm };
        const inputLayerNormHarnessBuffers: zml.Bufferized(InputLayerNormHarness) = .{ .inputLayerNorm = layer_buffers.input_layernorm };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            inputLayerNormHarness,
            .forward,
            activations_store.view(),
            input_layernorm_name,
            inputLayerNormHarnessBuffers,
            shardings,
            comp_opts,
        );

        // Testing attention (full or linear)

        switch (layer.attn) {
            .self_attn => |self_attn| {
                const attn_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.self_attn", .{layer_index});
                const selfAttnHarness: SelfAttnHarness = .{
                    .selfAttn = self_attn,
                };
                const selfAttnHarnessBuffers: zml.Bufferized(SelfAttnHarness) = .{
                    .selfAttn = switch (layer_buffers.attn) {
                        .self_attn => |buffered_self_attn| buffered_self_attn,
                        else => unreachable,
                    },
                };
                try zml.testing.testLayer(
                    allocator,
                    io,
                    platform,
                    selfAttnHarness,
                    .forward,
                    activations_store.view(),
                    attn_name,
                    selfAttnHarnessBuffers,
                    shardings,
                    comp_opts,
                );
            },
            .linear_attn => |linear_attn| {
                const attn_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn", .{layer_index});
                const linearAttnHarness: LinearAttnHarness = .{
                    .linearAttn = linear_attn,
                };
                const linearAttnHarnessBuffers: zml.Bufferized(LinearAttnHarness) = .{
                    .linearAttn = switch (layer_buffers.attn) {
                        .linear_attn => |buffered_linear_attn| buffered_linear_attn,
                        else => unreachable,
                    },
                };
                try zml.testing.testLayer(
                    allocator,
                    io,
                    platform,
                    linearAttnHarness,
                    .forward,
                    activations_store.view(),
                    attn_name,
                    linearAttnHarnessBuffers,
                    shardings,
                    comp_opts,
                );
            },
        }

        // Testing post_attention_layernorm

        const post_attention_layernorm_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.post_attention_layernorm", .{layer_index});
        const postAttentionLayerNormHarness: PostAttentionLayerNormHarness = .{ .postAttentionLayerNorm = layer.post_attention_layernorm };
        const postAttentionLayerNormHarnessBuffers: zml.Bufferized(PostAttentionLayerNormHarness) = .{
            .postAttentionLayerNorm = layer_buffers.post_attention_layernorm,
        };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            postAttentionLayerNormHarness,
            .forward,
            activations_store.view(),
            post_attention_layernorm_name,
            postAttentionLayerNormHarnessBuffers,
            shardings,
            comp_opts,
        );

        const mlp_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp", .{layer_index});
        const mlpHarness: MlpHarness = .{ .mlp = layer.mlp };
        const mlpHarnessBuffers: zml.Bufferized(MlpHarness) = .{ .mlp = layer_buffers.mlp };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            mlpHarness,
            .forward,
            activations_store.view(),
            mlp_name,
            mlpHarnessBuffers,
            shardings,
            comp_opts,
        );
    }

    // Testing text_model.norm

    const ModelNormHarness = struct {
        norm: qwen35.RmsNorm,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.norm.forward(x.withTags(.{ .b, .s, .d }));
        }
    };
    const modelNormHarness: ModelNormHarness = .{
        .norm = qwen_model.text_model.norm,
    };
    const modelNormHarnessBuffers: zml.Bufferized(ModelNormHarness) = .{
        .norm = qwen35_buffers.text_model.norm,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        modelNormHarness,
        .forward,
        activations_store.view(),
        "model.model.norm",
        modelNormHarnessBuffers,
        shardings,
        comp_opts,
    );

    // Testing full model

    const ModelHarness = struct {
        model: qwen35.TextModel,
        config: Qwen35.Config,
        cache: qwen35.KvCache,
        pub fn forward(
            self: @This(),
            tokens: Tensor,
            cos: Tensor,
            sin: Tensor,
            position_ids: Tensor,
        ) struct { Tensor } {
            _ = cos;
            _ = sin;
            _ = position_ids;
            const tagged_tokens = tokens.withTags(.{ .b, .s });
            return .{self.model.forward(tagged_tokens, Tensor.scalar(@as(u32, 0), .u32), self.cache)[0]};
        }
    };
    const modelHarness: ModelHarness = .{
        .model = qwen_model.text_model,
        .config = qwen_model.config,
        .cache = initKvCache(qwen_model.text_model.embed_tokens.weight.dtype(), qwen_model.config, qwen_model.gen_options),
    };
    var modelHarnessCacheBuffers = try modelHarness.cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&modelHarnessCacheBuffers);
    try zeroKvCacheBuffers(allocator, io, platform, replicated_sharding, &modelHarnessCacheBuffers);
    const modelHarnessBuffers: zml.Bufferized(ModelHarness) = .{
        .model = qwen35_buffers.text_model,
        .cache = modelHarnessCacheBuffers,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        modelHarness,
        .forward,
        activations_store.view(),
        "model.model",
        modelHarnessBuffers,
        shardings,
        comp_opts,
    );

    try checkMixedPrefillCaches(allocator, io, platform, qwen_model, qwen35_buffers, activations_store.view(), replicated_sharding, comp_opts);
}

fn checkMixedPrefillCaches(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    qwen_model: Qwen35,
    qwen35_buffers: zml.Bufferized(Qwen35),
    activations_store: zml.io.TensorStore.View,
    replicated_sharding: zml.sharding.Sharding,
    comp_opts: zml.testing.CompareOpts,
) !void {
    const prompt_tokens_shape = activations_store.getShape("model.model.in.0") orelse return error.NotFound;
    stdx.debug.assert(prompt_tokens_shape.rank() == 2, "Expected model.model.in.0 to be rank-2, got {f}", .{prompt_tokens_shape});

    const actual_seq_len: i64 = prompt_tokens_shape.dims()[1];
    const prefill_len: i64 = qwen_model.gen_options.max_seq_len;
    stdx.debug.assert(prefill_len > actual_seq_len, "Mixed prefill cache check needs max_seq_len > prompt len, got max_seq_len={} prompt_len={}", .{ prefill_len, actual_seq_len });

    const prompt_tokens = try loadSliceFromStore(allocator, io, activations_store, "model.model.in.0");
    defer prompt_tokens.free(allocator);

    var padded_tokens = try zml.Slice.alloc(allocator, zml.Shape.init(.{ 1, prefill_len }, prompt_tokens.shape.dtype()));
    defer padded_tokens.free(allocator);
    @memset(padded_tokens.data(), 0);
    @memcpy(padded_tokens.data()[0..prompt_tokens.data().len], prompt_tokens.data());

    var padded_tokens_buffer = try zml.Buffer.fromSlice(io, platform, padded_tokens, replicated_sharding);
    defer padded_tokens_buffer.deinit();
    var seq_len_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(actual_seq_len)), .u32, replicated_sharding);
    defer seq_len_buffer.deinit();

    const mixed_prefill_cache = initKvCache(qwen_model.text_model.embed_tokens.weight.dtype(), qwen_model.config, qwen_model.gen_options);
    var cache_buffers = try mixed_prefill_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&cache_buffers);
    try zeroKvCacheBuffers(allocator, io, platform, replicated_sharding, &cache_buffers);

    const MixedPrefillCacheHarness = struct {
        model: qwen35.TextModel,
        actual_seq_len: i64,

        pub fn forward(self: @This(), tokens: Tensor, seq_len: Tensor, kv_cache: qwen35.KvCache) struct { Tensor, Tensor, Tensor, Tensor } {
            _ = seq_len; // autofix
            const forward_result = self.model.forward(tokens.withTags(.{ .b, .s }), Tensor.scalar(@as(u32, 0), .u32), kv_cache);
            const updated_cache = forward_result[1];
            return .{
                updated_cache.self_attn.k.slice1d(.s, .{ .start = 0, .end = self.actual_seq_len }),
                updated_cache.self_attn.v.slice1d(.s, .{ .start = 0, .end = self.actual_seq_len }),
                updated_cache.gated_delta_net.conv_state,
                updated_cache.gated_delta_net.recurrent_state,
            };
        }
    };

    const harness: MixedPrefillCacheHarness = .{
        .model = qwen_model.text_model,
        .actual_seq_len = actual_seq_len,
    };
    const harness_buffers: zml.Bufferized(MixedPrefillCacheHarness) = .{
        .model = qwen35_buffers.text_model,
    };
    const padded_tokens_tensor = zml.Tensor.fromShape(padded_tokens.shape).withTags(.{ .b, .s });
    const seq_len_tensor = zml.Tensor.init(.{}, .u32);
    const exe = try platform.compile(
        allocator,
        io,
        harness,
        .forward,
        .{ padded_tokens_tensor, seq_len_tensor, mixed_prefill_cache },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();

    var actual_key_cache, var actual_value_cache, var actual_conv_state, var actual_recurrent_state = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        MixedPrefillCacheHarness.forward,
        .{ harness_buffers, padded_tokens_buffer, seq_len_buffer, cache_buffers },
    );
    defer actual_key_cache.deinit();
    defer actual_value_cache.deinit();
    defer actual_conv_state.deinit();
    defer actual_recurrent_state.deinit();
    var actual_recurrent_state_bf16 = try convertBufferF32ToBf16(allocator, io, platform, replicated_sharding, actual_recurrent_state);
    defer actual_recurrent_state_bf16.deinit();

    var expected_key_cache = try loadBufferFromStore(allocator, io, platform, activations_store, "model.model.cache_out.self_attn.k", replicated_sharding);
    defer expected_key_cache.deinit();
    var expected_value_cache = try loadBufferFromStore(allocator, io, platform, activations_store, "model.model.cache_out.self_attn.v", replicated_sharding);
    defer expected_value_cache.deinit();
    var expected_conv_state = try loadBufferFromStore(allocator, io, platform, activations_store, "model.model.cache_out.gated_delta_net.conv_state", replicated_sharding);
    defer expected_conv_state.deinit();
    var expected_recurrent_state = try loadBufferFromStore(allocator, io, platform, activations_store, "model.model.cache_out.gated_delta_net.recurrent_state", replicated_sharding);
    defer expected_recurrent_state.deinit();

    var failed = false;

    log.info("Comparing self_attn.k: actual={f} expected={f}", .{ actual_key_cache.shape(), expected_key_cache.shape() });
    zml.testing.expectClose(io, actual_key_cache, expected_key_cache, comp_opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            log.err("self_attn.k mismatch", .{});
            failed = true;
        },
        else => return err,
    };

    log.info("Comparing self_attn.v: actual={f} expected={f}", .{ actual_value_cache.shape(), expected_value_cache.shape() });
    zml.testing.expectClose(io, actual_value_cache, expected_value_cache, comp_opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            log.err("self_attn.v mismatch", .{});
            failed = true;
        },
        else => return err,
    };

    log.info("Comparing gated_delta_net.conv_state: actual={f} expected={f}", .{ actual_conv_state.shape(), expected_conv_state.shape() });
    zml.testing.expectClose(io, actual_conv_state, expected_conv_state, comp_opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            log.err("gated_delta_net.conv_state mismatch", .{});
            failed = true;
        },
        else => return err,
    };

    log.info("Comparing gated_delta_net.recurrent_state: actual={f} expected={f}", .{ actual_recurrent_state_bf16.shape(), expected_recurrent_state.shape() });
    zml.testing.expectClose(io, actual_recurrent_state_bf16, expected_recurrent_state, comp_opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            log.err("gated_delta_net.recurrent_state mismatch", .{});
            failed = true;
        },
        else => return err,
    };

    if (failed) {
        return error.TestUnexpectedResult;
    }
    log.info("✅ mixed prefill caches match reference for runtime seq_len={} with compile-time max_seq_len={}", .{ actual_seq_len, prefill_len });
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, buffer: *zml.Buffer) !void {
    const shape = buffer.shape();
    const zero_slice: zml.Slice = try .alloc(allocator, shape);
    defer zero_slice.free(allocator);
    @memset(zero_slice.data(), 0);

    var zeroed = try zml.Buffer.fromSlice(io, platform, zero_slice, sharding);
    errdefer zeroed.deinit();
    buffer.deinit();
    buffer.* = zeroed;
}

fn loadSliceFromStore(allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, key: []const u8) !zml.Slice {
    const shape = store.getShape(key) orelse return error.NotFound;
    const slice = try zml.Slice.alloc(allocator, shape);
    errdefer slice.free(allocator);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(slice.data());
    return slice;
}

fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const slice = try loadSliceFromStore(allocator, io, store, key);
    defer slice.free(allocator);
    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}

fn convertBufferF32ToBf16(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    buffer: zml.Buffer,
) !zml.Buffer {
    const input_slice = try buffer.toSliceAlloc(allocator, io);
    defer input_slice.free(allocator);

    stdx.debug.assert(input_slice.dtype() == .f32, "Expected f32 buffer, got {}", .{input_slice.dtype()});

    const bf16_shape = zml.Shape.init(input_slice.shape.dims(), .bf16);
    var bf16_slice = try zml.Slice.alloc(allocator, bf16_shape);
    errdefer bf16_slice.free(allocator);

    const src = input_slice.constItems(f32);
    const dst = bf16_slice.items(zml.floats.BFloat16);
    for (src, dst) |src_val, *dst_val| {
        dst_val.* = zml.floats.BFloat16.fromF32(src_val);
    }

    const bf16_buffer = try zml.Buffer.fromSlice(io, platform, bf16_slice, sharding);
    bf16_slice.free(allocator);
    return bf16_buffer;
}

fn zeroKvCacheBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    cache: *zml.Bufferized(qwen35.KvCache),
) !void {
    try zeroBuffer(allocator, io, platform, sharding, &cache.self_attn.k);
    try zeroBuffer(allocator, io, platform, sharding, &cache.self_attn.v);
    try zeroBuffer(allocator, io, platform, sharding, &cache.gated_delta_net.conv_state);
    try zeroBuffer(allocator, io, platform, sharding, &cache.gated_delta_net.recurrent_state);
}

fn initLinearAttnCache(dtype: zml.DataType, config: Qwen35.Config, options: Qwen35.GenOptions) qwen35.KvCache.GatedDeltaNetCache {
    return qwen35.KvCache.init(
        config,
        1,
        options.max_seq_len,
        dtype,
        .f32,
    ).gated_delta_net;
}

fn initSelfAttnCache(dtype: zml.DataType, config: Qwen35.Config, options: Qwen35.GenOptions) qwen35.KvCache.SelfAttnCache {
    return qwen35.KvCache.init(
        config,
        1,
        options.max_seq_len,
        dtype,
        .f32,
    ).self_attn;
}

fn initKvCache(dtype: zml.DataType, config: Qwen35.Config, options: Qwen35.GenOptions) qwen35.KvCache {
    return qwen35.KvCache.init(
        config,
        1,
        options.max_seq_len,
        dtype,
        .f32,
    );
}

fn zeroTensor(shape: zml.Shape) Tensor {
    return Tensor.constant(shape.dtype().zero()).broad(shape);
}

fn cachePositionToTokenIndex(cache_position: Tensor) Tensor {
    return cache_position.withTags(.{.s}).slice1d(.s, .{ .start = 0, .end = 1 }).squeeze(.s).convert(.u32);
}

fn emptySelfAttnCache(self_attn: qwen35.SelfAttn, x: Tensor) qwen35.KvCache.SelfAttnCache {
    const cache_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .s = x.dim(.s),
        .h = self_attn.num_kv_heads,
        .hd = self_attn.head_dim,
    }, x.dtype());

    return .{
        .k = zeroTensor(cache_shape),
        .v = zeroTensor(cache_shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

fn emptyLinearAttnCache(linear_attn: qwen35.GatedDeltaNet, x: Tensor) qwen35.KvCache.GatedDeltaNetCache {
    const conv_dim = 2 * linear_attn.num_k_heads * linear_attn.head_k_dim + linear_attn.num_v_heads * linear_attn.head_v_dim;
    const left_pad = linear_attn.conv_kernel_size - 1;
    const conv_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .s = left_pad,
        .mix = conv_dim,
    }, x.dtype());
    const recurrent_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .vh = linear_attn.num_v_heads,
        .khd = linear_attn.head_k_dim,
        .vhd = linear_attn.head_v_dim,
    }, .f32);

    return .{
        .conv_state = zeroTensor(conv_shape),
        .recurrent_state = zeroTensor(recurrent_shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

fn dummySelfAttnCache(dtype: zml.DataType, batch_dim: i64, seq_len: i64) qwen35.KvCache.SelfAttnCache {
    const shape = zml.Shape.init(.{ .b = batch_dim, .layer = 1, .s = seq_len, .h = 1, .hd = 1 }, dtype);
    return .{
        .k = zeroTensor(shape),
        .v = zeroTensor(shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

fn dummyLinearAttnCache(conv_dtype: zml.DataType, batch_dim: i64) qwen35.KvCache.GatedDeltaNetCache {
    return .{
        .conv_state = zeroTensor(zml.Shape.init(.{ .b = batch_dim, .layer = 1, .s = 1, .mix = 1 }, conv_dtype)),
        .recurrent_state = zeroTensor(zml.Shape.init(.{ .b = batch_dim, .layer = 1, .vh = 1, .khd = 1, .vhd = 1 }, .f32)),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

const single_full_layer_types = [_]qwen35.Qwen35.LayerType{.full_attention};
const single_linear_layer_types = [_]qwen35.Qwen35.LayerType{.linear_attention};

fn emptyLayerKvCache(layer: qwen35.TransformerLayer, x: Tensor) qwen35.KvCache {
    return .{
        .layer_types = switch (layer.attn) {
            .self_attn => single_full_layer_types[0..],
            .linear_attn => single_linear_layer_types[0..],
        },
        .self_attn = switch (layer.attn) {
            .self_attn => |self_attn| emptySelfAttnCache(self_attn, x),
            .linear_attn => dummySelfAttnCache(x.dtype(), x.dim(.b), x.dim(.s)),
        },
        .gated_delta_net = switch (layer.attn) {
            .linear_attn => |linear_attn| emptyLinearAttnCache(linear_attn, x),
            .self_attn => dummyLinearAttnCache(x.dtype(), x.dim(.b)),
        },
    };
}

fn flatKeyCache(cache: qwen35.KvCache.SelfAttnCache) Tensor {
    return cache.keys().transpose(.{ .b, .h, .s, .hd });
}

fn flatValueCache(cache: qwen35.KvCache.SelfAttnCache) Tensor {
    return cache.values().transpose(.{ .b, .h, .s, .hd });
}

fn flatConvState(cache: qwen35.KvCache.GatedDeltaNetCache) Tensor {
    return cache.convState().transpose(.{ .b, .mix, .s });
}

fn flatRecurrentState(cache: qwen35.KvCache.GatedDeltaNetCache) Tensor {
    return cache.recurrentState();
}
