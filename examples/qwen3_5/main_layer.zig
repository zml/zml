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
    prompt: []const u8 = "What is the capital of France ?",
    len: i64 = 1024,
    moe_backend: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    const options = Qwen35.GenOptions{
        .max_seq_len = args.len,
        .sampling_strategy = .{
            .topk = 3,
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
    try zml.tracer.register(platform);
    log.info("\n{f}", .{platform.fmtVerbose()});

    //======================= Model & cache init ========================

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;
    const prefill_len: usize = @intCast(options.max_seq_len);

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

    //======================= Moe backend init ========================

    const dtype = qwen_model.text_model.embed_tokens.weight.dtype();

    const moe_backend: zml.moe.Backend = if (args.moe_backend) |name| b: {
        if (std.mem.eql(u8, name, "flashinfer")) return error.FlashInferMoeBackendUnsupported;
        if (std.mem.eql(u8, name, "triton")) break :b .triton;
        log.err("Unknown MoE backend: {s}", .{name});
        return;
    } else try zml.moe.Backend.auto(platform, dtype);

    moe_backend.load(allocator) catch |err| {
        log.err("Failed to load MoE backend: {}", .{err});
        return err;
    };

    moe_backend.register(platform) catch |err| {
        log.err("Failed to register MoE backend: {}", .{err});
        return err;
    };

    const batch_size: u32 = 1;
    const prefill_moe_metadata = try initMoeMetadata(qwen_model, moe_backend, prefill_len, batch_size);
    const decode_moe_metadata = try initMoeMetadata(qwen_model, moe_backend, 1, batch_size);
    const moe_parameters: zml.moe.Parameters = .init(.fromBackend(moe_backend));
    _ = moe_parameters; // autofix
    var prefill_moe_metadata_buffers = try prefill_moe_metadata.initBuffer(io, platform);
    defer zml.moe.Metadata.deinitBuffer(&prefill_moe_metadata_buffers);
    var decode_moe_metadata_buffers = try decode_moe_metadata.initBuffer(io, platform);
    defer zml.moe.Metadata.deinitBuffer(&decode_moe_metadata_buffers);

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

    var tokenizer = try tokenizer_future.await(io);
    const input_token_ids = try tokenizePrompt(allocator, tokenizer, args.prompt, qwen_model);

    defer allocator.free(input_token_ids);

    //======================= Model compilation (async) ========================

    var compile_result_future = try io.concurrent(compileModel, .{ allocator, io, platform, qwen_model, kv_cache, prefill_moe_metadata, decode_moe_metadata, &progress, prefill_len });
    defer if (compile_result_future.cancel(io)) |v| {
        v.prefill_embedding_exe.deinit();
        v.decode_embedding_exe.deinit();
        if (v.prefill_full_layer_exe) |exe| exe.deinit();
        if (v.decode_full_layer_exe) |exe| exe.deinit();
        if (v.prefill_linear_layer_exe) |exe| exe.deinit();
        if (v.decode_linear_layer_exe) |exe| exe.deinit();
        v.prefill_sampling_exe.deinit();
        v.decode_sampling_exe.deinit();
    } else |_| {};

    //======================= Awaiting futures ========================

    const compile_result = try compile_result_future.await(io);
    const qwen35_buffers = try qwen35_buffers_future.await(io);

    progress.end();

    //======================= Running model for prompt ========================

    log.info("\n🕹️ ZML 🕹️ running model {s} on following prompt:\n{s}\n", .{ args.model, args.prompt });

    var stdout = std.Io.File.stdout().writer(io, &.{});
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    try runAndGenerate(allocator, io, platform, &tokenizer_decoder, qwen35_buffers, &kv_cache_buffers, &prefill_moe_metadata_buffers, &decode_moe_metadata_buffers, compile_result, input_token_ids, &stdout.interface, qwen_model, prefill_len);

    //======================= Output check (panics if fail) ========================
}

const CompileModelResult = struct {
    prefill_embedding_exe: zml.Exe,
    decode_embedding_exe: zml.Exe,
    prefill_full_layer_exe: ?zml.Exe,
    decode_full_layer_exe: ?zml.Exe,
    prefill_linear_layer_exe: ?zml.Exe,
    decode_linear_layer_exe: ?zml.Exe,
    prefill_sampling_exe: zml.Exe,
    decode_sampling_exe: zml.Exe,
};

fn findFirstLayerIndex(layer_types: []const Qwen35.LayerType, target: Qwen35.LayerType) ?usize {
    for (layer_types, 0..) |layer_type, index| {
        if (layer_type == target) return index;
    }
    return null;
}

fn compileSelfAttnLayerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: qwen35.TransformerLayer,
    hidden: Tensor,
    token_index: Tensor,
    cache: qwen35.KvCache.SelfAttnCache,
    config: Qwen35.Config,
    moe_metadata: zml.moe.Metadata,
    sharding: zml.sharding.Sharding,
    progress: *std.Progress.Node,
    label: []const u8,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(label, 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });

    return platform.compile(allocator, io, model, .forwardSelfAttnWithMoeMetadata, .{ hidden, token_index, cache, config, moe_metadata }, .{ .shardings = &.{sharding} });
}

fn compileLinearAttnLayerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: qwen35.TransformerLayer,
    hidden: Tensor,
    token_index: Tensor,
    cache: qwen35.KvCache.GatedDeltaNetCache,
    config: Qwen35.Config,
    moe_metadata: zml.moe.Metadata,
    sharding: zml.sharding.Sharding,
    progress: *std.Progress.Node,
    label: []const u8,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(label, 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });

    return platform.compile(allocator, io, model, .forwardLinearAttnWithMoeMetadata, .{ hidden, token_index, cache, config, moe_metadata }, .{ .shardings = &.{sharding} });
}

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, qwen35_model: Qwen35, kv_cache: qwen35.KvCache, prefill_moe_metadata: zml.moe.Metadata, decode_moe_metadata: zml.moe.Metadata, progress: *std.Progress.Node, prefill_len: usize) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});
    log.info("Compiling model for platform {any} with prefill length {d}...", .{ platform.target, prefill_len });

    const prefill_tokens = Tensor.init(.{ .b = 1, .s = prefill_len }, .u32);
    const decode_tokens = Tensor.init(.{ .b = 1, .s = 1 }, .u32);
    const prefill_hidden = Tensor.init(.{ .b = 1, .s = prefill_len, .d = qwen35_model.config.text_config.hidden_size }, qwen35_model.text_model.embed_tokens.weight.dtype());
    const decode_hidden = Tensor.init(.{ .b = 1, .s = 1, .d = qwen35_model.config.text_config.hidden_size }, qwen35_model.text_model.embed_tokens.weight.dtype());
    const token_index = Tensor.init(.{}, .u32);
    const self_attn_cache: qwen35.KvCache.SelfAttnCache = .{
        .k = kv_cache.self_attn.k,
        .v = kv_cache.self_attn.v,
        .layer_index = Tensor.init(.{}, .u32),
    };
    const linear_attn_cache: qwen35.KvCache.GatedDeltaNetCache = .{
        .conv_state = kv_cache.gated_delta_net.conv_state,
        .recurrent_state = kv_cache.gated_delta_net.recurrent_state,
        .layer_index = Tensor.init(.{}, .u32),
    };

    const rng = zml.Tensor.Rng.init();

    var prefill_embedding_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: zml.nn.TokenEmbedding,
            prefill_tokens_: Tensor,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .forward, .{prefill_tokens_}, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.embed_tokens, prefill_tokens, replicated_sharding, progress });
    errdefer if (prefill_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var decode_embedding_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: zml.nn.TokenEmbedding,
            decode_tokens_: Tensor,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .forward, .{decode_tokens_}, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.embed_tokens, decode_tokens, replicated_sharding, progress });
    errdefer if (decode_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var prefill_full_layer_exe: ?zml.Exe = null;
    errdefer if (prefill_full_layer_exe) |exe| exe.deinit();
    var decode_full_layer_exe: ?zml.Exe = null;
    errdefer if (decode_full_layer_exe) |exe| exe.deinit();
    var prefill_linear_layer_exe: ?zml.Exe = null;
    errdefer if (prefill_linear_layer_exe) |exe| exe.deinit();
    var decode_linear_layer_exe: ?zml.Exe = null;
    errdefer if (decode_linear_layer_exe) |exe| exe.deinit();

    if (findFirstLayerIndex(qwen35_model.config.text_config.layer_types, .full_attention)) |layer_index| {
        const layer_model = qwen35_model.text_model.layers[layer_index];
        prefill_full_layer_exe = try compileSelfAttnLayerExe(allocator, io, platform, layer_model, prefill_hidden, token_index, self_attn_cache, qwen35_model.config, prefill_moe_metadata, replicated_sharding, progress, "Compiling prefill full-attention layer...");
        decode_full_layer_exe = try compileSelfAttnLayerExe(allocator, io, platform, layer_model, decode_hidden, token_index, self_attn_cache, qwen35_model.config, decode_moe_metadata, replicated_sharding, progress, "Compiling decode full-attention layer...");
    }

    if (findFirstLayerIndex(qwen35_model.config.text_config.layer_types, .linear_attention)) |layer_index| {
        const layer_model = qwen35_model.text_model.layers[layer_index];
        prefill_linear_layer_exe = try compileLinearAttnLayerExe(allocator, io, platform, layer_model, prefill_hidden, token_index, linear_attn_cache, qwen35_model.config, prefill_moe_metadata, replicated_sharding, progress, "Compiling prefill linear-attention layer...");
        decode_linear_layer_exe = try compileLinearAttnLayerExe(allocator, io, platform, layer_model, decode_hidden, token_index, linear_attn_cache, qwen35_model.config, decode_moe_metadata, replicated_sharding, progress, "Compiling decode linear-attention layer...");
    }

    var prefill_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: qwen35.TextModel,
            prefill_hidden_: Tensor,
            rng_: zml.Tensor.Rng,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .sampleTokens, .{ prefill_hidden_, rng_ }, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model, prefill_hidden, rng, replicated_sharding, progress });
    errdefer if (prefill_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var decode_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: qwen35.TextModel,
            decode_hidden_: Tensor,
            rng_: zml.Tensor.Rng,
            replicated_sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .sampleTokens, .{ decode_hidden_, rng_ }, .{ .shardings = &.{replicated_sharding_} });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model, decode_hidden, rng, replicated_sharding, progress });
    errdefer if (decode_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const prefill_embedding_exe = try prefill_embedding_future.await(io);
    const decode_embedding_exe = try decode_embedding_future.await(io);
    const prefill_sampling_exe = try prefill_sampling_future.await(io);
    const decode_sampling_exe = try decode_sampling_future.await(io);

    return .{
        .prefill_embedding_exe = prefill_embedding_exe,
        .decode_embedding_exe = decode_embedding_exe,
        .prefill_full_layer_exe = prefill_full_layer_exe,
        .decode_full_layer_exe = decode_full_layer_exe,
        .prefill_linear_layer_exe = prefill_linear_layer_exe,
        .decode_linear_layer_exe = decode_linear_layer_exe,
        .prefill_sampling_exe = prefill_sampling_exe,
        .decode_sampling_exe = decode_sampling_exe,
    };
}

//======================= Generation setup ========================

fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, qwen_model: Qwen35) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    _ = qwen_model; // prompt already contains special tokens
    const rendered = try std.fmt.allocPrint(
        allocator,
        "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        .{prompt},
    );
    defer allocator.free(rendered);

    const encoded_prompt = try encoder.encode(rendered);
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, encoded_prompt.len);
    try tokens.appendSlice(allocator, encoded_prompt);
    return tokens.toOwnedSlice(allocator);
}

fn runAndGenerate(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    tokenizer_decoder: *zml.tokenizer.Tokenizer.Decoder,
    qwen35_buffers: zml.Bufferized(Qwen35),
    kv_cache_buffers: *zml.Bufferized(qwen35.KvCache),
    prefill_moe_metadata_buffers: *zml.Bufferized(zml.moe.Metadata),
    decode_moe_metadata_buffers: *zml.Bufferized(zml.moe.Metadata),
    compile_result: CompileModelResult,
    input_token_ids: []u32,
    writer: *std.Io.Writer,
    qwen_model: Qwen35,
    prefill_len: usize,
) !void {
    for (input_token_ids) |token_id| {
        if (try tokenizer_decoder.*.next(token_id)) |chunk| {
            try writer.writeAll(chunk);
        }
    }
    try writer.flush();

    const max_seq_len: usize = @intCast(qwen_model.text_model.gen_options.max_seq_len);
    if (input_token_ids.len > max_seq_len) return error.PromptTooLong;
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = prefill_len }, .u32);
    const decode_tokens_shape = zml.Shape.init(.{ .b = 1, .s = 1 }, .u32);
    const prefill_hidden_shape = zml.Shape.init(.{ .b = 1, .s = prefill_len, .d = qwen_model.config.text_config.hidden_size }, qwen_model.text_model.embed_tokens.weight.dtype());
    const decode_hidden_shape = zml.Shape.init(.{ .b = 1, .s = 1, .d = qwen_model.config.text_config.hidden_size }, qwen_model.text_model.embed_tokens.weight.dtype());

    const layer_types: []const Qwen35.LayerType = qwen_model.config.text_config.layer_types;

    const seed: u128 = 0;
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    var generated_token_slice: zml.Slice = try .alloc(allocator, decode_tokens_shape);
    defer generated_token_slice.free(allocator);

    const LayerIndexBuffer = union(enum) {
        self_attn: zml.Buffer,
        linear_attn: zml.Buffer,
    };

    var layer_cache_buffers = try allocator.alloc(LayerIndexBuffer, qwen_model.text_model.layers.len);
    defer {
        for (layer_cache_buffers) |*layer_cache_buffer| {
            switch (layer_cache_buffer.*) {
                .self_attn => |*buffer| buffer.deinit(),
                .linear_attn => |*buffer| buffer.deinit(),
            }
        }
        allocator.free(layer_cache_buffers);
    }

    {
        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types, 0..) |layer_type, layer_index| {
            switch (layer_type) {
                .full_attention => {
                    const layer_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(self_attn_layer_index)), .u32, replicated_sharding);
                    layer_cache_buffers[layer_index] = .{ .self_attn = layer_index_buffer };
                    self_attn_layer_index += 1;
                },
                .linear_attention => {
                    const layer_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(linear_attn_layer_index)), .u32, replicated_sharding);
                    layer_cache_buffers[layer_index] = .{ .linear_attn = layer_index_buffer };
                    linear_attn_layer_index += 1;
                },
            }
        }
    }

    var embedding_decode_args = try compile_result.decode_embedding_exe.args(allocator);
    defer embedding_decode_args.deinit(allocator);
    var embedding_decode_results = try compile_result.decode_embedding_exe.results(allocator);
    defer embedding_decode_results.deinit(allocator);

    var decode_full_layer_args: ?zml.Exe.Arguments = if (compile_result.decode_full_layer_exe) |exe| try exe.args(allocator) else null;
    defer if (decode_full_layer_args) |*args| args.deinit(allocator);
    var decode_full_layer_results: ?zml.Exe.Results = if (compile_result.decode_full_layer_exe) |exe| try exe.results(allocator) else null;
    defer if (decode_full_layer_results) |*results| results.deinit(allocator);

    var decode_linear_layer_args: ?zml.Exe.Arguments = if (compile_result.decode_linear_layer_exe) |exe| try exe.args(allocator) else null;
    defer if (decode_linear_layer_args) |*args| args.deinit(allocator);
    var decode_linear_layer_results: ?zml.Exe.Results = if (compile_result.decode_linear_layer_exe) |exe| try exe.results(allocator) else null;
    defer if (decode_linear_layer_results) |*results| results.deinit(allocator);

    var sampling_decode_args = try compile_result.decode_sampling_exe.args(allocator);
    defer sampling_decode_args.deinit(allocator);
    var sampling_decode_results = try compile_result.decode_sampling_exe.results(allocator);
    defer sampling_decode_results.deinit(allocator);

    var decode_hidden_buffer = try zml.Buffer.uninitialized(io, platform, decode_hidden_shape, replicated_sharding, .{});
    defer decode_hidden_buffer.deinit();

    {
        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..input_token_ids.len], input_token_ids);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();
        var prefill_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        var prefill_hidden_buffer = try zml.Buffer.uninitialized(io, platform, prefill_hidden_shape, replicated_sharding, .{});
        defer prefill_hidden_buffer.deinit();

        var embedding_prefill_args = try compile_result.prefill_embedding_exe.args(allocator);
        defer embedding_prefill_args.deinit(allocator);
        var embedding_prefill_results = try compile_result.prefill_embedding_exe.results(allocator);
        defer embedding_prefill_results.deinit(allocator);

        embedding_prefill_args.set(.{ qwen35_buffers.text_model.embed_tokens, prefill_tokens_buffer });
        compile_result.prefill_embedding_exe.call(embedding_prefill_args, &embedding_prefill_results);
        embedding_prefill_results.fill(.{&prefill_hidden_buffer});

        var prefill_full_layer_args: ?zml.Exe.Arguments = if (compile_result.prefill_full_layer_exe) |exe| try exe.args(allocator) else null;
        defer if (prefill_full_layer_args) |*args| args.deinit(allocator);
        var prefill_full_layer_results: ?zml.Exe.Results = if (compile_result.prefill_full_layer_exe) |exe| try exe.results(allocator) else null;
        defer if (prefill_full_layer_results) |*results| results.deinit(allocator);

        var prefill_linear_layer_args: ?zml.Exe.Arguments = if (compile_result.prefill_linear_layer_exe) |exe| try exe.args(allocator) else null;
        defer if (prefill_linear_layer_args) |*args| args.deinit(allocator);
        var prefill_linear_layer_results: ?zml.Exe.Results = if (compile_result.prefill_linear_layer_exe) |exe| try exe.results(allocator) else null;
        defer if (prefill_linear_layer_results) |*results| results.deinit(allocator);

        for (qwen35_buffers.text_model.layers, 0..) |layer_weights, i| {
            switch (layer_cache_buffers[i]) {
                .self_attn => |layer_index_buffer| {
                    const exe = compile_result.prefill_full_layer_exe orelse unreachable;
                    var layer_cache: zml.Bufferized(qwen35.KvCache.SelfAttnCache) = .{
                        .k = kv_cache_buffers.self_attn.k,
                        .v = kv_cache_buffers.self_attn.v,
                        .layer_index = layer_index_buffer,
                    };
                    prefill_full_layer_args.?.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_index_buffer, layer_cache, qwen_model.config, prefill_moe_metadata_buffers.* });
                    exe.call(prefill_full_layer_args.?, &prefill_full_layer_results.?);
                    prefill_full_layer_results.?.fill(.{ &prefill_hidden_buffer, &layer_cache });
                    kv_cache_buffers.self_attn.k = layer_cache.k;
                    kv_cache_buffers.self_attn.v = layer_cache.v;
                },
                .linear_attn => |layer_index_buffer| {
                    const exe = compile_result.prefill_linear_layer_exe orelse unreachable;
                    var layer_cache: zml.Bufferized(qwen35.KvCache.GatedDeltaNetCache) = .{
                        .conv_state = kv_cache_buffers.gated_delta_net.conv_state,
                        .recurrent_state = kv_cache_buffers.gated_delta_net.recurrent_state,
                        .layer_index = layer_index_buffer,
                    };
                    prefill_linear_layer_args.?.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_index_buffer, layer_cache, qwen_model.config, prefill_moe_metadata_buffers.* });
                    exe.call(prefill_linear_layer_args.?, &prefill_linear_layer_results.?);
                    prefill_linear_layer_results.?.fill(.{ &prefill_hidden_buffer, &layer_cache });
                    kv_cache_buffers.gated_delta_net.conv_state = layer_cache.conv_state;
                    kv_cache_buffers.gated_delta_net.recurrent_state = layer_cache.recurrent_state;
                },
            }
        }

        var sampling_prefill_args = try compile_result.prefill_sampling_exe.args(allocator);
        defer sampling_prefill_args.deinit(allocator);
        var sampling_prefill_results = try compile_result.prefill_sampling_exe.results(allocator);
        defer sampling_prefill_results.deinit(allocator);

        sampling_prefill_args.set(.{ qwen35_buffers.text_model, prefill_hidden_buffer, rng_buffers });
        compile_result.prefill_sampling_exe.call(sampling_prefill_args, &sampling_prefill_results);
        sampling_prefill_results.fill(.{ &prefill_tokens_buffer, &rng_buffers });

        try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[input_token_ids.len - 1];
        generated_token_slice.items(u32)[0] = generated_token;
    }

    const output_tokens_len = max_seq_len - input_token_ids.len - 1;
    var num_tokens_generated: usize = 1;

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, replicated_sharding);
    defer current_token_buffer.deinit();

    const now: std.Io.Timestamp = .now(io, .awake);

    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.*.next(generated_token)) |chunk| {
            try writer.writeAll(chunk);
            try writer.flush();
        }

        if (i == output_tokens_len) break :generation;
        if (generated_token == qwen_model.special_tokens.end_of_text_token_id) break :generation;

        var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(input_token_ids.len + i)), .u32, replicated_sharding);
        defer token_index_buffer.deinit();

        embedding_decode_args.set(.{ qwen35_buffers.text_model.embed_tokens, current_token_buffer });
        compile_result.decode_embedding_exe.call(embedding_decode_args, &embedding_decode_results);
        embedding_decode_results.fill(.{&decode_hidden_buffer});

        for (qwen35_buffers.text_model.layers, 0..) |layer_weights, layer_index| {
            switch (layer_cache_buffers[layer_index]) {
                .self_attn => |layer_index_buffer| {
                    const exe = compile_result.decode_full_layer_exe orelse unreachable;
                    var layer_cache: zml.Bufferized(qwen35.KvCache.SelfAttnCache) = .{
                        .k = kv_cache_buffers.self_attn.k,
                        .v = kv_cache_buffers.self_attn.v,
                        .layer_index = layer_index_buffer,
                    };
                    decode_full_layer_args.?.set(.{ layer_weights, decode_hidden_buffer, token_index_buffer, layer_cache, qwen_model.config, decode_moe_metadata_buffers.* });
                    exe.call(decode_full_layer_args.?, &decode_full_layer_results.?);
                    decode_full_layer_results.?.fill(.{ &decode_hidden_buffer, &layer_cache });
                    kv_cache_buffers.self_attn.k = layer_cache.k;
                    kv_cache_buffers.self_attn.v = layer_cache.v;
                },
                .linear_attn => |layer_index_buffer| {
                    const exe = compile_result.decode_linear_layer_exe orelse unreachable;
                    var layer_cache: zml.Bufferized(qwen35.KvCache.GatedDeltaNetCache) = .{
                        .conv_state = kv_cache_buffers.gated_delta_net.conv_state,
                        .recurrent_state = kv_cache_buffers.gated_delta_net.recurrent_state,
                        .layer_index = layer_index_buffer,
                    };
                    decode_linear_layer_args.?.set(.{ layer_weights, decode_hidden_buffer, token_index_buffer, layer_cache, qwen_model.config, decode_moe_metadata_buffers.* });
                    exe.call(decode_linear_layer_args.?, &decode_linear_layer_results.?);
                    decode_linear_layer_results.?.fill(.{ &decode_hidden_buffer, &layer_cache });
                    kv_cache_buffers.gated_delta_net.conv_state = layer_cache.conv_state;
                    kv_cache_buffers.gated_delta_net.recurrent_state = layer_cache.recurrent_state;
                },
            }
        }

        sampling_decode_args.set(.{ qwen35_buffers.text_model, decode_hidden_buffer, rng_buffers });
        compile_result.decode_sampling_exe.call(sampling_decode_args, &sampling_decode_results);
        sampling_decode_results.fill(.{ &current_token_buffer, &rng_buffers });

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

fn initMoeMetadata(qwen_model: Qwen35, moe_backend: zml.moe.Backend, token_len: usize, batch_size: u32) !zml.moe.Metadata {
    return switch (moe_backend) {
        .flashinfer => .init(.fromBackend(.flashinfer)),
        .triton => blk: {
            var w1_zero_bias_shape: ?zml.Shape = null;
            var w2_zero_bias_shape: ?zml.Shape = null;
            var first_out_shape: ?zml.Shape = null;
            var second_out_shape: ?zml.Shape = null;

            for (qwen_model.text_model.layers) |layer| {
                switch (layer.ffn) {
                    .dense => {},
                    .sparse => |sparse| {
                        const num_experts_per_tok = qwen_model.config.text_config.num_experts_per_tok.?;
                        const gate_up_shape = zml.Shape.init(.{
                            .expert = sparse.gate_up_proj.dim(.expert),
                            .out = sparse.gate_up_proj.dim(.dout),
                        }, sparse.gate_up_proj.dtype());
                        const down_shape = zml.Shape.init(.{
                            .expert = sparse.down_proj.dim(.expert),
                            .out = sparse.down_proj.dim(.d),
                        }, sparse.down_proj.dtype());
                        const first_out = zml.Shape.init(.{
                            .token = batch_size * token_len,
                            .topk = num_experts_per_tok,
                            .out = sparse.gate_up_proj.dim(.dout),
                        }, .bf16);
                        const second_out = zml.Shape.init(.{
                            .token = batch_size * token_len,
                            .topk = num_experts_per_tok,
                            .out = sparse.down_proj.dim(.d),
                        }, .bf16);

                        if (w1_zero_bias_shape == null) {
                            w1_zero_bias_shape = gate_up_shape;
                            w2_zero_bias_shape = down_shape;
                            first_out_shape = first_out;
                            second_out_shape = second_out;
                            continue;
                        }

                        if (!w1_zero_bias_shape.?.eql(gate_up_shape) or !w2_zero_bias_shape.?.eql(down_shape) or !first_out_shape.?.eql(first_out) or !second_out_shape.?.eql(second_out)) {
                            return error.UnsupportedMoeBiasShapes;
                        }
                    },
                }
            }

            break :blk .init(.{ .triton = .{
                .w1_zero_bias_shape = w1_zero_bias_shape,
                .w2_zero_bias_shape = w2_zero_bias_shape,
                .first_out_shape = first_out_shape,
                .second_out_shape = second_out_shape,
            } });
        },
    };
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
        .cache = initKvCache(qwen_model.text_model.embed_tokens.weight.dtype(), qwen_model.config, qwen_model.text_model.gen_options),
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
