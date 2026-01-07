const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;

const llama = @import("llama.zig");
const LlamaLM = llama.LlamaLM;
const Llama = llama.Llama;
const KvCache = llama.KvCache;
const TransformerLayer = llama.TransformerLayer;
const SelfAttn = llama.SelfAttn;

const log = std.log.scoped(.llama);

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Args = struct {
    model: ?[]const u8 = null,
    seqlen: u32 = 512,
};

pub fn main() !void {
    log.info("LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    const args: Args = blk: {
        var ret: Args = .{};
        var it = std.process.args();
        defer it.deinit();
        while (it.next()) |arg| {
            if (std.mem.startsWith(u8, arg, "--model=")) {
                ret.model = arg["--model=".len..];
            } else if (std.mem.startsWith(u8, arg, "--seqlen=")) {
                ret.seqlen = try std.fmt.parseUnsigned(u32, arg["--seqlen=".len..], 10);
            }
        }

        if (ret.model == null) {
            log.err("Missing --model", .{});
            return;
        }

        break :blk ret;
    };

    log.info("Resolving Model repo", .{});
    var repo = try zml.safetensors.resolveModelRepo(allocator, io, args.model.?);
    defer repo.deinit(allocator, io);

    const parsed_config = try parseConfig(allocator, io, repo.dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // Write metadata from the config file into the LlamaLm struct.
    const llama_options: llama.LlamaLM.Options = .{
        .max_seq_len = args.seqlen,
        .sampling_strategy = .{
            .topk = 2,
            .temperature = 1.0,
        },
    };

    // Initialize the Llama struct and map the content of the .safetensors to the model tensors.
    const llama_model: llama.LlamaLM = try .init(allocator, store.view(), config, llama_options);
    defer llama_model.deinit(allocator);

    // Specify shapes of input arguments
    const dtype = llama_model.model.embed_tokens.weight.dtype();
    const llama_parameters: LlamaParameters = .{
        .prefill_tokens = .init(.{ .s = llama_options.max_seq_len }, .i32),
        .decode_tokens = .init(.{ .s = 1 }, .i32),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(.init(.{
            .layer = llama_model.model.layers.len,
            .k = args.seqlen,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, dtype)),
        .rng = .init(),
    };

    var tokenizer_future = io.async(loadTokenizer, .{ allocator, io, repo.dir });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    var platform: zml.Platform = try .auto(io, .{});
    defer platform.deinit();
    var it = platform.devicesIterator();
    log.info("Devices:", .{});
    while (it.next()) |device| {
        log.info("\t- {f}", .{device});
    }

    var compiled_model_result_future = io.async(compile, .{ allocator, io, platform, llama_model, llama_parameters });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.embedding_prefill_exe.deinit();
        defer v.embedding_decode_exe.deinit();
        defer v.layer_prefill_exe.deinit();
        defer v.layer_decode_exe.deinit();
        defer v.sampling_prefill_exe.deinit();
        defer v.sampling_decode_exe.deinit();
    } else |_| {};

    var llama_buffers_future = io.async(loadModelBuffers, .{ allocator, io, platform, &store, llama_model });
    errdefer b: {
        var v = llama_buffers_future.cancel(io) catch break :b;
        LlamaLM.unloadBuffers(&v, allocator);
    }

    const compiled_model_result = try compiled_model_result_future.await(io);
    defer compiled_model_result.embedding_prefill_exe.deinit();
    defer compiled_model_result.embedding_decode_exe.deinit();
    defer compiled_model_result.layer_prefill_exe.deinit();
    defer compiled_model_result.layer_decode_exe.deinit();
    defer compiled_model_result.sampling_prefill_exe.deinit();
    defer compiled_model_result.sampling_decode_exe.deinit();

    var llama_buffers = try llama_buffers_future.await(io);
    defer LlamaLM.unloadBuffers(&llama_buffers, allocator);

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try llama_parameters.kv_cache.initBuffer(io, platform);
    defer llama.KvCache.deinitBuffer(&kv_cache_buffers);

    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    const prompt = blk: {
        var reader = std.Io.File.stdin().reader(io, &.{});
        break :blk try reader.interface.allocRemaining(allocator, .unlimited);
    };
    defer allocator.free(prompt);

    // const prompt = cli.args.prompt orelse "What is the capital of France?";
    log.info("✅\tPrompt: {s}", .{prompt});

    // Unbuffered writing of the tokens to stdout.
    var stdout = std.Io.File.stdout().writer(io, &.{});

    try generateText(
        allocator,
        io,
        llama_buffers,
        compiled_model_result.embedding_prefill_exe,
        compiled_model_result.embedding_decode_exe,
        compiled_model_result.layer_prefill_exe,
        compiled_model_result.layer_decode_exe,
        compiled_model_result.norm_prefill_exe,
        compiled_model_result.norm_decode_exe,
        compiled_model_result.sampling_prefill_exe,
        compiled_model_result.sampling_decode_exe,
        &kv_cache_buffers,
        tokenizer,
        config,
        llama_options,
        @intCast((try std.Io.Clock.now(.real, io)).toNanoseconds()),
        prompt[0..],
        false,
        &stdout.interface,
        platform,
    );
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(LlamaLM.Config) {
    var timer = try std.time.Timer.start();
    log.info("Loading model config", .{});
    defer log.info("Loaded model config [{D}]", .{timer.read()});

    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();

    return parsed_config;
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    log.info("Loading tokenizer", .{});
    var timer = try std.time.Timer.start();
    defer log.info("Loaded tokenizer [{D}]", .{timer.read()});
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

const LlamaParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: llama.KvCache,
    rng: zml.Tensor.Rng,
};

const CompileModelResult = struct {
    embedding_prefill_exe: zml.Exe,
    embedding_decode_exe: zml.Exe,
    layer_prefill_exe: zml.Exe,
    layer_decode_exe: zml.Exe,
    norm_prefill_exe: zml.Exe,
    norm_decode_exe: zml.Exe,
    sampling_prefill_exe: zml.Exe,
    sampling_decode_exe: zml.Exe,
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: zml.Platform,
    llama_model: llama.LlamaLM,
    parameters: LlamaParameters,
) !CompileModelResult {
    var timer = try std.time.Timer.start();
    log.info("Compiling model components", .{});
    defer log.info("Compiled all components [{D}]", .{timer.read()});

    const dtype = llama_model.model.embed_tokens.weight.dtype();
    const hidden_dim = llama_model.config.hidden_size;

    // Build hidden states tensors for prefill and decode compilation
    const hidden_shape_prefill = zml.Shape.init(.{ .s = parameters.prefill_tokens.dim(.s), .d = hidden_dim }, dtype);
    const hidden_states_prefill = zml.Tensor.init(hidden_shape_prefill, dtype);

    const hidden_shape_decode = zml.Shape.init(.{ .s = parameters.decode_tokens.dim(.s), .d = hidden_dim }, dtype);
    const hidden_states_decode = zml.Tensor.init(hidden_shape_decode, dtype);

    // embedding
    var embed_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: zml.nn.TokenEmbedding, input: zml.Tensor) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{input});
        }
    }.call, .{ allocator, io, platform, llama_model.model.embed_tokens, parameters.prefill_tokens });
    errdefer if (embed_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var embed_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: zml.nn.TokenEmbedding, input: zml.Tensor) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{input});
        }
    }.call, .{ allocator, io, platform, llama_model.model.embed_tokens, parameters.decode_tokens });
    errdefer if (embed_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // layer
    const layer_module = llama_model.model.layers[0];

    var layer_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: llama.TransformerLayer, hidden: zml.Tensor, idx: zml.Tensor, kv: llama.KvCache) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{ hidden, idx, kv });
        }
    }.call, .{ allocator, io, platform, layer_module, hidden_states_prefill, parameters.token_index, parameters.kv_cache });
    errdefer if (layer_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var layer_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: llama.TransformerLayer, hidden: zml.Tensor, idx: zml.Tensor, kv: llama.KvCache) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{ hidden, idx, kv });
        }
    }.call, .{ allocator, io, platform, layer_module, hidden_states_decode, parameters.token_index, parameters.kv_cache });
    errdefer if (layer_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // norm
    var norm_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: llama.RmsNorm, hidden: zml.Tensor) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{hidden});
        }
    }.call, .{ allocator, io, platform, llama_model.model.norm, hidden_states_prefill });
    errdefer if (norm_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var norm_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, mod: llama.RmsNorm, hidden: zml.Tensor) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{hidden});
        }
    }.call, .{ allocator, io, platform, llama_model.model.norm, hidden_states_decode });
    errdefer if (norm_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // --- SAMPLING ---
    var sampling_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, model: llama.LlamaLM, head: zml.nn.Linear, hidden: zml.Tensor, rng: zml.Tensor.Rng, opts: zml.nn.SamplingStrategy) !zml.Exe {
            return plt.compile(alloc, io_, model, .sampleTokens, .{ head, hidden, rng, opts });
        }
    }.call, .{ allocator, io, platform, llama_model, llama_model.lm_head.?, hidden_states_prefill, parameters.rng, llama_model.gen_opts });
    errdefer if (sampling_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var sampling_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: zml.Platform, model: llama.LlamaLM, head: zml.nn.Linear, hidden: zml.Tensor, rng: zml.Tensor.Rng, opts: zml.nn.SamplingStrategy) !zml.Exe {
            return plt.compile(alloc, io_, model, .sampleTokens, .{ head, hidden, rng, opts });
        }
    }.call, .{ allocator, io, platform, llama_model, llama_model.lm_head.?, hidden_states_decode, parameters.rng, llama_model.gen_opts });
    errdefer if (sampling_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // Wait for execs compiled
    const embedding_prefill_exe = try embed_prefill_future.await(io);
    const embedding_decode_exe = try embed_decode_future.await(io);

    const layer_prefill_exe = try layer_prefill_future.await(io);
    const layer_decode_exe = try layer_decode_future.await(io);

    const norm_prefill_exe = try norm_prefill_future.await(io);
    const norm_decode_exe = try norm_decode_future.await(io);

    const sampling_prefill_exe = try sampling_prefill_future.await(io);
    const sampling_decode_exe = try sampling_decode_future.await(io);

    return .{
        .embedding_prefill_exe = embedding_prefill_exe,
        .embedding_decode_exe = embedding_decode_exe,
        .layer_prefill_exe = layer_prefill_exe,
        .layer_decode_exe = layer_decode_exe,
        .norm_prefill_exe = norm_prefill_exe,
        .norm_decode_exe = norm_decode_exe,
        .sampling_prefill_exe = sampling_prefill_exe,
        .sampling_decode_exe = sampling_decode_exe,
    };
}

fn loadModelBuffers(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, store: *zml.io.TensorStore, llama_model: llama.LlamaLM) !zml.Bufferized(llama.LlamaLM) {
    var timer = try std.time.Timer.start();
    log.info("Loading model", .{});
    defer log.info("Loaded model [{D}]", .{timer.read()});
    return llama_model.loadBuffers(allocator, io, store.view(), platform);
}

pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: LlamaLM.Config, prompt: []const u8, skip_llama3_encoding: bool) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    if (skip_llama3_encoding) {
        // Copy so the ownership is the same in both branches.
        return try allocator.dupe(u32, try encoder.encode(prompt));
    }

    const start_header = tokenizer.tokenToId("<|start_header_id|>") orelse return error.NoSuchToken;
    const end_header = tokenizer.tokenToId("<|end_header_id|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const eot = tokenizer.tokenToId("<|eot_id|>") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, &.{ config.bos_token_id, start_header, user, end_header, newline });

    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ eot, newline });

    try tokens.appendSlice(allocator, &.{ start_header, assistant, end_header, newline });

    return tokens.toOwnedSlice(allocator);
}

pub fn generateText(
    allocator: std.mem.Allocator,
    io: std.Io,
    llama_buffers: zml.Bufferized(LlamaLM),
    embedding_prefill_exe: zml.Exe,
    embedding_decode_exe: zml.Exe,
    layer_prefill_exe: zml.Exe,
    layer_decode_exe: zml.Exe,
    norm_prefill_exe: zml.Exe,
    norm_decode_exe: zml.Exe,
    sampling_prefill_exe: zml.Exe,
    sampling_decode_exe: zml.Exe,
    kv_cache_buffers: *zml.Bufferized(llama.KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    config: LlamaLM.Config,
    options: LlamaLM.Options,
    seed: u128,
    prompt: []const u8,
    skip_llama3_encoding: bool,
    writer: *std.Io.Writer,
    platform: zml.Platform,
) !void {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    defer allocator.free(prompt_tok);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const max_seq_len = options.max_seq_len;
    const hidden_dim = config.hidden_size;
    const dtype = .bf16;

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .i32));
    defer generated_token_slice.free(allocator);

    const KvCacheLayerBuffer = struct {
        k: zml.Buffer,
        v: zml.Buffer,
        layer_index: zml.Buffer,
    };

    const num_layers = config.num_hidden_layers;
    var kv_layer_buffers = try allocator.alloc(KvCacheLayerBuffer, num_layers);
    defer {
        for (kv_layer_buffers) |input| {
            var idx = input.layer_index;
            idx.deinit();
        }
        allocator.free(kv_layer_buffers);
    }
    for (kv_layer_buffers, 0..) |*kv_layer, j| {
        const idx_buf = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(j)), .u32);
        kv_layer.* = .{
            .k = kv_cache_buffers.k,
            .v = kv_cache_buffers.v,
            .layer_index = idx_buf,
        };
    }

    // Prepare args and results for decode steps
    var embedding_decode_args = try embedding_decode_exe.args(allocator);
    defer embedding_decode_args.deinit(allocator);
    var embedding_decode_results = try embedding_decode_exe.results(allocator);
    defer embedding_decode_results.deinit(allocator);

    var layer_decode_args = try layer_decode_exe.args(allocator);
    defer layer_decode_args.deinit(allocator);
    var layer_decode_results = try layer_decode_exe.results(allocator);
    defer layer_decode_results.deinit(allocator);

    var norm_decode_args = try norm_decode_exe.args(allocator);
    defer norm_decode_args.deinit(allocator);
    var norm_decode_results = try norm_decode_exe.results(allocator);
    defer norm_decode_results.deinit(allocator);

    var sampling_decode_args = try sampling_decode_exe.args(allocator);
    defer sampling_decode_args.deinit(allocator);
    var sampling_decode_results = try sampling_decode_exe.results(allocator);
    defer sampling_decode_results.deinit(allocator);

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
    defer current_token_buffer.deinit();

    const decode_hidden_shape = zml.Shape.init(.{ .s = 1, .d = hidden_dim }, dtype);
    var decode_hidden_buffer = try zml.Buffer.uninitialized(io, platform, decode_hidden_shape, .{});
    defer decode_hidden_buffer.deinit();

    // Prefill
    {
        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .i32));
        defer prefill_tokens_slice.free(allocator);

        // Secure copy with cast to i32 because sampling output is i32
        @memset(prefill_tokens_slice.items(i32), 0);
        const prompt_i32 = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(prompt_tok));
        @memcpy(prefill_tokens_slice.items(i32)[0..prompt_tok.len], prompt_i32);

        // Prepare buffers
        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice);
        defer prefill_tokens_buffer.deinit();
        var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32);
        defer prefill_token_pos_buffer.deinit();

        const hidden_shape = zml.Shape.init(.{ .s = max_seq_len, .d = hidden_dim }, dtype);
        var prefill_hidden_buffer = try zml.Buffer.uninitialized(io, platform, hidden_shape, .{});
        defer prefill_hidden_buffer.deinit();

        // Execute embedding, layers, norm and sampling
        var embedding_args = try embedding_prefill_exe.args(allocator);
        defer embedding_args.deinit(allocator);
        var embedding_results = try embedding_prefill_exe.results(allocator);
        defer embedding_results.deinit(allocator);

        embedding_args.set(.{ llama_buffers.model.embed_tokens, prefill_tokens_buffer });
        embedding_prefill_exe.call(embedding_args, &embedding_results);
        embedding_results.fill(.{&prefill_hidden_buffer});

        var layer_args = try layer_prefill_exe.args(allocator);
        defer layer_args.deinit(allocator);
        var layer_results = try layer_prefill_exe.results(allocator);
        defer layer_results.deinit(allocator);

        for (llama_buffers.model.layers, 0..) |layer_weights, i| {
            kv_layer_buffers[i].k = kv_cache_buffers.k;
            kv_layer_buffers[i].v = kv_cache_buffers.v;
            layer_args.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_pos_buffer, kv_layer_buffers[i] });
            layer_prefill_exe.call(layer_args, &layer_results);

            layer_results.fill(.{ &prefill_hidden_buffer, kv_cache_buffers });
        }

        var norm_args = try norm_prefill_exe.args(allocator);
        defer norm_args.deinit(allocator);
        var norm_results = try norm_prefill_exe.results(allocator);
        defer norm_results.deinit(allocator);

        norm_args.set(.{ llama_buffers.model.norm, prefill_hidden_buffer });
        norm_prefill_exe.call(norm_args, &norm_results);
        norm_results.fill(.{&prefill_hidden_buffer});

        var sampling_args = try sampling_prefill_exe.args(allocator);
        defer sampling_args.deinit(allocator);
        var sampling_results = try sampling_prefill_exe.results(allocator);
        defer sampling_results.deinit(allocator);

        var full_sequence_output_buffer = try zml.Buffer.uninitialized(io, platform, zml.Shape.init(.{ .s = max_seq_len }, .u32), .{});
        defer full_sequence_output_buffer.deinit();

        sampling_args.set(.{ llama_buffers, llama_buffers.lm_head, prefill_hidden_buffer, rng_buffers });
        sampling_prefill_exe.call(sampling_args, &sampling_results);
        sampling_results.fill(.{ &full_sequence_output_buffer, &rng_buffers });

        try full_sequence_output_buffer.toSlice(io, prefill_tokens_slice);

        generated_token_slice.items(i32)[0] = prefill_tokens_slice.items(i32)[prompt_tok.len - 1];
    }

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    var timer = try stdx.time.Timer.start();
    var num_tokens_generated: usize = 1;

    // Decode
    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];

        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try writer.writeAll(chunk);
            try writer.flush();
        }

        if (i == output_tokens_len) break :generation;
        switch (config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }

        const token_pos_slice: zml.ConstSlice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        const token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice);
        defer token_pos_buffer.deinit();

        embedding_decode_args.set(.{ llama_buffers.model.embed_tokens, current_token_buffer });
        embedding_decode_exe.call(embedding_decode_args, &embedding_decode_results);
        embedding_decode_results.fill(.{&decode_hidden_buffer});

        for (llama_buffers.model.layers, 0..) |layer_weights, j| {
            kv_layer_buffers[j].k = kv_cache_buffers.k;
            kv_layer_buffers[j].v = kv_cache_buffers.v;
            layer_decode_args.set(.{ layer_weights, decode_hidden_buffer, token_pos_buffer, kv_layer_buffers[j] });
            layer_decode_exe.call(layer_decode_args, &layer_decode_results);
            layer_decode_results.fill(.{ &decode_hidden_buffer, kv_cache_buffers });
        }

        norm_decode_args.set(.{ llama_buffers.model.norm, decode_hidden_buffer });
        norm_decode_exe.call(norm_decode_args, &norm_decode_results);
        norm_decode_results.fill(.{&decode_hidden_buffer});

        sampling_decode_args.set(.{ llama_buffers, llama_buffers.lm_head, decode_hidden_buffer, rng_buffers });
        sampling_decode_exe.call(sampling_decode_args, &sampling_decode_results);
        sampling_decode_results.fill(.{ &current_token_buffer, &rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }

    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
