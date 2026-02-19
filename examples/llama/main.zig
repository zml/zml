const std = @import("std");
const log = std.log;

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

pub const std_options: std.Options = .{
    .log_level = .info,
};

const partitioner: zml.sharding.Partitioning.Partitioner = .shardy;

const CliArgs = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    seqlen: u32 = 512,
    reader_buffer_size: stdx.flags.ByteSize = .{ .value = 32, .unit = .mib },
    writer_buffer_size: stdx.flags.ByteSize = .{ .value = 32, .unit = .mib },
    async_limit: ?u32 = null,
    backend: ?zml.attention.Backend = null,

    pub const help =
        \\Usage: llama --model=<path> [options]
        \\
        \\Run LLaMA inference on a model.
        \\
        \\Options:
        \\  --model=<path>              Path to model (local path or HuggingFace repo)
        \\  --prompt=<text>             Input prompt (reads from stdin if not provided)
        \\  --seqlen=<n>                Maximum sequence length (default: 512)
        \\  --backend=<text>            Attention backend to use ([vanilla, cuda_fa2, cuda_fa3], default: auto-selection)
        \\  --reader-buffer-size=<size> Reader buffer size (default: 32MiB)
        \\  --writer-buffer-size=<size> Writer buffer size (default: 32MiB)
        \\  --async-limit=<n>           Async I/O concurrency limit
        \\  -h, --help                  Show this help message
        \\
        \\Examples:
        \\  llama --model=hf://meta-llama/Llama-3.1-8B-Instruct --prompt="Hello, world!"
        \\  llama --model=$(realpath ../Llama-3.1-8B-Instruct) --seqlen=1024
        \\  echo "What is 2+2?" | llama --model=meta-llama/Llama-3.1-8B-Instruct
        \\
    ;
};

pub fn main() !void {
    log.info("LLama was compiled with {}", .{@import("builtin").mode});

    // var debug_allocator: ?std.heap.DebugAllocator(.{}) = null;
    // const allocator = if (@import("builtin").mode == .Debug) blk: {
    //     debug_allocator = .init;
    //     break :blk debug_allocator.?.allocator();
    // } else std.heap.c_allocator;
    // defer if (debug_allocator) |*da| std.debug.assert(da.deinit() == .ok);
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const args = stdx.flags.parseProcessArgs(CliArgs);

    var vfs: zml.io.VFS = try .init(allocator, threaded.io());
    defer vfs.deinit();

    var http_client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
    };

    try http_client.initDefaultProxies(allocator);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    defer vfs_file.deinit();
    try vfs.register("file", vfs_file.io());

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), &http_client, .https);
    defer vfs_https.deinit();
    try vfs.register("https", vfs_https.io());

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, threaded.io(), &http_client);
    defer hf_vfs.deinit();
    try vfs.register("hf", hf_vfs.io());

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, threaded.io(), &http_client);
    defer s3_vfs.deinit();
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    log.info("Resolving model repo", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    const parsed_config = try parseConfig(allocator, io, repo);
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

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("\n{f}", .{platform.fmtVerbose()});
    // log.info("Devices:", .{});
    // for (platform.devices) |device| {
    //     log.info("\t- {f}", .{device});
    // }

    var profiler = try platform.profiler(allocator);

    // Initialize the Llama struct and map the content of the .safetensors to the model tensors.
    var llama_model: llama.LlamaLM = try .init(allocator, store.view(), config, llama_options);
    defer llama_model.deinit(allocator);

    const backend = args.backend orelse b: {
        const selected = zml.attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    // Specify shapes of input arguments
    const dtype = llama_model.model.embed_tokens.weight.dtype();
    const llama_parameters: LlamaParameters = .{
        .prefill_tokens = .init(.{ .s = llama_options.max_seq_len }, .u32),
        .decode_tokens = .init(.{ .s = 1 }, .u32),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(.init(.{
            .layer = llama_model.model.layers.len,
            .k = args.seqlen,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, dtype)),
        .rng = .init(),
        .attention_metadata = .init(.fromBackend(backend, @intCast(args.seqlen), @intCast(config.num_attention_heads))),
        .attention_parameters = .init(.fromBackend(backend)),
    };

    const physical_mesh: zml.sharding.PhysicalMesh = try .auto(allocator, platform);

    const embeddings_mesh: zml.sharding.LogicalMesh = try .init("embeddings_mesh", .{ .voc = .high_bandwidth });
    const tp_mesh: zml.sharding.LogicalMesh = try .init("tp_mesh", .{ .model = .high_bandwidth, .replicated = .low_bandwidth });

    const embeddings_strategy: zml.sharding.Strategy = try .suggest(embeddings_mesh, physical_mesh);
    const tp_strategy: zml.sharding.Strategy = try .suggest(tp_mesh, physical_mesh);

    const sharding_embeddings: zml.sharding.Sharding = try .initFromStrategy(embeddings_mesh, physical_mesh, embeddings_strategy);
    const sharding_tp: zml.sharding.Sharding = try .initFromStrategy(tp_mesh, physical_mesh, tp_strategy);
    const sharding_replicated: zml.sharding.Sharding = try .initFromStrategy(try .init("replica_mesh", .{ .replica = .low_bandwidth }), physical_mesh, .init);

    var shardings_array = [_]zml.sharding.Sharding{ sharding_embeddings, sharding_tp };
    const shardings: []zml.sharding.Sharding = shardings_array[0..];

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    var tokenizer_future = try io.concurrent(loadTokenizer, .{ allocator, io, repo, &progress });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    var compiled_model_result_future = try io.concurrent(compileModel, .{ allocator, io, platform, llama_model, llama_parameters, shardings, &progress });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.prefill_exe.deinit();
        defer v.decode_exe.deinit();
    } else |_| {};

    var load_group: stdx.Io.LimitedGroup = .init(16);
    defer load_group.cancel(io);

    var llama_buffers_future = try io.concurrent(LlamaLM.load, .{
        &llama_model,
        allocator,
        io,
        platform,
        &store,
        shardings,
        sharding_replicated,
        &progress,
    });
    errdefer b: {
        var v = llama_buffers_future.cancel(io) catch break :b;
        v = v; // autofix
        // LlamaLM.unloadBuffers(&v, allocator);
    }

    const compiled_model_result = try compiled_model_result_future.await(io);
    defer compiled_model_result.prefill_exe.deinit();
    defer compiled_model_result.decode_exe.deinit();

    var llama_buffers = try llama_buffers_future.await(io);
    llama_buffers = llama_buffers; // autofix
    // defer LlamaLM.unloadBuffers(&llama_buffers, allocator);
    progress.end();

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try llama_parameters.kv_cache.initBuffer(io, platform, sharding_tp);
    defer llama.KvCache.deinitBuffer(&kv_cache_buffers);

    var attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = try llama_parameters.attention_metadata.initBuffer(io, platform, sharding_tp);
    defer zml.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    const prompt = if (args.prompt) |p| p else blk: {
        var reader = std.Io.File.stdin().reader(io, &.{});
        break :blk try reader.interface.allocRemaining(allocator, .unlimited);
    };

    log.info("✅ Prompt: {s}", .{prompt});

    try profiler.?.start();
    defer {
        profiler.?.stop() catch unreachable;
        const data = profiler.?.collectData(allocator) catch unreachable;
        var file = std.Io.Dir.createFile(.cwd(), threaded.io(), "/home/hugo/zml/proto9.pb", .{ .read = true }) catch unreachable;
        defer file.close(threaded.io());
        file.writeStreamingAll(threaded.io(), data) catch unreachable;
        allocator.free(data);
    }

    // Unbuffered writing of the tokens to stdout.
    var stdout = std.Io.File.stdout().writer(io, &.{});

    try generateText(
        allocator,
        io,
        &llama_buffers,
        compiled_model_result.prefill_exe,
        compiled_model_result.decode_exe,
        &kv_cache_buffers,
        &attention_metadata_buffers,
        tokenizer,
        config,
        llama_options,
        @intCast((try std.Io.Clock.now(.real, io)).toNanoseconds()),
        prompt[0..],
        false,
        &stdout.interface,
        platform,
        sharding_replicated,
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

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    // log.info("Loading tokenizer", .{});
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();
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
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: llama.LlamaLM,
    parameters: LlamaParameters,
    shardings: []zml.sharding.Sharding,
    progress: *std.Progress.Node,
) !CompileModelResult {
    var timer = try std.time.Timer.start();
    defer log.info("Compiled model [{D}]", .{timer.read()});

    // Compile the model twice, one for prefill, one for generation.
    var prefill_future = try io.concurrent(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters, shardings_: []zml.sharding.Sharding, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();
            var timer_ = try std.time.Timer.start();
            defer log.info("Compiled prefill [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, llama_model_, .forward, .{
                parameters_.prefill_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                parameters_.attention_metadata,
                parameters_.attention_parameters,
            }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, shardings, progress });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters, shardings_: []zml.sharding.Sharding, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();
            var timer_ = try std.time.Timer.start();
            defer log.info("Compiled decode [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, llama_model_, .forward, .{
                parameters_.decode_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                parameters_.attention_metadata,
                parameters_.attention_parameters,
            }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, shardings, progress });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

fn loadModelBuffers(
    allocator: std.mem.Allocator,
    buffers_allocator: std.mem.Allocator,
    io: std.Io,
    progress: *std.Progress.Node,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    llama_model: llama.LlamaLM,
    pool: *zml.mem.DynamicBufferPool,
    async_limit: usize,
) !zml.Bufferized(llama.LlamaLM) {
    var transferred_bytes: usize = 0;

    var timer = try stdx.time.Timer.start();
    defer {
        const duration = timer.read();
        const seconds = @as(f64, @floatFromInt(duration.ns)) / 1e9;
        const gb_per_sec = @as(f64, @floatFromInt(transferred_bytes)) / (1024.0 * 1024.0 * 1024.0) / seconds;
        const gbps = gb_per_sec * 8.0;
        log.info("Loaded model [{D} {d:.3} GB/s {d:.3} gbps]", .{ duration, gb_per_sec, gbps });
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var group: stdx.Io.LimitedGroup = .init(async_limit);
    defer group.cancel(io);

    const bufferize_ctx: zml.io.BufferizeContext(llama.TransferCtx) = .{
        .allocator = allocator,
        .arena = &arena,
        .io = io,
        .platform = platform,
        .cb_ctx = .{
            .allocator = buffers_allocator,
            .pool = pool,
            .transferred_bytes = &transferred_bytes,
            .progress = progress,
        },
    };

    const bufferized = llama_model.loadBuffers(bufferize_ctx, &group, store.view(), transferBuffer);
    try group.await(io);

    return bufferized;
}

fn transferBuffer(ctx: zml.io.TensorBufferTransfer(llama.TransferCtx)) !void {
    var transfer_progress = ctx.cb_ctx.progress.start(ctx.tensor.name, ctx.tensor.byteSize());

    var reader = zml.safetensors.TensorReader.init(ctx.io, ctx.tensor, &.{}, .{}) catch unreachable;
    defer reader.deinit();

    var writer = zml.io.MemoryWriter.init(ctx.cb_ctx.allocator, ctx.io, ctx.platform, ctx.cb_ctx.pool, ctx.tensor.shape, ctx.buffer) catch unreachable;
    defer {
        writer.interface.flush() catch unreachable;
        writer.deinit();
    }

    _ = reader.interface.streamRemaining(&writer.interface) catch unreachable;
    writer.interface.flush() catch unreachable;

    transfer_progress.end();

    ctx.cb_ctx.transferred_bytes.* += ctx.tensor.byteSize();
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
    llama_buffers: *zml.Bufferized(LlamaLM),
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
    kv_cache_buffers: *zml.Bufferized(llama.KvCache),
    attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    tokenizer: zml.tokenizer.Tokenizer,
    config: LlamaLM.Config,
    options: LlamaLM.Options,
    seed: u128,
    prompt: []const u8,
    skip_llama3_encoding: bool,
    writer: *std.Io.Writer,
    platform: *const zml.Platform,
    replicated_sharding: zml.sharding.Sharding,
) !void {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    defer allocator.free(prompt_tok);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const max_seq_len = options.max_seq_len;

    // init RNG and buffers
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    {
        var prefill_args = try prefill_exe.args(allocator);
        defer prefill_args.deinit(allocator);

        var prefill_results = try prefill_exe.results(allocator);
        defer prefill_results.deinit(allocator);

        // prepare device buffers for the prefill tokens and their positions
        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .u32));
        defer prefill_tokens_slice.free(allocator);
        @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();
        var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, replicated_sharding);
        defer prefill_token_pos_buffer.deinit();

        prefill_args.set(.{ llama_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers, attention_metadata_buffers });

        prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers, &rng_buffers });
        try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
        generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];
    }

    // Prepare for token-by-token generation,

    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);

    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // start with the token generated based on the full prompt.
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, replicated_sharding);
    defer current_token_buffer.deinit();

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    var timer = try stdx.time.Timer.start();

    // One token has alreadyh been generated by the prefill.
    var num_tokens_generated: usize = 1;

    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try writer.writeAll(chunk);
            try writer.flush();
        }

        // check for eos
        if (i == output_tokens_len) break :generation;
        switch (config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }

        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, replicated_sharding);
        defer token_pos_buffer.deinit();

        // call to generate the next token
        decode_args.set(.{ llama_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers, attention_metadata_buffers });

        decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        // extract the generated token from the buffer
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
