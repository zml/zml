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

    var debug_allocator: ?std.heap.DebugAllocator(.{}) = null;
    const allocator = if (@import("builtin").mode == .Debug) blk: {
        debug_allocator = .init;
        break :blk debug_allocator.?.allocator();
    } else std.heap.c_allocator;
    defer if (debug_allocator) |*da| std.debug.assert(da.deinit() == .ok);
    // const allocator = std.heap.c_allocator;

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

    var compiled_model_result_future = try io.concurrent(compile, .{ allocator, io, platform, llama_model, llama_parameters, shardings, &progress });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        _ = v; // autofix
        // defer v.embedding_prefill_exe.deinit();
        // defer v.embedding_decode_exe.deinit();
        // defer v.layer_prefill_exe.deinit();
        // defer v.layer_decode_exe.deinit();
        // defer v.sampling_prefill_exe.deinit();
        // defer v.sampling_decode_exe.deinit();
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
    defer compiled_model_result.embedding_prefill_exe.deinit();
    defer compiled_model_result.embedding_decode_exe.deinit();
    defer compiled_model_result.layer_prefill_exe.deinit();
    defer compiled_model_result.layer_decode_exe.deinit();
    defer compiled_model_result.sampling_prefill_exe.deinit();
    defer compiled_model_result.sampling_decode_exe.deinit();

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
        // log.warn("data: {any}", .{data});
        var file = std.Io.Dir.createFile(.cwd(), threaded.io(), "/home/hugo/zml/proto8.pb", .{ .read = true }) catch unreachable;
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
        compiled_model_result.embedding_prefill_exe,
        compiled_model_result.embedding_decode_exe,
        compiled_model_result.layer_prefill_exe,
        compiled_model_result.layer_decode_exe,
        compiled_model_result.norm_prefill_exe,
        compiled_model_result.norm_decode_exe,
        compiled_model_result.sampling_prefill_exe,
        compiled_model_result.sampling_decode_exe,
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
        sharding_tp,
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
    platform: *const zml.Platform,
    llama_model: llama.LlamaLM,
    parameters: LlamaParameters,
    shardings: []zml.sharding.Sharding,
    _: *std.Progress.Node,
) !CompileModelResult {
    var timer = try std.time.Timer.start();
    log.info("Compiling model components", .{});
    defer log.info("Compiled all components [{D}]", .{timer.read()});

    const dtype = llama_model.model.embed_tokens.weight.dtype();
    const hidden_dim = llama_model.config.hidden_size;

    // Build hidden states tensors for prefill and decode compilation
    const hidden_shape_prefill = zml.Shape.init(.{ .s = parameters.prefill_tokens.dim(.s), .d = hidden_dim }, dtype)
        .withTags(.{ .s, .d })
        .withPartitioning(.{ .d = .model });
    const hidden_states_prefill: zml.Tensor = .fromShape(hidden_shape_prefill);

    const hidden_shape_decode = zml.Shape.init(.{ .s = parameters.decode_tokens.dim(.s), .d = hidden_dim }, dtype)
        .withTags(.{ .s, .d })
        .withPartitioning(.{ .d = .model });
    const hidden_states_decode: zml.Tensor = .fromShape(hidden_shape_decode);

    // embedding
    var embed_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: zml.nn.TokenEmbedding, input: zml.Tensor, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{input}, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model.model.embed_tokens, parameters.prefill_tokens, shardings });
    errdefer if (embed_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var embed_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: zml.nn.TokenEmbedding, input: zml.Tensor, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{input}, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model.model.embed_tokens, parameters.decode_tokens, shardings });
    errdefer if (embed_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // layer
    const layer_module = llama_model.model.layers[0];

    var layer_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: llama.TransformerLayer, hidden: zml.Tensor, idx: zml.Tensor, kv: llama.KvCache, param: LlamaParameters, shardings_: []zml.sharding.Sharding) !zml.Exe {
            log.info(">>>>>>>>>>>>>>>>> hidden: {f}", .{hidden});
            return plt.compile(alloc, io_, mod, .forward, .{ hidden, idx, kv, param.attention_metadata, param.attention_parameters }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, layer_module, hidden_states_prefill, parameters.token_index, parameters.kv_cache, parameters, shardings });
    errdefer if (layer_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var layer_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: llama.TransformerLayer, hidden: zml.Tensor, idx: zml.Tensor, kv: llama.KvCache, param: LlamaParameters, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{ hidden, idx, kv, param.attention_metadata, param.attention_parameters }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, layer_module, hidden_states_decode, parameters.token_index, parameters.kv_cache, parameters, shardings });
    errdefer if (layer_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // norm
    var norm_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: llama.RmsNorm, hidden: zml.Tensor, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{hidden}, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model.model.norm, hidden_states_prefill, shardings });
    errdefer if (norm_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var norm_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, mod: llama.RmsNorm, hidden: zml.Tensor, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, mod, .forward, .{hidden}, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model.model.norm, hidden_states_decode, shardings });
    errdefer if (norm_decode_future.cancel(io)) |v| v.deinit() else |_| {};

    // sampling
    var sampling_prefill_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, model: llama.LlamaLM, head: zml.nn.Linear, hidden: zml.Tensor, rng: zml.Tensor.Rng, opts: zml.nn.SamplingStrategy, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, model, .sampleTokens, .{ head, hidden, rng, opts }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model, llama_model.lm_head.?, hidden_states_prefill, parameters.rng, llama_model.gen_opts, shardings });
    errdefer if (sampling_prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var sampling_decode_future = io.async(struct {
        fn call(alloc: std.mem.Allocator, io_: std.Io, plt: *const zml.Platform, model: llama.LlamaLM, head: zml.nn.Linear, hidden: zml.Tensor, rng: zml.Tensor.Rng, opts: zml.nn.SamplingStrategy, shardings_: []zml.sharding.Sharding) !zml.Exe {
            return plt.compile(alloc, io_, model, .sampleTokens, .{ head, hidden, rng, opts }, .{ .partitioner = partitioner, .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, llama_model, llama_model.lm_head.?, hidden_states_decode, parameters.rng, llama_model.gen_opts, shardings });
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
    embedding_prefill_exe: zml.Exe,
    embedding_decode_exe: zml.Exe,
    layer_prefill_exe: zml.Exe,
    layer_decode_exe: zml.Exe,
    norm_prefill_exe: zml.Exe,
    norm_decode_exe: zml.Exe,
    sampling_prefill_exe: zml.Exe,
    sampling_decode_exe: zml.Exe,
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
    tp_sharding: zml.sharding.Sharding,
) !void {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    defer allocator.free(prompt_tok);

    const hidden_dim = config.hidden_size;
    const dtype = .bf16; // todo

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const max_seq_len = options.max_seq_len;

    // init RNG and buffers
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
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
        const idx_buf = try zml.Buffer.scalar(io, platform, @as(i32, @intCast(j)), .u32, replicated_sharding);
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

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, replicated_sharding);
    defer current_token_buffer.deinit();

    const decode_hidden_shape = zml.Shape.init(.{ .s = 1, .d = hidden_dim }, dtype)
        .withTags(.{ .s, .d })
        .withPartitioning(.{ .d = .model });
    var decode_hidden_buffer = try zml.Buffer.uninitialized(io, platform, decode_hidden_shape, tp_sharding, .{});
    defer decode_hidden_buffer.deinit();

    {
        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .i32));
        defer prefill_tokens_slice.free(allocator);

        // Secure copy with cast to i32 because sampling output is i32
        @memset(prefill_tokens_slice.items(i32), 0);
        const prompt_i32 = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(prompt_tok));
        @memcpy(prefill_tokens_slice.items(i32)[0..prompt_tok.len], prompt_i32);

        // Prepare buffers
        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();
        var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, replicated_sharding);
        defer prefill_token_pos_buffer.deinit();

        const prefill_hidden_shape = zml.Shape.init(.{ .s = max_seq_len, .d = hidden_dim }, dtype)
            .withTags(.{ .s, .d })
            .withPartitioning(.{ .d = .model });
        var prefill_hidden_buffer = try zml.Buffer.uninitialized(io, platform, prefill_hidden_shape, tp_sharding, .{});
        defer prefill_hidden_buffer.deinit();

        log.info("prefill_hidden_buffer before: {f}", .{prefill_hidden_buffer});

        // Execute embedding, layers, norm and sampling
        var embedding_args = try embedding_prefill_exe.args(allocator);
        defer embedding_args.deinit(allocator);
        var embedding_results = try embedding_prefill_exe.results(allocator);
        defer embedding_results.deinit(allocator);

        embedding_args.set(.{ llama_buffers.model.embed_tokens, prefill_tokens_buffer });
        embedding_prefill_exe.callOpts(io, embedding_args, &embedding_results, .{ .wait = true });
        embedding_results.fill(.{&prefill_hidden_buffer});

        var layer_args = try layer_prefill_exe.args(allocator);
        defer layer_args.deinit(allocator);
        var layer_results = try layer_prefill_exe.results(allocator);
        defer layer_results.deinit(allocator);

        for (llama_buffers.model.layers, 0..) |layer_weights, i| {
            kv_layer_buffers[i].k = kv_cache_buffers.k;
            kv_layer_buffers[i].v = kv_cache_buffers.v;
            log.info("prefill_hidden_shape: {f}", .{prefill_hidden_shape});
            log.info("prefill_hidden_buffer: {f}", .{prefill_hidden_buffer});
            layer_args.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_pos_buffer, kv_layer_buffers[i], attention_metadata_buffers });
            log.info("Running layer {d}/{d}", .{ i + 1, config.num_hidden_layers });
            layer_prefill_exe.callOpts(io, layer_args, &layer_results, .{ .wait = true });
            log.info("Completed layer {d}/{d}", .{ i + 1, config.num_hidden_layers });
            layer_results.fill(.{ &prefill_hidden_buffer, kv_cache_buffers });
            log.info("Updated prefill_hidden_buffer: {f}", .{prefill_hidden_buffer});
        }

        var norm_args = try norm_prefill_exe.args(allocator);
        defer norm_args.deinit(allocator);
        var norm_results = try norm_prefill_exe.results(allocator);
        defer norm_results.deinit(allocator);

        norm_args.set(.{ llama_buffers.model.norm, prefill_hidden_buffer });
        norm_prefill_exe.callOpts(io, norm_args, &norm_results, .{ .wait = true });
        norm_results.fill(.{&prefill_hidden_buffer});

        var sampling_args = try sampling_prefill_exe.args(allocator);
        defer sampling_args.deinit(allocator);
        var sampling_results = try sampling_prefill_exe.results(allocator);
        defer sampling_results.deinit(allocator);

        var full_sequence_output_buffer = try zml.Buffer.uninitialized(io, platform, zml.Shape.init(.{ .s = max_seq_len }, .u32), replicated_sharding, .{});
        defer full_sequence_output_buffer.deinit();

        sampling_args.set(.{ llama_buffers, llama_buffers.lm_head, prefill_hidden_buffer, rng_buffers });
        sampling_prefill_exe.callOpts(io, sampling_args, &sampling_results, .{ .wait = true });
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

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, replicated_sharding);
        defer token_pos_buffer.deinit();

        embedding_decode_args.set(.{ llama_buffers.model.embed_tokens, current_token_buffer });
        embedding_decode_exe.callOpts(io, embedding_decode_args, &embedding_decode_results, .{ .wait = true });
        embedding_decode_results.fill(.{&decode_hidden_buffer});

        for (llama_buffers.model.layers, 0..) |layer_weights, j| {
            kv_layer_buffers[j].k = kv_cache_buffers.k;
            kv_layer_buffers[j].v = kv_cache_buffers.v;
            layer_decode_args.set(.{ layer_weights, decode_hidden_buffer, token_pos_buffer, kv_layer_buffers[j], attention_metadata_buffers });
            layer_decode_exe.callOpts(io, layer_decode_args, &layer_decode_results, .{ .wait = true });
            layer_decode_results.fill(.{ &decode_hidden_buffer, kv_cache_buffers });
        }

        norm_decode_args.set(.{ llama_buffers.model.norm, decode_hidden_buffer });
        norm_decode_exe.callOpts(io, norm_decode_args, &norm_decode_results, .{ .wait = true });
        norm_decode_results.fill(.{&decode_hidden_buffer});

        sampling_decode_args.set(.{ llama_buffers, llama_buffers.lm_head, decode_hidden_buffer, rng_buffers });
        sampling_decode_exe.callOpts(io, sampling_decode_args, &sampling_decode_results, .{ .wait = true });
        sampling_decode_results.fill(.{ &current_token_buffer, &rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }

    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
