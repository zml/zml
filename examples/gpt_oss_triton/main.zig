const std = @import("std");

const zml = @import("zml");
const moe_triton = zml.moe_triton;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;
const MoeTriton = moe_triton.MoeTriton;

const gpt_oss = @import("gpt_oss_triton.zig");
const GptOss = gpt_oss.GptOss;
const KvCache = gpt_oss.KvCache;

// const cublas_gg = zml.cublas_grouped_gemm;
// const GemmGroupedBatched = cublas_gg.GemmGroupedBatched;

const log = std.log.scoped(.gpt_oss);

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Args = struct {
    model: ?[]const u8 = null,
    seqlen: u32 = 512,
};

pub fn main() !void {
    log.info("GptOss was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    // zml.init();
    // defer zml.deinit();

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
    const repo = try zml.safetensors.resolveModelRepo(io, args.model.?);
    //defer repo.deinit(allocator, io);

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // Write metadata from the config file into the GptOss struct.
    const gpt_options: gpt_oss.GptOss.Options = .{
        .max_seq_len = args.seqlen,
        .max_prompt_len = args.seqlen,
        .tokens_per_expert_ratio = 1.0,
        .sampling_strategy = .{
            .topk = 2,
            .temperature = 0.8,
        },
    };

    // Initialize the GptOss struct and map the content of the .safetensors to the model tensors.
    const gpt_model: gpt_oss.GptOss = try .init(allocator, store.view(), config, gpt_options);
    defer gpt_model.deinit(allocator);

    // Specify shapes of input arguments
    const dtype = gpt_model.model.embed_tokens.weight.dtype();
    const gpt_parameters: GptOssParameters = .{ .prefill_tokens = .init(.{ .s = gpt_options.max_seq_len }, .u32), .decode_tokens = .init(.{ .s = 1 }, .u32), .token_index = .init(.{}, .u32), .kv_cache = .init(.init(.{
        .layer = gpt_model.model.layers.len,
        .k = args.seqlen,
        .h = config.num_key_value_heads,
        .hd = config.head_dim,
    }, dtype)), .rng = .init(), .tokens_mask = .init(.{ .s = gpt_options.max_seq_len }, .bool), .metadata = .init(.{ .group_count = gpt_model.config.num_local_experts }) };

    var tokenizer_future = io.async(loadTokenizer, .{ allocator, io, repo });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    var platform: zml.Platform = try .auto(allocator, io, .{});
    platform = platform.withCompilationOptions(.{ .xla_dump_to = "/tmp/zml/gpt_oss/" });
    defer platform.deinit();
    var it = platform.devicesIterator();
    log.info("Devices:", .{});
    while (it.next()) |device| {
        log.info("\t- {f}", .{device});
    }

    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This example requires CUDA.", .{});
        return;
    }

    try MoeTriton.register(platform);
    try moe_triton.load(allocator, io);

    var compiled_model_result_future = io.async(compileModel, .{ allocator, io, platform, gpt_model, gpt_parameters });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.prefill_exe.deinit();
        defer v.decode_exe.deinit();
    } else |_| {};

    var load_model_buffers: zml.stdx.Io.AllocatingLimitedConcurrentGroup = try .init(allocator, threaded.async_limit.toInt() orelse 16);
    defer {
        load_model_buffers.cancel(io);
        load_model_buffers.deinit();
    }

    var read_pool, var write_pool = try zml.io.ConcurrentBufferPool.initRW(
        allocator,
        platform,
        .{
            .size = (64 * 1024 * 1024),
            .concurrency = load_model_buffers.limit,
            .dma = true,
        },
        .{
            .size = (64 * 1024 * 1024),
            .concurrency = load_model_buffers.limit,
            .dma = true,
        },
    );
    defer {
        read_pool.deinit();
        write_pool.deinit();
    }

    var progress = std.Progress.start(io, .{ .root_name = "gpt_oss", .estimated_total_items = store.view().count() });

    var gpt_buffers_future = io.async(loadModelBuffers, .{ allocator, io, &progress, &load_model_buffers, &read_pool, &write_pool, platform, &store, gpt_model });
    errdefer b: {
        var v = gpt_buffers_future.cancel(io) catch break :b;
        GptOss.unloadBuffers(&v, allocator);
    }

    const compiled_model_result = try compiled_model_result_future.await(io);
    defer compiled_model_result.prefill_exe.deinit();
    defer compiled_model_result.decode_exe.deinit();

    var gpt_buffers = try gpt_buffers_future.await(io);
    defer GptOss.unloadBuffers(&gpt_buffers, allocator);
    progress.end();

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try gpt_parameters.kv_cache.initBuffer(io, platform);
    defer gpt_oss.KvCache.deinitBuffer(&kv_cache_buffers);

    var host_buffer = try zml.Buffer.uninitialized(io, platform, gpt_parameters.metadata.host_buffer.shape(), .{ .memory = .host_pinned });
    log.info("HOST BUFER {f}", .{host_buffer});
    var device_buffer = try zml.Buffer.uninitialized(io, platform, gpt_parameters.metadata.device_buffer.shape(), .{ .memory = .device });
    defer host_buffer.deinit();
    defer device_buffer.deinit();
    // var metadata_buffers = try gpt_parameters.metadata.initBuffer(io, platform);
    // defer cublas_gg.GemmGroupedBatched.Metadata.deinitBuffer(&metadata_buffers);

    log.info("metadata host_buffer host ptr=0x{x}", .{@intFromPtr(host_buffer.devicePtr())});

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

    const prompt_tok_buf = try allocator.alloc(u32, gpt_options.max_seq_len);
    defer allocator.free(prompt_tok_buf);
    const prompt_tok: []const u32 = try tokenizePrompt(tokenizer, prompt, false, prompt_tok_buf);

    try generateText(
        allocator,
        io,
        gpt_buffers,
        compiled_model_result.prefill_exe,
        compiled_model_result.decode_exe,
        &kv_cache_buffers,
        &host_buffer,
        &device_buffer,
        tokenizer,
        config,
        gpt_options,
        @intCast((try std.Io.Clock.now(.real, io)).toNanoseconds()),
        prompt_tok,
        &stdout.interface,
        platform,
    );
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(GptOss.Config) {
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
        break :blk try std.json.parseFromTokenSource(gpt_oss.GptOss.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
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
    errdefer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

const GptOssParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: gpt_oss.KvCache,
    rng: zml.Tensor.Rng,
    tokens_mask: zml.Tensor,
    metadata: moe_triton.MoeTriton.Metadata,
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, gpt_model: gpt_oss.GptOss, parameters: GptOssParameters) !CompileModelResult {
    var timer = try std.time.Timer.start();
    log.info("Compiling model", .{});
    defer log.info("Compiled model [{D}]", .{timer.read()});

    // Compile the model twice, one for prefill, one for generation.
    var prefill_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling prefill", .{});
            defer log.info("Compiled prefill [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{
                parameters_.prefill_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                parameters_.tokens_mask,
                parameters_.metadata.host_buffer,
                parameters_.metadata.device_buffer,
            });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling decode", .{});
            defer log.info("Compiled decode [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{
                parameters_.decode_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                null,
                parameters_.metadata.host_buffer,
                parameters_.metadata.device_buffer,
            });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

fn loadModelBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    progress: *std.Progress.Node,
    group: *zml.stdx.Io.AllocatingLimitedConcurrentGroup,
    read_pool: *zml.io.ConcurrentBufferPool,
    write_pool: *zml.io.ConcurrentBufferPool,
    platform: zml.Platform,
    store: *zml.io.TensorStore,
    gpt_model: gpt_oss.GptOss,
) !zml.Bufferized(gpt_oss.GptOss) {
    var transferred_bytes: usize = 0;
    var timer = try stdx.time.Timer.start();
    log.info("Loading model", .{});
    defer {
        const duration = timer.read();
        const seconds = @as(f64, @floatFromInt(duration.ns)) / 1e9;
        const gb_per_sec = @as(f64, @floatFromInt(transferred_bytes)) / (1024.0 * 1024.0 * 1024.0) / seconds;
        const gbps = gb_per_sec * 8.0;
        log.info("Loaded model [{D} {d:.3} GB/s {d:.3} gbps]", .{ duration, gb_per_sec, gbps });
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const bufferize_ctx: zml.io.BufferizeContext(gpt_oss.TransferCtx) = .{
        .allocator = allocator,
        .arena = &arena,
        .io = io,
        .platform = platform,
        .cb_ctx = .{ .read_pool = read_pool, .write_pool = write_pool, .transferred_bytes = &transferred_bytes, .progress = progress },
    };

    const bufferized = gpt_model.loadBuffers(
        bufferize_ctx,
        group,
        store.view(),
        transferBuffer,
    );

    try group.await(io);

    return bufferized;
}

fn transferBuffer(ctx: zml.io.TensorBufferTransfer(gpt_oss.TransferCtx)) !void {
    const read_pool = ctx.cb_ctx.read_pool;
    const write_pool = ctx.cb_ctx.write_pool;
    const progress = ctx.cb_ctx.progress;

    const read_buffer = read_pool.acquire(ctx.io) catch unreachable;
    defer read_pool.release(ctx.io, read_buffer) catch {};

    const write_buffer = write_pool.acquire(ctx.io) catch unreachable;
    defer write_pool.release(ctx.io, write_buffer) catch {};

    var transfer_progress = progress.start(ctx.tensor.name, ctx.tensor.byteSize());

    var reader = zml.safetensors.TensorReader.init(ctx.io, ctx.tensor, read_buffer) catch unreachable;
    defer reader.deinit();

    var writer = zml.io.DeviceWriter.init(ctx.io, &transfer_progress, ctx.platform, ctx.tensor.shape, ctx.buffer, .device, write_buffer) catch unreachable;
    defer {
        writer.interface.flush() catch unreachable;
        writer.deinit();
    }

    _ = reader.interface.streamRemaining(&writer.interface) catch unreachable;

    transfer_progress.end();
    ctx.cb_ctx.transferred_bytes.* += ctx.tensor.byteSize();
}

pub fn tokenizePrompt(tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, no_chat: bool, out: []u32) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    if (no_chat) {
        const tokens = try encoder.encode(prompt);
        if (tokens.len > out.len) return error.PromptTooLong;
        @memcpy(out[0..tokens.len], tokens);
        return out[0..tokens.len];
    }

    const start_header = tokenizer.tokenToId("<|start|>") orelse return error.NoSuchToken;
    const end_header_start_message = tokenizer.tokenToId("<|message|>") orelse return error.NoSuchToken;
    const end_message = tokenizer.tokenToId("<|end|>") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = .initBuffer(out);

    const system_prompt = try encoder.encode("You are ChatGPT, a large language model trained by OpenAI.\n");
    if (system_prompt.len + 4 > tokens.unusedCapacitySlice().len) return error.PromptTooLong;
    tokens.appendSliceAssumeCapacity(&.{ start_header, tokenizer.tokenToId("system").?, end_header_start_message });
    tokens.appendSliceAssumeCapacity(system_prompt);
    tokens.appendAssumeCapacity(end_message);

    const user_prompt = try encoder.encode(prompt);
    if (user_prompt.len + 9 > tokens.unusedCapacitySlice().len) return error.PromptTooLong;
    tokens.appendSliceAssumeCapacity(&.{ start_header, tokenizer.tokenToId("user").?, end_header_start_message });
    tokens.appendSliceAssumeCapacity(user_prompt);
    tokens.appendSliceAssumeCapacity(&.{
        end_message,
        start_header,
        tokenizer.tokenToId("assistant").?,
        tokenizer.tokenToId("<|channel|>") orelse return error.NoSuchToken,
        tokenizer.tokenToId("analysis") orelse return error.NoSuchToken,
        end_header_start_message,
    });

    return tokens.items;
}

pub fn generateText(
    allocator: std.mem.Allocator,
    io: std.Io,
    gpt_buffers: zml.Bufferized(GptOss),
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
    kv_cache_buffers: *zml.Bufferized(gpt_oss.KvCache),
    host_buffer: *zml.Bufferized(Tensor),
    device_buffer: *zml.Bufferized(Tensor),
    tokenizer: zml.tokenizer.Tokenizer,
    config: GptOss.Config,
    options: GptOss.Options,
    seed: u128,
    prompt_tok: []const u32,
    writer: *std.Io.Writer,
    platform: zml.Platform,
) !void {
    var tokenizer_decoder = try tokenizer.decoder();

    const max_seq_len = options.max_seq_len;

    // init RNG and buffers
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io);
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

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32);
        defer prefill_token_pos_buffer.deinit();

        const tokens_mask: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .bool));
        defer tokens_mask.free(allocator);
        @memset(tokens_mask.items(bool)[0..prompt_tok.len], true);
        @memset(tokens_mask.items(bool)[prompt_tok.len..max_seq_len], false);

        var tokens_mask_buffer: zml.Buffer = try .fromSlice(io, platform, tokens_mask);
        defer tokens_mask_buffer.deinit();

        prefill_args.set(.{ gpt_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers, tokens_mask_buffer, host_buffer, device_buffer });

        prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{
            &prefill_tokens_buffer,
            kv_cache_buffers,
            &rng_buffers,
            host_buffer,
            device_buffer,
        });
        try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
        generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];
    }

    // Prepare for token-by-token generation,

    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);

    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // start with the token generated based on the full prompt.
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
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
        const token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice);
        defer token_pos_buffer.deinit();

        // call to generate the next token
        decode_args.set(.{ gpt_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers, host_buffer, device_buffer });

        decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers, host_buffer, device_buffer });

        // extract the generated token from the buffer
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
