const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const gpt_oss = @import("gpt_oss.zig");
const GptOss = gpt_oss.GptOss;
const KvCache = gpt_oss.KvCache;

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

    // Write metadata from the config file into the GptOss struct.
    const gpt_options: gpt_oss.GptOss.Options = .{
        .max_seq_len = args.seqlen,
        .max_prompt_len = args.seqlen,
        .tokens_per_expert_ratio = 1.0,
        .sampling_strategy = .{
            .topk = 5,
            .temperature = 0.8,
        },
    };

    // Initialize the GptOss struct and map the content of the .safetensors to the model tensors.
    const gpt_model: gpt_oss.GptOss = try .init(allocator, store.view(), config, gpt_options);
    defer gpt_model.deinit(allocator);

    // Specify shapes of input arguments
    const dtype = gpt_model.model.embed_tokens.weight.dtype();

    // We update GptParameters to handle the Mode union
    const gpt_parameters: GptOssParameters = .{
        .prefill_tokens = .init(.{ .s = gpt_options.max_seq_len }, .u32),
        .decode_tokens = .init(.{ .s = 1 }, .u32),
        // Define shape for the Mode union
        .mode_prefill = .{ .prefill = .init(.{}, .u32) }, // Scalar prompt length
        .mode_gen = .{ .gen = .init(.{ .s = 1 }, .u32) }, // Scalar token index

        .kv_cache = .init(.init(.{
            .layer = gpt_model.model.layers.len,
            .k = args.seqlen,
            .h = config.num_key_value_heads,
            .hd = config.head_dim,
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

    var compiled_model_result_future = io.async(compileModel, .{ allocator, io, platform, gpt_model, gpt_parameters });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.prefill_exe.deinit();
        defer v.decode_exe.deinit();
    } else |_| {};

    var gpt_buffers_future = io.async(loadModelBuffers, .{ allocator, io, platform, &store, gpt_model });

    const compiled_model_result = try compiled_model_result_future.await(io);
    defer compiled_model_result.prefill_exe.deinit();
    defer compiled_model_result.decode_exe.deinit();

    const gpt_buffers = try gpt_buffers_future.await(io);

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try gpt_parameters.kv_cache.initBuffer(io, platform);
    defer gpt_oss.KvCache.deinitBuffer(&kv_cache_buffers);

    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    const prompt = blk: {
        var reader = std.Io.File.stdin().reader(io, &.{});
        break :blk try reader.interface.allocRemaining(allocator, .unlimited);
    };
    defer allocator.free(prompt);

    log.info("✅\tPrompt: {s}", .{prompt});

    var stdout = std.Io.File.stdout().writer(io, &.{});

    const prompt_tok_buf = try allocator.alloc(u32, gpt_options.max_seq_len);
    defer allocator.free(prompt_tok_buf);
    const prompt_tok: []const u32 = try tokenizePrompt(tokenizer, prompt, false, prompt_tok_buf);
    log.info("Generated prompt tokens: {any}", .{prompt_tok});
    defer allocator.free(prompt_tok);

    try generateText(
        allocator,
        io,
        gpt_buffers,
        compiled_model_result.prefill_exe,
        compiled_model_result.decode_exe,
        &kv_cache_buffers,
        tokenizer,
        config,
        gpt_options,
        @intCast((try std.Io.Clock.now(.real, io)).toNanoseconds()),
        prompt_tok,
        false,
        &stdout.interface,
        platform,
    );
}

// ... parseConfig, loadTokenizer remain the same ...

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
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

const GptOssParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    mode_prefill: GptOss.Mode,
    mode_gen: GptOss.Mode,
    kv_cache: gpt_oss.KvCache,
    rng: zml.Tensor.Rng,
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, gpt_model: gpt_oss.GptOss, parameters: GptOssParameters) !CompileModelResult {
    var timer = try std.time.Timer.start();
    log.info("Compiling model", .{});
    defer log.info("Compiled model [{D}]", .{timer.read()});

    // Compile prefill with Mode.prefill
    var prefill_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling prefill", .{});
            defer log.info("Compiled prefill [{D}]", .{timer_.read()});
            // We pass parameters.mode_prefill here
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{ parameters_.prefill_tokens, parameters_.mode_prefill, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    // Compile decode with Mode.gen
    var decode_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling decode", .{});
            defer log.info("Compiled decode [{D}]", .{timer_.read()});
            // We pass parameters.mode_gen here
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{ parameters_.decode_tokens, parameters_.mode_gen, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

fn loadModelBuffers(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, store: *zml.io.TensorStore, gpt_model: gpt_oss.GptOss) !zml.Bufferized(gpt_oss.GptOss) {
    var timer = try std.time.Timer.start();
    log.info("Loading model", .{});
    defer log.info("Loaded model [{D}]", .{timer.read()});
    return gpt_model.loadBuffers(allocator, io, store.view(), platform);
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
    tokenizer: zml.tokenizer.Tokenizer,
    config: GptOss.Config,
    options: GptOss.Options,
    seed: u128,
    prompt_tok: []const u32,
    skip_llama3_encoding: bool,
    writer: *std.Io.Writer,
    platform: zml.Platform,
) !void {
    _ = skip_llama3_encoding;
    log.info("generateText START", .{});
    log.info("eos token {}", .{config.eos_token_id});

    var tokenizer_decoder = try tokenizer.decoder();

    const max_seq_len = options.max_seq_len;

    log.info("generateText init RNG", .{});
    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    log.info("generateText init generated_token_slice", .{});
    // This buffer holds the single token generated at each step
    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    // This buffer holds the single next-token output on device
    var current_token_buffer = try zml.Buffer.uninitialized(io, platform, zml.Shape.init(.{ .s = 1 }, .u32), .{});
    defer current_token_buffer.deinit();

    log.info("generateText prefill START", .{});
    {
        var prefill_args = try prefill_exe.args(allocator);
        defer prefill_args.deinit(allocator);

        var prefill_results = try prefill_exe.results(allocator);
        defer prefill_results.deinit(allocator);

        // Input tokens for prefill
        const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .u32));
        defer prefill_tokens_slice.free(allocator);
        @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice);
        defer prefill_tokens_buffer.deinit();

        log.info("generateText prefill prompt_len buffer", .{});
        // We pass prompt length for prefill logic
        var prefill_len_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(prompt_tok.len)), .u32);
        defer prefill_len_buffer.deinit();

        log.info("generateText prefill set args", .{});
        // Args correspond to: tokens, mode, kv, rng.
        // prefill_len_buffer corresponds to Mode.prefill
        prefill_args.set(.{ gpt_buffers, prefill_tokens_buffer, prefill_len_buffer, kv_cache_buffers, rng_buffers });

        log.info("generateText prefill call", .{});
        prefill_exe.call(prefill_args, &prefill_results);

        log.info("generateText prefill results fill", .{});
        // The output of prefill is {next_token, kv, rng}.
        // We put next_token into current_token_buffer.
        prefill_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        // Retrieve the generated token to print it
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    log.info("generateText prefill DONE", .{});
    log.info("first token generated: {}", .{generated_token_slice.items(u32)[0]});

    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);

    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    var timer = try stdx.time.Timer.start();

    var num_tokens_generated: usize = 1;

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

        const token_pos_slice: zml.ConstSlice = .init(zml.Shape.init(.{ .s = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        const token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice);
        defer token_pos_buffer.deinit();

        // Mode.gen corresponds to token_pos_buffer
        decode_args.set(.{ gpt_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers });

        decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
