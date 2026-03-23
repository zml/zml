const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;
const flashinfer_moe = zml.flashinfer_moe;

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
    moe_backend: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    log.info("GptOss was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;
    const io = init.io;

    const args: Args = blk: {
        var ret: Args = .{};
        var it = init.minimal.args.iterate();
        _ = it.next(); // program name
        while (it.next()) |arg| {
            if (std.mem.startsWith(u8, arg, "--model=")) {
                ret.model = arg["--model=".len..];
            } else if (std.mem.startsWith(u8, arg, "--seqlen=")) {
                ret.seqlen = try std.fmt.parseUnsigned(u32, arg["--seqlen=".len..], 10);
            } else if (std.mem.startsWith(u8, arg, "--moe_backend=")) {
                ret.moe_backend = arg["--moe_backend=".len..];
            } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
                ret.prompt = arg["--prompt=".len..];
            }
        }

        if (ret.model == null) {
            log.err("Missing --model", .{});
            return;
        }

        break :blk ret;
    };

    const prompt = if (args.prompt) |p| p else return error.NoPrompt;

    log.info("Resolving Model repo", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model.?);

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const gpt_options: gpt_oss.GptOss.Options = .{
        .max_seq_len = args.seqlen,
        .max_prompt_len = args.seqlen,
        .tokens_per_expert_ratio = 1.0,
        .sampling_strategy = .{
            .topk = 2,
            .temperature = 0.8,
        },
    };

    const gpt_model: gpt_oss.GptOss = try .init(allocator, store.view(), config, gpt_options);
    defer gpt_model.deinit(allocator);

    const dtype = gpt_model.model.embed_tokens.weight.dtype();

    var tokenizer_future = io.async(loadTokenizer, .{ allocator, io, repo });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    log.info("Selecting platform", .{});
    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This example requires CUDA.", .{});
        return;
    }

    log.info("Selecting backend", .{});

    const moe_backend: zml.moe.Backend = if (args.moe_backend) |name| b: {
        if (std.mem.eql(u8, name, "flashinfer")) break :b .flashinfer;
        if (std.mem.eql(u8, name, "triton")) break :b .triton;
        log.err("Unknown MoE backend: {s}", .{name});
        return;
    } else try zml.moe.Backend.auto(platform, dtype);

    log.info("Selected MoE backend: {}", .{moe_backend});

    moe_backend.load(allocator) catch |err| {
        log.err("Failed to load MoE backend: {}", .{err});
        return err;
    };

    moe_backend.register(platform) catch |err| {
        log.err("Failed to register MoE backend: {}", .{err});
        return err;
    };

    const tp_mesh: zml.sharding.LogicalMesh = try .init("tp_mesh", .{ .model = .high_bandwidth });
    const tp_strategy: zml.sharding.Strategy = try .suggest(tp_mesh, platform.physical_mesh);
    const sharding_tp: zml.sharding.Sharding = try .initFromStrategy(platform, tp_mesh, tp_strategy);

    const gpt_parameters: GptOssParameters = .{
        .prefill_tokens = .init(.{ .s = gpt_options.max_seq_len }, .u32),
        .decode_tokens = .init(.{ .s = 1 }, .u32),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(.init(.{
            .layer = gpt_model.model.layers.len,
            .k = args.seqlen,
            .h = config.num_key_value_heads,
            .hd = config.head_dim,
        }, dtype)),
        .rng = .init(),
        .tokens_mask = .init(.{ .s = gpt_options.max_seq_len }, .bool),
        .moe_metadata = .init(.fromBackend(moe_backend)),
        .moe_parameters = .init(.fromBackend(moe_backend)),
    };

    var progress = std.Progress.start(io, .{ .root_name = args.model.? });
    const compiled_model_result = try compileModel(allocator, io, platform, gpt_model, gpt_parameters, &.{sharding_tp});
    log.info("Compiled executables ready", .{});
    defer compiled_model_result.prefill_exe.deinit();
    defer compiled_model_result.decode_exe.deinit();

    log.info("Loading model buffers", .{});
    var gpt_buffers = try GptOss.load(&gpt_model, allocator, io, platform, &store, &.{sharding_tp}, &progress);
    log.info("Loaded buffers ready", .{});
    defer GptOss.unloadBuffers(&gpt_buffers, allocator);
    switch (moe_backend) {
        .flashinfer => {
            try gpt_oss.preprocessFlashinferSm90Mxfp4(allocator, io, platform, &gpt_buffers);
            log.info("Static SM90 MXFP4 preprocessing completed (flashinfer)", .{});
        },
        .triton => {
            try gpt_oss.preprocessTritonSm90Mxfp4(allocator, io, platform, &gpt_buffers);
            log.info("Static SM90 MXFP4 preprocessing completed (triton)", .{});
        },
    }
    log.info("Ending progress node", .{});
    progress.end();

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try gpt_parameters.kv_cache.initBuffer(io, platform, sharding_tp);
    defer gpt_oss.KvCache.deinitBuffer(&kv_cache_buffers);

    var moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata) = try gpt_parameters.moe_metadata.initBuffer(io, platform);
    defer zml.moe.Metadata.deinitBuffer(&moe_metadata_buffers);

    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

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
        &moe_metadata_buffers,
        tokenizer,
        config,
        gpt_options,
        @intCast(std.Io.Clock.now(.awake, io).toNanoseconds()),
        prompt_tok,
        &stdout.interface,
        platform,
        sharding_tp,
    );
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(GptOss.Config) {
    const now: std.Io.Timestamp = .now(io, .awake);
    log.info("Loading model config", .{});
    defer log.info("Loaded model config [{f}]", .{now.untilNow(io, .awake)});

    var parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(gpt_oss.GptOss.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();
    parsed_config.value.rope_scaling.setRopeTheta(parsed_config.value.rope_theta);

    return parsed_config;
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    log.info("Loading tokenizer", .{});
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Loaded tokenizer [{f}]", .{now.untilNow(io, .awake)});
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
    moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, gpt_model: gpt_oss.GptOss, parameters: GptOssParameters, shardings: []const zml.sharding.Sharding) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    log.info("Compiling model", .{});
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    // Compile the model twice, one for prefill, one for generation.
    var prefill_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters, shardings_: []const zml.sharding.Sharding) !zml.Exe {
            const now_: std.Io.Timestamp = .now(io_, .awake);
            log.info("Compiling prefill", .{});
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{
                parameters_.prefill_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                parameters_.tokens_mask,
                parameters_.moe_metadata,
                parameters_.moe_parameters,
            }, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters, shardings });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *zml.Platform, gpt_model_: gpt_oss.GptOss, parameters_: GptOssParameters, shardings_: []const zml.sharding.Sharding) !zml.Exe {
            const now_: std.Io.Timestamp = .now(io_, .awake);
            log.info("Compiling decode", .{});
            defer log.info("Compiled decode [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, gpt_model_, .forward, .{
                parameters_.decode_tokens,
                parameters_.token_index,
                parameters_.kv_cache,
                parameters_.rng,
                null,
                parameters_.moe_metadata,
                parameters_.moe_parameters,
            }, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, gpt_model, parameters, shardings });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
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
    moe_metadata_buffers: *zml.Bufferized(zml.moe.Metadata),
    tokenizer: zml.tokenizer.Tokenizer,
    config: GptOss.Config,
    options: GptOss.Options,
    seed: u128,
    prompt_tok: []const u32,
    writer: *std.Io.Writer,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
) !void {
    var tokenizer_decoder = try tokenizer.decoder();

    const max_seq_len = options.max_seq_len;
    // init RNG and buffers
    var rng_buffers = try zml.Tensor.Rng.initBuffer(
        platform,
        seed,
        io,
        sharding,
    );
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

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
        defer prefill_token_pos_buffer.deinit();

        const tokens_mask: zml.Slice = try .alloc(allocator, .init(.{max_seq_len}, .bool));
        defer tokens_mask.free(allocator);
        @memset(tokens_mask.items(bool)[0..prompt_tok.len], true);
        @memset(tokens_mask.items(bool)[prompt_tok.len..max_seq_len], false);

        var tokens_mask_buffer: zml.Buffer = try .fromSlice(io, platform, tokens_mask, sharding);
        defer tokens_mask_buffer.deinit();

        prefill_args.set(.{ gpt_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers, tokens_mask_buffer, moe_metadata_buffers });

        prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{
            &prefill_tokens_buffer,
            kv_cache_buffers,
            &rng_buffers,
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
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    const now: std.Io.Timestamp = .now(io, .awake);

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
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();

        // call to generate the next token
        decode_args.set(.{ gpt_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers, moe_metadata_buffers });

        decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        // extract the generated token from the buffer
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const duration = now.untilNow(io, .awake);
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {f}: {:.3}tok/s", .{
        num_tokens_generated,
        duration,
        stdx.Io.Duration.hzFloat(stdx.Io.Duration.div(duration, num_tokens_generated)),
    });
}
