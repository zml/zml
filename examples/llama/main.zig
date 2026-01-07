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

    const topology_description = try platform.pjrt_client.topologyDescription(platform.pjrt_api);

    const layout_extension = platform.layoutExtension() orelse return error.LayoutExtensionUnavailable;

    {
        const client_default_layout = try layout_extension.clientGetDefaultLayout(platform.pjrt_api, .{ .client = platform.pjrt_client, .type = zml.pjrt.BufferType.bf16, .dims = &.{ 16, 512, 8, 64 } });
        defer client_default_layout.deinit(layout_extension, platform.pjrt_api);

        const serialized_layout = try client_default_layout.serializeAlloc(allocator, layout_extension, platform.pjrt_api);
        defer allocator.free(serialized_layout);

        std.log.info("Client default layout: {s}", .{serialized_layout});
    }

    {
        const topology_default_layout = try layout_extension.topologyGetDefaultLayout(platform.pjrt_api, .{ .topology_description = topology_description, .type = zml.pjrt.BufferType.bf16, .dims = &.{ 16, 512, 8, 64 } });
        defer topology_default_layout.deinit(layout_extension, platform.pjrt_api);

        const serialized_layout = try topology_default_layout.serializeAlloc(allocator, layout_extension, platform.pjrt_api);
        defer allocator.free(serialized_layout);

        std.log.info("Topology default layout: {s}", .{serialized_layout});
    }

    var compiled_model_result_future = io.async(compileModel, .{ allocator, io, platform, llama_model, llama_parameters, layout_extension });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.prefill_exe.deinit();
        defer v.decode_exe.deinit();
    } else |_| {};

    var llama_buffers_future = io.async(loadModelBuffers, .{ allocator, io, platform, &store, llama_model });
    errdefer b: {
        var v = llama_buffers_future.cancel(io) catch break :b;
        LlamaLM.unloadBuffers(&v, allocator);
    }

    const compiled_model_result = try compiled_model_result_future.await(io);
    defer compiled_model_result.prefill_exe.deinit();
    defer compiled_model_result.decode_exe.deinit();

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
        compiled_model_result.prefill_exe,
        compiled_model_result.decode_exe,
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
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, llama_model: llama.LlamaLM, parameters: LlamaParameters, layout_extension: *const zml.pjrt.layout.LayoutExtension) !CompileModelResult {
    var timer = try std.time.Timer.start();
    log.info("Compiling model", .{});
    defer log.info("Compiled model [{D}]", .{timer.read()});

    // Compile the model twice, one for prefill, one for generation.
    var prefill_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling prefill", .{});
            defer log.info("Compiled prefill [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, llama_model_, .forward, .{ parameters_.prefill_tokens, parameters_.token_index, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters) !zml.Exe {
            var timer_ = try std.time.Timer.start();
            log.info("Compiling decode", .{});
            defer log.info("Compiled decode [{D}]", .{timer_.read()});
            return platform_.compile(allocator_, io_, llama_model_, .forward, .{ parameters_.decode_tokens, parameters_.token_index, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    {
        const exe: *zml.pjrt.Executable = try prefill_exe.exe.getExecutable(platform.pjrt_api);
        defer exe.deinit(platform.pjrt_api);

        const output_layouts = try layout_extension.executableGetOutputLayouts(platform.pjrt_api, exe);
        for (output_layouts, 0..) |layout, index| {
            const serialized_layout = try layout.serializeAlloc(allocator, layout_extension, platform.pjrt_api);
            defer allocator.free(serialized_layout);

            std.log.info("Prefill exe output #{d} layout: {s}", .{ index, serialized_layout });
        }
    }

    {
        const exe: *zml.pjrt.Executable = try decode_exe.exe.getExecutable(platform.pjrt_api);
        defer exe.deinit(platform.pjrt_api);

        const output_layouts = try layout_extension.executableGetOutputLayouts(platform.pjrt_api, exe);
        for (output_layouts, 0..) |layout, index| {
            const serialized_layout = try layout.serializeAlloc(allocator, layout_extension, platform.pjrt_api);
            defer allocator.free(serialized_layout);

            std.log.info("Decode exe output #{d} layout: {s}", .{ index, serialized_layout });
        }
    }

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
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
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
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

        prefill_args.set(.{ llama_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers });

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
        const token_pos_slice: zml.ConstSlice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        const token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice);
        defer token_pos_buffer.deinit();

        // call to generate the next token
        decode_args.set(.{ llama_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers });

        decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        // extract the generated token from the buffer
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const duration = timer.read();
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration, duration.div(num_tokens_generated).hzFloat() });
}
