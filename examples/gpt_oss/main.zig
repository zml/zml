const std = @import("std");

const async = @import("async");
const clap = @import("clap");
const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const GptOss = @import("GptOss.zig");

const log = std.log.scoped(.GptOss);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = async.logFn(std.log.defaultLog),
};

const cli_params = clap.parseParamsComptime(
    \\--help                      print this help
    \\--prompt         <STRING>   the prompt
    \\--hf-model-path  <STRING>   path to the directory containing model weights, config and tokenizer
    \\--seed           <UINT>     random seed (optional)
    \\--seq-len        <UINT>     sequence length
    \\--create-options <STRING>   platform creation options JSON, defaults to {}
    \\--no-llama3      <BOOL>     skip prompt template
    \\--sharding       <BOOL>     default: true: sharding on or off
);

pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: GptOss.Config, prompt: []const u8, skip_llama3_encoding: bool) ![]u32 {
    _ = skip_llama3_encoding; // autofix
    _ = config; // autofix
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // mine: 200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788, 17527, 558, 200007, 200006, 1428, 200008, 4827, 382, 290, 9029, 328, 10128, 30, 200007, 200006, 173781, 200008 }
    // transformer: 4827,   382,   290, 10574, 13983,    30]]
    if (true) {
        // Copy so the ownership is the same in both branches.
        return try allocator.dupe(u32, try encoder.encode(prompt));
    }

    const start_header = tokenizer.tokenToId("<|start|>") orelse return error.NoSuchToken;
    const end_header_start_message = tokenizer.tokenToId("<|message|>") orelse return error.NoSuchToken;
    const end_message = tokenizer.tokenToId("<|end|>") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    {
        try tokens.appendSlice(allocator, &.{ start_header, tokenizer.tokenToId("system").?, end_header_start_message });
        try tokens.appendSlice(allocator, try encoder.encode("You are ChatGPT, a large language model trained by OpenAI.\n"));
        try tokens.append(allocator, end_message);
    }

    {
        try tokens.appendSlice(allocator, &.{ start_header, tokenizer.tokenToId("user").?, end_header_start_message });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.append(allocator, end_message);
    }

    try tokens.appendSlice(allocator, &.{ start_header, tokenizer.tokenToId("assistant").?, end_header_start_message });

    return tokens.toOwnedSlice(allocator);
}

pub fn generateText(
    config: GptOss.Config,
    llama_: GptOss,
    mod_prefill: zml.ModuleExe(GptOss.forward),
    mod_generate: zml.ModuleExe(GptOss.forward),
    kv_cache_: zml.Bufferized(GptOss.KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seed: u128,
    prompt: []const u8,
    skip_llama3_encoding: bool,
    output: *std.Io.Writer,
) !void {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    log.info("\t Tokenized prompt: {any}", .{prompt_tok});
    defer allocator.free(prompt_tok);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const platform = mod_generate.platform();
    const max_seq_len = llama_.model.max_seq_len;

    // init RNG and buffers
    var rng = try zml.Tensor.Rng.init(platform, seed);
    var generated_token_buffer = [_]u32{undefined};

    var kv_cache = prefill: {
        // prepare device buffers for the prefill tokens and their positions
        const prefill_buffer = try allocator.alloc(u32, max_seq_len);
        @memcpy(prefill_buffer[0..prompt_tok.len], prompt_tok);

        var prefill_tokens = try zml.Buffer.fromSlice(platform, .{max_seq_len}, prefill_buffer);
        defer prefill_tokens.deinit();
        var prefill_token_pos = try zml.Buffer.scalar(platform, 0, .u32);
        defer prefill_token_pos.deinit();

        const prefilled_tokens, const kv_cache, rng = mod_prefill.call(.{ prefill_tokens, prefill_token_pos, kv_cache_, rng });
        _ = try prefilled_tokens.toHost(std.mem.sliceAsBytes(prefill_buffer));
        generated_token_buffer[0] = prefill_buffer[prompt_tok.len - 1];
        break :prefill kv_cache;
    };
    defer zml.aio.unloadBuffers(&kv_cache);

    // Prepare for token-by-token generation,
    // start with the token generated based on the full prompt.
    var current_token = try zml.Buffer.fromSlice(platform, .{1}, &generated_token_buffer);
    defer current_token.deinit();

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    const start = std.time.microTimestamp();

    // One token has alreadyh been generated by the prefill.
    var num_tokens_generated: usize = 1;

    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_buffer[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try output.writeAll(chunk);
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
        const token_pos_buffer = &[_]u32{@intCast(prompt_tok.len + i)};
        const token_pos = try zml.Buffer.fromSlice(platform, .{}, token_pos_buffer);
        defer token_pos.deinit();

        // call to generate the next token
        current_token, kv_cache, rng = mod_generate.call(.{ current_token, token_pos, kv_cache, rng });

        // extract the generated token from the buffer
        _ = try current_token.toHost(std.mem.sliceAsBytes(&generated_token_buffer));
    }
    const end = std.time.microTimestamp();
    const duration = stdx.math.divFloat(f64, end - start, std.time.us_per_s);
    const speed = @as(f64, @floatFromInt(num_tokens_generated)) / duration;

    log.info("✅ Generated {d} tokens in {:.3}s: {d:.3}tok/s", .{ num_tokens_generated, duration, speed });
}

pub fn main() !void {
    try async.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   GptOss was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.smp_allocator;
    const cli = ClapBoilerplate.parseCli(allocator);
    defer cli.deinit();

    const hf_model_path = cli.args.@"hf-model-path" orelse {
        log.err("Missing --hf-model-path", .{});
        return;
    };

    const model_config_path = try std.fs.path.join(allocator, &.{ hf_model_path, "config.json" });
    defer allocator.free(model_config_path);

    const model_weights_path = b: {
        const simple_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors" });
        if (async.File.access(simple_path, .{})) {
            break :b simple_path;
        } else |_| {
            allocator.free(simple_path);
        }

        const sharded_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors.index.json" });
        break :b sharded_path;
    };
    defer allocator.free(model_weights_path);

    const model_tokenizer_path = try std.fs.path.join(allocator, &.{ hf_model_path, "tokenizer.json" });
    defer allocator.free(model_tokenizer_path);

    const config = blk: {
        var config_json_file = try async.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(GptOss.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/GptOss",
        .sharding_enabled = cli.args.sharding orelse true,
    };

    // initialize ZML platform with optional create options
    // eg: --create-options='{"cuda":{"allocator":{"bfc":{"memory_fraction": 0.99}}}}'
    // or: --create-options='{"cpu":{"device_count":8}}'
    const create_opts_json = cli.args.@"create-options" orelse "{}";
    const create_opts = std.json.parseFromSlice(zml.Platform.CreateOptions, allocator, create_opts_json, .{}) catch |err| {
        log.err("Failed to parse --create-options as json ({}): {s}", .{ err, create_opts_json });
        return err;
    };

    const platform = context.autoPlatform(create_opts.value).withCompilationOptions(compilation_options);
    create_opts.deinit();
    context.printAvailablePlatforms(platform);

    var store = try zml.aio.detectFormatAndOpen(allocator, model_weights_path);
    defer store.deinit();

    const seq_len: u32 = cli.args.@"seq-len" orelse 256;
    const llama_options: GptOss.Options = .{
        .max_seq_len = seq_len,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };

    var compiler_arena = std.heap.ArenaAllocator.init(allocator);
    defer compiler_arena.deinit();

    const model_instance = try GptOss.init(compiler_arena.allocator(), store, config, llama_options);
    const dtype = model_instance.model.embed_tokens.weight.dtype();

    const tokens_shape_prefill = zml.Shape.init(.{ .s = llama_options.max_seq_len }, .u32);
    const tokens_shape = zml.Shape.init(.{ .s = 1 }, .u32);
    const token_idx_shape = zml.Shape.init(.{}, .u32);

    const kv_shape = zml.Shape.init(.{
        .layer = model_instance.model.layers.len,
        .k = seq_len,
        .h = config.num_key_value_heads,
        .hd = config.head_dim,
    }, dtype).withSharding(.{.h});

    const kv_cache_shape: zml.ShapeOf(GptOss.KvCache) = GptOss.KvCache.initShape(kv_shape);
    const rng_shape = zml.Tensor.Rng.shape();

    var start = try std.time.Timer.start();
    var fut_mod_prefill = try async.async(zml.compileModel, .{
        allocator, GptOss.forward, model_instance,
        .{
            tokens_shape_prefill,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        platform,
    });

    var fut_mod = try async.async(zml.compileModel, .{
        allocator, GptOss.forward, model_instance,
        .{
            tokens_shape,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        platform,
    });

    log.info("\tLoading GptOss weights from {s}...", .{model_weights_path});
    var mixtral_weights = try model_instance.loadBuffers(compiler_arena.allocator(), store, platform);
    defer zml.aio.unloadBuffers(&mixtral_weights);
    log.info("✅\tLoaded weights in {D}", .{start.read()});

    var mixtral_module_prefill = (try fut_mod_prefill.await()).prepare(mixtral_weights);
    defer mixtral_module_prefill.deinit();
    var mixtral_module = (try fut_mod.await()).prepare(mixtral_weights);
    defer mixtral_module.deinit();
    log.info("✅\tCompiled model in {D}", .{start.read()});

    log.info("Creating KvCache", .{});
    const kv_cache = try GptOss.KvCache.initBuffer(kv_shape, platform);

    var tokenizer = blk: {
        log.info("Loading tokenizer from {s}", .{model_tokenizer_path});
        var timer = try stdx.time.Timer.start();
        defer log.info("Loaded tokenizer from {s} [{f}]", .{ model_tokenizer_path, timer.read() });

        break :blk try zml.tokenizer.Tokenizer.fromFile(allocator, model_tokenizer_path);
    };
    errdefer tokenizer.deinit();

    const prompt = cli.args.prompt orelse "What is the largest animal?";
    log.info("✅\tPrompt: {s}", .{prompt});

    const seed = cli.args.seed orelse @as(u128, @bitCast(std.time.nanoTimestamp()));
    const skip_llama3_encoding = cli.args.@"no-llama3" orelse false;

    // Unbuffered writing of the tokens to stdout.
    var output = std.fs.File.stdout().writer(&.{});

    try generateText(config, model_instance, mixtral_module_prefill, mixtral_module, kv_cache, tokenizer, allocator, seed, prompt[0..], skip_llama3_encoding, &output.interface);
    // generated text will be printed token by token.
}

const ClapBoilerplate = struct {
    pub const Cli = clap.Result(clap.Help, &cli_params, parsers);

    fn bool_parser(in: []const u8) error{}!bool {
        return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
    }

    const parsers = .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(u32, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };

    pub fn parseCli(allocator: std.mem.Allocator) Cli {
        var diag: clap.Diagnostic = .{};
        var stderr_buffer: [1024]u8 = undefined;
        var stderr = std.fs.File.stderr().writer(&stderr_buffer);
        const cli = clap.parse(clap.Help, &cli_params, parsers, .{
            .diagnostic = &diag,
            .allocator = allocator,
        }) catch |err| {
            diag.report(&stderr.interface, err) catch {};
            stderr.interface.print("usage: ", .{}) catch {};
            clap.usage(&stderr.interface, clap.Help, &cli_params) catch {};
            stderr.interface.print("\n", .{}) catch {};
            std.process.exit(1);
        };
        if (cli.args.help != 0) {
            clap.help(&stderr.interface, clap.Help, &cli_params, .{}) catch {};
            std.process.exit(0);
        }
        return cli;
    }
};
