const std = @import("std");

const asynk = @import("async");
const clap = @import("clap");
const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const mixtral = @import("mixtral.zig");
const MixtralLM = mixtral.MixtralLM;
const Mixtral = mixtral.Llama;
const KvCache = mixtral.KvCache;
const TransformerLayer = mixtral.TransformerLayer;
const SelfAttn = mixtral.SelfAttn;

const log = std.log.scoped(.mixtral);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
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

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   mixtral was compiled with {}", .{@import("builtin").mode});

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
        if (asynk.File.access(simple_path, .{})) {
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
        var config_json_file = try asynk.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var reader = std.json.reader(allocator, config_json_file.reader());
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(mixtral.MixtralLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/mixtral",
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

    var model_arena = std.heap.ArenaAllocator.init(allocator);
    const llama_options: mixtral.MixtralLM.Options = .{
        .max_seq_len = @intCast(cli.args.@"seq-len" orelse 256),
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };
    const model_instance = try mixtral.MixtralLM.init(model_arena.allocator(), store, config, llama_options);

    const dims = model_instance.model.shape();
    const dtype = model_instance.model.embed_tokens.weight.dtype();

    const tokens_shape_prefill = zml.Shape.init(.{ .s = llama_options.max_seq_len }, .u32);
    const tokens_shape = zml.Shape.init(.{ .s = 1 }, .u32);
    const token_idx_shape = zml.Shape.init(.{}, .u32);

    const kv_shape = zml.Shape.init(.{ .layer = model_instance.model.layers.len, .k = dims.s, .h = dims.nkvh, .hd = dims.hd }, dtype).withSharding(.{.h});

    const kv_cache_shape: zml.ShapeOf(mixtral.KvCache) = mixtral.KvCache.initShape(kv_shape);
    const rng_shape = zml.Tensor.Rng.shape();

    var start = try std.time.Timer.start();
    var fut_mod_prefill = try asynk.asyncc(zml.compileModel, .{
        allocator, mixtral.MixtralLM.forward, model_instance,
        .{
            tokens_shape_prefill,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        platform,
    });

    var fut_mod = try asynk.asyncc(zml.compileModel, .{
        allocator, mixtral.MixtralLM.forward, model_instance,
        .{
            tokens_shape,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        platform,
    });

    log.info("\tLoading mixtral weights from {?s}...", .{model_weights_path});
    var mixtral_weights = try model_instance.loadBuffers(model_arena.allocator(), store, platform);
    defer zml.aio.unloadBuffers(&mixtral_weights);
    log.info("✅\tLoaded weights in {}", .{std.fmt.fmtDuration(start.read())});

    var mixtral_module_prefill = (try fut_mod_prefill.awaitt()).prepare(mixtral_weights);
    defer mixtral_module_prefill.deinit();
    var mixtral_module = (try fut_mod.awaitt()).prepare(mixtral_weights);
    defer mixtral_module.deinit();
    log.info("✅\tCompiled model in {}", .{std.fmt.fmtDuration(start.read())});

    log.info("Creating KvCache", .{});
    const kv_cache = try mixtral.KvCache.initBuffer(kv_shape, platform);

    var tokenizer = blk: {
        log.info("Loading tokenizer from {s}", .{model_tokenizer_path});
        var timer = try stdx.time.Timer.start();
        defer log.info("Loaded tokenizer from {s} [{}]", .{ model_tokenizer_path, timer.read() });

        break :blk try zml.tokenizer.Tokenizer.fromFile(model_arena.allocator(), model_tokenizer_path);
    };
    errdefer tokenizer.deinit();

    const prompt = cli.args.prompt orelse "What is the capital of France?";
    log.info("✅\tPrompt: {s}", .{prompt});

    const seed = cli.args.seed orelse @as(u128, @bitCast(std.time.nanoTimestamp()));
    const skip_llama3_encoding = cli.args.@"no-llama3" orelse false;
    const generated_text = try generateText(config, model_instance, mixtral_module_prefill, mixtral_module, kv_cache, tokenizer, allocator, seed, prompt[0..], skip_llama3_encoding);
    // generated text will be printed token by token.
    defer allocator.free(generated_text);
}

pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: MixtralLM.Config, prompt: []const u8, skip_llama3_encoding: bool) ![]u32 {
    var tokens = std.ArrayList(u32).init(allocator);
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    if (skip_llama3_encoding) {
        // Copy to the arraylist so the ownership is the same in both branches.
        try tokens.appendSlice(try encoder.encode(prompt));
        return tokens.toOwnedSlice();
    }

    const start_header_id = tokenizer.tokenToId("<|start_header_id|>") orelse return error.NoSuchToken;
    const end_header_id = tokenizer.tokenToId("<|end_header_id|>") orelse return error.NoSuchToken;
    const eot_id = tokenizer.tokenToId("<|eot_id|>") orelse return error.NoSuchToken;
    const newline_id = (try encoder.encode("\n"))[0];

    try tokens.append(config.bos_token_id);

    try tokens.append(start_header_id);
    try tokens.appendSlice(try encoder.encode("user"));
    try tokens.appendSlice(&.{ end_header_id, newline_id });

    try tokens.appendSlice(try encoder.encode(prompt));
    try tokens.appendSlice(&.{ eot_id, newline_id });
    try tokens.append(start_header_id);
    try tokens.appendSlice(try encoder.encode("assistant"));
    try tokens.appendSlice(&.{ end_header_id, newline_id });

    return tokens.toOwnedSlice();
}

pub fn generateText(
    config: MixtralLM.Config,
    llama_: MixtralLM,
    mod_prefill: zml.ModuleExe(MixtralLM.forward),
    mod_generate: zml.ModuleExe(MixtralLM.forward),
    kv_cache_: zml.Bufferized(mixtral.KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seed: u128,
    prompt: []const u8,
    skip_llama3_encoding: bool,
) ![]const u8 {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    defer allocator.free(prompt_tok);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const platform = mod_generate.platform();
    const max_seq_len = llama_.model.shape().s;

    // init RNG and buffers
    var rng = try zml.Tensor.Rng.init(platform, seed);
    var generated_token_buffer = [_]u32{undefined};

    var kv_cache = prefill: {
        // prepare device buffers for the prefill tokens and their positions
        const prefill_buffer = try allocator.alloc(u32, max_seq_len);
        @memcpy(prefill_buffer[0..prompt_tok.len], prompt_tok);

        var prefill_tokens = try zml.Buffer.fromSlice(platform, .{max_seq_len}, prefill_buffer);
        defer prefill_tokens.deinit();
        var prefill_token_pos = try zml.Buffer.constant(platform, zml.Shape.init(.{}, .u32), 0);
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

    // Here we collect the generated text
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    const output_tokens_len = max_seq_len - prompt_tok.len - 1;
    const start = std.time.microTimestamp();

    // One token has alreadyh been generated by the prefill.
    var num_tokens_generated: usize = 1;

    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_buffer[0];
        const chunk = try tokenizer_decoder.next(generated_token) orelse unreachable;
        try output.appendSlice(chunk);
        std.debug.print("{s}", .{chunk});

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
    std.debug.print("\n", .{});
    log.info("✅ Generated {d} tokens in {:.3}s: {d:.3}tok/s", .{ num_tokens_generated, duration, speed });
    return output.toOwnedSlice();
}

const ClapBoilerplate = struct {
    pub const Cli = clap.Result(clap.Help, &cli_params, parsers);

    fn bool_parser(in: []const u8) error{}!bool {
        return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
    }

    const parsers = .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(usize, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };

    pub fn parseCli(allocator: std.mem.Allocator) Cli {
        var diag: clap.Diagnostic = .{};
        const stderr = std.io.getStdErr().writer();
        const cli = clap.parse(clap.Help, &cli_params, parsers, .{
            .diagnostic = &diag,
            .allocator = allocator,
        }) catch |err| {
            diag.report(stderr, err) catch {};
            stderr.print("usage: ", .{}) catch {};
            clap.usage(stderr, clap.Help, &cli_params) catch {};
            stderr.print("\n", .{}) catch {};
            std.process.exit(1);
        };
        if (cli.args.help != 0) {
            clap.help(std.io.getStdErr().writer(), clap.Help, &cli_params, .{}) catch {};
            std.process.exit(0);
        }
        return cli;
    }
};
