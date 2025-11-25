const std = @import("std");

const async = @import("async");
const clap = @import("clap");
const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const llama = @import("llama.zig");
const LlamaLM = llama.LlamaLM;
const Llama = llama.Llama;
const KvCache = llama.KvCache;
const TransformerLayer = llama.TransformerLayer;
const SelfAttn = llama.SelfAttn;

const log = std.log.scoped(.llama);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = async.logFn(std.log.defaultLog),
};

const params = clap.parseParamsComptime(
    \\--help                      print this help
    \\--prompt         <STRING>   the prompt
    \\--hf-model-path  <STRING>   path to the directory containing model weights, config and tokenizer
    \\--seed           <UINT>     random seed (optional)
    \\--seq-len        <UINT>     sequence length
    \\--topk            <UINT>     top k
    \\--create-options <STRING>   platform creation options JSON, defaults to {}
    \\--no-llama3      <BOOL>     skip prompt template
    \\--sharding       <BOOL>     default: true: sharding on or off
);

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
    config: LlamaLM.Config,
    llama_: LlamaLM,
    mod_prefill: zml.ModuleExe(LlamaLM.forward),
    mod_generate: zml.ModuleExe(LlamaLM.forward),
    kv_cache_: zml.Bufferized(llama.KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seed: u128,
    prompt: []const u8,
    skip_llama3_encoding: bool,
    writer: *std.Io.Writer,
) !void {
    const prompt_tok: []const u32 = try tokenizePrompt(allocator, tokenizer, config, prompt, skip_llama3_encoding);
    defer allocator.free(prompt_tok);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const platform = mod_generate.platform();
    const max_seq_len = llama_.model.max_seq_len;

    // init RNG and buffers
    var generated_token_buffer = [_]u32{undefined};

    const probs_shape = mod_prefill.inner.result_shapes[0];
    std.debug.assert(probs_shape.eql(mod_generate.inner.result_shapes[0]));
    var probs_h: zml.HostBuffer = try .empty(allocator, probs_shape);

    const TokenSampler = @import("isaac/TokenSampler.zig");
    const strategy: TokenSampler.Strategy = .{ .top_k = 16, .seed = @truncate(seed) };
    var sampler: TokenSampler = try .init(allocator, strategy);
    defer sampler.deinit();

    var kv_cache = prefill: {
        // prepare device buffers for the prefill tokens and their positions
        const prefill_buffer = try allocator.alloc(u32, max_seq_len);
        @memcpy(prefill_buffer[0..prompt_tok.len], prompt_tok);

        var prefill_tokens = try zml.Buffer.fromSlice(platform, .{max_seq_len}, prefill_buffer);
        defer prefill_tokens.deinit();
        var prefill_token_pos = try zml.Buffer.scalar(platform, 0, .u32);
        defer prefill_token_pos.deinit();

        const probs_d, const kv_cache = mod_prefill.call(.{ prefill_tokens, prefill_token_pos, kv_cache_ });
        defer probs_d.deinit();

        probs_h = try probs_d.toHost(probs_h.mutBytes());
        _ = try sampler.vtable.writeFn(&sampler.state, @ptrCast(@alignCast(probs_h.items(f32))));
        generated_token_buffer[0] = sampler.vtable.pick(&sampler.state, strategy);
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
            try writer.writeAll(chunk);
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
        const probs_d, kv_cache = mod_generate.call(.{ current_token, token_pos, kv_cache });
        defer probs_d.deinit();

        {
            probs_h = try probs_d.toHost(probs_h.mutBytes());
            _ = try sampler.vtable.writeFn(&sampler.state, @ptrCast(@alignCast(probs_h.items(f32))));
            generated_token_buffer[0] = sampler.vtable.pick(&sampler.state, strategy);
        }
    }
    const end = std.time.microTimestamp();
    const duration = stdx.math.divFloat(f64, end - start, std.time.us_per_s);
    const speed = @as(f64, @floatFromInt(num_tokens_generated)) / duration;
    std.debug.print("\n", .{});
    log.info("✅ Generated {d} tokens in {:.3}s: {d:.3}tok/s", .{ num_tokens_generated, duration, speed });
}

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(u32, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    var stderr_buffer: [1024]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buffer);
    defer stderr.interface.flush() catch {};

    var cli = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(&stderr.interface, err) catch {};
        stderr.interface.writeAll("usage: ") catch {};
        clap.usage(&stderr.interface, clap.Help, &params) catch {};
        stderr.interface.writeAll("\n") catch {};
        return;
    };
    defer cli.deinit();

    if (cli.args.help != 0) {
        clap.help(&stderr.interface, clap.Help, &params, .{}) catch {};
        return;
    }

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
        const config_obj = try std.json.parseFromTokenSourceLeaky(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/llama",
        .sharding_enabled = cli.args.sharding orelse true,
    };

    // initialize ZML platform with optional create options
    // eg: --create-options='{"cuda":{"allocator":{"bfc":{"memory_fraction": 0.99}}}}'
    const create_opts_json = cli.args.@"create-options" orelse "{}";
    const create_opts = try std.json.parseFromSlice(zml.Platform.CreateOptions, allocator, create_opts_json, .{});
    const platform = context.autoPlatform(create_opts.value).withCompilationOptions(compilation_options);
    create_opts.deinit();
    context.printAvailablePlatforms(platform);

    var store = try zml.aio.detectFormatAndOpen(allocator, model_weights_path);
    defer store.deinit();

    // Write metadata from the config file into the LlamaLm struct.
    const seq_len: u32 = cli.args.@"seq-len" orelse 256;
    const llama_options: llama.LlamaLM.Options = .{
        .max_seq_len = seq_len,
        .sampling_strategy = .{
            .topk = cli.args.topk orelse 16,
            .temperature = 1.0,
        },
    };

    // Contains memory for llama_tensors and llama_buffers.
    var compiler_arena = std.heap.ArenaAllocator.init(allocator);
    defer compiler_arena.deinit();

    // Initialize the Llama struct and map the content of the .safetensors to the model tensors.
    const llama_tensors: llama.LlamaLM = try .init(compiler_arena.allocator(), config, llama_options, store);

    // Specify shapes of input arguments
    const prefill_tokens_shape = zml.Shape.init(.{ .s = llama_options.max_seq_len }, .u32);
    const gen_tokens_shape = zml.Shape.init(.{ .s = 1 }, .u32);
    const token_idx_shape = zml.Shape.init(.{}, .u32);

    const dtype = llama_tensors.model.embed_tokens.weight.dtype();
    const kv_shape = zml.Shape.init(.{
        .layer = llama_tensors.model.layers.len,
        .k = seq_len,
        .h = config.num_key_value_heads,
        .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
    }, dtype).withSharding(.{.h});
    const kv_cache_shape: zml.ShapeOf(llama.KvCache) = llama.KvCache.initShape(kv_shape);

    // Compile the model twice, one for prefill, one for generation.
    var start = try std.time.Timer.start();
    var fut_mod_prefill = try async.async(zml.compileModel, .{
        allocator, llama.LlamaLM.forward, llama_tensors,
        .{
            prefill_tokens_shape,
            token_idx_shape,
            kv_cache_shape,
        },
        platform,
    });

    var fut_mod = try async.async(zml.compileModel, .{
        allocator, llama.LlamaLM.forward, llama_tensors,
        .{
            gen_tokens_shape,
            token_idx_shape,
            kv_cache_shape,
        },
        platform,
    });

    // While we are still compiling load the weights to the device.
    log.info("\tLoading Llama weights from {s}...", .{model_weights_path});
    var llama_buffers = try store.loadModelById(llama.LlamaLM, compiler_arena.allocator(), llama_tensors, platform);
    defer zml.aio.unloadBuffers(&llama_buffers);
    log.info("✅\tLoaded weights in {D}", .{start.read()});

    var llama_module_prefill = (try fut_mod_prefill.await()).prepare(llama_buffers);
    defer llama_module_prefill.deinit();
    var llama_module = (try fut_mod.await()).prepare(llama_buffers);
    defer llama_module.deinit();
    log.info("✅\tCompiled model in {D}", .{start.read()});
    log.info("Creating KvCache", .{});
    const kv_cache = try llama.KvCache.initBuffer(kv_shape, platform);

    var tokenizer = blk: {
        log.info("Loading tokenizer from {s}", .{model_tokenizer_path});
        var timer = try stdx.time.Timer.start();
        defer log.info("Loaded tokenizer from {s} [{D}]", .{ model_tokenizer_path, timer.read() });

        break :blk try zml.tokenizer.Tokenizer.fromFile(allocator, model_tokenizer_path);
    };
    errdefer tokenizer.deinit();

    const prompt = cli.args.prompt orelse "What is the capital of France?";
    log.info("✅\tPrompt: {s}", .{prompt});

    // Unbuffered writing of the tokens to stdout.
    var stdout = std.fs.File.stdout().writer(&.{});

    const seed: u128 = cli.args.seed orelse @bitCast(std.time.nanoTimestamp());
    const skip_llama3_encoding = cli.args.@"no-llama3" orelse false;

    try generateText(
        config,
        llama_tensors,
        llama_module_prefill,
        llama_module,
        kv_cache,
        tokenizer,
        allocator,
        seed,
        prompt[0..],
        skip_llama3_encoding,
        &stdout.interface,
    );
}

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}
