const asynk = @import("async");
const clap = @import("clap");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const llama = @import("llama.zig");

const LlamaLM = llama.LlamaLM;
const Llama = llama.Llama;
const KvCache = llama.KvCache;
const TransformerLayer = llama.TransformerLayer;
const SelfAttn = llama.SelfAttn;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

// set this to false to disable the verbose logging
const show_mlir = false;

pub const std_options = .{
    .log_level = .warn,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/module", .level = if (show_mlir) .debug else .warn },
        .{ .scope = .llama, .level = .info },
    },
};
pub fn tokenize(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: LlamaLM.Config, prompt: []const u8) ![]u32 {
    var tokens = std.ArrayList(u32).init(allocator);
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const start_header_id = tokenizer.token_to_id("<|start_header_id|>");
    const end_header_id = tokenizer.token_to_id("<|end_header_id|>");
    const eot_id = tokenizer.token_to_id("<|eot_id|>");
    const newline_id = (try encoder.encode("\n"))[0];

    try tokens.append(config.bos_token_id);

    try tokens.append(start_header_id);
    try tokens.appendSlice(try encoder.encode("user"));
    try tokens.appendSlice(&.{ end_header_id, newline_id, newline_id });

    try tokens.appendSlice(try encoder.encode(prompt));
    try tokens.appendSlice(&.{ eot_id, newline_id });
    try tokens.appendSlice(try encoder.encode("\n"));
    try tokens.append(start_header_id);
    try tokens.appendSlice(try encoder.encode("assistant"));
    try tokens.append(end_header_id);

    return tokens.toOwnedSlice();
}

pub fn generateText(
    config: LlamaLM.Config,
    llama_: LlamaLM,
    mod_prefill: zml.ModuleExe(LlamaLM.forward),
    mod: zml.ModuleExe(LlamaLM.forward),
    kv_cache_: zml.Bufferized(llama.KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seed: u128,
    prompt: []const u8,
) ![]const u8 {
    var tokenizer_encoder = try tokenizer.encoder();
    defer tokenizer_encoder.deinit();

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    // const prompt_tok = try tokenizer_encoder.encode(prompt);
    const prompt_tok = try tokenize(allocator, tokenizer, config, prompt);
    log.info("Tokenized Prompt len={d}: {d}", .{ prompt_tok.len, prompt_tok });

    var decoded_prompt_tokens = std.ArrayList(u8).init(allocator);
    defer decoded_prompt_tokens.deinit();
    for (prompt_tok) |tok| {
        const chunk = try tokenizer_decoder.next(@intCast(tok)) orelse unreachable;
        try decoded_prompt_tokens.appendSlice(chunk);
    }
    log.info("Decoded prompt tokens: >>{s}<<", .{decoded_prompt_tokens.items});

    const dims = llama_.model.shape();
    const max_seq_len = dims.s;
    const token_buffer = try allocator.alloc(i32, @intCast(max_seq_len));
    @memset(token_buffer, 0);
    for (0..prompt_tok.len) |i| {
        token_buffer[i] = @intCast(prompt_tok[i]);
    }

    defer allocator.free(token_buffer);
    defer allocator.free(prompt_tok);
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    var tokens = try zml.Buffer.fromSlice(mod.platform(), .{max_seq_len}, token_buffer);
    var prefill_token_index = try zml.Buffer.fromSlice(mod.platform(), .{}, &[_]i32{0});
    defer prefill_token_index.deinit();

    var rng = try zml.Tensor.Rng.init(mod.platform(), seed);
    tokens, var kv_cache, rng = mod_prefill.call(.{ tokens, prefill_token_index, kv_cache_, rng });

    defer kv_cache.k.deinit();
    defer kv_cache.v.deinit();
    defer kv_cache.layer_index.deinit();

    var decode_progress = prompt_tok.len;

    // TODO: eos
    var eos_index: ?usize = null;
    eos_index = eos_index;

    _ = try tokens.toHost(std.mem.sliceAsBytes(token_buffer));

    var decoded_prefill_tokens = std.ArrayList(u8).init(allocator);
    defer decoded_prefill_tokens.deinit();
    for (token_buffer) |tok| {
        const chunk = try tokenizer_decoder.next(@intCast(tok)) orelse unreachable;
        try decoded_prefill_tokens.appendSlice(chunk);
    }

    var new_token_hostbuffer = [_]u32{prompt_tok[prompt_tok.len - 1]};
    var current_token = try zml.Buffer.fromSlice(mod.platform(), .{}, &new_token_hostbuffer);
    defer current_token.deinit();

    var new_token_out = [_]u32{0};

    for (0..10) |i| {
        const token_index_buffer = &[_]i32{@intCast(prompt_tok.len + i)};
        const token_index = try zml.Buffer.fromSlice(mod.platform(), .{}, token_index_buffer);
        defer token_index.deinit();

        current_token, kv_cache, rng = mod.call(.{ current_token, token_index, kv_cache, rng });

        _ = try current_token.toHost(std.mem.sliceAsBytes(&new_token_out));

        const chunk = try tokenizer_decoder.next(@intCast(new_token_out[0])) orelse unreachable;
        try output.appendSlice(chunk);
        decode_progress += 1;

        log.info("Generated #{d}:{d}=>>{s}<< ", .{ token_index_buffer[0], new_token_out[0], chunk });
    }
    return output.toOwnedSlice();
}

const params = clap.parseParamsComptime(
    \\--help                    print this help
    \\--prompt <STRING>         the prompt
    \\--model-config <PATH>     config.json path
    \\--model-name <STRING>     model name
    \\--model-weights <PATH>    model weights path
    \\--model-tokenizer <PATH>  tokenizer path
    \\--seed <UINT>             random seed (optional)
    \\--seq-len <UINT>          sequence length
);

pub fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(usize, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    const stderr = std.io.getStdErr().writer();
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(stderr, err) catch {};
        stderr.print("usage: ", .{}) catch {};
        clap.usage(stderr, clap.Help, &params) catch {};
        stderr.print("\n", .{}) catch {};
        return;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{}) catch {};
        return;
    }

    const config = blk: {
        if (res.args.@"model-config") |config_json_path| {
            var config_json_file = try asynk.File.open(config_json_path, .{ .mode = .read_only });
            defer config_json_file.close() catch unreachable;
            var reader = std.json.reader(allocator, config_json_file.reader());
            defer reader.deinit();
            const config_obj = try std.json.parseFromTokenSourceLeaky(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
            break :blk config_obj;
        } else {
            log.err("Missing --model-config", .{});
            return;
        }
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/llama",
        .sharding_enabled = true,
    };

    // TODO: add --create-opts <PATH> json string/file with creation options
    // const create_opts = try std.json.parseFromSlice(zml.Platform.CreateOptions, allocator, cli_args.create_options, .{});
    // const platform = context.autoPlatform(create_opts.value).withCompilationOptions(compilation_options);
    // create_opts.deinit();

    const platform = context.autoPlatform(.{}).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    var ts = try zml.aio.detectFormatAndOpen(allocator, res.args.@"model-weights".?);
    defer ts.deinit();

    var model_arena = std.heap.ArenaAllocator.init(allocator);
    var model_instance = try zml.aio.populateModel(llama.LlamaLM, model_arena.allocator(), ts);
    model_instance = model_instance; // autofix

    const llama_options: llama.LlamaLM.Options = .{
        .max_seq_len = @intCast(res.args.@"seq-len" orelse 256),
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };
    model_instance.init(config, llama_options);

    const dims = model_instance.model.shape();
    const dtype = model_instance.model.embed_tokens.weight.dtype();

    const batch_size = 1;

    // const tokens_shape = zml.Shape.init(.{ .b = batch_size, .s = llama_options.chunked_prefill_size }, .u32);
    const tokens_shape_prefill = zml.Shape.init(.{ .b = batch_size, .s = llama_options.max_seq_len }, .u32);
    const tokens_shape = zml.Shape.init(.{ .b = batch_size, .s = 1 }, .u32);
    const token_idx_shape = zml.Shape.init(.{ .b = batch_size }, .u32);

    // const kv_shape = zml.Shape.init(.{ .layer = model_instance.model.layers.len, .b = bs, .h = dims.nkvh, .k = dims.s, .hd = dims.hd }, dtype).withSharding(.{.h});
    const kv_shape = zml.Shape.init(.{ .layer = model_instance.model.layers.len, .b = batch_size, .k = dims.s, .h = dims.nkvh, .hd = dims.hd }, dtype).withSharding(.{.h});

    // needs to be optional
    const kv_cache_shape: zml.ShapeOf(llama.KvCache) = llama.KvCache.initShape(kv_shape);
    const rng_shape = zml.Tensor.Rng.shape();

    var start = try std.time.Timer.start();
    var fut_mod_prefill = try asynk.asyncc(zml.compile, .{
        allocator, llama.LlamaLM.forward, .{ config, llama_options },
        .{
            tokens_shape_prefill,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        ts,
        platform,
    });

    var fut_mod = try asynk.asyncc(zml.compile, .{
        allocator, llama.LlamaLM.forward, .{ config, llama_options },
        .{
            tokens_shape,
            token_idx_shape,
            kv_cache_shape,
            rng_shape,
        },
        ts,
        platform,
    });

    log.info("\tLoading Llama weights from {?s}...", .{res.args.@"model-weights"});
    var llama_weights = try zml.aio.loadBuffers(llama.LlamaLM, .{ config, llama_options }, ts, model_arena.allocator(), platform);
    defer zml.aio.unloadBuffers(&llama_weights);
    log.info("✅\tLoaded weights in {}", .{std.fmt.fmtDuration(start.read())});

    var llama_module_prefill = (try fut_mod_prefill.awaitt()).prepare(llama_weights);
    defer llama_module_prefill.deinit();
    var llama_module = (try fut_mod.awaitt()).prepare(llama_weights);
    defer llama_module.deinit();
    log.info("✅\tCompiled model in {}", .{std.fmt.fmtDuration(start.read())});

    log.info("Creating KvCache", .{});
    const kv_cache = try llama.KvCache.initBuffer(kv_shape, platform);

    const prompt = res.args.prompt orelse "Q: The capitol of France is?\nA: ";
    log.info("✅\tPrompt: {s}", .{prompt});

    var tokenizer = blk: {
        if (res.args.@"model-tokenizer") |tok| {
            log.info("Loading tokenizer from {s}", .{tok});
            var timer = try stdx.time.Timer.start();
            defer log.info("Loaded tokenizer from {s} [{}]", .{ tok, timer.read() });

            break :blk try zml.tokenizer.Tokenizer.from_file(model_arena.allocator(), tok);
        } else {
            log.err("Missing --model-tokenizer", .{});
            return;
        }
    };
    errdefer tokenizer.deinit();

    const seed = res.args.seed orelse @as(u128, @bitCast(std.time.nanoTimestamp()));
    // const story = try generateText(model_instance, llama_module_prefill, llama_module, kv_cache, tokenizer, allocator, seed, prompt[0..]);
    const story = try generateText(config, model_instance, llama_module_prefill, llama_module, kv_cache, tokenizer, allocator, seed, prompt[0..]);
    defer allocator.free(story);

    log.info("✅\tGenerated: {s}", .{story});
}
