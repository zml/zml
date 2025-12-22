const std = @import("std");

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
};

const params = clap.parseParamsComptime(
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
    log.info("LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;
    //var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //defer _ = gpa.deinit();

    //const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});

    var http_client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
        .connection_pool = .{
            .free_size = threaded.async_limit.toInt() orelse 16,
        },
    };

    try http_client.initDefaultProxies(allocator);
    defer http_client.deinit();

    var vfs_http: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), &http_client, .{});
    defer vfs_http.deinit();

    var hf_auth: zml.io.VFS.HF.Auth = try .auto(allocator, threaded.io());
    defer hf_auth.deinit(allocator);

    var hf_vfs: zml.io.VFS.HF = try .init(
        allocator,
        threaded.io(),
        &http_client,
        .{
            .request_range_min = 8 * 1024,
            .request_range_max = 128 * 1024 * 1024,
            .hf_pagination_limit = 100,
            .auth = hf_auth,
        },
    );
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("http", vfs_http.io());
    try vfs.register("https", vfs_http.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    zml.init();
    defer zml.deinit();

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

    log.info("Resolving Model repo", .{});
    var repo = try zml.safetensors.resolveModelRepo(allocator, io, hf_model_path);
    defer repo.deinit(allocator, io);

    const parsed_config = try parseConfig(allocator, io, repo.dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    // initialize ZML platform with optional create options
    // eg: --create-options='{"cuda":{"allocator":{"bfc":{"memory_fraction": 0.99}}}}'
    var platform = b: {
        const create_opts_json = cli.args.@"create-options" orelse "{}";
        const create_opts = try std.json.parseFromSlice(zml.platform.CreateOptions, allocator, create_opts_json, .{});
        defer create_opts.deinit();
        const platform: zml.Platform = try .auto(io, create_opts.value);
        break :b platform;
    };
    defer platform.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // Write metadata from the config file into the LlamaLm struct.
    const seq_len: u32 = cli.args.@"seq-len" orelse 256;
    const llama_options: llama.LlamaLM.Options = .{
        .max_seq_len = seq_len,
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
        .kv_cache = .init(zml.Shape.init(.{
            .layer = llama_model.model.layers.len,
            .k = seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, dtype)),
        .rng = .init(),
    };

    var compiled_model_result_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, llama_model_: llama.LlamaLM, llama_parameters_: LlamaParameters) !CompileModelResult {
            return compileModel(allocator_, io_, platform_, llama_model_, llama_parameters_);
        }
    }.call, .{ allocator, io, platform, llama_model, llama_parameters });
    errdefer if (compiled_model_result_future.cancel(io)) |v| {
        defer v.prefill_exe.deinit();
        defer v.decode_exe.deinit();
    } else |_| {};

    var llama_buffers_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, store_: *zml.io.TensorStore, llama_model_: llama.LlamaLM) !zml.Bufferized(llama.LlamaLM) {
            return loadModelBuffers(allocator_, io_, platform_, store_, llama_model_);
        }
    }.call, .{ allocator, io, platform, &store, llama_model });
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

    var tokenizer = try loadTokenizer(allocator, io, repo.dir);
    defer tokenizer.deinit();

    const prompt = cli.args.prompt orelse "What is the capital of France?";
    log.info("✅\tPrompt: {s}", .{prompt});

    // Unbuffered writing of the tokens to stdout.
    var stdout = std.fs.File.stdout().writer(&.{});

    const now = try std.Io.Clock.now(.boot, io);
    const seed: u128 = cli.args.seed orelse @intCast(now.nanoseconds);
    const skip_llama3_encoding = cli.args.@"no-llama3" orelse false;

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
        seed,
        prompt[0..],
        skip_llama3_encoding,
        &stdout.interface,
        platform,
    );
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(LlamaLM.Config) {
    var timer = try stdx.time.Timer.start();
    log.info("Loading model config", .{});
    defer log.info("Loaded model config [{D}]", .{timer.read()});

    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();

    return parsed_config;
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    var timer = try stdx.time.Timer.start();
    log.info("Loading tokenizer", .{});
    defer log.info("Loaded tokenizer [{D}]", .{timer.read()});
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);

        var reader = file.reader(io, &.{});
        var writer: std.Io.Writer.Allocating = .init(allocator);
        defer writer.deinit();
        _ = try reader.interface.streamRemaining(&writer.writer);
        break :b try writer.toOwnedSlice();
    };
    defer allocator.free(bytes);

    return .{ .hftokenizers = try zml.tokenizer.hftokenizers.HFTokenizer.fromBytes(bytes) };
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

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, llama_model: llama.LlamaLM, parameters: LlamaParameters) !CompileModelResult {
    var timer = try stdx.time.Timer.start();
    log.info("Compiling model", .{});
    defer log.info("Compiled model [{D}]", .{timer.read()});

    // Compile the model twice, one for prefill, one for generation.
    var prefill_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters) !zml.Exe {
            return platform_.compileModel(allocator_, io_, llama.LlamaLM.forward, llama_model_, .{ parameters_.prefill_tokens, parameters_.token_index, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = io.async(struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: zml.Platform, llama_model_: llama.LlamaLM, parameters_: LlamaParameters) !zml.Exe {
            return platform_.compileModel(allocator_, io_, llama.LlamaLM.forward, llama_model_, .{ parameters_.decode_tokens, parameters_.token_index, parameters_.kv_cache, parameters_.rng });
        }
    }.call, .{ allocator, io, platform, llama_model, parameters });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

fn loadModelBuffers(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, store: *zml.io.TensorStore, llama_model: llama.LlamaLM) !zml.Bufferized(llama.LlamaLM) {
    var timer = try stdx.time.Timer.start();
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
        try prefill_tokens_buffer.toSlice(prefill_tokens_slice, io);
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
        try current_token_buffer.toSlice(generated_token_slice, io);
    }
    const duration = timer.read();
    const duration_s = stdx.math.divFloat(f64, duration.ns, std.time.ns_per_s);
    const speed = @as(f64, @floatFromInt(num_tokens_generated)) / duration_s;
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration.ns, speed });
}

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}
