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

        prefill_exe.call(prefill_args, &prefill_results, io);

        prefill_results.fill(&.{ &prefill_tokens_buffer, kv_cache_buffers, &rng_buffers });
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

        decode_exe.call(decode_args, &decode_results, io);

        decode_results.fill(&.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });

        // extract the generated token from the buffer
        try current_token_buffer.toSlice(generated_token_slice, io);
    }
    const duration = timer.read();
    const duration_s = stdx.math.divFloat(f64, duration.ns, std.time.ns_per_s);
    const speed = @as(f64, @floatFromInt(num_tokens_generated)) / duration_s;
    std.debug.print("\n", .{});
    log.info("✅ Generated {} tokens in {D}: {:.3}tok/s", .{ num_tokens_generated, duration.ns, speed });
}

pub fn main() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    //const allocator = std.heap.c_allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    var vfs_file: zml.io.VFS.File = .init(threaded.io());

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

    var vfs: zml.io.VFS = .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("http", vfs_http.io());
    try vfs.register("https", vfs_http.io());

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

    var repo = try zml.safetensors.resolveModelRepo(allocator, io, hf_model_path);
    defer repo.deinit(allocator, io);

    const parsed_config = blk: {
        const config_json_file = try repo.dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
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
            .topk = 1,
            .temperature = 1.0,
        },
    };

    // Initialize the Llama struct and map the content of the .safetensors to the model tensors.
    const llama_model: llama.LlamaLM = try .init(allocator, store.view(), config, llama_options);
    defer llama_model.deinit(allocator);

    // Specify shapes of input arguments
    const prefill_tokens: zml.Tensor = .init(.{ .s = llama_options.max_seq_len }, .u32);
    const decode_tokens: zml.Tensor = .init(.{ .s = 1 }, .u32);
    const token_index: zml.Tensor = .init(.{}, .u32);

    const dtype = llama_model.model.embed_tokens.weight.dtype();
    const kv_cache: llama.KvCache = .init(zml.Shape.init(.{
        .layer = llama_model.model.layers.len,
        .k = seq_len,
        .h = config.num_key_value_heads,
        .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
    }, dtype));
    const rng: zml.Tensor.Rng = .init();

    // Compile the model twice, one for prefill, one for generation.
    var prefill_exe = try platform.compileModel(allocator, io, llama.LlamaLM.forward, llama_model, .{ prefill_tokens, token_index, kv_cache, rng });
    defer prefill_exe.deinit();
    var decode_exe = try platform.compileModel(allocator, io, llama.LlamaLM.forward, llama_model, .{ decode_tokens, token_index, kv_cache, rng });
    defer decode_exe.deinit();

    var llama_buffers = try llama_model.loadBuffers(allocator, io, store.view(), platform);
    defer LlamaLM.unloadBuffers(&llama_buffers, allocator);

    log.info("Creating KvCache", .{});
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer llama.KvCache.deinitBuffer(&kv_cache_buffers);

    var tokenizer = blk: {
        log.info("Loading tokenizer", .{});
        var timer = try stdx.time.Timer.start();
        defer log.info("Loaded tokenizer [{D}]", .{timer.read()});
        const bytes = b: {
            const file = try repo.dir.openFile(io, "tokenizer.json", .{});
            defer file.close(io);

            var reader = file.reader(io, &.{});
            var writer: std.Io.Writer.Allocating = .init(allocator);
            defer writer.deinit();
            _ = try reader.interface.streamRemaining(&writer.writer);
            break :b try writer.toOwnedSlice();
        };
        defer allocator.free(bytes);

        const tokenizer: zml.tokenizer.Tokenizer = .{ .hftokenizers = try zml.tokenizer.hftokenizers.HFTokenizer.fromBytes(bytes) };
        break :blk tokenizer;
    };
    errdefer tokenizer.deinit();

    const prompt = cli.args.prompt orelse "What is the capital of France?";
    log.info("✅\tPrompt: {s}", .{prompt});

    // Unbuffered writing of the tokens to stdout.
    var stdout = std.fs.File.stdout().writer(&.{});

    //const seed: u128 = cli.args.seed orelse @bitCast(std.time.nanoTimestamp());
    const seed: u128 = cli.args.seed orelse 0;
    const skip_llama3_encoding = cli.args.@"no-llama3" orelse false;

    try generateText(
        allocator,
        io,
        llama_buffers,
        prefill_exe,
        decode_exe,
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

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}
