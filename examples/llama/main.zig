const asynk = @import("async");
const flags = @import("tigerbeetle/flags");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const llama_mod = @import("llama.zig");

const LlamaLM = llama_mod.LlamaLM;
const Llama = llama_mod.Llama;
const KvCache = llama_mod.KvCache;
const TransformerLayer = llama_mod.TransformerLayer;
const SelfAttn = llama_mod.SelfAttn;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

// set this to false to disable the verbose logging
const show_mlir = true;

pub const std_options = .{
    .log_level = .err,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .pjrt, .level = if (show_mlir) .debug else .err },
        .{ .scope = .zml_module, .level = if (show_mlir) .debug else .err },
        .{ .scope = .zml, .level = if (show_mlir) .debug else .err },
        .{ .scope = .llama, .level = if (show_mlir) .debug else .info },
    },
};

pub fn generateText(
    llama: LlamaLM,
    mod_prefill: zml.module.ExeWithWeights(LlamaLM.forward),
    mod: zml.module.ExeWithWeights(LlamaLM.forward),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seed: u128,
    prompt: []const u8,
) ![]const u8 {
    const prompt_tok = tokenizer.encode(allocator, prompt, .{}) catch unreachable;
    log.debug("Tokenized Prompt {d}", .{prompt_tok});
    const dims = llama.model.shape();
    const max_seq_len = dims.s;
    const token_buffer = try allocator.alloc(i32, @intCast(max_seq_len));
    @memset(token_buffer, 0);
    for (0..prompt_tok.len) |i| {
        token_buffer[i] = @intCast(prompt_tok[i]);
    }

    const tracer_buffer = try allocator.alloc(u8, @intCast(max_seq_len));
    defer allocator.free(token_buffer);
    defer allocator.free(tracer_buffer);
    defer allocator.free(prompt_tok);
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    var tokens = try zml.Buffer.fromSlice(mod.platform(), .{max_seq_len}, token_buffer);
    var token_index = try zml.Buffer.fromSlice(mod.platform(), .{}, &[_]i32{@intCast(prompt_tok.len - 1)});

    var rng = try zml.Tensor.Rng.init(mod.platform(), seed);
    tokens, token_index, var kv_cache, rng = mod_prefill.call(.{ tokens, token_index, null, rng });
    defer kv_cache.k.deinit();
    defer kv_cache.v.deinit();
    defer kv_cache.layer_index.deinit();

    const tracer = zml.tools.Tracer.init("ai.zml.models.llama");
    var decode_progress = prompt_tok.len;
    const output_tokens_len = max_seq_len - prompt_tok.len - 1;

    const start = std.time.microTimestamp();
    const output_freq: u8 = 1;
    for (0..output_tokens_len) |i| {
        //_ = i;
        const frame_id = tracer.frameStart(try std.fmt.bufPrintZ(tracer_buffer, "Generate token {}/{}", .{ i + 1, output_tokens_len }));
        tokens, token_index, kv_cache, rng = mod.call(.{ tokens, token_index, kv_cache, rng });
        if ((i + 1) % output_freq == 0) {
            const n = output.items.len;
            _ = try tokens.toHost(std.mem.sliceAsBytes(token_buffer));
            try tokenizer.decodeWithOpts(&output, @ptrCast(token_buffer[decode_progress..][0..output_freq]), .{});
            decode_progress += output_freq;
            std.debug.print("{s}", .{output.items[n..]});
            tracer.frameEnd(frame_id, try std.fmt.bufPrintZ(tracer_buffer, "Decoded token {}/{} : {s}", .{ i + 1, output_tokens_len, output.items[n..] }));
        } else {
            tracer.frameEnd(frame_id, try std.fmt.bufPrintZ(tracer_buffer, "Generated token {}/{}", .{ i + 1, output_tokens_len }));
        }
    }
    std.debug.print("\n", .{});

    const n = output.items.len;
    try tokenizer.decodeWithOpts(&output, @ptrCast(token_buffer[decode_progress..]), .{});
    std.debug.print("{s}\n", .{output.items[n..]});
    const end = std.time.microTimestamp();

    const duration = stdx.math.divFloor(f64, end - start, std.time.us_per_s);
    const speed = @as(f64, @floatFromInt(max_seq_len)) / duration;
    log.info("✅ Generated {d} tokens in {:.3}s: {d:.3}tok/s", .{ max_seq_len, duration, speed });

    _ = try tokens.toHost(std.mem.sliceAsBytes(token_buffer));
    const end_index = std.mem.indexOfScalar(i32, token_buffer, 128001) orelse max_seq_len;
    output.clearRetainingCapacity();

    try tokenizer.decodeWithOpts(&output, @ptrCast(token_buffer[0..end_index]), .{});
    return output.toOwnedSlice();
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ llama --model=llama3.7B.safetensors --tokenizer=vocab.json --num_layers=2
        ;
        model: []const u8,
        tokenizer: ?[]const u8 = null,
        layer_start: u8 = 0,
        num_layers: ?u8 = null,
        seq_len: u32 = 256,
        topk: u32 = 2,
        temperature: u32 = 1,
        num_heads: ?i64 = null,
        num_kv_heads: ?i64 = null,
        rope_freq_base: ?i64 = null,
        prompt: ?[]const u8 = null,
        test_activations: ?[]const u8 = null,
        seed: ?u128 = null,
    };

    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/llama/cache");
    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .cache_location = "/tmp/zml/llama/cache",
        .xla_dump_to = "/tmp/zml/llama",
        .sharding_enabled = true,
    };

    const platform = context.autoPlatform().withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const model_file = cli_args.model;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    log.info("Model file: {s}", .{model_file});

    var ts = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer ts.deinit();

    var llama = try zml.aio.populateModel(LlamaLM, model_arena, ts);
    const num_heads = cli_args.num_heads orelse ts.metadata("num_heads", .int) orelse @panic("--num_heads is required for this model");
    const num_kv_heads = cli_args.num_kv_heads orelse ts.metadata("num_kv_heads", .int) orelse num_heads;

    const rope_impl = if (ts.metadata("rope_impl", .string)) |val|
        std.meta.stringToEnum(zml.nn.RopeOpts.Implementation, val).?
    else
        .sequential;

    const llama_options: llama_mod.LlamaOptions = .{
        .max_seq_len = cli_args.seq_len,
        .num_kv_heads = num_kv_heads,
        .num_heads = num_heads,
        .gen_opts = .{
            .topk = cli_args.topk,
            .temperature = @floatFromInt(cli_args.temperature),
        },
        .rms_norm_eps = @floatCast(ts.metadata("rms_norm_eps", .float) orelse 1e-5),
        .rope_opts = .{
            .impl = rope_impl,
            .freq_base = @floatCast(ts.metadata("rope_freq_base", .float) orelse @as(f32, @floatFromInt(cli_args.rope_freq_base orelse 10_000))),
        },
    };
    log.info("✅\tParsed llama config: {}", .{llama_options});
    llama.init(llama_options);

    if (cli_args.tokenizer == null and !std.mem.endsWith(u8, cli_args.model, ".gguf")) {
        log.err("Model doesn't have an embbedded tokenizer, please provide a path to a tokenizer.", .{});
        @panic("No tokenizer provided");
    }
    const tokenizer_path = cli_args.tokenizer orelse cli_args.model;
    log.info("\tLoading tokenizer from {s}", .{tokenizer_path});
    var tokenizer = try zml.aio.detectFormatAndLoadTokenizer(allocator, tokenizer_path);
    log.info("✅\tLoaded tokenizer from {s}", .{tokenizer_path});
    defer tokenizer.deinit();

    const dims = llama.model.shape();
    const dtype = llama.lm_head.weight.dtype();

    // Note: we compile the model without a batching dimension.
    // To do so, we would just need to add `.b = batch_size` to `token_shape` and `kv_shape`.
    const tokens_shape = zml.Shape.init(.{ .s = dims.s }, .i32);
    const token_idx_shape = zml.Shape.init(.{}, .i32);
    const kv_shape = zml.Shape.init(.{ .layer = llama.model.layers.len, .h = dims.nkvh, .k = dims.s, .hd = dims.hd }, dtype).withSharding(.{.h});
    // needs to be optional
    const kv_cache_shape: ?ShapeOf(KvCache) = KvCache.initShape(kv_shape);
    const rng_shape = Tensor.Rng.shape();

    var start = try std.time.Timer.start();
    var fut_mod_prefill = try asynk.asyncc(zml.compile, .{ allocator, LlamaLM.forward, .{llama_options}, .{ tokens_shape, token_idx_shape, null, rng_shape }, ts, platform });
    var fut_mod = try asynk.asyncc(zml.compile, .{ allocator, LlamaLM.forward, .{llama_options}, .{ tokens_shape, token_idx_shape, kv_cache_shape, rng_shape }, ts, platform });

    log.info("\tLoading Llama weights from {s}...", .{cli_args.model});
    var llama_weights = try zml.aio.loadBuffers(LlamaLM, .{llama_options}, ts, model_arena, platform);
    defer zml.aio.unloadBuffers(&llama_weights);
    log.info("✅\tLoaded weights in {d}ms", .{start.read() / std.time.ns_per_ms});

    var llama_module_prefill = try (try fut_mod_prefill.await_()).prepare(allocator, llama_weights);
    defer llama_module_prefill.deinit();
    var llama_module = try (try fut_mod.await_()).prepare(allocator, llama_weights);
    defer llama_module.deinit();
    log.info("✅\tCompiled model in {d}ms", .{start.read() / std.time.ns_per_ms});

    const prompt = cli_args.prompt orelse "Once upon a time, there was a little girl named Lily.";
    log.info("✅\tPrompt: {s}", .{prompt});

    const seed = cli_args.seed orelse @as(u128, @bitCast(std.time.nanoTimestamp()));
    const story = try generateText(llama, llama_module_prefill, llama_module, tokenizer, allocator, seed, prompt);
    defer allocator.free(story);
}
