const std = @import("std");

const async = @import("async");
const clap = @import("clap");
const stdx = @import("stdx");
const zml = @import("zml");
const Tensor = zml.Tensor;

const modernbert = @import("modernbert.zig");

const log = std.log.scoped(.modernbert);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .modernbert, .level = .info },
    },
    .logFn = async.logFn(std.log.defaultLog),
};

const params = clap.parseParamsComptime(
    \\--help                                print this help
    \\--text                    <STRING>    the prompt
    \\--model                   <PATH>      model path
    \\--tokenizer               <PATH>      tokenizer path
    \\--seq-len                 <UINT>      sequence length
    \\--num-attention-heads     <UINT>      number of attention heads
    \\--tie-word-embeddings     <BOOL>      default: false: tied weights
    \\--create-options          <STRING>    platform creation options JSON, defaults to {}
    \\--sharding                <BOOL>      default: true: sharding on or off
);

const clap_parsers = .{
    .BOOL = bool_parser,
    .UINT = clap.parsers.int(usize, 0),
    .STRING = clap.parsers.string,
    .PATH = clap.parsers.string,
};

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;
    const stderr = std.fs.File.stderr();

    var diag: clap.Diagnostic = .{};
    var cli = clap.parse(clap.Help, &params, clap_parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.reportToFile(stderr, err);
        try printUsageAndExit(stderr);
    };
    defer cli.deinit();

    if (cli.args.help != 0) {
        try clap.helpToFile(stderr, clap.Help, &params, .{});
        return;
    }

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/modernbert/cache");

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Platform and compilation options
    const create_opts_json = cli.args.@"create-options" orelse "{}";
    const create_opts = try std.json.parseFromSliceLeaky(zml.Platform.CreateOptions, allocator, create_opts_json, .{});
    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/modernbert",
        .sharding_enabled = cli.args.sharding orelse true,
    };

    // Auto-select platform
    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    // Detects the format of the model file (base on filename) and open it.
    const model_file = cli.args.model orelse {
        var buf: [256]u8 = undefined;
        var writer = stderr.writer(&buf);
        writer.interface.print("Error: missing --model=...\n\n", .{}) catch {};
        printUsageAndExit(stderr);
        unreachable;
    };
    var tensor_store = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer tensor_store.deinit();

    // Memory arena dedicated to model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    var tokenizer = blk: {
        if (cli.args.tokenizer) |tok| {
            log.info("\tLoading tokenizer from {s}", .{tok});
            var timer = try stdx.time.Timer.start();
            defer log.info("✅\tLoaded tokenizer from {s} [{D}]", .{ tok, timer.read() });

            break :blk try zml.tokenizer.Tokenizer.fromFile(model_arena, tok);
        } else {
            log.err("Error: missing --tokenizer", .{});
            return;
        }
    };
    defer tokenizer.deinit();

    // Create the model struct, with tensor shapes extracted from the tensor_store
    // TODO: read from config.json
    const modernbert_options = modernbert.ModernBertOptions{
        .pad_token = tokenizer.tokenToId("[PAD]") orelse return error.NoSuchToken,
        .num_attention_heads = @intCast(cli.args.@"num-attention-heads" orelse 12),
        .tie_word_embeddings = cli.args.@"tie-word-embeddings" orelse false,
        .local_attention = 128,
    };
    var modern_bert_for_masked_lm = try zml.aio.populateModel(modernbert.ModernBertForMaskedLM, model_arena, tensor_store);
    modern_bert_for_masked_lm.init(modernbert_options);

    log.info("\tModernBERT options: {}", .{modernbert_options});

    // Prepare shapes for compilation
    const seq_len = @as(i64, @intCast(cli.args.@"seq-len" orelse 256));
    const input_shape = zml.Shape.init(.{ .b = 1, .s = seq_len }, .u32);

    var start = try stdx.time.Timer.start();

    // Load weights
    log.info("\tLoading ModernBERT weights from {s}...", .{model_file});
    var bert_weights = try zml.aio.loadBuffers(modernbert.ModernBertForMaskedLM, .{modernbert_options}, tensor_store, model_arena, platform);
    defer zml.aio.unloadBuffers(&bert_weights);
    log.info("✅\tLoaded weights in {D}", .{start.read()});

    // Compile the model
    log.info("\tCompiling ModernBERT model...", .{});
    var fut_mod = try async.async(zml.compile, .{
        allocator,
        modernbert.ModernBertForMaskedLM.forward,
        .{modernbert_options},
        .{input_shape},
        tensor_store,
        platform,
    });
    var bert_module = (try fut_mod.await()).prepare(bert_weights);
    defer bert_module.deinit();
    log.info("✅\tLoaded weights and compiled model in {D}", .{start.read()});

    const text = cli.args.text orelse "Paris is the [MASK] of France.";
    log.info("\tInput text: {s}", .{text});

    try unmask(allocator, bert_module, tokenizer, seq_len, text);
}

/// fill-mask pipeline
/// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py
pub fn unmask(
    allocator: std.mem.Allocator,
    mod: zml.ModuleExe(modernbert.ModernBertForMaskedLM.forward),
    tokenizer: zml.tokenizer.Tokenizer,
    seq_len: i64,
    text: []const u8,
) !void {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const pad_token = tokenizer.tokenToId("[PAD]") orelse return error.NoSuchToken;
    const mask_token = tokenizer.tokenToId("[MASK]") orelse return error.NoSuchToken;

    // Tokenize input text
    const tokens: []const u32 = try tokenize(allocator, tokenizer, text);
    defer allocator.free(tokens);

    // Find "[MASK]" positions
    const mask_positions = try findMaskPositions(allocator, tokens, mask_token);
    defer allocator.free(mask_positions);

    // Prepare input tensors
    const inputs = try prepareTensorInputs(allocator, tokens, seq_len, pad_token);
    defer allocator.free(inputs);

    // Create input tensors (on the accelerator)
    const input_shape = zml.Shape.init(.{ .b = 1, .s = seq_len }, .i64);
    const input_ids_tensor = try zml.Buffer.fromSlice(mod.platform(), input_shape.dims(), inputs);
    defer input_ids_tensor.deinit();

    // Model inference (retrieve indices)
    var inference_timer = try std.time.Timer.start();
    var topk = mod.call(.{input_ids_tensor});
    defer zml.aio.unloadBuffers(&topk);
    const inference_time = inference_timer.read();

    // Transfer the result to host memory (CPU)
    var indices_host_buffer = try topk.indices.toHostAlloc(allocator);
    defer indices_host_buffer.deinit(allocator);
    var values_host_buffer = try topk.values.toHostAlloc(allocator);
    defer values_host_buffer.deinit(allocator);

    // We consider only the first occurrence of [MASK], which has five predictions
    const pred_offset = mask_positions[0] * 5;
    const predictions = indices_host_buffer.items(i32)[pred_offset..][0..5];
    const scores = values_host_buffer.items(f32)[pred_offset..][0..5];

    // Log timing information
    log.info("⏱️\tModel inference in  {d}ms", .{inference_time / std.time.ns_per_ms});

    log.info("✅\tTop 5 predictions:", .{});
    for (predictions, scores) |token_id, score| {
        const token_text = try tokenizer_decoder.next(@intCast(token_id));
        if (token_text) |word| {
            log.info("\t  • score: {d:.4}  word: '{s}' token: {}", .{ score, word, token_id });
        }
    }
}

pub fn tokenize(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const bos = tokenizer.tokenToId("[CLS]") orelse return error.NoSuchToken;
    const eos = tokenizer.tokenToId("[SEP]") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.append(allocator, bos);
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.append(allocator, eos);

    return tokens.toOwnedSlice(allocator);
}

fn findMaskPositions(allocator: std.mem.Allocator, tokens: []const u32, mask_token: u32) ![]usize {
    var mask_positions: std.ArrayList(usize) = .empty;

    for (tokens, 0..) |token, i| {
        if (token == mask_token) {
            try mask_positions.append(allocator, i);
        }
    }

    if (mask_positions.items.len == 0) {
        log.err("Input text must contains `[MASK]`", .{});
        return error.InvalidInput;
    }

    if (mask_positions.items.len > 1) log.warn("Currently only supporting one [MASK] per input", .{});

    return mask_positions.toOwnedSlice(allocator);
}

fn prepareTensorInputs(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    seq_len: i64,
    pad_token: u32,
) ![]u32 {
    const input_ids = try allocator.alloc(u32, @intCast(seq_len));

    @memset(input_ids, pad_token);
    for (tokens, 0..) |token, i| {
        input_ids[i] = @intCast(token);
    }

    return input_ids;
}

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

fn printUsageAndExit(stderr: std.fs.File) noreturn {
    clap.usageToFile(stderr, clap.Help, &params) catch {};
    std.process.exit(0);
}
