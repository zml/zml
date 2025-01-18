const flags = @import("tigerbeetle/flags");
const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log.scoped(.modernbert);
const Tensor = zml.Tensor;
const modernbert = @import("modernbert.zig");

// set this to false to disable the verbose logging
const show_mlir = true;
pub const std_options = .{
    .log_level = .warn,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zml_module, .level = if (show_mlir) .debug else .warn },
        .{ .scope = .modernbert, .level = .info },
    },
};

pub const max_seq_len = 64; // 8192

pub fn predictMaskedTokens(
    bert: modernbert.ModernBertForMaskedLM,
    mod: zml.ModuleExe(modernbert.ModernBertForMaskedLM.forward),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    text: []const u8,
) !void {
    _ = bert; // autofix

    // shapes
    const seq_len = 9;
    _ = seq_len; // autofix
    const input_shape = zml.Shape.init(.{ .b = 1, .s = max_seq_len }, .i64);
    const attention_mask_shape = input_shape;

    const tokens_u32 = try tokenizer.encode(allocator, text, .{});
    if (tokens_u32.len > max_seq_len) {
        log.err("Input text too long: {} tokens > {} max", .{ tokens_u32.len, max_seq_len });
        return error.InvalidInput;
    }

    log.info("Tokenized input text: {any} | length: {}", .{ tokens_u32, tokens_u32.len });

    // debug
    for (tokens_u32, 0..) |token, i| {
        if (i < tokenizer.tokens.len) {
            log.info("Token {}: {s} -> '{}'", .{ i, tokenizer.tokens[token], token });
        }
    }

    var input_ids_buffer = try allocator.alloc(i64, max_seq_len);
    var attention_mask_buffer = try allocator.alloc(i64, max_seq_len);

    // init all values to 0
    @memset(input_ids_buffer, 0);
    @memset(attention_mask_buffer, 0);

    // fill actual tokens and mask
    const copy_len = @min(tokens_u32.len, max_seq_len);
    log.info("Copying {} tokens into buffer of length {}", .{ copy_len, max_seq_len });

    for (0..copy_len) |index| {
        input_ids_buffer[index] = @intCast(tokens_u32[index]);
        attention_mask_buffer[index] = 1;
    }

    defer allocator.free(tokens_u32);
    defer allocator.free(input_ids_buffer);
    defer allocator.free(attention_mask_buffer);

    log.info("Input buffer: {any}", .{input_ids_buffer});
    log.info("Attention buffer: {any}", .{attention_mask_buffer});

    const input_ids = try zml.Buffer.fromSlice(mod.platform(), input_shape.dims(), input_ids_buffer);
    log.info("input_ids: {}", .{input_ids});

    const attention_mask = try zml.Buffer.fromSlice(mod.platform(), attention_mask_shape.dims(), attention_mask_buffer);
    log.info("attention_mask: {}", .{attention_mask});

    // const prediction_scores = mod.call(.{ input_ids, attention_mask });
    // log.info("prediction_scores: {}", .{prediction_scores});
    //
    // var output = std.ArrayList(u8).init(allocator);
    // defer output.deinit();
    //
    // const mask_token_id = tokenizer.special_tokens.mask;
    // log.info("mask_token_id : {}", .{mask_token_id});
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        model: []const u8,
        tokenizer: ?[]const u8 = null,
        num_attention_heads: ?i64 = null,
        text: ?[]const u8 = null,
        create_options: []const u8 = "{}",
    };

    const allocator = std.heap.c_allocator;

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/modernbert/cache");

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/modernbert",
        .sharding_enabled = true,
    };

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const model_file = cli_args.model;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    const create_opts = try std.json.parseFromSliceLeaky(zml.Platform.CreateOptions, model_arena, cli_args.create_options, .{});
    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    log.info("Model file: {s}", .{model_file});

    var ts = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer ts.deinit();

    const num_attention_heads = cli_args.num_attention_heads orelse ts.metadata("num_heads", .int) orelse @panic("--num-attention-heads is required for this model");
    const modernbert_options = modernbert.ModernBertOptions{
        .num_attention_heads = num_attention_heads,
        .tie_word_embeddings = true, // TODO: from cli_args
    };
    var modern_bert_for_masked_lm = try zml.aio.populateModel(modernbert.ModernBertForMaskedLM, model_arena, ts);

    if (cli_args.tokenizer == null) {
        log.err("Model doesn't have an embbedded tokenizer, please provide a path to a tokenizer.", .{});
        @panic("No tokenizer provided");
    }

    modern_bert_for_masked_lm.init(modernbert_options);
    log.info("✅\tParsed ModernBERT config: {}", .{modernbert_options});

    if (cli_args.tokenizer == null) {
        log.err("Model doesn't have an embbedded tokenizer, please provide a path to a tokenizer.", .{});
        @panic("No tokenizer provided");
    }
    const tokenizer_path = cli_args.tokenizer orelse cli_args.model;
    log.info("\tLoading tokenizer from {s}", .{tokenizer_path});
    var tokenizer = try zml.aio.detectFormatAndLoadTokenizer(allocator, tokenizer_path);
    log.info("✅\tLoaded tokenizer from {s}", .{tokenizer_path});
    defer tokenizer.deinit();

    // Prepare shapes for compilation
    // Note: we compile the model without a batching dimension ?
    const input_shape = zml.Shape.init(.{ .b = 1, .s = max_seq_len }, .i64);
    const attention_mask_shape = input_shape;

    // Compile the model
    log.info("\tCompiling model...", .{});
    var start = try std.time.Timer.start();
    var fut_mod = try asynk.asyncc(zml.compile, .{
        allocator,
        modernbert.ModernBertForMaskedLM.forward,
        .{modernbert_options},
        .{ input_shape, attention_mask_shape },
        ts,
        platform,
    });

    // Load weights
    log.info("\tLoading ModernBERT weights from {s}...", .{cli_args.model});
    var bert_weights = try zml.aio.loadBuffers(modernbert.ModernBertForMaskedLM, .{modernbert_options}, ts, model_arena, platform);
    defer zml.aio.unloadBuffers(&bert_weights);
    log.info("✅\tLoaded weights in {d}ms", .{start.read() / std.time.ns_per_ms});

    var bert_module = (try fut_mod.awaitt()).prepare(bert_weights);
    defer bert_module.deinit();
    log.info("✅\tCompiled model in {d}ms", .{start.read() / std.time.ns_per_ms});

    const text = cli_args.text orelse "Zig is the [MASK] programming language."; // this text does not contain any characters that would be affected by NFC norm
    log.info("\tInput text: {s}", .{text});

    try predictMaskedTokens(modern_bert_for_masked_lm, bert_module, tokenizer, allocator, text);
}
