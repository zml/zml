const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;

const main = @import("main.zig");
const llm_ = @import("llm.zig");

fn appendEncoded(allocator: std.mem.Allocator, encoder: *Tokenizer.Encoder, tokens: *std.ArrayList(u32), text: []const u8) !void {
    const encoded = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(encoded);
    try tokens.appendSlice(allocator, encoded);
}

pub fn tokenizePrompt(zml_handler: *Zml_handler, tokenizer: Tokenizer) ![]u32 {
    const allocator = zml_handler.allocator;
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse return error.NoSuchToken;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    const system_prompt = "# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n";
    const user_prompt = "Who are you ?";

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    errdefer tokens.deinit(allocator);

    try tokens.append(allocator, im_start);
    try appendEncoded(allocator, &encoder, &tokens, "system\n");
    try appendEncoded(allocator, &encoder, &tokens, system_prompt);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    try appendEncoded(allocator, &encoder, &tokens, "user\n");
    try appendEncoded(allocator, &encoder, &tokens, user_prompt);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    try appendEncoded(allocator, &encoder, &tokens, "assistant\n");

    return tokens.toOwnedSlice(allocator);
}

pub fn generateText(zml_handler: *Zml_handler, llm: *llm_.Llm_handler, prompt_tok: []const u32) ![]u8 {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding: zml.Sharding = .replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, sharding, 0);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var zero_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{}, .u32));
    defer zero_slice.free(allocator);
    zero_slice.items(u32)[0] = 0;
    var zero_buffer: zml.Buffer = try .fromSlice(io, platform, zero_slice, sharding);
    defer zero_buffer.deinit();

    const pred_slice: zml.Slice = try .alloc(allocator, .init(.{}, .u32));
    defer pred_slice.free(allocator);
    pred_slice.items(u32)[0] = @intCast(prompt_tok.len - 1);
    var pred_buffer: zml.Buffer = try .fromSlice(io, platform, pred_slice, sharding);
    defer pred_buffer.deinit();

    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);
    var token_buffer: zml.Buffer = undefined;

    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = llm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memset(prefill_tokens_slice.items(u32), llm.generation_config.pad_token_id);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);
    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();

    const layer_index_slices = try allocator.alloc(zml.Slice, llm.config.num_hidden_layers);
    defer {
        for (layer_index_slices) |*s| s.free(allocator);
        allocator.free(layer_index_slices);
    }
    for (0..llm.config.num_hidden_layers) |i| {
        layer_index_slices[i] = try zml.Slice.alloc(allocator, .init(.{}, .u32));
        layer_index_slices[i].items(u32)[0] = @intCast(i);
    }
    const layer_index_buffers = try allocator.alloc(zml.Buffer, llm.config.num_hidden_layers);
    defer {
        for (layer_index_buffers) |*s| s.deinit();
        allocator.free(layer_index_buffers);
    }
    for (0..llm.config.num_hidden_layers) |i| {
        layer_index_buffers[i] = try zml.Buffer.fromSlice(io, platform, layer_index_slices[i], sharding);
    }

    std.log.info("5Hz run prefill with seq_len/prompt_len of {d}/{d} tokens", .{ llm.options.seq_len, prompt_tok.len });
    zml_handler.tic(&zml_handler.timers.prefill);

    var prefill_embed_buffer: zml.Buffer = undefined;
    defer prefill_embed_buffer.deinit();
    var one_embed_buffer: zml.Buffer = undefined;
    defer one_embed_buffer.deinit();
    var logit_buffer: zml.Buffer = undefined;

    llm.exes.prefill_embed_args.set(.{ llm.model_buffers, prefill_tokens_buffer });
    llm.exes.prefill_embed_exe.call(llm.exes.prefill_embed_args, &llm.exes.prefill_embed_results);
    llm.exes.prefill_embed_results.fill(.{ &prefill_embed_buffer });
    for (0..llm.config.num_hidden_layers) |i| {
        llm.exes.prefill_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
        llm.exes.prefill_layer_exe.call(llm.exes.prefill_layer_args, &llm.exes.prefill_layer_results);
        llm.exes.prefill_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
    }
    llm.exes.prefill_select_args.set(.{ llm.model_buffers, prefill_embed_buffer, pred_buffer });
    llm.exes.prefill_select_exe.call(llm.exes.prefill_select_args, &llm.exes.prefill_select_results);
    llm.exes.prefill_select_results.fill(.{ &one_embed_buffer });
    llm.exes.logits_args.set(.{ llm.model_buffers, one_embed_buffer });
    llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
    llm.exes.logits_results.fill(.{ &logit_buffer });
    llm.exes.sample_args.set(.{ llm.model_buffers, logit_buffer, rng_buffers });
    llm.exes.sample_exe.call(llm.exes.sample_args, &llm.exes.sample_results);
    llm.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

    try token_buffer.toSlice(io, token_slice);
    token_buffer.deinit();
    logit_buffer.deinit();
    zml_handler.toc(&zml_handler.timers.prefill);

    std.log.info("5Hz run decode", .{});
    zml_handler.tic(&zml_handler.timers.decode);

    const decode_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = 1 }, .u32));
    defer decode_tokens_slice.free(allocator);
    var decode_embed_buffer: zml.Buffer = undefined;

    const output_tokens_len = llm.options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    const decoder_out_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(decoder_out_buffer);
    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = token_slice.items(u32)[0];
        const chunk = try tokenizer_decoder.feedOne(generated_token, decoder_out_buffer);
        try result.appendSlice(allocator, chunk);
        try writer.writeAll(chunk);
        try writer.flush();
        if (i == output_tokens_len) break :generation;
        if (llm.generation_config.isEosToken(generated_token)) break :generation;
        decode_tokens_slice.items(u32)[0] = generated_token;
        var decode_token_buffer: zml.Buffer = try .fromSlice(io, platform, decode_tokens_slice, sharding);
        defer decode_token_buffer.deinit();

        pred_slice.items(u32)[0] = @intCast(prompt_tok.len + i);
        var pos_buffer: zml.Buffer = try .fromSlice(io, platform, pred_slice, sharding);
        defer pos_buffer.deinit();

        // call to generate the next token
        llm.exes.decode_embed_args.set(.{ llm.model_buffers, decode_token_buffer });
        llm.exes.decode_embed_exe.call(llm.exes.decode_embed_args, &llm.exes.decode_embed_results);
        llm.exes.decode_embed_results.fill(.{ &decode_embed_buffer });
        for (0..llm.config.num_hidden_layers) |ii| {
            llm.exes.decode_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
            llm.exes.decode_layer_exe.call(llm.exes.decode_layer_args, &llm.exes.decode_layer_results);
            llm.exes.decode_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
        }
        llm.exes.logits_args.set(.{ llm.model_buffers, decode_embed_buffer });
        llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
        llm.exes.logits_results.fill(.{ &logit_buffer });

        llm.exes.sample_args.set(.{ llm.model_buffers, logit_buffer, rng_buffers });
        llm.exes.sample_exe.call(llm.exes.sample_args, &llm.exes.sample_results);
        llm.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

        // extract the generated token from the buffer
        try token_buffer.toSlice(io, token_slice);
        token_buffer.deinit();
        decode_embed_buffer.deinit();
        logit_buffer.deinit();
    }
    const final_chunk = try tokenizer_decoder.finalize(decoder_out_buffer);
    try result.appendSlice(allocator, final_chunk);
    try writer.writeAll(final_chunk);
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("LLM done, generated {d} tokens", .{ num_tokens_generated });
    zml_handler.toc(&zml_handler.timers.decode);
    return result.toOwnedSlice(allocator);
}