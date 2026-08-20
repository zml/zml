const std = @import("std");
const zml = @import("zml");

const main = @import("../main.zig");
const gemma = @import("gemma.zig");

const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;
const Gemma_handler = gemma.Gemma_handler;

pub fn tokenizePrompt(zml_handler: *Zml_handler, tokenizer: Tokenizer, prompt: []const u8) ![]u32 {
    const allocator = zml_handler.allocator;
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const bos = tokenizer.tokenId("<bos>") orelse return error.NoSuchToken;
    const start_turn = tokenizer.tokenId("<|turn>") orelse return error.NoSuchToken;
    const end_turn = tokenizer.tokenId("<turn|>") orelse return error.NoSuchToken;
    const start_channel = tokenizer.tokenId("<|channel>") orelse return error.NoSuchToken;
    const end_channel = tokenizer.tokenId("<channel|>") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    errdefer tokens.deinit(allocator);

    // Gemma 4 canonical template with thinking disabled.
    try tokens.appendSlice(allocator, &.{ bos, start_turn });
    try appendEncoded(allocator, &encoder, &tokens, "user\n");
    try appendEncoded(allocator, &encoder, &tokens, std.mem.trim(u8, prompt, " \t\r\n"));
    try tokens.append(allocator, end_turn);
    try appendEncoded(allocator, &encoder, &tokens, "\n");
    try tokens.append(allocator, start_turn);
    try appendEncoded(allocator, &encoder, &tokens, "model\n");
    try tokens.append(allocator, start_channel);
    try appendEncoded(allocator, &encoder, &tokens, "thought\n");
    try tokens.append(allocator, end_channel);

    return tokens.toOwnedSlice(allocator);
}

fn appendEncoded(allocator: std.mem.Allocator, encoder: *Tokenizer.Encoder, tokens: *std.ArrayList(u32), text: []const u8) !void {
    const encoded = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(encoded);
    try tokens.appendSlice(allocator, encoded);
}

pub fn generateText(zml_handler: *Zml_handler, llm: *Gemma_handler, prompt_tok: []const u32) ![]u8 {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding: zml.Sharding = .replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, sharding, 0);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    if (prompt_tok.len == 0) return error.EmptyPrompt;
    if (prompt_tok.len >= llm.options.seq_len) return error.PromptTooLong;

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
    var local_layer_index: u32 = 0;
    var global_layer_index: u32 = 0;
    for (0..llm.config.num_hidden_layers) |i| {
        layer_index_slices[i] = try zml.Slice.alloc(allocator, .init(.{}, .u32));
        switch (llm.config.layer_types[i]) {
            .sliding_attention => {
                layer_index_slices[i].items(u32)[0] = local_layer_index;
                local_layer_index += 1;
            },
            .full_attention => {
                layer_index_slices[i].items(u32)[0] = global_layer_index;
                global_layer_index += 1;
            },
        }
    }
    const layer_index_buffers = try allocator.alloc(zml.Buffer, llm.config.num_hidden_layers);
    defer {
        for (layer_index_buffers) |*s| s.deinit();
        allocator.free(layer_index_buffers);
    }
    for (0..llm.config.num_hidden_layers) |i| {
        layer_index_buffers[i] = try zml.Buffer.fromSlice(io, platform, layer_index_slices[i], sharding);
    }

    std.log.info("LLM run prefill with seq_len/prompt_len of {d}/{d} tokens", .{ llm.options.seq_len, prompt_tok.len });
    zml_handler.tic(&zml_handler.timers.prefill);

    var prefill_embed_buffer: zml.Buffer = undefined;
    defer prefill_embed_buffer.deinit();
    var one_embed_buffer: zml.Buffer = undefined;
    defer one_embed_buffer.deinit();
    var logit_buffer: zml.Buffer = undefined;

    llm.exes.prefill_embed_args.set(.{ llm.model_buffers, prefill_tokens_buffer });
    llm.exes.prefill_embed_exe.call(llm.exes.prefill_embed_args, &llm.exes.prefill_embed_results);
    llm.exes.prefill_embed_results.fill(.{&prefill_embed_buffer});
    for (0..llm.config.num_hidden_layers) |i| {
        switch (llm.config.layer_types[i]) {
            .sliding_attention => {
                llm.exes.prefill_local_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
                llm.exes.prefill_local_layer_exe.call(llm.exes.prefill_local_layer_args, &llm.exes.prefill_local_layer_results);
                llm.exes.prefill_local_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
            },
            .full_attention => {
                llm.exes.prefill_global_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
                llm.exes.prefill_global_layer_exe.call(llm.exes.prefill_global_layer_args, &llm.exes.prefill_global_layer_results);
                llm.exes.prefill_global_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
            },
        }
    }
    llm.exes.prefill_select_args.set(.{ llm.model_buffers, prefill_embed_buffer, pred_buffer });
    llm.exes.prefill_select_exe.call(llm.exes.prefill_select_args, &llm.exes.prefill_select_results);
    llm.exes.prefill_select_results.fill(.{&one_embed_buffer});
    llm.exes.logits_args.set(.{ llm.model_buffers, one_embed_buffer });
    llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
    llm.exes.logits_results.fill(.{&logit_buffer});
    llm.exes.sample_args.set(.{ llm.model_buffers, logit_buffer, llm.sampling_strategy_buffers, rng_buffers });
    llm.exes.sample_exe.call(llm.exes.sample_args, &llm.exes.sample_results);
    var next_rng_buffers: zml.Tensor.Rng.Buffer = undefined;
    llm.exes.sample_results.fill(.{ &token_buffer, &next_rng_buffers });
    try token_buffer.toSlice(io, token_slice);
    token_buffer.deinit();
    replaceRngBuffer(&rng_buffers, next_rng_buffers);
    logit_buffer.deinit();

    zml_handler.toc(&zml_handler.timers.prefill);

    std.log.info("LLM run decode", .{});
    const decode_start_ns = zml_handler.timers.decode.nanoseconds;
    zml_handler.tic(&zml_handler.timers.decode);

    const decode_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = 1 }, .u32));
    defer decode_tokens_slice.free(allocator);
    var decode_embed_buffer: zml.Buffer = undefined;

    const output_tokens_len = llm.options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    errdefer result.deinit(allocator);
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    const decoder_out_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(decoder_out_buffer);
    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = token_slice.items(u32)[0];
        if (llm.generation_config.isEosToken(generated_token)) break :generation;
        const chunk = try tokenizer_decoder.feedOne(generated_token, decoder_out_buffer);
        try result.appendSlice(allocator, chunk);
        try writer.writeAll(chunk);
        try writer.flush();
        if (i == output_tokens_len) break :generation;
        decode_tokens_slice.items(u32)[0] = generated_token;
        var decode_token_buffer: zml.Buffer = try .fromSlice(io, platform, decode_tokens_slice, sharding);
        defer decode_token_buffer.deinit();

        pred_slice.items(u32)[0] = @intCast(prompt_tok.len + i);
        var pos_buffer: zml.Buffer = try .fromSlice(io, platform, pred_slice, sharding);
        defer pos_buffer.deinit();

        // call to generate the next token
        llm.exes.decode_embed_args.set(.{ llm.model_buffers, decode_token_buffer });
        llm.exes.decode_embed_exe.call(llm.exes.decode_embed_args, &llm.exes.decode_embed_results);
        llm.exes.decode_embed_results.fill(.{&decode_embed_buffer});
        for (0..llm.config.num_hidden_layers) |ii| {
            switch (llm.config.layer_types[ii]) {
                .sliding_attention => {
                    llm.exes.decode_local_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
                    llm.exes.decode_local_layer_exe.call(llm.exes.decode_local_layer_args, &llm.exes.decode_local_layer_results);
                    llm.exes.decode_local_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
                },
                .full_attention => {
                    llm.exes.decode_global_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
                    llm.exes.decode_global_layer_exe.call(llm.exes.decode_global_layer_args, &llm.exes.decode_global_layer_results);
                    llm.exes.decode_global_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
                },
            }
        }
        llm.exes.logits_args.set(.{ llm.model_buffers, decode_embed_buffer });
        llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
        llm.exes.logits_results.fill(.{&logit_buffer});
        llm.exes.sample_args.set(.{ llm.model_buffers, logit_buffer, llm.sampling_strategy_buffers, rng_buffers });
        llm.exes.sample_exe.call(llm.exes.sample_args, &llm.exes.sample_results);
        llm.exes.sample_results.fill(.{ &token_buffer, &next_rng_buffers });
        try token_buffer.toSlice(io, token_slice);
        token_buffer.deinit();
        replaceRngBuffer(&rng_buffers, next_rng_buffers);
        decode_embed_buffer.deinit();
        logit_buffer.deinit();
    }
    const final_chunk = try tokenizer_decoder.finalize(decoder_out_buffer);
    try result.appendSlice(allocator, final_chunk);
    try writer.writeAll(final_chunk);
    try writer.writeAll("\n");
    try writer.flush();
    zml_handler.toc(&zml_handler.timers.decode);
    const decode_ns = zml_handler.timers.decode.nanoseconds - decode_start_ns;
    const tokens_per_second = @as(f64, @floatFromInt(num_tokens_generated)) / (@as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s);
    std.log.info("LLM done, generated {d} tokens ({d:.2} token/s)", .{ num_tokens_generated, tokens_per_second });
    return result.toOwnedSlice(allocator);
}

fn replaceRngBuffer(current: *zml.Tensor.Rng.Buffer, next: zml.Tensor.Rng.Buffer) void {
    if (!sameBufferHandles(current._state, next._state)) {
        current._state.deinit();
    }
    current.* = next;
}

fn sameBufferHandles(lhs: zml.Buffer, rhs: zml.Buffer) bool {
    var lhs_shards = lhs.shards();
    var rhs_shards = rhs.shards();
    while (lhs_shards.next()) |lhs_shard| {
        const rhs_shard = rhs_shards.next() orelse return false;
        if (lhs_shard._pjrt_buffer != rhs_shard._pjrt_buffer) return false;
    }
    return rhs_shards.next() == null;
}
