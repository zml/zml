const std = @import("std");
const zml = @import("zml");

const main = @import("../main.zig");
const gemma = @import("gemma.zig");

const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;
const Gemma_handler = gemma.Gemma_handler;

pub const ActivationBlock = struct {
    token_count: u32,
    slices: gemma.ActivationCache.Slices,
};

pub const Activations = struct {
    blocks: std.ArrayList(ActivationBlock) = .empty,
    total_tokens: usize = 0,

    pub fn deinit(self: *Activations, allocator: std.mem.Allocator) void {
        for (self.blocks.items) |*block| {
            gemma.ActivationCache.deinitSlices(&block.slices, allocator);
        }
        self.blocks.deinit(allocator);
    }
};

pub const GenerationResult = struct {
    text: []u8,
    activations: ?Activations,

    pub fn deinit(self: *GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        if (self.activations) |*activations| activations.deinit(allocator);
    }
};

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

pub fn generateText(zml_handler: *Zml_handler, llm: *Gemma_handler, prompt_tok: []const u32) !GenerationResult {
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
    if (llm.collect_activations and prompt_tok.len >= gemma.ActivationCache.capacity) return error.PromptTooLongForActivationCache;

    var activations: ?Activations = if (llm.collect_activations) .{} else null;
    errdefer if (activations) |*collected| collected.deinit(allocator);
    var activation_tokens_in_block: usize = prompt_tok.len;

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
                if (llm.collect_activations) {
                    llm.exes.prefill_local_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i], llm.activation_cache_buffers.? });
                } else {
                    llm.exes.prefill_local_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
                }
                llm.exes.prefill_local_layer_exe.call(llm.exes.prefill_local_layer_args, &llm.exes.prefill_local_layer_results);
                if (llm.collect_activations) {
                    var next_activation_cache_buffers: zml.Bufferized(gemma.ActivationCache) = undefined;
                    llm.exes.prefill_local_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers, &next_activation_cache_buffers });
                    replaceActivationCacheBuffers(&llm.activation_cache_buffers.?, next_activation_cache_buffers);
                } else {
                    llm.exes.prefill_local_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
                }
            },
            .full_attention => {
                if (llm.collect_activations) {
                    llm.exes.prefill_global_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i], llm.activation_cache_buffers.? });
                } else {
                    llm.exes.prefill_global_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
                }
                llm.exes.prefill_global_layer_exe.call(llm.exes.prefill_global_layer_args, &llm.exes.prefill_global_layer_results);
                if (llm.collect_activations) {
                    var next_activation_cache_buffers: zml.Bufferized(gemma.ActivationCache) = undefined;
                    llm.exes.prefill_global_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers, &next_activation_cache_buffers });
                    replaceActivationCacheBuffers(&llm.activation_cache_buffers.?, next_activation_cache_buffers);
                } else {
                    llm.exes.prefill_global_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
                }
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
                    if (llm.collect_activations) {
                        llm.exes.decode_local_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii], llm.activation_cache_buffers.? });
                    } else {
                        llm.exes.decode_local_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
                    }
                    llm.exes.decode_local_layer_exe.call(llm.exes.decode_local_layer_args, &llm.exes.decode_local_layer_results);
                    if (llm.collect_activations) {
                        var next_activation_cache_buffers: zml.Bufferized(gemma.ActivationCache) = undefined;
                        llm.exes.decode_local_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers, &next_activation_cache_buffers });
                        replaceActivationCacheBuffers(&llm.activation_cache_buffers.?, next_activation_cache_buffers);
                    } else {
                        llm.exes.decode_local_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
                    }
                },
                .full_attention => {
                    if (llm.collect_activations) {
                        llm.exes.decode_global_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii], llm.activation_cache_buffers.? });
                    } else {
                        llm.exes.decode_global_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
                    }
                    llm.exes.decode_global_layer_exe.call(llm.exes.decode_global_layer_args, &llm.exes.decode_global_layer_results);
                    if (llm.collect_activations) {
                        var next_activation_cache_buffers: zml.Bufferized(gemma.ActivationCache) = undefined;
                        llm.exes.decode_global_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers, &next_activation_cache_buffers });
                        replaceActivationCacheBuffers(&llm.activation_cache_buffers.?, next_activation_cache_buffers);
                    } else {
                        llm.exes.decode_global_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
                    }
                },
            }
        }
        if (activations) |*collected| {
            activation_tokens_in_block += 1;
            if (activation_tokens_in_block == gemma.ActivationCache.capacity) {
                try appendActivationBlock(zml_handler, llm, collected, activation_tokens_in_block);
                try llm.resetActivationCache(zml_handler);
                activation_tokens_in_block = 0;
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
    if (activations) |*collected| {
        if (activation_tokens_in_block != 0) {
            try appendActivationBlock(zml_handler, llm, collected, activation_tokens_in_block);
        }
    }
    return .{
        .text = try result.toOwnedSlice(allocator),
        .activations = activations,
    };
}

fn appendActivationBlock(zml_handler: *Zml_handler, llm: *Gemma_handler, activations: *Activations, token_count: usize) !void {
    const cache = llm.activation_cache orelse return;
    const buffers = &llm.activation_cache_buffers.?;
    var slices = try cache.copyToHost(buffers, zml_handler.allocator, zml_handler.io);
    errdefer gemma.ActivationCache.deinitSlices(&slices, zml_handler.allocator);
    try activations.blocks.append(zml_handler.allocator, .{
        .token_count = @intCast(token_count),
        .slices = slices,
    });
    activations.total_tokens += token_count;
    std.log.info("Collected activation block: tokens={d} total={d}", .{ token_count, activations.total_tokens });
}

const activation_fields = [_]struct { name: []const u8, field: []const u8 }{
    .{ .name = "layer_input_residual", .field = "layer_input" },
    .{ .name = "input_rmsnorm", .field = "input_norm" },
    .{ .name = "q_after_qknorm_rope", .field = "q" },
    .{ .name = "k_after_qknorm_rope", .field = "k" },
    .{ .name = "v", .field = "v" },
    .{ .name = "attention_context", .field = "attention_context" },
    .{ .name = "attention_output_projection", .field = "attention_output" },
    .{ .name = "post_attention_residual", .field = "post_attention_residual" },
    .{ .name = "pre_ffn_rmsnorm", .field = "pre_ff_norm" },
    .{ .name = "gate_projection", .field = "gate" },
    .{ .name = "up_projection", .field = "up" },
    .{ .name = "geglu", .field = "geglu" },
    .{ .name = "post_mlp_residual", .field = "post_mlp_residual" },
};

pub fn exportActivations(zml_handler: *Zml_handler, file_name: []const u8, config: gemma.Config, activations: *const Activations) !void {
    if (activations.blocks.items.len == 0) return error.NoActivations;

    const allocator = zml_handler.allocator;
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    try header.writer.writeByte('{');

    var data_offset: u64 = 0;
    var first_entry = true;
    var local_layer_index: i64 = 0;
    var global_layer_index: i64 = 0;
    for (config.layer_types, 0..) |layer_type, layer_index| {
        const is_global = layer_type == .full_attention;
        const cache_layer_index = if (is_global) global_layer_index else local_layer_index;
        if (is_global) global_layer_index += 1 else local_layer_index += 1;

        inline for (activation_fields) |activation| {
            const source = activationSlice(&activations.blocks.items[0].slices, is_global, activation.field);
            var output_shape = source.shape.drop(.layer);
            output_shape = output_shape.set(.a, @intCast(activations.total_tokens));
            const byte_len: u64 = @intCast(output_shape.byteSize());

            if (!first_entry) try header.writer.writeByte(',');
            first_entry = false;
            try header.writer.print("\"model.layers.{d}.{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{ layer_index, activation.name });
            for (output_shape.dims(), 0..) |dim, dim_index| {
                if (dim_index != 0) try header.writer.writeByte(',');
                try header.writer.print("{d}", .{dim});
            }
            try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ data_offset, data_offset + byte_len });
            data_offset += byte_len;
            _ = cache_layer_index;
        }
    }
    try header.writer.writeByte('}');

    const padded_header_len = std.mem.alignForward(usize, header.written().len, 8);
    const padded_header = try allocator.alloc(u8, padded_header_len);
    defer allocator.free(padded_header);
    @memcpy(padded_header[0..header.written().len], header.written());
    @memset(padded_header[header.written().len..], ' ');

    var header_len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_bytes, @intCast(padded_header.len), .little);

    const checkpoint_path = localPathFromFileUri(zml_handler.uris.checkpoint) orelse return error.NonLocalCheckpointPath;
    const output_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ checkpoint_path, file_name });
    defer allocator.free(output_path);

    var file = try std.Io.Dir.createFile(.cwd(), zml_handler.local_io, output_path, .{ .truncate = true });
    defer file.close(zml_handler.local_io);
    try file.writeStreamingAll(zml_handler.local_io, &header_len_bytes);
    try file.writeStreamingAll(zml_handler.local_io, padded_header);

    local_layer_index = 0;
    global_layer_index = 0;
    for (config.layer_types) |layer_type| {
        const is_global = layer_type == .full_attention;
        const cache_layer_index = if (is_global) global_layer_index else local_layer_index;
        if (is_global) global_layer_index += 1 else local_layer_index += 1;

        inline for (activation_fields) |activation| {
            for (activations.blocks.items) |*block| {
                const source = activationSlice(&block.slices, is_global, activation.field);
                const layer_slice = source.subSlice(source.shape.axis(.layer), cache_layer_index, 1);
                const token_slice = layer_slice.subSlice(layer_slice.shape.axis(.a), 0, block.token_count);
                try file.writeStreamingAll(zml_handler.local_io, token_slice.constData()[0..token_slice.shape.byteSize()]);
            }
        }
    }

    std.log.info("Exported Gemma activations: file={s} tokens={d} blocks={d} bytes={d}", .{
        file_name,
        activations.total_tokens,
        activations.blocks.items.len,
        data_offset,
    });
}

fn activationSlice(slices: *const gemma.ActivationCache.Slices, is_global: bool, comptime field: []const u8) zml.Slice {
    return if (is_global) @field(slices, "global_" ++ field) else @field(slices, "local_" ++ field);
}

fn localPathFromFileUri(uri: []const u8) ?[]const u8 {
    const prefix = "file://";
    if (!std.mem.startsWith(u8, uri, prefix)) return null;
    return uri[prefix.len..];
}

fn replaceRngBuffer(current: *zml.Tensor.Rng.Buffer, next: zml.Tensor.Rng.Buffer) void {
    if (!sameBufferHandles(current._state, next._state)) {
        current._state.deinit();
    }
    current.* = next;
}

fn replaceActivationCacheBuffers(current: *zml.Bufferized(gemma.ActivationCache), next: zml.Bufferized(gemma.ActivationCache)) void {
    @setEvalBranchQuota(10_000);
    inline for (std.meta.fields(gemma.ActivationCache)) |field| {
        const current_buffer = &@field(current, field.name);
        const next_buffer = @field(next, field.name);
        if (!sameBufferHandles(current_buffer.*, next_buffer)) {
            current_buffer.deinit();
        }
        current_buffer.* = next_buffer;
    }
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
