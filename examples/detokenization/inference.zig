const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;

const main = @import("main.zig");
const llm_ = @import("llm.zig");
const graph_ = @import("graph.zig");
const model_ = @import("model.zig");
const tokens_ = @import("tokens.zig");

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

    const user_prompt = "Write a python script that computes the n-th prime number";

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    errdefer tokens.deinit(allocator);

    try tokens.append(allocator, im_start);
    try appendEncoded(allocator, &encoder, &tokens, "user\n");
    try appendEncoded(allocator, &encoder, &tokens, user_prompt);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    try appendEncoded(allocator, &encoder, &tokens, "assistant\n");
    try appendEncoded(allocator, &encoder, &tokens, "<think>\n\n</think>\n\n");

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
        llm.exes.prefill_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
        llm.exes.prefill_layer_exe.call(llm.exes.prefill_layer_args, &llm.exes.prefill_layer_results);
        llm.exes.prefill_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
    }
    llm.exes.prefill_select_args.set(.{ llm.model_buffers, prefill_embed_buffer, pred_buffer });
    llm.exes.prefill_select_exe.call(llm.exes.prefill_select_args, &llm.exes.prefill_select_results);
    llm.exes.prefill_select_results.fill(.{&one_embed_buffer});
    llm.exes.logits_args.set(.{ llm.model_buffers, one_embed_buffer });
    llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
    llm.exes.logits_results.fill(.{&logit_buffer});
    llm.exes.sample_args.set(.{ llm.model_buffers, logit_buffer, rng_buffers });
    llm.exes.sample_exe.call(llm.exes.sample_args, &llm.exes.sample_results);
    llm.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

    try token_buffer.toSlice(io, token_slice);
    token_buffer.deinit();
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
        llm.exes.decode_embed_results.fill(.{&decode_embed_buffer});
        for (0..llm.config.num_hidden_layers) |ii| {
            llm.exes.decode_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
            llm.exes.decode_layer_exe.call(llm.exes.decode_layer_args, &llm.exes.decode_layer_results);
            llm.exes.decode_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
        }
        llm.exes.logits_args.set(.{ llm.model_buffers, decode_embed_buffer });
        llm.exes.logits_exe.call(llm.exes.logits_args, &llm.exes.logits_results);
        llm.exes.logits_results.fill(.{&logit_buffer});

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
    zml_handler.toc(&zml_handler.timers.decode);
    const decode_ns = zml_handler.timers.decode.nanoseconds - decode_start_ns;
    const tokens_per_second = @as(f64, @floatFromInt(num_tokens_generated)) / (@as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s);
    std.log.info("LLM done, generated {d} tokens ({d:.2} token/s)", .{ num_tokens_generated, tokens_per_second });
    return result.toOwnedSlice(allocator);
}

pub fn generateTextGraph(zml_handler: *Zml_handler, llm: *llm_.Llm_handler, model_handler: *model_.Model_handler, g: *graph_.Graph, prompt_tok: []const u32) ![]u8 {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding: zml.Sharding = .replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    var zero_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{}, .u32));
    defer zero_slice.free(allocator);
    zero_slice.items(u32)[0] = 0;
    var zero_buffer: zml.Buffer = try .fromSlice(io, platform, zero_slice, sharding);
    defer zero_buffer.deinit();

    const embed_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = 1, .d = llm.options.hidden_size }, .f32));
    defer embed_slice.free(allocator);

    const pred_slice: zml.Slice = try .alloc(allocator, .init(.{}, .u32));
    defer pred_slice.free(allocator);
    pred_slice.items(u32)[0] = @intCast(prompt_tok.len - 1);
    var pred_buffer: zml.Buffer = try .fromSlice(io, platform, pred_slice, sharding);
    defer pred_buffer.deinit();

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

    var norm_weight_slice = try llm.model_buffers.norm.weights.toSliceAlloc(allocator, io);
    defer norm_weight_slice.free(allocator);
    const hidden_size: usize = @intCast(llm.options.hidden_size);
    const query = try allocator.alloc(f32, hidden_size);
    defer allocator.free(query);
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    std.log.info("LLM run graph prefill with seq_len/prompt_len of {d}/{d} tokens", .{ llm.options.seq_len, prompt_tok.len });
    zml_handler.tic(&zml_handler.timers.prefill);

    var prefill_embed_buffer: zml.Buffer = undefined;
    defer prefill_embed_buffer.deinit();
    var one_embed_buffer: zml.Buffer = undefined;
    defer one_embed_buffer.deinit();

    llm.exes.prefill_embed_args.set(.{ llm.model_buffers, prefill_tokens_buffer });
    llm.exes.prefill_embed_exe.call(llm.exes.prefill_embed_args, &llm.exes.prefill_embed_results);
    llm.exes.prefill_embed_results.fill(.{&prefill_embed_buffer});
    for (0..llm.config.num_hidden_layers) |i| {
        llm.exes.prefill_layer_args.set(.{ llm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, llm.kv_cache_buffers, layer_index_buffers[i] });
        llm.exes.prefill_layer_exe.call(llm.exes.prefill_layer_args, &llm.exes.prefill_layer_results);
        llm.exes.prefill_layer_results.fill(.{ &prefill_embed_buffer, &llm.kv_cache_buffers });
    }
    llm.exes.prefill_select_args.set(.{ llm.model_buffers, prefill_embed_buffer, pred_buffer });
    llm.exes.prefill_select_exe.call(llm.exes.prefill_select_args, &llm.exes.prefill_select_results);
    llm.exes.prefill_select_results.fill(.{&one_embed_buffer});
    llm.exes.graph_embed_args.set(.{ llm.model_buffers, one_embed_buffer });
    llm.exes.graph_embed_exe.call(llm.exes.graph_embed_args, &llm.exes.graph_embed_results);
    llm.exes.graph_embed_results.fill(.{&one_embed_buffer});

    try one_embed_buffer.toSlice(io, embed_slice);
    var generated_token = try sampleTokenFromGraph(zml_handler, model_handler, llm.tokenizer, embed_slice, g, random, llm.generation_config.top_k);
    zml_handler.toc(&zml_handler.timers.prefill);

    std.log.info("LLM run graph decode", .{});
    const decode_start_ns = zml_handler.timers.decode.nanoseconds;
    zml_handler.tic(&zml_handler.timers.decode);

    const decode_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = 1 }, .u32));
    defer decode_tokens_slice.free(allocator);
    var decode_embed_buffer: zml.Buffer = undefined;

    const output_tokens_len = llm.options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    //var stdout = std.Io.File.stdout().writer(io, &.{});
    //var writer: *std.Io.Writer = &stdout.interface;
    const decoder_out_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(decoder_out_buffer);
    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const chunk = try tokenizer_decoder.feedOne(generated_token, decoder_out_buffer);
        try result.appendSlice(allocator, chunk);
        std.log.info("Generated {d}-th token {d}: {s}", .{ i, generated_token, chunk });
        //try writer.writeAll(chunk);
        //try writer.flush();
        if (i == output_tokens_len or i >= 5) break :generation;
        if (llm.generation_config.isEosToken(generated_token)) break :generation;
        decode_tokens_slice.items(u32)[0] = generated_token;
        var decode_token_buffer: zml.Buffer = try .fromSlice(io, platform, decode_tokens_slice, sharding);
        defer decode_token_buffer.deinit();

        pred_slice.items(u32)[0] = @intCast(prompt_tok.len + i);
        var pos_buffer: zml.Buffer = try .fromSlice(io, platform, pred_slice, sharding);
        defer pos_buffer.deinit();

        llm.exes.decode_embed_args.set(.{ llm.model_buffers, decode_token_buffer });
        llm.exes.decode_embed_exe.call(llm.exes.decode_embed_args, &llm.exes.decode_embed_results);
        llm.exes.decode_embed_results.fill(.{&decode_embed_buffer});
        for (0..llm.config.num_hidden_layers) |ii| {
            llm.exes.decode_layer_args.set(.{ llm.model_buffers.layers[ii], decode_embed_buffer, pos_buffer, llm.kv_cache_buffers, layer_index_buffers[ii] });
            llm.exes.decode_layer_exe.call(llm.exes.decode_layer_args, &llm.exes.decode_layer_results);
            llm.exes.decode_layer_results.fill(.{ &decode_embed_buffer, &llm.kv_cache_buffers });
        }
        llm.exes.graph_embed_args.set(.{ llm.model_buffers, decode_embed_buffer });
        llm.exes.graph_embed_exe.call(llm.exes.graph_embed_args, &llm.exes.graph_embed_results);
        llm.exes.graph_embed_results.fill(.{&decode_embed_buffer});
        try decode_embed_buffer.toSlice(io, embed_slice);
        generated_token = try sampleTokenFromGraph(zml_handler, model_handler, llm.tokenizer, embed_slice, g, random, llm.generation_config.top_k);
        decode_embed_buffer.deinit();
    }
    const final_chunk = try tokenizer_decoder.finalize(decoder_out_buffer);
    try result.appendSlice(allocator, final_chunk);
    //try writer.writeAll(final_chunk);
    //try writer.writeAll("\n");
    //try writer.flush();
    zml_handler.toc(&zml_handler.timers.decode);
    const decode_ns = zml_handler.timers.decode.nanoseconds - decode_start_ns;
    const tokens_per_second = @as(f64, @floatFromInt(num_tokens_generated)) / (@as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s);
    std.log.info("LLM graph done, generated {d} tokens ({d:.2} token/s)", .{ num_tokens_generated, tokens_per_second });
    return result.toOwnedSlice(allocator);
}

pub fn sampleTokenFromGraph(zml_handler: *Zml_handler, model_handler: *model_.Model_handler, tokenizer: Tokenizer, embed_slice: zml.Slice, g: *graph_.Graph, random: std.Random, top_k: u32) !u32 {
    g.greedySearch(embed_slice.constItems(f32));

    try analyzeSamplings(zml_handler, model_handler, tokenizer, embed_slice, g);

    const nb_found = @min(g.L, top_k);

    const row_norms = g.lm_head_row_norms.constItems(f32);
    var max_score: f32 = -std.math.inf(f32);
    for (0..nb_found) |i| {
        const score = g.visited[i].similarity * row_norms[g.visited[i].node];
        max_score = @max(max_score, score);
    }
    var total: f32 = 0.0;
    for (0..nb_found) |i| {
        const score = g.visited[i].similarity * row_norms[g.visited[i].node];
        total += @exp(score - max_score);
    }

    const threshold = random.float(f32) * total;
    var cumulative: f32 = 0.0;
    for (0..nb_found) |i| {
        const score = g.visited[i].similarity * row_norms[g.visited[i].node];
        cumulative += @exp(score - max_score);
        if (cumulative >= threshold) return @intCast(g.visited[i].node);
    }
    return @intCast(g.visited[nb_found - 1].node);
}

pub fn analyzeSamplings(zml_handler: *Zml_handler, model_handler: *model_.Model_handler, tokenizer: Tokenizer, embed_slice: zml.Slice, g: *graph_.Graph) !void {
    const nb_printed_max = 32;
    const allocator = zml_handler.allocator;
    const row_norms = g.lm_head_row_norms.constItems(f32);
    const is_junk = g.is_junk;
    const embedding_norm = embeddingNorm(embed_slice);

    var embed_buffer: zml.Buffer = try .fromSlice(zml_handler.io, zml_handler.platform, embed_slice, .replicated);
    defer embed_buffer.deinit();

    model_handler.exes.score_args.set(.{ model_handler.model_buffers, embed_buffer });
    model_handler.exes.score_exe.call(model_handler.exes.score_args, &model_handler.exes.score_results);

    var sorted_probas_buffer: zml.Buffer = undefined;
    var sorted_indices_buffer: zml.Buffer = undefined;
    model_handler.exes.score_results.fill(.{ &sorted_probas_buffer, &sorted_indices_buffer });
    defer sorted_probas_buffer.deinit();
    defer sorted_indices_buffer.deinit();

    const sorted_probas_slice = try sorted_probas_buffer.toSliceAlloc(allocator, zml_handler.io);
    defer sorted_probas_slice.free(allocator);
    const sorted_indices_slice = try sorted_indices_buffer.toSliceAlloc(allocator, zml_handler.io);
    defer sorted_indices_slice.free(allocator);

    const sorted_probas = sorted_probas_slice.constItems(f32);
    const sorted_indices = sorted_indices_slice.constItems(u32);
    std.debug.assert(sorted_probas.len == sorted_indices.len);

    const nb_real = @min(sorted_probas.len, nb_printed_max);
    log.info("Real sampling distribution (top {d})", .{nb_real});
    printSamplingHeader();
    for (0..nb_real) |i| {
        const token_id: usize = @intCast(sorted_indices[i]);
        try printSamplingRow(tokenizer, i + 1, token_id, is_junk[token_id], sorted_probas[i], row_norms[token_id], 0.0);
    }

    try g.greedySearchWithLog(embed_slice.constItems(f32), tokenizer);
    const nb_graph = g.L;
    const entries = try allocator.alloc(SamplingEntry, nb_graph);
    defer allocator.free(entries);

    var max_score: f32 = -std.math.inf(f32);
    for (0..nb_graph) |i| {
        const node = g.visited[i].node;
        const score = g.visited[i].similarity * row_norms[node];
        max_score = @max(max_score, score);
    }

    var total: f32 = 0.0;
    for (0..nb_graph) |i| {
        const node = g.visited[i].node;
        const score = g.visited[i].similarity * row_norms[node];
        total += @exp(score - max_score);
    }

    for (0..nb_graph) |i| {
        const node = g.visited[i].node;
        const score = g.visited[i].similarity * row_norms[node];
        entries[i] = .{
            .token_id = node,
            .proba = @exp(score - max_score) / total,
            .is_junk = is_junk[node],
            .row_norm = row_norms[node],
            .similarity = g.visited[i].similarity / embedding_norm,
        };
    }
    std.mem.sort(SamplingEntry, entries, {}, SamplingEntry.beforeThan);

    const nb_printed_graph = @min(nb_graph, nb_printed_max);
    log.info("Graph sampling distribution over {d} visited nodes (top {d})", .{ nb_graph, nb_printed_graph });
    printSamplingHeader();
    for (0..nb_printed_graph) |i| {
        const entry = entries[i];
        try printSamplingRow(tokenizer, i + 1, entry.token_id, entry.is_junk, entry.proba, entry.row_norm, entry.similarity);
    }
}

fn embeddingNorm(embed_slice: zml.Slice) f32 {
    var norm2: f32 = 0.0;
    for (embed_slice.constItems(f32)) |x| {
        norm2 += x * x;
    }
    return @sqrt(norm2);
}

const SamplingEntry = struct {
    token_id: usize,
    is_junk: bool,
    proba: f32,
    row_norm: f32,
    similarity: f32,

    fn beforeThan(_: void, lhs: SamplingEntry, rhs: SamplingEntry) bool {
        return lhs.proba > rhs.proba or (lhs.proba == rhs.proba and lhs.token_id < rhs.token_id);
    }
};

fn printSamplingHeader() void {
    log.info("{s:>6}  {s:>10}  {s:>7}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", "is_junk", "proba", "row_norm", "similarity", "token" });
    log.info("{s:>6}  {s:>10}  {s:>7}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "-------", "--------------", "--------------", "--------------", "-----" });
}

fn printSamplingRow(tokenizer: Tokenizer, rank: usize, token_id: usize, is_junk: bool, proba: f32, row_norm: f32, similarity: f32) !void {
    const token_str = try tokens_.tokenString(tokenizer, @intCast(token_id), std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(token_str);
    log.info("{d:>6}  {d:>10}  {d:>7}  {d:>14.8}  {d:>14.6}  {d:>14.8}  {s}", .{ rank, token_id, @intFromBool(is_junk), proba, row_norm, similarity, token_str });
}
