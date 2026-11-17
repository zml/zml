const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: *inference.CompiledModel,
    prefill: inference.KernelRunner,
    decode: inference.KernelRunner,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    attention_index_buffers: []zml.Buffer,
    mamba_cache_buffers: []model.Mamba2.Cache.Buffers,
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    seqlen: u32,
    /// Position of the sampled token, counting audio frames as context positions
    decode_position: u32 = 0,
    think_start: ?u32,
    think_end: ?u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.init().initBuffer(io, platform, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        const attention_count: usize = @intCast(compiled_model.params.kv_cache.k.dim(.layer));
        const attention_index_buffers = try allocator.alloc(zml.Buffer, attention_count);
        errdefer allocator.free(attention_index_buffers);

        var initialized_attention_indices: usize = 0;
        errdefer for (attention_index_buffers[0..initialized_attention_indices]) |*buffer|
            buffer.deinit();

        for (attention_index_buffers, 0..) |*buffer, index| {
            buffer.* = try .scalar(io, platform, @as(u32, @intCast(index)), .u32);
            initialized_attention_indices += 1;
        }

        var mamba_count: usize = 0;
        for (compiled_model.loaded_model.inner.config.llm_config.layers_block_type) |layer_type| {
            if (layer_type == .mamba)
                mamba_count += 1;
        }
        const mamba_cache_buffers = try allocator.alloc(model.Mamba2.Cache.Buffers, mamba_count);
        errdefer allocator.free(mamba_cache_buffers);

        var initialized_mamba_caches: usize = 0;
        errdefer for (mamba_cache_buffers[0..initialized_mamba_caches]) |*cache| {
            model.Mamba2.Cache.deinitBuffer(cache);
        };

        const cache_template = model.Mamba2.Cache.init(
            compiled_model.loaded_model.inner.config,
            1,
            compiled_model.loaded_model.inner.backbone.embed_tokens.weight.dtype(),
        );
        for (mamba_cache_buffers) |*cache| {
            cache.* = try cache_template.initBuffer(allocator, io, platform);
            initialized_mamba_caches += 1;
        }

        const generated_token_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
        errdefer generated_token_slice.free(allocator);

        var prefill = try inference.KernelRunner.init(allocator, &compiled_model.prefill, model_buffers, attention_index_buffers);
        errdefer prefill.deinit(allocator);
        const decode = try inference.KernelRunner.init(allocator, &compiled_model.decode, model_buffers, attention_index_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .prefill = prefill,
            .decode = decode,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .attention_index_buffers = attention_index_buffers,
            .mamba_cache_buffers = mamba_cache_buffers,
            .generated_token_slice = generated_token_slice,
            .tokenizer = tokenizer,
            .seqlen = compiled_model.params.seqlen,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);

        for (self.attention_index_buffers) |*buffer|
            buffer.deinit();
        self.allocator.free(self.attention_index_buffers);

        for (self.mamba_cache_buffers) |*cache|
            model.Mamba2.Cache.deinitBuffer(cache);

        self.allocator.free(self.mamba_cache_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        if (all_tokens.len == 0)
            return error.EmptyPrompt;

        if (all_tokens.len >= self.seqlen)
            return error.PromptTooLong;

        const config = self.compiled_model.loaded_model.inner.config.llm_config;
        if (config.pad_token_id >= config.vocab_size)
            return error.InvalidPromptToken;

        for (all_tokens) |token| {
            if (token >= config.vocab_size)
                return error.InvalidPromptToken;
        }

        const tokens_slice = try zml.Slice.alloc(self.allocator, .init(.{ .b = 1, .s = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);

        @memset(tokens_slice.items(u32), config.pad_token_id);
        @memcpy(tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        var tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, tokens_slice);
        defer tokens_buffer.deinit();

        try self.runPrefillInputs(&tokens_buffer, all_tokens.len, null);
    }

    /// Prefill projected audio [1, T, hidden_size] in the backbone dtype, using only `audio_length` frames.
    /// The audio buffer remains caller-owned; the expanded prompt must leave room for generation.
    pub fn runAudioPrefill(self: *Session, audio_hidden: *const zml.Buffer, audio_length: usize) !void {
        const backbone = self.compiled_model.loaded_model.inner.backbone;
        const config = self.compiled_model.loaded_model.inner.config.llm_config;
        const shape = audio_hidden.shape();
        if (shape.rank() != 3 or shape.dim(0) != 1 or shape.dim(1) <= 0 or
            shape.dim(2) != config.hidden_size)
            return error.InvalidAudioShape;

        if (shape.dtype() != backbone.embed_tokens.weight.dtype())
            return error.InvalidAudioDtype;

        if (audio_length == 0 or audio_length > @as(usize, @intCast(shape.dim(1))))
            return error.InvalidAudioLength;

        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();
        const prompt_tokens = try encoder.encodeAlloc(self.allocator, audio_prompt);
        defer self.allocator.free(prompt_tokens);

        const audio_token = self.tokenizer.tokenId("<|audio|>") orelse return error.MissingAudioToken;
        var marker_index: ?usize = null;

        for (prompt_tokens, 0..) |token, index| {
            if (token == audio_token) {
                if (marker_index != null)
                    return error.InvalidAudioPrompt;

                marker_index = index;
            } else if (token >= config.vocab_size)
                return error.InvalidPromptToken;
        }

        const marker = marker_index orelse return error.InvalidAudioPrompt;
        if (marker == 0 or marker + 1 == prompt_tokens.len)
            return error.InvalidAudioPrompt;

        const text_length = prompt_tokens.len - 1;
        if (text_length >= self.seqlen or audio_length >= self.seqlen - text_length)
            return error.PromptTooLong;

        const expanded_length = text_length + audio_length;
        if (config.pad_token_id >= config.vocab_size)
            return error.InvalidPromptToken;

        const tokens_slice = try zml.Slice.alloc(self.allocator, .init(.{ .b = 1, .s = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);

        @memset(tokens_slice.items(u32), config.pad_token_id);
        @memcpy(tokens_slice.items(u32)[0..prompt_tokens.len], prompt_tokens);

        // The locator is outside the embedding vocabulary and must never reach a gather.
        tokens_slice.items(u32)[marker] = config.pad_token_id;

        var tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, tokens_slice);
        defer tokens_buffer.deinit();

        const params = self.compiled_model.params;
        const text_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen, .d = config.hidden_size }, shape.dtype())
            .withPartitioning(params.meshes.model, .{ .b = .replicated, .s = .replicated, .d = .replicated });
        const splice_exe = try zml.FnExe(spliceAudio).compile(self.allocator, self.io, self.platform, .{
            .meshes = &params.meshes.all(),
        }, .{.{
            .text = .fromShape(text_shape),
            .audio = .fromShape(shape),
            .marker = @intCast(marker),
            .prompt_length = @intCast(prompt_tokens.len),
            .audio_length = @intCast(audio_length),
        }});
        defer splice_exe.deinit();

        var splice_runner = try splice_exe.runner(self.allocator);
        defer splice_runner.deinit(self.allocator);

        var text_buffer: zml.Buffer = undefined;
        self.prefill.embed.run(self.io, .{
            .inputs = .{ .tokens = tokens_buffer },
            .outputs = .{ .hidden = &text_buffer },
        }, .{});
        defer text_buffer.deinit();

        var hidden_buffer: zml.Buffer = undefined;
        splice_runner.run(self.io, .{
            .inputs = .{ .text = text_buffer, .audio = audio_hidden.* },
            .outputs = .{ .hidden = &hidden_buffer },
        }, .{});
        defer hidden_buffer.deinit();

        try self.runPrefillInputs(&tokens_buffer, expanded_length, &hidden_buffer);
    }

    fn resetMambaCaches(self: *Session) !void {
        self.decode_position = 0;
        const mdl = self.compiled_model.loaded_model.inner;
        const template = model.Mamba2.Cache.init(mdl.config, 1, mdl.backbone.embed_tokens.weight.dtype());

        for (self.mamba_cache_buffers) |*cache| {
            const fresh = try template.initBuffer(self.allocator, self.io, self.platform);
            model.Mamba2.Cache.deinitBuffer(cache);
            cache.* = fresh;
        }
        // KV slots are overwritten from position zero; causal attention excludes stale future slots.
    }

    fn runPrefillInputs(self: *Session, tokens_buffer: *zml.Buffer, active_length: usize, hidden_buffer: ?*zml.Buffer) !void {
        try self.resetMambaCaches();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var active_length_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(active_length)), .u32);
        defer active_length_buffer.deinit();

        const params = self.compiled_model.params;
        var attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = switch (params.attention_metadata) {
            .stablehlo => try params.attention_metadata.initBuffer(self.io, self.platform),
            .flash_attention => |metadata| .{ .flash_attention = switch (metadata) {
                .metal => .{ .metal = .{ .num_tokens = try .scalar(self.io, self.platform, active_length, .u32) } },
                else => try metadata.initBuffer(self.io, self.platform),
            } },
        };
        defer zml.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        var next_token_buffer: zml.Buffer = undefined;
        const args: inference.RunArgs = .{
            .io = self.io,
            .tokens_buffer = tokens_buffer,
            .next_token_buffer = &next_token_buffer,
            .token_index_buffer = &token_index_buffer,
            .active_length_buffer = &active_length_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
            .attention_metadata_buffers = &attention_metadata_buffers,
            .mamba_cache_buffers = self.mamba_cache_buffers,
        };

        if (hidden_buffer) |hidden| {
            inference.runHidden(&self.prefill, args, hidden);
        } else {
            inference.run(&self.prefill, args);
        }

        defer next_token_buffer.deinit();
        try next_token_buffer.toSlice(self.io, self.generated_token_slice);
        self.decode_position = @intCast(active_length);
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        if (self.decode_position == 0)
            return error.MissingPrefill;

        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();
        const out_tokens_buffer = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice);
        defer current_token_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, self.decode_position, .u32);
        defer token_index_buffer.deinit();
        var active_length_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 1), .u32);
        defer active_length_buffer.deinit();

        const params = self.compiled_model.params;
        var attention_metadata_buffers = try params.attention_metadata.initBuffer(self.io, self.platform);
        defer zml.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        generation: while (self.decode_position < self.seqlen) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (self.compiled_model.loaded_model.inner.config.isEosToken(token_id))
                break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start|
                if (token_id == think_start)
                    try stdout.writeAll("\x1b[2m");

            try stdout.writeAll(token);

            if (self.think_end) |think_end|
                if (token_id == think_end)
                    try stdout.writeAll("\x1b[0m");

            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            self.decode_position += 1;
            if (self.decode_position >= self.seqlen)
                break :generation;

            inference.run(&self.decode, .{
                .io = self.io,
                .tokens_buffer = &current_token_buffer,
                .next_token_buffer = &current_token_buffer,
                .token_index_buffer = &token_index_buffer,
                .active_length_buffer = &active_length_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .attention_metadata_buffers = &attention_metadata_buffers,
                .mamba_cache_buffers = self.mamba_cache_buffers,
            });
            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};

const audio_prompt = "<|im_start|>system\n<|im_end|>\n" ++
    "<|im_start|>user\nTranscribe the following audio. <|audio|><|im_end|>\n" ++
    "<|im_start|>assistant\n<think></think>";

const AudioSpliceInput = struct {
    text: zml.Tensor,
    audio: zml.Tensor,
    marker: i64,
    prompt_length: i64,
    audio_length: i64,
};

fn spliceAudio(input: AudioSpliceInput) model.EmbedTokens.Output {
    const text = input.text.withTags(.{ .b, .s, .d });
    const prefix = text.slice(.s, .{ .start = 0, .len = input.marker });
    const audio = input.audio.withTags(.{ .b, .s, .d }).slice(.s, .{ .start = 0, .len = input.audio_length });
    const suffix = text.slice(.s, .{ .start = input.marker + 1, .len = input.prompt_length - input.marker - 1 });
    const hidden = zml.Tensor.concatenate(&.{ prefix, audio, suffix }, .s);

    zml.stdx.debug.assert(hidden.dim(.s) == input.prompt_length - 1 + input.audio_length and hidden.dim(.s) <= text.dim(.s), "Audio splice must replace exactly one marker without truncating the expanded prompt", .{});

    return .{ .hidden = hidden.pad(0, .{ .s = zml.Tensor.Pad{ .high = text.dim(.s) - hidden.dim(.s) } }) };
}

fn tokenizeChatPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, is_first_turn: bool) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>").?;
    const im_end = tokenizer.tokenId("<|im_end|>").?;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len + 32);
    if (!is_first_turn) {
        try tokens.append(allocator, im_end);
        const newline = try encoder.encodeAlloc(allocator, "\n");
        defer allocator.free(newline);
        try tokens.appendSlice(allocator, newline);
    }

    try tokens.append(allocator, im_start);
    const user = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user);
    try tokens.appendSlice(allocator, user);

    const prompt_encoded = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_encoded);
    try tokens.appendSlice(allocator, prompt_encoded);

    try tokens.append(allocator, im_end);
    const newline = try encoder.encodeAlloc(allocator, "\n");
    defer allocator.free(newline);
    try tokens.appendSlice(allocator, newline);

    try tokens.append(allocator, im_start);
    const assistant = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant);
    try tokens.appendSlice(allocator, assistant);

    return tokens.toOwnedSlice(allocator);
}
