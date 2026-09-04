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
    prefill_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    decode_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    layer_index_buffers: []inference.LayerIndexBuffer,
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    seqlen: u32,
    eos_token_id: u32,
    special_tokens: model.Model.SpecialTokens,
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
        const shardings = compiled_model.params.shardings;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        var prefill_moe_metadata_buffers = try compiled_model.params.prefill_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&prefill_moe_metadata_buffers);
        var decode_moe_metadata_buffers = try compiled_model.params.decode_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&decode_moe_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        const layer_types = compiled_model.loaded_model.inner.config.text_config.layer_types;
        const layer_index_buffers = try allocator.alloc(inference.LayerIndexBuffer, layer_types.len);
        errdefer allocator.free(layer_index_buffers);
        var initialized_layer_index_buffers: usize = 0;
        errdefer for (layer_index_buffers[0..initialized_layer_index_buffers]) |*layer_index_buffer| {
            switch (layer_index_buffer.*) {
                .self_attn => |*buffer| buffer.deinit(),
                .linear_attn => |*buffer| buffer.deinit(),
            }
        };

        var self_attn_layer_index: u32 = 0;
        var linear_attn_layer_index: u32 = 0;
        for (layer_types, layer_index_buffers) |layer_type, *layer_index_buffer| {
            layer_index_buffer.* = switch (layer_type) {
                .full_attention => b: {
                    defer self_attn_layer_index += 1;
                    break :b .{ .self_attn = try .scalar(io, platform, self_attn_layer_index, .u32) };
                },
                .linear_attention => b: {
                    defer linear_attn_layer_index += 1;
                    break :b .{ .linear_attn = try .scalar(io, platform, linear_attn_layer_index, .u32) };
                },
            };
            initialized_layer_index_buffers += 1;
        }

        const generated_token_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
        errdefer generated_token_slice.free(allocator);

        var prefill = try inference.KernelRunner.init(allocator, &compiled_model.prefill, model_buffers);
        errdefer prefill.deinit(allocator);
        const decode = try inference.KernelRunner.init(allocator, &compiled_model.decode, model_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .prefill = prefill,
            .decode = decode,
            .kv_cache_buffers = kv_cache_buffers,
            .prefill_moe_metadata_buffers = prefill_moe_metadata_buffers,
            .decode_moe_metadata_buffers = decode_moe_metadata_buffers,
            .rng_buffers = rng_buffers,
            .layer_index_buffers = layer_index_buffers,
            .generated_token_slice = generated_token_slice,
            .tokenizer = tokenizer,
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = compiled_model.loaded_model.inner.special_tokens.end_of_text_token_id,
            .special_tokens = compiled_model.loaded_model.inner.special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.moe.Metadata.deinitBuffer(&self.prefill_moe_metadata_buffers);
        zml.moe.Metadata.deinitBuffer(&self.decode_moe_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        for (self.layer_index_buffers) |*layer_index_buffer| {
            switch (layer_index_buffer.*) {
                .self_attn => |*buffer| buffer.deinit(),
                .linear_attn => |*buffer| buffer.deinit(),
            }
        }
        self.allocator.free(self.layer_index_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const tokens_slice = try zml.Slice.alloc(self.allocator, .init(.{ .b = 1, .s = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);
        @memset(tokens_slice.items(u32), 0);
        @memcpy(tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        var tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, tokens_slice, .replicated);
        defer tokens_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var active_length_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.len)), .u32);
        defer active_length_buffer.deinit();

        inference.run(&self.prefill, .{
            .io = self.io,
            .tokens_buffer = &tokens_buffer,
            .token_index_buffer = &token_index_buffer,
            .active_length_buffer = &active_length_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .moe_metadata_buffers = self.prefill_moe_metadata_buffers,
            .rng_buffers = &self.rng_buffers,
            .layer_index_buffers = self.layer_index_buffers,
        });

        try tokens_buffer.toSlice(self.io, tokens_slice);
        self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();
        const out_tokens_buffer = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, .replicated);
        defer current_token_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
        defer token_index_buffer.deinit();
        var active_length_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 1), .u32);
        defer active_length_buffer.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start| if (token_id == think_start) try stdout.writeAll("\x1b[2m");
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) try stdout.writeAll("\x1b[0m");
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            inference.run(&self.decode, .{
                .io = self.io,
                .tokens_buffer = &current_token_buffer,
                .token_index_buffer = &token_index_buffer,
                .active_length_buffer = &active_length_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .moe_metadata_buffers = self.decode_moe_metadata_buffers,
                .rng_buffers = &self.rng_buffers,
                .layer_index_buffers = self.layer_index_buffers,
            });
            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};

fn tokenizeChatPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, special_tokens: model.Model.SpecialTokens, is_first_turn: bool) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id;

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
