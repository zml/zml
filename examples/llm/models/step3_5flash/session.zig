const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
const model = @import("model.zig");

fn initAttentionMetadataBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: *const inference.CompiledModel,
    phase: common.Phase,
) ![]zml.Bufferized(zml.attention.attention.Metadata) {
    const layers = compiled_model.loaded_model.inner.text_model.layers;
    const buffers = try allocator.alloc(zml.Bufferized(zml.attention.attention.Metadata), layers.len);
    errdefer allocator.free(buffers);

    var initialized: usize = 0;
    errdefer deinitAttentionMetadataBuffers(allocator, buffers[0..initialized]);

    const attention_parameters = switch (phase) {
        .prefill => compiled_model.params.prefill_attention_parameters,
        .decode => compiled_model.params.decode_attention_parameters,
    };
    const seqlen: i64 = switch (phase) {
        .prefill => @intCast(compiled_model.params.prefill_tokens.dim(.s)),
        .decode => @intCast(compiled_model.params.decode_tokens.dim(.s)),
    };

    for (layers, 0..) |layer, i| {
        const metadata: zml.attention.attention.Metadata = switch (attention_parameters) {
            .vanilla => compiled_model.params.attention_metadata,
            .attnd => compiled_model.params.attention_metadata,
            .nki => compiled_model.params.attention_metadata,
            .cuda_fa2 => .init(.fromBackend(.cuda_fa2, seqlen, layer.attn.num_q_heads)),
            .cuda_fa3 => .init(.fromBackend(.cuda_fa3, seqlen, layer.attn.num_q_heads)),
        };
        buffers[i] = try metadata.initBuffer(io, platform, compiled_model.params.shardings.model);
        initialized = i + 1;
    }

    return buffers;
}

fn deinitAttentionMetadataBuffers(
    allocator: std.mem.Allocator,
    buffers: []zml.Bufferized(zml.attention.attention.Metadata),
) void {
    for (buffers) |*buffer| {
        zml.attention.attention.Metadata.deinitBuffer(buffer);
    }
    allocator.free(buffers);
}

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    decode_runner: inference.KernelExe.Runner,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    prefill_attention_metadata_buffers: []zml.Bufferized(zml.attention.attention.Metadata),
    decode_attention_metadata_buffers: []zml.Bufferized(zml.attention.attention.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    generated_token_slice: zml.Slice,
    seqlen: u32,
    eos_token_id: u32,
    special_tokens: model.LoadedModel.SpecialTokens,
    think_start: ?u32,
    think_end: ?u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(allocator, io, platform, compiled_model.params.shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const prefill_attention_metadata_buffers = try initAttentionMetadataBuffers(allocator, io, platform, compiled_model, .prefill);
        errdefer deinitAttentionMetadataBuffers(allocator, prefill_attention_metadata_buffers);

        const decode_attention_metadata_buffers = try initAttentionMetadataBuffers(allocator, io, platform, compiled_model, .decode);
        errdefer deinitAttentionMetadataBuffers(allocator, decode_attention_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        var decode_runner = try compiled_model.decode.initRunner(
            allocator,
            io,
            platform,
            model_buffers,
        );
        errdefer decode_runner.deinit(allocator);

        const special_tokens = try compiled_model.loaded_model.specialTokens(tokenizer);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .decode_runner = decode_runner,
            .kv_cache_buffers = kv_cache_buffers,
            .prefill_attention_metadata_buffers = prefill_attention_metadata_buffers,
            .decode_attention_metadata_buffers = decode_attention_metadata_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = special_tokens.eos_token_id,
            .special_tokens = special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        self.decode_runner.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        deinitAttentionMetadataBuffers(self.allocator, self.prefill_attention_metadata_buffers);
        deinitAttentionMetadataBuffers(self.allocator, self.decode_attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        const prefill_tokens_slice = try zml.Slice.alloc(self.allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding: zml.Sharding = .replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        const prefill_token_index_slice: zml.Slice = .init(
            zml.Shape.init(.{ .s = 1 }, .u32),
            std.mem.sliceAsBytes(&[_]u32{0}),
        );
        var prefill_token_index_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_token_index_slice, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &prefill_tokens_buffer,
            .token_index_buf = &prefill_token_index_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
            .attention_metadata_buffers = self.prefill_attention_metadata_buffers,
        });

        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
        const replicated_sharding: zml.Sharding = .replicated;

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        const token_index_slice: zml.Slice = .init(
            zml.Shape.init(.{ .s = 1 }, .u32),
            std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}),
        );
        var token_index_buffer = try zml.Buffer.fromSlice(self.io, self.platform, token_index_slice, replicated_sharding);
        defer token_index_buffer.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start| if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            };
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            };
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            try self.decode_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &current_token_buffer,
                .token_index_buf = &token_index_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .attention_metadata_buffers = self.decode_attention_metadata_buffers,
            });

            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};

fn tokenizeChatPrompt(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    special_tokens: model.LoadedModel.SpecialTokens,
    is_first_turn: bool,
) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    if (!is_first_turn) {
        try tokens.appendSlice(allocator, &.{ im_end, newline });
    }

    try tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try tokens.appendSlice(allocator, user_tokens);
    const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_tokens);
    try tokens.appendSlice(allocator, prompt_tokens);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try tokens.appendSlice(allocator, assistant_tokens);

    return tokens.toOwnedSlice(allocator);
}
