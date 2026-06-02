const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    decode_runner: inference.KernelExe.Runner,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    token_index_buffers: []zml.Buffer,
    attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata),
    decode_attention_metadata_buffers: ?zml.Bufferized(zml.attention.attention.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    config: *const model.Config,
    seqlen: u32,
    last_generated_token: u32 = 0,
    conversation_id: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const shardings = &compiled_model.params.shardings;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const token_index_buffers = try allocator.alloc(zml.Buffer, compiled_model.params.seqlen);
        errdefer allocator.free(token_index_buffers);
        var initialized_token_index_buffers: usize = 0;
        errdefer {
            for (token_index_buffers[0..initialized_token_index_buffers]) |*token_index_buffer| {
                token_index_buffer.deinit();
            }
        }
        for (token_index_buffers, 0..) |*token_index_buffer, i| {
            token_index_buffer.* = try zml.Buffer.scalar(io, platform, i, .u32);
            initialized_token_index_buffers = i + 1;
        }

        var attention_metadata_buffers = try compiled_model.params.attention_metadata.initBuffer(io, platform, shardings.model);
        errdefer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        const conversation_id: u64 = @bitCast(std.Io.Clock.now(.real, io).toMicroseconds());

        var decode_attention_metadata_buffers: ?zml.Bufferized(zml.attention.attention.Metadata) = switch (compiled_model.params.decode_attention_parameters) {
            .attnd => b: {
                const buffers: zml.Bufferized(zml.attention.attention.Metadata) = .{ .attnd = .{
                    .conversation_id = try zml.Buffer.scalar(io, platform, conversation_id, .u64),
                    .layer_id = try zml.Buffer.scalar(io, platform, 0, .u16),
                    .num_tokens = try zml.Buffer.scalar(io, platform, 1, .u32),
                } };
                break :b buffers;
            },
            .vanilla, .cuda_fa2, .cuda_fa3, .nki => null,
        };
        errdefer if (decode_attention_metadata_buffers) |*buffers| zml.attention.attention.Metadata.deinitBuffer(buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        var decode_runner = try compiled_model.decode.initRunner(allocator, io, platform, model_buffers);
        errdefer decode_runner.deinit(allocator);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .decode_runner = decode_runner,
            .kv_cache_buffers = kv_cache_buffers,
            .token_index_buffers = token_index_buffers,
            .attention_metadata_buffers = attention_metadata_buffers,
            .decode_attention_metadata_buffers = decode_attention_metadata_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .config = &compiled_model.loaded_model.parsed_config.value,
            .seqlen = @intCast(compiled_model.params.seqlen),
            .conversation_id = conversation_id,
        };
    }

    pub fn deinit(self: *Session) void {
        self.decode_runner.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        for (self.token_index_buffers) |*token_index_buffer| {
            token_index_buffer.deinit();
        }
        self.allocator.free(self.token_index_buffers);
        zml.attention.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        if (self.decode_attention_metadata_buffers) |*buffers| zml.attention.attention.Metadata.deinitBuffer(buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const start_header = self.tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = self.tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = self.tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, &.{ self.config.bos_token_id, start_header });
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{ end_header, newline });
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "assistant");
        try encoder.appendTokens(w, &.{ end_header, newline });

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const start_header = self.tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = self.tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = self.tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{ end_header, newline });
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "assistant");
        try encoder.appendTokens(w, &.{ end_header, newline });

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{self.seqlen}, .u32));
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice, .replicated);
        defer prefill_tokens_buffer.deinit();

        const attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata) = switch (self.compiled_model.params.prefill_attention_parameters) {
            .attnd => .{ .attnd = .{
                .conversation_id = try zml.Buffer.scalar(self.io, self.platform, self.conversation_id, .u64),
                .layer_id = try zml.Buffer.scalar(self.io, self.platform, 0, .u16),
                .num_tokens = try zml.Buffer.scalar(self.io, self.platform, all_tokens.len, .u32),
            } },
            .vanilla, .cuda_fa2, .cuda_fa3, .nki => self.attention_metadata_buffers,
        };

        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &prefill_tokens_buffer,
            .token_index_buf = &self.token_index_buffers[0],
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
            .attention_metadata_buffers = &attention_metadata_buffers,
        });
        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

        self.last_generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder: zml.tokenizer.Tokenizer.Decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const decoder_out_buffer: []u8 = try self.allocator.alloc(u8, 256);
        defer self.allocator.free(decoder_out_buffer);

        var last_token_id: u32 = self.last_generated_token;
        var current_token_buffer: zml.Buffer = try .fromBytes(self.io, self.platform, .init(.{ .s = 1 }, .u32), .replicated, @ptrCast(&last_token_id));
        defer current_token_buffer.deinit();

        generation: while (true) {
            if (isEosToken(self.config, last_token_id)) break :generation;

            try stdout.writeAll(try decoder.feedOne(last_token_id, decoder_out_buffer));
            try stdout.flush();

            try all_tokens.append(self.allocator, last_token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata) =
                self.decode_attention_metadata_buffers orelse self.attention_metadata_buffers;

            try self.decode_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &current_token_buffer,
                .token_index_buf = &self.token_index_buffers[all_tokens.items.len],
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .attention_metadata_buffers = &attention_metadata_buffers,
            });
            last_token_id = try current_token_buffer.getValue(u32, self.io);
        }

        try stdout.writeAll(try decoder.finalize(decoder_out_buffer));
        try stdout.flush();
    }
};

fn isEosToken(config: *const model.Config, token_id: u32) bool {
    return switch (config.eos_token_id.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}
