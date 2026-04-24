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
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    config: *const model.Config,
    seqlen: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const shardings = compiled_model.params.shardings;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        var attention_metadata_buffers = try compiled_model.params.attention_metadata.initBuffer(io, platform, shardings.model);
        errdefer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, shardings.replicated);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .kv_cache_buffers = kv_cache_buffers,
            .attention_metadata_buffers = attention_metadata_buffers,
            .rng_buffers = rng_buffers,
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32)),
            .tokenizer = tokenizer,
            .config = &compiled_model.loaded_model.parsed_config.value,
            .seqlen = @intCast(compiled_model.params.seqlen),
        };
    }

    pub fn deinit(self: *Session) void {
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.attention.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const start_header = self.tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = self.tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = self.tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ self.config.bos_token_id, start_header });
        const user_tokens = try encoder.encodeAlloc(allocator, "user");
        defer allocator.free(user_tokens);
        try tokens.appendSlice(allocator, user_tokens);
        try tokens.appendSlice(allocator, &.{ end_header, newline });
        const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
        defer allocator.free(prompt_tokens);
        try tokens.appendSlice(allocator, prompt_tokens);
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header });
        const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant");
        defer allocator.free(assistant_tokens);
        try tokens.appendSlice(allocator, assistant_tokens);
        try tokens.appendSlice(allocator, &.{ end_header, newline });
        return tokens.toOwnedSlice(allocator);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const start_header = self.tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = self.tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = self.tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header });
        const user_tokens = try encoder.encodeAlloc(allocator, "user");
        defer allocator.free(user_tokens);
        try tokens.appendSlice(allocator, user_tokens);
        try tokens.appendSlice(allocator, &.{ end_header, newline });
        const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
        defer allocator.free(prompt_tokens);
        try tokens.appendSlice(allocator, prompt_tokens);
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header });
        const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant");
        defer allocator.free(assistant_tokens);
        try tokens.appendSlice(allocator, assistant_tokens);
        try tokens.appendSlice(allocator, &.{ end_header, newline });
        return tokens.toOwnedSlice(allocator);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        var prefill_args = try self.compiled_model.prefill_exe.args(self.allocator);
        defer prefill_args.deinit(self.allocator);

        var prefill_results = try self.compiled_model.prefill_exe.results(self.allocator);
        defer prefill_results.deinit(self.allocator);

        const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{self.seqlen}, .u32));
        defer prefill_tokens_slice.free(self.allocator);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_pos_buffer = try zml.Buffer.scalar(self.io, self.platform, 0, .u32, replicated_sharding);
        defer prefill_token_pos_buffer.deinit();

        prefill_args.set(.{
            self.model_buffers,
            prefill_tokens_buffer,
            prefill_token_pos_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
            &.{ .attnd = .{
                .conversation_id = try zml.Buffer.scalar(self.io, self.platform, 789, .u64, replicated_sharding),
                .layer_id = try zml.Buffer.scalar(self.io, self.platform, 0, .u16, replicated_sharding),
                .num_tokens = try zml.Buffer.scalar(self.io, self.platform, all_tokens.len, .u32, replicated_sharding),
            } },
        });
        self.compiled_model.prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{ &prefill_tokens_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

        self.generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        var decode_args = try self.compiled_model.decode_exe.args(self.allocator);
        defer decode_args.deinit(self.allocator);

        var decode_results = try self.compiled_model.decode_exe.results(self.allocator);
        defer decode_results.deinit(self.allocator);

        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

        var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];

            if (isEosToken(self.config, token_id)) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            try stdout.writeAll(token);
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
            var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, replicated_sharding);
            defer token_pos_buffer.deinit();

            decode_args.set(.{
                self.model_buffers,
                current_token_buffer,
                token_pos_buffer,
                &self.kv_cache_buffers,
                &self.rng_buffers,
                &.{ .attnd = .{
                    .conversation_id = try zml.Buffer.scalar(self.io, self.platform, 789, .u64, replicated_sharding),
                    .layer_id = try zml.Buffer.scalar(self.io, self.platform, 0, .u16, replicated_sharding),
                    .num_tokens = try zml.Buffer.scalar(self.io, self.platform, 1, .u32, replicated_sharding),
                } },
            });
            self.compiled_model.decode_exe.call(decode_args, &decode_results);

            decode_results.fill(.{ &current_token_buffer, &self.kv_cache_buffers, &self.rng_buffers });
            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
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
