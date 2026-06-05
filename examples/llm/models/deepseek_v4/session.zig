const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    seqlen: u32,
    cache_buffers: zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    rng_buf: zml.Bufferized(zml.Tensor.Rng),
    generated_token: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .seqlen = compiled_model.params.seqlen,
            .cache_buffers = try compiled_model.params.cache.initBuffers(io, platform, compiled_model.params.shardings.model),
            .attention_metadata_buffers = try compiled_model.params.attention_metadata.initBuffer(io, platform, compiled_model.params.shardings.model),
            .rng_buf = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed),
            .generated_token = try .alloc(allocator, zml.Shape.init(.{ .batch = 1, .seq = 1 }, .u32)),
            .tokenizer = tokenizer,
        };
    }

    pub fn deinit(self: *Session) void {
        model.Cache.unloadBuffers(&self.cache_buffers);
        attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buf);
        self.generated_token.free(self.allocator);
    }


    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const user_token = self.tokenizer.tokenId("<｜User｜>") orelse return error.NoSuchToken;
        const assistant_token = self.tokenizer.tokenId("<｜Assistant｜>") orelse return error.NoSuchToken;
        const think_token = self.tokenizer.tokenId("<think>") orelse return error.NoSuchToken;

        const bos_token = 0;

        const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
        defer allocator.free(prompt_tokens);

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ bos_token, user_token });
        try tokens.appendSlice(allocator, prompt_tokens);
        try tokens.appendSlice(allocator, &.{ assistant_token, think_token });
        return tokens.toOwnedSlice(allocator);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = prompt; // autofix
        return error.NotImplemented;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);
        const tokens = tokens_slice.items(u32);
        const pad_token_id = 2;
        @memset(tokens, pad_token_id);
        @memcpy(tokens[0..all_tokens.len], all_tokens);

        var tokens_buf: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice, .replicated);
        defer tokens_buf.deinit();

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var tokens_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, .replicated);
        defer tokens_pos_buf.deinit();

        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &tokens_buf,
            .tokens_pos_buf = &tokens_pos_buf,
            .rng_buf = &self.rng_buf,
            .cache_buffers = &self.cache_buffers,
            .attention_metadata_buffers = self.attention_metadata_buffers,
        });

        try tokens_buf.toSlice(self.io, tokens_slice);
        self.generated_token.items(u32)[0] = tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token, .replicated);
        defer current_token_buffer.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);

        const think_start = self.tokenizer.tokenId("<think>") orelse return error.NoSuchToken;
        const think_end = self.tokenizer.tokenId("</think>") orelse return error.NoSuchToken;

        const eos_token: u32 = 1;
        generation: while (true) {
            const token_id = self.generated_token.items(u32)[0];

            if (token_id == eos_token) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            }
            try stdout.writeAll(token);
            if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            }
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
            var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, .replicated);
            defer token_pos_buffer.deinit();

            try self.compiled_model.decode.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &current_token_buffer,
                .tokens_pos_buf = &token_pos_buffer,
                .rng_buf = &self.rng_buf,
                .cache_buffers = &self.cache_buffers,
                .attention_metadata_buffers = self.attention_metadata_buffers,
            });

            try current_token_buffer.toSlice(self.io, self.generated_token);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};
