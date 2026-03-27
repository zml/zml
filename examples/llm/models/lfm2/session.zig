const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const inference = @import("inference.zig");
const model = @import("model.zig");
const tracer = zml.tracer;

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    config: *const model.Config,
    seqlen: u32,
    cache_buffers: zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    rng_buf: zml.Bufferized(zml.Tensor.Rng),
    generated_token_slice: zml.Slice,
    think_start: ?u32,
    think_end: ?u32,
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
            .tokenizer = tokenizer,
            .config = &compiled_model.loaded_model.parsed_config.value,
            .seqlen = compiled_model.params.seqlen,
            .cache_buffers = try compiled_model.params.cache.initBuffers(allocator, io, platform, compiled_model.params.shardings.replicated),
            .attention_metadata_buffers = try compiled_model.params.attention_metadata.initBuffer(io, platform, compiled_model.params.shardings.model),
            .rng_buf = try zml.Tensor.Rng.initBuffer(platform, seed, io, compiled_model.params.shardings.replicated),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .batch = 1, .seq = 1 }, .u32)),
            .think_start = tokenizer.tokenToId("<think>") orelse unreachable,
            .think_end = tokenizer.tokenToId("</think>") orelse unreachable,
        };
    }

    pub fn deinit(self: *Session) void {
        model.Cache.unloadBuffers(&self.cache_buffers);
        attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buf);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var trace = tracer.scope("lfm2.tokenize_prompt");
        defer trace.end();

        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const im_start = self.tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = self.tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
        const user = self.tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = self.tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ self.config.bos_token_id, im_start, user, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ im_end, newline });
        try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });
        return tokens.toOwnedSlice(allocator);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var trace = tracer.scope("lfm2.tokenize_turn");
        defer trace.end();

        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const im_start = self.tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = self.tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
        const user = self.tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = self.tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ im_end, newline, im_start, user, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ im_end, newline });
        try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });
        return tokens.toOwnedSlice(allocator);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        var trace = try tracer.scopeWith(self.allocator, "lfm2.prefill", .{
            .tokens = all_tokens.len,
        });
        defer trace.end();

        const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);
        const tokens = tokens_slice.items(u32);
        @memset(tokens, self.config.pad_token_id);
        @memcpy(tokens[0..all_tokens.len], all_tokens);

        const sharding = try zml.sharding.replicatedSharding(self.platform);

        var tokens_buf: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice, sharding);
        defer tokens_buf.deinit();

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var tokens_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, sharding);
        defer tokens_pos_buf.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.len)}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice, sharding);
        defer actual_seq_len_buf.deinit();

        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &tokens_buf,
            .tokens_pos_buf = &tokens_pos_buf,
            .actual_seq_len_buf = &actual_seq_len_buf,
            .rng_buf = &self.rng_buf,
            .cache_buffers = &self.cache_buffers,
            .attention_metadata_buffers = self.attention_metadata_buffers,
        });

        try tokens_buf.toSlice(self.io, tokens_slice);
        self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var trace = tracer.scope("lfm2.decode");
        defer trace.end();

        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const sharding = try zml.sharding.replicatedSharding(self.platform);

        var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, sharding);
        defer current_token_buffer.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice, sharding);
        defer actual_seq_len_buf.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];

            if (token_id == self.config.eos_token_id) break :generation;

            if (try decoder.next(token_id)) |token| {
                if (self.think_start) |think_start| if (token_id == think_start) {
                    try stdout.writeAll("\x1b[2m");
                };
                try stdout.writeAll(token);
                if (self.think_end) |think_end| if (token_id == think_end) {
                    try stdout.writeAll("\x1b[0m");
                };
                try stdout.flush();
            }

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
            var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, sharding);
            defer token_pos_buffer.deinit();

            {
                var step_trace = try tracer.scopeWith(self.allocator, "lfm2.decode_step", .{
                    .token_index = all_tokens.items.len,
                });
                defer step_trace.end();

                try self.compiled_model.decode.run(.{
                    .allocator = self.allocator,
                    .io = self.io,
                    .platform = self.platform,
                    .model_buffers = self.model_buffers,
                    .tokens_buf = &current_token_buffer,
                    .tokens_pos_buf = &token_pos_buffer,
                    .actual_seq_len_buf = &actual_seq_len_buf,
                    .rng_buf = &self.rng_buf,
                    .cache_buffers = &self.cache_buffers,
                    .attention_metadata_buffers = self.attention_metadata_buffers,
                });

                try current_token_buffer.toSlice(self.io, self.generated_token_slice);
            }
        }
    }
};
