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
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    step_token_slice: zml.Slice,
    generated_token_slice: zml.Slice,
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
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const replicated_sharding = compiled_model.params.shardings.replicated;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, compiled_model.params.shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .step_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = compiled_model.loaded_model.inner.special_tokens.end_of_text_token_id,
            .special_tokens = compiled_model.loaded_model.inner.special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.step_token_slice.free(self.allocator);
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

        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        var args = try self.compiled_model.prefill_exe.args(self.allocator);
        defer args.deinit(self.allocator);

        var results = try self.compiled_model.prefill_exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{
            self.model_buffers,
            prefill_tokens_buffer,
            prefill_token_index_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
        });
        self.compiled_model.prefill_exe.call(args, &results);

        results.fill(.{ &prefill_tokens_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
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

            try self.runDecodeStep(token_id, @intCast(all_tokens.items.len));
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }

    fn runDecodeStep(self: *Session, token_id: u32, token_index: u32) !void {
        self.step_token_slice.items(u32)[0] = token_id;

        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.step_token_slice, replicated_sharding);
        defer token_buffer.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, token_index, .u32, replicated_sharding);
        defer token_index_buffer.deinit();

        var args = try self.compiled_model.decode_exe.args(self.allocator);
        defer args.deinit(self.allocator);

        var results = try self.compiled_model.decode_exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{
            self.model_buffers,
            token_buffer,
            token_index_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
        });
        self.compiled_model.decode_exe.call(args, &results);

        results.fill(.{ &token_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try token_buffer.toSlice(self.io, self.generated_token_slice);
    }
};

fn tokenizeChatPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, special_tokens: model.Model.SpecialTokens, is_first_turn: bool) ![]const u32 {
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
