const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const parseConfig = common.parseConfig;
const Shardings = common.Shardings;
const model = @import("model.zig");

pub const Repository = struct {
    inner: model.Model,
    parsed_config: std.json.Parsed(model.Config),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !Repository {
        const parsed_config = try parseConfig(model.Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: model.Model.GenOptions = .{
            .sampling_strategy = .{},
            .max_seq_len = parsed_config.value.text_config.max_position_embeddings,
        };

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *Repository, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(self: *Repository, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node, shardings: Shardings) !model.Buffers {
        const all_shardings = [_]zml.sharding.Sharding{shardings.replicated};
        return try self.inner.load(allocator, io, platform, store, &all_shardings, progress);
    }

    pub fn unloadBuffers(self: *model.Buffers, allocator: std.mem.Allocator) void {
        model.Model.unloadBuffers(self, allocator);
    }

    pub fn tokenizePrompt(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        return tokenizeChatPrompt(allocator, tokenizer, prompt, self.inner.special_tokens, true);
    }

    pub fn tokenizeTurn(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        return tokenizeChatPrompt(allocator, tokenizer, prompt, self.inner.special_tokens, false);
    }
};

fn tokenizeChatPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, special_tokens: model.Model.SpecialTokens, is_first_turn: bool) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenToId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenToId("<|im_end|>") orelse special_tokens.im_end_token_id;
    const think = tokenizer.tokenToId("<think>") orelse return error.NoSuchToken;
    const newline = try encodeSingleToken(&encoder, "\n");
    const user_prefix = try encoder.encode("user\n");
    const assistant_prefix = try encoder.encode("assistant\n");
    const encoded_prompt = try encoder.encode(prompt);

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, encoded_prompt.len + user_prefix.len + assistant_prefix.len + 8);
    if (!is_first_turn) {
        try tokens.appendSlice(allocator, &.{ im_end, newline });
    }

    try tokens.append(allocator, im_start);
    try tokens.appendSlice(allocator, user_prefix);
    try tokens.appendSlice(allocator, encoded_prompt);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    try tokens.appendSlice(allocator, assistant_prefix);
    try tokens.appendSlice(allocator, &.{ think, newline });

    return tokens.toOwnedSlice(allocator);
}

fn encodeSingleToken(encoder: *zml.tokenizer.Tokenizer.Encoder, text: []const u8) !u32 {
    const encoded = try encoder.encode(text);
    if (encoded.len != 1) return error.InvalidTokenizerEncoding;
    return encoded[0];
}
