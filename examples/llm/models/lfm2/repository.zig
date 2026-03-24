const std = @import("std");

const zml = @import("zml");

const model = @import("model.zig");
const common = @import("../common.zig");

const parseConfig = common.parseConfig;
const Shardings = common.Shardings;

pub const Repository = struct {
    inner: model.Model,
    parsed_config: std.json.Parsed(model.Config),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !Repository {
        const parsed_config = try parseConfig(model.Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .inner = .init(allocator, store, parsed_config.value),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *Repository, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(self: *Repository, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node, shardings: Shardings) !model.Buffers {
        return try self.inner.loadBuffers(allocator, io, platform, store, progress, shardings);
    }

    pub fn unloadBuffers(self: *model.Buffers, allocator: std.mem.Allocator) void {
        model.Model.unloadBuffers(self, allocator);
    }

    pub fn tokenizePrompt(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
        const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ self.parsed_config.value.bos_token_id, im_start, user, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ im_end, newline });
        try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });

        return tokens.toOwnedSlice(allocator);
    }

    pub fn tokenizeTurn(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        _ = self;
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
        const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ im_end, newline, im_start, user, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ im_end, newline });
        try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });

        return tokens.toOwnedSlice(allocator);
    }
};
