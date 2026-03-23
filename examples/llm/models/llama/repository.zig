const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const parseConfig = common.parseConfig;
const Shardings = common.Shardings;

pub const Repository = struct {
    inner: model.Model,
    parsed_config: std.json.Parsed(model.Config),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !Repository {
        const parsed_config = try parseConfig(model.Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: model.Options = .{
            .sampling_strategy = .{ .topk = 1, .temperature = 1.0 },
            .max_seq_len = parsed_config.value.max_position_embeddings,
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
        const all_shardings = shardings.all();
        return try self.inner.loadBuffers(allocator, io, platform, store, &all_shardings, progress);
    }

    pub fn unloadBuffers(self: *model.Buffers, allocator: std.mem.Allocator) void {
        model.Model.unloadBuffers(self, allocator);
    }

    pub fn tokenizePrompt(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const start_header = tokenizer.tokenToId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = tokenizer.tokenToId("<|end_header_id|>") orelse return error.NoSuchToken;
        const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const eot = tokenizer.tokenToId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ self.parsed_config.value.bos_token_id, start_header, user, end_header, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header, assistant, end_header, newline });

        return tokens.toOwnedSlice(allocator);
    }

    pub fn tokenizeTurn(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        _ = self;
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const start_header = tokenizer.tokenToId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = tokenizer.tokenToId("<|end_header_id|>") orelse return error.NoSuchToken;
        const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const eot = tokenizer.tokenToId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = (try encoder.encode("\n"))[0];

        var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header, user, end_header, newline });
        try tokens.appendSlice(allocator, try encoder.encode(prompt));
        try tokens.appendSlice(allocator, &.{ eot, newline, start_header, assistant, end_header, newline });

        return tokens.toOwnedSlice(allocator);
    }
};
