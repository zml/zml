const std = @import("std");

const zml = @import("zml");
const kimi_tokenizer = @import("tokenizer.zig");

const thinking_effort_max =
    "\x60thinking_effort\x60 guides on how much to think in your thinking channel (not including the response channel), " ++
    "supported values include \x60low\x60, \x60medium\x60, \x60high\x60, and \x60max\x60.\n" ++
    "Now the system is invoked with \x60thinking_effort=max\x60.";


const Builder = struct {
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    tokens: std.ArrayList(u32),

    fn init(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, capacity: usize) !Builder {
        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = try .initCapacity(allocator, capacity),
        };
    }

    fn deinit(self: *Builder) void {
        self.tokens.deinit(self.allocator);
    }

    fn finish(self: *Builder) ![]u32 {
        const owned = try self.tokens.toOwnedSlice(self.allocator);
        self.* = undefined;
        return owned;
    }

    fn control(self: *Builder, spelling: []const u8) !void {
        try self.tokens.append(
            self.allocator,
            self.tokenizer.tokenId(spelling) orelse return error.KimiK3MissingControlToken,
        );
    }

    /// Moonshot encodes every EncodeSegment independently. A fresh
    /// no-special encoder preserves those exact BPE boundaries and prevents a
    /// literal control spelling in user text from becoming an XTML token.
    fn text(self: *Builder, value: []const u8) !void {
        if (value.len == 0) return;
        const encoded = try kimi_tokenizer.encodeText(self.allocator, self.tokenizer, value);
        defer self.allocator.free(encoded);
        try self.tokens.appendSlice(self.allocator, encoded);
    }

    fn attribute(self: *Builder, key: []const u8, value: []const u8) !void {
        const key_with_space = try std.fmt.allocPrint(self.allocator, " {s}", .{key});
        defer self.allocator.free(key_with_space);
        try self.text(key_with_space);
        try self.text("=\"");

        var escaped: std.Io.Writer.Allocating = .init(self.allocator);
        defer escaped.deinit();
        for (value) |byte| switch (byte) {
            '&' => try escaped.writer.writeAll("&amp;"),
            '"' => try escaped.writer.writeAll("&quot;"),
            else => try escaped.writer.writeByte(byte),
        };
        try self.text(escaped.written());
        try self.text("\"");
    }

    fn openTag(self: *Builder, tag: []const u8, attributes: []const [2][]const u8) !void {
        try self.control("<|open|>");
        try self.text(tag);
        for (attributes) |attribute_| try self.attribute(attribute_[0], attribute_[1]);
        try self.control("<|sep|>");
    }

    fn closeTag(self: *Builder, tag: []const u8) !void {
        try self.control("<|close|>");
        try self.text(tag);
        try self.control("<|sep|>");
    }

    fn endMessage(self: *Builder) !void {
        try self.control("<|end_of_msg|>");
    }

    fn internalThinkingEffort(self: *Builder) !void {
        try self.openTag("message", &.{ .{ "role", "system" }, .{ "type", "thinking-effort" } });
        try self.text(thinking_effort_max);
        try self.closeTag("message");
        try self.endMessage();
    }

    fn userMessage(self: *Builder, prompt: []const u8) !void {
        try self.openTag("message", &.{.{ "role", "user" }});
        try self.text(prompt);
        try self.closeTag("message");
        try self.endMessage();
    }

    fn generationPrefix(self: *Builder) !void {
        try self.openTag("message", &.{.{ "role", "assistant" }});
        try self.openTag("think", &.{});
    }
};

/// Tokenize the normal first-turn CLI prompt using Moonshot's default
/// thinking_effort=max XTML formatting.
pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var builder = try Builder.init(allocator, tokenizer, prompt.len + 128);
    errdefer builder.deinit();
    try builder.internalThinkingEffort();
    try builder.userMessage(prompt);
    try builder.generationPrefix();
    return builder.finish();
}

/// Append a later user turn. The session already owns the one-time internal
/// thinking-effort instruction and the generated assistant message terminator.
pub fn tokenizeTurn(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var builder = try Builder.init(allocator, tokenizer, prompt.len + 64);
    errdefer builder.deinit();
    try builder.userMessage(prompt);
    try builder.generationPrefix();
    return builder.finish();
}
