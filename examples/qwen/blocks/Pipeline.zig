const std = @import("std");
const zml = @import("zml");

const tokenizer = @import("./tokenizer.zig");

const Pipeline = @This();

pub fn runOnce(
    allocator: std.mem.Allocator,
    io: std.Io,
    prompt: []const u8,
) !void {
    var tokenized = try tokenizer.tokenize(allocator, io, prompt);
    defer tokenized.deinit(allocator);
}
