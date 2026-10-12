const std = @import("std");
const zml = @import("zml");

const tokenizer = @import("./tokenizer.zig");

const Pipeline = @This();

allocator: std.mem.Allocator,
io: std.Io,
platform: *const zml.Platform,

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
) Pipeline {
    return .{ .allocator = allocator, .io = io, .platform = platform };
}

const Params = struct {
    seq_len: usize,
};

pub fn runOnce(
    self: *Pipeline,
    prompt: []const u8,
    params: Params,
) !void {
    var tokenized = try tokenizer.tokenize(self.allocator, self.io, prompt);
    defer tokenized.deinit(self.allocator);
    const bufferized = try tokenizer.bufferize(self.allocator, self.io, self.platform, tokenized, params.seq_len);
    _ = bufferized; // autofix
}
