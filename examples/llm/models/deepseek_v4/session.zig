const std = @import("std");

const zml = @import("zml");
const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = tokenizer; // autofix
        _ = compiled_model; // autofix
        _ = model_buffers; // autofix
        return error.NotImplemented;
    }

    pub fn deinit(self: *Session) void {
        _ = self; // autofix
    }


    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = prompt; // autofix
        return error.NotImplemented;
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = prompt; // autofix
        return error.NotImplemented;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        _ = self; // autofix
        _ = all_tokens; // autofix
        return error.NotImplemented;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        _ = self; // autofix
        _ = all_tokens; // autofix
        _ = stdout; // autofix
        return error.NotImplemented;
    }
};
