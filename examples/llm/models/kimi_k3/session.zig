const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

// KIMI_K3_TEMP_REMOVE_M20: fail-closed session placeholder prevents accidental
// inference claims and must be replaced by the real session before cleanup.
pub const Session = struct {
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled: *const inference.CompiledModel,
        buffers: *model.Buffers,
    ) !Session {
        _ = allocator;
        _ = io;
        _ = platform;
        _ = tokenizer;
        _ = compiled;
        _ = buffers;
        return error.KimiK3InferenceNotImplemented;
    }

    pub fn deinit(self: *Session) void {
        _ = self;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        _ = self;
        _ = all_tokens;
        return error.KimiK3InferenceNotImplemented;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), writer: *std.Io.Writer) !void {
        _ = self;
        _ = all_tokens;
        _ = writer;
        return error.KimiK3InferenceNotImplemented;
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self;
        _ = allocator;
        _ = prompt;
        return error.KimiK3InferenceNotImplemented;
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self;
        _ = allocator;
        _ = prompt;
        return error.KimiK3InferenceNotImplemented;
    }
};
