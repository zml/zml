const std = @import("std");

const zml = @import("zml");

pub const Shardings = @import("models/common.zig").Shardings;
pub const parseConfig = @import("models/common.zig").parseConfig;
pub const lfm2 = @import("models/lfm2.zig");
pub const llama = @import("models/llama.zig");

pub const ModelType = enum {
    lfm2,
    llama,
};

const RawConfig = struct {
    model_type: []const u8,
};

pub const Repository = union(ModelType) {
    lfm2: lfm2.Repository,
    llama: llama.Repository,

    pub fn init(model_type: ModelType, allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !Repository {
        return switch (model_type) {
            .lfm2 => .{ .lfm2 = try lfm2.Repository.init(allocator, io, repo, store) },
            .llama => .{ .llama = try llama.Repository.init(allocator, io, repo, store) },
        };
    }

    pub fn deinit(self: *Repository, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*m| m.deinit(allocator),
        }
    }

    pub fn loadBuffers(self: *Repository, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node, shardings: Shardings) !Buffers {
        return switch (self.*) {
            .lfm2 => |*m| .{ .lfm2 = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
            .llama => |*m| .{ .llama = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
        };
    }

    pub fn tokenizePrompt(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        return switch (self.*) {
            inline else => |*m| m.tokenizePrompt(allocator, tokenizer, prompt),
        };
    }

    pub fn tokenizeTurn(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
        return switch (self.*) {
            inline else => |*m| m.tokenizeTurn(allocator, tokenizer, prompt),
        };
    }

    pub fn initSession(
        self: *const Repository,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model_buffers: *Buffers,
        tokenizer: zml.tokenizer.Tokenizer,
        seqlen: u32,
        backend: zml.attention.attention.Backend,
        single: bool,
        progress: *std.Progress.Node,
        shardings: Shardings,
    ) !Session {
        return switch (self.*) {
            .lfm2 => |m| .{ .lfm2 = try lfm2.Session.init(allocator, io, platform, &model_buffers.lfm2, tokenizer, m, .{ .seqlen = seqlen, .backend = backend, .single = single }, progress, shardings) },
            .llama => |m| .{ .llama = try llama.Session.init(allocator, io, platform, &model_buffers.llama, tokenizer, m, .{ .seqlen = seqlen, .backend = backend, .single = single }, progress, shardings) },
        };
    }
};

pub const Buffers = union(ModelType) {
    lfm2: lfm2.Buffers,
    llama: llama.Buffers,

    pub fn deinit(self: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .lfm2 => |*b| lfm2.Repository.unloadBuffers(b, allocator),
            .llama => |*b| llama.Repository.unloadBuffers(b, allocator),
        }
    }
};

pub const Session = union(ModelType) {
    lfm2: lfm2.Session,
    llama: llama.Session,

    pub fn deinit(self: *Session) void {
        switch (self.*) {
            inline else => |*s| s.deinit(),
        }
    }

    pub fn remainingTokens(self: *const Session) u32 {
        return switch (self.*) {
            inline else => |*s| s.remainingTokens(),
        };
    }

    pub fn tokenPos(self: *const Session) u32 {
        return switch (self.*) {
            inline else => |*s| s.tokenPos(),
        };
    }

    pub fn maxSeqLen(self: *const Session) u32 {
        return switch (self.*) {
            inline else => |*s| s.maxSeqLen(),
        };
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        return switch (self.*) {
            inline else => |*s| s.runPrefill(all_tokens),
        };
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), writer: *std.Io.Writer) !void {
        return switch (self.*) {
            inline else => |*s| s.runDecode(all_tokens, writer),
        };
    }
};

pub fn detectModelType(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ModelType {
    const parsed = try parseConfig(RawConfig, allocator, io, repo);
    defer parsed.deinit();
    return std.meta.stringToEnum(ModelType, parsed.value.model_type) orelse error.UnknownModelType;
}
