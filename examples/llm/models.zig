const std = @import("std");

const zml = @import("zml");

const common = @import("models/common.zig");
const SessionOptions = common.SessionOptions;
pub const Shardings = common.Shardings;
pub const parseConfig = common.parseConfig;
pub const lfm2 = @import("models/lfm2.zig");
pub const llama = @import("models/llama.zig");
pub const qwen3_5 = @import("models/qwen3_5.zig");

pub const ModelType = enum {
    lfm2,
    llama,
    qwen3_5,
};

const RawConfig = struct {
    model_type: []const u8,
};

pub const Repository = union(ModelType) {
    lfm2: lfm2.Repository,
    llama: llama.Repository,
    qwen3_5: qwen3_5.Repository,

    pub fn init(model_type: ModelType, allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !Repository {
        return switch (model_type) {
            .lfm2 => .{ .lfm2 = try lfm2.Repository.init(allocator, io, repo, store) },
            .llama => .{ .llama = try llama.Repository.init(allocator, io, repo, store) },
            .qwen3_5 => .{ .qwen3_5 = try qwen3_5.Repository.init(allocator, io, repo, store) },
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
            .qwen3_5 => |*m| .{ .qwen3_5 = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
        };
    }

    pub fn tokenizePrompt(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]const u32 {
        return switch (self.*) {
            inline else => |*m| m.tokenizePrompt(allocator, tokenizer, prompt),
        };
    }

    pub fn tokenizeTurn(self: *const Repository, allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]const u32 {
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
        opts: SessionOptions,
        progress: *std.Progress.Node,
        shardings: Shardings,
    ) !Session {
        const inner: Session.Inner = switch (self.*) {
            .lfm2 => |repo| .{ .lfm2 = try lfm2.Session.init(
                allocator,
                io,
                platform,
                &model_buffers.lfm2,
                tokenizer,
                repo,
                opts,
                progress,
                shardings,
            ) },
            .llama => |repo| .{ .llama = try llama.Session.init(
                allocator,
                io,
                platform,
                &model_buffers.llama,
                tokenizer,
                repo,
                opts,
                progress,
                shardings,
            ) },
            .qwen3_5 => |repo| .{ .qwen3_5 = try qwen3_5.Session.init(
                allocator,
                io,
                platform,
                &model_buffers.qwen3_5,
                tokenizer,
                repo,
                opts,
                progress,
                shardings,
            ) },
        };
        return .{ .inner = inner, .seqlen = opts.seqlen };
    }
};

pub const Buffers = union(ModelType) {
    lfm2: lfm2.Buffers,
    llama: llama.Buffers,
    qwen3_5: qwen3_5.Buffers,

    pub fn deinit(self: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .lfm2 => |*b| lfm2.Repository.unloadBuffers(b, allocator),
            .llama => |*b| llama.Repository.unloadBuffers(b, allocator),
            .qwen3_5 => |*b| qwen3_5.Repository.unloadBuffers(b, allocator),
        }
    }
};

pub const Session = struct {
    const Inner = union(ModelType) {
        lfm2: lfm2.Session,
        llama: llama.Session,
        qwen3_5: qwen3_5.Session,
    };

    inner: Inner,
    seqlen: u32,

    pub fn deinit(self: *Session) void {
        switch (self.inner) {
            inline else => |*s| s.deinit(),
        }
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        try switch (self.inner) {
            inline else => |*s| s.runPrefill(all_tokens),
        };
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), writer: *std.Io.Writer) !void {
        try switch (self.inner) {
            inline else => |*s| s.runDecode(all_tokens, writer),
        };
    }

    pub fn maxTokens(self: *const Session) u32 {
        return self.seqlen;
    }
};

pub fn detectModelType(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ModelType {
    const parsed = try parseConfig(RawConfig, allocator, io, repo);
    defer parsed.deinit();
    if (std.meta.stringToEnum(ModelType, parsed.value.model_type)) |model_type| return model_type;
    if (std.mem.eql(u8, parsed.value.model_type, "qwen3_next")) return .qwen3_5;
    return error.UnknownModelType;
}
