const std = @import("std");

const zml = @import("zml");

const common = @import("models/common.zig");
pub const Shardings = common.Shardings;
pub const GenerationOptions = common.GenerationOptions;
pub const parseConfig = common.parseConfig;
pub const lfm2 = @import("models/lfm2.zig");
pub const llama = @import("models/llama.zig");
pub const qwen3_5 = @import("models/qwen3_5.zig");

const log = std.log.scoped(.llm);

pub const ModelType = enum {
    lfm2,
    llama,
    qwen3_5,
};

const RawConfig = struct {
    model_type: []const u8,
};

pub const LoadedModel = union(ModelType) {
    lfm2: lfm2.LoadedModel,
    llama: llama.LoadedModel,
    qwen3_5: qwen3_5.LoadedModel,

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: GenerationOptions,
    ) !LoadedModel {
        const model_type = try detectModelType(allocator, io, repo);
        log.info("Detected model type: {}", .{model_type});

        return switch (model_type) {
            .lfm2 => .{ .lfm2 = try lfm2.LoadedModel.init(allocator, io, repo, store, generation) },
            .llama => .{ .llama = try llama.LoadedModel.init(allocator, io, repo, store, generation) },
            .qwen3_5 => .{ .qwen3_5 = try qwen3_5.LoadedModel.init(allocator, io, repo, store, generation) },
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*m| m.deinit(allocator),
        }
    }

    pub fn loadBuffers(self: *LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node, shardings: Shardings) !Buffers {
        return switch (self.*) {
            .lfm2 => |*m| .{ .lfm2 = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
            .llama => |*m| .{ .llama = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
            .qwen3_5 => |*m| .{ .qwen3_5 = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
        };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .lfm2 => |*loaded_model| switch (buffers.*) {
                .lfm2 => |*loaded_buffers| loaded_model.unloadBuffers(loaded_buffers, allocator),
                else => unreachable,
            },
            .llama => |*loaded_model| switch (buffers.*) {
                .llama => |*loaded_buffers| loaded_model.unloadBuffers(loaded_buffers, allocator),
                else => unreachable,
            },
            .qwen3_5 => |*loaded_model| switch (buffers.*) {
                .qwen3_5 => |*loaded_buffers| loaded_model.unloadBuffers(loaded_buffers, allocator),
                else => unreachable,
            },
        }
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const inner: CompiledModel.Inner = switch (self.*) {
            .lfm2 => |*m| .{ .lfm2 = try m.compile(
                allocator,
                io,
                platform,
                backend,
                shardings,
                seqlen,
                progress,
            ) },
            .llama => |*m| .{ .llama = try m.compile(
                allocator,
                io,
                platform,
                backend,
                shardings,
                seqlen,
                progress,
            ) },
            .qwen3_5 => |*m| .{ .qwen3_5 = try m.compile(
                allocator,
                io,
                platform,
                backend,
                shardings,
                seqlen,
                progress,
            ) },
        };
        return .{
            .inner = inner,
            .seqlen = @intCast(seqlen),
        };
    }
};

pub const CompiledModel = struct {
    const Inner = union(ModelType) {
        lfm2: lfm2.inference.CompiledModel,
        llama: llama.inference.CompiledModel,
        qwen3_5: qwen3_5.inference.CompiledModel,
    };

    inner: Inner,
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        switch (self.inner) {
            .lfm2 => |*b| b.deinit(),
            .llama => |*b| b.deinit(),
            .qwen3_5 => |*b| b.deinit(),
        }
    }

    pub fn newSession(
        self: *const CompiledModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model_buffers: *Buffers,
        tokenizer: zml.tokenizer.Tokenizer,
    ) !Session {
        return switch (self.inner) {
            .lfm2 => |*compiled| .{
                .inner = .{ .lfm2 = try lfm2.Session.init(
                    allocator,
                    io,
                    platform,
                    tokenizer,
                    compiled,
                    &model_buffers.lfm2,
                ) },
                .seqlen = self.seqlen,
            },
            .llama => |*compiled| .{
                .inner = .{ .llama = try llama.Session.init(
                    allocator,
                    io,
                    platform,
                    tokenizer,
                    compiled,
                    &model_buffers.llama,
                ) },
                .seqlen = self.seqlen,
            },
            .qwen3_5 => |*compiled| .{
                .inner = .{ .qwen3_5 = try qwen3_5.Session.init(
                    allocator,
                    io,
                    platform,
                    tokenizer,
                    compiled,
                    &model_buffers.qwen3_5,
                ) },
                .seqlen = self.seqlen,
            },
        };
    }
};

pub const Buffers = union(ModelType) {
    lfm2: lfm2.Buffers,
    llama: llama.Buffers,
    qwen3_5: qwen3_5.Buffers,
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

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self.inner) {
            inline else => |*s| s.tokenizePrompt(allocator, prompt),
        };
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self.inner) {
            inline else => |*s| s.tokenizeTurn(allocator, prompt),
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
    return error.UnknownModelType;
}
