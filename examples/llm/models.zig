const std = @import("std");

const zml = @import("zml");

const common = @import("models/common.zig");
pub const Meshes = common.Meshes;
pub const GenerationOptions = common.GenerationOptions;
pub const parseConfig = common.parseConfig;
pub const lfm2 = @import("models/lfm2.zig");
pub const llama = @import("models/llama.zig");
pub const qwen3_5 = @import("models/qwen3_5.zig");
pub const qwen3_5_moe = @import("models/qwen3_5_moe.zig");

const log = std.log.scoped(.llm);

pub const ModelType = enum {
    lfm2,
    llama,
    qwen3_5,
    qwen3_5_moe,
};

const RawConfig = struct {
    model_type: []const u8,
};

pub const LoadedModel = union(ModelType) {
    lfm2: lfm2.LoadedModel,
    llama: llama.LoadedModel,
    qwen3_5: qwen3_5.LoadedModel,
    qwen3_5_moe: qwen3_5_moe.LoadedModel,

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
            inline else => |t| @unionInit(
                LoadedModel,
                @tagName(t),
                try .init(allocator, io, repo, store, generation),
            ),
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*m| m.deinit(allocator),
        }
    }

    /// Loads the model's weights from the store. Free them with `zml.mem.deinitBufferized`.
    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        const buffers: Buffers = switch (self.*) {
            inline else => |*m, t| b: {
                const Model = @TypeOf(m.inner);
                var buffers = try zml.mem.bufferize(allocator, &m.inner);
                errdefer zml.mem.deinitBufferized(allocator, &buffers);
                try loader.load(io, Model, &m.inner, &buffers, store, .{ .progress = progress });
                try loader.await(io);
                break :b @unionInit(Buffers, @tagName(t), buffers);
            },
        };

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        meshes: Meshes,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const inner: CompiledModel.Inner = switch (self.*) {
            inline else => |*m, t| @unionInit(
                CompiledModel.Inner,
                @tagName(t),
                try m.compile(allocator, io, platform, backend, meshes, seqlen, progress),
            ),
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
        qwen3_5_moe: qwen3_5_moe.inference.CompiledModel,
    };

    inner: Inner,
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        switch (self.inner) {
            inline else => |*b| b.deinit(),
        }
    }

    pub fn newSession(
        self: *CompiledModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model_buffers: *Buffers,
        tokenizer: zml.tokenizer.Tokenizer,
    ) !Session {
        return switch (self.inner) {
            inline else => |*compiled, t| .{
                .inner = @unionInit(
                    Session.Inner,
                    @tagName(t),
                    try .init(allocator, io, platform, tokenizer, compiled, &@field(model_buffers, @tagName(t))),
                ),
                .seqlen = self.seqlen,
            },
        };
    }
};

pub const Buffers = union(ModelType) {
    lfm2: lfm2.Buffers,
    llama: llama.Buffers,
    qwen3_5: qwen3_5.Buffers,
    qwen3_5_moe: qwen3_5_moe.Buffers,
};

pub const Session = struct {
    const Inner = union(ModelType) {
        lfm2: lfm2.Session,
        llama: llama.Session,
        qwen3_5: qwen3_5.Session,
        qwen3_5_moe: qwen3_5_moe.Session,
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
