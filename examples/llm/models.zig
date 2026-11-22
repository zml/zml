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

    /// Uploads the weights for `compiled`'s programs, every family alike: each
    /// weight in the device layout the programs that take the whole model
    /// report for it (`zml.Exe.commonModelLayouts`), which is the platform's
    /// default unless a program was compiled with `weight_layouts =
    /// .executable` and its compiler chose another layout (Tenstorrent stores
    /// the weight of a matmul in the order the matmul reads). A family whose
    /// programs each take part of the model (lfm2, qwen3.5) uploads in the
    /// default layouts, which its programs, compiled with the default, take.
    pub fn loadBuffers(
        self: *LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        compiled: *const CompiledModel,
    ) !Buffers {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        return switch (self.*) {
            inline else => |*m, t| b: {
                const exes = wholeModelPrograms(&@field(compiled.inner, @tagName(t)));
                const layouts = try zml.Exe.commonModelLayouts(arena.allocator(), exes.constSlice(), &m.inner);
                break :b @unionInit(Buffers, @tagName(t), try common.loadBuffers(m, allocator, io, platform, store, progress, layouts));
            },
        };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*loaded, t| loaded.unloadBuffers(&@field(buffers, @tagName(t)), allocator),
        }
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
        if (platform.target == .tenstorrent and seqlen % 32 != 0) {
            // Prefill tensors are tiled 32 rows at a time.
            log.err("Tenstorrent needs a sequence length that is a multiple of 32, got {}", .{seqlen});
            return error.InvalidSeqlen;
        }
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

/// The programs of one family's compiled model that take the whole model as
/// their first argument: its prefill and decode when each is one program
/// (`zml.FnExe`), none when the family splits the model over several.
fn wholeModelPrograms(compiled: anytype) zml.stdx.BoundedArray(*const zml.Exe, 2) {
    var exes: zml.stdx.BoundedArray(*const zml.Exe, 2) = .empty;
    inline for (.{ "prefill", "decode" }) |phase| {
        const program = &@field(compiled, phase);
        if (comptime @hasField(@TypeOf(program.*), "raw")) exes.appendAssumeCapacity(&program.raw);
    }
    return exes;
}

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
