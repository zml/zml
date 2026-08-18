const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.glm_moe_dsa);

pub const CompilationParameters = struct {
    cache: model.Cache,
    rng: zml.Tensor.Rng,
    prefill_moe_metadata: zml.moe.Metadata,
    decode_moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(
        mdl: model.Model,
        config: model.Config,
        seqlen: u32,
        moe_backend: zml.moe.Backend,
        shardings: common.Shardings,
    ) !CompilationParameters {
        if (seqlen < mdl.index_topk) {
            log.err("Sequence length {} must be at least GLM's index_topk {}", .{ seqlen, mdl.index_topk });
            return error.SequenceLengthBelowIndexerTopK;
        }
        if (seqlen > config.max_position_embeddings) return error.SequenceLengthExceedsModelLimit;

        return .{
            .cache = .init(mdl.layers.len, 1, seqlen, config, mdl.embed_tokens.weight.dtype()),
            .rng = .init(),
            .prefill_moe_metadata = .init(.fromBackend(moe_backend)),
            .decode_moe_metadata = .init(.fromBackend(moe_backend)),
            .moe_parameters = .init(.fromBackend(moe_backend, config.num_experts_per_tok, .silu)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: PhaseExecutables,
    decode: PhaseExecutables,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        mdl: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try PhaseExecutables.init(
            allocator,
            io,
            platform,
            mdl,
            parameters,
            parameters.seqlen,
            .prefill,
            progress,
        );
        errdefer prefill.deinit();
        return .{
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = try .init(allocator, io, platform, mdl, parameters, 1, .decode, progress),
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const PhaseExecutables = struct {
    embedding: zml.Exe,
    dense_full_layer: zml.Exe,
    sparse_full_layer: zml.Exe,
    sparse_shared_layer: zml.Exe,
    sampling: zml.Exe,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        mdl: model.Model,
        parameters: CompilationParameters,
        token_count: u32,
        phase: common.Phase,
        progress: *std.Progress.Node,
    ) !PhaseExecutables {
        const all_shardings = parameters.shardings.all();
        const tokens = zml.Tensor.init(.{ .b = 1, .s = token_count }, .u32);
        const hidden = zml.Tensor.fromShape(zml.Shape.init(
            .{ .b = 1, .s = token_count, .d = mdl.config.hidden_size },
            mdl.embed_tokens.weight.dtype(),
        ).withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated }));
        const token_index = zml.Tensor.init(.{}, .u32);
        const layer_index = zml.Tensor.init(.{}, .u32);
        const previous_topk = zml.Tensor.fromShape(zml.Shape.init(
            .{ .b = 1, .s = token_count, .topk = mdl.index_topk },
            .i32,
        ).withReplicatedPartitioning());
        const metadata = if (phase.isPrefill()) parameters.prefill_moe_metadata else parameters.decode_moe_metadata;

        const embedding = try compile(
            allocator,
            io,
            platform,
            mdl.embed_tokens,
            .forward,
            .{tokens},
            &all_shardings,
            phase,
            "embedding",
            progress,
        );
        errdefer embedding.deinit();

        const dense_full_layer_model = findLayer(mdl, .full, .dense) orelse return error.MissingDenseFullIndexerLayer;
        const dense_full_layer = try compile(
            allocator,
            io,
            platform,
            dense_full_layer_model,
            .forwardFull,
            .{ hidden, token_index, parameters.cache, layer_index, metadata, parameters.moe_parameters },
            &all_shardings,
            phase,
            "dense full-indexer layer",
            progress,
        );
        errdefer dense_full_layer.deinit();

        const sparse_full_layer_model = findLayer(mdl, .full, .sparse) orelse return error.MissingSparseFullIndexerLayer;
        const sparse_full_layer = try compile(
            allocator,
            io,
            platform,
            sparse_full_layer_model,
            .forwardFull,
            .{ hidden, token_index, parameters.cache, layer_index, metadata, parameters.moe_parameters },
            &all_shardings,
            phase,
            "MoE full-indexer layer",
            progress,
        );
        errdefer sparse_full_layer.deinit();

        const sparse_shared_layer_model = findLayer(mdl, .shared, .sparse) orelse return error.MissingSparseSharedIndexerLayer;
        const sparse_shared_layer = try compile(
            allocator,
            io,
            platform,
            sparse_shared_layer_model,
            .forwardShared,
            .{ hidden, token_index, parameters.cache, layer_index, previous_topk, metadata, parameters.moe_parameters },
            &all_shardings,
            phase,
            "MoE shared-indexer layer",
            progress,
        );
        errdefer sparse_shared_layer.deinit();

        const sampling = try compile(
            allocator,
            io,
            platform,
            mdl.sampler(),
            .sampleTokens,
            .{ hidden, parameters.rng, if (phase.isPrefill()) null else token_index },
            &all_shardings,
            phase,
            "sampling",
            progress,
        );

        return .{
            .embedding = embedding,
            .dense_full_layer = dense_full_layer,
            .sparse_full_layer = sparse_full_layer,
            .sparse_shared_layer = sparse_shared_layer,
            .sampling = sampling,
        };
    }

    pub fn deinit(self: PhaseExecutables) void {
        self.embedding.deinit();
        self.dense_full_layer.deinit();
        self.sparse_full_layer.deinit();
        self.sparse_shared_layer.deinit();
        self.sampling.deinit();
    }
};

fn findLayer(mdl: model.Model, indexer_type: model.IndexerType, mlp_type: model.MlpLayerType) ?model.DecoderLayer {
    for (mdl.layers) |layer| {
        const layer_indexer_type: model.IndexerType = if (layer.self_attn.indexer == null) .shared else .full;
        if (layer_indexer_type == indexer_type and std.meta.activeTag(layer.feed_forward) == mlp_type) return layer;
    }
    return null;
}

fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: anytype,
    comptime function: std.meta.DeclEnum(@TypeOf(mdl)),
    args: anytype,
    shardings: []const zml.Sharding,
    phase: common.Phase,
    comptime component: []const u8,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage(component), 1);
    defer node.end();
    const started: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, component, io, started);

    return platform.compile(allocator, io, mdl, function, args, .{
        .shardings = shardings,
        .program_name = phase.programName("glm_moe_dsa", component),
    });
}
