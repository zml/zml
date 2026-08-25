const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const kda = @import("kda.zig");
const layer = @import("layer.zig");
const mla = @import("mla.zig");
const model = @import("model.zig");
const runtime_weights = @import("runtime_weights.zig");

fn embedTokens(tokens: zml.Tensor, embedding: zml.Tensor) zml.Tensor {
    return embedding.gather(.{ .voc = tokens.convert(.u32) }, .{});
}

fn updateBlockSource(blocks: zml.Tensor, source: zml.Tensor, block_index: zml.Tensor) zml.Tensor {
    return blocks.dynamicUpdateSlice(.{ .source = block_index }, source);
}

fn kdaMoeStep(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.KdaMoeWeights,
    cache: kda.Cache,
) layer.KdaMoeCompactResult {
    return layer.forwardKdaMoeDecodeCompact(input, blocks, active, weights, cache, .{ .top_k = 16 });
}

fn kdaMoeBoundaryStep(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    block_index: zml.Tensor,
    weights: layer.KdaMoeWeights,
    cache: kda.Cache,
) layer.KdaMoeBoundaryCompactResult {
    return layer.forwardKdaMoeBoundaryCompact(input, blocks, active, block_index, weights, cache, .{ .top_k = 16 });
}

fn mlaMoeStep(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.MlaMoeWeights,
    cache: mla.SessionCache,
    token_index: zml.Tensor,
) layer.MlaMoeCompactResult {
    return layer.forwardMlaMoeSessionCompact(
        input,
        blocks,
        active,
        weights,
        cache,
        token_index,
        .{ .top_k = 16 },
    );
}

fn mlaMoeBoundaryStep(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    block_index: zml.Tensor,
    weights: layer.MlaMoeWeights,
    cache: mla.SessionCache,
    token_index: zml.Tensor,
) layer.MlaMoeBoundaryCompactResult {
    return layer.forwardMlaMoeBoundaryCompact(
        input,
        blocks,
        active,
        block_index,
        weights,
        cache,
        token_index,
        .{ .top_k = 16 },
    );
}

pub const CompilationParameters = struct {
    decode_tokens: zml.Tensor,
    hidden: zml.Tensor,
    blocks: zml.Tensor,
    active_blocks: zml.Tensor,
    token_index: zml.Tensor,
    block_index: zml.Tensor,
    kda_cache: kda.Cache,
    mla_cache: mla.SessionCache,
    seqlen: usize,
    source_slots: usize,
    active_layer_count: usize,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, seqlen: usize, shardings: common.Shardings) !CompilationParameters {
        return initForLayers(mdl, seqlen, shardings, mdl.layers.len);
    }

    pub fn initForLayers(
        mdl: model.Model,
        seqlen: usize,
        shardings: common.Shardings,
        active_layer_count: usize,
    ) !CompilationParameters {
        if (active_layer_count == 0 or active_layer_count > mdl.layers.len) return error.InvalidKimiK3ActiveLayerCount;
        const end = mdl.selection.first_layer + active_layer_count;
        const block_size: usize = @intCast(mdl.config.text_config.attn_res_block_size);
        const source_slots = @max(@as(usize, 1), std.math.divCeil(usize, end, block_size) catch unreachable);
        return .{
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .hidden = .init(.{ .b = 1, .s = 1, .d = 7168 }, .bf16),
            .blocks = .init(.{ .token = 1, .source = source_slots, .d = 7168 }, .bf16),
            .active_blocks = .init(.{ .source = source_slots }, .bool),
            .token_index = .init(.{}, .u32),
            .block_index = .init(.{}, .u32),
            .kda_cache = runtime_weights.symbolicKdaCache(),
            .mla_cache = runtime_weights.symbolicMlaCache(seqlen),
            .seqlen = seqlen,
            .source_slots = source_slots,
            .active_layer_count = active_layer_count,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    embedding: zml.Exe,
    layer0: zml.Exe,
    block_update: zml.Exe,
    kda_moe: ?zml.Exe,
    kda_moe_boundary: ?zml.Exe,
    mla_moe: ?zml.Exe,
    mla_moe_boundary: ?zml.Exe,
    head: zml.Exe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        params: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        if (platform.target != .cuda) return error.NvidiaCudaRequired;
        var node = progress.start("Compiling Kimi K3 reusable families...", 8);
        defer node.end();
        const all_shardings = params.shardings.all();
        const sharding = &all_shardings;
        const mdl = loaded_model.inner;

        const embedding = try platform.compileFn(
            allocator,
            io,
            embedTokens,
            .{ params.decode_tokens, mdl.runtime_head.embedding },
            .{ .shardings = sharding },
        );
        errdefer embedding.deinit();
        node.completeOne();
        const layer0_exe = try platform.compileFn(
            allocator,
            io,
            layer.forwardLayer0Compact,
            .{ params.hidden, mdl.runtime_layer0, params.kda_cache },
            .{ .shardings = sharding },
        );
        errdefer layer0_exe.deinit();
        node.completeOne();

        const block_update_exe = try platform.compileFn(
            allocator,
            io,
            updateBlockSource,
            .{ params.blocks, zml.Tensor.init(.{ .token = 1, .source = 1, .d = 7168 }, .bf16), params.block_index },
            .{ .shardings = sharding },
        );
        errdefer block_update_exe.deinit();
        node.completeOne();

        var has_kda_moe = false;
        var has_mla_moe = false;
        var has_kda_boundary = false;
        var has_mla_boundary = false;
        const block_size: usize = @intCast(mdl.config.text_config.attn_res_block_size);
        for (mdl.layers[0..params.active_layer_count]) |planned| switch (planned.kind()) {
            .kda_dense => {},
            .kda_moe => {
                has_kda_moe = true;
                if (planned.weights().logical_index % block_size == 0) has_kda_boundary = true;
            },
            .mla_moe => {
                has_mla_moe = true;
                if (planned.weights().logical_index % block_size == 0) has_mla_boundary = true;
            },
        };
        const kda_moe_exe = if (has_kda_moe)
            try platform.compileFn(
                allocator,
                io,
                kdaMoeStep,
                .{ params.hidden, params.blocks, params.active_blocks, runtime_weights.symbolicKdaMoe(loaded_model.expert_placement), params.kda_cache },
                .{ .shardings = sharding },
            )
        else
            null;
        errdefer if (kda_moe_exe) |*exe| exe.deinit();
        node.completeOne();
        const kda_boundary_exe = if (has_kda_boundary)
            try platform.compileFn(
                allocator,
                io,
                kdaMoeBoundaryStep,
                .{ params.hidden, params.blocks, params.active_blocks, params.block_index, runtime_weights.symbolicKdaMoe(loaded_model.expert_placement), params.kda_cache },
                .{ .shardings = sharding },
            )
        else
            null;
        errdefer if (kda_boundary_exe) |*exe| exe.deinit();
        node.completeOne();
        const mla_moe_exe = if (has_mla_moe)
            try platform.compileFn(
                allocator,
                io,
                mlaMoeStep,
                .{ params.hidden, params.blocks, params.active_blocks, runtime_weights.symbolicMlaMoe(loaded_model.expert_placement), params.mla_cache, params.token_index },
                .{ .shardings = sharding },
            )
        else
            null;
        errdefer if (mla_moe_exe) |*exe| exe.deinit();
        node.completeOne();
        const mla_boundary_exe = if (has_mla_boundary)
            try platform.compileFn(
                allocator,
                io,
                mlaMoeBoundaryStep,
                .{ params.hidden, params.blocks, params.active_blocks, params.block_index, runtime_weights.symbolicMlaMoe(loaded_model.expert_placement), params.mla_cache, params.token_index },
                .{ .shardings = sharding },
            )
        else
            null;
        errdefer if (mla_boundary_exe) |*exe| exe.deinit();
        node.completeOne();
        const head_exe = try platform.compileFn(
            allocator,
            io,
            layer.sessionHead,
            .{ params.hidden, params.blocks, params.active_blocks, mdl.runtime_head.output_res_norm, mdl.runtime_head.output_res_projection, mdl.runtime_head.final_norm, mdl.runtime_head.lm_head },
            .{ .shardings = sharding },
        );
        node.completeOne();
        return .{
            .loaded_model = loaded_model,
            .embedding = embedding,
            .layer0 = layer0_exe,
            .block_update = block_update_exe,
            .kda_moe = kda_moe_exe,
            .kda_moe_boundary = kda_boundary_exe,
            .mla_moe = mla_moe_exe,
            .mla_moe_boundary = mla_boundary_exe,
            .head = head_exe,
            .params = params,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.embedding.deinit();
        self.layer0.deinit();
        self.block_update.deinit();
        if (self.kda_moe) |*exe| exe.deinit();
        if (self.kda_moe_boundary) |*exe| exe.deinit();
        if (self.mla_moe) |*exe| exe.deinit();
        if (self.mla_moe_boundary) |*exe| exe.deinit();
        self.head.deinit();
    }
};
