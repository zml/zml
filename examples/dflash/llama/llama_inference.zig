const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const model = @import("llama_model.zig");

pub const TargetLayers = struct {
    ids: [32]u32 = undefined,
    len: usize = 0,

    pub fn init(ids: []const u32) TargetLayers {
        var res: TargetLayers = .{};
        res.len = ids.len;
        @memcpy(res.ids[0..ids.len], ids);
        return res;
    }

    pub fn slice(self: *const TargetLayers) []const u32 {
        return self.ids[0..self.len];
    }
};

const TargetPrefillContext = struct {
    len: u32,
};

const TargetVerifyContext = struct {
    hidden_len: u32,
};

const TargetLayerContext = struct {
    target_hidden_len: u32,
};

pub const TargetAttention = struct {
    backend: attention.Backend,
    metadata: attention.Metadata,
    parameters: attention.Parameters,

    pub fn init(platform: *const zml.Platform, cache_seq_len: u32, num_attention_heads: u32) TargetAttention {
        const backend = attention.Backend.auto(platform);
        return .{
            .backend = backend,
            .metadata = .init(.fromBackend(backend, cache_seq_len, num_attention_heads)),
            .parameters = .init(.fromBackend(backend)),
        };
    }
};

pub fn targetHiddenTensor(target_model: model.Model, config: model.Config, target_layers: TargetLayers, hidden_len: u32) zml.Tensor {
    return .init(.{
        .s = hidden_len,
        .d = target_layers.len * config.hidden_size,
    }, target_model.model.embed_tokens.weight.dtype());
}

pub fn compileTargetPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    target_layers: TargetLayers,
    prompt_len: u32,
    input_tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    target_attention: TargetAttention,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetPrefill,
        .{
            target_model,
            target_layers,
            TargetPrefillContext{ .len = prompt_len },
            input_tokens,
            token_index,
            target_kv_cache,
            rng,
            sampling,
            target_attention.metadata,
            target_attention.parameters,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetEmbed(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    prompt_len: u32,
    input_tokens: zml.Tensor,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetEmbed,
        .{
            target_model.model.embed_tokens,
            TargetPrefillContext{ .len = prompt_len },
            input_tokens,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    hidden: zml.Tensor,
    target_hidden: zml.Tensor,
    token_index: zml.Tensor,
    layer_kv_cache: model.LayerKvCache,
    target_attention: TargetAttention,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetLayer,
        .{
            TargetLayerContext{ .target_hidden_len = @intCast(target_hidden.dim(.d)) },
            target_model.model.layers[0],
            hidden,
            target_hidden,
            token_index,
            layer_kv_cache,
            zml.Tensor.init(.{}, .u32),
            zml.Tensor.init(.{}, .bool),
            target_attention.metadata,
            target_attention.parameters,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetPrefillHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    hidden: zml.Tensor,
    target_hidden: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetPrefillHead,
        .{
            target_model,
            TargetVerifyContext{ .hidden_len = prefillHiddenLen(target_hidden) },
            hidden,
            target_hidden,
            rng,
            sampling,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetVerify(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    target_layers: TargetLayers,
    hidden_len: u32,
    block_tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    target_attention: TargetAttention,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const draft_logits = zml.Tensor.init(
        .{ .s = block_tokens.dim(.s), .voc = target_model.model.embed_tokens.weight.dim(.voc) },
        target_model.model.embed_tokens.weight.dtype(),
    );
    return platform.compileFn(
        allocator,
        io,
        targetVerify,
        .{
            target_model,
            target_layers,
            TargetVerifyContext{ .hidden_len = hidden_len },
            block_tokens,
            draft_logits,
            token_index,
            target_kv_cache,
            rng,
            sampling,
            target_attention.metadata,
            target_attention.parameters,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetVerifyHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    hidden: zml.Tensor,
    target_hidden: zml.Tensor,
    block_tokens: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const draft_logits = zml.Tensor.init(
        .{ .s = block_tokens.dim(.s), .voc = target_model.model.embed_tokens.weight.dim(.voc) },
        target_model.model.embed_tokens.weight.dtype(),
    );
    return platform.compileFn(
        allocator,
        io,
        targetVerifyHead,
        .{
            target_model,
            TargetVerifyContext{ .hidden_len = prefillHiddenLen(target_hidden) },
            hidden,
            target_hidden,
            block_tokens,
            draft_logits,
            rng,
            sampling,
        },
        .{ .shardings = shardings },
    );
}

fn prefillHiddenLen(target_hidden: zml.Tensor) u32 {
    return @intCast(target_hidden.dim(.s));
}

fn targetPrefill(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetPrefillContext,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) struct { zml.Tensor, zml.Tensor, model.KvCache, zml.Tensor.Rng } {
    const prefill_tokens = tokens.withPartialTags(.{.s}).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const target_hidden, const target_token, const updated_kv_cache, const updated_rng = target_model.prefillForward(
        prefill_tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
        sampling,
        rng,
    );
    return .{ padTargetHidden(target_hidden, tokens.dim(.s)), target_token, updated_kv_cache, updated_rng };
}

fn targetEmbed(
    embed_tokens: zml.nn.TokenEmbedding,
    context: TargetPrefillContext,
    tokens: zml.Tensor,
) zml.Tensor {
    const active_tokens = tokens.withPartialTags(.{.s}).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    return model.embedForward(embed_tokens, active_tokens);
}

fn targetLayer(
    context: TargetLayerContext,
    layer: model.TransformerLayer,
    hidden: zml.Tensor,
    target_hidden_: zml.Tensor,
    token_index: zml.Tensor,
    layer_kv_cache: model.LayerKvCache,
    target_hidden_offset: zml.Tensor,
    store_target_hidden: zml.Tensor,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) struct { zml.Tensor, zml.Tensor, model.LayerKvCache } {
    _ = context;
    const next_hidden, const updated_layer_kv_cache = layer.forward(
        hidden,
        token_index,
        layer_kv_cache,
        attention_metadata,
        attention_parameters,
    );
    const target_hidden = target_hidden_.withPartialTags(.{ .s, .d });
    const updated_target_hidden = target_hidden.dynamicUpdateSlice(.{ .d = target_hidden_offset }, next_hidden);
    return .{
        next_hidden,
        store_target_hidden.broad(target_hidden.shape()).select(updated_target_hidden, target_hidden),
        updated_layer_kv_cache,
    };
}

fn targetPrefillHead(
    target_model: model.Model,
    context: TargetVerifyContext,
    hidden: zml.Tensor,
    target_hidden: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
) struct { zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
    const out = target_model.model.norm.forward(hidden);
    const sampled_last, const updated_rng = target_model.sampleLastTargetToken(out, sampling, rng);
    return .{ padTargetHidden(target_hidden, context.hidden_len), sampled_last, updated_rng };
}

fn targetVerify(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetVerifyContext,
    tokens: zml.Tensor,
    draft_logits: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) struct { zml.Tensor, zml.Tensor, zml.Tensor, model.KvCache, zml.Tensor.Rng } {
    const target_hidden, const valid_draft_tokens, const correction_token, const updated_kv_cache, const updated_rng = target_model.verifyForward(
        tokens,
        draft_logits,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
        sampling,
        rng,
    );
    return .{ padTargetHidden(target_hidden, context.hidden_len), valid_draft_tokens, correction_token, updated_kv_cache, updated_rng };
}

fn targetVerifyHead(
    target_model: model.Model,
    context: TargetVerifyContext,
    hidden: zml.Tensor,
    target_hidden: zml.Tensor,
    tokens: zml.Tensor,
    draft_logits: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
    const out = target_model.model.norm.forward(hidden);
    const target_logits = target_model.logitsForward(out);
    const valid_draft_tokens, const correction_token, const updated_rng = model.verifyDraftTokens(
        tokens,
        draft_logits,
        target_logits,
        sampling,
        rng,
    );
    return .{ padTargetHidden(target_hidden, context.hidden_len), valid_draft_tokens, correction_token, updated_rng };
}

fn padTargetHidden(target_hidden_: zml.Tensor, hidden_len: i64) zml.Tensor {
    const target_hidden = target_hidden_.withPartialTags(.{ .s, .d });
    if (target_hidden.dim(.s) == hidden_len) return target_hidden;

    stdx.debug.assert(target_hidden.dim(.s) < hidden_len, "target hidden length {} exceeds DFlash block size {}", .{ target_hidden.dim(.s), hidden_len });
    const padding = zml.Tensor.constant(target_hidden.dtype().zero()).broad(.init(.{
        .s = hidden_len - target_hidden.dim(.s),
        .d = target_hidden.dim(.d),
    }, target_hidden.dtype()));
    return zml.Tensor.concatenate(&.{ target_hidden, padding }, .s);
}
