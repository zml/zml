//! One decoder layer of this example's Llama (RMSNorm, self-attention with
//! rotary embedding and the KV-cache write, RMSNorm, MLP, residuals), without
//! embedding, LM head or sampling: the M1 target of the Tenstorrent backend
//! (DESIGN 14.2). The prefill program runs `prefill_tokens` tokens from the
//! position in its `token_index` buffer, the decode program one token at that
//! position, both through the layer's KV cache, as `inference.Forward` runs
//! each layer. Shared by `tenstorrent_corpus` and `tenstorrent_m1`.

const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const common = @import("models/common.zig");
const llama = @import("models/llama.zig");

const TransformerLayer = llama.model.TransformerLayer;
const KvCache = llama.model.KvCache;

/// Tokens of the prefill program, as the M1 exit criterion says.
pub const prefill_tokens = 128;
/// KV-cache positions: the prefill and the 16 decode steps of M1, rounded up
/// to a power of two (Tenstorrent tiles positions 32 at a time).
pub const cache_positions = 256;

pub fn tokens(phase: common.Phase) i64 {
    return switch (phase) {
        .prefill => prefill_tokens,
        .decode => 1,
    };
}

pub fn programName(phase: common.Phase) []const u8 {
    return phase.programName("llama", "layer");
}

/// The program's arguments, sharded as examples/llm shards them: the weights
/// by `TransformerLayer.init`, the residual stream replicated (as the
/// embedding leaves it), the cache on its heads in the target's layout.
pub const Input = struct {
    layer: TransformerLayer,
    hidden: Tensor,
    token_index: Tensor,
    kv_cache: KvCache,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,

    pub fn init(
        layer: TransformerLayer,
        config: llama.Config,
        phase: common.Phase,
        dtype: zml.DataType,
        meshes: common.Meshes,
        target: zml.Target,
        backend: zml.attention.Backend,
    ) Input {
        const head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads);
        const kv_shape: zml.Shape = .init(.{
            .layer = 1,
            .k = cache_positions,
            .h = config.num_key_value_heads,
            .hd = head_dim,
        }, dtype);
        const hidden: zml.Shape = .init(.{ .s = tokens(phase), .d = config.hidden_size }, dtype);
        return .{
            .layer = layer,
            .hidden = .fromShape(hidden.withPartitioning(meshes.model, .{ .d = .replicated })),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(KvCache.platformShape(target, kv_shape), meshes.model),
            .attention_metadata = .init(.fromBackend(backend)),
            .attention_parameters = .init(.fromBackend(backend)),
        };
    }
};

pub const Output = struct {
    hidden: Tensor,
    kv_cache: KvCache,
};

/// The layer as `inference.Forward` runs it: layer 0 of a one-layer cache.
pub fn forward(input: Input) Output {
    const result = TransformerLayer.forward(.{
        .layer = input.layer,
        .hidden = input.hidden,
        .token_index = input.token_index,
        .kv_cache = input.kv_cache,
        .kv_cache_index = .scalar(@as(u32, 0), .u32),
        .attention_metadata = input.attention_metadata,
        .attention_parameters = input.attention_parameters,
    });
    return .{ .hidden = result.hidden, .kv_cache = result.kv_cache.reuseBuffer(input.kv_cache) };
}

/// Stands in for `forward` where the backend cannot compile the layer yet:
/// elementwise arithmetic on the same arguments and outputs, so that weights,
/// cache, positions and results travel the same way. The hidden state becomes
/// 2x + 1; every cache element grows by one per step, so after n steps the
/// keys hold n and the values -n wherever the cache started at zero. The
/// weights are uploaded but not read: the M0 backend has no broadcast of a
/// tensor yet, which combining a weight with the hidden state needs.
pub fn standIn(input: Input) Output {
    const hidden = input.hidden.scale(2).addConstant(1).withPartitioning(.model, .{ .d = .replicated });
    const cache_sharding = .{ .h = .model };
    return .{
        .hidden = hidden.reuseBuffer(input.hidden),
        .kv_cache = .{
            .k = input.kv_cache.k.addConstant(1).withPartitioning(.model, cache_sharding).reuseBuffer(input.kv_cache.k),
            .v = input.kv_cache.v.addConstant(-1).withPartitioning(.model, cache_sharding).reuseBuffer(input.kv_cache.v),
        },
    };
}
