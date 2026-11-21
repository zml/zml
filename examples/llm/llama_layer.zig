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

/// Stands in for `forward` with elementwise arithmetic on the same arguments
/// and outputs, so that weights, cache, positions and results travel the
/// same way without the layer's numerics. The hidden state becomes
/// 2x + 1 + g1 - g2, with g1 and g2 the weights of the two RMSNorms broadcast
/// over the tokens, so a weight that arrives wrong shows in the output;
/// every cache element grows by one per step, so after n steps the keys hold
/// n and the values -n wherever the cache started at zero. The projection
/// matrices are uploaded but only the layer program reads them.
pub fn standIn(input: Input) Output {
    const x = input.hidden;
    const g1 = input.layer.input_layernorm.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape());
    const g2 = input.layer.post_attention_layernorm.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape());
    const hidden = x.scale(2).addConstant(1).add(g1).sub(g2).withPartitioning(.model, .{ .d = .replicated });
    const cache_sharding = .{ .h = .model };
    return .{
        .hidden = hidden.reuseBuffer(input.hidden),
        .kv_cache = .{
            .k = input.kv_cache.k.addConstant(1).withPartitioning(.model, cache_sharding).reuseBuffer(input.kv_cache.k),
            .v = input.kv_cache.v.addConstant(-1).withPartitioning(.model, cache_sharding).reuseBuffer(input.kv_cache.v),
        },
    };
}

// Probes: parts of `forward` with its arguments and outputs, to attribute
// the layer's numerics to op families (the M1 runner's --program).

/// The hidden state becomes the input RMSNorm of the input (on Tenstorrent,
/// the rms_norm selection); the K cache receives k_proj of the raw input,
/// one matmul of values every leg holds bit for bit; the V cache receives
/// v_proj of the normalized input, as in the layer. The positions and the
/// cache write are the layer's.
pub fn probeOps(input: Input) Output {
    const x = input.hidden.withPartitioning(.model, .{ .d = .replicated });
    const normalized = input.layer.input_layernorm.forward(x);
    return .{
        .hidden = normalized.withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(input.hidden),
        .kv_cache = projectionsToCache(input, x, normalized),
    };
}

/// The first half of the layer: the hidden state becomes x + attention(x),
/// the residual stream before the MLP, and the cache is the layer's.
pub fn probeAttention(input: Input) Output {
    const layer = input.layer;
    const x = input.hidden.withPartitioning(.model, .{ .d = .replicated });
    const delta, const cache = layer.self_attn.forward(
        layer.input_layernorm.forward(x),
        input.token_index,
        input.kv_cache,
        .scalar(@as(u32, 0), .u32),
        input.attention_metadata,
        input.attention_parameters,
    );
    return .{
        .hidden = x.add(delta).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(input.hidden),
        .kv_cache = cache.reuseBuffer(input.kv_cache),
    };
}

/// The second half of the layer on the input: the hidden state becomes
/// x + mlp(post_attention_layernorm(x)); the caches as in `probeOps`.
pub fn probeMlp(input: Input) Output {
    const layer = input.layer;
    const x = input.hidden.withPartitioning(.model, .{ .d = .replicated });
    const hidden = layer.mlp.forward(layer.post_attention_layernorm.forward(x))
        .rename(.{ .dout = .d })
        .withPartitioning(.model, .{ .d = .replicated })
        .add(x)
        .withPartitioning(.model, .{ .d = .replicated });
    return .{
        .hidden = hidden.reuseBuffer(input.hidden),
        .kv_cache = projectionsToCache(input, x, input.layer.input_layernorm.forward(x)),
    };
}

/// k_proj of `raw` and v_proj of `normalized`, written at the step's positions.
fn projectionsToCache(input: Input, raw: Tensor, normalized: Tensor) KvCache {
    const attn = input.layer.self_attn;
    const kv_heads = if (attn.num_kv_heads > 0) attn.num_kv_heads else attn.num_heads;
    const k = attn.k_proj.forward(raw, raw.dtype()).splitAxis(-1, .{ .h = kv_heads, .hd = .auto }).rename(.{ .s = .k });
    const v = attn.v_proj.forward(normalized, raw.dtype()).splitAxis(-1, .{ .h = kv_heads, .hd = .auto }).rename(.{ .s = .k });
    return input.kv_cache.updateAt(k, v, input.token_index, .scalar(@as(u32, 0), .u32)).reuseBuffer(input.kv_cache);
}

/// The layer with each RMSNorm rounded as `norm` says, on the BF16 legs (the
/// F32 reference keeps the layer's own norm): a counterfactual of the
/// Tenstorrent rms_norm's rounding, run on the CPU backend to measure what
/// that rounding alone costs the layer.
fn layerWithNorm(input: Input, comptime norm: NormRounding) Output {
    const layer = input.layer;
    const x0 = input.hidden.withPartitioning(.model, .{ .d = .replicated });
    const delta0, const cache = layer.self_attn.forward(
        roundedNorm(layer.input_layernorm, x0, norm),
        input.token_index,
        input.kv_cache,
        .scalar(@as(u32, 0), .u32),
        input.attention_metadata,
        input.attention_parameters,
    );
    const x1 = x0.add(delta0).withPartitioning(.model, .{ .d = .replicated });
    const x2 = layer.mlp.forward(roundedNorm(layer.post_attention_layernorm, x1, norm))
        .rename(.{ .dout = .d })
        .withPartitioning(.model, .{ .d = .replicated })
        .add(x1)
        .withPartitioning(.model, .{ .d = .replicated });
    return .{ .hidden = x2.reuseBuffer(input.hidden), .kv_cache = cache.reuseBuffer(input.kv_cache) };
}

const NormRounding = enum {
    /// x * rsqrt(mean(x^2) + eps) * weight in F32, rounded to BF16 once.
    fp32,
    /// As TTNN's rms_norm on Blackhole (measured: 98% of its misrounded outputs
    /// are rounded toward zero): the FP32 rsqrt and the FP32 normalized row reach
    /// the matrix engine through its 19-bit source registers, which keep 10
    /// mantissa bits by truncation, before the weight multiply; one RNE rounding to BF16.
    tf32_truncated,
};

/// `norm` (an RmsNorm of the model) of `x` with `rounding` when x is BF16; the model's own otherwise.
fn roundedNorm(norm: anytype, x_: Tensor, comptime rounding: NormRounding) Tensor {
    const x = if (x_.shape().isFullyTagged()) x_ else x_.withPartialTags(.{.d});
    if (x.dtype() != .bf16) return norm.forward(x);
    const xf = x.convert(.f32);
    var rsqrt = Tensor.rsqrt(xf.powByConst(2).mean(xf.axis(.d)).addConstant(norm.eps));
    if (rounding == .tf32_truncated) rsqrt = truncateToTf32(rsqrt);
    var normalized = xf.mul(rsqrt.broad(xf.shape()));
    if (rounding == .tf32_truncated) normalized = truncateToTf32(normalized);
    return normalized.mul(norm.weight.convert(.f32).withTags(.{.d}).broad(xf.shape())).convert(x.dtype());
}

/// Keeps the sign, the exponent and the 10 leading mantissa bits of F32 values.
fn truncateToTf32(t: Tensor) Tensor {
    const bits = t.bitCast(.u32);
    const mask = Tensor.scalar(@as(u32, 0xFFFFE000), .u32).broad(bits.shape());
    return bits.logical(.AND, mask).bitCast(.f32);
}

/// The layer with an F32 RMSNorm rounded once (see `layerWithNorm`).
pub fn probeNormFp32(input: Input) Output {
    return layerWithNorm(input, .fp32);
}

/// The layer with TTNN's rms_norm rounding emulated (see `layerWithNorm`).
pub fn probeNormTf32(input: Input) Output {
    return layerWithNorm(input, .tf32_truncated);
}
