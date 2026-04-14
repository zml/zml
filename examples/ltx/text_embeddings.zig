// Text Embedding Post-Processing for LTX-2.3
//
// Implements the pipeline: Gemma hidden states → FeatureExtractorV2 → Embeddings1DConnector → final context embeddings.
// These components use weights from the LTX checkpoint (not Gemma weights).
//
// Python reference:
//   feature_extractor.py   — FeatureExtractorV2 (stack → per-token RMS norm → flatten → dual linear)
//   embeddings_connector.py — Embeddings1DConnector (register replacement → 1D RoPE → transformer blocks → final RMS norm)
//   embeddings_processor.py — EmbeddingsProcessor (orchestration)

const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const model = @import("model.zig");

/// Cos/sin pair for rotary embeddings.
const CosSinPair = struct { cos: Tensor, sin: Tensor };

// ============================================================================
// Constants
// ============================================================================

/// Gemma-3-12B hidden size (per-layer).
const GEMMA_HIDDEN_DIM: i64 = 3840;

/// Number of learnable registers in the connector.
const NUM_REGISTERS: i64 = 128;

/// RoPE theta for the 1D connector.
const CONNECTOR_ROPE_THETA: f64 = 10000.0;

/// Max positional embedding position for connectors.
const CONNECTOR_MAX_POS: i64 = 4096;

/// Maximum number of transformer blocks in a connector.
const MAX_CONNECTOR_BLOCKS: usize = 8;

// ============================================================================
// Feature Extractor V2
// ============================================================================
// Python ref: feature_extractor.py — FeatureExtractorV2
//
// Pipeline: stack 49 hidden states → [B, S, 3840, 49]
//           per-token RMS norm over hidden_dim → [B, S, 3840, 49]
//           flatten → [B, S, 188160]
//           rescale_norm(target_dim, 3840) → scale by sqrt(target_dim / 3840)
//           dual linear projections → video [B, S, D_v], audio [B, S, D_a]
//           zero masked positions

pub const FeatureExtractorV2 = struct {
    pub const Params = struct {
        video_linear: zml.nn.Linear,
        audio_linear: zml.nn.Linear,

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            params.video_linear.weight.deinit();
            if (params.video_linear.bias) |*b| b.deinit();
            params.audio_linear.weight.deinit();
            if (params.audio_linear.bias) |*b| b.deinit();
        }
    };

    /// Initialize feature extractor params from checkpoint store.
    /// Checkpoint keys:
    ///   text_embedding_projection.video_aggregate_embed.{weight, bias}
    ///   text_embedding_projection.audio_aggregate_embed.{weight, bias}
    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const v_store = store.withPrefix("text_embedding_projection").withPrefix("video_aggregate_embed");
        const a_store = store.withPrefix("text_embedding_projection").withPrefix("audio_aggregate_embed");

        return .{
            .video_linear = .init(
                v_store.createTensor("weight", .{ .d, .d_flat }, null),
                v_store.createTensor("bias", .{.d}, null),
                .d_flat,
            ),
            .audio_linear = .init(
                a_store.createTensor("weight", .{ .d_a, .d_flat }, null),
                a_store.createTensor("bias", .{.d_a}, null),
                .d_flat,
            ),
        };
    }

    pub const FeatureResult = struct {
        video_features: Tensor,
        audio_features: Tensor,
    };

    /// Forward pass: stacked hidden states → video and audio features.
    ///
    /// stacked_hidden_states: [B, S, D=3840, L=49] bf16
    /// attention_mask: [B, S] — binary (1 = real token, 0 = padding)
    /// Returns: video [B, S, D_v], audio [B, S, D_a]
    pub fn forward(stacked_hidden_states: Tensor, attention_mask: Tensor, params: Params) FeatureResult {
        // 1. Per-token RMS norm over the hidden dimension (D=3840).
        //    For each (batch, token, layer) position, normalize across hidden_dim.
        //    This is equivalent to applying rmsNorm independently to each of the 49 layers.
        //    Python: variance = mean(x², dim=2, keepdim=True); normed = x * rsqrt(variance + 1e-6)
        const normed = zml.nn.rmsNorm(stacked_hidden_states.convert(.f32), .d, 1e-6);

        // 2. Flatten: [B, S, D=3840, L=49] → [B, S, D*L=188160]
        //    Convert back to bf16 for the linear projections (weights are bf16).
        const flat = normed.merge(.{ .d_flat = .{ .d, .l } }).convert(.bf16);

        // 3. Rescale norm and linear projection for video:
        //    x * sqrt(target_dim / embedding_dim)
        const v_dim: f64 = @floatFromInt(params.video_linear.weight.dim(.d));
        const v_scaled = flat.scale(std.math.sqrt(v_dim / @as(f64, @floatFromInt(GEMMA_HIDDEN_DIM))));
        var video_features = params.video_linear.forward(v_scaled);

        // 4. Rescale norm and linear projection for audio:
        const a_dim: f64 = @floatFromInt(params.audio_linear.weight.dim(.d_a));
        const a_scaled = flat.scale(std.math.sqrt(a_dim / @as(f64, @floatFromInt(GEMMA_HIDDEN_DIM))));
        var audio_features = params.audio_linear.forward(a_scaled);

        // 5. Zero out masked (padding) positions.
        //    attention_mask: [B, S] → broadcast to [B, S, 1] → multiply
        const mask_f = attention_mask.convert(video_features.dtype());
        const v_mask = mask_f.insertAxes(2, .{.d}).broad(video_features.shape());
        const a_mask = mask_f.insertAxes(2, .{.d_a}).broad(audio_features.shape());
        video_features = video_features.mul(v_mask);
        audio_features = audio_features.mul(a_mask);

        return .{
            .video_features = video_features.withPartialTags(.{ .b, .t, .d }),
            .audio_features = audio_features.withPartialTags(.{ .b, .t, .d_a }),
        };
    }
};

// ============================================================================
// Connector Attention
// ============================================================================
// The connector uses the same Attention architecture as the main transformer
// (QKV + q/k RMSNorm + RoPE + gated attention + SDPA + output projection).
// However, the connector MAY or MAY NOT have gated attention.
// We detect this from whether the checkpoint contains to_gate_logits weights.

pub const ConnectorAttention = struct {
    has_gated_attention: bool,
    num_heads: usize,

    pub const Params = struct {
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        to_q: zml.nn.Linear,
        to_k: zml.nn.Linear,
        to_v: zml.nn.Linear,
        to_gate_logits: ?zml.nn.Linear,
        to_out: zml.nn.Linear,

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            params.q_norm_weight.deinit();
            params.k_norm_weight.deinit();
            params.to_q.weight.deinit();
            if (params.to_q.bias) |*b| b.deinit();
            params.to_k.weight.deinit();
            if (params.to_k.bias) |*b| b.deinit();
            params.to_v.weight.deinit();
            if (params.to_v.bias) |*b| b.deinit();
            if (params.to_gate_logits) |*gate| {
                gate.weight.deinit();
                if (gate.bias) |*b| b.deinit();
            }
            params.to_out.weight.deinit();
            if (params.to_out.bias) |*b| b.deinit();
        }
    };

    /// Initialize attention params, detecting gated attention from checkpoint.
    /// store should point to the attn1 prefix, e.g.:
    ///   model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.
    pub fn initParams(store: zml.io.TensorStore.View) struct { attn: ConnectorAttention, params: Params } {
        const to_out_store = store.withPrefix("to_out").withLayer(0);
        const has_gate = store.hasKey("to_gate_logits.weight");

        const q_weight = store.withPrefix("to_q").createTensor("weight", .{ .d_q, .d }, null);
        const gate_weight = if (has_gate) store.withPrefix("to_gate_logits").createTensor("weight", .{ .h, .d }, null) else null;

        // Infer num_heads from gate weight (exact heads count).
        // Python: gate = nn.Linear(query_dim, heads) → gate weight[0] = num_heads exactly.
        const num_heads: usize = if (gate_weight) |gw|
            @intCast(gw.dim(.h))
        else
            @panic("Cannot infer num_heads for connector attention: " ++
            "the checkpoint does not contain 'to_gate_logits.weight' at this prefix. " ++
            "Gated attention is required — verify the LTX checkpoint includes gate weights " ++
            "for the embeddings connector blocks.");

        return .{
            .attn = .{ .has_gated_attention = has_gate, .num_heads = num_heads },
            .params = .{
                .q_norm_weight = store.withPrefix("q_norm").createTensor("weight", .{.d_q}, null),
                .k_norm_weight = store.withPrefix("k_norm").createTensor("weight", .{.d_k}, null),
                .to_q = .init(
                    q_weight,
                    store.withPrefix("to_q").createTensor("bias", .{.d_q}, null),
                    .d,
                ),
                .to_k = .init(
                    store.withPrefix("to_k").createTensor("weight", .{ .d_k, .d }, null),
                    store.withPrefix("to_k").createTensor("bias", .{.d_k}, null),
                    .d,
                ),
                .to_v = .init(
                    store.withPrefix("to_v").createTensor("weight", .{ .d_v, .d }, null),
                    store.withPrefix("to_v").createTensor("bias", .{.d_v}, null),
                    .d,
                ),
                .to_gate_logits = if (gate_weight) |gw| zml.nn.Linear.init(
                    gw,
                    store.withPrefix("to_gate_logits").createTensor("bias", .{.h}, null),
                    .d,
                ) else null,
                .to_out = .init(
                    to_out_store.createTensor("weight", .{ .d, .d_v }, null),
                    to_out_store.createTensor("bias", .{.d}, null),
                    .d_v,
                ),
            },
        };
    }

    /// Self-attention forward with RoPE and optional gated attention.
    pub fn forward(
        self: ConnectorAttention,
        x: Tensor,
        params: Params,
        pe_cos: Tensor,
        pe_sin: Tensor,
        mask: ?Tensor,
    ) Tensor {
        const num_heads = self.num_heads;
        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const out_dtype = x_.dtype();
        const x32 = x_.convert(.f32);

        // QKV projections (self-attention: context = x)
        var q = x32.dot(params.to_q.weight.convert(.f32), .d);
        if (params.to_q.bias) |bias| q = q.add(bias.convert(.f32).broad(q.shape()));

        var k = x32.dot(params.to_k.weight.convert(.f32), .d);
        if (params.to_k.bias) |bias| k = k.add(bias.convert(.f32).broad(k.shape()));

        var v = x32.dot(params.to_v.weight.convert(.f32), .d);
        if (params.to_v.bias) |bias| v = v.add(bias.convert(.f32).broad(v.shape()));

        // Q/K RMSNorm with learned weights
        q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.convert(.f32).broad(q.shape()));
        k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.convert(.f32).broad(k.shape()));

        // Split into heads
        var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        const vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

        // Apply RoPE (SPLIT format — cos/sin are [B, H, T, HD/2])
        const q_cos = pe_cos.rename(.{ .t = .q }).convert(.f32);
        const q_sin = pe_sin.rename(.{ .t = .q }).convert(.f32);
        const k_cos = q_cos.rename(.{ .q = .k });
        const k_sin = q_sin.rename(.{ .q = .k });

        qh = applyRotaryEmb(qh.withPartialTags(.{ .b, .q, .h, .hd }), q_cos, q_sin);
        kh = applyRotaryEmb(kh.withPartialTags(.{ .b, .k, .h, .hd }), k_cos, k_sin);

        // SDPA
        // Retag mask from [batch, _h1, _h2, t] → [batch, h, q, k] for sdpa compatibility.
        const sdpa_mask = if (mask) |m| m.rename(.{ .b = .batch, ._h1 = .h, ._h2 = .q, .t = .k }) else null;
        var attn = zml.nn.sdpa(
            qh.rename(.{ .b = .batch }),
            kh.rename(.{ .b = .batch }),
            vh.rename(.{ .b = .batch }).withPartialTags(.{ .batch, .k, .h, .hd }),
            .{ .attn_mask = sdpa_mask },
        ).rename(.{ .batch = .b }); // [B, Q, H, HD]

        // Optional gated attention: gates = 2 * sigmoid(logits)
        // Python: gate_logits = self.to_gate_logits(x) → [B, T, H]
        //         out = out.view(B, T, H, D) * (2 * sigmoid(gate_logits)).unsqueeze(-1)
        //         out = out.view(B, T, H*D)
        if (self.has_gated_attention) {
            if (params.to_gate_logits) |gate_linear| {
                var gate_logits = x32.dot(gate_linear.weight.convert(.f32), .d);
                if (gate_linear.bias) |bias| {
                    gate_logits = gate_logits.add(bias.convert(.f32).broad(gate_logits.shape()));
                }
                // gate_logits is [B, T, H] — rename .t → .q and add .hd=1 for broadcasting
                const gate = gate_logits.sigmoid().scale(2.0)
                    .rename(.{ .t = .q })
                    .insertAxes(3, .{.hd}); // [B, Q, H, HD=1]
                attn = attn.mul(gate.broad(attn.shape()));
            }
        }

        // Merge heads and output projection
        const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t });
        var out = merged.dot(params.to_out.weight.convert(.f32), .d_v);
        if (params.to_out.bias) |bias| {
            out = out.add(bias.convert(.f32).broad(out.shape()));
        }
        return out.convert(out_dtype).withPartialTags(.{ .b, .t, .d });
    }
};

// ============================================================================
// Connector Block (1D Transformer Block)
// ============================================================================
// Python ref: embeddings_connector.py — _BasicTransformerBlock1D
//
// Pipeline: rmsNorm(x) → self-attention → residual → rmsNorm → FF → residual

pub const ConnectorBlock = struct {
    attn: ConnectorAttention,

    pub const Params = struct {
        attn: ConnectorAttention.Params,
        ff: model.FeedForward.Params,

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            ConnectorAttention.Params.unloadBuffers(&params.attn);
            model.FeedForward.Params.unloadBuffers(&params.ff);
        }
    };

    pub const InitResult = struct { block: ConnectorBlock, params: Params };

    pub fn initParams(store: zml.io.TensorStore.View) InitResult {
        const attn_init = ConnectorAttention.initParams(store.withPrefix("attn1"));
        const ff_params = model.FeedForward.initParams(store.withPrefix("ff"));
        return .{
            .block = .{ .attn = attn_init.attn },
            .params = .{
                .attn = attn_init.params,
                .ff = ff_params,
            },
        };
    }

    pub fn forward(
        self: ConnectorBlock,
        hidden_states: Tensor,
        params: Params,
        pe_cos: Tensor,
        pe_sin: Tensor,
        mask: ?Tensor,
    ) Tensor {
        // 1. RMSNorm → self-attention → residual
        const norm1 = zml.nn.rmsNorm(hidden_states, .d, 1e-6);
        const attn_out = self.attn.forward(norm1, params.attn, pe_cos, pe_sin, mask);
        const h1 = attn_out.add(hidden_states);

        // 2. RMSNorm → feed-forward → residual
        const norm2 = zml.nn.rmsNorm(h1, .d, 1e-6);
        const ff_out = (model.FeedForward{}).forward(norm2, params.ff);
        return ff_out.add(h1);
    }
};

// ============================================================================
// Embeddings 1D Connector
// ============================================================================
// Python ref: embeddings_connector.py — Embeddings1DConnector
//
// Pipeline: replace padding with learnable registers → 1D RoPE → N× block → final RMSNorm

pub const Embeddings1DConnector = struct {
    num_heads: usize,
    inner_dim: i64,
    num_blocks: usize,
    blocks: [MAX_CONNECTOR_BLOCKS]ConnectorBlock,

    pub const Params = struct {
        learnable_registers: Tensor,
        block_params: [MAX_CONNECTOR_BLOCKS]ConnectorBlock.Params,
    };

    pub fn unloadBuffers(self: Embeddings1DConnector, params: *zml.Bufferized(Params)) void {
        params.learnable_registers.deinit();
        for (params.block_params[0..self.num_blocks]) |*bp| {
            ConnectorBlock.Params.unloadBuffers(bp);
        }
    }

    /// Initialize connector from checkpoint store.
    /// store should point to e.g. model.diffusion_model.video_embeddings_connector.
    pub fn initParams(store: zml.io.TensorStore.View) struct { connector: Embeddings1DConnector, params: Params } {
        const blocks_store = store.withPrefix("transformer_1d_blocks");

        // Detect number of blocks by probing for layer keys.
        var num_blocks: usize = 0;
        var block_inits: [MAX_CONNECTOR_BLOCKS]ConnectorBlock.InitResult = undefined;
        while (num_blocks < MAX_CONNECTOR_BLOCKS) : (num_blocks += 1) {
            const layer_store = blocks_store.withLayer(num_blocks).withPrefix("attn1").withPrefix("to_q");
            if (!layer_store.hasKey("weight")) break;
            block_inits[num_blocks] = ConnectorBlock.initParams(blocks_store.withLayer(num_blocks));
        }

        if (num_blocks == 0) @panic("No connector blocks found in checkpoint");

        // Use num_heads from block 0's attention (inferred from gate or Q weight).
        const num_heads = block_inits[0].block.attn.num_heads;
        const inner_dim = block_inits[0].params.attn.to_q.weight.dim(.d_q);

        std.log.info("Connector: num_heads={}, inner_dim={}, head_dim={}, num_blocks={}, has_gate={}", .{
            num_heads,  inner_dim,                                     @divExact(inner_dim, @as(i64, @intCast(num_heads))),
            num_blocks, block_inits[0].block.attn.has_gated_attention,
        });

        var connector: Embeddings1DConnector = .{
            .num_heads = num_heads,
            .inner_dim = inner_dim,
            .num_blocks = num_blocks,
            .blocks = undefined,
        };
        var params: Params = .{
            .learnable_registers = store.createTensor("learnable_registers", .{ .n_reg, .d }, null),
            .block_params = undefined,
        };
        for (0..num_blocks) |i| {
            connector.blocks[i] = block_inits[i].block;
            params.block_params[i] = block_inits[i].params;
        }

        return .{ .connector = connector, .params = params };
    }

    pub const ConnectorResult = struct {
        encoded: Tensor,
        mask: Tensor,
    };

    /// Forward pass: features + additive mask → encoded + mask.
    ///
    /// features: [B, S, D] — projected features from FeatureExtractorV2
    /// additive_mask: [B, 1, 1, S] — additive attention mask (0 = attend, -inf = don't)
    /// Returns: (encoded [B, S, D], mask [B, 1, 1, S])
    pub fn forward(self: Embeddings1DConnector, features: Tensor, additive_mask: Tensor, binary_mask_2d: Tensor, params: Params) ConnectorResult {
        var hidden_states = features.withPartialTags(.{ .b, .t, .d });
        var attn_mask = additive_mask;

        // 1. Replace padded tokens with learnable registers.
        const replaced = replacePaddedWithRegisters(hidden_states, attn_mask, binary_mask_2d, params.learnable_registers);
        hidden_states = replaced.hidden_states;
        attn_mask = replaced.mask;

        // 2. Compute 1D RoPE for the connector.
        const seq_len = hidden_states.dim(.t);
        const rope = compute1dRope(
            seq_len,
            @intCast(self.num_heads),
            self.inner_dim,
            CONNECTOR_ROPE_THETA,
            CONNECTOR_MAX_POS,
            hidden_states.dtype(),
        );

        // 3. Run transformer blocks.
        //    After register replacement, mask is all zeros (attend everywhere) → pass null.
        for (0..self.num_blocks) |i| {
            hidden_states = self.blocks[i].forward(
                hidden_states,
                params.block_params[i],
                rope.cos,
                rope.sin,
                null,
            );
        }

        // 4. Final RMSNorm (no learned weight).
        hidden_states = zml.nn.rmsNorm(hidden_states, .d, 1e-6);

        return .{
            .encoded = hidden_states,
            .mask = attn_mask,
        };
    }
};

// ============================================================================
// Register Replacement
// ============================================================================
// Python ref: embeddings_connector.py — _replace_padded_with_learnable_registers
//
// Input layout (left-padded):  [pad, pad, ..., real_0, real_1, ..., real_{n-1}]
// Output layout:               [real_0, real_1, ..., real_{n-1}, reg_0, ..., reg_{p-1}]
// Output mask: all zeros (attend everywhere, including registers).
//
// Implementation uses a cyclic roll: shift left by num_pad positions.
// Then blend with tiled registers using the flipped mask.

fn replacePaddedWithRegisters(
    hidden_states: Tensor,
    additive_mask: Tensor,
    binary_mask: Tensor,
    learnable_registers: Tensor,
) struct { hidden_states: Tensor, mask: Tensor } {
    const seq_len = hidden_states.dim(.t);
    const num_register_tiles: i64 = @divExact(seq_len, NUM_REGISTERS);

    // binary_mask: [B, T] — 1 for real tokens, 0 for padding

    // Compute num_pad = S - sum(mask) for the cyclic roll.
    // sum gives per-batch count of real tokens.
    // Note: ZML sum keeps the reduced dim as size 1, so num_real is [B, T=1].
    const num_real = binary_mask.convert(.i32).sum(.t); // [B, T=1]
    const s_tensor = Tensor.scalar(@as(i32, @intCast(seq_len)), .i32).broad(num_real.shape());
    const num_pad = s_tensor.sub(num_real); // [B, T=1]

    // Create rolled indices: (iota(S) + num_pad) % S
    // Create iota [B, S] with values 0..S-1 along .t, then shift by num_pad.
    const iota_2d = Tensor.iota(binary_mask.shape(), .t); // [B, S] with values 0..S-1 along .t
    // num_pad is already [B, T=1] — broadcast directly to [B, T=S]
    const num_pad_expanded = num_pad.broad(iota_2d.shape()); // [B, S]
    const s_broad = s_tensor.broad(iota_2d.shape());
    const rolled_indices = iota_2d.convert(.i32).add(num_pad_expanded).remainder(s_broad);

    // Gather: rolled hidden_states along .t using rolled_indices.
    // Rename .t → ._idx in indices to avoid tag conflict with gather axis .t.
    const rolled = hidden_states.gather(.{ .t = rolled_indices.withTags(.{ .b, ._idx }) }, .{})
        .rename(.{ ._idx = .t });

    // Tile learnable registers: [N_REG, D] → repeat → [S, D]
    const tiled_registers = learnable_registers
        .repeat1d(.n_reg, @intCast(num_register_tiles))
        .rename(.{ .n_reg = .t })
        .withPartialTags(.{ .t, .d });

    // Flipped binary mask: reverse along .t → [B, S]
    const flipped_mask = binary_mask.reverse(.{.t});

    // Blend: where flipped_mask=1 use rolled (real tokens), where 0 use registers
    const flipped_f = flipped_mask.convert(hidden_states.dtype()).insertAxes(2, .{.d}).broad(hidden_states.shape());
    const one_minus = Tensor.scalar(@as(f32, 1.0), hidden_states.dtype()).broad(flipped_f.shape()).sub(flipped_f);
    const tiled_broad = tiled_registers.broad(hidden_states.shape());
    const blended = rolled.mul(flipped_f).add(tiled_broad.mul(one_minus));

    // Output mask: all zeros (attend everywhere)
    const zero_mask = Tensor.constant(additive_mask.dtype().zero()).broad(additive_mask.shape());

    return .{
        .hidden_states = blended.withPartialTags(.{ .b, .t, .d }),
        .mask = zero_mask,
    };
}

// ============================================================================
// 1D RoPE for Connector
// ============================================================================
// Python ref: embeddings_connector.py — precompute_freqs_cis with indices_grid = arange(S)
//
// Uses the same frequency formula as model.zig (generateFreqGrid + generateFreqs + splitFreqsCis):
//   freq_basis[i] = theta^(linspace(0, 1, num_freqs)[i]) * π/2
//   positions: frac = arange(S) / max_pos, scaled = 2*frac - 1
//
// Output: SPLIT format cos/sin shaped [B, H, T, HD/2].

pub fn compute1dRope(
    seq_len: i64,
    num_heads: i64,
    inner_dim: i64,
    theta: f64,
    max_pos: i64,
    dtype: zml.dtype.DataType,
) CosSinPair {
    const head_dim = @divExact(inner_dim, num_heads);
    const half_hd = @divExact(head_dim, 2);
    const num_freqs = @divExact(inner_dim, 2); // n_pos_dims=1 → inner_dim / (2*1)

    // 1. Frequency basis (matches generateFreqGrid with n_pos_dims=1):
    //    freq_basis[i] = exp(i/(num_freqs-1) * ln(theta)) * π/2
    const log_theta = std.math.log(f64, std.math.e, theta);
    const step: f64 = if (num_freqs > 1) 1.0 / @as(f64, @floatFromInt(num_freqs - 1)) else 0.0;
    const idx = Tensor.arange(.{ .end = num_freqs }, .f32);
    const freq_basis = idx.scale(step * log_theta).exp().scale(std.math.pi / 2.0);
    // freq_basis: [num_freqs] f32

    // 2. Fractional positions: frac = arange(S) / max_pos, scaled = 2*frac - 1
    const positions = Tensor.arange(.{ .end = seq_len }, .f32);
    const frac = positions.scale(1.0 / @as(f64, @floatFromInt(max_pos)));
    const scaled = frac.scale(2.0).addConstant(-1.0);
    const scaled_t = scaled.withTags(.{.t}); // [S(.t)]

    // 3. Outer product: freqs[t, k] = scaled[t] * freq_basis[k] → [S, num_freqs]
    const pos_2d = scaled_t.appendAxes(.{.freq}).broad(
        scaled_t.shape().append(.{ .freq = num_freqs }),
    );
    const fb = freq_basis.withTags(.{.freq});
    const freqs = pos_2d.mul(fb.broad(pos_2d.shape())); // [S(.t), num_freqs(.freq)]

    // 4. cos/sin → [S, num_freqs]
    var cos_freq = freqs.cos();
    var sin_freq = freqs.sin();

    // 5. Add batch dim → [1, S, num_freqs]
    cos_freq = cos_freq.insertAxes(0, .{.b});
    sin_freq = sin_freq.insertAxes(0, .{.b});

    // 6. SPLIT format: reshape [B, T, num_freqs] → [B, T, H, HD/2] → transpose → [B, H, T, HD/2]
    //    Pad if num_freqs < num_heads * half_hd (Python PREPENDS padding: cat([padding, freqs]))
    const total_needed = num_heads * half_hd;
    if (cos_freq.dim(.freq) < total_needed) {
        const pad_size = total_needed - cos_freq.dim(.freq);
        const bt_shape = cos_freq.shape().set(.freq, pad_size);
        const cos_pad = Tensor.scalar(@as(f32, 1.0), .f32).broad(bt_shape);
        const sin_pad = Tensor.scalar(@as(f32, 0.0), .f32).broad(bt_shape);
        cos_freq = Tensor.concatenate(&.{ cos_pad, cos_freq }, .freq);
        sin_freq = Tensor.concatenate(&.{ sin_pad, sin_freq }, .freq);
    }

    const cos_split = cos_freq.reshape(cos_freq.shape().splitAxis(.freq, .{ .h = num_heads, .hd = half_hd }));
    const sin_split = sin_freq.reshape(sin_freq.shape().splitAxis(.freq, .{ .h = num_heads, .hd = half_hd }));

    const rope_shape = Shape.init(.{ .b = @as(i64, 1), .h = num_heads, .t = seq_len, .hd = half_hd }, .f32);
    return .{
        .cos = cos_split.transpose(rope_shape).convert(dtype),
        .sin = sin_split.transpose(rope_shape).convert(dtype),
    };
}

// ============================================================================
// Embeddings Processor (Top-Level)
// ============================================================================
// Python ref: embeddings_processor.py — EmbeddingsProcessor
//
// Orchestrates: feature extraction → mask conversion → connectors → binary mask

pub const EmbeddingsProcessor = struct {
    video_connector: Embeddings1DConnector,
    audio_connector: Embeddings1DConnector,

    pub const Params = struct {
        feature_extractor: FeatureExtractorV2.Params,
        video_connector: Embeddings1DConnector.Params,
        audio_connector: Embeddings1DConnector.Params,
    };

    pub fn unloadBuffers(self: EmbeddingsProcessor, params: *zml.Bufferized(Params)) void {
        FeatureExtractorV2.Params.unloadBuffers(&params.feature_extractor);
        self.video_connector.unloadBuffers(&params.video_connector);
        self.audio_connector.unloadBuffers(&params.audio_connector);
    }

    pub const Result = struct {
        v_context: Tensor,
        a_context: Tensor,
        binary_mask: Tensor,
    };

    /// Initialize all params from the LTX checkpoint.
    pub fn initParams(store: zml.io.TensorStore.View) struct { processor: EmbeddingsProcessor, params: Params } {
        const v_conn_init = Embeddings1DConnector.initParams(
            store.withPrefix("model.diffusion_model").withPrefix("video_embeddings_connector"),
        );
        const a_conn_init = Embeddings1DConnector.initParams(
            store.withPrefix("model.diffusion_model").withPrefix("audio_embeddings_connector"),
        );

        return .{
            .processor = .{
                .video_connector = v_conn_init.connector,
                .audio_connector = a_conn_init.connector,
            },
            .params = .{
                .feature_extractor = FeatureExtractorV2.initParams(store),
                .video_connector = v_conn_init.params,
                .audio_connector = a_conn_init.params,
            },
        };
    }

    /// Full forward pass: stacked hidden states → context embeddings.
    pub fn forward(
        self: EmbeddingsProcessor,
        stacked_hidden_states: Tensor,
        attention_mask: Tensor,
        params: Params,
    ) Result {
        // 1. Feature extraction
        const features = FeatureExtractorV2.forward(stacked_hidden_states, attention_mask, params.feature_extractor);

        // 2. Convert binary mask → additive mask for transformer attention.
        //    (mask - 1) * dtype_max → 0 for real tokens, -inf for padding
        //    Shape: [B, S] → [B, 1, 1, T]   (rename .s → .t for connector convention)
        const mask_t = attention_mask.rename(.{ .s = .t });
        const mask_f = mask_t.convert(features.video_features.dtype());
        const one = Tensor.scalar(@as(f32, 1.0), mask_f.dtype()).broad(mask_f.shape());
        const dtype_max = Tensor.scalar(std.math.floatMax(f32), mask_f.dtype()).broad(mask_f.shape());
        const additive_mask_2d = mask_f.sub(one).mul(dtype_max); // [B, T]
        // Reshape to [B, 1, 1, S] for attention
        const additive_mask = additive_mask_2d
            .insertAxes(1, .{._h1})
            .insertAxes(2, .{._h2}); // [B, 1, 1, S]

        // 2D binary mask for register replacement logic [B, T]
        const binary_mask_2d = mask_t.convert(.f32).cmp(.GT, Tensor.scalar(@as(f32, 0.5), .f32).broad(mask_t.shape()));

        // 3. Video connector
        const v_result = self.video_connector.forward(
            features.video_features,
            additive_mask,
            binary_mask_2d,
            params.video_connector,
        );

        // 4. Audio connector
        const a_result = self.audio_connector.forward(
            features.audio_features,
            additive_mask,
            binary_mask_2d,
            params.audio_connector,
        );

        // 5. Binary mask from connector output.
        //    After register replacement, the mask is all zeros → binary_mask is all 1s.
        //    _to_binary_mask: binary = abs(mask) < eps → True for zero-valued positions.
        //    We reproduce this from the 4D mask by reducing to 2D: take one element along singleton _h dims.
        //    v_result.mask is [B, _h1=1, _h2=1, T] all zeros → threshold check → all True.
        const v_mask_4d = v_result.mask;
        const eps_4d = Tensor.scalar(@as(f32, 0.000001), v_mask_4d.dtype()).broad(v_mask_4d.shape());
        const binary_mask_4d = v_mask_4d.abs().cmp(.LT, eps_4d); // [B, 1, 1, T]
        // Reduce singleton dims: merge [B, _h1, _h2, T] → [B, T*1*1] then it's already [B, T]
        // Since _h1=_h2=1 the merge is lossless.
        const binary_mask = binary_mask_4d
            .merge(.{ .t = .{ ._h1, ._h2, .t } }); // [B, T]
        const binary_f = binary_mask.convert(v_result.encoded.dtype())
            .insertAxes(2, .{.d}).broad(v_result.encoded.shape());
        const v_context = v_result.encoded.mul(binary_f);

        return .{
            .v_context = v_context,
            .a_context = a_result.encoded,
            .binary_mask = binary_mask.convert(.f32),
        };
    }
};

// ============================================================================
// Graph Function for Compilation
// ============================================================================

/// Top-level graph function: compile this with platform.compileFn.
///
/// Arguments:
///   stacked_hidden_states: [B=1, S=1024, D=3840, L=49] bf16
///   attention_mask: [B=1, S=1024] i64 (binary: 1=real, 0=pad)
///   params: EmbeddingsProcessor.Params (loaded from LTX checkpoint)
///
/// Returns:
///   .v_context: [B=1, S=1024, D_v] bf16
///   .a_context: [B=1, S=1024, D_a] bf16
///   .binary_mask: [B=1, S=1024] f32
pub fn forwardEmbeddingsProcessor(
    processor: *const EmbeddingsProcessor,
    stacked_hidden_states: Tensor,
    attention_mask: Tensor,
    params: EmbeddingsProcessor.Params,
) EmbeddingsProcessor.Result {
    // Tag untagged inputs: [B, S, D=3840, L=49] and [B, S]
    const hs = stacked_hidden_states.withTags(.{ .b, .s, .d, .l });
    const mask = attention_mask.withTags(.{ .b, .s });
    return processor.forward(hs, mask, params);
}

// ============================================================================
// Rotary Embedding (local copy from model.zig — private there)
// ============================================================================

/// Applies split RoPE (cos/sin have half the head dimension of x).
fn applyRotaryEmb(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
    const half: i64 = @divExact(x.dim(.hd), 2);
    const x_first = x.slice1d(.hd, .{ .start = 0, .end = half });
    const x_second = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });

    const cos_b = cos.transpose(x_first.shape()).broad(x_first.shape());
    const sin_b = sin.transpose(x_first.shape()).broad(x_first.shape());

    const out_first = x_first.mul(cos_b).sub(x_second.mul(sin_b));
    const out_second = x_second.mul(cos_b).add(x_first.mul(sin_b));
    return Tensor.concatenate(&.{ out_first, out_second }, .hd);
}
