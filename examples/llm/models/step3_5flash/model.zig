const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;
const common = @import("../common.zig");

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,

    auto_map: ?AutoMap = null,

    rope_scaling: RopeScaling,
    yarn_only_types: []const []const u8 = &.{},

    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    max_seq_len: u32,
    vocab_size: u32,

    torch_dtype: []const u8 = "bfloat16",

    use_qk_norm: bool = false,

    moe_layers_enum: []const u8 = "",
    num_attention_heads: u32,
    num_attention_groups: u32,
    head_dim: u32,

    use_moe: bool = false,
    moe_num_experts: u32 = 0,
    moe_top_k: u32 = 0,
    moe_intermediate_size: u32 = 0,
    share_expert_dim: u32 = 0,
    moe_layer_offset: u32 = 0,
    moe_every_n_layer: u32 = 1,
    norm_expert_weight: bool = false,
    moe_router_activation: []const u8 = "sigmoid",
    moe_router_scaling_factor: f32 = 1.0,

    att_impl_type: []const u8 = "GQA",
    tie_word_embeddings: bool = false,

    rope_theta: []const f32,

    use_head_wise_attn_gate: bool = false,
    sliding_window: u32 = 0,

    use_moe_router_bias: bool = false,
    need_fp32_gate: bool = false,
    sink: bool = false,

    layer_types: []const []const u8 = &.{},
    use_rope_layers: []const u32 = &.{},

    num_nextn_predict_layers: u32 = 0,
    partial_rotary_factors: []const f32 = &.{},

    attention_other_setting: ?AttentionOtherSetting = null,

    swiglu_limits: []const f32 = &.{},
    swiglu_limits_shared: []const f32 = &.{},

    zero_centered: bool = false,
    max_position_embeddings: u32,

    pub const AutoMap = struct {
        AutoConfig: []const u8,
        AutoModelForCausalLM: []const u8,
    };

    pub const RopeScaling = struct {
        rope_type: []const u8,
        factor: f32,
        original_max_position_embeddings: u32,
        low_freq_factor: f32,
        high_freq_factor: f32,
    };

    pub const AttentionOtherSetting = struct {
        attention_type: []const u8,
        num_attention_heads: u32,
        num_attention_groups: u32,
        head_dim: u32,
        true_head_dim: u32,
    };

    pub fn numKeyValueHeads(self: Config) u32 {
        return self.num_attention_groups;
    }
};

// Options
pub const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,
};

// TODO: Model struct. Do we need a LoadedModel?

// TODO: buffers

// TODO: Transformer Layer

// TODO: retrieve hardcoded values from config
const num_q_heads_full: i64 = 64;
const num_q_heads_swa: i64 = 96;
const num_kv_heads: i64 = 8;
const head_dim: i64 = 128;
const rotary_dim: i64 = head_dim;
const main_layer_count: usize = 45;

fn isFullAttentionLayer(layer_idx: usize) bool {
    // Step 3.5 Flash pattern: Full, SWA, SWA, SWA, repeating, for the 45 main layers.
    // Layers >= 45 are MTP/speculative blocks and use the SWA shape.
    return layer_idx < main_layer_count and layer_idx % 4 == 0;
}

const rope_theta_full: f32 = 5000000.0;
const rope_theta_swa: f32 = 10000.0;
const rope_original_max_position: u32 = 131072;
const rope_scaling_factor: f32 = 2.0;
const rope_low_freq_factor: f32 = 1.0;
const rope_high_freq_factor: f32 = 32.0;

fn ropeScalingForLayer(layer_idx: usize) zml.nn.RopeOpts.Scaling {
    const theta: f32 = if (isFullAttentionLayer(layer_idx)) rope_theta_full else rope_theta_swa;
    return .{ .llama3 = .{
        .factor = rope_scaling_factor,
        .high_freq_factor = rope_high_freq_factor,
        .low_freq_factor = rope_low_freq_factor,
        .original_max_position_embeddings = rope_original_max_position,
        .rope_theta = theta,
    } };
}

// TODO: Attention struct wrapping SelfAttn and SwAttn
pub const SelfAttn = struct {
    layer_idx: usize,
    enable_sliding_window: bool,

    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    g_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    q_size: i64,
    kv_size: i64,

    head_dim: i64,
    scaling: f32,

    num_q_heads: i64,
    num_kv_heads: i64,
    num_kv_groups: i64,
    rotary_dim: i64,
    rotary_emb: TextRotaryEmbedding,

    // do we need initProj
    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
            .d,
        );
    }

    // TODO: add config here
    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize) !SelfAttn {
        const is_full = isFullAttentionLayer(layer_idx);
        const num_q_heads: i64 = if (is_full) num_q_heads_full else num_q_heads_swa;
        return .{
            .layer_idx = layer_idx,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .enable_sliding_window = !is_full,
            //TODO: hardcoded head dim
            .head_dim = head_dim,
            .num_kv_groups = @divExact(num_q_heads, num_kv_heads),
            //TODO: hardcoded rotary emb
            .rotary_dim = rotary_dim,
            .rotary_emb = .init(rotary_dim, ropeScalingForLayer(layer_idx), 1.0),
            .q_size = num_q_heads * head_dim,
            .kv_size = num_kv_heads * head_dim,
            .scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .g_proj = initProj(store.withPrefix("g_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            //TODO: hardcoded eps
            .q_norm = .init(store.withPrefix("q_norm"), 1e-5),
            .k_norm = .init(store.withPrefix("k_norm"), 1e-5),
        };
    }

    // unloadBuffers
    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*b| b.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*b| b.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*b| b.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*b| b.deinit();
        self.g_proj.weight.deinit();
        if (self.g_proj.bias) |*b| b.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    // project Q, Gate
    // Step 3.5 Flash keeps q and gate as separate projections (q_proj and g_proj). q_proj outputs num_q_heads * head_dim (= 12288 for layer 30).
    // g_proj is a head-wise attention gate: one scalar per query head (dout = num_q_heads), broadcast over .hd and applied to the per-head attention output before merging heads.
    fn projectQAndGate(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q = self.q_proj.forward(x)
            .splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });

        const gate = self.g_proj.forward(x).rename(.{ .dout = .h });

        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const k = self.k_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        const v = self.v_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        return .{ k, v };
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        // kv_cache: zml.Tensor,
    ) struct { zml.Tensor } {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        var q, const gate = self.projectQAndGate(input);
        var k, var v = self.projectKV(input);

        q = self.q_norm.forward(q, .hd);
        k = self.k_norm.forward(k, .hd);

        const dtype = q.dtype();
        // see how long the sequence is? what is x.dim(.s) for?
        const position_ids = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});

        const cos, const sin = self.rotary_emb.getCosAndSin(position_ids, dtype);

        q = self.rotary_emb.applyRope(q, cos, sin);
        k = self.rotary_emb.applyRope(k, cos, sin);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        // The vanilla attention expects a scalar start index into the KV
        // sequence; the recorded `cache_position` is the full per-token
        // positions array, so take its first element.
        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            attn_start,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, input.dim(.s), self.num_q_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        );

        // Head-wise gate: sigmoid(gate) is {b, s, h}; broadcast over .hd and
        // apply before merging heads and o_proj.
        const gate_b = gate.sigmoid().rename(.{ .s = .q }).broad(attn_output.shape());
        const gated_attn = attn_output.mul(gate_b);

        const projected_output = self.o_proj.forward(
            gated_attn.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s }),
        );

        return .{projected_output};
    }

    pub const Stages = struct {
        q_proj: zml.Tensor,
        k_proj: zml.Tensor,
        v_proj: zml.Tensor,
        g_proj: zml.Tensor,
        q_norm: zml.Tensor,
        k_norm: zml.Tensor,
        // HF dumper records q/k pre-rope in [b,h,s,hd] layout (rope.q_in, rope.k_in)
        q_pre_rope_hf: zml.Tensor,
        k_pre_rope_hf: zml.Tensor,
        // cos/sin in HF layout [1,s,hd]
        cos: zml.Tensor,
        sin: zml.Tensor,
        // post-rope, HF layout [b,h,s,hd] (rope.q_embed, rope.k_embed)
        q_rope_hf: zml.Tensor,
        k_rope_hf: zml.Tensor,
        attn: zml.Tensor,
        gate_sig: zml.Tensor,
        gated: zml.Tensor,
        // input to o_proj: merged head output [b,s,h*hd]
        o_proj_in: zml.Tensor,
        out: zml.Tensor,
    };

    pub fn forwardStages(self: SelfAttn, x: zml.Tensor, token_index: zml.Tensor) Stages {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const q_proj_raw = self.q_proj.forward(input);
        const k_proj_raw = self.k_proj.forward(input);
        const v_proj_raw = self.v_proj.forward(input);
        const g_proj_raw = self.g_proj.forward(input);

        var q = q_proj_raw.splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });
        var k = k_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = v_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const gate = g_proj_raw.rename(.{ .dout = .h });

        const q_after_norm = self.q_norm.forward(q, .hd);
        const k_after_norm = self.k_norm.forward(k, .hd);
        q = q_after_norm;
        k = k_after_norm;

        const q_pre_rope_hf = q.transpose(.{ .b, .h, .s, .hd });
        const k_pre_rope_hf = k.transpose(.{ .b, .h, .s, .hd });

        const dtype = q.dtype();
        const position_ids = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});
        const cos_raw, const sin_raw = self.rotary_emb.getCosAndSin(position_ids, dtype);
        const cos = cos_raw.insertAxes(0, .{.b});
        const sin = sin_raw.insertAxes(0, .{.b});

        const q_rope = self.rotary_emb.applyRope(q, cos_raw, sin_raw);
        const k_rope = self.rotary_emb.applyRope(k, cos_raw, sin_raw);

        const q_rope_hf = q_rope.transpose(.{ .b, .h, .s, .hd });
        const k_rope_hf = k_rope.transpose(.{ .b, .h, .s, .hd });

        const q_for_attn = q_rope.rename(.{ .s = .q });
        const k_for_attn = k_rope.rename(.{ .s = .k });
        const v_for_attn = v.rename(.{ .s = .k });

        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });
        const attn_output = zml.attention.attention.attention(
            q_for_attn,
            k_for_attn,
            v_for_attn,
            attn_start,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, input.dim(.s), self.num_q_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        );

        const gate_sig = gate.sigmoid().rename(.{ .s = .q });
        const gated = attn_output.mul(gate_sig.broad(attn_output.shape()));

        const o_proj_in = gated.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const out = self.o_proj.forward(o_proj_in);

        return .{
            .q_proj = q_proj_raw,
            .k_proj = k_proj_raw,
            .v_proj = v_proj_raw,
            .g_proj = g_proj_raw,
            .q_norm = q_after_norm,
            .k_norm = k_after_norm,
            .q_pre_rope_hf = q_pre_rope_hf,
            .k_pre_rope_hf = k_pre_rope_hf,
            .cos = cos,
            .sin = sin,
            .q_rope_hf = q_rope_hf,
            .k_rope_hf = k_rope_hf,
            .attn = attn_output,
            .gate_sig = gate_sig,
            .gated = gated,
            .o_proj_in = o_proj_in,
            .out = out,
        };
    }
};

pub const TextRotaryEmbedding = struct {
    rotary_dim: i64,
    scaling: zml.nn.RopeOpts.Scaling,
    attention_scaling: f32 = 1.0,

    pub fn init(rotary_dimension: i64, scaling: zml.nn.RopeOpts.Scaling, attention_scaling: f32) TextRotaryEmbedding {
        return .{
            .rotary_dim = rotary_dimension,
            .scaling = scaling,
            .attention_scaling = attention_scaling,
        };
    }

    /// Expects `position_ids` tagged `{.b, .s}` and returns cos/sin tagged `{.b, .s, .hd}`
    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const opts: zml.nn.RopeOpts = .{ .scaling = self.scaling };
        const inv_freq = zml.nn.invFreq(self.rotary_dim, opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs_t, freqs_t }, .hd);

        const cos = emb.cos().scale(self.attention_scaling).convert(dtype);
        const sin = emb.sin().scale(self.attention_scaling).convert(dtype);
        return .{ cos, sin };
    }

    pub fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(.hd, .{ .start = half_dim, .end = x.dim(.hd) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    }

    /// Expects `q`/`k` tagged `{.b, .s, .h, .hd}` and `cos`/`sin` tagged `{.b, .s, .hd}`.
    pub fn applyRope(
        self: TextRotaryEmbedding,
        x: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        const x_rot = x.slice1d(.hd, .{ .start = 0, .end = self.rotary_dim });

        // Insert head axis so cos/sin broadcast over .h.
        const cos_b = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_b = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const x_rotated = x_rot.mul(cos_b).add(rotateHalf(x_rot).mul(sin_b));

        if (self.rotary_dim == x.dim(.hd)) {
            return x_rotated;
        }

        const x_pass = x.slice1d(.hd, .{ .start = self.rotary_dim, .end = x.dim(.hd) });

        return zml.Tensor.concatenate(&.{ x_rotated, x_pass }, .hd);
    }
};

// TODO: KV Cache

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    limit: ?i32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: ?i32) Mlp {
        return .{
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }),
                null,
                .dout,
            ),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically
        if (self.limit) |limit| {
            if (limit != 0) {
                const lim_f = @as(f32, @floatFromInt(limit));
                const max_t = zml.Tensor.scalar(lim_f, gate.dtype());
                const min_t = zml.Tensor.scalar(-lim_f, gate.dtype());

                // Step 3.5 Flash has asymmetric clamping of gate projection
                gate = gate.minimum(max_t);
                up_proj = up_proj.clamp(min_t, max_t);
            }
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

// RmsNorm
pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along the given axis.
    /// Step 3.5 Flash uses offset-style RMSNorm: the effective scale is `1 + weight`,
    /// not just `weight`.
    pub fn forward(self: RmsNorm, input: zml.Tensor, comptime axis: anytype) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{axis});
        const normalized = zml.nn.rmsNorm(x, axis, self.eps);
        const scale = self.weight.convert(.f32).addConstant(1).convert(x.dtype()).withTags(.{axis});
        return normalized.mul(scale.broad(normalized.shape()));
    }
};

pub const Router = struct {
    gate: zml.nn.Linear, // no bias inside the Linear; router_bias is applied post-sigmoid
    router_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,

    // k = num_experts_per_tok
    // `store` is the view at the parent `...moe` prefix; HF layout is:
    //   <prefix>.gate.weight
    //   <prefix>.router_bias   (optional)
    pub fn init(
        store: zml.io.TensorStore.View,
        num_experts_per_tok: u32,
        routed_scaling_factor: f32,
    ) Router {
        return .{
            .gate = .init(
                store.createTensor("gate.weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .router_bias = store.createTensor("router_bias", .{.expert}, .{ .expert = .replicated }),
            .num_experts_per_tok = num_experts_per_tok,
            .routed_scaling_factor = routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.gate.weight.deinit();
        self.router_bias.deinit();
    }

    /// Returns (topk weights, topk indices) with shapes {.b, .s, .topk}.
    /// Matches `Step3p5MoEMLP.router_bias_func`:
    ///   logits = x @ W^T
    ///   probs  = sigmoid(logits)        (in fp32)
    ///   biased = probs + router_bias    (only used to choose indices)
    ///   ids    = topk(biased)
    ///   wts    = gather(probs, ids)     (from UNBIASED probs)
    ///   if renormalize: wts /= sum(wts) + 1e-20
    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // Match HF reference: always run gate matmul + sigmoid in fp32.
        const tagged = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = tagged.convert(.f32);

        const gate_f32: zml.nn.Linear = .init(
            self.gate.weight.convert(.f32),
            null,
            .d,
        );

        // calculate raw logits
        const logits = gate_f32.forward(input);

        // calculate scores of how strongly a token matches an expert
        const expert_scores = logits.sigmoid();

        // add bias to influence how experts are distributed
        const expert_scores_with_bias = expert_scores.add(self.router_bias.convert(.f32).broad(expert_scores.shape()));

        // with N=288 total experts, pick top_k (k=8)
        const topk = expert_scores_with_bias.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);

        // Gather from UNBIASED probs — bias is only used to pick indices.
        const router_scores = expert_scores.gather(.{ .expert = topk_ids }, .{});

        // Renormalize so the topk weights sum to 1. Scaling by
        // routed_scaling_factor happens in the caller (Moe.forward), matching
        // the HF reference where `route()` returns unscaled weights.
        const denom = router_scores.sum(.topk).addConstant(1e-20);
        const normalized = router_scores.div(denom);

        return .{ normalized, topk_ids };
    }
};

// Moe
pub const Moe = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,
    router: Router,
    layer_idx: usize,
    limit: ?f32,

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize) !Moe {
        // init the up, gate, down tensors

        const up_proj_tensor = store.createTensor(
            "up_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const gate_proj_tensor = store.createTensor(
            "gate_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const down_proj_tensor = store.createTensor(
            "down_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        return .{
            // swiglu limit temporarily hardcoded to 0
            // .shared_expert = .init(store.parent().withPrefix("share_expert"), 0),
            .up_proj = up_proj_tensor,
            .gate_proj = gate_proj_tensor,
            .down_proj = down_proj_tensor,
            .router = .init(store, 8, 3.0),
            .layer_idx = layer_idx,
            .limit = switch (layer_idx) {
                43, 44 => 7.0,
                else => null,
            },
        };
    }

    pub fn deinit(self: *zml.Bufferized(Moe)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: Moe, x: zml.Tensor) zml.Tensor {
        if (self.layer_idx >= 42 and self.layer_idx <= 44) {
            return self.forwardLoop(x);
        }
        return self.forwardTriton(x);
    }

    fn forwardTriton(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const routing_scores, const topk_ids = self.router.forward(input);
        const scaled = routing_scores.scale(self.router.routed_scaling_factor);

        const topk_ids_tensor = topk_ids.rename(.{ .topk = .top_expert });
        const scaled_tensor = scaled.rename(.{ .topk = .top_expert });

        const gate_up_proj = zml.Tensor.concatenate(&.{ self.gate_proj, self.up_proj }, .dout).rename(.{ .dout = .out, .d = .in });
        const down_proj = self.down_proj.rename(.{ .dout = .out, .d = .in });

        // TODO: hardcoded zml.moe.metadata, zml.moe.parameters
        const moe_metadata = zml.moe.Metadata.init(.{ .triton = .{} });
        const moe_parameters = zml.moe.Parameters.init(.{ .triton = .{ .num_experts_per_tok = self.router.num_experts_per_tok, .activation = .silu } });

        const moe_output = zml.moe.forwardMoe(
            input,
            topk_ids_tensor,
            scaled_tensor,
            gate_up_proj,
            null,
            null,
            down_proj,
            null,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        return moe_output;
    }

    fn forwardLoop(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const routing_scores, const topk_ids = self.router.forward(input);
        const routing_scaled = routing_scores.scale(self.router.routed_scaling_factor);

        const x_f32 = input.convert(.f32);
        const gate_w = self.gate_proj.convert(.f32);
        const up_w = self.up_proj.convert(.f32);
        const down_w = self.down_proj.convert(.f32);

        const gate_all = x_f32.dot(gate_w, .d);
        const up_all = x_f32.dot(up_w, .d);

        var gate_act = gate_all.silu();
        var up_act = up_all;
        if (self.limit) |lim| {
            gate_act = gate_act.minimum(.scalar(lim, .f32));
            up_act = up_act.clamp(.scalar(-lim, .f32), .scalar(lim, .f32));
        }
        const act_d = gate_act.mul(up_act).rename(.{ .dout = .d });
        const down_all = act_d.dot(down_w, .d).transpose(.{ .b, .s, .expert, .dout });
        const down_topk = down_all.gather(.{ .expert = topk_ids }, .{});
        const routing = routing_scaled.convert(.f32).broad(down_topk.shape());
        const weighted = down_topk.mul(routing).convert(x.dtype());
        const sort_order = topk_ids.argsort(.topk, .{ .descending = false })
            .rename(.{ .topk = .topk_sorted });
        const weighted_sorted = weighted
            .gather(.{ .topk = sort_order }, .{})
            .rename(.{ .topk_sorted = .topk });

        var acc = weighted_sorted.slice1d(.topk, .single(0));
        inline for (1..8) |i| {
            const contrib = weighted_sorted.slice1d(.topk, .single(@as(i64, @intCast(i))));
            acc = acc.add(contrib);
        }
        return acc;
    }
};
