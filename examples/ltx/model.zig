const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

/// Which parameter set to use at runtime.
pub const Stage = enum {
    stage1,
    stage2,
};

pub const Config = struct {
    num_transformer_blocks: usize,
};

/// FeedForward module shared by stage 1 and stage 2.
///
/// Forward contract for LTX 2.3 video FF:
/// [B, T, 4096] -> Linear(4096->16384) -> GELU(tanh approx) -> Linear(16384->4096)
pub const FeedForward = struct {
    pub const Params = struct {
        proj: zml.nn.Linear,
        out: zml.nn.Linear,
    };

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const proj_store = store.withPrefix("net").withLayer(0).withPrefix("proj");
        const out_store = store.withPrefix("net").withLayer(2);

        return .{
            .proj = .init(
                proj_store.createTensor("weight", .{ .d_ff, .d }, null),
                proj_store.createTensor("bias", .{.d_ff}, null),
                .d,
            ),
            .out = .init(
                out_store.createTensor("weight", .{ .d, .d_ff }, null),
                out_store.createTensor("bias", .{.d}, null),
                .d_ff,
            ),
        };
    }

    pub fn forward(self: FeedForward, x: Tensor, params: Params) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const h1 = params.proj.forward(x_);
        const h2 = h1.gelu();
        return params.out.forward(h2);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.proj.weight.deinit();
        if (params.proj.bias) |*b| b.deinit();
        params.out.weight.deinit();
        if (params.out.bias) |*b| b.deinit();
    }
};

/// Patchify projection used before transformer blocks.
/// Forward contract for LTX video patchify:
/// [B, T, 128] -> Linear(128->4096)
pub const Patchify = struct {
    pub const Params = zml.nn.Linear;

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const proj_store = store.withPrefix("patchify_proj");
        return .init(
            proj_store.createTensor("weight", .{ .d, .patch }, null),
            proj_store.createTensor("bias", .{.d}, null),
            .patch,
        );
    }

    pub fn forward(self: Patchify, x: Tensor, params: Params) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .patch });
        return params.forward(x_);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.weight.deinit();
        if (params.bias) |*b| b.deinit();
    }
};

pub const AttentionKind = enum {
    attn1,
    attn2,
    audio_attn1,
    audio_attn2,
    audio_to_video_attn,
    video_to_audio_attn,
};

pub const Attention = struct {
    pub const ForwardOpts = struct {
        context: ?Tensor = null,
        mask: ?Tensor = null,
        pe_cos: ?Tensor = null,
        pe_sin: ?Tensor = null,
        k_pe_cos: ?Tensor = null,
        k_pe_sin: ?Tensor = null,
    };

    pub const Params = struct {
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        to_q: zml.nn.Linear,
        to_k: zml.nn.Linear,
        to_v: zml.nn.Linear,
        to_gate_logits: zml.nn.Linear,
        to_out: zml.nn.Linear,
    };

    pub fn initParams(store: zml.io.TensorStore.View, kind: AttentionKind) Params {
        const path = kindStorePath(kind);
        const attn_store = store.withPrefix(path);
        const to_out_store = attn_store.withPrefix("to_out").withLayer(0);

        return .{
            .q_norm_weight = attn_store.withPrefix("q_norm").createTensor("weight", .{.d_q}, null),
            .k_norm_weight = attn_store.withPrefix("k_norm").createTensor("weight", .{.d_k}, null),
            .to_q = .init(
                attn_store.withPrefix("to_q").createTensor("weight", .{ .d_q, .d }, null),
                attn_store.withPrefix("to_q").createTensor("bias", .{.d_q}, null),
                .d,
            ),
            .to_k = .init(
                attn_store.withPrefix("to_k").createTensor("weight", .{ .d_k, .d }, null),
                attn_store.withPrefix("to_k").createTensor("bias", .{.d_k}, null),
                .d,
            ),
            .to_v = .init(
                attn_store.withPrefix("to_v").createTensor("weight", .{ .d_v, .d }, null),
                attn_store.withPrefix("to_v").createTensor("bias", .{.d_v}, null),
                .d,
            ),
            .to_gate_logits = .init(
                attn_store.withPrefix("to_gate_logits").createTensor("weight", .{ .h, .d }, null),
                attn_store.withPrefix("to_gate_logits").createTensor("bias", .{.h}, null),
                .d,
            ),
            .to_out = .init(
                to_out_store.createTensor("weight", .{ .d, .d_v }, null),
                to_out_store.createTensor("bias", .{.d}, null),
                .d_v,
            ),
        };
    }

    pub fn forward(self: Attention, x: Tensor, params: Params, num_heads: usize, opts: ForwardOpts) Tensor {
        _ = self;

        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const context = if (opts.context) |ctx| ctx.withPartialTags(.{ .b, .t, .d }) else x_;
        var q = params.to_q.forward(x_);
        var k = params.to_k.forward(context);
        var v = params.to_v.forward(context);

        // LTX 2.3 attention includes RMSNorms on q and k with learned weights, applied before splitting into heads.
        // In PyTorch, multiplying by the learned weight is done within the `RMSNorm` module after the normalization step. 
        // On the other hand, zml.nn.rmsNorm() returns a normalized tensor without applying the learned weight, so we multiply manually.
        q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
        k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

        // Split embedding dimension into heads
        // `q`: `[B, Tq, heads * head_dim]` --> `qh`: `[B, Q, H, HD]`
        // `k`: `[B, Tk, heads * head_dim]` --> `kh`: `[B, K, H, HD]`
        // `v`: `[B, Tk, heads * head_dim]` --> `vh`: `[B, K, H, HD]`
        var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

        // Apply RoPE if present in checkpoint.
        // Checkpoint tensors come in two layouts: rank-4 [B, H, T, HD] (tagged head-first as [B, H, Q/K, HD])
        // or rank-2 [T, H*HD] (split and tagged token-first as [B, Q/K, H, HD]). Both are equivalent under
        // ZML's named-axis system; applyLtxRotaryEmb handles alignment via transpose().
        // When no k-specific RoPE tensors are present, the q tensors are reused for k.
        if (opts.pe_cos) |pe_cos| {
            if (opts.pe_sin) |pe_sin| {
                const q_cos = if (pe_cos.rank() == 4)
                    pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
                else
                    pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });

                const q_sin = if (pe_sin.rank() == 4)
                    pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
                else
                    pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });

                const k_cos = if (opts.k_pe_cos) |k_pe_cos|
                    if (k_pe_cos.rank() == 4)
                        k_pe_cos.withPartialTags(.{ .b, .h, .k, .hd })
                    else
                        k_pe_cos.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd })
                else
                    q_cos.rename(.{ .q = .k });

                const k_sin = if (opts.k_pe_sin) |k_pe_sin|
                    if (k_pe_sin.rank() == 4)
                        k_pe_sin.withPartialTags(.{ .b, .h, .k, .hd })
                    else
                        k_pe_sin.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd })
                else
                    q_sin.rename(.{ .q = .k });

                // LTX is a video generation model needing coordinates for temporal + spatial (height/width) dimensions, not flat sequential 
                // token positions, so the RoPE implementation is customized but mathematically equivalent to standard RoPE.
                qh = applyLtxRotaryEmb(qh, q_cos, q_sin);
                kh = applyLtxRotaryEmb(kh, k_cos, k_sin);
            }
        }

        qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
        kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
        vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

        // ZML uses named axes. The sdpa primitive expects a specific batch axis name, `.batch`, so it is temporarily renamed here. 
        var attn = zml.nn.sdpa(
            qh.rename(.{ .b = .batch }),
            kh.rename(.{ .b = .batch }),
            vh.rename(.{ .b = .batch }),
            .{ .attn_mask = if (opts.mask) |m| m.rename(.{ .b = .batch }) else null },
        ).rename(.{ .batch = .b }); // [B, Q, H, HD]

        // Compute per-head gates as 2 * sigmoid(logits) so zero-initialized logits preserve identity.
        const gate_logits = params.to_gate_logits.forward(x_); // [B, T, H]
        const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate); // [B, Q, H]
        attn = attn.mul(gate.broad(attn.shape())); // [B, Q, H, HD]

        const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }); // [B, Q, D_V] with D_V = H * HD
        return params.to_out.forward(merged).withPartialTags(.{ .b, .t, .d });
    }

    fn applyLtxRotaryEmb(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_hd = x.dim(.hd);
        const rope_hd = cos.dim(.hd);

        if (rope_hd == x_hd) {
            return applyLtxRotaryEmbInterleaved(x, cos, sin);
        }
        if (rope_hd * 2 == x_hd) {
            return applyLtxRotaryEmbSplit(x, cos, sin);
        }

        std.debug.panic("Unsupported RoPE shapes: x={f} cos={f} sin={f}", .{ x, cos, sin });
    }

    fn applyLtxRotaryEmbInterleaved(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_even, const x_odd = zml.nn.splitRealImg(x, .interleaved);
        const x_rot = zml.nn.mergeRealImg(x_odd.scale(-1.0), x_even, .interleaved);
        const cos_aligned = cos.transpose(x.shape());
        const sin_aligned = sin.transpose(x.shape());
        return x.mul(cos_aligned.broad(x.shape())).add(x_rot.mul(sin_aligned.broad(x.shape())));
    }

    fn applyLtxRotaryEmbSplit(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const half: i64 = @divExact(x.dim(.hd), 2);
        const x_first = x.slice1d(.hd, .{ .start = 0, .end = half });
        const x_second = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });

        const cos_b = cos.transpose(x_first.shape()).broad(x_first.shape());
        const sin_b = sin.transpose(x_first.shape()).broad(x_first.shape());

        const out_first = x_first.mul(cos_b).sub(x_second.mul(sin_b));
        const out_second = x_second.mul(cos_b).add(x_first.mul(sin_b));
        return Tensor.concatenate(&.{ out_first, out_second }, .hd);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.q_norm_weight.deinit();
        params.k_norm_weight.deinit();

        params.to_q.weight.deinit();
        if (params.to_q.bias) |*b| b.deinit();

        params.to_k.weight.deinit();
        if (params.to_k.bias) |*b| b.deinit();

        params.to_v.weight.deinit();
        if (params.to_v.bias) |*b| b.deinit();

        params.to_gate_logits.weight.deinit();
        if (params.to_gate_logits.bias) |*b| b.deinit();

        params.to_out.weight.deinit();
        if (params.to_out.bias) |*b| b.deinit();
    }
};

fn kindStorePath(kind: AttentionKind) []const u8 {
    return switch (kind) {
        .attn1 => "attn1",
        .attn2 => "attn2",
        .audio_attn1 => "audio_attn1",
        .audio_attn2 => "audio_attn2",
        .audio_to_video_attn => "audio_to_video_attn",
        .video_to_audio_attn => "video_to_audio_attn",
    };
}

fn kindNumHeads(kind: AttentionKind) usize {
    return switch (kind) {
        .attn1, .attn2, .audio_attn1, .audio_attn2, .audio_to_video_attn, .video_to_audio_attn => 32,
    };
}

/// One AV transformer block. Shared implementation across stages.
pub const BasicAVTransformerBlock = struct {
    ff: FeedForward,
    attn: Attention,

    pub const Params = struct {
        ff: FeedForward.Params,
        attn: Attention.Params,
    };

    pub fn init() BasicAVTransformerBlock {
        return .{
            .ff = .{},
            .attn = .{},
        };
    }

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        return .{
            .ff = FeedForward.initParams(store.withPrefix("ff")),
            .attn = Attention.initParams(store, .attn1),
        };
    }

    pub fn forward(self: BasicAVTransformerBlock, x: Tensor, params: Params) Tensor {
        _ = self.attn.forward(x, params.attn, kindNumHeads(.attn1), .{});
        // NOTE: residual/gating/adaln wiring belongs at block level and will be added here.
        return self.ff.forward(x, params.ff);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        FeedForward.unloadBuffers(&params.ff);
        Attention.unloadBuffers(&params.attn);
    }
};

/// Velocity model (transformer stack) shared across stages.
pub const LTXModel = struct {
    blocks: []BasicAVTransformerBlock,

    pub const Params = struct {
        blocks: []BasicAVTransformerBlock.Params,

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            allocator.free(self.blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
            for (self.blocks) |*bp| {
                BasicAVTransformerBlock.unloadBuffers(bp);
            }
            allocator.free(self.blocks);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !LTXModel {
        const blocks = try allocator.alloc(BasicAVTransformerBlock, config.num_transformer_blocks);
        for (blocks) |*b| b.* = BasicAVTransformerBlock.init();

        return .{ .blocks = blocks };
    }

    pub fn deinit(self: *LTXModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const blocks = try allocator.alloc(BasicAVTransformerBlock.Params, config.num_transformer_blocks);

        const blocks_store = store.withPrefix("transformer_blocks");
        for (blocks, 0..) |*bp, i| {
            bp.* = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
        }

        return .{ .blocks = blocks };
    }

    pub fn forward(self: LTXModel, x: Tensor, params: Params) Tensor {
        std.debug.assert(self.blocks.len == params.blocks.len);

        var h = x;
        for (self.blocks, params.blocks) |block, block_params| {
            h = block.forward(h, block_params);
        }
        return h;
    }
};

/// Top-level wrapper used by the LTX pipeline.
pub const X0Model = struct {
    velocity_model: LTXModel,

    pub const Params = struct {
        velocity_model: LTXModel.Params,

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            self.velocity_model.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !X0Model {
        return .{
            .velocity_model = try LTXModel.init(allocator, config),
        };
    }

    pub fn deinit(self: *X0Model, allocator: std.mem.Allocator) void {
        self.velocity_model.deinit(allocator);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const root = selectTransformerRoot(store);
        return .{
            .velocity_model = try LTXModel.initParams(allocator, root, config),
        };
    }

    pub fn forward(self: X0Model, x: Tensor, params: Params) Tensor {
        return self.velocity_model.forward(x, params.velocity_model);
    }

    /// Focused entrypoint used for FF parity bring-up.
    pub fn forwardBlock0FF(_: X0Model, x: Tensor, params: Params) Tensor {
        const ff = FeedForward{};
        return ff.forward(x, params.velocity_model.blocks[0].ff);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
        LTXModel.Params.unloadBuffers(&params.velocity_model, allocator);
    }
};

/// Active runtime parameter set (single stage at a time).
/// This keeps code shared while allowing atomic stage-by-stage validation.
pub const ActiveParams = struct {
    stage: Stage,
    params: X0Model.Params,

    pub fn deinit(self: *ActiveParams, allocator: std.mem.Allocator) void {
        self.params.deinit(allocator);
    }
};

/// Build one active stage-specific parameter set from one checkpoint/store.
pub fn initActiveParams(
    allocator: std.mem.Allocator,
    stage: Stage,
    store: zml.io.TensorStore.View,
    config: Config,
) !ActiveParams {
    return .{
        .stage = stage,
        .params = try X0Model.initParams(allocator, store, config),
    };
}

fn selectTransformerRoot(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("model.velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("velocity_model");
    }
    if (store.hasKey("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("diffusion_model");
    }
    if (store.hasKey("velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("velocity_model");
    }
    if (store.hasKey("diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("diffusion_model");
    }
    // Keep previous behavior as a fallback.
    return store.withPrefix("velocity_model");
}

/// Free-function entrypoint for parity tooling: block0 FF only.
pub fn forwardBlock0FF(x: Tensor, params: X0Model.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params.velocity_model.blocks[0].ff);
}

/// Focused FF-only entrypoint for parity bring-up.
pub fn forwardFF(x: Tensor, params: FeedForward.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params);
}

/// Focused FF net.0 (Linear 4096->16384) entrypoint.
pub fn forwardFFLinear1(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    return params.proj.forward(x_);
}

/// Focused FF net.2 (Linear 16384->4096) entrypoint.
pub fn forwardFFLinear2(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d_ff });
    return params.out.forward(x_);
}

/// Focused GELU entrypoint matching current model behavior.
pub fn forwardFFGeluBf16(x: Tensor) Tensor {
    return x.gelu();
}

/// Focused GELU entrypoint that upcasts to f32 for GELU math then casts back.
pub fn forwardFFGeluF32(x: Tensor) Tensor {
    const dt = x.dtype();
    return x.convert(.f32).gelu().convert(dt);
}

/// Focused composed path: ff.net.0 projection followed by GELU.
pub fn forwardFFLinear1Gelu(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const h1 = params.proj.forward(x_);
    return h1.gelu();
}

/// Build only block0.ff params from the selected transformer root.
pub fn initBlock0FFParams(store: zml.io.TensorStore.View) FeedForward.Params {
    const root = selectTransformerRoot(store);
    return FeedForward.initParams(root.withPrefix("transformer_blocks").withLayer(0).withPrefix("ff"));
}

pub fn unloadBlock0FFBuffers(params: *zml.Bufferized(FeedForward.Params)) void {
    FeedForward.unloadBuffers(params);
}

/// Build patchify params from the selected transformer root.
pub fn initPatchifyParams(store: zml.io.TensorStore.View) Patchify.Params {
    const root = selectTransformerRoot(store);
    return Patchify.initParams(root);
}

/// Focused patchify-only entrypoint for parity bring-up.
pub fn forwardPatchify(x: Tensor, params: Patchify.Params) Tensor {
    const patchify = Patchify{};
    return patchify.forward(x, params);
}

pub fn unloadPatchifyBuffers(params: *zml.Bufferized(Patchify.Params)) void {
    Patchify.unloadBuffers(params);
}

/// Build audio patchify params from the selected transformer root.
pub fn initAudioPatchifyParams(store: zml.io.TensorStore.View) Patchify.Params {
    const root = selectTransformerRoot(store);
    return .init(
        root.withPrefix("audio_patchify_proj").createTensor("weight", .{ .d, .patch }, null),
        root.withPrefix("audio_patchify_proj").createTensor("bias", .{.d}, null),
        .patch,
    );
}

pub fn initBlock0AttentionParams(store: zml.io.TensorStore.View, kind: AttentionKind) Attention.Params {
    const root = selectTransformerRoot(store);
    return Attention.initParams(root.withPrefix("transformer_blocks").withLayer(0), kind);
}

pub fn unloadBlock0AttentionBuffers(params: *zml.Bufferized(Attention.Params)) void {
    Attention.unloadBuffers(params);
}

pub fn forwardBlock0Attn1(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{});
}

pub fn forwardBlock0Attn1WithPeCosSin(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin });
}

pub fn forwardBlock0Attn1WithPeCosSinMask(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, mask: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin, .mask = mask });
}

/// Diagnostic: returns q after to_q projection + RMS norm.
/// Output shape: [B, T, D_Q] — identical layout to Python's q_norm_diag0.
pub fn forwardBlock0Attn1DiagQNorm(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    return zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
}

/// Diagnostic: returns q after projection (pre-RMSNorm).
/// Output shape: [B, T, D_Q].
pub fn forwardBlock0Attn1DiagQProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    return q.withPartialTags(.{ .b, .t, .d_q });
}

/// Diagnostic: q projection with f32 accumulation, cast back to input dtype.
/// Output shape: [B, T, D_Q].
pub fn forwardBlock0Attn1DiagQProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_q.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d);
    if (params.to_q.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(x.dtype()).withPartialTags(.{ .b, .t, .d_q });
}

/// Diagnostic: returns qh after projection + norm + head-split + RoPE.
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagQRot(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    var qh = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    qh = qh.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    return qh.withPartialTags(.{ .b, .q, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns kh after projection + norm + head-split + RoPE.
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagKRot(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const k = params.to_k.forward(x_);
    var kh = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));
    kh = kh.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const k_cos = (if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd })).rename(.{ .q = .k });
    const k_sin = (if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd })).rename(.{ .q = .k });

    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);
    return kh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns k after projection (pre-RMSNorm).
/// Output shape: [B, T, D_K].
pub fn forwardBlock0Attn1DiagKProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const k = params.to_k.forward(x_);
    return k.withPartialTags(.{ .b, .t, .d_k });
}

/// Diagnostic: k projection with f32 accumulation, cast back to input dtype.
/// Output shape: [B, T, D_K].
pub fn forwardBlock0Attn1DiagKProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_k.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d);
    if (params.to_k.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(x.dtype()).withPartialTags(.{ .b, .t, .d_k });
}

/// Diagnostic: returns vh after projection + head-split (no RoPE for V).
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagVProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const v = params.to_v.forward(x_);
    return v.withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic helper: split a precomputed V projection [B, T, Dv] into heads.
/// Output shape: [B, T, H, HD].
pub fn forwardBlock0Attn1DiagVHeadFromProj(v_proj: Tensor) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const v = v_proj.withPartialTags(.{ .b, .t, .d_v });
    const vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd });
}

/// Diagnostic: returns vh after projection + head-split (no RoPE for V).
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagVHead(x: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const v = params.to_v.forward(x_);
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: V projection using f32 accumulation (precision-corrected variant).
/// This upcasts context to f32 before matmul for accumulated precision, then downcasts back.
/// Returns: [B, T, Dv]
pub fn forwardBlock0Attn1DiagVProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    // Upcast to f32 for matmul accumulation
    const x_f32 = x_.convert(.f32);
    // Weights must be upcast too; assume weight stored in same dtype as bias (typically bf16)
    const weight_f32 = params.to_v.weight.convert(.f32);
    // Perform matmul in f32
    var y_f32 = x_f32.dot(weight_f32, .d);
    // Add bias if present
    if (params.to_v.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    // Downcast back to original dtype
    const v = y_f32.convert(x.dtype());
    return v.withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic: V projection head-split using f32 accumulation (precision-corrected variant).
/// Returns: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagVHeadF32(x: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const v_f32 = forwardBlock0Attn1DiagVProjF32(x, params);
    var vh = v_f32.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns raw SDPA output before gate and before merge/to_out.
/// Output shape: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagSdpaOut(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd });
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    const attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    return attn.withPartialTags(.{ .b, .q, .h, .hd });
}

/// Diagnostic: returns raw SDPA output computed manually in f32 (before gate/merge).
/// Output shape: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagSdpaOutManualF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const out_dtype = x.dtype();
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var scores = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = scores.dot(vh, .k);
    attn = attn.transpose(qh.shape()).convert(out_dtype);
    return attn.withPartialTags(.{ .b, .q, .h, .hd });
}

/// Diagnostic: returns gate logits before sigmoid/scale.
/// Output shape: [B, T, H]
pub fn forwardBlock0Attn1DiagGateLogits(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    return params.to_gate_logits.forward(x_).withPartialTags(.{ .b, .t, .h });
}

/// Diagnostic: applies only to_out projection on a pre_to_out tensor.
/// Input shape: [B, T, D_V], output shape: [B, T, D].
pub fn forwardBlock0Attn1DiagToOutOnly(pre_to_out: Tensor, params: Attention.Params) Tensor {
    const x_ = pre_to_out.withPartialTags(.{ .b, .t, .d_v });
    return params.to_out.forward(x_).withPartialTags(.{ .b, .t, .d });
}


/// Diagnostic: to_out projection with f32 accumulation, cast back to input dtype.
/// Used to isolate whether PyTorch uses f32 intermediate accumulation in to_out.
/// Input shape: [B, T, D_V], output shape: [B, T, D].
pub fn forwardBlock0Attn1DiagToOutOnlyF32(pre_to_out: Tensor, params: Attention.Params) Tensor {
    const x_ = pre_to_out.withPartialTags(.{ .b, .t, .d_v });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_out.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d_v);
    if (params.to_out.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(pre_to_out.dtype()).withPartialTags(.{ .b, .t, .d });
}

/// Diagnostic: full attn1 path in f32 for all linear/attention intermediates, cast back at output.
/// This isolates the best-case gain obtainable from precision-only changes.
pub fn forwardBlock0Attn1DiagAllF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const out_dtype = x.dtype();
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);

    var q = x_f32.dot(params.to_q.weight.convert(.f32), .d);
    if (params.to_q.bias) |bias_orig| {
        q = q.add(bias_orig.convert(.f32).broad(q.shape()));
    }

    var k = x_f32.dot(params.to_k.weight.convert(.f32), .d);
    if (params.to_k.bias) |bias_orig| {
        k = k.add(bias_orig.convert(.f32).broad(k.shape()));
    }

    var v = x_f32.dot(params.to_v.weight.convert(.f32), .d);
    if (params.to_v.bias) |bias_orig| {
        v = v.add(bias_orig.convert(.f32).broad(v.shape()));
    }

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.convert(.f32).broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.convert(.f32).broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    var gate_logits = x_f32.dot(params.to_gate_logits.weight.convert(.f32), .d);
    if (params.to_gate_logits.bias) |bias_orig| {
        gate_logits = gate_logits.add(bias_orig.convert(.f32).broad(gate_logits.shape()));
    }
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t });
    var out_f32 = merged.dot(params.to_out.weight.convert(.f32), .d_v);
    if (params.to_out.bias) |bias_orig| {
        out_f32 = out_f32.add(bias_orig.convert(.f32).broad(out_f32.shape()));
    }

    return out_f32.convert(out_dtype).withPartialTags(.{ .b, .t, .d });
}

/// Diagnostic: returns merged attention after SDPA and gate, before to_out.
/// Output shape: [B, T, D_V]
pub fn forwardBlock0Attn1DiagPreToOut(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: same as forwardBlock0Attn1DiagPreToOut, but runs SDPA core in f32.
/// This helps isolate whether bf16 attention precision is the source of parity drift.
pub fn forwardBlock0Attn1DiagPreToOutF32Sdpa(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    var gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: pre-to_out with no gate applied.
pub fn forwardBlock0Attn1DiagPreToOutNoGate(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    const attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: pre-to_out with sigmoid(gate_logits) gate (without *2 scaling).
pub fn forwardBlock0Attn1DiagPreToOutSigmoidGate(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: explicit SDPA math (f32) instead of zml.nn.sdpa.
/// This isolates potential semantic differences inside the shared sdpa primitive.
pub fn forwardBlock0Attn1DiagPreToOutManualSdpaF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var attn_weights = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = attn_weights.dot(vh, .k);
    attn = attn.transpose(qh.shape());

    const gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: run sdpa with head-first layout [b,h,q,hd] like PyTorch,
/// then swap back to [b,q,h,hd] before merge.
pub fn forwardBlock0Attn1DiagPreToOutHeadFirstSdpa(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).swapAxes(.q, .h);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).swapAxes(.k, .h);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).swapAxes(.k, .h);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    // Bring back to [b,q,h,hd] so gating + merge match the main pipeline.
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: same as pre_to_out with sigmoid gate, but flatten heads as [hd, h].
/// Uses mergeTranspose to safely merge non-contiguous axes.
pub fn forwardBlock0Attn1DiagPreToOutAltMergeTranspose(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.mergeTranspose(.{ .hd, .h }, .d_v).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: head-first layout [b,h,q,hd] with native zml.nn.sdpa only
/// (no manual SDPA, no f32 accumulation). Tests if layout alone explains the gap.
pub fn forwardBlock0Attn1DiagPreToOutHeadFirstOnly(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd });
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    // Reorder to head-first [b,h,q,hd] before SDPA
    qh = qh.swapAxes(.q, .h);
    kh = kh.swapAxes(.k, .h);
    vh = vh.swapAxes(.k, .h);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    // Reorder back to [b,q,h,hd] for gating + merge
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: manual SDPA f32 logic with head-first layout [b,h,q,hd].
/// Combines both fixes: explicit scaling/softmax in f32 + PyTorch head-first layout.
pub fn forwardBlock0Attn1DiagPreToOutManualSdpaHeadFirst(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    // Reorder to head-first [b,h,q,hd] before manual SDPA
    qh = qh.swapAxes(.q, .h);
    kh = kh.swapAxes(.k, .h);
    vh = vh.swapAxes(.k, .h);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var attn_weights = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = attn_weights.dot(vh, .k);

    // Reorder back to [b,q,h,hd] for gating + merge
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

pub fn forwardBlock0Attn2(x: Tensor, context: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn2), .{ .context = context });
}

pub fn forwardBlock0AudioAttn1(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn1), .{});
}

pub fn forwardBlock0AudioAttn1WithPeCosSin(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin });
}

pub fn forwardBlock0AudioAttn2(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn2), .{});
}

pub fn forwardBlock0AudioAttn2WithContext(x: Tensor, context: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn2), .{ .context = context });
}

pub fn forwardBlock0AudioToVideoAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_to_video_attn), .{});
}

pub fn forwardBlock0AudioToVideoAttnWithContextPeKPe(x: Tensor, context: Tensor, pe_cos: Tensor, pe_sin: Tensor, k_pe_cos: Tensor, k_pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_to_video_attn), .{
        .context = context,
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
        .k_pe_cos = k_pe_cos,
        .k_pe_sin = k_pe_sin,
    });
}

pub fn forwardBlock0VideoToAudioAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.video_to_audio_attn), .{});
}

pub fn forwardBlock0VideoToAudioAttnWithContextPeKPe(x: Tensor, context: Tensor, pe_cos: Tensor, pe_sin: Tensor, k_pe_cos: Tensor, k_pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.video_to_audio_attn), .{
        .context = context,
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
        .k_pe_cos = k_pe_cos,
        .k_pe_sin = k_pe_sin,
    });
}
