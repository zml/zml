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

    pub fn forward(self: Attention, x: Tensor, params: Params, num_heads: usize) Tensor {
        _ = self;

        const x_ = x.withPartialTags(.{ .b, .t, .d });
        var q = params.to_q.forward(x_);
        var k = params.to_k.forward(x_);
        var v = params.to_v.forward(x_);

        q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
        k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

        var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

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

        const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t });
        return params.to_out.forward(merged).withPartialTags(.{ .b, .t, .d });
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
        _ = self.attn.forward(x, params.attn, kindNumHeads(.attn1));
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
    return attn.forward(x, params, kindNumHeads(.attn1));
}

pub fn forwardBlock0Attn2(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn2));
}

pub fn forwardBlock0AudioAttn1(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn1));
}

pub fn forwardBlock0AudioAttn2(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn2));
}

pub fn forwardBlock0AudioToVideoAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_to_video_attn));
}

pub fn forwardBlock0VideoToAudioAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.video_to_audio_attn));
}
