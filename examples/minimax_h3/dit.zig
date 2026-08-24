const std = @import("std");

const zml = @import("zml");

const config_mod = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

pub const Config = config_mod.Config;

pub const Buffers = zml.Bufferized(Model);

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", tagz, .replicated),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor, axis: anytype) zml.Tensor {
        const normalized = zml.nn.rmsNorm(input, axis, self.eps);
        return normalized.mul(self.weight.convert(input.dtype()).broad(input.shape()));
    }
};

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
    return .init(
        store.createTensor(weight_name, .{ .dout, .d }, partitions),
        if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, bias_partitions) else null,
        .d,
    );
}

fn unloadLinear(lin: *zml.Bufferized(zml.nn.Linear)) void {
    lin.weight.deinit();
    if (lin.bias) |*bias| bias.deinit();
}

const SwiGlu = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    /// Official `mlp.fc1` is `[gate; value]`. Diffusers `ff.net.0.proj` is `[value; gate]`.
    gate_first: bool,

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        if (store.hasKey("fc1.weight")) {
            return .{
                .fc1 = linear(store, "fc1.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
                .fc2 = linear(store, "fc2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
                .gate_first = true,
            };
        }
        return .{
            .fc1 = linear(store, "net.0.proj.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
            .fc2 = linear(store, "net.2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            .gate_first = false,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
        unloadLinear(&self.fc1);
        unloadLinear(&self.fc2);
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const uv = self.fc1.forward(x);
        const first, const second = uv.chunkExact(-1, 2);
        const gated = if (self.gate_first) first.silu().mul(second) else second.silu().mul(first);
        return self.fc2.forward(gated.rename(.{ .dout = .d }));
    }
};

const Attention = struct {
    qkv: ?zml.nn.Linear,
    to_q: ?zml.nn.Linear,
    to_k: ?zml.nn.Linear,
    to_v: ?zml.nn.Linear,
    out: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
        const qkv_part = .{ .dout = .model, .d = .replicated };
        const out_part = .{ .dout = .replicated, .d = .model };
        const fused = store.hasKey("qkv_proj.weight");
        return .{
            .qkv = if (fused) linear(store, "qkv_proj.weight", null, qkv_part, .replicated) else null,
            .to_q = if (!fused) linear(store, if (store.hasKey("to_q.weight")) "to_q.weight" else "q_proj.weight", null, qkv_part, .replicated) else null,
            .to_k = if (!fused) linear(store, if (store.hasKey("to_k.weight")) "to_k.weight" else "k_proj.weight", null, qkv_part, .replicated) else null,
            .to_v = if (!fused) linear(store, if (store.hasKey("to_v.weight")) "to_v.weight" else "v_proj.weight", null, qkv_part, .replicated) else null,
            .out = linear(store, if (store.hasKey("out_proj.weight")) "out_proj.weight" else if (store.hasKey("to_out.0.weight")) "to_out.0.weight" else "o_proj.weight", null, out_part, .replicated),
            .q_norm = .init(store.withPrefix(if (store.hasKey("q_norm.weight")) "q_norm" else "norm_q"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = .init(store.withPrefix(if (store.hasKey("k_norm.weight")) "k_norm" else "norm_k"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        if (self.qkv) |*qkv| unloadLinear(qkv);
        if (self.to_q) |*q| unloadLinear(q);
        if (self.to_k) |*k| unloadLinear(k);
        if (self.to_v) |*v| unloadLinear(v);
        unloadLinear(&self.out);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q: zml.Tensor = undefined;
        var k: zml.Tensor = undefined;
        var v: zml.Tensor = undefined;
        if (self.qkv) |qkv| {
            const split = qkv.forward(x_qkv).splitAxis(.dout, .{ .p = 3, .h = self.num_heads, .hd = self.head_dim })
                .withPartitioning(.{ .h = .model });
            const parts = split.chunkExact(.p, 3);
            q = parts[0].squeeze(.p);
            k = parts[1].squeeze(.p);
            v = parts[2].squeeze(.p);
        } else {
            q = self.to_q.?.forward(x_qkv).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
            k = self.to_k.?.forward(x_qkv).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
            v = self.to_v.?.forward(x_qkv).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        }

        q = self.q_norm.forward(q, .hd);
        k = self.k_norm.forward(k, .hd);
        if (rotary) |pe| {
            q = applyRotary(q, pe[0], pe[1]);
            k = applyRotary(k, pe[0], pe[1]);
        }
        const attn = zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{},
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        return self.out.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

fn rotateHalf(x: zml.Tensor) zml.Tensor {
    const half = @divExact(x.dim(-1), 2);
    const x1 = x.slice1d(-1, .{ .start = 0, .end = half });
    const x2 = x.slice1d(-1, .{ .start = half, .end = x.dim(-1) });
    return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
}

fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const rotary_dim = cos.dim(-1);
    const x_rot = x.slice1d(-1, .{ .start = 0, .end = rotary_dim });
    const x_pass = x.slice1d(-1, .{ .start = rotary_dim, .end = x.dim(-1) });
    const cos_x = cos.rename(.{ .f = .hd }).broad(x_rot.shape());
    const sin_x = sin.rename(.{ .f = .hd }).broad(x_rot.shape());
    const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));
    return zml.Tensor.concatenate(&.{ rotated, x_pass }, -1);
}

pub fn mmRope(position_ids: zml.Tensor, rope_freq_dim: i64, rope_theta: f32) struct { zml.Tensor, zml.Tensor } {
    const pos = position_ids.convert(.f32).withPartialTags(.{ .s, .ax });
    const inv = zml.nn.invFreq(2 * rope_freq_dim, .{
        .layout = .real_im_pass,
        .scaling = .{ .default = .{ .rope_theta = rope_theta } },
    }).withTags(.{.f});
    const freqs = pos.outer(inv);
    const parts = freqs.chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    const emb = zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
    return .{ emb.cos(), emb.sin() };
}

const TimeEmbedder = struct {
    proj_in: zml.nn.Linear,
    proj_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) TimeEmbedder {
        const prefix = store.withPrefix("time_embedder");
        if (prefix.hasKey("proj_in.weight")) {
            return .{
                .proj_in = linear(prefix, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
                .proj_out = linear(prefix, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
            };
        }
        return .{
            .proj_in = linear(prefix, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .proj_out = linear(prefix, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TimeEmbedder)) void {
        unloadLinear(&self.proj_in);
        unloadLinear(&self.proj_out);
    }

    pub fn forward(self: TimeEmbedder, features: zml.Tensor) zml.Tensor {
        return self.proj_out.forward(self.proj_in.forward(features).silu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

pub fn timestepFeatures(t: zml.Tensor, dim: i64) zml.Tensor {
    const inv = zml.nn.invFreq(dim, .{
        .layout = .real_im_pass,
        .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
    }).withTags(.{.f});
    const angles = t.convert(.f32).withPartialTags(.{.n}).outer(inv);
    return zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
}

const AdaLn = struct {
    linear: zml.nn.Linear,
    hidden_size: i64,
    expand: i64,
    modalities: i64,

    pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        const prefix = if (store.hasKey("linear.weight")) store else store.withPrefix("linear");
        return .{
            .linear = linear(prefix, "linear.weight", "linear.bias", .replicated, .replicated),
            .hidden_size = hidden_size,
            .expand = expand,
            .modalities = modalities,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AdaLn)) void {
        unloadLinear(&self.linear);
    }

    pub fn forward(self: AdaLn, temb: zml.Tensor) zml.Tensor {
        const raw = self.linear.forward(temb.silu().convert(self.linear.weight.dtype()));
        if (self.modalities == 1) {
            return raw.splitAxis(.dout, .{ .k = self.expand, .d = self.hidden_size });
        }
        return raw.splitAxis(.dout, .{
            .mod = self.modalities,
            .k = self.expand,
            .d = self.hidden_size,
        });
    }
};

pub const TransformerBlock = struct {
    norm1: RmsNorm,
    attn: Attention,
    norm2: RmsNorm,
    mlp: SwiGlu,
    adaln: AdaLn,
    hidden_size: i64,

    pub const Input = struct {
        layer: TransformerBlock,
        hidden: zml.Tensor,
        temb: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TransformerBlock {
        const attn_store = if (store.hasKey("attn.qkv_proj.weight") or store.hasKey("attn.to_q.weight") or store.hasKey("attn.q_proj.weight"))
            store.withPrefix("attn")
        else
            store;
        const mlp_store = if (store.hasKey("mlp.fc1.weight")) store.withPrefix("mlp") else if (store.hasKey("ff.net.0.proj.weight")) store.withPrefix("ff") else store.withPrefix("mlp");
        const adaln_store = store.withPrefix("adaln_proj");
        return .{
            .norm1 = .init(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(attn_store, cfg),
            .norm2 = .init(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(mlp_store),
            .adaln = .init(adaln_store, cfg.hidden_size, 6, config_mod.modality_count),
            .hidden_size = cfg.hidden_size,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
        RmsNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        RmsNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.mlp);
        AdaLn.unloadBuffers(&self.adaln);
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const table = self.adaln.forward(input.temb);
        const mods = table.merge(.{ .n = .{ .n, .mod } });
        const selected = mods.gather(.{ .n = input.adaln_indices }, .{});
        const parts = selected.chunkExact(.k, 6);
        const shift_msa = parts[0].squeeze(.k);
        const scale_msa = parts[1].squeeze(.k);
        const gate_msa = parts[2].squeeze(.k);
        const shift_mlp = parts[3].squeeze(.k);
        const scale_mlp = parts[4].squeeze(.k);
        const gate_mlp = parts[5].squeeze(.k);

        const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
        const n1 = self.norm1.forward(residual0, .d);
        const one = zml.Tensor.scalar(1.0, n1.dtype());
        const attn_in = n1.mul(one.add(scale_msa.convert(n1.dtype()).broad(n1.shape()))).add(shift_msa.convert(n1.dtype()).broad(n1.shape()));
        const attn_out = self.attn.forward(attn_in, .{ input.cos, input.sin });
        const x1 = residual0.add(gate_msa.convert(attn_out.dtype()).broad(attn_out.shape()).mul(attn_out)).withPartitioning(.{ .d = .replicated });

        const n2 = self.norm2.forward(x1, .d);
        const mlp_in = n2.mul(one.add(scale_mlp.convert(n2.dtype()).broad(n2.shape()))).add(shift_mlp.convert(n2.dtype()).broad(n2.shape()));
        const mlp_out = self.mlp.forward(mlp_in).rename(.{ .dout = .d });
        const x2 = x1.add(gate_mlp.convert(mlp_out.dtype()).broad(mlp_out.shape()).mul(mlp_out)).withPartitioning(.{ .d = .replicated });
        return .{ .hidden = x2.reuseBuffer(input.hidden) };
    }
};

const TokenRefinerBlock = struct {
    norm1: RmsNorm,
    attn: Attention,
    norm2: RmsNorm,
    mlp: SwiGlu,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TokenRefinerBlock {
        const attn_store = store.withPrefix("attn");
        const mlp_store = if (store.hasKey("mlp.fc1.weight")) store.withPrefix("mlp") else store.withPrefix("ff");
        return .{
            .norm1 = .init(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(attn_store, cfg),
            .norm2 = .init(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(mlp_store),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenRefinerBlock)) void {
        RmsNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        RmsNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: TokenRefinerBlock, x: zml.Tensor) zml.Tensor {
        const residual = x.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual, .d), null));
        return x1.add(self.mlp.forward(self.norm2.forward(x1, .d)).rename(.{ .dout = .d })).withPartitioning(.{ .d = .replicated }).reuseBuffer(x);
    }
};

const TokenRefiner = struct {
    blocks: []TokenRefinerBlock,
    final_norm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !TokenRefiner {
        const block_store = if (store.hasKey("blocks.0.norm1.weight")) store.withPrefix("blocks") else store.withPrefix("refiner_blocks");
        const blocks = try allocator.alloc(TokenRefinerBlock, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, i| {
            block.* = .init(block_store.withLayer(i), cfg);
        }
        return .{
            .blocks = blocks,
            .final_norm = .init(store.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
        };
    }

    pub fn deinit(self: TokenRefiner, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenRefiner), allocator: std.mem.Allocator) void {
        for (self.blocks) |*block| TokenRefinerBlock.unloadBuffers(block);
        allocator.free(self.blocks);
        RmsNorm.unloadBuffers(&self.final_norm);
    }

    pub fn forward(self: TokenRefiner, x: zml.Tensor) zml.Tensor {
        var hidden = x;
        for (self.blocks) |block| {
            hidden = block.forward(hidden);
        }
        return self.final_norm.forward(hidden, .d);
    }
};

const FinalLayer = struct {
    norm: RmsNorm,
    adaln: AdaLn,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) FinalLayer {
        return .{
            .norm = .init(store.withPrefix("norm"), .{.d}, cfg.final_norm_eps),
            .adaln = .init(store.withPrefix("adaln_proj"), cfg.hidden_size, 2, 1),
            .video_out = linear(store, "video_out.weight", "video_out.bias", .replicated, .replicated),
            .audio_out = linear(store, "audio_out.weight", "audio_out.bias", .replicated, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FinalLayer)) void {
        RmsNorm.unloadBuffers(&self.norm);
        AdaLn.unloadBuffers(&self.adaln);
        unloadLinear(&self.video_out);
        unloadLinear(&self.audio_out);
    }
};

pub const Model = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    condition_proj: zml.nn.Linear,
    time_embedder: TimeEmbedder,
    token_refiner: TokenRefiner,
    blocks: []TransformerBlock,
    final_layer: FinalLayer,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
        const store = rootView(store_);
        const blocks_store = if (store.hasKey("blocks.0.norm1.weight")) store.withPrefix("blocks") else store.withPrefix("transformer_blocks");
        const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.num_layers));
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, i| {
            block.* = .init(blocks_store.withLayer(i), cfg);
        }

        const video_name, const video_bias = if (store.hasKey("video_patch_proj.weight"))
            .{ "video_patch_proj.weight", "video_patch_proj.bias" }
        else
            .{ "proj_in.weight", "proj_in.bias" };
        const audio_name, const audio_bias = if (store.hasKey("audio_patch_proj.weight"))
            .{ "audio_patch_proj.weight", "audio_patch_proj.bias" }
        else
            .{ "audio_proj_in.weight", "audio_proj_in.bias" };
        const cond_name, const cond_bias = if (store.hasKey("condition_proj.weight"))
            .{ "condition_proj.weight", "condition_proj.bias" }
        else
            .{ "context_embedder.weight", "context_embedder.bias" };

        const token_refiner = try TokenRefiner.init(allocator, store.withPrefix("token_refiner"), cfg);
        errdefer token_refiner.deinit(allocator);

        const final_layer = if (store.hasKey("final_layer.norm.weight") or store.hasKey("final_layer.adaln_proj.linear.weight") or store.hasKey("final_layer.video_out.weight"))
            FinalLayer.init(store.withPrefix("final_layer"), cfg)
        else
            FinalLayer{
                .norm = .init(store.withPrefix("norm_out"), .{.d}, cfg.final_norm_eps),
                .adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size, 2, 1),
                .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
                .audio_out = linear(store, "audio_proj_out.weight", "audio_proj_out.bias", .replicated, .replicated),
            };

        return .{
            .video_proj = linear(store, video_name, video_bias, .replicated, .replicated),
            .audio_proj = linear(store, audio_name, audio_bias, .replicated, .replicated),
            .condition_proj = linear(store, cond_name, cond_bias, .replicated, .replicated),
            .time_embedder = .init(store),
            .token_refiner = token_refiner,
            .blocks = blocks,
            .final_layer = final_layer,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.token_refiner.deinit(allocator);
        allocator.free(self.blocks);
    }

    pub fn unloadBuffers(self: *const Model, buffers: *Buffers, allocator: std.mem.Allocator) void {
        unloadLinear(&buffers.video_proj);
        unloadLinear(&buffers.audio_proj);
        unloadLinear(&buffers.condition_proj);
        TimeEmbedder.unloadBuffers(&buffers.time_embedder);
        TokenRefiner.unloadBuffers(&buffers.token_refiner, allocator);
        for (buffers.blocks) |*block| TransformerBlock.unloadBuffers(block);
        allocator.free(buffers.blocks);
        FinalLayer.unloadBuffers(&buffers.final_layer);
        _ = self;
    }

    pub fn embedPart(self: Model) EmbedModel {
        return .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
            .condition_proj = self.condition_proj,
            .time_embedder = self.time_embedder,
            .token_refiner = self.token_refiner,
            .cfg = self.cfg,
        };
    }

    pub fn finishPart(self: Model) FinishModel {
        return .{
            .final_layer = self.final_layer,
            .cfg = self.cfg,
        };
    }
};

pub const EmbedModel = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    condition_proj: zml.nn.Linear,
    time_embedder: TimeEmbedder,
    token_refiner: TokenRefiner,
    cfg: Config,

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedModel), allocator: std.mem.Allocator) void {
        unloadLinear(&self.video_proj);
        unloadLinear(&self.audio_proj);
        unloadLinear(&self.condition_proj);
        TimeEmbedder.unloadBuffers(&self.time_embedder);
        TokenRefiner.unloadBuffers(&self.token_refiner, allocator);
    }
};

pub const FinishModel = struct {
    final_layer: FinalLayer,
    cfg: Config,

    pub fn unloadBuffers(self: *zml.Bufferized(FinishModel)) void {
        FinalLayer.unloadBuffers(&self.final_layer);
    }
};

fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("video_patch_proj.weight") or store.hasKey("proj_in.weight") or store.hasKey("blocks.0.norm1.weight")) return store;
    if (store.hasKey("transformer.video_patch_proj.weight")) return store.withPrefix("transformer");
    if (store.hasKey("model.diffusion_model.video_patch_proj.weight")) return store.withPrefix("model.diffusion_model");
    return store;
}

pub const EmbedInput = struct {
    model: EmbedModel,
    video: zml.Tensor,
    audio: zml.Tensor,
    text: zml.Tensor,
    timestep: zml.Tensor,
    position_ids: zml.Tensor,
    video_indices: zml.Tensor,
    audio_indices: zml.Tensor,
    text_indices: zml.Tensor,
};

pub const EmbedOutput = struct {
    hidden: zml.Tensor,
    temb: zml.Tensor,
    cos: zml.Tensor,
    sin: zml.Tensor,
};

pub fn embed(input: EmbedInput) EmbedOutput {
    const self = input.model;
    const video = self.video_proj.forward(input.video.convert(self.video_proj.weight.dtype())).rename(.{ .dout = .d });
    const audio = self.audio_proj.forward(input.audio.convert(self.audio_proj.weight.dtype())).rename(.{ .dout = .d });
    var text = self.condition_proj.forward(input.text.convert(self.condition_proj.weight.dtype())).rename(.{ .dout = .d });
    text = self.token_refiner.forward(text.convert(self.token_refiner.final_norm.weight.dtype()));

    const seq = input.position_ids.withTags(.{ .s, .ax }).dim(.s);
    const batch = text.dim(.b);
    var hidden = zml.Tensor.zeroes(zml.Shape.init(.{ .b = batch, .s = seq, .d = self.cfg.hidden_size }, text.dtype()));
    hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
    hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
    hidden = hidden.scatterSlices(.{ .s = input.audio_indices.withTags(.{.s}) }, audio.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
    hidden = hidden.withPartitioning(.{ .d = .replicated });

    const features = timestepFeatures(input.timestep, self.cfg.freq_dim);
    const temb = self.time_embedder.forward(features);
    const cos, const sin = mmRope(input.position_ids, self.cfg.rope_freq_dim, self.cfg.rope_theta);
    return .{
        .hidden = hidden,
        .temb = temb,
        .cos = cos.convert(text.dtype()),
        .sin = sin.convert(text.dtype()),
    };
}

pub const FinishInput = struct {
    model: FinishModel,
    hidden: zml.Tensor,
    temb: zml.Tensor,
    timestep_indices: zml.Tensor,
    video_indices: zml.Tensor,
    audio_indices: zml.Tensor,
};

pub const FinishOutput = struct {
    video: zml.Tensor,
    audio: zml.Tensor,
};

pub fn finish(input: FinishInput) FinishOutput {
    const self = input.model;
    const hidden = input.hidden.withPartitioning(.{ .d = .replicated });
    const n = self.final_layer.norm.forward(hidden, .d);
    const raw = self.final_layer.adaln.linear.forward(input.temb.silu().convert(self.final_layer.adaln.linear.weight.dtype()));
    const table = raw.splitAxis(.dout, .{ .k = 2, .d = self.cfg.hidden_size });
    const selected = table.gather(.{ .n = input.timestep_indices }, .{});
    const parts = selected.chunkExact(.k, 2);
    const shift = parts[0].squeeze(.k);
    const scale = parts[1].squeeze(.k);
    const one = zml.Tensor.scalar(1.0, n.dtype());
    const modulated = n.mul(one.add(scale.convert(n.dtype()).broad(n.shape()))).add(shift.convert(n.dtype()).broad(n.shape()));
    const aligned = modulated.convert(self.final_layer.video_out.weight.dtype());
    const video_all = self.final_layer.video_out.forward(aligned);
    const audio_all = self.final_layer.audio_out.forward(aligned);
    return .{
        .video = video_all.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }),
        .audio = audio_all.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }),
    };
}

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const parsed = try config_mod.parseConfig(allocator, io, repo);
        errdefer parsed.deinit();
        const cfg = parsed.value.resolve();
        return .{
            .inner = try .init(allocator, store, cfg),
            .parsed_config = parsed,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.inner.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        loader.load(io, Model, &self.inner, &buffers, store, shardings, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded DiT weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        return buffers;
    }

    pub fn loadEmbed(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(EmbedModel) {
        const part = self.inner.embedPart();
        return loadPart(allocator, io, platform, store, shardings, EmbedModel, &part, progress);
    }

    pub fn loadFinish(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(FinishModel) {
        const part = self.inner.finishPart();
        return loadPart(allocator, io, platform, store, shardings, FinishModel, &part, progress);
    }

    pub fn loadBlock(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(TransformerBlock) {
        return loadPart(allocator, io, platform, store, shardings, TransformerBlock, &self.inner.blocks[index], progress);
    }
};

fn loadPart(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    comptime T: type,
    model: *const T,
    progress: *std.Progress.Node,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(allocator, T, model);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 256 * zml.MiB,
        .parallelism = 16,
    });
    defer loader.deinit();
    loader.load(io, T, model, &buffers, store, shardings, .{ .progress = progress });
    try loader.await(io);
    return buffers;
}
