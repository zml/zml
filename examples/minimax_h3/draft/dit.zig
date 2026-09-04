const std = @import("std");

const zml = @import("zml");

const config = @import("../recipe/config.zig");
const lora = @import("../recipe/lora.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

const Config = config.Config;

const linear = weights.linear;
const rmsNorm = weights.rmsNorm;

// =============================================================================
// draft/dit.zig — H3 diffusion transformer
//
// Megatron TP on QKV/O/MLP. AdaLN tables are baked once per SKU.
// =============================================================================

const SwiGlu = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        const in_part = .{ .dout = .model, .d = .replicated };
        const out_part = .{ .dout = .replicated, .d = .model };
        return .{
            .fc1 = linear(store, "net.0.proj.weight", null, in_part, .replicated),
            .fc2 = linear(store, "net.2.weight", null, out_part, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
        zml.nn.Linear.unloadBuffers(&self.fc1);
        zml.nn.Linear.unloadBuffers(&self.fc2);
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const uv = self.fc1.forward(x);
        const value, const gate = uv.chunkExact(-1, 2);
        return self.fc2.forward(gate.silu().mul(value).rename(.{ .dout = .d }));
    }
};

const Attention = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    out: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    num_heads: i64,
    head_dim: i64,
    attn_backend: zml.attention.Backend = .vanilla,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
        const qkv_part = .{ .dout = .model, .d = .replicated };
        const out_part = .{ .dout = .replicated, .d = .model };
        return .{
            .q = linear(store, "to_q.weight", null, qkv_part, .replicated),
            .k = linear(store, "to_k.weight", null, qkv_part, .replicated),
            .v = linear(store, "to_v.weight", null, qkv_part, .replicated),
            .out = linear(store, "to_out.0.weight", null, out_part, .replicated),
            .q_norm = rmsNorm(store.withPrefix("norm_q"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = rmsNorm(store.withPrefix("norm_k"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        zml.nn.Linear.unloadBuffers(&self.q);
        zml.nn.Linear.unloadBuffers(&self.k);
        zml.nn.Linear.unloadBuffers(&self.v);
        zml.nn.Linear.unloadBuffers(&self.out);
        zml.nn.RmsNorm.unloadBuffers(&self.q_norm);
        zml.nn.RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn projectQkv(self: Attention, x: zml.Tensor) struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor } {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        return .{
            .q = self.q.forward(x).splitAxis(.dout, heads).withPartitioning(.{ .h = .model }),
            .k = self.k.forward(x).splitAxis(.dout, heads).withPartitioning(.{ .h = .model }),
            .v = self.v.forward(x).splitAxis(.dout, heads).withPartitioning(.{ .h = .model }),
        };
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        const qk = projectQkv(self, x_qkv);
        var q = qk.q;
        var k = qk.k;
        const v = qk.v;

        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);
        if (rotary) |pe| {
            q = zml.nn.applyRotary(q, pe[0], pe[1]);
            k = zml.nn.applyRotary(k, pe[0], pe[1]);
        }
        const q_s = q.rename(.{ .s = .q });
        const k_s = k.rename(.{ .s = .k });
        const v_s = v.rename(.{ .s = .k });
        const attn = zml.attention.dense(q_s, k_s, v_s, self.attn_backend, .{ .is_causal = false })
            .rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        return self.out.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

fn mmRope(position_ids: zml.Tensor, rope_freq_dim: i64, rope_theta: f32) struct { zml.Tensor, zml.Tensor } {
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

pub const TimeEmbedder = struct {
    proj_in: zml.nn.Linear,
    proj_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) TimeEmbedder {
        const prefix = store.withPrefix("time_embedder");
        return .{
            .proj_in = linear(prefix, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .proj_out = linear(prefix, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
        };
    }

    pub fn outDim(self: TimeEmbedder) i64 {
        return self.proj_out.weight.dim(.dout);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TimeEmbedder)) void {
        zml.nn.Linear.unloadBuffers(&self.proj_in);
        zml.nn.Linear.unloadBuffers(&self.proj_out);
    }

    pub fn forward(self: TimeEmbedder, t: zml.Tensor, freq_dim: i64) zml.Tensor {
        const features = timestepFeatures(t, freq_dim);
        return self.proj_out.forward(self.proj_in.forward(features).silu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

fn timestepFeatures(t: zml.Tensor, dim: i64) zml.Tensor {
    const inv = zml.nn.invFreq(dim, .{
        .layout = .real_im_pass,
        .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
    }).withTags(.{.f});
    const angles = t.convert(.f32).withPartialTags(.{.n}).outer(inv);
    return zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
}

pub const AdaLn = struct {
    linear: zml.nn.Linear,
    hidden_size: i64,
    expand: i64,
    modalities: i64,

    pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        return .{
            .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
            .hidden_size = hidden_size,
            .expand = expand,
            .modalities = modalities,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AdaLn)) void {
        zml.nn.Linear.unloadBuffers(&self.linear);
    }

    pub fn forward(self: AdaLn, temb: zml.Tensor) zml.Tensor {
        const cond = temb.silu();
        const raw = self.linear.forward(cond.convert(self.linear.weight.dtype()));
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

pub const BlockCore = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,
    hidden_size: i64,

    pub const Input = struct {
        layer: BlockCore,
        hidden: zml.Tensor,
        mods: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn unloadBuffers(self: *zml.Bufferized(BlockCore)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        zml.nn.RmsNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.mlp);
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const mods = if (input.mods.shape().hasTag(.mod)) |_|
            input.mods.merge(.{ .n = .{ .n, .mod } })
        else
            input.mods;
        const selected = mods.gather(.{ .n = input.adaln_indices }, .{});
        const parts = selected.chunkExact(.k, 6);
        const shift_msa = parts[0].squeeze(.k);
        const scale_msa = parts[1].squeeze(.k);
        const gate_msa = parts[2].squeeze(.k);
        const shift_mlp = parts[3].squeeze(.k);
        const scale_mlp = parts[4].squeeze(.k);
        const gate_mlp = parts[5].squeeze(.k);

        const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
        const n1 = self.norm1.forward(residual0);
        const one = zml.Tensor.scalar(1.0, n1.dtype());
        const attn_in = n1.mul(one.add(scale_msa.convert(n1.dtype()).broad(n1.shape()))).add(shift_msa.convert(n1.dtype()).broad(n1.shape()));
        const attn_out = self.attn.forward(attn_in, .{ input.cos, input.sin });
        const x1 = residual0.add(gate_msa.convert(attn_out.dtype()).broad(attn_out.shape()).mul(attn_out)).withPartitioning(.{ .d = .replicated });

        const n2 = self.norm2.forward(x1);
        const mlp_in = n2.mul(one.add(scale_mlp.convert(n2.dtype()).broad(n2.shape()))).add(shift_mlp.convert(n2.dtype()).broad(n2.shape()));
        const mlp_out = self.mlp.forward(mlp_in).rename(.{ .dout = .d });
        const x2 = x1.add(gate_mlp.convert(mlp_out.dtype()).broad(mlp_out.shape()).mul(mlp_out)).withPartitioning(.{ .d = .replicated });
        return .{ .hidden = x2.reuseBuffer(input.hidden) };
    }
};

pub const StepBlockInput = struct {
    layer: BlockCore,
    hidden: zml.Tensor,
    table: zml.Tensor,
    step: zml.Tensor,
    adaln_indices: zml.Tensor,
    cos: zml.Tensor,
    sin: zml.Tensor,
};

pub fn stepBlock(input: StepBlockInput) BlockCore.Output {
    const mods = input.table.gather(.{ .t = input.step }, .{});
    return BlockCore.forward(.{
        .layer = input.layer,
        .hidden = input.hidden,
        .mods = mods,
        .adaln_indices = input.adaln_indices,
        .cos = input.cos,
        .sin = input.sin,
    });
}

pub const BlockGroup = struct {
    layers: []BlockCore,

    pub const Input = struct {
        group: BlockGroup,
        hidden: zml.Tensor,
        tables: []zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn unloadBuffers(self: *zml.Bufferized(BlockGroup), allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| BlockCore.unloadBuffers(layer);
        allocator.free(self.layers);
    }

    pub fn forward(input: Input) Output {
        var hidden = input.hidden;
        for (input.group.layers, input.tables) |layer, table| {
            const mods = table.gather(.{ .t = input.step }, .{});
            hidden = BlockCore.forward(.{
                .layer = layer,
                .hidden = hidden,
                .mods = mods,
                .adaln_indices = input.adaln_indices,
                .cos = input.cos,
                .sin = input.sin,
            }).hidden;
        }
        return .{ .hidden = hidden };
    }
};

pub const TransformerBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,
    adaln: AdaLn,
    hidden_size: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TransformerBlock {
        const attn_store = store.withPrefix("attn");
        const mlp_store = store.withPrefix("ff");
        const adaln_store = store.withPrefix("adaln_proj");
        return .{
            .norm1 = rmsNorm(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(attn_store, cfg),
            .norm2 = rmsNorm(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(mlp_store),
            .adaln = .init(adaln_store, cfg.hidden_size, 6, config.modality_count),
            .hidden_size = cfg.hidden_size,
        };
    }

    pub fn corePart(self: TransformerBlock) BlockCore {
        return .{
            .norm1 = self.norm1,
            .attn = self.attn,
            .norm2 = self.norm2,
            .mlp = self.mlp,
            .hidden_size = self.hidden_size,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        zml.nn.RmsNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.mlp);
        AdaLn.unloadBuffers(&self.adaln);
    }
};

const TokenRefinerBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TokenRefinerBlock {
        const attn_store = store.withPrefix("attn");
        const mlp_store = store.withPrefix("ff");
        return .{
            .norm1 = rmsNorm(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(attn_store, cfg),
            .norm2 = rmsNorm(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(mlp_store),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenRefinerBlock)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        zml.nn.RmsNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: TokenRefinerBlock, x: zml.Tensor) zml.Tensor {
        const residual = x.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual), null));
        return x1.add(self.mlp.forward(self.norm2.forward(x1)).rename(.{ .dout = .d })).withPartitioning(.{ .d = .replicated }).reuseBuffer(x);
    }
};

const TokenRefiner = struct {
    blocks: []TokenRefinerBlock,
    final_norm: zml.nn.RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !TokenRefiner {
        const block_store = store.withPrefix("refiner_blocks");
        const blocks = try allocator.alloc(TokenRefinerBlock, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, i| {
            block.* = .init(block_store.withLayer(i), cfg);
        }
        return .{
            .blocks = blocks,
            .final_norm = rmsNorm(store.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
        };
    }

    pub fn deinit(self: TokenRefiner, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenRefiner), allocator: std.mem.Allocator) void {
        for (self.blocks) |*block| TokenRefinerBlock.unloadBuffers(block);
        allocator.free(self.blocks);
        zml.nn.RmsNorm.unloadBuffers(&self.final_norm);
    }

    pub fn forward(self: TokenRefiner, x: zml.Tensor) zml.Tensor {
        var hidden = x;
        for (self.blocks) |block| {
            hidden = block.forward(hidden);
        }
        return self.final_norm.forward(hidden);
    }
};

const FinalLayer = struct {
    norm: zml.nn.RmsNorm,
    adaln: AdaLn,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) FinalLayer {
        return .{
            .norm = rmsNorm(store.withPrefix("norm_out.norm"), .{.d}, cfg.final_norm_eps),
            .adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size, 2, 1),
            .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
            .audio_out = linear(store, "audio_proj_out.weight", "audio_proj_out.bias", .replicated, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FinalLayer)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm);
        AdaLn.unloadBuffers(&self.adaln);
        zml.nn.Linear.unloadBuffers(&self.video_out);
        zml.nn.Linear.unloadBuffers(&self.audio_out);
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

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
        const blocks_store = store.withPrefix("transformer_blocks");
        const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.num_layers));
        errdefer allocator.free(blocks);

        const token_refiner = try TokenRefiner.init(allocator, store.withPrefix("token_refiner"), cfg);
        errdefer token_refiner.deinit(allocator);

        const time_embedder: TimeEmbedder = .init(store);
        for (blocks, 0..) |*block, i| {
            block.* = .init(blocks_store.withLayer(i), cfg);
        }

        const video_proj = linear(store, "proj_in.weight", "proj_in.bias", .replicated, .replicated);
        const audio_proj = linear(store, "audio_proj_in.weight", "audio_proj_in.bias", .replicated, .replicated);
        const condition_proj = linear(store, "context_embedder.weight", "context_embedder.bias", .replicated, .replicated);

        return .{
            .video_proj = video_proj,
            .audio_proj = audio_proj,
            .condition_proj = condition_proj,
            .time_embedder = time_embedder,
            .token_refiner = token_refiner,
            .blocks = blocks,
            .final_layer = FinalLayer.init(store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.token_refiner.deinit(allocator);
        allocator.free(self.blocks);
    }

    pub fn textPrep(self: Model) TextPrep {
        return .{
            .condition_proj = self.condition_proj,
            .token_refiner = self.token_refiner,
        };
    }

    pub fn patchEmbed(self: Model) PatchEmbed {
        return .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
            .hidden_size = self.cfg.hidden_size,
            .seq = 0,
        };
    }

    pub fn finishCore(self: Model) FinishCore {
        return .{
            .norm = self.final_layer.norm,
            .video_out = self.final_layer.video_out,
            .audio_out = self.final_layer.audio_out,
        };
    }

    pub fn applyBackend(self: *Model, dit_kind: zml.attention.Backend, refiner_kind: zml.attention.Backend) void {
        for (self.blocks) |*block| block.attn.attn_backend = dit_kind;
        for (self.token_refiner.blocks) |*block| block.attn.attn_backend = refiner_kind;
    }
};

pub const TextPrep = struct {
    condition_proj: zml.nn.Linear,
    token_refiner: TokenRefiner,

    pub fn unloadBuffers(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
        zml.nn.Linear.unloadBuffers(&self.condition_proj);
        TokenRefiner.unloadBuffers(&self.token_refiner, allocator);
    }
};

pub const PatchEmbed = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    hidden_size: i64,
    seq: i64 = 0,

    pub fn unloadBuffers(self: *zml.Bufferized(PatchEmbed)) void {
        zml.nn.Linear.unloadBuffers(&self.video_proj);
        zml.nn.Linear.unloadBuffers(&self.audio_proj);
    }
};

pub const FinishCore = struct {
    norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(FinishCore)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm);
        zml.nn.Linear.unloadBuffers(&self.video_out);
        zml.nn.Linear.unloadBuffers(&self.audio_out);
    }
};

pub const TextPrepInput = struct {
    model: TextPrep,
    text: zml.Tensor,
};

pub const TextPrepOutput = struct {
    text: zml.Tensor,
};

pub fn prepareText(input: TextPrepInput) TextPrepOutput {
    const self = input.model;
    var text = self.condition_proj.forward(input.text.convert(self.condition_proj.weight.dtype())).rename(.{ .dout = .d });
    text = self.token_refiner.forward(text.convert(self.token_refiner.final_norm.weight.dtype()));
    return .{ .text = text };
}

pub const RopeModel = struct {
    rope_freq_dim: i64,
    rope_theta: f32,
    out_dtype: zml.DataType,
};

pub const RopeInput = struct {
    model: RopeModel,
    position_ids: zml.Tensor,
};

pub const RopeOutput = struct {
    cos: zml.Tensor,
    sin: zml.Tensor,
};

pub fn prepareRope(input: RopeInput) RopeOutput {
    const cos, const sin = mmRope(input.position_ids, input.model.rope_freq_dim, input.model.rope_theta);
    return .{
        .cos = cos.convert(input.model.out_dtype),
        .sin = sin.convert(input.model.out_dtype),
    };
}

pub const PatchInput = struct {
    model: PatchEmbed,
    video: zml.Tensor,
    audio: zml.Tensor,
    text: zml.Tensor,
    video_indices: zml.Tensor,
    audio_indices: zml.Tensor,
    text_indices: zml.Tensor,
};

pub const PatchOutput = struct {
    hidden: zml.Tensor,
};

pub fn embedPatches(input: PatchInput) PatchOutput {
    const self = input.model;
    const video = self.video_proj.forward(input.video.convert(self.video_proj.weight.dtype())).rename(.{ .dout = .d });
    const audio = self.audio_proj.forward(input.audio.convert(self.audio_proj.weight.dtype())).rename(.{ .dout = .d });
    const text = input.text;
    const batch = text.dim(.b);
    var hidden = zml.Tensor.zeroes(zml.Shape.init(.{ .b = batch, .s = self.seq, .d = self.hidden_size }, text.dtype()));
    hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
    hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
    hidden = hidden.scatterSlices(.{ .s = input.audio_indices.withTags(.{.s}) }, audio.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
    return .{ .hidden = hidden.withPartitioning(.{ .d = .replicated }) };
}

pub const TembInput = struct {
    model: TimeEmbedder,
    timestep: zml.Tensor,
    freq_dim: i64,
};

pub const TembOutput = struct {
    temb: zml.Tensor,
};

pub fn prepareTemb(input: TembInput) TembOutput {
    return .{ .temb = input.model.forward(input.timestep, input.freq_dim) };
}

pub const AdaLnPrep = struct {
    adaln: AdaLn,
    steps: i64,
    slots: i64,
};

pub const AdaLnPrepInput = struct {
    model: AdaLnPrep,
    temb: zml.Tensor,
};

pub const AdaLnPrepOutput = struct {
    table: zml.Tensor,
};

pub fn prepareAdaln(input: AdaLnPrepInput) AdaLnPrepOutput {
    const raw = input.model.adaln.forward(input.temb);
    return .{ .table = raw.splitAxis(.n, .{ .t = input.model.steps, .n = input.model.slots }) };
}

pub const ScatterInput = struct {
    hidden: zml.Tensor,
    values: zml.Tensor,
    indices: zml.Tensor,
};

pub const ScatterOutput = struct {
    hidden: zml.Tensor,
};

pub fn scatterRows(input: ScatterInput) ScatterOutput {
    const hidden = input.hidden.scatterSlices(
        .{ .s = input.indices.withTags(.{.s}) },
        input.values.convert(input.hidden.dtype()),
        .{ .update_fn = zml.Tensor.ScatterOpts.override },
    );
    return .{ .hidden = hidden };
}

pub const FinishInput = struct {
    model: FinishCore,
    hidden: zml.Tensor,
    table: zml.Tensor,
    step: zml.Tensor,
    timestep_indices: zml.Tensor,
    video_indices: zml.Tensor,
    audio_indices: zml.Tensor,
};

pub const FinishOutput = struct {
    video: zml.Tensor,
    audio: zml.Tensor,
};

fn modulateRows(norm: zml.nn.RmsNorm, hidden: zml.Tensor, mods: zml.Tensor, timestep_indices: zml.Tensor) zml.Tensor {
    const n = norm.forward(hidden.withPartitioning(.{ .d = .replicated }));
    const selected = mods.gather(.{ .n = timestep_indices }, .{});
    const parts = selected.chunkExact(.k, 2);
    const shift = parts[0].squeeze(.k);
    const scale = parts[1].squeeze(.k);
    const one = zml.Tensor.scalar(1.0, n.dtype());
    return n.mul(one.add(scale.convert(n.dtype()).broad(n.shape()))).add(shift.convert(n.dtype()).broad(n.shape()));
}

pub fn finish(input: FinishInput) FinishOutput {
    const self = input.model;
    const mods = input.table.gather(.{ .t = input.step }, .{});
    const video_h = input.hidden.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
    const audio_h = input.hidden.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
    const video_t = input.timestep_indices.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
    const audio_t = input.timestep_indices.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
    const video_m = modulateRows(self.norm, video_h, mods, video_t);
    const audio_m = modulateRows(self.norm, audio_h, mods, audio_t);
    return .{
        .video = self.video_out.forward(video_m.convert(self.video_out.weight.dtype())),
        .audio = self.audio_out.forward(audio_m.convert(self.audio_out.weight.dtype())),
    };
}

pub const LoadedModel = struct {
    inner: Model,
    cfg: Config,
    lora: ?*const lora.Bundle = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const cfg = try config.loadDitConfig(allocator, io, repo);
        log.info("dit: {d} layers hidden={d} heads={d} text_dim={d}", .{
            cfg.num_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.text_dim,
        });
        return .{
            .inner = try .init(allocator, store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }

    pub fn loadCore(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
        loader: ?*zml.io.Loader,
    ) !zml.Bufferized(BlockCore) {
        const core = self.inner.blocks[index].corePart();
        var bufs = try weights.load(allocator, io, platform, store, shardings, BlockCore, &core, progress, loader);
        if (self.lora) |bundle| try bundle.mergeCore(allocator, io, platform, &bufs, index);
        return bufs;
    }

    pub fn loadAdaln(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
        loader: ?*zml.io.Loader,
    ) !zml.Bufferized(AdaLn) {
        var bufs = try weights.load(allocator, io, platform, store, shardings, AdaLn, &self.inner.blocks[index].adaln, progress, loader);
        if (self.lora) |bundle| try bundle.mergeAdaln(allocator, io, platform, &bufs, index);
        return bufs;
    }

    pub fn loadTextPrep(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(TextPrep) {
        const part = self.inner.textPrep();
        const bufs = try weights.load(allocator, io, platform, store, shardings, TextPrep, &part, progress, null);
        if (self.lora) |bundle| {
            for (bufs.token_refiner.blocks, 0..) |*block, i| {
                try bundle.mergeRefiner(allocator, io, platform, block, i);
            }
        }
        return bufs;
    }

    pub fn loadPatchEmbed(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(PatchEmbed) {
        const part = self.inner.patchEmbed();
        return weights.load(allocator, io, platform, store, shardings, PatchEmbed, &part, progress, null);
    }

    pub fn loadTimeEmbedder(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(TimeEmbedder) {
        return weights.load(allocator, io, platform, store, shardings, TimeEmbedder, &self.inner.time_embedder, progress, null);
    }

    pub fn loadFinishCore(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(FinishCore) {
        const part = self.inner.finishCore();
        return weights.load(allocator, io, platform, store, shardings, FinishCore, &part, progress, null);
    }

    pub fn loadFinalAdaln(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(AdaLn) {
        var bufs = try weights.load(allocator, io, platform, store, shardings, AdaLn, &self.inner.final_layer.adaln, progress, null);
        if (self.lora) |bundle| try bundle.mergeFinalAdaln(allocator, io, platform, &bufs);
        return bufs;
    }
};
