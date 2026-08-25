const std = @import("std");

const zml = @import("zml");

const config_mod = @import("../core/config.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3);

pub const Config = config_mod.Config;

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

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        return .{
            .fc1 = linear(store, "fc1.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
            .fc2 = linear(store, "fc2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
        unloadLinear(&self.fc1);
        unloadLinear(&self.fc2);
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const uv = self.fc1.forward(x);
        const gate, const value = uv.chunkExact(-1, 2);
        return self.fc2.forward(gate.silu().mul(value).rename(.{ .dout = .d }));
    }
};

const Attention = struct {
    qkv: zml.nn.Linear,
    out: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
        const qkv_part = .{ .dout = .model, .d = .replicated };
        const out_part = .{ .dout = .replicated, .d = .model };
        return .{
            .qkv = linear(store, "qkv_proj.weight", null, qkv_part, .replicated),
            .out = linear(store, "out_proj.weight", null, out_part, .replicated),
            .q_norm = .init(store.withPrefix("q_norm"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        unloadLinear(&self.qkv);
        unloadLinear(&self.out);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        // Fused `qkv_proj` is `(heads, 3, head_dim)`: per-head `[Q|K|V]`.
        const split = self.qkv.forward(x_qkv).splitAxis(.dout, .{ .h = self.num_heads, .p = 3, .hd = self.head_dim })
            .withPartitioning(.{ .h = .model });
        const parts = split.chunkExact(.p, 3);
        var q = parts[0].squeeze(.p);
        var k = parts[1].squeeze(.p);
        const v = parts[2].squeeze(.p);

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
        return .{
            .proj_in = linear(prefix, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
            .proj_out = linear(prefix, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
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
    kind: enum { full, curve } = .full,
    linear: zml.nn.Linear,
    hidden_size: i64,
    expand: i64,
    modalities: i64,

    pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        return .{
            .kind = .full,
            .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
            .hidden_size = hidden_size,
            .expand = expand,
            .modalities = modalities,
        };
    }

    pub fn initCurve(table: zml.Tensor, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        return .{
            .kind = .curve,
            .linear = .{ .weight = table, .tag = zml.Shape.toTag(.d) },
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
        const attn_store = store.withPrefix("attn");
        const mlp_store = store.withPrefix("mlp");
        const adaln_store = store.withPrefix("adaln_proj");
        const curve_table = store.maybeCreateTensor("adaln_t_table", .{ .t, .d }, .replicated);
        return .{
            .norm1 = .init(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(attn_store, cfg),
            .norm2 = .init(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(mlp_store),
            .adaln = if (curve_table) |table|
                .initCurve(table, cfg.hidden_size, 6, config_mod.modality_count)
            else
                .init(adaln_store, cfg.hidden_size, 6, config_mod.modality_count),
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
        const selected = if (self.adaln.kind == .curve) blk: {
            const raw = self.adaln.linear.weight.splitAxis(.d, .{
                .k = self.adaln.expand,
                .d = self.adaln.hidden_size,
            });
            break :blk raw.gather(.{ .t = input.adaln_indices }, .{});
        } else blk: {
            const table = self.adaln.forward(input.temb);
            const mods = table.merge(.{ .n = .{ .n, .mod } });
            break :blk mods.gather(.{ .n = input.adaln_indices }, .{});
        };
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
        const mlp_store = store.withPrefix("mlp");
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
        const block_store = store.withPrefix("blocks");
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

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
        const blocks_store = store.withPrefix("blocks");
        const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.num_layers));
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, i| {
            block.* = .init(blocks_store.withLayer(i), cfg);
        }

        const token_refiner = try TokenRefiner.init(allocator, store.withPrefix("token_refiner"), cfg);
        errdefer token_refiner.deinit(allocator);

        return .{
            .video_proj = linear(store, "video_patch_proj.weight", "video_patch_proj.bias", .replicated, .replicated),
            .audio_proj = linear(store, "audio_patch_proj.weight", "audio_patch_proj.bias", .replicated, .replicated),
            .condition_proj = linear(store, "condition_proj.weight", "condition_proj.bias", .replicated, .replicated),
            .time_embedder = .init(store),
            .token_refiner = token_refiner,
            .blocks = blocks,
            .final_layer = FinalLayer.init(store.withPrefix("final_layer"), cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.token_refiner.deinit(allocator);
        allocator.free(self.blocks);
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
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const parsed = try config_mod.parseConfig(allocator, io, repo);
        defer parsed.deinit();
        const cfg = parsed.value.resolve();
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
        return weights.load(allocator, io, platform, store, shardings, EmbedModel, &part, progress, null);
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
        return weights.load(allocator, io, platform, store, shardings, FinishModel, &part, progress, null);
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
        loader: ?*zml.io.Loader,
    ) !zml.Bufferized(TransformerBlock) {
        return weights.load(allocator, io, platform, store, shardings, TransformerBlock, &self.inner.blocks[index], progress, loader);
    }
};
