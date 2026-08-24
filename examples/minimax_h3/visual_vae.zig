const std = @import("std");

const zml = @import("zml");

const config_mod = @import("config.zig");
const vae = @import("vae.zig");

const log = std.log.scoped(.minimax_h3_visual_vae);

pub const Config = struct {
    latent_channels: i64 = 24,
    out_channels: i64 = 3,
    decoder_num_layers: i64 = 36,
    decoder_num_attention_heads: i64 = 32,
    decoder_attention_head_dim: i64 = 64,
    decoder_num_register_tokens: i64 = 4,
    decoder_ffn_mult: i64 = 4,
    decoder_rope_theta: f32 = 100.0,
    decoder_rope_dim_ratio: f32 = 0.75,
    decoder_norm_eps: f32 = 1e-5,
    clip_length: u32 = 17,
    token_drop: u32 = 3,
    tile_px: u32 = 256,
    tile_overlap_px: u32 = 64,
    latents_mean: [24]f32 = @splat(0),
    latents_std: [24]f32 = @splat(1),

    pub fn dim(self: Config) i64 {
        return self.decoder_num_attention_heads * self.decoder_attention_head_dim;
    }

    pub fn rotaryDim(self: Config) i64 {
        return @intFromFloat(@as(f32, @floatFromInt(self.decoder_attention_head_dim)) * self.decoder_rope_dim_ratio);
    }

    pub fn spec(self: Config) vae.VisualSpec {
        return .{
            .channels = @intCast(self.latent_channels),
            .clip_length = self.clip_length,
            .token_drop = self.token_drop,
            .tile_px = self.tile_px,
            .tile_overlap_px = self.tile_overlap_px,
        };
    }

    pub fn official() Config {
        return .{};
    }
};

const FileConfig = struct {
    latent_channels: ?i64 = null,
    out_channels: ?i64 = null,
    decoder_num_layers: ?i64 = null,
    num_layers: ?i64 = null,
    decoder_num_attention_heads: ?i64 = null,
    heads: ?i64 = null,
    decoder_attention_head_dim: ?i64 = null,
    dim_head: ?i64 = null,
    decoder_num_register_tokens: ?i64 = null,
    num_register_tokens: ?i64 = null,
    decoder_ffn_mult: ?i64 = null,
    decoder_rope_theta: ?f32 = null,
    rope_theta: ?f32 = null,
    decoder_rope_dim_ratio: ?f32 = null,
    rope_dim_ratio: ?f32 = null,
    decoder_norm_eps: ?f32 = null,
    clip_length: ?u32 = null,
    vae_clip_length: ?u32 = null,
    token_drop: ?u32 = null,
    vae_token_drop: ?u32 = null,
    vae_tile_size: ?u32 = null,
    vae_tile_overlap_min: ?u32 = null,
    latents_mean: ?[]const f32 = null,
    latents_std: ?[]const f32 = null,

    fn resolve(self: FileConfig) Config {
        var out = Config.official();
        if (self.latent_channels) |v| out.latent_channels = v;
        if (self.out_channels) |v| out.out_channels = v;
        if (self.decoder_num_layers orelse self.num_layers) |v| out.decoder_num_layers = v;
        if (self.decoder_num_attention_heads orelse self.heads) |v| out.decoder_num_attention_heads = v;
        if (self.decoder_attention_head_dim orelse self.dim_head) |v| out.decoder_attention_head_dim = v;
        if (self.decoder_num_register_tokens orelse self.num_register_tokens) |v| out.decoder_num_register_tokens = v;
        if (self.decoder_ffn_mult) |v| out.decoder_ffn_mult = v;
        if (self.decoder_rope_theta orelse self.rope_theta) |v| out.decoder_rope_theta = v;
        if (self.decoder_rope_dim_ratio orelse self.rope_dim_ratio) |v| out.decoder_rope_dim_ratio = v;
        if (self.decoder_norm_eps) |v| out.decoder_norm_eps = v;
        if (self.clip_length orelse self.vae_clip_length) |v| out.clip_length = v;
        if (self.token_drop orelse self.vae_token_drop) |v| out.token_drop = v;
        if (self.vae_tile_size) |v| out.tile_px = v;
        if (self.vae_tile_overlap_min) |v| out.tile_overlap_px = v;
        if (self.latents_mean) |mean| {
            for (0..@min(mean.len, out.latents_mean.len)) |i| out.latents_mean[i] = mean[i];
        }
        if (self.latents_std) |stddev| {
            for (0..@min(stddev.len, out.latents_std.len)) |i| out.latents_std[i] = stddev[i];
        }
        return out;
    }
};

fn tensorRank(store: zml.io.TensorStore.View, name: []const u8) u8 {
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", name }) catch return 2;
    return if (store.store.getShape(key)) |shape| shape.rank() else 2;
}

fn linearWeight(store: zml.io.TensorStore.View, weight_name: []const u8) zml.Tensor {
    return switch (tensorRank(store, weight_name)) {
        5 => store.createTensor(weight_name, .{ .dout, .d, .kt, .kh, .kw }, .replicated),
        4 => store.createTensor(weight_name, .{ .dout, .d, .kh, .kw }, .replicated),
        3 => store.createTensor(weight_name, .{ .dout, .d, .k }, .replicated),
        else => store.createTensor(weight_name, .{ .dout, .d }, .replicated),
    };
}

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
    return .init(
        linearWeight(store, weight_name),
        if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, .replicated) else null,
        .d,
    );
}

fn unloadLinear(lin: *zml.Bufferized(zml.nn.Linear)) void {
    lin.weight.deinit();
    if (lin.bias) |*bias| bias.deinit();
}

const LayerNorm = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) LayerNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .replicated),
            .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LayerNorm)) void {
        self.weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: LayerNorm, x: zml.Tensor) zml.Tensor {
        return (zml.nn.LayerNorm{
            .weight = self.weight,
            .bias = self.bias,
            .eps = self.eps,
        }).forward(x.convert(.f32)).convert(x.dtype());
    }
};

const SwiGlu = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        if (store.hasKey("w1.weight")) {
            return .{
                .w1 = linear(store, "w1.weight", "w1.bias"),
                .w2 = linear(store, "w2.weight", "w2.bias"),
            };
        }
        return .{
            .w1 = linear(store, "net.0.proj.weight", "net.0.proj.bias"),
            .w2 = linear(store, "net.2.weight", "net.2.bias"),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
        unloadLinear(&self.w1);
        unloadLinear(&self.w2);
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const fused = self.w1.forward(x);
        const gate, const value = fused.chunkExact(-1, 2);
        return self.w2.forward(gate.silu().mul(value).rename(.{ .dout = .d }));
    }
};

const Attention = struct {
    qkv: ?zml.nn.Linear,
    to_q: ?zml.nn.Linear,
    to_k: ?zml.nn.Linear,
    to_v: ?zml.nn.Linear,
    out: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
        const fused = store.hasKey("to_qkv.weight");
        return .{
            .qkv = if (fused) linear(store, "to_qkv.weight", "to_qkv.bias") else null,
            .to_q = if (!fused) linear(store, "to_q.weight", "to_q.bias") else null,
            .to_k = if (!fused) linear(store, "to_k.weight", "to_k.bias") else null,
            .to_v = if (!fused) linear(store, "to_v.weight", "to_v.bias") else null,
            .out = linear(store, if (store.hasKey("to_out.weight")) "to_out.weight" else "to_out.0.weight", if (store.hasKey("to_out.bias")) "to_out.bias" else "to_out.0.bias"),
            .num_heads = cfg.decoder_num_attention_heads,
            .head_dim = cfg.decoder_attention_head_dim,
            .eps = cfg.decoder_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        if (self.qkv) |*qkv| unloadLinear(qkv);
        if (self.to_q) |*q| unloadLinear(q);
        if (self.to_k) |*k| unloadLinear(k);
        if (self.to_v) |*v| unloadLinear(v);
        unloadLinear(&self.out);
    }

    pub fn forward(self: Attention, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        var q: zml.Tensor = undefined;
        var k: zml.Tensor = undefined;
        var v: zml.Tensor = undefined;
        if (self.qkv) |qkv| {
            const split = qkv.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 3 * self.head_dim });
            const parts = split.chunkExact(.hd, 3);
            q = parts[0];
            k = parts[1];
            v = parts[2];
        } else {
            q = self.to_q.?.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
            k = self.to_k.?.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
            v = self.to_v.?.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
        }
        q = zml.nn.rmsNorm(q.convert(.f32), .hd, self.eps).convert(x.dtype());
        k = zml.nn.rmsNorm(k.convert(.f32), .hd, self.eps).convert(x.dtype());
        q = applyRotary(q, cos, sin);
        k = applyRotary(k, cos, sin);
        const attn = zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{},
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        return self.out.forward(attn).rename(.{ .dout = .d });
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

pub fn vitRope(position_ids: zml.Tensor, rotary_dim: i64, theta: f32) struct { zml.Tensor, zml.Tensor } {
    const n_dim: i64 = 3;
    const freq_len = @divExact(rotary_dim, 2 * n_dim);
    const step = @as(f32, @floatFromInt(2 * n_dim)) / @as(f32, @floatFromInt(rotary_dim));
    const idx = zml.Tensor.arange(.{ .end = freq_len }, .f32).withTags(.{.f});
    const inv = zml.Tensor.scalar(theta, .f32).pow(idx.scale(-step));
    const pos = position_ids.convert(.f32).withPartialTags(.{ .s, .ax });
    const freqs = pos.outer(inv).scale(2.0 * std.math.pi);
    const parts = freqs.chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    const emb = zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
    return .{ emb.cos(), emb.sin() };
}

pub const TransformerBlock = struct {
    norm1: LayerNorm,
    attn: Attention,
    scale1: zml.Tensor,
    norm2: LayerNorm,
    ff: SwiGlu,
    scale2: zml.Tensor,

    pub const Input = struct {
        layer: TransformerBlock,
        hidden: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TransformerBlock {
        const attn_store = if (store.hasKey("attn.to_qkv.weight") or store.hasKey("attn.to_q.weight")) store.withPrefix("attn") else store;
        const ff_store = if (store.hasKey("ff.w1.weight") or store.hasKey("ff.net.0.proj.weight")) store.withPrefix("ff") else store;
        return .{
            .norm1 = .init(store.withPrefix("norm1"), cfg.decoder_norm_eps),
            .attn = .init(attn_store, cfg),
            .scale1 = store.createTensor("scale1", .{.d}, .replicated),
            .norm2 = .init(store.withPrefix("norm2"), cfg.decoder_norm_eps),
            .ff = .init(ff_store),
            .scale2 = store.createTensor("scale2", .{.d}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
        LayerNorm.unloadBuffers(&self.norm1);
        Attention.unloadBuffers(&self.attn);
        self.scale1.deinit();
        LayerNorm.unloadBuffers(&self.norm2);
        SwiGlu.unloadBuffers(&self.ff);
        self.scale2.deinit();
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual = input.hidden;
        const attn = self.attn.forward(self.norm1.forward(residual), input.cos, input.sin);
        const x1 = residual.add(attn.mul(self.scale1.convert(attn.dtype()).broad(attn.shape())));
        const ff = self.ff.forward(self.norm2.forward(x1)).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff.mul(self.scale2.convert(ff.dtype()).broad(ff.shape()))).reuseBuffer(input.hidden) };
    }
};

pub const EmbedModel = struct {
    post_quant: zml.nn.Linear,
    proj: zml.nn.Linear,
    register_tokens: zml.Tensor,
    cfg: Config,

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedModel)) void {
        unloadLinear(&self.post_quant);
        unloadLinear(&self.proj);
        self.register_tokens.deinit();
    }
};

pub const FinishModel = struct {
    norm_out: LayerNorm,
    proj_out: zml.nn.Linear,
    cfg: Config,

    pub fn unloadBuffers(self: *zml.Bufferized(FinishModel)) void {
        LayerNorm.unloadBuffers(&self.norm_out);
        unloadLinear(&self.proj_out);
    }
};

pub const Model = struct {
    embed: EmbedModel,
    blocks: []TransformerBlock,
    finish: FinishModel,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
        const store = rootView(store_);
        const dec = decoderView(store);
        const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.decoder_num_layers));
        errdefer allocator.free(blocks);
        const block_store = if (dec.hasKey("transformer_blocks.0.norm1.weight")) dec.withPrefix("transformer_blocks") else dec.withPrefix("blocks");
        for (blocks, 0..) |*block, i| block.* = .init(block_store.withLayer(i), cfg);

        const proj_name = if (dec.hasKey("x_embedder.weight")) "x_embedder" else "proj_in";
        const post = if (store.hasKey("post_quant_conv.weight")) store.withPrefix("post_quant_conv") else store.withPrefix("decoder.post_quant_conv");
        return .{
            .embed = .{
                .post_quant = linear(post, "weight", "bias"),
                .proj = linear(dec.withPrefix(proj_name), "weight", "bias"),
                .register_tokens = dec.createTensor("register_tokens", .{ .b, .s, .d }, .replicated),
                .cfg = cfg,
            },
            .blocks = blocks,
            .finish = .{
                .norm_out = .init(dec.withPrefix("norm_out"), cfg.decoder_norm_eps),
                .proj_out = linear(dec.withPrefix("proj_out"), "weight", "bias"),
                .cfg = cfg,
            },
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }
};

pub fn ready(store: zml.io.TensorStore.View) bool {
    return store.hasKey("decoder.x_embedder.weight") or
        store.hasKey("decoder.proj_in.weight") or
        store.hasKey("x_embedder.weight") or
        store.hasKey("proj_in.weight") or
        store.hasKey("post_quant_conv.weight") or
        store.hasKey("model.decoder.x_embedder.weight");
}

fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("decoder.x_embedder.weight") or store.hasKey("decoder.proj_in.weight") or store.hasKey("post_quant_conv.weight")) return store;
    if (store.hasKey("model.decoder.x_embedder.weight")) return store.withPrefix("model");
    return store;
}

fn decoderView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("decoder.x_embedder.weight") or store.hasKey("decoder.proj_in.weight")) return store.withPrefix("decoder");
    return store;
}

pub const EmbedInput = struct {
    model: EmbedModel,
    latents: zml.Tensor,
    position_ids: zml.Tensor,
};

pub const EmbedOutput = struct {
    hidden: zml.Tensor,
    cos: zml.Tensor,
    sin: zml.Tensor,
};

fn conv1x1(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var weight = lin.weight;
    while (weight.rank() > 2) {
        weight = weight.squeeze(-1);
    }
    return (zml.nn.Linear.init(weight.withTags(.{ .dout, .d }), lin.bias, .d)).forward(x);
}

pub fn embed(input: EmbedInput) EmbedOutput {
    const self = input.model;
    const x = input.latents.withPartialTags(.{ .b, .s, .d });
    const quantized = conv1x1(self.post_quant, x).rename(.{ .dout = .d });
    const tokens = self.proj.forward(quantized.convert(self.proj.weight.dtype())).rename(.{ .dout = .d });
    const registers = self.register_tokens.convert(tokens.dtype()).broad(zml.Shape.init(.{
        .b = tokens.dim(.b),
        .s = self.register_tokens.dim(.s),
        .d = tokens.dim(.d),
    }, tokens.dtype()));
    const cls = zml.Tensor.zeroes(zml.Shape.init(.{ .b = tokens.dim(.b), .s = 1, .d = tokens.dim(.d) }, tokens.dtype()));
    const hidden = zml.Tensor.concatenate(&.{ tokens, registers, cls }, .s);
    const cos, const sin = vitRope(input.position_ids, self.cfg.rotaryDim(), self.cfg.decoder_rope_theta);
    return .{
        .hidden = hidden,
        .cos = cos.convert(tokens.dtype()),
        .sin = sin.convert(tokens.dtype()),
    };
}

pub const FinishInput = struct {
    model: FinishModel,
    hidden: zml.Tensor,
};

pub const FinishOutput = struct {
    patches: zml.Tensor,
};

pub fn finish(input: FinishInput) FinishOutput {
    const self = input.model;
    const hidden = self.norm_out.forward(input.hidden);
    const proj = self.proj_out.forward(hidden).rename(.{ .dout = .d });
    const keep = proj.dim(.s) - self.cfg.decoder_num_register_tokens - 1;
    return .{ .patches = proj.slice1d(.s, .{ .start = 0, .end = keep }) };
}

pub const LoadedModel = struct {
    inner: Model,
    parsed: ?std.json.Parsed(FileConfig),
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const parsed: ?std.json.Parsed(FileConfig) = config_mod.parseJson(FileConfig, allocator, io, repo, "config.json") catch
            config_mod.parseJson(FileConfig, allocator, io, repo, "source/config.json") catch null;
        const cfg = if (parsed) |p| p.value.resolve() else Config.official();
        return .{
            .inner = try .init(allocator, store, cfg),
            .parsed = parsed,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        if (self.parsed) |*parsed| parsed.deinit();
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
        return loadPart(allocator, io, platform, store, shardings, EmbedModel, &self.inner.embed, progress);
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
        return loadPart(allocator, io, platform, store, shardings, FinishModel, &self.inner.finish, progress);
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

pub const TileShape = struct {
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,

    pub fn tokens(self: TileShape) u32 {
        return self.latent_t * self.latent_h * self.latent_w;
    }

    pub fn seq(self: TileShape, registers: u32) u32 {
        return self.tokens() + registers + 1;
    }

    pub fn fromGeometry(cfg: Config, latent_t: u32, latent_h: u32, latent_w: u32) TileShape {
        const spec = cfg.spec();
        const tile = vae.decodeTileLatent(spec, latent_h, latent_w);
        return .{
            .latent_t = vae.decodeClipTokens(spec, latent_t),
            .latent_h = tile.h,
            .latent_w = tile.w,
        };
    }
};

pub fn hostPositions(allocator: std.mem.Allocator, t: u32, h: u32, w: u32, registers: u32) ![]f32 {
    const patches = t * h * w;
    const seq = patches + registers + 1;
    const out = try allocator.alloc(f32, seq * 3);
    const t_axis = try allocator.alloc(f32, t);
    defer allocator.free(t_axis);
    const h_axis = try allocator.alloc(f32, h);
    defer allocator.free(h_axis);
    const w_axis = try allocator.alloc(f32, w);
    defer allocator.free(w_axis);
    _ = vae.vitCoords(t, t_axis);
    _ = vae.vitCoords(h, h_axis);
    _ = vae.vitCoords(w, w_axis);
    var i: usize = 0;
    for (0..t) |tt| {
        for (0..h) |hh| {
            for (0..w) |ww| {
                out[i * 3 + 0] = t_axis[tt];
                out[i * 3 + 1] = h_axis[hh];
                out[i * 3 + 2] = w_axis[ww];
                i += 1;
            }
        }
    }
    @memset(out[patches * 3 ..], 0);
    return out;
}

/// Host unpack of ViT patch tokens `{s, 3*pt*ph*pw}` into NCHW `{3, T, H, W}`.
pub fn unpackPatches(
    allocator: std.mem.Allocator,
    patches: []const f32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    patch_t: u32,
    patch: u32,
    channels: u32,
) ![]f32 {
    const pixel_t = latent_t * patch_t;
    const pixel_h = latent_h * patch;
    const pixel_w = latent_w * patch;
    const out = try allocator.alloc(f32, channels * pixel_t * pixel_h * pixel_w);
    @memset(out, 0);
    const width = channels * patch_t * patch * patch;
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < latent_t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < latent_h) : (hh += 1) {
            var ww: u32 = 0;
            while (ww < latent_w) : (ww += 1) {
                var src: usize = 0;
                for (0..channels) |c| {
                    for (0..patch_t) |dt| {
                        for (0..patch) |dh| {
                            for (0..patch) |dw| {
                                const pt = tt * patch_t + @as(u32, @intCast(dt));
                                const ph = hh * patch + @as(u32, @intCast(dh));
                                const pw = ww * patch + @as(u32, @intCast(dw));
                                const dst = (((c * pixel_t + pt) * pixel_h + ph) * pixel_w + pw);
                                out[dst] = patches[row * width + src];
                                src += 1;
                            }
                        }
                    }
                }
                row += 1;
            }
        }
    }
    return out;
}
