const std = @import("std");

const zml = @import("zml");

const config_mod = @import("config.zig");
const vae = @import("vae.zig");
const weights = @import("weights.zig");

const log = std.log.scoped(.minimax_h3_visual_vae);

/// Per-channel latent moments from the released `video_vae/config.json`.
pub const official_latents_mean = [24]f32{
    0.858090341091156,    -0.9606591463088989, 1.0661640167236328,   -0.5090325474739075,
    -0.2727581858634949,  -1.3675414323806763, -0.2553254961967468,  -0.26907554268836975,
    -0.5376840829849243,  -0.0464097298681736, 0.6657370328903198,   0.19690127670764923,
    -0.5460608005523682,  -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607,   -1.1206848621368408,  0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942,  0.22864092886447906,  0.7056031823158264,
};

pub const official_latents_std = [24]f32{
    1.2223774194717407, 1.2767263650894165,  1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665,   0.96531379222869875, 1.05698859691619875,
    0.841948926448822,  0.7729952931404114,  1.8955937623977661,  0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745,  0.69362932443618775,
    2.961095094680786,  2.7694199085235595,  3.0496184825897215,  2.1088054180145265,
    3.276226282119751,  3.1627357006073,     2.28168129920959475, 2.6127843856811525,
};

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
    latents_mean: [24]f32 = official_latents_mean,
    latents_std: [24]f32 = official_latents_std,

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

    fn overlay(self: FileConfig, out: *Config) void {
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
        return .{
            .w1 = linear(store, "w1.weight", "w1.bias"),
            .w2 = linear(store, "w2.weight", "w2.bias"),
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
    qkv: zml.nn.Linear,
    out: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
        return .{
            .qkv = linear(store, "to_qkv.weight", "to_qkv.bias"),
            .out = linear(store, "to_out.weight", "to_out.bias"),
            .num_heads = cfg.decoder_num_attention_heads,
            .head_dim = cfg.decoder_attention_head_dim,
            .eps = cfg.decoder_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        unloadLinear(&self.qkv);
        unloadLinear(&self.out);
    }

    pub fn forward(self: Attention, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const split = self.qkv.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 3 * self.head_dim });
        const parts = split.chunkExact(.hd, 3);
        var q = parts[0];
        var k = parts[1];
        const v = parts[2];
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
        const attn_store = store.withPrefix("attn");
        const ff_store = store.withPrefix("ff");
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

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
        const dec = decoderView(store);
        const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.decoder_num_layers));
        errdefer allocator.free(blocks);
        const block_store = dec.withPrefix("transformer_blocks");
        for (blocks, 0..) |*block, i| block.* = .init(block_store.withLayer(i), cfg);

        const post = store.withPrefix("post_quant_conv");
        return .{
            .embed = .{
                .post_quant = linear(post, "weight", "bias"),
                .proj = linear(dec.withPrefix("x_embedder"), "weight", "bias"),
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
    return store.hasKey("decoder.x_embedder.weight") and store.hasKey("post_quant_conv.weight");
}

fn decoderView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    return store.withPrefix("decoder");
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
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        var parsed_root = config_mod.parseJson(FileConfig, allocator, io, repo, "config.json") catch null;
        defer if (parsed_root) |*parsed| parsed.deinit();
        var parsed_source = config_mod.parseJson(FileConfig, allocator, io, repo, "source/config.json") catch null;
        defer if (parsed_source) |*parsed| parsed.deinit();
        var cfg = Config.official();
        if (parsed_source) |parsed| parsed.value.overlay(&cfg);
        if (parsed_root) |parsed| parsed.value.overlay(&cfg);
        log.info("visual vae: {d} layers latent_c={d} tile={d} mean0={d:.3} std0={d:.3}", .{
            cfg.decoder_num_layers,
            cfg.latent_channels,
            cfg.spec().tile_px,
            cfg.latents_mean[0],
            cfg.latents_std[0],
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
        return weights.load(allocator, io, platform, store, shardings, EmbedModel, &self.inner.embed, progress, null);
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
        return weights.load(allocator, io, platform, store, shardings, FinishModel, &self.inner.finish, progress, null);
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
