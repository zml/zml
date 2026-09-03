// video-only 768p t2v
//
//   tokenize (main)
//     → encoder     tokens → text hidden
//     → pack        text + video grid, rope / adaln / schedule
//     → dit         noise → 50 blocks → euler → video tokens
//     → unpatchify  tokens → THWC latents
//     → vae         tiled decoder → rgb
//

// ---------------------------------------------------------------------------
// shared helpers (load / compile / tiny tensor ops)
// ---------------------------------------------------------------------------

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

const EncCfg = config.EncoderConfig;
const DitCfg = config.Config;
const VaeCfg = config.VisualConfig;

const loader_opts: zml.io.Loader.Opts = .{
    .dma_chunks = 8,
    .dma_chunk_size = 64 * zml.MiB,
    .parallelism = 8,
};

fn linear(
    store: zml.io.TensorStore.View,
    weight_name: []const u8,
    bias_name: ?[]const u8,
    partitions: anytype,
    bias_partitions: anytype,
) zml.nn.Linear {
    return .init(
        store.createTensor(weight_name, .{ .dout, .d }, partitions),
        if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, bias_partitions) else null,
        .d,
    );
}

fn rms(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) zml.nn.RmsNorm {
    return .{ .weight = store.createTensor("weight", tagz, .replicated), .eps = eps };
}

fn ln(store: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = store.createTensor("weight", .{.d}, .replicated),
        .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
        .eps = eps,
    };
}

fn drop(comptime T: type, m: *zml.Bufferized(T)) void {
    zml.Buffer.deinitAll(T, m);
}

fn initLoader(allocator: std.mem.Allocator, platform: *const zml.Platform) !zml.io.Loader {
    return .init(allocator, platform, loader_opts);
}

fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    comptime T: type,
    m: *const T,
    progress: *std.Progress.Node,
    loader: ?*zml.io.Loader,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(allocator, T, m);
    if (loader) |shared| {
        shared.load(io, T, m, &buffers, store, shardings, .{ .progress = progress });
        try shared.await(io);
        return buffers;
    }
    var owned = try initLoader(allocator, platform);
    defer owned.deinit();
    owned.load(io, T, m, &buffers, store, shardings, .{ .progress = progress });
    try owned.await(io);
    return buffers;
}

fn host(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(items));
}

fn hostF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    values: []const f32,
) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return host(io, platform, shape, values),
        .bf16 => {
            const converted = try allocator.alloc(zml.floats.BFloat16, values.len);
            defer allocator.free(converted);
            for (converted, values) |*dst, src| dst.* = .fromF32(src);
            return host(io, platform, shape, converted);
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

fn compileFn(
    comptime function: anytype,
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) !zml.FnExe(function) {
    progress.increaseEstimatedTotalItems(1);
    const now: std.Io.Timestamp = .now(io, .awake);
    const exe = try zml.FnExe(function).compile(allocator, io, platform, .{ .shardings = shardings, .program_name = name }, args);
    log.info("compile {s}: ok [{f}]", .{ name, now.untilNow(io, .awake) });
    return exe;
}

fn applyLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    return lin.forward(x.convert(lin.weight.dtype())).convert(x.dtype());
}

fn shiftScale(x: zml.Tensor, shift: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const dt = x.dtype();
    return x.mul(zml.Tensor.scalar(1.0, dt).add(scale.squeeze(.k).convert(dt).broad(x.shape())))
        .add(shift.squeeze(.k).convert(dt).broad(x.shape()));
}

fn residualGate(x: zml.Tensor, g: zml.Tensor, y: zml.Tensor) zml.Tensor {
    return x.add(g.squeeze(.k).convert(y.dtype()).broad(y.shape()).mul(y));
}

// 3-axis rope: concat t/h/w freqs, then duplicate (official mm-rope)
fn ropeCat3(pos: zml.Tensor, inv: zml.Tensor) zml.Tensor {
    const parts = pos.convert(.f32).withPartialTags(.{ .s, .ax }).outer(inv).chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    return zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
}

// ===========================================================================
// 1. ENCODER
//    qwen, 50 layers. prompt tokens in, 5120-d hidden out
// ===========================================================================

const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = linear(store, "up_proj.weight", null, .{ .dout = .model }, .replicated),
            .gate_proj = linear(store, "gate_proj.weight", null, .{ .dout = .model }, .replicated),
            .down_proj = linear(store, "down_proj.weight", null, .{ .d = .model }, .replicated),
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.down_proj.forward(
            self.gate_proj.forward(x).silu().mul(self.up_proj.forward(x)).rename(.{ .dout = .d }),
        );
    }
};

// GQA 64q/8kv, causal — this is the llm, not the dit
const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: EncCfg) SelfAttn {
        return .{
            .q_proj = linear(store, "q_proj.weight", null, .{ .dout = .model }, .replicated),
            .k_proj = linear(store, "k_proj.weight", null, .{ .dout = .model }, .replicated),
            .v_proj = linear(store, "v_proj.weight", null, .{ .dout = .model }, .replicated),
            .o_proj = linear(store, "o_proj.weight", null, .{ .d = .model }, .replicated),
            .q_norm = rms(store.withPrefix("q_norm"), .{.hd}, cfg.rms_norm_eps),
            .k_norm = rms(store.withPrefix("k_norm"), .{.hd}, cfg.rms_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
        };
    }

    pub fn forward(self: SelfAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        const v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        q = zml.nn.applyRotary(self.q_norm.forward(q), cos, sin);
        k = zml.nn.applyRotary(self.k_norm.forward(k), cos, sin);
        return self.o_proj.forward(zml.attention.dense(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .vanilla,
            .{ .is_causal = true },
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } })).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const TransformerLayer = struct {
    input_layernorm: zml.nn.RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: zml.nn.RmsNorm,
    mlp: Mlp,
    pub const Input = struct { layer: TransformerLayer, hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: EncCfg) TransformerLayer {
        return .{
            .input_layernorm = rms(store.withPrefix("input_layernorm"), .{.d}, cfg.rms_norm_eps),
            .self_attn = .init(store.withPrefix("self_attn"), cfg),
            .post_attention_layernorm = rms(store.withPrefix("post_attention_layernorm"), .{.d}, cfg.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual = input.hidden.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.self_attn.forward(self.input_layernorm.forward(residual), input.cos, input.sin))
            .withPartitioning(.{ .d = .replicated });
        return .{
            .hidden = x1.add(self.mlp.forward(self.post_attention_layernorm.forward(x1)).rename(.{ .dout = .d }))
                .withPartitioning(.{ .d = .replicated })
                .reuseBuffer(input.hidden),
        };
    }
};

pub const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    pub const Input = struct { embedding: EmbedTokens, tokens: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        return .{
            .hidden = input.embedding.embed_tokens.forward(input.tokens.withPartialTags(.{.s}))
                .withPartialTags(.{.d})
                .withPartitioning(.{ .d = .replicated }),
        };
    }
};

pub const Encoder = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    cfg: EncCfg,

    pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View) !Encoder {
        const cfg: EncCfg = .{};
        const store = store_.withPrefix("model.language_model");
        const layers = try allocator.alloc(TransformerLayer, @intCast(cfg.used_hidden_layers));
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| layer.* = .init(store.withPrefix("layers").withLayer(i), cfg);
        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Encoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

pub const EncoderCompiled = struct {
    embed: zml.FnExe(EmbedTokens.forward),
    layer: zml.FnExe(TransformerLayer.forward),

    pub fn compile(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model: Encoder,
        text_len: u32,
        shardings: config.Shardings,
        progress: *std.Progress.Node,
    ) !EncoderCompiled {
        var all = shardings.all();
        var node = progress.start("Compiling MiniMax-H3 encoder", 2);
        defer node.end();
        const dt = model.embed_tokens.weight.dtype();
        const embed = try compileFn(EmbedTokens.forward, "minimax_h3_encoder_embed", allocator, io, platform, &all, progress, .{.{
            .embedding = .{ .embed_tokens = model.embed_tokens },
            .tokens = .init(.{ .b = 1, .s = text_len }, .u32),
        }});
        errdefer embed.deinit();
        const layer = try compileFn(TransformerLayer.forward, "minimax_h3_encoder_layer", allocator, io, platform, &all, progress, .{.{
            .layer = model.layers[0],
            .hidden = .init(.{ .b = 1, .s = text_len, .d = model.cfg.hidden_size }, dt),
            .cos = .init(.{ .s = text_len, .hd = model.cfg.head_dim }, dt),
            .sin = .init(.{ .s = text_len, .hd = model.cfg.head_dim }, dt),
        }});
        return .{ .embed = embed, .layer = layer };
    }

    pub fn deinit(self: *EncoderCompiled) void {
        self.embed.deinit();
        self.layer.deinit();
    }

    pub fn encodeText(
        self: *const EncoderCompiled,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model: *const Encoder,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        tokens: []const u32,
        progress: *std.Progress.Node,
    ) !zml.Buffer {
        const seq: u32 = @intCast(tokens.len);
        const head_dim: u32 = @intCast(model.cfg.head_dim);
        var token_buf = try host(io, platform, .init(.{ .b = 1, .s = tokens.len }, .u32), tokens);
        defer token_buf.deinit();

        const embed_part = EmbedTokens{ .embed_tokens = model.embed_tokens };
        var embed_bufs = try load(allocator, io, platform, store, shardings, EmbedTokens, &embed_part, progress, null);
        defer drop(EmbedTokens, &embed_bufs);
        var embed_runner = try zml.FnExe(EmbedTokens.forward).Runner(.{.embedding}).init(&self.embed, allocator, .{ .embedding = embed_bufs });
        defer embed_runner.deinit(allocator);
        var hidden: zml.Buffer = undefined;
        embed_runner.run(io, .{ .inputs = .{ .tokens = token_buf }, .outputs = .{ .hidden = &hidden }, .opts = .{ .wait = true } });
        errdefer hidden.deinit();

        // interleaved rope: each freq written to both halves of the head
        const cos = try allocator.alloc(f32, seq * head_dim);
        defer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq * head_dim);
        defer allocator.free(sin);
        const half = head_dim / 2;
        var pos_i: u32 = 0;
        while (pos_i < seq) : (pos_i += 1) {
            var f: u32 = 0;
            while (f < half) : (f += 1) {
                const ang = @as(f32, @floatFromInt(pos_i)) / std.math.pow(
                    f32,
                    model.cfg.rope_theta,
                    @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)),
                );
                cos[pos_i * head_dim + f] = @cos(ang);
                cos[pos_i * head_dim + half + f] = @cos(ang);
                sin[pos_i * head_dim + f] = @sin(ang);
                sin[pos_i * head_dim + half + f] = @sin(ang);
            }
        }
        var cos_buf = try hostF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, model.embed_tokens.weight.dtype()), cos);
        defer cos_buf.deinit();
        var sin_buf = try hostF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, model.embed_tokens.weight.dtype()), sin);
        defer sin_buf.deinit();

        var loader = try initLoader(allocator, platform);
        defer loader.deinit();
        const LayerRunner = zml.FnExe(TransformerLayer.forward).Runner(.{.layer});
        var layer_runner: ?LayerRunner = null;
        defer if (layer_runner) |*r| r.deinit(allocator);
        for (0..model.layers.len) |layer_i| {
            var layer_bufs = try load(allocator, io, platform, store, shardings, TransformerLayer, &model.layers[layer_i], progress, &loader);
            defer drop(TransformerLayer, &layer_bufs);
            if (layer_runner) |*r| r.rebake(.{ .layer = layer_bufs }) else layer_runner = try LayerRunner.init(&self.layer, allocator, .{ .layer = layer_bufs });
            var next: zml.Buffer = undefined;
            layer_runner.?.run(io, .{
                .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden.deinit();
            hidden = next;
        }
        return hidden;
    }
};

// ===========================================================================
// 2. PACK
//    sequence order: text tokens, then the video grid.
//    also builds rope positions, modality tags, and the rf sigma schedule
// ===========================================================================

pub const Layout = struct {
    positions: []f32, // [s, 3] t/h/w
    token_tags: []u8,
    text_indices: []u32,
    video_indices: []u32,

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.token_tags);
        allocator.free(self.text_indices);
        allocator.free(self.video_indices);
    }

    pub fn seqLen(self: Layout) u32 {
        return @intCast(self.token_tags.len);
    }

    pub fn writeAdalnIndices(self: Layout, out: []u32, timestep_indices: []const u32) void {
        for (out, timestep_indices, self.token_tags) |*a, t, tag| {
            a.* = t * @as(u32, @intCast(config.modality_count)) + tag;
        }
    }
};

const video_spans = [_]u32{ 1, 4, 4, 4, 4 };
const frame_rescale: f64 = 5.0 / 3.0;

fn spatialAxis(dim: u32, sqrt_area: f64, out: []f32) []f32 {
    const count = dim / 2;
    const ratio = @as(f64, @floatFromInt(dim)) / sqrt_area;
    const left = (1.0 - ratio) / 2.0;
    const step = ratio / @as(f64, @floatFromInt(count));
    for (0..count) |i| out[i] = @floatCast((left + @as(f64, @floatFromInt(i)) * step) * 32.0);
    return out[0..count];
}

pub const Schedule = struct {
    sigmas: []f32,

    pub fn init(allocator: std.mem.Allocator, shift: f32, n: u32) !Schedule {
        const sigmas = try allocator.alloc(f32, n);
        for (sigmas, 0..) |*sigma, i| {
            const base = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
            sigma.* = shift * base / (1.0 + (shift - 1.0) * base);
        }
        return .{ .sigmas = sigmas };
    }

    pub fn deinit(self: Schedule, allocator: std.mem.Allocator) void {
        allocator.free(self.sigmas);
    }

    pub fn stepCount(self: Schedule) usize {
        return self.sigmas.len - 1;
    }

    pub fn t(self: Schedule, i: usize) f32 {
        return 1.0 - self.sigmas[i];
    }
};

pub const Packed = struct {
    layout: Layout,
    video: Schedule,

    pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
        self.layout.deinit(allocator);
        self.video.deinit(allocator);
    }
};

// text tokens first, then video grid. video rope t starts at L.
pub fn pack(allocator: std.mem.Allocator, geo: config.Geometry, text_len: u32, steps: u32) !Packed {
    const video = try Schedule.init(allocator, config.video_shift, steps);
    errdefer video.deinit(allocator);
    const n = text_len + geo.video_tokens;
    const positions = try allocator.alloc(f32, n * 3);
    errdefer allocator.free(positions);
    const token_tags = try allocator.alloc(u8, n);
    errdefer allocator.free(token_tags);
    const text_indices = try allocator.alloc(u32, text_len);
    errdefer allocator.free(text_indices);
    const video_indices = try allocator.alloc(u32, geo.video_tokens);
    errdefer allocator.free(video_indices);

    const sqrt_area = @sqrt(@as(f64, @floatFromInt(geo.latent_h * geo.latent_w)));
    var h_buf: [256]f32 = undefined;
    var w_buf: [256]f32 = undefined;
    const h_axis = spatialAxis(geo.latent_h, sqrt_area, &h_buf);
    const w_axis = spatialAxis(geo.latent_w, sqrt_area, &w_buf);

    for (0..text_len) |i| {
        positions[i * 3 + 0] = @floatFromInt(i);
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = 0;
        token_tags[i] = 1; // text
        text_indices[i] = @intCast(i);
    }

    var cursor: f64 = @floatFromInt(text_len);
    var v: u32 = 0;
    for (0..geo.latent_t) |ti| {
        for (h_axis) |h| {
            for (w_axis) |w| {
                const idx = text_len + v;
                video_indices[v] = idx;
                positions[idx * 3 + 0] = @floatCast(cursor);
                positions[idx * 3 + 1] = h;
                positions[idx * 3 + 2] = w;
                token_tags[idx] = 0; // video
                v += 1;
            }
        }
        cursor += frame_rescale * @as(f64, @floatFromInt(video_spans[ti % video_spans.len]));
    }

    return .{
        .layout = .{ .positions = positions, .token_tags = token_tags, .text_indices = text_indices, .video_indices = video_indices },
        .video = video,
    };
}

// ===========================================================================
// 3. DIT
//    start from N(0,1) latents (official nchw order so --seed matches python),
//    patchify 1x2x2, run 50 blocks in groups of 10, euler step
// ===========================================================================

fn thwcAt(tt: u32, hh: u32, ww: u32, ch: usize, h: u32, w: u32, c: u32) usize {
    return (((@as(usize, tt) * h + hh) * w + ww) * c) + ch;
}

fn patchWalk(t: u32, h: u32, w: u32, c: u32, patch: [3]i64, comptime write: bool, src: []const f32, dst: []f32) void {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    const width = c * pt * ph * pw;
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < t) : (tt += pt) {
        var hh: u32 = 0;
        while (hh < h) : (hh += ph) {
            var ww: u32 = 0;
            while (ww < w) : (ww += pw) {
                var i: usize = 0;
                for (0..c) |ch| {
                    for (0..pt) |dt| {
                        for (0..ph) |dh| {
                            for (0..pw) |dw| {
                                const base = thwcAt(tt + @as(u32, @intCast(dt)), hh + @as(u32, @intCast(dh)), ww + @as(u32, @intCast(dw)), ch, h, w, c);
                                if (write) dst[row * width + i] = src[base] else dst[base] = src[row * width + i];
                                i += 1;
                            }
                        }
                    }
                }
                row += 1;
            }
        }
    }
}

fn patchify(allocator: std.mem.Allocator, src: []const f32, t: u32, h: u32, w: u32, c: u32, patch: [3]i64) ![]f32 {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    const out = try allocator.alloc(f32, (t / pt) * (h / ph) * (w / pw) * c * pt * ph * pw);
    patchWalk(t, h, w, c, patch, true, src, out);
    return out;
}

// official torch.randn is NCHW (24,T,H,W). walk that order so --seed matches python, then 1x2x2 patchify.
fn noise(allocator: std.mem.Allocator, seed: u64, t: u32, h: u32, w: u32, patch: [3]i64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const c: u32 = 24;
    const thwc = try allocator.alloc(f32, @as(usize, c) * t * h * w);
    defer allocator.free(thwc);
    for (0..c) |ci| {
        for (0..t) |ti| {
            for (0..h) |hi| {
                for (0..w) |wi| {
                    thwc[thwcAt(@intCast(ti), @intCast(hi), @intCast(wi), ci, h, w, c)] = rng.floatNorm(f32);
                }
            }
        }
    }
    return patchify(allocator, thwc, t, h, w, c, patch);
}

// --- weights: time embed, adaln, 50 blocks, text refiner, final ---

const group_size: u32 = 10;

const SwiGlu = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        return .{
            .fc1 = linear(store, "net.0.proj.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
            .fc2 = linear(store, "net.2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
        };
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const value, const gate = self.fc1.forward(x).chunkExact(-1, 2);
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

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) Attention {
        const qkv = .{ .dout = .model, .d = .replicated };
        return .{
            .q = linear(store, "to_q.weight", null, qkv, .replicated),
            .k = linear(store, "to_k.weight", null, qkv, .replicated),
            .v = linear(store, "to_v.weight", null, qkv, .replicated),
            .out = linear(store, "to_out.0.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            .q_norm = rms(store.withPrefix("norm_q"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = rms(store.withPrefix("norm_k"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        var k = self.k.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        const v = self.v.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);
        if (rotary) |pe| {
            q = zml.nn.applyRotary(q, pe[0], pe[1]);
            k = zml.nn.applyRotary(k, pe[0], pe[1]);
        }
        return self.out.forward(zml.attention.dense(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            self.attn_backend,
            .{ .is_causal = false },
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } })).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const TimeEmbedder = struct {
    proj_in: zml.nn.Linear,
    proj_out: zml.nn.Linear,
    pub const Input = struct { model: TimeEmbedder, timestep: zml.Tensor, freq_dim: i64 };
    pub const Output = struct { temb: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) TimeEmbedder {
        const p = store.withPrefix("time_embedder");
        return .{
            .proj_in = linear(p, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .proj_out = linear(p, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
        };
    }

    pub fn outDim(self: TimeEmbedder) i64 {
        return self.proj_out.weight.dim(.dout);
    }

    pub fn forward(input: Input) Output {
        const inv = zml.nn.invFreq(input.freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
        }).withTags(.{.f});
        const angles = input.timestep.convert(.f32).withPartialTags(.{.n}).outer(inv);
        const features = zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
        return .{
            .temb = input.model.proj_out.forward(
                input.model.proj_in.forward(features).silu().rename(.{ .dout = .d }),
            ).rename(.{ .dout = .d }),
        };
    }
};

pub const AdaLn = struct {
    linear: zml.nn.Linear,
    hidden_size: i64,
    expand: i64,
    modalities: i64,
    pub const PrepareInput = struct { adaln: AdaLn, temb: zml.Tensor, steps: i64, slots: i64 };
    pub const PrepareOutput = struct { table: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        return .{
            .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
            .hidden_size = hidden_size,
            .expand = expand,
            .modalities = modalities,
        };
    }

    pub fn forward(self: AdaLn, temb: zml.Tensor) zml.Tensor {
        const raw = self.linear.forward(temb.silu().convert(self.linear.weight.dtype()));
        return if (self.modalities == 1)
            raw.splitAxis(.dout, .{ .k = self.expand, .d = self.hidden_size })
        else
            raw.splitAxis(.dout, .{ .mod = self.modalities, .k = self.expand, .d = self.hidden_size });
    }

    pub fn prepare(input: PrepareInput) PrepareOutput {
        return .{ .table = input.adaln.forward(input.temb).splitAxis(.n, .{ .t = input.steps, .n = input.slots }) };
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
        table: zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const gathered = input.table.gather(.{ .t = input.step }, .{});
        const mods = if (gathered.shape().hasTag(.mod)) |_| gathered.merge(.{ .n = .{ .n, .mod } }) else gathered;
        const parts = mods.gather(.{ .n = input.adaln_indices }, .{}).chunkExact(.k, 6); // shift/scale/gate × attn,mlp
        const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
        const attn_out = self.attn.forward(shiftScale(self.norm1.forward(residual0), parts[0], parts[1]), .{ input.cos, input.sin });
        const x1 = residualGate(residual0, parts[2], attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(shiftScale(self.norm2.forward(x1), parts[3], parts[4])).rename(.{ .dout = .d });
        return .{
            .hidden = residualGate(x1, parts[5], mlp_out).withPartitioning(.{ .d = .replicated }).reuseBuffer(input.hidden),
        };
    }
};

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
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        var hidden = input.hidden;
        for (input.group.layers, input.tables) |layer, table| {
            hidden = BlockCore.forward(.{
                .layer = layer,
                .hidden = hidden,
                .table = table,
                .step = input.step,
                .adaln_indices = input.adaln_indices,
                .cos = input.cos,
                .sin = input.sin,
            }).hidden;
        }
        return .{ .hidden = hidden };
    }
};

pub const DitBlock = struct {
    core: BlockCore,
    adaln: AdaLn,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) DitBlock {
        return .{
            .core = .{
                .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
                .attn = .init(store.withPrefix("attn"), cfg),
                .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
                .mlp = .init(store.withPrefix("ff")),
                .hidden_size = cfg.hidden_size,
            },
            .adaln = .init(store.withPrefix("adaln_proj"), cfg.hidden_size, 6, config.modality_count),
        };
    }
};

const TokenRefinerBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) TokenRefinerBlock {
        return .{
            .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(store.withPrefix("attn"), cfg),
            .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(store.withPrefix("ff")),
        };
    }

    pub fn forward(self: TokenRefinerBlock, x: zml.Tensor) zml.Tensor {
        const residual = x.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual), null));
        return x1.add(self.mlp.forward(self.norm2.forward(x1)).rename(.{ .dout = .d }))
            .withPartitioning(.{ .d = .replicated })
            .reuseBuffer(x);
    }
};

const FinalLayer = struct {
    norm: zml.nn.RmsNorm,
    adaln: AdaLn,
    video_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) FinalLayer {
        return .{
            .norm = rms(store.withPrefix("norm_out.norm"), .{.d}, cfg.final_norm_eps),
            .adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size, 2, 1),
            .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
        };
    }
};

pub const Dit = struct {
    video_proj: zml.nn.Linear,
    condition_proj: zml.nn.Linear,
    time_embedder: TimeEmbedder,
    refiner_blocks: []TokenRefinerBlock,
    refiner_norm: zml.nn.RmsNorm,
    blocks: []DitBlock,
    final_layer: FinalLayer,
    cfg: DitCfg,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Dit {
        const cfg: DitCfg = .{};
        const blocks = try allocator.alloc(DitBlock, @intCast(cfg.num_layers));
        errdefer allocator.free(blocks);
        const refiner = store.withPrefix("token_refiner");
        const refiner_blocks = try allocator.alloc(TokenRefinerBlock, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(refiner_blocks);
        for (refiner_blocks, 0..) |*block, i| block.* = .init(refiner.withPrefix("refiner_blocks").withLayer(i), cfg);
        for (blocks, 0..) |*block, i| block.* = .init(store.withPrefix("transformer_blocks").withLayer(i), cfg);
        return .{
            .video_proj = linear(store, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
            .condition_proj = linear(store, "context_embedder.weight", "context_embedder.bias", .replicated, .replicated),
            .time_embedder = .init(store),
            .refiner_blocks = refiner_blocks,
            .refiner_norm = rms(refiner.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
            .blocks = blocks,
            .final_layer = FinalLayer.init(store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Dit, allocator: std.mem.Allocator) void {
        allocator.free(self.refiner_blocks);
        allocator.free(self.blocks);
    }

    pub fn textPrep(self: Dit) TextPrep {
        return .{ .condition_proj = self.condition_proj, .blocks = self.refiner_blocks, .final_norm = self.refiner_norm };
    }

    pub fn patchEmbed(self: Dit) PatchEmbed {
        return .{ .video_proj = self.video_proj, .hidden_size = self.cfg.hidden_size, .seq = 0 };
    }

    pub fn finishCore(self: Dit) FinishCore {
        return .{ .norm = self.final_layer.norm, .video_out = self.final_layer.video_out };
    }
};

pub const TextPrep = struct {
    condition_proj: zml.nn.Linear,
    blocks: []TokenRefinerBlock,
    final_norm: zml.nn.RmsNorm,
    pub const Input = struct { model: TextPrep, text: zml.Tensor };
    pub const Output = struct { text: zml.Tensor };

    fn unload(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
        zml.nn.Linear.unloadBuffers(&self.condition_proj);
        for (self.blocks) |*block| drop(TokenRefinerBlock, block);
        allocator.free(self.blocks);
        self.final_norm.weight.deinit();
    }

    pub fn forward(input: Input) Output {
        var text = input.model.condition_proj.forward(input.text.convert(input.model.condition_proj.weight.dtype())).rename(.{ .dout = .d });
        text = text.convert(input.model.final_norm.weight.dtype());
        for (input.model.blocks) |block| text = block.forward(text);
        return .{ .text = input.model.final_norm.forward(text) };
    }
};

pub const PatchEmbed = struct {
    video_proj: zml.nn.Linear,
    hidden_size: i64,
    seq: i64 = 0,
    pub const Input = struct {
        model: PatchEmbed,
        video: zml.Tensor,
        text: zml.Tensor,
        video_indices: zml.Tensor,
        text_indices: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const video = input.model.video_proj.forward(input.video.convert(input.model.video_proj.weight.dtype())).rename(.{ .dout = .d });
        var hidden = zml.Tensor.zeroes(zml.Shape.init(
            .{ .b = input.text.dim(.b), .s = input.model.seq, .d = input.model.hidden_size },
            input.text.dtype(),
        ));
        hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, input.text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video.convert(input.text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
        return .{ .hidden = hidden.withPartitioning(.{ .d = .replicated }) };
    }
};

pub const FinishCore = struct {
    norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    pub const Input = struct {
        model: FinishCore,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
        video_indices: zml.Tensor,
    };
    pub const Output = struct { video: zml.Tensor };

    pub fn forward(input: Input) Output {
        const n = input.model.norm.forward(
            input.hidden.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }).withPartitioning(.{ .d = .replicated }),
        );
        const selected = input.table.gather(.{ .t = input.step }, .{}).gather(.{
            .n = input.timestep_indices.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }),
        }, .{});
        const parts = selected.chunkExact(.k, 2);
        return .{ .video = input.model.video_out.forward(shiftScale(n, parts[0], parts[1]).convert(input.model.video_out.weight.dtype())) };
    }
};

const Rope = struct {
    pub const Input = struct { position_ids: zml.Tensor, rope_freq_dim: i64, rope_theta: f32, out_dtype: zml.DataType };
    pub const Output = struct { cos: zml.Tensor, sin: zml.Tensor };

    pub fn forward(input: Input) Output {
        const emb = ropeCat3(input.position_ids, zml.nn.invFreq(2 * input.rope_freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = input.rope_theta } },
        }).withTags(.{.f}));
        return .{ .cos = emb.cos().convert(input.out_dtype), .sin = emb.sin().convert(input.out_dtype) };
    }
};

// x0 = x + sigma*v, then lerp toward sigma_next (eta=0)
const Euler = struct {
    pub const Input = struct { sample: zml.Tensor, velocity: zml.Tensor, sigma: zml.Tensor, sigma_next: zml.Tensor };
    pub const Output = struct { sample: zml.Tensor };

    pub fn apply(input: Input) Output {
        const sample = input.sample;
        const denoised = sample.add(input.velocity.convert(sample.dtype()).mul(input.sigma.convert(sample.dtype()).broad(sample.shape())));
        const ratio = input.sigma_next.convert(.f32).div(input.sigma.convert(.f32)).broad(sample.convert(.f32).shape());
        return .{
            .sample = ratio.mul(sample.convert(.f32))
                .add(zml.Tensor.scalar(1.0, .f32).sub(ratio).mul(denoised.convert(.f32)))
                .convert(sample.dtype())
                .reuseBuffer(input.sample),
        };
    }
};

pub const DitCompiled = struct {
    prepare_text: zml.FnExe(TextPrep.forward),
    prepare_rope: zml.FnExe(Rope.forward),
    embed_patches: zml.FnExe(PatchEmbed.forward),
    prepare_temb: zml.FnExe(TimeEmbedder.forward),
    prepare_adaln: zml.FnExe(AdaLn.prepare),
    prepare_final_adaln: zml.FnExe(AdaLn.prepare),
    block_group: zml.FnExe(BlockGroup.forward),
    finish: zml.FnExe(FinishCore.forward),
    apply_video: zml.FnExe(Euler.apply),

    pub fn deinit(self: *DitCompiled) void {
        self.prepare_text.deinit();
        self.prepare_rope.deinit();
        self.embed_patches.deinit();
        self.prepare_temb.deinit();
        self.prepare_adaln.deinit();
        self.prepare_final_adaln.deinit();
        self.block_group.deinit();
        self.finish.deinit();
        self.apply_video.deinit();
    }
};

// --- compile + 30-step euler (attn is fa2; sdpa OOM on 37k tokens) ---

pub fn compileDit(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    dit_model: Dit,
    geo: config.Geometry,
    text_len: u32,
    seq_len: u32,
    steps: u32,
    shardings: config.Shardings,
    text_dt: zml.DataType,
    progress: *std.Progress.Node,
) !DitCompiled {
    var model = dit_model;
    const attn = zml.attention.Backend.auto(platform);
    for (model.blocks) |*block| block.core.attn.attn_backend = attn;
    log.info("dit attn={s} group={d} seq={d} devices={d}", .{
        @tagName(attn),
        group_size,
        seq_len,
        platform.devices.len,
    });
    var all = shardings.all();
    var node = progress.start("Compiling MiniMax-H3 DiT", 9);
    defer node.end();
    const dt = model.blocks[0].core.norm1.weight.dtype();
    var patch_part = model.patchEmbed();
    patch_part.seq = seq_len;

    const prepare_text = try compileFn(TextPrep.forward, "minimax_h3_prepare_text", allocator, io, platform, &all, progress, .{.{
        .model = model.textPrep(),
        .text = .init(.{ .b = 1, .s = text_len, .d = model.cfg.text_dim }, text_dt),
    }});
    errdefer prepare_text.deinit();
    const prepare_rope = try compileFn(Rope.forward, "minimax_h3_prepare_rope", allocator, io, platform, &all, progress, .{.{
        .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
        .rope_freq_dim = model.cfg.rope_freq_dim,
        .rope_theta = model.cfg.rope_theta,
        .out_dtype = dt,
    }});
    errdefer prepare_rope.deinit();
    const embed_patches = try compileFn(PatchEmbed.forward, "minimax_h3_embed_patches", allocator, io, platform, &all, progress, .{.{
        .model = patch_part,
        .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .text = .init(.{ .b = 1, .s = text_len, .d = model.cfg.hidden_size }, dt),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .text_indices = .init(.{ .s = text_len }, .u32),
    }});
    errdefer embed_patches.deinit();
    const prepare_temb = try compileFn(TimeEmbedder.forward, "minimax_h3_prepare_temb", allocator, io, platform, &all, progress, .{.{
        .model = model.time_embedder,
        .timestep = .init(.{ .n = steps }, .f32),
        .freq_dim = model.cfg.freq_dim,
    }});
    errdefer prepare_temb.deinit();
    const prepare_adaln = try compileFn(AdaLn.prepare, "minimax_h3_prepare_adaln", allocator, io, platform, &all, progress, .{.{
        .adaln = model.blocks[0].adaln,
        .temb = .init(.{ .n = steps, .d = model.time_embedder.outDim() }, .f32),
        .steps = steps,
        .slots = 1,
    }});
    errdefer prepare_adaln.deinit();
    const prepare_final_adaln = try compileFn(AdaLn.prepare, "minimax_h3_prepare_final_adaln", allocator, io, platform, &all, progress, .{.{
        .adaln = model.final_layer.adaln,
        .temb = .init(.{ .n = steps, .d = model.time_embedder.outDim() }, .f32),
        .steps = steps,
        .slots = 1,
    }});
    errdefer prepare_final_adaln.deinit();

    const layers = try allocator.alloc(BlockCore, group_size);
    defer allocator.free(layers);
    const tables = try allocator.alloc(zml.Tensor, group_size);
    defer allocator.free(tables);
    for (layers, tables, 0..) |*layer, *tab, i| {
        layer.* = model.blocks[i].core;
        tab.* = zml.Tensor.init(.{ .t = steps, .n = 1, .mod = config.modality_count, .k = 6, .d = model.cfg.hidden_size }, dt);
    }
    const block_group = try compileFn(BlockGroup.forward, "minimax_h3_block_group", allocator, io, platform, &all, progress, .{.{
        .group = .{ .layers = layers },
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = model.cfg.hidden_size }, dt),
        .tables = tables,
        .step = zml.Tensor.init(.{}, .u32),
        .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
        .cos = zml.Tensor.init(.{ .s = seq_len, .f = model.cfg.rotaryDim() }, dt),
        .sin = zml.Tensor.init(.{ .s = seq_len, .f = model.cfg.rotaryDim() }, dt),
    }});
    errdefer block_group.deinit();
    const finish_exe = try compileFn(FinishCore.forward, "minimax_h3_finish", allocator, io, platform, &all, progress, .{.{
        .model = model.finishCore(),
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = model.cfg.hidden_size }, dt),
        .table = zml.Tensor.init(.{ .t = steps, .n = 1, .k = 2, .d = model.cfg.hidden_size }, dt),
        .step = zml.Tensor.init(.{}, .u32),
        .timestep_indices = .init(.{ .s = seq_len }, .u32),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
    }});
    errdefer finish_exe.deinit();
    const apply_video = try compileFn(Euler.apply, "minimax_h3_apply_video", allocator, io, platform, &all, progress, .{.{
        .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
    }});
    return .{
        .prepare_text = prepare_text,
        .prepare_rope = prepare_rope,
        .embed_patches = embed_patches,
        .prepare_temb = prepare_temb,
        .prepare_adaln = prepare_adaln,
        .prepare_final_adaln = prepare_final_adaln,
        .block_group = block_group,
        .finish = finish_exe,
        .apply_video = apply_video,
    };
}

fn scalarF32(io: std.Io, platform: *const zml.Platform, value: f32) !zml.Buffer {
    var item: f32 = value;
    return zml.Buffer.fromBytes(io, platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&item));
}

pub fn denoise(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const DitCompiled,
    model: *const Dit,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    geo: config.Geometry,
    text: zml.Buffer,
    text_len: u32,
    packed_run: Packed,
    seed: u64,
    progress: *std.Progress.Node,
) ![]f32 {
    const video = try noise(allocator, seed, geo.latent_t, geo.latent_h, geo.latent_w, model.cfg.patch_size);
    errdefer allocator.free(video);
    const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
    const layout = packed_run.layout;
    const seq = layout.seqLen();
    const steps = packed_run.video.stepCount();
    const n_blocks = model.blocks.len;
    std.debug.assert(n_blocks % group_size == 0);

    const flat_t = try allocator.alloc(f32, steps);
    defer allocator.free(flat_t);
    for (flat_t, packed_run.video.sigmas[0..steps]) |*tv, s| tv.* = 1.0 - s;
    const tidx = try allocator.alloc(u32, seq);
    defer allocator.free(tidx);
    const adaln = try allocator.alloc(u32, seq);
    defer allocator.free(adaln);
    @memset(tidx, 0);
    layout.writeAdalnIndices(adaln, tidx);

    var pos_buf = try host(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), layout.positions);
    defer pos_buf.deinit();
    var video_idx = try host(io, platform, .init(.{ .s = geo.video_tokens }, .u32), layout.video_indices);
    defer video_idx.deinit();
    var text_idx = try host(io, platform, .init(.{ .s = text_len }, .u32), layout.text_indices);
    defer text_idx.deinit();
    var adaln_buf = try host(io, platform, .init(.{ .s = seq }, .u32), adaln);
    defer adaln_buf.deinit();
    var time_idx = try host(io, platform, .init(.{ .s = seq }, .u32), tidx);
    defer time_idx.deinit();

    const text_part = model.textPrep();
    var text_bufs = try load(allocator, io, platform, store, shardings, TextPrep, &text_part, progress, null);
    defer TextPrep.unload(&text_bufs, allocator);
    var text_runner = try zml.FnExe(TextPrep.forward).Runner(.{.model}).init(&compiled.prepare_text, allocator, .{ .model = text_bufs });
    defer text_runner.deinit(allocator);
    var refined_text: zml.Buffer = undefined;
    text_runner.run(io, .{ .inputs = .{ .text = text }, .outputs = .{ .text = &refined_text }, .opts = .{ .wait = true } });
    defer refined_text.deinit();

    var rope_runner = try zml.FnExe(Rope.forward).Runner(.{}).init(&compiled.prepare_rope, allocator, .{});
    defer rope_runner.deinit(allocator);
    var cos: zml.Buffer = undefined;
    var sin: zml.Buffer = undefined;
    rope_runner.run(io, .{ .inputs = .{ .position_ids = pos_buf }, .outputs = .{ .cos = &cos, .sin = &sin }, .opts = .{ .wait = true } });
    defer cos.deinit();
    defer sin.deinit();

    var flat_buf = try host(io, platform, .init(.{ .n = steps }, .f32), flat_t);
    defer flat_buf.deinit();
    var time_bufs = try load(allocator, io, platform, store, shardings, TimeEmbedder, &model.time_embedder, progress, null);
    var all_temb: zml.Buffer = undefined;
    {
        var temb_runner = try zml.FnExe(TimeEmbedder.forward).Runner(.{.model}).init(&compiled.prepare_temb, allocator, .{ .model = time_bufs });
        defer temb_runner.deinit(allocator);
        temb_runner.run(io, .{ .inputs = .{ .timestep = flat_buf }, .outputs = .{ .temb = &all_temb }, .opts = .{ .wait = true } });
    }
    defer all_temb.deinit();
    drop(TimeEmbedder, &time_bufs);

    var tables = try allocator.alloc(zml.Buffer, n_blocks);
    var tables_filled: usize = 0;
    errdefer {
        for (tables[0..tables_filled]) |*tb| tb.deinit();
        allocator.free(tables);
    }
    var cores = try allocator.alloc(zml.Bufferized(BlockCore), n_blocks);
    var cores_filled: usize = 0;
    errdefer {
        for (cores[0..cores_filled]) |*core| drop(BlockCore, core);
        allocator.free(cores);
    }
    var loader = try initLoader(allocator, platform);
    defer loader.deinit();
    const AdaLnRunner = zml.FnExe(AdaLn.prepare).Runner(.{.adaln});
    var adaln_runner: ?AdaLnRunner = null;
    defer if (adaln_runner) |*r| r.deinit(allocator);
    var prev_adaln: ?zml.Bufferized(AdaLn) = null;
    defer if (prev_adaln) |*a| drop(AdaLn, a);
    for (0..n_blocks) |block_i| {
        const adaln_bufs = try load(allocator, io, platform, store, shardings, AdaLn, &model.blocks[block_i].adaln, progress, &loader);
        if (adaln_runner) |*r| {
            r.rebake(.{ .adaln = adaln_bufs });
            if (prev_adaln) |*a| drop(AdaLn, a);
        } else {
            adaln_runner = try AdaLnRunner.init(&compiled.prepare_adaln, allocator, .{ .adaln = adaln_bufs });
        }
        prev_adaln = adaln_bufs;
        var table: zml.Buffer = undefined;
        adaln_runner.?.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = &table }, .opts = .{ .wait = true } });
        tables[block_i] = table;
        tables_filled += 1;
        cores[block_i] = try load(allocator, io, platform, store, shardings, BlockCore, &model.blocks[block_i].core, progress, &loader);
        cores_filled += 1;
    }

    var final_table: zml.Buffer = undefined;
    {
        var final_adaln = try load(allocator, io, platform, store, shardings, AdaLn, &model.final_layer.adaln, progress, null);
        var final_runner = try AdaLnRunner.init(&compiled.prepare_final_adaln, allocator, .{ .adaln = final_adaln });
        defer final_runner.deinit(allocator);
        final_runner.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = &final_table }, .opts = .{ .wait = true } });
        drop(AdaLn, &final_adaln);
    }
    defer final_table.deinit();

    const patch_part = model.patchEmbed();
    var patch_bufs = try load(allocator, io, platform, store, shardings, PatchEmbed, &patch_part, progress, null);
    defer drop(PatchEmbed, &patch_bufs);
    var patch_runner = try zml.FnExe(PatchEmbed.forward).Runner(.{.model}).init(&compiled.embed_patches, allocator, .{ .model = patch_bufs });
    defer patch_runner.deinit(allocator);
    const finish_part = model.finishCore();
    var finish_bufs = try load(allocator, io, platform, store, shardings, FinishCore, &finish_part, progress, null);
    defer drop(FinishCore, &finish_bufs);
    var finish_runner = try zml.FnExe(FinishCore.forward).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = finish_bufs });
    defer finish_runner.deinit(allocator);

    const GroupRunner = zml.FnExe(BlockGroup.forward).Runner(.{.group});
    const n_groups = n_blocks / group_size;
    const group_runners = try allocator.alloc(GroupRunner, n_groups);
    var runners_ready: usize = 0;
    defer {
        for (group_runners[0..runners_ready]) |*r| r.deinit(allocator);
        allocator.free(group_runners);
    }
    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        group_runners[g] = try GroupRunner.init(&compiled.block_group, allocator, .{
            .group = .{ .layers = cores[g * group_size ..][0..group_size] },
        });
        runners_ready += 1;
    }
    var apply_v = try zml.FnExe(Euler.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
    defer apply_v.deinit(allocator);
    var video_buf = try host(io, platform, video_shape, video);
    defer video_buf.deinit();

    log.info("denoise: blocks={d} groups={d} seq={d} devices={d}", .{
        n_blocks,
        n_groups,
        seq,
        platform.devices.len,
    });
    const denoise_start: std.Io.Timestamp = .now(io, .awake);
    var step_i: usize = 0;
    while (step_i < steps) : (step_i += 1) {
        const step_start: std.Io.Timestamp = .now(io, .awake);
        var step_u32: u32 = @intCast(step_i);
        var step_buf = try zml.Buffer.fromBytes(io, platform, .init(.{}, .u32), .replicated, std.mem.asBytes(&step_u32));
        defer step_buf.deinit();
        var sigma_v = try scalarF32(io, platform, packed_run.video.sigmas[step_i]);
        defer sigma_v.deinit();
        var sigma_v_next = try scalarF32(io, platform, packed_run.video.sigmas[step_i + 1]);
        defer sigma_v_next.deinit();

        var hidden: zml.Buffer = undefined;
        patch_runner.run(io, .{
            .inputs = .{ .video = video_buf, .text = refined_text, .video_indices = video_idx, .text_indices = text_idx },
            .outputs = .{ .hidden = &hidden },
            .opts = .{ .wait = true },
        });
        var held: std.ArrayList(zml.Buffer) = .empty;
        defer {
            for (held.items) |*buf| buf.deinit();
            held.deinit(allocator);
        }
        try held.append(allocator, hidden);

        g = 0;
        while (g < n_groups) : (g += 1) {
            var next: zml.Buffer = undefined;
            group_runners[g].run(io, .{
                .inputs = .{
                    .hidden = hidden,
                    .tables = tables[g * group_size ..][0..group_size],
                    .step = step_buf,
                    .adaln_indices = adaln_buf,
                    .cos = cos,
                    .sin = sin,
                },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden = next;
            try held.append(allocator, next);
        }

        var video_out: zml.Buffer = undefined;
        finish_runner.run(io, .{
            .inputs = .{ .hidden = hidden, .table = final_table, .step = step_buf, .timestep_indices = time_idx, .video_indices = video_idx },
            .outputs = .{ .video = &video_out },
            .opts = .{ .wait = true },
        });
        defer video_out.deinit();

        var next_video: zml.Buffer = undefined;
        apply_v.run(io, .{
            .inputs = .{ .sample = video_buf, .velocity = video_out, .sigma = sigma_v, .sigma_next = sigma_v_next },
            .outputs = .{ .sample = &next_video },
            .opts = .{ .wait = true },
        });
        video_buf.deinit();
        video_buf = next_video;
        log.info("denoise {d}/{d} t={d:.4} [{f}]", .{
            step_i + 1,
            steps,
            packed_run.video.t(step_i),
            step_start.untilNow(io, .awake),
        });
    }

    try video_buf.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video)));
    log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });
    for (tables) |*tb| tb.deinit();
    allocator.free(tables);
    for (cores) |*core| drop(BlockCore, core);
    allocator.free(cores);
    return video;
}

// ===========================================================================
// 4. UNPATCHIFY
//    dit tokens {s, 96} → THWC latents the vae wants
// ===========================================================================

pub fn unpatchify(allocator: std.mem.Allocator, src: []const f32, t: u32, h: u32, w: u32, c: u32, patch: [3]i64) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, t) * h * w * c);
    patchWalk(t, h, w, c, patch, false, src, out);
    return out;
}

// ===========================================================================
// 5. VAE
//    tiled vit decoder. denorm latents, 256px tiles, imagenet undo → rgb
// ===========================================================================

// --- tile / stitch (256px, 64 overlap) ---

const imagenet_mean = [_]f32{ 0.485, 0.456, 0.406 };
const imagenet_std = [_]f32{ 0.229, 0.224, 0.225 };

const tile_px: u32 = 256;
const tile_overlap_px: u32 = 64;
const token_drop: u32 = 3;
const chunk: u32 = 5;
const frame_pre: u32 = 3;
const frame_ov: u32 = 5;
const vae_t: u32 = 7;
const vae_h: u32 = 16;
const vae_w: u32 = 16;

const TilePlan = struct {
    starts: []u32,
    overlaps: []u32,

    pub fn deinit(self: TilePlan, allocator: std.mem.Allocator) void {
        allocator.free(self.starts);
        allocator.free(self.overlaps);
    }
};

fn splitTiles(allocator: std.mem.Allocator, length: u32, tile_size: u32, min_overlap: u32, align_to: u32) !TilePlan {
    if (tile_size >= length) {
        const starts = try allocator.alloc(u32, 1);
        starts[0] = 0;
        return .{ .starts = starts, .overlaps = try allocator.alloc(u32, 0) };
    }
    var num_tiles = std.math.divCeil(u32, length, tile_size) catch unreachable;
    while (tile_size * num_tiles < min_overlap * (num_tiles - 1) + length) num_tiles += 1;
    const overlaps = try allocator.alloc(u32, num_tiles - 1);
    errdefer allocator.free(overlaps);
    @memset(overlaps, min_overlap);
    var remaining: i64 = @as(i64, tile_size) * num_tiles - @as(i64, min_overlap) * (num_tiles - 1) - length;
    var i: usize = 0;
    while (remaining >= align_to) : (i += 1) {
        overlaps[i % overlaps.len] += align_to;
        remaining -= align_to;
    }
    const starts = try allocator.alloc(u32, num_tiles);
    starts[0] = 0;
    for (1..num_tiles) |ti| starts[ti] = starts[ti - 1] + tile_size - overlaps[ti - 1];
    return .{ .starts = starts, .overlaps = overlaps };
}

fn nchwIndex(c: u32, t: u32, y: u32, x: u32, tt: u32, h: u32, w: u32) usize {
    return ((((c * tt + t) * h) + y) * w) + x;
}

fn blend(a: []const f32, b: []f32, channels: u32, t: u32, h: u32, w: u32, extent: u32, along_h: bool) void {
    const e = @min(if (along_h) h else w, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < (if (along_h) e else h)) : (y += 1) {
                var x: u32 = 0;
                while (x < (if (along_h) w else e)) : (x += 1) {
                    const k = if (along_h) y else x;
                    const wb = @as(f32, @floatFromInt(k)) / ef;
                    const ai = if (along_h)
                        nchwIndex(c, ti, h - e + y, x, t, h, w)
                    else
                        nchwIndex(c, ti, y, w - e + x, t, h, w);
                    const bi = nchwIndex(c, ti, y, x, t, h, w);
                    b[bi] = a[ai] * (1.0 - wb) + b[bi] * wb;
                }
            }
        }
    }
}

fn copyNchwCrop(
    dst: []f32,
    dst_h: u32,
    dst_w: u32,
    out_y: u32,
    out_x: u32,
    src: []const f32,
    src_h: u32,
    src_w: u32,
    use_h: u32,
    use_w: u32,
    channels: u32,
    t: u32,
) void {
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < use_h) : (y += 1) {
                @memcpy(
                    dst[nchwIndex(c, ti, out_y + y, out_x, t, dst_h, dst_w)..][0..use_w],
                    src[nchwIndex(c, ti, y, 0, t, src_h, src_w)..][0..use_w],
                );
            }
        }
    }
}

const NchwStitcher = struct {
    acc: []f32,
    prev_row: []f32,
    curr_row: []f32,
    work: []f32,
    channels: u32,
    t: u32,
    acc_h: u32,
    acc_w: u32,
    tile_h: u32,
    tile_w: u32,
    n_y: u32,
    n_x: u32,
    y_overlaps: []u32,
    x_overlaps: []u32,
    out_y: u32,
    out_x: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        acc: []f32,
        channels: u32,
        t: u32,
        acc_h: u32,
        acc_w: u32,
        tile_h: u32,
        tile_w: u32,
        y: TilePlan,
        x: TilePlan,
    ) !NchwStitcher {
        const n_y: u32 = @intCast(y.starts.len);
        const n_x: u32 = @intCast(x.starts.len);
        const tile_n = @as(usize, channels) * t * tile_h * tile_w;
        return .{
            .acc = acc,
            .prev_row = try allocator.alloc(f32, n_x * tile_n),
            .curr_row = try allocator.alloc(f32, n_x * tile_n),
            .work = try allocator.alloc(f32, tile_n),
            .channels = channels,
            .t = t,
            .acc_h = acc_h,
            .acc_w = acc_w,
            .tile_h = tile_h,
            .tile_w = tile_w,
            .n_y = n_y,
            .n_x = n_x,
            .y_overlaps = y.overlaps,
            .x_overlaps = x.overlaps,
            .out_y = 0,
            .out_x = 0,
        };
    }

    pub fn deinit(self: *NchwStitcher, allocator: std.mem.Allocator) void {
        allocator.free(self.prev_row);
        allocator.free(self.curr_row);
        allocator.free(self.work);
    }

    fn tileN(self: NchwStitcher) usize {
        return @as(usize, self.channels) * self.t * self.tile_h * self.tile_w;
    }

    pub fn push(self: *NchwStitcher, yi: u32, xi: u32, tile: []const f32) void {
        const n = self.tileN();
        @memcpy(self.curr_row[xi * n ..][0..n], tile[0..n]);
        @memcpy(self.work[0..n], tile[0..n]);
        if (yi > 0) blend(self.prev_row[xi * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.y_overlaps[yi - 1], true);
        if (xi > 0) blend(self.curr_row[(xi - 1) * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.x_overlaps[xi - 1], false);
        const use_h = if (yi + 1 < self.n_y) self.tile_h - self.y_overlaps[yi] else self.tile_h;
        const use_w = if (xi + 1 < self.n_x) self.tile_w - self.x_overlaps[xi] else self.tile_w;
        copyNchwCrop(self.acc, self.acc_h, self.acc_w, self.out_y, self.out_x, self.work, self.tile_h, self.tile_w, use_h, use_w, self.channels, self.t);
        self.out_x += use_w;
        if (xi + 1 == self.n_x) {
            const tmp = self.prev_row;
            self.prev_row = self.curr_row;
            self.curr_row = tmp;
            self.out_y += use_h;
            self.out_x = 0;
        }
    }
};

fn applyLatentNorm(values: []f32, channels: u32, mean: []const f32, stddev: []const f32) void {
    for (values, 0..) |*v, i| v.* = v.* * stddev[i % channels] + mean[i % channels];
}

fn vitCoords(dim: u32, out: []f32) []f32 {
    const d: f32 = @floatFromInt(dim);
    for (0..dim) |i| out[i] = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / d) - 1.0;
    return out[0..dim];
}

fn withModelBatch(like: zml.Tensor, t: zml.Tensor) zml.Tensor {
    return if (like.shape().partition(.b).eql(.init(.model))) t.withPartitioning(.{ .b = .model }) else t;
}

fn vaeTokens() u32 {
    return vae_t * vae_h * vae_w;
}

fn vaeSeq(registers: u32) u32 {
    return vaeTokens() + registers + 1;
}

fn vaePositions(allocator: std.mem.Allocator, registers: u32) ![]f32 {
    const patches = vaeTokens();
    const out = try allocator.alloc(f32, (patches + registers + 1) * 3);
    var t_axis: [vae_t]f32 = undefined;
    var h_axis: [vae_h]f32 = undefined;
    var w_axis: [vae_w]f32 = undefined;
    _ = vitCoords(vae_t, &t_axis);
    _ = vitCoords(vae_h, &h_axis);
    _ = vitCoords(vae_w, &w_axis);
    var i: usize = 0;
    for (0..vae_t) |tt| {
        for (0..vae_h) |hh| {
            for (0..vae_w) |ww| {
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

// --- decoder weights: embed → 36 vit blocks → finish ---

const VitFf = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) VitFf {
        return .{
            .w1 = linear(store, "net.0.proj.weight", "net.0.proj.bias", .replicated, .replicated),
            .w2 = linear(store, "net.2.weight", "net.2.bias", .replicated, .replicated),
        };
    }

    pub fn forward(self: VitFf, x: zml.Tensor) zml.Tensor {
        const value, const gate = applyLinear(self.w1, x).chunkExact(-1, 2);
        return applyLinear(self.w2, gate.silu().mul(value).rename(.{ .dout = .d }));
    }
};

const VitAttn = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    out: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, cfg: VaeCfg) VitAttn {
        return .{
            .q = linear(store, "to_q.weight", "to_q.bias", .replicated, .replicated),
            .k = linear(store, "to_k.weight", "to_k.bias", .replicated, .replicated),
            .v = linear(store, "to_v.weight", "to_v.bias", .replicated, .replicated),
            .out = linear(store, "to_out.0.weight", "to_out.0.bias", .replicated, .replicated),
            .num_heads = cfg.decoder_num_attention_heads,
            .head_dim = cfg.decoder_attention_head_dim,
            .eps = cfg.decoder_norm_eps,
        };
    }

    pub fn forward(self: VitAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        var q = applyLinear(self.q, x).splitAxis(.dout, heads);
        var k = applyLinear(self.k, x).splitAxis(.dout, heads);
        const v = applyLinear(self.v, x).splitAxis(.dout, heads);
        q = zml.nn.applyRotary(zml.nn.rmsNorm(q, .hd, self.eps), cos, sin);
        k = zml.nn.applyRotary(zml.nn.rmsNorm(k, .hd, self.eps), cos, sin);
        return applyLinear(self.out, zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{},
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } })).rename(.{ .dout = .d });
    }
};

pub const VitBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: VitAttn,
    scale1: zml.Tensor,
    norm2: zml.nn.RmsNorm,
    ff: VitFf,
    scale2: zml.Tensor,
    pub const Input = struct { layer: VitBlock, hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: VaeCfg) VitBlock {
        return .{
            .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.decoder_norm_eps),
            .attn = .init(store.withPrefix("attn"), cfg),
            .scale1 = store.createTensor("scale1", .{.d}, .replicated),
            .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.decoder_norm_eps),
            .ff = .init(store.withPrefix("ff")),
            .scale2 = store.createTensor("scale2", .{.d}, .replicated),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const attn = self.attn.forward(self.norm1.forward(input.hidden), input.cos, input.sin);
        const x1 = input.hidden.add(attn.mul(self.scale1.convert(attn.dtype()).broad(attn.shape())));
        const ff = self.ff.forward(self.norm2.forward(x1)).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff.mul(self.scale2.convert(ff.dtype()).broad(ff.shape()))).reuseBuffer(input.hidden) };
    }
};

pub const EmbedModel = struct {
    post_quant: zml.nn.Linear,
    proj: zml.nn.Linear,
    register_tokens: zml.Tensor,
    cfg: VaeCfg,
    pub const Input = struct { model: EmbedModel, latents: zml.Tensor, position_ids: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };

    pub fn forward(input: Input) Output {
        const self = input.model;
        const x = input.latents.withPartialTags(.{ .b, .s, .d });
        var post_w = self.post_quant.weight;
        while (post_w.rank() > 2) post_w = post_w.squeeze(-1);
        const quantized = (zml.nn.Linear.init(post_w.withTags(.{ .dout, .d }), self.post_quant.bias, .d))
            .forward(x.convert(post_w.dtype()))
            .convert(x.dtype())
            .rename(.{ .dout = .d });
        const tokens = withModelBatch(x, applyLinear(self.proj, quantized).rename(.{ .dout = .d }));
        const hidden = withModelBatch(x, zml.Tensor.concatenate(&.{
            tokens,
            self.register_tokens.convert(tokens.dtype()).broad(tokens.shape().setDim(.s, self.register_tokens.dim(.s))),
            zml.Tensor.zeroes(tokens.shape().setDim(.s, 1)),
        }, .s));
        const rotary_dim = self.cfg.rotaryDim();
        const inv = zml.Tensor.scalar(self.cfg.decoder_rope_theta, .f32)
            .pow(zml.Tensor.arange(.{ .end = @divExact(rotary_dim, 6) }, .f32).withTags(.{.f}).scale(-@as(f32, 6) / @as(f32, @floatFromInt(rotary_dim))));
        const emb = ropeCat3(input.position_ids, inv).scale(2.0 * std.math.pi);
        return .{ .hidden = hidden, .cos = emb.cos().convert(tokens.dtype()), .sin = emb.sin().convert(tokens.dtype()) };
    }
};

pub const FinishModel = struct {
    norm_out: zml.nn.LayerNorm,
    proj_out: zml.nn.Linear,
    cfg: VaeCfg,
    pub const Input = struct { model: FinishModel, hidden: zml.Tensor };
    pub const Output = struct { patches: zml.Tensor };

    pub fn forward(input: Input) Output {
        const proj = applyLinear(input.model.proj_out, input.model.norm_out.forward(input.hidden)).rename(.{ .dout = .d });
        return .{ .patches = proj.slice(.s, .{ .start = 0, .end = proj.dim(.s) - input.model.cfg.decoder_num_register_tokens - 1 }) };
    }
};

pub const VisualModel = struct {
    embed: EmbedModel,
    blocks: []VitBlock,
    finish: FinishModel,
    cfg: VaeCfg,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !VisualModel {
        const cfg: VaeCfg = .{};
        const dec = store.withPrefix("decoder");
        const blocks = try allocator.alloc(VitBlock, @intCast(cfg.decoder_num_layers));
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*block, i| block.* = .init(dec.withPrefix("transformer_blocks").withLayer(i), cfg);
        const post = store.withPrefix("post_quant_conv");
        return .{
            .embed = .{
                .post_quant = .init(post.createTensor("weight", .{ .dout, .d, .kt, .kh, .kw }, .replicated), post.maybeCreateTensor("bias", .{.dout}, .replicated), .d),
                .proj = linear(dec.withPrefix("proj_in"), "weight", "bias", .replicated, .replicated),
                .register_tokens = dec.createTensor("register_tokens", .{ .b, .s, .d }, .replicated),
                .cfg = cfg,
            },
            .blocks = blocks,
            .finish = .{
                .norm_out = ln(dec.withPrefix("norm_out"), cfg.decoder_norm_eps),
                .proj_out = linear(dec.withPrefix("proj_out"), "weight", "bias", .replicated, .replicated),
                .cfg = cfg,
            },
            .cfg = cfg,
        };
    }

    pub fn deinit(self: VisualModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }
};

fn unpackPatches(allocator: std.mem.Allocator, patches: []const f32, patch_t: u32, patch: u32, channels: u32) ![]f32 {
    const pixel_t = vae_t * patch_t;
    const pixel_h = vae_h * patch;
    const pixel_w = vae_w * patch;
    const out = try allocator.alloc(f32, channels * pixel_t * pixel_h * pixel_w);
    const width = channels * patch_t * patch * patch;
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < vae_t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < vae_h) : (hh += 1) {
            var ww: u32 = 0;
            while (ww < vae_w) : (ww += 1) {
                var src: usize = 0;
                for (0..channels) |c| {
                    for (0..patch_t) |dt| {
                        const pt = tt * patch_t + @as(u32, @intCast(dt));
                        var dh: u32 = 0;
                        while (dh < patch) : (dh += 1) {
                            @memcpy(
                                out[(((c * pixel_t + pt) * pixel_h + (hh * patch + dh)) * pixel_w + (ww * patch))..][0..patch],
                                patches[row * width + src ..][0..patch],
                            );
                            src += patch;
                        }
                    }
                }
                row += 1;
            }
        }
    }
    return out;
}

pub const VaeCompiled = struct {
    embed: zml.FnExe(EmbedModel.forward),
    block: zml.FnExe(VitBlock.forward),
    finish: zml.FnExe(FinishModel.forward),
    tile_batch: u32 = 1,
    partition_b: bool = false,

    pub fn deinit(self: *VaeCompiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.finish.deinit();
    }
};

// --- compile tiles + decode ---

fn vaeBatchShape(tags: anytype, dt: zml.DataType, partition_b: bool) zml.Tensor {
    const t = zml.Tensor.init(tags, dt);
    return if (partition_b) t.withPartitioning(.{ .b = .model }) else t;
}

pub fn compileVae(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    visual: VisualModel,
    geo: config.Geometry,
    shardings: config.Shardings,
    progress: *std.Progress.Node,
) !VaeCompiled {
    _ = geo;
    const tile_batch: u32 = 28;
    var all = shardings.all();
    const seq = vaeSeq(@intCast(visual.cfg.decoder_num_register_tokens));
    const batch = @max(1, tile_batch);
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const partition_b = batch > 1 and tp > 1 and batch % tp == 0;
    var node = progress.start("Compiling MiniMax-H3 VAE", 3);
    defer node.end();
    const vae_dt = visual.embed.proj.weight.dtype();
    const embed_exe = try compileFn(EmbedModel.forward, "minimax_h3_vae_embed", allocator, io, platform, &all, progress, .{.{
        .model = visual.embed,
        .latents = vaeBatchShape(.{ .b = batch, .s = vaeTokens(), .d = visual.cfg.latent_channels }, .f32, partition_b),
        .position_ids = .init(.{ .s = seq, .ax = 3 }, .f32),
    }});
    errdefer embed_exe.deinit();
    const block_exe = try compileFn(VitBlock.forward, "minimax_h3_vae_block", allocator, io, platform, &all, progress, .{.{
        .layer = visual.blocks[0],
        .hidden = vaeBatchShape(.{ .b = batch, .s = seq, .d = visual.cfg.dim() }, vae_dt, partition_b),
        .cos = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, vae_dt),
        .sin = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, vae_dt),
    }});
    errdefer block_exe.deinit();
    const finish_exe = try compileFn(FinishModel.forward, "minimax_h3_vae_finish", allocator, io, platform, &all, progress, .{.{
        .model = visual.finish,
        .hidden = vaeBatchShape(.{ .b = batch, .s = seq, .d = visual.cfg.dim() }, vae_dt, partition_b),
    }});
    return .{ .embed = embed_exe, .block = block_exe, .finish = finish_exe, .tile_batch = batch, .partition_b = partition_b };
}

fn copyLatentTile(src: []const f32, src_t: u32, src_h: u32, src_w: u32, channels: u32, t0: u32, h0: u32, w0: u32, dst: []f32) void {
    @memset(dst, 0);
    const copy_t = @min(vae_t, src_t - t0);
    const copy_h = @min(vae_h, src_h - h0);
    const copy_w = @min(vae_w, src_w - w0);
    const row_n = @as(usize, copy_w) * channels;
    var tt: u32 = 0;
    while (tt < copy_t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < copy_h) : (hh += 1) {
            @memcpy(
                dst[(((tt * vae_h + hh) * vae_w) * channels)..][0..row_n],
                src[((((t0 + tt) * src_h + (h0 + hh)) * src_w + w0) * channels)..][0..row_n],
            );
        }
    }
}

const VisualCache = struct {
    embed: zml.Bufferized(EmbedModel),
    blocks: []zml.Bufferized(VitBlock),
    finish: zml.Bufferized(FinishModel),

    pub fn deinit(self: *VisualCache, allocator: std.mem.Allocator) void {
        drop(EmbedModel, &self.embed);
        for (self.blocks) |*block| drop(VitBlock, block);
        allocator.free(self.blocks);
        drop(FinishModel, &self.finish);
    }
};

fn loadVisualCache(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *const VisualModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
) !VisualCache {
    var embed_bufs = try load(allocator, io, platform, store, shardings, EmbedModel, &loaded.embed, progress, null);
    errdefer drop(EmbedModel, &embed_bufs);
    var finish_bufs = try load(allocator, io, platform, store, shardings, FinishModel, &loaded.finish, progress, null);
    errdefer drop(FinishModel, &finish_bufs);
    const blocks = try allocator.alloc(zml.Bufferized(VitBlock), loaded.blocks.len);
    errdefer allocator.free(blocks);
    var filled: usize = 0;
    errdefer for (blocks[0..filled]) |*block| drop(VitBlock, block);
    var loader = try initLoader(allocator, platform);
    defer loader.deinit();
    for (0..loaded.blocks.len) |i| {
        blocks[i] = try load(allocator, io, platform, store, shardings, VitBlock, &loaded.blocks[i], progress, &loader);
        filled += 1;
    }
    return .{ .embed = embed_bufs, .blocks = blocks, .finish = finish_bufs };
}

const EmbedRunner = zml.FnExe(EmbedModel.forward).Runner(.{.model});
const VitRunner = zml.FnExe(VitBlock.forward).Runner(.{.layer});
const FinishRunner = zml.FnExe(FinishModel.forward).Runner(.{.model});

fn runVisualBatch(
    allocator: std.mem.Allocator,
    io: std.Io,
    compiled: *const VaeCompiled,
    loaded: *const VisualModel,
    embed: *EmbedRunner,
    block: *VitRunner,
    finish: *FinishRunner,
    layers: []const zml.Bufferized(VitBlock),
    pos: zml.Buffer,
    packed_latents: []const f32,
    shardings: []const zml.Sharding,
    platform: *const zml.Platform,
) ![]f32 {
    const batch = compiled.tile_batch;
    var latent_shape: zml.Shape = .init(.{ .b = batch, .s = vaeTokens(), .d = loaded.cfg.latent_channels }, .f32);
    const latent_sharding: zml.Sharding = if (compiled.partition_b) blk: {
        latent_shape = latent_shape.withPartitioning(.{ .b = .model });
        break :blk shardings[0];
    } else .replicated;
    var latent_buf = try zml.Buffer.fromBytes(io, platform, latent_shape, latent_sharding, std.mem.sliceAsBytes(packed_latents));
    defer latent_buf.deinit();
    var hidden: zml.Buffer = undefined;
    var cos: zml.Buffer = undefined;
    var sin: zml.Buffer = undefined;
    embed.run(io, .{ .inputs = .{ .latents = latent_buf, .position_ids = pos }, .outputs = .{ .hidden = &hidden, .cos = &cos, .sin = &sin } });
    defer cos.deinit();
    defer sin.deinit();
    // keep every hidden alive until finish waits — block.run is async
    var held: std.ArrayList(zml.Buffer) = .empty;
    defer {
        for (held.items) |*buf| buf.deinit();
        held.deinit(allocator);
    }
    try held.append(allocator, hidden);
    for (layers) |layer| {
        block.rebake(.{ .layer = layer });
        var next: zml.Buffer = undefined;
        block.run(io, .{ .inputs = .{ .hidden = hidden, .cos = cos, .sin = sin }, .outputs = .{ .hidden = &next } });
        hidden = next;
        try held.append(allocator, next);
    }
    var patches: zml.Buffer = undefined;
    finish.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .patches = &patches }, .opts = .{ .wait = true } });
    defer patches.deinit();
    const raw = try allocator.alloc(f32, @as(usize, batch) * vaeTokens() * @as(usize, @intCast(loaded.cfg.out_channels * config.visual_temporal * config.visual_spatial * config.visual_spatial)));
    errdefer allocator.free(raw);
    try patches.toSlice(io, .init(patches.shape(), std.mem.sliceAsBytes(raw)));
    return raw;
}

pub fn decodeVideo(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const VaeCompiled,
    loaded: *const VisualModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    geo: config.Geometry,
    video_thwc: []f32,
    progress: *std.Progress.Node,
) ![]f32 {
    const cfg = loaded.cfg;
    applyLatentNorm(video_thwc, @intCast(cfg.latent_channels), &cfg.latents_mean, &cfg.latents_std);
    const channels: u32 = @intCast(cfg.latent_channels);
    const y_plan = try splitTiles(allocator, geo.pixel_h, tile_px, tile_overlap_px, config.visual_spatial);
    defer y_plan.deinit(allocator);
    const x_plan = try splitTiles(allocator, geo.pixel_w, tile_px, tile_overlap_px, config.visual_spatial);
    defer x_plan.deinit(allocator);
    const num_chunks = (geo.latent_t + token_drop) / chunk - 1;
    const chunk_frames = chunk * config.visual_temporal;
    const out_frames = geo.frames;
    const out = try allocator.alloc(f32, 3 * out_frames * geo.pixel_h * geo.pixel_w);
    errdefer allocator.free(out);
    @memset(out, 0);

    var cache = try loadVisualCache(allocator, io, platform, loaded, store, shardings, progress);
    defer cache.deinit(allocator);
    const registers: u32 = @intCast(loaded.cfg.decoder_num_register_tokens);
    const positions = try vaePositions(allocator, registers);
    defer allocator.free(positions);
    var embed = try EmbedRunner.init(&compiled.embed, allocator, .{ .model = cache.embed });
    defer embed.deinit(allocator);
    var block = try VitRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[0] });
    defer block.deinit(allocator);
    var finish = try FinishRunner.init(&compiled.finish, allocator, .{ .model = cache.finish });
    defer finish.deinit(allocator);
    var pos = try host(io, platform, .init(.{ .s = vaeSeq(registers), .ax = 3 }, .f32), positions);
    defer pos.deinit();

    const plane = geo.pixel_h * geo.pixel_w;
    const pending = try allocator.alloc(f32, 3 * frame_ov * plane);
    defer allocator.free(pending);
    var has_overlap = false;
    var written: u32 = 0;
    var chunk_i: u32 = 0;
    while (chunk_i < num_chunks) : (chunk_i += 1) {
        const start_t = chunk_i * chunk;
        const tile_n = vaeTokens() * channels;
        const n_tiles: u32 = @intCast(y_plan.starts.len * x_plan.starts.len);
        const tile_lats = try allocator.alloc(f32, n_tiles * tile_n);
        defer allocator.free(tile_lats);
        const jobs = try allocator.alloc(struct { yi: usize, xi: usize }, n_tiles);
        defer allocator.free(jobs);
        var job_i: usize = 0;
        for (y_plan.starts, 0..) |y0, yi| {
            for (x_plan.starts, 0..) |x0, xi| {
                copyLatentTile(
                    video_thwc,
                    geo.latent_t,
                    geo.latent_h,
                    geo.latent_w,
                    channels,
                    start_t,
                    y0 / config.visual_spatial,
                    x0 / config.visual_spatial,
                    tile_lats[job_i * tile_n ..][0..tile_n],
                );
                jobs[job_i] = .{ .yi = yi, .xi = xi };
                job_i += 1;
            }
        }

        const clip_t = vae_t * config.visual_temporal;
        const clip = try allocator.alloc(f32, 3 * clip_t * geo.pixel_h * geo.pixel_w);
        defer allocator.free(clip);
        @memset(clip, 0);
        var stitcher = try NchwStitcher.init(
            allocator,
            clip,
            3,
            clip_t,
            geo.pixel_h,
            geo.pixel_w,
            vae_h * config.visual_spatial,
            vae_w * config.visual_spatial,
            y_plan,
            x_plan,
        );
        defer stitcher.deinit(allocator);

        const batch = @max(1, compiled.tile_batch);
        const packed_lat = try allocator.alloc(f32, batch * tile_n);
        defer allocator.free(packed_lat);
        const tile_patch = vaeTokens() * @as(usize, @intCast(loaded.cfg.out_channels * config.visual_temporal * config.visual_spatial * config.visual_spatial));
        var off: usize = 0;
        while (off < jobs.len) {
            @memset(packed_lat, 0);
            const take = @min(batch, @as(u32, @intCast(jobs.len - off)));
            var b: u32 = 0;
            while (b < take) : (b += 1) {
                @memcpy(packed_lat[b * tile_n ..][0..tile_n], tile_lats[(off + b) * tile_n ..][0..tile_n]);
            }
            const patches = try runVisualBatch(
                allocator,
                io,
                compiled,
                loaded,
                &embed,
                &block,
                &finish,
                cache.blocks,
                pos,
                packed_lat,
                shardings,
                platform,
            );
            defer allocator.free(patches);
            b = 0;
            while (b < take) : (b += 1) {
                const pix = try unpackPatches(
                    allocator,
                    patches[b * tile_patch ..][0..tile_patch],
                    config.visual_temporal,
                    config.visual_spatial,
                    3,
                );
                defer allocator.free(pix);
                stitcher.push(@intCast(jobs[off + b].yi), @intCast(jobs[off + b].xi), pix);
            }
            off += take;
        }

        const take = @min(chunk_frames - frame_pre, out_frames - written);
        var f: u32 = 0;
        while (f < take) : (f += 1) {
            if (has_overlap and f < frame_ov) {
                const w = @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(frame_ov));
                var c: u32 = 0;
                while (c < 3) : (c += 1) {
                    var p: usize = 0;
                    while (p < plane) : (p += 1) {
                        out[((c * out_frames + written + f) * plane) + p] =
                            pending[((c * frame_ov + f) * plane) + p] * (1.0 - w) +
                            clip[((c * clip_t + frame_pre + f) * plane) + p] * w;
                    }
                }
            } else {
                var c: u32 = 0;
                while (c < 3) : (c += 1) {
                    @memcpy(out[(c * out_frames + written + f) * plane ..][0..plane], clip[(c * clip_t + frame_pre + f) * plane ..][0..plane]);
                }
            }
        }
        written += take;
        const overlap_src = chunk_frames + frame_pre;
        if (frame_ov > 0 and overlap_src < clip_t) {
            const avail = @min(frame_ov, clip_t - overlap_src);
            var c: u32 = 0;
            while (c < 3) : (c += 1) {
                var of: u32 = 0;
                while (of < avail) : (of += 1) {
                    @memcpy(pending[(c * frame_ov + of) * plane ..][0..plane], clip[(c * clip_t + overlap_src + of) * plane ..][0..plane]);
                }
            }
            has_overlap = true;
        }
        if (written >= out_frames) break;
    }

    if (has_overlap and written < out_frames) {
        const take = @min(frame_ov, out_frames - written);
        var c: u32 = 0;
        while (c < 3) : (c += 1) {
            var f: u32 = 0;
            while (f < take) : (f += 1) {
                @memcpy(out[(c * out_frames + written + f) * plane ..][0..plane], pending[(c * frame_ov + f) * plane ..][0..plane]);
            }
        }
    }

    const rgb_plane = out.len / 3;
    for (0..3) |c| {
        for (0..rgb_plane) |pi| {
            out[c * rgb_plane + pi] = std.math.clamp(out[c * rgb_plane + pi] * imagenet_std[c] + imagenet_mean[c], 0.0, 1.0);
        }
    }
    return out;
}
