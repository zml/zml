//! Tiled ViT decoder. Python: `AutoencoderKLMiniMaxH3`.
//!
//!   1. denormalize latents with `vae/config.json` moments
//!   2. split the canvas into 256 px tiles (64 px overlap)
//!   3. for each temporal chunk of 5 latent frames:
//!        extract tiles → embed → 36 ViT blocks → unpatch pixels → stitch
//!   4. blend overlapping chunks in time
//!   5. undo ImageNet mean/std, clamp to `[0, 1]`

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

const VaeCfg = config.VisualConfig;
const linear = ops.linear;
const rms = ops.rms;
const ln = ops.ln;
const load = ops.load;
const ropeCat3 = ops.ropeCat3;
const Run = ops.Run;

fn applyLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    return lin.forward(x.convert(lin.weight.dtype())).convert(x.dtype());
}

// =============================================================================
// Tile / stitch  (256 px tiles, 64 px overlap — official VAE tiling)
// =============================================================================

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

/// Evenly spaced tile origins along one axis, overlaps aligned to `align_to`.
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

/// Linear blend of two NCHW tiles along H (`along_h`) or W.
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

/// Places decoded tiles into the canvas, blending 64 px overlaps.
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

    /// Blend this tile with its top/left neighbors and copy the unique region into `acc`.
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

/// `v ← v * std + mean` per latent channel (official VAE denorm).
fn applyLatentNorm(values: []f32, channels: u32, mean: []const f32, stddev: []const f32) void {
    for (values, 0..) |*v, i| v.* = v.* * stddev[i % channels] + mean[i % channels];
}

fn vitCoords(dim: u32, out: []f32) void {
    const d: f32 = @floatFromInt(dim);
    for (0..dim) |i| out[i] = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / d) - 1.0;
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

/// RoPE (t,h,w) for the 7×16×16 latent tile, plus zeros for register/pad tokens.
fn vaePositions(allocator: std.mem.Allocator, registers: u32) ![]f32 {
    const patches = vaeTokens();
    const out = try allocator.alloc(f32, (patches + registers + 1) * 3);
    var t_axis: [vae_t]f32 = undefined;
    var h_axis: [vae_h]f32 = undefined;
    var w_axis: [vae_w]f32 = undefined;
    vitCoords(vae_t, &t_axis);
    vitCoords(vae_h, &h_axis);
    vitCoords(vae_w, &w_axis);
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

fn rgbPlane(c: u32, f: u32, frames: u32, plane: usize) usize {
    return (@as(usize, c) * frames + f) * plane;
}

fn copyRgbFrames(dst: []f32, dst_frames: u32, dst_off: u32, src: []const f32, src_frames: u32, src_off: u32, n: u32, plane: usize) void {
    var c: u32 = 0;
    while (c < 3) : (c += 1) {
        var f: u32 = 0;
        while (f < n) : (f += 1) {
            @memcpy(dst[rgbPlane(c, dst_off + f, dst_frames, plane)..][0..plane], src[rgbPlane(c, src_off + f, src_frames, plane)..][0..plane]);
        }
    }
}

fn blendRgbFrames(
    dst: []f32,
    dst_frames: u32,
    dst_off: u32,
    a: []const f32,
    a_frames: u32,
    a_off: u32,
    b: []const f32,
    b_frames: u32,
    b_off: u32,
    n: u32,
    blend_span: u32,
    plane: usize,
) void {
    var f: u32 = 0;
    while (f < n) : (f += 1) {
        const w = @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(blend_span));
        var c: u32 = 0;
        while (c < 3) : (c += 1) {
            const d = dst[rgbPlane(c, dst_off + f, dst_frames, plane)..][0..plane];
            const aa = a[rgbPlane(c, a_off + f, a_frames, plane)..][0..plane];
            const bb = b[rgbPlane(c, b_off + f, b_frames, plane)..][0..plane];
            for (d, aa, bb) |*o, av, bv| o.* = av * (1.0 - w) + bv * w;
        }
    }
}

// =============================================================================
// Decoder  (`MiniMaxH3VideoViTDecoder3d`: embed → 36 blocks → finish)
// =============================================================================

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

/// Python: `MiniMaxH3VideoAttention`
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

/// Python: `MiniMaxH3VideoTransformerBlock`
const VitBlock = struct {
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
        // h ← h + scale1 * Attn(RMS(h));  h ← h + scale2 * FF(RMS(h))
        const attn = self.attn.forward(self.norm1.forward(input.hidden), input.cos, input.sin);
        const x1 = input.hidden.add(attn.mul(self.scale1.convert(attn.dtype()).broad(attn.shape())));
        const ff = self.ff.forward(self.norm2.forward(x1)).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff.mul(self.scale2.convert(ff.dtype()).broad(ff.shape()))).reuseBuffer(input.hidden) };
    }
};

/// Post-quant conv, patch project, register tokens, and ViT RoPE.
const EmbedModel = struct {
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

/// LayerNorm + linear; drops register and pad tokens from the sequence.
const FinishModel = struct {
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

/// Python: `AutoencoderKLMiniMaxH3` (tiled ViT decoder)
pub const Vae = struct {
    embed: EmbedModel,
    blocks: []VitBlock,
    finish: FinishModel,
    cfg: VaeCfg,
    compiled: ?Compiled = null,

    const Compiled = struct {
        embed: zml.FnExe(EmbedModel.forward),
        block: zml.FnExe(VitBlock.forward),
        finish: zml.FnExe(FinishModel.forward),
        tile_batch: u32 = 1,
        partition_b: bool = false,

        fn deinit(self: *Compiled) void {
            self.embed.deinit();
            self.block.deinit();
            self.finish.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Vae {
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

    pub fn deinit(self: *Vae, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        allocator.free(self.blocks);
    }

    pub fn compile(self: *Vae, run: *const Run) !void {
        return compileVae(self, run);
    }

    /// THWC latents → NCHW RGB in `[0, 1]`, tiled and temporally chunked.
    pub fn decodeVideo(
        self: *const Vae,
        run: *const Run,
        store: *zml.io.TensorStore,
        geo: config.Geometry,
        video_thwc: []f32,
    ) ![]f32 {
        return decodeVae(self, run, store, geo, video_thwc);
    }
};

/// Fold ViT patch tokens back into an NCHW pixel tile.
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

// =============================================================================
// Compile / run
// =============================================================================

fn vaeBatchShape(tags: anytype, dt: zml.DataType, partition_b: bool) zml.Tensor {
    const t = zml.Tensor.init(tags, dt);
    return if (partition_b) t.withPartitioning(.{ .b = .model }) else t;
}

fn compileVae(self: *Vae, run: *const Run) !void {
    const model = self.*;
    const tile_batch: u32 = 28;
    const seq = vaeSeq(@intCast(model.cfg.decoder_num_register_tokens));
    const batch = @max(1, tile_batch);
    const tp: u32 = @intCast(run.shardings.model.numPartitionsForLogicalAxis(.model));
    const partition_b = batch > 1 and tp > 1 and batch % tp == 0;
    var node = run.progress.start("Compiling MiniMax-H3 VAE", 3);
    defer node.end();
    const vae_dt = model.embed.proj.weight.dtype();
    const embed_exe = try zml.FnExe(EmbedModel.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_vae_embed",
    }, .{.{
        .model = model.embed,
        .latents = vaeBatchShape(.{ .b = batch, .s = vaeTokens(), .d = model.cfg.latent_channels }, .f32, partition_b),
        .position_ids = .init(.{ .s = seq, .ax = 3 }, .f32),
    }});
    errdefer embed_exe.deinit();
    const block_exe = try zml.FnExe(VitBlock.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_vae_block",
    }, .{.{
        .layer = model.blocks[0],
        .hidden = vaeBatchShape(.{ .b = batch, .s = seq, .d = model.cfg.dim() }, vae_dt, partition_b),
        .cos = .init(.{ .s = seq, .f = model.cfg.rotaryDim() }, vae_dt),
        .sin = .init(.{ .s = seq, .f = model.cfg.rotaryDim() }, vae_dt),
    }});
    errdefer block_exe.deinit();
    const finish_exe = try zml.FnExe(FinishModel.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_vae_finish",
    }, .{.{
        .model = model.finish,
        .hidden = vaeBatchShape(.{ .b = batch, .s = seq, .d = model.cfg.dim() }, vae_dt, partition_b),
    }});
    self.compiled = .{ .embed = embed_exe, .block = block_exe, .finish = finish_exe, .tile_batch = batch, .partition_b = partition_b };
}

/// Copy a 7×16×16 latent window at `(t0,h0,w0)` into `dst` (zero-padded at edges).
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

const VaeCache = struct {
    embed: zml.Bufferized(EmbedModel),
    blocks: []zml.Bufferized(VitBlock),
    finish: zml.Bufferized(FinishModel),

    pub fn deinit(self: *VaeCache, allocator: std.mem.Allocator) void {
        zml.Buffer.deinitAll(EmbedModel, &self.embed);
        for (self.blocks) |*block| zml.Buffer.deinitAll(VitBlock, block);
        allocator.free(self.blocks);
        zml.Buffer.deinitAll(FinishModel, &self.finish);
    }
};

fn loadVaeCache(run: *const Run, loaded: *const Vae, store: *zml.io.TensorStore) !VaeCache {
    var embed_bufs = try load(run, store, EmbedModel, &loaded.embed, null);
    errdefer zml.Buffer.deinitAll(EmbedModel, &embed_bufs);
    var finish_bufs = try load(run, store, FinishModel, &loaded.finish, null);
    errdefer zml.Buffer.deinitAll(FinishModel, &finish_bufs);
    const blocks = try run.allocator.alloc(zml.Bufferized(VitBlock), loaded.blocks.len);
    errdefer run.allocator.free(blocks);
    var filled: usize = 0;
    errdefer for (blocks[0..filled]) |*block| zml.Buffer.deinitAll(VitBlock, block);
    var loader: zml.io.Loader = try .init(run.allocator, run.platform, ops.loader_opts);
    defer loader.deinit();
    for (0..loaded.blocks.len) |i| {
        blocks[i] = try load(run, store, VitBlock, &loaded.blocks[i], &loader);
        filled += 1;
    }
    return .{ .embed = embed_bufs, .blocks = blocks, .finish = finish_bufs };
}

const EmbedRunner = zml.FnExe(EmbedModel.forward).Runner(.{.model});
const VitRunner = zml.FnExe(VitBlock.forward).Runner(.{.layer});
const FinishRunner = zml.FnExe(FinishModel.forward).Runner(.{.model});

/// Embed → 36 ViT blocks → unpatch one GPU batch of latent tiles.
fn runVaeBatch(
    run: *const Run,
    loaded: *const Vae,
    embed: *EmbedRunner,
    block: *VitRunner,
    finish: *FinishRunner,
    layers: []const zml.Bufferized(VitBlock),
    pos: zml.Buffer,
    packed_latents: []const f32,
) ![]f32 {
    const compiled = if (loaded.compiled) |*c| c else return error.NotCompiled;
    const batch = compiled.tile_batch;
    var latent_shape: zml.Shape = .init(.{ .b = batch, .s = vaeTokens(), .d = loaded.cfg.latent_channels }, .f32);
    const latent_sharding: zml.Sharding = if (compiled.partition_b) blk: {
        latent_shape = latent_shape.withPartitioning(.{ .b = .model });
        break :blk run.mesh()[0];
    } else .replicated;
    var latent_buf = try zml.Buffer.fromBytes(run.io, run.platform, latent_shape, latent_sharding, std.mem.sliceAsBytes(packed_latents));
    defer latent_buf.deinit();
    var hidden: zml.Buffer = undefined;
    var cos: zml.Buffer = undefined;
    var sin: zml.Buffer = undefined;
    embed.run(run.io, .{ .inputs = .{ .latents = latent_buf, .position_ids = pos }, .outputs = .{ .hidden = &hidden, .cos = &cos, .sin = &sin } });
    defer cos.deinit();
    defer sin.deinit();
    // keep every hidden alive until finish waits — block.run is async
    var held: std.ArrayList(zml.Buffer) = .empty;
    defer {
        for (held.items) |*buf| buf.deinit();
        held.deinit(run.allocator);
    }
    try held.append(run.allocator, hidden);
    for (layers) |layer| {
        block.rebake(.{ .layer = layer });
        var next: zml.Buffer = undefined;
        block.run(run.io, .{ .inputs = .{ .hidden = hidden, .cos = cos, .sin = sin }, .outputs = .{ .hidden = &next } });
        hidden = next;
        try held.append(run.allocator, next);
    }
    var patches: zml.Buffer = undefined;
    finish.run(run.io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .patches = &patches }, .opts = .{ .wait = true } });
    defer patches.deinit();
    const raw = try run.allocator.alloc(f32, @as(usize, batch) * vaeTokens() * @as(usize, @intCast(loaded.cfg.out_channels * config.visual_temporal * config.visual_spatial * config.visual_spatial)));
    errdefer run.allocator.free(raw);
    try patches.toSlice(run.io, .init(patches.shape(), std.mem.sliceAsBytes(raw)));
    return raw;
}

/// Denorm → tile → chunked ViT decode → temporal blend → ImageNet undo.
fn decodeVae(
    self: *const Vae,
    run: *const Run,
    store: *zml.io.TensorStore,
    geo: config.Geometry,
    video_thwc: []f32,
) ![]f32 {
    const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
    const cfg = self.cfg;
    applyLatentNorm(video_thwc, @intCast(cfg.latent_channels), &cfg.latents_mean, &cfg.latents_std);
    const channels: u32 = @intCast(cfg.latent_channels);
    const y_plan = try splitTiles(run.allocator, geo.pixel_h, tile_px, tile_overlap_px, config.visual_spatial);
    defer y_plan.deinit(run.allocator);
    const x_plan = try splitTiles(run.allocator, geo.pixel_w, tile_px, tile_overlap_px, config.visual_spatial);
    defer x_plan.deinit(run.allocator);
    const num_chunks = (geo.latent_t + token_drop) / chunk - 1;
    const chunk_frames = chunk * config.visual_temporal;
    const out_frames = geo.frames;
    const out = try run.allocator.alloc(f32, 3 * out_frames * geo.pixel_h * geo.pixel_w);
    errdefer run.allocator.free(out);
    @memset(out, 0);

    var cache = try loadVaeCache(run, self, store);
    defer cache.deinit(run.allocator);
    const registers: u32 = @intCast(self.cfg.decoder_num_register_tokens);
    const positions = try vaePositions(run.allocator, registers);
    defer run.allocator.free(positions);
    var embed = try EmbedRunner.init(&compiled.embed, run.allocator, .{ .model = cache.embed });
    defer embed.deinit(run.allocator);
    var block = try VitRunner.init(&compiled.block, run.allocator, .{ .layer = cache.blocks[0] });
    defer block.deinit(run.allocator);
    var finish = try FinishRunner.init(&compiled.finish, run.allocator, .{ .model = cache.finish });
    defer finish.deinit(run.allocator);
    var pos = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .s = vaeSeq(registers), .ax = 3 }, .f32), .replicated, std.mem.sliceAsBytes(positions));
    defer pos.deinit();

    const plane = geo.pixel_h * geo.pixel_w;
    const pending = try run.allocator.alloc(f32, 3 * frame_ov * plane);
    defer run.allocator.free(pending);
    var has_overlap = false;
    var written: u32 = 0;
    var chunk_i: u32 = 0;
    while (chunk_i < num_chunks) : (chunk_i += 1) {
        const start_t = chunk_i * chunk;
        const tile_n = vaeTokens() * channels;
        const n_tiles: u32 = @intCast(y_plan.starts.len * x_plan.starts.len);
        const tile_lats = try run.allocator.alloc(f32, n_tiles * tile_n);
        defer run.allocator.free(tile_lats);
        const jobs = try run.allocator.alloc(struct { yi: usize, xi: usize }, n_tiles);
        defer run.allocator.free(jobs);
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
        const clip = try run.allocator.alloc(f32, 3 * clip_t * geo.pixel_h * geo.pixel_w);
        defer run.allocator.free(clip);
        @memset(clip, 0);
        var stitcher = try NchwStitcher.init(
            run.allocator,
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
        defer stitcher.deinit(run.allocator);

        const batch = @max(1, compiled.tile_batch);
        const packed_lat = try run.allocator.alloc(f32, batch * tile_n);
        defer run.allocator.free(packed_lat);
        const tile_patch = vaeTokens() * @as(usize, @intCast(self.cfg.out_channels * config.visual_temporal * config.visual_spatial * config.visual_spatial));
        var off: usize = 0;
        while (off < jobs.len) {
            @memset(packed_lat, 0);
            const take = @min(batch, @as(u32, @intCast(jobs.len - off)));
            var b: u32 = 0;
            while (b < take) : (b += 1) {
                @memcpy(packed_lat[b * tile_n ..][0..tile_n], tile_lats[(off + b) * tile_n ..][0..tile_n]);
            }
            const patches = try runVaeBatch(run, self, &embed, &block, &finish, cache.blocks, pos, packed_lat);
            defer run.allocator.free(patches);
            b = 0;
            while (b < take) : (b += 1) {
                const pix = try unpackPatches(
                    run.allocator,
                    patches[b * tile_patch ..][0..tile_patch],
                    config.visual_temporal,
                    config.visual_spatial,
                    3,
                );
                defer run.allocator.free(pix);
                stitcher.push(@intCast(jobs[off + b].yi), @intCast(jobs[off + b].xi), pix);
            }
            off += take;
        }

        const take = @min(chunk_frames - frame_pre, out_frames - written);
        const overlap_n: u32 = if (has_overlap) @min(take, frame_ov) else 0;
        if (overlap_n > 0) {
            blendRgbFrames(out, out_frames, written, pending, frame_ov, 0, clip, clip_t, frame_pre, overlap_n, frame_ov, plane);
        }
        if (take > overlap_n) {
            copyRgbFrames(out, out_frames, written + overlap_n, clip, clip_t, frame_pre + overlap_n, take - overlap_n, plane);
        }
        written += take;
        const overlap_src = chunk_frames + frame_pre;
        if (frame_ov > 0 and overlap_src < clip_t) {
            copyRgbFrames(pending, frame_ov, 0, clip, clip_t, overlap_src, @min(frame_ov, clip_t - overlap_src), plane);
            has_overlap = true;
        }
        if (written >= out_frames) break;
    }

    if (has_overlap and written < out_frames) {
        copyRgbFrames(out, out_frames, written, pending, frame_ov, 0, @min(frame_ov, out_frames - written), plane);
    }

    const rgb_plane = out.len / 3;
    for (0..3) |c| {
        for (0..rgb_plane) |pi| {
            out[c * rgb_plane + pi] = std.math.clamp(out[c * rgb_plane + pi] * imagenet_std[c] + imagenet_mean[c], 0.0, 1.0);
        }
    }
    return out;
}
