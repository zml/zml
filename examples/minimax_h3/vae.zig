//! Tiled ViT decoder.
//!
//!   1. unpatch DiT tokens and denormalize with `vae/config.json` moments
//!   2. split the canvas into 256 px tiles (64 px overlap)
//!   3. for each temporal chunk of 5 latent frames:
//!        extract tiles → embed → 36 ViT blocks → unpatch pixels → stitch
//!   4. blend overlapping chunks in time
//!   5. undo ImageNet mean/std, clamp to `[0, 1]`

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

const VisualConfig = config.VisualConfig;
const linear = ops.linear;
const rms = ops.rms;
const ln = ops.ln;
const ropeCat3 = ops.ropeCat3;
const Run = ops.Run;

const log = std.log.scoped(.minimax_h3);

fn applyLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    return lin.forward(x.convert(lin.weight.dtype())).convert(x.dtype());
}

// =============================================================================
// Decoder  (embed → 36 blocks → finish)
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
        const value, const gate = applyLinear(self.w1, x).chunkExact(.dout, 2);
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

    pub fn init(store: zml.io.TensorStore.View, cfg: VisualConfig) VitAttn {
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
        const q = zml.nn.applyRotary(zml.nn.rmsNorm(applyLinear(self.q, x).splitAxis(.dout, heads), .hd, self.eps), cos, sin);
        const k = zml.nn.applyRotary(zml.nn.rmsNorm(applyLinear(self.k, x).splitAxis(.dout, heads), .hd, self.eps), cos, sin);
        const v = applyLinear(self.v, x).splitAxis(.dout, heads);
        // Portable SDPA (CPU / CUDA / …). DiT uses `attention.dense` for the long packed seq.
        return applyLinear(self.out, zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{},
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } })).rename(.{ .dout = .d });
    }
};

const VitBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: VitAttn,
    scale1: zml.Tensor,
    norm2: zml.nn.RmsNorm,
    ff: VitFf,
    scale2: zml.Tensor,
    pub const Input = struct { layer: VitBlock, hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: VisualConfig) VitBlock {
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
    cfg: VisualConfig,
    pub const Input = struct { model: EmbedModel, latents: zml.Tensor, position_ids: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };

    pub fn forward(input: Input) Output {
        const self = input.model;
        const x = input.latents.withPartialTags(.{ .b, .s, .d });
        const post_w = self.post_quant.weight.merge(.{ .d = .{ .d, .kt, .kh, .kw } });
        const quantized = (zml.nn.Linear.init(post_w, self.post_quant.bias, .d))
            .forward(x.convert(post_w.dtype()))
            .convert(x.dtype())
            .rename(.{ .dout = .d });
        const tokens = applyLinear(self.proj, quantized).rename(.{ .dout = .d });
        const hidden = zml.Tensor.concatenate(&.{
            tokens,
            self.register_tokens.convert(tokens.dtype()).broad(tokens.shape().setDim(.s, self.register_tokens.dim(.s))),
            zml.Tensor.zeroes(tokens.shape().setDim(.s, 1)),
        }, .s);
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
    cfg: VisualConfig,
    pub const Input = struct { model: FinishModel, hidden: zml.Tensor };
    pub const Output = struct { patches: zml.Tensor };

    pub fn forward(input: Input) Output {
        const proj = applyLinear(input.model.proj_out, input.model.norm_out.forward(input.hidden)).rename(.{ .dout = .d });
        return .{ .patches = proj.slice(.s, .{ .start = 0, .end = proj.dim(.s) - input.model.cfg.decoder_num_register_tokens - 1 }) };
    }
};

fn vitCoords(dim: u32, out: []f32) void {
    const d: f32 = @floatFromInt(dim);
    for (0..dim) |i| out[i] = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / d) - 1.0;
}

fn vaeTokens() u32 {
    return config.vae_latent_t * config.vae_latent_h * config.vae_latent_w;
}

fn vaeSeq(registers: u32) u32 {
    return vaeTokens() + registers + 1;
}

/// RoPE (t,h,w) for the 7×16×16 latent tile, plus zeros for register/pad tokens.
fn vaePositions(allocator: std.mem.Allocator, registers: u32) ![]f32 {
    const patches = vaeTokens();
    const out = try allocator.alloc(f32, (patches + registers + 1) * 3);
    var t_axis: [config.vae_latent_t]f32 = undefined;
    var h_axis: [config.vae_latent_h]f32 = undefined;
    var w_axis: [config.vae_latent_w]f32 = undefined;
    vitCoords(config.vae_latent_t, &t_axis);
    vitCoords(config.vae_latent_h, &h_axis);
    vitCoords(config.vae_latent_w, &w_axis);
    var i: usize = 0;
    for (0..config.vae_latent_t) |tt| {
        for (0..config.vae_latent_h) |hh| {
            for (0..config.vae_latent_w) |ww| {
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

const Unpatch = struct {
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    channels: u32,
    patch: [3]i64,
    mean: [24]f32,
    std: [24]f32,
    pub const Input = struct { spec: Unpatch, tokens: zml.Tensor };
    pub const Output = struct { thwc: zml.Tensor };

    pub fn forward(input: Input) Output {
        const spec = input.spec;
        const pt: u32 = @intCast(spec.patch[0]);
        const ph: u32 = @intCast(spec.patch[1]);
        const pw: u32 = @intCast(spec.patch[2]);
        const x = input.tokens.withPartialTags(.{ .b, .s, .d })
            .splitAxis(.s, .{
                .tp = spec.latent_t / pt,
                .hp = spec.latent_h / ph,
                .wp = spec.latent_w / pw,
            })
            .splitAxis(.d, .{
                .c = spec.channels,
                .pt = pt,
                .ph = ph,
                .pw = pw,
            })
            .transpose(.{ .b, .tp, .pt, .hp, .ph, .wp, .pw, .c })
            .merge(.{ .t = .{ .tp, .pt }, .h = .{ .hp, .ph }, .w = .{ .wp, .pw } })
            .squeeze(.b);
        return .{ .thwc = ops.denorm(x, spec.mean[0..spec.channels], spec.std[0..spec.channels]) };
    }
};

fn unpackNchw(patches: zml.Tensor, temporal: u32, spatial: u32, channels: i64) zml.Tensor {
    return patches.withPartialTags(.{ .b, .s, .d })
        .splitAxis(.s, .{
            .lt = config.vae_latent_t,
            .lh = config.vae_latent_h,
            .lw = config.vae_latent_w,
        })
        .splitAxis(.d, .{
            .c = channels,
            .pt = temporal,
            .ph = spatial,
            .pw = spatial,
        })
        .transpose(.{ .b, .c, .lt, .pt, .lh, .ph, .lw, .pw })
        .merge(.{ .t = .{ .lt, .pt }, .h = .{ .lh, .ph }, .w = .{ .lw, .pw } });
}

const Decode = struct {
    embed: EmbedModel,
    blocks: []VitBlock,
    finish: FinishModel,
    y_starts: []const u32,
    x_starts: []const u32,
    partition_b: bool,
    pub const Input = struct { model: Decode, thwc: zml.Tensor, start_t: zml.Tensor, position_ids: zml.Tensor };
    pub const Output = struct { tiles: zml.Tensor };

    pub fn unload(self: *zml.Bufferized(Decode), allocator: std.mem.Allocator) void {
        zml.Buffer.deinitAll(EmbedModel, &self.embed);
        for (self.blocks) |*block| zml.Buffer.deinitAll(VitBlock, block);
        allocator.free(self.blocks);
        zml.Buffer.deinitAll(FinishModel, &self.finish);
    }

    pub fn forward(input: Input) Output {
        const self = input.model;
        const channels = self.embed.cfg.latent_channels;
        const temporal: u32 = self.embed.cfg.temporal();
        const spatial: u32 = self.embed.cfg.spatial();
        const thwc = input.thwc.withPartialTags(.{ .t, .h, .w, .c });
        const padded = thwc.pad(0, .{
            .t = zml.Tensor.Pad{ .high = @as(i64, config.vae_latent_t) },
            .h = zml.Tensor.Pad{ .high = @as(i64, config.vae_latent_h) },
            .w = zml.Tensor.Pad{ .high = @as(i64, config.vae_latent_w) },
        });
        var parts: [config.vae_tile_batch]zml.Tensor = undefined;
        var n: usize = 0;
        for (self.y_starts) |y0| {
            for (self.x_starts) |x0| {
                const window = padded.slices(.{ .t, .h, .w }, &.{
                    .dyn(input.start_t, @as(i64, config.vae_latent_t)),
                    .{ .start = @as(i64, @intCast(y0 / spatial)), .len = @as(i64, config.vae_latent_h) },
                    .{ .start = @as(i64, @intCast(x0 / spatial)), .len = @as(i64, config.vae_latent_w) },
                });
                parts[n] = window.reshape(.{ .b = 1, .s = vaeTokens(), .d = channels });
                n += 1;
            }
        }
        var latents = zml.Tensor.concatenate(parts[0..n], .b);
        if (self.partition_b) latents = latents.withPartitioning(.{ .b = .model });
        const emb = EmbedModel.forward(.{ .model = self.embed, .latents = latents, .position_ids = input.position_ids });
        var hidden = emb.hidden;
        for (self.blocks) |block| {
            hidden = VitBlock.forward(.{ .layer = block, .hidden = hidden, .cos = emb.cos, .sin = emb.sin }).hidden;
        }
        const patches = FinishModel.forward(.{ .model = self.finish, .hidden = hidden }).patches;
        return .{ .tiles = unpackNchw(patches, temporal, spatial, self.embed.cfg.out_channels) };
    }
};

pub const Loaded = struct {
    decode: zml.Bufferized(Decode),
    loader: ?zml.io.Loader = null,

    pub fn wait(self: *Loaded, io: std.Io) !void {
        if (self.loader) |*loader| {
            try loader.await(io);
            loader.deinit();
            self.loader = null;
        }
    }

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator, io: std.Io) void {
        self.wait(io) catch {};
        Decode.unload(&self.decode, allocator);
        allocator.destroy(self);
    }
};

pub const Vae = struct {
    embed: EmbedModel,
    blocks: []VitBlock,
    finish: FinishModel,
    cfg: VisualConfig,
    compiled: ?Compiled = null,

    const Compiled = struct {
        unpatch: zml.FnExe(Unpatch.forward),
        decode: zml.FnExe(Decode.forward),
        y_plan: TilePlan,
        x_plan: TilePlan,

        fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
            self.unpatch.deinit();
            self.decode.deinit();
            self.y_plan.deinit(allocator);
            self.x_plan.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: VisualConfig) !Vae {
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
        if (self.compiled) |*c| c.deinit(allocator);
        allocator.free(self.blocks);
    }

    pub fn compile(self: *Vae, run: *const Run, geo: config.Geometry, patch: [3]i64) !void {
        const spatial = self.cfg.spatial();
        var y_plan = try splitTiles(run.allocator, geo.pixel_h, config.vae_tile_px, config.vae_tile_overlap_px, spatial);
        errdefer y_plan.deinit(run.allocator);
        var x_plan = try splitTiles(run.allocator, geo.pixel_w, config.vae_tile_px, config.vae_tile_overlap_px, spatial);
        errdefer x_plan.deinit(run.allocator);
        const n_tiles: u32 = @intCast(y_plan.starts.len * x_plan.starts.len);
        if (n_tiles > config.vae_tile_batch) return error.TooManyVaeTiles;
        const tp: u32 = @intCast(run.shardings.model.numPartitionsForLogicalAxis(.model));
        const partition_b = tp > 1 and n_tiles % tp == 0;
        const seq_len = vaeSeq(@intCast(self.cfg.decoder_num_register_tokens));
        var node = run.progress.start("Compiling MiniMax-H3 VAE", 2);
        defer node.end();
        const unpatch_exe = try zml.FnExe(Unpatch.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_vae_unpatch",
        }, .{.{
            .spec = Unpatch{
                .latent_t = geo.latent_t,
                .latent_h = geo.latent_h,
                .latent_w = geo.latent_w,
                .channels = @intCast(self.cfg.latent_channels),
                .patch = patch,
                .mean = self.cfg.latents_mean,
                .std = self.cfg.latents_std,
            },
            .tokens = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        }});
        errdefer unpatch_exe.deinit();
        const decode_model = Decode{
            .embed = self.embed,
            .blocks = self.blocks,
            .finish = self.finish,
            .y_starts = y_plan.starts,
            .x_starts = x_plan.starts,
            .partition_b = partition_b,
        };
        const decode_exe = try zml.FnExe(Decode.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_vae_decode",
        }, .{.{
            .model = decode_model,
            .thwc = .init(.{ .t = geo.latent_t, .h = geo.latent_h, .w = geo.latent_w, .c = self.cfg.latent_channels }, .f32),
            .start_t = .init(.{}, .i32),
            .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
        }});
        self.compiled = .{
            .unpatch = unpatch_exe,
            .decode = decode_exe,
            .y_plan = y_plan,
            .x_plan = x_plan,
        };
    }

    pub fn startLoad(self: *const Vae, run: *const Run, store: *zml.io.TensorStore) !*Loaded {
        const loaded = try run.allocator.create(Loaded);
        errdefer run.allocator.destroy(loaded);
        const blocks_buf = try run.allocator.alloc(zml.Bufferized(VitBlock), self.blocks.len);
        var blocks_owned = false;
        errdefer if (!blocks_owned) run.allocator.free(blocks_buf);
        for (self.blocks, blocks_buf) |*src, *dst| {
            dst.* = try zml.mem.bufferize(run.allocator, VitBlock, src);
        }
        loaded.* = .{
            .decode = .{
                .embed = try zml.mem.bufferize(run.allocator, EmbedModel, &self.embed),
                .blocks = blocks_buf,
                .finish = try zml.mem.bufferize(run.allocator, FinishModel, &self.finish),
            },
            .loader = try .init(run.allocator, run.platform, ops.loader_opts),
        };
        blocks_owned = true;
        errdefer Decode.unload(&loaded.decode, run.allocator);
        errdefer loaded.loader.?.deinit();
        if (loaded.loader) |*loader| {
            try loader.load(run.io, EmbedModel, &self.embed, &loaded.decode.embed, store, &run.mesh, .{ .progress = run.progress });
            for (self.blocks, loaded.decode.blocks) |*src, *dst| {
                try loader.load(run.io, VitBlock, src, dst, store, &run.mesh, .{ .progress = run.progress });
            }
            try loader.load(run.io, FinishModel, &self.finish, &loaded.decode.finish, store, &run.mesh, .{ .progress = run.progress });
        }
        return loaded;
    }

    /// Packed DiT video tokens → NCHW RGB in `[0, 1]`.
    pub fn decodeVideo(
        self: *const Vae,
        run: *const Run,
        geo: config.Geometry,
        video_tokens: zml.Buffer,
        loaded: *Loaded,
    ) ![]f32 {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        try loaded.wait(run.io);
        const cfg = self.cfg;
        const spatial = cfg.spatial();
        const temporal = cfg.temporal();
        const token_drop: u32 = @intCast(cfg.token_drop);
        const y_plan = compiled.y_plan;
        const x_plan = compiled.x_plan;
        const n_tiles: u32 = @intCast(y_plan.starts.len * x_plan.starts.len);
        const num_chunks = (geo.latent_t + token_drop) / config.visual_latents_per_chunk - 1;
        const chunk_frames = config.visual_latents_per_chunk * temporal;
        const out_frames = geo.frames;
        const out = try run.allocator.alloc(f32, 3 * out_frames * geo.pixel_h * geo.pixel_w);
        errdefer run.allocator.free(out);
        @memset(out, 0);

        const registers: u32 = @intCast(cfg.decoder_num_register_tokens);
        const positions = try vaePositions(run.allocator, registers);
        defer run.allocator.free(positions);
        var pos = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .s = vaeSeq(registers), .ax = 3 }, .f32), .replicated, std.mem.sliceAsBytes(positions));
        defer pos.deinit();

        var unpatch_runner = try zml.FnExe(Unpatch.forward).Runner(.{}).init(&compiled.unpatch, run.allocator, .{});
        defer unpatch_runner.deinit(run.allocator);
        var thwc: zml.Buffer = undefined;
        unpatch_runner.run(run.io, .{
            .inputs = .{ .tokens = video_tokens },
            .outputs = .{ .thwc = &thwc },
            .opts = .{ .wait = true },
        });
        defer thwc.deinit();

        var decode_runner = try zml.FnExe(Decode.forward).Runner(.{.model}).init(&compiled.decode, run.allocator, .{ .model = loaded.decode });
        defer decode_runner.deinit(run.allocator);

        const clip_t = config.vae_latent_t * temporal;
        const tile_h = config.vae_latent_h * spatial;
        const tile_w = config.vae_latent_w * spatial;
        const tile_n = @as(usize, 3) * clip_t * tile_h * tile_w;
        const host = try run.allocator.alloc(f32, @as(usize, n_tiles) * tile_n);
        defer run.allocator.free(host);

        const plane = geo.pixel_h * geo.pixel_w;
        const pending = try run.allocator.alloc(f32, 3 * config.vae_frame_overlap * plane);
        defer run.allocator.free(pending);
        var has_overlap = false;
        var written: u32 = 0;
        const decode_start: std.Io.Timestamp = .now(run.io, .awake);

        for (0..num_chunks) |chunk_i| {
            const start_t: i32 = @intCast(@as(u32, @intCast(chunk_i)) * config.visual_latents_per_chunk);
            var start_buf = try zml.Buffer.scalar(run.io, run.platform, start_t, .i32);
            defer start_buf.deinit();
            var tiles: zml.Buffer = undefined;
            decode_runner.run(run.io, .{
                .inputs = .{ .thwc = thwc, .start_t = start_buf, .position_ids = pos },
                .outputs = .{ .tiles = &tiles },
                .opts = .{ .wait = true },
            });
            defer tiles.deinit();
            try tiles.toSlice(run.io, .init(tiles.shape(), std.mem.sliceAsBytes(host)));

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
                tile_h,
                tile_w,
                y_plan,
                x_plan,
            );
            defer stitcher.deinit(run.allocator);
            const n_x = x_plan.starts.len;
            for (0..n_tiles) |b| {
                stitcher.push(@intCast(b / n_x), @intCast(b % n_x), host[b * tile_n ..][0..tile_n]);
            }

            const take = @min(chunk_frames - config.vae_frame_pre, out_frames - written);
            const overlap_n: u32 = if (has_overlap) @min(take, config.vae_frame_overlap) else 0;
            if (overlap_n > 0) {
                blendRgbFrames(out, out_frames, written, pending, config.vae_frame_overlap, 0, clip, clip_t, config.vae_frame_pre, overlap_n, config.vae_frame_overlap, plane);
            }
            if (take > overlap_n) {
                copyRgbFrames(out, out_frames, written + overlap_n, clip, clip_t, config.vae_frame_pre + overlap_n, take - overlap_n, plane);
            }
            written += take;
            const overlap_src = chunk_frames + config.vae_frame_pre;
            if (overlap_src < clip_t) {
                copyRgbFrames(pending, config.vae_frame_overlap, 0, clip, clip_t, overlap_src, @min(config.vae_frame_overlap, clip_t - overlap_src), plane);
                has_overlap = true;
            }
            if (written >= out_frames) break;
        }
        if (has_overlap and written < out_frames) {
            copyRgbFrames(out, out_frames, written, pending, config.vae_frame_overlap, 0, @min(config.vae_frame_overlap, out_frames - written), plane);
        }

        const rgb_plane = out.len / 3;
        for (0..3) |c| {
            for (0..rgb_plane) |pi| {
                out[c * rgb_plane + pi] = std.math.clamp(out[c * rgb_plane + pi] * imagenet_std[c] + imagenet_mean[c], 0.0, 1.0);
            }
        }
        log.info("decode video: ok [{f}]", .{decode_start.untilNow(run.io, .awake)});
        return out;
    }
};

// =============================================================================
// Tile / stitch  (256 px tiles, 64 px overlap)
// =============================================================================

const imagenet_mean = [_]f32{ 0.485, 0.456, 0.406 };
const imagenet_std = [_]f32{ 0.229, 0.224, 0.225 };

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

fn nchwIndex(c: usize, t: usize, y: usize, x: usize, tt: usize, h: usize, w: usize) usize {
    return ((((c * tt + t) * h) + y) * w) + x;
}

const Axis = enum { h, w };

/// Linear blend of two NCHW tiles along H or W.
fn blend(a: []const f32, b: []f32, channels: u32, t: u32, h: u32, w: u32, extent: u32, axis: Axis) void {
    const e = @min(if (axis == .h) h else w, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    const t_n: usize = t;
    const h_n: usize = h;
    const w_n: usize = w;
    const e_n: usize = e;
    for (0..channels) |c| {
        for (0..t_n) |ti| {
            for (0..if (axis == .h) e_n else h_n) |y| {
                for (0..if (axis == .h) w_n else e_n) |x| {
                    const k = if (axis == .h) y else x;
                    const wb = @as(f32, @floatFromInt(k)) / ef;
                    const ai = if (axis == .h)
                        nchwIndex(c, ti, h_n - e_n + y, x, t_n, h_n, w_n)
                    else
                        nchwIndex(c, ti, y, w_n - e_n + x, t_n, h_n, w_n);
                    const bi = nchwIndex(c, ti, y, x, t_n, h_n, w_n);
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
    const dst_h_n: usize = dst_h;
    const dst_w_n: usize = dst_w;
    const src_h_n: usize = src_h;
    const src_w_n: usize = src_w;
    const out_y_n: usize = out_y;
    const out_x_n: usize = out_x;
    const t_n: usize = t;
    const use_h_n: usize = use_h;
    const use_w_n: usize = use_w;
    for (0..channels) |c| {
        for (0..t_n) |ti| {
            for (0..use_h_n) |y| {
                @memcpy(
                    dst[nchwIndex(c, ti, out_y_n + y, out_x_n, t_n, dst_h_n, dst_w_n)..][0..use_w_n],
                    src[nchwIndex(c, ti, y, 0, t_n, src_h_n, src_w_n)..][0..use_w_n],
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

    /// Blend this tile with its top/left neighbors and copy the unique region into `acc`.
    pub fn push(self: *NchwStitcher, yi: u32, xi: u32, tile: []const f32) void {
        const n = @as(usize, self.channels) * self.t * self.tile_h * self.tile_w;
        @memcpy(self.curr_row[xi * n ..][0..n], tile[0..n]);
        @memcpy(self.work[0..n], tile[0..n]);
        if (yi > 0) blend(self.prev_row[xi * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.y_overlaps[yi - 1], .h);
        if (xi > 0) blend(self.curr_row[(xi - 1) * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.x_overlaps[xi - 1], .w);
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

fn rgbPlane(c: usize, f: usize, frames: usize, plane: usize) usize {
    return (c * frames + f) * plane;
}

fn copyRgbFrames(dst: []f32, dst_frames: u32, dst_off: u32, src: []const f32, src_frames: u32, src_off: u32, n: u32, plane: usize) void {
    const dst_frames_n: usize = dst_frames;
    const src_frames_n: usize = src_frames;
    const dst_off_n: usize = dst_off;
    const src_off_n: usize = src_off;
    for (0..3) |c| {
        for (0..n) |f| {
            @memcpy(dst[rgbPlane(c, dst_off_n + f, dst_frames_n, plane)..][0..plane], src[rgbPlane(c, src_off_n + f, src_frames_n, plane)..][0..plane]);
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
    const dst_frames_n: usize = dst_frames;
    const a_frames_n: usize = a_frames;
    const b_frames_n: usize = b_frames;
    const dst_off_n: usize = dst_off;
    const a_off_n: usize = a_off;
    const b_off_n: usize = b_off;
    const span: f32 = @floatFromInt(blend_span);
    for (0..n) |f| {
        const w = @as(f32, @floatFromInt(f)) / span;
        for (0..3) |c| {
            const d = dst[rgbPlane(c, dst_off_n + f, dst_frames_n, plane)..][0..plane];
            const aa = a[rgbPlane(c, a_off_n + f, a_frames_n, plane)..][0..plane];
            const bb = b[rgbPlane(c, b_off_n + f, b_frames_n, plane)..][0..plane];
            for (d, aa, bb) |*o, av, bv| o.* = av * (1.0 - w) + bv * w;
        }
    }
}
