// Visual VAE encode/decode and audio VAE.

// --- vae/geom.zig ---
pub const geom = struct {
    const std = @import("std");

    const config = @import("model.zig").config;
    const noise = @import("model.zig").noise;

    const LatentHw = config.LatentHw;

    pub const imagenet_mean = [_]f32{ 0.485, 0.456, 0.406 };
    pub const imagenet_std = [_]f32{ 0.229, 0.224, 0.225 };

    pub const VisualSpec = struct {
        spatial: u32 = config.visual_spatial,
        temporal: u32 = config.visual_temporal,
        channels: u32 = 24,
        patch: [3]i64 = .{ 1, 2, 2 },
        clip_length: u32 = 17,
        token_drop: u32 = 3,
        tile_px: u32 = 256,
        tile_overlap_px: u32 = 64,

        pub fn latentFromPixels(self: VisualSpec, pixel_h: u32, pixel_w: u32, frames: u32) LatentHw {
            _ = self;
            return config.visualLatentSize(pixel_h, pixel_w, frames);
        }

        pub fn patchDim(self: VisualSpec) u32 {
            return self.channels * @as(u32, @intCast(self.patch[0] * self.patch[1] * self.patch[2]));
        }

        pub fn tokensChunkSize(self: VisualSpec) u32 {
            return std.math.divCeil(u32, self.clip_length, self.temporal) catch unreachable;
        }

        pub fn tokenOverlap(self: VisualSpec) u32 {
            const chunk = self.tokensChunkSize();
            return (chunk - (self.token_drop % chunk)) % chunk;
        }

        pub fn framePrePadding(self: VisualSpec) u32 {
            return (self.temporal - (self.clip_length % self.temporal)) % self.temporal;
        }

        pub fn frameOverlap(self: VisualSpec) u32 {
            const raw = self.tokenOverlap() * self.temporal;
            const pad = self.framePrePadding();
            if (raw <= pad) return 0;
            return raw - pad;
        }
    };

    pub const AudioSpec = struct {
        channels: u32 = 32,
        stereo: u32 = 2,
        hz: f32 = config.audio_hz,
        sample_rate: u32 = config.audio_sample_rate,
        hop: u32 = 800,

        pub fn tokenCount(self: AudioSpec, latent_t: u32) u32 {
            return latent_t * self.stereo;
        }

        pub fn sampleCount(self: AudioSpec, latent_t: u32) u32 {
            return latent_t * self.hop;
        }
    };

    pub const official_visual: VisualSpec = .{};
    pub const official_audio: AudioSpec = .{};

    pub const TilePlan = struct {
        starts: []u32,
        lengths: []u32,
        overlaps: []u32,

        pub fn deinit(self: TilePlan, allocator: std.mem.Allocator) void {
            allocator.free(self.starts);
            allocator.free(self.lengths);
            allocator.free(self.overlaps);
        }

        pub fn count(self: TilePlan) usize {
            return self.starts.len;
        }
    };

    pub fn tileCount(length: u32, tile_size: u32, min_overlap: u32, align_to: u32) u32 {
        _ = align_to;
        if (tile_size >= length) return 1;
        var num_tiles = std.math.divCeil(u32, length, tile_size) catch unreachable;
        while (tile_size * num_tiles < min_overlap * (num_tiles - 1) + length) {
            num_tiles += 1;
        }
        return num_tiles;
    }

    /// Cover `length` with `tile_size` tiles, overlap at least `min_overlap`,
    /// slack in `align_to` steps.
    pub fn splitTiles(allocator: std.mem.Allocator, length: u32, tile_size: u32, min_overlap: u32, align_to: u32) !TilePlan {
        if (tile_size >= length) {
            const starts = try allocator.alloc(u32, 1);
            errdefer allocator.free(starts);
            starts[0] = 0;
            const lengths = try allocator.alloc(u32, 1);
            lengths[0] = length;
            return .{
                .starts = starts,
                .lengths = lengths,
                .overlaps = try allocator.alloc(u32, 0),
            };
        }

        var num_tiles = std.math.divCeil(u32, length, tile_size) catch unreachable;
        while (tile_size * num_tiles < min_overlap * (num_tiles - 1) + length) {
            num_tiles += 1;
        }

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
        errdefer allocator.free(starts);
        const lengths = try allocator.alloc(u32, num_tiles);
        starts[0] = 0;
        lengths[0] = tile_size;
        for (1..num_tiles) |t| {
            starts[t] = starts[t - 1] + tile_size - overlaps[t - 1];
            lengths[t] = tile_size;
        }
        return .{ .starts = starts, .lengths = lengths, .overlaps = overlaps };
    }

    pub fn decodeTileLatent(spec: VisualSpec, latent_h: u32, latent_w: u32) struct { h: u32, w: u32 } {
        const tile_lat = spec.tile_px / spec.spatial;
        return .{
            .h = @min(tile_lat, latent_h),
            .w = @min(tile_lat, latent_w),
        };
    }

    pub fn decodeClipTokens(spec: VisualSpec, latent_t: u32) u32 {
        return @min(spec.tokensChunkSize() + spec.tokenOverlap(), latent_t + tokenDropPad(spec, latent_t));
    }

    pub fn tokenDropPad(spec: VisualSpec, latent_t: u32) u32 {
        const num_tokens = latent_t + spec.token_drop;
        const chunk = spec.tokensChunkSize();
        return (chunk - (num_tokens % chunk)) % chunk;
    }

    /// Packed DiT audio is channel-major stereo: `(2 * T, C)` = left then right, each `(T, C)`.
    /// The mono audio VAE consumes `(2, C, T)`.
    pub fn audioRowsToBct(dst: []f32, rows: []const f32, channels: u32, t: u32) void {
        std.debug.assert(dst.len == rows.len);
        std.debug.assert(rows.len == 2 * @as(usize, channels) * t);
        var ear: usize = 0;
        while (ear < 2) : (ear += 1) {
            const src = rows[ear * t * channels ..][0 .. t * channels];
            const out = dst[ear * channels * t ..][0 .. channels * t];
            var ti: usize = 0;
            while (ti < t) : (ti += 1) {
                var c: usize = 0;
                while (c < channels) : (c += 1) {
                    out[c * t + ti] = src[ti * channels + c];
                }
            }
        }
    }

    pub fn audioBctToRows(dst: []f32, bct: []const f32, channels: u32, t: u32) void {
        std.debug.assert(dst.len == bct.len);
        std.debug.assert(bct.len == 2 * @as(usize, channels) * t);
        var ear: usize = 0;
        while (ear < 2) : (ear += 1) {
            const src = bct[ear * channels * t ..][0 .. channels * t];
            const out = dst[ear * t * channels ..][0 .. t * channels];
            var c: usize = 0;
            while (c < channels) : (c += 1) {
                var ti: usize = 0;
                while (ti < t) : (ti += 1) {
                    out[ti * channels + c] = src[c * t + ti];
                }
            }
        }
    }

    pub fn f32ToF16Bits(value: f32) u16 {
        return @as(u16, @bitCast(@as(f16, @floatCast(value))));
    }

    pub fn f16BitsToF32(bits: u16) f32 {
        return @as(f16, @bitCast(bits));
    }

    /// Visual-condition posterior: mean + std * randn(seed=42), then FP16 round-trip.
    pub fn sampleVisualPosteriorNchw(
        allocator: std.mem.Allocator,
        moments_nchw: []const f32,
        t: u32,
        h: u32,
        w: u32,
        policy: config.PosteriorPolicy,
    ) ![]f32 {
        const spatial = @as(usize, t) * h * w;
        std.debug.assert(moments_nchw.len >= spatial * 48);
        const out = try allocator.alloc(f32, spatial * 24);
        if (policy == .mean) {
            @memcpy(out, moments_nchw[0..out.len]);
            return out;
        }
        var gen = noise.Generator.init(config.visual_encode_seed);
        const eps = try allocator.alloc(f32, out.len);
        defer allocator.free(eps);
        noise.randn(&gen, eps);
        var i: usize = 0;
        while (i < out.len) : (i += 1) {
            const mean = moments_nchw[i];
            var logvar = moments_nchw[spatial * 24 + i];
            logvar = std.math.clamp(logvar, -30.0, 20.0);
            const stddev = @exp(0.5 * logvar);
            const sampled = mean + stddev * eps[i];
            out[i] = f16BitsToF32(f32ToF16Bits(sampled));
        }
        return out;
    }

    pub fn applyLatentNorm(values: []f32, channels: u32, mean: []const f32, stddev: []const f32, decode: bool) void {
        std.debug.assert(values.len % channels == 0);
        std.debug.assert(mean.len >= channels and stddev.len >= channels);
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            const c = i % channels;
            if (decode) {
                values[i] = values[i] * stddev[c] + mean[c];
            } else {
                values[i] = (values[i] - mean[c]) / stddev[c];
            }
        }
    }

    pub fn denormImagenetRgb(pixels: []f32) void {
        std.debug.assert(pixels.len % 3 == 0);
        const plane = pixels.len / 3;
        var c: usize = 0;
        while (c < 3) : (c += 1) {
            const mean = imagenet_mean[c];
            const stddev = imagenet_std[c];
            var i: usize = 0;
            while (i < plane) : (i += 1) {
                const v = pixels[c * plane + i] * stddev + mean;
                pixels[c * plane + i] = std.math.clamp(v, 0.0, 1.0);
            }
        }
    }

    pub fn nchwToThwc(allocator: std.mem.Allocator, nchw: []const f32, channels: u32, t: u32, h: u32, w: u32) ![]f32 {
        const plane = @as(usize, t) * h * w;
        std.debug.assert(nchw.len >= plane * channels);
        const out = try allocator.alloc(f32, plane * channels);
        var tt: u32 = 0;
        while (tt < t) : (tt += 1) {
            var hh: u32 = 0;
            while (hh < h) : (hh += 1) {
                var ww: u32 = 0;
                while (ww < w) : (ww += 1) {
                    var c: u32 = 0;
                    while (c < channels) : (c += 1) {
                        const src = ((@as(usize, c) * t + tt) * h + hh) * w + ww;
                        const dst = ((@as(usize, tt) * h + hh) * w + ww) * channels + c;
                        out[dst] = nchw[src];
                    }
                }
            }
        }
        return out;
    }

    /// Pixel frames after last-clip pad, then latent frames after one tail `token_drop`.
    pub fn encodeVideoLatentT(spec: VisualSpec, frames: u32) u32 {
        const padded = frames + (spec.clip_length - (frames % spec.clip_length)) % spec.clip_length;
        const clips = padded / spec.clip_length;
        const tokens = clips * spec.tokensChunkSize();
        if (spec.token_drop >= tokens) return 0;
        return tokens - spec.token_drop;
    }

    pub fn vitCoords(dim: u32, out: []f32) []f32 {
        std.debug.assert(out.len >= dim);
        const d: f32 = @floatFromInt(dim);
        for (0..dim) |i| {
            out[i] = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / d) - 1.0;
        }
        return out[0..dim];
    }
};

// --- vae/visual.zig ---
pub const visual = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config_mod = @import("model.zig").config;
    const vae = @import("vae.zig").geom;
    const weights = @import("model.zig").weights;

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
        if (tensorRank(store, weight_name) == 2)
            return .fromStore(store, weight_name, bias_name, .replicated, .replicated, .d);
        return .init(
            linearWeight(store, weight_name),
            if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, .replicated) else null,
            .d,
        );
    }

    const LayerNorm = struct {
        weight: zml.Tensor,
        bias: ?zml.Tensor,
        eps: f32,
        rms: bool = false,

        pub fn init(store: zml.io.TensorStore.View, eps: f32, rms: bool) LayerNorm {
            return .{
                .weight = store.createTensor("weight", .{.d}, .replicated),
                .bias = if (rms) null else store.maybeCreateTensor("bias", .{.d}, .replicated),
                .eps = eps,
                .rms = rms,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(LayerNorm)) void {
            self.weight.deinit();
            if (self.bias) |*bias| bias.deinit();
        }

        pub fn forward(self: LayerNorm, x: zml.Tensor) zml.Tensor {
            if (self.rms) {
                const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
                return normalized.mul(self.weight.convert(.f32).broad(normalized.shape())).convert(x.dtype());
            }
            return (zml.nn.LayerNorm{
                .weight = self.weight.convert(.f32),
                .bias = if (self.bias) |b| b.convert(.f32) else null,
                .eps = self.eps,
            }).forward(x.convert(.f32)).convert(x.dtype());
        }
    };

    const SwiGlu = struct {
        w1: zml.nn.Linear,
        w2: zml.nn.Linear,
        value_first: bool = false,

        pub fn init(store: zml.io.TensorStore.View) SwiGlu {
            if (store.getShape("w1.weight") != null) {
                return .{
                    .w1 = linear(store, "w1.weight", "w1.bias"),
                    .w2 = linear(store, "w2.weight", "w2.bias"),
                };
            }
            return .{
                .w1 = linear(store, "net.0.proj.weight", "net.0.proj.bias"),
                .w2 = linear(store, "net.2.weight", "net.2.bias"),
                .value_first = true,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
            zml.nn.Linear.unloadBuffers(&self.w1);
            zml.nn.Linear.unloadBuffers(&self.w2);
        }

        pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
            const fused = applyLinear(self.w1, x);
            const a, const b = fused.chunkExact(-1, 2);
            const gated = if (self.value_first) b.silu().mul(a) else a.silu().mul(b);
            return applyLinear(self.w2, gated.rename(.{ .dout = .d }));
        }
    };

    const Attention = struct {
        qkv: ?zml.nn.Linear = null,
        q: ?zml.nn.Linear = null,
        k: ?zml.nn.Linear = null,
        v: ?zml.nn.Linear = null,
        out: zml.nn.Linear,
        num_heads: i64,
        head_dim: i64,
        eps: f32,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
            if (store.getShape("to_qkv.weight") != null) {
                return .{
                    .qkv = linear(store, "to_qkv.weight", "to_qkv.bias"),
                    .out = linear(store, "to_out.weight", "to_out.bias"),
                    .num_heads = cfg.decoder_num_attention_heads,
                    .head_dim = cfg.decoder_attention_head_dim,
                    .eps = cfg.decoder_norm_eps,
                };
            }
            return .{
                .q = linear(store, "to_q.weight", "to_q.bias"),
                .k = linear(store, "to_k.weight", "to_k.bias"),
                .v = linear(store, "to_v.weight", "to_v.bias"),
                .out = linear(store, "to_out.0.weight", "to_out.0.bias"),
                .num_heads = cfg.decoder_num_attention_heads,
                .head_dim = cfg.decoder_attention_head_dim,
                .eps = cfg.decoder_norm_eps,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
            if (self.qkv) |*layer| zml.nn.Linear.unloadBuffers(layer);
            if (self.q) |*layer| zml.nn.Linear.unloadBuffers(layer);
            if (self.k) |*layer| zml.nn.Linear.unloadBuffers(layer);
            if (self.v) |*layer| zml.nn.Linear.unloadBuffers(layer);
            zml.nn.Linear.unloadBuffers(&self.out);
        }

        fn projectQkv(self: Attention, x: zml.Tensor) struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor } {
            if (self.qkv) |qkv| {
                const split = applyLinear(qkv, x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 3 * self.head_dim });
                const parts = split.chunkExact(.hd, 3);
                return .{ .q = parts[0], .k = parts[1], .v = parts[2] };
            }
            const heads = .{ .h = self.num_heads, .hd = self.head_dim };
            return .{
                .q = applyLinear(self.q.?, x).splitAxis(.dout, heads),
                .k = applyLinear(self.k.?, x).splitAxis(.dout, heads),
                .v = applyLinear(self.v.?, x).splitAxis(.dout, heads),
            };
        }

        pub fn forward(self: Attention, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
            const qkv = projectQkv(self, x);
            var q = qkv.q;
            var k = qkv.k;
            const v = qkv.v;
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
            return applyLinear(self.out, attn).rename(.{ .dout = .d });
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

        pub fn init(store: zml.io.TensorStore.View, cfg: Config, rms: bool) TransformerBlock {
            const attn_store = store.withPrefix("attn");
            const ff_store = store.withPrefix("ff");
            return .{
                .norm1 = .init(store.withPrefix("norm1"), cfg.decoder_norm_eps, rms),
                .attn = .init(attn_store, cfg),
                .scale1 = store.createTensor("scale1", .{.d}, .replicated),
                .norm2 = .init(store.withPrefix("norm2"), cfg.decoder_norm_eps, rms),
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
            zml.nn.Linear.unloadBuffers(&self.post_quant);
            zml.nn.Linear.unloadBuffers(&self.proj);
            self.register_tokens.deinit();
        }
    };

    pub const FinishModel = struct {
        norm_out: LayerNorm,
        proj_out: zml.nn.Linear,
        cfg: Config,

        pub fn unloadBuffers(self: *zml.Bufferized(FinishModel)) void {
            LayerNorm.unloadBuffers(&self.norm_out);
            zml.nn.Linear.unloadBuffers(&self.proj_out);
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
            const block_rms = !dec.hasKey("x_embedder");
            for (blocks, 0..) |*block, i| block.* = .init(block_store.withLayer(i), cfg, block_rms);

            const post = store.withPrefix("post_quant_conv");
            return .{
                .embed = .{
                    .post_quant = linear(post, "weight", "bias"),
                    .proj = linear(dec.withPrefix(if (dec.hasKey("x_embedder")) "x_embedder" else "proj_in"), "weight", "bias"),
                    .register_tokens = dec.createTensor("register_tokens", .{ .b, .s, .d }, .replicated),
                    .cfg = cfg,
                },
                .blocks = blocks,
                .finish = .{
                    .norm_out = .init(dec.withPrefix("norm_out"), cfg.decoder_norm_eps, false),
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
        const has_embed = store.hasKey("decoder.x_embedder.weight") or store.hasKey("decoder.proj_in.weight");
        return has_embed and store.hasKey("post_quant_conv.weight");
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

    fn applyLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
        const dt = lin.weight.dtype();
        return lin.forward(x.convert(dt)).convert(x.dtype());
    }

    fn conv1x1(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
        var weight = lin.weight;
        while (weight.rank() > 2) {
            weight = weight.squeeze(-1);
        }
        const dt = weight.dtype();
        return (zml.nn.Linear.init(weight.withTags(.{ .dout, .d }), lin.bias, .d))
            .forward(x.convert(dt))
            .convert(x.dtype());
    }

    pub fn embed(input: EmbedInput) EmbedOutput {
        const self = input.model;
        const x = input.latents.withPartialTags(.{ .b, .s, .d });
        const quantized = conv1x1(self.post_quant, x).rename(.{ .dout = .d });
        const tokens = applyLinear(self.proj, quantized).rename(.{ .dout = .d });
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
        const proj = applyLinear(self.proj_out, hidden).rename(.{ .dout = .d });
        const keep = proj.dim(.s) - self.cfg.decoder_num_register_tokens - 1;
        return .{ .patches = proj.slice1d(.s, .{ .start = 0, .end = keep }) };
    }

    pub const LoadedModel = struct {
        inner: Model,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
            var parsed_root = try config_mod.parseOptional(FileConfig, allocator, io, repo, "config.json");
            defer if (parsed_root) |*parsed| parsed.deinit();
            var parsed_source = try config_mod.parseOptional(FileConfig, allocator, io, repo, "source/config.json");
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

    /// Unpack ViT patch tokens `{s, 3*pt*ph*pw}` into NCHW `{3, T, H, W}`.
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
};

// --- vae/visual_enc.zig ---
pub const visual_enc = struct {
    const std = @import("std");

    const zml = @import("zml");

    const visual_vae = @import("vae.zig").visual;
    const weights = @import("model.zig").weights;

    const log = std.log.scoped(.minimax_h3_visual_enc);

    pub const block_out_channels = [_]i64{ 128, 256, 256, 512, 512, 1024 };
    pub const spatial_downsample = [_]i64{ 2, 2, 2, 2, 1, 1 };
    pub const temporal_downsample = [_]i64{ 1, 2, 2, 1, 1, 1 };
    pub const layers_per_block: usize = 2;
    pub const norm_groups: i64 = 32;
    pub const norm_eps: f32 = 1e-6;

    fn tensorRank(store: zml.io.TensorStore.View, name: []const u8) u8 {
        var buffer: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", name }) catch return 5;
        return if (store.store.getShape(key)) |shape| shape.rank() else 5;
    }

    fn convWeight(store: zml.io.TensorStore.View, name: []const u8) zml.Tensor {
        return switch (tensorRank(store, name)) {
            5 => store.createTensor(name, .{ .co, .ci, .kt, .kh, .kw }, .replicated),
            4 => store.createTensor(name, .{ .co, .ci, .kh, .kw }, .replicated),
            else => store.createTensor(name, .{ .co, .ci, .k }, .replicated),
        };
    }

    fn unloadConv(weight: *zml.Buffer, bias: *?zml.Buffer) void {
        weight.deinit();
        if (bias.*) |*b| b.deinit();
    }

    fn reflectPadBoth(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
        if (pad <= 0) return x;
        const n = x.dim(axis);
        if (n <= 1) {
            const first = x.slice1d(axis, .{ .start = 0, .end = 1 });
            const extra = first.broad(first.shape().setDim(axis, pad));
            return zml.Tensor.concatenate(&.{ extra, x, extra }, axis);
        }
        const left = x.slice1d(axis, .{ .start = 1, .end = 1 + pad }).reverse(.{axis});
        const right = x.slice1d(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
        return zml.Tensor.concatenate(&.{ left, x, right }, axis);
    }

    fn reflectPadHigh(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
        if (pad <= 0) return x;
        const n = x.dim(axis);
        if (n <= 1) {
            const last = x.slice1d(axis, .{ .start = n - 1, .end = n });
            return zml.Tensor.concatenate(&.{ x, last.broad(last.shape().setDim(axis, pad)) }, axis);
        }
        const tail = x.slice1d(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
        return zml.Tensor.concatenate(&.{ x, tail }, axis);
    }

    fn causalPadT(x: zml.Tensor, pad: i64) zml.Tensor {
        if (pad <= 0) return x;
        const zeros = zml.Tensor.zeroes(x.shape().setDim(.t, pad));
        return zml.Tensor.concatenate(&.{ zeros, x }, .t);
    }

    fn isolatedGroupNorm(x: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor, groups: i64, eps: f32) zml.Tensor {
        const xf = x.convert(.f32).withPartialTags(.{ .b, .c, .t, .h, .w });
        const b = xf.dim(.b);
        const c = xf.dim(.c);
        const t = xf.dim(.t);
        const h = xf.dim(.h);
        const w = xf.dim(.w);
        const cg = @divExact(c, groups);
        var y = xf.transpose(.{ .b, .t, .c, .h, .w });
        y = y.merge(.{ .bt = .{ .b, .t } }).splitAxis(.c, .{ .g = groups, .cg = cg });
        y = y.merge(.{ .n = .{ .cg, .h, .w } });
        const mean = y.mean(.n);
        const centered = y.sub(mean.broad(y.shape()));
        const variance = centered.mul(centered).mean(.n);
        y = centered.mul(variance.addConstant(eps).rsqrt().broad(y.shape()));
        y = y.splitAxis(.n, .{ .cg = cg, .h = h, .w = w });
        y = y.merge(.{ .c = .{ .g, .cg } }).splitAxis(.bt, .{ .b = b, .t = t });
        y = y.transpose(.{ .b, .c, .t, .h, .w });
        const scale = weight.convert(.f32).withTags(.{.c}).broad(y.shape());
        const shift = bias.convert(.f32).withTags(.{.c}).broad(y.shape());
        return y.mul(scale).add(shift).convert(x.dtype());
    }

    const CausalConv3d = struct {
        weight: zml.Tensor,
        bias: ?zml.Tensor,
        stride_t: i64,
        stride_hw: i64,
        spatial_pad: i64,
        temporal_pad: i64,

        pub fn init(store: zml.io.TensorStore.View, stride_t: i64, stride_hw: i64, spatial_pad: i64, temporal_pad: i64) CausalConv3d {
            return .{
                .weight = convWeight(store, "weight"),
                .bias = store.maybeCreateTensor("bias", .{.co}, .replicated),
                .stride_t = stride_t,
                .stride_hw = stride_hw,
                .spatial_pad = spatial_pad,
                .temporal_pad = temporal_pad,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(CausalConv3d)) void {
            unloadConv(&self.weight, &self.bias);
        }

        pub fn forward(self: CausalConv3d, x: zml.Tensor) zml.Tensor {
            var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t, .h, .w });
            y = reflectPadBoth(y, .h, self.spatial_pad);
            y = reflectPadBoth(y, .w, self.spatial_pad);
            y = causalPadT(y, self.temporal_pad);
            var w = self.weight.convert(.f32);
            if (w.rank() < 5) {
                while (w.rank() < 5) w = w.appendAxes(.{.kt});
            }
            w = w.withPartialTags(.{ .co, .ci, .kt, .kh, .kw });
            y = y.conv3d(w, .{
                .window_strides = &.{ self.stride_t, self.stride_hw, self.stride_hw },
            });
            if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
            return y.convert(x.dtype());
        }
    };

    const GroupNorm = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) GroupNorm {
            return .{
                .weight = store.createTensor("weight", .{.c}, .replicated),
                .bias = store.createTensor("bias", .{.c}, .replicated),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(GroupNorm)) void {
            self.weight.deinit();
            self.bias.deinit();
        }

        pub fn forward(self: GroupNorm, x: zml.Tensor) zml.Tensor {
            return isolatedGroupNorm(x, self.weight, self.bias, norm_groups, norm_eps);
        }
    };

    const Resnet = struct {
        norm1: GroupNorm,
        conv1: CausalConv3d,
        norm2: GroupNorm,
        conv2: CausalConv3d,
        shortcut: ?CausalConv3d,

        pub fn init(store: zml.io.TensorStore.View) Resnet {
            return .{
                .norm1 = .init(store.withPrefix("norm1")),
                .conv1 = .init(store.withPrefix("conv1"), 1, 1, 1, 2),
                .norm2 = .init(store.withPrefix("norm2")),
                .conv2 = .init(store.withPrefix("conv2"), 1, 1, 1, 2),
                .shortcut = if (store.hasKey("conv_shortcut.weight"))
                    .init(store.withPrefix("conv_shortcut"), 1, 1, 0, 0)
                else if (store.hasKey("nin_shortcut.weight"))
                    .init(store.withPrefix("nin_shortcut"), 1, 1, 0, 0)
                else
                    null,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Resnet)) void {
            GroupNorm.unloadBuffers(&self.norm1);
            CausalConv3d.unloadBuffers(&self.conv1);
            GroupNorm.unloadBuffers(&self.norm2);
            CausalConv3d.unloadBuffers(&self.conv2);
            if (self.shortcut) |*s| CausalConv3d.unloadBuffers(s);
        }

        pub fn forward(self: Resnet, x: zml.Tensor) zml.Tensor {
            var h = self.conv1.forward(self.norm1.forward(x).silu());
            h = self.conv2.forward(self.norm2.forward(h).silu());
            var residual = x;
            if (self.shortcut) |s| residual = s.forward(residual);
            return residual.add(h);
        }
    };

    const Downsample = struct {
        conv: CausalConv3d,
        spatial_stride: i64,

        pub fn init(store: zml.io.TensorStore.View, temporal_stride: i64, spatial_stride: i64) Downsample {
            const inner = store.withPrefix("conv");
            return .{
                .conv = .init(inner, temporal_stride, spatial_stride, 0, 2),
                .spatial_stride = spatial_stride,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Downsample)) void {
            CausalConv3d.unloadBuffers(&self.conv);
        }

        pub fn forward(self: Downsample, x: zml.Tensor) zml.Tensor {
            var y = x.withPartialTags(.{ .b, .c, .t, .h, .w });
            if (self.spatial_stride == 2) {
                y = reflectPadHigh(y, .h, 1);
                y = reflectPadHigh(y, .w, 1);
            }
            return self.conv.forward(y);
        }
    };

    const DownBlock = struct {
        block0: Resnet,
        block1: Resnet,
        downsample: ?Downsample,

        pub fn init(store: zml.io.TensorStore.View, temporal_factor: i64, spatial_factor: i64) DownBlock {
            const blocks = store.withPrefix(if (store.hasKey("resnets.0.norm1.weight")) "resnets" else "block");
            return .{
                .block0 = .init(blocks.withLayer(0)),
                .block1 = .init(blocks.withLayer(1)),
                .downsample = if (temporal_factor * spatial_factor > 1)
                    .init(
                        if (store.hasKey("downsamplers.0.conv.weight"))
                            store.withPrefix("downsamplers").withLayer(0)
                        else
                            store.withPrefix("downsample"),
                        temporal_factor,
                        spatial_factor,
                    )
                else
                    null,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(DownBlock)) void {
            Resnet.unloadBuffers(&self.block0);
            Resnet.unloadBuffers(&self.block1);
            if (self.downsample) |*d| Downsample.unloadBuffers(d);
        }

        pub fn forward(self: DownBlock, x: zml.Tensor) zml.Tensor {
            var h = self.block1.forward(self.block0.forward(x));
            if (self.downsample) |d| h = d.forward(h);
            return h;
        }
    };

    fn encoderView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
        return store.withPrefix("encoder");
    }

    pub fn ready(store: zml.io.TensorStore.View) bool {
        return store.hasKey("encoder.conv_in.weight") and store.hasKey("quant_conv.weight");
    }

    pub const Model = struct {
        conv_in: CausalConv3d,
        downs: [6]DownBlock,
        norm_out: GroupNorm,
        conv_out: CausalConv3d,
        quant_conv: CausalConv3d,

        pub fn init(store: zml.io.TensorStore.View) Model {
            const root = store;
            const enc = encoderView(root);
            const down_root = enc.withPrefix(if (enc.hasKey("down_blocks.0.resnets.0.norm1.weight")) "down_blocks" else "down");
            var downs: [6]DownBlock = undefined;
            for (&downs, 0..) |*block, i| {
                block.* = .init(down_root.withLayer(i), temporal_downsample[i], spatial_downsample[i]);
            }
            const quant = root.withPrefix("quant_conv");
            return .{
                .conv_in = .init(enc.withPrefix("conv_in"), 1, 1, 1, 2),
                .downs = downs,
                .norm_out = .init(enc.withPrefix("norm_out")),
                .conv_out = .init(enc.withPrefix("conv_out"), 1, 1, 1, 2),
                .quant_conv = .init(quant, 1, 1, 0, 0),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Model)) void {
            CausalConv3d.unloadBuffers(&self.conv_in);
            for (&self.downs) |*block| DownBlock.unloadBuffers(block);
            GroupNorm.unloadBuffers(&self.norm_out);
            CausalConv3d.unloadBuffers(&self.conv_out);
            CausalConv3d.unloadBuffers(&self.quant_conv);
        }
    };

    pub const EncodeInput = struct {
        model: Model,
        pixels: zml.Tensor,
    };

    pub const EncodeOutput = struct {
        moments: zml.Tensor,
    };

    pub fn encode(input: EncodeInput) EncodeOutput {
        const self = input.model;
        var h = self.conv_in.forward(input.pixels);
        for (self.downs) |block| h = block.forward(h);
        h = self.conv_out.forward(self.norm_out.forward(h).silu());
        return .{ .moments = self.quant_conv.forward(h) };
    }

    pub const LoadedModel = struct {
        inner: Model,
        cfg: visual_vae.Config,

        pub fn init(store: zml.io.TensorStore.View, cfg: visual_vae.Config) LoadedModel {
            return .{ .inner = .init(store), .cfg = cfg };
        }

        pub fn loadBuffers(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(Model) {
            var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
            errdefer Model.unloadBuffers(&buffers);
            var loader = try weights.initLoader(allocator, platform);
            defer loader.deinit();
            const now: std.Io.Timestamp = .now(io, .awake);
            try weights.populate(&loader, io, store, shardings, Model, &self.inner, &buffers, progress);
            log.info("loaded visual VAE encoder [{f}]", .{now.untilNow(io, .awake)});
            return buffers;
        }
    };
};

// --- vae/audio.zig ---
pub const audio = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config_mod = @import("model.zig").config;
    const vae = @import("vae.zig").geom;
    const weights = @import("model.zig").weights;

    const log = std.log.scoped(.minimax_h3_audio_vae);

    /// Per-channel latent moments from the released `audio_vae/config.json`.
    pub const official_latents_mean = [32]f32{
        -0.020211687488382354, 0.3876466479950502,   -0.04398279799186767, -0.28591514936373,
        0.08179686214561671,   -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
        -0.223362481667332,    0.1821006842509091,   0.2941778783780663,   -0.07901167601970885,
        -0.056815072777201,    -0.3699028221860095,  -0.31616315591624855, 0.5905951377425391,
        -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
        -0.3394662328788498,   -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
        0.02868032184767538,   -0.21217779266454084, -0.1678263169941987,  0.3221287889040614,
        -0.1223055851554907,   0.4356604928128464,   -0.0502599202236253,  0.3979258376211797,
    };

    pub const official_latents_std = [32]f32{
        1.6895524230479284, 2.76263727217653,   1.7945344281264435, 1.6801681847309828,
        1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
        2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
        1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
        1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
        1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
        1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
        1.718045989838149,  1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
    };

    pub const Config = struct {
        latent_channels: i64 = 32,
        latent_dim: i64 = 2048,
        encoder_dim: i64 = 64,
        decoder_dim: i64 = 1024,
        sample_rate: u32 = 32_000,
        hop: u32 = 800,
        upsample_rates: [7]i64 = .{ 5, 5, 2, 2, 2, 2, 2 },
        upsample_kernels: [7]i64 = .{ 9, 9, 4, 4, 4, 4, 4 },
        encoder_rates: [5]i64 = .{ 2, 4, 4, 5, 5 },
        resblock_kernels: [3]i64 = .{ 3, 7, 11 },
        resblock_dilations: [3][3]i64 = .{ .{ 1, 3, 5 }, .{ 1, 3, 5 }, .{ 1, 3, 5 } },
        latents_mean: [32]f32 = official_latents_mean,
        latents_std: [32]f32 = official_latents_std,

        pub fn official() Config {
            return .{};
        }

        pub fn spec(self: Config) vae.AudioSpec {
            return .{
                .channels = @intCast(self.latent_channels),
                .sample_rate = self.sample_rate,
                .hop = self.hop,
            };
        }
    };

    const FileConfig = struct {
        latent_channels: ?i64 = null,
        vae_latent_channels: ?i64 = null,
        latent_dim: ?i64 = null,
        encoder_dim: ?i64 = null,
        decoder_dim: ?i64 = null,
        sample_rate: ?u32 = null,
        sampling_rate: ?u32 = null,
        latents_mean: ?[]const f32 = null,
        latents_std: ?[]const f32 = null,

        fn resolve(self: FileConfig) Config {
            var out = Config.official();
            if (self.latent_channels orelse self.vae_latent_channels) |v| out.latent_channels = v;
            if (self.latent_dim) |v| out.latent_dim = v;
            if (self.encoder_dim) |v| out.encoder_dim = v;
            if (self.decoder_dim) |v| out.decoder_dim = v;
            if (self.sample_rate orelse self.sampling_rate) |v| out.sample_rate = v;
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

    fn pick(store: zml.io.TensorStore.View, names: []const []const u8, tagz: anytype) zml.Tensor {
        for (names) |name| {
            if (store.hasKey(name)) return store.createTensor(name, tagz, .replicated);
        }
        return store.createTensor(names[0], tagz, .replicated);
    }

    fn firstKey(store: zml.io.TensorStore.View, names: []const []const u8) []const u8 {
        for (names) |name| {
            if (store.hasKey(name)) return name;
        }
        return names[0];
    }

    fn pickByRank(store: zml.io.TensorStore.View, names: []const []const u8) zml.Tensor {
        const name = firstKey(store, names);
        return switch (tensorRank(store, name)) {
            5 => store.createTensor(name, .{ .co, .ci, .k, .unused_a, .unused_b }, .replicated),
            3 => store.createTensor(name, .{ .co, .ci, .k }, .replicated),
            2 => store.createTensor(name, .{ .co, .ci }, .replicated),
            else => store.createTensor(name, .{.co}, .replicated),
        };
    }

    fn pickTranspose(store: zml.io.TensorStore.View, names: []const []const u8) zml.Tensor {
        const name = firstKey(store, names);
        return switch (tensorRank(store, name)) {
            3 => store.createTensor(name, .{ .ci, .co, .k }, .replicated),
            2 => store.createTensor(name, .{ .ci, .co }, .replicated),
            else => store.createTensor(name, .{.ci}, .replicated),
        };
    }

    fn pickChannel(store: zml.io.TensorStore.View, names: []const []const u8) zml.Tensor {
        const name = firstKey(store, names);
        return switch (tensorRank(store, name)) {
            3 => store.createTensor(name, .{ .unused_a, .c, .unused_b }, .replicated),
            2 => store.createTensor(name, .{ .unused_a, .c }, .replicated),
            else => store.createTensor(name, .{.c}, .replicated),
        };
    }

    fn squeezeToTag(t: zml.Tensor, comptime tag: anytype) zml.Tensor {
        var out = t.convert(.f32);
        var changed = true;
        while (changed and out.rank() > 1) {
            changed = false;
            var ax: i8 = 0;
            while (ax < @as(i8, @intCast(out.rank()))) : (ax += 1) {
                if (out.dim(ax) == 1) {
                    out = out.squeeze(ax);
                    changed = true;
                    break;
                }
            }
        }
        return out.withTags(.{tag});
    }

    fn padRepeatT(x: zml.Tensor, low: i64, high: i64) zml.Tensor {
        var y = x;
        if (low > 0) {
            const first = x.slice1d(.t, .{ .start = 0, .end = 1 });
            y = zml.Tensor.concatenate(&.{ first.broad(first.shape().setDim(.t, low)), y }, .t);
        }
        if (high > 0) {
            const last = x.slice1d(.t, .{ .start = x.dim(.t) - 1, .end = x.dim(.t) });
            y = zml.Tensor.concatenate(&.{ y, last.broad(last.shape().setDim(.t, high)) }, .t);
        }
        return y;
    }

    fn maybe(store: zml.io.TensorStore.View, names: []const []const u8, tagz: anytype) ?zml.Tensor {
        for (names) |name| {
            if (store.hasKey(name)) return store.createTensor(name, tagz, .replicated);
        }
        return null;
    }

    fn unloadOpt(t: *?zml.Buffer) void {
        if (t.*) |*buf| buf.deinit();
    }

    fn fusedKernel(store: zml.io.TensorStore.View) bool {
        return store.hasKey("weight") and !store.hasKey("weight_v");
    }

    fn loadWn(store: zml.io.TensorStore.View, comptime transpose: bool) struct { v: zml.Tensor, g: ?zml.Tensor } {
        const fused = fusedKernel(store);
        const names: []const []const u8 = if (fused) &.{"weight"} else &.{"weight_v"};
        return .{
            .v = if (transpose) pickTranspose(store, names) else pickByRank(store, names),
            .g = if (fused) null else pickByRank(store, &.{"weight_g"}),
        };
    }

    const WNConv1d = struct {
        weight_v: zml.Tensor,
        weight_g: ?zml.Tensor,
        bias: ?zml.Tensor,
        stride: i64,
        dilation: i64,
        padding: i64,

        pub fn init(store: zml.io.TensorStore.View, stride: i64, dilation: i64, padding: i64) WNConv1d {
            const wn = loadWn(store, false);
            return .{
                .weight_v = wn.v,
                .weight_g = wn.g,
                .bias = maybe(store, &.{"bias"}, .{.co}),
                .stride = stride,
                .dilation = dilation,
                .padding = padding,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(WNConv1d)) void {
            self.weight_v.deinit();
            unloadOpt(&self.weight_g);
            unloadOpt(&self.bias);
        }

        pub fn forward(self: WNConv1d, x: zml.Tensor) zml.Tensor {
            const v = self.weight_v.convert(.f32).withPartialTags(.{ .co, .ci, .k });
            const fused = if (self.weight_g) |g| blk: {
                const gs = squeezeToTag(g, .co);
                const sq = squeezeToTag(v.mul(v).sum(.k).sum(.ci), .co).addConstant(1e-9);
                break :blk v.mul(gs.mul(sq.rsqrt()).broad(v.shape()));
            } else v;
            var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t }).conv1d(fused, .{
                .window_strides = self.stride,
                .rhs_dilation = self.dilation,
                .padding = &.{ self.padding, self.padding },
            });
            if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
            return y.convert(x.dtype());
        }
    };

    const TransposeConv = struct {
        weight_v: zml.Tensor,
        weight_g: ?zml.Tensor,
        bias: ?zml.Tensor,
        stride: i64,
        kernel: i64,

        pub fn init(store: zml.io.TensorStore.View, stride: i64, kernel: i64) TransposeConv {
            const inner = store.withPrefix("0");
            const wn = loadWn(inner, true);
            return .{
                .weight_v = wn.v,
                .weight_g = wn.g,
                .bias = maybe(inner, &.{"bias"}, .{.co}),
                .stride = stride,
                .kernel = kernel,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TransposeConv)) void {
            self.weight_v.deinit();
            unloadOpt(&self.weight_g);
            unloadOpt(&self.bias);
        }

        pub fn forward(self: TransposeConv, x: zml.Tensor) zml.Tensor {
            const v = self.weight_v.convert(.f32).withPartialTags(.{ .ci, .co, .k });
            const kernel = if (self.weight_g) |g| blk: {
                const gs = squeezeToTag(g, .ci);
                const sq = squeezeToTag(v.mul(v).sum(.k).sum(.co), .ci).addConstant(1e-9);
                break :blk v.mul(gs.mul(sq.rsqrt()).broad(v.shape()));
            } else v;
            // Reverse along `k`: this conv is conv_transpose1d.
            const fused = kernel.reverse(.{.k});
            const official_pad = @divFloor(self.kernel - self.stride, 2);
            const xla_pad = self.kernel - 1 - official_pad;
            var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t }).conv1d(fused, .{
                .window_strides = 1,
                .lhs_dilation = self.stride,
                .padding = &.{ xla_pad, xla_pad },
                .kernel_input_feature_dimension = 0,
                .kernel_output_feature_dimension = 1,
                .kernel_spatial_dimensions = 2,
            });
            if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
            return y.convert(x.dtype());
        }
    };

    const SnakeBeta = struct {
        alpha: zml.Tensor,
        beta: zml.Tensor,
        logscale: bool = true,

        pub fn init(store: zml.io.TensorStore.View) SnakeBeta {
            const act = store.withPrefix("act");
            return .{
                .alpha = pickChannel(act, &.{"alpha"}),
                .beta = pickChannel(act, &.{"beta"}),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(SnakeBeta)) void {
            self.alpha.deinit();
            self.beta.deinit();
        }

        pub fn forward(self: SnakeBeta, x: zml.Tensor) zml.Tensor {
            var alpha = self.alpha.convert(.f32);
            var beta = self.beta.convert(.f32);
            if (self.logscale) {
                alpha = alpha.exp();
                beta = beta.exp();
            }
            const xf = x.convert(.f32);
            const shaped = alpha.broad(xf.shape());
            const mag = zml.Tensor.scalar(1.0, .f32).div(beta.addConstant(1e-9)).broad(xf.shape());
            const s = xf.mul(shaped).sin();
            return xf.add(mag.mul(s.mul(s))).convert(x.dtype());
        }
    };

    const Activation1d = struct {
        act: SnakeBeta,
        up_filter: zml.Tensor,
        down_filter: zml.Tensor,
        ratio: i64 = 2,
        kernel: i64 = 12,

        pub fn init(store: zml.io.TensorStore.View) Activation1d {
            return .{
                .act = .init(store),
                .up_filter = pick(store, &.{ "upsample.filter", "upsample.lowpass.filter" }, .{ .co, .ci, .k }),
                .down_filter = pick(store, &.{ "downsample.lowpass.filter", "downsample.filter" }, .{ .co, .ci, .k }),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Activation1d)) void {
            SnakeBeta.unloadBuffers(&self.act);
            self.up_filter.deinit();
            self.down_filter.deinit();
        }

        pub fn forward(self: Activation1d, x: zml.Tensor) zml.Tensor {
            const xt = x.withPartialTags(.{ .b, .c, .t });
            const channels = xt.dim(.c);
            const up = self.up_filter.convert(.f32).broad(zml.Shape.init(.{
                .co = channels,
                .ci = 1,
                .k = self.up_filter.dim(-1),
            }, .f32));
            const down = self.down_filter.convert(.f32).broad(zml.Shape.init(.{
                .co = channels,
                .ci = 1,
                .k = self.down_filter.dim(-1),
            }, .f32));
            const pad = @divFloor(self.kernel, self.ratio) - 1;
            const crop_left = pad * self.ratio + @divFloor(self.kernel - self.ratio, 2);
            const crop_right = pad * self.ratio + @divFloor(self.kernel - self.ratio + 1, 2);
            var y = padRepeatT(xt.convert(.f32), pad, pad);
            y = y.conv1d(up, .{
                .window_strides = 1,
                .lhs_dilation = self.ratio,
                .feature_group_count = channels,
                .padding = &.{ self.kernel - 1, self.kernel - 1 },
            }).scale(@as(f32, @floatFromInt(self.ratio)));
            y = y.slice1d(.t, .{ .start = crop_left, .end = y.dim(.t) - crop_right });
            y = self.act.forward(y.convert(x.dtype())).convert(.f32);
            const even = @mod(self.kernel, 2) == 0;
            const pad_left = @divFloor(self.kernel, 2) - @intFromBool(even);
            const pad_right = @divFloor(self.kernel, 2);
            y = padRepeatT(y, pad_left, pad_right);
            return y.conv1d(down, .{
                .window_strides = self.ratio,
                .feature_group_count = channels,
                .padding = &.{ 0, 0 },
            }).convert(x.dtype());
        }
    };

    const AMPBlock = struct {
        convs1: [3]WNConv1d,
        convs2: [3]WNConv1d,
        acts: [6]Activation1d,

        pub fn init(store: zml.io.TensorStore.View, kernel: i64, dilations: [3]i64) AMPBlock {
            var convs1: [3]WNConv1d = undefined;
            var convs2: [3]WNConv1d = undefined;
            var acts: [6]Activation1d = undefined;
            for (dilations, 0..) |d, i| {
                const pad1 = @divFloor(kernel * d - d, 2);
                const pad2 = @divFloor(kernel - 1, 2);
                convs1[i] = .init(store.withPrefix("convs1").withLayer(i), 1, d, pad1);
                convs2[i] = .init(store.withPrefix("convs2").withLayer(i), 1, 1, pad2);
                acts[i * 2] = .init(store.withPrefix("activations").withLayer(i * 2));
                acts[i * 2 + 1] = .init(store.withPrefix("activations").withLayer(i * 2 + 1));
            }
            return .{ .convs1 = convs1, .convs2 = convs2, .acts = acts };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(AMPBlock)) void {
            for (&self.convs1) |*c| WNConv1d.unloadBuffers(c);
            for (&self.convs2) |*c| WNConv1d.unloadBuffers(c);
            for (&self.acts) |*a| Activation1d.unloadBuffers(a);
        }

        pub fn forward(self: AMPBlock, x: zml.Tensor) zml.Tensor {
            var hidden = x;
            for (0..3) |i| {
                var residual = self.acts[i * 2].forward(hidden);
                residual = self.convs1[i].forward(residual);
                residual = self.acts[i * 2 + 1].forward(residual);
                residual = self.convs2[i].forward(residual);
                hidden = hidden.add(residual);
            }
            return hidden;
        }
    };

    fn conv1x1(store: zml.io.TensorStore.View) zml.nn.Linear {
        const weight = switch (tensorRank(store, "weight")) {
            5 => store.createTensor("weight", .{ .dout, .d, .kt, .kh, .kw }, .replicated),
            3 => store.createTensor("weight", .{ .dout, .d, .k }, .replicated),
            else => store.createTensor("weight", .{ .dout, .d }, .replicated),
        };
        return .init(weight, store.maybeCreateTensor("bias", .{.dout}, .replicated), .d);
    }

    pub const Model = struct {
        dec_in_proj: zml.nn.Linear,
        conv_pre: WNConv1d,
        ups: []TransposeConv,
        resblocks: []AMPBlock,
        activation_post: Activation1d,
        conv_post: WNConv1d,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
            const dec = store.withPrefix("decoder");

            const ups = try allocator.alloc(TransposeConv, cfg.upsample_rates.len);
            errdefer allocator.free(ups);
            for (ups, 0..) |*up, i| {
                up.* = .init(dec.withPrefix("ups").withLayer(i), cfg.upsample_rates[i], cfg.upsample_kernels[i]);
            }

            const n_res = cfg.upsample_rates.len * cfg.resblock_kernels.len;
            const resblocks = try allocator.alloc(AMPBlock, n_res);
            errdefer allocator.free(resblocks);
            for (0..cfg.upsample_rates.len) |i| {
                for (0..cfg.resblock_kernels.len) |j| {
                    resblocks[i * cfg.resblock_kernels.len + j] = .init(
                        dec.withPrefix("resblocks").withLayer(i * cfg.resblock_kernels.len + j),
                        cfg.resblock_kernels[j],
                        cfg.resblock_dilations[j],
                    );
                }
            }

            const proj_store = store.withPrefix("dec_in_proj");
            return .{
                .dec_in_proj = conv1x1(proj_store),
                .conv_pre = .init(dec.withPrefix("conv_pre"), 1, 1, 3),
                .ups = ups,
                .resblocks = resblocks,
                .activation_post = .init(dec.withPrefix("activation_post")),
                .conv_post = .init(dec.withPrefix("conv_post"), 1, 1, 3),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
            allocator.free(self.ups);
            allocator.free(self.resblocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
            self.dec_in_proj.weight.deinit();
            if (self.dec_in_proj.bias) |*bias| bias.deinit();
            WNConv1d.unloadBuffers(&self.conv_pre);
            for (self.ups) |*up| TransposeConv.unloadBuffers(up);
            allocator.free(self.ups);
            for (self.resblocks) |*block| AMPBlock.unloadBuffers(block);
            allocator.free(self.resblocks);
            Activation1d.unloadBuffers(&self.activation_post);
            WNConv1d.unloadBuffers(&self.conv_post);
        }
    };

    pub const DecodeInput = struct {
        model: Model,
        latents: zml.Tensor,
    };

    pub const DecodeOutput = struct {
        wav: zml.Tensor,
    };

    fn projectIn(self: Model, latents: zml.Tensor) zml.Tensor {
        const x = latents.withPartialTags(.{ .b, .c, .t }).convert(.f32);
        var weight = self.dec_in_proj.weight;
        while (weight.rank() > 2) weight = weight.squeeze(-1);
        return (zml.nn.Linear.init(weight.withTags(.{ .dout, .d }), self.dec_in_proj.bias, .d))
            .forward(x.rename(.{ .c = .d }))
            .rename(.{ .dout = .c })
            .transpose(.{ .b, .c, .t });
    }

    pub fn decode(input: DecodeInput) DecodeOutput {
        const self = input.model;
        var x = projectIn(self, input.latents);
        x = self.conv_pre.forward(x);
        const n_up = self.ups.len;
        const n_k: usize = 3;
        for (0..n_up) |i| {
            x = self.ups[i].forward(x);
            var acc = self.resblocks[i * n_k].forward(x);
            var j: usize = 1;
            while (j < n_k) : (j += 1) {
                acc = acc.add(self.resblocks[i * n_k + j].forward(x));
            }
            x = acc.scale(1.0 / @as(f32, @floatFromInt(n_k)));
        }
        x = self.activation_post.forward(x);
        x = self.conv_post.forward(x);
        const one = zml.Tensor.scalar(1.0, x.dtype());
        const neg = zml.Tensor.scalar(-1.0, x.dtype());
        return .{ .wav = x.minimum(one).maximum(neg) };
    }

    const Snake1d = struct {
        alpha: zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) Snake1d {
            return .{ .alpha = pickChannel(store, &.{ "alpha", "act.alpha" }) };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Snake1d)) void {
            self.alpha.deinit();
        }

        pub fn forward(self: Snake1d, x: zml.Tensor) zml.Tensor {
            const xf = x.convert(.f32).withPartialTags(.{ .b, .c, .t });
            const a = squeezeToTag(self.alpha.convert(.f32), .c).broad(xf.shape());
            const s = xf.mul(a).sin();
            return xf.add(s.mul(s).div(a.addConstant(1e-9))).convert(x.dtype());
        }
    };

    const ResidualUnit = struct {
        snake0: Snake1d,
        conv0: WNConv1d,
        snake1: Snake1d,
        conv1: WNConv1d,

        pub fn init(store: zml.io.TensorStore.View, dilation: i64) ResidualUnit {
            const inner = store.withPrefix("block");
            const pad = @divFloor(6 * dilation, 2);
            return .{
                .snake0 = .init(inner.withLayer(0)),
                .conv0 = .init(inner.withLayer(1), 1, dilation, pad),
                .snake1 = .init(inner.withLayer(2)),
                .conv1 = .init(inner.withLayer(3), 1, 1, 0),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(ResidualUnit)) void {
            Snake1d.unloadBuffers(&self.snake0);
            WNConv1d.unloadBuffers(&self.conv0);
            Snake1d.unloadBuffers(&self.snake1);
            WNConv1d.unloadBuffers(&self.conv1);
        }

        pub fn forward(self: ResidualUnit, x: zml.Tensor) zml.Tensor {
            var y = self.conv1.forward(self.snake1.forward(self.conv0.forward(self.snake0.forward(x))));
            const xt = x.withPartialTags(.{ .b, .c, .t });
            const yt = y.withPartialTags(.{ .b, .c, .t });
            if (xt.dim(.t) != yt.dim(.t)) {
                const pad = @divFloor(xt.dim(.t) - yt.dim(.t), 2);
                return yt.add(xt.slice1d(.t, .{ .start = pad, .end = xt.dim(.t) - pad }));
            }
            return yt.add(xt);
        }
    };

    const EncoderBlock = struct {
        unit0: ResidualUnit,
        unit1: ResidualUnit,
        unit2: ResidualUnit,
        snake: Snake1d,
        conv: WNConv1d,

        pub fn init(store: zml.io.TensorStore.View, stride: i64) EncoderBlock {
            const inner = store.withPrefix("block");
            const pad = std.math.divCeil(i64, stride, 2) catch stride;
            return .{
                .unit0 = .init(inner.withLayer(0), 1),
                .unit1 = .init(inner.withLayer(1), 3),
                .unit2 = .init(inner.withLayer(2), 9),
                .snake = .init(inner.withLayer(3)),
                .conv = .init(inner.withLayer(4), stride, 1, pad),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(EncoderBlock)) void {
            ResidualUnit.unloadBuffers(&self.unit0);
            ResidualUnit.unloadBuffers(&self.unit1);
            ResidualUnit.unloadBuffers(&self.unit2);
            Snake1d.unloadBuffers(&self.snake);
            WNConv1d.unloadBuffers(&self.conv);
        }

        pub fn forward(self: EncoderBlock, x: zml.Tensor) zml.Tensor {
            return self.conv.forward(self.snake.forward(self.unit2.forward(self.unit1.forward(self.unit0.forward(x)))));
        }
    };

    const GeGluMlp = struct {
        norm: LayerNormEnc,
        w0: zml.nn.Linear,
        w1: zml.nn.Linear,
        w2: zml.nn.Linear,

        pub fn init(store: zml.io.TensorStore.View) GeGluMlp {
            return .{
                .norm = .init(store.withPrefix("norm")),
                .w0 = conv1x1(store.withPrefix("w0")),
                .w1 = conv1x1(store.withPrefix("w1")),
                .w2 = conv1x1(store.withPrefix("w2")),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(GeGluMlp)) void {
            LayerNormEnc.unloadBuffers(&self.norm);
            zml.nn.Linear.unloadBuffers(&self.w0);
            zml.nn.Linear.unloadBuffers(&self.w1);
            zml.nn.Linear.unloadBuffers(&self.w2);
        }

        pub fn forward(self: GeGluMlp, x: zml.Tensor) zml.Tensor {
            const n = self.norm.forward(x);
            return self.w2.forward(self.w0.forward(n).gelu().mul(self.w1.forward(n)).rename(.{ .dout = .d })).rename(.{ .dout = .d });
        }
    };

    const LayerNormEnc = struct {
        weight: zml.Tensor,
        bias: ?zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) LayerNormEnc {
            return .{
                .weight = store.createTensor("weight", .{.d}, .replicated),
                .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(LayerNormEnc)) void {
            self.weight.deinit();
            if (self.bias) |*b| b.deinit();
        }

        pub fn forward(self: LayerNormEnc, x: zml.Tensor) zml.Tensor {
            return (zml.nn.LayerNorm{ .weight = self.weight, .bias = self.bias, .eps = 1e-5 }).forward(x.convert(.f32)).convert(x.dtype());
        }
    };

    const CausalAttn = struct {
        qkv: zml.nn.Linear,
        q_bias: zml.Tensor,
        v_bias: zml.Tensor,
        k_bias: zml.Tensor,
        proj: zml.nn.Linear,
        num_heads: i64,
        head_dim: i64,
        out_dim: i64,

        pub fn init(store: zml.io.TensorStore.View, in_dim: i64, out_dim: i64, num_heads: i64) CausalAttn {
            return .{
                .qkv = conv1x1(store.withPrefix("qkv")),
                .q_bias = store.createTensor("q_bias", .{.d}, .replicated),
                .v_bias = store.createTensor("v_bias", .{.d}, .replicated),
                .k_bias = pick(store, &.{ "zero_k_bias", "k_bias" }, .{.d}),
                .proj = conv1x1(store.withPrefix("proj")),
                .num_heads = num_heads,
                .head_dim = @divExact(in_dim, num_heads),
                .out_dim = out_dim,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(CausalAttn)) void {
            zml.nn.Linear.unloadBuffers(&self.qkv);
            self.q_bias.deinit();
            self.v_bias.deinit();
            self.k_bias.deinit();
            zml.nn.Linear.unloadBuffers(&self.proj);
        }

        pub fn forward(self: CausalAttn, x: zml.Tensor) zml.Tensor {
            const xt = x.withPartialTags(.{ .b, .s, .d });
            const seq = xt.dim(.s);
            var qkv = self.qkv.forward(xt);
            const bias = zml.Tensor.concatenate(&.{
                self.q_bias.convert(xt.dtype()).withTags(.{.dout}),
                self.k_bias.convert(xt.dtype()).withTags(.{.dout}),
                self.v_bias.convert(xt.dtype()).withTags(.{.dout}),
            }, .dout);
            qkv = qkv.add(bias.broad(qkv.shape()));
            const parts = qkv.chunkExact(.dout, 3);
            const q = parts[0].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .q });
            const k = parts[1].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .k });
            const v = parts[2].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .k });
            const q_i = zml.Tensor.arange(.{ .end = seq }, .f32).withTags(.{.q});
            const k_i = zml.Tensor.arange(.{ .end = seq }, .f32).withTags(.{.k});
            const neg = zml.Tensor.scalar(-1.0e9, .f32);
            const zero = zml.Tensor.scalar(0.0, .f32);
            const qk = zml.Shape.init(.{ .q = seq, .k = seq }, .f32);
            const mask = q_i.broad(qk).cmp(.GE, k_i.broad(qk)).select(zero, neg);
            var attn = zml.nn.sdpa(q, k, v, .{ .attn_mask = mask }).rename(.{ .q = .s });
            attn = attn.mean(.h).squeeze(.h);
            const pool = @divExact(self.head_dim, self.out_dim);
            attn = attn.splitAxis(.hd, .{ .d = self.out_dim, .k = pool }).mean(.k).squeeze(.k);
            return self.proj.forward(attn).rename(.{ .dout = .d });
        }
    };

    const AttnProjection = struct {
        norm1: LayerNormEnc,
        attn: CausalAttn,
        proj: zml.nn.Linear,
        norm3: LayerNormEnc,
        norm2: LayerNormEnc,
        mlp: GeGluMlp,

        pub fn init(store: zml.io.TensorStore.View, in_dim: i64, out_dim: i64) AttnProjection {
            return .{
                .norm1 = .init(store.withPrefix("norm1")),
                .attn = .init(store.withPrefix("attn"), in_dim, out_dim, 8),
                .proj = conv1x1(store.withPrefix("proj")),
                .norm3 = .init(store.withPrefix("norm3")),
                .norm2 = .init(store.withPrefix("norm2")),
                .mlp = .init(store.withPrefix("mlp")),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(AttnProjection)) void {
            LayerNormEnc.unloadBuffers(&self.norm1);
            CausalAttn.unloadBuffers(&self.attn);
            zml.nn.Linear.unloadBuffers(&self.proj);
            LayerNormEnc.unloadBuffers(&self.norm3);
            LayerNormEnc.unloadBuffers(&self.norm2);
            GeGluMlp.unloadBuffers(&self.mlp);
        }

        pub fn forward(self: AttnProjection, x: zml.Tensor) zml.Tensor {
            const xt = x.withPartialTags(.{ .b, .s, .d });
            var y = self.proj.forward(self.norm3.forward(xt)).rename(.{ .dout = .d });
            y = y.add(self.attn.forward(self.norm1.forward(xt)));
            return y.add(self.mlp.forward(self.norm2.forward(y)));
        }
    };

    pub fn decodeReady(store: zml.io.TensorStore.View) bool {
        return store.hasKey("dec_in_proj.weight") and
            (store.hasKey("decoder.conv_pre.weight_v") or store.hasKey("decoder.conv_pre.weight"));
    }

    pub fn encodeReady(store: zml.io.TensorStore.View) bool {
        return (store.hasKey("encoder.block.0.weight_v") or store.hasKey("encoder.block.0.weight")) and
            store.hasKey("mean_proj.weight");
    }

    pub const EncoderModel = struct {
        conv_in: WNConv1d,
        blocks: [5]EncoderBlock,
        snake: Snake1d,
        conv_out: WNConv1d,
        pre_block: AttnProjection,
        mean_proj: zml.nn.Linear,
        cfg: Config,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) EncoderModel {
            const enc = store.withPrefix("encoder.block");
            return .{
                .conv_in = .init(enc.withLayer(0), 1, 1, 3),
                .blocks = .{
                    .init(enc.withLayer(1), cfg.encoder_rates[0]),
                    .init(enc.withLayer(2), cfg.encoder_rates[1]),
                    .init(enc.withLayer(3), cfg.encoder_rates[2]),
                    .init(enc.withLayer(4), cfg.encoder_rates[3]),
                    .init(enc.withLayer(5), cfg.encoder_rates[4]),
                },
                .snake = .init(enc.withLayer(6)),
                .conv_out = .init(enc.withLayer(7), 1, 1, 1),
                .pre_block = .init(store.withPrefix("pre_block"), cfg.latent_dim, cfg.latent_channels),
                .mean_proj = conv1x1(store.withPrefix("mean_proj")),
                .cfg = cfg,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(EncoderModel)) void {
            WNConv1d.unloadBuffers(&self.conv_in);
            for (&self.blocks) |*block| EncoderBlock.unloadBuffers(block);
            Snake1d.unloadBuffers(&self.snake);
            WNConv1d.unloadBuffers(&self.conv_out);
            AttnProjection.unloadBuffers(&self.pre_block);
            zml.nn.Linear.unloadBuffers(&self.mean_proj);
        }
    };

    pub const EncodeInput = struct {
        model: EncoderModel,
        wav: zml.Tensor,
    };

    pub const EncodeOutput = struct {
        latents: zml.Tensor,
    };

    pub fn encode(input: EncodeInput) EncodeOutput {
        const self = input.model;
        var x = input.wav.withPartialTags(.{ .b, .c, .t }).convert(.f32);
        x = self.conv_in.forward(x);
        for (self.blocks) |block| x = block.forward(x);
        x = self.conv_out.forward(self.snake.forward(x));
        x = x.transpose(.{ .b, .t, .c }).rename(.{ .c = .d, .t = .s });
        x = self.pre_block.forward(x);
        x = self.mean_proj.forward(x).rename(.{ .dout = .c });
        if (x.shape().hasTag(.k) != null) x = x.squeeze(.k);
        return .{ .latents = x.transpose(.{ .b, .c, .s }).rename(.{ .s = .t }) };
    }

    pub const LoadedEncoder = struct {
        inner: EncoderModel,
        cfg: Config,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) LoadedEncoder {
            return .{ .inner = .init(store, cfg), .cfg = cfg };
        }

        pub fn loadBuffers(
            self: *const LoadedEncoder,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(EncoderModel) {
            var buffers = try zml.mem.bufferize(allocator, EncoderModel, &self.inner);
            errdefer EncoderModel.unloadBuffers(&buffers);
            var loader = try weights.initLoader(allocator, platform);
            defer loader.deinit();
            const now: std.Io.Timestamp = .now(io, .awake);
            try weights.populate(&loader, io, store, shardings, EncoderModel, &self.inner, &buffers, progress);
            log.info("loaded audio VAE encoder [{f}]", .{now.untilNow(io, .awake)});
            return buffers;
        }
    };

    pub const LoadedModel = struct {
        inner: Model,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
            var cfg = Config.official();
            if (try config_mod.parseOptional(FileConfig, allocator, io, repo, "config.json")) |parsed| {
                defer parsed.deinit();
                cfg = parsed.value.resolve();
            }
            log.info("audio vae: hop={d} latent_c={d} mean0={d:.4} std0={d:.4}", .{
                cfg.hop,
                cfg.latent_channels,
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

        pub fn loadBuffers(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(Model) {
            var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
            errdefer Model.unloadBuffers(&buffers, allocator);
            var loader = try weights.initLoader(allocator, platform);
            defer loader.deinit();
            const now: std.Io.Timestamp = .now(io, .awake);
            try weights.populate(&loader, io, store, shardings, Model, &self.inner, &buffers, progress);
            log.info("loaded audio VAE [{f}]", .{now.untilNow(io, .awake)});
            return buffers;
        }
    };

    pub fn snake(x: f32, alpha: f32) f32 {
        const a = alpha + 1e-9;
        const s = @sin(alpha * x);
        return x + (1.0 / a) * (s * s);
    }
};
