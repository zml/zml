const std = @import("std");

const config = @import("../core/config.zig");
const noise = @import("../model/noise.zig");

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
