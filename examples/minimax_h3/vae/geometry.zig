const std = @import("std");

const config = @import("../core/config.zig");
const noise = @import("../model/noise.zig");
const packing = @import("../model/packing.zig");

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

fn nchwTileN(channels: u32, t: u32, h: u32, w: u32) usize {
    return @as(usize, channels) * t * h * w;
}

fn nchwIndex(c: u32, t: u32, y: u32, x: u32, tt: u32, h: u32, w: u32) usize {
    return ((((c * tt + t) * h) + y) * w) + x;
}

/// Official `_blend` along H (`dim=-2`): `b[y] = a[tail+y]*(1-y/E) + b[y]*(y/E)`.
fn blendH(a: []const f32, b: []f32, channels: u32, t: u32, h: u32, w: u32, extent: u32) void {
    const e = @min(h, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < e) : (y += 1) {
                const wb = @as(f32, @floatFromInt(y)) / ef;
                const wa = 1.0 - wb;
                const ay = h - e + y;
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const ai = nchwIndex(c, ti, ay, x, t, h, w);
                    const bi = nchwIndex(c, ti, y, x, t, h, w);
                    b[bi] = a[ai] * wa + b[bi] * wb;
                }
            }
        }
    }
}

/// Official `_blend` along W (`dim=-1`).
fn blendW(a: []const f32, b: []f32, channels: u32, t: u32, h: u32, w: u32, extent: u32) void {
    const e = @min(w, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < e) : (x += 1) {
                    const wb = @as(f32, @floatFromInt(x)) / ef;
                    const wa = 1.0 - wb;
                    const ai = nchwIndex(c, ti, y, w - e + x, t, h, w);
                    const bi = nchwIndex(c, ti, y, x, t, h, w);
                    b[bi] = a[ai] * wa + b[bi] * wb;
                }
            }
        }
    }
}

pub fn copyNchwCrop(
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
    std.debug.assert(out_y + use_h <= dst_h and out_x + use_w <= dst_w);
    std.debug.assert(use_h <= src_h and use_w <= src_w);
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < use_h) : (y += 1) {
                const si = nchwIndex(c, ti, y, 0, t, src_h, src_w);
                const di = nchwIndex(c, ti, out_y + y, out_x, t, dst_h, dst_w);
                @memcpy(dst[di..][0..use_w], src[si..][0..use_w]);
            }
        }
    }
}

/// Official `_stitch_tiles`: blend each tile with the *original* neighbor tiles,
/// crop the outgoing overlap, concatenate. Canvas bilinear is not the same.
pub const NchwStitcher = struct {
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
        n_y: u32,
        n_x: u32,
        y_overlaps: []const u32,
        x_overlaps: []const u32,
    ) !NchwStitcher {
        std.debug.assert(n_y >= 1 and n_x >= 1);
        std.debug.assert(y_overlaps.len + 1 == n_y);
        std.debug.assert(x_overlaps.len + 1 == n_x);
        const tile_n = nchwTileN(channels, t, tile_h, tile_w);
        const y_ov = try allocator.dupe(u32, y_overlaps);
        errdefer allocator.free(y_ov);
        const x_ov = try allocator.dupe(u32, x_overlaps);
        errdefer allocator.free(x_ov);
        const prev_row = try allocator.alloc(f32, n_x * tile_n);
        errdefer allocator.free(prev_row);
        const curr_row = try allocator.alloc(f32, n_x * tile_n);
        errdefer allocator.free(curr_row);
        const work = try allocator.alloc(f32, tile_n);
        return .{
            .acc = acc,
            .prev_row = prev_row,
            .curr_row = curr_row,
            .work = work,
            .channels = channels,
            .t = t,
            .acc_h = acc_h,
            .acc_w = acc_w,
            .tile_h = tile_h,
            .tile_w = tile_w,
            .n_y = n_y,
            .n_x = n_x,
            .y_overlaps = y_ov,
            .x_overlaps = x_ov,
            .out_y = 0,
            .out_x = 0,
        };
    }

    pub fn deinit(self: *NchwStitcher, allocator: std.mem.Allocator) void {
        allocator.free(self.prev_row);
        allocator.free(self.curr_row);
        allocator.free(self.work);
        allocator.free(self.y_overlaps);
        allocator.free(self.x_overlaps);
    }

    fn tileN(self: NchwStitcher) usize {
        return nchwTileN(self.channels, self.t, self.tile_h, self.tile_w);
    }

    pub fn push(self: *NchwStitcher, yi: u32, xi: u32, tile: []const f32) void {
        const n = self.tileN();
        std.debug.assert(yi < self.n_y and xi < self.n_x);
        std.debug.assert(tile.len >= n);
        @memcpy(self.curr_row[xi * n ..][0..n], tile[0..n]);
        @memcpy(self.work[0..n], tile[0..n]);
        if (yi > 0) {
            blendH(self.prev_row[xi * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.y_overlaps[yi - 1]);
        }
        if (xi > 0) {
            blendW(self.curr_row[(xi - 1) * n ..][0..n], self.work, self.channels, self.t, self.tile_h, self.tile_w, self.x_overlaps[xi - 1]);
        }
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

fn f32ToF16Bits(value: f32) u16 {
    return @as(u16, @bitCast(@as(f16, @floatCast(value))));
}

fn f16BitsToF32(bits: u16) f32 {
    return @as(f16, @bitCast(bits));
}

/// Visual-condition posterior: mean + std * randn(seed=42), then FP16 round-trip.
pub fn sampleVisualPosteriorNchw(
    allocator: std.mem.Allocator,
    moments_nchw: []const f32,
    t: u32,
    h: u32,
    w: u32,
) ![]f32 {
    const spatial = @as(usize, t) * h * w;
    std.debug.assert(moments_nchw.len >= spatial * 48);
    const out = try allocator.alloc(f32, spatial * 24);
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
    packing.nchwToThwc(out, nchw[0..out.len], channels, t, h, w);
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

/// Reference videos follow the official encoder's snap-down rule. The vision
/// conditioner still sees every normalized frame; only the visual VAE input is
/// shortened to the largest `clip_length * n + tokens_chunk_size` prefix.
pub fn referenceVideoFrameCount(spec: VisualSpec, frames: u32) u32 {
    const tail = spec.tokensChunkSize();
    const minimum = spec.clip_length + tail;
    if (frames < minimum) return frames;
    return ((frames - tail) / spec.clip_length) * spec.clip_length + tail;
}

pub fn vitCoords(dim: u32, out: []f32) []f32 {
    std.debug.assert(out.len >= dim);
    const d: f32 = @floatFromInt(dim);
    for (0..dim) |i| {
        out[i] = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / d) - 1.0;
    }
    return out[0..dim];
}
