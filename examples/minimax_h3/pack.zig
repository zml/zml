//! Packed sequence, rectified-flow schedule, and 1×2×2 patchify.
//!
//!   1. Layout — text tokens, then the video patch grid; RoPE (t,h,w) per row
//!   2. Schedule — σ from t ∈ [1→0] with time-shift 12
//!   3. Patchify — THWC latents ↔ DiT tokens of width 96

const std = @import("std");
const config = @import("config.zig");

/// AdaLN modality tags. Audio (2) is in the checkpoint but unused here.
pub const tag_video: u8 = 0;
pub const tag_text: u8 = 1;

/// Official temporal span pattern along latent frames for video RoPE `t`.
const video_spans = [_]u32{ 1, 4, 4, 4, 4 };
/// Maps 24 fps latent frames onto the RoPE time axis (official packer uses 5/3).
const frame_rescale: f64 = 5.0 / 3.0;

// =============================================================================
// Sequence layout
// =============================================================================

/// One packed sequence: text rows, then video-patch rows.
///
/// `positions` is `[seq, 3]` (t, h, w). Text uses `(token_index, 0, 0)`.
/// Video RoPE `t` continues from `text_len` (official packing).
pub const Layout = struct {
    positions: []f32,
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

    /// AdaLN row = `timestep * n_modalities + token_tag`.
    pub fn writeAdalnIndices(self: Layout, out: []u32, timestep_indices: []const u32) void {
        for (out, timestep_indices, self.token_tags) |*a, t, tag| {
            a.* = t * @as(u32, @intCast(config.modality_count)) + tag;
        }
    }
};

/// Spatial RoPE axis for one latent dimension, scaled onto the official 32-unit canvas.
fn spatialAxis(dim: u32, sqrt_area: f64, out: []f32) []f32 {
    const count = dim / 2;
    const ratio = @as(f64, @floatFromInt(dim)) / sqrt_area;
    const left = (1.0 - ratio) / 2.0;
    const step = ratio / @as(f64, @floatFromInt(count));
    for (0..count) |i| out[i] = @floatCast((left + @as(f64, @floatFromInt(i)) * step) * 32.0);
    return out[0..count];
}

// =============================================================================
// Rectified-flow schedule
// =============================================================================

/// σ points for Euler. `--steps=N` is N values including terminal 0, so the
/// DiT runs N−1 times.
///
///   t ∈ [1 → 0],   σ = shift·t / (1 + (shift−1)·t)
pub const Schedule = struct {
    sigmas: []f32,

    pub fn init(allocator: std.mem.Allocator, shift: f32, n: u32) !Schedule {
        const sigmas = try allocator.alloc(f32, n);
        for (sigmas, 0..) |*sigma, i| {
            const t_lin = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
            sigma.* = shift * t_lin / (1.0 + (shift - 1.0) * t_lin);
        }
        return .{ .sigmas = sigmas };
    }

    pub fn deinit(self: Schedule, allocator: std.mem.Allocator) void {
        allocator.free(self.sigmas);
    }

    pub fn stepCount(self: Schedule) usize {
        return self.sigmas.len - 1;
    }

    /// Flow time `1 − σ` at step `i` (logging).
    pub fn time(self: Schedule, i: usize) f32 {
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

/// Build the packed sequence and the video σ schedule for this prompt length.
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
        token_tags[i] = tag_text;
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
                token_tags[idx] = tag_video;
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

// =============================================================================
// Patchify
// =============================================================================

fn thwcAt(tt: u32, hh: u32, ww: u32, ch: usize, h: u32, w: u32, c: u32) usize {
    return (((@as(usize, tt) * h + hh) * w + ww) * c) + ch;
}

/// Walk the 1×2×2 patch grid. `write=true` is patchify; `false` is unpatchify.
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

/// Gaussian noise in official `torch.randn` NCHW order (24, T, H, W), then patchify.
/// Walking NCHW first makes `--seed` match Python.
pub fn noise(allocator: std.mem.Allocator, seed: u64, t: u32, h: u32, w: u32, patch: [3]i64) ![]f32 {
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

/// DiT video tokens `{s, 96}` → THWC latents for the VAE.
pub fn unpatchify(allocator: std.mem.Allocator, src: []const f32, t: u32, h: u32, w: u32, c: u32, patch: [3]i64) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, t) * h * w * c);
    patchWalk(t, h, w, c, patch, false, src, out);
    return out;
}
