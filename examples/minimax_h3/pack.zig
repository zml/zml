//! Packed sequence, rectified-flow schedule, noise, and 1×2×2 patchify.
//!
//!   1. Layout — text, then audio, then the video patch grid; RoPE (t,h,w) per row
//!   2. Schedule — σ from t ∈ [1→0] (video shift 12, audio shift 3)
//!   3. Noise — N(0,1) video tokens then audio tokens
//!   4. Patchify — THWC latents ↔ DiT tokens of width 96

const std = @import("std");
const config = @import("config.zig");

/// AdaLN modality tags: video=0, text=1, audio=2.
const tag_video: u8 = 0;
const tag_text: u8 = 1;
const tag_audio: u8 = 2;

/// Temporal span pattern along latent frames for video RoPE `t`.
const video_spans = [_]u32{ 1, 4, 4, 4, 4 };
/// Maps 24 fps latent frames onto the RoPE time axis (5/3).
const frame_rescale: f64 = 5.0 / 3.0;

/// AdaLN / time-embed table capacity (at most 4 distinct row times).
pub const timestep_slot_count: u32 = 4;

/// One packed sequence: text rows, then audio rows, then video-patch rows.
///
/// `positions` is `[seq, 3]` (t, h, w). Text uses `(token_index, 0, 0)`.
/// Audio uses `(text_len + t, 0, w_low|w_high)`. Video RoPE `t` continues after audio.
pub const Layout = struct {
    positions: []f32,
    token_tags: []u8,
    text_indices: []u32,
    audio_indices: []u32,
    video_indices: []u32,

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.token_tags);
        allocator.free(self.text_indices);
        allocator.free(self.audio_indices);
        allocator.free(self.video_indices);
    }

    pub fn seqLen(self: Layout) u32 {
        return @intCast(self.token_tags.len);
    }
};

fn padUnique(out: []f32, unique: []const f32) void {
    if (out.len == 0 or unique.len == 0) return;
    const n = @min(out.len, unique.len);
    @memcpy(out[0..n], unique[0..n]);
    for (n..out.len) |i| out[i] = unique[n - 1];
}

fn sortAscending(values: []f32) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const key = values[i];
        var j: usize = i;
        while (j > 0 and values[j - 1] > key) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = key;
    }
}

/// Distinct row times, sorted, at most 4.
fn uniqueSorted(values: []const f32, out: *[timestep_slot_count]f32) u32 {
    var n: u32 = 0;
    for (values) |v| {
        var seen = false;
        for (out[0..n]) |u| {
            if (u == v) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        if (n >= timestep_slot_count) std.debug.panic("too many unique timesteps", .{});
        out[n] = v;
        n += 1;
    }
    sortAscending(out[0..n]);
    return n;
}

fn indexOfEqual(values: []const f32, needle: f32) u32 {
    for (values, 0..) |v, i| {
        if (v == needle) return @intCast(i);
    }
    std.debug.panic("timestep missing from unique set", .{});
}

/// Per-row times: video/text at `video_t`, audio at `audio_t`. Then unique-sort
/// into 4 AdaLN slots.
pub fn writeRowPlan(
    layout: Layout,
    video_t: f32,
    audio_t: f32,
    row_ts: []f32,
    timestep_indices: []u32,
    unique_out: []f32,
) void {
    std.debug.assert(row_ts.len == layout.seqLen());
    @memset(row_ts, video_t);
    for (layout.audio_indices) |idx| row_ts[idx] = audio_t;
    var unique: [timestep_slot_count]f32 = undefined;
    const n = uniqueSorted(row_ts, &unique);
    padUnique(unique_out, unique[0..n]);
    for (timestep_indices, row_ts) |*idx, t| idx.* = indexOfEqual(unique[0..n], t);
}

/// AdaLN row = `slot * n_modalities + token_tag`.
pub fn writeAdalnIndices(out: []u32, timestep_indices: []const u32, token_tags: []const u8) void {
    for (out, timestep_indices, token_tags) |*a, t, tag| {
        a.* = t * @as(u32, @intCast(config.modality_count)) + tag;
    }
}

/// Spatial RoPE axis for one latent dimension, scaled onto a 32-unit canvas.
fn spatialAxis(dim: u32, sqrt_area: f64, out: []f32) []f32 {
    const count = dim / 2;
    const ratio = @as(f64, @floatFromInt(dim)) / sqrt_area;
    const left = (1.0 - ratio) / 2.0;
    const step = ratio / @as(f64, @floatFromInt(count));
    for (0..count) |i| out[i] = @floatCast((left + @as(f64, @floatFromInt(i)) * step) * 32.0);
    return out[0..count];
}

/// σ points for Euler. `--steps=N` is N values including terminal 0, so the
/// DiT runs N−1 times.
///
///   t ∈ [1 → 0],   σ = shift·t / (1 + (shift−1)·t)
pub const Schedule = struct {
    sigmas: []f32,

    pub fn init(allocator: std.mem.Allocator, shift: f32, n: u32) error{ OutOfMemory, TooFewSteps }!Schedule {
        if (n < 2) return error.TooFewSteps;
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

    /// Flow time `1 − σ` at step `i`.
    pub fn time(self: Schedule, i: usize) f32 {
        return 1.0 - self.sigmas[i];
    }
};

pub const Packed = struct {
    layout: Layout,
    video: Schedule,
    audio: Schedule,

    pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
        self.layout.deinit(allocator);
        self.video.deinit(allocator);
        self.audio.deinit(allocator);
    }
};

/// Build the packed sequence and video/audio σ schedules for this prompt length.
pub fn pack(allocator: std.mem.Allocator, geo: config.Geometry, text_len: u32, steps: u32) !Packed {
    const video = try Schedule.init(allocator, config.video_shift, steps);
    errdefer video.deinit(allocator);
    const audio = try Schedule.init(allocator, config.audio_shift, steps);
    errdefer audio.deinit(allocator);

    const n = text_len + geo.audio_tokens + geo.video_tokens;
    const positions = try allocator.alloc(f32, n * 3);
    errdefer allocator.free(positions);
    const token_tags = try allocator.alloc(u8, n);
    errdefer allocator.free(token_tags);
    const text_indices = try allocator.alloc(u32, text_len);
    errdefer allocator.free(text_indices);
    const audio_indices = try allocator.alloc(u32, geo.audio_tokens);
    errdefer allocator.free(audio_indices);
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
    const widths = [_]f32{ w_axis[0], w_axis[w_axis.len - 1] };
    var a: u32 = 0;
    for (widths) |w| {
        for (0..geo.audio_t) |t| {
            const idx = text_len + a;
            audio_indices[a] = idx;
            positions[idx * 3 + 0] = @floatCast(cursor + @as(f64, @floatFromInt(t)));
            positions[idx * 3 + 1] = 0;
            positions[idx * 3 + 2] = w;
            token_tags[idx] = tag_audio;
            a += 1;
        }
    }
    cursor += @as(f64, @floatFromInt(geo.audio_t));

    var v: u32 = 0;
    for (0..geo.latent_t) |ti| {
        for (h_axis) |h| {
            for (w_axis) |w| {
                const idx = text_len + geo.audio_tokens + v;
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
        .layout = .{
            .positions = positions,
            .token_tags = token_tags,
            .text_indices = text_indices,
            .audio_indices = audio_indices,
            .video_indices = video_indices,
        },
        .video = video,
        .audio = audio,
    };
}

pub const Latents = struct {
    video: []f32,
    audio: []f32,

    pub fn deinit(self: Latents, allocator: std.mem.Allocator) void {
        allocator.free(self.video);
        allocator.free(self.audio);
    }
};

/// N(0,1) video tokens `{s, 96}` then audio `{s, 32}`.
pub fn noise(allocator: std.mem.Allocator, seed: u64, geo: config.Geometry) !Latents {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    const video = try allocator.alloc(f32, @as(usize, geo.video_tokens) * geo.video_patch_dim);
    errdefer allocator.free(video);
    for (video) |*x| x.* = r.floatNorm(f32);
    const audio = try allocator.alloc(f32, @as(usize, geo.audio_tokens) * geo.audio_dim);
    errdefer allocator.free(audio);
    for (audio) |*x| x.* = r.floatNorm(f32);
    return .{ .video = video, .audio = audio };
}

fn thwcAt(tt: u32, hh: u32, ww: u32, ch: usize, h: u32, w: u32, c: u32) usize {
    return (((@as(usize, tt) * h + hh) * w + ww) * c) + ch;
}

/// Scatter 1×2×2 DiT tokens `{s, 96}` back onto the THWC latent grid.
fn unpatchWalk(t: u32, h: u32, w: u32, c: u32, patch: [3]i64, src: []const f32, dst: []f32) void {
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
                                dst[base] = src[row * width + i];
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

/// DiT video tokens `{s, 96}` → THWC latents for the VAE.
pub fn unpatchify(allocator: std.mem.Allocator, src: []const f32, t: u32, h: u32, w: u32, c: u32, patch: [3]i64) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, t) * h * w * c);
    unpatchWalk(t, h, w, c, patch, src, out);
    return out;
}
