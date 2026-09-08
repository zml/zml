//! Packed sequence, rectified-flow schedule, noise, and 1×2×2 unpatchify.
//!
//!   1. Layout — text, then audio, then the video patch grid; RoPE (t,h,w) per row
//!   2. Schedule — σ from t ∈ [1→0] (video shift 12, audio shift 3)
//!   3. Noise — N(0,1) video tokens then audio tokens
//!   4. Unpatchify — DiT video tokens `{s, 96}` → THWC latents for the VAE

const std = @import("std");
const config = @import("config.zig");

const Modality = config.Modality;

/// Temporal span pattern along latent frames for video RoPE `t` (official 24 fps
/// schedule: 1 then repeating 4s, scaled by 5/3 onto the RoPE time axis).
const video_spans = [_]u32{ 1, 4, 4, 4, 4 };
/// Maps 24 fps latent frames onto the RoPE time axis (5/3).
const frame_rescale: f64 = 5.0 / 3.0;

/// One packed sequence: text rows, then audio rows, then video-patch rows.
///
/// `positions` is `[seq][t, h, w]`. Text uses `(token_index, 0, 0)`.
/// Audio uses `(text_len + t, 0, w_low|w_high)`. Video RoPE `t` shares that origin.
pub const Layout = struct {
    positions: [][3]f32,
    text_len: u32,
    audio_len: u32,
    video_len: u32,

    pub fn seqLen(self: Layout) u32 {
        return self.text_len + self.audio_len + self.video_len;
    }

    pub fn videoStart(self: Layout) u32 {
        return self.text_len + self.audio_len;
    }

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
    }
};

/// Video/text share one time, audio another. Smaller time first, then pad to
/// `config.timestep_slot_count` (checkpoint table width).
fn fillSlots(out: []f32, video_t: f32, audio_t: f32) struct { video: u32, audio: u32 } {
    std.debug.assert(out.len == config.timestep_slot_count);
    if (video_t == audio_t) {
        @memset(out, video_t);
        return .{ .video = 0, .audio = 0 };
    }
    if (video_t < audio_t) {
        out[0] = video_t;
        @memset(out[1..], audio_t);
        return .{ .video = 0, .audio = 1 };
    }
    out[0] = audio_t;
    @memset(out[1..], video_t);
    return .{ .video = 1, .audio = 0 };
}

/// Per-row times: video/text at `video_t`, audio at `audio_t`.
/// AdaLN row = `slot * n_modalities + modality`.
pub fn writeRowPlan(
    layout: Layout,
    video_t: f32,
    audio_t: f32,
    timestep_indices: []u32,
    adaln_indices: []u32,
    unique_out: []f32,
) void {
    const seq = layout.seqLen();
    std.debug.assert(timestep_indices.len == seq);
    std.debug.assert(adaln_indices.len == seq);

    const slots = fillSlots(unique_out, video_t, audio_t);
    const n_mod: u32 = @intCast(config.modality_count);
    const text_end = layout.text_len;
    const audio_end = layout.videoStart();

    @memset(timestep_indices[0..text_end], slots.video);
    @memset(timestep_indices[text_end..audio_end], slots.audio);
    @memset(timestep_indices[audio_end..seq], slots.video);

    @memset(adaln_indices[0..text_end], slots.video * n_mod + @intFromEnum(Modality.text));
    @memset(adaln_indices[text_end..audio_end], slots.audio * n_mod + @intFromEnum(Modality.audio));
    @memset(adaln_indices[audio_end..seq], slots.video * n_mod + @intFromEnum(Modality.video));
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
    const positions = try allocator.alloc([3]f32, n);
    errdefer allocator.free(positions);

    const sqrt_area = @sqrt(@as(f64, @floatFromInt(geo.latent_h * geo.latent_w)));
    var h_buf: [256]f32 = undefined;
    var w_buf: [256]f32 = undefined;
    std.debug.assert(geo.latent_h / 2 <= h_buf.len);
    std.debug.assert(geo.latent_w / 2 <= w_buf.len);
    const h_axis = spatialAxis(geo.latent_h, sqrt_area, &h_buf);
    const w_axis = spatialAxis(geo.latent_w, sqrt_area, &w_buf);

    for (0..text_len) |i| {
        positions[i] = .{ @floatFromInt(i), 0, 0 };
    }

    var cursor: f64 = @floatFromInt(text_len);
    const widths = [_]f32{ w_axis[0], w_axis[w_axis.len - 1] };
    var a: u32 = 0;
    for (widths) |w| {
        for (0..geo.audio_t) |t| {
            positions[text_len + a] = .{
                @floatCast(cursor + @as(f64, @floatFromInt(t))),
                0,
                w,
            };
            a += 1;
        }
    }

    var v: u32 = 0;
    for (0..geo.latent_t) |ti| {
        for (h_axis) |h| {
            for (w_axis) |w| {
                positions[text_len + geo.audio_tokens + v] = .{ @floatCast(cursor), h, w };
                v += 1;
            }
        }
        cursor += frame_rescale * @as(f64, @floatFromInt(video_spans[ti % video_spans.len]));
    }

    return .{
        .layout = .{
            .positions = positions,
            .text_len = text_len,
            .audio_len = geo.audio_tokens,
            .video_len = geo.video_tokens,
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
    for (0..t / pt) |ti| {
        const tt: u32 = @as(u32, @intCast(ti)) * pt;
        for (0..h / ph) |hi| {
            const hh: u32 = @as(u32, @intCast(hi)) * ph;
            for (0..w / pw) |wi| {
                const ww: u32 = @as(u32, @intCast(wi)) * pw;
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
