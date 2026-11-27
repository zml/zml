//! Packed sequence, rectified-flow schedule, and noise.
//!
//!   1. Layout — text, then audio, then the video patch grid; RoPE (t,h,w) per row
//!   2. Schedule — σ from t ∈ [1→0] (video/audio shifts from scheduler JSON)
//!   3. Noise — N(0,1) video tokens then audio tokens

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

    fn init(allocator: std.mem.Allocator, spec: ScheduleSpec, shift: f32) error{ OutOfMemory, TooFewSteps }!Schedule {
        if (spec == .h3 and spec.h3.steps < 2) return error.TooFewSteps;
        const evaluations = spec.evaluations();
        const sigmas = try allocator.alloc(f32, evaluations + 1);
        for (sigmas[0..evaluations], 0..) |*sigma, i| {
            const raw_t: f32 = switch (spec) {
                .h3 => 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(evaluations)),
                .pi2, .pi4 => blk: {
                    // The reference uses numpy.linspace(dtype=float32): compute raw
                    // positions in double, then apply the shift in float32.
                    const start = spec.maxRawTime();
                    const intervals: f64 = @floatFromInt(evaluations - 1);
                    const end = start * 0.6 / (intervals + 0.6);
                    break :blk @floatCast(if (i == evaluations - 1) end else start + @as(f64, @floatFromInt(i)) * ((end - start) / intervals));
                },
            };
            sigma.* = shift * raw_t / (1.0 + (shift - 1.0) * raw_t);
        }
        sigmas[evaluations] = 0;
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

pub const ScheduleSpec = union(enum) {
    h3: struct { steps: u32, video_shift: f32, audio_shift: f32 },
    pi2,
    pi4,

    pub fn evaluations(self: ScheduleSpec) u32 {
        return switch (self) {
            .h3 => |h3| h3.steps - 1,
            .pi2 => 2,
            .pi4 => 4,
        };
    }

    /// Starting raw timestep, before applying the sigma shift.
    /// Source: LakonLab https://github.com/Lakonik/LakonLab/blob/2ee2b334374ec91b7cee371a85a524bf4ba644d3/demo/example_pih3_pipeline.py#L14
    /// Those values are the Pi-H3 inference recipe’s fixed MAX_RAW_T choices
    fn maxRawTime(self: ScheduleSpec) f64 {
        return switch (self) {
            .h3 => 1.0,
            .pi2 => 0.93,
            .pi4 => 0.96,
        };
    }
};

pub const Variant = std.meta.Tag(ScheduleSpec);

pub const Packed = struct {
    layout: Layout,
    video: Schedule,
    audio: Schedule,

    pub fn init(allocator: std.mem.Allocator, geo: config.Geometry, text_len: u32, spec: ScheduleSpec) !Packed {
        const shifts: struct { video: f32, audio: f32 } = switch (spec) {
            .h3 => |h3| .{
                .video = h3.video_shift,
                .audio = h3.audio_shift,
            },
            .pi2, .pi4 => .{
                .video = 12,
                .audio = 3,
            },
        };
        const video = try Schedule.init(allocator, spec, shifts.video);
        errdefer video.deinit(allocator);
        const audio = try Schedule.init(allocator, spec, shifts.audio);
        errdefer audio.deinit(allocator);

        return .{
            .video = video,
            .audio = audio,
            .layout = try makeLayout(allocator, geo, text_len),
        };
    }

    pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
        self.layout.deinit(allocator);
        self.video.deinit(allocator);
        self.audio.deinit(allocator);
    }
};

fn makeLayout(allocator: std.mem.Allocator, geo: config.Geometry, text_len: u32) !Layout {
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
        .positions = positions,
        .text_len = text_len,
        .audio_len = geo.audio_tokens,
        .video_len = geo.video_tokens,
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

    @memset(adaln_indices[0..text_end], slots.video * n_mod + @backingInt(Modality.text));
    @memset(adaln_indices[text_end..audio_end], slots.audio * n_mod + @backingInt(Modality.audio));
    @memset(adaln_indices[audio_end..seq], slots.video * n_mod + @backingInt(Modality.video));
}
