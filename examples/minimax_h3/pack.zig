//! Packed sequence, rectified-flow schedule, and noise.
//!
//!   1. Layout — T2VA is text|audio|video. Cond modes append in official order
//!      (text, each ref block, target audio, target video) and scatter into DiT.
//!   2. Schedule — σ from t ∈ [1→0] (video/audio shifts from scheduler JSON)
//!   3. Noise — N(0,1) video tokens then audio. Cond video mixes at 0.999.

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
    cond_video_len: u32 = 0,
    cond_audio_len: u32 = 0,
    token_tags: []u8 = &.{},
    text_indices: []u32 = &.{},
    video_indices: []u32 = &.{},
    audio_indices: []u32 = &.{},

    pub fn seqLen(self: Layout) u32 {
        return self.text_len + self.audio_len + self.video_len;
    }

    pub fn videoStart(self: Layout) u32 {
        return self.text_len + self.audio_len;
    }

    pub fn scatter(self: Layout) bool {
        return self.video_indices.len != 0;
    }

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        if (self.token_tags.len != 0) allocator.free(self.token_tags);
        if (self.text_indices.len != 0) allocator.free(self.text_indices);
        if (self.video_indices.len != 0) allocator.free(self.video_indices);
        if (self.audio_indices.len != 0) allocator.free(self.audio_indices);
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
    /// Encoded patches. Borrowed from `mode.Encoded`.
    cond_video: []const f32 = &.{},
    cond_audio: []const f32 = &.{},

    pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
        self.layout.deinit(allocator);
        self.video.deinit(allocator);
        self.audio.deinit(allocator);
    }
};

/// Build the packed sequence and video/audio σ schedules for this prompt length.
pub fn pack(allocator: std.mem.Allocator, geo: config.Geometry, text_len: u32, steps: u32, video_shift: f32, audio_shift: f32) !Packed {
    const video = try Schedule.init(allocator, video_shift, steps);
    errdefer video.deinit(allocator);
    const audio = try Schedule.init(allocator, audio_shift, steps);
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
        cursor += videoSpan(@intCast(ti));
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

/// Condition rows sit at `max(video_t, 0.999)`.
const visual_cond_timestep: f32 = 0.999;

/// N(0,1) video tokens `{s, 96}` then audio `{s, 32}`. Conditioned video
/// rows mix `0.999 * clean + 0.001 * noise`; cond audio is copied through.
pub fn noise(allocator: std.mem.Allocator, seed: u64, geo: config.Geometry, packed_run: Packed) !Latents {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    const video = try allocator.alloc(f32, @as(usize, geo.video_tokens) * geo.video_patch_dim);
    errdefer allocator.free(video);
    for (video) |*x| x.* = r.floatNorm(f32);
    if (packed_run.cond_video.len != 0) {
        if (packed_run.cond_video.len > video.len) return error.ConditionPatchSize;
        for (video[0..packed_run.cond_video.len], packed_run.cond_video) |*dst, clean| {
            dst.* = visual_cond_timestep * clean + (1.0 - visual_cond_timestep) * dst.*;
        }
    }
    const audio = try allocator.alloc(f32, @as(usize, geo.audio_tokens) * geo.audio_dim);
    errdefer allocator.free(audio);
    if (packed_run.cond_audio.len != 0) {
        if (packed_run.cond_audio.len > audio.len) return error.AudioNoiseSize;
        @memcpy(audio[0..packed_run.cond_audio.len], packed_run.cond_audio);
    }
    for (audio[packed_run.cond_audio.len..]) |*x| x.* = r.floatNorm(f32);
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

fn indexOf(values: []const f32, n: u32, needle: f32) u32 {
    for (values[0..n], 0..) |v, i| {
        if (v == needle) return @intCast(i);
    }
    unreachable;
}

fn fillUnique(out: []f32, values: []const f32) u32 {
    std.debug.assert(out.len == config.timestep_slot_count);
    var unique: [config.timestep_slot_count]f32 = undefined;
    var n: u32 = 0;
    for (values) |v| {
        var seen = false;
        for (unique[0..n]) |u| {
            if (u == v) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        unique[n] = v;
        n += 1;
    }
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const key = unique[i];
        var j: usize = i;
        while (j > 0 and unique[j - 1] > key) : (j -= 1) unique[j] = unique[j - 1];
        unique[j] = key;
    }
    @memcpy(out[0..n], unique[0..n]);
    for (n..out.len) |k| out[k] = unique[n - 1];
    return n;
}

fn videoSpan(frame: u32) f64 {
    return frame_rescale * @as(f64, @floatFromInt(video_spans[frame % video_spans.len]));
}

fn videoDuration(latent_t: u32) f64 {
    var total: f64 = 0;
    for (0..latent_t) |t| total += videoSpan(@intCast(t));
    return total;
}

const Packer = struct {
    allocator: std.mem.Allocator,
    positions: std.ArrayList([3]f32) = .empty,
    tags: std.ArrayList(u8) = .empty,
    text_indices: std.ArrayList(u32) = .empty,
    video_indices: std.ArrayList(u32) = .empty,
    audio_indices: std.ArrayList(u32) = .empty,

    fn init(allocator: std.mem.Allocator) Packer {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Packer) void {
        self.positions.deinit(self.allocator);
        self.tags.deinit(self.allocator);
        self.text_indices.deinit(self.allocator);
        self.video_indices.deinit(self.allocator);
        self.audio_indices.deinit(self.allocator);
    }

    fn row(self: *const Packer) u32 {
        return @intCast(self.positions.items.len);
    }

    fn append(self: *Packer, pos: [3]f32, tag: Modality) !void {
        try self.positions.append(self.allocator, pos);
        try self.tags.append(self.allocator, @intFromEnum(tag));
    }

    fn appendText(self: *Packer, text_len: u32, tags: []const u8) !void {
        const start = self.row();
        for (0..text_len) |i| {
            const tag: Modality = if (i < tags.len) @enumFromInt(tags[i]) else .text;
            try self.append(.{ @floatFromInt(i), 0, 0 }, tag);
            try self.text_indices.append(self.allocator, start + @as(u32, @intCast(i)));
        }
    }

    fn appendGrid(self: *Packer, h_axis: []const f32, w_axis: []const f32, latent_t: u32, start_t: f64) !f64 {
        var cursor = start_t;
        for (0..latent_t) |ti| {
            for (h_axis) |h| {
                for (w_axis) |w| {
                    try self.video_indices.append(self.allocator, self.row());
                    try self.append(.{ @floatCast(cursor), h, w }, .video);
                }
            }
            cursor += videoSpan(@intCast(ti));
        }
        return cursor;
    }

    fn appendAudio(self: *Packer, length: u32, cursor: f64, w_low: f32, w_high: f32) !u32 {
        const start = self.row();
        const widths = [_]f32{ w_low, w_high };
        for (widths) |w| {
            for (0..length) |t| {
                try self.audio_indices.append(self.allocator, self.row());
                try self.append(.{
                    @floatCast(cursor + @as(f64, @floatFromInt(t))),
                    0,
                    w,
                }, .audio);
            }
        }
        return self.row() - start;
    }

    fn finish(self: *Packer, text_len: u32, audio_len: u32, video_len: u32, cond_n: u32, cond_a: u32) !Layout {
        return .{
            .positions = try self.positions.toOwnedSlice(self.allocator),
            .text_len = text_len,
            .audio_len = audio_len,
            .video_len = video_len,
            .cond_video_len = cond_n,
            .cond_audio_len = cond_a,
            .token_tags = try self.tags.toOwnedSlice(self.allocator),
            .text_indices = try self.text_indices.toOwnedSlice(self.allocator),
            .video_indices = try self.video_indices.toOwnedSlice(self.allocator),
            .audio_indices = try self.audio_indices.toOwnedSlice(self.allocator),
        };
    }
};

/// Official cond packing: append text, then each ref (or keyframes), then target audio, then target video.
pub fn packCond(
    allocator: std.mem.Allocator,
    geo: config.Geometry,
    text_len: u32,
    steps: u32,
    video_shift: f32,
    audio_shift: f32,
    conds: []const CondClip,
    audios: []const CondAudio,
    refs: []const RefBlock,
    text_tags: []const u8,
) !Packed {
    const video = try Schedule.init(allocator, video_shift, steps);
    errdefer video.deinit(allocator);
    const audio = try Schedule.init(allocator, audio_shift, steps);
    errdefer audio.deinit(allocator);

    var cond_n: u32 = 0;
    for (conds) |c| cond_n += condTokens(c);
    const cond_a = condAudioTokens(audios);
    const video_len = cond_n + geo.video_tokens;
    const audio_len = cond_a + geo.audio_tokens;

    var p = Packer.init(allocator);
    errdefer p.deinit();

    const sqrt_area = @sqrt(@as(f64, @floatFromInt(geo.latent_h * geo.latent_w)));
    var h_buf: [256]f32 = undefined;
    var w_buf: [256]f32 = undefined;
    std.debug.assert(geo.latent_h / 2 <= h_buf.len);
    std.debug.assert(geo.latent_w / 2 <= w_buf.len);
    const h_axis = spatialAxis(geo.latent_h, sqrt_area, &h_buf);
    const w_axis = spatialAxis(geo.latent_w, sqrt_area, &w_buf);

    try p.appendText(text_len, text_tags);

    if (refs.len == 0) {
        // First/last-frame only: conds share the target timeline. Image --refs must
        // go through `refs` so the canvas starts after each still's rotary slot.
        const duration = videoDuration(geo.latent_t);
        for (conds) |cond| {
            var ch_buf: [256]f32 = undefined;
            var cw_buf: [256]f32 = undefined;
            const area = @sqrt(@as(f64, @floatFromInt(cond.latent_h * cond.latent_w)));
            const ch = spatialAxis(cond.latent_h, area, &ch_buf);
            const cw = spatialAxis(cond.latent_w, area, &cw_buf);
            const start_t: f64 = if (cond.keyframe_index == 0)
                @floatFromInt(text_len)
            else
                @as(f64, @floatFromInt(text_len)) + duration - frame_rescale;
            _ = try p.appendGrid(ch, cw, cond.latent_t, start_t);
        }
        _ = try p.appendAudio(geo.audio_t, @floatFromInt(text_len), w_axis[0], w_axis[w_axis.len - 1]);
        _ = try p.appendGrid(h_axis, w_axis, geo.latent_t, @floatFromInt(text_len));
    } else {
        var rotary: f64 = @floatFromInt(text_len);
        for (refs) |block| {
            var block_end = rotary;
            var w_low = w_axis[0];
            var w_high = w_axis[w_axis.len - 1];
            if (block.video_index >= 0) {
                const cond = conds[@intCast(block.video_index)];
                var cw_buf: [256]f32 = undefined;
                const area = @sqrt(@as(f64, @floatFromInt(cond.latent_h * cond.latent_w)));
                const cw = spatialAxis(cond.latent_w, area, &cw_buf);
                w_low = cw[0];
                w_high = cw[cw.len - 1];
            }
            if (block.kind == .audio or block.kind == .video_audio) {
                const a = audios[@intCast(block.audio_index)];
                _ = try p.appendAudio(a.latent_t, rotary, w_low, w_high);
                block_end = @max(block_end, rotary + @as(f64, @floatFromInt(a.latent_t)));
            }
            if (block.kind != .audio) {
                const cond = conds[@intCast(block.video_index)];
                var ch_buf: [256]f32 = undefined;
                var cw_buf: [256]f32 = undefined;
                const area = @sqrt(@as(f64, @floatFromInt(cond.latent_h * cond.latent_w)));
                const ch = spatialAxis(cond.latent_h, area, &ch_buf);
                const cw = spatialAxis(cond.latent_w, area, &cw_buf);
                const placed = try p.appendGrid(ch, cw, cond.latent_t, rotary);
                block_end = if (block.kind == .image)
                    @max(block_end, rotary + 1.0)
                else
                    @max(block_end, placed);
            }
            rotary = block_end;
        }
        _ = try p.appendAudio(geo.audio_t, rotary, w_axis[0], w_axis[w_axis.len - 1]);
        _ = try p.appendGrid(h_axis, w_axis, geo.latent_t, rotary);
    }

    std.debug.assert(p.video_indices.items.len == video_len);
    std.debug.assert(p.audio_indices.items.len == audio_len);
    return .{
        .layout = try p.finish(text_len, audio_len, video_len, cond_n, cond_a),
        .video = video,
        .audio = audio,
    };
}

pub const CondClip = struct {
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    /// 0 = first frame, 1 = last frame, -1 = reference still.
    keyframe_index: i32,
};

pub const CondAudio = struct {
    latent_t: u32,
};

pub const RefKind = enum { image, video, audio, video_audio };

pub const RefBlock = struct {
    kind: RefKind,
    video_index: i32 = -1,
    audio_index: i32 = -1,
};

fn condTokens(clip: CondClip) u32 {
    return clip.latent_t * (clip.latent_h / 2) * (clip.latent_w / 2);
}

fn condAudioTokens(audios: []const CondAudio) u32 {
    var n: u32 = 0;
    for (audios) |a| n += a.latent_t * 2;
    return n;
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

    const n_mod: u32 = @intCast(config.modality_count);
    const text_end = layout.text_len;
    const audio_end = layout.videoStart();

    if (layout.cond_video_len == 0 and layout.cond_audio_len == 0) {
        const slots = fillSlots(unique_out, video_t, audio_t);
        @memset(timestep_indices[0..text_end], slots.video);
        @memset(timestep_indices[text_end..audio_end], slots.audio);
        @memset(timestep_indices[audio_end..seq], slots.video);
        @memset(adaln_indices[0..text_end], slots.video * n_mod + @intFromEnum(Modality.text));
        @memset(adaln_indices[text_end..audio_end], slots.audio * n_mod + @intFromEnum(Modality.audio));
        @memset(adaln_indices[audio_end..seq], slots.video * n_mod + @intFromEnum(Modality.video));
        return;
    }

    const vis_cond = @max(video_t, visual_cond_timestep);
    const aud_cond: f32 = 1.0;
    var times_buf: [4]f32 = .{ video_t, audio_t, vis_cond, aud_cond };
    var n_times: u32 = 2;
    if (layout.cond_video_len != 0) {
        times_buf[n_times] = vis_cond;
        n_times += 1;
    }
    if (layout.cond_audio_len != 0) {
        times_buf[n_times] = aud_cond;
        n_times += 1;
    }
    const n = fillUnique(unique_out, times_buf[0..n_times]);
    const v_slot = indexOf(unique_out, n, video_t);
    const a_slot = indexOf(unique_out, n, audio_t);
    const vc_slot = if (layout.cond_video_len != 0) indexOf(unique_out, n, vis_cond) else v_slot;
    const ac_slot = if (layout.cond_audio_len != 0) indexOf(unique_out, n, aud_cond) else a_slot;

    std.debug.assert(layout.scatter());
    const cond_v = layout.cond_video_len;
    const cond_a = layout.cond_audio_len;
    for (layout.text_indices) |idx| {
        timestep_indices[idx] = v_slot;
        const tag = if (idx < layout.token_tags.len) layout.token_tags[idx] else @intFromEnum(Modality.text);
        adaln_indices[idx] = v_slot * n_mod + tag;
    }
    for (layout.video_indices[0..cond_v]) |idx| {
        timestep_indices[idx] = vc_slot;
        adaln_indices[idx] = vc_slot * n_mod + @intFromEnum(Modality.video);
    }
    for (layout.video_indices[cond_v..]) |idx| {
        timestep_indices[idx] = v_slot;
        adaln_indices[idx] = v_slot * n_mod + @intFromEnum(Modality.video);
    }
    for (layout.audio_indices[0..cond_a]) |idx| {
        timestep_indices[idx] = ac_slot;
        adaln_indices[idx] = ac_slot * n_mod + @intFromEnum(Modality.audio);
    }
    for (layout.audio_indices[cond_a..]) |idx| {
        timestep_indices[idx] = a_slot;
        adaln_indices[idx] = a_slot * n_mod + @intFromEnum(Modality.audio);
    }
}
