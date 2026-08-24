const std = @import("std");

const config = @import("config.zig");

pub const Modality = enum(u8) {
    video = 0,
    text = 1,
    audio = 2,
};

pub const SegmentKind = enum {
    text,
    condition_video,
    condition_audio,
    target_audio,
    target_video,
};

pub const Position = struct { t: f32, h: f32, w: f32 };

pub const SequenceSegment = struct {
    start: u32,
    end: u32,
    kind: SegmentKind,
    source_index: i32 = -1,
};

pub const ConditionVideo = struct {
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    /// 0 = first frame, last frame otherwise. Ignored for Ref2VA.
    keyframe_index: i32 = 0,
};

pub const ConditionAudio = struct {
    latent_t: u32,
};

pub const ReferenceKind = enum { image, video, audio, video_audio };

pub const ReferenceBlock = struct {
    kind: ReferenceKind,
    video_index: i32 = -1,
    audio_index: i32 = -1,
};

/// Fixed AdaLN / time-embed slots. Distinct rows stay valid when video and
/// condition times diverge; equal values still produce equal `temb` rows.
pub const timestep_slot_count: u32 = 4;
pub const TimeSlot = enum(u32) {
    video = 0,
    audio = 1,
    visual_cond = 2,
    audio_cond = 3,
};

pub fn timestepValues(video_t: f32, audio_t: f32) [timestep_slot_count]f32 {
    return .{
        video_t,
        audio_t,
        @max(video_t, config.visual_cond_timestep),
        @max(audio_t, 1.0),
    };
}

pub fn writeTimesteps(out: []f32, video_t: f32, audio_t: f32) void {
    const values = timestepValues(video_t, audio_t);
    const n = @min(out.len, values.len);
    @memcpy(out[0..n], values[0..n]);
    if (n > 0) {
        for (n..out.len) |i| out[i] = values[n - 1];
    }
}

pub const Layout = struct {
    positions: []Position,
    token_tags: []u8,
    timestep_indices: []u32,
    timesteps: []f32,
    segments: []SequenceSegment,
    text_indices: []u32,
    video_indices: []u32,
    audio_indices: []u32,
    target_video_start: u32,
    target_video_end: u32,
    target_audio_start: u32,
    target_audio_end: u32,

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.token_tags);
        allocator.free(self.timestep_indices);
        allocator.free(self.timesteps);
        allocator.free(self.segments);
        allocator.free(self.text_indices);
        allocator.free(self.video_indices);
        allocator.free(self.audio_indices);
    }

    pub fn seqLen(self: Layout) u32 {
        return @intCast(self.positions.len);
    }

    pub fn adalnIndex(self: Layout, row: usize) u32 {
        return self.timestep_indices[row] * @as(u32, @intCast(config.modality_count)) + self.token_tags[row];
    }
};

pub const BuildArgs = struct {
    text_len: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    audio_t: u32,
    video_t: f32,
    audio_t_noise: f32,
    condition_videos: []const ConditionVideo = &.{},
    condition_audios: []const ConditionAudio = &.{},
    references: []const ReferenceBlock = &.{},
    text_tags: []const u8 = &.{},
};

const video_spans = [_]u32{ 1, 4, 4, 4, 4 };

pub fn videoSpan(frame: u32) f32 {
    return config.frame_rescale * @as(f32, @floatFromInt(video_spans[frame % video_spans.len]));
}

pub fn videoDuration(latent_t: u32) f32 {
    var total: f32 = 0;
    for (0..latent_t) |t| total += videoSpan(@intCast(t));
    return total;
}

pub fn spatialAxis(dim: u32, sqrt_area: f32, out: []f32) []f32 {
    const count = dim / 2;
    std.debug.assert(out.len >= count);
    const ratio = @as(f32, @floatFromInt(dim)) / sqrt_area;
    const step = ratio / @as(f32, @floatFromInt(count));
    for (0..count) |i| {
        out[i] = (@as(f32, @floatFromInt(i)) * step + (1.0 - ratio) * 0.5) * 32.0;
    }
    return out[0..count];
}

const Builder = struct {
    allocator: std.mem.Allocator,
    positions: std.ArrayList(Position),
    token_tags: std.ArrayList(u8),
    timestep_indices: std.ArrayList(u32),
    timesteps: std.ArrayList(f32),
    segments: std.ArrayList(SequenceSegment),
    text_indices: std.ArrayList(u32),
    video_indices: std.ArrayList(u32),
    audio_indices: std.ArrayList(u32),

    fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .positions = .empty,
            .token_tags = .empty,
            .timestep_indices = .empty,
            .timesteps = .empty,
            .segments = .empty,
            .text_indices = .empty,
            .video_indices = .empty,
            .audio_indices = .empty,
        };
    }

    fn deinit(self: *Builder) void {
        self.positions.deinit(self.allocator);
        self.token_tags.deinit(self.allocator);
        self.timestep_indices.deinit(self.allocator);
        self.timesteps.deinit(self.allocator);
        self.segments.deinit(self.allocator);
        self.text_indices.deinit(self.allocator);
        self.video_indices.deinit(self.allocator);
        self.audio_indices.deinit(self.allocator);
    }

    fn row(self: *const Builder) u32 {
        return @intCast(self.positions.items.len);
    }

    fn appendRow(self: *Builder, pos: Position, tag: Modality, time_row: u32) !void {
        try self.positions.append(self.allocator, pos);
        try self.token_tags.append(self.allocator, @intFromEnum(tag));
        try self.timestep_indices.append(self.allocator, time_row);
    }

    fn finish(self: *Builder, target_video: struct { start: u32, end: u32 }, target_audio: struct { start: u32, end: u32 }) !Layout {
        return .{
            .positions = try self.positions.toOwnedSlice(self.allocator),
            .token_tags = try self.token_tags.toOwnedSlice(self.allocator),
            .timestep_indices = try self.timestep_indices.toOwnedSlice(self.allocator),
            .timesteps = try self.timesteps.toOwnedSlice(self.allocator),
            .segments = try self.segments.toOwnedSlice(self.allocator),
            .text_indices = try self.text_indices.toOwnedSlice(self.allocator),
            .video_indices = try self.video_indices.toOwnedSlice(self.allocator),
            .audio_indices = try self.audio_indices.toOwnedSlice(self.allocator),
            .target_video_start = target_video.start,
            .target_video_end = target_video.end,
            .target_audio_start = target_audio.start,
            .target_audio_end = target_audio.end,
        };
    }
};

fn appendText(b: *Builder, text_len: u32, text_tags: []const u8, time_row: u32) !void {
    const start = b.row();
    var run_start: u32 = 0;
    var current: u8 = if (text_tags.len == 0) @intFromEnum(Modality.text) else text_tags[0];
    for (0..text_len) |i| {
        const tag: u8 = if (i < text_tags.len) text_tags[i] else @intFromEnum(Modality.text);
        if (i > 0 and tag != current) {
            try b.segments.append(b.allocator, .{
                .start = start + run_start,
                .end = start + @as(u32, @intCast(i)),
                .kind = .text,
            });
            run_start = @intCast(i);
            current = tag;
        }
        try b.appendRow(.{ .t = @floatFromInt(i), .h = 0, .w = 0 }, @enumFromInt(tag), time_row);
        try b.text_indices.append(b.allocator, start + @as(u32, @intCast(i)));
    }
    try b.segments.append(b.allocator, .{
        .start = start + run_start,
        .end = start + text_len,
        .kind = .text,
    });
}

fn appendVideoGrid(
    b: *Builder,
    h_axis: []const f32,
    w_axis: []const f32,
    latent_t: u32,
    start_t: f32,
    time_row: u32,
    kind: SegmentKind,
    source_index: i32,
) !struct { start: u32, end: u32, cursor: f32 } {
    const start = b.row();
    var cursor = start_t;
    for (0..latent_t) |t| {
        for (h_axis) |h| {
            for (w_axis) |w| {
                const idx = b.row();
                try b.appendRow(.{ .t = cursor, .h = h, .w = w }, .video, time_row);
                try b.video_indices.append(b.allocator, idx);
            }
        }
        cursor += videoSpan(@intCast(t));
    }
    const end = b.row();
    try b.segments.append(b.allocator, .{
        .start = start,
        .end = end,
        .kind = kind,
        .source_index = source_index,
    });
    return .{ .start = start, .end = end, .cursor = cursor };
}

fn appendAudioRows(
    b: *Builder,
    length: u32,
    cursor: f32,
    w_low: f32,
    w_high: f32,
    time_row: u32,
    kind: SegmentKind,
    source_index: i32,
) !struct { start: u32, end: u32 } {
    const start = b.row();
    const widths = [_]f32{ w_low, w_high };
    for (widths) |w| {
        for (0..length) |t| {
            const idx = b.row();
            try b.appendRow(.{ .t = cursor + @as(f32, @floatFromInt(t)), .h = 0, .w = w }, .audio, time_row);
            try b.audio_indices.append(b.allocator, idx);
        }
    }
    const end = b.row();
    try b.segments.append(b.allocator, .{
        .start = start,
        .end = end,
        .kind = kind,
        .source_index = source_index,
    });
    return .{ .start = start, .end = end };
}

pub fn build(allocator: std.mem.Allocator, args: BuildArgs) !Layout {
    var b = Builder.init(allocator);
    errdefer b.deinit();

    const sqrt_area = @sqrt(@as(f32, @floatFromInt(args.latent_h * args.latent_w)));
    var h_buf: [256]f32 = undefined;
    var w_buf: [256]f32 = undefined;
    const h_axis = spatialAxis(args.latent_h, sqrt_area, &h_buf);
    const w_axis = spatialAxis(args.latent_w, sqrt_area, &w_buf);

    const times = timestepValues(args.video_t, args.audio_t_noise);
    try b.timesteps.appendSlice(b.allocator, &times);
    const video_time = @intFromEnum(TimeSlot.video);
    const audio_time = @intFromEnum(TimeSlot.audio);
    const cond_time = @intFromEnum(TimeSlot.visual_cond);
    const audio_cond_time = @intFromEnum(TimeSlot.audio_cond);

    try appendText(&b, args.text_len, args.text_tags, video_time);

    if (args.references.len == 0) {
        const duration = videoDuration(args.latent_t);
        for (args.condition_videos, 0..) |cond, index| {
            var ch_buf: [256]f32 = undefined;
            var cw_buf: [256]f32 = undefined;
            const area = @sqrt(@as(f32, @floatFromInt(cond.latent_h * cond.latent_w)));
            const ch = spatialAxis(cond.latent_h, area, &ch_buf);
            const cw = spatialAxis(cond.latent_w, area, &cw_buf);
            const is_first = cond.keyframe_index == 0;
            const keyframe_t = if (is_first)
                @as(f32, @floatFromInt(args.text_len))
            else
                @as(f32, @floatFromInt(args.text_len)) + duration - config.frame_rescale;
            _ = try appendVideoGrid(&b, ch, cw, cond.latent_t, keyframe_t, cond_time, .condition_video, @intCast(index));
        }
    } else {
        var cursor: f32 = @floatFromInt(args.text_len);
        for (args.references) |block| {
            var block_end = cursor;
            if (block.kind == .audio or block.kind == .video_audio) {
                if (block.audio_index < 0) return error.MissingReferenceAudio;
                const audio = args.condition_audios[@intCast(block.audio_index)];
                var w_low = w_axis[0];
                var w_high = w_axis[w_axis.len - 1];
                if (block.video_index >= 0) {
                    const video = args.condition_videos[@intCast(block.video_index)];
                    var cw_buf: [256]f32 = undefined;
                    const area = @sqrt(@as(f32, @floatFromInt(video.latent_h * video.latent_w)));
                    const cw = spatialAxis(video.latent_w, area, &cw_buf);
                    w_low = cw[0];
                    w_high = cw[cw.len - 1];
                }
                _ = try appendAudioRows(&b, audio.latent_t, cursor, w_low, w_high, audio_cond_time, .condition_audio, block.audio_index);
                block_end = @max(block_end, cursor + @as(f32, @floatFromInt(audio.latent_t)));
            }
            if (block.kind != .audio) {
                if (block.video_index < 0) return error.MissingReferenceVideo;
                const video = args.condition_videos[@intCast(block.video_index)];
                var ch_buf: [256]f32 = undefined;
                var cw_buf: [256]f32 = undefined;
                const area = @sqrt(@as(f32, @floatFromInt(video.latent_h * video.latent_w)));
                const ch = spatialAxis(video.latent_h, area, &ch_buf);
                const cw = spatialAxis(video.latent_w, area, &cw_buf);
                const placed = try appendVideoGrid(&b, ch, cw, video.latent_t, cursor, cond_time, .condition_video, block.video_index);
                block_end = if (block.kind == .image)
                    @max(block_end, cursor + 1.0)
                else
                    @max(block_end, placed.cursor);
            }
            cursor = block_end;
        }
    }

    var cursor: f32 = @floatFromInt(args.text_len);
    if (args.references.len != 0) {
        cursor = 0;
        for (b.positions.items) |pos| cursor = @max(cursor, pos.t);
    }

    const audio = try appendAudioRows(
        &b,
        args.audio_t,
        cursor,
        w_axis[0],
        w_axis[w_axis.len - 1],
        audio_time,
        .target_audio,
        -1,
    );
    const video = try appendVideoGrid(&b, h_axis, w_axis, args.latent_t, cursor, video_time, .target_video, -1);

    return b.finish(.{ .start = video.start, .end = video.end }, .{ .start = audio.start, .end = audio.end });
}

pub fn buildT2va(
    allocator: std.mem.Allocator,
    text_len: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    audio_t: u32,
    video_t: f32,
    audio_t_noise: f32,
) !Layout {
    return build(allocator, .{
        .text_len = text_len,
        .latent_t = latent_t,
        .latent_h = latent_h,
        .latent_w = latent_w,
        .audio_t = audio_t,
        .video_t = video_t,
        .audio_t_noise = audio_t_noise,
    });
}

/// Host patchify of `{t,h,w,c}` with patch `(pt,ph,pw)` into rows of `c*pt*ph*pw`.
pub fn patchify(
    allocator: std.mem.Allocator,
    src: []const f32,
    t: u32,
    h: u32,
    w: u32,
    c: u32,
    patch: [3]i64,
) ![]f32 {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    std.debug.assert(t % pt == 0 and h % ph == 0 and w % pw == 0);
    const rows = (t / pt) * (h / ph) * (w / pw);
    const width = c * pt * ph * pw;
    const out = try allocator.alloc(f32, rows * width);
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < t) : (tt += pt) {
        var hh: u32 = 0;
        while (hh < h) : (hh += ph) {
            var ww: u32 = 0;
            while (ww < w) : (ww += pw) {
                var dst: usize = 0;
                for (0..pt) |dt| {
                    for (0..ph) |dh| {
                        for (0..pw) |dw| {
                            const src_t = tt + @as(u32, @intCast(dt));
                            const src_h = hh + @as(u32, @intCast(dh));
                            const src_w = ww + @as(u32, @intCast(dw));
                            const base = (((@as(usize, src_t) * h + src_h) * w + src_w) * c);
                            @memcpy(out[row * width + dst ..][0..c], src[base..][0..c]);
                            dst += c;
                        }
                    }
                }
                row += 1;
            }
        }
    }
    return out;
}

pub fn unpatchify(
    allocator: std.mem.Allocator,
    src: []const f32,
    t: u32,
    h: u32,
    w: u32,
    c: u32,
    patch: [3]i64,
) ![]f32 {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    const width = c * pt * ph * pw;
    const out = try allocator.alloc(f32, @as(usize, t) * h * w * c);
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < t) : (tt += pt) {
        var hh: u32 = 0;
        while (hh < h) : (hh += ph) {
            var ww: u32 = 0;
            while (ww < w) : (ww += pw) {
                var src_i: usize = 0;
                for (0..pt) |dt| {
                    for (0..ph) |dh| {
                        for (0..pw) |dw| {
                            const dst_t = tt + @as(u32, @intCast(dt));
                            const dst_h = hh + @as(u32, @intCast(dh));
                            const dst_w = ww + @as(u32, @intCast(dw));
                            const base = (((@as(usize, dst_t) * h + dst_h) * w + dst_w) * c);
                            @memcpy(out[base..][0..c], src[row * width + src_i ..][0..c]);
                            src_i += c;
                        }
                    }
                }
                row += 1;
            }
        }
    }
    return out;
}
