//! Image / video / audio modes: ffmpeg load, Qwen vision prompt, visual-VAE + audio encode.

const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const audio = @import("audio.zig");
const config = @import("config.zig");
const encoder = @import("encoder.zig");
const ops = @import("ops.zig");
const pack = @import("pack.zig");
const vision = @import("vision.zig");
const visual_enc = @import("visual_encoder.zig");

const log = std.log.scoped(.minimax_h3);

const max_ref_files: u32 = 12;
const max_ref_images: u32 = 9;
const max_ref_videos: u32 = 3;
const max_ref_audios: u32 = 3;

pub const PictureKind = enum { first, last, ref };

pub const Picture = struct {
    path: []const u8,
    kind: PictureKind,
    media: pack.RefKind = .image,
    has_audio: bool = false,
    vis_w: u32,
    vis_h: u32,
    vis_seq: u32,
    vis_merged: u32,
    temporal: u32 = 1,
    timestamps: []f32 = &.{},
    rgb: []u8 = &.{},
    frames: u32 = 1,
    vae_w: u32 = 0,
    vae_h: u32 = 0,
    vae_frames: u32 = 1,
    latent_t: u32 = 1,
};

pub const AudioRef = struct {
    stereo: []f32,
    latent_t: u32,
};

pub const Plan = struct {
    pictures: []Picture,
    audios: []AudioRef,
    refs: []pack.RefBlock,
    tokens: []u32,
    tags: []u8 = &.{},
    spans: []encoder.VisionSpan,
    ref2va: bool,

    pub fn deinit(self: Plan, allocator: std.mem.Allocator) void {
        for (self.pictures) |p| {
            if (p.timestamps.len != 0) allocator.free(p.timestamps);
            if (p.rgb.len != 0) allocator.free(p.rgb);
        }
        allocator.free(self.pictures);
        for (self.audios) |a| {
            if (a.stereo.len != 0) allocator.free(a.stereo);
        }
        allocator.free(self.audios);
        allocator.free(self.refs);
        allocator.free(self.tokens);
        allocator.free(self.spans);
        if (self.tags.len != 0) allocator.free(self.tags);
    }

    pub fn hasVideo(self: Plan) bool {
        for (self.pictures) |p| {
            if (p.media == .video or p.media == .video_audio) return true;
        }
        return false;
    }

    pub fn clips(self: Plan, allocator: std.mem.Allocator, geo: config.Geometry, spatial: u32) ![]pack.CondClip {
        var n: usize = 0;
        for (self.pictures) |p| {
            if (p.media != .audio) n += 1;
        }
        const out = try allocator.alloc(pack.CondClip, n);
        var i: usize = 0;
        for (self.pictures) |p| {
            if (p.media == .audio) continue;
            if (p.media == .video or p.media == .video_audio) {
                out[i] = .{
                    .latent_t = p.latent_t,
                    .latent_h = p.vae_h / spatial,
                    .latent_w = p.vae_w / spatial,
                    .keyframe_index = -1,
                };
            } else {
                out[i] = .{
                    .latent_t = 1,
                    .latent_h = if (p.kind == .ref) p.vis_h / spatial else geo.latent_h,
                    .latent_w = if (p.kind == .ref) p.vis_w / spatial else geo.latent_w,
                    .keyframe_index = switch (p.kind) {
                        .first => 0,
                        .last => 1,
                        .ref => -1,
                    },
                };
            }
            i += 1;
        }
        return out;
    }

    pub fn audioClips(self: Plan, allocator: std.mem.Allocator) ![]pack.CondAudio {
        const out = try allocator.alloc(pack.CondAudio, self.audios.len);
        for (self.audios, out) |a, *c| c.* = .{ .latent_t = a.latent_t };
        return out;
    }
};

pub const Encoded = struct {
    patches: []f32,
    merged: []f32,
    deepstack: [3][]f32,
    audio_patches: []f32 = &.{},

    pub fn deinit(self: Encoded, allocator: std.mem.Allocator) void {
        allocator.free(self.patches);
        allocator.free(self.merged);
        for (self.deepstack) |d| allocator.free(d);
        if (self.audio_patches.len != 0) allocator.free(self.audio_patches);
    }
};

fn isMediaExt(path: []const u8, exts: []const []const u8) bool {
    const ext = std.fs.path.extension(path);
    for (exts) |e| {
        if (std.ascii.eqlIgnoreCase(ext, e)) return true;
    }
    return false;
}

fn guessKind(path: []const u8) pack.RefKind {
    if (isMediaExt(path, &.{ ".wav", ".mp3", ".flac", ".m4a" })) return .audio;
    if (isMediaExt(path, &.{ ".mp4", ".mov", ".webm", ".mkv", ".avi" })) return .video;
    return .image;
}

fn probeSize(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { w: u32, h: u32 } {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "ffmpeg", "-hide_banner", "-i", path },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(8192),
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    const size = findWxH(result.stderr) orelse return error.ImageLoadFailed;
    return .{ .w = size.w, .h = size.h };
}

const VideoMeta = struct { w: u32, h: u32, fps: f32, has_audio: bool };

fn probeVideo(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !VideoMeta {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "ffmpeg", "-hide_banner", "-i", path },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(8192),
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return parseFfmpegProbe(result.stderr);
}

fn parseFfmpegProbe(text: []const u8) !VideoMeta {
    const video = std.mem.indexOf(u8, text, "Video:") orelse return error.VideoLoadFailed;
    const rest = text[video..];
    const size = findWxH(rest) orelse return error.VideoLoadFailed;
    const fps = findFps(rest) orelse return error.VideoLoadFailed;
    return .{
        .w = size.w,
        .h = size.h,
        .fps = fps,
        .has_audio = std.mem.indexOf(u8, text, "Audio:") != null,
    };
}

fn findWxH(text: []const u8) ?struct { w: u32, h: u32 } {
    var i: usize = 0;
    while (i + 3 < text.len) : (i += 1) {
        if (!std.ascii.isDigit(text[i])) continue;
        const x = std.mem.indexOfScalarPos(u8, text, i + 1, 'x') orelse return null;
        if (x == i) continue;
        var end = x + 1;
        while (end < text.len and std.ascii.isDigit(text[end])) end += 1;
        if (end == x + 1) continue;
        const w = std.fmt.parseInt(u32, text[i..x], 10) catch continue;
        const h = std.fmt.parseInt(u32, text[x + 1 .. end], 10) catch continue;
        if (w > 0 and h > 0) return .{ .w = w, .h = h };
    }
    return null;
}

fn findFps(text: []const u8) ?f32 {
    for ([_][]const u8{ " fps", " tbr" }) |needle| {
        if (std.mem.indexOf(u8, text, needle)) |at| {
            var start = at;
            while (start > 0 and (std.ascii.isDigit(text[start - 1]) or text[start - 1] == '.')) start -= 1;
            if (start < at) {
                if (std.fmt.parseFloat(f32, text[start..at])) |v| {
                    if (v > 0) return v;
                } else |_| {}
            }
        }
    }
    return null;
}

const RgbImage = struct { w: u32, h: u32, rgb: []u8 };

fn parsePpmHeader(bytes: []const u8) !struct { w: u32, h: u32, data_off: usize } {
    var rest = bytes;
    const magic_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.ImageLoadFailed;
    if (!std.mem.eql(u8, std.mem.trim(u8, rest[0..magic_end], " \r"), "P6")) return error.ImageLoadFailed;
    rest = rest[magic_end + 1 ..];
    var w: usize = 0;
    var h: usize = 0;
    var maxv: usize = 0;
    while (maxv == 0) {
        const line_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.ImageLoadFailed;
        const line = std.mem.trim(u8, rest[0..line_end], " \r");
        rest = rest[line_end + 1 ..];
        if (line.len == 0 or line[0] == '#') continue;
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        if (w == 0) w = try std.fmt.parseInt(usize, it.next() orelse return error.ImageLoadFailed, 10);
        if (h == 0) h = try std.fmt.parseInt(usize, it.next() orelse return error.ImageLoadFailed, 10);
        if (it.next()) |mv| maxv = try std.fmt.parseInt(usize, mv, 10);
    }
    if (maxv != 255) return error.ImageLoadFailed;
    return .{ .w = @intCast(w), .h = @intCast(h), .data_off = bytes.len - rest.len };
}

fn readPpmRgb(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const header = try parsePpmHeader(bytes);
    const need = @as(usize, header.w) * header.h * 3;
    if (bytes.len < header.data_off + need) return error.ImageLoadFailed;
    return .{ .w = header.w, .h = header.h, .rgb = try allocator.dupe(u8, bytes[header.data_off..][0..need]) };
}

fn loadRgbRaw(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
    if (readPpmRgb(allocator, io, path)) |img| return img else |_| {}
    const tmp = try scratchPath(allocator);
    defer allocator.free(tmp);
    try std.Io.Dir.cwd().createDirPath(io, tmp);
    defer {
        if (std.Io.Dir.openDirAbsolute(io, "/tmp", .{})) |parent_dir| {
            var parent = parent_dir;
            defer parent.close(io);
            parent.deleteTree(io, std.fs.path.basename(tmp)) catch {};
        } else |_| {}
    }
    const tmp_name = try std.fs.path.join(allocator, &.{ tmp, "in.ppm" });
    defer allocator.free(tmp_name);
    const result = std.process.run(allocator, io, .{
        .argv = &.{
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", path, "-frames:v", "1",
            "-sws_flags", "area+accurate_rnd+full_chroma_int+full_chroma_inp",
            "-pix_fmt", "rgb24", tmp_name,
        },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(4096),
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.ImageLoadFailed,
        else => return error.ImageLoadFailed,
    }
    return readPpmRgb(allocator, io, tmp_name);
}

fn sincFilter(x: f64) f64 {
    if (x == 0.0) return 1.0;
    const px = std.math.pi * x;
    return @sin(px) / px;
}

fn lanczosFilter(x: f64) f64 {
    if (x < -3.0 or x >= 3.0) return 0.0;
    return sincFilter(x) * sincFilter(x / 3.0);
}

const lanczos_precision_bits: u5 = 22;

fn clip8(ss: i64) u8 {
    const v = ss >> lanczos_precision_bits;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return @intCast(v);
}

fn quantizeLanczosCoeff(w: f64) i32 {
    const scaled = w * @as(f64, @floatFromInt(@as(i32, 1) << lanczos_precision_bits));
    if (w < 0) return @intFromFloat(-0.5 + scaled);
    return @intFromFloat(0.5 + scaled);
}

const LanczosAxis = struct {
    xmin: []u32,
    n: []u32,
    k: []i32,
    ksize: u32,

    fn deinit(self: LanczosAxis, allocator: std.mem.Allocator) void {
        allocator.free(self.xmin);
        allocator.free(self.n);
        allocator.free(self.k);
    }
};

fn precomputeLanczos(allocator: std.mem.Allocator, in_size: u32, out_size: u32) !LanczosAxis {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(1.0, scale);
    const support = 3.0 * filterscale;
    const inv = 1.0 / filterscale;
    const ksize: u32 = @as(u32, @intFromFloat(@ceil(support))) * 2 + 1;
    const xmin = try allocator.alloc(u32, out_size);
    errdefer allocator.free(xmin);
    const n = try allocator.alloc(u32, out_size);
    errdefer allocator.free(n);
    const k = try allocator.alloc(i32, @as(usize, out_size) * ksize);
    errdefer allocator.free(k);
    @memset(k, 0);
    const tmp = try allocator.alloc(f64, ksize);
    defer allocator.free(tmp);
    var xx: u32 = 0;
    while (xx < out_size) : (xx += 1) {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        var x0: i32 = @intFromFloat(center - support + 0.5);
        if (x0 < 0) x0 = 0;
        var x1: i32 = @intFromFloat(center + support + 0.5);
        if (x1 > @as(i32, @intCast(in_size))) x1 = @intCast(in_size);
        const count: u32 = @intCast(x1 - x0);
        xmin[xx] = @intCast(x0);
        n[xx] = count;
        var ww: f64 = 0;
        var t: u32 = 0;
        while (t < count) : (t += 1) {
            const w = lanczosFilter((@as(f64, @floatFromInt(t + xmin[xx])) - center + 0.5) * inv);
            tmp[t] = w;
            ww += w;
        }
        const row = k[@as(usize, xx) * ksize ..];
        t = 0;
        while (t < count) : (t += 1) {
            const w = if (ww != 0) tmp[t] / ww else tmp[t];
            row[t] = quantizeLanczosCoeff(w);
        }
    }
    return .{ .xmin = xmin, .n = n, .k = k, .ksize = ksize };
}

fn resize1dLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_len: u32, horizontal: bool) ![]u8 {
    const in_size: u32 = if (horizontal) src_w else src_h;
    const axis = try precomputeLanczos(allocator, in_size, dst_len);
    defer axis.deinit(allocator);
    const out_w = if (horizontal) dst_len else src_w;
    const out_h = if (horizontal) src_h else dst_len;
    const out = try allocator.alloc(u8, @as(usize, out_w) * out_h * 3);
    const bias: i64 = 1 << (lanczos_precision_bits - 1);
    var dy: u32 = 0;
    while (dy < out_h) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < out_w) : (dx += 1) {
            const i: u32 = if (horizontal) dx else dy;
            const origin = axis.xmin[i];
            const count = axis.n[i];
            const kk = axis.k[@as(usize, i) * axis.ksize ..];
            var acc = [3]i64{ bias, bias, bias };
            var t: u32 = 0;
            while (t < count) : (t += 1) {
                const sx: u32 = if (horizontal) origin + t else dx;
                const sy: u32 = if (horizontal) dy else origin + t;
                const si = (@as(usize, sy) * src_w + sx) * 3;
                const kv: i64 = kk[t];
                inline for (0..3) |c| acc[c] += kv * src[si + c];
            }
            const di = (@as(usize, dy) * out_w + dx) * 3;
            inline for (0..3) |c| out[di + c] = clip8(acc[c]);
        }
    }
    return out;
}

fn resizeLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    std.debug.assert(src.len == @as(usize, src_w) * src_h * 3);
    if (src_w == dst_w and src_h == dst_h) return allocator.dupe(u8, src);
    const mid = try resize1dLanczos(allocator, src, src_w, src_h, dst_w, true);
    defer allocator.free(mid);
    return resize1dLanczos(allocator, mid, dst_w, src_h, dst_h, false);
}

fn loadRgb(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: u32, h: u32) ![]u8 {
    const raw = try loadRgbRaw(allocator, io, path);
    defer allocator.free(raw.rgb);
    if (raw.w == w and raw.h == h) return allocator.dupe(u8, raw.rgb);
    return resizeLanczos(allocator, raw.rgb, raw.w, raw.h, w, h);
}

fn snapMultiple(value: u32, multiple: u32) u32 {
    if (value == 0) return multiple;
    return @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(multiple))))) * multiple);
}

fn videoCanvas(src_w: u32, src_h: u32) struct { w: u32, h: u32 } {
    const multiple = config.canvas_multiple;
    const short: f32 = 768;
    const cap: u64 = config.canvas_max_pixels;
    const ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(src_h));
    var width: f32 = undefined;
    var height: f32 = undefined;
    if (ratio >= 1.0) {
        width = short * ratio;
        height = short;
    } else {
        width = short;
        height = short / ratio;
    }
    if (width * height > @as(f32, @floatFromInt(cap))) {
        const scale = @sqrt(@as(f32, @floatFromInt(cap)) / (width * height));
        width *= scale;
        height *= scale;
    }
    var w = @max(multiple, @as(u32, @intFromFloat(@round(width / @as(f32, @floatFromInt(multiple))))) * multiple);
    var h = @max(multiple, @as(u32, @intFromFloat(@round(height / @as(f32, @floatFromInt(multiple))))) * multiple);
    while (@as(u64, w) * h > cap and (w > multiple or h > multiple)) {
        if (w > multiple and h > multiple) {
            const rw = @as(f32, @floatFromInt(w - multiple)) / @as(f32, @floatFromInt(h));
            const rh = @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h - multiple));
            if (@abs(rw - ratio) <= @abs(rh - ratio)) w -= multiple else h -= multiple;
        } else if (w > multiple) w -= multiple else h -= multiple;
    }
    if (@as(u64, src_w) * src_h < @as(u64, w) * h) {
        return .{ .w = snapMultiple(src_w, multiple), .h = snapMultiple(src_h, multiple) };
    }
    return .{ .w = w, .h = h };
}

fn resampleFrameIndices(src_frames: u32, src_fps: f32, dst_fps: f32, allocator: std.mem.Allocator) ![]u32 {
    if (src_frames == 0) return error.EmptyVideo;
    if (src_fps <= 0 or dst_fps <= 0) return error.InvalidFps;
    if (src_fps == dst_fps) {
        const out = try allocator.alloc(u32, src_frames);
        for (out, 0..) |*d, i| d.* = @intCast(i);
        return out;
    }
    const scale = dst_fps / src_fps;
    const out_len: u32 = @intFromFloat(@floor(@as(f32, @floatFromInt(src_frames)) * scale + 0.5));
    const out = try allocator.alloc(u32, out_len);
    var src: u32 = 0;
    var written: u32 = 0;
    while (src < src_frames) : (src += 1) {
        const slot: u32 = @intFromFloat(@floor(@as(f32, @floatFromInt(src)) * scale + 0.5));
        const next: u32 = if (src + 1 == src_frames)
            out_len
        else
            @intFromFloat(@floor(@as(f32, @floatFromInt(src + 1)) * scale + 0.5));
        const hold = if (next > slot) next - slot else 0;
        var h: u32 = 0;
        while (h < hold and written < out_len) : (h += 1) {
            out[written] = src;
            written += 1;
        }
    }
    if (written < out_len) {
        const last = src_frames - 1;
        while (written < out_len) : (written += 1) out[written] = last;
    }
    return out;
}

fn sampleVideoConditionFrames(frames: u32, fps: f32, sample_fps: f32, temporal_patch: u32) !struct { indices_len: u32, block_count: u32 } {
    if (frames == 0 or fps <= 0 or sample_fps <= 0) return error.EmptyVideo;
    const stride = fps / sample_fps;
    var count: u32 = 0;
    var last: i64 = -1;
    var cursor: f32 = 0;
    while (@round(cursor) < @as(f32, @floatFromInt(frames))) {
        const idx: i64 = @intFromFloat(@round(cursor));
        if (last < 0 or idx > last) {
            count += 1;
            last = idx;
        }
        cursor += stride;
    }
    if (count < temporal_patch) return error.VideoTooShort;
    const padded = count + (temporal_patch - (count % temporal_patch)) % temporal_patch;
    return .{ .indices_len = count, .block_count = padded / temporal_patch };
}

fn fillVideoConditionIndices(frames: u32, fps: f32, sample_fps: f32, out: []u32) u32 {
    const stride = fps / sample_fps;
    var n: u32 = 0;
    var last: i64 = -1;
    var cursor: f32 = 0;
    while (@round(cursor) < @as(f32, @floatFromInt(frames)) and n < out.len) {
        const idx: u32 = @intFromFloat(@round(cursor));
        if (last < 0 or @as(i64, idx) > last) {
            out[n] = @min(frames - 1, idx);
            n += 1;
            last = idx;
        }
        cursor += stride;
    }
    return n;
}

fn fillVideoTimestamps(sample_count: u32, out: []f32) void {
    const n = @min(sample_count, @as(u32, @intCast(out.len)));
    var i: u32 = 0;
    while (i < n) : (i += 1) out[i] = @as(f32, @floatFromInt(i)) / 2.0;
}

fn formatSeconds1(value: f32, buf: []u8) []const u8 {
    const scaled = @as(f64, value) * 10.0;
    const whole = @floor(scaled);
    const frac = scaled - whole;
    var tenths: i64 = @intFromFloat(whole);
    if (frac > 0.5) {
        tenths += 1;
    } else if (frac == 0.5 and @mod(tenths, 2) != 0) {
        tenths += 1;
    }
    const ip = @divTrunc(tenths, 10);
    const frac_digit = @mod(tenths, 10);
    return std.fmt.bufPrint(buf, "{d}.{d}", .{ ip, if (frac_digit < 0) -frac_digit else frac_digit }) catch buf[0..0];
}

fn hopAlign(n: u32, hop: u32) u32 {
    if (hop == 0) return n;
    return n + (hop - (n % hop)) % hop;
}

fn applyRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, indices: []const u32) ![]u8 {
    const plane = @as(usize, src_w) * src_h * 3;
    const out = try allocator.alloc(u8, indices.len * plane);
    for (indices, 0..) |src_i, i| {
        const si = @min(src_i, if (src.len == 0) 0 else @as(u32, @intCast(src.len / plane - 1)));
        @memcpy(out[i * plane ..][0..plane], src[si * plane ..][0..plane]);
    }
    return out;
}

fn truncateStereo(allocator: std.mem.Allocator, stereo: []const f32, max_samples: u32) ![]f32 {
    const have: u32 = @intCast(stereo.len / 2);
    const keep = @min(have, max_samples);
    const out = try allocator.alloc(f32, @as(usize, keep) * 2);
    @memcpy(out, stereo[0..out.len]);
    return out;
}

fn resampleLinear(allocator: std.mem.Allocator, stereo: []const f32, src_rate: u32, dst_rate: u32) ![]f32 {
    const src_n: u32 = @intCast(stereo.len / 2);
    if (src_rate == 0 or dst_rate == 0) return error.InvalidRate;
    if (src_rate == dst_rate) return allocator.dupe(f32, stereo);
    const dst_n: u32 = @intFromFloat(@round(@as(f64, src_n) * @as(f64, dst_rate) / @as(f64, src_rate)));
    const out = try allocator.alloc(f32, @as(usize, dst_n) * 2);
    if (src_n == 0 or dst_n == 0) {
        @memset(out, 0);
        return out;
    }
    if (dst_n == 1) {
        @memcpy(out[0..2], stereo[0..2]);
        return out;
    }
    var i: u32 = 0;
    while (i < dst_n) : (i += 1) {
        const src_pos = @as(f64, i) * @as(f64, src_n - 1) / @as(f64, dst_n -| 1);
        const lo: u32 = @intFromFloat(@floor(src_pos));
        const hi = @min(src_n - 1, lo + 1);
        const a: f32 = @floatCast(src_pos - @floor(src_pos));
        inline for (0..2) |c| {
            const a0 = stereo[@as(usize, lo) * 2 + c];
            const a1 = stereo[@as(usize, hi) * 2 + c];
            out[@as(usize, i) * 2 + c] = a0 * (1 - a) + a1 * a;
        }
    }
    return out;
}

var tmp_seq: u32 = 0;

fn scratchPath(allocator: std.mem.Allocator) ![]u8 {
    tmp_seq += 1;
    return std.fmt.allocPrint(allocator, "/tmp/minimax_h3_{x}_{d}", .{ @as(u64, @intFromPtr(&tmp_seq)), tmp_seq });
}

fn loadVideo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, dst_w: u32, dst_h: u32, geo_frames: u32) !struct { rgb: []u8, frames: u32 } {
    const meta = try probeVideo(allocator, io, path);
    const scratch = try scratchPath(allocator);
    defer allocator.free(scratch);
    try std.Io.Dir.cwd().createDirPath(io, scratch);
    defer {
        if (std.Io.Dir.openDirAbsolute(io, "/tmp", .{})) |parent_dir| {
            var parent = parent_dir;
            defer parent.close(io);
            parent.deleteTree(io, std.fs.path.basename(scratch)) catch {};
        } else |_| {}
    }
    const tmp_pat = try std.fs.path.join(allocator, &.{ scratch, "f_%04d.ppm" });
    defer allocator.free(tmp_pat);
    const result = std.process.run(allocator, io, .{
        .argv = &.{
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", path, "-vsync", "0", "-pix_fmt", "rgb24", tmp_pat,
        },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(4096),
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.VideoLoadFailed,
        else => return error.VideoLoadFailed,
    }

    var frames_list: std.ArrayList([]u8) = .empty;
    defer {
        for (frames_list.items) |f| allocator.free(f);
        frames_list.deinit(allocator);
    }
    var loaded: u32 = 0;
    var fw: u32 = meta.w;
    var fh: u32 = meta.h;
    while (loaded < 4096) : (loaded += 1) {
        var name_buf: [16]u8 = undefined;
        const frame_name = try std.fmt.bufPrint(&name_buf, "f_{d:0>4}.ppm", .{loaded + 1});
        const name = try std.fs.path.join(allocator, &.{ scratch, frame_name });
        defer allocator.free(name);
        const img = readPpmRgb(allocator, io, name) catch break;
        fw = img.w;
        fh = img.h;
        try frames_list.append(allocator, img.rgb);
    }
    if (frames_list.items.len == 0) return error.VideoLoadFailed;
    const fps = if (meta.fps > 0) meta.fps else config.video_fps;
    const indices = try resampleFrameIndices(@intCast(frames_list.items.len), fps, config.video_fps, allocator);
    defer allocator.free(indices);
    const keep = @min(geo_frames, @as(u32, @intCast(indices.len)));
    const plane = @as(usize, dst_w) * dst_h * 3;
    const out = try allocator.alloc(u8, keep * plane);
    errdefer allocator.free(out);
    var fi: u32 = 0;
    while (fi < keep) : (fi += 1) {
        const src_i = @min(indices[fi], @as(u32, @intCast(frames_list.items.len - 1)));
        const resized = try resizeLanczos(allocator, frames_list.items[src_i], fw, fh, dst_w, dst_h);
        defer allocator.free(resized);
        @memcpy(out[fi * plane ..][0..plane], resized);
    }
    return .{ .rgb = out, .frames = keep };
}

const WavInfo = struct { samples: u32, ch: u16, rate: u32, bits: u16, data_off: usize };

fn parseWavHeader(bytes: []const u8) !WavInfo {
    if (bytes.len < 44) return error.BadWav;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.BadWav;
    var off: usize = 12;
    var data_off: usize = 0;
    var data_len: usize = 0;
    var ch: u16 = 0;
    var rate: u32 = 0;
    var bits: u16 = 0;
    while (off + 8 <= bytes.len) {
        const id = bytes[off..][0..4];
        const n = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
        off += 8;
        if (std.mem.eql(u8, id, "fmt ")) {
            if (n < 16 or off + 16 > bytes.len) return error.BadWav;
            ch = std.mem.readInt(u16, bytes[off + 2 ..][0..2], .little);
            rate = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
            bits = std.mem.readInt(u16, bytes[off + 14 ..][0..2], .little);
        } else if (std.mem.eql(u8, id, "data")) {
            data_off = off;
            data_len = n;
            break;
        }
        off += n;
    }
    if (data_off == 0 or ch == 0 or bits == 0) return error.BadWav;
    return .{
        .samples = @intCast(data_len / (ch * (bits / 8))),
        .ch = ch,
        .rate = rate,
        .bits = bits,
        .data_off = data_off,
    };
}

fn decodeWavStereo(allocator: std.mem.Allocator, bytes: []const u8, info: WavInfo) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, info.samples) * 2);
    var i: usize = 0;
    while (i < info.samples) : (i += 1) {
        var c: u16 = 0;
        while (c < 2) : (c += 1) {
            const src_c = if (c < info.ch) c else 0;
            const idx = info.data_off + (i * info.ch + src_c) * (info.bits / 8);
            const s: f32 = switch (info.bits) {
                16 => @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[idx..][0..2], .little))) / 32768.0,
                32 => std.mem.bytesAsValue(f32, bytes[idx..][0..4]).*,
                else => return error.UnsupportedWav,
            };
            out[i * 2 + c] = s;
        }
    }
    return out;
}

fn loadAudio(allocator: std.mem.Allocator, io: std.Io, path: []const u8, duration_s: f32, dst_rate: u32) ![]f32 {
    const scratch = try scratchPath(allocator);
    defer allocator.free(scratch);
    try std.Io.Dir.cwd().createDirPath(io, scratch);
    defer {
        if (std.Io.Dir.openDirAbsolute(io, "/tmp", .{})) |parent_dir| {
            var parent = parent_dir;
            defer parent.close(io);
            parent.deleteTree(io, std.fs.path.basename(scratch)) catch {};
        } else |_| {}
    }
    const tmp = try std.fs.path.join(allocator, &.{ scratch, "native.wav" });
    defer allocator.free(tmp);
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "2", tmp },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(4096),
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.AudioLoadFailed,
        else => return error.AudioLoadFailed,
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, tmp, allocator, .unlimited);
    defer allocator.free(bytes);
    const info = try parseWavHeader(bytes);
    const stereo = try decodeWavStereo(allocator, bytes, info);
    defer allocator.free(stereo);
    const max_pcm: u32 = @intFromFloat(@as(f64, duration_s) * @as(f64, @floatFromInt(info.rate)));
    const truncated = try truncateStereo(allocator, stereo, max_pcm);
    defer allocator.free(truncated);
    return resampleLinear(allocator, truncated, info.rate, dst_rate);
}

fn addPicture(
    allocator: std.mem.Allocator,
    io: std.Io,
    pictures: *std.ArrayList(Picture),
    path: []const u8,
    kind: PictureKind,
    vcfg: vision.Config,
) !void {
    if (isMediaExt(path, &.{ ".mp4", ".mov", ".webm", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".m4a" })) {
        stdx.flags.fatal("video/audio is not valid for --first-frame/--last-frame: {s}", .{path});
    }
    const size = probeSize(allocator, io, path) catch |err| switch (err) {
        error.FfmpegMissing => stdx.flags.fatal("ffmpeg not found; needed to read {s}", .{path}),
        else => stdx.flags.fatal("failed to read image {s}", .{path}),
    };
    const sized: config.Size = if (kind == .ref) config.refImageSize(size.w, size.h) catch
        stdx.flags.fatal("invalid ref image aspect {s} ({d}x{d})", .{ path, size.w, size.h })
    else
        .{ .w = size.w, .h = size.h };
    const vis = vision.spatialTokens(vcfg, sized.h, sized.w, false);
    try pictures.append(allocator, .{
        .path = path,
        .kind = kind,
        .vis_w = vis.grid.w * @as(u32, @intCast(vcfg.patch_size)),
        .vis_h = vis.grid.h * @as(u32, @intCast(vcfg.patch_size)),
        .vis_seq = vis.seq,
        .vis_merged = vis.merged,
    });
    if (kind == .ref) {
        log.info("mode: ref image {s} {d}x{d} -> {d}x{d} qwen={d}x{d}", .{
            path, size.w, size.h, sized.w, sized.h, vis.grid.h, vis.grid.w,
        });
    }
}

fn assemble(
    allocator: std.mem.Allocator,
    encode_text: anytype,
    pictures: []const Picture,
    prompt: []const u8,
    ref2va: bool,
    patch: u32,
) !struct {
    tokens: []u32,
    tags: []u8,
    spans: []encoder.VisionSpan,
} {
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(allocator);
    var tags: std.ArrayList(u8) = .empty;
    errdefer tags.deinit(allocator);
    var spans: std.ArrayList(encoder.VisionSpan) = .empty;
    errdefer spans.deinit(allocator);

    const text_tag = @intFromEnum(config.Modality.text);
    const video_tag = @intFromEnum(config.Modality.video);

    if (ref2va) {
        var n_pic: u32 = 0;
        var n_vid: u32 = 0;
        var n_aud: u32 = 0;
        for (pictures) |pic| {
            if (pic.has_audio or pic.media == .audio or pic.media == .video_audio) {
                n_aud += 1;
                var label_buf: [32]u8 = undefined;
                const label = try std.fmt.bufPrint(&label_buf, "<Audio {d}>: ", .{n_aud});
                const ids = try encode_text.encodeAlloc(allocator, label);
                defer allocator.free(ids);
                try tokens.appendSlice(allocator, ids);
                try tags.appendNTimes(allocator, text_tag, ids.len);
            }
            if (pic.media == .image) {
                n_pic += 1;
                var label_buf: [32]u8 = undefined;
                const label = try std.fmt.bufPrint(&label_buf, "<Picture {d}>: ", .{n_pic});
                const ids = try encode_text.encodeAlloc(allocator, label);
                defer allocator.free(ids);
                try tokens.appendSlice(allocator, ids);
                try tags.appendNTimes(allocator, text_tag, ids.len);
                try tokens.append(allocator, vision.VISION_START);
                try tags.append(allocator, video_tag);
                const start: u32 = @intCast(tokens.items.len);
                try tokens.appendNTimes(allocator, vision.IMAGE_PAD, pic.vis_merged);
                try tags.appendNTimes(allocator, video_tag, pic.vis_merged);
                try tokens.append(allocator, vision.VISION_END);
                try tags.append(allocator, video_tag);
                try spans.append(allocator, .{
                    .start = start,
                    .tokens = pic.vis_merged,
                    .grid_h = pic.vis_h / patch,
                    .grid_w = pic.vis_w / patch,
                    .temporal = 1,
                });
            } else if (pic.media == .video or pic.media == .video_audio) {
                n_vid += 1;
                var label_buf: [32]u8 = undefined;
                const label = try std.fmt.bufPrint(&label_buf, "<Video {d}>: ", .{n_vid});
                const ids = try encode_text.encodeAlloc(allocator, label);
                defer allocator.free(ids);
                try tokens.appendSlice(allocator, ids);
                try tags.appendNTimes(allocator, text_tag, ids.len);
                for (pic.timestamps) |ts| {
                    var tbuf: [32]u8 = undefined;
                    const rendered = formatSeconds1(ts, &tbuf);
                    var sbuf: [48]u8 = undefined;
                    const stamp = try std.fmt.bufPrint(&sbuf, "<{s} seconds>", .{rendered});
                    const stamp_ids = try encode_text.encodeAlloc(allocator, stamp);
                    defer allocator.free(stamp_ids);
                    try tokens.appendSlice(allocator, stamp_ids);
                    try tags.appendNTimes(allocator, text_tag, stamp_ids.len);
                    try tokens.append(allocator, vision.VISION_START);
                    try tags.append(allocator, video_tag);
                    const start: u32 = @intCast(tokens.items.len);
                    try tokens.appendNTimes(allocator, vision.VIDEO_PAD, pic.vis_merged);
                    try tags.appendNTimes(allocator, video_tag, pic.vis_merged);
                    try tokens.append(allocator, vision.VISION_END);
                    try tags.append(allocator, video_tag);
                    try spans.append(allocator, .{
                        .start = start,
                        .tokens = pic.vis_merged,
                    .grid_h = pic.vis_h / patch,
                    .grid_w = pic.vis_w / patch,
                        .temporal = 1,
                    });
                }
            }
        }
    } else {
        var n: u32 = 0;
        for (pictures) |pic| {
            n += 1;
            var label_buf: [32]u8 = undefined;
            const label = try std.fmt.bufPrint(&label_buf, "<Picture {d}>: ", .{n});
            const ids = try encode_text.encodeAlloc(allocator, label);
            defer allocator.free(ids);
            try tokens.appendSlice(allocator, ids);
            try tags.appendNTimes(allocator, text_tag, ids.len);
            try tokens.append(allocator, vision.VISION_START);
            try tags.append(allocator, video_tag);
            const start: u32 = @intCast(tokens.items.len);
            try tokens.appendNTimes(allocator, vision.IMAGE_PAD, pic.vis_merged);
            try tags.appendNTimes(allocator, video_tag, pic.vis_merged);
            try tokens.append(allocator, vision.VISION_END);
            try tags.append(allocator, video_tag);
            try spans.append(allocator, .{
                .start = start,
                .tokens = pic.vis_merged,
                .grid_h = pic.vis_h / patch,
                .grid_w = pic.vis_w / patch,
                .temporal = 1,
            });
        }
    }
    const ids = try encode_text.encodeAlloc(allocator, prompt);
    defer allocator.free(ids);
    try tokens.appendSlice(allocator, ids);
    try tags.appendNTimes(allocator, text_tag, ids.len);
    std.debug.assert(tags.items.len == tokens.items.len);
    return .{
        .tokens = try tokens.toOwnedSlice(allocator),
        .tags = try tags.toOwnedSlice(allocator),
        .spans = try spans.toOwnedSlice(allocator),
    };
}

pub fn plan(
    allocator: std.mem.Allocator,
    io: std.Io,
    encode_text: anytype,
    prompt: []const u8,
    first: []const u8,
    last: []const u8,
    refs: []const u8,
    vis_cfg: vision.Config,
    geo: config.Geometry,
    vae: config.VisualConfig,
    audio_cfg: config.AudioConfig,
) !?Plan {
    if (first.len == 0 and last.len == 0 and refs.len == 0) return null;
    if (refs.len != 0 and (first.len != 0 or last.len != 0)) {
        stdx.flags.fatal("use --first-frame/--last-frame or --refs, not both", .{});
    }
    const vcfg = vis_cfg;
    var pictures: std.ArrayList(Picture) = .empty;
    errdefer {
        for (pictures.items) |p| {
            if (p.timestamps.len != 0) allocator.free(p.timestamps);
            if (p.rgb.len != 0) allocator.free(p.rgb);
        }
        pictures.deinit(allocator);
    }
    var audios: std.ArrayList(AudioRef) = .empty;
    errdefer {
        for (audios.items) |a| {
            if (a.stereo.len != 0) allocator.free(a.stereo);
        }
        audios.deinit(allocator);
    }
    var blocks: std.ArrayList(pack.RefBlock) = .empty;
    errdefer blocks.deinit(allocator);

    if (first.len != 0) try addPicture(allocator, io, &pictures, first, .first, vcfg);
    if (last.len != 0) try addPicture(allocator, io, &pictures, last, .last, vcfg);
    if (refs.len != 0) {
        var n_img: u32 = 0;
        var n_vid: u32 = 0;
        var n_aud: u32 = 0;
        var n_files: u32 = 0;
        var vis_i: i32 = 0;
        var aud_i: i32 = 0;
        var it = std.mem.splitScalar(u8, refs, ',');
        while (it.next()) |raw| {
            const path = std.mem.trim(u8, raw, " \t");
            if (path.len == 0) continue;
            n_files += 1;
            if (n_files > max_ref_files) stdx.flags.fatal("--refs: at most {d} files", .{max_ref_files});
            const kind = guessKind(path);
            switch (kind) {
                .image => {
                    n_img += 1;
                    if (n_img > max_ref_images) stdx.flags.fatal("--refs: at most {d} images", .{max_ref_images});
                    try addPicture(allocator, io, &pictures, path, .ref, vcfg);
                    try blocks.append(allocator, .{ .kind = .image, .video_index = vis_i });
                    vis_i += 1;
                },
                .video, .video_audio => {
                    n_vid += 1;
                    if (n_vid > max_ref_videos) stdx.flags.fatal("--refs: at most {d} videos", .{max_ref_videos});
                    const meta = probeVideo(allocator, io, path) catch |err| switch (err) {
                        error.FfmpegMissing => stdx.flags.fatal("ffmpeg not found; needed to read {s}", .{path}),
                        else => stdx.flags.fatal("failed to read video {s}", .{path}),
                    };
                    if (meta.has_audio) {
                        n_aud += 1;
                        if (n_aud > max_ref_audios) stdx.flags.fatal("--refs: at most {d} audios", .{max_ref_audios});
                    }
                    const canvas = videoCanvas(meta.w, meta.h);
                    const loaded = loadVideo(allocator, io, path, canvas.w, canvas.h, geo.frames) catch |err| switch (err) {
                        error.FfmpegMissing => stdx.flags.fatal("ffmpeg not found; needed to read {s}", .{path}),
                        else => stdx.flags.fatal("failed to load video {s}", .{path}),
                    };
                    const vae_frames = config.referenceVideoFrameCount(vae, loaded.frames);
                    const spec = vision.spatialTokens(vcfg, canvas.h, canvas.w, true);
                    const sampled = sampleVideoConditionFrames(loaded.frames, config.video_fps, config.qwen_video_fps, 2) catch
                        stdx.flags.fatal("video too short for Qwen sampling: {s}", .{path});
                    const timestamps = try allocator.alloc(f32, sampled.block_count);
                    fillVideoTimestamps(sampled.block_count, timestamps);
                    try pictures.append(allocator, .{
                        .path = path,
                        .kind = .ref,
                        .media = if (meta.has_audio) .video_audio else .video,
                        .has_audio = meta.has_audio,
                        .vis_w = spec.grid.w * @as(u32, @intCast(vcfg.patch_size)),
                        .vis_h = spec.grid.h * @as(u32, @intCast(vcfg.patch_size)),
                        .vis_seq = spec.seq * sampled.block_count,
                        .vis_merged = spec.merged,
                        .temporal = sampled.block_count,
                        .timestamps = timestamps,
                        .rgb = loaded.rgb,
                        .frames = loaded.frames,
                        .vae_w = canvas.w,
                        .vae_h = canvas.h,
                        .vae_frames = vae_frames,
                        .latent_t = config.encodeVideoLatentT(vae, vae_frames),
                    });
                    var audio_index: i32 = -1;
                    if (meta.has_audio) {
                        const duration_s = @as(f32, @floatFromInt(geo.frames)) / config.video_fps;
                        const stereo = loadAudio(allocator, io, path, duration_s, audio_cfg.sampling_rate) catch |err| switch (err) {
                            error.FfmpegMissing => stdx.flags.fatal("ffmpeg not found; needed to read {s}", .{path}),
                            else => stdx.flags.fatal("failed to load audio from {s}", .{path}),
                        };
                        const samples = hopAlign(@intCast(stereo.len / 2), audio_cfg.hop());
                        try audios.append(allocator, .{
                            .stereo = stereo,
                            .latent_t = samples / audio_cfg.hop(),
                        });
                        audio_index = aud_i;
                        aud_i += 1;
                    }
                    try blocks.append(allocator, .{
                        .kind = if (meta.has_audio) .video_audio else .video,
                        .video_index = vis_i,
                        .audio_index = audio_index,
                    });
                    vis_i += 1;
                    log.info(
                        "mode: video {s} {d}x{d} -> canvas {d}x{d} frames={d} vae_t={d} qwen={d}x{d}x{d}",
                        .{ path, meta.w, meta.h, canvas.w, canvas.h, loaded.frames, config.encodeVideoLatentT(vae, vae_frames), spec.grid.h, spec.grid.w, sampled.block_count },
                    );
                },
                .audio => {
                    n_aud += 1;
                    if (n_aud > max_ref_audios) stdx.flags.fatal("--refs: at most {d} audios", .{max_ref_audios});
                    const duration_s = @as(f32, @floatFromInt(geo.frames)) / config.video_fps;
                    const stereo = loadAudio(allocator, io, path, duration_s, audio_cfg.sampling_rate) catch |err| switch (err) {
                        error.FfmpegMissing => stdx.flags.fatal("ffmpeg not found; needed to read {s}", .{path}),
                        else => stdx.flags.fatal("failed to load audio {s}", .{path}),
                    };
                    const samples = hopAlign(@intCast(stereo.len / 2), audio_cfg.hop());
                    try audios.append(allocator, .{
                        .stereo = stereo,
                        .latent_t = samples / audio_cfg.hop(),
                    });
                    try pictures.append(allocator, .{
                        .path = path,
                        .kind = .ref,
                        .media = .audio,
                        .has_audio = true,
                        .vis_w = 0,
                        .vis_h = 0,
                        .vis_seq = 0,
                        .vis_merged = 0,
                    });
                    try blocks.append(allocator, .{ .kind = .audio, .audio_index = aud_i });
                    aud_i += 1;
                    log.info("mode: audio {s} latent_t={d}", .{ path, samples / audio_cfg.hop() });
                },
            }
        }
        if (pictures.items.len == 0) stdx.flags.fatal("--refs needs at least one image or video", .{});
        var has_visual = false;
        for (pictures.items) |p| {
            if (p.media != .audio) has_visual = true;
        }
        if (audios.items.len != 0 and !has_visual) stdx.flags.fatal("--refs audio needs an image or video", .{});
    }
    const assembled = try assemble(allocator, encode_text, pictures.items, std.mem.trimEnd(u8, prompt, "\r\n"), refs.len != 0, @intCast(vcfg.patch_size));
    return .{
        .pictures = try pictures.toOwnedSlice(allocator),
        .audios = try audios.toOwnedSlice(allocator),
        .refs = try blocks.toOwnedSlice(allocator),
        .tokens = assembled.tokens,
        .tags = assembled.tags,
        .spans = assembled.spans,
        .ref2va = refs.len != 0,
    };
}

pub fn encode(
    run: *const ops.Run,
    geo: config.Geometry,
    vae_cfg: config.VisualConfig,
    planned: Plan,
    vis_model: *const vision.LoadedModel,
    vis_cache: *const vision.WeightCache,
    vae_compiled: *const visual_enc.Compiled,
    vae_bufs: *const zml.Bufferized(visual_enc.Model),
    audio_enc: ?audio.EncoderModel,
    audio_bufs: ?*const zml.Bufferized(audio.EncoderModel),
) !Encoded {
    const allocator = run.allocator;
    const io = run.io;
    const hidden_dim: u32 = @intCast(vis_model.cfg.out_hidden_size);
    var merged_all: std.ArrayList(f32) = .empty;
    errdefer merged_all.deinit(allocator);
    var ds_host: [3][]f32 = .{ &.{}, &.{}, &.{} };
    errdefer for (ds_host) |d| if (d.len != 0) allocator.free(d);
    for (&ds_host) |*d| {
        d.* = try allocator.alloc(f32, planned.tokens.len * hidden_dim);
        @memset(d.*, 0);
    }

    var patches: std.ArrayList(f32) = .empty;
    errdefer patches.deinit(allocator);

    var compiled_v: ?vision.Compiled = null;
    defer if (compiled_v) |*c| c.deinit();

    var span_i: usize = 0;
    for (planned.pictures) |pic| {
        if (pic.media == .audio) continue;
        const is_video = pic.media == .video or pic.media == .video_audio;
        if (compiled_v == null or compiled_v.?.seq != pic.vis_seq) {
            if (compiled_v) |*c| {
                c.deinit();
                compiled_v = null;
            }
            compiled_v = try vision.compile(
                allocator,
                io,
                run.platform,
                vis_model.inner,
                pic.vis_seq,
                run.mesh[0..],
                run.progress,
            );
        }

        if (is_video) {
            const vis_frames = pic.temporal * 2;
            const sampled = try sampleVideoConditionFrames(pic.frames, config.video_fps, config.qwen_video_fps, 2);
            const idx_buf = try allocator.alloc(u32, sampled.indices_len);
            defer allocator.free(idx_buf);
            const nidx = fillVideoConditionIndices(pic.frames, config.video_fps, config.qwen_video_fps, idx_buf);
            var qwen_idx = try allocator.alloc(u32, vis_frames);
            defer allocator.free(qwen_idx);
            var qi: u32 = 0;
            while (qi < vis_frames) : (qi += 1) qwen_idx[qi] = idx_buf[@min(nidx - 1, qi)];
            const qwen_rgb = try applyRgb(allocator, pic.rgb, pic.vae_w, pic.vae_h, qwen_idx);
            defer allocator.free(qwen_rgb);
            if (pic.vis_h != pic.vae_h or pic.vis_w != pic.vae_w) return error.VisionNeedsResize;
            var encoded_v = try vision.runVideo(allocator, io, run.platform, &compiled_v.?, vis_model, vis_cache, qwen_rgb, vis_frames, pic.vis_h, pic.vis_w);
            defer encoded_v.deinit(allocator);
            try merged_all.appendSlice(allocator, encoded_v.merged);
            const block_tokens = pic.vis_merged;
            var bi: usize = 0;
            while (bi < pic.temporal and span_i < planned.spans.len) : (bi += 1) {
                const span = planned.spans[span_i];
                span_i += 1;
                for (0..3) |di| {
                    if (encoded_v.deepstack[di].len != 0) {
                        const src_off = bi * block_tokens * hidden_dim;
                        @memcpy(
                            ds_host[di][@as(usize, span.start) * hidden_dim ..][0 .. span.tokens * hidden_dim],
                            encoded_v.deepstack[di][src_off..][0 .. span.tokens * hidden_dim],
                        );
                    }
                }
            }
            const nchw = try visual_enc.rgbVideoToNchw(allocator, pic.rgb, pic.vae_frames, pic.vae_h, pic.vae_w);
            defer allocator.free(nchw);
            const still = try visual_enc.encodeVideo(run, vae_compiled, vae_bufs, vae_cfg, nchw, pic.vae_frames, pic.vae_h, pic.vae_w);
            defer allocator.free(still);
            try patches.appendSlice(allocator, still);
            log.info("mode: video {d}x{d}x{d} -> {d} patches", .{ pic.vae_frames, pic.vae_w, pic.vae_h, still.len / 96 });
        } else {
            const vis_rgb = try loadRgb(allocator, io, pic.path, pic.vis_w, pic.vis_h);
            defer allocator.free(vis_rgb);
            var encoded_v = try vision.runImage(allocator, io, run.platform, &compiled_v.?, vis_model, vis_cache, vis_rgb, pic.vis_h, pic.vis_w);
            defer encoded_v.deinit(allocator);
            try merged_all.appendSlice(allocator, encoded_v.merged);
            if (span_i < planned.spans.len) {
                const span = planned.spans[span_i];
                span_i += 1;
                for (0..3) |di| {
                    if (encoded_v.deepstack[di].len != 0) {
                        @memcpy(
                            ds_host[di][@as(usize, span.start) * hidden_dim ..][0 .. span.tokens * hidden_dim],
                            encoded_v.deepstack[di][0 .. span.tokens * hidden_dim],
                        );
                    }
                }
            }
            const vae_w = if (pic.kind == .ref) pic.vis_w else geo.pixel_w;
            const vae_h = if (pic.kind == .ref) pic.vis_h else geo.pixel_h;
            const vae_rgb = if (pic.kind == .ref) vis_rgb else try loadRgb(allocator, io, pic.path, vae_w, vae_h);
            defer if (pic.kind != .ref) allocator.free(vae_rgb);
            const still = try visual_enc.encodeStill(run, vae_compiled, vae_bufs, vae_cfg, vae_rgb, vae_h, vae_w);
            defer allocator.free(still);
            try patches.appendSlice(allocator, still);
            log.info("mode: {s} {d}x{d} -> {d} patches", .{ @tagName(pic.kind), vae_w, vae_h, still.len / 96 });
        }
    }

    var audio_patches: std.ArrayList(f32) = .empty;
    errdefer audio_patches.deinit(allocator);
    if (planned.audios.len != 0) {
        const model = audio_enc orelse return error.AudioEncodeMissing;
        const bufs = audio_bufs orelse return error.AudioEncodeMissing;
        const AudioExe = struct { samples: u32, exe: zml.FnExe(audio.encode) };
        var exes: std.ArrayList(AudioExe) = .empty;
        defer {
            for (exes.items) |*e| e.exe.deinit();
            exes.deinit(allocator);
        }
        for (planned.audios) |item| {
            const samples = item.latent_t * audio_enc.?.cfg.hop();
            var exe_i: ?usize = null;
            for (exes.items, 0..) |entry, index| {
                if (entry.samples == samples) {
                    exe_i = index;
                    break;
                }
            }
            if (exe_i == null) {
                const exe = try audio.compileAudioEncode(run, model, samples);
                try exes.append(allocator, .{ .samples = samples, .exe = exe });
                exe_i = exes.items.len - 1;
            }
            const encoded_a = try audio.encodeAudio(run, &exes.items[exe_i.?].exe, bufs, audio_enc.?.cfg, item.stereo);
            defer allocator.free(encoded_a.values);
            std.debug.assert(encoded_a.latent_t == item.latent_t);
            try audio_patches.appendSlice(allocator, encoded_a.values);
        }
    }

    return .{
        .patches = try patches.toOwnedSlice(allocator),
        .merged = try merged_all.toOwnedSlice(allocator),
        .deepstack = ds_host,
        .audio_patches = try audio_patches.toOwnedSlice(allocator),
    };
}
