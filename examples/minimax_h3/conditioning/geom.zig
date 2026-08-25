const std = @import("std");

const config = @import("../core/config.zig");

pub const Size = config.Size;

pub fn snapMultiple(value: u32, multiple: u32) u32 {
    if (value == 0) return multiple;
    return @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(multiple))))) * multiple);
}

/// Aspect-preserving, down only, to the generation pixel area.
pub fn refImageSize(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) error{InvalidAspect}!Size {
    if (src_w == 0 or src_h == 0) return error.InvalidAspect;
    const ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(src_h));
    if (ratio < config.min_aspect or ratio > config.max_aspect) return error.InvalidAspect;
    const src_area = @as(f32, @floatFromInt(src_w)) * @as(f32, @floatFromInt(src_h));
    const scale = @min(1.0, @sqrt(@as(f32, @floatFromInt(canvas_w)) * @as(f32, @floatFromInt(canvas_h)) / src_area));
    const multiple = config.canvas_multiple;
    const w = @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_w)) * scale / @as(f32, @floatFromInt(multiple))))) * multiple);
    const h = @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_h)) * scale / @as(f32, @floatFromInt(multiple))))) * multiple);
    return .{ .w = w, .h = h };
}

/// Own-aspect canvas with 768-short-edge + area cap. Never upscale the source.
pub fn videoCanvas(src_w: u32, src_h: u32) error{InvalidAspect}!Size {
    const adapted = try config.resolveCanvas(@floatFromInt(src_w), @floatFromInt(src_h), config.default_short_side, config.canvas_max_pixels);
    if (@as(u64, src_w) * src_h < @as(u64, adapted.w) * adapted.h) {
        return .{ .w = snapMultiple(src_w, config.canvas_multiple), .h = snapMultiple(src_h, config.canvas_multiple) };
    }
    return adapted;
}

pub fn fillVideoTimestamps(sample_count: u32, out: []f32) u32 {
    const n = @min(sample_count, @as(u32, @intCast(out.len)));
    var i: u32 = 0;
    while (i < n) : (i += 1) out[i] = @as(f32, @floatFromInt(i)) / 2.0;
    return n;
}

pub fn coverCropBox(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) struct { w: u32, h: u32, x: u32, y: u32 } {
    const scale = @max(
        @as(f32, @floatFromInt(dst_w)) / @as(f32, @floatFromInt(src_w)),
        @as(f32, @floatFromInt(dst_h)) / @as(f32, @floatFromInt(src_h)),
    );
    const rw = @max(dst_w, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_w)) * scale))));
    const rh = @max(dst_h, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_h)) * scale))));
    const x = @max(0, (rw - dst_w) / 2);
    const y = @max(0, (rh - dst_h) / 2);
    return .{ .w = rw, .h = rh, .x = x, .y = y };
}

fn sinc(x: f32) f32 {
    if (x == 0) return 1.0;
    const px = std.math.pi * x;
    return @sin(px) / px;
}

fn lanczos3(x: f32) f32 {
    if (x <= -3.0 or x >= 3.0) return 0;
    return sinc(x) * sinc(x / 3.0);
}

fn resize1d(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, horizontal: bool) ![]u8 {
    const dst_h = src_h;
    const out_w = if (horizontal) dst_w else src_w;
    const out_h = if (horizontal) src_h else dst_w;
    _ = dst_h;
    const out = try allocator.alloc(u8, @as(usize, out_w) * out_h * 3);
    const src_len: u32 = if (horizontal) src_w else src_h;
    const dst_len: u32 = if (horizontal) dst_w else dst_w;
    const scale = @as(f32, @floatFromInt(src_len)) / @as(f32, @floatFromInt(dst_len));
    const filterscale = @max(1.0, scale);
    const support = 3.0 * filterscale;

    var dy: u32 = 0;
    while (dy < out_h) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < out_w) : (dx += 1) {
            const dst_i: f32 = @floatFromInt(if (horizontal) dx else dy);
            const center = (dst_i + 0.5) * @as(f32, @floatFromInt(src_len)) / @as(f32, @floatFromInt(dst_len));
            const xmin = @as(i32, @intFromFloat(@floor(center - support)));
            const xmax = @as(i32, @intFromFloat(@ceil(center + support)));
            var acc = [3]f32{ 0, 0, 0 };
            var wsum: f32 = 0;
            var xi = xmin;
            while (xi < xmax) : (xi += 1) {
                const src_pos = std.math.clamp(xi, 0, @as(i32, @intCast(src_len - 1)));
                const weight = lanczos3(((@as(f32, @floatFromInt(src_pos)) + 0.5) - center) / filterscale);
                if (weight == 0) continue;
                wsum += weight;
                const sx: u32 = if (horizontal) @intCast(src_pos) else dx;
                const sy: u32 = if (horizontal) dy else @intCast(src_pos);
                const si = (@as(usize, sy) * src_w + sx) * 3;
                inline for (0..3) |c| acc[c] += weight * @as(f32, @floatFromInt(src[si + c]));
            }
            const di = (@as(usize, dy) * out_w + dx) * 3;
            if (wsum == 0) wsum = 1;
            inline for (0..3) |c| {
                out[di + c] = @intFromFloat(std.math.clamp(@round(acc[c] / wsum), 0, 255));
            }
        }
    }
    return out;
}

pub fn resizeLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    std.debug.assert(src.len == @as(usize, src_w) * src_h * 3);
    if (src_w == dst_w and src_h == dst_h) return allocator.dupe(u8, src);
    const mid = try resize1d(allocator, src, src_w, src_h, dst_w, true);
    defer allocator.free(mid);
    return resize1d(allocator, mid, dst_w, src_h, dst_h, false);
}

pub fn cropRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, x: u32, y: u32, dst_w: u32, dst_h: u32) ![]u8 {
    const out = try allocator.alloc(u8, @as(usize, dst_w) * dst_h * 3);
    var row: u32 = 0;
    while (row < dst_h) : (row += 1) {
        const sy = @min(src_h - 1, y + row);
        var col: u32 = 0;
        while (col < dst_w) : (col += 1) {
            const sx = @min(src_w - 1, x + col);
            const si = (@as(usize, sy) * src_w + sx) * 3;
            const di = (@as(usize, row) * dst_w + col) * 3;
            @memcpy(out[di..][0..3], src[si..][0..3]);
        }
    }
    return out;
}

pub fn stretchLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    return resizeLanczos(allocator, src, src_w, src_h, dst_w, dst_h);
}

pub fn coverCropLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    const box = coverCropBox(src_w, src_h, dst_w, dst_h);
    const resized = try resizeLanczos(allocator, src, src_w, src_h, box.w, box.h);
    defer allocator.free(resized);
    return cropRgb(allocator, resized, box.w, box.h, box.x, box.y, dst_w, dst_h);
}

/// 24 fps hold-resample: each source frame is held until the next slot.
pub fn resampleFrameIndices(src_frames: u32, src_fps: f32, dst_fps: f32, allocator: std.mem.Allocator) ![]u32 {
    if (src_frames == 0) return error.EmptyVideo;
    if (src_fps <= 0 or dst_fps <= 0) return error.InvalidFps;
    if (src_fps == dst_fps) {
        const out = try allocator.alloc(u32, src_frames);
        for (out, 0..) |*d, i| d.* = @intCast(i);
        return out;
    }
    const scale = dst_fps / src_fps;
    const out_len_f = @floor(@as(f32, @floatFromInt(src_frames)) * scale + 0.5);
    const out_len: u32 = @intFromFloat(out_len_f);
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
        const last = if (src_frames == 0) 0 else src_frames - 1;
        while (written < out_len) : (written += 1) out[written] = last;
    }
    return out;
}

pub fn sampleVideoConditionFrames(frames: u32, fps: f32, sample_fps: f32, temporal_patch: u32) !struct { indices_len: u32, block_count: u32 } {
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

pub fn fillVideoConditionIndices(frames: u32, fps: f32, sample_fps: f32, out: []u32) u32 {
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

pub fn fillBlockTimestamps(sample_count: u32, sample_fps: f32, temporal_patch: u32, out: []f32) u32 {
    const padded = sample_count + (temporal_patch - (sample_count % temporal_patch)) % temporal_patch;
    const blocks = padded / temporal_patch;
    std.debug.assert(out.len >= blocks);
    var i: u32 = 0;
    while (i < blocks) : (i += 1) {
        const a = @as(f32, @floatFromInt(i * temporal_patch)) / sample_fps;
        const last_idx = @min(sample_count - 1, (i + 1) * temporal_patch - 1);
        const b = @as(f32, @floatFromInt(last_idx)) / sample_fps;
        out[i] = (a + b) / 2;
    }
    return blocks;
}

/// One decimal place, round half to even.
pub fn formatSeconds1(value: f32, buf: []u8) []const u8 {
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

pub fn hopAlign(n: u32, hop: u32) u32 {
    if (hop == 0) return n;
    return n + (hop - (n % hop)) % hop;
}

pub fn applyRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, indices: []const u32) ![]u8 {
    const plane = @as(usize, src_w) * src_h * 3;
    const out = try allocator.alloc(u8, indices.len * plane);
    for (indices, 0..) |src_i, i| {
        const si = @min(src_i, if (src.len == 0) 0 else @as(u32, @intCast(src.len / plane - 1)));
        @memcpy(out[i * plane ..][0..plane], src[si * plane ..][0..plane]);
    }
    return out;
}

pub fn truncateStereo(allocator: std.mem.Allocator, stereo: []const f32, max_samples: u32) ![]f32 {
    const have: u32 = @intCast(stereo.len / 2);
    const keep = @min(have, max_samples);
    const out = try allocator.alloc(f32, @as(usize, keep) * 2);
    @memcpy(out, stereo[0..out.len]);
    return out;
}

pub fn resampleLinear(allocator: std.mem.Allocator, stereo: []const f32, src_rate: u32, dst_rate: u32) ![]f32 {
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

pub fn monoToStereo(allocator: std.mem.Allocator, mono: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, mono.len * 2);
    for (mono, 0..) |s, i| {
        out[i * 2] = s;
        out[i * 2 + 1] = s;
    }
    return out;
}
