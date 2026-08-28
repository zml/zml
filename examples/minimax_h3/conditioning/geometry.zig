const std = @import("std");

const config = @import("../core/config.zig");

pub const Size = config.Size;

fn snapMultiple(value: u32, multiple: u32) u32 {
    if (value == 0) return multiple;
    return @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(multiple))))) * multiple);
}

/// Official ref2va image geometry: short edge 2048, snap-32, upscale allowed, no area cap.
pub fn refImageSize(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) error{InvalidAspect}!Size {
    _ = canvas_w;
    _ = canvas_h;
    if (src_w == 0 or src_h == 0) return error.InvalidAspect;
    const ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(src_h));
    if (ratio < config.min_aspect or ratio > config.max_aspect) return error.InvalidAspect;
    const short = @min(src_w, src_h);
    const scale = @as(f32, @floatFromInt(config.reference_image_short_edge)) / @as(f32, @floatFromInt(short));
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

/// Pillow `lanczos_filter`: `-3 <= x < 3`, `sinc(x)*sinc(x/3)`.
fn sincFilter(x: f64) f64 {
    if (x == 0.0) return 1.0;
    const px = std.math.pi * x;
    return @sin(px) / px;
}

fn lanczosFilter(x: f64) f64 {
    if (x < -3.0 or x >= 3.0) return 0.0;
    return sincFilter(x) * sincFilter(x / 3.0);
}

/// Pillow `Resample.c` 8bpc path: Q22 weights, uint8 after each 1-D pass.
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

/// PIL/torch `_upsample_bicubic2d_aa` Keys cubic. Official Qwen2VL Fast (`antialias=True`) uses a=-0.5.
fn bicubicKeysAa(x: f64) f64 {
    const a: f64 = -0.5;
    const ax = @abs(x);
    if (ax < 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
    if (ax < 2.0) return ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a;
    return 0;
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

/// Torchvision `tvF.resize(..., BICUBIC, antialias=True)` on uint8: separable `_upsample_bicubic2d_aa`.
fn resize1dBicubicAa(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_len: u32, horizontal: bool) ![]u8 {
    const out_w = if (horizontal) dst_len else src_w;
    const out_h = if (horizontal) src_h else dst_len;
    const out = try allocator.alloc(u8, @as(usize, out_w) * out_h * 3);
    const src_len: u32 = if (horizontal) src_w else src_h;
    const scale = @as(f64, @floatFromInt(src_len)) / @as(f64, @floatFromInt(dst_len));
    const support = if (scale >= 1.0) 2.0 * scale else 2.0;
    const invscale = if (scale >= 1.0) 1.0 / scale else 1.0;
    const max_interp: i64 = @as(i64, @intFromFloat(@ceil(support))) * 2 + 1;

    var dy: u32 = 0;
    while (dy < out_h) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < out_w) : (dx += 1) {
            const i: f64 = @floatFromInt(if (horizontal) dx else dy);
            const center = scale * (i + 0.5);
            const xmin = @max(@as(i64, @intFromFloat(center - support + 0.5)), 0);
            var xsize = @min(@as(i64, @intFromFloat(center + support + 0.5)), @as(i64, @intCast(src_len))) - xmin;
            if (xsize < 0) xsize = 0;
            if (xsize > max_interp) xsize = max_interp;
            var acc = [3]f64{ 0, 0, 0 };
            var wsum: f64 = 0;
            var j: i64 = 0;
            while (j < xsize) : (j += 1) {
                const weight = bicubicKeysAa((@as(f64, @floatFromInt(j + xmin)) - center + 0.5) * invscale);
                wsum += weight;
                const src_pos: u32 = @intCast(xmin + j);
                const sx: u32 = if (horizontal) src_pos else dx;
                const sy: u32 = if (horizontal) dy else src_pos;
                const si = (@as(usize, sy) * src_w + sx) * 3;
                inline for (0..3) |c| acc[c] += weight * @as(f64, @floatFromInt(src[si + c]));
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
    const mid = try resize1dLanczos(allocator, src, src_w, src_h, dst_w, true);
    defer allocator.free(mid);
    return resize1dLanczos(allocator, mid, dst_w, src_h, dst_h, false);
}

pub fn resizeBicubic(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    std.debug.assert(src.len == @as(usize, src_w) * src_h * 3);
    if (src_w == dst_w and src_h == dst_h) return allocator.dupe(u8, src);
    const mid = try resize1dBicubicAa(allocator, src, src_w, src_h, dst_w, true);
    defer allocator.free(mid);
    return resize1dBicubicAa(allocator, mid, dst_w, src_h, dst_h, false);
}

fn cropRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, x: u32, y: u32, dst_w: u32, dst_h: u32) ![]u8 {
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
