const std = @import("std");

const geom = @import("../conditioning/geometry.zig");
const packing = @import("../model/packing.zig");
const presentation = @import("../conditioning/presentation.zig");
const vision = @import("../model/vision.zig");

pub fn run(allocator: std.mem.Allocator) !void {
    try testVisionSpatial();
    try testPatchify(allocator);
    try testOfficialVisionRope(allocator);
    try testGeomHost(allocator);
    try testPresentation(allocator);
    try testLastOnlyFl2va(allocator);
    try testRefSize();
    try testPixelCrc(allocator);
    try testStandaloneAudio(allocator);
    try testFirstLastFl2va(allocator);
}

fn testVisionSpatial() !void {
    var cfg = vision.Config{};
    cfg.out_hidden_size = 5120;
    const spec = vision.spatialTokens(cfg, 256, 256, false);
    try std.testing.expectEqual(@as(u32, 0), spec.seq % 4);
    try std.testing.expectEqual(spec.seq / 4, spec.merged);
    var cursor: f32 = 0;
    var pos: [12]f32 = undefined;
    vision.applyVisionPositions(&pos, 0, 4, 4, 4, 1, &cursor);
    try std.testing.expect(cursor > 0);

    // Official Qwen2VL smart_resize on the canvas keyframe, not the source file.
    const tiny = vision.chooseGrid(cfg, 128, 224, false);
    try std.testing.expectEqual(@as(u32, 224), tiny.h);
    try std.testing.expectEqual(@as(u32, 352), tiny.w);
    const preview = vision.chooseGrid(cfg, 352, 640, false);
    try std.testing.expectEqual(@as(u32, 352), preview.h);
    try std.testing.expectEqual(@as(u32, 640), preview.w);
    const full = vision.chooseGrid(cfg, 768, 1344, false);
    try std.testing.expectEqual(@as(u32, 768), full.h);
    try std.testing.expectEqual(@as(u32, 1344), full.w);

    // Python 3 even ties: 144/32=4.5→4, 208/32=6.5→6, then min_pixels / in-range snap.
    const even_lo = vision.chooseGrid(cfg, 144, 144, false);
    try std.testing.expectEqual(@as(u32, 256), even_lo.h);
    try std.testing.expectEqual(@as(u32, 256), even_lo.w);
    const even_in = vision.chooseGrid(cfg, 208, 512, false);
    try std.testing.expectEqual(@as(u32, 192), even_in.h);
    try std.testing.expectEqual(@as(u32, 512), even_in.w);
    const vid = vision.chooseGrid(cfg, 8, 8, true);
    try std.testing.expectEqual(@as(u32, 64), vid.h);
    try std.testing.expectEqual(@as(u32, 64), vid.w);
}
fn testPatchify(allocator: std.mem.Allocator) !void {
    const t: u32 = 2;
    const h: u32 = 4;
    const w: u32 = 4;
    const c: u32 = 2;
    const src = try allocator.alloc(f32, t * h * w * c);
    defer allocator.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);

    const rows = try packing.patchify(allocator, src, t, h, w, c, .{ 1, 2, 2 });
    defer allocator.free(rows);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * (2 * 1 * 2 * 2)), rows.len);
    // First 2×2 patch, channel-major: ch0 of four voxels, then ch1.
    try std.testing.expectEqual(@as(f32, 0), rows[0]);
    try std.testing.expectEqual(@as(f32, 2), rows[1]);
    try std.testing.expectEqual(@as(f32, 8), rows[2]);
    try std.testing.expectEqual(@as(f32, 10), rows[3]);

    const back = try packing.unpatchify(allocator, rows, t, h, w, c, .{ 1, 2, 2 });
    defer allocator.free(back);
    try std.testing.expectEqualSlices(f32, src, back);
}
fn testOfficialVisionRope(allocator: std.mem.Allocator) !void {
    const head_dim: u32 = 72;
    const rope = try vision.visionRope(allocator, 2, 2, head_dim);
    defer allocator.free(rope.cos);
    defer allocator.free(rope.sin);
    try std.testing.expectEqual(@as(usize, 4 * head_dim), rope.cos.len);

    const half: u32 = head_dim / 2;
    const n_freq: u32 = half / 2;
    const hpos: f32 = 1;
    const wpos: f32 = 1;
    const freq0 = 1.0;
    const freq1 = 1.0 / std.math.pow(f32, 10000.0, 2.0 / @as(f32, @floatFromInt(half)));
    const row = 3 * head_dim;
    try std.testing.expectApproxEqAbs(@cos(hpos * freq0), rope.cos[row], 1e-6);
    try std.testing.expectApproxEqAbs(@cos(hpos * freq1), rope.cos[row + 1], 1e-6);
    try std.testing.expectApproxEqAbs(@cos(wpos * freq0), rope.cos[row + n_freq], 1e-6);
    try std.testing.expectApproxEqAbs(rope.cos[row], rope.cos[row + half], 1e-6);
    try std.testing.expectApproxEqAbs(rope.sin[row + 1], rope.sin[row + half + 1], 1e-6);
}
fn testGeomHost(allocator: std.mem.Allocator) !void {
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("0.2", geom.formatSeconds1(0.25, &buf));
    try std.testing.expectEqualStrings("0.8", geom.formatSeconds1(0.75, &buf));
    try std.testing.expectEqualStrings("1.2", geom.formatSeconds1(1.25, &buf));

    const ref = try geom.refImageSize(2048, 2048, 640, 352);
    try std.testing.expectEqual(@as(u32, 2048), ref.w);
    try std.testing.expectEqual(@as(u32, 2048), ref.h);
    try std.testing.expectError(error.InvalidAspect, geom.refImageSize(100, 10, 640, 352));

    const box = geom.coverCropBox(100, 50, 32, 32);
    try std.testing.expectEqual(@as(u32, 64), box.w);
    try std.testing.expectEqual(@as(u32, 32), box.h);
    try std.testing.expectEqual(@as(u32, 16), box.x);
    try std.testing.expectEqual(@as(u32, 0), box.y);

    const idx = try geom.resampleFrameIndices(2, 12, 24, allocator);
    defer allocator.free(idx);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1 }, idx);

    const sampled = try geom.sampleVideoConditionFrames(24, 24, 2, 2);
    try std.testing.expectEqual(@as(u32, 2), sampled.indices_len);
    try std.testing.expectEqual(@as(u32, 1), sampled.block_count);
    var qidx: [4]u32 = undefined;
    try std.testing.expectEqual(@as(u32, 2), geom.fillVideoConditionIndices(24, 24, 2, &qidx));
    try std.testing.expectEqual(@as(u32, 0), qidx[0]);
    try std.testing.expectEqual(@as(u32, 12), qidx[1]);
    var ts: [1]f32 = undefined;
    try std.testing.expectEqual(@as(u32, 1), geom.fillBlockTimestamps(2, 2, 2, &ts));
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), ts[0], 1e-6);
}
const StubEnc = struct {
    pub fn encodeAlloc(_: @This(), allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try allocator.alloc(u32, text.len);
        for (text, out) |c, *d| d.* = c;
        return out;
    }
};
fn testPresentation(allocator: std.mem.Allocator) !void {
    const enc = StubEnc{};
    var t2 = try presentation.assembleT2va(allocator, enc, "hello");
    defer t2.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 'h', 'e', 'l', 'l', 'o' }, t2.tokens);
    try std.testing.expectEqual(@as(usize, 0), t2.spans.len);

    const fl_specs = [_]presentation.VisualSpec{.{
        .kind = .image,
        .merged = 4,
        .grid_h = 2,
        .grid_w = 2,
    }};
    var fl = try presentation.assembleFl2va(allocator, enc, &fl_specs, "ZZ");
    defer fl.deinit(allocator);
    try std.testing.expectEqual(@as(u32, '<'), fl.tokens[0]);
    try std.testing.expect(std.mem.indexOfScalar(u32, fl.tokens, vision.VISION_START) != null);
    try std.testing.expect(std.mem.indexOfScalar(u32, fl.tokens, vision.IMAGE_PAD) != null);
    try std.testing.expectEqual(@as(u32, 'Z'), fl.tokens[fl.tokens.len - 2]);
    try std.testing.expectEqual(@as(u32, 'Z'), fl.tokens[fl.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), fl.spans.len);

    const ts = [_]f32{0.25};
    const ref_specs = [_]presentation.VisualSpec{.{
        .kind = .video_audio,
        .merged = 2,
        .grid_h = 1,
        .grid_w = 2,
        .temporal = 1,
        .timestamps = &ts,
        .has_audio = true,
    }};
    var ref = try presentation.assembleRef2va(allocator, enc, &ref_specs, "p");
    defer ref.deinit(allocator);
    try std.testing.expect(containsAscii(ref.tokens, "<Audio 1>: "));
    try std.testing.expect(containsAscii(ref.tokens, "<Video 1>: "));
    try std.testing.expect(containsAscii(ref.tokens, "<0.2 seconds>"));
    const audio_at = indexOfAscii(ref.tokens, "<Audio 1>: ").?;
    const video_at = indexOfAscii(ref.tokens, "<Video 1>: ").?;
    try std.testing.expect(audio_at < video_at);
    try std.testing.expect(std.mem.indexOfScalar(u32, ref.tokens, vision.VIDEO_PAD) != null);
    try std.testing.expectEqual(@as(u32, 'p'), ref.tokens[ref.tokens.len - 1]);
}
fn containsAscii(tokens: []const u32, text: []const u8) bool {
    return indexOfAscii(tokens, text) != null;
}
fn indexOfAscii(tokens: []const u32, text: []const u8) ?usize {
    if (text.len == 0 or text.len > tokens.len) return null;
    var i: usize = 0;
    while (i + text.len <= tokens.len) : (i += 1) {
        var ok = true;
        for (text, 0..) |c, j| {
            if (tokens[i + j] != c) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}
const OfficialEnc = struct {
    pub fn encodeAlloc(_: @This(), allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (std.mem.eql(u8, text, "<Picture 1>: ")) return allocator.dupe(u32, &.{ 21604, 3826, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<Audio 1>: ")) return allocator.dupe(u32, &.{ 65406, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<Video 1>: ")) return allocator.dupe(u32, &.{ 27, 10724, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<0.2 seconds>")) return allocator.dupe(u32, &.{ 27, 15, 13, 17, 6486, 29 });
        if (std.mem.eql(u8, text, "hello")) return allocator.dupe(u32, &.{14990});
        const out = try allocator.alloc(u32, text.len);
        for (text, out) |c, *d| d.* = c;
        return out;
    }
};
fn testLastOnlyFl2va(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{.{
        .kind = .image,
        .merged = 2,
        .grid_h = 1,
        .grid_w = 2,
    }};
    var assembled = try presentation.assembleFl2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 21604, 3826, 220, 16, 26818, 220 }, assembled.tokens[0..6]);
    try std.testing.expectEqual(@as(u32, 14990), assembled.tokens[assembled.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), assembled.spans.len);
}
fn testRefSize() !void {
    const match = try geom.refImageSize(2048, 2048, 640, 352);
    try std.testing.expectEqual(@as(u32, 2048), match.w);
    try std.testing.expectEqual(@as(u32, 2048), match.h);
    const up = try geom.refImageSize(256, 256, 640, 352);
    try std.testing.expectEqual(@as(u32, 2048), up.w);
    try std.testing.expectEqual(@as(u32, 2048), up.h);
    const wide = try geom.refImageSize(1920, 1080, 224, 128);
    try std.testing.expectEqual(@as(u32, 3648), wide.w);
    try std.testing.expectEqual(@as(u32, 2048), wide.h);
    const small_vid = try geom.videoCanvas(320, 180);
    try std.testing.expect(small_vid.w <= 320 + 32);
    var ts: [3]f32 = undefined;
    try std.testing.expectEqual(@as(u32, 3), geom.fillVideoTimestamps(3, &ts));
    try std.testing.expectEqual(@as(f32, 0), ts[0]);
    try std.testing.expectEqual(@as(f32, 1.0), ts[2]);
}
fn testPixelCrc(allocator: std.mem.Allocator) !void {
    const src = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    const a = try geom.stretchLanczos(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(a);
    const b = try geom.stretchLanczos(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(b);
    try std.testing.expectEqual(std.hash.Crc32.hash(a), std.hash.Crc32.hash(b));
    try std.testing.expectEqual(@as(usize, 48), a.len);
    const crop = try geom.coverCropLanczos(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(crop);
    try std.testing.expectEqual(@as(usize, 12), crop.len);
    const same = try geom.resizeBicubic(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(same);
    try std.testing.expectEqualSlices(u8, &src, same);
    const up = try geom.resizeBicubic(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(up);
    const up2 = try geom.resizeBicubic(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(up2);
    try std.testing.expectEqual(std.hash.Crc32.hash(up), std.hash.Crc32.hash(up2));
    try std.testing.expect(std.hash.Crc32.hash(up) != std.hash.Crc32.hash(a));
    // torchvision tvF.resize uint8 BICUBIC antialias=True
    const torch_aa = [_]u8{
        255, 0, 0,   215, 53, 0,   40,  202, 0,   0,   255, 0,
        202, 0, 53,  171, 53, 53,  84,  202, 53,  53,  255, 53,
        53,  0, 202, 84,  53, 202, 171, 202, 202, 202, 255, 202,
        0,   0, 255, 40,  53, 255, 215, 202, 255, 255, 255, 255,
    };
    try std.testing.expectEqualSlices(u8, &torch_aa, up);
}
fn testStandaloneAudio(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{.{
        .kind = .audio,
        .merged = 0,
        .grid_h = 1,
        .grid_w = 1,
        .has_audio = true,
    }};
    var assembled = try presentation.assembleRef2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 65406, 220, 16, 26818, 220 }, assembled.tokens[0..5]);
    try std.testing.expectEqual(@as(u32, 14990), assembled.tokens[assembled.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 0), assembled.spans.len);
}
fn testFirstLastFl2va(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{
        .{ .kind = .image, .merged = 2, .grid_h = 1, .grid_w = 2 },
        .{ .kind = .image, .merged = 2, .grid_h = 1, .grid_w = 2 },
    };
    var assembled = try presentation.assembleFl2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), assembled.spans.len);
    try std.testing.expectEqualSlices(u32, &.{ 21604, 3826, 220, 16, 26818, 220 }, assembled.tokens[0..6]);
}
