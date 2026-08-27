const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const config = @import("../core/config.zig");
const vae = @import("../vae/geometry.zig");
const visual_vae = @import("../vae/visual.zig");

pub fn run(allocator: std.mem.Allocator) !void {
    try testEncodeVideoLatentT();
    try testOfficialVaeNativeTile(allocator);
    try testUnpackPatches(allocator);
    try testVaeGeometry();
    try testVaeTiling(allocator);
    try testOfficialStitch(allocator);
    try testVitCoords();
    try testImagenet();
    try testOfficialAudioLatents();
    try testOfficialVisualLatents();
    try testTokenDrop();
    try testAudioRowBct();
    try testPosterior(allocator);
    try testVaeEmbedBatchShapes();
}

/// `embed` builds register/cls/concat shapes with `setDim` on the token batch.
fn testVaeEmbedBatchShapes() !void {
    const tokens = zml.Shape.init(.{ .b = 8, .s = 10, .d = 16 }, .bf16).withPartitioning(.{ .b = .model });
    const registers = tokens.setDim(.s, 4);
    const cls = tokens.setDim(.s, 1);
    const hidden = tokens.setDim(.s, tokens.dim(.s) + registers.dim(.s) + cls.dim(.s));
    try std.testing.expect(tokens.partition(.b).eql(.init(.model)));
    try std.testing.expect(registers.partition(.b).eql(.init(.model)));
    try std.testing.expect(cls.partition(.b).eql(.init(.model)));
    try std.testing.expect(hidden.partition(.b).eql(.init(.model)));
    try std.testing.expect(!zml.Shape.init(.{ .b = 8, .s = 15, .d = 16 }, .bf16).partition(.b).eql(.init(.model)));
    try std.testing.expectEqual(@as(i64, 15), hidden.dim(.s));
}

fn testEncodeVideoLatentT() !void {
    try std.testing.expectEqual(@as(u32, 2), vae.encodeVideoLatentT(vae.official_visual, 5));
    try std.testing.expectEqual(@as(u32, 2), vae.encodeVideoLatentT(vae.official_visual, 17));
    try std.testing.expectEqual(@as(u32, 7), vae.encodeVideoLatentT(vae.official_visual, 34));
    try std.testing.expectEqual(@as(u32, 37), vae.encodeVideoLatentT(vae.official_visual, 120));
    try std.testing.expectEqual(
        config.videoLatentFrames(config.alignFrameCount(120)),
        vae.encodeVideoLatentT(vae.official_visual, 120),
    );
}
fn testOfficialVaeNativeTile(allocator: std.mem.Allocator) !void {
    const spec = vae.official_visual;
    const y = try vae.splitTiles(allocator, 128, spec.tile_px, spec.tile_overlap_px, spec.spatial);
    defer y.deinit(allocator);
    const x = try vae.splitTiles(allocator, 224, spec.tile_px, spec.tile_overlap_px, spec.spatial);
    defer x.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), y.starts.len);
    try std.testing.expectEqual(@as(u32, 128), y.lengths[0]);
    try std.testing.expectEqual(@as(usize, 1), x.starts.len);
    try std.testing.expectEqual(@as(u32, 224), x.lengths[0]);
    try std.testing.expectEqual(@as(u32, 128), @min(spec.tile_px, @as(u32, 128)));
    try std.testing.expectEqual(@as(u32, 224), @min(spec.tile_px, @as(u32, 224)));
    try std.testing.expectEqual(@as(u32, 256), @min(spec.tile_px, @as(u32, 768)));
}
fn testUnpackPatches(allocator: std.mem.Allocator) !void {
    const patches = [_]f32{ 1, 2, 3, 4 };
    const out = try visual_vae.unpackPatches(allocator, &patches, 1, 1, 1, 1, 2, 1);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, out);

    const two = [_]f32{ 10, 11, 12, 13, 20, 21, 22, 23 };
    const nchw = try visual_vae.unpackPatches(allocator, &two, 1, 1, 1, 1, 2, 2);
    defer allocator.free(nchw);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 13, 20, 21, 22, 23 }, nchw);
}
fn testVaeGeometry() !void {
    const lat = vae.official_visual.latentFromPixels(768, 1376, 120);
    try std.testing.expectEqual(@as(u32, 37), lat.t);
    try std.testing.expectEqual(@as(u32, 48), lat.h);
    try std.testing.expectEqual(@as(u32, 86), lat.w);
    const official = vae.official_visual.latentFromPixels(768, 1344, 120);
    try std.testing.expectEqual(@as(u32, 84), official.w);
    try std.testing.expectEqual(@as(u32, 96), vae.official_visual.patchDim());
    try std.testing.expectEqual(@as(u32, 200), vae.official_audio.tokenCount(100));
}
fn testVaeTiling(allocator: std.mem.Allocator) !void {
    const one = try vae.splitTiles(allocator, 128, 256, 64, 16);
    defer one.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), one.count());
    try std.testing.expectEqual(@as(u32, 128), one.lengths[0]);

    const many = try vae.splitTiles(allocator, 640, 256, 64, 16);
    defer many.deinit(allocator);
    try std.testing.expect(many.count() >= 3);
    try std.testing.expectEqual(@as(u32, 0), many.starts[0]);
}
fn testOfficialStitch(allocator: std.mem.Allocator) !void {
    var tiles: [4 * 16]f32 = undefined;
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const yi: f32 = @floatFromInt(i / 2);
        const xi: f32 = @floatFromInt(i % 2);
        var k: usize = 0;
        while (k < 16) : (k += 1) {
            tiles[i * 16 + k] = @as(f32, @floatFromInt(k)) + 100.0 * yi + 10.0 * xi;
        }
    }
    var acc: [36]f32 = undefined;
    @memset(&acc, 0);
    const y_ov = [_]u32{2};
    const x_ov = [_]u32{2};
    var stitcher = try vae.NchwStitcher.init(allocator, &acc, 1, 1, 6, 6, 4, 4, 2, 2, &y_ov, &x_ov);
    defer stitcher.deinit(allocator);
    stitcher.push(0, 0, tiles[0..16]);
    stitcher.push(0, 1, tiles[16..32]);
    stitcher.push(1, 0, tiles[32..48]);
    stitcher.push(1, 1, tiles[48..64]);
    const want = [_]f32{
        0.0,   1.0,   2.0,   7.0,   12.0,  13.0,
        4.0,   5.0,   6.0,   11.0,  16.0,  17.0,
        8.0,   9.0,   102.0, 61.0,  20.0,  21.0,
        58.0,  59.0,  106.0, 88.0,  70.0,  71.0,
        108.0, 109.0, 110.0, 115.0, 120.0, 121.0,
        112.0, 113.0, 114.0, 119.0, 124.0, 125.0,
    };
    try std.testing.expectEqualSlices(f32, &want, &acc);
}
fn testVitCoords() !void {
    var buf: [4]f32 = undefined;
    const axis = vae.vitCoords(4, &buf);
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), axis[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), axis[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), axis[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), axis[3], 1e-6);
}
fn testImagenet() !void {
    var px = [_]f32{ 0.0, 0.0, 0.0 };
    vae.denormImagenetRgb(&px);
    try std.testing.expectApproxEqAbs(vae.imagenet_mean[0], px[0], 1e-5);
}
fn testOfficialAudioLatents() !void {
    const cfg = audio_vae.Config.official();
    try std.testing.expectEqualSlices(f32, &audio_vae.official_latents_mean, &cfg.latents_mean);
    try std.testing.expectEqualSlices(f32, &audio_vae.official_latents_std, &cfg.latents_std);
    try std.testing.expect(cfg.latents_std[0] != 1.0);
}
fn testOfficialVisualLatents() !void {
    const cfg = visual_vae.Config.official();
    try std.testing.expectEqualSlices(f32, &visual_vae.official_latents_mean, &cfg.latents_mean);
    try std.testing.expectEqualSlices(f32, &visual_vae.official_latents_std, &cfg.latents_std);
    try std.testing.expectEqual(@as(i64, 48), cfg.rotaryDim());
    try std.testing.expect(cfg.latents_std[0] != 1.0);
}
fn testTokenDrop() !void {
    const spec = vae.official_visual;
    try std.testing.expectEqual(@as(u32, 5), spec.tokensChunkSize());
    try std.testing.expectEqual(@as(u32, 2), spec.tokenOverlap());
    try std.testing.expectEqual(@as(u32, 3), spec.framePrePadding());
    try std.testing.expectEqual(@as(u32, 5), spec.frameOverlap());
}
fn testAudioRowBct() !void {
    const channels: u32 = 2;
    const t: u32 = 3;
    const rows = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var bct: [12]f32 = undefined;
    vae.audioRowsToBct(&bct, &rows, channels, t);
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11 }, &bct);
    var back: [12]f32 = undefined;
    vae.audioBctToRows(&back, &bct, channels, t);
    try std.testing.expectEqualSlices(f32, &rows, &back);
}
fn testPosterior(allocator: std.mem.Allocator) !void {
    var moments: [48]f32 = undefined;
    @memset(moments[0..24], 1.0);
    @memset(moments[24..], 0.0);
    const a = try vae.sampleVisualPosteriorNchw(allocator, &moments, 1, 1, 1);
    defer allocator.free(a);
    const b = try vae.sampleVisualPosteriorNchw(allocator, &moments, 1, 1, 1);
    defer allocator.free(b);
    try std.testing.expectEqualSlices(f32, a, b);
    try std.testing.expect(!std.mem.eql(f32, a, moments[0..24]));
}
