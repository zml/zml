const std = @import("std");

const geom = @import("../conditioning/geometry.zig");
const packing = @import("../model/packing.zig");
const conditions = @import("../runtime/conditions.zig");
const encode = @import("../runtime/encode.zig");
const media = @import("../runtime/media.zig");

pub fn run(allocator: std.mem.Allocator) !void {
    try testResample(allocator);
    try testReferenceIndexDomains();
    try testOfficialAudioTruncate();
    try testMediaErrors();
    try testFfmpegProbe();
    try testOutputTarget();
    try testExportVideo(allocator);
    try testNchwTimeLayout();
}

fn expectReferenceIndices(
    kinds: []const packing.ReferenceKind,
    video_has_audio: []const bool,
    expected: []const packing.ReferenceBlock,
) !void {
    try std.testing.expectEqual(kinds.len, video_has_audio.len);
    try std.testing.expectEqual(kinds.len, expected.len);
    var domains: conditions.ReferenceIndexDomains = .{};
    for (kinds, video_has_audio, expected) |kind, has_audio, want| {
        const got = domains.next(kind, has_audio);
        try std.testing.expectEqual(want.kind, got.kind);
        try std.testing.expectEqual(want.video_index, got.video_index);
        try std.testing.expectEqual(want.audio_index, got.audio_index);
    }
}

fn testReferenceIndexDomains() !void {
    try expectReferenceIndices(&.{.image}, &.{false}, &.{.{ .kind = .image, .video_index = 0 }});
    try expectReferenceIndices(&.{.video}, &.{false}, &.{.{ .kind = .video, .video_index = 0 }});
    try expectReferenceIndices(
        &.{ .audio, .video },
        &.{ true, false },
        &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video, .video_index = 0 },
        },
    );
    try expectReferenceIndices(
        &.{ .audio, .image },
        &.{ true, false },
        &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .image, .video_index = 0 },
        },
    );
    try expectReferenceIndices(
        &.{ .image, .audio },
        &.{ false, true },
        &.{
            .{ .kind = .image, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
        },
    );
    try expectReferenceIndices(
        &.{ .video, .audio },
        &.{ false, true },
        &.{
            .{ .kind = .video, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
        },
    );
    const triples = [_][3]packing.ReferenceKind{
        .{ .image, .audio, .video },
        .{ .audio, .image, .video },
        .{ .video, .audio, .image },
        .{ .image, .video, .audio },
    };
    for (triples) |kinds| {
        var domains: conditions.ReferenceIndexDomains = .{};
        var visual_index: i32 = 0;
        var audio_index: i32 = 0;
        for (kinds) |kind| {
            const block = domains.next(kind, false);
            if (kind == .audio) {
                try std.testing.expectEqual(audio_index, block.audio_index);
                try std.testing.expectEqual(@as(i32, -1), block.video_index);
                audio_index += 1;
            } else {
                try std.testing.expectEqual(visual_index, block.video_index);
                try std.testing.expectEqual(@as(i32, -1), block.audio_index);
                visual_index += 1;
            }
        }
    }

    // A video's soundtrack shares its reference block, but occupies its own
    // encoded-audio slot and does not perturb later visual indices.
    try expectReferenceIndices(
        &.{ .audio, .video_audio, .image },
        &.{ true, true, false },
        &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video_audio, .video_index = 0, .audio_index = 1 },
            .{ .kind = .image, .video_index = 1 },
        },
    );
}

fn testOfficialAudioTruncate() !void {
    // Official: int(max_duration * sample_rate) toward zero, not rounded.
    try std.testing.expectEqual(@as(u32, 0), media.officialTruncateSamples(0.5, 1));
    try std.testing.expectEqual(@as(u32, 2), media.officialTruncateSamples(2.9, 1));
    try std.testing.expectEqual(@as(u32, 160000), media.officialTruncateSamples(5.0, 32000));
}

fn testResample(allocator: std.mem.Allocator) !void {
    const stereo = [_]f32{ 0, 0, 1, 1 };
    const out = try geom.resampleLinear(allocator, &stereo, 2, 4);
    defer allocator.free(out);
    try std.testing.expectEqual(@as(usize, 8), out.len);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 1), out[out.len - 1]);
}
fn testMediaErrors() !void {
    try std.testing.expectError(error.BadWav, media.parseWavHeader("not a wav"));
}
fn testFfmpegProbe() !void {
    const sample =
        \\Input #0, mov, from 'clip.mp4':
        \\  Stream #0:0: Video: h264 (High), yuv420p, 320x180, 24 fps, 24 tbr, 12288 tbn
        \\  Stream #0:1: Audio: aac, 48000 Hz, stereo
    ;
    const meta = try media.parseFfmpegProbe(sample);
    try std.testing.expectEqual(@as(u32, 320), meta.w);
    try std.testing.expectEqual(@as(u32, 180), meta.h);
    try std.testing.expectEqual(@as(f32, 24), meta.fps);
    try std.testing.expect(meta.has_audio);
    try std.testing.expectError(error.VideoLoadFailed, media.parseFfmpegProbe("no streams"));
}
fn testOutputTarget() !void {
    const def = media.Output.parse("");
    try std.testing.expectEqualStrings("output", def.dir);
    try std.testing.expectEqualStrings("output.mp4", def.mp4_name);
    try std.testing.expect(!def.isCwd());

    const dir = media.Output.parse("out_t2va");
    try std.testing.expectEqualStrings("out_t2va", dir.dir);
    try std.testing.expectEqualStrings("output.mp4", dir.mp4_name);

    const file = media.Output.parse("clips/waves.mp4");
    try std.testing.expectEqualStrings("clips", file.dir);
    try std.testing.expectEqualStrings("waves.mp4", file.mp4_name);

    const cwd = media.Output.parse("output.mp4");
    try std.testing.expectEqualStrings(".", cwd.dir);
    try std.testing.expectEqualStrings("output.mp4", cwd.mp4_name);
    try std.testing.expect(cwd.isCwd());
}
fn testExportVideo(allocator: std.mem.Allocator) !void {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var scratch = try media.Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const dest_path = scratch.path;
    var dest = try media.openPath(io, dest_path);
    defer dest.close(io);

    const nchw = [_]f32{1} ** (16 * 16 * 3);
    const pcm = [_]i16{0} ** 16000;
    const muxed = try media.writeGeneratedVideo(
        allocator,
        io,
        dest,
        dest_path,
        "clip.mp4",
        &nchw,
        1,
        16,
        16,
        &pcm,
        32000,
    );
    if (dest.openFile(io, "frame_0000.ppm", .{ .mode = .read_only })) |f| {
        f.close(io);
        return error.TestUnexpectedResult;
    } else |_| {}
    if (muxed) {
        var mp4 = dest.openFile(io, "clip.mp4", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        mp4.close(io);
    } else {
        var frame = dest.openFile(io, "frames/frame_0000.ppm", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        frame.close(io);
        var wav = dest.openFile(io, "audio.wav", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        wav.close(io);
    }
}

fn testNchwTimeLayout() !void {
    const padded_src = [_]f32{ 10, 11, 20, 21, 30, 31 };
    var padded: [12]f32 = undefined;
    encode.padTimeNchw(&padded, &padded_src, 3, 2, 4, 1, 1);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 11, 11, 20, 21, 21, 21, 30, 31, 31, 31 }, &padded);

    const clip0 = [_]f32{ 10, 11, 20, 21 };
    const clip1 = [_]f32{ 12, 13, 22, 23 };
    var concat: [8]f32 = undefined;
    @memset(&concat, 0);
    encode.copyTimeChunkNchw(&concat, 4, 0, &clip0, 2, 2, 1, 1);
    encode.copyTimeChunkNchw(&concat, 4, 2, &clip1, 2, 2, 1, 1);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 13, 20, 21, 22, 23 }, &concat);

    const prefix_src = [_]f32{ 10, 11, 12, 13, 20, 21, 22, 23 };
    var prefix: [4]f32 = undefined;
    encode.compactTimePrefixNchw(&prefix, &prefix_src, 2, 4, 2, 1, 1);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 20, 21 }, &prefix);

    var combined = concat;
    encode.compactTimePrefixNchw(&combined, &combined, 2, 4, 2, 1, 1);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 20, 21 }, combined[0..4]);

    var overlap = prefix_src;
    encode.compactTimePrefixNchw(&overlap, &overlap, 2, 4, 3, 1, 1);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 20, 21, 22 }, overlap[0..6]);
}
