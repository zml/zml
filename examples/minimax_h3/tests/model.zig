const std = @import("std");

const policy_mod = @import("../core/policy.zig");
const config = @import("../core/config.zig");
const noise = @import("../model/noise.zig");
const packing = @import("../model/packing.zig");
const scheduler = @import("../model/scheduler.zig");

pub fn run(allocator: std.mem.Allocator) !void {
    try testScheduler(allocator);
    try testTimestepEmbedding();
    try testPackingT2va(allocator);
    try testPackingTimestepSlots(allocator);
    try testOfficialRowTimesteps(allocator);
    try testPackingFl2va(allocator);
    try testPackingRef2va(allocator);
    try testRef2vaOrderPermutations(allocator);
    try testUnequalReferenceAudioLengths(allocator);
    try testAdalnResidualHost();
    try testNchwToThwc();
    try testMmRopeHost();
    try testOfficialSpatialGrid();
    try testTorchNoise(allocator);
    try testMultistepSampler();
    try testAdalnIndexLayout(allocator);
    try testSchedulerFormula(allocator);
}

fn testAudioLengthOrder(allocator: std.mem.Allocator, lengths: []const u32) !void {
    var audios: [3]packing.ConditionAudio = undefined;
    var refs: [4]packing.ReferenceBlock = undefined;
    for (lengths, 0..) |length, i| {
        audios[i] = .{ .latent_t = length };
        refs[i] = .{ .kind = .audio, .audio_index = @intCast(i) };
    }
    refs[lengths.len] = .{ .kind = .image, .video_index = 0 };
    const videos = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
    }};
    const text_len: u32 = 2;
    const target_audio_t: u32 = 3;
    const target_video_t: u32 = 2;
    const layout = try packing.build(allocator, .{
        .text_len = text_len,
        .latent_t = target_video_t,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = target_audio_t,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = &videos,
        .condition_audios = audios[0..lengths.len],
        .references = refs[0 .. lengths.len + 1],
    });
    defer layout.deinit(allocator);

    var cursor: f32 = @floatFromInt(text_len);
    for (lengths, 0..) |length, i| {
        const segment = layout.segments[1 + i];
        try std.testing.expectEqual(packing.SegmentKind.condition_audio, segment.kind);
        try std.testing.expectEqual(@as(i32, @intCast(i)), segment.source_index);
        try std.testing.expectEqual(length * 2, segment.end - segment.start);
        try std.testing.expectApproxEqAbs(cursor, layout.positions[segment.start].t, 1e-6);
        cursor += @floatFromInt(length);
    }
    const image_segment = layout.segments[1 + lengths.len];
    try std.testing.expectEqual(packing.SegmentKind.condition_video, image_segment.kind);
    try std.testing.expectApproxEqAbs(cursor, layout.positions[image_segment.start].t, 1e-6);
    cursor += 1;
    try std.testing.expectApproxEqAbs(cursor, layout.positions[layout.target_audio_start].t, 1e-6);
    try std.testing.expectApproxEqAbs(cursor, layout.positions[layout.target_video_start].t, 1e-6);

    var reference_audio_rows: u32 = 0;
    for (lengths) |length| reference_audio_rows += length * 2;
    const condition_video_rows: u32 = 4;
    const target_audio_rows = target_audio_t * 2;
    const target_video_rows = config.videoTokenCount(target_video_t, 4, 4, .{ 1, 2, 2 });
    try std.testing.expectEqual(reference_audio_rows, layout.conditionAudioRows());
    try std.testing.expectEqual(condition_video_rows, layout.conditionVideoRows());
    try std.testing.expectEqual(
        text_len + reference_audio_rows + condition_video_rows + target_audio_rows + target_video_rows,
        layout.seqLen(),
    );
}

fn testUnequalReferenceAudioLengths(allocator: std.mem.Allocator) !void {
    // 40 audio latents per second at the official 32 kHz / hop-800 geometry.
    try testAudioLengthOrder(allocator, &.{ 40, 160 });
    try testAudioLengthOrder(allocator, &.{ 160, 40 });
    try testAudioLengthOrder(allocator, &.{ 80, 200, 120 });

    const bad_audio = [_]packing.ReferenceBlock{.{ .kind = .audio, .audio_index = 1 }};
    try std.testing.expectError(error.InvalidReferenceAudioIndex, packing.build(allocator, .{
        .text_len = 1,
        .latent_t = 1,
        .latent_h = 2,
        .latent_w = 2,
        .audio_t = 1,
        .video_t = 0,
        .audio_t_noise = 0,
        .condition_audios = &.{.{ .latent_t = 1 }},
        .references = &bad_audio,
    }));
    const bad_video = [_]packing.ReferenceBlock{.{ .kind = .image, .video_index = 1 }};
    try std.testing.expectError(error.InvalidReferenceVideoIndex, packing.build(allocator, .{
        .text_len = 1,
        .latent_t = 1,
        .latent_h = 2,
        .latent_w = 2,
        .audio_t = 1,
        .video_t = 0,
        .audio_t_noise = 0,
        .condition_videos = &.{.{ .latent_t = 1, .latent_h = 2, .latent_w = 2 }},
        .references = &bad_video,
    }));
}

fn testScheduler(allocator: std.mem.Allocator) !void {
    const sched = try scheduler.Schedule.init(allocator, 12.0, 8);
    defer sched.deinit(allocator);
    try std.testing.expect(sched.sigmas[0] > sched.sigmas[sched.sigmas.len - 1]);
    try std.testing.expectEqual(@as(f32, 0.0), sched.sigmas[sched.sigmas.len - 1]);
    try std.testing.expectEqual(sched.timesteps.len + 1, sched.sigmas.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sched.timesteps[0] + sched.sigmas[0], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scheduler.shiftSigma(1.0, 12.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0 / 13.0), scheduler.shiftSigma(0.5, 12.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), scheduler.shiftSigma(0.0, 12.0));

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), scheduler.Schedule.scaleNoise(0.5, 1.0, 0.0), 1e-6);

    const dual = try scheduler.DualSchedule.init(allocator, 10, config.video_shift, config.audio_shift);
    defer dual.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 12.0), dual.video.shift);
    try std.testing.expectEqual(@as(f32, 3.0), dual.audio.shift);
}
fn testTimestepEmbedding() !void {
    const t = [_]f32{ 0.0, 1.0 };
    var out: [512]f32 = undefined;
    scheduler.timestepEmbedding(&t, 256, true, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[128], 1e-5);
}
fn testPackingT2va(allocator: std.mem.Allocator) !void {
    const layout = try packing.build(allocator, .{
        .text_len = 4,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 3,
        .video_t = 0.25,
        .audio_t_noise = 0.6,
    });
    defer layout.deinit(allocator);

    const video_tokens = config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 });
    const audio_tokens: u32 = 3 * 2;
    try std.testing.expectEqual(4 + audio_tokens + video_tokens, layout.seqLen());
    try std.testing.expectEqual(@as(usize, 4), layout.text_indices.len);
    try std.testing.expectEqual(@as(usize, video_tokens), layout.video_indices.len);
    try std.testing.expectEqual(@as(usize, audio_tokens), layout.audio_indices.len);
    try std.testing.expectEqual(@as(u8, 1), layout.token_tags[0]);
    try std.testing.expectEqual(@as(u8, 2), layout.token_tags[layout.target_audio_start]);
    try std.testing.expectEqual(@as(u8, 0), layout.token_tags[layout.target_video_start]);

    const first_video = layout.adalnIndex(layout.target_video_start);
    try std.testing.expectEqual(first_video % 3, 0);
    const first_text = layout.adalnIndex(0);
    try std.testing.expectEqual(first_text % 3, 1);
    const first_audio = layout.adalnIndex(layout.target_audio_start);
    try std.testing.expectEqual(first_audio % 3, 2);
    try std.testing.expectEqual(packing.timestep_slot_count, @as(u32, @intCast(layout.timesteps.len)));
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[layout.target_video_start]);
    try std.testing.expectEqual(@as(u32, 1), layout.timestep_indices[layout.target_audio_start]);
}
fn testPackingTimestepSlots(allocator: std.mem.Allocator) !void {
    const early = packing.timestepValues(0.99, 0.8);
    const late = packing.timestepValues(0.1, 0.2);
    try std.testing.expectEqual(@as(usize, 4), early.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.99), early[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), early[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), late[3], 1e-6);

    const a = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.99,
        .audio_t_noise = 0.8,
    });
    defer a.deinit(allocator);
    const b = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.1,
        .audio_t_noise = 0.2,
    });
    defer b.deinit(allocator);
    try std.testing.expectEqualSlices(u32, a.video_indices, b.video_indices);
    try std.testing.expectEqual(@as(u32, 1), a.timestep_indices[a.target_video_start]);
    try std.testing.expectEqual(@as(u32, 0), a.timestep_indices[a.target_audio_start]);
    try std.testing.expectEqual(@as(u32, 0), b.timestep_indices[b.target_video_start]);
    try std.testing.expectEqual(@as(u32, 1), b.timestep_indices[b.target_audio_start]);
    try std.testing.expectEqualSlices(f32, &late, &packing.timestepValues(0.1, 0.2));
}
fn testOfficialRowTimesteps(allocator: std.mem.Allocator) !void {
    const videos = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
    }};
    const refs = [_]packing.ReferenceBlock{.{
        .kind = .image,
        .video_index = 0,
    }};
    const layout = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.0,
        .audio_t_noise = 0.0,
        .condition_videos = &videos,
        .references = &refs,
    });
    defer layout.deinit(allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), layout.timesteps[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), layout.timesteps[1], 1e-6);
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[0]);
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[layout.target_video_start]);
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[layout.target_audio_start]);
    try std.testing.expectEqual(@as(u32, 1), layout.timestep_indices[layout.video_indices[0]]);

    const row_ts = try allocator.alloc(f32, layout.seqLen());
    defer allocator.free(row_ts);
    const idx = try allocator.alloc(u32, layout.seqLen());
    defer allocator.free(idx);
    var unique: [4]f32 = undefined;
    const n = packing.writeRowPlan(layout, 0.04, 0.143, row_ts, idx, &unique);
    try std.testing.expectEqual(@as(u32, 3), n);
    try std.testing.expectApproxEqAbs(@as(f32, 0.04), unique[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.143), unique[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), unique[2], 1e-6);
    try std.testing.expectEqual(@as(u32, 0), idx[layout.target_video_start]);
    try std.testing.expectEqual(@as(u32, 1), idx[layout.target_audio_start]);
    try std.testing.expectEqual(@as(u32, 2), idx[layout.video_indices[0]]);

    const audios = [_]packing.ConditionAudio{.{ .latent_t = 2 }};
    const av_refs = [_]packing.ReferenceBlock{.{
        .kind = .video_audio,
        .video_index = 0,
        .audio_index = 0,
    }};
    const av = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.0,
        .audio_t_noise = 0.0,
        .condition_videos = &videos,
        .condition_audios = &audios,
        .references = &av_refs,
    });
    defer av.deinit(allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), av.timesteps[2], 1e-6);
    try std.testing.expectEqual(@as(u32, 2), av.timestep_indices[av.audio_indices[0]]);
}
fn testPackingFl2va(allocator: std.mem.Allocator) !void {
    const first = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
        .keyframe_index = 0,
    }};
    const first_layout = try packing.build(allocator, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.2,
        .audio_t_noise = 0.4,
        .condition_videos = &first,
    });
    defer first_layout.deinit(allocator);
    try std.testing.expect(first_layout.seqLen() > 2 + 4 + 8);
    try std.testing.expect(first_layout.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), first_layout.positions[first_layout.video_indices[0]].t, 1e-5);

    const last = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
        .keyframe_index = 1,
    }};
    const last_layout = try packing.build(allocator, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.2,
        .audio_t_noise = 0.4,
        .condition_videos = &last,
    });
    defer last_layout.deinit(allocator);
    const last_t: f32 = @floatCast(2.0 + packing.videoDuration(2) - (5.0 / 3.0));
    try std.testing.expectApproxEqAbs(last_t, last_layout.positions[last_layout.video_indices[0]].t, 1e-5);
    try std.testing.expect(last_layout.positions[last_layout.video_indices[0]].t > first_layout.positions[first_layout.video_indices[0]].t);
}
fn testPackingRef2va(allocator: std.mem.Allocator) !void {
    const videos = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
    }};
    const refs = [_]packing.ReferenceBlock{.{
        .kind = .image,
        .video_index = 0,
    }};
    const layout = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = &videos,
        .references = &refs,
    });
    defer layout.deinit(allocator);
    try std.testing.expect(layout.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
    try std.testing.expect(layout.seqLen() > 3);
    // Image ref occupies one integer rotary slot. Target T is text_len+1, not max placed T.
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), layout.positions[layout.video_indices[0]].t, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), layout.positions[layout.target_audio_start].t, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), layout.positions[layout.target_video_start].t, 1e-5);

    const audios = [_]packing.ConditionAudio{.{ .latent_t = 3 }};
    const av_refs = [_]packing.ReferenceBlock{.{
        .kind = .video_audio,
        .video_index = 0,
        .audio_index = 0,
    }};
    const av = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = &videos,
        .condition_audios = &audios,
        .references = &av_refs,
    });
    defer av.deinit(allocator);
    try std.testing.expect(av.audio_indices.len > 4);
    try std.testing.expect(av.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
}

const RefCase = struct {
    refs: []const packing.ReferenceBlock,
    videos: []const packing.ConditionVideo,
    audios: []const packing.ConditionAudio,
    kinds: []const packing.SegmentKind,
    sources: []const i32,
    target_t: f32,
};

fn expectRefCase(allocator: std.mem.Allocator, case: RefCase) !void {
    const layout = try packing.build(allocator, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = case.videos,
        .condition_audios = case.audios,
        .references = case.refs,
    });
    defer layout.deinit(allocator);
    try packing.checkConditionRows(layout, case.videos, case.audios);
    try std.testing.expectEqual(case.kinds.len + 3, layout.segments.len);
    try std.testing.expectEqual(packing.SegmentKind.text, layout.segments[0].kind);
    for (case.kinds, case.sources, 1..) |kind, source, i| {
        try std.testing.expectEqual(kind, layout.segments[i].kind);
        try std.testing.expectEqual(source, layout.segments[i].source_index);
    }
    try std.testing.expectEqual(packing.SegmentKind.target_audio, layout.segments[case.kinds.len + 1].kind);
    try std.testing.expectEqual(packing.SegmentKind.target_video, layout.segments[case.kinds.len + 2].kind);
    try std.testing.expectApproxEqAbs(case.target_t, layout.positions[layout.target_audio_start].t, 1e-5);
    try std.testing.expectApproxEqAbs(case.target_t, layout.positions[layout.target_video_start].t, 1e-5);
}

fn testRef2vaOrderPermutations(allocator: std.mem.Allocator) !void {
    const image = packing.ConditionVideo{ .latent_t = 1, .latent_h = 4, .latent_w = 4 };
    const clip = packing.ConditionVideo{ .latent_t = 2, .latent_h = 4, .latent_w = 4 };
    const tone = packing.ConditionAudio{ .latent_t = 3 };
    const short_tone = packing.ConditionAudio{ .latent_t = 3 };
    const long_tone = packing.ConditionAudio{ .latent_t = 3 };
    const video_span: f32 = @floatCast(packing.videoDuration(2));
    const t_image: f32 = 3;
    const t_video: f32 = 2 + video_span;
    const t_audio_image: f32 = 6;
    const t_audio_video: f32 = 5 + video_span;
    const t_video_audio: f32 = t_video + 3;
    const t_triple: f32 = t_audio_video + 1;

    try expectRefCase(allocator, .{
        .refs = &.{.{ .kind = .image, .video_index = 0 }},
        .videos = &.{image},
        .audios = &.{},
        .kinds = &.{.condition_video},
        .sources = &.{0},
        .target_t = t_image,
    });
    try expectRefCase(allocator, .{
        .refs = &.{.{ .kind = .video, .video_index = 0 }},
        .videos = &.{clip},
        .audios = &.{},
        .kinds = &.{.condition_video},
        .sources = &.{0},
        .target_t = t_video,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .image, .video_index = 0 },
        },
        .videos = &.{image},
        .audios = &.{tone},
        .kinds = &.{ .condition_audio, .condition_video },
        .sources = &.{ 0, 0 },
        .target_t = t_audio_image,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .image, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
        },
        .videos = &.{image},
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_audio },
        .sources = &.{ 0, 0 },
        .target_t = t_audio_image,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video, .video_index = 0 },
        },
        .videos = &.{clip},
        .audios = &.{tone},
        .kinds = &.{ .condition_audio, .condition_video },
        .sources = &.{ 0, 0 },
        .target_t = t_audio_video,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .video, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
        },
        .videos = &.{clip},
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_audio },
        .sources = &.{ 0, 0 },
        .target_t = t_video_audio,
    });
    try expectRefCase(allocator, .{
        .refs = &.{.{ .kind = .video_audio, .video_index = 0, .audio_index = 0 }},
        .videos = &.{clip},
        .audios = &.{tone},
        .kinds = &.{ .condition_audio, .condition_video },
        .sources = &.{ 0, 0 },
        .target_t = t_video,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .image, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video, .video_index = 1 },
        },
        .videos = &.{ image, clip },
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_audio, .condition_video },
        .sources = &.{ 0, 0, 1 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .image, .video_index = 0 },
            .{ .kind = .video, .video_index = 1 },
        },
        .videos = &.{ image, clip },
        .audios = &.{tone},
        .kinds = &.{ .condition_audio, .condition_video, .condition_video },
        .sources = &.{ 0, 0, 1 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .video, .video_index = 0 },
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .image, .video_index = 1 },
        },
        .videos = &.{ clip, image },
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_audio, .condition_video },
        .sources = &.{ 0, 0, 1 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .image, .video_index = 0 },
            .{ .kind = .video, .video_index = 1 },
            .{ .kind = .audio, .audio_index = 0 },
        },
        .videos = &.{ image, clip },
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_video, .condition_audio },
        .sources = &.{ 0, 1, 0 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video, .video_index = 0 },
            .{ .kind = .image, .video_index = 1 },
        },
        .videos = &.{ clip, image },
        .audios = &.{tone},
        .kinds = &.{ .condition_audio, .condition_video, .condition_video },
        .sources = &.{ 0, 0, 1 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .video, .video_index = 0 },
            .{ .kind = .image, .video_index = 1 },
            .{ .kind = .audio, .audio_index = 0 },
        },
        .videos = &.{ clip, image },
        .audios = &.{tone},
        .kinds = &.{ .condition_video, .condition_video, .condition_audio },
        .sources = &.{ 0, 1, 0 },
        .target_t = t_triple,
    });
    try expectRefCase(allocator, .{
        .refs = &.{
            .{ .kind = .audio, .audio_index = 0 },
            .{ .kind = .video_audio, .video_index = 0, .audio_index = 1 },
            .{ .kind = .image, .video_index = 1 },
        },
        .videos = &.{ clip, image },
        .audios = &.{ short_tone, long_tone },
        .kinds = &.{ .condition_audio, .condition_audio, .condition_video, .condition_video },
        .sources = &.{ 0, 1, 0, 1 },
        .target_t = t_triple,
    });
}

fn testAdalnResidualHost() !void {
    const x: f32 = 2.0;
    const shift_msa: f32 = -0.5;
    const scale_msa: f32 = 0.25;
    const gate_msa: f32 = 0.5;
    const attn: f32 = 4.0;
    const attn_in = x * (1.0 + scale_msa) + shift_msa;
    const after_attn = x + gate_msa * attn;
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), attn_in, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), after_attn, 1e-6);

    const shift_mlp: f32 = 1.0;
    const scale_mlp: f32 = -0.5;
    const gate_mlp: f32 = 2.0;
    const mlp: f32 = 0.75;
    const mlp_in = after_attn * (1.0 + scale_mlp) + shift_mlp;
    const after_mlp = after_attn + gate_mlp * mlp;
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), mlp_in, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), after_mlp, 1e-6);

    const value: f32 = 3.0;
    const gate: f32 = 1.0;
    const silu = gate / (1.0 + @exp(-gate));
    try std.testing.expectApproxEqAbs(silu * value, 3.0 * silu, 1e-6);
}
fn testNchwToThwc() !void {
    const src = [_]f32{ 0, 1, 2, 3, 10, 11, 12, 13 };
    var dst: [8]f32 = undefined;
    packing.nchwToThwc(&dst, &src, 2, 2, 1, 2);
    try std.testing.expectEqualSlices(f32, &.{ 0, 10, 1, 11, 2, 12, 3, 13 }, &dst);
}
fn testMmRopeHost() !void {
    const cfg = config.Config.official();
    const theta: f32 = cfg.rope_theta;
    const freq: f32 = @floatFromInt(cfg.rope_freq_dim);
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, i| {
        f.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) / freq);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), inv[0], 1e-6);
    try std.testing.expect(inv[15] < inv[0]);
    try std.testing.expectEqual(@as(i64, 96), cfg.rotaryDim());
    try std.testing.expect(cfg.rotaryDim() < cfg.attention_head_dim);
}
fn testOfficialSpatialGrid() !void {
    var buf: [8]f32 = undefined;
    const axis = packing.spatialAxis(8, 8, &buf);
    try std.testing.expectEqual(@as(usize, 4), axis.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), axis[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), axis[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), axis[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), axis[3], 1e-6);
}
fn testTorchNoise(allocator: std.mem.Allocator) !void {
    var gen_v = noise.Generator.init(1);
    const video = try noise.drawVideo(allocator, &gen_v, &.{}, &.{}, 2, 4, 4, .{ 1, 2, 2 });
    defer allocator.free(video);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * 96), video.len);

    const audio = try noise.drawAudio(allocator, &gen_v, &.{}, 32, 3);
    defer allocator.free(audio);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 32), audio.len);

    var gen_c = noise.Generator.init(7);
    const conds = [_]packing.ConditionVideo{.{ .latent_t = 1, .latent_h = 2, .latent_w = 2 }};
    const clean = [_]f32{0} ** 96;
    const mixed = try noise.drawVideo(allocator, &gen_c, &conds, &clean, 2, 4, 4, .{ 1, 2, 2 });
    defer allocator.free(mixed);
    try std.testing.expectEqual(@as(usize, 96 + 2 * 2 * 2 * 96), mixed.len);
}
fn testMultistepSampler() !void {
    var x = [_]f32{1.0};
    const v = [_]f32{1.0};
    const sig = [_]f32{ 1.0, 0.5, 0.0 };
    const ts = [_]f32{ 0.0, 0.5 };
    scheduler.eulerStep(&sig, &ts, 0, &x, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), x[0], 1e-6);
    x[0] = 1.0;
    scheduler.eulerStep(&sig, &ts, 1, &x, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), x[0], 1e-6);
}
fn testAdalnIndexLayout(allocator: std.mem.Allocator) !void {
    const layout = try packing.build(allocator, .{
        .text_len = 4,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 3,
        .video_t = 0.25,
        .audio_t_noise = 0.6,
    });
    defer layout.deinit(allocator);
    const idx = try allocator.alloc(u32, layout.seqLen());
    defer allocator.free(idx);
    packing.writeAdalnIndices(idx, layout.timestep_indices, layout.token_tags);
    try std.testing.expectEqual(layout.seqLen(), idx.len);
    try std.testing.expectEqual(layout.adalnIndex(0), idx[0]);
    try std.testing.expectEqual(@as(u32, 1), idx[0] % 3);
    try std.testing.expectEqual(@as(u32, 2), idx[layout.target_audio_start] % 3);
    try std.testing.expectEqual(@as(u32, 0), idx[layout.target_video_start] % 3);

    const slots = packing.timestep_slot_count;
    const steps: u32 = 4;
    const hidden: i64 = 16;
    const bytes = policy_mod.adalnTableBytes(steps, hidden, 2, 2);
    const per_block = @as(u64, steps) * slots * 3 * 6 * 16 * 2;
    const final = @as(u64, steps) * slots * 2 * 16 * 2;
    try std.testing.expectEqual(per_block * 2 + final, bytes);
}
fn testSchedulerFormula(allocator: std.mem.Allocator) !void {
    const sched = try scheduler.Schedule.init(allocator, 12.0, 8);
    defer sched.deinit(allocator);
    const want = [_]f32{ 1.0, 0.98630137, 0.96774194, 0.94117647, 0.9, 0.82758621, 0.66666667, 0.0 };
    try std.testing.expectEqual(want.len, sched.sigmas.len);
    try std.testing.expectEqual(want.len - 1, sched.timesteps.len);
    for (want, sched.sigmas) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sched.timesteps[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), sched.timesteps[6], 1e-6);

    const audio = try scheduler.Schedule.init(allocator, 3.0, 8);
    defer audio.deinit(allocator);
    const want_a = [_]f32{ 1.0, 0.94736842, 0.88235294, 0.8, 0.69230769, 0.54545455, 1.0 / 3.0, 0.0 };
    try std.testing.expectEqual(want_a.len, audio.sigmas.len);
    for (want_a, audio.sigmas) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    var x = [_]f32{1.0};
    const v = [_]f32{1.0};
    const sig = [_]f32{ 1.0, 0.5, 0.0 };
    const ts = [_]f32{ 0.0, 0.5 };
    scheduler.eulerStep(&sig, &ts, 0, &x, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), x[0], 1e-6);
}
