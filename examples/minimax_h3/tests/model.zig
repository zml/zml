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
    try testNchwToThwc();
    try testMmRopeHost();
    try testOfficialSpatialGrid();
    try testOfficialRotateHalf();
    try testTorchNoise(allocator);
    try testMultistepSampler();
    try testRngReset();
    try testOfficialEuler();
    try testAdalnIndexLayout(allocator);
    try testSchedulerFormula(allocator);
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

    // Official 1344×768 canvas: latent 48×84, width axis after f32 cast of f64 linspace.
    var wide: [42]f32 = undefined;
    const w_axis = packing.spatialAxis(84, @sqrt(@as(f64, 48 * 84)), &wide);
    try std.testing.expectEqual(@as(usize, 42), w_axis.len);
    try std.testing.expectApproxEqAbs(@as(f32, -5.16601038), w_axis[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.15810537), w_axis[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.15019989), w_axis[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.14229465), w_axis[3], 1e-6);
}
fn testOfficialRotateHalf() !void {
    const x = [_]f32{ 1, 2, 3, 4 };
    const rotated = [_]f32{ -3, -4, 1, 2 };
    var out: [4]f32 = undefined;
    const half = x.len / 2;
    for (0..half) |i| {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
    try std.testing.expectEqualSlices(f32, &rotated, &out);
}
fn testTorchNoise(allocator: std.mem.Allocator) !void {
    var gen = noise.Generator.init(1);
    const want_u = [_]f32{ 0.7576315999031067, 0.2793108820915222, 0.40306925773620605, 0.7346844673156738, 0.029281556606292725, 0.7998586297035217, 0.3971373438835144, 0.7543719410896301 };
    for (want_u) |w| try std.testing.expectApproxEqAbs(w, gen.uniform01(), 1e-7);

    var gen_n = noise.Generator.init(1);
    var n16: [16]f32 = undefined;
    noise.randn(&gen_n, &n16);
    const want_n = [_]f32{ -1.5255959033966064, -0.7502318024635315, -0.6539809107780457, -1.6094847917556763, -0.1001671776175499, -0.6091889142990112, -0.9797722697257996, -1.6090962886810303 };
    for (want_n, n16[0..8]) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-5);

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
fn testRngReset() !void {
    var a = noise.Generator.init(3);
    const first = a.uniform01();
    a.reset();
    try std.testing.expectEqual(first, a.uniform01());
}
fn testOfficialEuler() !void {
    var x = [_]f32{1.0};
    const v = [_]f32{2.0};
    const sig = [_]f32{ 1.0, 0.5, 0.25 };
    const ts = [_]f32{ 0.0, 0.5 };
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
