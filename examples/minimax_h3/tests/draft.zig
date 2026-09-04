const std = @import("std");

const policy_mod = @import("../recipe/policy.zig");
const config = @import("../recipe/config.zig");
const noise = @import("../draft/noise.zig");
const packing = @import("../draft/packing.zig");
const scheduler = @import("../draft/scheduler.zig");

// =============================================================================
// tests/draft.zig — packing, scheduler, noise
// =============================================================================

pub fn run(allocator: std.mem.Allocator) !void {
    try testScheduler(allocator);
    try testPackingT2va(allocator);
    try testPackingTimestepSlots(allocator);
    try testNchwToThwc();
    try testMmRopeHost();
    try testOfficialSpatialGrid();
    try testTorchNoise(allocator);
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

    const official = try scheduler.Schedule.init(allocator, config.video_shift, 30);
    defer official.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 30), official.sigmas.len);
    try std.testing.expectEqual(@as(usize, 29), official.timesteps.len);
    try std.testing.expectEqual(@as(usize, 29), official.stepCount());
    try std.testing.expectEqual(@as(f32, 0.0), official.sigmas[official.sigmas.len - 1]);
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
    try std.testing.expectEqual(packing.timestep_slot_count, @as(u32, @intCast(layout.timesteps.len)));
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[layout.target_video_start]);
    try std.testing.expectEqual(@as(u32, 1), layout.timestep_indices[layout.target_audio_start]);
}
fn testPackingTimestepSlots(allocator: std.mem.Allocator) !void {
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
    const video = try noise.drawVideo(allocator, &gen_v, 2, 4, 4, .{ 1, 2, 2 });
    defer allocator.free(video);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * 96), video.len);

    const audio = try noise.drawAudio(allocator, &gen_v, 32, 3);
    defer allocator.free(audio);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 32), audio.len);
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
}
