const std = @import("std");

const scheduler = @import("../model/scheduler.zig");

/// Second-order multistep on one flow stream. Data-ward velocity:
/// `x_next = x + (sigma - sigma_next) * (1.5 v - 0.5 v_prev)` after step 0.
pub fn resMultistep(
    sigmas: []const f32,
    step_index: usize,
    sample: []f32,
    velocity: []const f32,
    prev_velocity: ?[]const f32,
) void {
    std.debug.assert(step_index + 1 < sigmas.len);
    std.debug.assert(sample.len == velocity.len);
    const sigma = sigmas[step_index];
    const sigma_next = sigmas[step_index + 1];
    const dt = sigma - sigma_next;
    if (step_index == 0 or prev_velocity == null or sigma_next == 0) {
        for (sample, velocity) |*x, v| x.* += dt * v;
        return;
    }
    const prev = prev_velocity.?;
    std.debug.assert(prev.len == sample.len);
    for (sample, velocity, prev) |*x, v, pv| {
        x.* += dt * (1.5 * v - 0.5 * pv);
    }
}

pub const State = struct {
    prev_video: ?[]f32 = null,
    prev_audio: ?[]f32 = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) State {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *State) void {
        if (self.prev_video) |p| self.allocator.free(p);
        if (self.prev_audio) |p| self.allocator.free(p);
        self.prev_video = null;
        self.prev_audio = null;
    }

    pub fn remember(self: *State, video: []const f32, audio: []const f32) !void {
        if (self.prev_video) |p| self.allocator.free(p);
        if (self.prev_audio) |p| self.allocator.free(p);
        self.prev_video = try self.allocator.dupe(f32, video);
        self.prev_audio = try self.allocator.dupe(f32, audio);
    }
};

pub fn dualResMultistep(
    schedules: scheduler.DualSchedule,
    step_index: usize,
    video: []f32,
    audio: []f32,
    video_vel: []const f32,
    audio_vel: []const f32,
    state: *State,
) !void {
    resMultistep(schedules.video.sigmas, step_index, video, video_vel, state.prev_video);
    resMultistep(schedules.audio.sigmas, step_index, audio, audio_vel, state.prev_audio);
    try state.remember(video_vel, audio_vel);
}
