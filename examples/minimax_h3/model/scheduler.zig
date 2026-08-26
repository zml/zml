const std = @import("std");

/// Rectified-flow Euler (`eta = 0`). Transformer predicts data-ward velocity:
/// `x0 = x_t + sigma * v`, `t = 1 - sigma` in `[0, 1]`.
pub const Schedule = struct {
    shift: f32,
    sigmas: []f32,
    timesteps: []f32,

    pub fn init(allocator: std.mem.Allocator, shift: f32, num_inference_steps: u32) !Schedule {
        if (shift <= 0) return error.InvalidShift;
        if (num_inference_steps < 2) return error.TooFewSteps;

        const raw = try allocator.alloc(f32, num_inference_steps);
        defer allocator.free(raw);
        for (raw, 0..) |*sigma, i| {
            const base = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(num_inference_steps - 1));
            sigma.* = shiftSigma(base, shift);
        }

        var unique = try std.ArrayList(f32).initCapacity(allocator, raw.len);
        errdefer unique.deinit(allocator);
        unique.appendAssumeCapacity(raw[0]);
        for (raw[1..]) |sigma| {
            if (sigma != unique.items[unique.items.len - 1]) {
                unique.appendAssumeCapacity(sigma);
            }
        }
        if (unique.items[unique.items.len - 1] != 0.0) {
            try unique.append(allocator, 0.0);
        }
        if (unique.items.len < 2) return error.DegenerateSchedule;

        const sigmas = try unique.toOwnedSlice(allocator);
        errdefer allocator.free(sigmas);

        const timesteps = try allocator.alloc(f32, sigmas.len - 1);
        for (timesteps, sigmas[0 .. sigmas.len - 1]) |*t, sigma| {
            t.* = 1.0 - sigma;
        }

        return .{
            .shift = shift,
            .sigmas = sigmas,
            .timesteps = timesteps,
        };
    }

    pub fn deinit(self: Schedule, allocator: std.mem.Allocator) void {
        allocator.free(self.sigmas);
        allocator.free(self.timesteps);
    }

    pub fn stepCount(self: Schedule) usize {
        return self.timesteps.len;
    }

    pub fn scaleNoise(t: f32, clean: f32, noisy: f32) f32 {
        return t * clean + (1.0 - t) * noisy;
    }
};

pub const DualSchedule = struct {
    video: Schedule,
    audio: Schedule,

    pub fn init(allocator: std.mem.Allocator, steps: u32, video_shift: f32, audio_shift: f32) !DualSchedule {
        const video = try Schedule.init(allocator, video_shift, steps);
        errdefer video.deinit(allocator);
        const audio = try Schedule.init(allocator, audio_shift, steps);
        return .{ .video = video, .audio = audio };
    }

    pub fn deinit(self: DualSchedule, allocator: std.mem.Allocator) void {
        self.video.deinit(allocator);
        self.audio.deinit(allocator);
    }
};

pub fn shiftSigma(sigma: f32, shift: f32) f32 {
    return shift * sigma / (1.0 + (shift - 1.0) * sigma);
}

/// Maps a video-schedule sigma onto the audio schedule (`from_shift` → `to_shift`).
pub fn timeShiftSigma(sigma: f32, from_shift: f32, to_shift: f32) f32 {
    const base = sigma / (from_shift + sigma * (1.0 - from_shift));
    return to_shift * base / (1.0 + (to_shift - 1.0) * base);
}

pub fn timestepEmbedding(timesteps: []const f32, dim: usize, flip_sin_to_cos: bool, out: []f32) void {
    std.debug.assert(out.len == timesteps.len * dim);
    std.debug.assert(dim % 2 == 0);
    const half = dim / 2;
    for (timesteps, 0..) |t, row| {
        const dst = out[row * dim ..][0..dim];
        for (0..half) |i| {
            const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
            const angle = t * freq;
            if (flip_sin_to_cos) {
                dst[i] = @cos(angle);
                dst[half + i] = @sin(angle);
            } else {
                dst[i] = @sin(angle);
                dst[half + i] = @cos(angle);
            }
        }
    }
}
