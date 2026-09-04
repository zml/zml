const std = @import("std");

const zml = @import("zml");

// =============================================================================
// draft/scheduler.zig — FlowMatch Euler schedule
//
// Turbo 4-step uses video shift 12 / audio shift 3.
// =============================================================================

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

pub const StepInput = struct {
    sample: zml.Tensor,
    velocity: zml.Tensor,
    sigma: zml.Tensor,
    sigma_next: zml.Tensor,
    sigma_t: zml.Tensor,
};

pub const StepOutput = struct {
    sample: zml.Tensor,
};

pub fn apply(input: StepInput) StepOutput {
    const sample = input.sample;
    const vel = input.velocity.convert(sample.dtype());
    const sigma_t = input.sigma_t.convert(sample.dtype()).broad(sample.shape());
    const denoised = sample.add(vel.mul(sigma_t));
    const sample_f = sample.convert(.f32);
    const denoised_f = denoised.convert(.f32);
    const ratio = input.sigma_next.convert(.f32).div(input.sigma.convert(.f32)).broad(sample_f.shape());
    const next = ratio.mul(sample_f).add(zml.Tensor.scalar(1.0, .f32).sub(ratio).mul(denoised_f));
    return .{ .sample = next.convert(sample.dtype()).reuseBuffer(input.sample) };
}
