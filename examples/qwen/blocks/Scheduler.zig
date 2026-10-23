const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;

pub const FlowMatchEulerScheduler = @This();

num_train_steps: f32,
num_train_timesteps: f32 = 1000.0,
shift: f32 = 1.0,
shift_terminal: f32 = 0.02,
use_dynamic_shifting: bool = true,
base_seq_len: f32 = 256.0,
max_seq_len: f32 = 8192.0,
base_shift: f32 = 0.5,
max_shift: f32 = 0.9,

sigmas: []f32 = undefined,
timesteps: []f32 = undefined,

pub fn deinit(self: *FlowMatchEulerScheduler, allocator: std.mem.Allocator) void {
    allocator.free(self.timesteps);
    allocator.free(self.sigmas);
}

pub fn init_timesteps(self: *FlowMatchEulerScheduler, allocator: std.mem.Allocator, steps: usize, latent_size: f32) ![]f32 {
    const mu = self.calculate_shift(latent_size);

    var sigmas = try allocator.alloc(f32, steps + 1);
    for (0..steps) |i| {
        sigmas[i] = 1.0 - 1.0 / @as(f32, @floatFromInt(steps)) * @as(f32, @floatFromInt(i));
    }
    sigmas = time_shift(mu, sigmas);
    sigmas = stretch_shift_to_terminal(self.shift_terminal, sigmas);
    sigmas[steps] = 0;

    const timesteps = try allocator.alloc(f32, steps);
    for (0..steps) |i| {
        timesteps[i] = sigmas[i] * self.num_train_timesteps;
    }

    self.sigmas = sigmas;
    self.timesteps = timesteps;
    return self.timesteps;
}

/// Euler step. `sigmas` is the {.step = steps + 1} f32 tensor from `sigmasTensor` and `step_index` a runtime
/// scalar, so the step can be traced once inside a `zml.ops.while`.
pub fn step(sigmas: zml.Tensor, step_index: zml.Tensor, model_output: zml.Tensor, sample_: zml.Tensor) zml.Tensor {
    const sample = sample_.convert(.f32);

    const sigma = sigmas.slice(.step, .dynSingle(step_index));
    const sigma_next = sigmas.slice(.step, .dynSingle(step_index.addConstant(1)));
    const delta = sigma_next.sub(sigma);

    const prev_sample = sample.add(delta.broad(model_output.shape()).mul(model_output.convert(.f32))).convert(model_output.dtype());

    return prev_sample;
}

/// Embeds the sigmas computed by `init_timesteps` in the graph, as {.step = steps + 1} f32.
pub fn sigmasTensor(self: FlowMatchEulerScheduler) zml.Tensor {
    return .constantTensor(.init(.{ .step = self.sigmas.len }, .f32), std.mem.sliceAsBytes(self.sigmas));
}

fn calculate_shift(self: FlowMatchEulerScheduler, image_seq_len: f32) f32 {
    if (!self.use_dynamic_shifting) {
        return self.shift;
    }

    const m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len);
    const b = self.base_shift - m * self.base_seq_len;
    return m * image_seq_len + b;
}

// Overwrites sigmas to reuse buffers
// Uses self.config.time_shift_type == "exponential"
fn time_shift(mu: f32, sigmas: []f32) []f32 {
    const e_mu = std.math.exp(mu);
    for (sigmas) |*s| {
        s.* = e_mu / (e_mu + (1 / s.* - 1));
    }
    return sigmas;
}

// Overwrites sigmas to reuse buffers
fn stretch_shift_to_terminal(shift_terminal: f32, sigmas: []f32) []f32 {
    const scale_factor = (1 - sigmas[sigmas.len - 1]) / (1 - shift_terminal);
    for (sigmas) |*s| {
        s.* = 1 - ((1 - s.*) / scale_factor);
    }
    return sigmas;
}
