const std = @import("std");

const zml = @import("zml");

/// Official `MiniMaxH3Scheduler.step`: data-ward Euler (`eta = 0`).
/// `x0 = x_t + (1 - t) * v`, then `x_next = r*x_t + (1-r)*x0` with `r = sigma_next / sigma`.
pub fn eulerStep(
    sigmas: []const f32,
    timesteps: []const f32,
    step_index: usize,
    sample: []f32,
    velocity: []const f32,
) void {
    std.debug.assert(step_index + 1 < sigmas.len);
    std.debug.assert(step_index < timesteps.len);
    std.debug.assert(sample.len == velocity.len);
    const sigma = sigmas[step_index];
    const sigma_next = sigmas[step_index + 1];
    const sigma_t = 1.0 - timesteps[step_index];
    const ratio = sigma_next / sigma;
    for (sample, velocity) |*x, v| {
        const denoised = x.* + sigma_t * v;
        x.* = ratio * x.* + (1.0 - ratio) * denoised;
    }
}

pub const StepModel = struct {
    hold: i64,
};

pub const StepInput = struct {
    model: StepModel,
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
    var next = ratio.mul(sample_f).add(zml.Tensor.scalar(1.0, .f32).sub(ratio).mul(denoised_f));
    if (input.model.hold > 0) {
        const seq = next.dim(.s);
        const prefix = sample_f.slice1d(.s, .{ .start = 0, .end = input.model.hold });
        const rest = next.slice1d(.s, .{ .start = input.model.hold, .end = seq });
        next = zml.Tensor.concatenate(&.{ prefix, rest }, .s);
    }
    return .{ .sample = next.convert(sample.dtype()).reuseBuffer(input.sample) };
}
