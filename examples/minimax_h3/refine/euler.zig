const sku = @import("../recipe/sku.zig");

// =============================================================================
// refine/euler.zig — LTX 3-step distilled sigmas
//
// 0.909375 / 0.725 / 0.421875, then terminal 0.
// =============================================================================

pub const evals: u32 = sku.ltx_refine_evals;

pub fn sigmaAt(step: u32) f32 {
    return if (step < sku.ltx_stage2_sigmas.len) sku.ltx_stage2_sigmas[step] else 0;
}

pub fn tauAt(step: u32) f32 {
    return if (step < sku.ltx_stage2_taus.len) sku.ltx_stage2_taus[step] else 0;
}

/// Comfy CONST / FLUX `noise_scaling`: `x = σ·ε + (1-σ)·latent`.
pub fn mixConst(sigma: f32, clean: f32, noisy: f32) f32 {
    return sigma * noisy + (1.0 - sigma) * clean;
}

pub fn mixConstInPlace(sigma: f32, clean: []const f32, noisy: []f32) void {
    for (noisy, clean) |*n, c| n.* = mixConst(sigma, c, n.*);
}

/// Euler step on flow-matching latents: `x := x + (sigma_next - sigma) * v`.
pub fn apply(x: anytype, v: anytype, sigma: f32, sigma_next: f32) @TypeOf(x) {
    const dt = sigma_next - sigma;
    return x.add(v.scale(dt));
}
