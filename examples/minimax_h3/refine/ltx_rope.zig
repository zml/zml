const std = @import("std");

// =============================================================================
// refine/ltx_rope.zig — LTX video RoPE tables
// =============================================================================

pub const theta: f32 = 10_000.0;
pub const max_pos = [_]f32{ 20, 2048, 2048 };
pub const vae_scale = [_]f32{ 8, 32, 32 };
pub const inner_dim: u32 = 4096;
pub const heads: u32 = 32;
pub const head_dim: u32 = 128;

fn logBase(x: f64, base: f64) f64 {
    return @log(x) / @log(base);
}

/// Official Super Accel video RoPE pixel midpoint.
/// `SymmetricPatchifier(1, start_end=True)` + `latent_to_pixel_coords(causal_fix)`
/// + `use_middle_indices_grid` + time `/ fps`.
pub fn videoPixelMid(tt: u32, hh: u32, ww: u32, fps: f32) [3]f64 {
    const t_scale: f64 = vae_scale[0];
    const h_scale: f64 = vae_scale[1];
    const w_scale: f64 = vae_scale[2];
    const t_start = @max(@as(f64, @floatFromInt(tt)) * t_scale + 1.0 - t_scale, 0.0);
    const t_end = @max(@as(f64, @floatFromInt(tt + 1)) * t_scale + 1.0 - t_scale, 0.0);
    const h_start = @as(f64, @floatFromInt(hh)) * h_scale;
    const h_end = @as(f64, @floatFromInt(hh + 1)) * h_scale;
    const w_start = @as(f64, @floatFromInt(ww)) * w_scale;
    const w_end = @as(f64, @floatFromInt(ww + 1)) * w_scale;
    return .{
        (t_start + t_end) * 0.5 / @as(f64, fps),
        (h_start + h_end) * 0.5,
        (w_start + w_end) * 0.5,
    };
}

/// Official `generate_freq_grid_np` + `generate_freqs` + split-RoPE pad.
/// Writes per-head cos/sin of length `head_dim` (64 freqs duplicated) so
/// `zml.nn.applyRotary` matches `apply_rope_split_half`.
pub fn fillVideo(out_cos: []f32, out_sin: []f32, latent_t: u32, latent_h: u32, latent_w: u32, fps: f32) void {
    const tokens = latent_t * latent_h * latent_w;
    std.debug.assert(out_cos.len == tokens * heads * head_dim);
    std.debug.assert(out_sin.len == tokens * heads * head_dim);

    const n_pos: u32 = 3;
    const n_elem = 2 * n_pos;
    const n_freq = inner_dim / n_elem;
    var indices: [682]f64 = undefined;
    std.debug.assert(n_freq == 682);
    const th: f64 = theta;
    var k: u32 = 0;
    while (k < n_freq) : (k += 1) {
        const t = if (n_freq == 1) 0 else @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n_freq - 1));
        const expo = logBase(1.0, th) * (1.0 - t) + logBase(th, th) * t;
        indices[k] = std.math.pow(f64, th, expo) * (std.math.pi / 2.0);
    }

    const expected: u32 = inner_dim / 2;
    const raw: u32 = n_freq * n_pos;
    const pad = expected - raw;

    var tok: u32 = 0;
    var tt: u32 = 0;
    while (tt < latent_t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < latent_h) : (hh += 1) {
            var ww: u32 = 0;
            while (ww < latent_w) : (ww += 1) {
                const pix = videoPixelMid(tt, hh, ww, fps);
                const signed = [_]f64{
                    pix[0] / max_pos[0] * 2.0 - 1.0,
                    pix[1] / max_pos[1] * 2.0 - 1.0,
                    pix[2] / max_pos[2] * 2.0 - 1.0,
                };
                var angles: [2048]f32 = undefined;
                var i: u32 = 0;
                while (i < pad) : (i += 1) angles[i] = 0;
                var f: u32 = 0;
                while (f < n_freq) : (f += 1) {
                    var ax: u32 = 0;
                    while (ax < n_pos) : (ax += 1) {
                        angles[i] = @floatCast(indices[f] * signed[ax]);
                        i += 1;
                    }
                }
                std.debug.assert(i == expected);
                var h: u32 = 0;
                while (h < heads) : (h += 1) {
                    const half = head_dim / 2;
                    const base = ((tok * heads + h) * head_dim);
                    var d: u32 = 0;
                    while (d < half) : (d += 1) {
                        const ang = angles[h * half + d];
                        const c = @cos(ang);
                        const s = @sin(ang);
                        out_cos[base + d] = c;
                        out_cos[base + half + d] = c;
                        out_sin[base + d] = s;
                        out_sin[base + half + d] = s;
                    }
                }
                tok += 1;
            }
        }
    }
}

pub fn fillConnector(out_cos: []f32, out_sin: []f32, tokens: u32) void {
    const dim: u32 = 4096;
    const n_heads: u32 = 32;
    const hd: u32 = 128;
    std.debug.assert(out_cos.len == tokens * n_heads * hd);
    const n_freq = dim / 2;
    var indices: [2048]f64 = undefined;
    const th: f64 = theta;
    var k: u32 = 0;
    while (k < n_freq) : (k += 1) {
        const t = if (n_freq == 1) 0 else @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n_freq - 1));
        const expo = logBase(1.0, th) * (1.0 - t) + logBase(th, th) * t;
        indices[k] = std.math.pow(f64, th, expo) * (std.math.pi / 2.0);
    }
    const maxp: f64 = 4096;
    var tok: u32 = 0;
    while (tok < tokens) : (tok += 1) {
        const signed = @as(f64, @floatFromInt(tok)) / maxp * 2.0 - 1.0;
        var h: u32 = 0;
        while (h < n_heads) : (h += 1) {
            const half = hd / 2;
            const base = ((tok * n_heads + h) * hd);
            var d: u32 = 0;
            while (d < half) : (d += 1) {
                const ang: f32 = @floatCast(indices[h * half + d] * signed);
                const c = @cos(ang);
                const s = @sin(ang);
                out_cos[base + d] = c;
                out_cos[base + half + d] = c;
                out_sin[base + d] = s;
                out_sin[base + half + d] = s;
            }
        }
    }
}
