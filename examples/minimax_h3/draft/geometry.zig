const std = @import("std");

const config = @import("../recipe/config.zig");

// =============================================================================
// draft/geometry.zig — latent / audio layout helpers
// =============================================================================

pub const AudioSpec = struct {
    channels: u32 = 32,
    stereo: u32 = 2,
    hz: f32 = config.audio_hz,
    sample_rate: u32 = config.audio_sample_rate,
    hop: u32 = 800,

    pub fn tokenCount(self: AudioSpec, latent_t: u32) u32 {
        return latent_t * self.stereo;
    }

    pub fn sampleCount(self: AudioSpec, latent_t: u32) u32 {
        return latent_t * self.hop;
    }
};

pub const official_audio: AudioSpec = .{};

/// Packed DiT audio is channel-major stereo: `(2 * T, C)` = left then right, each `(T, C)`.
/// The mono audio VAE consumes `(2, C, T)`.
pub fn audioRowsToBct(dst: []f32, rows: []const f32, channels: u32, t: u32) void {
    std.debug.assert(dst.len == rows.len);
    std.debug.assert(rows.len == 2 * @as(usize, channels) * t);
    var ear: usize = 0;
    while (ear < 2) : (ear += 1) {
        const src = rows[ear * t * channels ..][0 .. t * channels];
        const out = dst[ear * channels * t ..][0 .. channels * t];
        var ti: usize = 0;
        while (ti < t) : (ti += 1) {
            var c: usize = 0;
            while (c < channels) : (c += 1) {
                out[c * t + ti] = src[ti * channels + c];
            }
        }
    }
}

pub fn applyLatentNorm(values: []f32, channels: u32, mean: []const f32, stddev: []const f32, decode: bool) void {
    std.debug.assert(values.len % channels == 0);
    std.debug.assert(mean.len >= channels and stddev.len >= channels);
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        const c = i % channels;
        if (decode) {
            values[i] = values[i] * stddev[c] + mean[c];
        } else {
            values[i] = (values[i] - mean[c]) / stddev[c];
        }
    }
}
