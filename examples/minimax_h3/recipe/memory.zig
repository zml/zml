const std = @import("std");

const zml = @import("zml");

const packing = @import("../draft/packing.zig");
const policy = @import("policy.zig");
const vae = @import("../draft/geometry.zig");

// =============================================================================
// recipe/memory.zig — per-SKU peak plan
//
// Chooses resident DiT cores, group size, and attention backend so the job fits BFC.
// =============================================================================

pub const Plan = struct {
    peak_bytes: u64,
    fixed_denoise_bytes: u64,
    resident_core_bytes: u64,
    transient_core_bytes: u64,
    denoise_peak_bytes: u64,
    encoder_peak_bytes: u64,
    audio_vae_peak_bytes: u64,
    refine_weight_bytes: u64,
    refine_peak_bytes: u64,
    attention: zml.attention.Backend,
    resident_blocks: u32,
    group_size: u32,
    safe: bool,
    reason: []const u8,
};

pub const streamed_block_bytes: u64 = 768 * 1024 * 1024;

pub const Geometry = struct {
    pixel_w: u32,
    pixel_h: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    video_tokens: u32,
    audio_tokens: u32,
    video_patch_dim: u32,
    audio_dim: u32,

    pub fn init(geo: anytype) Geometry {
        return .{
            .pixel_w = geo.pixel_w,
            .pixel_h = geo.pixel_h,
            .latent_t = geo.latent_t,
            .latent_h = geo.latent_h,
            .latent_w = geo.latent_w,
            .video_tokens = geo.video_tokens,
            .audio_tokens = geo.audio_tokens,
            .video_patch_dim = geo.video_patch_dim,
            .audio_dim = geo.audio_dim,
        };
    }
};

pub const Opts = struct {
    geo: Geometry,
    layout: packing.Layout,
    hidden: i64,
    steps: u32,
    device_bytes: u64,
    tp: u32,
    heads: i64 = 56,
    head_dim: i64 = 128,
    layers: u32 = 50,
    dtype: zml.DataType = .bf16,
    target: zml.Target = .cpu,
    block_core_bytes: u64 = 0,
    flash: zml.attention.Backend = .cuda_fa2,
    fixed_denoise_weight_bytes: u64 = 0,
    encoder_weight_bytes: u64 = 0,
    audio_vae_weight_bytes: u64 = 0,
    refine_weight_bytes: u64 = 0,
};

/// Stage-2 resident bytes per GPU: TP-split transformers + one copy of replicated VAE/decoders.
pub fn refineWeightBytes(tp: u32) u64 {
    const n = @max(tp, 1);
    const ltx_bf16: u64 = 44 * 1024 * 1024 * 1024;
    const gemma_bf16: u64 = 24 * 1024 * 1024 * 1024;
    const connector_bf16: u64 = 1 * 1024 * 1024 * 1024;
    // VAE + upsampler + TAEHV stay fully replicated on every rank.
    const replicated: u64 = 3 * 1024 * 1024 * 1024;
    return (ltx_bf16 + gemma_bf16 + connector_bf16) / n +| replicated;
}

pub fn plan(opts: Opts) Plan {
    const seq: u64 = opts.layout.seqLen();
    const dtype_bytes = policy.dtypeBytes(opts.dtype);
    const seq_indices = seq * (3 * 4 + 3 * 4);
    const rope = seq * @as(u64, @intCast(opts.head_dim)) * dtype_bytes * 2;
    const latent_inputs = @as(u64, opts.geo.video_tokens) * opts.geo.video_patch_dim * 4 +
        @as(u64, opts.geo.audio_tokens) * opts.geo.audio_dim * 4;
    const fixed_runtime = seq_indices +| rope +| latent_inputs +| opts.fixed_denoise_weight_bytes;
    const encoder_activation = @as(u64, @intCast(opts.layout.text_indices.len)) *|
        @as(u64, @intCast(@max(opts.hidden, 1))) *| dtype_bytes *| 8;
    const encoder_peak = policy.allocatorPeak(opts.encoder_weight_bytes +| encoder_activation);
    const decision = policy.decide(.{
        .target = opts.target,
        .seq = seq,
        .hidden = opts.hidden,
        .heads = opts.heads,
        .head_dim = opts.head_dim,
        .layers = opts.layers,
        .steps = opts.steps,
        .dtype = opts.dtype,
        .device_bytes = opts.device_bytes,
        .tp = opts.tp,
        .block_core_bytes = if (opts.block_core_bytes == 0)
            streamed_block_bytes
        else
            opts.block_core_bytes,
        .dtype_bytes = dtype_bytes,
        .flash = opts.flash,
        .fixed_bytes = fixed_runtime +| opts.refine_weight_bytes,
    });
    const budget = decision.budget_bytes;
    const audio_activation = @as(u64, opts.geo.audio_tokens) *| vae.official_audio.hop *| 4 *| 8;
    const audio_vae_peak = policy.allocatorPeak(opts.audio_vae_weight_bytes +| audio_activation);
    const refine_peak = policy.allocatorPeak(opts.refine_weight_bytes);
    const peak = @max(encoder_peak, @max(decision.denoise_peak_bytes, @max(audio_vae_peak, refine_peak)));

    var result: Plan = .{
        .peak_bytes = peak,
        .fixed_denoise_bytes = decision.fixed_bytes,
        .resident_core_bytes = decision.resident_core_bytes,
        .transient_core_bytes = decision.transient_core_bytes,
        .denoise_peak_bytes = decision.denoise_peak_bytes,
        .encoder_peak_bytes = encoder_peak,
        .audio_vae_peak_bytes = audio_vae_peak,
        .refine_weight_bytes = opts.refine_weight_bytes,
        .refine_peak_bytes = refine_peak,
        .attention = decision.attention,
        .resident_blocks = decision.resident_blocks,
        .group_size = decision.group_size,
        .safe = true,
        .reason = "ok",
    };
    if (opts.device_bytes != 0 and decision.denoise_peak_bytes > budget) {
        result.safe = false;
        result.reason = "estimated denoising peak exceeds 85% of device memory";
        return result;
    }
    if (opts.device_bytes != 0 and encoder_peak > budget) {
        result.safe = false;
        result.reason = "estimated text-encoder peak exceeds 85% of device memory";
        return result;
    }
    if (opts.device_bytes != 0 and audio_vae_peak > budget) {
        result.safe = false;
        result.reason = "estimated audio VAE peak exceeds 85% of device memory";
        return result;
    }
    if (opts.device_bytes != 0 and refine_peak > budget) {
        result.safe = false;
        result.reason = "estimated Stage 2 weight peak exceeds 85% of device memory";
        return result;
    }
    return result;
}
