const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");
const packing = @import("../model/packing.zig");
const policy = @import("policy.zig");
const vae = @import("../vae/geometry.zig");

pub const Plan = struct {
    peak_bytes: u64,
    score_bytes: u64,
    fa2_scratch_bytes: u64,
    adaln_table_bytes: u64,
    fixed_denoise_bytes: u64,
    resident_core_bytes: u64,
    transient_core_bytes: u64,
    denoise_peak_bytes: u64,
    encoder_peak_bytes: u64,
    vae_cache_bytes: u64,
    vae_peak_bytes: u64,
    audio_vae_peak_bytes: u64,
    attention: zml.attention.Backend,
    resident_blocks: u32,
    group_size: u32,
    tile_batch: u32,
    prefetch_vae: bool,
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
    vae_cache_bytes: u64 = 0,
    audio_vae_weight_bytes: u64 = 0,
};

pub fn plan(opts: Opts) Plan {
    const seq: u64 = opts.layout.seqLen();
    const dtype_bytes = policy.dtypeBytes(opts.dtype);
    const spec = vae.official_visual;
    const tiles = vae.tileCount(opts.geo.pixel_h, spec.tile_px, spec.tile_overlap_px, spec.spatial) *
        vae.tileCount(opts.geo.pixel_w, spec.tile_px, spec.tile_overlap_px, spec.spatial);
    const tile_lat = vae.decodeTileLatent(spec, opts.geo.latent_h, opts.geo.latent_w);
    const tile_t = vae.decodeClipTokens(spec, opts.geo.latent_t);
    const tile_seq = @as(u64, tile_t) * tile_lat.h * tile_lat.w + 5;
    const tile_act = tile_seq * 2048 * dtype_bytes * 8;
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
        .fixed_bytes = fixed_runtime,
    });
    const budget = decision.budget_bytes;
    const vae_base = opts.vae_cache_bytes;
    const vae_headroom = if (opts.device_bytes == 0)
        @as(u64, 0)
    else
        budget -| policy.allocatorPeak(vae_base);
    const tile_batch = if (opts.device_bytes == 0)
        @as(u32, 1)
    else
        policy.tileBatch(tiles, tile_act, vae_headroom, @max(1, opts.tp));
    const vae_peak = policy.allocatorPeak(vae_base +| @as(u64, tile_batch) *| tile_act);
    const audio_activation = @as(u64, opts.geo.audio_tokens) *| vae.official_audio.hop *| 4 *| 8;
    const audio_vae_peak = policy.allocatorPeak(vae_base +| opts.audio_vae_weight_bytes +| audio_activation);
    const prefetch_vae = policy.canPrefetchVae(opts.device_bytes, budget, decision.denoise_live_bytes, vae_base);
    const denoise_with_prefetch = if (prefetch_vae)
        policy.allocatorPeak(decision.denoise_live_bytes +| vae_base)
    else
        decision.denoise_peak_bytes;
    const peak = @max(encoder_peak, @max(denoise_with_prefetch, @max(vae_peak, audio_vae_peak)));
    const full_floor = config.full_canvas_min_device_bytes;
    const needs_full_floor = @min(opts.geo.pixel_w, opts.geo.pixel_h) > config.preview_short_side;

    var result: Plan = .{
        .peak_bytes = peak,
        .score_bytes = decision.score_bytes,
        .fa2_scratch_bytes = decision.fa2_scratch_bytes,
        .adaln_table_bytes = decision.adaln_table_bytes,
        .fixed_denoise_bytes = decision.fixed_bytes,
        .resident_core_bytes = decision.resident_core_bytes,
        .transient_core_bytes = decision.transient_core_bytes,
        .denoise_peak_bytes = decision.denoise_peak_bytes,
        .encoder_peak_bytes = encoder_peak,
        .vae_cache_bytes = vae_base,
        .vae_peak_bytes = vae_peak,
        .audio_vae_peak_bytes = audio_vae_peak,
        .attention = decision.attention,
        .resident_blocks = decision.resident_blocks,
        .group_size = decision.group_size,
        .tile_batch = tile_batch,
        .prefetch_vae = prefetch_vae,
        .safe = true,
        .reason = "ok",
    };
    if (needs_full_floor and opts.device_bytes != 0 and opts.device_bytes < full_floor) {
        result.safe = false;
        result.reason = "requested size is below the full-canvas device-memory floor";
        return result;
    }
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
    if (opts.device_bytes != 0 and vae_peak > budget) {
        result.safe = false;
        result.reason = "estimated visual VAE peak exceeds 85% of device memory";
        return result;
    }
    if (opts.device_bytes != 0 and audio_vae_peak > budget) {
        result.safe = false;
        result.reason = "estimated audio VAE peak exceeds 85% of device memory";
        return result;
    }
    return result;
}
