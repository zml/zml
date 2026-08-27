const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");
const packing = @import("../model/packing.zig");
const policy = @import("policy.zig");
const pipeline = @import("../runtime/pipeline.zig");
const vae = @import("../vae/geometry.zig");

pub const Plan = struct {
    activation_bytes: u64,
    streamed_block_bytes: u64,
    peak_bytes: u64,
    device_bytes: u64,
    score_bytes: u64,
    fa2_scratch_bytes: u64,
    adaln_table_bytes: u64,
    attention: zml.attention.Backend,
    resident_blocks: u32,
    group_size: u32,
    tile_batch: u32,
    safe: bool,
    reason: []const u8,
};

/// Bytes reserved for one streamed transformer block.
pub const streamed_block_bytes: u64 = 768 * 1024 * 1024;

pub const Opts = struct {
    geo: pipeline.Geometry,
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
    devices: u32 = 1,
    tile_count: u32 = 0,
    tile_act_bytes: u64 = 0,
    flash: zml.attention.Backend = .cuda_fa2,
};

pub fn plan(opts: Opts) Plan {
    const seq: u64 = opts.layout.seqLen();
    const dtype_bytes = policy.dtypeBytes(opts.dtype);
    const spec = vae.official_visual;
    const tiles = if (opts.tile_count != 0) opts.tile_count else vae.tileCount(opts.geo.pixel_h, spec.tile_px, spec.tile_overlap_px, spec.spatial) *
        vae.tileCount(opts.geo.pixel_w, spec.tile_px, spec.tile_overlap_px, spec.spatial);
    const tile_lat = vae.decodeTileLatent(spec, opts.geo.latent_h, opts.geo.latent_w);
    const tile_t = vae.decodeClipTokens(spec, opts.geo.latent_t);
    const tile_seq = @as(u64, tile_t) * tile_lat.h * tile_lat.w + 5;
    const tile_act = if (opts.tile_act_bytes != 0) opts.tile_act_bytes else tile_seq * 2048 * dtype_bytes * 8;
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
        .devices = opts.devices,
        .block_core_bytes = if (opts.block_core_bytes == 0)
            streamed_block_bytes / @max(1, opts.tp)
        else
            opts.block_core_bytes,
        .dtype_bytes = dtype_bytes,
        .tile_count = tiles,
        .tile_act_bytes = tile_act,
        .flash = opts.flash,
    });
    const host = (@as(u64, opts.geo.video_tokens) + opts.geo.audio_tokens) * 4 * 4;
    const block = if (opts.block_core_bytes == 0)
        streamed_block_bytes / @max(1, opts.tp)
    else
        opts.block_core_bytes;
    const attn_scratch = if (policy.isFlash(decision.attention)) decision.fa2_scratch_bytes else decision.score_bytes;
    const peak = decision.activation_bytes + host + block * 2 + attn_scratch + decision.adaln_table_bytes;
    const budget = if (opts.device_bytes == 0) std.math.maxInt(u64) else opts.device_bytes * policy.safety_numer / policy.safety_denom;
    const full_floor = config.full_canvas_min_device_bytes;
    const needs_full_floor = @min(opts.geo.pixel_w, opts.geo.pixel_h) > config.preview_short_side;

    var result: Plan = .{
        .activation_bytes = decision.activation_bytes,
        .streamed_block_bytes = block,
        .peak_bytes = peak,
        .device_bytes = opts.device_bytes,
        .score_bytes = decision.score_bytes,
        .fa2_scratch_bytes = decision.fa2_scratch_bytes,
        .adaln_table_bytes = decision.adaln_table_bytes,
        .attention = decision.attention,
        .resident_blocks = decision.resident_blocks,
        .group_size = decision.group_size,
        .tile_batch = decision.tile_batch,
        .safe = true,
        .reason = "ok",
    };
    if (needs_full_floor and opts.device_bytes != 0 and opts.device_bytes < full_floor) {
        result.safe = false;
        result.reason = "requested size needs a measured 40 GiB-class device";
        return result;
    }
    if (opts.device_bytes != 0 and peak > budget) {
        result.safe = false;
        result.reason = "estimated peak exceeds 85% of device memory";
        return result;
    }
    return result;
}
