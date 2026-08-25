const std = @import("std");

const config = @import("config.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("../runtime/pipeline.zig");

pub const Plan = struct {
    activation_bytes: u64,
    streamed_block_bytes: u64,
    host_latent_bytes: u64,
    peak_bytes: u64,
    device_bytes: u64,
    safe: bool,
    reason: []const u8,
};

/// Bytes reserved for one streamed transformer block.
pub const streamed_block_bytes: u64 = 768 * 1024 * 1024;
pub const safety_numer: u64 = 85;
pub const safety_denom: u64 = 100;

pub fn plan(
    geo: pipeline.Geometry,
    layout: packing.Layout,
    hidden: i64,
    steps: u32,
    device_bytes: u64,
    tp: u32,
) Plan {
    _ = steps;
    const seq: u64 = layout.seqLen();
    const hid: u64 = @intCast(hidden);
    const act = seq * hid * 2 * 8;
    const host = (@as(u64, geo.video_tokens) + geo.audio_tokens) * 4 * 4;
    const block = streamed_block_bytes / @max(1, tp);
    const peak = act + host + block * 2;
    const budget = if (device_bytes == 0) std.math.maxInt(u64) else device_bytes * safety_numer / safety_denom;
    const full_floor = config.full_canvas_min_device_bytes;
    const needs_full_floor = @min(geo.pixel_w, geo.pixel_h) > config.preview_short_side;
    if (needs_full_floor and device_bytes != 0 and device_bytes < full_floor) {
        return .{
            .activation_bytes = act,
            .streamed_block_bytes = block,
            .host_latent_bytes = host,
            .peak_bytes = peak,
            .device_bytes = device_bytes,
            .safe = false,
            .reason = "canvas above preview needs a measured 40 GiB-class device",
        };
    }
    if (device_bytes != 0 and peak > budget) {
        return .{
            .activation_bytes = act,
            .streamed_block_bytes = block,
            .host_latent_bytes = host,
            .peak_bytes = peak,
            .device_bytes = device_bytes,
            .safe = false,
            .reason = "estimated peak exceeds 85% of device memory",
        };
    }
    return .{
        .activation_bytes = act,
        .streamed_block_bytes = block,
        .host_latent_bytes = host,
        .peak_bytes = peak,
        .device_bytes = device_bytes,
        .safe = true,
        .reason = "ok",
    };
}
