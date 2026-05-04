const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const conv_ops = @import("conv_ops.zig");
const Conv3dWeight = conv_ops.Conv3dWeight;
const PerChannelStats = conv_ops.PerChannelStats;

// ============================================================================
// Video VAE Decoder
// Python ref: ltx_core/model/video_vae/decoder.py — Decoder3d
//             ltx_core/model/video_vae/ops.py — CausalConv3d, PixelNorm,
//                                                DepthToSpaceUpsample
// Architecture: 9 up_blocks (alternating ResBlock groups + DepthToSpace),
//   conv_in → up_blocks.0..8 → PixelNorm → SiLU → conv_out → unpatchify
// ============================================================================

// --- Parameter structs ---

/// Weight struct for a VAE ResnetBlock3D. Each block is:
///   PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU → CausalConv3d + residual
/// No timestep conditioning or noise injection in this checkpoint.
pub const VaeResBlock = struct {
    conv1: Conv3dWeight, // checkpoint: conv1.conv.{weight,bias}
    conv2: Conv3dWeight, // checkpoint: conv2.conv.{weight,bias}
};

/// Weight struct for a DepthToSpaceUpsample block.
/// CausalConv3d → 3D pixel-shuffle → optionally remove first frame.
pub const VaeDepthToSpaceBlock = struct {
    conv: Conv3dWeight, // checkpoint: conv.conv.{weight,bias}
};

/// Full parameter set for the video VAE decoder.
/// Checkpoint keys: vae.decoder.* (42 weight tensors).
pub const VideoVaeDecoderParams = struct {
    conv_in: Conv3dWeight,

    // up_blocks.0: 2 ResBlocks @ 1024ch
    up0_res0: VaeResBlock,
    up0_res1: VaeResBlock,

    // up_blocks.1: DepthToSpace (2,2,2) → 1024→512
    up1: VaeDepthToSpaceBlock,

    // up_blocks.2: 2 ResBlocks @ 512ch
    up2_res0: VaeResBlock,
    up2_res1: VaeResBlock,

    // up_blocks.3: DepthToSpace (2,2,2) → 512→512
    up3: VaeDepthToSpaceBlock,

    // up_blocks.4: 4 ResBlocks @ 512ch
    up4_res0: VaeResBlock,
    up4_res1: VaeResBlock,
    up4_res2: VaeResBlock,
    up4_res3: VaeResBlock,

    // up_blocks.5: DepthToSpace (2,1,1) → 512→256
    up5: VaeDepthToSpaceBlock,

    // up_blocks.6: 6 ResBlocks @ 256ch
    up6_res0: VaeResBlock,
    up6_res1: VaeResBlock,
    up6_res2: VaeResBlock,
    up6_res3: VaeResBlock,
    up6_res4: VaeResBlock,
    up6_res5: VaeResBlock,

    // up_blocks.7: DepthToSpace (1,2,2) → 256→128
    up7: VaeDepthToSpaceBlock,

    // up_blocks.8: 4 ResBlocks @ 128ch
    up8_res0: VaeResBlock,
    up8_res1: VaeResBlock,
    up8_res2: VaeResBlock,
    up8_res3: VaeResBlock,

    conv_out: Conv3dWeight,
};

// --- Weight loading ---

fn initVaeResBlock(store: zml.io.TensorStore.View) VaeResBlock {
    const c1 = store.withPrefix("conv1").withPrefix("conv");
    const c2 = store.withPrefix("conv2").withPrefix("conv");
    return .{
        .conv1 = .{
            .weight = c1.createTensor("weight", null, null),
            .bias = c1.createTensor("bias", null, null),
        },
        .conv2 = .{
            .weight = c2.createTensor("weight", null, null),
            .bias = c2.createTensor("bias", null, null),
        },
    };
}

fn initVaeConv3d(store: zml.io.TensorStore.View) Conv3dWeight {
    const c = store.withPrefix("conv");
    return .{
        .weight = c.createTensor("weight", null, null),
        .bias = c.createTensor("bias", null, null),
    };
}

/// Load VideoVaeDecoderParams from the main model checkpoint.
/// Keys: vae.decoder.{conv_in,up_blocks.*,conv_out}.conv.{weight,bias}
pub fn initVideoVaeDecoderParams(store: zml.io.TensorStore.View) VideoVaeDecoderParams {
    const dec = store.withPrefix("vae").withPrefix("decoder");
    const ub = dec.withPrefix("up_blocks");

    return .{
        .conv_in = initVaeConv3d(dec.withPrefix("conv_in")),

        // up_blocks.0: 2 ResBlocks @ 1024
        .up0_res0 = initVaeResBlock(ub.withLayer(0).withPrefix("res_blocks").withLayer(0)),
        .up0_res1 = initVaeResBlock(ub.withLayer(0).withPrefix("res_blocks").withLayer(1)),

        // up_blocks.1: DepthToSpace
        .up1 = .{ .conv = initVaeConv3d(ub.withLayer(1).withPrefix("conv")) },

        // up_blocks.2: 2 ResBlocks @ 512
        .up2_res0 = initVaeResBlock(ub.withLayer(2).withPrefix("res_blocks").withLayer(0)),
        .up2_res1 = initVaeResBlock(ub.withLayer(2).withPrefix("res_blocks").withLayer(1)),

        // up_blocks.3: DepthToSpace
        .up3 = .{ .conv = initVaeConv3d(ub.withLayer(3).withPrefix("conv")) },

        // up_blocks.4: 4 ResBlocks @ 512
        .up4_res0 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(0)),
        .up4_res1 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(1)),
        .up4_res2 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(2)),
        .up4_res3 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(3)),

        // up_blocks.5: DepthToSpace
        .up5 = .{ .conv = initVaeConv3d(ub.withLayer(5).withPrefix("conv")) },

        // up_blocks.6: 6 ResBlocks @ 256
        .up6_res0 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(0)),
        .up6_res1 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(1)),
        .up6_res2 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(2)),
        .up6_res3 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(3)),
        .up6_res4 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(4)),
        .up6_res5 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(5)),

        // up_blocks.7: DepthToSpace
        .up7 = .{ .conv = initVaeConv3d(ub.withLayer(7).withPrefix("conv")) },

        // up_blocks.8: 4 ResBlocks @ 128
        .up8_res0 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(0)),
        .up8_res1 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(1)),
        .up8_res2 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(2)),
        .up8_res3 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(3)),

        .conv_out = initVaeConv3d(dec.withPrefix("conv_out")),
    };
}

// --- Forward ops ---

/// CausalConv3d forward (causal=False mode): symmetric temporal replicate-pad + conv3d.
/// For kernel_size=3: pad 1 frame at start/end (replicate), zero-pad 1 on H,W.
/// Python CausalConv3d with causal=False uses Conv3d built-in zero-padding for spatial.
fn forwardCausalConv3dNonCausal(input: Tensor, w: Conv3dWeight) Tensor {
    // 1. Temporal replicate-padding: duplicate first and last frame
    const first_frame = input.slice1d(2, .{ .end = 1 }); // [B,C,1,H,W]
    const last_frame = input.slice1d(2, .{ .start = -1 }); // [B,C,1,H,W]
    const padded = Tensor.concatenate(&.{ first_frame, input, last_frame }, 2); // [B,C,F+2,H,W]

    // 2. Conv3d with zero-padding on H,W only (temporal already handled)
    const conv_out = padded.conv3d(w.weight, .{
        .padding = &.{ 0, 0, 1, 1, 1, 1 }, // D:0, H:1, W:1
    });

    // 3. Add bias: [C_out] → [1, C_out, 1, 1, 1]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1),
    ).broad(conv_out.shape()));
}

/// PixelNorm: x / sqrt(mean(x², dim=channels) + eps).
/// Per-location RMS normalization over the channel dimension. No learnable params.
/// Computed in f32 for precision (bf16 squaring can lose significant bits).
fn forwardPixelNorm(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    // mean over dim 1 (channels), keeping dim → [B, 1, F, H, W]
    const mean_sq = x_sq.mean(1);
    const rms = mean_sq.addConstant(1e-8).sqrt();
    return x_f32.div(rms.broad(x_f32.shape())).convert(x.dtype());
}

/// VAE ResnetBlock3D forward:
///   PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU → CausalConv3d + residual.
/// No channel change (in_ch == out_ch), so no conv_shortcut needed.
fn forwardVaeResBlock(x: Tensor, rb: VaeResBlock) Tensor {
    var h = forwardPixelNorm(x);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv1);
    h = forwardPixelNorm(h);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv2);
    return h.add(x); // residual
}

/// DepthToSpace 3D (pixel-shuffle): CausalConv3d → rearrange → optionally remove first frame.
/// stride = (p1, p2, p3): spatial/temporal upsample factors.
/// Rearrange: [B, C*p1*p2*p3, F, H, W] → [B, C, F*p1, H*p2, W*p3]
fn forwardDepthToSpace(x: Tensor, w: VaeDepthToSpaceBlock, comptime stride: [3]i64) Tensor {
    var h = forwardCausalConv3dNonCausal(x, w.conv);

    const B = h.dim(0);
    const C_total = h.dim(1);
    const F = h.dim(2);
    const H = h.dim(3);
    const W = h.dim(4);
    const p1 = stride[0];
    const p2 = stride[1];
    const p3 = stride[2];
    const C = @divExact(C_total, p1 * p2 * p3);

    // [B, C*p1*p2*p3, F, H, W] → [B, C, p1, p2, p3, F, H, W]
    h = h.reshape(.{ B, C, p1, p2, p3, F, H, W });
    // → [B, C, F, p1, H, p2, W, p3]
    h = h.transpose(.{ 0, 1, 5, 2, 6, 3, 7, 4 });
    // → [B, C, F*p1, H*p2, W*p3]
    h = h.reshape(.{ B, C, F * p1, H * p2, W * p3 });

    // Remove first frame if temporal upsample (p1 == 2)
    if (p1 == 2) {
        h = h.slice1d(2, .{ .start = 1 }); // drop frame 0
    }
    return h;
}

/// Unpatchify for VAE output: [B, 48, F, H, W] → [B, 3, F, 4H, 4W].
/// "b (c p r q) f h w -> b c (f p) (h q) (w r)" with c=3, p=1, q=4, r=4
/// Channel decomposition: 48 = c(3) * p(1) * r(4) * q(4), where q→H and r→W.
fn forwardUnpatchifyVae(x: Tensor) Tensor {
    const B = x.dim(0);
    const F = x.dim(2);
    const H = x.dim(3);
    const W = x.dim(4);
    // 48 = 3 * 1 * 4 * 4 → (c=3, p=1, r=4, q=4) in channel decomposition order
    // reshape [B, 48, F, H, W] → [B, 3, 1, 4, 4, F, H, W]
    //   dims:                      [B, c, p, r, q, F, H, W]
    var h = x.reshape(.{ B, 3, 1, 4, 4, F, H, W });
    // transpose → [B, c, F, p, H, q, W, r]  (q pairs with H, r pairs with W)
    h = h.transpose(.{ 0, 1, 5, 2, 6, 4, 7, 3 });
    // reshape → [B, 3, F*p, H*q, W*r] = [B, 3, F, 4H, 4W]
    return h.reshape(.{ B, 3, F * 1, H * 4, W * 4 });
}

/// Full video VAE decoder forward pass.
/// Input: latent [B, 128, F', H', W'] bf16
/// Output: decoded video [B, 3, 8(F'-1)+1, 32H', 32W'] bf16
pub fn forwardVideoVaeDecode(
    latent: Tensor,
    stats: PerChannelStats,
    params: VideoVaeDecoderParams,
) Tensor {
    // 1. Denormalize: x = latent * std_of_means + mean_of_means
    const stats_shape = latent.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(latent.shape());
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(latent.shape());
    var x = latent.mul(std_broad).add(mean_broad);

    // 2. conv_in: CausalConv3d(128 → 1024)
    x = forwardCausalConv3dNonCausal(x, params.conv_in);

    // 3. up_blocks.0: 2 ResBlocks @ 1024ch
    x = forwardVaeResBlock(x, params.up0_res0);
    x = forwardVaeResBlock(x, params.up0_res1);

    // 4. up_blocks.1: DepthToSpace (2,2,2) → 1024→512
    x = forwardDepthToSpace(x, params.up1, .{ 2, 2, 2 });

    // 5. up_blocks.2: 2 ResBlocks @ 512ch
    x = forwardVaeResBlock(x, params.up2_res0);
    x = forwardVaeResBlock(x, params.up2_res1);

    // 6. up_blocks.3: DepthToSpace (2,2,2) → 512→512
    x = forwardDepthToSpace(x, params.up3, .{ 2, 2, 2 });

    // 7. up_blocks.4: 4 ResBlocks @ 512ch
    x = forwardVaeResBlock(x, params.up4_res0);
    x = forwardVaeResBlock(x, params.up4_res1);
    x = forwardVaeResBlock(x, params.up4_res2);
    x = forwardVaeResBlock(x, params.up4_res3);

    // 8. up_blocks.5: DepthToSpace (2,1,1) → 512→256
    x = forwardDepthToSpace(x, params.up5, .{ 2, 1, 1 });

    // 9. up_blocks.6: 6 ResBlocks @ 256ch
    x = forwardVaeResBlock(x, params.up6_res0);
    x = forwardVaeResBlock(x, params.up6_res1);
    x = forwardVaeResBlock(x, params.up6_res2);
    x = forwardVaeResBlock(x, params.up6_res3);
    x = forwardVaeResBlock(x, params.up6_res4);
    x = forwardVaeResBlock(x, params.up6_res5);

    // 10. up_blocks.7: DepthToSpace (1,2,2) → 256→128
    x = forwardDepthToSpace(x, params.up7, .{ 1, 2, 2 });

    // 11. up_blocks.8: 4 ResBlocks @ 128ch
    x = forwardVaeResBlock(x, params.up8_res0);
    x = forwardVaeResBlock(x, params.up8_res1);
    x = forwardVaeResBlock(x, params.up8_res2);
    x = forwardVaeResBlock(x, params.up8_res3);

    // 12. PixelNorm → SiLU → conv_out
    x = forwardPixelNorm(x);
    x = x.silu();
    x = forwardCausalConv3dNonCausal(x, params.conv_out);

    // 13. Unpatchify: [B, 48, F, H, W] → [B, 3, F, 4H, 4W]
    return forwardUnpatchifyVae(x);
}

// ============================================================================
// Temporal tiling for large-frame VAE decode
// ============================================================================

/// Tiling config for temporal chunking of the VAE decoder.
/// tile_latent_frames=9 gives 65 pixel frames per tile.
/// overlap_latent_frames=3 gives 17 pixel frames of blending overlap.
pub const TemporalTilingConfig = struct {
    tile_latent_frames: i64 = 9,
    overlap_latent_frames: i64 = 3,

    pub fn stride(self: TemporalTilingConfig) i64 {
        return self.tile_latent_frames - self.overlap_latent_frames;
    }

    /// Number of pixel frames in the overlap region.
    /// Each latent frame maps to 8 pixel frames, but the formula for a
    /// contiguous range of N latent frames gives 8*(N-1)+1 pixel frames.
    pub fn overlapPixelFrames(self: TemporalTilingConfig) i64 {
        return 8 * (self.overlap_latent_frames - 1) + 1;
    }
};

/// A single temporal tile: half-open range [lat_start, lat_end) in latent frames.
pub const TemporalTile = struct {
    lat_start: i64,
    lat_end: i64, // exclusive, may exceed F' for the padded last tile
    lat_actual_end: i64, // exclusive, clamped to F' (before padding)
    px_start: i64,
    px_end: i64, // exclusive, pixel frames for the actual (non-padded) content
    is_first: bool,
    is_last: bool,
};

/// Compute the list of temporal tiles for a given number of latent frames.
/// Returns at most MAX_TILES tiles (enough for ~500 latent frames).
const MAX_TILES = 64;

pub fn computeTemporalTiles(
    f_lat: i64,
    config: TemporalTilingConfig,
) struct { tiles: [MAX_TILES]TemporalTile, count: usize } {
    const s = config.stride();
    var result: [MAX_TILES]TemporalTile = undefined;
    var count: usize = 0;

    var start: i64 = 0;
    while (start < f_lat) : (start += s) {
        const end = @min(start + config.tile_latent_frames, f_lat);
        const actual_end = end;

        // Pixel frame range for the actual content of this tile.
        // Latent frames [start..end) decode to pixel frames [px_start..px_end).
        // Formula: latent frame i → pixel frames 8*i .. 8*i+1 (for a single frame).
        // A contiguous range [a..b) of latent frames decodes to pixel [8*a .. 8*(b-1)+1).
        // But the first latent frame of the whole sequence starts at pixel 0, and
        // the VAE output for N latent frames is 8*(N-1)+1 pixel frames.
        // So tile [start..end) in latent → pixel [8*start .. 8*start + 8*(end-start-1)+1).
        const tile_actual_lat = actual_end - start;
        const px_start = 8 * start;
        const px_end = px_start + 8 * (tile_actual_lat - 1) + 1;

        const is_first = (start == 0);
        const is_last = (end >= f_lat);

        stdx.debug.assert(count < MAX_TILES, "too many temporal tiles", .{});
        result[count] = .{
            .lat_start = start,
            .lat_end = if (is_last and end < start + config.tile_latent_frames)
                start + config.tile_latent_frames // padded to full tile size
            else
                end,
            .lat_actual_end = actual_end,
            .px_start = px_start,
            .px_end = px_end,
            .is_first = is_first,
            .is_last = is_last,
        };
        count += 1;

        if (is_last) break;
    }

    return .{ .tiles = result, .count = count };
}

/// Compute per-pixel-frame blend weights for a tile.
/// Returns a weight array of length `px_count` (the tile's actual pixel frames).
/// Overlap regions use a linear ramp [0..1]; non-overlap regions have weight 1.
pub fn computeBlendWeights(
    allocator: std.mem.Allocator,
    tile: TemporalTile,
    config: TemporalTilingConfig,
) ![]f32 {
    const px_count: usize = @intCast(tile.px_end - tile.px_start);
    const weights = try allocator.alloc(f32, px_count);

    // Fill with 1.0
    @memset(weights, 1.0);

    const overlap_px: usize = @intCast(config.overlapPixelFrames());

    // Left ramp (except for the first tile — no blending needed on the left)
    if (!tile.is_first and overlap_px > 0) {
        const ramp_len = overlap_px;
        for (0..@min(ramp_len, px_count)) |i| {
            weights[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(ramp_len - 1));
        }
    }

    // Right ramp (except for the last tile — no blending needed on the right)
    if (!tile.is_last and overlap_px > 0) {
        const ramp_len = overlap_px;
        for (0..@min(ramp_len, px_count)) |i| {
            const idx = px_count - 1 - i;
            const w = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(ramp_len - 1));
            // Take the minimum of existing weight and the right ramp
            // (handles the case where both ramps overlap in a very short tile)
            weights[idx] = @min(weights[idx], w);
        }
    }

    return weights;
}

// ============================================================================
// Tests
// ============================================================================

test "blend weights sum to 1.0 for every pixel frame (F'=16)" {
    const allocator = std.testing.allocator;
    const config: TemporalTilingConfig = .{};
    const f_lat: i64 = 16;
    const f_px: usize = @intCast(8 * (f_lat - 1) + 1); // 121

    const tile_plan = computeTemporalTiles(f_lat, config);
    try std.testing.expectEqual(@as(usize, 3), tile_plan.count);

    // Accumulate total weight per pixel frame
    var total_weight: [121]f32 = .{0.0} ** 121;

    for (0..tile_plan.count) |i| {
        const tile = tile_plan.tiles[i];
        const weights = try computeBlendWeights(allocator, tile, config);
        defer allocator.free(weights);

        const px_offset: usize = @intCast(tile.px_start);
        for (0..weights.len) |f| {
            total_weight[px_offset + f] += weights[f];
        }
    }

    // Every pixel frame must have total weight ≈ 1.0
    for (0..f_px) |f| {
        if (@abs(total_weight[f] - 1.0) > 1e-5) {
            std.debug.print("Frame {d}: weight sum = {d:.6}\n", .{ f, total_weight[f] });
            return error.TestUnexpectedResult;
        }
    }
}
