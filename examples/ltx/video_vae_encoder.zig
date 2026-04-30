const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const conv_ops = @import("conv_ops.zig");
const Conv3dWeight = conv_ops.Conv3dWeight;
const PerChannelStats = conv_ops.PerChannelStats;

const video_vae = @import("video_vae.zig");
const VaeResBlock = video_vae.VaeResBlock;

// ============================================================================
// Video VAE Encoder
// Python ref: ltx_core/model/video_vae/video_vae.py — VideoEncoder
//             ltx_core/model/video_vae/ops.py       — patchify, PerChannelStatistics
//             ltx_core/model/video_vae/sampling.py   — SpaceToDepthDownsample
//             ltx_core/model/video_vae/convolution.py — CausalConv3d
//
// Architecture (LTX-2.3, 84 weight tensors):
//   patchify(4) → conv_in → down_blocks.0..8 → PixelNorm → SiLU → conv_out
//     → extract means (ch 0..127) → normalize
//
// Block structure:
//   down_blocks.0: 4 ResBlocks @ 128ch
//   down_blocks.1: SpaceToDepth (1,2,2) 128→256
//   down_blocks.2: 6 ResBlocks @ 256ch
//   down_blocks.3: SpaceToDepth (2,1,1) 256→512
//   down_blocks.4: 4 ResBlocks @ 512ch
//   down_blocks.5: SpaceToDepth (2,2,2) 512→1024
//   down_blocks.6: 2 ResBlocks @ 1024ch
//   down_blocks.7: SpaceToDepth (2,2,2) 1024→1024
//   down_blocks.8: 2 ResBlocks @ 1024ch
// ============================================================================

// --- Parameter struct ---

/// Full parameter set for the video VAE encoder.
/// Checkpoint keys: vae.encoder.* (84 weight tensors).
pub const VideoVaeEncoderParams = struct {
    conv_in: Conv3dWeight,

    // down_blocks.0: 4 ResBlocks @ 128ch
    down0_res0: VaeResBlock,
    down0_res1: VaeResBlock,
    down0_res2: VaeResBlock,
    down0_res3: VaeResBlock,

    // down_blocks.1: SpaceToDepth (1,2,2) conv — 128→256ch
    down1_conv: Conv3dWeight,

    // down_blocks.2: 6 ResBlocks @ 256ch
    down2_res0: VaeResBlock,
    down2_res1: VaeResBlock,
    down2_res2: VaeResBlock,
    down2_res3: VaeResBlock,
    down2_res4: VaeResBlock,
    down2_res5: VaeResBlock,

    // down_blocks.3: SpaceToDepth (2,1,1) conv — 256→512ch
    down3_conv: Conv3dWeight,

    // down_blocks.4: 4 ResBlocks @ 512ch
    down4_res0: VaeResBlock,
    down4_res1: VaeResBlock,
    down4_res2: VaeResBlock,
    down4_res3: VaeResBlock,

    // down_blocks.5: SpaceToDepth (2,2,2) conv — 512→1024ch
    down5_conv: Conv3dWeight,

    // down_blocks.6: 2 ResBlocks @ 1024ch
    down6_res0: VaeResBlock,
    down6_res1: VaeResBlock,

    // down_blocks.7: SpaceToDepth (2,2,2) conv — 1024→1024ch
    down7_conv: Conv3dWeight,

    // down_blocks.8: 2 ResBlocks @ 1024ch
    down8_res0: VaeResBlock,
    down8_res1: VaeResBlock,

    conv_out: Conv3dWeight,
};

// --- Weight loading ---

fn initResBlock(store: zml.io.TensorStore.View) VaeResBlock {
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

fn initConv3d(store: zml.io.TensorStore.View) Conv3dWeight {
    const c = store.withPrefix("conv");
    return .{
        .weight = c.createTensor("weight", null, null),
        .bias = c.createTensor("bias", null, null),
    };
}

/// Load VideoVaeEncoderParams from the main model checkpoint.
/// Keys: vae.encoder.{conv_in,down_blocks.*,conv_out}.conv.{weight,bias}
pub fn initVideoVaeEncoderParams(store: zml.io.TensorStore.View) VideoVaeEncoderParams {
    const enc = store.withPrefix("vae").withPrefix("encoder");
    const db = enc.withPrefix("down_blocks");

    return .{
        .conv_in = initConv3d(enc.withPrefix("conv_in")),

        // down_blocks.0: 4 ResBlocks @ 128ch
        .down0_res0 = initResBlock(db.withLayer(0).withPrefix("res_blocks").withLayer(0)),
        .down0_res1 = initResBlock(db.withLayer(0).withPrefix("res_blocks").withLayer(1)),
        .down0_res2 = initResBlock(db.withLayer(0).withPrefix("res_blocks").withLayer(2)),
        .down0_res3 = initResBlock(db.withLayer(0).withPrefix("res_blocks").withLayer(3)),

        // down_blocks.1: SpaceToDepth (1,2,2)
        .down1_conv = initConv3d(db.withLayer(1).withPrefix("conv")),

        // down_blocks.2: 6 ResBlocks @ 256ch
        .down2_res0 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(0)),
        .down2_res1 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(1)),
        .down2_res2 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(2)),
        .down2_res3 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(3)),
        .down2_res4 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(4)),
        .down2_res5 = initResBlock(db.withLayer(2).withPrefix("res_blocks").withLayer(5)),

        // down_blocks.3: SpaceToDepth (2,1,1)
        .down3_conv = initConv3d(db.withLayer(3).withPrefix("conv")),

        // down_blocks.4: 4 ResBlocks @ 512ch
        .down4_res0 = initResBlock(db.withLayer(4).withPrefix("res_blocks").withLayer(0)),
        .down4_res1 = initResBlock(db.withLayer(4).withPrefix("res_blocks").withLayer(1)),
        .down4_res2 = initResBlock(db.withLayer(4).withPrefix("res_blocks").withLayer(2)),
        .down4_res3 = initResBlock(db.withLayer(4).withPrefix("res_blocks").withLayer(3)),

        // down_blocks.5: SpaceToDepth (2,2,2)
        .down5_conv = initConv3d(db.withLayer(5).withPrefix("conv")),

        // down_blocks.6: 2 ResBlocks @ 1024ch
        .down6_res0 = initResBlock(db.withLayer(6).withPrefix("res_blocks").withLayer(0)),
        .down6_res1 = initResBlock(db.withLayer(6).withPrefix("res_blocks").withLayer(1)),

        // down_blocks.7: SpaceToDepth (2,2,2)
        .down7_conv = initConv3d(db.withLayer(7).withPrefix("conv")),

        // down_blocks.8: 2 ResBlocks @ 1024ch
        .down8_res0 = initResBlock(db.withLayer(8).withPrefix("res_blocks").withLayer(0)),
        .down8_res1 = initResBlock(db.withLayer(8).withPrefix("res_blocks").withLayer(1)),

        .conv_out = initConv3d(enc.withPrefix("conv_out")),
    };
}

// --- Forward ops ---

/// CausalConv3d (causal=True mode): front-only temporal replicate-pad + conv3d.
/// For kernel_size=3: replicate first frame twice at front, zero-pad H,W by 1.
/// Python: CausalConv3d.forward(x, causal=True) — used by encoder.
pub fn forwardCausalConv3d(input: Tensor, w: Conv3dWeight) Tensor {
    // Temporal causal padding: replicate first frame (kernel_size - 1 = 2 times)
    const first_frame = input.slice1d(2, .{ .end = 1 }); // [B,C,1,H,W]
    const padded = Tensor.concatenate(&.{ first_frame, first_frame, input }, 2); // [B,C,F+2,H,W]

    // Conv3d: no temporal padding (already handled), zero-pad H,W by 1
    const conv_out = padded.conv3d(w.weight, .{
        .padding = &.{ 0, 0, 1, 1, 1, 1 }, // D:0, H:1, W:1
    });

    // Add bias: [C_out] → [1, C_out, 1, 1, 1]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1),
    ).broad(conv_out.shape()));
}

/// PixelNorm: x / sqrt(mean(x², dim=channels) + eps).
/// Computed in f32 for numerical stability (matches Python's effective behavior
/// through XLA — the bf16 variant produces identical results on CUDA due to
/// XLA's internal promotion, but f32 is safer for other backends).
pub fn forwardPixelNorm(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    const mean_sq = x_sq.mean(1);
    const rms = mean_sq.addConstant(1e-8).sqrt();
    return x_f32.div(rms.broad(x_f32.shape())).convert(x.dtype());
}

/// VAE ResnetBlock3D forward (causal=True convolutions for encoder):
///   PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU → CausalConv3d + residual.
pub fn forwardResBlock(x: Tensor, rb: VaeResBlock) Tensor {
    var h = forwardPixelNorm(x);
    h = h.silu();
    h = forwardCausalConv3d(h, rb.conv1);
    h = forwardPixelNorm(h);
    h = h.silu();
    h = forwardCausalConv3d(h, rb.conv2);
    return h.add(x);
}

/// Space-to-depth rearrange: [B, C, D, H, W] → [B, C*p1*p2*p3, D/p1, H/p2, W/p3].
/// Inverse of DepthToSpace rearrange in the decoder.
/// einops: "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"
fn forwardSpaceToDepthRearrange(x: Tensor, comptime stride: [3]i64) Tensor {
    const B = x.dim(0);
    const C = x.dim(1);
    const D = x.dim(2);
    const H = x.dim(3);
    const W = x.dim(4);
    const p1 = stride[0];
    const p2 = stride[1];
    const p3 = stride[2];
    const d = @divExact(D, p1);
    const h = @divExact(H, p2);
    const w = @divExact(W, p3);

    // [B, C, D, H, W] → [B, C, d, p1, h, p2, w, p3]
    var t = x.reshape(.{ B, C, d, p1, h, p2, w, p3 });
    // → [B, C, p1, p2, p3, d, h, w]  (inverse of decoder's transpose)
    t = t.transpose(.{ 0, 1, 3, 5, 7, 2, 4, 6 });
    // → [B, C*p1*p2*p3, d, h, w]
    return t.reshape(.{ B, C * p1 * p2 * p3, d, h, w });
}

/// SpaceToDepthDownsample: residual downsampling block.
/// Python: SpaceToDepthDownsample.forward
///
/// 1. If temporal stride=2: duplicate first frame (pad F → F+1)
/// 2. Skip path: space-to-depth rearrange → group mean (reduce channels)
/// 3. Conv path: causal conv3d → space-to-depth rearrange
/// 4. Return conv + skip
pub fn forwardSpaceToDepthDownsample(
    input: Tensor,
    conv_w: Conv3dWeight,
    comptime stride: [3]i64,
    comptime group_size: i64,
) Tensor {
    // Temporal padding: duplicate first frame if temporal stride is 2
    var x = input;
    if (stride[0] == 2) {
        const first_frame = x.slice1d(2, .{ .end = 1 });
        x = Tensor.concatenate(&.{ first_frame, x }, 2);
    }

    // Skip connection: space-to-depth rearrange + group mean
    var x_in = forwardSpaceToDepthRearrange(x, stride);
    if (group_size > 1) {
        // [B, C_total, d, h, w] → [B, out_ch, group_size, d, h, w] → mean(dim=2) → [B, out_ch, d, h, w]
        const B = x_in.dim(0);
        const C_total = x_in.dim(1);
        const d = x_in.dim(2);
        const h = x_in.dim(3);
        const w = x_in.dim(4);
        const out_ch = @divExact(C_total, group_size);
        x_in = x_in.reshape(.{ B, out_ch, group_size, d, h, w });
        x_in = x_in.mean(2); // zml mean keeps dim → [B, out_ch, 1, d, h, w]
        x_in = x_in.reshape(.{ B, out_ch, d, h, w }); // collapse back to 5D
    }
    // group_size == 1: no-op (x_in already has correct channels)

    // Conv path: causal conv3d → same rearrange
    var x_conv = forwardCausalConv3d(x, conv_w);
    x_conv = forwardSpaceToDepthRearrange(x_conv, stride);

    return x_conv.add(x_in);
}

/// Patchify for encoder input: [B, 3, F, H, W] → [B, 48, F, H/4, W/4].
/// einops: "b c (f p) (h q) (w r) -> b (c p r q) f h w" with p=1, q=4, r=4
/// Inverse of forwardUnpatchifyVae in video_vae.zig.
pub fn forwardPatchifyVae(input: Tensor) Tensor {
    const B = input.dim(0);
    const F = input.dim(2);
    const H = input.dim(3);
    const W = input.dim(4);
    const h = @divExact(H, 4);
    const w = @divExact(W, 4);

    // [B, 3, F, H, W] → [B, 3, F, 1, H/4, 4, W/4, 4]
    //   dims:            [B, c, f, p,  h,   q,  w,   r]
    var x = input.reshape(.{ B, 3, F, 1, h, 4, w, 4 });
    // → [B, c, p, r, q, f, h, w]   channel order matches einops (c p r q)
    x = x.transpose(.{ 0, 1, 3, 7, 5, 2, 4, 6 });
    // → [B, 3*1*4*4, F, H/4, W/4] = [B, 48, F, H/4, W/4]
    return x.reshape(.{ B, 48, F, h, w });
}

// --- Top-level encoder ---

/// Full video VAE encoder forward pass.
/// Encodes a pixel-space image into a normalized latent representation.
///
/// Input:  pixel_input [B, 3, 1, H, W] bf16 — single frame normalized to [-1, 1]
/// Output: [B, 128, 1, H/32, W/32] bf16
pub fn forwardVideoVaeEncode(
    pixel_input: Tensor,
    stats: PerChannelStats,
    params: VideoVaeEncoderParams,
) Tensor {
    // 1. Patchify: [B, 3, 1, H, W] → [B, 48, 1, H/4, W/4]
    var x = forwardPatchifyVae(pixel_input);

    // 2. conv_in: CausalConv3d(48 → 128)
    x = forwardCausalConv3d(x, params.conv_in);

    // 3. down_blocks.0: 4 ResBlocks @ 128ch
    x = forwardResBlock(x, params.down0_res0);
    x = forwardResBlock(x, params.down0_res1);
    x = forwardResBlock(x, params.down0_res2);
    x = forwardResBlock(x, params.down0_res3);

    // 4. down_blocks.1: SpaceToDepth (1,2,2) → 128→256ch  (group_size=2)
    x = forwardSpaceToDepthDownsample(x, params.down1_conv, .{ 1, 2, 2 }, 2);

    // 5. down_blocks.2: 6 ResBlocks @ 256ch
    x = forwardResBlock(x, params.down2_res0);
    x = forwardResBlock(x, params.down2_res1);
    x = forwardResBlock(x, params.down2_res2);
    x = forwardResBlock(x, params.down2_res3);
    x = forwardResBlock(x, params.down2_res4);
    x = forwardResBlock(x, params.down2_res5);

    // 6. down_blocks.3: SpaceToDepth (2,1,1) → 256→512ch  (group_size=1)
    x = forwardSpaceToDepthDownsample(x, params.down3_conv, .{ 2, 1, 1 }, 1);

    // 7. down_blocks.4: 4 ResBlocks @ 512ch
    x = forwardResBlock(x, params.down4_res0);
    x = forwardResBlock(x, params.down4_res1);
    x = forwardResBlock(x, params.down4_res2);
    x = forwardResBlock(x, params.down4_res3);

    // 8. down_blocks.5: SpaceToDepth (2,2,2) → 512→1024ch  (group_size=4)
    x = forwardSpaceToDepthDownsample(x, params.down5_conv, .{ 2, 2, 2 }, 4);

    // 9. down_blocks.6: 2 ResBlocks @ 1024ch
    x = forwardResBlock(x, params.down6_res0);
    x = forwardResBlock(x, params.down6_res1);

    // 10. down_blocks.7: SpaceToDepth (2,2,2) → 1024→1024ch  (group_size=8)
    x = forwardSpaceToDepthDownsample(x, params.down7_conv, .{ 2, 2, 2 }, 8);

    // 11. down_blocks.8: 2 ResBlocks @ 1024ch
    x = forwardResBlock(x, params.down8_res0);
    x = forwardResBlock(x, params.down8_res1);

    // 12. PixelNorm → SiLU → conv_out
    x = forwardPixelNorm(x);
    x = x.silu();
    x = forwardCausalConv3d(x, params.conv_out);

    // 13. Extract means: [B, 129, F, H, W] → [B, 128, F, H, W]  (drop logvar channel)
    x = x.slice1d(1, .{ .end = 128 });

    // 14. Normalize: (x - mean_of_means) / std_of_means
    const stats_shape = x.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(x.shape());
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(x.shape());
    return x.sub(mean_broad).div(std_broad);
}
