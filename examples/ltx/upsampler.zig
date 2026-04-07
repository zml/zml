const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const conv_ops = @import("conv_ops.zig");
const Conv3dWeight = conv_ops.Conv3dWeight;
const Conv2dWeight = conv_ops.Conv2dWeight;
const GroupNormWeight = conv_ops.GroupNormWeight;
const PerChannelStats = conv_ops.PerChannelStats;

// ============================================================================
// Latent Upsampler (stage 1 → stage 2 bridge)
// Python ref: ltx_core/model/upsampler/model.py — LatentUpsampler
//             ltx_core/model/upsampler/res_block.py — ResBlock
//             ltx_core/model/video_vae/ops.py — PerChannelStatistics
// ============================================================================

/// ResBlock for the upsampler: conv1→norm1→silu→conv2→norm2→silu(x+residual).
/// Python ref: ltx_core/model/upsampler/res_block.py
pub const UpsamplerResBlock = struct {
    conv1: Conv3dWeight,
    norm1: GroupNormWeight,
    conv2: Conv3dWeight,
    norm2: GroupNormWeight,
};

/// Full parameter set for the LatentUpsampler CNN.
/// Checkpoint: ltx-2.3-spatial-upscaler-x2-1.1.safetensors (72 keys)
pub const UpsamplerParams = struct {
    initial_conv: Conv3dWeight,
    initial_norm: GroupNormWeight,
    res_block_0: UpsamplerResBlock,
    res_block_1: UpsamplerResBlock,
    res_block_2: UpsamplerResBlock,
    res_block_3: UpsamplerResBlock,
    /// Conv2d for spatial upsample (operates on (B*F, C, H, W), then PixelShuffle).
    upsampler_conv: Conv2dWeight,
    post_res_block_0: UpsamplerResBlock,
    post_res_block_1: UpsamplerResBlock,
    post_res_block_2: UpsamplerResBlock,
    post_res_block_3: UpsamplerResBlock,
    final_conv: Conv3dWeight,
};

// --- Weight loading ---

/// Load UpsamplerParams from an upsampler safetensors checkpoint.
pub fn initUpsamplerParams(store: zml.io.TensorStore.View) UpsamplerParams {
    const ic = store.withPrefix("initial_conv");
    const in_ = store.withPrefix("initial_norm");
    const fc = store.withPrefix("final_conv");
    // upsampler.0 is the Conv2d inside nn.Sequential
    const us = store.withPrefix("upsampler").withLayer(0);

    return .{
        .initial_conv = .{
            .weight = ic.createTensor("weight", null, null),
            .bias = ic.createTensor("bias", null, null),
        },
        .initial_norm = .{
            .weight = in_.createTensor("weight", null, null),
            .bias = in_.createTensor("bias", null, null),
        },
        .res_block_0 = initResBlockParams(store.withPrefix("res_blocks").withLayer(0)),
        .res_block_1 = initResBlockParams(store.withPrefix("res_blocks").withLayer(1)),
        .res_block_2 = initResBlockParams(store.withPrefix("res_blocks").withLayer(2)),
        .res_block_3 = initResBlockParams(store.withPrefix("res_blocks").withLayer(3)),
        .upsampler_conv = .{
            .weight = us.createTensor("weight", null, null),
            .bias = us.createTensor("bias", null, null),
        },
        .post_res_block_0 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(0)),
        .post_res_block_1 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(1)),
        .post_res_block_2 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(2)),
        .post_res_block_3 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(3)),
        .final_conv = .{
            .weight = fc.createTensor("weight", null, null),
            .bias = fc.createTensor("bias", null, null),
        },
    };
}

fn initResBlockParams(store: zml.io.TensorStore.View) UpsamplerResBlock {
    const c1 = store.withPrefix("conv1");
    const n1 = store.withPrefix("norm1");
    const c2 = store.withPrefix("conv2");
    const n2 = store.withPrefix("norm2");
    return .{
        .conv1 = .{
            .weight = c1.createTensor("weight", null, null),
            .bias = c1.createTensor("bias", null, null),
        },
        .norm1 = .{
            .weight = n1.createTensor("weight", null, null),
            .bias = n1.createTensor("bias", null, null),
        },
        .conv2 = .{
            .weight = c2.createTensor("weight", null, null),
            .bias = c2.createTensor("bias", null, null),
        },
        .norm2 = .{
            .weight = n2.createTensor("weight", null, null),
            .bias = n2.createTensor("bias", null, null),
        },
    };
}

// --- Forward ops ---

/// Forward pass for a single upsampler ResBlock.
/// Python: conv1→norm1→silu→conv2→norm2→silu(x+residual)
fn forwardResBlock(x: Tensor, rb: UpsamplerResBlock) Tensor {
    const residual = x;
    var h = conv_ops.forwardConv3d(x, rb.conv1);
    h = conv_ops.forwardGroupNorm(h, rb.norm1, 32);
    h = h.silu();
    h = conv_ops.forwardConv3d(h, rb.conv2);
    h = conv_ops.forwardGroupNorm(h, rb.norm2, 32);
    return h.add(residual).silu();
}

/// Full upsampler forward: un_normalize → LatentUpsampler CNN → normalize.
///
/// Input: [B, 128, F, H, W] bf16 (un-patchified Stage 1 latent)
/// Output: [B, 128, F, H*2, W*2] bf16 (spatially upsampled)
///
/// Python ref: upsample_video() in model.py
pub fn forwardUpsample(
    input: Tensor,
    params: UpsamplerParams,
    stats: PerChannelStats,
) Tensor {
    const B = input.dim(0);
    const F = input.dim(2);

    // Step 1: un_normalize — x * std_of_means + mean_of_means (per-channel)
    // Broadcast stats [128] → [B, 128, F, H, W]
    const stats_shape = input.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(input.shape());
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(input.shape());
    var x = input.mul(std_broad).add(mean_broad);

    // Step 2: LatentUpsampler.forward()
    // initial_conv → initial_norm → silu
    x = conv_ops.forwardConv3d(x, params.initial_conv);
    x = conv_ops.forwardGroupNorm(x, params.initial_norm, 32);
    x = x.silu();

    // 4x ResBlock (pre-upsample)
    x = forwardResBlock(x, params.res_block_0);
    x = forwardResBlock(x, params.res_block_1);
    x = forwardResBlock(x, params.res_block_2);
    x = forwardResBlock(x, params.res_block_3);

    // Spatial upsample: rearrange to 2D → Conv2d → PixelShuffle(2) → back to 5D
    // "b c f h w → (b f) c h w" requires transpose before reshape
    const H = x.dim(3);
    const W = x.dim(4);
    const C = x.dim(1);
    const x_bfchw = x.transpose(.{ 0, 2, 1, 3, 4 }); // [B, C, F, H, W] → [B, F, C, H, W]
    const x_4d = x_bfchw.reshape(.{ B * F, C, H, W });
    var upsampled_4d = conv_ops.forwardConv2d(x_4d, params.upsampler_conv);
    upsampled_4d = conv_ops.forwardPixelShuffle2d(upsampled_4d, 2);
    // "(b f) c h w → b c f h w"
    const C_out = upsampled_4d.dim(1);
    const H2 = upsampled_4d.dim(2);
    const W2 = upsampled_4d.dim(3);
    x = upsampled_4d.reshape(.{ B, F, C_out, H2, W2 }).transpose(.{ 0, 2, 1, 3, 4 });

    // 4x ResBlock (post-upsample)
    x = forwardResBlock(x, params.post_res_block_0);
    x = forwardResBlock(x, params.post_res_block_1);
    x = forwardResBlock(x, params.post_res_block_2);
    x = forwardResBlock(x, params.post_res_block_3);

    // final_conv
    x = conv_ops.forwardConv3d(x, params.final_conv);

    // Step 3: normalize — (x - mean_of_means) / std_of_means
    const out_stats_shape = x.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const out_std = stats.std_of_means.reshape(out_stats_shape).broad(x.shape());
    const out_mean = stats.mean_of_means.reshape(out_stats_shape).broad(x.shape());
    return x.sub(out_mean).div(out_std);
}

/// Unpatchify a video latent: [1, T, 128] → [1, 128, F, H, W].
/// VideoLatentPatchifier(patch_size=1): patchify = "b c f h w → b (f h w) c"
/// target_shape carries the [1, 128, F, H, W] dimensions.
pub fn forwardUnpatchifyVideo(input: Tensor, target_shape: zml.Shape) Tensor {
    const B = target_shape.dim(0);
    const C = target_shape.dim(1);
    const F = target_shape.dim(2);
    const H = target_shape.dim(3);
    const W = target_shape.dim(4);
    // [1, F*H*W, 128] → [1, F, H, W, 128] → [1, 128, F, H, W]
    return input
        .reshape(.{ B, F, H, W, C })
        .transpose(.{ 0, 4, 1, 2, 3 });
}

/// Patchify a video latent: [1, 128, F, H, W] → [1, F*H*W, 128].
/// VideoLatentPatchifier(patch_size=1): patchify = "b c f h w → b (f h w) c"
/// Inverse of forwardUnpatchifyVideo.
pub fn forwardPatchifyVideo(input: Tensor) Tensor {
    const B = input.dim(0);
    const C = input.dim(1);
    const F = input.dim(2);
    const H = input.dim(3);
    const W = input.dim(4);
    // [1, 128, F, H, W] → [1, F, H, W, 128] → [1, F*H*W, 128]
    return input
        .transpose(.{ 0, 2, 3, 4, 1 })
        .reshape(.{ B, F * H * W, C });
}
