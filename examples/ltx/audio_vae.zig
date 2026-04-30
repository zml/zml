const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const conv_ops = @import("conv_ops.zig");
const Conv2dWeight = conv_ops.Conv2dWeight;

// ============================================================================
// Audio VAE Decoder
// Python ref: ltx_core/model/audio_vae/decoder.py
// ============================================================================

// --- Parameter structs ---

/// Audio VAE ResBlock: PixelNorm → SiLU → CausalConv2d → PixelNorm → SiLU → CausalConv2d + residual.
/// Some blocks have a nin_shortcut (1×1 conv) when in_channels != out_channels.
pub const AudioVaeResBlock = struct {
    conv1: Conv2dWeight, // conv1.conv.{weight,bias}
    conv2: Conv2dWeight, // conv2.conv.{weight,bias}
    nin_shortcut: ?Conv2dWeight, // nin_shortcut.conv.{weight,bias}, null if same channels
};

/// Audio VAE upsample block: nearest 2× interpolation + CausalConv2d + drop first row.
pub const AudioVaeUpsample = struct {
    conv: Conv2dWeight, // upsample.conv.conv.{weight,bias}
};

/// Full Audio VAE Decoder parameter set.
/// Architecture: conv_in → mid_block_1 → mid_block_2 → up.2 → up.1 → up.0 → norm_out → conv_out
///
/// Config: ch=128, ch_mult=(1,2,4), num_res_blocks=2, z_channels=8, out_ch=2
/// Base channels = ch * ch_mult[-1] = 128 * 4 = 512
/// No attention at all.
pub const AudioVaeDecoderParams = struct {
    // conv_in: CausalConv2d(8 → 512, k=3)
    conv_in: Conv2dWeight,

    // mid.block_1, mid.block_2: ResBlock(512)
    mid_block_1: AudioVaeResBlock,
    mid_block_2: AudioVaeResBlock,

    // up.2: 3× ResBlock(512) + Upsample(512)
    up2_block0: AudioVaeResBlock,
    up2_block1: AudioVaeResBlock,
    up2_block2: AudioVaeResBlock,
    up2_upsample: AudioVaeUpsample,

    // up.1: ResBlock(512→256, nin_shortcut) + 2× ResBlock(256) + Upsample(256)
    up1_block0: AudioVaeResBlock, // has nin_shortcut 512→256
    up1_block1: AudioVaeResBlock,
    up1_block2: AudioVaeResBlock,
    up1_upsample: AudioVaeUpsample,

    // up.0: ResBlock(256→128, nin_shortcut) + 2× ResBlock(128) — NO upsample
    up0_block0: AudioVaeResBlock, // has nin_shortcut 256→128
    up0_block1: AudioVaeResBlock,
    up0_block2: AudioVaeResBlock,

    // conv_out: CausalConv2d(128 → 2, k=3)
    conv_out: Conv2dWeight,
};

/// Audio per-channel statistics (separate from video).
/// Keys: audio_vae.per_channel_statistics.{mean-of-means, std-of-means} [128]
pub const AudioPerChannelStats = struct {
    mean_of_means: Tensor, // [128]
    std_of_means: Tensor, // [128]
};

// --- Weight loading ---

fn initAudioConv2d(store: zml.io.TensorStore.View) Conv2dWeight {
    const c = store.withPrefix("conv");
    return .{
        .weight = c.createTensor("weight", null, null),
        .bias = c.createTensor("bias", null, null),
    };
}

fn initAudioVaeResBlock(store: zml.io.TensorStore.View) AudioVaeResBlock {
    return .{
        .conv1 = initAudioConv2d(store.withPrefix("conv1")),
        .conv2 = initAudioConv2d(store.withPrefix("conv2")),
        .nin_shortcut = null,
    };
}

fn initAudioVaeResBlockWithShortcut(store: zml.io.TensorStore.View) AudioVaeResBlock {
    return .{
        .conv1 = initAudioConv2d(store.withPrefix("conv1")),
        .conv2 = initAudioConv2d(store.withPrefix("conv2")),
        .nin_shortcut = initAudioConv2d(store.withPrefix("nin_shortcut")),
    };
}

pub fn initAudioVaeDecoderParams(store: zml.io.TensorStore.View) AudioVaeDecoderParams {
    const dec = store.withPrefix("audio_vae").withPrefix("decoder");

    return .{
        .conv_in = initAudioConv2d(dec.withPrefix("conv_in")),

        .mid_block_1 = initAudioVaeResBlock(dec.withPrefix("mid").withPrefix("block_1")),
        .mid_block_2 = initAudioVaeResBlock(dec.withPrefix("mid").withPrefix("block_2")),

        // up.2: 3× ResBlock(512), all same channels
        .up2_block0 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(0)),
        .up2_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(1)),
        .up2_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(2)),
        .up2_upsample = .{ .conv = initAudioConv2d(dec.withPrefix("up").withLayer(2).withPrefix("upsample").withPrefix("conv")) },

        // up.1: first block has nin_shortcut (512→256)
        .up1_block0 = initAudioVaeResBlockWithShortcut(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(0)),
        .up1_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(1)),
        .up1_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(2)),
        .up1_upsample = .{ .conv = initAudioConv2d(dec.withPrefix("up").withLayer(1).withPrefix("upsample").withPrefix("conv")) },

        // up.0: first block has nin_shortcut (256→128)
        .up0_block0 = initAudioVaeResBlockWithShortcut(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(0)),
        .up0_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(1)),
        .up0_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(2)),

        .conv_out = initAudioConv2d(dec.withPrefix("conv_out")),
    };
}

pub fn initAudioPerChannelStats(store: zml.io.TensorStore.View) AudioPerChannelStats {
    const pcs = store.withPrefix("audio_vae").withPrefix("per_channel_statistics");
    return .{
        .mean_of_means = pcs.createTensor("mean-of-means", null, null),
        .std_of_means = pcs.createTensor("std-of-means", null, null),
    };
}

// --- Forward ops ---

/// CausalConv2d forward with HEIGHT causality, kernel_size=3.
/// Padding: (pad_h_before=2, pad_h_after=0, pad_w_before=1, pad_w_after=1).
/// input [B, C_in, H, W], kernel [C_out, C_in, 3, 3].
fn forwardCausalConv2dHeight(input: Tensor, w: Conv2dWeight) Tensor {
    // conv2d padding: {pad_h_before, pad_h_after, pad_w_before, pad_w_after}
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 2, 0, 1, 1 },
    });
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// CausalConv2d 1×1 forward (nin_shortcut) with HEIGHT causality.
/// kernel_size=1 → no padding needed.
fn forwardCausalConv2d1x1(input: Tensor, w: Conv2dWeight) Tensor {
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 0, 0, 0, 0 },
    });
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// Audio VAE ResBlock forward: PixelNorm → SiLU → CausalConv2d → PixelNorm → SiLU → CausalConv2d + residual.
fn forwardAudioVaeResBlock(x: Tensor, rb: AudioVaeResBlock) Tensor {
    var h = forwardPixelNorm2d(x);
    h = h.silu();
    h = forwardCausalConv2dHeight(h, rb.conv1);
    h = forwardPixelNorm2d(h);
    h = h.silu();
    h = forwardCausalConv2dHeight(h, rb.conv2);

    // nin_shortcut for channel mismatch
    var residual = x;
    if (rb.nin_shortcut) |shortcut| {
        residual = forwardCausalConv2d1x1(x, shortcut);
    }
    return h.add(residual);
}

/// PixelNorm for 4D tensors [B, C, H, W].
/// y = x / sqrt(mean(x², dim=C) + eps)
fn forwardPixelNorm2d(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    // mean over dim=1 (channels): sum returns [B, 1, H, W]
    const c: i64 = x.shape().dim(1);
    const sum_sq = x_sq.sum(1);
    const mean_sq = sum_sq.divByConst(c);
    const rms = mean_sq.addConstant(1e-6).sqrt();
    return x_f32.div(rms.broad(x_f32.shape())).convert(x.dtype());
}

/// Nearest-neighbor 2× upsample for [B, C, H, W] → [B, C, 2H, 2W].
/// Implemented as reshape + broadcast (no interpolation kernel needed).
fn forwardNearest2x(x: Tensor) Tensor {
    const B = x.shape().dim(0);
    const C = x.shape().dim(1);
    const H = x.shape().dim(2);
    const W = x.shape().dim(3);

    // [B, C, H, W] → [B, C, H, 1, W, 1] → broadcast to [B, C, H, 2, W, 2] → reshape [B, C, 2H, 2W]
    const expanded = x.reshape(.{ B, C, H, 1, W, 1 })
        .broad(zml.Shape.init(.{ B, C, H, 2, W, 2 }, x.dtype()));
    return expanded.reshape(.{ B, C, H * 2, W * 2 });
}

/// Audio VAE Upsample: nearest 2× → CausalConv2d → drop first row (HEIGHT causality).
fn forwardAudioVaeUpsample(x: Tensor, us: AudioVaeUpsample) Tensor {
    var h = forwardNearest2x(x);
    h = forwardCausalConv2dHeight(h, us.conv);
    // Drop first row to undo causal padding (HEIGHT causality)
    h = h.slice1d(2, .{ .start = 1 });
    return h;
}

/// Unpatchify audio latent: [B, T_aud, 128] → [B, 8, T_aud, 16].
/// rearrange "b t (c f) -> b c t f" with c=8, f=16.
pub fn forwardUnpatchifyAudio(patchified: Tensor) Tensor {
    const B = patchified.shape().dim(0);
    const T = patchified.shape().dim(1);
    // [B, T, 128] → [B, T, 8, 16] → transpose → [B, 8, T, 16]
    return patchified.reshape(.{ B, T, 8, 16 }).transpose(.{ 0, 2, 1, 3 });
}

/// Denormalize audio latent using per-channel statistics.
/// Patchify → denorm → unpatchify (operates on the 128-dim patchified representation).
fn forwardAudioDenormalize(latent: Tensor, stats: AudioPerChannelStats) Tensor {
    const B = latent.shape().dim(0);
    const T = latent.shape().dim(2);

    // Patchify: [B, 8, T, 16] → [B, T, 128]
    const patchified = latent.transpose(.{ 0, 2, 1, 3 }).reshape(.{ B, T, 128 });

    // Denormalize: x * std + mean (broadcast [128] over [B, T, 128])
    const stats_shape = zml.Shape.init(.{ 1, 1, 128 }, latent.dtype());
    const std_val = stats.std_of_means.reshape(stats_shape).broad(patchified.shape());
    const mean = stats.mean_of_means.reshape(stats_shape).broad(patchified.shape());
    const denormed = patchified.mul(std_val).add(mean);

    // Unpatchify back: [B, T, 128] → [B, 8, T, 16]
    return denormed.reshape(.{ B, T, 8, 16 }).transpose(.{ 0, 2, 1, 3 });
}

/// Full audio VAE decoder forward.
/// Input: [B, 8, T, 16] bf16 (audio latent after unpatchify)
/// Output: [B, 2, T_out, 64] bf16 (mel spectrogram)
///
/// T_out = max(T * 4 - 3, 1) (causal, LATENT_DOWNSAMPLE_FACTOR=4)
pub fn forwardAudioVaeDecode(
    latent: Tensor,
    stats: AudioPerChannelStats,
    params: AudioVaeDecoderParams,
) Tensor {
    // 1. Denormalize
    var x = forwardAudioDenormalize(latent, stats);

    // 2. conv_in: [B, 8, T, 16] → [B, 512, T, 16]
    x = forwardCausalConv2dHeight(x, params.conv_in);

    // 3. Mid block
    x = forwardAudioVaeResBlock(x, params.mid_block_1);
    x = forwardAudioVaeResBlock(x, params.mid_block_2);

    // 4. up.2: 3× ResBlock(512) + Upsample → [B, 512, 2T-1, 32]
    x = forwardAudioVaeResBlock(x, params.up2_block0);
    x = forwardAudioVaeResBlock(x, params.up2_block1);
    x = forwardAudioVaeResBlock(x, params.up2_block2);
    x = forwardAudioVaeUpsample(x, params.up2_upsample);

    // 5. up.1: ResBlock(512→256) + 2× ResBlock(256) + Upsample → [B, 256, 4T-3, 64]
    x = forwardAudioVaeResBlock(x, params.up1_block0);
    x = forwardAudioVaeResBlock(x, params.up1_block1);
    x = forwardAudioVaeResBlock(x, params.up1_block2);
    x = forwardAudioVaeUpsample(x, params.up1_upsample);

    // 6. up.0: ResBlock(256→128) + 2× ResBlock(128) — no upsample
    x = forwardAudioVaeResBlock(x, params.up0_block0);
    x = forwardAudioVaeResBlock(x, params.up0_block1);
    x = forwardAudioVaeResBlock(x, params.up0_block2);

    // 7. norm_out → SiLU → conv_out
    x = forwardPixelNorm2d(x);
    x = x.silu();
    x = forwardCausalConv2dHeight(x, params.conv_out);

    // Output: [B, 2, T_out, F_out] where T_out ≈ 4T-3, F_out ≈ 64
    return x;
}
