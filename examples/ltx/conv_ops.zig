const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const ops = zml.ops;

// ============================================================================
// Shared convolution types and operations
// Used by upsampler, video VAE, and audio VAE subsystems.
// ============================================================================

/// Weight struct for a 3D convolution (Conv3d).
/// Shape: weight [C_out, C_in, D, H, W], bias [C_out].
pub const Conv3dWeight = struct {
    weight: Tensor,
    bias: Tensor,
};

/// Weight struct for a 2D convolution (Conv2d).
/// Shape: weight [C_out, C_in, H, W], bias [C_out].
pub const Conv2dWeight = struct {
    weight: Tensor,
    bias: Tensor,
};

/// GroupNorm weight struct (gamma/beta).
/// Python: torch.nn.GroupNorm(num_groups, num_channels).
pub const GroupNormWeight = struct {
    weight: Tensor, // gamma, shape [C]
    bias: Tensor, // beta, shape [C]
};

/// Per-channel statistics for normalize / un_normalize.
/// From main checkpoint: vae.per_channel_statistics.{mean-of-means, std-of-means}
pub const PerChannelStats = struct {
    mean_of_means: Tensor, // [128]
    std_of_means: Tensor, // [128]
};

/// Load PerChannelStats from the main model checkpoint.
/// Keys: vae.per_channel_statistics.mean-of-means, vae.per_channel_statistics.std-of-means
pub fn initPerChannelStats(store: zml.io.TensorStore.View) PerChannelStats {
    const pcs = store.withPrefix("vae").withPrefix("per_channel_statistics");
    return .{
        .mean_of_means = pcs.createTensor("mean-of-means", null, .replicated),
        .std_of_means = pcs.createTensor("std-of-means", null, .replicated),
    };
}

// --- Forward ops ---

/// Conv3d forward: input [B, C_in, D, H, W], kernel [C_out, C_in, kD, kH, kW].
/// Padding=1 on all spatial dims (same padding for kernel_size=3).
pub fn forwardConv3d(input: Tensor, w: Conv3dWeight) Tensor {
    const conv_out = input.conv3d(w.weight, .{
        .padding = &.{ 1, 1, 1, 1, 1, 1 }, // padding=1 on each side of D, H, W
    });
    // Broadcast bias [C_out] → [B, C_out, D, H, W]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1),
    ).broad(conv_out.shape()));
}

/// Conv2d forward: input [B, C_in, H, W], kernel [C_out, C_in, kH, kW].
/// Padding=1 on all spatial dims (same padding for kernel_size=3).
pub fn forwardConv2d(input: Tensor, w: Conv2dWeight) Tensor {
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 1, 1, 1, 1 },
    });
    // Broadcast bias [C_out] → [B, C_out, H, W]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// GroupNorm forward for 5D tensors [B, C, D, H, W].
/// Python: torch.nn.GroupNorm(num_groups, C).
pub fn forwardGroupNorm(input: Tensor, w: GroupNormWeight, comptime num_groups: i64) Tensor {
    const B = input.dim(0);
    const C = input.dim(1);
    const D = input.dim(2);
    const H = input.dim(3);
    const W = input.dim(4);
    const channels_per_group = @divExact(C, num_groups);
    const group_size = channels_per_group * D * H * W;

    // Reshape: [B, C, D, H, W] → [B, G, C/G, D, H, W]
    const grouped = input.reshape(.{ B, num_groups, channels_per_group, D, H, W });
    const grouped_f32 = grouped.convert(.f32);

    // Multi-axis sum over dims {2,3,4,5} (C/G, D, H, W) in a single HLO reduce.
    // This avoids sequential single-axis reductions that cause TPU layout conflicts.
    const reduced_sum = multiAxisSum(grouped_f32, &.{ 2, 3, 4, 5 });
    const reduced_mean = reduced_sum.divByConst(group_size);
    // reduced_mean shape: [B, G, 1, 1, 1, 1]

    // Variance: E[(x - mean)^2]
    const centered = grouped_f32.sub(reduced_mean.broadcastLeft(grouped_f32.shape()));
    const var_sum = multiAxisSum(centered.mul(centered), &.{ 2, 3, 4, 5 });
    const reduced_var = var_sum.divByConst(group_size);

    // Normalize: (x - mean) / sqrt(var + eps)
    const eps: f32 = 1e-5;
    const inv_std = reduced_var.addConstant(eps).rsqrt();
    const normed = centered.mul(inv_std.broadcastLeft(grouped_f32.shape()));

    // Reshape back in f32: [B, G, C/G, D, H, W] → [B, C, D, H, W]
    const normed_5d = normed.reshape(.{ B, C, D, H, W });

    // Apply affine in f32: gamma * x + beta, then convert to input dtype.
    // Keeping f32 through the affine avoids precision loss when gamma is large.
    const affine_shape = input.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const gamma = w.weight.convert(.f32).reshape(affine_shape).broad(normed_5d.shape());
    const beta = w.bias.convert(.f32).reshape(affine_shape).broad(normed_5d.shape());
    return normed_5d.mul(gamma).add(beta).convert(input.dtype());
}

/// PixelShuffle 2D: rearrange (B, C*r*r, H, W) → (B, C, H*r, W*r).
/// Pure reshape + transpose, no learnable parameters.
pub fn forwardPixelShuffle2d(input: Tensor, comptime upscale_factor: i64) Tensor {
    const B = input.dim(0);
    const C_total = input.dim(1);
    const H = input.dim(2);
    const W = input.dim(3);
    const C = @divExact(C_total, upscale_factor * upscale_factor);
    const r = upscale_factor;

    // [B, C*r*r, H, W] → [B, C, r, r, H, W] → [B, C, H, r, W, r] → [B, C, H*r, W*r]
    return input
        .reshape(.{ B, C, r, r, H, W })
        .transpose(.{ 0, 1, 4, 2, 5, 3 })
        .reshape(.{ B, C, H * r, W * r });
}

/// Multi-axis sum: reduce over all given axes in a single HLO reduce op.
/// Avoids sequential single-axis reductions that can cause TPU layout conflicts.
fn multiAxisSum(tensor: Tensor, comptime axes: []const i64) Tensor {
    return ops.reduce(
        .{tensor},
        .{Tensor.constant(tensor.dtype().zero())},
        axes,
        struct {
            pub fn add(args: ops.ReduceArgs) struct { Tensor } {
                return .{args.right.add(args.left.convert(args.right.dtype()))};
            }
        }.add,
        .{},
    )[0];
}
