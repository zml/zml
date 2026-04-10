const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

// ============================================================================
// Vocoder + BWE (mel spectrogram → 48kHz stereo waveform)
// ============================================================================
//
// Architecture (from checkpoint config):
//   VocoderWithBWE:
//     1. Vocoder (BigVGAN v2): mel[B,2,T,64] → waveform[B,2,T*160] at 16kHz
//        conv_pre(128→1536,k=7) → 6× [ConvTranspose1d + 3× AMPBlock1] →
//        act_post(SnakeBeta) → conv_post(24→2,k=7) → clamp(-1,1)
//        Upsample rates: [5,2,2,2,2,2]=160×, channels: 1536→768→384→192→96→48→24
//
//     2. BWE (bandwidth extension): 16kHz → 48kHz
//        mel_stft on vocoder output → bwe_generator (2nd Vocoder, 240×) → + sinc_resample(3×) skip
//        BWE channels: 512→256→128→64→32→16, upsample rates: [6,5,2,2,2]
//
// CRITICAL: Entire forward pass runs in f32 (bf16 causes 40-90% spectral degradation).
// Weight checkpoint stores bf16 — converted to f32 at each op (mimics PyTorch autocast).
//
// Checkpoint keys: vocoder.vocoder.*, vocoder.bwe_generator.*, vocoder.mel_stft.*
// Total: 1227 weight tensors.

// --- Parameter structs ---

/// Conv1d weight + bias.
pub const Conv1dWeight = struct {
    weight: Tensor, // [C_out, C_in, K]
    bias: Tensor, // [C_out]
};

/// Conv1d weight only (no bias).
pub const Conv1dWeightNoBias = struct {
    weight: Tensor, // [C_out, C_in, K]
};

/// ConvTranspose1d weight + bias.
/// PyTorch stores shape [in_ch, out_ch, K] (dims swapped vs Conv1d).
pub const ConvTranspose1dWeight = struct {
    weight: Tensor, // [C_in, C_out, K]
    bias: Tensor, // [C_out]
};

/// SnakeBeta activation: x + (1/exp(beta)) * sin(exp(alpha) * x)²
/// Stored with alpha_logscale=True.
pub const SnakeBetaParams = struct {
    alpha: Tensor, // [C]
    beta: Tensor, // [C]
};

/// Anti-aliased activation: upsample 2× → SnakeBeta → downsample 2×.
/// Kaiser-sinc filters stored in checkpoint.
pub const Activation1dParams = struct {
    act: SnakeBetaParams,
    upsample_filter: Tensor, // [1, 1, 12]
    downsample_filter: Tensor, // [1, 1, 12]
};

/// AMPBlock1: 3 dilated conv pairs with anti-aliased SnakeBeta activations.
/// Each pair: acts1[i](x) → convs1[i] → acts2[i] → convs2[i], residual added.
pub const AMPBlock1Params = struct {
    convs1: [3]Conv1dWeight,
    convs2: [3]Conv1dWeight,
    acts1: [3]Activation1dParams,
    acts2: [3]Activation1dParams,
};

/// STFT as conv1d with precomputed DFT bases (from checkpoint).
pub const STFTParams = struct {
    forward_basis: Tensor, // [n_fft+2, 1, n_fft] = [514, 1, 512]
};

/// Causal log-mel spectrogram: STFT + mel filterbank projection.
pub const MelSTFTParams = struct {
    mel_basis: Tensor, // [n_mels, n_freqs] = [64, 257]
    stft_fn: STFTParams,
};

/// Main vocoder parameters (6 upsample stages, 18 AMPBlock1 resblocks).
pub const MainVocoderParams = struct {
    conv_pre: Conv1dWeight, // 128 → 1536, k=7
    ups: [6]ConvTranspose1dWeight,
    resblocks: [18]AMPBlock1Params, // 6 stages × 3 kernels
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias, // 24 → 2, k=7
};

/// BWE vocoder parameters (5 upsample stages, 15 AMPBlock1 resblocks).
pub const BWEVocoderParams = struct {
    conv_pre: Conv1dWeight, // 128 → 512, k=7
    ups: [5]ConvTranspose1dWeight,
    resblocks: [15]AMPBlock1Params, // 5 stages × 3 kernels
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias, // 16 → 2, k=7
};

/// Top-level VocoderWithBWE parameters.
pub const VocoderWithBWEParams = struct {
    vocoder: MainVocoderParams,
    bwe_generator: BWEVocoderParams,
    mel_stft: MelSTFTParams,
};

// --- Weight loading ---

fn initConv1dWeight(store: zml.io.TensorStore.View) Conv1dWeight {
    return .{
        .weight = store.createTensor("weight", null, null),
        .bias = store.createTensor("bias", null, null),
    };
}

fn initConv1dWeightNoBias(store: zml.io.TensorStore.View) Conv1dWeightNoBias {
    return .{
        .weight = store.createTensor("weight", null, null),
    };
}

fn initConvTranspose1dWeight(store: zml.io.TensorStore.View) ConvTranspose1dWeight {
    return .{
        .weight = store.createTensor("weight", null, null),
        .bias = store.createTensor("bias", null, null),
    };
}

fn initSnakeBetaParams(store: zml.io.TensorStore.View) SnakeBetaParams {
    return .{
        .alpha = store.createTensor("alpha", null, null),
        .beta = store.createTensor("beta", null, null),
    };
}

fn initActivation1dParams(store: zml.io.TensorStore.View) Activation1dParams {
    return .{
        .act = initSnakeBetaParams(store.withPrefix("act")),
        .upsample_filter = store.withPrefix("upsample").createTensor("filter", null, null),
        .downsample_filter = store.withPrefix("downsample").withPrefix("lowpass").createTensor("filter", null, null),
    };
}

fn initAMPBlock1Params(result: *AMPBlock1Params, store: zml.io.TensorStore.View) void {
    inline for (0..3) |i| {
        result.convs1[i] = initConv1dWeight(store.withPrefix("convs1").withLayer(i));
        result.convs2[i] = initConv1dWeight(store.withPrefix("convs2").withLayer(i));
        result.acts1[i] = initActivation1dParams(store.withPrefix("acts1").withLayer(i));
        result.acts2[i] = initActivation1dParams(store.withPrefix("acts2").withLayer(i));
    }
}

pub fn initMainVocoderParams(result: *MainVocoderParams, store: zml.io.TensorStore.View) void {
    result.conv_pre = initConv1dWeight(store.withPrefix("conv_pre"));
    inline for (0..6) |i| {
        result.ups[i] = initConvTranspose1dWeight(store.withPrefix("ups").withLayer(i));
    }
    inline for (0..18) |i| {
        initAMPBlock1Params(&result.resblocks[i], store.withPrefix("resblocks").withLayer(i));
    }
    result.act_post = initActivation1dParams(store.withPrefix("act_post"));
    result.conv_post = initConv1dWeightNoBias(store.withPrefix("conv_post"));
}

fn initBWEVocoderParams(result: *BWEVocoderParams, store: zml.io.TensorStore.View) void {
    result.conv_pre = initConv1dWeight(store.withPrefix("conv_pre"));
    inline for (0..5) |i| {
        result.ups[i] = initConvTranspose1dWeight(store.withPrefix("ups").withLayer(i));
    }
    inline for (0..15) |i| {
        initAMPBlock1Params(&result.resblocks[i], store.withPrefix("resblocks").withLayer(i));
    }
    result.act_post = initActivation1dParams(store.withPrefix("act_post"));
    result.conv_post = initConv1dWeightNoBias(store.withPrefix("conv_post"));
}

fn initMelSTFTParams(store: zml.io.TensorStore.View) MelSTFTParams {
    return .{
        .mel_basis = store.createTensor("mel_basis", null, null),
        .stft_fn = .{
            .forward_basis = store.withPrefix("stft_fn").createTensor("forward_basis", null, null),
        },
    };
}

pub fn initVocoderWithBWEParams(result: *VocoderWithBWEParams, store: zml.io.TensorStore.View) void {
    const voc = store.withPrefix("vocoder");
    initMainVocoderParams(&result.vocoder, voc.withPrefix("vocoder"));
    initBWEVocoderParams(&result.bwe_generator, voc.withPrefix("bwe_generator"));
    result.mel_stft = initMelSTFTParams(voc.withPrefix("mel_stft"));
}

// --- Forward ops ---

/// Ensure tensor is f32 (for vocoder precision requirement).
fn ensureF32(t: Tensor) Tensor {
    return if (t.dtype() == .f32) t else t.convert(.f32);
}

/// Replicate-pad along the last dimension (spatial dim for 1D signals).
/// Input [B, C, T] → output [B, C, pad_left + T + pad_right].
fn replicatePad1d(x: Tensor, pad_left: i64, pad_right: i64) Tensor {
    const spatial_axis = x.rank() - 1;
    const t_dim = x.dim(spatial_axis);
    // First element replicated pad_left times
    const first = x.slice1d(spatial_axis, .{ .end = 1 }); // [B, C, 1]
    const first_pad = first.broad(x.shape().set(spatial_axis, pad_left));
    // Last element replicated pad_right times
    const last = x.slice1d(spatial_axis, .{ .start = t_dim - 1 }); // [B, C, 1]
    const last_pad = last.broad(x.shape().set(spatial_axis, pad_right));
    return Tensor.concatenate(&.{ first_pad, x, last_pad }, spatial_axis);
}

/// Conv1d forward with f32 conversion, input [B, C_in, T], weight [C_out, C_in, K], bias [C_out].
fn forwardVocConv1d(input: Tensor, w: Conv1dWeight, opts: struct {
    padding: i64 = 0,
    dilation: i64 = 1,
}) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight);
    const bias = ensureF32(w.bias);
    const result = x.conv1d(weight, .{
        .padding = &.{ opts.padding, opts.padding },
        .rhs_dilation = opts.dilation,
    });
    // Add bias: reshape [C_out] → [1, C_out, 1]
    return result.add(bias.reshape(result.shape().set(0, 1).set(2, 1)));
}

/// Conv1d forward without bias.
fn forwardVocConv1dNoBias(input: Tensor, w: Conv1dWeightNoBias, opts: struct {
    padding: i64 = 0,
}) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight);
    return x.conv1d(weight, .{
        .padding = &.{ opts.padding, opts.padding },
    });
}

/// ConvTranspose1d forward with f32 conversion.
/// PyTorch weight [in_ch, out_ch, K] → MLIR: swap kernel dims, use lhs_dilation=stride, explicit kernel flip.
/// PyTorch padding p → MLIR padding = K - p - 1 on each side.
fn forwardVocConvTranspose1d(input: Tensor, w: ConvTranspose1dWeight, stride: i64, pytorch_padding: i64) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight).reverse(.{2}); // flip kernel along spatial dim
    const bias = ensureF32(w.bias);
    const k = weight.dim(2);
    const mlir_pad = k - pytorch_padding - 1;
    const result = x.conv1d(weight, .{
        .lhs_dilation = stride,
        .padding = &.{ mlir_pad, mlir_pad },
        .kernel_output_feature_dimension = 1,
        .kernel_input_feature_dimension = 0,
    });
    return result.add(bias.reshape(result.shape().set(0, 1).set(2, 1)));
}

/// SnakeBeta activation: x + (1/exp(beta)) * sin(exp(alpha) * x)²
/// alpha_logscale=True: alpha, beta are in log-space.
fn forwardSnakeBeta(x: Tensor, params: SnakeBetaParams) Tensor {
    const alpha = ensureF32(params.alpha).exp().reshape(x.shape().set(0, 1).set(2, 1)); // [1, C, 1]
    const beta = ensureF32(params.beta).exp().reshape(x.shape().set(0, 1).set(2, 1)); // [1, C, 1]
    const eps: f32 = 1e-9;
    // x + (1 / (beta + eps)) * sin(alpha * x)²
    const sin_val = x.mul(alpha).sin();
    return x.add(sin_val.mul(sin_val).div(beta.addConstant(eps)));
}

/// UpSample1d: replicate-pad → depthwise conv_transpose1d → scale → trim.
/// Kaiser-sinc filter from checkpoint [1, 1, 12], expanded to [C, 1, 12] for grouped conv.
/// Ratio=2, kernel_size=12, pad=5, pad_left=15, pad_right=15.
fn forwardUpSample1d(x: Tensor, filter: Tensor) Tensor {
    const ratio: i64 = 2;
    const kernel_size: i64 = 12;
    const pad: i64 = kernel_size / ratio - 1; // 5
    const pad_left_trim: i64 = pad * ratio + @divTrunc(kernel_size - ratio, 2); // 15
    const pad_right_trim: i64 = pad * ratio + @divTrunc(kernel_size - ratio + 1, 2); // 15

    // 1. Replicate-pad input by 'pad' on each side
    var y = replicatePad1d(x, pad, pad);

    // 2. Depthwise ConvTranspose1d: expand filter [1,1,12] → [C,1,12], stride=2, groups=C
    const n_channels = y.dim(1);
    const filt = ensureF32(filter).reverse(.{2}).broad(filter.shape().set(0, n_channels)); // [C, 1, 12] flipped
    // For depthwise transposed conv: kernel_output=0 (C), kernel_input=1 (1), feature_group_count=C
    // PyTorch ConvTranspose1d(groups=C): no padding → MLIR padding = K - 0 - 1 = 11
    const mlir_pad = kernel_size - 1;
    y = ensureF32(y).conv1d(filt, .{
        .lhs_dilation = ratio,
        .padding = &.{ mlir_pad, mlir_pad },
        .feature_group_count = n_channels,
    });

    // 3. Scale by ratio
    y = y.scale(ratio);

    // 4. Trim padded edges
    const t_out = y.dim(2);
    y = y.slice1d(2, .{ .start = pad_left_trim, .end = t_out - pad_right_trim });
    return y;
}

/// DownSample1d via LowPassFilter1d: replicate-pad → depthwise conv1d with stride.
/// Kaiser-sinc filter from checkpoint [1, 1, 12], expanded to [C, 1, 12].
/// Ratio=2, kernel_size=12, pad_left=5 (even: k//2 - 1), pad_right=6 (k//2).
fn forwardDownSample1d(x: Tensor, filter: Tensor) Tensor {
    const stride: i64 = 2;
    const pad_left: i64 = 5; // kernel_size // 2 - 1 (even kernel)
    const pad_right: i64 = 6; // kernel_size // 2

    // 1. Replicate-pad
    var y = replicatePad1d(x, pad_left, pad_right);

    // 2. Depthwise conv1d with stride=2
    const n_channels = y.dim(1);
    const filt = ensureF32(filter).broad(filter.shape().set(0, n_channels)); // [C, 1, 12]
    y = ensureF32(y).conv1d(filt, .{
        .window_strides = stride,
        .feature_group_count = n_channels,
    });
    return y;
}

/// Activation1d: upsample 2× → SnakeBeta → downsample 2× (anti-aliased activation).
fn forwardActivation1d(x: Tensor, params: Activation1dParams) Tensor {
    var y = forwardUpSample1d(x, params.upsample_filter);
    y = forwardSnakeBeta(y, params.act);
    y = forwardDownSample1d(y, params.downsample_filter);
    return y;
}

/// AMPBlock1 forward: 3 dilated conv pairs with residual connections.
/// For each pair i: xt = acts1[i](x) → convs1[i](xt) → acts2[i](xt) → convs2[i](xt); x = x + xt.
fn forwardAMPBlock1(x_in: Tensor, params: AMPBlock1Params, dilations: [3]i64) Tensor {
    var x = x_in;
    inline for (0..3) |i| {
        var xt = forwardActivation1d(x, params.acts1[i]);
        xt = forwardVocConv1d(xt, params.convs1[i], .{
            .padding = @divTrunc(params.convs1[i].weight.dim(2) * dilations[i] - dilations[i], 2),
            .dilation = dilations[i],
        });
        xt = forwardActivation1d(xt, params.acts2[i]);
        xt = forwardVocConv1d(xt, params.convs2[i], .{
            .padding = @divTrunc(params.convs2[i].weight.dim(2) - 1, 2),
        });
        x = x.add(xt);
    }
    return x;
}

/// Vocoder forward (shared by main vocoder and BWE generator).
/// Input: mel [B, 2, T, 64] → rearrange to [B, 128, T] → upsample → waveform [B, 2, T_out].
fn forwardVocoderGeneric(
    mel: Tensor,
    conv_pre: Conv1dWeight,
    ups: anytype,
    resblocks: anytype,
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias,
    comptime num_ups: usize,
    apply_final_activation: bool,
) Tensor {
    // Rearrange [B, 2, T, 64] → [B, 128, T]
    // First transpose to [B, 2, 64, T], then reshape to [B, 128, T]
    var x = ensureF32(mel).transpose(.{ 0, 1, 3, 2 }); // [B, 2, T, 64] → [B, 2, 64, T]
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) }); // [B, 128, T]

    // conv_pre (k=7, pad=3)
    x = forwardVocConv1d(x, conv_pre, .{ .padding = 3 });

    // Upsample stages: each stage has 1 ConvTranspose1d + num_kernels AMPBlock1 resblocks
    const num_kernels: usize = 3;
    const dilations = [3]i64{ 1, 3, 5 };

    inline for (0..num_ups) |i| {
        // ConvTranspose1d upsample
        // stride = kernel_size // 2 (holds for this checkpoint: [11→5, 4→2, 12→6, 11→5])
        const k = ups[i].weight.dim(2);
        const stride = @divTrunc(k, 2);
        const pytorch_padding = @divTrunc(k - stride, 2);
        x = forwardVocConvTranspose1d(x, ups[i], stride, pytorch_padding);

        // AMPBlock1 resblocks: evaluate all 3 kernel variants, average their outputs
        const start = i * num_kernels;
        var block_sum = forwardAMPBlock1(x, resblocks[start], dilations);
        inline for (1..num_kernels) |j| {
            block_sum = block_sum.add(forwardAMPBlock1(x, resblocks[start + j], dilations));
        }
        x = block_sum.divByConst(num_kernels);
    }

    // Final activation + conv
    x = forwardActivation1d(x, act_post);
    x = forwardVocConv1dNoBias(x, conv_post, .{ .padding = 3 });

    if (apply_final_activation) {
        x = x.clamp(Tensor.scalar(-1.0, .f32), Tensor.scalar(1.0, .f32));
    }

    return x;
}

/// Causal STFT via conv1d with precomputed DFT bases.
/// Input: waveform [B, T_samples], output: magnitude [B, n_freqs, T_frames].
/// Causal: left-only padding of (win_length - hop_length) samples.
fn forwardSTFT(y_in: Tensor, params: STFTParams) Tensor {
    const hop_length: i64 = 80;
    const win_length: i64 = 512;

    // Add channel dim: [B, T] → [B, 1, T]
    var y = ensureF32(y_in).reshape(.{ y_in.dim(0), 1, y_in.dim(1) });

    // Causal left-only padding (prepend zeros)
    const left_pad = win_length - hop_length; // 432
    const left_zeros = Tensor.zeroes(y.shape().set(2, left_pad));
    y = Tensor.concatenate(&.{ left_zeros, y }, 2);

    // Conv1d with precomputed DFT bases [514, 1, 512], stride=hop_length
    const basis = ensureF32(params.forward_basis); // [514, 1, 512]
    const spec = y.conv1d(basis, .{ .window_strides = hop_length }); // [B, 514, T_frames]

    // Split into real and imaginary: first 257 and last 257 channels
    const n_freqs = @divTrunc(spec.dim(1), 2); // 257
    const real = spec.slice1d(1, .{ .end = n_freqs }); // [B, 257, T_frames]
    const imag = spec.slice1d(1, .{ .start = n_freqs }); // [B, 257, T_frames]

    // magnitude = sqrt(real² + imag²)
    return real.mul(real).add(imag.mul(imag)).sqrt();
}

/// Compute log-mel spectrogram from magnitude spectrogram.
/// magnitude [B, n_freqs, T_frames] → log_mel [B, n_mels, T_frames].
fn forwardMelProjection(magnitude: Tensor, mel_basis: Tensor) Tensor {
    // mel = mel_basis @ magnitude
    // mel_basis [n_mels=64, n_freqs=257], magnitude [B, n_freqs=257, T_frames]
    // Use conv1d with kernel_size=1 to implement matmul:
    // Treat mel_basis [64, 257] as conv1d kernel [64, 257, 1]
    const basis = ensureF32(mel_basis).reshape(.{ mel_basis.dim(0), mel_basis.dim(1), 1 }); // [64, 257, 1]
    const mel = ensureF32(magnitude).conv1d(basis, .{}); // [B, 64, T_frames]

    // log(clamp(mel, min=1e-5))
    const clamped = mel.clamp(Tensor.scalar(1e-5, .f32), Tensor.scalar(1e30, .f32));
    return clamped.log();
}

/// Compute causal log-mel spectrogram from stereo waveform.
/// audio [B, 2, T_samples] → mel [B, 2, n_mels, T_frames].
fn forwardComputeMel(audio: Tensor, mel_stft: MelSTFTParams) Tensor {
    const batch = audio.dim(0);
    const n_channels = audio.dim(1); // 2
    const t_samples = audio.dim(2);

    // Flatten: [B, 2, T] → [B*2, T]
    const flat = audio.reshape(.{ batch * n_channels, t_samples });

    // STFT → magnitude [B*2, 257, T_frames]
    const magnitude = forwardSTFT(flat, mel_stft.stft_fn);

    // Mel projection → log_mel [B*2, 64, T_frames]
    const log_mel = forwardMelProjection(magnitude, mel_stft.mel_basis);

    // Reshape back: [B*2, 64, T_frames] → [B, 2, 64, T_frames]
    return log_mel.reshape(.{ batch, n_channels, log_mel.dim(1), log_mel.dim(2) });
}

/// Kaiser-windowed sinc resampler for BWE skip connection.
/// Upsamples by ratio=3 (16kHz → 48kHz). Filter not in checkpoint — computed here.
///
/// UpSample1d(ratio=3, window_type="kaiser"):  (kaiser is the default)
///   kernel_size=18, pad=5, pad_left=22, pad_right=23
fn forwardSincResample3x(x: Tensor) Tensor {
    const ratio: i64 = 3;
    const kernel_size: i64 = 18;
    const pad: i64 = 5;
    const pad_left_trim: i64 = 22;
    const pad_right_trim: i64 = 23;

    // Precomputed Kaiser-windowed sinc filter (18 taps, ratio=3)
    // Generated from: UpSample1d(ratio=3, window_type="kaiser")
    //   cutoff=0.5/ratio, half_width=0.6/ratio, kernel_size=18
    //   filter.sum() ≈ 1.0, filter.sum()*ratio ≈ 3.0
    const filter_data = [18]f32{
        7.0040696301e-04,  4.6405289322e-03,  5.3038536571e-03,  -1.0276262648e-02,
        -3.6353457719e-02, -3.0759338289e-02, 5.2384063601e-02,  1.9813767076e-01,
        3.1622257829e-01,  3.1622257829e-01,  1.9813767076e-01,  5.2384063601e-02,
        -3.0759338289e-02, -3.6353457719e-02, -1.0276262648e-02, 5.3038536571e-03,
        4.6405289322e-03,  7.0040696301e-04,
    };

    // 1. Replicate-pad
    var y = replicatePad1d(ensureF32(x), pad, pad);

    // 2. Depthwise conv_transpose1d with sinc filter
    const n_channels = y.dim(1);
    // Create constant filter tensor [1, 1, 18] then broadcast to [C, 1, 18]
    const filter_shape = zml.Shape.init(.{ 1, 1, kernel_size }, .f32);
    const filt_1 = Tensor.constantTensor(filter_shape, std.mem.sliceAsBytes(&filter_data));
    const filt = filt_1.reverse(.{2}).broad(filter_shape.set(0, n_channels)); // [C, 1, 18] flipped (symmetric)

    // MLIR padding for depthwise transposed conv: kernel_size - 1 = 42
    const mlir_pad = kernel_size - 1;
    y = y.conv1d(filt, .{
        .lhs_dilation = ratio,
        .padding = &.{ mlir_pad, mlir_pad },
        .feature_group_count = n_channels,
    });

    // 3. Scale by ratio
    y = y.scale(ratio);

    // 4. Trim
    const t_out = y.dim(2);
    y = y.slice1d(2, .{ .start = pad_left_trim, .end = t_out - pad_right_trim });
    return y;
}

/// BWE pipeline parameters (bwe_generator + mel_stft). 559 tensors total.
pub const BWEPipelineParams = struct {
    bwe_generator: BWEVocoderParams,
    mel_stft: MelSTFTParams,
};

pub fn initBWEPipelineParams(result: *BWEPipelineParams, store: zml.io.TensorStore.View) void {
    const voc = store.withPrefix("vocoder");
    initBWEVocoderParams(&result.bwe_generator, voc.withPrefix("bwe_generator"));
    result.mel_stft = initMelSTFTParams(voc.withPrefix("mel_stft"));
}

/// Stage 1: Main vocoder — mel [B, 2, T, 64] → waveform [B, 2, T*160] at 16kHz.
/// 667 tensor parameters (+ 1 input = 668 MLIR args; well under 1024 limit).
pub fn forwardMainVocoder(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    return forwardVocoderGeneric(
        mel_spec,
        params.conv_pre,
        params.ups,
        params.resblocks,
        params.act_post,
        params.conv_post,
        6,
        true, // apply clamp(-1,1)
    );
}

/// Debug: rearrange + conv_pre + ups[0] for isolating numerical issues.
pub fn forwardAfterUps0(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    var x = ensureF32(mel_spec).transpose(.{ 0, 1, 3, 2 });
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) });
    x = forwardVocConv1d(x, params.conv_pre, .{ .padding = 3 });

    // ups[0]: ConvTranspose1d
    const k = params.ups[0].weight.dim(2);
    const stride = @divTrunc(k, 2);
    const pytorch_padding = @divTrunc(k - stride, 2);
    x = forwardVocConvTranspose1d(x, params.ups[0], stride, pytorch_padding);
    return x;
}

/// Debug: rearrange + conv_pre + ups[0] + first resblock stage.
pub fn forwardAfterStage0(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    var x = ensureF32(mel_spec).transpose(.{ 0, 1, 3, 2 });
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) });
    x = forwardVocConv1d(x, params.conv_pre, .{ .padding = 3 });

    // ups[0]
    const k = params.ups[0].weight.dim(2);
    const stride = @divTrunc(k, 2);
    const pytorch_padding = @divTrunc(k - stride, 2);
    x = forwardVocConvTranspose1d(x, params.ups[0], stride, pytorch_padding);

    // resblocks 0,1,2 → mean
    const dilations = [3]i64{ 1, 3, 5 };
    var block_sum = forwardAMPBlock1(x, params.resblocks[0], dilations);
    block_sum = block_sum.add(forwardAMPBlock1(x, params.resblocks[1], dilations));
    block_sum = block_sum.add(forwardAMPBlock1(x, params.resblocks[2], dilations));
    x = block_sum.divByConst(3);
    return x;
}

/// Stage 2: BWE pipeline — 16kHz waveform → 48kHz waveform.
/// Takes the main vocoder output, computes mel-STFT, runs BWE generator,
/// adds sinc-resampled skip connection, clamps to [-1, 1].
/// 559 tensor parameters (+ 1 input = 560 MLIR args; well under 1024 limit).
pub fn forwardBWEPipeline(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const input_sr: i64 = 16000;
    const output_sr: i64 = 48000;
    const hop_length: i64 = 80;
    const sr_ratio = @divTrunc(output_sr, input_sr); // 3

    var x = waveform_16k;
    const length_low_rate = x.dim(2);
    const output_length = length_low_rate * sr_ratio;

    // 1. Pad vocoder output to multiple of hop_length for exact mel frame count
    const remainder = @mod(length_low_rate, hop_length);
    const pad_amount = if (remainder != 0) hop_length - remainder else 0;
    if (pad_amount > 0) {
        const right_zeros = Tensor.zeroes(x.shape().set(2, pad_amount));
        x = Tensor.concatenate(&.{ x, right_zeros }, 2);
    }

    // 2. Compute mel spectrogram from vocoder output: [B, 2, n_mels, T_frames]
    const mel = forwardComputeMel(x, params.mel_stft);

    // 3. Vocoder.forward expects [B, 2, T_frames, mel_bins] — transpose
    const mel_for_bwe = mel.transpose(.{ 0, 1, 3, 2 });

    // 4. BWE generator: mel → residual waveform [B, 2, T_bwe]
    const residual = forwardVocoderGeneric(
        mel_for_bwe,
        params.bwe_generator.conv_pre,
        params.bwe_generator.ups,
        params.bwe_generator.resblocks,
        params.bwe_generator.act_post,
        params.bwe_generator.conv_post,
        5,
        false, // no final activation
    );

    // 5. Sinc-resample vocoder output by 3× for skip connection
    const skip = forwardSincResample3x(x);

    // 6. Add residual + skip, clamp, trim to output length
    var output = residual.add(skip);
    output = output.clamp(Tensor.scalar(-1.0, .f32), Tensor.scalar(1.0, .f32));
    output = output.slice1d(2, .{ .end = output_length });

    return output;
}

// ====================================================================
// Debug forward functions for BWE pipeline bisection
// ====================================================================

/// Helper: pad waveform to multiple of hop_length (shared by BWE debug fns).
fn bwePadInput(waveform_16k: Tensor) Tensor {
    const hop_length: i64 = 80;
    var x = waveform_16k;
    const length_low_rate = x.dim(2);
    const remainder = @mod(length_low_rate, hop_length);
    const pad_amount = if (remainder != 0) hop_length - remainder else 0;
    if (pad_amount > 0) {
        const right_zeros = Tensor.zeroes(x.shape().set(2, pad_amount));
        x = Tensor.concatenate(&.{ x, right_zeros }, 2);
    }
    return x;
}

/// Debug: BWE compute mel only — returns log-mel [B, 2, n_mels, T_frames] (before transpose).
pub fn forwardBWEComputeMel(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const x = bwePadInput(waveform_16k);
    return forwardComputeMel(x, params.mel_stft);
}

/// Debug: BWE sinc resample skip only — returns [B, 2, T_skip].
pub fn forwardBWESincSkip(waveform_16k: Tensor) Tensor {
    const x = bwePadInput(waveform_16k);
    return forwardSincResample3x(x);
}

/// Debug: BWE residual only (mel → BWE generator) — returns [B, 2, T_bwe].
pub fn forwardBWEResidual(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const x = bwePadInput(waveform_16k);
    const mel = forwardComputeMel(x, params.mel_stft);
    const mel_for_bwe = mel.transpose(.{ 0, 1, 3, 2 });
    return forwardVocoderGeneric(
        mel_for_bwe,
        params.bwe_generator.conv_pre,
        params.bwe_generator.ups,
        params.bwe_generator.resblocks,
        params.bwe_generator.act_post,
        params.bwe_generator.conv_post,
        5,
        false,
    );
}
