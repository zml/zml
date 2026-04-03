# ZML window_reversal Bug

## Issue
`window_reversal = true` in ZML's `Tensor.conv1d()` does NOT correctly flip the convolution kernel.
When simulating PyTorch's `ConvTranspose1d` via `conv1d` with `lhs_dilation`, using `window_reversal = true` produces wrong results.

## Workaround
Explicitly flip the kernel before the convolution:
```zig
const weight_flipped = weight.reverse(.{2}); // flip spatial dim
const result = x.conv1d(weight_flipped, .{
    .lhs_dilation = stride,
    .padding = &.{ mlir_pad, mlir_pad },
    // NO window_reversal
    .kernel_output_feature_dimension = 1,
    .kernel_input_feature_dimension = 0,
});
```

## Verified
- Confirmed by comparing Python F.conv_transpose1d output against manual dilated conv
- With flip: max diff 0.00008 (match)
- Without flip: max diff 1.35 (mismatch)
- After fix: Stage 1 vocoder went from 16 dB to 63.74 dB PSNR

## Affected Sites in LTX Vocoder
1. `forwardVocConvTranspose1d` — vocoder/BWE upsample layers
2. `forwardUpSample1d` — anti-aliased activation upsampling (Activation1d)
3. `forwardSincResample3x` — BWE skip connection resampler
