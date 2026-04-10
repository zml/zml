# Noise Generation & Sigma Schedule Migration

Migrate noise initialization and sigma schedule from Python export into the Zig
unified pipeline, eliminating two runtime dependencies on Python-generated data.

## Summary of changes

### Task 1: Sigma schedule hardcoding

**Before**: Stage 2 distilled sigmas were exported in `pipeline_meta.json` and
parsed at runtime. Stage 1 sigmas were also exported (but never used by Zig).

**After**: Both sigma schedules are computed/hardcoded in Zig:
- Stage 1: computed from `computeSigmaSchedule()` (already existed)
- Stage 2: `stage2_distilled_sigmas = [4]f32{ 0.909375, 0.725, 0.421875, 0.0 }`
  hardcoded in `model.zig` (matches Python's `STAGE_2_DISTILLED_SIGMA_VALUES`)

Removed from `pipeline_meta.json`: `sigmas`, `sigma_0` fields.
`PipelineMeta` no longer needs an allocator or `deinit()`.

### Task 2: Noise generation via Box-Muller

**Before**: All noise tensors were pre-generated in Python (`torch.randn`) and
exported as safetensors files (`stage1_inputs.safetensors` contained pre-noised
latents, `stage2_noise.safetensors` for Stage 2).

**After**: Noise is generated in Zig using `forwardGenerateNoise()`:
- Uses `Tensor.Rng` (StableHLO `rng_bit_generator`) for deterministic seeded RNG
- Box-Muller transform: `Z = sqrt(-2 * ln(U1)) * cos(2π * U2)` converts two
  uniform samples to N(0,1)
- RNG state is threaded: seed → draw #1 (S1 video) → draw #2 (S1 audio) →
  draw #3 (S2 video) → draw #4 (S2 audio)
- `forwardNoiseInit()` mixes noise with clean latent using the denoise mask

### CLI changes

- **Removed**: `--stage2-noise <path>` (no longer needed)
- **Added**: `--seed <int>` (optional, default 42) — controls RNG seed for all
  noise generation

## Files changed

### `model.zig`

- Added `forwardGenerateNoise()` — Box-Muller noise generation (~15 lines)
- Added `stage2_distilled_sigmas` constant

### `inference.zig`

- `CliArgs`: removed `stage2_noise`, added `seed` (default 42)
- `runStage1()`: accepts seed, generates noise via `forwardGenerateNoise`,
  runs `forwardNoiseInit` with `noise_scale=1.0`, returns `rng_state` buffer
- `runBridge()`: generates Stage 2 noise from RNG state (no file loading)
- `--dump-intermediates` now saves noise buffers:
  `s1_video_noise.bin`, `s1_audio_noise.bin`, `s2_video_noise.bin`, `s2_audio_noise.bin`

### `export_pipeline.py`

- Removed `sigmas`, `sigma_0` from `pipeline_meta.json` output

## Noise validation

Zig-generated noise verified as correct N(0,1) on GPU (seed=42):

| Buffer | Shape | mean | std | min | max | kurtosis |
|--------|-------|------|-----|-----|-----|----------|
| s1_video_noise | [1, 6144, 128] | -0.002 | 0.999 | -4.63 | 5.00 | 2.99 |
| s1_audio_noise | [1, 126, 128] | 0.003 | 0.993 | -3.59 | 3.80 | 3.03 |
| s2_video_noise | [1, 24576, 128] | 0.001 | 1.000 | -5.28 | 5.13 | 3.00 |
| s2_audio_noise | [1, 126, 128] | 0.002 | 0.998 | -4.16 | 3.58 | 2.96 |

Python reference noise (same prompt, different seed/RNG):

| Buffer | Shape | mean | std | min | max | kurtosis |
|--------|-------|------|-----|-----|-----|----------|
| video_noise_s2 | [1, 24576, 128] | -0.001 | 0.968 | -5.25 | 4.69 | 3.20 |
| audio_noise_s2 | [1, 126, 128] | -0.009 | 1.003 | -4.34 | 3.80 | 2.97 |

Both produce proper Gaussian noise. The Zig RNG (StableHLO `rng_bit_generator`)
and Python's `torch.Generator` use different algorithms, so identical seeds
produce different random values — this is expected.

### Analysis gotcha: bf16 vs fp16

The noise buffers are stored in **bf16** (bfloat16). When analyzing raw `.bin`
dumps, use `torch.bfloat16` — NOT `numpy.float16` (which is IEEE fp16).
Reading bf16 bytes as fp16 produces bogus statistics (std≈1.77, kurtosis≈1.04)
because the bit layouts differ:

- **bf16**: 1 sign + 8 exponent + 7 mantissa bits
- **fp16**: 1 sign + 5 exponent + 10 mantissa bits

Correct analysis script:

```python
import torch

def stats(path, shape, name):
    data = open(path, 'rb').read()
    t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).float()
    kurt = ((t - t.mean())**4).mean() / t.std()**4
    print(f"{name}: mean={t.mean():.6f} std={t.std():.6f} min={t.min():.4f} max={t.max():.4f} kurtosis={kurt:.4f}")
```

## Usage

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
    --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --stage1-inputs /root/e2e_demo/stage1_inputs.safetensors \
    --meta /root/e2e_demo/pipeline_meta.json \
    --output-dir /root/e2e_demo/output/ \
    --seed 42 \
    --bf16-attn-stage2
```

To dump noise for analysis, add `--dump-intermediates`.
