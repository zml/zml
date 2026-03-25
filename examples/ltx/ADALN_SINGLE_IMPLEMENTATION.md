# AdaLayerNormSingle Implementation

## Overview

Implementation and parity verification of all 8 `AdaLayerNormSingle` modules from the LTX-2.3 22B distilled model in Zig/ZML.

**Date**: 2026-03-25  
**Checkpoint**: `/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors`  
**Checkpoint prefix**: `model.diffusion_model.*`

---

## Architecture

The `AdaLayerNormSingle` module conditions the transformer on the diffusion timestep via a small MLP applied to a sinusoidal embedding.

### Full chain

```
sigma ∈ [0, 1]
  → × 1000                          (timestep_scale_multiplier)
  → sinusoidal_embedding(d=256)     (half=128 frequencies)
  → cast(bf16)
  → linear_1(d_sin=256 → d)  + silu
  → linear_2(d → d)          + silu   ← embedded_timestep captured here (before silu+linear_out)
  → linear_out(d → N×d)
  → modulation  [B, N×d]
```

### Sinusoidal embedding formula

```
half = 128
freq[i] = exp(−ln(10000) × i / 128),  i ∈ [0, 128)
emb[b, i] = sigma_scaled[b] × freq[i]
output = cat([cos(emb), sin(emb)], axis=-1)  →  shape [B, 256]
```
`flip_sin_to_cos=True` means cosine comes first in the output.

### Captured tensors

| Tensor | When | Shape |
|--------|------|-------|
| `embedded_timestep` | after `linear_2`, before final `silu+linear_out` | `[B, d]` |
| `modulation` | after `linear_out` | `[B, N×d]` |

---

## Module Inventory

All 8 adaln modules and their output dimensions:

| Module field | Checkpoint suffix | `d` (hidden) | `N` | `N×d` (linear_out) |
|---|---|---|---|---|
| `adaln_single` | `adaln_single` | 4096 | 9 | 36864 |
| `audio_adaln_single` | `audio_adaln_single` | 2048 | 9 | 18432 |
| `prompt_adaln_single` | `prompt_adaln_single` | 4096 | 2 | 8192 |
| `audio_prompt_adaln_single` | `audio_prompt_adaln_single` | 2048 | 2 | 4096 |
| `av_ca_video_scale_shift_adaln_single` | `av_cross_attention.video_scale_shift_adaln_single` | 4096 | 4 | 16384 |
| `av_ca_audio_scale_shift_adaln_single` | `av_cross_attention.audio_scale_shift_adaln_single` | 2048 | 4 | 8192 |
| `av_ca_a2v_gate_adaln_single` | `av_cross_attention.a2v_gate_adaln_single` | 4096 | 1 | 4096 |
| `av_ca_v2a_gate_adaln_single` | `av_cross_attention.v2a_gate_adaln_single` | 2048 | 1 | 2048 |

---

## Implementation

### New code in `model.zig`

**`sinusoidalTimestepEmbedding`** (pure math, no weights):
```zig
fn sinusoidalTimestepEmbedding(sigma: Tensor) Tensor {
    // half=128, freq[i] = exp(-ln(10000) * i / 128)
    const log_max = comptime std.math.log(f64, std.math.e, 10000.0);
    const idx = Tensor.arange(.{ .end = half }, .f32);
    const freq = idx.scale(-log_max / 128.0).exp();
    // outer product, then cat([cos, sin])
    return Tensor.concatenate(&.{ emb.cos(), emb.sin() }, .hf)
                 .withPartialTags(.{ .b, .d_sin });
}
```

**`AdaLayerNormSingle` struct**:
```zig
pub const AdaLayerNormSingle = struct {
    pub const Params = struct {
        linear_1: zml.nn.Linear,    // d_sin=256 → d
        linear_2: zml.nn.Linear,    // d → d_emb
        linear_out: zml.nn.Linear,  // d_emb → d_ada (N×d)
    };
    pub const ForwardResult = struct { modulation: Tensor, embedded_timestep: Tensor };
    pub fn forward(sigma: Tensor, params: Params) ForwardResult { ... }
};
```

**`forwardAdalnSingle`** free function (for compilation entry point):
```zig
pub fn forwardAdalnSingle(sigma: Tensor, params: AdaLayerNormSingle.Params) AdaLayerNormSingle.ForwardResult
```

**`LTXModel.Params`** — 8 new fields added with full `initParams` / `unloadBuffers` wiring.

**`selectTransformerRoot`** — changed to `pub fn` to expose it to checker programs.

---

## Parity Verification

### Fixture export

**Script**: `examples/ltx/export_adaln_single_fixture.py`  
**Output**: `/root/repos/LTX-2/trace_run/adaln_single_fixture.safetensors`

Test inputs (same for all 8 modules):
```python
sigma_scaled = torch.tensor([0.0, 128.0, 500.0, 1000.0], dtype=torch.float32)
# already pre-multiplied by 1000
```

Tensors saved (3 per module, 24 total):
- `{prefix}.sigma_scaled` — f32 [4]
- `{prefix}.modulation` — bf16 [4, N×d]
- `{prefix}.embedded_timestep` — bf16 [4, d]

### Zig checker

**Target**: `//examples/ltx:adaln_single_check`  
**Source**: `examples/ltx/adaln_single_check.zig`

Pass criteria per module:
- `close_fraction >= 0.999`
- `mean_abs_error < 0.1`

### Run command

```bash
bazel run //examples/ltx:adaln_single_check -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/repos/LTX-2/trace_run/adaln_single_fixture.safetensors
```

---

## Results (2026-03-25)

All 8 modules **PASSED**. Results: **8 passed, 0 failed**.

| Module | Tensor | mean_abs | max_abs | cos_sim | close |
|---|---|---|---|---|---|
| `adaln_single` | modulation | 0.00022 | 0.03125 | 0.999997 | 1.0000 |
| `adaln_single` | embedded_timestep | 0.00002 | 0.00781 | 0.999997 | 1.0000 |
| `audio_adaln_single` | modulation | 0.00036 | 0.03125 | 0.999998 | 1.0000 |
| `audio_adaln_single` | embedded_timestep | 0.00005 | 0.00781 | 0.999998 | 1.0000 |
| `prompt_adaln_single` | modulation | 0.00019 | 0.00195 | 0.999994 | 1.0000 |
| `prompt_adaln_single` | embedded_timestep | 0.00023 | 0.00195 | 0.999990 | 1.0000 |
| `audio_prompt_adaln_single` | modulation | 0.00010 | 0.00098 | 0.999991 | 1.0000 |
| `audio_prompt_adaln_single` | embedded_timestep | 0.00017 | 0.00098 | 0.999990 | 1.0000 |
| `av_ca_video_scale_shift_adaln_single` | modulation | 0.00082 | 0.12500 | 0.999999 | 1.0000 |
| `av_ca_video_scale_shift_adaln_single` | embedded_timestep | 0.00021 | 0.03125 | 0.999999 | 1.0000 |
| `av_ca_audio_scale_shift_adaln_single` | modulation | 0.00074 | 0.12500 | 0.999998 | 1.0000 |
| `av_ca_audio_scale_shift_adaln_single` | embedded_timestep | 0.00016 | 0.00781 | 0.999998 | 1.0000 |
| `av_ca_a2v_gate_adaln_single` | modulation | 0.00007 | 0.00195 | 0.999999 | 1.0000 |
| `av_ca_a2v_gate_adaln_single` | embedded_timestep | 0.00005 | 0.00391 | 0.999996 | 1.0000 |
| `av_ca_v2a_gate_adaln_single` | modulation | 0.00002 | 0.00098 | 0.999999 | 1.0000 |
| `av_ca_v2a_gate_adaln_single` | embedded_timestep | 0.00002 | 0.00049 | 0.999997 | 1.0000 |

Errors are within expected bf16 rounding tolerance (bf16 has ~1/128 ≈ 0.008 precision at values near 1).

---

## Live-Pipeline Validation (2026-03-25)

All 8 modules re-verified using **real denoising-loop activations** captured from a full `TI2VidTwoStagesPipeline` forward pass (step 0, token-limited to 512).

**Script**: `examples/ltx/export_live_capture_fixture.py`  
**Output**: `/root/repos/LTX-2/trace_run/live_capture_fixture_step_000_t512.safetensors`  
**Checker**: `//examples/ltx:live_capture_check`

```bash
bazel run //examples/ltx:live_capture_check -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/repos/LTX-2/trace_run/live_capture_fixture_step_000_t512.safetensors
```

Results: **8/8 passed** (10/10 total including OutputProjection).

| Module | modulation cos_sim | embedded_timestep cos_sim | close |
|---|---|---|---|
| `adaln_single` | 0.999996 | 0.999995 | 1.0000 |
| `audio_adaln_single` | 0.999996 | 0.999996 | 1.0000 |
| `prompt_adaln_single` | 0.999991 | 0.999986 | 1.0000 |
| `audio_prompt_adaln_single` | 0.999991 | 0.999982 | 1.0000 |
| `av_ca_video_scale_shift_adaln_single` | 0.999997 | 0.999994 | 1.0000 |
| `av_ca_audio_scale_shift_adaln_single` | 0.999996 | 0.999995 | 1.0000 |
| `av_ca_a2v_gate_adaln_single` | 0.999997 | 0.999996 | 1.0000 |
| `av_ca_v2a_gate_adaln_single` | 0.999997 | 0.999996 | 1.0000 |

---

## Next Steps

- [x] `norm_out` / `proj_out` — video output projection (`[4096]` norm, `[128, 4096]` proj)
- [x] `audio_norm_out` / `audio_proj_out` — audio output projection (`[2048]` norm, `[128, 2048]` proj)
- [ ] Patchify wired into full pipeline
- [ ] Denoising scheduler loop
- [ ] VAE decode
