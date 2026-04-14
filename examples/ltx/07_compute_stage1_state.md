# Compute Stage 1 Initial State in Zig — DONE ✅

## Goal

Remove `--stage1-inputs` safetensors dependency. Compute positions, masks, and
clean latents directly in `runStage1` using existing helper functions. Handles
both unconditioned and image-conditioned paths (image conditioning already
applies *after* the initial state is created).

## Previous State

`runStage1` loaded 6 tensors from `unconditioned_stage1_inputs.safetensors`:

| Tensor                | Shape               | Dtype | Value (unconditioned) |
|-----------------------|---------------------|-------|-----------------------|
| `video_positions`     | `[1, 3, T_v1, 2]`  | bf16  | 3D pixel-coord grid   |
| `audio_positions`     | `[1, 1, T_a, 2]`   | f32   | time intervals (sec)  |
| `video_denoise_mask`  | `[1, T_v1, 1]`     | f32   | all 1.0               |
| `audio_denoise_mask`  | `[1, T_a, 1]`      | f32   | all 1.0               |
| `video_clean_latent`  | `[1, T_v1, 128]`   | bf16  | all zeros             |
| `audio_clean_latent`  | `[1, T_a, 128]`    | bf16  | all zeros             |

Where:
- `T_v1 = f_lat × h_lat_s1 × w_lat_s1` (from `pipe_meta.stage1`)
- `T_a = pipe_meta.stage1.t_audio`

All dimension info is already available in `PipelineMeta`.

## Existing Zig Helpers (proven in `runBridge` for Stage 2)

| Helper                   | Produces                     | Used in Stage 2 bridge |
|--------------------------|------------------------------|------------------------|
| `computeVideoPositions(F, H, W, fps)` | `[1, 3, T_v, 2]` bf16 bytes | ✅ with S2 dims |
| `computeAudioPositions(T_a)`          | `[1, 1, T_a, 2]` f32 bytes  | ✅ identical     |
| `fillOnesF32(buf)`                    | all-ones f32 bytes           | ✅ for S2 masks  |

### What's NOT shared with bridge

Clean latents: the bridge feeds real data (upsampled S1 output for video,
S1 denoised passthrough for audio). Stage 1 needs **all zeros**. This is
trivial: `@memset(host_bytes, 0)`.

## Exact Mapping: Stage 1 vs Stage 2 (bridge)

```
Stage 2 bridge code (inference.zig ~L1713):          Stage 1 equivalent:
─────────────────────────────────────                 ─────────────────────
computeVideoPositions(F, H_s2, W_s2, fps)     →  computeVideoPositions(F, H_s1, W_s1, fps)
computeAudioPositions(T_a)                     →  computeAudioPositions(T_a)          [identical]
fillOnesF32([1, T_v2, 1])                     →  fillOnesF32([1, T_v1, 1])
fillOnesF32([1, T_a, 1])                      →  fillOnesF32([1, T_a, 1])            [identical]
video_clean = re-patchified upsampled S1       →  video_clean = zeros [1, T_v1, 128] bf16
audio_clean = S1 denoised passthrough          →  audio_clean = zeros [1, T_a, 128]  bf16
```

Same functions, just different dimensions and zeros instead of real data for
clean latents.

## Image Conditioning

Already handled post-initial-state by existing code at inference.zig ~L800:

```zig
if (image_path) |img_path| {
    // loadAndPreprocess → encodeImageToTokens → applyConditioning
    // Modifies v_latent_buf, v_clean_buf, v_mask_buf in-place
}
```

This runs *after* initial state creation, so the change is transparent to it.

## Plan

### M0: Replace loadBuf calls with host computation — DONE ✅

In `runStage1`, replaced the 6 `loadBuf` calls with host computation:

1. `T_v1 = pipe_meta.stage1.f_lat * pipe_meta.stage1.h_lat * pipe_meta.stage1.w_lat`
2. `computeVideoPositions(f_lat, h_lat_s1, w_lat_s1, fps)` → `Buffer.fromBytes`
3. `computeAudioPositions(t_audio)` → `Buffer.fromBytes`
4. Mask buffers `[1, T_v1, 1]` and `[1, T_a, 1]` f32, `fillOnesF32`
5. Clean latent buffers `[1, T_v1, 128]` and `[1, T_a, 128]` bf16, `@memset(_, 0)`
6. Removed `--stage1-inputs` CLI arg and `inputs_store` code
7. Added `--dump-intermediates` output for the computed tensors

Also updated: README.md, ARCHITECTURE.md, export_pipeline.py docstring.

### M1: Validate parity — DONE ✅

**Level 1 — Bitwise tensor check:**

Validated using `validate_stage1_state.py`:

| Tensor | Result |
|--------|--------|
| `video_positions` | **bitwise identical** |
| `audio_positions` | **near-identical** (max 4.8e-7, f32 rounding < 1 ULP) |
| `video_denoise_mask` | **bitwise identical** |
| `audio_denoise_mask` | **bitwise identical** |
| `video_clean_latent` | **bitwise identical** |
| `audio_clean_latent` | **bitwise identical** |

The audio positions diff is from different intermediate rounding of
`mel * 160 / 16000` between PyTorch and Zig — 0.5 microsecond in a 5s clip.

**Level 2 — End-to-end denoising:** Stage 1 ran all 30 steps successfully
with the computed initial state. Full pipeline E2E validation pending
(requires `--bf16-attn-stage2` for Stage 2).

### M2: Clean up Python export — DONE ✅

Removed Stage 1 inputs export from `export_pipeline.py`:
- Removed `conditioned_stage1_inputs.safetensors` and
  `unconditioned_stage1_inputs.safetensors` export blocks
- Updated docstring and summary print section

## Files Modified

- `examples/ltx/inference.zig` — `runStage1` (host computation replaces loadBuf), `CliArgs` / `parseArgs` (removed `--stage1-inputs`), doc comment
- `examples/ltx/export_pipeline.py` — removed S1 inputs export blocks (M2), updated docstring and summary
- `examples/ltx/README.md` — pipeline overview, data flow diagram, tensor table, CLI examples, migration history
- `examples/ltx/ARCHITECTURE.md` — Section 4.1 inputs table
- `examples/ltx/export_pipeline.py` — docstring (outputs marked reference-only)
- `examples/ltx/validate_stage1_state.py` — new validation script
- `examples/ltx/07_compute_stage1_state.md` — this file
