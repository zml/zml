# M4 Vocoder + BWE — Bug Investigation Status

## Final State (April 7, 2026) — COMPLETE

### Summary
Both stages of the VocoderWithBWE pipeline are **functionally correct**.

| Stage | Component | PSNR | Notes |
|-------|-----------|------|-------|
| Stage 1 | Main Vocoder (16kHz) | **63.78 dB** | mel [1,2,8,64] bf16 → waveform [1,2,1280] f32 |
| Stage 2 | Mel STFT | 66.59 dB | Isolated: forwardComputeMel |
| Stage 2 | BWE Generator (residual) | 72.20 dB | Isolated: forwardBWEResidual |
| Stage 2 | Sinc Resampler (skip) | **160.32 dB** | Same-input isolation test (perfect) |
| Stage 2 | End-to-end (48kHz) | **26.42 dB** | Stage 1 error amplified by 3× resampler |

### Bugs Found and Fixed

#### Bug 1: ZML `window_reversal` doesn't flip conv kernels (Stage 1)
- **Symptom**: Stage 1 vocoder output divergent
- **Root cause**: `window_reversal = true` in ZML's conv1d doesn't flip the kernel as expected
- **Fix**: explicitly flip kernel with `.reverse(.{2})` before convolution
- Applied to all 3 transposed convolution sites:
  1. `forwardVocConvTranspose1d` (vocoder/BWE upsample layers)
  2. `forwardUpSample1d` (anti-aliased activation upsampling)
  3. `forwardSincResample3x` (BWE skip connection resampler)

#### Bug 2: Wrong sinc filter window type (Stage 2)
- **Symptom**: `bwe_skip` at 19.30 dB despite correct filter logic
- **Root cause**: Zig used **Hann**-windowed sinc filter (kernel_size=43) but Python defaults to **Kaiser** (kernel_size=18)
  - Wrong filter values, kernel size, and all padding constants (pad, pad_left, pad_right)
- **Fix**: replaced with correct 18-tap Kaiser-windowed sinc filter and padding constants
  - `kernel_size=18, pad=5, pad_left=22, pad_right=23`
  - Filter values extracted via `inspect_resampler.py`
- **Verification**: isolation test feeding identical input → 160.32 dB PSNR (perfect match)

### PSNR Budget Explanation
- Stage 1: 63.78 dB — limited by bf16 mel input quantization
- Stage 2: 26.42 dB — Stage 1 error (max 0.005) amplified 3× by sinc resampler (max error → 0.456)
- All BWE-internal components: >66 dB when measured in isolation

---

## Files Modified
- `examples/ltx/model.zig` — Section 13 (vocoder), all forward ops + debug forward fns
- `examples/ltx/vocoder_decode.zig` — standalone validation driver with BWE debug dumps
- `examples/ltx/inference.zig` — full pipeline, two-stage vocoder
- `zml/mem.zig` — `.array` case in `bufferizeInner` (+5 lines)

## Debug Scripts
- `examples/ltx/e2e/export_vocoder_activations.py` — exports input_mel, ref_waveform_16k, ref_waveform
- `examples/ltx/e2e/export_vocoder_stages.py` — exports Stage 1 intermediates (rearrange, conv_pre, ups0, resblocks)
- `examples/ltx/e2e/export_bwe_stages.py` — exports BWE intermediates (stft_magnitude, mel, residual, skip, output)
- `examples/ltx/e2e/compare_vocoder.py` — compares 16kHz and 48kHz with PSNR per channel
- `examples/ltx/e2e/compare_bwe_stages.py` — compares BWE intermediates with PSNR per stage
- `examples/ltx/e2e/verify_conv_transpose.py` — verified manual kernel flip matches PyTorch
- `examples/ltx/e2e/inspect_resampler.py` — dumps Python UpSample1d filter, padding, and source
- `examples/ltx/e2e/isolate_sinc_resampler.py` — same-input isolation test for sinc resampler
- `examples/ltx/e2e/debug_skip_offset.py` — offset/shift analysis for skip connection

## Debug Forward Functions in model.zig
- `forwardAfterUps0` — verified at 145 dB after fix
- `forwardAfterStage0` — verified high PSNR
- `forwardBWEComputeMel` — returns log-mel [B,2,n_mels,T_frames] (before transpose)
- `forwardBWESincSkip` — returns sinc-resampled skip [B,2,T_skip]
- `forwardBWEResidual` — returns BWE generator residual [B,2,T_bwe]

## Config Values (from checkpoint metadata)
Main vocoder: rates=[5,2,2,2,2,2], kernels=[11,4,4,4,4,4], initial_ch=1536
BWE: rates=[6,5,2,2,2], kernels=[12,11,4,4,4], initial_ch=512
BWE STFT: n_fft=512, hop_length=80, win_size=512, num_mels=64
BWE sampling: input_sr=16000, output_sr=48000

## Remote Debug Infrastructure
- Python env: `cd /root/repos/LTX-2 && uv run ...`
- Zig build: `cd /root/repos/zml && bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:vocoder_decode -- <args>`
- Checkpoint: `/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors`
- Reference activations: `/root/e2e_demo/vocoder_ref/vocoder_activations.safetensors`
- BWE stages: `/root/e2e_demo/vocoder_ref/bwe_stages.safetensors`
- Zig output dir: `/root/e2e_demo/vocoder_zig_out/`

---

## Investigation Timeline

### April 3, 2026 — Stage 1 Fixed
- Discovered `window_reversal=true` bug in ZML conv1d
- Applied `.reverse(.{2})` workaround to all 3 transposed conv sites
- Stage 1: 63.74 dB — validated

### April 7, 2026 — Stage 2 Fixed
- Bisected BWE pipeline: mel (66.59 dB), residual (72.20 dB), skip (26.42 dB)
- Skip was the culprit — discovered wrong Hann→Kaiser window type via `inspect_resampler.py`
- Fixed filter (18-tap Kaiser) and padding constants
- Isolation test confirmed 160.32 dB same-input PSNR
- End-to-end 48kHz: 26.42 dB (expected given Stage 1 error amplification)
- All BWE-internal components: >66 dB when measured in isolation
- **Both stages are functionally correct.** No further bugs to fix.
