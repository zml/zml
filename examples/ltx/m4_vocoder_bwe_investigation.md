# M4 Vocoder + BWE — Bug Investigation Status

## Current State (end of April 3, 2026)

### What's Working
- **Stage 1 (Main Vocoder)**: 63.74 dB PSNR — **FIXED and validated**
  - mel [1,2,8,64] bf16 → waveform [1,2,1280] f32 at 16kHz
  - Max abs error: 0.004396, mean: 0.001021

### What's Broken
- **Stage 2 (BWE Pipeline)**: 19.30 dB PSNR — **still divergent**
  - waveform_16k [1,2,1280] f32 → waveform_48k [1,2,3840] f32
  - Max abs error: 0.974543, mean: 0.138034
  - The 16kHz input to BWE is correct (63.74 dB), so the bug is inside `forwardBWEPipeline`

### Root Cause Found and Fixed (Stage 1)
- **`window_reversal = true` in ZML's conv1d doesn't flip the kernel as expected**
- Fix: explicitly flip kernel with `.reverse(.{2})` before convolution
- Applied to all 3 transposed convolution sites:
  1. `forwardVocConvTranspose1d` (vocoder/BWE upsample layers)
  2. `forwardUpSample1d` (anti-aliased activation upsampling)
  3. `forwardSincResample3x` (BWE skip connection resampler)

### Next Steps: Debug BWE Pipeline (Stage 2)

The `forwardBWEPipeline` in model.zig (line ~4030) does:
1. **Pad** vocoder output to multiple of hop_length=80
2. **forwardComputeMel**: STFT + mel projection on the vocoder output
3. **Transpose**: mel [B,2,n_mels,T_frames] → [B,2,T_frames,mel_bins]
4. **forwardVocoderGeneric** (BWE generator): mel → residual waveform
5. **forwardSincResample3x**: upsample vocoder output by 3×
6. **Add** residual + skip, clamp, trim

**Likely suspects** (in order of probability):
- **forwardComputeMel** — uses `forwardSTFT` (causal conv1d with DFT basis) and `forwardMelProjection`. These are unique to BWE (not tested by Stage 1). The STFT uses a conv1d with `window_strides=hop_length` and the mel projection uses a conv1d with kernel_size=1. No transposed convs here, so the kernel flip fix doesn't apply.
- **forwardSincResample3x** — the hardcoded sinc filter is symmetric so the flip doesn't matter much, but check the pad/trim arithmetic
- **forwardVocoderGeneric for BWE** — same code as Stage 1 (which is now 63.74 dB), so unlikely to be buggy per se, but could have wrong input from mel computation

**Debugging approach**: Same bisection as before:
1. Export Python intermediate after `_compute_mel` (the mel fed to BWE generator)
2. Export Python intermediate after `bwe_generator(mel_for_bwe)` (residual)
3. Export Python intermediate after `resampler(x)` (skip connection)
4. Compare each against Zig equivalents

### Key Python Source References
- `VocoderWithBWE.forward()`: `/root/repos/LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py` line 551
- `_compute_mel()`: same file, around line 490
- `_STFTFn.forward()`: same file, around line 435
- `UpSample1d` (hann window variant for sinc resampler): line ~82

### Files Modified
- `examples/ltx/model.zig` — Section 13 (vocoder), all forward ops
- `examples/ltx/vocoder_decode.zig` — standalone validation driver (has debug dump code)
- `examples/ltx/inference.zig` — full pipeline, two-stage vocoder
- `zml/mem.zig` — `.array` case in `bufferizeInner` (+5 lines)
- `zml/io.zig` — **UNCHANGED** (reverted to original)

### Debug Infrastructure on Remote
- Remote server: `root@dev-oboulant` (vast.ai, SSH port 6378, IP 34.48.171.202)
- `/root/.local/bin/uv` for Python
- Python env: `cd /root/repos/LTX-2 && /root/.local/bin/uv run ...`
- Zig build: `cd /root/repos/zml && bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:vocoder_decode -- <args>`
- Checkpoint: `/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors`
- Reference activations: `/root/e2e_demo/vocoder_ref/vocoder_activations.safetensors`
- Zig output dir: `/root/e2e_demo/vocoder_zig_out/`
- Python stage export: `/root/e2e_demo/vocoder_ref/vocoder_stages.safetensors`

### Debug Scripts Created
- `examples/ltx/e2e/export_vocoder_activations.py` — exports input_mel, ref_waveform_16k, ref_waveform
- `examples/ltx/e2e/export_vocoder_stages.py` — exports after_rearrange, after_conv_pre, after_ups0, after_stage0_resblocks
- `examples/ltx/e2e/compare_vocoder.py` — compares 16kHz and 48kHz with PSNR per channel
- `examples/ltx/e2e/verify_conv_transpose.py` — verified that manual kernel flip matches PyTorch

### Debug Forward Functions in model.zig
- `forwardAfterUps0` — verified at 145 dB after fix
- `forwardAfterStage0` — should now be high PSNR too

### Config Values (from checkpoint metadata)
Main vocoder: rates=[5,2,2,2,2,2], kernels=[11,4,4,4,4,4], initial_ch=1536
BWE: rates=[6,5,2,2,2], kernels=[12,11,4,4,4], initial_ch=512
BWE STFT: n_fft=512, hop_length=80, win_size=512, num_mels=64
BWE sampling: input_sr=16000, output_sr=48000
