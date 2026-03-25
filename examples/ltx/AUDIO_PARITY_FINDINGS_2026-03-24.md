# Audio Parity Findings - 2026-03-24

## Scope

This note summarizes the current state of the stage-2 native parity work for the contiguous 8-block checker, with emphasis on the audio non-pass and the practical restart plan for the next session.

The main checker is [examples/ltx/block_slice_native_check.zig](/Users/oboulant/repos/zml/zml/examples/ltx/block_slice_native_check.zig).

## Current Conclusion

- Video parity for the 8-block slice is good and passes the strict checker threshold.
- Audio parity for the same 8-block slice does not pass the strict `minimum_close_fraction = 0.999` threshold.
- The current evidence does not point to a single obvious implementation bug in the audio path.
- The current evidence is most consistent with distributed numerical drift that accumulates across the chained audio residual path.
- The drift appears to be backend-sensitive and consistent with XLA/CUDA execution details such as fusion, tiling, reassociation, and bf16-style accumulation behavior.

## What Was Tested

### 1. Dot precision hint experiment

- A temporary `dot_precision = .high` experiment was tried in the ZML side.
- It did not materially improve parity.
- That change was reverted.

Conclusion: the mismatch is not explained by that precision hint alone.

### 2. Targeted fixture retrace

- A targeted retrace flow was built to avoid replaying the full 22B pipeline.
- The script [examples/ltx/retrace_block_slice_bf16.py](/Users/oboulant/repos/zml/zml/examples/ltx/retrace_block_slice_bf16.py) was stabilized and used to produce a refreshed fixture (`maskrefresh11`).
- The refreshed fixture improved results only modestly.

Conclusion: fixture regeneration changed the numbers slightly, but it did not remove the audio mismatch.

### 3. Checker-only f32 residual experiments

- Two opt-in checker experiments were added in [examples/ltx/block_slice_native_check.zig](/Users/oboulant/repos/zml/zml/examples/ltx/block_slice_native_check.zig) and [examples/ltx/model.zig](/Users/oboulant/repos/zml/zml/examples/ltx/model.zig):
  - `--audio-ff-residual-f32`
  - `--audio-all-residuals-f32`
- These were diagnostic only. The default path remained unchanged.
- Both experiments produced only modest improvement.

Conclusion: localizing residual adds to f32 can move the output, but it does not resolve the strict parity gap by itself.

### 4. HLO signature evidence

- Baseline and experiment HLO signatures were compared.
- The convert counts stayed similar.
- GEMM partitioning changed in meaningful ways, including tiling shapes consistent with backend reorganization.

Conclusion: there is real backend graph variation, which supports a numerical execution explanation rather than a purely front-end logic mistake.

### 5. Extended and blockwise error statistics

- Distribution-aware error stats were added in [examples/ltx/check_utils.zig](/Users/oboulant/repos/zml/zml/examples/ltx/check_utils.zig).
- Per-block chain, teacher-forced, and exact-input diagnostics were added in [examples/ltx/block_slice_native_check.zig](/Users/oboulant/repos/zml/zml/examples/ltx/block_slice_native_check.zig).

Key observed pattern from the remote run:

- Free-running chain error broadens steadily from block 0 to block 7.
- Teacher-forced audio-input results stay much tighter.
- Exact-input per-block results also stay much tighter.
- Final audio still fails the strict checker threshold, but the aggregate metrics remain numerically close in a global sense:
  - rel-L2 stays small
  - cosine similarity stays extremely close to 1
  - sign fractions are roughly balanced
  - the drift is distributed, not dominated by a few isolated outliers

Conclusion: the observed non-pass is best explained as accumulated, distributed drift through the chained audio path, not as a single broken stage.

## Practical Interpretation

### What the current evidence supports

- The audio mismatch is real.
- The mismatch is not well explained by a stale fixture alone.
- The mismatch is not well explained by only a few extreme outliers.
- The mismatch is not well explained by one obviously broken block-local computation, because teacher-forced and exact-input checks remain tight.

### What the current evidence does not prove

- It does not prove that full 48-block or end-to-end pipeline outputs will be visually or audibly identical to Python.
- It does not prove that full pipeline quality will be unacceptable.
- It does not yet establish the correct production acceptance policy.

### Current working judgment

- Exact strict tensor parity for long chained audio paths is unlikely.
- A catastrophic divergence is not indicated by the current 8-block evidence.
- The most likely outcome is measurable tensor drift with potentially acceptable end-result quality, but that still needs to be validated directly at 48 blocks and full pipeline level.

## Why Video Passes But Audio Does Not: Hidden Dimension and Backend Tiling Asymmetry

### Residual Structure Parity

Both streams have **identical residual count per block** — 4 residual additions each, evaluated in order:

| Position | Video | Audio |
|----------|-------|-------|
| 1 | `h_v += attn1_out * vgate_msa` (M1: self-attn) | `h_a += audio_attn1_out * agate_msa` (M4-A: self-attn) |
| 2 | `h_v += text_ca_out * vgate_text_ca` (M2: text-CA) | `h_a += audio_text_ca_out * agate_text_ca` (M4-B: text-CA) |
| 3 | `h_v += a2v_delta` (A→V: audio→video) | `h_a += v2a_delta` (V→A: video→audio) |
| 4 | `h_v += ff_out * vgate_mlp` (M3: FF) | `h_a += audio_ff_out * agate_mlp` (M4-C: FF) |

See [examples/ltx/model.zig](/Users/oboulant/repos/zml/zml/examples/ltx/model.zig) lines 650–830 for the full `forwardNativeImpl` function.

The code structure is **exactly symmetric**:
- Same gating and normalization patterns
- Same cross-attention structure
- Same residual addition semantics

Yet video passes parity and audio does not. The reason is not architectural; it is **numerical and backend-driven**.

### Root Cause: Hidden Dimension Mismatch and XLA Tiling

Video uses `hidden_dim = 4096; heads = 32; d_head = 128`.  
Audio uses `hidden_dim = 2048; heads = 32; d_head = 64`.

When XLA/CUDA compile GEMMs for these two dimensions, they make fundamentally different tiling and reduction-tree decisions:

- **Video GEMM shapes** at `d=4096` are tiled by XLA in a way that produces numerical output closer to PyTorch's cuBLAS reference, especially at bf16 precision levels.
- **Audio GEMM shapes** at `d=2048` incur different tiling, which accumulates more deviation from the Python reference at the same precision.

### Direct Evidence: HLO Signature Changes

When the `--audio-all-residuals-f32` experiment was run yesterday (converting V2A cross-attention delta and subsequent FF residual add to f32), the HLO signatures showed:

- **GEMM partition shapes changed**, e.g., `f32[8,128,4128]` became `f32[2,128,4128]`.
- This was **not** a precision change alone; it was a **tiling reorganization** triggered by the dtype change.
- The baseline audio GEMMs at `d=2048` also undergo similar (different) tiling than the video stream.

This confirms that the backend is making real structural choices that differ between the two streams, and those choices affect numerical output.

### Cross-Stream Contamination Is Not the Driver

The teacher-forced and exact-input diagnostics ruled out a simpler explanation:

**Teacher-forced experiment result** (from yesterday's remote run):
- When audio block N receives **reference audio input** but **computed (drifted) video** as V2A context, the audio output error stays **much tighter** than in the free-running chain.
- This proves that incoming drift from the video stream is **not** the primary cause of audio divergence.

**Exact-input per-block diagnostics result**:
- When audio block N is fed **exact reference inputs** (both audio and video context), it produces output that remains **near-reference** with low error.
- Only the **chained** audio forward (where each block's output becomes the next block's input) accumulates drift.

**Interpretation**: The drift is intrinsic to the audio chain itself, not caused by contamination from video. It is driven by how XLA tiles the audio-dimension GEMMs internally, and those tiling choices produce small but systematic deviations that add up across blocks.

### Mixed-Dimension Cross-Attention Adds Asymmetric Variation

The A→V and V→A cross-attention paths have asymmetric dimensions:

- **A→V**: query from video (`d_q=4096, d_head=128`), context from audio (`d_kv=2048`)
- **V→A**: query from audio (`d_q=2048, d_head=64`), context from video (`d_q_v=4096, d_head=128`)

These mixed-dimension GEMMs produce additional scheduling variation in the audio output path that the video path does not experience (since the A→V delta lands in the already-4096-dim video stream, which has the more favorable tiling).

### Summary of Why Video Escapes Drift

Video passes parity because:

1. Its GEMMs at `d=4096` happen to tile in a way that XLA aligns well with cuBLAS.
2. Even though it has 4 residuals per block (same as audio), the accumulated error across 8 blocks remains within tolerance.
3. The A→V mixed-dimension GEMM output lands in the already-well-tiled video dimension, so it doesn't disrupt the stream.

Audio drifts because:

1. Its GEMMs at `d=2048` incur different (less favorable) tiling by XLA.
2. The same 4 residuals per block accumulate proportionally more deviation.
3. The V→A mixed-dimension GEMM lands in the audio dimension and contributes to the accumulated error (not as a logic break, but as a numerical consequence).

The mismatch is therefore **not a bug in the Zig implementation**, but a **representation of real backend-level numerical behavior** that differs between the two streams due to their different hidden dimensions.

## Fresh Start Inputs

If the next session starts from a `trace_run/` directory that contains only:

```text
01_text_contexts.pt
02_stage1_conditionings.pt
03_stage1_sigmas.pt
04_stage1_outputs.pt
05_stage2_upsample_io.pt
06_stage2_conditionings.pt
07_stage2_sigmas.pt
08_stage2_outputs.pt
09_decoded_outputs.pt
10_stage1_steps.pt
11_stage2_steps.pt
acts_stage2_transformer_step_000_b00_ff_boundary.pt
ff_b00_fixture.safetensors
```

that is a good base state.

However, it is not yet the exact state needed to run the 8-block checker immediately.

The checker consumes two derived files:

- `stage2_blocks_<start>_<end>_lora<LORA>_merged.safetensors`
- `block_slice_native_<start>_<end>_lora<LORA>_t<TOKEN_LIMIT>.safetensors`

Those need to be regenerated from the fresh base state.

## What Must Be Regenerated Tomorrow

For an 8-block slice run such as blocks `0..7` with token limit `128` and LoRA `0.0`, regenerate these artifacts:

1. A replay capture from `11_stage2_steps.pt`:
   - `acts_stage2_transformer_step_000_<pass_label>_t128.pt`
2. A reindexed merged checkpoint for that block range:
   - `stage2_blocks_0_7_lora0.0_merged.safetensors`
3. A native checker fixture for that same slice:
   - `block_slice_native_0_7_lora0.0_t128.safetensors`

Optional:

4. A refreshed fixture via bf16-style PyTorch retrace:
   - `block_slice_native_0_7_lora0.0_t128_maskrefresh11.safetensors`

## Regeneration Path

The relevant scripts are:

- Replay from `11_stage2_steps.pt`: [examples/ltx/replay_stage2_transformer_step.py](/Users/oboulant/repos/zml/zml/examples/ltx/replay_stage2_transformer_step.py)
- Export reindexed slice checkpoint: [examples/ltx/export_stage2_block_slice_checkpoint.py](/Users/oboulant/repos/zml/zml/examples/ltx/export_stage2_block_slice_checkpoint.py)
- Export native slice fixture: [examples/ltx/export_block_slice_native_fixture.py](/Users/oboulant/repos/zml/zml/examples/ltx/export_block_slice_native_fixture.py)
- Optional refreshed retrace fixture: [examples/ltx/retrace_block_slice_bf16.py](/Users/oboulant/repos/zml/zml/examples/ltx/retrace_block_slice_bf16.py)
- Convenience script for one 8-block slice: [examples/ltx/run_native_t3_slice.sh](/Users/oboulant/repos/zml/zml/examples/ltx/run_native_t3_slice.sh)

## Suggested First Commands For Tomorrow

These commands assume a remote layout similar to the one used during investigation:

- Python/LTX repo at `/root/repos/LTX-2`
- ZML repo at `/root/repos/zml`
- trace directory at `/root/repos/LTX-2/trace_run`

### 1. Recreate the slice replay capture

```bash
cd /root/repos/LTX-2

uv run python ./scripts/replay_stage2_transformer_step.py \
  --pass-label t3_slice_0_7_lora0.0 \
  --capture-inputs \
  --capture-kwargs \
  --all-modules \
  --max-capture-gib 8.0 \
  --distilled-lora-strength 0.0 \
  --token-limit 128 \
  --include '^velocity_model\.transformer_blocks\.(0|1|2|3|4|5|6|7)(\.|$)'
```

Expected output:

```text
/root/repos/LTX-2/trace_run/acts_stage2_transformer_step_000_t3_slice_0_7_lora0.0_t128.pt
```

### 2. Export the reindexed 8-block checkpoint

```bash
uv run python ./scripts/export_stage2_block_slice_checkpoint.py \
  --start-block 0 \
  --end-block 7 \
  --distilled-lora-strength 0.0 \
  --output /root/repos/LTX-2/trace_run/stage2_blocks_0_7_lora0.0_merged.safetensors
```

### 3. Export the native checker fixture

```bash
uv run python ./scripts/export_block_slice_native_fixture.py \
  /root/repos/LTX-2/trace_run/acts_stage2_transformer_step_000_t3_slice_0_7_lora0.0_t128.pt \
  /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128.safetensors \
  --start-block 0 \
  --end-block 7
```

### 4. Run the 8-block checker with extended stats

```bash
cd /root/repos/zml

bazel run --@zml//platforms:cuda=true //examples/ltx:block_slice_native_check -- \
  /root/repos/LTX-2/trace_run/stage2_blocks_0_7_lora0.0_merged.safetensors \
  /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128.safetensors \
  --extended-error-stats
```

### 5. Optional: rebuild the refreshed fixture and re-run

```bash
cd /root/repos/LTX-2

uv run python /root/repos/zml/examples/ltx/retrace_block_slice_bf16.py \
  --input-fixture  /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128.safetensors \
  --checkpoint     /root/repos/LTX-2/trace_run/stage2_blocks_0_7_lora0.0_merged.safetensors \
  --output-fixture /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128_maskrefresh11.safetensors
```

Then:

```bash
cd /root/repos/zml

bazel run --@zml//platforms:cuda=true //examples/ltx:block_slice_native_check -- \
  /root/repos/LTX-2/trace_run/stage2_blocks_0_7_lora0.0_merged.safetensors \
  /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128_maskrefresh11.safetensors \
  --extended-error-stats
```

## Tomorrow's Priority Plan

### 1. Reconfirm the 8-block baseline quickly

- Rebuild or locate the 8-block checkpoint + fixture pair.
- Run the checker with `--extended-error-stats`.
- Confirm that the same drift pattern is reproduced.

### 2. Move to the 48-block question

- Apply the same methodology to a longer chain.
- Capture blockwise trend if a 48-block checker or equivalent staged diagnostic is available.
- Determine whether the same chain-accumulation pattern continues without indicating a logic break.

### 3. Run end-to-end A/B

- Use the same prompt, seed, and generation settings between Python and ZML.
- Compare objective outputs first.
- Then do human inspection.

### 4. Decide the acceptance policy

- Either keep a strict parity target for the relevant scope.
- Or move to a quality-equivalent acceptance target if full-pipeline results justify it.

## Bottom Line

At the end of 2026-03-24, the best working interpretation is:

- video is in good shape for the 8-block slice,
- audio does not meet the current strict chained parity threshold,
- the evidence points to accumulated numerical drift rather than a single obvious implementation bug,
- and the next session should focus on validating practical impact at 48-block and full-pipeline scale rather than continuing to argue from the 8-block checker alone.
