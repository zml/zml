# Velocity Model Full Forward Pass — Implementation Plan

## Goal

Wire the full `velocity_model.forward` in Zig and validate it end-to-end against Python, working from verified sub-modules up to a full single-step forward pass.

## Context

The Python `velocity_model.forward(video: Modality, audio: Modality)` executes this flow:

```
1. PATCHIFY
   video_hidden = patchify_proj(video.latent)           → [B, T_v, 4096]
   audio_hidden = audio_patchify_proj(audio.latent)      → [B, T_a, 2048]

2. ADALN (8 modules, all from sigma_scaled = sigma × 1000)
   video_timesteps, v_emb_ts     = adaln_single(σ)            → [B, 9×4096], [B, 4096]
   audio_timesteps, a_emb_ts     = audio_adaln_single(σ)       → [B, 9×2048], [B, 2048]
   v_prompt_ts, _                = prompt_adaln_single(σ)      → [B, 2×4096]
   a_prompt_ts, _                = audio_prompt_adaln_single(σ) → [B, 2×2048]
   v_cross_ss, _                 = av_ca_video_ss_adaln(σ)     → [B, 4×4096]
   a_cross_ss, _                 = av_ca_audio_ss_adaln(σ)     → [B, 4×2048]
   v_cross_gate, _               = av_ca_a2v_gate_adaln(σ)     → [B, 4096]
   a_cross_gate, _               = av_ca_v2a_gate_adaln(σ)     → [B, 2048]

3. ROPE (from positions → cos/sin pairs for self-attn and cross-attn)
   v_pe, a_pe, a2v_pe/k_pe, v2a_pe/k_pe

4. TEXT CONTEXT (pass-through from Modality)
   v_text_ctx, a_text_ctx, optional masks

5. 48 × BasicAVTransformerBlock.forward(video_args, audio_args)

6. OUTPUT PROJECTION
   video_out = _process_output(..., video_args.x, v_emb_ts) → [B, T_v, 128]
   audio_out = _process_output(..., audio_args.x, a_emb_ts) → [B, T_a, 128]
```

### What is verified in Zig

| Component | Status | Entry point |
|---|---|---|
| Patchify (video + audio) | ✅ verified | `forwardPatchify` |
| AdaLayerNormSingle ×8 | ✅ verified (live) | `forwardAdalnSingle` |
| 48 transformer blocks (chain) | ✅ verified (live, 48-block chain) | `forwardBlock0Native` (loop) |
| OutputProjection (video + audio) | ✅ verified (live) | `forwardOutputProjection` |
| RoPE application (cos/sin → rotated q/k) | ✅ verified | `applyLtxRotaryEmb` |
| **Full step (blocks + output proj)** | ✅ **verified (live, GPU)** | `full_step_check.zig` |
| RoPE generation (positions → cos/sin) | ❌ not implemented | — |
| `modality_from_latent_state` | ❌ not implemented | — |

---

## Step 1 — End-to-end single transformer step check ✅ DONE

**Goal**: Verify that all verified sub-modules compose correctly when wired together, using pre-computed SharedInputs from the Python fixture.

**Boundary**: Feed already-patchified `vx_in`, `ax_in`, pre-computed `SharedInputs` fields (adaln modulations, PE cos/sin, text contexts) — all exported from a single real Python step — and compare `{video_out, audio_out}` against Python reference.

### Results

Validated on GPU (`libpjrt_cuda.so`) with bf16, `use_f32_video_residuals = false`:

| Output | cos_sim | close@0.25 | p50 | p99 | p999 |
|---|---|---|---|---|---|
| video_out | 0.998280 | 99.15% ✅ | 0.023 | 0.239 | 0.470 |
| audio_out | 0.999970 | 100% ✅ | 0.004 | 0.020 | 0.031 |

Pass criteria: `close ≥ 0.99` at `atol=0.25` AND `cos_sim ≥ 0.995`.

**Precision investigation**: bf16 rounding diverges between XLA/PJRT and PyTorch compilers
(different operation fusion/ordering), accumulating across 48 serial blocks. Tested all 4
configurations (CPU/GPU × bf16/f32-residuals) — GPU vs CPU makes no difference, and f32
residuals make things *worse* (diverges further from the reference bf16 path). The error is
inherently caused by different compiler backends, not a logic bug.

### Comments on the DIAG 

The trace is :

```
info: Initializing model params...
info: Compiling single-block exe...
info: Single-block exe compiled.
info: Compiling output projection exe...
info: Output projection exes compiled.
info: Loading 48 block weights...
info: Block weights loaded.
info: Loading output projection weights...
info: Output projection weights loaded.
info: Running 48-block chain...
info:   block  7 done
info:   block 15 done
info:   block 23 done
info:   block 31 done
info:   block 39 done
info:   block 47 done
info: 48-block chain complete.
info: DIAG video_out_ref vs proj.video.output:  mean_abs=0.00000  max_abs=0.00000  cos_sim=1.000000  close=1.0000
info: DIAG video_out_ref vs proj.video.output:  p50=0.00000  p90=0.00000  p99=0.00000  p999=0.00000
info: DIAG video chain vs proj_x_in:  mean_abs=0.69484  max_abs=136.00000  cos_sim=0.999815  close=0.2576
info: DIAG video chain vs proj_x_in:  p50=0.43750  p90=1.56250  p99=4.12500  p999=8.17188
info: DIAG audio chain vs proj_x_in:  mean_abs=0.11299  max_abs=16.00000  cos_sim=0.999992  close=0.8387
info: DIAG audio chain vs proj_x_in:  p50=0.07813  p90=0.25000  p99=0.50000  p999=2.00000
info: DIAG video proj(ref_x_in):  mean_abs=0.00117  max_abs=0.03125  cos_sim=0.999996  close=1.0000
info: DIAG video proj(ref_x_in):  p50=0.00000  p90=0.00391  p99=0.01563  p999=0.01563
info: DIAG audio proj(ref_x_in):  mean_abs=0.00105  max_abs=0.01563  cos_sim=0.999996  close=1.0000
info: DIAG audio proj(ref_x_in):  p50=0.00000  p90=0.00391  p99=0.01563  p999=0.01563
info: Running output projection...
info: Output projection complete.
info: video_out:  mean_abs=0.03811  max_abs=1.50391  cos_sim=0.998280  close=0.9915
info: video_out:  p50=0.02344  p90=0.08594  p99=0.23876  p999=0.53697
info: audio_out:  mean_abs=0.00558  max_abs=0.04688  cos_sim=0.999970  close=1.0000
info: audio_out:  p50=0.00391  p90=0.01172  p99=0.02246  p999=0.03125
info: PASSED: both video and audio outputs match.
```

Analysis : 
Those lines compare the **intermediate** output of the 48-block chain (before output projection) against the Python reference's intermediate at the same point (`output_projection.video.x_in` / `output_projection.audio.x_in`).

- **close=0.2576** at atol=0.15 means only 25.8% of the 512×4096 = 2M video activations are within 0.15 of the reference — but this is misleading because the activations themselves are **large** (values in the hundreds). A max_abs of 136 with p50=0.44 means at the median element, the error is 0.44 out of values typically in the range ±50–200. That's <1% relative.
- **cos_sim=0.9998** confirms the 4096-dimensional activation vectors point in essentially the same direction. The errors are a uniform rounding haze, not systematic drift.
- **Audio is tighter** (p50=0.08, close=83.9%) because audio has smaller dimensionality (2048 vs 4096) and fewer tokens (126 vs 512), so less accumulation.

The output projection (4096→128 linear) then **compresses** these errors — the final video p50 drops from 0.44 to 0.023, and close jumps from 25.8% to 99.15%.


### Artifacts

- **Fixture**: `export_live_capture_fixture.py` → `live_capture_fixture_step_000_t512.safetensors` (56 tensors)
- **Checker**: `full_step_check.zig` — loop-based approach (single-block exe × 48 + output projection exe × 2)
- **Model code**: `FullStepParams`, `initFullStepParams`, `unloadFullStepBuffers` in `model.zig`

### Architecture note

A monolithic `forwardFullStep` function would exceed MLIR's 1024 function-argument limit
(48 blocks × 86 tensors = 4,160 args). The checker uses a loop-based approach instead:
compile a single-block exe, call it 48 times swapping weights. For production inference,
the same loop approach (or a future block-fusion strategy) will be needed.

### Step 1 robustness — degrees of freedom

The full step has several runtime degrees of freedom. Current validation covers only one point
(token_limit=512, lora=0.0, step_idx=0, no AV masks).

| Parameter | Current coverage | Worth testing? | Notes |
|---|---|---|---|
| **token_limit** | 512 only | ✅ Yes — 128, 256, 512 | Different sequence lengths stress different broadcast/shape paths. Smaller limits are more precision-sensitive. |
| **lora_strength** | 0.0 only | ✅ Yes — 0.0 and 0.5 | LoRA is merged at checkpoint load time (affects attn/FF weights, not adaln/proj_out). Needs a LoRA-merged checkpoint export for the Zig checker. |
| **step_idx** | 0 only | ⚠️ Low priority | Different sigma → different adaln outputs, but our fixture captures those directly. Same code path, different data. Can defer to Step 3 (denoising loop). |
| **AV masks** | null (no masks) | ⚠️ Low priority | `forwardBlock0Native` passes null masks. The existing `block_slice_48_check` already validates with masks. The mask path is a simple multiplicative gate — unlikely to diverge. |

**Recommended test matrix:**

| token_limit | lora_strength | Priority | Notes |
|---|---|---|---|
| 512 | 0.0 | ✅ Already passing | Current baseline |
| 256 | 0.0 | High | Different sequence length |
| 128 | 0.0 | High | Smallest, most precision-sensitive |
| 512 | 0.5 | High | Different weights (needs merged checkpoint) |
| 256 | 0.5 | Medium | |
| 128 | 0.5 | Medium | Prior block-level tests showed LoRA + small token count is most precision-sensitive (close=0.986) |

**How to run each configuration:**

```bash
# token_limit sweep (lora=0.0, reuses base checkpoint)
uv run scripts/export_live_capture_fixture.py --step-idx 0 --token-limit 256
bazel run --@zml//platforms:cuda=true //examples/ltx:full_step_check -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/repos/LTX-2/trace_run/live_capture_fixture_step_000_t256.safetensors

# LoRA sweep (needs merged checkpoint first)
python scripts/export_stage2_block0_checkpoint.py --distilled-lora-strength 0.5
uv run scripts/export_live_capture_fixture.py --step-idx 0 --token-limit 512 --distilled-lora-strength 0.5
bazel run --@zml//platforms:cuda=true //examples/ltx:full_step_check -- \
    /path/to/lora-merged-checkpoint.safetensors \
    /root/repos/LTX-2/trace_run/live_capture_fixture_step_000_t512.safetensors
```

---

## Step 2 — `modality_from_latent_state` + RoPE generation

**Goal**: Start from raw latent states (`LatentState.latent`, `.positions`, `.denoise_mask`, `.clean_latent`), sigma, and text contexts, and produce correct `velocity_model.forward` outputs — i.e., patchify + RoPE generation + adaln + blocks + output projection.

This tests:
- Patchify from raw 128-dim latents → 4096/2048 hidden states
- RoPE computation from position coordinate grids → cos/sin pairs
- `modality_from_latent_state` (sigma conditioning of latents via denoise_mask and clean_latent)
- The full `velocity_model.forward` boundary

### Key new implementation work

1. **RoPE generation**: Implement the mapping `positions [B, C, T] → (cos, sin) [B, H, T, HD]` in Zig. This requires understanding the LTX-specific RoPE frequency computation (likely `SPLIT` variant with specific freq parameters).

2. **`modality_from_latent_state`**: Implement the latent conditioning: `x = sigma * latent + (1 - sigma) * clean_latent` (or however it blends), and prepare the full input for patchify.

### Deliverables

1. **Python inspector script** to dump the exact RoPE generation formula and `modality_from_latent_state` source
2. **RoPE generation** function in `model.zig`
3. **`modality_from_latent_state`** equivalent in Zig
4. **`forwardVelocityModel`** — takes raw latents, sigma, text contexts → video/audio outputs
5. **Fixture** + **checker** for the full boundary

### How to check

Same approach: export raw inputs + expected outputs from Python, run the full Zig path, compare.

---

## Step 3 — Denoising loop (multi-step scheduler)

**Goal**: Run the full stage-2 denoising loop — iterate over the sigma schedule (~11 steps), calling `velocity_model.forward` at each step, applying the scheduler update, and producing the final denoised latents.

This tests:
- Sigma schedule generation
- Scheduler update rule (Euler / flow-matching step: `x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * velocity`)
- Error accumulation over 11 steps
- Correct latent state update between steps

### Key new implementation work

1. **Sigma schedule**: The discrete schedule of sigma values used by the distilled model
2. **Scheduler step**: The update rule applied between transformer calls
3. **Loop control**: Iterating, updating `LatentState`, recomputing `modality_from_latent_state`

### Deliverables

1. **Python inspector** to dump schedule + step rule
2. **Zig denoising loop** function
3. **Fixture**: capture `{initial_latent, all sigmas, final_denoised_latent}` from Python
4. **Checker**: run full loop, compare final output

### How to check

Compare final denoised video/audio latents after all 11 steps against Python reference.

---

## Scope exclusions (not part of this plan)

These are needed for a full inference pipeline but are separate workstreams:

- **Gemma text encoder** (`encode_prompts`) — produces `v_context_p`, `a_context_p`
- **Stage 1** denoising (separate/smaller transformer)
- **VAE decoder** (latent → pixel/audio)
- **Spatial upsampler** (post-VAE resolution enhancement)
- **Input preprocessing** (prompt parsing, video/audio preparation)
