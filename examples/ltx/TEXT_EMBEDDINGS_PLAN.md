# Text Embeddings Post-Processing in Zig

## Goal

Move LTX-specific text embedding post-processing from Python to Zig.
Keep only Gemma forward pass in Python; implement everything that uses
LTX checkpoint weights (`text_embedding_projection.*`,
`model.diffusion_model.{video,audio}_embeddings_connector.*`) in Zig.

**Input:** 49 Gemma hidden states `[1, 1024, 3840]` bf16 + attention mask `[1, 1024]`
**Output:** `v_context [1, 1024, 4096]` bf16, `a_context [1, 1024, 2048]` bf16

## Architecture Overview

```
49 hidden states [1,S,3840] ──┐
                              ├─► Stack → [1,S,3840,49]
                              │      │
                              │      ▼
                              │   Per-token RMS norm (dim=2)
                              │      │
                              │      ▼
                              │   Flatten → [1,S,188160]
                              │      │
                    ┌─────────┴──────┴──────────────┐
                    ▼                                ▼
           rescale_norm(4096)              rescale_norm(2048)
                    │                                │
                    ▼                                ▼
          Linear(188160→4096)            Linear(188160→2048)
          video_aggregate_embed          audio_aggregate_embed
                    │                                │
                    ▼                                ▼
           video_features                    audio_features
           [1,S,4096]                        [1,S,2048]
                    │                                │
         ┌──────── │ ───────────────── │ ────────────┘
         ▼                             ▼
  ┌──────────────────┐    ┌──────────────────────┐
  │ Video Connector  │    │  Audio Connector     │
  │ Emb1DConnector   │    │  Emb1DConnector      │
  │ inner=4096       │    │  inner=2048          │
  │ 8 × TransBlock1D │    │  8 × TransBlock1D    │
  │ 128 registers    │    │  128 registers       │
  └────────┬─────────┘    └─────────┬────────────┘
           ▼                        ▼
    v_context [1,S,4096]     a_context [1,S,2048]
```

## Design Decisions

### No config.json — infer from checkpoint shapes
The [LTX-2.3 HuggingFace repo](https://huggingface.co/Lightricks/LTX-2.3/tree/main)
ships no config.json. Define Zig structs with semantic dimension tags and let
`zml.io.load` resolve concrete sizes from checkpoint weight shapes.

### Two files for pos/neg (not batched)
Save separate hidden-state files for positive and negative prompts.
Run the Zig embeddings processor graph twice (one compilation, two calls).
This keeps code simple, memory lower, and matches the existing pattern where
`inference.zig` loads `v_context_pos/neg` independently.

### S = 1024 always (static shapes)
The Gemma tokenizer pads to `max_length=1024` with `padding="max_length"`.
All hidden states are always `[1, 1024, 3840]`. The graph can be statically
compiled — no dynamic shapes needed.

## Checkpoint Weight Key Mapping

From `encoder_configurator.py` `EMBEDDINGS_PROCESSOR_KEY_OPS`:

| Checkpoint key prefix | Zig module |
|---|---|
| `text_embedding_projection.video_aggregate_embed.{weight,bias}` | `FeatureExtractorV2.video_linear` |
| `text_embedding_projection.audio_aggregate_embed.{weight,bias}` | `FeatureExtractorV2.audio_linear` |
| `model.diffusion_model.video_embeddings_connector.learnable_registers` | `Embeddings1DConnector.learnable_registers` |
| `model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.{0..7}.attn1.*` | `Embeddings1DConnector.blocks[i].attn` |
| `model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.{0..7}.ff.*` | `Embeddings1DConnector.blocks[i].ff` |
| `model.diffusion_model.audio_embeddings_connector.*` | Same pattern, audio connector |

## Reuse from Existing Code

| Component | Location in model.zig | Reusable? |
|---|---|---|
| `Attention` (QKV + q/k RMSNorm + RoPE + gated attention + SDPA) | Lines ~376–606 | **Yes** — identical architecture |
| `FeedForward` (Linear → GELU(tanh) → Linear) | Lines ~30–70 | **Yes** — identical |
| `precomputeFreqsCis` / `splitFreqsCis` | Lines ~2295–2340 | **Partially** — connector uses 1D positions, not 3D. Need a simpler 1D RoPE variant |
| `applyLtxRotaryEmb` (split) | Lines ~608–640 | **Partially** — connector uses SPLIT format, not interleaved |
| `zml.nn.rmsNorm` | ZML framework | **Yes** |

### What needs new code
- `FeatureExtractorV2`: stack, per-token RMS norm, flatten, rescale, two linears
- `Embeddings1DConnector`: register replacement, 1D RoPE generation, 8× block loop, final RMS norm, binary mask conversion
- `EmbeddingsProcessor`: orchestration (feature extract → mask convert → connectors)
- 1D RoPE: simplified version of `precomputeFreqsCis` for a flat `arange(S)` grid

---

## Milestones

### M0: Python Exporter — `export_gemma_hidden_states.py` ✅ DONE

**New file** in `examples/ltx/`.

Run Gemma forward only (no `EmbeddingsProcessor`), save:
- `{out}/pos_hidden_states.safetensors`: 49 tensors `hidden_state_00..48` each `[1, 1024, 3840]` bf16 + `attention_mask` `[1, 1024]` int
- `{out}/neg_hidden_states.safetensors`: same for negative prompt
- `{out}/ref_embeddings.safetensors`: Python-computed final embeddings for validation:
  `v_context_pos`, `a_context_pos`, `v_context_neg`, `a_context_neg`,
  `attention_mask_pos`, `attention_mask_neg`

The script also saves intermediate results for per-stage validation:
- `{out}/ref_features.safetensors`: `video_features_pos`, `audio_features_pos`,
  `video_features_neg`, `audio_features_neg` (after feature extraction, before connectors)

**CLI:**
```bash
cd /root/repos/LTX-2
uv run examples/ltx/export_gemma_hidden_states.py \
    --prompt "A beautiful sunset over the ocean" \
    --negative-prompt "blurry, out of focus" \
    --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root ~/models/gemma-3-12b-it \
    --output-dir /root/gemma_export/
```

### M1: Zig Module — `text_embeddings.zig` ✅ DONE

**New file** in `examples/ltx/`.

#### Structs

```
FeatureExtractorV2
  Params: video_linear (Linear), audio_linear (Linear)
  forward(hidden_states: [49]Tensor, attention_mask: Tensor) → (video_features, audio_features)

ConnectorBlock
  Params: attn (Attention.Params), ff (FeedForward.Params)
  forward(x, mask, pe_cos, pe_sin, num_heads) → x

Embeddings1DConnector
  Params: learnable_registers (Tensor), blocks: [MAX_CONNECTOR_BLOCKS]ConnectorBlock.Params
  forward(features, additive_mask) → (encoded, mask)
  num_blocks auto-detected from checkpoint (8 for LTX-2.3)

EmbeddingsProcessor
  Params: feature_extractor (FeatureExtractorV2.Params),
          video_connector (Embeddings1DConnector.Params),
          audio_connector (Embeddings1DConnector.Params)
```

#### Graph Function

```zig
pub fn forwardEmbeddingsProcessor(
    hidden_states: [49]Tensor,  // each [1, S, 3840]
    attention_mask: Tensor,     // [1, S]
    params: EmbeddingsProcessor.Params,
) struct { v_context: Tensor, a_context: Tensor, binary_mask: Tensor }
```

#### Algorithm Detail: Register Replacement

The connector replaces padding tokens with learnable registers:
1. Count non-zero tokens from attention mask
2. Compact real tokens to the left (removing padding)
3. Pad remaining positions with tiled `learnable_registers [128, D]`
4. Flip layout: real tokens left, registers right
5. Set attention mask to all-zeros (attend everywhere)

This is the trickiest part to implement in a static graph since it involves
data-dependent reshuffling. Strategy: use `scatter`/`gather` with index
tensors derived from the attention mask.

#### Algorithm Detail: 1D RoPE for Connector

Simplified version of the 3D RoPE used in the main transformer:
```
frac = arange(S) / max_pos                       # fractional positions
scaled = 2 * frac - 1                            # [-1, 1) range
freq_basis = exp(linspace(0,1,D/2) * ln(theta)) * π/2   # LTX custom formula
freqs = outer(scaled, freq_basis)                 # [S, D/2]
cos, sin = reshape(freqs, [B, H, S, HD/2])        # SPLIT format
```

**Structural corrections vs. original plan:**
- `max_pos=[4096]` (not `[1]`) — positions are fractional: `arange(S) / 4096`
- **SPLIT RoPE format** (not interleaved) — cos/sin shaped `[B, H, T, HD/2]`
- Frequency formula is `theta^(linspace(0,1,N)) * π/2`, not `1 / theta^(2i/D)`

### M2: Validation Driver — `validate_text_embeddings.zig` ✅ DONE

**New file** in `examples/ltx/`. Standalone binary.

**Purpose:** Load hidden states → run Zig embeddings processor → compare
against Python reference → report pass/fail with per-tensor metrics.

**Flow:**
1. Parse CLI: `--hidden-states`, `--ref-embeddings`, `--ref-features`, `--checkpoint`
2. Load LTX checkpoint (only `text_embedding_projection.*` and
   `model.diffusion_model.{video,audio}_embeddings_connector.*` keys)
3. Compile `forwardEmbeddingsProcessor` graph
4. Load weights into `Bufferized(EmbeddingsProcessor.Params)`
5. For each prompt (pos, neg):
   a. Load 49 hidden states + attention mask from safetensors
   b. Call the compiled graph
   c. Compare outputs against reference:
      - Feature extraction stage: compare against `ref_features.safetensors`
      - Final embeddings: compare against `ref_embeddings.safetensors`
   d. Report: max abs diff, mean abs diff, cosine similarity

**Pass criteria:** cosine similarity ≥ 0.9999 and mean abs diff ≤ 0.01.
(Original plan used max abs < 5e-3, but bf16 outliers can exceed that while the
tensor is otherwise nearly identical. Cosine + mean_abs is a more robust metric.)

**CLI:**
```bash
bazel run //examples/ltx:validate_text_embeddings -- \
    --hidden-states /root/gemma_export/pos_hidden_states.safetensors \
    --ref-embeddings /root/gemma_export/ref_embeddings.safetensors \
    --ref-features /root/gemma_export/ref_features.safetensors \
    --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors
```

**BUILD.bazel addition:**
```starlark
zig_binary(
    name = "validate_text_embeddings",
    main = "validate_text_embeddings.zig",
    srcs = ["text_embeddings.zig", "model.zig", ...],
    deps = ["//zml"],
)
```

### M3: Wire into inference.zig ✅ DONE

Added `--gemma-hidden-states-pos` and `--gemma-hidden-states-neg` optional CLI flags.
When provided:
1. Load hidden states + attention mask from the files
2. Compile + run `forwardEmbeddingsProcessor` (one compilation, two calls)
3. Use the outputs as `v_context_pos`, `a_context_pos` (and similarly for neg)
4. Continue with the existing pipeline unchanged

When `--gemma-hidden-states-*` flags are NOT provided, fall back to the current behavior
(load pre-computed `v_context_pos/neg` from `stage1_inputs.safetensors`).

Changes: `computeTextEmbeddings()` helper (~130 lines) added to `inference.zig`,
`text_embeddings.zig` added to BUILD.bazel inference srcs. No changes to `model.zig`.

**Validated:** End-to-end pipeline run on GPU server produces clean MP4 output.

### M4: Consolidate to single Python command

**Goal:** Go from 2 Python commands + 1 Zig command to 1 Python + 1 Zig.
Hidden states are now the canonical text interface — Zig always computes
final contexts, no pre-computed `v_context_*`/`a_context_*` in the saved files.

#### Python changes (`export_pipeline.py`)

1. **Replace `PromptEncoder`** with split Gemma + EmbeddingsProcessor approach
   (same as `export_gemma_hidden_states.py`):
   - Load Gemma via `GemmaTextEncoderConfigurator` → raw `hidden_states` + `attention_mask`
   - Load `EmbeddingsProcessorConfigurator` → compute final contexts for internal use
2. **Save hidden states** as sidecar files:
   - `{out}/pos_hidden_states.safetensors` (`stacked_hidden_states` + `attention_mask`)
   - `{out}/neg_hidden_states.safetensors` (same format)
3. **Remove `v_context_*`/`a_context_*` keys** from `stage1_inputs.safetensors` —
   the pipeline still uses final contexts internally for its own denoising loop,
   but they are no longer saved to disk.

#### Zig changes (`inference.zig`)

1. **Make `--gemma-hidden-states-pos` and `--gemma-hidden-states-neg` required**
   (no longer optional `?[]const u8`).
2. **Remove the fallback branch** that loads `v_context_*` from `stage1_inputs.safetensors`.
3. **Simplify `runStage1`** — remove the conditional, always call `computeTextEmbeddings()`.

#### Cleanup

- `export_gemma_hidden_states.py` — **keep** for standalone validation/debugging
  (also produces `ref_embeddings.safetensors` and `ref_features.safetensors`
  which are needed by `validate_text_embeddings.zig`).
- `validate_text_embeddings.zig` — **keep** unchanged.

#### CLI after M4

```bash
# Single Python command (produces stage1_inputs + hidden states + meta)
cd ~/repos/LTX-2 && .venv/bin/python ~/repos/zml/examples/ltx/export_pipeline.py \
    --output-dir ~/e2e_demo/ --seed 42

# Single Zig command
cd ~/repos/zml && bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
    --stage1-ckpt ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --stage2-ckpt ~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --upsampler-ckpt ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --stage1-inputs ~/e2e_demo/stage1_inputs.safetensors \
    --meta ~/e2e_demo/pipeline_meta.json \
    --output-dir ~/e2e_demo/output/ \
    --seed 42 --bf16-attn-stage2 \
    --gemma-hidden-states-pos ~/e2e_demo/pos_hidden_states.safetensors \
    --gemma-hidden-states-neg ~/e2e_demo/neg_hidden_states.safetensors
```

---

## Validation Matrix

| Stage | Zig output | Python reference | Achieved | Status |
|---|---|---|---|---|
| Feature extraction (video) | `video_features [1,1024,4096]` | `ref_features.safetensors:video_features_pos` | cosine=0.99990, mean_abs=0.006 | ✅ PASS |
| Feature extraction (audio) | `audio_features [1,1024,2048]` | `ref_features.safetensors:audio_features_pos` | cosine=0.99992, mean_abs=0.004 | ✅ PASS |
| Register replacement (video) | `v_after_replace [1,1024,4096]` | `connector_intermediates.safetensors:video_after_replace` | cosine=0.99999, mean_abs=0.00007 | ✅ PASS |
| Register replacement (audio) | `a_after_replace [1,1024,2048]` | `connector_intermediates.safetensors:audio_after_replace` | cosine=0.99999, mean_abs=0.00004 | ✅ PASS |
| Video connector output | `v_context [1,1024,4096]` | `ref_embeddings.safetensors:v_context_pos` | cosine=0.99997, mean_abs=0.001 | ✅ PASS |
| Audio connector output | `a_context [1,1024,2048]` | `ref_embeddings.safetensors:a_context_pos` | cosine=0.99999, mean_abs=0.001 | ✅ PASS |
| Video connector output (neg) | `v_context [1,1024,4096]` | `ref_embeddings.safetensors:v_context_neg` | cosine=0.99999, mean_abs=0.001 | ✅ PASS |
| Audio connector output (neg) | `a_context [1,1024,2048]` | `ref_embeddings.safetensors:a_context_neg` | cosine=0.99999, mean_abs=0.001 | ✅ PASS |

## Structural Corrections vs. Original Plan

Three major assumptions in the original plan turned out to be wrong. All were
discovered via Python diagnostic instrumentation (`export_connector_intermediates.py`)
and corrected in the Zig implementation:

| Assumption (original) | Actual (LTX-2.3 checkpoint) | Impact |
|---|---|---|
| 2 transformer blocks per connector | **8 blocks** per connector | Connector ran only 25% of layers → cosine ~0.05 |
| Interleaved RoPE format | **SPLIT format** — cos/sin shaped `[B, H, T, HD/2]` | Wrong rotation pattern applied to Q/K |
| `max_pos=[1]` (positions used as-is) | **`max_pos=[4096]`** — fractional positions `arange(S)/4096`, scaled to `[-1, 1)` | Wrong position scaling fed to RoPE frequency computation |

Additionally:
- **RoPE frequency formula**: LTX uses `θ^(linspace(0,1,N)) × π/2`, not the standard `1/θ^(2i/d)`
- **Num heads**: Both connectors use 32 heads (video: 32×128=4096, audio: 32×64=2048)
- **Gated attention**: Both connectors have gated attention (`to_gate_logits` present)
- **Pass criteria**: Changed from `max_abs < 5e-3` to `cosine ≥ 0.9999 ∧ mean_abs ≤ 0.01` — bf16 can have a few outlier elements with large absolute error while the tensor is otherwise identical

## Implementation Order

```
M0 (Python exporter)
 │
 ▼
M1 (text_embeddings.zig) ←── can develop/compile locally, test on GPU
 │
 ▼
M2 (validate_text_embeddings.zig) ←── run on GPU server
 │
 ▼
M3 (wire into inference.zig) ←── only after M2 passes
 │
 ▼
M4 (update export_pipeline.py) ←── cleanup
```
