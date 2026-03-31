# 01 — Cleaning: Stage 1 + Stage 2 Codebase

**Goal:** Reduce ~120 files (41 .zig, 77 .py, 18 .md) to a clean, production-ready set.
Keep only what's needed to run the mixed pipeline. Remove all diagnostic/checker scaffolding.

**Validation gate at every step:** After each step, run a local `bazel build` on the two
production targets. After Step 7, run a full mixed pipeline on the server.

```bash
# Quick build gate (run after every step):
bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e

# Full validation gate (run after Step 7):
# Run the full mixed pipeline from full_e2e_mixed.md "Full Command Reference"
```

---

## Current Inventory

| Category | Count | Notes |
|----------|-------|-------|
| Zig drivers (production) | 4 | `model.zig`, `denoise_stage1.zig`, `denoise_e2e.zig`, `main.zig` |
| Zig checkers (diagnostic) | 37 | `block0_*_check.zig`, `ff_*_check.zig`, `block_slice_*_check.zig`, etc. |
| Zig utils | 1 | `check_utils.zig` |
| Python fixture exports | ~35 | `export_block0_*.py`, `export_ff_*.py`, etc. |
| Python diagnostics | ~25 | `debug_*.py`, `analyze_*.py`, `inspect_*.py`, `trace_*.py`, etc. |
| Python pipeline scripts | 7 | `export_mixed_pipeline.py`, `bridge_s1_to_s2.py`, `validate_mixed_pipeline.py`, `export_stage1_inputs.py`, `e2e/export_stage2_inputs.py`, `e2e/decode_latents.py`, `export_all_step_latents.py` |
| Python comparison | ~7 | `compare_*.py`, `verify_*.py` |
| Markdown docs | 18 | Mix of plans, analyses, status reports |
| Shell scripts | 3 | `run_mixed_pipeline.sh`, `run_native_t2_matrix.sh`, `run_native_t3_slice.sh` |
| model.zig | ~5263 lines | Includes ~20 experimental `forwardBlock0*` variants |

---

## Steps

### Step 1: Archive diagnostic files (Zig checkers + Python fixtures)

Move all checker/diagnostic/fixture files to `examples/ltx/_archive/`.
This is reversible — nothing is deleted.

**Zig files to archive (37 checkers + 1 utility):**
```
check_utils.zig
ff_parity.zig
attention_forward_check.zig
ff_forward_check.zig ff_gelu_check.zig ff_linear1_check.zig ff_linear2_check.zig
ff_linear1_gelu_check.zig ff_forward_staged_check.zig ff_forward_staged_h2_validation_check.zig
patchify_forward_check.zig output_projection_check.zig adaln_single_check.zig
sigma_schedule_check.zig guider_combine_check.zig check_noise_init.zig
step2_check.zig full_step_check.zig step3_check.zig
live_capture_check.zig model_check.zig
block0_forward_check.zig block0_ff_residual_check.zig block0_self_attn_check.zig
block0_full_check.zig block0_native_check.zig block0_text_ca_check.zig
block0_audio_ff_residual_check.zig block0_audio_self_attn_check.zig
block0_audio_text_ca_check.zig block0_av_a2v_check.zig block0_av_v2a_check.zig
audio_ff_forward_check.zig block_slice_48_check.zig block_slice_native_check.zig
perblock_check.zig stg_block_check.zig
```

**Python files to archive (~60+ fixture/diagnostic/analysis scripts):**
All `export_block0_*.py`, `export_ff_*.py`, `export_adaln_*.py`, `export_attention_*.py`,
`export_patchify_*.py`, `export_audio_*.py`, `export_output_projection_*.py`,
`export_guider_combine_*.py`, `export_live_capture_*.py`, `export_noise_init_*.py`,
`export_activation_*.py`, `export_full_step_*.py`, `export_step2_*.py`, `export_step3_*.py`,
`export_perblock_*.py`, `export_stg_block_*.py`, `export_block_slice_*.py`,
`export_stage2_block*.py`, `export_stage1_step0_reference.py`,
`debug_*.py`, `analyze_*.py`, `inspect_*.py`, `trace_*.py`, `diag_*.py`,
`diagnostic_*.py`, `attention_kind_suite.py`, `compare_*.py`, `verify_*.py`,
`check_*.py` (Python ones), `retrace_*.py`, `replay_*.py`,
`sampling-activations*.py`, `inspect-pt.py`, `zml_utils.py`

**Shell scripts to archive:**
```
run_native_t2_matrix.sh run_native_t3_slice.sh
```

**Markdown files to archive:**
```
ZIG_DIAGNOSTIC_USAGE.md ZIG_IMPLEMENTATION_COMPLETE.md
PIPELINE_STATUS.md DIAGNOSTIC_QUICKSTART.md DIAGNOSTIC_CHECKLIST.md
DIAGNOSTIC_COMPLETE.md DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md DIAGNOSTIC_PIPELINE.md
STAGE1_IMPLEMENTATION_PLAN.md ADALN_SINGLE_IMPLEMENTATION.md
OUTPUT_PROJECTION_IMPLEMENTATION.md VELOCITY_MODEL_PLAN.md
block0_reverse_engineering_map.md transformer_threading_progress.md
AUDIO_PARITY_FINDINGS_2026-03-24.md
```

**Keep in place:**
- `model.zig`, `denoise_stage1.zig`, `denoise_e2e.zig`, `main.zig`
- `export_mixed_pipeline.py`, `bridge_s1_to_s2.py`, `validate_mixed_pipeline.py`
- `export_stage1_inputs.py`, `export_all_step_latents.py`, `export_step0_perpass.py`
- `e2e/decode_latents.py`, `e2e/export_stage2_inputs.py`
- `run_mixed_pipeline.sh`
- `BUILD.bazel`
- `README.md`, `full_e2e_mixed.md`, `STAGE1_DIVERGENCE_ANALYSIS.md`
- This file (`01_cleaning_stage1_stage2.md`)

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 2: Update BUILD.bazel

Remove all `zig_test` / `zig_binary` targets for archived checker files.
Keep only:
- `denoise_stage1`
- `denoise_e2e`
- `main` (safetensors inspector)

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 3: Remove experimental forwardBlock0* variants from model.zig

Remove all checker-only / experimental forward functions and their associated
`BasicAVTransformerBlock` method wrappers:

**Block-level wrappers to remove:**
- `forwardBlock0NativeSTGWithAVMasks` — unused (STG pass doesn't use AV masks in production)
- `forwardBlock0NativeWithAVMasksVideoAllResidualsF32` — checker-only
- `forwardBlock0NativeVideoIntermediatesWithAVMasks` — debug helper
- `forwardBlock0NativeAudioIntermediatesWithAVMasks` — debug helper (if exists)
- All `forwardBlock0*Diag*` variants
- `forwardBlock0FF`, `forwardBlock0FFBoundary` — FF-only (checker)
- `forwardFF`, `forwardFFLinear1`, `forwardFFLinear2`, `forwardFFGeluBf16`, `forwardFFGeluF32`, `forwardFFLinear1Gelu`, `forwardFFLinear1GeluF32` — FF sub-op (checker)
- `castToF32` — checker utility

**BasicAVTransformerBlock methods to remove:**
- `forwardNativeAudioFFResidualF32` — checker-only
- `forwardNativeAudioAllResidualsF32` — checker-only
- `forwardNativeVideoAllResidualsF32` — checker-only

**Structs to remove:**
- `Block0NativeAudioIntermediates` — debug only
- `Block0NativeVideoIntermediates` — debug only
- `Block0VideoStageOutputs`, `Block0AudioStageOutputs` — checker stage outputs
- `BlockSlice8FullParams` — unused
- `BlockSlice48FullParams` — checker only

**Keep:**
- `forwardBlock0Native` — production (passes 1, 2, and non-STG blocks of pass 3)
- `forwardBlock0NativeBf16Attn` — production (bf16 attention variant)
- `forwardBlock0NativeSTG` — production (pass 3 block 28)
- `forwardBlock0NativeSTGBf16Attn` — production (pass 3 block 28 bf16)
- `forwardBlock0NativeWithAVMasks` — production (pass 4, isolated modality)
- `forwardBlock0NativeWithAVMasksBf16Attn` — production (pass 4 bf16)
- `forwardPreprocess` — production
- `forwardOutputProjection` — production
- `forwardGuiderCombine` — production
- `forwardToDenoised` — production
- `forwardDenoisingStep` — production
- `forwardDenoisingStepFromX0` — production
- `forwardNoiseInit` — production
- `computeSigmaSchedule` — production
- All core structs: `Config`, `FeedForward`, `Attention`, `BasicAVTransformerBlock`,
  `LTXModel`, `Patchify`, `AdaLayerNormSingle`, `OutputProjection`,
  `PreprocessOutput`, `PreprocessParams`, `SharedInputs`, `FullStepParams`,
  `Block0FullParams`, `GuiderCombineResult`, `DenoisingStepResult`

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 4: Add section comments to model.zig mapping to Python reference

Annotate the remaining code with clear section headers mapping to the Python reference:

```
// ============================================================================
// Section 1: Configuration & Constants
// Python ref: ltx_core/models/ltx_model.py — LTXConfig
// ============================================================================

// ============================================================================
// Section 2: Patchification (Video + Audio)
// Python ref: ltx_core/models/patchifiers/ — VideoLatentPatchifier, AudioPatchifier
// ============================================================================

// ============================================================================
// Section 3: Positional Embeddings (RoPE)
// Python ref: ltx_core/models/embeddings.py — RoPE3D
// ============================================================================

// ============================================================================
// Section 4: AdaLayerNormSingle (timestep modulation)
// Python ref: ltx_core/models/normalization.py — AdaLayerNormSingle
// ============================================================================

// ============================================================================
// Section 5: Attention (self-attention, cross-attention, AV cross-attention)
// Python ref: ltx_core/models/attention.py — Attention, CrossAttention
// ============================================================================

// ============================================================================
// Section 6: FeedForward
// Python ref: ltx_core/models/attention.py — FeedForward
// ============================================================================

// ============================================================================
// Section 7: BasicAVTransformerBlock (single block forward)
// Python ref: ltx_core/models/transformer_ltx_2.py — BasicAVTransformerBlock.forward()
// ============================================================================

// ============================================================================
// Section 8: LTXModel (full transformer stack)
// Python ref: ltx_core/models/ltx_model.py — LTXModel.forward()
// ============================================================================

// ============================================================================
// Section 9: Preprocessing (patchify + embed + RoPE + AV mask computation)
// Python ref: ltx_core/models/ltx_model.py — LTXModel._prepare_inputs()
// ============================================================================

// ============================================================================
// Section 10: Output Projection
// Python ref: ltx_core/models/ltx_model.py — LTXModel._postprocess()
// ============================================================================

// ============================================================================
// Section 11: Denoising Step (sigma schedule + Euler step + mask blending)
// Python ref: ltx_pipelines/scheduler.py — RectifiedFlowScheduler
// ============================================================================

// ============================================================================
// Section 12: Guidance Combine (CFG + STG + modality isolation + rescale)
// Python ref: ltx_pipelines/guiders.py — LTXGuider.combine()
// ============================================================================

// ============================================================================
// Section 13: Block-Level Entrypoints (forwardBlock0* family)
// These are the ZML-compilable entrypoints — each wraps a Section 7 method
// with explicit tensor arguments (no struct, for MLIR arg flattening).
// ============================================================================
```

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 5: Add section comments to denoise_stage1.zig and denoise_e2e.zig

Annotate both drivers with clear section headers:

```
// Section A: CLI argument parsing
// Section B: Sigma schedule computation
// Section C: Open stores + load inputs
// Section D: Compile executables (preprocessing, block, projection, denoising, guider)
// Section E: Load weights
// Section F: Denoising loop
//   F.1: Preprocessing
//   F.2: Pass 1 — Conditional (positive context)
//   F.3: Pass 2 — Negative/CFG (negative context)
//   F.4: Pass 3 — STG (V-passthrough at block 28)
//   F.5: Pass 4 — Isolated (zero AV masks)
//   F.6: Guider combine
//   F.7: Euler step
// Section G: Write output
```

For `denoise_e2e.zig` (simpler — single pass, 3 steps):
```
// Section A: CLI argument parsing
// Section B: Open stores + load inputs
// Section C: Compile executables
// Section D: Load weights
// Section E: Denoising loop (3-step Euler)
// Section F: Write output
```

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 6: Clean up README.md

Rewrite `README.md` as a concise guide to the production codebase:

- **What it does** (mixed LTX-2 pipeline)
- **File map** (model.zig, denoise_stage1.zig, denoise_e2e.zig + Python scripts)
- **How to run** (pointer to `full_e2e_mixed.md` command reference)
- **Architecture overview** (Python ↔ Zig boundary flow)

**Validate:** `bazel build //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e`

---

### Step 7: Full pipeline validation on server

Run the complete mixed pipeline with a new prompt to confirm nothing broke:

```bash
scp examples/ltx/model.zig examples/ltx/denoise_stage1.zig \
    examples/ltx/denoise_e2e.zig root@dev-oboulant:/root/repos/zml/examples/ltx/

# Then on the server: run full mixed pipeline (see full_e2e_mixed.md command reference)
```

**Expected:** identical output to pre-cleaning run (same binary, just fewer source files).

---

## Post-Cleaning Inventory (expected)

| Category | Count | Files |
|----------|-------|-------|
| Zig production | 4 | `model.zig`, `denoise_stage1.zig`, `denoise_e2e.zig`, `main.zig` |
| Python pipeline | 8 | `export_mixed_pipeline.py`, `bridge_s1_to_s2.py`, `validate_mixed_pipeline.py`, `export_stage1_inputs.py`, `export_all_step_latents.py`, `export_step0_perpass.py`, `e2e/decode_latents.py`, `e2e/export_stage2_inputs.py` |
| Shell | 1 | `run_mixed_pipeline.sh` |
| Build | 1 | `BUILD.bazel` |
| Docs | 4 | `README.md`, `full_e2e_mixed.md`, `STAGE1_DIVERGENCE_ANALYSIS.md`, `01_cleaning_stage1_stage2.md` |
| Archive | ~100 | `_archive/` (all diagnostic/checker/fixture files, recoverable) |
| model.zig | ~3000 lines (est.) | Down from ~5263 — removed ~2000 lines of experimental variants |
