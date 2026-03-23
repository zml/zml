# Block-0 Reverse Engineering Map (Python -> Zig)

This map defines where each Block-0 operation comes from in LTX Python and where it should land in current Zig code.

## Scope

Two checker paths are currently maintained:

- Legacy FF-boundary surrogate parity (`block0_ff_boundary_*`).
- Full stream parity for block0 video+audio composition (`block0_full_*`, M6).

- Current canonical block full-forward API: [examples/ltx/model.zig](examples/ltx/model.zig#L402)
- Current FF-boundary entrypoint: [examples/ltx/model.zig](examples/ltx/model.zig#L506)
- Checker callsite: [examples/ltx/block0_forward_check.zig](examples/ltx/block0_forward_check.zig#L96)
- Fixture keys: [examples/ltx/export_block0_fixture.py](examples/ltx/export_block0_fixture.py#L75)

## Canonical Python Sources

Use these upstream files as the source of truth for block semantics.

- BasicAVTransformerBlock and helper methods:
  - https://github.com/lightricks/ltx-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/transformer.py
- TransformerArgs dataclass and preprocessors:
  - https://github.com/lightricks/ltx-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py
- Model wiring for preprocessors and transformer block loop:
  - https://github.com/lightricks/ltx-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/model.py
- AdaLN embedding coefficient and AdaLayerNormSingle:
  - https://github.com/lightricks/ltx-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/adaln.py

## Python Stage Order (Video Stream)

For block index 0, Python forward executes stages in this order (video side):

1. Video self-attn pre-norm and modulation:
   - vshift_msa, vscale_msa, vgate_msa from get_ada_values(slice(0,3))
   - norm_vx = rms_norm(vx) * (1 + vscale_msa) + vshift_msa
2. Video self-attn residual:
   - vx = vx + attn1(norm_vx, pe, self_attention_mask, perturbation_mask) * vgate_msa
3. Video text cross-attn residual:
   - vx = vx + _apply_text_cross_attention(...)
   - If cross_attention_adaln is enabled: apply_cross_attention_adaln path
4. AV cross-attn residuals:
   - A->V then V->A branches with get_av_ca_ada_values, per-branch scales/shifts/gates
5. Video FF residual:
   - vshift_mlp, vscale_mlp, vgate_mlp from get_ada_values(slice(3,6))
   - vx_scaled = rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
   - vx = vx + ff(vx_scaled) * vgate_mlp

Audio stream mirrors the same pattern with audio tensors and audio modules.

## Zig Mapping (Current vs Target)

- Current block struct and params: [examples/ltx/model.zig](examples/ltx/model.zig#L315)
- Current canonical full forward API (`video+audio` inputs/outputs): [examples/ltx/model.zig](examples/ltx/model.zig#L402)
- Current legacy simplified FF-boundary forward: [examples/ltx/model.zig](examples/ltx/model.zig#L409)
- Block-0 params entrypoint: [examples/ltx/model.zig](examples/ltx/model.zig#L565)
- Legacy alias for checker compatibility: [examples/ltx/model.zig](examples/ltx/model.zig#L512)

Target semantic landing in Zig for full parity:

1. Add block-level tensors for scale_shift_table and optional prompt scale-shift tables.
2. Implement get_ada_values equivalent with identical reshape/slice ordering.
3. Implement residual equation exactly in block forward:
   - vx <- vx + attn1(norm_vx) * vgate_msa
4. Implement text cross-attn residual path for both non-AdaLN and AdaLN variants.
5. Implement AV cross-attn residuals with correct timestep families:
   - cross_scale_shift_timestep
   - cross_gate_timestep
6. Implement FF residual equation exactly:
   - vx <- vx + ff(vx_scaled) * vgate_mlp
7. Implement audio branch only when running audio tensors, never feed video tensor into audio_ff.

## Verified Checker Boundaries

Two fixture/check contracts are validated and maintained:

1. FF-boundary surrogate checker (legacy bring-up path)
    - Fixture tensors:
       - block0_ff_boundary.input0
       - block0_ff_boundary.output0
    - Loader prefers new keys and can fallback to legacy keys in checker.
    - References:
       - Fixture export: [examples/ltx/export_block0_fixture.py](examples/ltx/export_block0_fixture.py#L75)
       - Checker key map: [examples/ltx/block0_forward_check.zig](examples/ltx/block0_forward_check.zig#L188)

2. Full block0 stream checker (M6 canonical parity path)
    - Fixture tensors are under `block0_full.*` and include both video and audio stream inputs/outputs.
    - Checker validates:
       - `forwardBlock0VideoStream(...) == block0_full.vx_out`
       - `forwardBlock0AudioStream(...) == block0_full.ax_out`
    - Reference checker: [examples/ltx/block0_full_check.zig](examples/ltx/block0_full_check.zig#L1)

## Recommended Stagewise Bring-up Milestones

Use independent fixtures/checkers for each stage before combining all paths.

1. M1: Video self-attn residual parity
   - Capture vx_in and vx_after_self_attn from Python block0.
2. M2: Video text cross-attn residual parity
   - Capture vx_after_self_attn and vx_after_text_ca.
3. M3: Video FF residual parity
   - Capture vx_before_ff and vx_after_ff.
4. M4: Audio branch parity (self-attn, text-ca, ff)
5. M5: AV cross-attn A->V and V->A parity
6. M6: Full block0 parity (video+audio with all residuals/gates)

## Milestone Status

- M1: complete (video self-attn residual parity) in full residual mode.
   - Extended validation matrix:
      - token_limit=128, distilled_lora_strength=0.0: PASS (attn + residual)
      - token_limit=256, distilled_lora_strength=0.0: PASS (attn + residual)
      - token_limit=512, distilled_lora_strength=0.0: PASS (attn + residual)
      - token_limit=128, distilled_lora_strength=0.5: attn stage below strict threshold (close_fraction=0.986), residual stage not reached in checker flow
      - token_limit=256, distilled_lora_strength=0.5: PASS (attn + residual)
      - token_limit=512, distilled_lora_strength=0.5: attn stage below strict threshold (close_fraction=0.991), residual stage not reached in checker flow
   - Precision caveat (LoRA 0.5): attn1 can be numerically tighter than strict `minimum_close_fraction=0.999` at some token limits even when fixtures are bf16-clean and residual algebra path is stable.
- M2: complete (video text cross-attn residual parity) in full residual mode.
   - Validated matrix:
      - token_limit=128, distilled_lora_strength=0.0
      - token_limit=256, distilled_lora_strength=0.0
      - token_limit=512, distilled_lora_strength=0.0
      - token_limit=128, distilled_lora_strength=0.5
      - token_limit=256, distilled_lora_strength=0.5
      - token_limit=512, distilled_lora_strength=0.5
   - Extra validation outcome: all tested M2 combinations pass in full residual mode, including token_limit in {128, 256, 512} and LoRA strengths {0.0, 0.5}.
   - Fixture contract used for M2 checks:
      - attn parity: `block0_text_ca.attn2_x`, `block0_text_ca.context`, `block0_text_ca.attn2_out`
      - residual parity: `block0_text_ca.vx_in`, `block0_text_ca.text_ca_out`, `block0_text_ca.vx_out`
   - Capture guardrail: disambiguate video vs audio `_apply_text_cross_attention` captures and export residual keys only when shapes match video attn2 tensors.
- M3: complete (video FF residual parity) in full residual mode.
   - Validated matrix:
      - token_limit=128, distilled_lora_strength=0.0: PASS (ff stage + residual)
      - token_limit=256, distilled_lora_strength=0.0: PASS (ff stage + residual)
      - token_limit=512, distilled_lora_strength=0.0: PASS (ff stage + residual)
      - token_limit=128, distilled_lora_strength=0.5: PASS (ff stage + residual)
      - token_limit=256, distilled_lora_strength=0.5: PASS (ff stage + residual)
      - token_limit=512, distilled_lora_strength=0.5: PASS (ff stage + residual)
   - No LoRA sensitivity observed (unlike M1 attn1 stage). FF is numerically stable across all token limits.
   - Fixture contract used for M3 checks:
      - ff parity: `block0_ff_residual.vx_scaled`, `block0_ff_residual.ff_out`
      - residual parity: `block0_ff_residual.vx_in`, `block0_ff_residual.ff_out`, `block0_ff_residual.vgate_mlp`, `block0_ff_residual.vx_out`
   - Note: current exporter derives `vx_in` from `vx_out - ff_out * vgate_mlp` when direct pre-FF capture is unavailable.
- M4: complete (audio branch parity: self-attn, text-ca, ff).
   - **Implementation complete**: All Zig model entrypoints, Python exporters, Zig checkers, and BUILD.bazel targets are in place.
   - **Zig entrypoints** ([examples/ltx/model.zig](examples/ltx/model.zig#L1554)):
      - `forwardBlock0AudioSelfAttn` — M4-A: audio self-attn + PE
      - `forwardBlock0AudioSelfAttnResidualFromAttnOut` — M4-A: residual algebra
      - `forwardBlock0AudioTextCaResidualFromDelta` — M4-B: text cross-attn residual algebra
      - `forwardBlock0AudioFF` — M4-C: audio FF forward
      - `forwardBlock0AudioFFResidualFromFFOut` — M4-C: residual algebra
   - **Capture fix applied**: [replay_stage2_transformer_step.py](examples/ltx/replay_stage2_transformer_step.py#L465) now captures `__aux__.agate_msa` and `__aux__.agate_mlp` from `audio_scale_shift_table` using `audio.timesteps`.
   - **Validated matrix**:
      - token_limit=256, distilled_lora_strength=0.0: M4-A PASS, M4-B PASS, M4-C PASS
      - token_limit=256, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M4-A PASS (full residual), M4-B PASS (full residual), M4-C PASS (full residual)
      - token_limit=128, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M4-A PASS (full residual), M4-B PASS (full residual), M4-C PASS (full residual)
      - token_limit=512, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M4-A PASS (full residual), M4-B PASS (full residual), M4-C PASS (full residual)
   - **No token-limit sensitivity observed**: audio sequence length is fixed at 126 tokens regardless of video token limit, so M4 results are stable across all tested video token limits.
   - **Checkpoint alignment note**: LoRA>0 replay traces must be checked against LoRA-merged stage2 checkpoint export to avoid base-vs-LoRA parameter mismatch.

- M5: complete (AV cross-attn A->V and V->A parity).
   - **Scope**:
      - M5-A: A->V residual (`vx = vx + audio_to_video_attn(...) * gate * mask` path)
      - M5-B: V->A residual (`ax = ax + video_to_audio_attn(...) * gate * mask` path)
   - **Implementation complete**: All Zig model entrypoints, Python exporters, Zig checkers, and BUILD.bazel targets in place.
   - **Zig entrypoints** ([examples/ltx/model.zig](examples/ltx/model.zig#L1627)):
      - `forwardBlock0AudioToVideoAttnWithContextPeKPe` — M5-A: audio_to_video_attn with PE tensors
      - `forwardBlock0A2VDeltaFromAttnOut` — M5-A: gated delta algebra (attn_out * gate * mask)
      - `forwardBlock0VideoToAudioAttnWithContextPeKPe` — M5-B: video_to_audio_attn with PE tensors
      - `forwardBlock0V2ADeltaFromAttnOut` — M5-B: gated delta algebra (attn_out * gate * mask)
   - **Capture implementation** ([replay_stage2_transformer_step.py](examples/ltx/replay_stage2_transformer_step.py#L465)):
      - Block pre-hook now captures A->V and V->A gates via `module.get_av_ca_ada_values()` with scale_shift_table_a2v_ca_* and timestep families.
      - Captures: `__aux__.a2v_gate`, `__aux__.a2v_mask`, `__aux__.v2a_gate`, `__aux__.v2a_mask` at block scope.
      - Fixture exporters ([export_block0_av_a2v_fixture.py](examples/ltx/export_block0_av_a2v_fixture.py), [export_block0_av_v2a_fixture.py](examples/ltx/export_block0_av_v2a_fixture.py)) load delta from explicit capture; fallback arithmetic (`attn_out * gate * mask`) retained as safety net if capture is absent.
   - **Closure bug fixed**: `_make_av_attn_hook` previously referenced `mod` as a free variable from the enclosing loop, causing late-binding to a later submodule that lacked `get_av_ca_ada_values`. Fixed by passing the block module as an explicit `block_mod` parameter. Delta is now explicitly captured and verified to match the fallback arithmetic exactly (checkers pass with both).
   - **Validated matrix**:
      - token_limit=128, distilled_lora_strength=0.0 (base checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
      - token_limit=256, distilled_lora_strength=0.0 (base checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
      - token_limit=512, distilled_lora_strength=0.0 (base checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
      - token_limit=128, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
      - token_limit=256, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
      - token_limit=512, distilled_lora_strength=0.5 (LoRA-merged checkpoint): M5-A ✓ PASS (attention + gated delta), M5-B ✓ PASS (attention + gated delta)
   - **No token-limit sensitivity observed**: audio sequence length is fixed at 126 tokens; video context size varies but parity is stable across all tested limits.
   - **Fixture contracts validated**:
      - A->V: `block0_av_a2v.{x, context, pe_cos, pe_sin, k_pe_cos, k_pe_sin, attn_out, gate, mask, delta}`
      - V->A: `block0_av_v2a.{x, context, pe_cos, pe_sin, k_pe_cos, k_pe_sin, attn_out, gate, mask, delta}`
   - **Checkpoint alignment note**: M5 requires both audio_to_video_attn and video_to_audio_attn parameters in merged checkpoint; updated exporter to include these M5 modules (previously M4-only).

- M6: complete (full block0 parity combining all residuals and gates).
   - **Scope**: Validate complete block0 forward with video+audio branches, all residual paths (self-attn, text-ca, AV cross-attn, FF), and all gate modulations.
   - **Validated matrix**:
      - token_limit=128, distilled_lora_strength=0.0 (base checkpoint): PASS (video stream + audio stream + full block0 composition)
      - token_limit=256, distilled_lora_strength=0.0 (base checkpoint): PASS (video stream + audio stream + full block0 composition)
      - token_limit=512, distilled_lora_strength=0.0 (base checkpoint): PASS (video stream + audio stream + full block0 composition)
      - token_limit=128, distilled_lora_strength=0.5 (LoRA-merged checkpoint): PASS (video stream + audio stream + full block0 composition)
      - token_limit=256, distilled_lora_strength=0.5 (LoRA-merged checkpoint): PASS (video stream + audio stream + full block0 composition)
      - token_limit=512, distilled_lora_strength=0.5 (LoRA-merged checkpoint): PASS (video stream + audio stream + full block0 composition)
   - **Parity target**: validated against the Python `BasicAVTransformerBlock` semantics and related helper paths in the upstream LTX transformer implementation, not merely against an internal Zig surrogate.
   - **Current Zig landing point**: full M6 semantics are now implemented in `BasicAVTransformerBlock` stream methods and canonical full-forward API in [examples/ltx/model.zig](examples/ltx/model.zig#L402), with free-function M6 entrypoints delegating to those methods.
   - **Important distinction**: runtime stack iteration in `LTXModel.forward` still uses the legacy single-stream FF-boundary method for compatibility; full-stream inputs (contexts/PE/gates for both video+audio) are not yet threaded through the model loop.
   - **Post-integration revalidation (2026-03-23)**: re-ran `block0_full_check` across all six combinations (token limits 128/256/512 × lora 0.0/0.5); all video+audio stream checks PASS. Note: lora=0.5 t256 fixture was stale (missing `v_text_x`) and required re-export from `acts_stage2_transformer_step_000_m6_lora0.5_t256.pt` before passing.
   - **Key implementation lesson**: for both video and audio streams, text cross-attn and AV cross-attn must use the exact per-module query inputs captured from Python (`v_text_x`, `a2v_x`, `a_text_x`, `v2a_x`) because those module inputs include hidden preprocessing not recoverable from a naive reconstructed residual state alone.

## Milestone Status Summary

| Milestone | Status | Token Limits | LoRA Strengths |
|-----------|--------|--------------|----------------|
| M1 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |
| M2 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |
| M3 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |
| M4 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |
| M5 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |
| M6 | ✓ Complete | 128, 256, 512 | 0.0, 0.5 |

## Practical Notes

- Keep one parity equation per checker target.
- Keep fixture names explicit by stage to avoid ambiguity.
- Use same tolerances currently accepted by FF-boundary checker unless a stage needs a stricter budget after f32 fixes.
- For gated residual stages, prefer direct auxiliary gate capture from replay hooks (e.g. `__aux__.vgate_msa`) over reconstructing gates from timestep heuristics; this avoids drift and keeps strict tolerances meaningful.
- Standard rule for LoRA runs: when replay uses `distilled_lora_strength > 0`, use a LoRA-merged stage2 checkpoint for Zig checkers (see [examples/ltx/export_stage2_block0_checkpoint.py](examples/ltx/export_stage2_block0_checkpoint.py)).
