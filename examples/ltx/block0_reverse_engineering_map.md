# Block-0 Reverse Engineering Map (Python -> Zig)

This map defines where each Block-0 operation comes from in LTX Python and where it should land in current Zig code.

## Scope

Current checker path validates a simplified FF-boundary surrogate, not full BasicAVTransformerBlock equivalence.

- Current Zig block forward: [examples/ltx/model.zig](examples/ltx/model.zig#L344)
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
- Current simplified forward (attn result ignored, returns ff(x)): [examples/ltx/model.zig](examples/ltx/model.zig#L344)
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

## Verified Current Boundary

Current fixture/check path validates only the FF-boundary surrogate:

- Fixture tensors:
  - block0_ff_boundary.input0
  - block0_ff_boundary.output0
- Loader prefers new keys and can fallback to legacy keys in checker.

References:

- Fixture export: [examples/ltx/export_block0_fixture.py](examples/ltx/export_block0_fixture.py#L75)
- Checker key map: [examples/ltx/block0_forward_check.zig](examples/ltx/block0_forward_check.zig#L188)

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

## Practical Notes

- Keep one parity equation per checker target.
- Keep fixture names explicit by stage to avoid ambiguity.
- Use same tolerances currently accepted by FF-boundary checker unless a stage needs a stricter budget after f32 fixes.
- For gated residual stages, prefer direct auxiliary gate capture from replay hooks (e.g. `__aux__.vgate_msa`) over reconstructing gates from timestep heuristics; this avoids drift and keeps strict tolerances meaningful.
