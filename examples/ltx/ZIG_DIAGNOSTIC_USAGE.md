## Zig Diagnostic Integration Usage

After generating the Python reference file (`attn1_reference.safetensors`), use it with the Zig checker to validate stage-by-stage.

### Command Structure

```bash
bazel run --@zml//platforms:cuda=true //examples/ltx:attention_forward_check -- \
    <stage2_checkpoint.safetensors> \
    <fixture_with_diagnostics.safetensors> \
    <mode> \
    [token_limit] \
    [diagnostic_reference.safetensors]
```

### Example: With Diagnostic Reference

```bash
cd /root/repos/zml/zml

bazel run --@zml//platforms:cuda=true //examples/ltx:attention_forward_check -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors \
    attn1 \
    256 \
    /root/models/ltx-2.3/attn1_reference.safetensors
```

### Example: Without Diagnostic Reference (baseline)

```bash
bazel run --@zml//platforms:cuda=true //examples/ltx:attention_forward_check -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors \
    attn1 \
    256
```

### Example: Full-Context With Diagnostic Reference

The checker also accepts a diagnostic reference without a `token_limit`.
This argument shape works for every mode accepted by `attention_forward_check`.
Today, the detailed stage-by-stage diagnostic breakdown is implemented for `attn1`.
For the other modes, the checker still performs full end-to-end parity using the same CLI.

```bash
bazel run --@zml//platforms:cuda=true //examples/ltx:attention_forward_check -- \
   /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
   /root/repos/LTX-2/trace_run/attn1_fixture.safetensors \
   attn1 \
   /root/repos/LTX-2/trace_run/attn1_reference.safetensors
```

## AttentionKind Suite

To produce repeatable proof that every `AttentionKind` wiring can be exported and checked,
use `attention_kind_suite.py`. It exports one fixture per mode from a single replay trace and
runs `attention_forward_check` across all six modes:

- `attn1`
- `attn2`
- `audio_attn1`
- `audio_attn2`
- `audio_to_video_attn`
- `video_to_audio_attn`

### Full-Context Proof Run

```bash
python examples/ltx/attention_kind_suite.py \
   /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
   /root/repos/LTX-2/trace_run/acts_stage2_transformer_step.pt \
   /root/repos/LTX-2/trace_run/attention_kind_suite \
   --repo-root /root/repos/zml \
   --bazel bazel \
   --cuda
```

This writes:

- `<output_dir>/<mode>_fixture.safetensors`
- `<output_dir>/<mode>_export.log`
- `<output_dir>/<mode>_check.log`

and prints a final PASS/FAIL summary for every mode.

Scope of proof:

- All six modes are exercised end-to-end through the same `AttentionKind` mapping, parameter loading path, and forward entrypoints.
- `attn1` has the deepest diagnostic coverage today, including intermediate-stage comparisons when matching references are available.
- `attn2`, `audio_attn1`, `audio_attn2`, `audio_to_video_attn`, and `video_to_audio_attn` are currently validated end-to-end rather than stage-by-stage.

### Token-Limited Proof Run

```bash
python examples/ltx/attention_kind_suite.py \
   /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
   /root/repos/LTX-2/trace_run/acts_stage2_transformer_step.pt \
   /root/repos/LTX-2/trace_run/attention_kind_suite_t256 \
   --repo-root /root/repos/zml \
   --bazel bazel \
   --cuda \
   --token-limit 256
```

For token-limited runs, only token-local diagnostics are strict. Full attention outputs are not
directly comparable to references captured from full-context attention unless the references were
generated from a genuinely truncated forward pass.

### Output

The command will:
1. Load checkpoint and fixture tensors
2. If diagnostic reference is provided:
   - Load q_head_split, k_head_split, v_head_split, q_rotated, k_rotated from reference
   - Log availability of each reference tensor
3. Compile the attention graph
4. Execute forward pass
5. Compare output against expected (always done)
6. Log diagnostic stage information if reference was loaded

### Current Implementation Status

✅ **Done:**
- Diagnostic reference loading infrastructure
- Support for passing diagnostic reference file path
- Head-split tensor slicing for token limits
- Logging of available diagnostic tensors
- Output comparison (existing functionality)

⚠️ **In Progress (Next Phase):**
- Extract intermediate values (q, k, v, q_rotated, k_rotated) from computation graph
- Element-wise comparison with reference at each stage
- Per-stage parity metrics and pass/fail reporting

### Files Modified

**check_utils.zig:**
- Added `DiagnosticStage` enum
- Added `StageMetrics` struct
- Added `compareBuffers()` function for element-wise comparison
- Added `applyHeadSplit()` to compute head-split transformation
- Added `DiagnosticReference` struct
- Added `loadDiagnosticReference()` to load reference tensors
- Added `reportStageMetrics()` convenience function

**attention_forward_check.zig:**
- Updated usage to include optional diagnostic reference path
- Added diagnostic registry and store loading
- Added loading of diagnostic reference tensors
- Added token-limit slicing for diagnostic tensors
- Added logging section for diagnostic stage comparison

### Expected Workflow

```
1. Run replay with --capture-kwargs --capture-inputs
   → Captures q, k, v, pe, mask

2. Export fixture with diagnostics
   → Creates attn1_fixture_diagnostic.safetensors with q/k/v tensors

3. Run Python diagnostic validation
   → Generates attn1_reference.safetensors with post-head-split and post-rope tensors

4. Run Zig checker WITH reference
   → Loads reference
   → Executes attention
   → (Next: Compare intermediate values at each stage)
   → Reports stage-by-stage parity

5. Debug divergence
   → If stage N diverges, problem is in operation N
```

### Notes

- Diagnostic reference file is optional; checker works without it
- Token limits are applied to all tensors (fixture, reference, outputs)
- Reference tensors are loaded from store but not yet extracted from computation
- Full comparison requires extending the attention computation to output intermediates
