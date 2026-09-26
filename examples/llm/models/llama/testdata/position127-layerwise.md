# Position-127 precision audit

`llama_tests --compare-cpu --layerwise` now diagnoses each transformer layer
using both the exact CPU hidden input and the accumulated device hidden input.
It starts at token 1000 and uses the same deterministic nonzero cache history
as the whole-forward comparison. Untouched cache storage remains bit-checked.
It reports errors only over the newly computed cache row, so unchanged history
does not dilute the statistics. Original hidden/KV tolerances are preserved,
and any failed layer keeps the process failing after all layers are reported.

`--layerwise-stages` additionally exposes attention, residual, normalization,
MLP projections, activation and product outputs. Exact-bit reports for these
stages are observations, not a replacement correctness gate. RMS F32
intermediates and independent FP64 dots for mismatching projection outputs help
separate reduction, conversion and accumulation differences. Dot checks require
bit-identical post-norm inputs and skip otherwise. Extra output roots can affect
fusion, so these diagnostics do not replace the original whole-forward test.
Production remains one public executable per complete forward pass.

Build with the Furiosa platform explicitly enabled:

```sh
bazel build //examples/llm:llama_tests --@zml//platforms:furiosa=true --jobs=8
```

With the usual Furiosa runtime/compiler/plugin environment configured:

```sh
bazel-bin/examples/llm/llama_tests --platform=furiosa \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct --compare-cpu \
  --layerwise --layers=32 --seqlen=1 --cache-seqlen=128 --token-offset=127
```

Use `--layers=1 --layerwise-stages` for the first-layer stage audit. For a
separate strict-reference experiment, add `--xla_allow_excess_precision=false`
to `XLA_FLAGS`. This flag is not made a new default.

Observed with units mode, budget 1024, cache-view and update fusion enabled:

- All 32 local-input layer comparisons pass existing tolerances. Accumulated
  keys first fail at layer 5; this reproduces the whole-forward failure pattern
  without requiring fusion across layer boundaries.
- With the default CPU policy, layer 0's residual is bit-identical but its
  post-attention RMSNorm differs in 796/4096 values. CPU optimized HLO preserves
  the residual in F32 internally, although its exposed residual output is BF16.
- Disabling excess precision preserves that BF16 rounding boundary. Post-norm
  becomes bit-identical and layer-0 hidden differences fall from 1910 to 8.
- The remaining strict-reference gate/up mismatches are 4/14336 and 1/14336.
  Independent FP64 sums are closer to CPU for three and to Furiosa for two;
  these do not support assuming every CPU/device disagreement is a Furiosa bug.
- The **whole-forward strict comparison still fails KV tolerances**, beginning
  at key layer 7. Exact argmax now agrees at token 323. Neither this partial
  improvement nor stage diagnostic exit 0 is counted as a whole-forward pass.

The unchanged component/single-layer comparison passed. The 32-layer
accumulation diagnostic and strict whole-forward KV comparison correctly
returned 1. Final first-layer runs with default/strict precision returned 0
under the original local tolerances; exact-bit differences remain recorded.

Initial setup failures are retained in the XLA experiment evidence: the first
build used mutable slice bindings and encountered an existing unused
`compareFloats` wrapper signature error. The diagnostic now computes its own
explicit statistics; no shared testing helper changed. A subsequent build
omitted the Furiosa feature flag and returned `Unavailable`; enabling the flag
fixed that setup error. Final builds and `zig fmt --check` pass.

Complete commands, binary/plugin/runtime hashes, raw logs, optimized CPU HLO
and analysis live in the companion XLA checkout under
`xla/pjrt/furiosa/experiments/2026-09-26-compilation-units/position127-audit/`.
Local compiler dumps remain under
`~/.local/state/xla-rngd/internals/compilation-units/position127-audit/`.
No production precision, model weights, tolerances or throughput claim changed.
