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

The diagnostic also supports multi-token `--seqlen` values. It initializes IDs
`1000..1000+seqlen-1` and checks all updated cache rows, preserving the original
tolerances. Optional `--layerwise-dump-dir=<path>` writes BF16 stage/hidden
buffers and the already-returned KV caches for cross-run comparisons; default
runs do not write these files. Reading those cache outputs does not add graph
roots or change the compiled forward function.
Stage outputs can change native fusion and rounding. Compare their results
with the separately executed layer before attributing a discrepancy.

The XLA repository's `batch-prefix-audit` experiment records widths 1, 2, and 4
at offset zero. CPU width-two/width-four prefix hidden states are bit-identical
through all 32 layers. Device differences start in layer zero, row zero, in
the standalone layer but disappear when intermediate stage outputs are added.
All local CPU-input checks pass tolerance; propagated checks still fail.
These observations do not waive the whole-forward or position-127 gates.

The follow-up `first-position-attention` audit narrows the first standalone
width-two/width-four cache difference to V element 758 at position zero,
one BF16 step (0.00000762939453125). Instrumented V agrees across widths.
With the instrumented normalization input and original weight row, the exact
rational dot is just below the midpoint between those adjacent BF16 values.
CPU width one also chooses the upper value. This supports accumulation-order
sensitivity, but the standalone normalization input is not exposed, and this
does not prove the cause of every propagated or whole-forward discrepancy.
All 75 previously captured stage/hidden buffers remain byte-identical
after adding the cache dumps; all three first-layer runs pass local tolerances.

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

## Model-computed history diagnostic

`--forward-only --compare-cpu --prefill-history` replaces the synthetic cache
prefix with a CPU-computed prefix, then supplies identical initial K/V bytes
to both decode paths. It requires a one-token query at a nonzero offset.
The prefix uses deterministic valid token IDs `1000..1000+offset-1`; these are
not a natural-language prompt. The query remains token 1000 so that history
is the only input changed from the synthetic test. Unused cache rows retain
their original bit pattern and are checked before and after decoding.

With the same Furiosa environment and `--xla_allow_excess_precision=false`:

```sh
bazel-bin/examples/llm/llama_tests --platform=furiosa \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct --compare-cpu \
  --forward-only --prefill-history --seqlen=1 --cache-seqlen=128 --token-offset=127
```

The 32-layer run agrees on argmax token 198 and passes the value-cache and
untouched-cache checks. Keys still fail the existing tolerance, starting at
layer 9, with maximum error 0.09375. This rules out synthetic history as the
sole cause; it does not establish the cause of every numerical discrepancy.
CPU-created history deliberately isolates one decode query and does not test
Furiosa prefill or full session feedback.

Regression runs without the flag preserve the previous results: position 127
fails with argmax 323 and maximum key/value errors 0.109375/0.04296875;
position 0 passes exact argmax 76944, KV tolerance and untouched-cache checks.
The initial helper run aborted before decoding because it passed a replicated
sharding for a cache tensor tagged with the model partition. The helper now
uses the registered model sharding. The final build and Zig format check pass.
Complete run records, including that setup failure, are retained in the XLA
checkout's `prefill-history-audit` experiment. No gate or tolerance was waived.
