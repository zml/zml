# Fixed-block full-forward timing

`llama_tests --compare-cpu --forward-only --benchmark-block-iterations=32`
times three trials of the complete existing forward executable, with four
warmups per trial. `--seqlen` selects the query width; `--cache-seqlen` and
`--token-offset` keep their existing meanings. This diagnostic is separate
from the original autoregressive `--benchmark-iterations` mode.

Every timed call uses fixed token IDs `1000..1000+seqlen-1` at the same position,
with a cache initialized to zero at the start of the trial. Token buffers are
prepared before timing because the executable donates them. Each call includes
all model layers, all query positions' head projections and argmax operations,
and synchronous readback of all output token IDs. Repeated outputs must match
exactly. Compilation, model upload, allocation and initialization are excluded.

The report uses **milliseconds per call** and **positions per second**. These
are not generated tokens per second: it does not construct proposals, test
acceptance, roll back a cache, or feed predictions into the next input block.
It is a feasibility diagnostic for amortizing weight transfers across query
positions. Generation behavior is unchanged.

After timing, the existing independent CPU comparison runs with its original
deterministic nonzero cache fixture. Exact token, KV tolerance, and untouched
cache checks are preserved. A failed comparison still fails the process even
if the timing loop completed successfully.

Build with the Furiosa platform enabled:

```sh
bazel build //examples/llm:llama_tests --@zml//platforms:furiosa=true --jobs=8
```

The XLA repository's
`xla/pjrt/furiosa/experiments/2026-09-26-compilation-units/block-forward-probe`
contains the host's exact commands, plugin/runtime hashes, logs, comparison
results, and timings for query widths 1, 2, and 4. It uses strict BF16 CPU
boundaries (`--xla_allow_excess_precision=false`) and retains failures.

Measured medians are 15.048, 15.071, and 17.137 ms per call for widths 1, 2,
and 4 respectively. Width one passes the CPU comparison. Width two fails key
and value tolerances; width four fails the token and key comparisons. At width
four, row 1 selects token 278 on Furiosa and 315 on CPU. CPU scores are 6.0 and
6.0625 respectively, with a unique maximum. This is not an exact-score tie.
Per-position reference-logit diagnostics preserve the failures and do not
establish that batched execution is a validated greedy verifier.
