# Whole-model Llama forward

Generation compiles embedding, all transformer layers, final normalization,
output projection and sampling together. Prefill and decode each have one XLA
executable. Weights remain separate tensors loaded through the ordinary loader;
there is no host-side layer grouping or packed-weight graph.
Prefill selects the final prompt hidden row before final normalization,
vocabulary projection and sampling. Transformer/KV computation covers the full
prefill shape. Greedy sampling uses `--topk=1`.

On Furiosa, cached RoPE is bounded by KV-cache length. Layer residual outputs
retain their own buffer, and standalone layer callers release the previous
hidden buffer. Preserve this ownership when changing the forward graph.

## Numerical validation

Use the [Furiosa setup](../../../../platforms/furiosa/README.md) and SDK environment.
CPU must be enabled for the reference comparison:

```sh
bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --config=debug --@zml//platforms:furiosa=true --@zml//platforms:cpu=true \
  //examples/llm:llama_tests -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --platform=furiosa --furiosa-pe-count=4 --compare-cpu --forward-only \
  --layers=32 --seqlen=1 --cache-seqlen=128
```

Repeat with eight PEs in a new process. Exercise nonzero positions, prefilled
history and autoregressive decode using the harness options. Check exact argmax,
existing numeric tolerances, finite cache values and exact unchanged KV entries.
Do not relax tolerances to mask backend errors. A benchmark's throughput is not
a correctness result; report compilation, transfer and execution timing separately.

## Historical conclusions

Earlier experiments exposed cached-position, residual ownership, session feedback
and prefill-memory problems. The reusable diagnostics and their fixes remain in
source. Raw machine-specific captures have been removed; Git history retains them.
Old packed-weight, bridge/11 and vISA measurements do not validate the current
implementation. Current migration evidence belongs in
[VALIDATION.md](../../../../platforms/furiosa/VALIDATION.md).
