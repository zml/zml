# Whole-model Llama forward

Llama now compiles embedding, all transformer layers, final normalization,
output projection and sampling together. Prefill and decode each have one XLA
executable. There is no host-side layer grouping. Greedy sampling is the CLI
default (`--topk=1`). The final normalization runs once in `LmHead`.

The current Furiosa opt-runtime launch protocol permits 119 address arguments,
including runtime statics. Packing equally shaped checkpoint weights into 32
buffers keeps the full BF16 Llama 3.1 8B invocation below that limit. Upload uses
`BufferedMemoryWriter`; constant slices recover individual weights inside the
compiled graph. Packing preserves dtype and original partition annotations and
does not quantize weights. Component tests retain the unpacked loader.

## Validation and current limitation

The packed whole-forward graph passes the unpacked whole-forward CPU reference
at query length 1, cache length 128 and position 127: exact argmax, existing KV
tolerances (absolute 0.03, relative 0.02, all elements), and exact preservation
of untouched cache entries. CPU reference and actual weights are loaded and
released sequentially to bound host memory.

The corresponding RNGD executable compiles and runs, but **does not pass the
CPU comparison**: it returns token 323 where the CPU returns 311. The test exits
with an error; no tolerance was relaxed. Full-prefill compilation and normal
prompt generation with the new architecture remain unverified. These results
do not establish correct full-model inference on RNGD.

## Provisional decode benchmark, 2026-09-25

Original BF16 `/var/models/meta-llama/Llama-3.1-8B-Instruct`, one RNGD card,
vanilla attention, argmax, cache length 128, one complete forward executable.
Staged fallback, execution tracing and profiling were disabled. No concurrent
build or compiler job ran during timing.

Three trials of 100 autoregressive decode steps measured **52.93, 52.73 and
52.81 tok/s**, median **52.81 tok/s**. Each trial starts from BOS with zeroed KV,
runs four warmup steps, then advances positions and feeds each predicted token
back to the executable. Timing includes synchronous token readback. EOS does
not terminate these fixed-length trials. Compilation, uploads, prefill,
tokenization and terminal rendering are excluded. This is a decode benchmark,
not a validated chat-generation rate; the correctness failure above remains.

After sourcing the SDK environment and setting `XLA_FURIOSA_PJRT_LIBRARY` and
`XLA_FURIOSA_COMPILER_CACHE` as usual:

```sh
bazel-bin/examples/llm/llama_tests \
  --platform=furiosa \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --compare-cpu --forward-only --seqlen=1 --cache-seqlen=128 \
  --token-offset=127 --benchmark-iterations=100
```

The command logs throughput before performing the correctness comparison and
currently exits nonzero. Evidence is retained locally in
`/home/steeve/.local/state/xla-rngd/internals/full-forward/decode-benchmark.log`.
The successful CPU-only check is `packing-cpu-sequential.log` in that directory.

Build validation passed for `//examples/llm`, `//examples/llm:llama_tests` and
`//examples/llm:llama_logits` with `--@zml//platforms:furiosa=true --config=debug`.
The benchmark changes were rebuilt with the `llama_tests` target. The captured
decode TCL SHA256 is
`efa80d98893c6605d47af8b8291bd1604e9ae79c0b6834575247c91b55ad8848`.
