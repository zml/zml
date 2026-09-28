# Furiosa3: native TCL MLIR backend

Enable with `--@zml//platforms:furiosa3=true`, and point
`XLA_FURIOSA3_PJRT_LIBRARY` at the XLA checkout's
`bazel-bin/xla/pjrt/furiosa3/libpjrt_c_api_furiosa3_plugin.so`.
This platform has a distinct PJRT identity. It shares device allocation,
layouts, asynchronous execution and sharding behavior with the other Furiosa
platforms, and selects vanilla attention. PE count defaults to four; use
`--furiosa-pe-count=8` for the whole card.

Llama retains one XLA executable per whole forward and separate weight
arguments. Cached RoPE and declined-donation cleanup apply to Furiosa3 too.
The plugin lowers post-fusion HLO through verified TCL MLIR operations and
outlines private native units within that public executable.

The companion XLA checkout contains `xla/pjrt/furiosa3/run_llama.sh`, which
builds both components and selects the SDK and plugin paths for this host:

```sh
/home/steeve/xla-private/xla/pjrt/furiosa3/run_llama.sh \
  --prompt='Count from 1 to 100, separated by commas.'
```

## Current handoff (2026-09-28)

Use ZML branch `steeve/wip/furiosa` together with `steeve/furiosa` in
`zml/xla-private`. The tested ZML implementation is `3866ec6`; subsequent
handoff changes were documentation only until the RNG statistical-test update below. Both checkouts are required.

The companion [engineering handoff](https://github.com/zml/xla-private/blob/steeve/furiosa/xla/pjrt/furiosa3/HANDOFF.md)
contains pinned setup versions, architecture, source ownership, current flags,
exact test commands, benchmark distinctions, known gaps and next steps.
The [documentation index](https://github.com/zml/xla-private/blob/steeve/furiosa/xla/pjrt/furiosa3/DOCUMENTATION_INDEX.md)
links the complete experiment history. These links require private-repository access.

The current launcher uses BF16 weights, vanilla SDPA, top-k 16, seqlen 256 and
eight PEs per card. Two-card tensor parallelism uses XLA/Shardy and device-only
collectives. The linked default reaches a matched 63.7–63.8 tok/s; experimental
producer/collective fusion regresses to 59.7 and is off by default. The older
argmax measurements below are a different configuration.

The XLA backend now uses ThreeFry for ZML's two-word `RNG_DEFAULT` state;
explicit algorithm requests are unchanged. This changes the default sequence
for the same seed. The [latest dispatch report](https://github.com/zml/xla-private/blob/steeve/furiosa/xla/pjrt/furiosa3/experiments/2026-09-28-coalesced-dispatch/README.md)
records an **83.5 tok/s median** candidate with opt-in task coalescing and shared
code, against a matched old Philox control of 63.6. That rate requires rebuilding
the runtime and setting the report's experimental flags. Dispatch experiments
remain off by default, and the installed runtime has not been replaced. The
100 tok/s target is not met; older measurements above use the prior default.

Set `XLA_FURIOSA_VISIBLE_DEVICES=0` or `=1` for one card, or `=0,1` for both.
The runtime requires ascending physical IDs. Set `ZML_CHECKOUT`,
`FURIOSA_SDK_DIR` and `LLAMA_MODEL` when using the companion launcher elsewhere.

The latest `bazel test //zml/...` rerun passes all three test targets on card 0
with CPU disabled (177 core passes, 18 platform skips, both tokenizer targets
passing). It uses the ThreeFry default plus experimental runtime coalescing,
code sharing and paired argument copies. The two RNG statistical tests now
use sample-size-derived tolerances: their original tight bounds failed for
ThreeFry even though its bits and states exactly match XLA's CPU evaluator.
The companion report retains both the failures and passing rerun. The older
full `//...` validation covered 24 Bazel test targets before the latest TP work.
On shared machines, set `RULES_ZIG_CACHE_PREFIX_LINUX` to a private writable
path, for example `/tmp/zig-cache-$USER`, to avoid the shared cache's ownership. Use `--nocache_test_results` after changing the external plugin library.
Qwen 3.5/Gated DeltaNet and general scaled-dot/MXFP4 are not validated by this path.

## Historical initial Llama validation

Validation: `bazel build //... --@zml//platforms:furiosa3=true --jobs=16`
builds all 169 targets. The exact launcher above generates the requested
counting sequence at 63.7–64.4 tok/s on all eight PEs, with original BF16
weights, vanilla attention, argmax and cache length 128. The context limit
includes the prompt; the example stops at 34. With compiled units cached,
prefill and decode compilation take about 3.4s and 3.7s respectively.

Embedding, batched RMSNorm, Q/K/V/O projections, MLP and an eight-token
transformer pass CPU comparisons including KV. A whole 32-layer decode check
using BOS + "Hello" and four continuation predictions matches CPU argmax at
all five tested positions, with finite active caches and unchanged padding.
These do not establish long-context CPU equivalence. Hardware regression
suites pass all 16 tests on both four and eight PEs. Further evidence and
compiler failure reproductions are in the companion XLA checkout's
`xla/pjrt/furiosa3/experiments/2026-09-27-llama` directory.
