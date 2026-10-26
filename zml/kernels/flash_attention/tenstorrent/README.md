# Tenstorrent attention

The backend emits `ttir.scaled_dot_product_attention_decode` through
the TT-MLIR custom-call builder. It supports causal BF16 decode with one query,
grouped query heads, optional batches, and head/cache dimensions divisible by 32.
Prefill and other shapes use the general StableHLO implementation.

On two cards, a manual computation partitions query and KV heads together,
keeping each GQA group local. Computed cache slices with open partitioning use
their mesh's model axis. Explicitly replicated heads and unsupported partitions
retain StableHLO attention. The XLA compiler preserves these local kernels while
partitioning the surrounding graph.

The XLA plugin configures HiFi4 arithmetic, FP32 accumulation, and accurate
exponentials. Its Metalium patch makes the softmax loop honor the exponential
precision setting; the upstream loop otherwise hardcodes approximate math.

`//zml:tenstorrent_test` checks every output element against an independent f64
causal-attention calculation using BF16 inputs. Cases cover Llama's 32-query/8-KV
heads, batched GQA, Qwen's 24-query/4-KV heads with head dimension 256, and cache
lengths 128 and 1024. Positions span tile and chunk boundaries, plus values
beyond the cache including the largest u32. Bounds clamp before signed conversion.
The two-card cases cover 27 position/shape combinations, including computed
inputs whose partitioning is open. An operation trace confirms native decode
attention and no all-gathers within these kernels.

Build the example with the Tenstorrent platform enabled:

```sh
bazel build //examples/llm --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false
```

Select attention explicitly with `--backend=tenstorrent`; it is also selected
automatically for one or two Tenstorrent cards. The attention flag alone does
not enable the platform in a CPU-only build.

Llama stores its Tenstorrent KV cache in `[layer, head, sequence, head_dim]`
order. Keeping the attention kernel's order across tokens removes full-cache
permutations and avoids padding the eight KV heads to a 32-row tile. The same
tagged update path handles prefill and decode and preserves buffer donation.
`//examples/llm:llama_cache_test` checks every cache element after 32-token
prefills and one-token updates at positions 0, 31, 32, and 63, including both
layers, untouched positions, and preservation of update inputs.

With this layout, XLA's tuned BF8 down-projection and output head, eight-core RMS
normalization, FP32 matrix accumulation, and cached DRAM-sharded BF16 FFN/QKV
weights with SiLU and QKV head splitting on L1 shards, the single-card Llama 3.1 8B
counting smoke test at sequence length 128, including the cache-update fixes
described below, measured:

| Weights | Decode time (73 tokens) | Throughput | Kernel cache hits |
| --- | ---: | ---: | ---: |
| BF16 | 2.892 s | 25.2 tokens/s | 518/518 |
| BF8 | 2.162 s | 33.8 tokens/s | 531/531 |

Both runs produced the correct count from 1 to 20. Before the layout change,
the corresponding BF16/BF8 runs measured 18.0/24.2 tokens/s; the layout alone
raised them to 20.2/28.3 tokens/s. BF8 is enabled by
`XLA_TT_EXPERIMENTAL_BF8_WEIGHTS=1`; activations and KV caches remain BF16.
These measurements supersede the earlier 22.2/32.9 tokens/s results with BF16
accumulation, which failed a long-dot numerical regression. Trace replay is
disabled in these measurements. These are short application
smoke tests, not model-quality evaluations.

With `--warmup-runs=1` and `XLA_TT_EXPERIMENTAL_TRACE=1`, a complete discarded
generation warms the executables and weights before a fresh session is timed.
Using the same counting prompt, sequence length 128, and top-k 1:

| Weights | XLA layout optimization | Decode time (73 tokens) | Throughput | Kernel-cache hits |
| --- | --- | ---: | ---: | ---: |
| BF16 | Level 1 (default) | 2.713 s | 26.9 tokens/s | 518/518 |
| BF16 | Level 2 (experimental) | 2.763 s | 26.4 tokens/s | 918/918 |
| BF8 | Level 1 (default) | 2.049 s | 35.6 tokens/s | 531/531 |

All responses are correct. These warm-generation rates include token transfer,
text decoding, and output, and use a different warm-up/trace configuration from
the preceding table. The XLA plugin's level-2 option is
`XLA_TT_EXPERIMENTAL_L1_LAYOUT=1`. Its bounded layout search, FP32 scheduling,
and measured decode schedules now compile and run this model, but level 1
remains faster and compiles much more quickly, so it remains the default.

XLA's subsequent cached vocabulary split improves the paired level-1 BF16
comparison to **2.678 s / 27.3 tokens/s** (520/520 kernel-cache hits), versus
**2.712 s / 26.9 tokens/s** with `XLA_TT_EXPERIMENTAL_CHUNKED_HEAD=0` in the
same build. Both generate the correct 73-token response with the warm-up/trace
settings above. BF8 retains its original schedule and repeats at **2.049 s /
35.6 tokens/s** (531/531 hits). The split pads and caches eight BF16 weight
chunks, uses the measured DRAM-sharded matrix schedule with FP32 partial sums,
and trims the final vocabulary output. It tracks weight versions and preserves
prefill's original preprocessing cache.

The subsequent native attention schedule uses 128-token chunks and one core
per KV head for this length-128 Llama case, preserving HiFi4 and FP32
accumulation. With the vocabulary split enabled, a paired run measured
**2.684 s / 27.2 tokens/s** with `XLA_TT_EXPERIMENTAL_ATTENTION_SCHEDULE=0`
and **2.673 s / 27.3 tokens/s** with selection enabled. BF8 measured
**2.031 s / 35.9 tokens/s**. All responses were correct, and timings follow
one discarded warm-up; kernel changes caused startup/warm-up JIT misses.
The attention oracle suite passes with tracing off and on, including the
1024-token-cache schedule. The plugin also fixes an SDK softmax correction
that used five destination tiles when FP32 half synchronization permits four;
a native regression fails before the fix and passes afterward.

The confirmed minimum is 29.4 tokens/s with **BF16** weights, with approximately
33 tokens/s as the performance objective; BF8 is expected to exceed that.
The BF16 target remains unmet. XLA now selects DRAM-sharded
BF16 FFN/QKV weights by default for this card and keeps the FFN activation
chain in L1. Sharding consumes prefill's cached preprocessing in a separate cache
entry, preserving reuse while tracking weight-buffer versions. Set
`XLA_TT_EXPERIMENTAL_DRAM_WEIGHTS=0` to compare with the interleaved schedule.

Separate execution-stage profiles measured median steady calls of 40.737 ms
versus 43.641 ms before sharding. The original sharding experiment repeated
prefill preprocessing, pushing first decode to 701.376 ms and regressing the
short run to 20.1 tokens/s. The layered cache reduces first decode to 224.658 ms,
so the complete run improves from 22.0 to 22.9 tokens/s. Running SiLU separately
on the 16 L1 output shards instead of fusing it onto the eight DRAM reader
cores improves BF16 further to 23.8 tokens/s; QKV sharding raises it to 24.0.
With both changes, a separate execution profile measured 233.100 ms for first
decode and 38.813 ms median steady calls. Tuning BF8's output head reduced
its isolated kernel time from 1.763 to 1.546 ms; the full counting run measured
32.7 tokens/s versus the prior 32.6. That application-level difference is small.

Keeping BF16 QKV projections in eight L1 width shards through head splitting
raises the warm run to 24.5 tokens/s. The BF8 path stages the small BF16
projection into L1 before splitting and measures 33.4 tokens/s. The native
head-splitting benchmark checks every Q/K/V value exactly and measures 23.2
microseconds from those shards, versus 64.0 microseconds from DRAM. XLA's
integration regression checks the projection-plus-head-split graph, input and
weight preservation, changing inputs, and the compiled L1 layout.

Spreading the BF16 FFN gate/up outputs, SiLU, and multiply over 64 L1 shards
raises the warm run from 24.5 to 25.2 tokens/s. The projections keep their
16-shard inputs, eight DRAM reader cores, and FP32 arithmetic; the down projection
reshards back to 16 cores. The unchanged BF8 schedule repeated at 33.5 tokens/s,
consistent with 33.4 previously. XLA's seven default single-card suites and
focused BF16/BF8 trace/cache/donation regressions pass with this layout. A separate
BF16 trace execution profile measured 228.343 ms for first decode and 36.801 ms
median steady calls, versus 38.737 ms before this change.

XLA also guards TT-MLIR's in-place cache lowering: only private partial slices
may be mutated, while shared cache values and input buffers keep functional
scatter semantics. A regression returns both the original and updated cache,
checking preservation with donation enabled and disabled and with trace replay.
All seven default single-card suites pass. The BF16 counting run after this fix
measured 2.912 seconds / 25.1 tokens/s with all 488 kernels cached, close to the
25.2 result above. BF8 repeated correctly at 2.168 seconds / 33.7 tokens/s
with 501/501 kernel-cache hits.

XLA now preserves the computed, clamped cache position and combines eligible
K/V updates into the native fused cache kernel. It retargets K's existing layout
conversion to a separate core, avoiding an extra reshard. Llama emits 31 fused
updates; the last pair retains separate writes because a K-cache observer lies
between them. The latest unprofiled warm runs are in the table above, with all
kernel artifacts already materialized and correct counts. Separate execution
profiles measured BF16 steady calls of 36.748 ms with fusion and 36.877 ms without
it. Trace replay measured 36.345 ms steady and 25.5 tokens/s overall with profiling
enabled, a small improvement that does not close the BF16 performance gap.

At initial integration, Llama 3.1 8B's counting prompt with sequence length 128 and greedy
decoding completed in 4.489 seconds (16.3 tokens/s), versus 4.778 seconds
(15.3 tokens/s) for StableHLO attention on the same build. Both runs produced the
same answer and hit all 495 cached device kernels. The first native-attention run
had six cache misses and took 5.423 seconds; compare warm runs for throughput.

The two-card Llama run with head partitions and KV-cache donation completed in
3.503 seconds (20.8 tokens/s), hitting all 541 cached kernels. Its first run took
3.709 seconds (19.7 tokens/s), with two kernel compiles. The preceding two-card
StableHLO-attention run measured 3.889 seconds (18.8 tokens/s). These are
individual smoke-test measurements with the same prompt and decoding settings.
