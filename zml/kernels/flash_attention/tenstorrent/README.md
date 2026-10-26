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

Select it explicitly with `//examples/llm -- --backend=tenstorrent`; it is also
selected automatically for one or two Tenstorrent cards.

Llama stores its Tenstorrent KV cache in `[layer, head, sequence, head_dim]`
order. Keeping the attention kernel's order across tokens removes full-cache
permutations and avoids padding the eight KV heads to a 32-row tile. The same
tagged update path handles prefill and decode and preserves buffer donation.
`//examples/llm:llama_cache_test` checks every cache element after 32-token
prefills and one-token updates at positions 0, 31, 32, and 63, including both
layers, untouched positions, and preservation of update inputs.

With this layout and XLA's tuned BF8 down-projection, the single-card Llama
3.1 8B counting smoke test at sequence length 128 measured:

| Weights | Decode time (73 tokens) | Throughput | Kernel cache hits |
| --- | ---: | ---: | ---: |
| BF16 | 3.618 s | 20.2 tokens/s | 462/462 |
| BF8 | 2.574 s | 28.3 tokens/s | 488/488 |

Both runs produced the correct count from 1 to 20. Before the layout change,
the corresponding BF16/BF8 runs measured 18.0/24.2 tokens/s. BF8 is enabled by
`XLA_TT_EXPERIMENTAL_BF8_WEIGHTS=1`; activations and KV caches remain BF16.
Trace replay is disabled in these measurements. These are short application
smoke tests, not model-quality evaluations.

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
