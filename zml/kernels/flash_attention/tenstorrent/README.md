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

On the P150a, Llama 3.1 8B's counting prompt with sequence length 128 and greedy
decoding completed in 4.489 seconds (16.3 tokens/s), versus 4.778 seconds
(15.3 tokens/s) for StableHLO attention on the same build. Both runs produced the
same answer and hit all 495 cached device kernels. The first native-attention run
had six cache misses and took 5.423 seconds; compare warm runs for throughput.

The two-card Llama run with head partitions and KV-cache donation completed in
3.503 seconds (20.8 tokens/s), hitting all 541 cached kernels. Its first run took
3.709 seconds (19.7 tokens/s), with two kernel compiles. The preceding two-card
StableHLO-attention run measured 3.889 seconds (18.8 tokens/s). These are
individual smoke-test measurements with the same prompt and decoding settings.
