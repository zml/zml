# Tenstorrent attention

The single-card backend emits `ttir.scaled_dot_product_attention_decode` through
the TT-MLIR custom-call builder. It supports causal BF16 decode with one query,
grouped query heads, optional batches, and head/cache dimensions divisible by 32.
Prefill and other shapes use the general StableHLO implementation. Multi-card
automatic selection continues to use StableHLO attention.

The XLA plugin configures HiFi4 arithmetic, FP32 accumulation, and accurate
exponentials. Its Metalium patch makes the softmax loop honor the exponential
precision setting; the upstream loop otherwise hardcodes approximate math.

`//zml:tenstorrent_test` checks every output element against an independent f64
causal-attention calculation using BF16 inputs. Cases cover Llama's 32-query/8-KV
heads, batched GQA, Qwen's 24-query/4-KV heads with head dimension 256, and cache
lengths 128 and 1024. Positions span tile and chunk boundaries, plus values
beyond the cache including the largest u32. Bounds clamp before signed conversion.

Select it explicitly with `//examples/llm -- --backend=tenstorrent`; it is also
selected automatically for a single Tenstorrent card.
