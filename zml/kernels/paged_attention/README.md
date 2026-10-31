# Paged attention semantics

[paged_attention.zig](paged_attention.zig) evaluates attention over a KV cache
whose logical sequences are stored in fixed-size physical pages. Paging changes
how keys and values are addressed; softmax still normalizes over all permitted
keys in one sequence, independently for each query and query head.

The public entry point is:

```zig
pagedAttention(parameters, q, k, v, kv_cache, opts)
```

The result has the query's shape and dtype. The `k` and `v` arguments are currently
ignored: attention reads exclusively from `kv_cache`. Populate the cache,
including the current queries' keys and values, **before** calling attention.
The attention call does not update the cache or allocate pages. Projections,
RoPE, and any Q/K normalization happen before this operation; the output
projection happens afterward.

## Tensor layouts and heads

Let `T` be the packed query token count, `B` the sequence capacity, `N` the
physical page count, `P` the tokens per page, `M` the page-table capacity per
sequence, `Hkv` the KV head count, `G` the query heads per KV head, and `D` the
features per head.

| Input | Tagged shape | Meaning |
| --- | --- | --- |
| `q` | `{ .b = T, .hkv = Hkv, .hg = G, .hd = D }` | Queries packed in sequence order. Here `.b` counts tokens, not sequences. |
| Split cache `k`, `v` | `{ .page = N, .k_chunk = P, .hkv = Hkv, .hd = D }` | Physical key/value storage. |
| `parameters.block_table` | `{ .b = B, .p = M }`, `i32` | Logical page to physical page mapping for each sequence. |
| `parameters.seq_lens` | `{ .b = B }`, `i32` | Valid KV token counts, including the current queries. |
| `parameters.query_start_len` | `{ .b = B + 1 }`, `i32` | Cumulative offsets into packed queries. |

The total query head count is `Hq = Hkv * G`. Flattening the two head axes gives
`h = hkv * G + hg`, so query head `h` reads KV head `floor(h / G)`. Each query
head has its own scores and softmax. This covers MHA (`G = 1`), GQA, and MQA
(`Hkv = 1`). `D` is the feature count of one head, not the number of heads or
the sequence length. Use matching floating-point query and cache dtypes.

The TPU backend instead requires `KvCache.dense`, a fused tensor with layout
`{ .page = N, .k_chunk = P, .hkv = 2 * Hkv, .hd = Dpad }`. The combined head
axis interleaves `K0, V0, K1, V1, ...`. `Dpad` must equal `D` rounded up to a
multiple of 128. Pad unused cache features with zeros. The wrapper pads queries,
uses the logical `D` for the default scale, and trims the output back to `D`.
Its accepted query/cache dtypes are `bf16`, `f16`, and `f32`.

## Packed sequences and positions

For sequence `s`, define:

```text
start       = query_start_len[s]
end         = query_start_len[s + 1]
Lq          = end - start
Lk          = seq_lens[s]
context_len = Lk - Lq
position(i) = context_len + i          # 0 <= i < Lq
```

Queries `q[start:end]` represent the final `Lq` tokens of that sequence. The
kernel infers their positions from lengths; there is no separate position input.
An initial prefill has `Lq = Lk`, a chunked prefill has `Lq > 1` after a cached
prefix, and a decode normally has `Lq = 1` at position `Lk - 1`. A batch may mix
prefill and decode sequences. Set `is_prefill = true` for such mixed batches;
`false` selects the all-decode execution configuration on Triton. It does not
change the attention equation. `max_seqlen_q` is a configuration bound for the
largest query chunk, not a runtime sequence length.

Offsets must begin at zero and be nondecreasing. Active sequences require
`0 < Lq <= Lk <= M * P`. Place inactive sequences after the active sequences,
with repeated offsets and normally `seq_lens = 0`. This ordering matters:
StableHLO stops at the first sequence with no queries, and TPU passes a count of
active sequences to its kernel. The final offset is the active packed token
count; any extra query capacity is padding whose output should not be consumed.

## Page addressing and cache updates

A logical key position `j` in sequence `s` is addressed as:

```text
logical_page  = floor(j / P)
physical_page = block_table[s, logical_page]
offset        = j % P
K(s, j)       = k_cache[physical_page, offset]
V(s, j)       = v_cache[physical_page, offset]
```

Physical pages need not be contiguous or ordered. Different sequences may map
to the same physical pages to share a prefix. Every page covering a valid token
must have a valid physical ID in `[0, N)` and populated KV data. Entries beyond
`ceil(Lk / P)` are unused; tests use `-1` there. A negative ID is not a portable
way to mask a valid logical page. `seq_lens`, rather than allocated page count,
determines which tokens in the last page are valid.

`KvCache.update(new_k, new_v, slot_mapping, chunk_size, backend)` writes split
cache entries using physical flattened slots:

```text
chunk_size    = P
physical_page = slot_mapping[t] / P
offset        = slot_mapping[t] % P
```

`new_k` and `new_v` have packed token/head/feature axes matching the split
cache. `slot_mapping` is a physical address, whereas `block_table` translates
logical sequence positions. The caller maintains both mappings and lengths;
`update` does not derive or update sequence metadata. Writes override existing
entries and request buffer reuse. Use valid, distinct slots for writes requiring
deterministic results; conflicting scatter writes have no ordering guarantee.
The generic update method currently panics for `.dense`.

## Attention equation and masks

For local query row `i`, KV head `a`, and query group head `g`:

```text
p       = Lk - Lq + i
score_j = scale * dot(q[start + i, a, g], K(s, j, a))
weight  = softmax(score over permitted j)
out     = sum_j(weight_j * V(s, j, a))
```

`opts.scale` defaults to `1 / sqrt(D)`. There is no dropout or returned
probability matrix. Internal paging, tiling, or splitting of the key range must
preserve one softmax normalization; summing separately normalized page outputs
would compute a different result. Accumulation order and dtype conversions can
cause numerical differences between backends.

The default mask is `.standard = .{ .causal = true }`:

| Mask | Permitted logical key positions |
| --- | --- |
| Default causal | `0 <= j < Lk` and `j <= p`. Includes the query's own key. |
| `AttentionMask.none` | Every `0 <= j < Lk`. |
| Causal with `sliding_window = W` | `0 <= j < Lk` and `p - W < j <= p`. |
| Non-causal with `sliding_window = W` | `0 <= j < Lk` and `p - W < j`. Includes all later valid keys. |

Use a positive `W`; it counts the current token as part of the left window.
Use `null` for no window. A non-causal window is a left bound, not a symmetric
window around the query.

Triton also accepts `.indices`, an `i32` tensor shaped
`{ .b = T, .topk = K }` with `K > 0`. Each packed query row supplies logical
key positions in its own sequence, shared across its heads. Positions outside
`[0, Lk)` are ignored. This replaces the standard mask, so future valid keys
are permitted. Duplicate positions are processed repeatedly and therefore
change their weight; provide unique positions to represent a set of keys.

On Triton, `opts.sink` supplies one softmax logit per flattened query head.
It adds `exp(sink[h])` to the denominator and contributes zero to the weighted
value numerator. It is not a cache token and does not consume a page or sequence
position. Other wrappers currently ignore `sink`. For portable use, ensure each
active query has at least one permitted key; fully masked rows are not given a
shared cross-backend output contract.

## Example

With `P = 4`, consider a two-query chunk after a four-token prefix and a decode
after a two-token prefix, followed by one inactive sequence:

```text
query_start_len = [0, 2, 3, 3]
seq_lens        = [6, 3, 0]
block_table     = [[5, 2], [8, -1], [-1, -1]]
```

Packed queries 0 and 1 belong to sequence 0 at positions 4 and 5. Their current
KV entries occupy physical page 2, offsets 0 and 1 (slots 8 and 9). Packed query
2 belongs to sequence 1 at position 2, physical page 8, offset 2 (slot 34).
The default causal mask permits keys 0–4, 0–5, and 0–2 respectively, always
within the query's own sequence. With a window of 3, the permitted positions
become 2–4, 3–5, and 0–2.

## Backend implementations

See each backend's README for execution details, supported options, and current
limitations:

- [Triton](triton/README.md) — CUDA, ROCm, and oneAPI.
- [CuTe](cute/README.md) — CUDA benchmark exports and Nsight Compute capture.
- [Mosaic TPU](mosaic_tpu/README.md) — TPU.
- [Metal](metal/README.md) — Metal.
- [StableHLO](stablehlo/README.md) — CPU and Furiosa fallback.
