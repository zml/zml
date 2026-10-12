# Flash attention semantics

This directory exposes causal scaled dot-product attention through
[`zml.flash_attention.attention`](flash_attention.zig). This document describes
the attention operation, the FlashAttention algorithm, and ZML's current semantic
contract. Shapes below are logical tensor shapes. Sharding describes ownership
of those tensors across devices; platform-specific execution layouts are outside
the scope of this document.

## The attention operation

For one sequence and one head, let `Q` contain `Lq` query vectors, and let `K` and
`V` contain `Lk` key/value vectors. Each query is compared with each permitted key;
the resulting probabilities weight the corresponding values:

```text
S = scale * Q @ transpose(K) + M
P = softmax(S, axis=keys)
O = P @ V
```

The usual scale is `1 / sqrt(D)`, where `D` is the query/key head dimension.
An additive mask `M` is zero for allowed pairs and negative infinity for forbidden
pairs. Softmax normalizes each query row independently over keys, so a masked key
has zero probability. Heads and batch elements are independent. ZML's
[`sdpa`](../../nn.zig) implements this reference computation.

Attention in general permits a value dimension `Dv` different from the query/key
dimension `D`; the output then has dimension `Dv`. This module uses the same
`.hd = D` for queries, keys, values, and output.

FlashAttention can evaluate either unrestricted attention or masked attention.
Causality is a choice of mask, rather than an inherent property of the algorithm.
Other attention APIs may offer local windows, additive score biases, a custom
scale, and dropout on the probabilities during training. These options change
the operation and must be matched when comparing results. The
[upstream FlashAttention API](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)
documents such options; ZML's interface here exposes a narrower operation.

## What makes attention "flash"

FlashAttention computes the same mathematical attention result while avoiding
storage of the complete `Lq × Lk` score and probability matrices. It processes
query/key/value tiles and combines partial results using an online softmax.
This reduces intermediate memory and memory traffic. Dense attention still
requires `O(Lq * Lk * D)` arithmetic per head; for self-attention with `Lq = Lk`,
that remains quadratic in sequence length. "Exact" means the attention formula
is preserved, with ordinary floating-point rounding differences. See the
[FlashAttention paper](https://arxiv.org/abs/2205.14135).

For each query row, online softmax maintains a maximum `m`, an exponential sum
`l`, and an unnormalized weighted-value vector `u`. Given scores `s_j` for a new
tile of permitted keys, update:

```text
m_new = max(m, max_j(s_j))
a     = exp(m - m_new)
p_j   = exp(s_j - m_new)
l_new = a * l + sum_j(p_j)
u_new = a * u + sum_j(p_j * V_j)
O     = u / l                         # after all tiles
```

Initialize with `m = -infinity`, `l = 0`, and `u = 0`; skip tiles with no permitted
keys. Rescaling the previous accumulators preserves a single normalization across
all tiles. Independently normalizing each tile and adding its output would give
a different result. The [FlashAttention-2 paper, sections 2–3](https://arxiv.org/html/2307.08691v1)
describes this recurrence and how training can recompute probability tiles during
the backward pass instead of retaining the full matrix. This module exposes only
the attention output, with no backward entry point or returned probabilities.

## Heads and head dimensions

`D` counts features in **one head**. It is distinct from the number of heads and
the sequence length. In a conventional multi-head projection,
`D = hidden_size / Hq`; models with an explicit head dimension need not follow
that relation. The query projection has width `Hq * D`, while each key/value
projection has width `Hkv * D`.

Typical dimensions include:

| Head dimension | Context |
| --- | --- |
| `64` | A common transformer head size; also the size used by this module's shared tests. |
| `128` | Another common size; Qwen2.5-7B uses `3584 / 28 = 128`. |
| `256` | Used by models such as Gemma 2 2B and 9B. |
| `512` | Used by Gemma 4 global attention layers, for example in Gemma 4 31B. |

The [FlashAttention-2 paper](https://arxiv.org/html/2307.08691v1) describes typical
head sizes around 64–128. The model examples follow the
[Qwen2.5-7B configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json)
and [Gemma configurations](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py).
Gemma can also use different head dimensions within one model: the
[Gemma 4 31B configuration](https://huggingface.co/google/gemma-4-31B/blob/main/config.json)
specifies `head_dim = 256` for local attention and `global_head_dim = 512` for
global attention. Use the dimension configured for the specific layer.
These are model dimensions, not a list of supported kernel sizes or a mathematical
requirement that `D` be a power of two.

The number of query heads `Hq` may differ from the number of key/value heads
`Hkv`. ZML uses contiguous groups of query heads:

```text
G = Hq / Hkv
kv_head(h) = floor(h / G)
```

`Hq` must be divisible by `Hkv`, and keys and values must have the same head count.
This covers multi-head attention (MHA: `Hq = Hkv`), grouped-query attention
(GQA: `1 < Hkv < Hq`), and multi-query attention (MQA: `Hkv = 1`). For example,
with `Hq = 64` and `Hkv = 8`, query heads 0–7 use KV head 0, heads 8–15 use KV
head 1, and so on. Sharing keys and values does not merge query heads: each query
head still has its own scores, normalization, and output.

## ZML inputs and output

The entry point is:

```zig
attention(q, k, v, token_index, metadata, parameters)
```

The reference contract uses these tagged shapes, with an optional `.b` batch axis:

| Argument | Unbatched shape | Batched shape | Meaning |
| --- | --- | --- | --- |
| `q` | `{ .q = Lq, .h = Hq, .hd = D }` | `{ .b = B, .q = Lq, .h = Hq, .hd = D }` | Queries to evaluate. |
| `k`, `v` | `{ .k = Lk, .h = Hkv, .hd = D }` | `{ .b = B, .k = Lk, .h = Hkv, .hd = D }` | Keys and values in sequence order. |
| `token_index` | Scalar | `{ .b = B }` or scalar | Zero-based position of the first query in the key sequence; a scalar applies to the whole batch. |
| Result | Same as `q` | Same as `q` | Weighted values for every query head. |

Use matching floating-point dtypes for `q`, `k`, and `v`; the output retains the
query dtype. The shared tests use `bf16` tensors and `u32` positions. `metadata`
and `parameters` provide execution configuration rather than additional masks,
score biases, or model projections.

### Causality and query positions

Let `t = token_index` for a sequence. Query row `i` represents position `t + i`,
and key row `j` represents position `j`. The inclusive causal rule is:

```text
allowed(i, j) = (j <= t + i)
M[i, j]      = 0 if allowed(i, j), otherwise -infinity
```

The reference path constructs a square causal mask over `Lk` positions, gathers
`Lq` consecutive rows starting at `t`, and passes that mask to `sdpa`. Valid inputs
require `Lq > 0`, `t >= 0`, and `t + Lq <= Lk`. The test helper checks the upper
bound; callers must satisfy it rather than rely on out-of-range gather behavior.
Each valid query can attend to at least its own key, so fully masked rows are
outside this contract.

| Use case | Query positions | Keys visible to query row `i` |
| --- | --- | --- |
| Initial prefill | `t = 0`, usually `Lq = Lk` | `0` through `i`. |
| Chunked prefill | `Lq > 1`, starting at `t` | `0` through `t + i`. |
| Single-token decode | `Lq = 1`, current token at `t` | `0` through `t`, including the current token. |

For `Lq = 2`, `Lk = 5`, and `t = 2`, the mask is:

```text
             key position
             0 1 2 3 4
query at 2   1 1 1 0 0
query at 3   1 1 1 1 0
```

Here `1` means permitted. `Lk` may be the capacity of a preallocated KV buffer;
it does not determine the current query position. All permitted KV entries must
already be populated. Positions after the last query are masked, and the call
does not append to or update a KV cache.

For unequal query and key lengths, causal alignment must be specified explicitly.
The [upstream FlashAttention API](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag)
uses bottom-right alignment: `j <= i + Lk - Lq`. ZML's reference rule agrees with
that convention when `t = Lk - Lq`. An explicit `token_index` also describes
queries earlier in a larger buffer, as in the example above.

### Sharding and local attention

Sharding changes where attention is evaluated, not its mathematical result.
The global output still has the query's tagged shape and dtype. Each local call
must receive the matching queries, keys, values, and positions for the batch
elements and heads it owns.

| Axis or input | Sharding semantics |
| --- | --- |
| Batch `.b` | Sequences are independent. Q, K, V, and the output must assign the same batch elements to each shard. |
| Heads `.h` | Each shard must have the KV heads needed by its query heads, preserving `kv_head(h) = floor(h / G)`. K and V must have matching head ownership. |
| Query sequence `.q` | Query rows are independent once the required KV entries are available. Their causal positions remain positions in the original sequence. |
| Key sequence `.k` | Softmax normalizes over all permitted keys for a query, including keys on other shards if the sequence is distributed. Separate local softmax outputs cannot simply be added or averaged. |
| Features `.hd` | Every score uses the complete head dimension and the global scale `1 / sqrt(D)`. Splitting features requires combining partial dot products before softmax. |
| `token_index` | A scalar applies to every shard. Per-sequence positions follow the batch partition and are replicated across head shards of the same sequence. Head or batch sharding does not change their values. |

For head sharding without KV replication, a partition count `P` must divide
`Hkv`, and therefore also `Hq`. Each shard owns complete, contiguous GQA groups:
`Hq_local = Hq / P`, `Hkv_local = Hkv / P`, and the local ratio remains `G`.
For example, with `Hq = 32`, `Hkv = 8`, and two shards, the first shard owns query
heads 0–15 and KV heads 0–3; the second owns query heads 16–31 and KV heads 4–7.
Each query still uses exactly the same KV head as in the unsharded computation.
MQA has only one KV head, so distributing its query heads over multiple devices
requires replicating that KV head rather than dividing it into independent heads.

The local-kernel execution contract keeps `.q`, `.k`, and `.hd` complete on each
shard, with parallelism across batches and/or complete head groups. Thus each
local call has the full sequence coordinates and feature vectors needed by the
causal mask, dot products, and normalization. The mathematical possibilities for
sequence and feature sharding above do not imply that these local kernels
implement distributed sequence or feature reductions.

An **open** partition means placement is unresolved; it does not mean the tensor
must be replicated. Operations such as cache slicing and RoPE can reopen
partition annotations even when their inputs were sharded. Before entering a
local kernel, resolve compatible partitions on a common mesh and align Q, K, V,
and positions. The KV layout anchors this alignment: an explicit KV head
partition is retained, while an open head partition can use a compatible mesh
axis whose partition count divides `Hkv`. Preserve explicit batch ownership and
do not assign the same device partition independently to both batch and heads.
An explicit replicated KV head axis remains replicated; open heads can also
remain replicated when no compatible head partition is available. Automatic
partition selection is an execution choice, not an additional attention argument.

The output follows the **resolved** query partition. Its partition annotations
need not match the incoming query's unresolved annotations, and aligning operands
may require communication. In particular, unnecessarily resolving open KV heads
to replicated can gather the full cache on every attention call even though the
head groups could be evaluated independently. This changes cost, not attention
semantics, and does not modify the caller's KV cache.

Splitting KV work into internal tiles or partitions is distinct from sharding the
input tensors across devices. Either way, combining partial attention states
must preserve the single softmax normalization. For disjoint key subsets with
states `(m_r, l_r, u_r)` as defined above:

```text
m = max_r(m_r)
l = sum_r(exp(m_r - m) * l_r)
u = sum_r(exp(m_r - m) * u_r)
O = u / l
```

An empty subset contributes zero (`m_r = -infinity`, `l_r = 0`, `u_r = 0`).
Every valid query has at least one permitted key overall, so its merged
normalization is nonempty.

### Scope and current conformance

The public call always requests causal attention with scale `1 / sqrt(D)` and no
dropout. It has no arguments for an arbitrary mask, padding lengths, packed
sequence boundaries, a local window, ALiBi, score soft-capping, a softmax sink, or
a custom scale. The lower-level `sdpa` and `causalAttnMask` helpers expose some
additional options, but this wrapper does not pass them through. Separate batch
elements can have different `token_index` values; concatenating independent
sequences along `.k` does not isolate them from each other.

Q/K/V projections, positional transformations such as RoPE, and any Q/K
normalization must happen before this call. Head concatenation and the output
projection happen afterward. A model's use of a familiar head dimension alone
does not establish compatibility with this operation.

The reference contract above is broader than current kernel coverage. In
particular, nonzero-offset multi-query prefill and the optional batch dimension
are not uniformly supported by the current kernels; an existing prefill path
requires equal query/key lengths and assumes positions begin at zero. Treat
these as current conformance gaps, not alternative definitions of causality.

The tests in [flash_attention.zig](flash_attention.zig) compare results
against `sdpa` with the gathered causal mask. The original shared cases cover
`D = 64`, MHA and GQA,
single-token decode at full and partial context lengths, batched decode with
different positions, and an eight-query chunk beginning at position 56 in a
64-position KV buffer. Additional conformance cases exercise head dimensions
through `512`, MQA, `bf16`/`f16`/`f32`, initial and offset prefill, partial tiles,
transposed tagged layouts, scalar batch positions, head/batch sharding, open
query/KV partitions, and split decode with empty partitions and cache tails. These
cases are enabled for kernels that support them; their presence does not imply
uniform coverage across every kernel. FP32 results are also checked against an
independent host calculation using FP64 arithmetic.

Numerical agreement is tolerance-based. The reference evaluates softmax in
`f32`, then converts probabilities to the query dtype before multiplying by
values. Different accumulation orders and rounding points can produce small
differences even when the mathematical operation is identical. The shared tests
use absolute tolerance `5e-3`, relative tolerance `1e-2`, relative epsilon `1e-3`,
and a minimum close fraction of `0.99`; these are test criteria rather than a
universal error bound.
