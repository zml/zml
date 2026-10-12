# CuTe flash attention scheduling

The [parent README](../README.md) defines attention and sharding semantics.
This document describes decode scheduling in this implementation and the
FlashInfer heuristics that inform it. Scheduling changes parallelism and
intermediate storage, not the attention formula or public arguments.

## What FlashInfer does

These references are pinned to FlashInfer commit
`2a2cceab07c7bfca0525065fbfdc5fb2cff0c33a`; FlashInfer has multiple execution paths,
so these rules describe the cited CUDA decode paths rather than every backend.

[Single-sequence decode](https://github.com/flashinfer-ai/flashinfer/blob/2a2cceab07c7bfca0525065fbfdc5fb2cff0c33a/include/flashinfer/attention/decode.cuh#L705-L744)
uses one attention launch when the live sequence has at most 256 tokens or no
scratch buffer is supplied. Otherwise it queries the compiled kernel's resident
blocks per SM and the device's SM count, then computes:

```text
grid_budget = resident_blocks_per_SM * SM_count
split_budget = floor(grid_budget / KV_heads)
chunk_tokens = max(ceil(live_tokens / split_budget), 256)
splits = ceil(live_tokens / chunk_tokens)
```

It launches partial attention over these chunks followed by `MergeStates`.
The merge preserves a single softmax normalization across the chunks.

[Batched paged decode](https://github.com/flashinfer-ai/flashinfer/blob/2a2cceab07c7bfca0525065fbfdc5fb2cff0c33a/include/flashinfer/attention/scheduler.cuh#L75-L98)
uses the per-request lengths to search for a chunk size whose total block count
fits the occupancy budget. Its
[work estimator](https://github.com/flashinfer-ai/flashinfer/blob/2a2cceab07c7bfca0525065fbfdc5fb2cff0c33a/include/flashinfer/attention/scheduler.cuh#L174-L205)
disables splitting if `batch_size * KV_heads` already fills that budget. The
minimum chunk is `max(floor(128 / page_size), 1)` pages. Without CUDA graphs it
also avoids splitting when partitioning adds no requests. Graph execution can
retain a padded split schedule to keep launch and allocation sizes stable.

## Adaptation to this kernel

[`decodeSplits`](attention.zig) runs after sharding has established each device's
local dimensions. A block owns at most four query heads belonging to one KV
head, so our unsplit decode block count is:

```text
G = local_query_heads / local_KV_heads
base_blocks = local_batch * local_KV_heads * ceil(G / 4)
```

Counting query-head groups matters: a large GQA group already launches multiple
blocks per KV head in this kernel. Using global batch or head counts would
underestimate the need for splitting after partitioning across devices.

XLA compiles and owns the CUDA function after this planner runs. We cannot call
CUDA's occupancy calculator on that function here. Instead, the planner uses
PJRT's `core_count` and `shared_memory_per_block_optin` attributes and a
static limit of twelve 128-thread blocks per SM:

```text
padded_D = next_power_of_two(max(D, 32))
shared_bytes_per_block = 2 * 16 * padded_D * sizeof(dtype)  # K and V tiles
blocks_per_SM = max(1, min(12, floor(shared_budget / shared_bytes_per_block)))
grid_budget = 4 * SM_count * blocks_per_SM
```

The per-block shared-memory limit serves as a conservative shared-memory budget;
it is not a measurement of SM capacity or compiled-kernel occupancy. The
twelve-block limit is a scheduling estimate, not a measured occupancy or a
hardware constant. It provides enough resident work to hide latency in the
128-thread kernel; register usage and other hardware limits can reduce actual
occupancy. Four scheduling waves provide additional work for the serial
KV-tile loop; this is a measured adaptation, whereas the cited FlashInfer CUDA
planner targets one resident grid. Larger heads and FP32 reduce the budget through their shared-memory requirements. For
multiple visible devices, the smallest reported SM count and shared-memory
budget are used. Missing or invalid attributes disable splitting.

The selection rules are:

1. Prefill (`Lq > 1`) remains unsplit.
2. A cache capacity of at most 64 tokens remains unsplit.
3. If `base_blocks >= grid_budget`, keep one partition per block's head group.
4. Otherwise set `split_budget = ceil(grid_budget / base_blocks)`,
   `minimum_chunk = 16` for capacities through 256, otherwise `32`,
   `chunk = max(minimum_chunk, ceil(cache_capacity / split_budget))`, and
   `splits = ceil(cache_capacity / chunk)`.
5. If the result is one split, launch attention directly into the output. Other
   cases allocate FP32 partial states and launch attention followed by a merge.

The chunk floors are one or two of this kernel's 16-key tiles. They adapt
FlashInfer's minimum-work-per-partition heuristic to our serial per-tile score
loop. Applying its literal 128/256-token floors and a smaller block budget
regressed several measured cases. The workload-based budget and split/merge
structure follow FlashInfer, but these constants are specific to this kernel.
The split budget rounds up a partial wave: rounding 32 partitions down to 31,
for example, can add a whole tile to every block. The grid budget is therefore
a target, and the launched grid may exceed it by less than one split's worth
of `base_blocks`.

## Capacity and live lengths

Our graph receives `token_index` on device, potentially with a different value
for each batch element. Planning uses static cache capacity to bound the grid
and scratch allocation without reading positions back to the host. At runtime,
each sequence distributes its **live** prefix of `token_index + 1` keys evenly
across the planned splits. Therefore the actual chunk can be smaller than the
planning floor. Empty partitions contribute zero to the merge.

This is not FlashInfer's host-side ragged-length search. A short live sequence
in a large cache can still use the split path, and the merge launch remains in
the graph even when only one partition has keys. The schedule does not change
from token to token, so it can be reused during generation and graph capture.

Partial states contain the weighted-value accumulator, maximum, and exponential
sum in FP32. Their storage is
`4 * local_batch * splits * local_query_heads * (D + 2)` bytes. Reducing splits
as the batch grows reduces both scratch traffic and merge work. Scores are
scaled by `log2(e) / sqrt(D)` so attention and merging use base-2 exponentials.

## Validation and timing

The existing attention harness covers GQA/MQA, different positions within a
batch, partial tiles, empty partitions, sharding, three dtypes, and head
dimensions through 512. Planner tests cover the short-cache boundary, decreasing
split counts as batch size grows, resource limits, and extra query-head groups.
Runtime checks include batches through 2048 at the 64/65-token scheduling
boundary; they establish correctness, not throughput at that batch size.

An opt-in benchmark in [cute.zig](cute.zig) compares the previous 32-split policy
with the planner at batches 1, 4, 16, and 64, and full contexts of 128, 2048, and
8192 tokens. It uses BF16, 16 query heads, four KV heads, and `D = 128` per device.
Sixteen dependent attention calls amortize host overhead; reported microseconds
per attention call are the median of 16 samples after three warmups. This is a
repeated-cache kernel benchmark, not end-to-end model throughput.

```sh
bazel test //zml:test --//platforms:cuda=true --//platforms:cpu=false \
  --@rules_zig//zig/settings:zigopt=--test-filter \
  --@rules_zig//zig/settings:zigopt=zml.test.zml \
  --@rules_zig//zig/settings:zigopt=--test-filter \
  '--@rules_zig//zig/settings:zigopt=CuTe decode' \
  --test_env=CUDA_VISIBLE_DEVICES=0 \
  --test_env=ZML_CUTE_ATTENTION_BENCHMARK=1 --test_output=all
```

`CUTE_DECODE_BENCH` rows contain batch, context length, policy, split count, and
microseconds per call. Timing is not a correctness assertion.
