# Mosaic TPU paged attention

This backend is automatically selected for TPU. See the
[parent README](../README.md) for the attention equation and packed sequence
metadata contract.

[The wrapper](mosaic_tpu.zig) requires fused `KvCache.dense` storage with shape
`{ .page = N, .k_chunk = P, .hkv = 2 * Hkv, .hd = Dpad }`. The combined head
axis interleaves `K0, V0, K1, V1, ...`. `Dpad` must equal the logical query head
dimension rounded up to a multiple of 128, with unused cache features zeroed.
Queries are padded before execution and outputs trimmed afterward. The default
scale uses the logical head dimension. Accepted query/cache dtypes are `bf16`,
`f16`, and `f32`.

Causal and non-causal standard masks and left sliding windows are supported.
Explicit key indices are rejected. `opts.sink` is currently ignored. Split
caches are rejected by ordinary paged attention, and the generic
`KvCache.update` method does not implement fused-cache writes.

The wrapper partitions heads on the model mesh and replicates sequence metadata.
It counts sequences with nonempty query intervals and passes that count to the
ragged kernel, so active sequences must precede trailing inactive entries.
Each local head computation must have the full required key range and features.

The shared split-cache comparison test does not exercise this backend. Wrapper
unit tests cover head-dimension alignment and configuration, including scale
and standard mask options; they do not establish runtime support for every
shape or dtype.
