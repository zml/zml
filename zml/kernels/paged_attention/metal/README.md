# Metal paged attention

This backend is automatically selected for Metal. See the
[parent README](../README.md) for the attention equation, packed sequence
metadata, and page addressing contract.

[The wrapper](metal.zig) requires split KV caches. It merges query head groups,
transposes queries and caches into the kernels' layouts, launches the
[Metal DSL kernels](paged_attn.zig), and restores the query's tagged layout for
the output.

Queries and caches are `bf16` or `f16`; other dtypes panic. Causal `bf16`
decode with head dimension 32, 64, 128, 256 or 512 runs `fa_vec_paged`, which
is `bf16` only: a threadgroup holds up to four query heads of a KV head, times
splits of the KV positions, at most 256 threads. Rows split over the positions
are merged by `fa_vec_paged_reduce`; the split count follows the GPU core count
and the number of rows. Everything else, `f16` decode included, runs
`paged_attention_tiled` (head dimension 32, 64, 96, 128, 256 or 512, pages of
8, 16 or 32 positions), then `zero_padding_rows`.

Causal and non-causal standard masks, left sliding windows, and a custom scale
are passed to the kernels. Explicit key indices are rejected, and
`opts.sink` is currently ignored. Fused caches are rejected by
ordinary paged attention.

When sharding heads, keep complete GQA groups together with matching query and
cache head ownership. Each local computation needs the full required key range,
features, and matching sequence metadata.
