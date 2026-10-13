# Metal paged attention

This backend is automatically selected for Metal. See the
[parent README](../README.md) for the attention equation, packed sequence
metadata, and page addressing contract.

[The wrapper](metal.zig) requires split KV caches. It merges query head groups,
transposes queries and caches into the custom call's layouts, invokes
`zml$paged_attn`, and restores the query's tagged layout for the output.

Causal and non-causal standard masks, left sliding windows, and a custom scale
are passed to the custom call. Explicit key indices are rejected, and
`opts.sink` is currently ignored. Fused caches are rejected by
ordinary paged attention.

When sharding heads, keep complete GQA groups together with matching query and
cache head ownership. Each local computation needs the full required key range,
features, and matching sequence metadata.

The shared tests in [paged_attention.zig](../paged_attention.zig) compare this
backend against StableHLO for causal attention and include option-specific
comparisons. They exercise `bf16`, mixed prefill/decode, shared prefix pages,
partial final KV pages, and trailing inactive sequences. These cases do not
establish support for every shape or dtype.
