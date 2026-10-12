# StableHLO paged attention

This fallback is automatically selected for CPU and Furiosa. See the
[parent README](../README.md) for the general attention equation, packed
sequence metadata, and page addressing contract.

[The implementation](stablehlo.zig) iterates over sequences, query chunks, and
KV pages using StableHLO control flow. It merges partial softmax states with
`f32` maxima, exponential sums, and weighted-value accumulators, then converts
the final output to the query dtype. Query-key dot products use the input dtype.

## Current limitations

- Only split KV caches work. Although public dispatch forwards `.dense`, this
  implementation accesses `.split`. Latent caches are rejected by dispatch.
- Attention is always causal: `causal = false` is currently ignored. Sliding
  windows and explicit key indices are rejected. `opts.sink` is ignored; custom
  scale is supported.
- Multi-query chunks are processed a full page at a time, and the packed output
  cursor advances by the page size. Use page-multiple query chunks; arbitrary
  partial query chunks are not covered by this implementation. Single-query
  decode uses a separate one-row path.
- The sequence loop stops at the first empty query interval. Active sequences
  must precede trailing inactive entries.

The shared tests in [paged_attention.zig](../paged_attention.zig) use this path
as the causal comparison reference with `bf16`, page size 16, mixed
prefill/decode, shared prefix pages, partial final KV pages, and trailing
inactive sequences. Those cases do not establish support for every shape or
dtype, or conformance for the unsupported options above.
