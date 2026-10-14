# Triton paged attention

This backend is automatically selected for CUDA, ROCm, and oneAPI. See the
[parent README](../README.md) for the attention equation, packed sequence
metadata, and page addressing contract.

[Triton's wrapper](triton.zig) uses split KV caches and supports causal and
non-causal attention, a left sliding window, explicit key indices, and softmax
sinks. Query heads are grouped by KV head. Keep complete GQA groups together
when sharding `.hkv`, with matching query and cache ownership and complete
`.hg` and `.hd` axes. Each local computation needs its full key range and the
matching sequence metadata.

The wrapper chooses between a 2D kernel and a 3D split-key kernel followed by
segment reduction. Prefill uses the 2D path; all-decode batches select a path
based on batch size and KV head count. oneAPI forces split-key execution for
all-decode batches and has a specialized path for standard masks. Segment
reduction combines softmax maxima, exponential sums, and weighted values to
preserve one normalization over the key range.

Set `is_prefill = true` for mixed prefill/decode batches. `max_seqlen_q` guides
kernel configuration. These options do not change logical query positions.

The shared tests in [paged_attention.zig](../paged_attention.zig) exercise
`bf16`, mixed prefill/decode, shared prefix pages, partial final KV pages,
trailing inactive sequences, and option-specific comparisons. Kernel selection
and numerical behavior depend on the target; these cases do not establish
support for every shape or dtype.
