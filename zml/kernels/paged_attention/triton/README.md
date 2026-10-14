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
based on batch size and KV head count. Sparse decode always uses the
existing 3D kernel, specialized by the Zig emitter. oneAPI forces split-key execution for
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

## Sparse decode tuning

Sparse decode uses 16- or 32-position gather tiles, four attention warps (eight for oneAPI decode),
and an occupancy-based split count capped by the selection length. Only useful
segments launch; scratch capacity remains a power of two for reduction. A
single-token decode grid addresses sequences directly, including inactive slots.
Small segments scan indices before loading Q, wholly empty tiles skip the KV
loads and dot products, and empty segments omit partial-output stores when
sinks are disabled. The reducer masks these unwritten outputs using segment mass.

The wrapper specializes known strides and scale in the existing kernel. Sparse
gathers use Triton’s default cache policy, store statistics with contiguous
heads, and reduce head dimensions in tiles of at most 64 features.
Producer and reducer must agree on the statistics layout. These are emitter
specializations, not additional kernels.

The sparse regression test covers holes in packed sequence metadata, negative
and out-of-range selections, empty rows, valid selections after padding, sinks,
non-power-of-two GQA groups, padded head dimensions, and partial selection tiles.
It runs both prefill and decode on platforms whose default attention backend is
Triton; other platforms skip this test.

### Validation on GB300

On 2026-10-14, the supplied patch and this integration were compared using only
physical GPU 3 on `gb300-2` (152 SMs), with Triton 3.7.1. The workload uses BF16,
12 query heads, one KV head, head dimension 256, page size 16, and 2051 selection
slots. The table reports median attention-plus-reduction latency across six
alternating CUDA-graph measurements per variant. Each measurement uses a 200 ms
replay budget after warmup.

| Batch | Valid keys | Supplied patch (µs) | Portable integration (µs) |
| --- | --- | --- | --- |
| 1 | 0 | 4.568 | 4.511 |
| 1 | 129 | 6.231 | 6.280 |
| 1 | 2049 | 6.432 | 6.435 |
| 16 | 0 | 7.431 | 7.419 |
| 16 | 129 | 9.254 | 9.208 |
| 16 | 2049 | 20.339 | 20.355 |
| 32 | 0 | 8.590 | 8.658 |
| 32 | 129 | 10.469 | 10.455 |
| 32 | 2049 | 29.594 | 29.969 |

The portable integration emits no inline assembly and has no CUDA-specific
sparse dispatch or tuning gates. Registers and shared-memory usage match the
supplied patch. Measured combined latency differs by at most 1.3%. These are direct
kernel measurements with identical inputs and launch configurations; performance
and correctness on ROCm and oneAPI have not been validated in this run.

All nine cases produce bit-identical outputs between variants and pass comparison
with FP32 reference attention; maximum absolute error is 0.002262. Scratch buffers
were initialized to NaN to verify that empty and unlaunched segments are masked.
The focused CUDA attention run passes, including the sparse regression's sinks,
packed-sequence holes, padded heads, and both prefill and decode paths. The broad
CUDA suite stops in MoE tests outside the modified attention files.
