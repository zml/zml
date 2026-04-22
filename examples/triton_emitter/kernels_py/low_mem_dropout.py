"""Triton low-memory dropout — direct copy of tritonbench's `_triton_dropout`.

Uses an externally-supplied keep-mask buffer; the seeded variant from the same
operator depends on `tl.rand` and is intentionally left out here.
"""
import triton
import triton.language as tl


@triton.jit
def _triton_dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
