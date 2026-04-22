"""Triton vector-add kernel — direct copy of tritonbench's reference kernel.

Used by `dump_python_ir.py` to produce the Python-side baseline TTIR/TTGIR/LLIR/PTX
that is diffed against the IR our Zig DSL produces in `kernels_zig/vector_add.zig`.
"""
import triton
import triton.language as tl


@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
