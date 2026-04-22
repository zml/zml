"""Triton scalar-sum reduction — direct copy of tritonbench's
`triton_sum_kernel_scalar_result`. Each program sums one BLOCK_SIZE_M slice and
atomically adds the partial into a single scalar output."""
import triton
import triton.language as tl


@triton.jit
def triton_sum_kernel_scalar_result(
    input_ptr,
    output_ptr,
    M,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < M
    x = tl.load(input_ptr + offsets, mask=mask, other=mask)
    output = tl.sum(x)
    output_offsets = tl.arange(0, 1)
    tl.atomic_add(output_ptr + output_offsets, output)
