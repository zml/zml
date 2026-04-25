"""Triton vector-exp + backward — direct copy of tritonbench's reference kernels.

The original carries an optional `profile_mem` pointer hooked to a
`tritonbench.kernels.profile.time()` helper; that's a benchmarking concern, not
part of the kernel's math. We drop it here so the IR comparison is purely the
forward/backward computation.
"""
import triton
import triton.language as tl


@triton.jit
def triton_exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_exp_backward_kernel(
    grad_output_ptr,
    output_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    output = tl.load(output_ptr + offsets, mask=mask)
    grad_input = grad_output * output
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
