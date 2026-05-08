"""Triton vector-exp backward kernel: grad_input = grad_output * output."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import fake


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


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    return [
        fake("fp32", 1024),  # grad_output_ptr
        fake("fp32", 1024),  # output_ptr
        fake("fp32", 1024),  # grad_input_ptr
        1024,                # n_elements
    ], {
        "BLOCK_SIZE": int(cfg.get("BLOCK_SIZE", 1024)),
        "num_warps": 4,
        "num_stages": 1,
    }
