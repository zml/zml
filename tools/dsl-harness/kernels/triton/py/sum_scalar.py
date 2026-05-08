"""Triton scalar-sum reduction. Each program sums one BLOCK_SIZE_M slice
and atomically adds the partial into a single scalar output."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import fake


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


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    return [
        fake("fp32", 1024),  # input_ptr
        fake("fp32", 1),     # output_ptr
        1024,                # M
    ], {
        "BLOCK_SIZE_M": int(cfg.get("BLOCK_SIZE_M", 1024)),
        "num_warps": 4,
        "num_stages": 1,
    }
