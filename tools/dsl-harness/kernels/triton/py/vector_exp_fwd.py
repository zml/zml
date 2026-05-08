"""Triton vector-exp forward kernel."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import fake


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


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    return [
        fake("fp32", 1024),  # x_ptr
        fake("fp32", 1024),  # output_ptr
        1024,                # n_elements
    ], {
        "BLOCK_SIZE": int(cfg.get("BLOCK_SIZE", 1024)),
        "num_warps": 4,
        "num_stages": 1,
    }
