"""Triton low-memory dropout — externally-supplied keep-mask buffer, no
`tl.rand`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import fake


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


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    return [
        fake("fp32", 1024),  # x_ptr
        fake("fp32", 1024),  # x_keep_ptr
        fake("fp32", 1024),  # output_ptr
        1024,                # n_elements
        0.5,                 # p (default keep-prob)
    ], {
        "BLOCK_SIZE": int(cfg.get("BLOCK_SIZE", 1024)),
        "num_warps": 4,
        "num_stages": 1,
    }
