"""Triton row-softmax."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import fake


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    return [
        fake("fp32", 1024 * 64),  # output_ptr
        fake("fp32", 1024 * 64),  # input_ptr
        1024,                     # input_row_stride
        1024,                     # output_row_stride
        1024,                     # n_cols
    ], {
        "BLOCK_SIZE": int(cfg.get("BLOCK_SIZE", 1024)),
        "num_warps": 4,
        "num_stages": 1,
    }
