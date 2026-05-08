"""Python reference for `reduce_segments_ptr`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from triton_helpers import dtype_str, fake as _t
from unified_attention import reduce_segments_ptr  # type: ignore

__all__ = ["reduce_segments_ptr", "build_args"]


_NUM_TOKENS = 64
_NUM_SEQS = 1


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    o_dtype = dtype_str(cfg.get("o_dtype"), "bf16")

    num_query_heads = int(cfg.get("num_query_heads", 32))
    tile_size = int(cfg.get("tile_size", 64))
    head_size = int(cfg.get("head_size", 128))
    head_size_padded = int(cfg.get("head_size_padded", head_size))
    block_q = int(cfg.get("block_q", 16))
    num_segments_per_seq = int(cfg.get("num_segments_per_seq", 4))
    use_fp8 = bool(cfg.get("use_fp8", False))
    fp8_min = float(cfg.get("fp8_min", -448.0))
    fp8_max = float(cfg.get("fp8_max", 448.0))

    num_tokens = _NUM_TOKENS
    segm_n = num_tokens * num_query_heads * num_segments_per_seq

    args = [
        # Inputs from the 3d pass: segm_output, segm_max, segm_expsum.
        _t("fp32", segm_n * head_size_padded),
        _t("fp32", segm_n),
        _t("fp32", segm_n),
        # seq_lens, num_seqs, num_query_heads (constexpr), out_scale_inv.
        _t("i32", _NUM_SEQS),
        _t("i32", 1),
        num_query_heads,            # NUM_QUERY_HEADS
        _t("fp32", 1),              # out_scale_inv_ptr
        # output_stride_0/1, block_table_stride.
        _t("i64", 1), _t("i64", 1), _t("i64", 1),
        tile_size,                  # TILE_SIZE
        head_size,                  # HEAD_SIZE
        head_size_padded,           # HEAD_SIZE_PADDED
        _t("i32", _NUM_SEQS + 1),   # query_start_len_ptr
        block_q,                    # BLOCK_Q
        num_segments_per_seq,       # NUM_SEGMENTS_PER_SEQ
        use_fp8,
        fp8_min,
        fp8_max,
        # Output buffer placeholder.
        _t(o_dtype, num_tokens * num_query_heads * head_size),
    ]
    return args, {}
