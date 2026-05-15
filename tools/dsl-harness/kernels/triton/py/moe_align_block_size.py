"""Python reference for `moe_align_block_size_kernel`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from triton_helpers import fake as _t
from moe import moe_align_block_size_kernel  # type: ignore

__all__ = ["moe_align_block_size_kernel", "build_args"]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    numel = int(cfg.get("numel", 1024))
    num_experts = int(cfg.get("num_experts", 8))
    padded_num_experts = int(cfg.get("padded_num_experts", num_experts))
    max_num_tokens_padded = int(cfg.get("max_num_tokens_padded", 2048))
    max_num_m_blocks = int(cfg.get("max_num_m_blocks", 32))
    block_size_m = int(cfg.get("block_size_m", 64))
    hist_block = int(cfg.get("hist_block", 64))

    args = [
        _t("i32", numel),                       # topk_ids_ptr
        _t("i32", max_num_tokens_padded),       # sorted_token_ids_ptr
        _t("i32", max_num_m_blocks),            # expert_ids_ptr
        _t("i32", 1),                           # num_tokens_post_pad_ptr
        _t("i32", num_experts + 1),             # cumsum_ptr
        block_size_m,                           # BLOCK_SIZE_M (constexpr)
        numel,                                  # NUMEL (constexpr)
        num_experts,                            # NUM_EXPERTS (constexpr)
        padded_num_experts,                     # PADDED_NUM_EXPERTS (constexpr)
        max_num_tokens_padded,                  # MAX_NUM_TOKENS_PADDED (constexpr)
        max_num_m_blocks,                       # MAX_NUM_M_BLOCKS (constexpr)
        hist_block,                             # HIST_BLOCK (constexpr)
        _t("i32", 1),                           # out0_ptr
        _t("i32", 1),                           # out1_ptr
        _t("i32", 1),                           # out2_ptr
        _t("i32", 1),                           # out3_ptr
    ]
    return args, {}
