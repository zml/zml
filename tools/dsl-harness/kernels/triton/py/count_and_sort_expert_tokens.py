"""Python reference for `count_and_sort_expert_tokens_kernel`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from triton_helpers import fake as _t
from moe import count_and_sort_expert_tokens_kernel  # type: ignore

__all__ = ["count_and_sort_expert_tokens_kernel", "build_args"]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    numel = int(cfg.get("numel", 1024))
    num_experts = int(cfg.get("num_experts", 8))
    sort_block_size = int(cfg.get("sort_block_size", 256))
    max_num_tokens_padded = int(cfg.get("max_num_tokens_padded", 2048))

    args = [
        _t("i32", numel),                       # topk_ids_ptr
        _t("i32", max_num_tokens_padded),       # sorted_token_ids_ptr
        _t("i32", num_experts + 1),             # cumsum_ptr
        sort_block_size,                        # BLOCK_SIZE (constexpr)
        numel,                                  # NUMEL (constexpr)
        num_experts,                            # NUM_EXPERTS (constexpr)
        _t("i32", 1),                           # out0_ptr
        _t("i32", 1),                           # out1_ptr
    ]
    return args, {}
