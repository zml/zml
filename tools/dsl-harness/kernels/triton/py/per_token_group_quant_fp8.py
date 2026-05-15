"""Python reference for `per_token_group_quant_fp8` (MoE fp8 per-token-group quant)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from triton_helpers import dtype_str, fake as _t
from moe import per_token_group_quant_fp8  # type: ignore

# Re-export so the harness runner's `getattr(mod, args.kernel_fn)` resolves it.
__all__ = ["per_token_group_quant_fp8", "build_args"]


# Synthetic shape budget — XLA never launches; only dtype + rank matter.
_NUM_TOKENS = 64
_NUM_COLS = 1024


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    input_dtype = dtype_str(cfg.get("input_dtype"), "bf16")
    output_dtype = dtype_str(cfg.get("output_dtype"), "fp8e5")
    scale_dtype = dtype_str(cfg.get("scale_dtype"), "bf16")
    block = int(cfg.get("block", 128))
    fp8_min = float(cfg.get("fp8_min", -57344.0))
    fp8_max = float(cfg.get("fp8_max", 57344.0))
    use_ue8m0 = bool(cfg.get("use_ue8m0", False))

    groups_per_row = max(1, _NUM_COLS // block)
    num_groups = _NUM_TOKENS * groups_per_row

    args = [
        _t(input_dtype, _NUM_TOKENS * _NUM_COLS),  # y_ptr
        _t("i64", 1),                              # group_size_ptr
        _t("i64", 1),                              # y_num_columns_ptr
        _t("i64", 1),                              # y_row_stride_ptr
        _t("fp32", 1),                             # eps_ptr
        fp8_min,                                   # fp8_min (constexpr)
        fp8_max,                                   # fp8_max (constexpr)
        use_ue8m0,                                 # use_ue8m0 (constexpr)
        block,                                     # BLOCK (constexpr)
        _t(output_dtype, num_groups * block),      # y_q_ptr
        _t(scale_dtype, num_groups),               # y_s_ptr
    ]
    return args, {}
