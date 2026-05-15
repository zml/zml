"""Python reference for `fused_moe_kernel` (bf16 / no-quant / no-bias path)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import triton.language as tl

from triton_helpers import dtype_str, fake as _t
from moe import fused_moe_kernel, write_zeros_to_output  # type: ignore  # noqa: F401

__all__ = ["fused_moe_kernel", "build_args"]

_TL_DTYPES = {"bf16": tl.bfloat16, "fp16": tl.float16, "fp32": tl.float32}

# Synthetic shape budget — XLA never launches; sizes are placeholders.
_M = 256
_N = 1024
_K = 1024
_E = 8


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    a_dtype = dtype_str(cfg.get("a_dtype"), "bf16")
    b_dtype = dtype_str(cfg.get("b_dtype"), "bf16")
    c_dtype = dtype_str(cfg.get("c_dtype"), "bf16")
    a_scale_dtype = dtype_str(cfg.get("a_scale_dtype"), "fp32")
    b_scale_dtype = dtype_str(cfg.get("b_scale_dtype"), "fp32")
    b_bias_dtype = dtype_str(cfg.get("b_bias_dtype"), c_dtype)
    topk_weights_dtype = dtype_str(cfg.get("topk_weights_dtype"), "fp32")

    block_size_m = int(cfg.get("block_size_m", 64))
    block_size_n = int(cfg.get("block_size_n", 64))
    block_size_k = int(cfg.get("block_size_k", 32))
    group_size_m = int(cfg.get("group_size_m", 4))
    top_k = int(cfg.get("top_k", 2))
    naive_block_assignment = bool(cfg.get("naive_block_assignment", False))
    mul_routed_weight = bool(cfg.get("mul_routed_weight", True))
    compute_type = _TL_DTYPES[dtype_str(cfg.get("compute_type"), "bf16")]
    use_fp8_w8a8 = bool(cfg.get("use_fp8_w8a8", False))
    use_int8_w8a8 = bool(cfg.get("use_int8_w8a8", False))
    use_int8_w8a16 = bool(cfg.get("use_int8_w8a16", False))
    per_channel_quant = bool(cfg.get("per_channel_quant", False))
    has_bias = bool(cfg.get("has_bias", False))

    def p() -> Any:
        return _t("i64", 1)

    args = [
        _t(a_dtype, _M * _K),                       # a_ptr
        _t(b_dtype, _E * _N * _K),                  # b_ptr
        _t(b_bias_dtype, _E * _N),                  # b_bias_ptr
        _t(a_scale_dtype, 1),                       # a_scale_ptr
        _t(b_scale_dtype, 1),                       # b_scale_ptr
        _t(topk_weights_dtype, _M),                 # topk_weights_ptr
        _t("i32", _M),                              # sorted_token_ids_ptr
        _t("i32", _E * 4),                          # expert_ids_ptr
        _t("i32", 1),                               # num_tokens_post_padded_ptr
        p(), p(), p(), p(),                         # N_ptr, K_ptr, EM_ptr, num_valid_tokens_ptr
        p(), 1, p(), 1, p(), p(), 1,                # stride_am_ptr, stride_ak=1, stride_be_ptr, stride_bk=1, stride_bn_ptr, stride_cm_ptr, stride_cn=1
        p(), p(), p(), p(), p(), p(), p(),          # stride_asm/ask/bse/bsk/bsn/bbe/bbn ptrs
        0, 0, naive_block_assignment,               # group_n, group_k, naive_block_assignment (constexpr)
        block_size_m, block_size_n, block_size_k,   # BLOCK_SIZE_M/N/K (constexpr)
        group_size_m, 1, mul_routed_weight, top_k,  # GROUP_SIZE_M, SPLIT_K=1, MUL_ROUTED_WEIGHT, top_k (constexpr)
        compute_type,                               # compute_type (constexpr)
        use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, per_channel_quant, has_bias,  # (constexpr)
        _t(c_dtype, _M * _N),                       # c_ptr
    ]
    return args, {}
