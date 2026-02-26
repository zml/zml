# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_ROOT = Path(__file__).resolve().parents[2]
_VLLM_KERNELS_DIR = _ROOT / "vllm_kernels"
if str(_VLLM_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_VLLM_KERNELS_DIR))

from triton_prefill_attention import RCP_LN2, _fwd_kernel, get_block_size


@triton.jit
def wrapped_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    sm_scale_ptr,
    b_start_loc_ptr,
    b_seq_len_ptr,
    stride_qbs_ptr,
    stride_qh_ptr,
    stride_kbs_ptr,
    stride_kh_ptr,
    stride_vbs_ptr,
    stride_vh_ptr,
    stride_obs_ptr,
    stride_oh_ptr,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_Q: tl.constexpr,
    SLIDING_WINDOW_K: tl.constexpr,
    Lk: tl.constexpr,
    out_ptr,
):
    sm_scale = tl.load(sm_scale_ptr)
    stride_qbs = tl.load(stride_qbs_ptr)
    stride_qh = tl.load(stride_qh_ptr)
    stride_kbs = tl.load(stride_kbs_ptr)
    stride_kh = tl.load(stride_kh_ptr)
    stride_vbs = tl.load(stride_vbs_ptr)
    stride_vh = tl.load(stride_vh_ptr)
    stride_obs = tl.load(stride_obs_ptr)
    stride_oh = tl.load(stride_oh_ptr)

    _fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        sm_scale,
        b_start_loc_ptr,
        b_seq_len_ptr,
        out_ptr,
        stride_qbs,
        stride_qh,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        stride_obs,
        stride_oh,
        kv_group_num,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        IS_CAUSAL,
        SLIDING_WINDOW_Q,
        SLIDING_WINDOW_K,
        Lk,
    )


def scalar_ptr(
    val: int | float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor([val], dtype=dtype, device=device)


def run_prefill_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
):
    BLOCK = get_block_size(q.dtype)

    Lq, Lk = q.shape[-1], k.shape[-1]

    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0

    device = q.device
    compiled_kernel = wrapped_fwd_kernel[grid](
        q,
        k,
        v,
        scalar_ptr(sm_scale, torch.float32, device),
        b_start_loc,
        b_seq_len,
        scalar_ptr(q.stride(0), torch.int64, device),
        scalar_ptr(q.stride(1), torch.int64, device),
        scalar_ptr(k.stride(0), torch.int64, device),
        scalar_ptr(k.stride(1), torch.int64, device),
        scalar_ptr(v.stride(0), torch.int64, device),
        scalar_ptr(v.stride(1), torch.int64, device),
        scalar_ptr(out.stride(0), torch.int64, device),
        scalar_ptr(out.stride(1), torch.int64, device),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=sliding_window_q,
        SLIDING_WINDOW_K=sliding_window_k,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
        out_ptr=out,
    )

    print(f"kernel_wrapped_prefill: {compiled_kernel.src.constants}")
    print(f"kernel_wrapped_prefill: {compiled_kernel.asm['ttir']}")
    return compiled_kernel
