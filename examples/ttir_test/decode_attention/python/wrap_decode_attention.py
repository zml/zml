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

from triton_decode_attention import _fwd_kernel_stage1, is_hip_


@triton.jit
def wrapped_fwd_kernel_stage1(
    q_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    sm_scale_ptr,
    req_to_tokens_ptr,
    b_seqlen_ptr,
    stride_req_to_tokens_b_ptr,
    stride_qbs_ptr,
    stride_qh_ptr,
    stride_buf_kbs_ptr,
    stride_buf_kh_ptr,
    stride_buf_vbs_ptr,
    stride_buf_vh_ptr,
    stride_mid_ob_ptr,
    stride_mid_oh_ptr,
    stride_mid_os_ptr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    out_ptr,
):
    sm_scale = tl.load(sm_scale_ptr)
    stride_req_to_tokens_b = tl.load(stride_req_to_tokens_b_ptr)
    stride_qbs = tl.load(stride_qbs_ptr)
    stride_qh = tl.load(stride_qh_ptr)
    stride_buf_kbs = tl.load(stride_buf_kbs_ptr)
    stride_buf_kh = tl.load(stride_buf_kh_ptr)
    stride_buf_vbs = tl.load(stride_buf_vbs_ptr)
    stride_buf_vh = tl.load(stride_buf_vh_ptr)
    stride_mid_ob = tl.load(stride_mid_ob_ptr)
    stride_mid_oh = tl.load(stride_mid_oh_ptr)
    stride_mid_os = tl.load(stride_mid_os_ptr)

    _fwd_kernel_stage1(
        q_ptr,
        k_buffer_ptr,
        v_buffer_ptr,
        sm_scale,
        req_to_tokens_ptr,
        b_seqlen_ptr,
        out_ptr,
        stride_req_to_tokens_b,
        stride_qbs,
        stride_qh,
        stride_buf_kbs,
        stride_buf_kh,
        stride_buf_vbs,
        stride_buf_vh,
        stride_mid_ob,
        stride_mid_oh,
        stride_mid_os,
        kv_group_num,
        BLOCK_DMODEL,
        BLOCK_DV,
        BLOCK_N,
        NUM_KV_SPLITS,
        PAGE_SIZE,
        logit_cap,
        Lk,
        Lv,
    )


def scalar_ptr(val: int | float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor([val], dtype=dtype, device=device)


def run_decode_attention_stage1_kernel(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    att_out: torch.Tensor,
    req_to_tokens: torch.Tensor,
    b_seqlen: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
):
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, num_kv_splits)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    block_n = 64 if not is_hip_ else 8
    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    device = q.device
    compiled_kernel = wrapped_fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        scalar_ptr(sm_scale, torch.float32, device),
        req_to_tokens,
        b_seqlen,
        scalar_ptr(req_to_tokens.stride(0), torch.int64, device),
        scalar_ptr(q.stride(0), torch.int64, device),
        scalar_ptr(q.stride(1), torch.int64, device),
        scalar_ptr(k_buffer.stride(-3), torch.int64, device),
        scalar_ptr(k_buffer.stride(-2), torch.int64, device),
        scalar_ptr(v_buffer.stride(-3), torch.int64, device),
        scalar_ptr(v_buffer.stride(-2), torch.int64, device),
        scalar_ptr(att_out.stride(0), torch.int64, device),
        scalar_ptr(att_out.stride(1), torch.int64, device),
        scalar_ptr(att_out.stride(2), torch.int64, device),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_N=block_n,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        out_ptr=att_out,
    )

    print(f"kernel_wrapped_decode_stage1: {compiled_kernel.src.constants}")
    print(f"kernel_wrapped_decode_stage1: {compiled_kernel.asm['ttir']}")
    return compiled_kernel
