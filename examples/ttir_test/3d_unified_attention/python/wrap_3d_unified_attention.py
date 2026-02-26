# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_ROOT = Path(__file__).resolve().parents[2]
_VLLM_KERNELS_DIR = _ROOT / "vllm_kernels"
if str(_VLLM_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_VLLM_KERNELS_DIR))

from triton_unified_attention import _get_tile_size, kernel_unified_attention_3d, reduce_segments

float8_info = torch.finfo(torch.float8_e4m3fn)  # potentially not adapted


@triton.jit
def wrapped_kernel_unified_attention_3d(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    softcap_ptr,
    block_table_stride_ptr,
    query_stride_0_ptr,
    query_stride_1_ptr,
    qq_bias_stride_0_ptr,
    stride_k_cache_0_ptr,
    stride_k_cache_1_ptr,
    stride_k_cache_2_ptr,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0_ptr,
    stride_v_cache_1_ptr,
    stride_v_cache_2_ptr,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    num_seqs_ptr,
    mm_prefix_range_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    segm_max_ptr,
    segm_expsum_ptr,
    segm_output_ptr,
):
    scale = tl.load(scale_ptr)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    softcap = tl.load(softcap_ptr)
    block_table_stride = tl.load(block_table_stride_ptr)
    query_stride_0 = tl.load(query_stride_0_ptr)
    query_stride_1 = tl.load(query_stride_1_ptr)
    qq_bias_stride_0 = tl.load(qq_bias_stride_0_ptr)
    stride_k_cache_0 = tl.load(stride_k_cache_0_ptr)
    stride_k_cache_1 = tl.load(stride_k_cache_1_ptr)
    stride_k_cache_2 = tl.load(stride_k_cache_2_ptr)
    stride_v_cache_0 = tl.load(stride_v_cache_0_ptr)
    stride_v_cache_1 = tl.load(stride_v_cache_1_ptr)
    stride_v_cache_2 = tl.load(stride_v_cache_2_ptr)
    num_seqs = tl.load(num_seqs_ptr)

    kernel_unified_attention_3d(
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        sink_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        alibi_slopes_ptr,
        qq_bias_ptr,
        scale,
        k_scale,
        v_scale,
        softcap,
        num_query_heads,
        num_queries_per_kv,
        block_table_stride,
        query_stride_0,
        query_stride_1,
        qq_bias_stride_0,
        BLOCK_SIZE,
        TILE_SIZE,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES,
        USE_ALIBI_SQRT,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        SLIDING_WINDOW,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
        query_start_len_ptr,
        BLOCK_Q,
        num_seqs,
        BLOCK_M,
        NUM_SEGMENTS_PER_SEQ,
        USE_MM_PREFIX,
        MAX_MM_RANGES,
        mm_prefix_range_ptr,
    )


@triton.jit
def wrapped_reduce_segments(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs_ptr,
    out_scale_inv_ptr,
    output_stride_0_ptr,
    output_stride_1_ptr,
    block_table_stride_ptr,
    query_start_len_ptr,
    num_query_heads: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    output_ptr,
):
    out_scale_inv = tl.load(out_scale_inv_ptr)
    output_stride_0 = tl.load(output_stride_0_ptr)
    output_stride_1 = tl.load(output_stride_1_ptr)
    block_table_stride = tl.load(block_table_stride_ptr)
    num_seqs = tl.load(num_seqs_ptr)

    reduce_segments(
        output_ptr,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        seq_lens_ptr,
        num_seqs,
        num_query_heads,
        out_scale_inv,
        output_stride_0,
        output_stride_1,
        block_table_stride,
        TILE_SIZE,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
        query_start_len_ptr,
        BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ,
        USE_FP8,
        FP8_MIN,
        FP8_MAX,
    )


def scalar_ptr(val: int | float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor([val], dtype=dtype, device=device)


def run_3d_unified_attention_kernels(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"
    should_log = os.environ.get("SHOULD_LOG", None) is not None

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )

    device = q.device
    compiled_kernel = wrapped_kernel_unified_attention_3d[
        (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
    ](
        q,  # query_ptr
        k,  # key_cache_ptr
        v,  # value_cache_ptr
        sinks,  # sink_ptr
        block_table,  # block_tables_ptr
        seqused_k,  # seq_lens_ptr
        alibi_slopes,  # alibi_slopes_ptr
        qq_bias,  # qq_bias_ptr
        scalar_ptr(softmax_scale, torch.float32, device),  # scale_ptr
        scalar_ptr(
            k_descale if k_descale is not None else 1.0, torch.float32, device
        ),  # k_scale_ptr
        scalar_ptr(
            v_descale if v_descale is not None else 1.0, torch.float32, device
        ),  # v_scale_ptr
        scalar_ptr(softcap, torch.float32, device),  # softcap_ptr
        scalar_ptr(block_table.stride(0), torch.int64, device),  # block_table_stride_ptr
        scalar_ptr(q.stride(0), torch.int64, device),  # query_stride_0_ptr
        scalar_ptr(q.stride(1), torch.int64, device),  # query_stride_1_ptr
        scalar_ptr(
            qq_bias.stride(0) if use_qq_bias else 0, torch.int64, device
        ),  # qq_bias_stride_0_ptr
        scalar_ptr(k.stride(0), torch.int64, device),  # stride_k_cache_0_ptr
        scalar_ptr(k.stride(1), torch.int64, device),  # stride_k_cache_1_ptr
        scalar_ptr(k.stride(2), torch.int64, device),  # stride_k_cache_2_ptr
        k.stride(3),  # stride_k_cache_3 (constexpr)
        scalar_ptr(v.stride(0), torch.int64, device),  # stride_v_cache_0_ptr
        scalar_ptr(v.stride(1), torch.int64, device),  # stride_v_cache_1_ptr
        scalar_ptr(v.stride(2), torch.int64, device),  # stride_v_cache_2_ptr
        v.stride(3),  # stride_v_cache_3 (constexpr)
        cu_seqlens_q,  # query_start_len_ptr
        scalar_ptr(num_seqs, torch.int32, device),  # num_seqs_ptr
        mm_prefix_range,  # mm_prefix_range_ptr
        num_query_heads,  # num_query_heads (constexpr)
        num_queries_per_kv,  # num_queries_per_kv (constexpr)
        block_size,  # BLOCK_SIZE (constexpr)
        TILE_SIZE_DECODE,  # TILE_SIZE (constexpr)
        head_size,  # HEAD_SIZE (constexpr)
        triton.next_power_of_2(head_size),  # HEAD_SIZE_PADDED (constexpr)
        int(use_alibi_slopes),  # USE_ALIBI_SLOPES (constexpr)
        int(use_alibi_sqrt),  # USE_ALIBI_SQRT (constexpr)
        int(use_qq_bias),  # USE_QQ_BIAS (constexpr)
        int(softcap > 0),  # USE_SOFTCAP (constexpr)
        int(sinks is not None),  # USE_SINKS (constexpr)
        1 + window_size[0],  # SLIDING_WINDOW (constexpr)
        BLOCK_Q,  # BLOCK_Q (constexpr)
        BLOCK_M,  # BLOCK_M (constexpr)
        num_par_softmax_segments,  # NUM_SEGMENTS_PER_SEQ (constexpr)
        int(use_mm_prefix),  # USE_MM_PREFIX (constexpr)
        max_mm_ranges,  # MAX_MM_RANGES (constexpr)
        softmax_segm_max,  # segm_max_ptr
        softmax_segm_expsum,  # segm_expsum_ptr
        softmax_segm_output,  # segm_output_ptr
    )

    if should_log:
        print(f"kernel_wrapped_3d: {compiled_kernel.src.constants}")
        print(f"kernel_wrapped_3d: {compiled_kernel.asm['ttir']}")

    compiled_reduce = wrapped_reduce_segments[(q.shape[0], num_query_heads)](
        softmax_segm_output,  # segm_output_ptr
        softmax_segm_max,  # segm_max_ptr
        softmax_segm_expsum,  # segm_expsum_ptr
        seqused_k,  # seq_lens_ptr
        scalar_ptr(num_seqs, torch.int32, device),  # num_seqs_ptr
        scalar_ptr(
            1 / output_scale if output_scale is not None else 1.0,
            torch.float32,
            device,
        ),  # out_scale_inv_ptr
        scalar_ptr(out.stride(0), torch.int64, device),  # output_stride_0_ptr
        scalar_ptr(out.stride(1), torch.int64, device),  # output_stride_1_ptr
        scalar_ptr(block_table.stride(0), torch.int64, device),  # block_table_stride_ptr
        cu_seqlens_q,  # query_start_len_ptr
        num_query_heads,  # num_query_heads (constexpr)
        TILE_SIZE_DECODE,  # TILE_SIZE (constexpr)
        head_size,  # HEAD_SIZE (constexpr)
        triton.next_power_of_2(head_size),  # HEAD_SIZE_PADDED (constexpr)
        BLOCK_Q,  # BLOCK_Q (constexpr)
        num_par_softmax_segments,  # NUM_SEGMENTS_PER_SEQ (constexpr)
        int(output_scale is not None),  # USE_FP8 (constexpr)
        -448.0,  # FP8_MIN (constexpr, replace if needed)
        448.0,  # FP8_MAX (constexpr, replace if needed)
        out,  # output_ptr
    )

    if should_log:
        print(f"kernel_wrapped_reduce: {compiled_reduce.src.constants}")
        print(f"kernel_wrapped_reduce: {compiled_reduce.asm['ttir']}")
