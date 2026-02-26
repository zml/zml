# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

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

from triton_unified_attention import _get_tile_size, kernel_unified_attention_2d

float8_info = torch.finfo(torch.float8_e4m3fn)  # potentially not adapted


def _env_flag_enabled(name: str) -> bool:
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


@triton.jit
def wrapped_kernel_unified_attention_2d(
    output_ptr,
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
    out_scale_ptr,
    softcap_ptr,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride_ptr,
    query_stride_0_ptr,
    query_stride_1_ptr,
    output_stride_0_ptr,
    output_stride_1_ptr,
    qq_bias_stride_0_ptr,
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0_ptr,
    stride_k_cache_1_ptr,
    stride_k_cache_2_ptr,
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0_ptr,
    stride_v_cache_1_ptr,
    stride_v_cache_2_ptr,
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,  # int
    num_seqs_ptr,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # Load all scalars from pointers
    scale = tl.load(scale_ptr)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    out_scale = tl.load(out_scale_ptr)
    softcap = tl.load(softcap_ptr)
    block_table_stride = tl.load(block_table_stride_ptr)
    query_stride_0 = tl.load(query_stride_0_ptr)
    query_stride_1 = tl.load(query_stride_1_ptr)
    output_stride_0 = tl.load(output_stride_0_ptr)
    output_stride_1 = tl.load(output_stride_1_ptr)
    qq_bias_stride_0 = tl.load(qq_bias_stride_0_ptr)
    stride_k_cache_0 = tl.load(stride_k_cache_0_ptr)
    stride_k_cache_1 = tl.load(stride_k_cache_1_ptr)
    stride_k_cache_2 = tl.load(stride_k_cache_2_ptr)
    stride_v_cache_0 = tl.load(stride_v_cache_0_ptr)
    stride_v_cache_1 = tl.load(stride_v_cache_1_ptr)
    stride_v_cache_2 = tl.load(stride_v_cache_2_ptr)
    num_seqs = tl.load(num_seqs_ptr)

    # Call the original kernel with loaded values
    kernel_unified_attention_2d(
        output_ptr,
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
        out_scale,
        softcap,
        num_query_heads,
        num_queries_per_kv,
        block_table_stride,
        query_stride_0,
        query_stride_1,
        output_stride_0,
        output_stride_1,
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
        USE_MM_PREFIX,
        MAX_MM_RANGES,
        mm_prefix_range_ptr,
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
        USE_FP8,
        FP8_MIN,
        FP8_MAX,
    )


# Helper to create scalar pointers
def scalar_ptr(val, dtype=torch.float32):
    return torch.tensor([val], dtype=dtype).cuda()


def run_2d_unified_attention_kernel(
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
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"
    should_log = _env_flag_enabled("SHOULD_LOG")

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

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile size for 2D (prefill) kernel.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    compiled_kernel = wrapped_kernel_unified_attention_2d[
        (
            total_num_q_blocks,
            num_kv_heads,
        )
    ](
        out,  # output_ptr
        q,  # query_ptr
        k,  # key_cache_ptr
        v,  # value_cache_ptr
        sinks,  # sink_ptr
        block_table,  # block_tables_ptr
        seqused_k,  # seq_lens_ptr
        alibi_slopes,  # alibi_slopes_ptr
        qq_bias,  # qq_bias_ptr
        scalar_ptr(softmax_scale, dtype=torch.float32),  # scale_ptr
        scalar_ptr(
            k_descale if k_descale is not None else 1.0, dtype=torch.float32
        ),  # k_scale_ptr
        scalar_ptr(
            v_descale if v_descale is not None else 1.0, dtype=torch.float32
        ),  # v_scale_ptr
        scalar_ptr(
            1 / output_scale if output_scale is not None else 1.0, dtype=torch.float32
        ),  # out_scale_ptr
        scalar_ptr(softcap, dtype=torch.float32),  # softcap_ptr
        num_query_heads,  # num_query_heads (constexpr)
        num_queries_per_kv,  # num_queries_per_kv (constexpr)
        scalar_ptr(block_table.stride(0), dtype=torch.int64),  # block_table_stride_ptr
        scalar_ptr(q.stride(0), dtype=torch.int64),  # query_stride_0_ptr
        scalar_ptr(q.stride(1), dtype=torch.int64),  # query_stride_1_ptr
        scalar_ptr(out.stride(0), dtype=torch.int64),  # output_stride_0_ptr
        scalar_ptr(out.stride(1), dtype=torch.int64),  # output_stride_1_ptr
        scalar_ptr(
            qq_bias.stride(0) if use_qq_bias else 0, dtype=torch.int64
        ),  # qq_bias_stride_0_ptr
        block_size,  # BLOCK_SIZE (constexpr)
        TILE_SIZE_PREFILL,  # TILE_SIZE (constexpr)
        head_size,  # HEAD_SIZE (constexpr)
        triton.next_power_of_2(head_size),  # HEAD_SIZE_PADDED (constexpr)
        int(use_alibi_slopes),  # USE_ALIBI_SLOPES (constexpr)
        int(use_alibi_sqrt),  # USE_ALIBI_SQRT (constexpr)
        int(use_qq_bias),  # USE_QQ_BIAS (constexpr)
        int(softcap > 0),  # USE_SOFTCAP (constexpr)
        int(sinks is not None),  # USE_SINKS (constexpr)
        1 + window_size[0],  # SLIDING_WINDOW (constexpr)
        int(use_mm_prefix),  # USE_MM_PREFIX (constexpr)
        max_mm_ranges,  # MAX_MM_RANGES (constexpr)
        mm_prefix_range,  # mm_prefix_range_ptr
        scalar_ptr(k.stride(0), dtype=torch.int64),  # stride_k_cache_0_ptr
        scalar_ptr(k.stride(1), dtype=torch.int64),  # stride_k_cache_1_ptr
        scalar_ptr(k.stride(2), dtype=torch.int64),  # stride_k_cache_2_ptr
        k.stride(3),  # stride_k_cache_3 (constexpr)
        scalar_ptr(v.stride(0), dtype=torch.int64),  # stride_v_cache_0_ptr
        scalar_ptr(v.stride(1), dtype=torch.int64),  # stride_v_cache_1_ptr
        scalar_ptr(v.stride(2), dtype=torch.int64),  # stride_v_cache_2_ptr
        v.stride(3),  # stride_v_cache_3 (constexpr)
        cu_seqlens_q,  # query_start_len_ptr
        BLOCK_Q,  # BLOCK_Q (constexpr)
        scalar_ptr(num_seqs, dtype=torch.int32),  # num_seqs_ptr
        BLOCK_M,  # BLOCK_M (constexpr)
        int(output_scale is not None),  # USE_FP8 (constexpr)
        -448.0,  # FP8_MIN (constexpr, replace if needed)
        448.0,  # FP8_MAX (constexpr, replace if needed)
    )

    if should_log:
        print(f"kernel_wrapped_2d: {compiled_kernel.src.constants}")
        print(f"kernel_wrapped_2d: {compiled_kernel.asm['ttir']}")
