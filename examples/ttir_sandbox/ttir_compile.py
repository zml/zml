import json
import sys
from pathlib import Path

import torch

_WRAPPED_KERNELS_DIR = Path(__file__).resolve().parent / "wrapped_kernels"
if str(_WRAPPED_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_WRAPPED_KERNELS_DIR))

from wrap_2d_unified_attention import compile_ttir_2d_unified_attention_kernel
from wrap_3d_unified_attention import compile_ttir_3d_unified_attention_kernels
from wrap_decode_attention import compile_ttir_decode_attention_stage1_kernel
from wrap_prefill_attention import compile_ttir_prefill_attention_kernel


def _params(args_json: str) -> dict:
    return json.loads(args_json) if args_json else {}


def _int(params: dict, key: str, default: int) -> int:
    return int(params.get(key, default))


def _float(params: dict, key: str, default: float) -> float:
    return float(params.get(key, default))


def _bool(params: dict, key: str, default: bool) -> bool:
    return bool(params.get(key, default))


def _device() -> torch.device:
    return torch.device("cuda")


def _common_attn_shapes(params: dict) -> tuple[int, int, int, int, int, int]:
    num_seqs = _int(params, "num_seqs", 2)
    seq_len_q = _int(params, "seq_len_q", 64)
    seq_len_k = _int(params, "seq_len_k", 64)
    num_query_heads = _int(params, "num_query_heads", 8)
    num_kv_heads = _int(params, "num_kv_heads", 8)
    head_size = _int(params, "head_size", 128)
    return num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size


def compile_2d_unified_attention_ttir(args_json: str) -> str:
    params = _params(args_json)
    dev = _device()

    num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size = _common_attn_shapes(params)
    block_size = _int(params, "block_size", 16)
    max_blocks_per_seq = _int(params, "max_blocks_per_seq", max(1, seq_len_k // block_size))

    total_q = num_seqs * seq_len_q
    num_blocks = num_seqs * max_blocks_per_seq

    q = torch.zeros((total_q, num_query_heads, head_size), dtype=torch.float16, device=dev)
    k = torch.zeros((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    v = torch.zeros((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    out = torch.zeros_like(q)

    cu_seqlens_q = torch.arange(0, total_q + 1, seq_len_q, dtype=torch.int32, device=dev)
    seqused_k = torch.full((num_seqs,), seq_len_k, dtype=torch.int32, device=dev)
    block_table = torch.zeros((num_seqs, max_blocks_per_seq), dtype=torch.int32, device=dev)

    return compile_ttir_2d_unified_attention_kernel(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seq_len_q,
        seqused_k=seqused_k,
        max_seqlen_k=seq_len_k,
        softmax_scale=_float(params, "softmax_scale", head_size ** -0.5),
        causal=_bool(params, "causal", True),
        window_size=(_int(params, "window_size", -1), _int(params, "window_size", -1)),
        block_table=block_table,
        softcap=_float(params, "softcap", 0.0),
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )


def _compile_ttir_3d_pair(params: dict) -> tuple[str, str]:
    dev = _device()

    num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size = _common_attn_shapes(params)
    block_size = _int(params, "block_size", 16)
    max_blocks_per_seq = _int(params, "max_blocks_per_seq", max(1, seq_len_k // block_size))
    num_par_softmax_segments = _int(params, "num_par_softmax_segments", 2)

    total_q = num_seqs * seq_len_q
    num_blocks = num_seqs * max_blocks_per_seq

    q = torch.zeros((total_q, num_query_heads, head_size), dtype=torch.float16, device=dev)
    k = torch.zeros((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    v = torch.zeros((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    out = torch.zeros_like(q)

    cu_seqlens_q = torch.arange(0, total_q + 1, seq_len_q, dtype=torch.int32, device=dev)
    seqused_k = torch.full((num_seqs,), seq_len_k, dtype=torch.int32, device=dev)
    block_table = torch.zeros((num_seqs, max_blocks_per_seq), dtype=torch.int32, device=dev)

    softmax_segm_output = torch.zeros(
        (total_q, num_query_heads, num_par_softmax_segments, head_size),
        dtype=torch.float32,
        device=dev,
    )
    softmax_segm_max = torch.zeros((total_q, num_query_heads, num_par_softmax_segments), dtype=torch.float32, device=dev)
    softmax_segm_expsum = torch.zeros((total_q, num_query_heads, num_par_softmax_segments), dtype=torch.float32, device=dev)

    return compile_ttir_3d_unified_attention_kernels(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seq_len_q,
        seqused_k=seqused_k,
        max_seqlen_k=seq_len_k,
        softmax_scale=_float(params, "softmax_scale", head_size ** -0.5),
        causal=_bool(params, "causal", True),
        window_size=(_int(params, "window_size", -1), _int(params, "window_size", -1)),
        block_table=block_table,
        softcap=_float(params, "softcap", 0.0),
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )


def compile_3d_unified_attention_ttir(args_json: str) -> str:
    return _compile_ttir_3d_pair(_params(args_json))[0]


def compile_3d_reduce_segments_ttir(args_json: str) -> str:
    return _compile_ttir_3d_pair(_params(args_json))[1]


def compile_decode_attention_stage1_ttir(args_json: str) -> str:
    params = _params(args_json)
    dev = _device()

    batch = _int(params, "batch", 2)
    num_query_heads = _int(params, "num_query_heads", 8)
    num_kv_heads = _int(params, "num_kv_heads", 8)
    head_size = _int(params, "head_size", 128)
    page_size = _int(params, "page_size", 16)
    num_pages = _int(params, "num_pages", 16)
    num_kv_splits = _int(params, "num_kv_splits", 2)
    max_ctx = _int(params, "max_ctx", page_size * num_pages)
    seq_len = _int(params, "seq_len", min(max_ctx, 64))

    q = torch.zeros((batch, num_query_heads, head_size), dtype=torch.float16, device=dev)
    k_buffer = torch.zeros((num_pages, page_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    v_buffer = torch.zeros((num_pages, page_size, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    att_out = torch.zeros((batch, num_query_heads, num_kv_splits, head_size), dtype=torch.float32, device=dev)
    req_to_tokens = torch.zeros((batch, max_ctx), dtype=torch.int32, device=dev)
    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)

    return compile_ttir_decode_attention_stage1_kernel(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        att_out=att_out,
        req_to_tokens=req_to_tokens,
        b_seqlen=b_seqlen,
        num_kv_splits=num_kv_splits,
        sm_scale=_float(params, "softmax_scale", head_size ** -0.5),
        page_size=page_size,
        logit_cap=_float(params, "logit_cap", 0.0),
    )


def compile_prefill_attention_ttir(args_json: str) -> str:
    params = _params(args_json)
    dev = _device()

    batch = _int(params, "batch", 2)
    seq_len = _int(params, "seq_len", 64)
    num_query_heads = _int(params, "num_query_heads", 8)
    num_kv_heads = _int(params, "num_kv_heads", 8)
    head_size = _int(params, "head_size", 128)

    total_tokens = batch * seq_len
    q = torch.zeros((total_tokens, num_query_heads, head_size), dtype=torch.float16, device=dev)
    k = torch.zeros((total_tokens, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    v = torch.zeros((total_tokens, num_kv_heads, head_size), dtype=torch.float16, device=dev)
    out = torch.zeros((total_tokens, num_query_heads, head_size), dtype=torch.float16, device=dev)

    b_start_loc = torch.arange(0, total_tokens, seq_len, dtype=torch.int32, device=dev)
    b_seq_len = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)

    return compile_ttir_prefill_attention_kernel(
        q=q,
        k=k,
        v=v,
        out=out,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=seq_len,
        is_causal=_bool(params, "causal", True),
        softmax_scale=_float(params, "softmax_scale", head_size ** -0.5),
        sliding_window_q=_int(params, "sliding_window_q", 0),
        sliding_window_k=_int(params, "sliding_window_k", 0),
    )
 