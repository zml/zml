"""Dump Python Triton's IR (TTIR/TTGIR/LLIR/PTX) for every kernel in
`kernels_py/`.

For each `@triton.jit` function found, we build synthetic args matching its
signature and call `JITFunction.warmup(*args, grid=(1,))` with default passes —
no autotune, no operator wrappers. Run from a venv that has `triton` + `torch`
installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_KERNELS_DIR = _HERE / "kernels_py"

_FLOAT_HINTS = {"p", "scale", "eps", "alpha", "beta", "dropout"}

# Helper / inner kernels that are inlined at JIT time. Either they have no
# user-launchable top-level signature (`write_zeros_to_output`), or they're
# tiny pure helpers (`fast_exp`, `cdiv_fn`, `find_seq_idx`) that Triton's
# `ConvertTritonGPUToLLVM` pass refuses to lower as standalone kernels, or
# they're the inner bodies of `_ptr` wrapper kernels (we dump the wrappers).
_SKIP = {
    "write_zeros_to_output",
    "fast_exp",
    "cdiv_fn",
    "apply_softcap",
    "find_seq_idx",
    "kernel_unified_attention_2d",
    "kernel_unified_attention_3d",
    "reduce_segments",
    # `gdn.py` helpers (inlined into `fused_recurrent_gated_delta_rule_fwd_kernel_ptr`).
    "exp",
    "exp2",
    "fused_recurrent_gated_delta_rule_fwd_kernel",
}


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per-kernel arg overrides for kernels whose signatures the heuristic below
# can't satisfy (mixed dtypes, bool/dtype constexprs, etc.). Must match the
# Tensor shapes in `dump_via_xla.zig`'s per-kernel `args()`.
def _per_token_group_quant_fp8():
    import torch
    return ([
        torch.empty(64 * 1024, dtype=torch.bfloat16, device="cuda"),
        torch.empty(1, dtype=torch.int64, device="cuda"),
        torch.empty(1, dtype=torch.int64, device="cuda"),
        torch.empty(1, dtype=torch.int64, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        -57344.0, 57344.0, False, 128,
        torch.empty(64 * 1024, dtype=torch.float8_e5m2, device="cuda"),
        torch.empty(64 * 8, dtype=torch.bfloat16, device="cuda"),
    ], {})


def _fused_moe_kernel():
    import torch
    import triton.language as tl
    p = lambda: torch.empty(1, dtype=torch.int64, device="cuda")
    return ([
        torch.empty(128 * 1024, dtype=torch.bfloat16, device="cuda"),       # a_ptr
        torch.empty(8 * 1024 * 1024, dtype=torch.bfloat16, device="cuda"),  # b_ptr
        torch.empty(8 * 1024, dtype=torch.bfloat16, device="cuda"),         # b_bias_ptr
        torch.empty(1, dtype=torch.float32, device="cuda"),                 # a_scale_ptr
        torch.empty(1, dtype=torch.float32, device="cuda"),                 # b_scale_ptr
        torch.empty(256, dtype=torch.float32, device="cuda"),               # topk_weights_ptr
        torch.empty(256, dtype=torch.int32, device="cuda"),                 # sorted_token_ids_ptr
        torch.empty(32, dtype=torch.int32, device="cuda"),                  # expert_ids_ptr
        torch.empty(1, dtype=torch.int32, device="cuda"),                   # num_tokens_post_padded_ptr
        p(), p(), p(), p(),                                                 # N, K, EM, num_valid_tokens
        p(), 1, p(), 1, p(), p(), 1,                                        # strides am..cn
        p(), p(), p(), p(), p(), p(), p(),                                  # strides asm..bbn
        0, 0, False, 64, 64, 32, 4, 1, True, 2, tl.bfloat16,                # constexprs
        False, False, False, False, False,                                  # use_*, HAS_BIAS
        torch.empty(128 * 2 * 1024, dtype=torch.bfloat16, device="cuda"),   # c_ptr
    ], {})


def _moe_align_block_size_kernel():
    import torch
    return ([
        torch.empty(1024, dtype=torch.int32, device="cuda"),
        torch.empty(2048, dtype=torch.int32, device="cuda"),
        torch.empty(32, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(9, dtype=torch.int32, device="cuda"),
        64, 1024, 8, 8, 2048, 32, 64,
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.int32, device="cuda"),
    ], {})


def _count_and_sort_expert_tokens_kernel():
    import torch
    return ([
        torch.empty(1024, dtype=torch.int32, device="cuda"),
        torch.empty(2048, dtype=torch.int32, device="cuda"),
        torch.empty(9, dtype=torch.int32, device="cuda"),
        256, 1024, 8,
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.int32, device="cuda"),
    ], {})


# Unified attention `_ptr` overrides — the kernels mix `tl.constexpr` params
# with runtime `*_ptr` args throughout the signature, so we hand-write the
# full positional arg list (no kwargs). Constexpr values match the
# `withConfig(...)` call in `kernels_zig.zig` for byte-for-byte comparison.
_NUM_QUERY_HEADS = 32
_NUM_QUERIES_PER_KV = 4
_NUM_KV_HEADS = _NUM_QUERY_HEADS // _NUM_QUERIES_PER_KV  # 8
_BLOCK_SIZE = 16
_TILE_SIZE = 64
_HEAD_SIZE = 128
_HEAD_SIZE_PADDED = 128
_BLOCK_Q = 16
_BLOCK_M = 16
_NUM_SEGMENTS_PER_SEQ = 4
_NUM_SEQS = 1
_NUM_TOKENS = 64
_NUM_BLOCKS = 64

# fp8 magic numbers from `attention/triton_kernels.zig` defaults.
_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0


def _kernel_unified_attention_2d_ptr():
    import torch
    bf16 = lambda n: torch.empty(n, dtype=torch.bfloat16, device="cuda")
    f32 = lambda n: torch.empty(n, dtype=torch.float32, device="cuda")
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device="cuda")
    i64 = lambda n: torch.empty(n, dtype=torch.int64, device="cuda")
    return ([
        bf16(_NUM_TOKENS * _NUM_QUERY_HEADS * _HEAD_SIZE),  # query_ptr
        bf16(_NUM_BLOCKS * _NUM_KV_HEADS * _BLOCK_SIZE * _HEAD_SIZE),  # key_cache_ptr
        bf16(_NUM_BLOCKS * _NUM_KV_HEADS * _BLOCK_SIZE * _HEAD_SIZE),  # value_cache_ptr
        f32(1),                                  # sink_ptr
        i32(_NUM_SEQS * _NUM_BLOCKS),            # block_tables_ptr
        i32(_NUM_SEQS),                          # seq_lens_ptr
        f32(_NUM_QUERY_HEADS),                   # alibi_slopes_ptr
        f32(1),                                  # qq_bias_ptr
        f32(1),                                  # scale_ptr
        f32(1),                                  # k_scale_ptr
        f32(1),                                  # v_scale_ptr
        f32(1),                                  # out_scale_ptr
        f32(1),                                  # softcap_ptr
        _NUM_QUERY_HEADS,                        # num_query_heads (constexpr)
        _NUM_QUERIES_PER_KV,                     # num_queries_per_kv (constexpr)
        i64(1),                                  # block_table_stride_ptr
        i64(1),                                  # query_stride_0_ptr
        i64(1),                                  # query_stride_1_ptr
        i64(1),                                  # output_stride_0_ptr
        i64(1),                                  # output_stride_1_ptr
        i64(1),                                  # qq_bias_stride_0_ptr
        _BLOCK_SIZE,                             # BLOCK_SIZE
        _TILE_SIZE,                              # TILE_SIZE
        _HEAD_SIZE,                              # HEAD_SIZE
        _HEAD_SIZE_PADDED,                       # HEAD_SIZE_PADDED
        False,                                   # USE_ALIBI_SLOPES
        False,                                   # USE_QQ_BIAS
        False,                                   # USE_SOFTCAP
        False,                                   # USE_SINKS
        0,                                       # SLIDING_WINDOW
        i64(1),                                  # stride_k_cache_0_ptr
        i64(1),                                  # stride_k_cache_1_ptr
        i64(1),                                  # stride_k_cache_2_ptr
        1,                                       # stride_k_cache_3 (constexpr)
        i64(1),                                  # stride_v_cache_0_ptr
        i64(1),                                  # stride_v_cache_1_ptr
        i64(1),                                  # stride_v_cache_2_ptr
        1,                                       # stride_v_cache_3 (constexpr)
        i32(_NUM_SEQS + 1),                      # query_start_len_ptr
        _BLOCK_Q,                                # BLOCK_Q
        i32(1),                                  # num_seqs_ptr
        _BLOCK_M,                                # BLOCK_M
        False,                                   # USE_FP8
        _FP8_E4M3_MIN,                           # FP8_MIN
        _FP8_E4M3_MAX,                           # FP8_MAX
        False,                                   # ALL_DECODE
        bf16(_NUM_TOKENS * _NUM_QUERY_HEADS * _HEAD_SIZE),  # output_ptr
    ], {})


def _kernel_unified_attention_3d_ptr():
    import torch
    bf16 = lambda n: torch.empty(n, dtype=torch.bfloat16, device="cuda")
    f32 = lambda n: torch.empty(n, dtype=torch.float32, device="cuda")
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device="cuda")
    i64 = lambda n: torch.empty(n, dtype=torch.int64, device="cuda")
    segm_n = _NUM_TOKENS * _NUM_QUERY_HEADS * _NUM_SEGMENTS_PER_SEQ
    return ([
        bf16(_NUM_TOKENS * _NUM_QUERY_HEADS * _HEAD_SIZE),  # query_ptr
        bf16(_NUM_BLOCKS * _NUM_KV_HEADS * _BLOCK_SIZE * _HEAD_SIZE),  # key_cache_ptr
        bf16(_NUM_BLOCKS * _NUM_KV_HEADS * _BLOCK_SIZE * _HEAD_SIZE),  # value_cache_ptr
        f32(1),                                  # sink_ptr
        i32(_NUM_SEQS * _NUM_BLOCKS),            # block_tables_ptr
        i32(_NUM_SEQS),                          # seq_lens_ptr
        f32(_NUM_QUERY_HEADS),                   # alibi_slopes_ptr
        f32(1),                                  # qq_bias_ptr
        f32(1),                                  # scale_ptr
        f32(1),                                  # k_scale_ptr
        f32(1),                                  # v_scale_ptr
        f32(1),                                  # softcap_ptr
        _NUM_QUERY_HEADS,                        # num_query_heads (constexpr)
        _NUM_QUERIES_PER_KV,                     # num_queries_per_kv (constexpr)
        i64(1),                                  # block_table_stride_ptr
        i64(1),                                  # query_stride_0_ptr
        i64(1),                                  # query_stride_1_ptr
        i64(1),                                  # qq_bias_stride_0_ptr
        _BLOCK_SIZE,                             # BLOCK_SIZE
        _TILE_SIZE,                              # TILE_SIZE
        _HEAD_SIZE,                              # HEAD_SIZE
        _HEAD_SIZE_PADDED,                       # HEAD_SIZE_PADDED
        False,                                   # USE_ALIBI_SLOPES
        False,                                   # USE_QQ_BIAS
        False,                                   # USE_SOFTCAP
        False,                                   # USE_SINKS
        0,                                       # SLIDING_WINDOW
        i64(1),                                  # stride_k_cache_0_ptr
        i64(1),                                  # stride_k_cache_1_ptr
        i64(1),                                  # stride_k_cache_2_ptr
        1,                                       # stride_k_cache_3 (constexpr)
        i64(1),                                  # stride_v_cache_0_ptr
        i64(1),                                  # stride_v_cache_1_ptr
        i64(1),                                  # stride_v_cache_2_ptr
        1,                                       # stride_v_cache_3 (constexpr)
        i32(_NUM_SEQS + 1),                      # query_start_len_ptr
        _BLOCK_Q,                                # BLOCK_Q
        i32(1),                                  # num_seqs_ptr
        _BLOCK_M,                                # BLOCK_M
        _NUM_SEGMENTS_PER_SEQ,                   # NUM_SEGMENTS_PER_SEQ
        False,                                   # ALL_DECODE
        f32(segm_n * _HEAD_SIZE_PADDED),         # segm_output_ptr
        f32(segm_n),                             # segm_max_ptr
        f32(segm_n),                             # segm_expsum_ptr
    ], {})


def _reduce_segments_ptr():
    import torch
    bf16 = lambda n: torch.empty(n, dtype=torch.bfloat16, device="cuda")
    f32 = lambda n: torch.empty(n, dtype=torch.float32, device="cuda")
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device="cuda")
    i64 = lambda n: torch.empty(n, dtype=torch.int64, device="cuda")
    segm_n = _NUM_TOKENS * _NUM_QUERY_HEADS * _NUM_SEGMENTS_PER_SEQ
    return ([
        f32(segm_n * _HEAD_SIZE_PADDED),         # segm_output_ptr
        f32(segm_n),                             # segm_max_ptr
        f32(segm_n),                             # segm_expsum_ptr
        i32(_NUM_SEQS),                          # seq_lens_ptr
        i32(1),                                  # num_seqs_ptr
        _NUM_QUERY_HEADS,                        # num_query_heads (constexpr)
        f32(1),                                  # out_scale_inv_ptr
        i64(1),                                  # output_stride_0_ptr
        i64(1),                                  # output_stride_1_ptr
        i64(1),                                  # block_table_stride_ptr
        _TILE_SIZE,                              # TILE_SIZE
        _HEAD_SIZE,                              # HEAD_SIZE
        _HEAD_SIZE_PADDED,                       # HEAD_SIZE_PADDED
        i32(_NUM_SEQS + 1),                      # query_start_len_ptr
        _BLOCK_Q,                                # BLOCK_Q
        _NUM_SEGMENTS_PER_SEQ,                   # NUM_SEGMENTS_PER_SEQ
        False,                                   # USE_FP8
        _FP8_E4M3_MIN,                           # FP8_MIN
        _FP8_E4M3_MAX,                           # FP8_MAX
        bf16(_NUM_TOKENS * _NUM_QUERY_HEADS * _HEAD_SIZE),  # output_ptr
    ], {})


def _fused_recurrent_gated_delta_rule_fwd_kernel_ptr():
    """Mirrors monorepo's `compile_gated_delta_net_kernel` for the constexpr
    config used in qwen3_5 (USE_G=True, IS_VARLEN=True, …). Constexpr ints
    and tensor shapes match the Zig `withConfig(...)` call in `kernels_zig.zig`
    so the dumped TTIR is byte-comparable."""
    import torch
    import math
    num_tokens = 64
    num_qk_heads = 4
    num_v_heads = 16
    key_dim = 32
    value_dim = 64
    num_sequences = 2
    bk = 32
    bv = 8
    qk_shape = (1, num_tokens, num_qk_heads, key_dim)
    v_shape = (1, num_tokens, num_v_heads, value_dim)
    gbeta_shape = (1, num_tokens, num_v_heads)
    state_shape = (num_sequences, num_v_heads, key_dim, value_dim)
    cu_seqlens_shape = (num_sequences + 1,)
    return ([
        torch.empty(qk_shape, dtype=torch.bfloat16, device="cuda"),       # q_ptr
        torch.empty(qk_shape, dtype=torch.bfloat16, device="cuda"),       # k_ptr
        torch.empty(v_shape, dtype=torch.bfloat16, device="cuda"),        # v_ptr
        torch.empty(gbeta_shape, dtype=torch.float32, device="cuda"),     # g_ptr
        torch.empty(gbeta_shape, dtype=torch.bfloat16, device="cuda"),    # beta_ptr
        torch.empty(state_shape, dtype=torch.float32, device="cuda"),     # h0_ptr
        torch.empty(cu_seqlens_shape, dtype=torch.int32, device="cuda"),  # cu_seqlens_ptr
        torch.empty(v_shape, dtype=torch.bfloat16, device="cuda"),        # o_ptr
        torch.empty(state_shape, dtype=torch.float32, device="cuda"),     # ht_ptr
        None,                                # gk (constexpr)
        None,                                # gv (constexpr)
        1.0 / math.sqrt(key_dim),            # scale (constexpr float)
        num_tokens,                          # T (constexpr)
        num_qk_heads,                        # H (constexpr)
        num_v_heads,                         # HV (constexpr)
        key_dim,                             # K (constexpr)
        value_dim,                           # V (constexpr)
        bk,                                  # BK (constexpr)
        bv,                                  # BV (constexpr)
        True,                                # USE_G
        False,                               # USE_GK
        False,                               # USE_GV
        True,                                # USE_QK_L2NORM_IN_KERNEL
        True,                                # IS_BETA_HEADWISE
        True,                                # USE_INITIAL_STATE
        True,                                # STORE_FINAL_STATE
        False,                               # USE_EXP2
        False,                               # TRANSPOSE_STATE
        True,                                # IS_VARLEN
    ], {})


_OVERRIDES = {
    "per_token_group_quant_fp8": _per_token_group_quant_fp8,
    "fused_moe_kernel": _fused_moe_kernel,
    "moe_align_block_size_kernel": _moe_align_block_size_kernel,
    "count_and_sort_expert_tokens_kernel": _count_and_sort_expert_tokens_kernel,
    "kernel_unified_attention_2d_ptr": _kernel_unified_attention_2d_ptr,
    "kernel_unified_attention_3d_ptr": _kernel_unified_attention_3d_ptr,
    "reduce_segments_ptr": _reduce_segments_ptr,
    "fused_recurrent_gated_delta_rule_fwd_kernel_ptr": _fused_recurrent_gated_delta_rule_fwd_kernel_ptr,
}


# =============================================================================
# Unified-attention fuzzer — variant configurations matching the entries in
# `kernels_zig.zig`. Each variant overrides a subset of the constexpr fields
# of `_kernel_unified_attention_*_ptr` (or `_reduce_segments_ptr`); every
# other arg falls back to the base override. The dump driver runs
# `JITFunction.warmup` once per (kernel, variant) pair and writes the result
# to `<kernel>__<label>.<stage>` so it pairs with Zig's variant TTIR by stem.
#
# Field set drawn from monorepo/llmd's `select2dConfig` / `select3dConfig`
# call space (head_dim ∈ {64,128,256}, num_queries_per_kv ∈ {1,4,8},
# block_m ∈ {16,64,128}, sliding_window ∈ {0,4096}, all_decode ∈ {true,false}).
# =============================================================================
_UNIFIED_ATTENTION_2D_VARIANTS = {
    "dec_h128_g4": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "BLOCK_Q": 4, "BLOCK_M": 16, "ALL_DECODE": True},
    "pre_h128_g4": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 64, "BLOCK_Q": 4, "BLOCK_M": 16, "ALL_DECODE": False},
    "pre_h128_g8": {"NUM_QUERY_HEADS": 64, "NUM_QUERIES_PER_KV": 8, "TILE_SIZE": 64, "BLOCK_Q": 2, "BLOCK_M": 16, "ALL_DECODE": False},
    "pre_h128_g4_long": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 64, "BLOCK_Q": 32, "BLOCK_M": 128, "ALL_DECODE": False},
    "dec_h256_swa": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "HEAD_SIZE": 256, "HEAD_SIZE_PADDED": 256, "SLIDING_WINDOW": 4096, "BLOCK_Q": 16, "BLOCK_M": 64, "ALL_DECODE": True},
    "pre_h256_swa": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "HEAD_SIZE": 256, "HEAD_SIZE_PADDED": 256, "SLIDING_WINDOW": 4096, "BLOCK_Q": 4, "BLOCK_M": 16, "ALL_DECODE": False},
    "dec_h64_g1": {"NUM_QUERY_HEADS": 16, "NUM_QUERIES_PER_KV": 1, "TILE_SIZE": 16, "HEAD_SIZE": 64, "HEAD_SIZE_PADDED": 64, "BLOCK_Q": 16, "BLOCK_M": 16, "ALL_DECODE": True},
}

_UNIFIED_ATTENTION_3D_VARIANTS = {
    "pre_h128_g4_seg16": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "BLOCK_Q": 4, "BLOCK_M": 16, "NUM_SEGMENTS_PER_SEQ": 16},
    "pre_h128_g8_seg32": {"NUM_QUERY_HEADS": 64, "NUM_QUERIES_PER_KV": 8, "TILE_SIZE": 16, "BLOCK_Q": 2, "BLOCK_M": 16, "NUM_SEGMENTS_PER_SEQ": 32},
    "dec_h128_g4_seg64": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "BLOCK_Q": 4, "BLOCK_M": 16, "NUM_SEGMENTS_PER_SEQ": 64, "ALL_DECODE": True},
    "pre_h256_seg16": {"NUM_QUERIES_PER_KV": 4, "TILE_SIZE": 16, "HEAD_SIZE": 256, "HEAD_SIZE_PADDED": 256, "BLOCK_Q": 4, "BLOCK_M": 16, "NUM_SEGMENTS_PER_SEQ": 16},
}

_REDUCE_VARIANTS = {
    "h128_qh32_seg16": {"NUM_QUERY_HEADS": 32, "TILE_SIZE": 16, "BLOCK_Q": 4, "NUM_SEGMENTS_PER_SEQ": 16},
    "h128_qh64_seg32": {"NUM_QUERY_HEADS": 64, "TILE_SIZE": 16, "BLOCK_Q": 2, "NUM_SEGMENTS_PER_SEQ": 32},
    "h256_qh32_seg16": {"NUM_QUERY_HEADS": 32, "TILE_SIZE": 16, "HEAD_SIZE": 256, "HEAD_SIZE_PADDED": 256, "BLOCK_Q": 4, "NUM_SEGMENTS_PER_SEQ": 16},
}


# Position of each constexpr in the matching `_kernel_*` override's args
# tuple. Mirrors the order the override builds the tuple in. Used by
# `_apply_overrides` to splice variant values into the base args.
_UAT_2D_CONSTEXPR_INDEX = {
    "NUM_QUERY_HEADS": 13, "NUM_QUERIES_PER_KV": 14,
    "BLOCK_SIZE": 21, "TILE_SIZE": 22, "HEAD_SIZE": 23, "HEAD_SIZE_PADDED": 24,
    "USE_ALIBI_SLOPES": 25, "USE_QQ_BIAS": 26, "USE_SOFTCAP": 27, "USE_SINKS": 28,
    "SLIDING_WINDOW": 29,
    "BLOCK_Q": 39, "BLOCK_M": 41,
    "USE_FP8": 42, "ALL_DECODE": 45,
}

_UAT_3D_CONSTEXPR_INDEX = {
    "NUM_QUERY_HEADS": 12, "NUM_QUERIES_PER_KV": 13,
    "BLOCK_SIZE": 18, "TILE_SIZE": 19, "HEAD_SIZE": 20, "HEAD_SIZE_PADDED": 21,
    "USE_ALIBI_SLOPES": 22, "USE_QQ_BIAS": 23, "USE_SOFTCAP": 24, "USE_SINKS": 25,
    "SLIDING_WINDOW": 26,
    "BLOCK_Q": 36, "BLOCK_M": 38,
    "NUM_SEGMENTS_PER_SEQ": 39, "ALL_DECODE": 40,
}

_REDUCE_CONSTEXPR_INDEX = {
    "NUM_QUERY_HEADS": 5,
    "TILE_SIZE": 10, "HEAD_SIZE": 11, "HEAD_SIZE_PADDED": 12,
    "BLOCK_Q": 14, "NUM_SEGMENTS_PER_SEQ": 15,
    "USE_FP8": 16,
}


_UNIFIED_ATTENTION_VARIANT_TABLE = [
    ("kernel_unified_attention_2d_ptr", _kernel_unified_attention_2d_ptr,
     _UNIFIED_ATTENTION_2D_VARIANTS, _UAT_2D_CONSTEXPR_INDEX),
    ("kernel_unified_attention_3d_ptr", _kernel_unified_attention_3d_ptr,
     _UNIFIED_ATTENTION_3D_VARIANTS, _UAT_3D_CONSTEXPR_INDEX),
    ("reduce_segments_ptr", _reduce_segments_ptr,
     _REDUCE_VARIANTS, _REDUCE_CONSTEXPR_INDEX),
]


def _apply_overrides(base_pos, base_kw, overrides, index_map):
    """Splice `overrides` (dict of constexpr name → value) into a base
    override's positional args at the indices in `index_map`. Both the
    Python overrides and the Zig variants pin everything as positional, so
    we only touch positions named in `index_map`."""
    pos = list(base_pos)
    for name, val in overrides.items():
        idx = index_map.get(name)
        if idx is None:
            raise KeyError(f"unknown constexpr {name!r} for variant override")
        pos[idx] = val
    return pos, dict(base_kw)


def _enumerate_variant_jobs(jit_fns_by_name, kernel_filter):
    """Yield `(out_name, jit_fn, pos_args, kw_args)` for every fuzzer
    variant whose base kernel was loaded. Skips variants whose base kernel
    is missing (e.g. `kernels_dir` was filtered to a different file)."""
    for base_name, base_factory, variants, index_map in _UNIFIED_ATTENTION_VARIANT_TABLE:
        if base_name not in jit_fns_by_name:
            continue
        for label, overrides in variants.items():
            out_name = f"{base_name}__{label}"
            if kernel_filter and kernel_filter != out_name and kernel_filter != base_name:
                continue
            base_pos, base_kw = base_factory()
            pos, kw = _apply_overrides(base_pos, base_kw, overrides, index_map)
            yield out_name, jit_fns_by_name[base_name], pos, kw


def _make_synthetic_args(jit_fn) -> Tuple[List[Any], Dict[str, Any]]:
    """Heuristic args for simple kernels: `*_ptr` → f32 tensor, `tl.constexpr`
    → 1024 kwarg, names in `_FLOAT_HINTS` → 0.5, anything else → 1024 int."""
    name = getattr(jit_fn.fn, "__name__", "")
    if name in _OVERRIDES:
        return _OVERRIDES[name]()

    import torch

    sig = inspect.signature(jit_fn.fn)
    args, kwargs = [], {}
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is not inspect.Parameter.empty and getattr(ann, "__name__", "") == "constexpr":
            kwargs[pname] = 1024
        elif pname.endswith("_ptr"):
            args.append(torch.empty(1024, dtype=torch.float32, device="cuda"))
        elif pname in _FLOAT_HINTS:
            args.append(0.5)
        else:
            args.append(1024)
    return args, kwargs


def _dump_compiled(compiled, out_dir: Path, name: str) -> int:
    asm = getattr(compiled, "asm", None) or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        if stage not in asm:
            continue
        blob = asm[stage]
        path = out_dir / f"{name}.{stage}"
        if isinstance(blob, bytes):
            path.write_bytes(blob)
        else:
            path.write_text(str(blob))
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=str(_HERE / "py_ir"))
    p.add_argument("--kernels-dir", default=str(_KERNELS_DIR))
    p.add_argument("--kernel", default="", help="Only dump kernels matching this name.")
    args = p.parse_args(argv)

    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")

    try:
        from triton.runtime import JITFunction
    except Exception as e:
        print(f"dump_python: cannot import triton — {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    kdir = Path(args.kernels_dir).resolve()
    files = sorted(p for p in kdir.glob("*.py") if not p.name.startswith("_"))
    if not files:
        print(f"dump_python: no .py files in {kdir}", file=sys.stderr)
        return 1

    total = 0
    jit_fns_by_name: Dict[str, "JITFunction"] = {}
    for src in files:
        mod = _load_module(src)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if not isinstance(obj, JITFunction):
                continue
            kname = getattr(obj.fn, "__name__", "")
            if kname in _SKIP:
                continue
            jit_fns_by_name[kname] = obj
            if args.kernel and args.kernel != attr and args.kernel != kname:
                # Even if the user filtered out this kernel by name, keep it
                # in the table so a variant filter like
                # `kernel_unified_attention_2d_ptr__dec_h128_g4` can still
                # find its base kernel below.
                continue
            pos, kw = _make_synthetic_args(obj)
            try:
                compiled = obj.warmup(*pos, **kw, grid=(1,))
            except Exception as e:
                print(f"dump_python: warmup failed for {attr}: {e}", file=sys.stderr)
                continue
            total += _dump_compiled(compiled, out_dir, attr)

    # Unified-attention fuzzer variants — runs the same `_ptr` kernels with
    # constexpr overrides drawn from monorepo's `select{2,3}dConfig` call
    # space, writes each result to `<base>__<label>.<stage>` for byte-by-byte
    # comparison against `kernels_zig.zig`'s `variantOf(...)` entries.
    for out_name, jit_fn, pos, kw in _enumerate_variant_jobs(jit_fns_by_name, args.kernel):
        try:
            compiled = jit_fn.warmup(*pos, **kw, grid=(1,))
        except Exception as e:
            print(f"dump_python: variant warmup failed for {out_name}: {e}", file=sys.stderr)
            continue
        total += _dump_compiled(compiled, out_dir, out_name)

    if total == 0:
        print("dump_python: no IR captured", file=sys.stderr)
        return 1
    print(f"dump_python: wrote {total} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
