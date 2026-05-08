"""Python reference for `kernel_unified_attention_2d_ptr`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from triton_helpers import dtype_str, fake as _t
from unified_attention import kernel_unified_attention_2d_ptr  # type: ignore

# Re-export so the harness runner's `getattr(mod, args.kernel_fn)` resolves it.
__all__ = ["kernel_unified_attention_2d_ptr", "build_args"]


# Synthetic shape budget — chosen to span every sweep config (max
# num_query_heads=64, head_size_padded=256). XLA never launches the
# kernel; only dtype + rank matter for the lowering pipeline.
_NUM_TOKENS = 64
_NUM_BLOCKS = 64
_NUM_SEQS = 1


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    q_dtype = dtype_str(cfg.get("q_dtype"), "bf16")
    kv_dtype = dtype_str(cfg.get("kv_dtype"), "bf16")
    o_dtype = dtype_str(cfg.get("o_dtype"), "bf16")

    num_query_heads = int(cfg.get("num_query_heads", 32))
    num_queries_per_kv = int(cfg.get("num_queries_per_kv", 4))
    num_kv_heads = max(1, num_query_heads // num_queries_per_kv)

    block_size = int(cfg.get("block_size", 16))
    tile_size = int(cfg.get("tile_size", 64))
    head_size = int(cfg.get("head_size", 128))
    head_size_padded = int(cfg.get("head_size_padded", head_size))

    use_alibi_slopes = bool(cfg.get("use_alibi_slopes", False))
    use_qq_bias = bool(cfg.get("use_qq_bias", False))
    use_softcap = bool(cfg.get("use_softcap", False))
    use_sinks = bool(cfg.get("use_sinks", False))
    sliding_window = int(cfg.get("sliding_window", 0))

    stride_k_cache_3 = int(cfg.get("stride_k_cache_3", 1))
    stride_v_cache_3 = int(cfg.get("stride_v_cache_3", 1))

    block_q = int(cfg.get("block_q", 16))
    block_m = int(cfg.get("block_m", 16))
    use_fp8 = bool(cfg.get("use_fp8", False))
    fp8_min = float(cfg.get("fp8_min", -448.0))
    fp8_max = float(cfg.get("fp8_max", 448.0))
    all_decode = bool(cfg.get("all_decode", False))

    num_tokens = _NUM_TOKENS
    num_blocks = _NUM_BLOCKS

    args = [
        # query / key / value caches.
        _t(q_dtype, num_tokens * num_query_heads * head_size),
        _t(kv_dtype, num_blocks * num_kv_heads * block_size * head_size),
        _t(kv_dtype, num_blocks * num_kv_heads * block_size * head_size),
        # sink, block_tables, seq_lens, alibi, qq_bias.
        _t("fp32", 1),
        _t("i32", _NUM_SEQS * num_blocks),
        _t("i32", _NUM_SEQS),
        _t("fp32", num_query_heads),
        _t("fp32", 1),
        # scale, k_scale, v_scale, out_scale, softcap.
        _t("fp32", 1), _t("fp32", 1), _t("fp32", 1), _t("fp32", 1), _t("fp32", 1),
        # constexprs interleaved with stride pointers (matches Python signature).
        num_query_heads,           # NUM_QUERY_HEADS
        num_queries_per_kv,        # NUM_QUERIES_PER_KV
        _t("i64", 1),              # block_table_stride_ptr
        _t("i64", 1), _t("i64", 1),  # query_stride_0/1
        _t("i64", 1), _t("i64", 1),  # output_stride_0/1
        _t("i64", 1),              # qq_bias_stride_0
        block_size,                # BLOCK_SIZE
        tile_size,                 # TILE_SIZE
        head_size,                 # HEAD_SIZE
        head_size_padded,          # HEAD_SIZE_PADDED
        use_alibi_slopes,
        use_qq_bias,
        use_softcap,
        use_sinks,
        sliding_window,
        _t("i64", 1), _t("i64", 1), _t("i64", 1),  # stride_k_cache_0..2
        stride_k_cache_3,          # constexpr
        _t("i64", 1), _t("i64", 1), _t("i64", 1),  # stride_v_cache_0..2
        stride_v_cache_3,          # constexpr
        _t("i32", _NUM_SEQS + 1),  # query_start_len_ptr
        block_q,                   # BLOCK_Q
        _t("i32", 1),              # num_seqs_ptr
        block_m,                   # BLOCK_M
        use_fp8,
        fp8_min,
        fp8_max,
        all_decode,
        # output_ptr (last positional).
        _t(o_dtype, num_tokens * num_query_heads * head_size),
    ]
    return args, {}
