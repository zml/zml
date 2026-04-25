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


_OVERRIDES = {
    "per_token_group_quant_fp8": _per_token_group_quant_fp8,
    "fused_moe_kernel": _fused_moe_kernel,
    "moe_align_block_size_kernel": _moe_align_block_size_kernel,
    "count_and_sort_expert_tokens_kernel": _count_and_sort_expert_tokens_kernel,
    "kernel_unified_attention_2d_ptr": _kernel_unified_attention_2d_ptr,
    "kernel_unified_attention_3d_ptr": _kernel_unified_attention_3d_ptr,
    "reduce_segments_ptr": _reduce_segments_ptr,
}


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
    for src in files:
        mod = _load_module(src)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if not isinstance(obj, JITFunction):
                continue
            kname = getattr(obj.fn, "__name__", "")
            if kname in _SKIP:
                continue
            if args.kernel and args.kernel != attr and args.kernel != kname:
                continue
            pos, kw = _make_synthetic_args(obj)
            try:
                compiled = obj.warmup(*pos, **kw, grid=(1,))
            except Exception as e:
                print(f"dump_python: warmup failed for {attr}: {e}", file=sys.stderr)
                continue
            total += _dump_compiled(compiled, out_dir, attr)

    if total == 0:
        print("dump_python: no IR captured", file=sys.stderr)
        return 1
    print(f"dump_python: wrote {total} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
