import json
import importlib.util
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

def _resolve_wrapper_file(file_name: str) -> Path:
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []

    for base in (script_path.parent, *script_path.parents):
        candidates.append(base / "wrapped_kernels" / file_name)
        candidates.append(base / "examples" / "ttir_sandbox" / "wrapped_kernels" / file_name)

    for env_key in ("RUNFILES_DIR", "RUNFILES_DIRECTORY"):
        runfiles = os.environ.get(env_key)
        if runfiles:
            runfiles_root = Path(runfiles)
            candidates.append(runfiles_root / "_main" / "examples" / "ttir_sandbox" / "wrapped_kernels" / file_name)
            candidates.append(runfiles_root / "examples" / "ttir_sandbox" / "wrapped_kernels" / file_name)

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    raise FileNotFoundError(f"Unable to locate {file_name}; tried: {', '.join(sorted(seen))}")


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec: {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_wrap_2d = _load_module("wrap_2d_unified_attention", _resolve_wrapper_file("wrap_2d_unified_attention.py"))
_wrap_3d = _load_module("wrap_3d_unified_attention", _resolve_wrapper_file("wrap_3d_unified_attention.py"))
_wrap_decode = _load_module("wrap_decode_attention", _resolve_wrapper_file("wrap_decode_attention.py"))
_wrap_prefill = _load_module("wrap_prefill_attention", _resolve_wrapper_file("wrap_prefill_attention.py"))

compile_ttir_2d_unified_attention_kernel = _wrap_2d.compile_ttir_2d_unified_attention_kernel
compile_ttir_3d_unified_attention_kernels = _wrap_3d.compile_ttir_3d_unified_attention_kernels
compile_ttir_decode_attention_stage1_kernel = _wrap_decode.compile_ttir_decode_attention_stage1_kernel
compile_ttir_prefill_attention_kernel = _wrap_prefill.compile_ttir_prefill_attention_kernel


@triton.jit
def matmul_fixed_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M_C: tl.constexpr,
    N_C: tl.constexpr,
    K_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    A = tl.make_block_ptr(
        base=A_ptr,
        shape=(M_C, K_C),
        strides=(K_C, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B = tl.make_block_ptr(
        base=B_ptr,
        shape=(K_C, N_C),
        strides=(N_C, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    C = tl.make_block_ptr(
        base=C_ptr,
        shape=(M_C, N_C),
        strides=(N_C, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_C, BLOCK_K):
        a = tl.load(A, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        A = tl.advance(A, (0, BLOCK_K))
        B = tl.advance(B, (BLOCK_K, 0))
    tl.store(C, acc, boundary_check=(0, 1))


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


def compile_hello_world_matmul_ttir(args_json: str) -> str:
    params = _params(args_json)
    dev = _device()

    m = _int(params, "M", 256)
    n = _int(params, "N", 256)
    k = _int(params, "K", 256)
    block_m = _int(params, "BLOCK_M", 128)
    block_n = _int(params, "BLOCK_N", 128)
    block_k = _int(params, "BLOCK_K", 32)
    num_warps = _int(params, "num_warps", 8)

    a = torch.zeros((m, k), dtype=torch.float32, device=dev)
    b = torch.zeros((k, n), dtype=torch.float32, device=dev)
    c = torch.zeros((m, n), dtype=torch.float32, device=dev)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    compiled_kernel = matmul_fixed_kernel[grid](
        a,
        b,
        c,
        M_C=m,
        N_C=n,
        K_C=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return str(compiled_kernel.asm["ttir"])
 
