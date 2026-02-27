import json

import torch

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
