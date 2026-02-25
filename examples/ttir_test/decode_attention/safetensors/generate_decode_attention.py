import os

import torch
from safetensors.torch import save_file


torch.manual_seed(0)

batch_size = 8
num_heads = 32
num_kv_heads = 8
head_size = 128
page_size = 16
num_pages = 128
max_num_pages_per_req = 64
num_kv_splits = 8
sm_scale = 1.0 / (head_size**0.5)


def main() -> None:
    q = torch.randn(batch_size, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k_buffer = torch.randn(num_pages, page_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v_buffer = torch.randn(num_pages, page_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")

    req_to_tokens = torch.randint(0, num_pages, (batch_size, max_num_pages_per_req), dtype=torch.int32, device="cuda")
    req_to_tokens[:, 0] = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seqlen = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    out_path = os.path.join(os.path.dirname(__file__), "decode_attention_inputs.safetensors")
    save_file(
        {
            "q": q.cpu(),
            "k_buffer": k_buffer.cpu(),
            "v_buffer": v_buffer.cpu(),
            "req_to_tokens": req_to_tokens.cpu(),
            "b_seqlen": b_seqlen.cpu(),
        },
        out_path,
    )

    print(f"Wrote: {out_path}")
    print(f"q shape: {q.shape}, dtype: {q.dtype}")
    print(f"k_buffer shape: {k_buffer.shape}, dtype: {k_buffer.dtype}")
    print(f"v_buffer shape: {v_buffer.shape}, dtype: {v_buffer.dtype}")
    print(f"req_to_tokens shape: {req_to_tokens.shape}, dtype: {req_to_tokens.dtype}")
    print(f"b_seqlen shape: {b_seqlen.shape}, dtype: {b_seqlen.dtype}")
    print(f"num_kv_splits: {num_kv_splits}")
    print(f"sm_scale: {sm_scale}")
    print(f"page_size: {page_size}")


if __name__ == "__main__":
    main()
