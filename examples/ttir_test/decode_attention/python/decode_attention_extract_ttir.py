import os
from pathlib import Path

import torch

from wrap_decode_attention import run_decode_attention_stage1_kernel


def main() -> None:
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"

    batch_size = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    head_size_padded = 1 << (head_size - 1).bit_length()
    num_kv_splits = 8
    page_size = 16
    num_pages = 128
    max_num_pages_per_req = 64
    sm_scale = 1.0 / (head_size**0.5)

    q = torch.zeros(batch_size, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k_buffer = torch.zeros(num_pages, page_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v_buffer = torch.zeros(num_pages, page_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    req_to_tokens = torch.zeros((batch_size, max_num_pages_per_req), dtype=torch.int32, device="cuda")
    req_to_tokens[:, 0] = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seqlen = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")
    att_out = torch.empty(batch_size, num_heads, num_kv_splits, head_size_padded + 1, dtype=torch.float32, device="cuda")

    compiled_kernel = run_decode_attention_stage1_kernel(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        att_out=att_out,
        req_to_tokens=req_to_tokens,
        b_seqlen=b_seqlen,
        num_kv_splits=num_kv_splits,
        sm_scale=sm_scale,
        page_size=page_size,
        logit_cap=0.0,
    )

    out_path = Path(__file__).resolve().parents[1] / "decode_attention.ttir"
    out_path.write_text(compiled_kernel.asm["ttir"])
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
