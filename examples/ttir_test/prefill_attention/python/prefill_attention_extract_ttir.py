import os
from pathlib import Path

import torch

from wrap_prefill_attention import run_prefill_attention_kernel


def main() -> None:
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"

    token_count = 8
    batch_size = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128

    query = torch.zeros(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    key = torch.zeros(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(query)
    b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")

    compiled_kernel = run_prefill_attention_kernel(
        q=query,
        k=key,
        v=value,
        out=out,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=1,
        is_causal=True,
    )

    out_path = Path(__file__).resolve().parents[1] / "prefill_attention.ttir"
    out_path.write_text(compiled_kernel.asm["ttir"])
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
