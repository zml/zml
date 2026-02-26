import os
from pathlib import Path

import torch

from wrap_2d_unified_attention import run_2d_unified_attention_kernel


def main() -> None:
    os.environ["SHOULD_RUN_3D"] = "0"
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"

    token_count = 8
    batch_size = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 4096
    max_input_len = 1
    max_seq_len = 8192
    scale = 0.08838834765

    query = torch.zeros(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(query)
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.zeros((batch_size, max_seq_len // block_size), dtype=torch.int32, device="cuda")
    context_seq_lens = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")
    start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

    compiled_kernel = run_2d_unified_attention_kernel(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=start_loc,
        max_seqlen_q=max_input_len,
        seqused_k=context_seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=1.0,
        v_descale=1.0,
    )

    out_path = Path(__file__).resolve().parents[1] / "2d_unified_attention_kernel.ttir"
    out_path.write_text(compiled_kernel.asm["ttir"])
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
