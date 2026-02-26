import os
from pathlib import Path

import torch

from wrap_3d_unified_attention import run_3d_unified_attention_kernels


def main() -> None:
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"

    token_count = 8
    batch_size = 8
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    head_size_padded = 1 << (head_size - 1).bit_length()
    block_size = 16
    num_blocks = 4096
    max_input_len = 1
    max_seq_len = block_size
    num_par_softmax_segments = 8
    scale = 0.08838834765

    query = torch.zeros(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(query)
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.zeros((batch_size, max_seq_len // block_size), dtype=torch.int32, device="cuda")
    context_seq_lens = torch.full((batch_size,), max_input_len, dtype=torch.int32, device="cuda")
    start_loc = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    softmax_segm_output = torch.empty(
        (token_count, num_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
        device="cuda",
    )
    softmax_segm_max = torch.empty(
        (token_count, num_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device="cuda",
    )
    softmax_segm_expsum = torch.empty(
        (token_count, num_heads, num_par_softmax_segments),
        dtype=torch.float32,
        device="cuda",
    )

    compiled_kernel, compiled_reduce = run_3d_unified_attention_kernels(
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
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    example_dir = Path(__file__).resolve().parents[1]
    out_main = example_dir / "3d_unified_attention_kernel.ttir"
    out_reduce = example_dir / "reduce_segments_kernel.ttir"
    out_main.write_text(compiled_kernel.asm["ttir"])
    out_reduce.write_text(compiled_reduce.asm["ttir"])
    print(f"Wrote: {out_main}")
    print(f"Wrote: {out_reduce}")


if __name__ == "__main__":
    main()
