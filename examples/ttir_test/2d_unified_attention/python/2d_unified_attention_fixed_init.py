import os

import torch
from wrap_2d_unified_attention import wrapped_unified_attention
from safetensors.torch import load_file

torch.set_printoptions(threshold=torch.inf)

token_count = 8
batch_size = 8
num_heads = 32
num_kv_heads = 8
head_size = 128
head_size_padded = 1 << (head_size - 1).bit_length()
block_size = 16
num_blocks = 4096
max_input_len = 1
max_seq_len = 8192
scale = 0.08838834765
alibi_slopes = None
k_scale = 1.0
v_scale = 1.0


def main() -> None:
    inputs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "safetensors",
        "2d_unified_attention_inputs.safetensors",
    )
    inputs = load_file(inputs_path)

    query = inputs["query"].to(device="cuda", dtype=torch.bfloat16)
    key_cache = inputs["key_cache"].to(device="cuda", dtype=torch.bfloat16)
    value_cache = inputs["value_cache"].to(device="cuda", dtype=torch.bfloat16)
    block_tables = inputs["block_tables"].to(device="cuda", dtype=torch.int32)
    context_seq_lens = inputs["context_seq_lens"].to(device="cuda", dtype=torch.int32)
    start_loc = inputs["start_loc"].to(device="cuda", dtype=torch.int32)

    o = torch.empty_like(query)

    print("Parameters for unified_attention:")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"key_cache shape: {key_cache.shape}, dtype: {key_cache.dtype}")
    print(f"value_cache shape: {value_cache.shape}, dtype: {value_cache.dtype}")
    print(f"block_tables shape: {block_tables.shape}, dtype: {block_tables.dtype}")
    print(f"start_loc shape: {start_loc.shape}, dtype: {start_loc.dtype}")
    print(
        f"context_seq_lens shape: {context_seq_lens.shape}, dtype: {context_seq_lens.dtype}"
    )
    print(f"max_input_len: {max_input_len}")
    print(f"k_scale: {k_scale}")
    print(f"v_scale: {v_scale}")
    print(f"alibi_slopes: {alibi_slopes}")
    print(f"scale: {scale}")

    wrapped_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=o,
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
        k_descale=k_scale,
        v_descale=v_scale,
    )

    sample = o[:, 0, 0].float().cpu()[:8]
    print("Output o[:,0,0] first 8:", " ".join(f"{v.item():.5f}" for v in sample))


if __name__ == "__main__":
    os.environ["SHOULD_RUN_3D"] = "0"
    os.environ["TRITON_BACKEND_DEBUG"] = "1"
    # os.environ["SHOULD_LOG"] = "1"
    main()
