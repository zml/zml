import os
import torch
from safetensors.torch import save_file


torch.manual_seed(0)

# Same constants as 2d_unified_attention_random_init.py
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
    # Random inputs (2D path) matching 2d_unified_attention_random_init.py
    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    # Use valid block indices to avoid out-of-range accesses in kernels.
    block_tables = torch.zeros((batch_size, max_seq_len // block_size), dtype=torch.int32, device="cuda")
    block_tables[:, 0] = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    context_seq_lens = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")
    start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

    out_path = os.path.join(os.path.dirname(__file__), "2d_unified_attention_inputs.safetensors")
    save_file(
        {
            "query": query.cpu(),
            "k": k.cpu(),
            "v": v.cpu(),
            "key_cache": key_cache.cpu(),
            "value_cache": value_cache.cpu(),
            "block_tables": block_tables.cpu(),
            "context_seq_lens": context_seq_lens.cpu(),
            "start_loc": start_loc.cpu(),
        },
        out_path,
    )

    print(f"Wrote: {out_path}")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"k shape: {k.shape}, dtype: {k.dtype}")
    print(f"v shape: {v.shape}, dtype: {v.dtype}")
    print(f"key_cache shape: {key_cache.shape}, dtype: {key_cache.dtype}")
    print(f"value_cache shape: {value_cache.shape}, dtype: {value_cache.dtype}")
    print(f"block_tables shape: {block_tables.shape}, dtype: {block_tables.dtype}")
    print(f"context_seq_lens shape: {context_seq_lens.shape}, dtype: {context_seq_lens.dtype}")
    print(f"start_loc shape: {start_loc.shape}, dtype: {start_loc.dtype}")
    print(f"max_input_len: {max_input_len}")
    print(f"k_scale: {k_scale}")
    print(f"v_scale: {v_scale}")
    print(f"alibi_slopes: {alibi_slopes}")
    print(f"scale: {scale}")


if __name__ == "__main__":
    main()
