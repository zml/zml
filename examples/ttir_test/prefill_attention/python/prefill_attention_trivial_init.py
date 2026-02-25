import os

import torch

from wrap_prefill_attention import run_prefill_attention_kernel


torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(0)

os.environ["TRITON_BACKEND_DEBUG"] = "1"
os.environ["SHOULD_LOG"] = "1"

batch_size = 8
token_count = 8
num_heads = 32
num_kv_heads = 8
head_size = 128
max_input_len = 1


def main() -> None:
    query = torch.zeros(token_count, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    key = torch.zeros(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros(token_count, num_kv_heads, head_size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(query)

    b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    print("Parameters for prefill attention:")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"key shape: {key.shape}, dtype: {key.dtype}")
    print(f"value shape: {value.shape}, dtype: {value.dtype}")
    print(f"out shape: {out.shape}, dtype: {out.dtype}")
    print(f"b_start_loc shape: {b_start_loc.shape}, dtype: {b_start_loc.dtype}")
    print(f"b_seq_len shape: {b_seq_len.shape}, dtype: {b_seq_len.dtype}")
    print(f"max_input_len: {max_input_len}")

    run_prefill_attention_kernel(
        q=query,
        k=key,
        v=value,
        out=out,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=max_input_len,
        is_causal=True,
    )

    sample = out[:, 0, 0].float().cpu()[:8]
    print("Output out[:,0,0] first 8:", " ".join(f"{v.item():.5f}" for v in sample))


if __name__ == "__main__":
    main()
