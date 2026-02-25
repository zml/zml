import os

import torch
from safetensors.torch import save_file


torch.manual_seed(0)

batch_size = 8
num_heads = 32
num_kv_heads = 8
head_size = 128


def main() -> None:
    # Random valid sequence lengths per batch element.
    b_seq_len = torch.randint(1, 9, (batch_size,), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)

    token_count = int(b_seq_len.sum().item())
    max_input_len = int(b_seq_len.max().item())

    query = torch.randn(
        token_count,
        num_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(
        token_count,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn(
        token_count,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    out_path = os.path.join(os.path.dirname(__file__), "prefill_attention_inputs.safetensors")
    save_file(
        {
            "query": query.cpu(),
            "key": key.cpu(),
            "value": value.cpu(),
            "b_start_loc": b_start_loc.cpu(),
            "b_seq_len": b_seq_len.cpu(),
            "max_input_len": torch.tensor([max_input_len], dtype=torch.int32),
        },
        out_path,
    )

    print(f"Wrote: {out_path}")
    print(f"query shape: {query.shape}, dtype: {query.dtype}")
    print(f"key shape: {key.shape}, dtype: {key.dtype}")
    print(f"value shape: {value.shape}, dtype: {value.dtype}")
    print(f"b_start_loc shape: {b_start_loc.shape}, dtype: {b_start_loc.dtype}")
    print(f"b_seq_len shape: {b_seq_len.shape}, dtype: {b_seq_len.dtype}")
    print(f"token_count: {token_count}")
    print(f"max_input_len: {max_input_len}")


if __name__ == "__main__":
    main()
