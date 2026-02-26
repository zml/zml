import os

import torch
from safetensors.torch import load_file, save_file

from wrap_prefill_attention import run_prefill_attention_kernel


torch.set_printoptions(threshold=torch.inf)


def main() -> None:
    inputs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "safetensors",
        "prefill_attention_inputs.safetensors",
    )
    inputs = load_file(inputs_path)

    query = inputs["query"].to(device="cuda", dtype=torch.bfloat16)
    key = inputs["key"].to(device="cuda", dtype=torch.bfloat16)
    value = inputs["value"].to(device="cuda", dtype=torch.bfloat16)
    b_start_loc = inputs["b_start_loc"].to(device="cuda", dtype=torch.int32)
    b_seq_len = inputs["b_seq_len"].to(device="cuda", dtype=torch.int32)
    max_input_len = int(inputs["max_input_len"].item())

    out = torch.empty_like(query)

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

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "safetensors",
        "prefill_attention_output.safetensors",
    )
    save_file({"out": out.detach().cpu()}, out_path)

    sample = out[:, 0, 0].float().cpu()[:8]
    print("Output out[:,0,0] first 8:", " ".join(f"{v.item():.5f}" for v in sample))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"
    main()
