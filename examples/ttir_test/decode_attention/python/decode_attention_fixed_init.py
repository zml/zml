import os

import torch
from safetensors.torch import load_file, save_file

from wrap_decode_attention import run_decode_attention_stage1_kernel


torch.set_printoptions(threshold=torch.inf)

batch_size = 8
num_heads = 32
head_size = 128
head_size_padded = 1 << (head_size - 1).bit_length()
num_kv_splits = 8
page_size = 16
sm_scale = 1.0 / (head_size**0.5)


def main() -> None:
    inputs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "safetensors",
        "decode_attention_inputs.safetensors",
    )
    inputs = load_file(inputs_path)

    q = inputs["q"].to(device="cuda", dtype=torch.bfloat16)
    k_buffer = inputs["k_buffer"].to(device="cuda", dtype=torch.bfloat16)
    v_buffer = inputs["v_buffer"].to(device="cuda", dtype=torch.bfloat16)
    req_to_tokens = inputs["req_to_tokens"].to(device="cuda", dtype=torch.int32)
    b_seqlen = inputs["b_seqlen"].to(device="cuda", dtype=torch.int32)

    att_out = torch.empty(batch_size, num_heads, num_kv_splits, head_size_padded + 1, dtype=torch.float32, device="cuda")

    print("Parameters for decode attention stage1:")
    print(f"q shape: {q.shape}, dtype: {q.dtype}")
    print(f"k_buffer shape: {k_buffer.shape}, dtype: {k_buffer.dtype}")
    print(f"v_buffer shape: {v_buffer.shape}, dtype: {v_buffer.dtype}")
    print(f"req_to_tokens shape: {req_to_tokens.shape}, dtype: {req_to_tokens.dtype}")
    print(f"b_seqlen shape: {b_seqlen.shape}, dtype: {b_seqlen.dtype}")
    print(f"att_out shape: {att_out.shape}, dtype: {att_out.dtype}")

    run_decode_attention_stage1_kernel(
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

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "safetensors",
        "decode_attention_output.safetensors",
    )
    save_file({"att_out": att_out.detach().cpu()}, out_path)

    sample = att_out[:, 0, 0, 0].float().cpu()[:8]
    print("Output att_out[:,0,0,0] first 8:", " ".join(f"{v.item():.5f}" for v in sample))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    os.environ["TRITON_BACKEND_DEBUG"] = "1"
    os.environ["SHOULD_LOG"] = "1"
    main()
