"""Export a block-0 FF-boundary fixture for the current simplified Zig parity checker.

Preferred sources, in order:
    1. transformer_blocks.0 container input/output
    2. transformer_blocks.0.ff input/output
    3. transformer_blocks.0.ff.net.0.proj input + transformer_blocks.0.ff.net.2 output

The last fallback avoids tracing the full block container, which is too large for
the replay activation collector on stage-2 runs.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export block0 FF-boundary activation fixture from replay .pt to safetensors"
    )
    parser.add_argument("input_pt", type=Path, help="Path to replay .pt file")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--block-index", type=int, default=0, help="Transformer block index (default: 0)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_idx = args.block_index

    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    # Prefer the direct block capture. When that is unavailable, reconstruct the
    # current simplified checker boundary from the FF subtree.
    block_key = f"velocity_model.transformer_blocks.{block_idx}"
    ff_key = f"velocity_model.transformer_blocks.{block_idx}.ff"
    ff_proj_key = f"{ff_key}.net.0.proj"
    ff_out_key = f"{ff_key}.net.2"

    block_entry = acts.get(block_key)
    ff_entry = acts.get(ff_key)
    ff_proj_entry = acts.get(ff_proj_key)
    ff_out_entry = acts.get(ff_out_key)

    if block_entry is not None:
        block_input = block_entry["input"][0].detach().cpu().contiguous()
        block_output = block_entry["output"][0].detach().cpu().contiguous()
    elif ff_entry is not None:
        # Compatibility path for traces captured before the container module was included.
        block_input = ff_entry["input"][0].detach().cpu().contiguous()
        block_output = ff_entry["output"][0].detach().cpu().contiguous()
        print(
            f"Note: block-level key '{block_key}' not found; "
            f"using ff entry as simplified block0 boundary (input=ff.input0, output=ff.output0)"
        )
    elif ff_proj_entry is not None and ff_out_entry is not None:
        block_input = ff_proj_entry["input"][0].detach().cpu().contiguous()
        block_output = ff_out_entry["output"][0].detach().cpu().contiguous()
        print(
            f"Note: block-level and ff keys not found; using leaf entries "
            f"'{ff_proj_key}' input and '{ff_out_key}' output as simplified block0 boundary"
        )
    else:
        available = sorted(k for k in acts.keys() if f"transformer_blocks.{block_idx}" in k)
        raise KeyError(
            f"Could not find block {block_idx} entry in activations.\n"
            f"Available keys for block {block_idx}: {available[:30]}"
        )

    tensors = {
        "block0_ff_boundary.input0": block_input,
        "block0_ff_boundary.output0": block_output,
    }

    metadata = {
        "source_pt": str(args.input_pt),
        "block_index": str(block_idx),
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "dtype": str(block_output.dtype),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"saved: {args.output_st}")
    print(f"block0_ff_boundary.input0  shape={list(block_input.shape)}  dtype={block_input.dtype}")
    print(f"block0_ff_boundary.output0 shape={list(block_output.shape)} dtype={block_output.dtype}")


if __name__ == "__main__":
    main()
