import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export block0.ff activation fixture from replay .pt to safetensors")
    parser.add_argument("input_pt", type=Path, help="Path to acts_stage2_transformer_step_..._b00_ff_boundary.pt")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--activation-key",
        default="velocity_model.transformer_blocks.0.ff",
        help="Activation key inside obj['activations']",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)

    acts = obj.get("activations", {})
    if args.activation_key not in acts:
        keys = sorted(acts.keys())
        raise KeyError(f"Activation key not found: {args.activation_key}. Available keys: {keys[:20]}")

    entry = acts[args.activation_key]
    if not isinstance(entry, dict):
        raise TypeError(f"Unexpected activation entry type: {type(entry)}")

    inputs = entry.get("input", [])
    outputs = entry.get("output", [])
    if len(inputs) < 1 or len(outputs) < 1:
        raise ValueError(
            f"Expected non-empty input/output for {args.activation_key}, got input={len(inputs)} output={len(outputs)}"
        )

    # Keep bf16 as-is; parity binary will convert to f32 only for metrics.
    ff_input0 = inputs[0].detach().cpu().contiguous()
    ff_output0 = outputs[0].detach().cpu().contiguous()

    tensors = {
        "ff.input0": ff_input0,
        "ff.output0": ff_output0,
    }

    metadata = {
        "source_pt": str(args.input_pt),
        "activation_key": args.activation_key,
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "dtype": str(ff_output0.dtype),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"saved: {args.output_st}")
    print(f"ff.input0 shape={tuple(ff_input0.shape)} dtype={ff_input0.dtype} nbytes={ff_input0.nelement() * ff_input0.element_size()}")
    print(f"ff.output0 shape={tuple(ff_output0.shape)} dtype={ff_output0.dtype} nbytes={ff_output0.nelement() * ff_output0.element_size()}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
