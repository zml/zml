import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from export_activation_fixture import resolve_activation_key


ACTIVATION_KEYS = {
    "attn1": "velocity_model.transformer_blocks.0.attn1",
    "attn2": "velocity_model.transformer_blocks.0.attn2",
    "audio_attn1": "velocity_model.transformer_blocks.0.audio_attn1",
    "audio_attn2": "velocity_model.transformer_blocks.0.audio_attn2",
    "audio_to_video_attn": "velocity_model.transformer_blocks.0.audio_to_video_attn",
    "video_to_audio_attn": "velocity_model.transformer_blocks.0.video_to_audio_attn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attention activation fixture from replay .pt to safetensors")
    parser.add_argument("input_pt", type=Path, help="Path to acts_stage2_transformer_step_... .pt")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(ACTIVATION_KEYS.keys()),
        help="Attention component to export",
    )
    parser.add_argument(
        "--activation-key",
        default=None,
        help="Override activation key inside obj['activations']",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation_key = args.activation_key or ACTIVATION_KEYS[args.mode]

    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    input0 = None
    output0 = None
    resolved_key = None

    # Preferred path: full attention module was hooked.
    if activation_key in acts:
        resolved_key = activation_key
        entry = acts[activation_key]
        if not isinstance(entry, dict):
            raise TypeError(f"Unexpected activation entry type: {type(entry)}")

        inputs = entry.get("input", [])
        outputs = entry.get("output", [])
        if len(inputs) < 1 or len(outputs) < 1:
            raise ValueError(
                f"Expected non-empty input/output for {activation_key}, got input={len(inputs)} output={len(outputs)}"
            )

        input0 = inputs[0].detach().cpu().contiguous()
        output0 = outputs[0].detach().cpu().contiguous()
    else:
        # Fallback path: capture usually contains leaf modules only.
        # Attention input is identical to to_q input, and attention output is to_out.0 output.
        q_key = resolve_activation_key(acts, activation_key + ".to_q", allow_proj_suffix=False)

        out_key = None
        out_err = None
        for candidate in (
            activation_key + ".to_out.0",
            activation_key + ".to_out",
            activation_key,
        ):
            try:
                out_key = resolve_activation_key(acts, candidate, allow_proj_suffix=False)
                break
            except KeyError as err:
                out_err = err

        if out_key is None:
            # Some traces only include q/k/v projections. That is insufficient for full attention parity,
            # because checker output is post-attention and post-to_out.
            available = sorted([k for k in acts.keys() if k.startswith(activation_key + ".")])
            raise ValueError(
                "Trace is missing required attention output hook for full parity. "
                f"Could not find any of: '{activation_key}.to_out.0', '{activation_key}.to_out', '{activation_key}'. "
                f"Available under prefix: {available}. "
                "Regenerate replay with --all-modules and an include regex that explicitly captures "
                "to_gate_logits and to_out(.0), e.g. "
                rf"^({activation_key}\\.(q_norm|k_norm|to_q|to_k|to_v|to_gate_logits|to_out(\\.0)?))(\\.|$)"
            ) from out_err

        q_entry = acts[q_key]
        out_entry = acts[out_key]
        if not isinstance(q_entry, dict) or not isinstance(out_entry, dict):
            raise TypeError("Unexpected activation entry type for attention fallback")

        q_inputs = q_entry.get("input", [])
        out_outputs = out_entry.get("output", [])
        if len(q_inputs) < 1 or len(out_outputs) < 1:
            raise ValueError(
                f"Expected non-empty fallback entries for {activation_key}, got to_q.input={len(q_inputs)} to_out.0.output={len(out_outputs)}"
            )

        input0 = q_inputs[0].detach().cpu().contiguous()
        output0 = out_outputs[0].detach().cpu().contiguous()
        resolved_key = f"{activation_key} [fallback: input={q_key}, output={out_key}]"

    input_name = f"{args.mode}.input0"
    output_name = f"{args.mode}.output0"
    tensors = {input_name: input0, output_name: output0}

    metadata = {
        "source_pt": str(args.input_pt),
        "activation_key": resolved_key,
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "dtype": str(output0.dtype),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"saved: {args.output_st}")
    print(f"activation_key={resolved_key}")
    print(f"{input_name} shape={tuple(input0.shape)} dtype={input0.dtype} nbytes={input0.nelement() * input0.element_size()}")
    print(f"{output_name} shape={tuple(output0.shape)} dtype={output0.dtype} nbytes={output0.nelement() * output0.element_size()}")


if __name__ == "__main__":
    main()
