import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def resolve_activation_key(acts: dict, requested_key: str, allow_proj_suffix: bool) -> str:
    if requested_key in acts:
        return requested_key

    candidates = [k for k in acts.keys() if k.startswith(requested_key)]
    if len(candidates) == 1:
        return candidates[0]

    if allow_proj_suffix:
        proj_key = requested_key + ".proj"
        if proj_key in acts:
            return proj_key

    if len(candidates) > 1:
        raise KeyError(
            f"Activation key is ambiguous: {requested_key}. "
            f"Candidates: {candidates[:20]}"
        )

    keys = sorted(acts.keys())
    raise KeyError(
        f"Activation key not found: {requested_key}. "
        f"Available keys: {keys[:20]}"
    )


def export_fixture(
    input_pt: Path,
    output_st: Path,
    activation_key: str,
    tensor_prefix: str,
    allow_proj_suffix: bool,
) -> None:
    obj = torch.load(input_pt, map_location="cpu", weights_only=False)

    acts = obj.get("activations", {})
    resolved_key = resolve_activation_key(acts, activation_key, allow_proj_suffix)

    entry = acts[resolved_key]
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

    input_name = f"{tensor_prefix}.input0"
    output_name = f"{tensor_prefix}.output0"

    tensors = {
        input_name: input0,
        output_name: output0,
    }

    metadata = {
        "source_pt": str(input_pt),
        "activation_key": resolved_key,
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "dtype": str(output0.dtype),
    }

    output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_st), metadata=metadata)

    print(f"saved: {output_st}")
    print(f"activation_key={resolved_key}")
    print(f"{input_name} shape={tuple(input0.shape)} dtype={input0.dtype} nbytes={input0.nelement() * input0.element_size()}")
    print(f"{output_name} shape={tuple(output0.shape)} dtype={output0.dtype} nbytes={output0.nelement() * output0.element_size()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export activation fixture from replay .pt to safetensors")
    parser.add_argument("input_pt", type=Path, help="Path to acts_stage2_transformer_step_... .pt")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument("--activation-key", required=True, help="Activation key inside obj['activations']")
    parser.add_argument("--tensor-prefix", required=True, help="Output tensor key prefix, e.g. ff or patchify")
    parser.add_argument(
        "--allow-proj-suffix",
        action="store_true",
        help="If activation key is missing, also try activation_key + '.proj'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_fixture(
        input_pt=args.input_pt,
        output_st=args.output_st,
        activation_key=args.activation_key,
        tensor_prefix=args.tensor_prefix,
        allow_proj_suffix=args.allow_proj_suffix,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
