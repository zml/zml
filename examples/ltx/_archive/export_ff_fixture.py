import argparse
from pathlib import Path
from export_activation_fixture import export_fixture


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
    export_fixture(
        input_pt=args.input_pt,
        output_st=args.output_st,
        activation_key=args.activation_key,
        tensor_prefix="ff",
        allow_proj_suffix=True,
    )


if __name__ == "__main__":
    main()
