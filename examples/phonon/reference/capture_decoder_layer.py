"""Capture a full decoder layer using an existing norm1 input and cached weights.

Run with this directory's virtualenv Python. Paths default relative to this script.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import GradiumPhononTTSForConditionalGeneration


def main():
    assets = Path(__file__).resolve().parent.parent / "assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_path = args.input or assets / f"model.layers.{args.layer}.input_layernorm.activations.safetensors"
    output_path = args.output or assets / f"model.layers.{args.layer}.activations.safetensors"

    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
        "gradium/phonon-31m-en",
        revision="92741c5b07d085ff840e26fea8ce43110b87852f",
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()

    with torch.inference_mode():
        hidden_states = load_file(input_path)["in"]
        positions = torch.arange(hidden_states.shape[1]).unsqueeze(0).expand(hidden_states.shape[0], -1)
        position_embeddings = model.model.rotary_emb(hidden_states, positions)
        output = model.model.layers[args.layer](
            hidden_states,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        save_file({"in": hidden_states.contiguous(), "out": output.contiguous()}, output_path)
    print(f"Saved {output_path}: {tuple(hidden_states.shape)} -> {tuple(output.shape)}")


if __name__ == "__main__":
    main()
