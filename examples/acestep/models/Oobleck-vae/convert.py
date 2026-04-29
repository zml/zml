#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def main() -> None:
    input_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffusion_pytorch_model.safetensors"
    output_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffusion_pytorch_model_f32.safetensors"

    tensors = load_file(input_path)
    tensors_f32 = {name: tensor.to(dtype=torch.float32) for name, tensor in tensors.items()}

    save_file(tensors_f32, output_path)

    print(f"Loaded {len(tensors)} tensors from: {input_path}")
    print(f"Saved float32 tensors to: {output_path}")

if __name__ == "__main__":
    main()