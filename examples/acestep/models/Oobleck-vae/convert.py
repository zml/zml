#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def main() -> None:
    input_path_1 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00001-of-00004.safetensors"
    input_path_2 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00002-of-00004.safetensors"
    input_path_3 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00003-of-00004.safetensors"
    input_path_4 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00004-of-00004.safetensors"
    
    output_path_1 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00001-of-00004_bf16.safetensors"
    output_path_2 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00002-of-00004_bf16.safetensors"
    output_path_3 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00003-of-00004_bf16.safetensors"
    output_path_4 = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-xl-turbo//model-00004-of-00004_bf16.safetensors"

    tensors_1 = load_file(input_path_1)
    tensors_1_bf16 = { name: tensor.to(dtype=torch.bfloat16) for name, tensor in tensors_1.items()}
    save_file(tensors_1_bf16, output_path_1)
    print(f"Loaded {len(tensors_1)} tensors from: {input_path_1}")
    print(f"Saved bf16 tensors to: {output_path_1}")

    tensors_2 = load_file(input_path_2)
    tensors_2_bf16 = { name: tensor.to(dtype=torch.bfloat16) for name, tensor in tensors_2.items()}
    save_file(tensors_2_bf16, output_path_2)
    print(f"Loaded {len(tensors_2)} tensors from: {input_path_2}")
    print(f"Saved bf16 tensors to: {output_path_2}")

    tensors_3 = load_file(input_path_3)
    tensors_3_bf16 = { name: tensor.to(dtype=torch.bfloat16) for name, tensor in tensors_3.items()}
    save_file(tensors_3_bf16, output_path_3)
    print(f"Loaded {len(tensors_3)} tensors from: {input_path_3}")
    print(f"Saved bf16 tensors to: {output_path_3}")

    tensors_4 = load_file(input_path_4)
    tensors_4_bf16 = { name: tensor.to(dtype=torch.bfloat16) for name, tensor in tensors_4.items()}
    save_file(tensors_4_bf16, output_path_4)
    print(f"Loaded {len(tensors_4)} tensors from: {input_path_4}")
    print(f"Saved bf16 tensors to: {output_path_4}")

if __name__ == "__main__":
    main()