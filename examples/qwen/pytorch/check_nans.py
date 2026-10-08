#!/usr/bin/env python3
import sys
import numpy as np
import torch
from safetensors import safe_open


def inspect_tensor(f, tensor_name):
    """Load a tensor slice and count NaNs and Infs."""
    try:
        # Load via PyTorch framework to support bfloat16
        tensor_pt = f.get_tensor(tensor_name)

        # Cast bfloat16 / float16 to float32 for NumPy inspection
        if tensor_pt.dtype in (torch.bfloat16, torch.float16):
            tensor_pt = tensor_pt.to(torch.float32)

        tensor = tensor_pt.numpy()
    except Exception as e:
        print(f"\n[Error] Could not load tensor '{tensor_name}': {e}")
        return

    # Handle float vs non-float types
    if np.issubdtype(tensor.dtype, np.floating):
        nan_count = np.isnan(tensor).sum()
        inf_count = np.isinf(tensor).sum()
        total_elements = tensor.size

        print(f"\n--- Tensor: {tensor_name} ---")
        print(f"Shape:            {tensor.shape}")
        print(f"Dtype:            {tensor_pt.dtype}")
        print(f"Total Elements:   {total_elements:,}")
        print(f"NaN Count:        {nan_count:,} ({nan_count / total_elements:.2%})")
        print(f"Inf Count:        {inf_count:,} ({inf_count / total_elements:.2%})")

        if nan_count == 0 and inf_count == 0:
            print(f"Min / Max:        {tensor.min():.6f} / {tensor.max():.6f}")
            print(f"Mean / StdDev:    {tensor.mean():.6f} / {tensor.std():.6f}")
    else:
        print(f"\n--- Tensor: {tensor_name} ---")
        print(f"Shape:          {tensor.shape}")
        print(f"Dtype:          {tensor_pt.dtype} (Non-floating point format)")
        print(f"Total Elements: {tensor.size:,}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_nans.py <file.safetensors> [tensor_name]")
        sys.exit(1)

    filepath = sys.argv[1]
    target_tensor = sys.argv[2] if len(sys.argv) > 2 else None

    with safe_open(filepath, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        keys.sort()

        if not keys:
            print("[Error] The safetensors file contains no keys.")
            return

        # Direct evaluation if tensor name passed as argument
        if target_tensor:
            if target_tensor in keys:
                inspect_tensor(f, target_tensor)
            else:
                print(f"[Error] Tensor '{target_tensor}' not found in file.")
                print("Available keys matching search pattern:")
                matches = [k for k in keys if target_tensor.lower() in k.lower()]
                for m in matches[:10]:
                    print(f"  - {m}")
            return

        # Interactive selection menu if no target_tensor argument was supplied
        print(f"\nLoaded '{filepath}' ({len(keys)} tensors available)\n")

        filter_str = (
            input("Filter tensor names (press Enter to show all): ").strip().lower()
        )
        filtered_keys = (
            [k for k in keys if filter_str in k.lower()] if filter_str else keys
        )

        if not filtered_keys:
            print("No tensors matched your filter.")
            return

        print("\nSelect a tensor buffer to evaluate:")
        for idx, key in enumerate(filtered_keys[:50]):
            slice_obj = f.get_slice(key)
            print(
                f" [{idx + 1}] {key} -> {slice_obj.get_shape()} ({slice_obj.get_dtype()})"
            )

        if len(filtered_keys) > 50:
            print(f"... and {len(filtered_keys) - 50} more tensors.")

        try:
            choice = input(
                f"\nEnter number [1-{min(50, len(filtered_keys))}]: "
            ).strip()
            selected_idx = int(choice) - 1
            if 0 <= selected_idx < len(filtered_keys):
                inspect_tensor(f, filtered_keys[selected_idx])
            else:
                print("Invalid selection.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")


if __name__ == "__main__":
    main()
