import os
import sys
import torch
from safetensors.torch import save_file

THIS_DIR = os.path.dirname(__file__)
PY_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "python"))
sys.path.append(PY_DIR)

from hello_world_kernel import M, N, K


def main() -> None:
    # Initialize random inputs for the matmul kernel.
    a = torch.randn((M, K), dtype=torch.float32, device="cuda")
    b = torch.randn((K, N), dtype=torch.float32, device="cuda")

    # Save to safetensors on CPU for portability.
    out_path = os.path.join(os.path.dirname(__file__), "hello_world_inputs.safetensors")
    save_file({"a": a.cpu(), "b": b.cpu()}, out_path)

    print(f"Wrote: {out_path}")
    print(f"a shape: {a.shape}, dtype: {a.dtype}")
    print(f"b shape: {b.shape}, dtype: {b.dtype}")


if __name__ == "__main__":
    main()
