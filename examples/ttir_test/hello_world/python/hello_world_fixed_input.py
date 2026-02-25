import os
import torch
from safetensors.torch import load_file

from hello_world_kernel import M, N, K, hello_world


def main() -> None:
    inputs_path = os.path.join(
        os.path.dirname(__file__), "..", "safetensors", "hello_world_inputs.safetensors"
    )
    inputs = load_file(inputs_path)

    a = inputs["a"].to(device="cuda", dtype=torch.float32)
    b = inputs["b"].to(device="cuda", dtype=torch.float32)
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    print(f"a shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")
    print(f"b shape: {b.shape}, dtype: {b.dtype}, device: {b.device}")
    print(f"c shape: {c.shape}, dtype: {c.dtype}, device: {c.device}")

    hello_world(a, b, c)

    c_cpu = c.detach().cpu()
    grid = c_cpu[:10, :10]
    print("c[0:10,0:10] after matmul:")
    for row in grid:
        print(" ".join(f"{v.item():.5f}" for v in row))


if __name__ == "__main__":
    main()
