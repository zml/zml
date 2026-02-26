import os
from pathlib import Path

import torch

from hello_world_kernel import K, M, N, hello_world


def main() -> None:
    os.environ["TRITON_BACKEND_DEBUG"] = "0"
    os.environ["SHOULD_LOG"] = "0"

    a = torch.zeros((M, K), dtype=torch.float32, device="cuda")
    b = torch.zeros((K, N), dtype=torch.float32, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    compiled_kernel = hello_world(a, b, c)

    out_path = Path(__file__).resolve().parents[1] / "hello_world_kernel.ttir"
    out_path.write_text(compiled_kernel.asm["ttir"])
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
