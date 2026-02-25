import torch
from hello_world_kernel import K, M, N, hello_world


def run():
    a = torch.zeros((M, K), dtype=torch.float32, device="cuda")
    b = torch.zeros((K, N), dtype=torch.float32, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    print(f"a shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")
    print(f"b shape: {b.shape}, dtype: {b.dtype}, device: {b.device}")
    print(f"c shape: {c.shape}, dtype: {c.dtype}, device: {c.device}")

    hello_world(a, b, c)

    c_cpu = c.detach().cpu()
    grid = c_cpu[:10, :10]
    print("c[0:10,0:10] after matmul:")
    for row in grid:
        print(" ".join(f"{v.item():.6g}" for v in row))


if __name__ == "__main__":
    run()
