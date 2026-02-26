import argparse
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fixed_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M_C: tl.constexpr,
    N_C: tl.constexpr,
    K_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    A = tl.make_block_ptr(
        base=A_ptr,
        shape=(M_C, K_C),
        strides=(K_C, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B = tl.make_block_ptr(
        base=B_ptr,
        shape=(K_C, N_C),
        strides=(N_C, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    C = tl.make_block_ptr(
        base=C_ptr,
        shape=(M_C, N_C),
        strides=(N_C, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K_C, BLOCK_K):
        a = tl.load(A, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        A = tl.advance(A, (0, BLOCK_K))
        B = tl.advance(B, (BLOCK_K, 0))

    tl.store(C, acc, boundary_check=(0, 1))


def compile_hello_world_ttir(args_json: str) -> str:
    params = json.loads(args_json)

    m = int(params.get("M", 256))
    n = int(params.get("N", 256))
    k = int(params.get("K", 256))
    block_m = int(params.get("BLOCK_M", 128))
    block_n = int(params.get("BLOCK_N", 128))
    block_k = int(params.get("BLOCK_K", 32))
    num_warps = int(params.get("num_warps", 8))

    os.environ.setdefault("TRITON_BACKEND_DEBUG", "0")
    os.environ.setdefault("SHOULD_LOG", "0")

    a = torch.zeros((m, k), dtype=torch.float32, device="cuda")
    b = torch.zeros((k, n), dtype=torch.float32, device="cuda")
    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    compiled_kernel = matmul_fixed_kernel[grid](
        a,
        b,
        c,
        M_C=m,
        N_C=n,
        K_C=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return str(compiled_kernel.asm["ttir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton hello_world and print TTIR")
    parser.add_argument(
        "--kernel-params",
        type=str,
        default='{"M":256,"N":256,"K":256,"BLOCK_M":128,"BLOCK_N":128,"BLOCK_K":32,"num_warps":8}',
        help="JSON arguments used to compile the kernel",
    )
    args = parser.parse_args()
    print(compile_hello_world_ttir(args.kernel_params), end="")


if __name__ == "__main__":
    main()
