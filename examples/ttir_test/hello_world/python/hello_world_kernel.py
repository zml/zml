# triton_hello_world_matmul.py
import torch
import triton
import triton.language as tl

# Hardcoded sizes
M = 256
N = 256
K = 256


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

    # Row-major contiguous assumptions:
    # A: [M, K] strides (K, 1)
    # B: [K, N] strides (N, 1)
    # C: [M, N] strides (N, 1)
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


def hello_world(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Simplest possible Triton 'hello world' matmul.

    Constraints:
      - a.shape == (256, 256)
      - b.shape == (256, 256)
      - dtype == torch.float32
      - CUDA tensors
      - row-major contiguous
    """
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.shape == (M, K)
    assert b.shape == (K, N)

    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    compiled_kernel = matmul_fixed_kernel[grid](
        a,
        b,
        c,
        M_C=M,
        N_C=N,
        K_C=K,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
    )

    print(
        f"---------------------\n ttir:\n\n{compiled_kernel.asm['ttir']}\n\n---------------------\n"
    )
    return compiled_kernel
