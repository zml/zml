import jax
import jax.numpy as jnp
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------
# 1. NKI Kernel Definition
# ---------------------------------------------------------
@nki.jit
def nki_matmul_jax_(lhsT, rhs):
    """
    Computes a single-tile matrix multiplication.
    
    Args:
        lhsT: JAX array of shape [K, M]. This is the transposed Left-Hand Side matrix.
        rhs: JAX array of shape [K, N]. This is the Right-Hand Side matrix.
    Returns:
        result: JAX array of shape [M, N].
    """
    # Extract dimensions
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "Inner contraction dimensions must match!"

    # 1. Allocate output tensor in High Bandwidth Memory (HBM)
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # 2. Allocate on-chip SRAM memory (SBUF) for inputs
    lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

    # 3. Load input data from HBM to SBUF (DMA copy)
    nisa.dma_copy(dst=lhs_tile, src=lhsT)
    nisa.dma_copy(dst=rhs_tile, src=rhs)

    # 4. Allocate space in the Accumulator memory (PSUM)
    # The matrix multiplier always outputs fp32 into the PSUM
    res_psum = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)

    # 5. Execute the hardware matrix multiplication instruction
    nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)

    # 6. Allocate SBUF for the result to cast it back to the original dtype
    res_sbuf = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res_sbuf, src=res_psum)

    # 7. Copy the final result from SBUF back out to HBM
    nisa.dma_copy(dst=result, src=res_sbuf)

    return result


# uv pip install jax-neuronx libneuronxla neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match

# NEURON_RT_INSPECT_ENABLE=1 \
# NEURON_RT_INSPECT_OUTPUT_DIR=/home/kevin/profiling \
# NEURON_RT_INSPECT_SYSTEM_PROFILE=1 \
# NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
# NEURON_FRAMEWORK_DEBUG=1 \
# XLA_HLO_DEBUG=1 \
# XLA_IR_DEBUG=1 \
# python run_matmul.py

# Note: You may need to point it to the specific .ntff file inside your profiling folder
# neuron-profile view --output-format summary-text -d /home/kevin/profiling/ 

# ---------------------------------------------------------
# 2. JAX Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Setting up JAX environment and generating data...")
    
    # We use bfloat16 as it is highly optimized for Trainium/Inferentia matrix multipliers.
    # We must stay within the single-tile limits for this basic kernel (K<=128, M<=128, N<=512)
    M, K, N = 64, 128, 512
    
    # Create standard JAX matrices
    # JAX will automatically allocate these on the Neuron device if libneuronxla is installed
    lhs_jax = jnp.ones((M, K), dtype=jnp.bfloat16)
    rhs_jax = jnp.ones((K, N), dtype=jnp.bfloat16) * 2.0  # Multiply by 2 so the math is obvious
    
    # Transpose LHS to match the kernel's stationary requirement (K x M)
    lhsT_jax = jnp.transpose(lhs_jax)
    
    print("Executing NKI kernel natively as a JAX operation...")
    
    # Call the kernel. JAX handles the compilation and dispatches it to the Neuron core.
    nki_output = nki_matmul_jax_(lhsT_jax, rhs_jax)
    nki_output = nki_matmul_jax_(lhsT_jax, rhs_jax)
    nki_output = nki_matmul_jax_(lhsT_jax, rhs_jax)
    
    print(f"Execution complete. Output shape: {nki_output.shape}")
    
    # Let's verify the math against JAX's standard compiler
    print("\nVerifying against standard jnp.matmul...")
    expected_output = jnp.matmul(lhs_jax, rhs_jax)
    expected_output = jnp.matmul(lhs_jax, rhs_jax)
    expected_output = jnp.matmul(lhs_jax, rhs_jax)
    
    # Since K=128, each element in the output should be 128 * (1.0 * 2.0) = 256.0
    print(f"Sample NKI value: {nki_output[0, 0]}")
    print(f"Sample JAX value: {expected_output[0, 0]}")
    
    is_match = jnp.allclose(nki_output, expected_output)
    print(f"Results Match: {'✅ Yes' if is_match else '❌ No'}")
