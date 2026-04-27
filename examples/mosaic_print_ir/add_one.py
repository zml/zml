"""JAX Pallas TPU counterpart to `main.zig` — same `add_one` kernel, same
shape. With `debug=True`, `pl.pallas_call` prints the Mosaic IR module
during lowering so we can compare it side-by-side with what `zml/mosaic`
emits.

Run:
    uv run --with "jax" --with "jaxlib" python add_one.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BLOCK_M = 128
BLOCK_N = 128


def add_one_kernel(x_ref, y_ref):
    # Pallas style: read the full ref ([...]) into a vector, add 1.0, store
    # back. Mirrors the Zig DSL: `k.refStore(y, k.refLoad(x).add(1.0))`.
    y_ref[...] = x_ref[...] + jnp.bfloat16(1.0)


def add_one(x):
    return pl.pallas_call(
        add_one_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        debug=True,  # makes JAX print the Mosaic IR during lowering
    )(x)


def main():
    x = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.bfloat16)
    # `lowering_platforms=("tpu",)` forces the TPU lowering even on a
    # non-TPU host — pallas_call would otherwise error with
    # "Only interpret mode is supported on CPU backend."
    jax.jit(add_one).trace(x).lower(lowering_platforms=("tpu",))


if __name__ == "__main__":
    main()
