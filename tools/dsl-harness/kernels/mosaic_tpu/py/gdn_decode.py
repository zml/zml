"""Gated DeltaNet — decode kernel (single-token recurrent step).

Standalone Pallas Mosaic TPU kernel for the GDN per-token decode. L2 norm of
q,k is INSIDE the kernel (matches FLA Triton's `USE_QK_L2NORM_IN_KERNEL=true`),
so callers pass un-normed q,k.

Adapted from `gdn-impl/zml/kernels.py` (the GDN-Research consolidated copy).
This is the harness-friendly form: standalone, has `build_args(cfg)`, has
`main()` that lowers the kernel under an abstract TPU mesh + CPU shim.

Pair file in zig: `kernels/mosaic_tpu/zig/gdn_decode.zig`.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_DECODE_CLIP = 20.0


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------


def _decode_kernel(
    q_ref,       # [M_pp//gva_factor, K]
    k_ref,       # [M_pp//gva_factor, K]
    v_ref,       # [M_pp, V]
    g_ref,       # [M_pp, 1]   fp32   (rank-2 to satisfy (8, 128) tiling)
    beta_ref,    # [M_pp, 1]   fp32
    s_in_ref,    # [M_pp, K, V]   aliased with s_out_ref
    o_ref,       # [M_pp, V]
    s_out_ref,   # [M_pp, K, V]
    *,
    scale: float,
    gva_factor: int,
):
    q = q_ref[...].astype(jnp.float32)                       # NOT scaled yet
    k = k_ref[...].astype(jnp.float32)
    # L2 norm q,k along K axis (matches Triton's USE_QK_L2NORM_IN_KERNEL).
    # Done before GVA-replication so we normalize per q/k-head, not per slot.
    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q * scale
    if gva_factor > 1:
        q = jnp.repeat(q, gva_factor, axis=0)
        k = jnp.repeat(k, gva_factor, axis=0)
    v = v_ref[...].astype(jnp.float32)
    g = g_ref[:, 0].astype(jnp.float32)
    beta = beta_ref[:, 0].astype(jnp.float32)
    s = s_in_ref[...].astype(jnp.float32)

    g = jnp.clip(g, -_DECODE_CLIP, _DECODE_CLIP)
    alpha = jnp.exp(g)

    # Per-slot batched matmul: out[m, v] = sum_k k[m, k] * s[m, k, v].
    # `matmul(...)[:, 0, :]` instead of `einsum("mk,mkv->mv", ...)` — the
    # einsum form trips a Mosaic `lhs_non_contracting_dims` lowering bug.
    kS = jnp.matmul(k[:, None, :], s)[:, 0, :]
    qS = jnp.matmul(q[:, None, :], s)[:, 0, :]
    kv = alpha[:, None] * kS
    y_alpha = alpha[:, None] * qS

    delta = beta[:, None] * (v - kv)
    kq = jnp.sum(k * q, axis=-1)
    out = y_alpha + delta * kq[:, None]

    s_new = alpha[:, None, None] * s + k[:, :, None] * delta[:, None, :]

    o_ref[...] = out.astype(o_ref.dtype)
    s_out_ref[...] = s_new.astype(s_out_ref.dtype)


# ---------------------------------------------------------------------------
# Wrapper: padding + GVA dispatch + pallas_call
# ---------------------------------------------------------------------------


def _pallas_decode_step(
    q, k, v, g, beta, state,
    *, M_pp, gva_factor, scale, interpret, vmem_limit_bytes,
):
    N, K = v.shape[0], q.shape[-1]
    V = v.shape[-1]
    assert N % M_pp == 0, f"N={N} must be a multiple of M_pp={M_pp}"
    M_pp_qk = M_pp // gva_factor
    n_progs = N // M_pp

    out_shape = (
        jax.ShapeDtypeStruct((N, V), v.dtype),
        jax.ShapeDtypeStruct((N, K, V), state.dtype),
    )
    compiler_params = (pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes)
                       if vmem_limit_bytes is not None else None)

    return pl.pallas_call(
        partial(_decode_kernel, scale=scale, gva_factor=gva_factor),
        out_shape=out_shape,
        grid=(n_progs,),
        in_specs=[
            pl.BlockSpec((M_pp_qk, K), lambda i: (i, 0)),
            pl.BlockSpec((M_pp_qk, K), lambda i: (i, 0)),
            pl.BlockSpec((M_pp, V), lambda i: (i, 0)),
            pl.BlockSpec((M_pp, 1), lambda i: (i, 0)),
            pl.BlockSpec((M_pp, 1), lambda i: (i, 0)),
            pl.BlockSpec((M_pp, K, V), lambda i: (i, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((M_pp, V), lambda i: (i, 0)),
            pl.BlockSpec((M_pp, K, V), lambda i: (i, 0, 0)),
        ],
        input_output_aliases={5: 1},
        interpret=interpret,
        compiler_params=compiler_params,
        name="gdn_decode",
        debug=True,                   # required by the dsl-harness mosaic_tpu runner
    )(q, k, v, g, beta, state)


def gdn_decode(
    q: jax.Array,            # [B, T=1, H_qk, K]
    k: jax.Array,            # [B, T=1, H_qk, K]
    v: jax.Array,            # [B, T=1, H_v, V]
    g: jax.Array,            # [B, T=1, H_v]
    beta: jax.Array,         # [B, T=1, H_v]
    state: jax.Array,        # [B, H_v, K, V]
    *,
    M_pp: int = 8,
    scale: float | None = None,
    interpret: bool = False,
    vmem_limit_bytes: int | None = None,
):
    """Single-token Pallas decode. Returns `(o [B, 1, H_v, V], state' [B, H_v, K, V])`."""
    if q.shape[1] != 1:
        raise ValueError(f"This kernel is single-token. Got T={q.shape[1]}.")
    B, _, H_qk, K = q.shape
    H_v = v.shape[2]
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be divisible by H_qk={H_qk}")
    factor = H_v // H_qk

    if factor > 1 and M_pp % factor == 0:
        gva_factor = factor
        H_qk_kernel = H_qk
    else:
        q = jnp.repeat(q, factor, axis=2) if factor > 1 else q
        k = jnp.repeat(k, factor, axis=2) if factor > 1 else k
        gva_factor = 1
        H_qk_kernel = H_v

    N = B * H_v
    N_qk = B * H_qk_kernel
    pad = (-N) % M_pp
    pad_qk = pad // gva_factor

    q_flat = q.reshape(N_qk, K)
    k_flat = k.reshape(N_qk, K)
    v_flat = v.reshape(N, V)
    g_flat = g.reshape(N, 1)
    beta_flat = beta.reshape(N, 1)
    state_flat = state.reshape(N, K, V)

    if pad:
        q_flat = jnp.pad(q_flat, ((0, pad_qk), (0, 0)))
        k_flat = jnp.pad(k_flat, ((0, pad_qk), (0, 0)))
        v_flat = jnp.pad(v_flat, ((0, pad), (0, 0)))
        g_flat = jnp.pad(g_flat, ((0, pad), (0, 0)))
        beta_flat = jnp.pad(beta_flat, ((0, pad), (0, 0)))
        state_flat = jnp.pad(state_flat, ((0, pad), (0, 0), (0, 0)))

    o_flat, s_flat = _pallas_decode_step(
        q_flat, k_flat, v_flat, g_flat, beta_flat, state_flat,
        M_pp=M_pp, gva_factor=gva_factor, scale=scale, interpret=interpret,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if pad:
        o_flat = o_flat[:N]
        s_flat = s_flat[:N]
    return o_flat.reshape(B, 1, H_v, V), s_flat.reshape(B, H_v, K, V)


# ---------------------------------------------------------------------------
# ZML harness adapter
# ---------------------------------------------------------------------------


__all__ = ["gdn_decode", "build_args"]


_DTYPE_MAP = {
    "i32": jnp.int32, "f16": jnp.float16, "bf16": jnp.bfloat16, "f32": jnp.float32,
}


def _dtype(name: Any, default: jnp.dtype) -> jnp.dtype:
    return _DTYPE_MAP.get(name, default) if isinstance(name, str) else default


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Returns `(args, static_kwargs)` for `gdn_decode`."""
    dtype = _dtype(cfg.get("dtype"), jnp.bfloat16)
    s_dtype = _dtype(cfg.get("state_dtype"), jnp.float32)
    K = int(cfg.get("head_k_dim", 128))
    V = int(cfg.get("head_v_dim", 128))
    H_qk = int(cfg.get("num_k_heads", 16))
    H_v = int(cfg.get("num_v_heads", 32))
    B = int(cfg.get("batch", 1))

    # M_pp_qk = M_pp / gva_factor must be ≥ 8 to satisfy (8, 128) tile rule.
    gva_factor = H_v // H_qk
    default_M_pp = max(8, 8 * gva_factor)

    q = jax.ShapeDtypeStruct((B, 1, H_qk, K), dtype)
    k = jax.ShapeDtypeStruct((B, 1, H_qk, K), dtype)
    v = jax.ShapeDtypeStruct((B, 1, H_v, V), dtype)
    g = jax.ShapeDtypeStruct((B, 1, H_v), jnp.float32)
    beta = jax.ShapeDtypeStruct((B, 1, H_v), jnp.float32)
    state = jax.ShapeDtypeStruct((B, H_v, K, V), s_dtype)
    return ([q, k, v, g, beta, state],
            {"M_pp": int(cfg.get("M_pp", default_M_pp)),
             "interpret": bool(cfg.get("interpret", False)),
             "vmem_limit_bytes": cfg.get("vmem_limit_bytes")})


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import contextlib
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--state-dtype", default="f32")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--m-pp", type=int, default=None,
                   help="Default: 8 * gva_factor (= 16 for Qwen3.5).")
    p.add_argument("--vmem-limit-bytes", type=int, default=None)
    p.add_argument("--print-mlir", action="store_true")
    args = p.parse_args()

    cfg = {
        "dtype": args.dtype,
        "state_dtype": args.state_dtype,
        "batch": args.batch,
        "num_k_heads": args.num_k_heads,
        "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim,
        "head_v_dim": args.head_v_dim,
        **({"M_pp": args.m_pp} if args.m_pp is not None else {}),
        "vmem_limit_bytes": args.vmem_limit_bytes,
        "interpret": False,
    }
    pos_args, static_kwargs = build_args(cfg)

    from jax._src.interpreters import mlir
    from jax._src.pallas import pallas_call

    def _cpu_lowering_tpu_wrapper(ctx, *in_nodes, **params):
        params.pop("backend", None)
        params.pop("which_linear", None)
        if "out_shapes" in params:
            params["out_avals"] = params.pop("out_shapes")
        from jax._src.pallas.mosaic.pallas_call_registration import (
            pallas_call_tpu_lowering_rule,
        )
        return pallas_call_tpu_lowering_rule(ctx, *in_nodes, **params)

    mlir.register_lowering(pallas_call.pallas_call_p,
                           _cpu_lowering_tpu_wrapper, platform="cpu")

    @contextlib.contextmanager
    def _tpu_abstract_mesh_context():
        mesh = jax.sharding.AbstractMesh(
            axis_sizes=(1,), axis_names=("tpu_core",),
            abstract_device=jax.sharding.AbstractDevice(
                device_kind="TPU v5 lite", num_cores=1,
            ),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            yield

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(
            gdn_decode, backend="cpu",
            static_argnames=tuple(static_kwargs.keys()),
        ).lower(*pos_args, **static_kwargs)

    if args.print_mlir:
        print(lowered.compiler_ir())


if __name__ == "__main__":
    main()
