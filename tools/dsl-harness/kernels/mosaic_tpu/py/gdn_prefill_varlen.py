"""Gated DeltaNet — packed-varlen prefill kernel (chunkwise, single Pallas call).

Standalone Pallas Mosaic TPU kernel for the GDN packed-varlen prefill. L2
norm of q,k is INSIDE the kernel (matches FLA Triton's
`USE_QK_L2NORM_IN_KERNEL=true`), so callers pass un-normed q,k.

Adapted from `gdn-impl/zml/kernels.py`. Single-batch (uniform-length) prefill
is just a special case with `cu_seqlens=(0, T)`.

Pair file in zig: `kernels/mosaic_tpu/zig/gdn_prefill_varlen.zig`.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ---------------------------------------------------------------------------
# Triangular solve via nilpotent doubling: (I + A)^-1 for strictly-lower A.
# ---------------------------------------------------------------------------


def nilpotent_doubling_rounds(BT: int) -> int:
    """`ceil(log2(BT))` rounds, equivalently `(BT-1).bit_length()` for BT >= 1."""
    if BT < 1:
        raise ValueError(f"BT must be >= 1, got {BT}")
    if BT == 1:
        return 0
    return (BT - 1).bit_length()


def inv_eye_plus_A_jax(A: jax.Array) -> jax.Array:
    """`(I + A)^-1` for strictly-lower-triangular `A` of shape `[..., BT, BT]`.

    Nilpotent doubling on `Ā = -A`:
        T_2 = I + Ā;  P_2 = Ā²
        T = T + T @ P;  P = P @ P    (repeat ceil(log2(BT)) - 1 times)
    """
    BT = A.shape[-1]
    rounds = nilpotent_doubling_rounds(BT)
    Abar = -A
    eye = jnp.eye(BT, dtype=A.dtype)
    eye = jnp.broadcast_to(eye, A.shape)
    T = eye + Abar
    if rounds <= 1:
        return T
    P = Abar @ Abar
    for stage in range(1, rounds):
        T = T + T @ P
        if stage + 1 < rounds:
            P = P @ P
    return T


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------


def _varlen_fused_kernel(
    seq_id_smem,    # i32[total_chunks]   prefetched scalar
    is_first_smem,  # i32[total_chunks]   prefetched scalar (1 = first chunk of its seq)
    is_last_smem,   # i32[total_chunks]   prefetched scalar (1 = last chunk of its seq)
    q_ref,          # [1, BT, K]
    k_ref,          # [1, BT, K]
    v_ref,          # [1, BT, V]
    g_cum_ref,      # [1, BT, 1]
    beta_ref,       # [1, BT, 1]
    s_in_ref,       # [1, 1, K, V]   per-seq initial state (read at is_first)
    o_ref,          # [1, BT, V]
    s_out_ref,      # [1, 1, K, V]   per-seq final state (written at is_last)
    s_scratch,      # VMEM[K, V]     persistent across "arbitrary" axis iters
    *,
    BT: int,
    scale: float,
):
    c = pl.program_id(1)

    @pl.when(is_first_smem[c] == 1)
    def _init():
        s_scratch[...] = s_in_ref[0, 0].astype(jnp.float32)

    q = q_ref[0].astype(jnp.float32)                        # NOT scaled yet
    k = k_ref[0].astype(jnp.float32)
    # L2 norm q,k along K axis per token (matches Triton USE_QK_L2NORM_IN_KERNEL).
    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q * scale
    v = v_ref[0].astype(jnp.float32)
    g_cum = g_cum_ref[0, :, 0].astype(jnp.float32)
    beta = beta_ref[0, :, 0].astype(jnp.float32)
    S = s_scratch[...]

    # Cap upper triangle at 0 to avoid `0 * exp(huge) = NaN` when g cumsums
    # to large negatives (Qwen3.5 routinely emits g ≈ -38, cumsumming to -300+).
    decay_diff = jnp.minimum(g_cum[:, None] - g_cum[None, :], 0.0)
    decay = jnp.exp(decay_diff)
    idx = jnp.arange(BT)
    strict_lower = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    diag_or_below = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    kkt = k @ k.T
    A_strict = (kkt * strict_lower) * decay * beta[:, None]
    T_inv = inv_eye_plus_A_jax(A_strict)

    decay_g = jnp.exp(g_cum)
    beta_v = beta[:, None] * v
    beta_k_decay = (beta * decay_g)[:, None] * k
    u = T_inv @ beta_v
    w = T_inv @ beta_k_decay

    wS = w @ S
    v_new = u - wS
    qkT = q @ k.T
    attn = (qkT * diag_or_below) * decay
    o_intra = attn @ v_new
    o_inter = (q * decay_g[:, None]) @ S
    o = o_intra + o_inter

    g_last = g_cum[-1]
    decay_to_end = jnp.exp(g_last - g_cum)
    k_decayed = k * decay_to_end[:, None]
    S_new = jnp.exp(g_last) * S + k_decayed.T @ v_new

    s_scratch[...] = S_new
    o_ref[0] = o.astype(o_ref.dtype)

    @pl.when(is_last_smem[c] == 1)
    def _final():
        s_out_ref[0, 0] = s_scratch[...].astype(s_out_ref.dtype)


# ---------------------------------------------------------------------------
# Host-side helpers (build chunk metadata, padded packing, chunked cumsum)
# ---------------------------------------------------------------------------


def _build_chunk_metadata(seqlens, BT):
    seq_id, is_first, is_last, chunks_per_seq = [], [], [], []
    for i, t in enumerate(seqlens):
        n_chunks = (t + BT - 1) // BT if t > 0 else 0
        chunks_per_seq.append(n_chunks)
        for c in range(n_chunks):
            seq_id.append(i)
            is_first.append(1 if c == 0 else 0)
            is_last.append(1 if c == n_chunks - 1 else 0)
    return seq_id, is_first, is_last, chunks_per_seq


def _materialize_cu(cu_seqlens):
    if isinstance(cu_seqlens, (list, tuple)):
        return [int(x) for x in cu_seqlens]
    return np.asarray(cu_seqlens).astype(np.int64).tolist()


def _chunked_cumsum(x: jax.Array, BT: int) -> jax.Array:
    *leading, T = x.shape
    n_chunks = T // BT
    x_c = x.reshape(*leading, n_chunks, BT)
    x_c = jnp.cumsum(x_c, axis=-1)
    return x_c.reshape(*leading, T)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def gdn_prefill_varlen(
    q: jax.Array,            # [total_tokens, H_qk, K]
    k: jax.Array,            # [total_tokens, H_qk, K]
    v: jax.Array,            # [total_tokens, H_v, V]
    g: jax.Array,            # [total_tokens, H_v]
    beta: jax.Array,         # [total_tokens, H_v]
    cu_seqlens,              # [num_seqs + 1]
    *,
    chunk_size: int = 64,
    initial_states: jax.Array | None = None,
    state_dtype: jnp.dtype | None = None,
    scale: float | None = None,
    interpret: bool = False,
    output_final_state: bool = True,
    vmem_limit_bytes: int | None = None,
):
    """Packed varlen prefill — one Pallas call covers all sequences."""
    cu = _materialize_cu(cu_seqlens)
    num_seqs = len(cu) - 1
    if num_seqs < 1:
        raise ValueError(f"cu_seqlens must have at least 2 entries; got {cu}")
    if cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0; got {cu[0]}")
    total_tokens = q.shape[0]
    if cu[-1] != total_tokens:
        raise ValueError(f"cu_seqlens[-1]={cu[-1]} != total_tokens={total_tokens}")

    H_qk = q.shape[1]
    H_v = v.shape[1]
    K = q.shape[-1]
    V = v.shape[-1]
    BT = chunk_size
    if scale is None:
        scale = K ** -0.5
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be divisible by H_qk={H_qk}")
    gva_factor = H_v // H_qk

    seqlens = [cu[i + 1] - cu[i] for i in range(num_seqs)]
    seq_id_list, is_first_list, is_last_list, chunks_per_seq = _build_chunk_metadata(
        seqlens, BT)
    total_chunks = sum(chunks_per_seq)
    if total_chunks == 0:
        out_dtype = q.dtype
        s_dt = state_dtype or jnp.float32
        return (
            jnp.zeros((0, H_v, V), dtype=out_dtype),
            jnp.zeros((num_seqs, H_v, K, V), dtype=s_dt) if output_final_state else None,
        )

    L = total_chunks * BT

    def _pack_packed(x_packed):
        out = []
        for i, t in enumerate(seqlens):
            s, e = cu[i], cu[i + 1]
            chunk_count = chunks_per_seq[i]
            if t == 0:
                shape = (chunk_count * BT, *x_packed.shape[1:])
                out.append(jnp.zeros(shape, dtype=x_packed.dtype))
                continue
            seq = lax.dynamic_slice_in_dim(x_packed, s, t, axis=0)
            pad = chunk_count * BT - t
            if pad > 0:
                pad_widths = [(0, pad)] + [(0, 0)] * (x_packed.ndim - 1)
                seq = jnp.pad(seq, pad_widths)
            out.append(seq)
        return jnp.concatenate(out, axis=0)

    q_p = _pack_packed(q); k_p = _pack_packed(k); v_p = _pack_packed(v)
    g_p = _pack_packed(g); beta_p = _pack_packed(beta)

    q_p = jnp.transpose(q_p, (1, 0, 2)).astype(jnp.float32)
    k_p = jnp.transpose(k_p, (1, 0, 2)).astype(jnp.float32)
    v_p = jnp.transpose(v_p, (1, 0, 2)).astype(jnp.float32)
    g_p = jnp.transpose(g_p, (1, 0)).astype(jnp.float32)
    beta_p = jnp.transpose(beta_p, (1, 0)).astype(jnp.float32)

    g_cum_p = _chunked_cumsum(g_p, BT)[..., None]
    beta_p = beta_p[..., None]

    if initial_states is not None:
        s_dtype = initial_states.dtype if state_dtype is None else state_dtype
        S_in = initial_states.astype(s_dtype)
    else:
        s_dtype = state_dtype or jnp.float32
        S_in = jnp.zeros((num_seqs, H_v, K, V), dtype=s_dtype)

    seq_id_arr = jnp.asarray(seq_id_list, dtype=jnp.int32)
    is_first_arr = jnp.asarray(is_first_list, dtype=jnp.int32)
    is_last_arr = jnp.asarray(is_last_list, dtype=jnp.int32)

    out_dtype = q.dtype
    out_shape = (
        jax.ShapeDtypeStruct((H_v, L, V), out_dtype),
        jax.ShapeDtypeStruct((num_seqs, H_v, K, V), s_dtype),
    )

    def q_idx(h_v, c, *_): return (h_v // gva_factor, c, 0)
    def v_idx(h_v, c, *_): return (h_v, c, 0)
    def s_idx(h_v, c, seq_ref, *_): return (seq_ref[c], h_v, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(H_v, total_chunks),
        in_specs=[
            pl.BlockSpec((1, BT, K), q_idx),
            pl.BlockSpec((1, BT, K), q_idx),
            pl.BlockSpec((1, BT, V), v_idx),
            pl.BlockSpec((1, BT, 1), v_idx),
            pl.BlockSpec((1, BT, 1), v_idx),
            pl.BlockSpec((1, 1, K, V), s_idx),
        ],
        out_specs=[
            pl.BlockSpec((1, BT, V), v_idx),
            pl.BlockSpec((1, 1, K, V), s_idx),
        ],
        scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
    )
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary"),
        **({"vmem_limit_bytes": vmem_limit_bytes} if vmem_limit_bytes is not None else {}),
    )

    o_p, s_final = pl.pallas_call(
        partial(_varlen_fused_kernel, BT=BT, scale=scale),
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=interpret,
        compiler_params=compiler_params,
        name="gdn_prefill_varlen",
        debug=True,                  # required by the dsl-harness mosaic_tpu runner
    )(seq_id_arr, is_first_arr, is_last_arr,
      q_p, k_p, v_p, g_cum_p, beta_p, S_in)

    o_p = jnp.transpose(o_p, (1, 0, 2)).astype(out_dtype)
    out_chunks = []
    cursor = 0
    for i, t in enumerate(seqlens):
        chunk_count = chunks_per_seq[i]
        if t > 0:
            out_chunks.append(o_p[cursor:cursor + t])
        cursor += chunk_count * BT
    o = jnp.concatenate(out_chunks, axis=0) if out_chunks else \
        jnp.zeros((0, H_v, V), dtype=out_dtype)
    if not output_final_state:
        s_final = None
    return o, s_final


# ---------------------------------------------------------------------------
# ZML harness adapter
# ---------------------------------------------------------------------------


__all__ = ["gdn_prefill_varlen", "build_args"]


_DTYPE_MAP = {
    "i32": jnp.int32, "f16": jnp.float16, "bf16": jnp.bfloat16, "f32": jnp.float32,
}


def _dtype(name: Any, default: jnp.dtype) -> jnp.dtype:
    return _DTYPE_MAP.get(name, default) if isinstance(name, str) else default


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Returns `(args, static_kwargs)` for `gdn_prefill_varlen`.

    The Zig harness passes a fixed-size `seqlens` array padded with zero
    trailing entries plus `num_seqs` indicating the count of valid entries.
    Slice accordingly so the kernel sees only real sequences.
    """
    dtype = _dtype(cfg.get("dtype"), jnp.bfloat16)
    s_dtype = _dtype(cfg.get("state_dtype"), jnp.float32)
    K = int(cfg.get("head_k_dim", 128))
    V = int(cfg.get("head_v_dim", 128))
    H_qk = int(cfg.get("num_k_heads", 16))
    H_v = int(cfg.get("num_v_heads", 32))
    seqlens_full = list(cfg.get("seqlens", [256, 192, 128, 64]))
    num_seqs = int(cfg.get("num_seqs", len(seqlens_full)))
    seqlens = [int(s) for s in seqlens_full[:num_seqs]]
    BT = int(cfg.get("chunk_size", 64))

    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    total = cu[-1]

    q = jax.ShapeDtypeStruct((total, H_qk, K), dtype)
    k = jax.ShapeDtypeStruct((total, H_qk, K), dtype)
    v = jax.ShapeDtypeStruct((total, H_v, V), dtype)
    g = jax.ShapeDtypeStruct((total, H_v), jnp.float32)
    beta = jax.ShapeDtypeStruct((total, H_v), jnp.float32)
    return ([q, k, v, g, beta],
            {"cu_seqlens": tuple(cu),
             "chunk_size": BT,
             "interpret": bool(cfg.get("interpret", False)),
             "output_final_state": True,
             "state_dtype": s_dtype,
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
    p.add_argument("--seqlens", default="256,192,128,64")
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--vmem-limit-bytes", type=int, default=None)
    p.add_argument("--print-mlir", action="store_true")
    args = p.parse_args()

    cfg = {
        "dtype": args.dtype,
        "state_dtype": args.state_dtype,
        "seqlens": [int(s) for s in args.seqlens.split(",") if s],
        "num_k_heads": args.num_k_heads,
        "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim,
        "head_v_dim": args.head_v_dim,
        "chunk_size": args.chunk_size,
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
            gdn_prefill_varlen, backend="cpu",
            static_argnames=tuple(static_kwargs.keys()),
        ).lower(*pos_args, **static_kwargs)

    if args.print_mlir:
        print(lowered.compiler_ir())


if __name__ == "__main__":
    main()
