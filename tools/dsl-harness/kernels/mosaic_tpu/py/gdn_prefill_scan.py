"""dsl-harness reference for GDN "P5" — `recurrent_scan` (the v2 chunked-prefill
+ recurrent-decode Mosaic kernel: schedule-table-driven `emit_pipeline`, fused
SiLU + gate, sublane-aligned chunking, token-by-token transition blocks at
sequence boundaries, in-kernel gather-DMA of state, inlined triangular inverse).

The kernel body itself lives in `gdn_fast.py` (a *verbatim* concatenation of the
relevant tpu-inference modules). This file is the harness adapter: it

  * monkeypatches `pl.pallas_call` so the one call inside `recurrent_scan` gets
    `debug=True` (the `mosaic_tpu/runner.py` captures stdout and extracts the
    post-lowering Mosaic module) and a stable `name="recurrent_scan"` (so the
    lowered `func.func` symbol matches what the Zig side emits, and the harness
    `--kernel="recurrent_scan"` filter lines up). Both are *cosmetic* — they
    don't change the kernel's semantics; cf. `gdn_decode_fast.py` which likewise
    just adds `debug=True`/`name=` to the verbatim `fused_decoding_gdn`.
  * re-exports `recurrent_scan` as the harness `py_kernel`,
  * exposes `build_args(cfg)` → `(args, static_kwargs)`. The kernel only ever
    sees *abstract* shapes (the harness lowers, never runs), so the seq layout
    boils down to two ints — `num_tokens` and `num_seqs` — plus `chunk_size`,
    `num_blocks`, the head dims, and the dtypes. (`compute_schedule_table_v2`
    runs in StableHLO on the abstract `query_start_loc` / `distribution`; the
    schedule-table *shape* is `(ceil(num_tokens/chunk_size) + 2*num_seqs,
    11 + 3*sublanesize)` — static — and the values never enter the IR.)
  * has a tiny standalone `main()` for `python gdn_prefill_scan.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# ── make `recurrent_scan`'s lone `pl.pallas_call` print + name itself ────────
# `gdn_fast.py` does `from jax.experimental import pallas as pl`, so it resolves
# `pl.pallas_call` off the *same* module object at call time — patching it here
# (before any lowering) is enough. One subprocess per kernel ⇒ no cross-talk.
_orig_pallas_call = pl.pallas_call


def _pallas_call_harness(*args: Any, **kwargs: Any):  # noqa: ANN401
    kwargs.setdefault("debug", True)
    kwargs.setdefault("name", "recurrent_scan")
    return _orig_pallas_call(*args, **kwargs)


pl.pallas_call = _pallas_call_harness  # type: ignore[assignment]

from gdn_fast import recurrent_scan  # noqa: E402  (after the monkeypatch)


_DTYPE_MAP = {
    "i8": jnp.int8, "i16": jnp.int16, "i32": jnp.int32,
    "f16": jnp.float16, "bf16": jnp.bfloat16, "f32": jnp.float32,
}


def _dtype(name: Any, default: Any) -> Any:  # noqa: ANN401
    return _DTYPE_MAP.get(name, default) if isinstance(name, str) else default


def _prefill_shapes(cfg: Dict[str, Any]):
    """(ShapeDtypeStruct args, static_kwargs={n_kq,...}, + a `_meta` echo) for
    `recurrent_scan`.

    cfg keys (all optional):
      dtype / state_dtype : "bf16" | "f16" | "f32"
      num_k_heads (n_kq) / num_v_heads (n_v) / head_k_dim (d_k) / head_v_dim (d_v)
      chunk_size  : v2 chunk size; BT (decode batch) == chunk_size (mult. of 16)
      num_tokens  : total scheduled tokens this step (>= num_seqs)
      num_seqs    : number of requests (decode + prefill)
      num_blocks  : recurrent-state pool size (default num_seqs + 1)
    """
    dtype = _dtype(cfg.get("dtype"), jnp.bfloat16)
    state_dtype = _dtype(cfg.get("state_dtype"), jnp.bfloat16)
    n_kq = int(cfg.get("num_k_heads", 16))
    n_v = int(cfg.get("num_v_heads", 32))
    d_k = int(cfg.get("head_k_dim", 128))
    d_v = int(cfg.get("head_v_dim", 128))
    chunk_size = int(cfg.get("chunk_size", 64))
    num_tokens = int(cfg.get("num_tokens", 388))
    num_seqs = int(cfg.get("num_seqs", 6))
    if n_v % n_kq != 0:
        raise ValueError(f"n_v={n_v} must be a multiple of n_kq={n_kq}")
    if chunk_size % 16 != 0:
        raise ValueError(f"chunk_size={chunk_size} must be a multiple of 16 "
                         "(invert_triangular_matrix block_size=16)")
    if num_tokens < num_seqs:
        raise ValueError(f"num_tokens={num_tokens} must be >= num_seqs={num_seqs}")
    key_dim = n_kq * d_k
    dim = 2 * key_dim + n_v * d_v
    _nb = cfg.get("num_blocks")
    num_blocks = (num_seqs + 1) if _nb is None else int(_nb)
    if num_blocks < num_seqs + 1:
        raise ValueError(f"num_blocks={num_blocks} must be >= num_seqs+1={num_seqs + 1}")

    mixed_qkv = jax.ShapeDtypeStruct((num_tokens, dim), dtype)
    b = jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32)
    a = jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32)
    recurrent_state = jax.ShapeDtypeStruct((num_blocks, n_v, d_k, d_v), state_dtype)
    A_log = jax.ShapeDtypeStruct((n_v,), jnp.float32)
    dt_bias = jax.ShapeDtypeStruct((n_v,), jnp.float32)
    query_start_loc = jax.ShapeDtypeStruct((num_seqs + 1,), jnp.int32)
    state_indices = jax.ShapeDtypeStruct((num_seqs,), jnp.int32)
    distribution = jax.ShapeDtypeStruct((3,), jnp.int32)

    args = [mixed_qkv, b, a, recurrent_state, A_log, dt_bias,
            query_start_loc, state_indices, distribution]
    static_kwargs: Dict[str, Any] = {
        "n_kq": n_kq, "n_v": n_v, "d_k": d_k, "d_v": d_v,
        "chunk_size": chunk_size, "BT": chunk_size, "use_qk_norm_in_gdn": True,
    }
    static_kwargs["_meta"] = {
        "num_tokens": num_tokens, "dim": dim, "num_seqs": num_seqs,
        "num_blocks": num_blocks, "chunk_size": chunk_size,
    }
    return args, static_kwargs


__all__ = ["recurrent_scan", "build_args"]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Harness entry: (args, static_kwargs) for `recurrent_scan`. `_meta` stripped."""
    args, static_kwargs = _prefill_shapes(cfg)
    static_kwargs = {k: v for k, v in static_kwargs.items() if k != "_meta"}
    return args, static_kwargs


def main() -> None:
    import argparse
    import contextlib
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--state-dtype", default="bf16")
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--num-tokens", type=int, default=388)
    p.add_argument("--num-seqs", type=int, default=6)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--print-mlir", action="store_true",
                   help="print compiler_ir() (StableHLO). The Mosaic module is "
                        "printed by debug=True regardless.")
    args = p.parse_args()

    cfg: Dict[str, Any] = {
        "dtype": args.dtype, "state_dtype": args.state_dtype,
        "num_k_heads": args.num_k_heads, "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim, "head_v_dim": args.head_v_dim,
        "chunk_size": args.chunk_size,
        "num_tokens": args.num_tokens, "num_seqs": args.num_seqs,
    }
    if args.num_blocks is not None:
        cfg["num_blocks"] = args.num_blocks

    pos_args, static_full = _prefill_shapes(cfg)
    meta = static_full.pop("_meta")

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

    static_names = list(static_full.keys())
    with _tpu_abstract_mesh_context():
        lowered = jax.jit(recurrent_scan, backend="cpu",
                          static_argnames=tuple(static_names)).lower(
                              *pos_args, **static_full)

    if args.print_mlir:
        print(lowered.compiler_ir())
    else:
        print(f"lowered ok: {meta}")


if __name__ == "__main__":
    main()
