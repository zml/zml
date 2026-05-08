"""Mosaic-TPU reference-IR generator. Loaded as a long-lived subprocess.
Lowers `pl.pallas_call(...)` from a CPU host via an abstract TPU mesh.
The user's kernel must set `debug=True` on its `pl.pallas_call` so the
post-lowering Mosaic module prints to stdout — we capture and extract it."""

from __future__ import annotations

import contextlib
import io
import os
import re
from typing import Any, Dict

import runtime


_MODULE_RE = re.compile(r"(?ms)^module \{$.*?^\}$")


def _setup() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Register the TPU lowering rule on the cpu backend so `jax.jit(...,
    # backend="cpu").lower(...)` invokes Mosaic on a non-TPU host.
    from jax._src.interpreters import mlir
    from jax._src.pallas import pallas_call

    def _wrapper(ctx, *in_nodes, **params):
        params.pop("backend", None)
        params.pop("which_linear", None)
        if "out_shapes" in params:
            params["out_avals"] = params.pop("out_shapes")
        from jax._src.pallas.mosaic.pallas_call_registration import (
            pallas_call_tpu_lowering_rule,
        )

        return pallas_call_tpu_lowering_rule(ctx, *in_nodes, **params)

    mlir.register_lowering(pallas_call.pallas_call_p, _wrapper, platform="cpu")


@contextlib.contextmanager
def _abstract_tpu_mesh():
    import jax

    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1,),
        axis_names=("tpu_core",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU v5 lite",
            num_cores=1,
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        yield


def _compile_one(kernel_fn, build_args, cfg: Dict[str, Any]) -> Dict[str, str]:
    import jax

    args, static_kwargs = build_args(cfg)
    static_argnames = list(static_kwargs.keys())

    # `debug=True` on the kernel's pallas_call makes JAX print the jaxpr
    # (and the post-lowering Mosaic module on newer JAX) to stdout —
    # capture so it doesn't leak onto our JSON protocol stdout.
    capture = io.StringIO()
    with _abstract_tpu_mesh(), contextlib.redirect_stdout(capture):
        jax.jit(
            kernel_fn,
            backend="cpu",
            static_argnames=static_argnames,
        ).lower(*args, **static_kwargs)

    matches = _MODULE_RE.findall(capture.getvalue())
    # Older JAX (0.9.x) doesn't print the Mosaic module — return empty
    # so the harness records this as `missing_py` instead of garbage.
    return {"mosaic": matches[-1]} if matches else {}


if __name__ == "__main__":
    raise SystemExit(runtime.serve_kernel(_setup, _compile_one))
