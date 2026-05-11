"""Triton-side reference-IR generator. Loaded as a long-lived subprocess
by `harness/py_runner.zig`. The user's `--kernel-module` must expose a
`@triton.jit` function named `--kernel-fn` plus `build_args(cfg) ->
(positional, kwargs)`. The `fake_plugin` driver lets `warmup` lower
without launching a Python-side kernel."""

from __future__ import annotations

import os
from typing import Any, Dict

import runtime


def _setup() -> None:
    # Must be set before triton import. Otherwise `loc(...)` annotations
    # pick up bazel sandbox paths that differ between runs and drown the
    # diff in noise.
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")

    from triton.runtime.driver import driver as runtime_driver

    from fake_plugin.driver import Driver as FakeDriver

    runtime_driver.set_active(FakeDriver())


def _compile_one(jit_fn, build_args, cfg: Dict[str, Any]) -> Dict[str, str]:
    args, kwargs = build_args(cfg)
    kwargs.setdefault("grid", (1,))
    compiled = jit_fn.warmup(*args, **kwargs)
    # Only TTIR is consumed Zig-side — post-XLA TTGIR/LLIR/PTX are
    # extracted from the XLA dump dir, not from this response.
    blob = (getattr(compiled, "asm", {}) or {}).get("ttir")
    if blob is None:
        return {}
    return {"ttir": blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)}


if __name__ == "__main__":
    raise SystemExit(runtime.serve_kernel(_setup, _compile_one))
