"""Shared runtime support for both runner.py files: the JSON RPC loop
that drives Triton/Pallas lowering on demand. Diffing lives Zig-side."""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
from typing import Any, Callable, Dict


def serve(compile_handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Read JSON requests on stdin, write JSON responses on stdout, until
    EOF. `compile_handler(cfg)` returns the asm dict. Exceptions become
    `{"ok": false, "error": "..."}`."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            response = {"ok": True, **compile_handler(req.get("cfg", {}))}
        except Exception as exc:  # surface every failure to Zig
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


def serve_kernel(
    setup: Callable[[], None],
    compile_one: Callable[[Any, Callable[..., Any], Dict[str, Any]], Dict[str, str]],
) -> int:
    """Entry-point shared by `triton/runner.py` and `mosaic_tpu/runner.py`.
    Parses `--kernel-module`/`--kernel-fn`, runs `setup()` once for
    backend-specific env + plugin registration, imports the kernel
    module, then loops via `serve(compile_one)`. `compile_one` receives
    `(fn, build_args, cfg)`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-module", required=True)
    parser.add_argument("--kernel-fn", required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    setup()

    mod = importlib.import_module(args.kernel_module)
    if not hasattr(mod, "build_args"):
        raise AttributeError(
            f"module {args.kernel_module!r} must export `build_args(cfg)`"
        )
    fn = getattr(mod, args.kernel_fn)

    serve(lambda cfg: compile_one(fn, mod.build_args, cfg))
    return 0
