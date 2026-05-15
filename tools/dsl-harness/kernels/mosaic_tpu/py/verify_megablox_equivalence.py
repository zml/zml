#!/usr/bin/env python3
"""Verify non-Qwix (unquantized) equivalence between:

- Original MaxText gmm kernel
- Harness gmm adapter kernel

The script intentionally feeds plain JAX arrays only, so it checks the
unquantized path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import pathlib
import re
import sys
import types

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def find_workspace_root() -> pathlib.Path:
    """Find workspace root containing both maxtext/ and zml/."""
    here = pathlib.Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "maxtext").exists() and (candidate / "zml").exists():
            return candidate
    raise RuntimeError("Could not locate workspace root containing maxtext/ and zml/")


def install_qwix_stub_if_missing() -> None:
    """Install a minimal qwix.pallas stub for importing original backend.

    The equivalence test still exercises only non-Qwix paths because inputs are
    plain JAX arrays. The stub is just to satisfy import-time dependency.
    """
    try:
        import qwix.pallas  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    qwix_pkg = types.ModuleType("qwix")
    qwix_pallas = types.ModuleType("qwix.pallas")

    class DummyQArray:
        pass

    def dot_general(lhs, rhs, *, preferred_element_type, dimension_numbers):
        return jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type,
        )

    def pallas_call(*args, **kwargs):
        return pl.pallas_call(*args, **kwargs)

    def dot(lhs, rhs, *, preferred_element_type):
        return jnp.dot(lhs, rhs, preferred_element_type=preferred_element_type)

    qwix_pallas.QArray = DummyQArray
    qwix_pallas.dot_general = dot_general
    qwix_pallas.pallas_call = pallas_call
    qwix_pallas.dot = dot

    sys.modules["qwix"] = qwix_pkg
    sys.modules["qwix.pallas"] = qwix_pallas


def import_module_from_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stablehlo_signature(fn, *args, **kwargs) -> tuple[collections.Counter, str]:
    lowered = fn.lower(*args, **kwargs)
    ir = lowered.compiler_ir(dialect="stablehlo")
    text = str(ir)
    ops = re.findall(r"\b(stablehlo\.[A-Za-z0-9_]+)\b", text)
    return collections.Counter(ops), hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_case(
    seed: int,
    m: int,
    k: int,
    n: int,
    group_sizes: list[int],
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    existing_out: bool,
) -> dict:
    return {
        "seed": seed,
        "m": m,
        "k": k,
        "n": n,
        "group_sizes": group_sizes,
        "tiling": tiling,
        "transpose_rhs": transpose_rhs,
        "existing_out": existing_out,
    }


def run_case(
    case: dict,
    orig_mod,
    new_mod,
    atol: float,
    rtol: float,
    check_ir: bool,
) -> tuple[bool, str]:
    m, k, n = int(case["m"]), int(case["k"]), int(case["n"])
    tm, tk, tn = tuple(case["tiling"])
    transpose_rhs = bool(case["transpose_rhs"])
    use_existing_out = bool(case["existing_out"])

    group_sizes = jnp.asarray(case["group_sizes"], dtype=jnp.int32)
    num_groups = int(group_sizes.shape[0])
    if int(group_sizes.sum()) != m:
        return False, f"invalid case: sum(group_sizes)={int(group_sizes.sum())} != m={m}"

    key = jax.random.PRNGKey(int(case["seed"]))
    k1, k2, k3 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
    rhs_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    rhs = jax.random.normal(k2, rhs_shape, dtype=jnp.float32)
    existing = jax.random.normal(k3, (m, n), dtype=jnp.float32) if use_existing_out else None

    kwargs = {
        "preferred_element_type": jnp.float32,
        "tiling": (tm, tk, tn),
        "group_offset": None,
        "existing_out": existing,
        "transpose_rhs": transpose_rhs,
        "interpret": True,
    }

    out_ref = orig_mod.gmm(lhs, rhs, group_sizes, **kwargs)
    out_new = new_mod.gmm(lhs, rhs, group_sizes, **kwargs)

    out_ref.block_until_ready()
    out_new.block_until_ready()

    if out_ref.shape != out_new.shape:
        return False, f"shape mismatch: {out_ref.shape} vs {out_new.shape}"
    if out_ref.dtype != out_new.dtype:
        return False, f"dtype mismatch: {out_ref.dtype} vs {out_new.dtype}"

    diff = jnp.abs(out_ref - out_new)
    max_abs = float(jnp.max(diff))
    denom = jnp.maximum(jnp.abs(out_ref), jnp.array(1e-12, dtype=out_ref.dtype))
    max_rel = float(jnp.max(diff / denom))
    close = bool(jnp.allclose(out_ref, out_new, atol=atol, rtol=rtol))

    detail = f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, allclose={close}"

    if check_ir:
        ops_ref, hash_ref = stablehlo_signature(orig_mod.gmm, lhs, rhs, group_sizes, **kwargs)
        ops_new, hash_new = stablehlo_signature(new_mod.gmm, lhs, rhs, group_sizes, **kwargs)
        detail += (
            f", stablehlo_ops_equal={ops_ref == ops_new}"
            f", hash_ref={hash_ref[:12]}"
            f", hash_new={hash_new[:12]}"
        )

    return close, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare original vs harness gmm (non-Qwix path)")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--check-ir", action="store_true", help="also compare StableHLO op histograms")
    parser.add_argument(
        "--orig",
        type=str,
        default="",
        help="optional absolute path to original backend.py",
    )
    parser.add_argument(
        "--new",
        type=str,
        default="",
        help="optional absolute path to new megablox_gmm.py",
    )
    args = parser.parse_args()

    root = find_workspace_root()
    orig_path = pathlib.Path(args.orig) if args.orig else root / "maxtext/src/maxtext/kernels/megablox/backend.py"
    new_path = pathlib.Path(args.new) if args.new else root / "zml/tools/dsl-harness/kernels/mosaic_tpu/py/megablox_gmm.py"

    if not orig_path.exists():
        raise FileNotFoundError(f"Original file not found: {orig_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New file not found: {new_path}")

    install_qwix_stub_if_missing()

    orig_mod = import_module_from_path("orig_megablox_backend", orig_path)
    new_mod = import_module_from_path("new_megablox_gmm", new_path)

    # Covers:
    # - transpose rhs on/off
    # - existing_out on/off
    # - zero-sized group edge case
    cases = [
        make_case(0, 256, 192, 128, [64, 0, 96, 96], (64, 64, 64), False, False),
        make_case(1, 256, 192, 128, [64, 0, 96, 96], (64, 64, 64), False, True),
        make_case(2, 256, 192, 128, [64, 0, 96, 96], (64, 64, 64), True, False),
        make_case(3, 256, 192, 128, [64, 0, 96, 96], (64, 64, 64), True, True),
    ]

    failures = 0
    print("Running non-Qwix equivalence checks...")
    print(f"original: {orig_path}")
    print(f"new     : {new_path}")

    for i, case in enumerate(cases):
        ok, detail = run_case(case, orig_mod, new_mod, args.atol, args.rtol, args.check_ir)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case={i} {detail}")
        if not ok:
            failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s) mismatched")
        return 1

    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
