#!/usr/bin/env python3
"""Fixture integrity check for LTX block-0 parity fixtures.

Verifies that every tensor in a safetensors fixture is:
  - dtype bfloat16
  - For token-major keys (shape [B, T, D]): D == 4096
  - When --token-limit is given: T == token_limit for token-major keys
    (skips gate tensors whose T==1 by design)

Usage:
    python check_fixture_integrity.py <fixture.safetensors> [--token-limit T]

Exit 0 on success, 1 on any integrity failure.
"""
import argparse
import sys
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check safetensors fixture dtype/shape integrity")
    p.add_argument("fixture", type=Path, help="Path to fixture .safetensors file")
    p.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Expected token axis size for [B, T, D] tensors (skips T==1 gate tensors)",
    )
    return p.parse_args()


# Suffixes of keys whose tensor has a token axis at dim 1.
_TOKEN_MAJOR_SUFFIXES = (
    ".norm_vx",
    ".attn1_out",
    ".attn2_x",
    ".attn2_out",
    ".context",
    ".vx_in",
    ".vx_out",
    ".text_ca_out",
    ".ff_out",
    ".pe_cos",
    ".pe_sin",
)

_MODEL_DIM = 4096


def main() -> None:
    args = parse_args()

    if not args.fixture.exists():
        print(f"ERROR: fixture file not found: {args.fixture}", file=sys.stderr)
        sys.exit(1)

    errors: list[str] = []
    print(f"Fixture : {args.fixture}")
    if args.token_limit is not None:
        print(f"Expected: dtype=bfloat16  D={_MODEL_DIM}  T={args.token_limit}")
    else:
        print(f"Expected: dtype=bfloat16  D={_MODEL_DIM}")
    print()

    with safe_open(str(args.fixture), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        max_key_len = max(len(k) for k in keys)
        for k in keys:
            t = f.get_tensor(k)
            dtype_str = str(t.dtype)
            shape = list(t.shape)
            tag = ""

            if dtype_str != "torch.bfloat16":
                errors.append(f"  DTYPE FAIL  {k}: expected bfloat16, got {dtype_str}")
                tag += " [DTYPE!]"

            if any(k.endswith(s) for s in _TOKEN_MAJOR_SUFFIXES) and t.ndim == 3:
                D = shape[2]
                T = shape[1]
                if D != _MODEL_DIM:
                    errors.append(f"  DIM FAIL    {k}: D={D} expected {_MODEL_DIM}")
                    tag += f" [D={D}!]"
                if args.token_limit is not None and T not in (1, args.token_limit):
                    errors.append(f"  TOKEN FAIL  {k}: T={T} expected {args.token_limit}")
                    tag += f" [T={T}!]"

            print(f"  {k:<{max_key_len}}  shape={str(shape):<20}  {dtype_str}{tag}")

    print()
    if errors:
        print("INTEGRITY CHECK FAILED:")
        for e in errors:
            print(e)
        sys.exit(1)

    summary = f"{len(keys)} keys, all bfloat16"
    if args.token_limit is not None:
        summary += f", T={args.token_limit}"
    print(f"✓ Integrity OK  ({summary})")


if __name__ == "__main__":
    main()
