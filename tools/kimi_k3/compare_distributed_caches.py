#!/usr/bin/env python3
"""Compare reconstructed GPU-0 and TP4 four-layer session caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MINIMUM_CLOSE_FRACTION = 0.995
# dumpSessionCache writes all four tensors for each KDA layer before advancing
# to the next layer. Keep the public report ordered the same way.
def segments(
    capacity: int,
    kda_count: int = 3,
    mla_count: int = 1,
) -> tuple[tuple[str, str, int, float, float], ...]:
    if capacity < 1:
        raise ValueError(f"cache capacity must be positive: {capacity}")
    if kda_count < 0 or mla_count < 0 or kda_count + mla_count == 0:
        raise ValueError(
            f"cache family counts must be non-negative and non-empty: "
            f"kda={kda_count} mla={mla_count}"
        )
    kda_segments = tuple(
        item
        for ordinal in range(kda_count)
        for item in (
            (f"kda{ordinal}.q_conv", "bf16", 1 * 12_288 * 4, 5e-2, 2e-2),
            (f"kda{ordinal}.k_conv", "bf16", 1 * 12_288 * 4, 5e-2, 2e-2),
            (f"kda{ordinal}.v_conv", "bf16", 1 * 12_288 * 4, 5e-2, 2e-2),
            (f"kda{ordinal}.recurrent", "f32", 1 * 96 * 128 * 128, 5e-3, 2e-2),
        )
    )
    mla_segments = tuple(
        item
        for ordinal in range(mla_count)
        for item in (
            (f"mla{ordinal}.compressed", "bf16", capacity * 512, 5e-2, 2e-2),
            (f"mla{ordinal}.extra_key", "bf16", capacity * 64, 5e-2, 2e-2),
        )
    )
    return kda_segments + mla_segments


SEGMENTS = segments(1)


def _bf16(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (words << 16).view(np.float32)


def compare(
    gpu0_path: Path,
    tp4_path: Path,
    capacity: int = 1,
    kda_count: int = 3,
    mla_count: int = 1,
) -> dict[str, Any]:
    gpu0_raw = gpu0_path.read_bytes()
    tp4_raw = tp4_path.read_bytes()
    cache_segments = segments(capacity, kda_count, mla_count)
    expected_bytes = sum(count * (2 if dtype == "bf16" else 4) for _, dtype, count, _, _ in cache_segments)
    if len(gpu0_raw) != expected_bytes or len(tp4_raw) != expected_bytes:
        raise ValueError(
            f"cache byte-size mismatch: expected={expected_bytes} "
            f"gpu0={len(gpu0_raw)} tp4={len(tp4_raw)}"
        )

    offset = 0
    records: list[dict[str, Any]] = []
    for name, dtype, count, atol, rtol in cache_segments:
        byte_count = count * (2 if dtype == "bf16" else 4)
        gpu0_bytes = gpu0_raw[offset : offset + byte_count]
        tp4_bytes = tp4_raw[offset : offset + byte_count]
        gpu0 = _bf16(gpu0_bytes) if dtype == "bf16" else np.frombuffer(gpu0_bytes, dtype=np.float32)
        tp4 = _bf16(tp4_bytes) if dtype == "bf16" else np.frombuffer(tp4_bytes, dtype=np.float32)
        finite = np.isfinite(gpu0) & np.isfinite(tp4)
        absolute = np.abs(gpu0 - tp4)
        close = finite & (absolute <= atol + rtol * np.abs(gpu0))
        close_fraction = float(close.sum() / count)
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "elements": count,
                "atol": atol,
                "rtol": rtol,
                "minimum_close_fraction": MINIMUM_CLOSE_FRACTION,
                "finite_fraction": float(finite.sum() / count),
                "close_fraction": close_fraction,
                "max_abs": float(absolute[finite].max(initial=0)),
                "mean_abs": float(absolute[finite].mean() if finite.any() else 0),
                "passed": bool(finite.all() and close_fraction >= MINIMUM_CLOSE_FRACTION),
            }
        )
        offset += byte_count

    result = {
        "schema_version": 1,
        "status": "pass" if all(record["passed"] for record in records) else "fail",
        "kda_count": kda_count,
        "mla_count": mla_count,
        "scope": (
            "four_layer_prefill_cache_gpu0_vs_tp4_ep1"
            if capacity == 1
            else "four_layer_continuation_cache_gpu0_vs_tp4_ep1"
        ),
        "capacity": capacity,
        "bytes_per_dump": expected_bytes,
        "gpu0_sha256": hashlib.sha256(gpu0_raw).hexdigest(),
        "tp4_sha256": hashlib.sha256(tp4_raw).hexdigest(),
        "exact_byte_match": gpu0_raw == tp4_raw,
        "minimum_close_fraction": min(record["close_fraction"] for record in records),
        "maximum_abs": max(record["max_abs"] for record in records),
        "segments": records,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu0", type=Path, required=True)
    parser.add_argument("--tp4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--kda-count", type=int, default=3)
    parser.add_argument("--mla-count", type=int, default=1)
    args = parser.parse_args()
    result = compare(args.gpu0, args.tp4, args.capacity, args.kda_count, args.mla_count)
    if result["status"] != "pass":
        failed = [record["name"] for record in result["segments"] if not record["passed"]]
        raise SystemExit(f"distributed cache comparison failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
