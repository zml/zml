from __future__ import annotations

from pathlib import Path

import pytest

from compare_distributed_caches import SEGMENTS, compare


def cache_bytes() -> int:
    return sum(count * (2 if dtype == "bf16" else 4) for _, dtype, count, _, _ in SEGMENTS)


def test_identical_zero_caches_pass(tmp_path: Path) -> None:
    raw = bytes(cache_bytes())
    gpu0 = tmp_path / "gpu0.bin"
    tp4 = tmp_path / "tp4.bin"
    gpu0.write_bytes(raw)
    tp4.write_bytes(raw)
    result = compare(gpu0, tp4)
    assert result["status"] == "pass"
    assert result["exact_byte_match"] is True
    assert result["minimum_close_fraction"] == 1.0
    assert len(result["segments"]) == 14


def test_wrong_cache_size_fails_closed(tmp_path: Path) -> None:
    gpu0 = tmp_path / "gpu0.bin"
    tp4 = tmp_path / "tp4.bin"
    gpu0.write_bytes(b"short")
    tp4.write_bytes(b"short")
    with pytest.raises(ValueError, match="cache byte-size mismatch"):
        compare(gpu0, tp4)
