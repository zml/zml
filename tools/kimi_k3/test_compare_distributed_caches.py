from __future__ import annotations

from pathlib import Path

import pytest

from compare_distributed_caches import SEGMENTS, compare, segments


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


def test_capacity_two_cache_layout_passes(tmp_path: Path) -> None:
    capacity = 2
    sized_segments = segments(capacity)
    raw = bytes(sum(count * (2 if dtype == "bf16" else 4) for _, dtype, count, _, _ in sized_segments))
    gpu0 = tmp_path / "gpu0-capacity2.bin"
    tp4 = tmp_path / "tp4-capacity2.bin"
    gpu0.write_bytes(raw)
    tp4.write_bytes(raw)
    result = compare(gpu0, tp4, capacity)
    assert result["status"] == "pass"
    assert result["capacity"] == capacity
    assert result["scope"] == "four_layer_continuation_cache_gpu0_vs_tp4_ep1"
    assert result["bytes_per_dump"] == cache_bytes() + 2 * (512 + 64)


def test_configurable_kda_mla_layout_passes(tmp_path: Path) -> None:
    capacity = 3
    kda_count = 2
    mla_count = 2
    configured = segments(capacity, kda_count, mla_count)
    raw = bytes(
        sum(
            count * (2 if dtype == "bf16" else 4)
            for _, dtype, count, _, _ in configured
        )
    )
    gpu0 = tmp_path / "gpu0-configured.bin"
    distributed = tmp_path / "distributed-configured.bin"
    gpu0.write_bytes(raw)
    distributed.write_bytes(raw)
    result = compare(
        gpu0,
        distributed,
        capacity,
        kda_count,
        mla_count,
    )
    assert result["status"] == "pass"
    assert result["kda_count"] == kda_count
    assert result["mla_count"] == mla_count
    assert len(result["segments"]) == kda_count * 4 + mla_count * 2


def test_wrong_cache_size_fails_closed(tmp_path: Path) -> None:
    gpu0 = tmp_path / "gpu0.bin"
    tp4 = tmp_path / "tp4.bin"
    gpu0.write_bytes(b"short")
    tp4.write_bytes(b"short")
    with pytest.raises(ValueError, match="cache byte-size mismatch"):
        compare(gpu0, tp4)
