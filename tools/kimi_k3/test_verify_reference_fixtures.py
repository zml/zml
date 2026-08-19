from __future__ import annotations

import json
from pathlib import Path

import pytest

from verify_reference_fixtures import EXPECTED, FixtureError, verify


def test_missing_aggregate_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="missing aggregate"):
        verify(tmp_path, None)


def test_incomplete_fixture_set_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"fixtures": []}))
    with pytest.raises(FixtureError, match="fixture set mismatch"):
        verify(tmp_path, None)


def test_expected_fixture_contract() -> None:
    assert EXPECTED == {
        "s0-operators",
        "s1-layer0-len1",
        "s1-layer0-len4",
        "s1-layer0-len8",
        "s1-layer0-len16",
        "s1-layer0-prefill4-decode1",
    }
