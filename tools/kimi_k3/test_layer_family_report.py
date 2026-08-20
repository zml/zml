from __future__ import annotations

import pytest

from summarize_layer_family_report import EXPECTED, summarize


def manifest():
    return {
        "device": "NVIDIA H100 80GB HBM3",
        "moonshot_revision": "revision",
        "tensor_file_sha256": "file-hash",
        "tensor_semantic_sha256": "semantic-hash",
        "timing": {"repeat": {"gpu_ms": 1.0}},
    }


def valid_log() -> str:
    rows = []
    for (layer, mode), (attention, experts, boundaries) in EXPECTED.items():
        extra = (
            " warm_compile_us=7 warm_execute_us=8"
            if layer == 3 and mode == "decode"
            else ""
        )
        rows.append(
            "KIMI_K3_LAYER_FAMILY_PASS "
            f"layer={layer} attention={attention} mode={mode} experts={experts} "
            f"boundaries={boundaries}{extra} load_us=1 compile_us=2 execute_us=3"
        )
    rows.append(
        "KIMI_K3_LAYER_FAMILY_ALL_PASS "
        "layers=1,2,3 prefill=3 decode=3 backend=cuda global_routes=exact"
    )
    return "\n".join(rows) + "\n"


def test_summarize_accepts_complete_cuda_inventory():
    report = summarize(valid_log(), manifest())
    assert report["status"] == "PASS"
    assert report["backend"] == "cuda"
    assert report["global_route_sets"] == "exact"
    assert len(report["cuda_cases"]) == 6
    decode = next(
        case
        for case in report["cuda_cases"]
        if case["layer"] == 3 and case["mode"] == "decode"
    )
    assert decode["warm_compile_us"] == 7
    assert decode["execute_us"] == 3


def test_summarize_rejects_missing_case():
    lines = valid_log().splitlines()
    with pytest.raises(ValueError, match="incomplete"):
        summarize("\n".join(lines[1:]) + "\n", manifest())


def test_summarize_rejects_non_cuda_completion():
    log = valid_log().replace("backend=cuda", "backend=cpu")
    with pytest.raises(ValueError, match="completion"):
        summarize(log, manifest())


def test_summarize_rejects_inventory_drift():
    log = valid_log().replace("experts=61", "experts=60", 1)
    with pytest.raises(ValueError, match="inventory"):
        summarize(log, manifest())
