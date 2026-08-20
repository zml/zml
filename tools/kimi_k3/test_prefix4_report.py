from pathlib import Path

from summarize_prefix4_report import summarize


def test_summarize_prefix4_report(tmp_path: Path) -> None:
    lines = ["KIMI_K3_PREFIX4_REFERENCE_HEAD_PASS inputs=official tolerance=strict"]
    lines += [
        f"KIMI_K3_PREFIX4_LAYER_PASS layer={layer} attention={'kda_dense' if layer == 0 else 'mla' if layer == 3 else 'kda'}"
        f"{' experts=896' if layer else ''} load_us={100 + layer} compile_us={200 + layer} execute_us={300 + layer} routing=global"
        for layer in range(4)
    ]
    for prefix, matched, total in (
        ("layer1", 64, 64), ("layer1.decode", 16, 16),
        ("layer2", 63, 64), ("layer2.decode", 16, 16),
        ("layer3", 62, 64), ("layer3.decode", 14, 16),
    ):
        lines.append(
            f"KIMI_K3_PREFIX4_ROUTE_OVERLAP prefix={prefix} matched={matched} total={total} fraction={matched / total:.4f}"
        )
    for prefix in ("prefix", "decode"):
        lines.append(
            f"KIMI_K3_PREFIX4_GREEDY_TIE prefix={prefix} actual_token=8821 official_token=95385 "
            "actual_logit=8.75 official_logit_at_actual=8.5625 actual_logit_at_official=8.75 "
            "actual_max=8.75 official_max=8.6875"
        )
    lines += [
        "KIMI_K3_PREFIX4_HEAD_PASS load_us=400 compile_us=500 execute_us=600",
        "KIMI_K3_PREFIX4_ALL_PASS layers=0,1,2,3 experts_per_moe=896 prefill=4 warm=3 decode=1 backend=cuda routing=global",
    ]
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    report = summarize(
        "\n".join(lines),
        {
            "tensor_file_sha256": "a" * 64,
            "tensor_semantic_sha256": "b" * 64,
            "prefill_greedy_token": 95385,
            "decode_greedy_token": 95385,
        },
        trace,
    )
    assert report["status"] == "PASS"
    assert report["layers"]["3"]["experts"] == 896
    assert report["routing"]["chained_overlap"]["layer3.decode"]["matched"] == 14
    assert report["head"]["greedy"]["prefix"]["acceptance"] == "maximum_tie_sets_intersect"
