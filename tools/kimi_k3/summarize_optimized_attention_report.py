#!/usr/bin/env python3
"""Build and enforce the Milestone 18 optimized-attention report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KDA_CASES = {
    "small_s1", "small_s3", "small_s4", "small_s5", "small_s31",
    "small_s32", "small_s33", "small_s63", "small_s64", "small_s65",
    "small_s257", "production_decode", "production_prefill64",
}
MLA_CASES = {
    "capacity1_valid1", "capacity32_valid31", "capacity32_valid32",
    "capacity64_valid33", "capacity64_valid63", "capacity64_valid64",
    "capacity128_valid65", "capacity128_valid127", "capacity128_valid128",
    "capacity4096_valid4096",
}


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", line))


def parse_kda(text: str) -> dict:
    rows: dict[str, dict] = {}
    for line in text.splitlines():
        if line.startswith("KIMI_K3_KDA_OPT_PASS "):
            row = fields(line)
            name = row["case"]
            rows[name] = {
                "sequence": int(row["sequence"]),
                "optimized_execute_us": int(row["optimized_execute_us"]),
                "reference_execute_us": int(row["reference_execute_us"]),
            }
    if set(rows) != KDA_CASES or "KIMI_K3_KDA_OPT_ALL_PASS" not in text:
        raise ValueError(f"incomplete optimized KDA cases: {sorted(rows)}")
    for name in ("production_decode", "production_prefill64"):
        row = rows[name]
        if row["optimized_execute_us"] * 100 > row["reference_execute_us"] * 105:
            raise ValueError(f"KDA regression budget exceeded: {name}")
        row["speedup"] = row["reference_execute_us"] / row["optimized_execute_us"]
    errors = [float(value) for value in re.findall(r"max_absolute_error:\s*([0-9.eE+-]+)", text)]
    return {
        "cases": rows,
        "max_recorded_absolute_error": max(errors, default=0.0),
        "all_finite": "finite=true" in text and "nan_or_inf: true" not in text,
    }


def parse_mla(text: str) -> dict:
    rows: dict[str, dict] = {}
    benches: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("KIMI_K3_MLA_OPT_BENCH "):
            row = fields(line)
            benches[row["case"]] = int(row["mean_execute_us"])
        if line.startswith("KIMI_K3_MLA_OPT_PASS "):
            row = fields(line)
            rows[row["case"]] = {
                "capacity": int(row["capacity"]),
                "valid_tokens": int(row["valid_tokens"]),
                "execute_us": int(row["execute_us"]),
            }
    if set(rows) != MLA_CASES or "KIMI_K3_MLA_OPT_ALL_PASS" not in text:
        raise ValueError(f"incomplete optimized MLA cases: {sorted(rows)}")
    ceilings = {"capacity64_valid64": 750, "capacity4096_valid4096": 900}
    for name, ceiling in ceilings.items():
        if benches.get(name, ceiling + 1) > ceiling:
            raise ValueError(f"MLA regression ceiling exceeded: {name}")
    return {"cases": rows, "benchmarks_us": benches, "ceilings_us": ceilings, "all_finite": "finite=true" in text}


def trace_summary(path: Path, family_span: str) -> dict:
    events = json.loads(path.read_text()).get("traceEvents", [])
    names = [str(event.get("name", "")) for event in events]
    cuda = sum("cuda" in name.lower() or "gpu" in name.lower() for name in names)
    if not events or cuda == 0 or not any(family_span in name for name in names):
        raise ValueError(f"invalid family trace: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "events": len(events), "cuda_named_events": cuda, "family_span": family_span}


def summarize(args: argparse.Namespace) -> dict:
    official_kda = args.official_kda_log.read_text()
    mla_cache = args.mla_cache_log.read_text()
    layer_family = args.layer_family_log.read_text()
    if "KIMI_K3_KDA_PREFILL_ALL_PASS" not in official_kda:
        raise ValueError("prior official KDA fixture did not pass")
    if "KIMI_K3_MLA_CACHE_ALL_PASS" not in mla_cache or "KIMI_K3_MLA_SESSION_CACHE_PASS reset=1" not in mla_cache:
        raise ValueError("prior latent MLA cache fixture did not pass")
    if "KIMI_K3_LAYER_FAMILY_ALL_PASS" not in layer_family:
        raise ValueError("real-weight layer-family fixture did not pass")

    kda_manifest = json.loads(args.kda_manifest.read_text())
    mla_manifest = json.loads(args.mla_manifest.read_text())
    memory = json.loads(args.memory_baseline.read_text())["memory"]
    if set(case["name"] for case in kda_manifest["cases"]) != KDA_CASES:
        raise ValueError("KDA manifest case inventory mismatch")
    if set(case["name"] for case in mla_manifest["cases"]) != MLA_CASES:
        raise ValueError("MLA manifest case inventory mismatch")
    if any(manifest.get("cpu_inference_fallback") is not False or manifest.get("checkpoint_downloaded") is not False for manifest in (kda_manifest, mla_manifest)):
        raise ValueError("fixture policy violation")
    if mla_manifest["tensor_file_sha256"] != "70595a6921f668872f7f963bab7a33b2fa5c6dfa4c0ace25c418d17d2e6c939a":
        raise ValueError("canonical MLA fixture SHA mismatch")

    kda = parse_kda(args.kda_log.read_text())
    mla = parse_mla(args.mla_log.read_text())
    kda_state_bytes = 96 * 128 * 128 * 4
    kda_conv_bytes = 3 * 12288 * 4 * 2
    return {
        "schema_version": 1,
        "milestone": 18,
        "status": "PASS",
        "backend": "cuda",
        "device": mla_manifest["device"],
        "cpu_inference_fallback": False,
        "checkpoint_downloaded": False,
        "official_reference": {"moonshot_revision": "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721", "kda_prefill_fixture": "PASS", "real_weight_layer_family": "PASS"},
        "kda": {
            **kda,
            "production_default": "recurrentOptimized",
            "test_reference": "recurrentReference",
            "cache": {
                "reference_bytes_batch1": kda_state_bytes + kda_conv_bytes,
                "optimized_bytes_batch1": kda_state_bytes + kda_conv_bytes,
                "state_layout": "b,h,v,k FP32 plus three BF16 convolution windows",
            },
            "budget": "optimized synchronized mean <= 105% of sequential reference",
        },
        "mla": {
            **mla,
            "production_default": "latentPrefillCompact/latentContinueCompact/latentSessionCompact",
            "test_references": "expanded prefill/decode and diagnostic latent paths",
            "cache": memory,
            "expanded_kv_materialized": False,
            "long_context_tokens": 4096,
            "rejected_custom_kernels": [
                {"kind": "scalar", "context": 64, "candidate_us": 474, "stablehlo_us": 443},
                {"kind": "scalar", "context": 4096, "candidate_us": 1180, "stablehlo_us": 525},
                {"kind": "tensor_core", "context": 64, "candidate_us": 554, "stablehlo_us": 521},
                {"kind": "tensor_core", "context": 4096, "candidate_us": 782, "stablehlo_us": 499},
            ],
        },
        "prior_correctness": {"official_kda": "PASS", "latent_mla_cache": "PASS", "real_weight_kda_mla_layers": "PASS"},
        "traces": {
            "kda": trace_summary(args.kda_trace, "kimi_k3.kda.optimized_case"),
            "mla": trace_summary(args.mla_trace, "kimi_k3.mla.optimized_case"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("official_kda_log", "kda_log", "mla_cache_log", "mla_log", "layer_family_log", "kda_manifest", "mla_manifest", "memory_baseline", "kda_trace", "mla_trace", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"], "kda_prefill64_speedup": report["kda"]["cases"]["production_prefill64"]["speedup"], "mla_4096_us": report["mla"]["benchmarks_us"]["capacity4096_valid4096"]}))


if __name__ == "__main__":
    main()
