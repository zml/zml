#!/usr/bin/env python3
"""Summarize the Milestone 16 tokenizer and staged CUDA session gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


TOKEN = re.compile(
    r"token_index=(?P<index>\d+) input=(?P<input>\d+) greedy=(?P<greedy>\d+)"
    r" load_us=(?P<load>\d+) execute_us=(?P<execute>\d+) total_us=(?P<total>\d+)"
)
SESSION = re.compile(
    r"KIMI_K3_SESSION_PASS repeat=(?P<repeat>\d+) tokens=(?P<tokens>\d+)"
    r" greedy=(?P<greedy>\d+) compile_us=(?P<compile>\d+)"
    r" session_us=(?P<session>\d+) backend=(?P<backend>\w+)"
)
DECODE = re.compile(
    r"KIMI_K3_SESSION_DECODE_PASS streamed=(?P<streamed>\d+)"
    r" next=(?P<next>\d+) history_tokens=(?P<history>\d+) capacity=(?P<capacity>\d+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_rows(log: str) -> list[dict[str, int]]:
    return [
        {
            "index": int(match["index"]),
            "input": int(match["input"]),
            "greedy": int(match["greedy"]),
            "load_us": int(match["load"]),
            "execute_us": int(match["execute"]),
            "total_us": int(match["total"]),
        }
        for match in TOKEN.finditer(log)
    ]


def summarize(
    prefix_log: str,
    decode_log: str,
    tokenizer_log: str,
    tokenizer_fixture: dict,
    prefix_manifest: dict,
    lock: dict,
    implementation_commit: str,
) -> dict:
    if "KIMI_K3_SESSION_ALL_PASS reset_deterministic=true official_prefix_checked=true" not in prefix_log:
        raise ValueError("missing official-prefix/reset pass marker")
    if "KIMI_K3_SESSION_DECODE_PASS" not in decode_log:
        raise ValueError("missing streaming decode pass marker")
    if "KIMI_K3_TOKENIZER_ALL_PASS" not in tokenizer_log:
        raise ValueError("missing tokenizer pass marker")

    official_tokens = prefix_manifest["token_ids"]
    official_greedy = prefix_manifest["prefill_greedy_token"]
    if official_tokens != lock["raw_token_ids"] or official_greedy != lock["official_greedy_token"]:
        raise ValueError("official prefix manifest disagrees with Milestone 16 lock")

    rows = token_rows(prefix_log)
    sessions = [
        {
            "repeat": int(match["repeat"]),
            "tokens": int(match["tokens"]),
            "greedy": int(match["greedy"]),
            "compile_us": int(match["compile"]),
            "session_us": int(match["session"]),
            "backend": match["backend"],
        }
        for match in SESSION.finditer(prefix_log)
    ]
    if len(rows) != 8 or len(sessions) != 2:
        raise ValueError("expected two complete four-token sessions")
    runs = [rows[:4], rows[4:]]
    for repeat, run in enumerate(runs):
        if [row["index"] for row in run] != list(range(4)):
            raise ValueError(f"invalid cache positions in repeat {repeat}")
        if [row["input"] for row in run] != official_tokens:
            raise ValueError(f"raw input mismatch in repeat {repeat}")
        if run[-1]["greedy"] != official_greedy:
            raise ValueError(f"official greedy mismatch in repeat {repeat}")
    if sessions[0]["greedy"] != sessions[1]["greedy"] or sessions[0]["greedy"] != official_greedy:
        raise ValueError("session reset is not deterministic")
    if any(row["backend"] != "cuda" for row in sessions):
        raise ValueError("non-CUDA session result")

    decode_match = DECODE.search(decode_log)
    if decode_match is None:
        raise ValueError("incomplete decode telemetry")
    decode = {key: int(value) for key, value in decode_match.groupdict().items()}
    if decode["history"] != decode["capacity"] or decode["streamed"] != runs[0][0]["greedy"]:
        raise ValueError("bounded decode history mismatch")

    counts = {
        "ordinary": len(tokenizer_fixture["ordinary"]),
        "fuzz": len(tokenizer_fixture["fuzz"]),
        "structural": len(tokenizer_fixture["structural"]),
        "first_turn": len(tokenizer_fixture["first_turn"]),
        "continuation": len(tokenizer_fixture["continuation"]),
    }
    if counts != lock["tokenizer_cases"]:
        raise ValueError("tokenizer corpus count mismatch")

    all_rows = [row for run in runs for row in run]
    mean_load = statistics.mean(row["load_us"] for row in all_rows)
    mean_execute = statistics.mean(row["execute_us"] for row in all_rows)
    mean_total = statistics.mean(row["total_us"] for row in all_rows)
    return {
        "schema_version": 1,
        "milestone": 16,
        "status": "PASS",
        "backend": "cuda",
        "implementation_commit": implementation_commit,
        "official": {
            "authority": "pinned Moonshot source and equally truncated S4 fixture",
            "moonshot_revision": lock["moonshot_revision"],
            "raw_token_ids": official_tokens,
            "greedy_token": official_greedy,
            "reference_wall_ms": prefix_manifest["timing"]["repeat"]["wall_ms"],
            "prefix_fixture_sha256": lock["prefix_fixture_sha256"],
            "prefix_fixture_semantic_sha256": lock["prefix_fixture_semantic_sha256"],
        },
        "tokenizer": {
            "authority": tokenizer_fixture["authority"],
            "exact_corpus_parity": True,
            "cases": counts,
            "source": tokenizer_fixture["source"],
            "tokenizer_reference_sha256": lock["tokenizer_reference_sha256"],
            "tokenizer_json_sha256": lock["tokenizer_json_sha256"],
        },
        "zml_session": {
            "compiled_families": ["embedding", "kda_dense", "kda_moe", "mla_moe", "diagnostic_head"],
            "layer_selection": lock["layer_selection"],
            "routing": "normal_global_896_experts",
            "runs": [
                {
                    **sessions[index],
                    "token_greedy_sequence": [row["greedy"] for row in run],
                    "tokens_detail": run,
                }
                for index, run in enumerate(runs)
            ],
            "reset_deterministic": True,
            "official_final_greedy_exact": True,
            "streaming_decode": {
                **decode,
                "status": "PASS",
                "official_numeric_comparison": "not_claimed; API/cache/tokenizer plumbing gate",
            },
        },
        "performance": {
            "scope": "four-layer correctness oracle; one full expert bank resident at a time",
            "mean_staged_weight_load_us_per_token": round(mean_load),
            "mean_synchronized_execute_us_per_token": round(mean_execute),
            "mean_total_us_per_token": round(mean_total),
            "execute_tokens_per_second": 1_000_000 / mean_execute,
            "end_to_end_tokens_per_second": 1_000_000 / mean_total,
            "comparison_note": "Moonshot reference timing is full-prefix execution; ZML total includes repeated disk/host/GPU staging and is not an optimized throughput claim.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-log", type=Path, required=True)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--tokenizer-log", type=Path, required=True)
    parser.add_argument("--tokenizer-fixture", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if sha256(args.tokenizer_fixture) != lock["tokenizer_reference_sha256"]:
        raise ValueError("tokenizer reference file hash mismatch")
    tokenizer_json = args.tokenizer_fixture.parents[2] / "tokenizers/milestone-16/tokenizer.json"
    if sha256(tokenizer_json) != lock["tokenizer_json_sha256"]:
        raise ValueError("converted tokenizer file hash mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = summarize(
        args.prefix_log.read_text(),
        args.decode_log.read_text(),
        args.tokenizer_log.read_text(),
        json.loads(args.tokenizer_fixture.read_text()),
        json.loads(args.prefix_manifest.read_text()),
        lock,
        commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
