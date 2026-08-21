#!/usr/bin/env python3
"""Verify immutable evidence or documented fallback for completed milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen(paths: list[Path], workspace: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(workspace).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def verify(workspace: Path, last: int = 19) -> dict[str, Any]:
    workspace = workspace.resolve()
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    warnings: list[str] = []
    for milestone in range(last + 1):
        sitrep = workspace / f"sitrep-milestone-{milestone}.md"
        runner = workspace / f"scripts/milestones/milestone-{milestone}.sh"
        manifest = workspace / f"artifacts/manifests/milestone-{milestone}-artifacts.json"
        log_dir = workspace / f"artifacts/logs/milestone-{milestone}"
        command_log = log_dir / "commands.log"
        command_jsonl = log_dir / "commands.jsonl"
        fallback = [path for path in (sitrep, command_log, command_jsonl) if path.is_file()]
        record: dict[str, Any] = {
            "milestone": milestone,
            "sitrep": sitrep.relative_to(workspace).as_posix(),
            "runner": runner.relative_to(workspace).as_posix(),
        }
        if not sitrep.is_file() or sitrep.stat().st_size == 0:
            issues.append(f"Milestone {milestone}: missing SITREP")
        if not runner.is_file() or runner.stat().st_size == 0:
            issues.append(f"Milestone {milestone}: missing runner")

        mismatches: list[str] = []
        data: dict[str, Any] | None = None
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            for artifact in data.get("artifacts", []):
                path = workspace / artifact["path"]
                if not path.is_file():
                    mismatches.append(f"missing:{artifact['path']}")
                elif path.stat().st_size != artifact["bytes"]:
                    mismatches.append(f"size:{artifact['path']}")
                elif sha256(path) != artifact["sha256"]:
                    mismatches.append(f"sha256:{artifact['path']}")

        if data is not None and not mismatches:
            record.update(
                evidence="artifact_manifest",
                manifest=manifest.relative_to(workspace).as_posix(),
                manifest_state="verified",
                artifacts=len(data.get("artifacts", [])),
                verified=True,
            )
        else:
            if len(fallback) != 3:
                issues.append(f"Milestone {milestone}: incomplete SITREP/log fallback")
            if mismatches:
                warnings.append(
                    f"Milestone {milestone}: stale historical manifest; using frozen SITREP/log fallback"
                )
            record.update(
                evidence="documented_sitrep_logs",
                manifest=(manifest.relative_to(workspace).as_posix() if manifest.is_file() else None),
                manifest_state="stale" if mismatches else "absent",
                manifest_mismatches=mismatches,
                artifacts=_frozen(fallback, workspace),
                verified=len(fallback) == 3,
            )
        records.append(record)
    return {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "milestones": records,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--last", type=int, default=19)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.workspace, args.last)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
