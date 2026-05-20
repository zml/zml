#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = {
    "swe_bench_lite": {
        "source": "SWE-bench_Lite/data/test-00000-of-00001.parquet",
        "output": "SWE-bench_Lite/test.jsonl",
        "required": ("instance_id", "problem_statement"),
    },
    "alpaca": {
        "source": "alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        "output": "alpaca/train.jsonl",
        "required": ("instruction",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded benchmark parquet datasets to JSONL."
    )
    parser.add_argument(
        "dataset",
        choices=sorted(DATASETS.keys()) + ["all"],
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--data-root",
        default=str(Path.home() / "data"),
        help="Root containing downloaded dataset directories. Defaults to ~/data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing JSONL output file.",
    )
    return parser.parse_args()


def convert_one(name: str, data_root: Path, overwrite: bool) -> None:
    config = DATASETS[name]
    source = data_root / config["source"]
    output = data_root / config["output"]

    if not source.exists():
        raise FileNotFoundError(f"missing parquet source: {source}")
    if output.exists() and not overwrite:
        print(f"{name}: {output} already exists; use --overwrite to replace it")
        return

    table = pq.read_table(source)
    columns = set(table.column_names)
    missing = [column for column in config["required"] if column not in columns]
    if missing:
        raise ValueError(f"{source} is missing expected columns: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = table.to_pylist()
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"{name}: wrote {len(rows)} rows to {output}")
    print(f"{name}: columns: {', '.join(table.column_names)}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    names = DATASETS.keys() if args.dataset == "all" else (args.dataset,)
    for name in names:
        convert_one(name, data_root, args.overwrite)


if __name__ == "__main__":
    main()
