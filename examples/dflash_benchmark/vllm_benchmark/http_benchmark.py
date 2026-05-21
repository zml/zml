#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    sample_id: str
    prompt: str


@dataclass
class Result:
    sample_id: str
    completion_tokens: int
    first_token_s: float
    generation_s: float
    tokens_per_s: float
    finish_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a running vLLM HTTP server on JSONL dataset samples."
    )
    parser.add_argument(
        "--base-url",
        default="http://100.70.254.89:8000/v1",
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--model",
        default="/var/models/meta-llama/Llama-3.1-8B-Instruct/",
        help="Model id to send in chat completion requests.",
    )
    parser.add_argument(
        "dataset_file",
        help=(
            "JSONL file under examples/dflash_benchmark/data, or an explicit path. "
            "Examples: math500.jsonl, stories.jsonl"
        ),
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--curl",
        default="curl",
        help="curl binary to use for HTTP requests.",
    )
    return parser.parse_args()


def sample_id(value: dict[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        if key in value and str(value[key]).strip():
            return str(value[key]).strip()
    return str(row_index)


def resolve_dataset_path(dataset_file: str) -> Path:
    path = Path(dataset_file).expanduser()
    if path.is_absolute() or path.exists():
        return path

    repo_data_path = Path(__file__).resolve().parents[1] / "data" / dataset_file
    if repo_data_path.exists():
        return repo_data_path

    return path


def format_prompt(value: dict[str, Any]) -> str:
    if str(value.get("prompt") or "").strip():
        return str(value["prompt"]).strip()

    if str(value.get("problem") or "").strip():
        problem = str(value.get("problem") or "").strip()
        return (
            f"{problem}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    if str(value.get("instruction") or "").strip():
        instruction = str(value.get("instruction") or "").strip()
        input_text = str(value.get("input") or "").strip()
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}"
        return instruction

    return ""


def load_samples(dataset_file: str, count: int, seed: int) -> list[Sample]:
    rows: list[Sample] = []
    dataset_path = resolve_dataset_path(dataset_file)
    with Path(dataset_path).expanduser().open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = format_prompt(value)
            if not prompt:
                continue
            rows.append(
                Sample(
                    sample_id=sample_id(value, row_index),
                    prompt=prompt,
                )
            )

    if len(rows) < count:
        raise SystemExit(f"dataset has {len(rows)} valid samples, need {count}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:count]


def run_sample(args: argparse.Namespace, sample: Sample) -> Result:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": sample.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    cmd = [
        args.curl,
        "-N",
        "-sS",
        "-m",
        "600",
        f"{args.base_url.rstrip('/')}/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload, separators=(",", ":")),
    ]

    start_s = time.perf_counter()
    first_token_s: float | None = None
    end_s = start_s
    completion_tokens: int | None = None
    finish_reason = ""

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    try:
        for line in proc.stdout:
            now_s = time.perf_counter()
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                end_s = now_s
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content and first_token_s is None:
                    first_token_s = now_s
    finally:
        stderr = proc.stderr.read()
        return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"curl failed for sample {sample.sample_id}: {stderr.strip()}")
    if first_token_s is None:
        raise RuntimeError(f"sample {sample.sample_id} did not stream any content")
    if completion_tokens is None:
        raise RuntimeError(
            "server did not return streamed usage; vLLM may not support "
            "stream_options.include_usage"
        )

    generation_s = max(end_s - first_token_s, 1e-9)
    return Result(
        sample_id=sample.sample_id,
        completion_tokens=completion_tokens,
        first_token_s=first_token_s - start_s,
        generation_s=generation_s,
        tokens_per_s=completion_tokens / generation_s,
        finish_reason=finish_reason,
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset_file, args.samples, args.seed)

    results: list[Result] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)
        result = run_sample(args, sample)
        results.append(result)
        print(
            "  "
            f"tokens={result.completion_tokens} "
            f"ttft={result.first_token_s:.3f}s "
            f"gen={result.generation_s:.3f}s "
            f"tps={result.tokens_per_s:.2f} "
            f"finish={result.finish_reason}",
            flush=True,
        )

    total_tokens = sum(result.completion_tokens for result in results)
    total_generation_s = sum(result.generation_s for result in results)
    average_tps = total_tokens / total_generation_s if total_generation_s else 0.0
    mean_sample_tps = (
        sum(result.tokens_per_s for result in results) / len(results) if results else 0.0
    )

    print()
    print(f"samples: {len(results)}")
    print(f"completion tokens: {total_tokens}")
    print(f"total generation time after first token: {total_generation_s:.3f}s")
    print(f"average tokens/sec: {average_tps:.2f}")
    print(f"mean per-sample tokens/sec: {mean_sample_tps:.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
