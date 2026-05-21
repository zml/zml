#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "/var/models/google/gemma-4-31B-it"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str


@dataclass(frozen=True)
class Result:
    sample_id: str
    completion_tokens: int
    first_token_s: float
    generation_s: float
    tokens_per_s: float
    finish_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JSONL dataset requests against an OpenAI-compatible vLLM server."
    )
    parser.add_argument("dataset_file", nargs="?", default="math500.jsonl")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def resolve_dataset_path(dataset_file: str) -> Path:
    path = Path(dataset_file).expanduser()
    if path.is_absolute() or path.exists():
        return path

    candidates = [
        Path(__file__).resolve().parents[1] / "data" / dataset_file,
        (
            Path(__file__).resolve().parents[1].parent
            / "dflash_benchmark"
            / "data"
            / dataset_file
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def sample_id(value: dict[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        if key in value and str(value[key]).strip():
            return str(value[key]).strip()
    return str(row_index)


def format_prompt(value: dict[str, Any]) -> str:
    if str(value.get("prompt") or "").strip():
        return str(value["prompt"]).strip()

    if str(value.get("problem") or "").strip():
        problem = str(value["problem"]).strip()
        return (
            f"{problem}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    if str(value.get("story") or "").strip():
        return str(value["story"]).strip()

    if str(value.get("text") or "").strip():
        return str(value["text"]).strip()

    return ""


def load_samples(dataset_file: str, count: int, seed: int) -> list[Sample]:
    rows: list[Sample] = []
    dataset_path = resolve_dataset_path(dataset_file)
    with dataset_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = format_prompt(value)
            if prompt:
                rows.append(Sample(sample_id(value, row_index), prompt))

    if len(rows) < count:
        raise SystemExit(f"dataset has {len(rows)} valid samples, need {count}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:count]


def post_stream_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[float, float, int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_s = time.perf_counter()
    first_token_s: float | None = None
    end_s = start_s
    completion_tokens: int | None = None
    finish_reason = ""

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            now_s = time.perf_counter()
            line = raw_line.decode("utf-8", errors="replace").strip()
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
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if delta.get("content") and first_token_s is None:
                first_token_s = now_s

    if first_token_s is None:
        raise RuntimeError("server streamed no content")
    if completion_tokens is None:
        raise RuntimeError("server did not return streamed usage")

    return (
        first_token_s - start_s,
        max(end_s - first_token_s, 1e-9),
        completion_tokens,
        finish_reason,
    )


def run_sample(
    base_url: str,
    model: str,
    sample: Sample,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": sample.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    first_token_s, generation_s, completion_tokens, finish_reason = post_stream_json(
        url, payload, timeout_s
    )
    return Result(
        sample_id=sample.sample_id,
        completion_tokens=completion_tokens,
        first_token_s=first_token_s,
        generation_s=generation_s,
        tokens_per_s=completion_tokens / generation_s,
        finish_reason=finish_reason,
    )


def run_dataset_requests(
    dataset_file: str,
    base_url: str,
    model: str,
    samples: int = 10,
    seed: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
) -> list[Result]:
    loaded_samples = load_samples(dataset_file, samples, seed)
    results: list[Result] = []
    for index, sample in enumerate(loaded_samples, start=1):
        print(f"[{index}/{len(loaded_samples)}] {sample.sample_id}", flush=True)
        result = run_sample(base_url, model, sample, max_tokens, temperature, timeout_s)
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
    return results


def summarize(results: list[Result]) -> dict[str, Any]:
    total_tokens = sum(result.completion_tokens for result in results)
    total_generation_s = sum(result.generation_s for result in results)
    average_tps = total_tokens / total_generation_s if total_generation_s else 0.0
    mean_sample_tps = (
        sum(result.tokens_per_s for result in results) / len(results) if results else 0.0
    )
    return {
        "samples": len(results),
        "completion_tokens": total_tokens,
        "total_generation_s": total_generation_s,
        "average_tokens_per_second": average_tps,
        "mean_sample_tokens_per_second": mean_sample_tps,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print(f"samples: {summary['samples']}")
    print(f"completion tokens: {summary['completion_tokens']}")
    print(f"total generation time after first token: {summary['total_generation_s']:.3f}s")
    print(f"average tokens/sec: {summary['average_tokens_per_second']:.2f}")
    print(f"mean per-sample tokens/sec: {summary['mean_sample_tokens_per_second']:.2f}")


def results_to_json(results: list[Result], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": [result.__dict__ for result in results],
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    results = run_dataset_requests(
        dataset_file=args.dataset_file,
        base_url=args.base_url,
        model=args.model,
        samples=args.samples,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )
    summary = summarize(results)
    print_summary(summary)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results_to_json(results, summary), indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
