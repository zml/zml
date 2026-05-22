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


DEFAULT_MODEL = "/var/models/google/gemma-4-31B-it"
DEFAULT_DATASETS = ("math500.jsonl", "stories.jsonl")
TOKENIZER_CACHE: dict[str, Any] = {}


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
    meta_info: dict[str, Any] | None = None


def add_request_args(parser: argparse.ArgumentParser, include_connection: bool = True) -> None:
    parser.add_argument(
        "dataset_files",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=(
            "JSONL file(s), or names under examples/dflash_benchmark_gemma4/data. "
            "Default: math500.jsonl stories.jsonl"
        ),
    )
    if include_connection:
        parser.add_argument(
            "--base-url",
            default="http://127.0.0.1:30000/v1",
            help="OpenAI-compatible SGLang base URL, or server root with --verbose.",
        )
        parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--curl", default="curl", help="curl binary to use.")
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_MODEL,
        help="Tokenizer used to render chat prompts in --verbose mode.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="With --verbose, send raw prompt text instead of chat-template input_ids.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Use SGLang's native /generate endpoint and print meta_info fields, "
            "including speculative decoding metrics when available."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JSONL dataset requests against a running SGLang server."
    )
    add_request_args(parser)
    return parser.parse_args()


def sample_id(value: dict[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id", "question_id", "unique_id"):
        if key in value and str(value[key]).strip():
            return str(value[key]).strip()
    return str(row_index)


def resolve_dataset_path(dataset_file: str) -> Path:
    path = Path(dataset_file).expanduser()
    if path.is_absolute() or path.exists():
        return path

    local_data_path = Path(__file__).resolve().parents[1] / "data" / dataset_file
    if local_data_path.exists():
        return local_data_path

    repo_root = Path(__file__).resolve().parents[3]
    benchmark_data_path = repo_root / "examples" / "dflash_benchmark" / "data" / dataset_file
    if benchmark_data_path.exists():
        return benchmark_data_path

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
            if not prompt:
                continue
            rows.append(Sample(sample_id=sample_id(value, row_index), prompt=prompt))

    if len(rows) < count:
        raise SystemExit(f"{dataset_path} has {len(rows)} valid samples, need {count}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:count]


def server_root(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def run_curl(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def get_tokenizer(tokenizer_path: str) -> Any:
    tokenizer = TOKENIZER_CACHE.get(tokenizer_path)
    if tokenizer is not None:
        return tokenizer

    try:
        from transformers import AutoTokenizer
    except ImportError as err:
        raise RuntimeError(
            "--verbose chat-template rendering requires transformers; use "
            "--raw-prompt to send the dataset prompt directly."
        ) from err

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


def native_payload(args: argparse.Namespace, sample: Sample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sampling_params": {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }
    if args.raw_prompt:
        payload["text"] = sample.prompt
        return payload

    tokenizer = get_tokenizer(args.tokenizer_path)
    payload["input_ids"] = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return payload


def run_native_sample(args: argparse.Namespace, sample: Sample) -> Result:
    payload = native_payload(args, sample)
    cmd = [
        args.curl,
        "-sS",
        "-m",
        "900",
        f"{server_root(args.base_url)}/generate",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload, separators=(",", ":")),
    ]

    start_s = time.perf_counter()
    return_code, stdout, stderr = run_curl(cmd)
    end_s = time.perf_counter()
    if return_code != 0:
        raise RuntimeError(f"curl failed for sample {sample.sample_id}: {stderr.strip()}")

    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"sample {sample.sample_id} returned invalid JSON: {err}") from err

    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise RuntimeError(f"sample {sample.sample_id} did not return meta_info")

    completion_tokens = int(
        meta_info.get("completion_tokens") or len(response.get("output_ids") or [])
    )
    if completion_tokens <= 0:
        raise RuntimeError(f"sample {sample.sample_id} returned no completion tokens")

    finish_reason_value = meta_info.get("finish_reason") or ""
    if isinstance(finish_reason_value, dict):
        finish_reason = str(finish_reason_value.get("type") or finish_reason_value)
    else:
        finish_reason = str(finish_reason_value)

    total_s = max(end_s - start_s, 1e-9)
    return Result(
        sample_id=sample.sample_id,
        completion_tokens=completion_tokens,
        first_token_s=float(meta_info.get("time_to_first_token") or 0.0),
        generation_s=total_s,
        tokens_per_s=completion_tokens / total_s,
        finish_reason=finish_reason,
        meta_info=meta_info,
    )


def run_openai_sample(args: argparse.Namespace, sample: Sample) -> Result:
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
        "900",
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
        raise RuntimeError("server did not return streamed usage")

    generation_s = max(end_s - first_token_s, 1e-9)
    return Result(
        sample_id=sample.sample_id,
        completion_tokens=completion_tokens,
        first_token_s=first_token_s - start_s,
        generation_s=generation_s,
        tokens_per_s=completion_tokens / generation_s,
        finish_reason=finish_reason,
    )


def run_sample(args: argparse.Namespace, sample: Sample) -> Result:
    if args.verbose:
        return run_native_sample(args, sample)
    return run_openai_sample(args, sample)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_histograms(results: list[Result]) -> list[int]:
    histogram: list[int] = []
    for result in results:
        meta_info = result.meta_info or {}
        values = meta_info.get("spec_accept_histogram")
        if values is None:
            values = meta_info.get("spec_correct_drafts_histogram")
        if not isinstance(values, list):
            continue
        if len(histogram) < len(values):
            histogram.extend([0] * (len(values) - len(histogram)))
        for index, value in enumerate(values):
            histogram[index] += int(value)
    return histogram


def histogram_mean(histogram: list[int]) -> float:
    total = sum(histogram)
    if total == 0:
        return 0.0
    return sum(index * count for index, count in enumerate(histogram)) / total


def print_summary(dataset_name: str, results: list[Result]) -> None:
    total_tokens = sum(result.completion_tokens for result in results)
    total_generation_s = sum(result.generation_s for result in results)
    average_tps = total_tokens / total_generation_s if total_generation_s else 0.0
    mean_sample_tps = mean([result.tokens_per_s for result in results])

    print()
    print(f"== {dataset_name} summary ==")
    print(f"samples: {len(results)}")
    print(f"completion tokens: {total_tokens}")
    print(f"total generation time after first token: {total_generation_s:.3f}s")
    print(f"average tokens/sec: {average_tps:.2f}")
    print(f"mean per-sample tokens/sec: {mean_sample_tps:.2f}")

    spec_accept_lengths = [
        float(result.meta_info["spec_accept_length"])
        for result in results
        if result.meta_info and result.meta_info.get("spec_accept_length") is not None
    ]
    if spec_accept_lengths:
        print(
            "average committed tokens per verify step: "
            f"{mean(spec_accept_lengths):.4f}"
        )

    spec_accept_rates = [
        float(result.meta_info["spec_accept_rate"])
        for result in results
        if result.meta_info and result.meta_info.get("spec_accept_rate") is not None
    ]
    if spec_accept_rates:
        print(f"average draft acceptance rate: {mean(spec_accept_rates):.4f}")

    histogram = aggregate_histograms(results)
    if histogram:
        print(
            "average accepted draft tokens per verify step: "
            f"{histogram_mean(histogram):.4f}"
        )
        print(f"candidate sequence length acceptance histogram: {histogram}")


def run_requests(args: argparse.Namespace) -> list[tuple[str, list[Result]]]:
    all_results: list[tuple[str, list[Result]]] = []
    for dataset_index, dataset_file in enumerate(args.dataset_files):
        samples = load_samples(dataset_file, args.samples, args.seed + dataset_index)
        dataset_name = resolve_dataset_path(dataset_file).name
        results: list[Result] = []
        print(f"== {dataset_name} ==")
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
            if args.verbose and result.meta_info:
                spec_parts = []
                for key in (
                    "spec_accept_length",
                    "spec_accept_rate",
                    "spec_accept_histogram",
                    "spec_verify_ct",
                    "spec_num_correct_drafts",
                    "spec_num_proposed_drafts",
                    "spec_correct_drafts_histogram",
                ):
                    if key in result.meta_info:
                        spec_parts.append(f"{key}={result.meta_info[key]}")
                if spec_parts:
                    print("  " + " ".join(spec_parts), flush=True)
        all_results.append((dataset_name, results))
        print_summary(dataset_name, results)
        print()

    if len(all_results) > 1:
        print("== Combined table ==")
        print("dataset\tsamples\tavg_tps\tmean_sample_tps\tcompletion_tokens")
        for dataset_name, results in all_results:
            total_tokens = sum(result.completion_tokens for result in results)
            total_generation_s = sum(result.generation_s for result in results)
            average_tps = total_tokens / total_generation_s if total_generation_s else 0.0
            mean_sample_tps = mean([result.tokens_per_s for result in results])
            print(
                f"{dataset_name}\t{len(results)}\t{average_tps:.2f}\t"
                f"{mean_sample_tps:.2f}\t{total_tokens}"
            )

    return all_results


def main() -> None:
    run_requests(parse_args())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
