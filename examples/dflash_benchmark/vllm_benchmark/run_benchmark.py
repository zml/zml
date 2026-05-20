#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from stats import (
        MethodResult,
        SampleResult,
        benchmark_to_dict,
        compare_token_ids,
        compute_summary,
        format_report,
    )
except ImportError as exc:  # pragma: no cover - exercised on hosts without local module path.
    raise SystemExit(
        "Could not import sibling stats.py. Run this script from its folder or keep "
        "examples/dflash_benchmark/vllm_benchmark/stats.py next to it."
    ) from exc


@dataclass
class DatasetSample:
    sample_id: str
    dataset: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM DFlash-vs-baseline benchmark.")
    parser.add_argument("--model", required=True, help="Draft/DFlash model path.")
    parser.add_argument("--target-model", required=True, help="Target model path.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["math500", "swe_bench_lite", "alpaca", "mt_bench", "generic_jsonl"],
    )
    parser.add_argument("--dataset-path", required=True, help="JSONL dataset path.")
    parser.add_argument("--split", default="", help="Dataset split label.")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-speculative-tokens", type=int, default=10)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[DatasetSample]:
    path = Path(args.dataset_path).expanduser()
    rows: list[DatasetSample] = []
    split = args.split or default_split(args.dataset)
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                prompt = format_prompt(args.dataset, value)
            except Exception:
                continue
            if not prompt.strip():
                continue
            rows.append(
                DatasetSample(
                    sample_id=sample_id(value, row_index),
                    dataset=f"{args.dataset}/{split}",
                    prompt=prompt.strip(),
                )
            )
    if len(rows) < args.samples:
        raise SystemExit(f"dataset has {len(rows)} valid rows, need {args.samples}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    return rows[: args.samples]


def default_split(dataset: str) -> str:
    return {
        "alpaca": "train",
        "generic_jsonl": "default",
    }.get(dataset, "test")


def format_prompt(dataset: str, value: dict[str, Any]) -> str:
    if dataset == "math500":
        return f"{required_str(value, 'problem')}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    if dataset == "swe_bench_lite":
        return f"Problem Statement:\n{required_str(value, 'problem_statement')}\nPlease fix the issue described above."
    if dataset == "alpaca":
        instruction = required_str(value, "instruction")
        input_text = str(value.get("input") or "").strip()
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}"
        return instruction
    if dataset == "mt_bench":
        turns = value.get("prompt")
        if not isinstance(turns, list) or not turns:
            raise ValueError("mt_bench row missing prompt turns")
        return "\n\n".join(
            f"Turn {i + 1}:\n{str(turn).strip()}"
            for i, turn in enumerate(turns)
            if str(turn).strip()
        )
    if dataset == "generic_jsonl":
        for key in ("prompt", "text", "input"):
            if str(value.get(key) or "").strip():
                return str(value[key]).strip()
    raise ValueError(f"unsupported or malformed {dataset} row")


def required_str(value: dict[str, Any], key: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"missing {key}")
    return text


def sample_id(value: dict[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id", "instance_id", "question_id"):
        if key in value and str(value[key]).strip():
            return str(value[key]).strip()
    return str(row_index)


def make_llm(args: argparse.Namespace, speculative: bool):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit("vLLM is not installed. Install requirements.txt in this folder.") from exc

    kwargs: dict[str, Any] = {
        "model": args.target_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if speculative:
        kwargs["speculative_config"] = {
            "model": args.model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "method": "draft_model",
        }
    return LLM(**kwargs)


def sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )


def unload_llm(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_one(llm: Any, params: Any, prompt: str) -> tuple[list[int], str, float]:
    start = time.perf_counter()
    outputs = llm.generate([prompt], params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    output = outputs[0].outputs[0]
    token_ids = list(getattr(output, "token_ids", []) or [])
    return token_ids, output.text, elapsed


def prompt_token_count(llm: Any, prompt: str) -> int:
    tokenizer = llm.get_tokenizer()
    return len(tokenizer.encode(prompt))


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    params = sampling_params(args)

    baseline_llm = make_llm(args, speculative=False)
    prompt_tokens = [prompt_token_count(baseline_llm, sample.prompt) for sample in samples]
    baseline_outputs = [run_one(baseline_llm, params, sample.prompt) for sample in samples]
    unload_llm(baseline_llm)

    dflash_llm = make_llm(args, speculative=True)
    dflash_outputs = [run_one(dflash_llm, params, sample.prompt) for sample in samples]
    unload_llm(dflash_llm)

    results: list[SampleResult] = []
    for index, sample in enumerate(samples):
        baseline_ids, baseline_text, baseline_elapsed = baseline_outputs[index]
        dflash_ids, dflash_text, dflash_elapsed = dflash_outputs[index]
        result = SampleResult(
            id=sample.sample_id,
            dataset=sample.dataset,
            prompt_tokens=prompt_tokens[index],
            baseline=MethodResult(
                token_ids=baseline_ids,
                generated_text=baseline_text,
                elapsed_s=baseline_elapsed,
            ),
            dflash=MethodResult(
                token_ids=dflash_ids,
                generated_text=dflash_text,
                elapsed_s=dflash_elapsed,
            ),
            quality=compare_token_ids(baseline_ids, dflash_ids),
        )
        print_sample(index + 1, len(samples), result)
        if args.verbose:
            print("  Prompt:")
            print_indented(sample.prompt)
            print("  Baseline answer:")
            print_indented(baseline_text)
            print("  DFlash answer:")
            print_indented(dflash_text)
        results.append(result)

    summary = compute_summary(results)
    print(format_report(summary), end="")

    if args.output_json:
        payload = benchmark_to_dict(results, summary, vars(args))
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_indented(text: str) -> None:
    for line in text.splitlines() or [""]:
        print(f"    {line}")


def print_sample(index: int, total: int, sample: SampleResult) -> None:
    quality = sample.comparison()
    print(f"[{index}/{total}] dataset={sample.dataset} id={sample.id} prompt_tokens={sample.prompt_tokens}")
    print(
        "  Baseline: "
        f"{sample.baseline.token_count()} tokens, "
        f"TPOT={sample.baseline.tpot_ms():.2f}ms, "
        f"TPS={sample.baseline.tokens_per_second():.1f}"
    )
    print(
        "  DFlash:   "
        f"{sample.dflash.token_count()} tokens, "
        f"TPOT={sample.dflash.tpot_ms():.2f}ms, "
        f"TPS={sample.dflash.tokens_per_second():.1f}, "
        f"tau={sample.dflash.tau():.2f}"
    )
    if quality.exact_match:
        print(f"  Quality:  MATCH first {quality.compared_tokens} output tokens")
    elif quality.first_mismatch_index is not None:
        print(
            "  Quality:  "
            f"MISMATCH at output token {quality.first_mismatch_index} "
            f"({quality.compared_tokens} compared)"
        )
    else:
        print(f"  Quality:  MISMATCH ({quality.compared_tokens} compared)")


if __name__ == "__main__":
    main()
