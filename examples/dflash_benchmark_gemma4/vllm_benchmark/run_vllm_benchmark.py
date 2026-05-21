#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from request_benchmark import (
    DEFAULT_MODEL,
    print_summary,
    results_to_json,
    run_dataset_requests,
    summarize,
)


DEFAULT_DFLASH_MODEL = "/var/models/z-lab/gemma-4-31B-it-DFlash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a Gemma4 vLLM server, run dataset requests, then shut it down."
    )
    parser.add_argument("dataset_file", nargs="?", default="math500.jsonl")
    parser.add_argument("--mode", choices=["baseline", "dflash"], default="baseline")
    parser.add_argument("--target-model", default=DEFAULT_MODEL)
    parser.add_argument("--dflash-model", default=DEFAULT_DFLASH_MODEL)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--serve-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-bin", default="")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="Extra argument passed to 'vllm serve'. Repeat for multiple args.",
    )
    return parser.parse_args()


def default_vllm_bin() -> str:
    local_bin = Path(__file__).resolve().parent / ".venv" / "bin" / "vllm"
    if local_bin.exists():
        return str(local_bin)
    return "vllm"


def base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1"


def server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.vllm_bin or default_vllm_bin(),
        "serve",
        args.target_model,
        "--host",
        args.serve_host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.mode == "dflash":
        cmd.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dflash",
                        "model": args.dflash_model,
                        "num_speculative_tokens": args.num_speculative_tokens,
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    cmd.extend(args.serve_arg)
    return cmd


def start_server(args: argparse.Namespace) -> tuple[subprocess.Popen[bytes], Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    log_handle = None
    stdout: Any = subprocess.DEVNULL
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("wb")
        stdout = log_handle

    cmd = server_command(args)
    print(
        f"Starting {args.mode} vLLM server on CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
        flush=True,
    )
    print("Command:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle


def wait_for_ready(proc: subprocess.Popen[bytes], url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{url.rstrip('/')}/models"
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(models_url, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    print("vLLM server is ready", flush=True)
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server did not become ready: {last_error}")


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    print("Stopping vLLM server", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10.0)


def main() -> None:
    args = parse_args()
    url = base_url(args)
    proc, log_handle = start_server(args)
    try:
        wait_for_ready(proc, url, args.ready_timeout_s)
        results = run_dataset_requests(
            dataset_file=args.dataset_file,
            base_url=url,
            model=args.target_model,
            samples=args.samples,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.request_timeout_s,
        )
        summary = summarize(results)
        print_summary(summary)

        if args.output_json:
            payload = {
                "config": {
                    "mode": args.mode,
                    "target_model": args.target_model,
                    "dflash_model": args.dflash_model if args.mode == "dflash" else "",
                    "dataset_file": args.dataset_file,
                    "samples": args.samples,
                },
                **results_to_json(results, summary),
            }
            Path(args.output_json).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    finally:
        stop_server(proc)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
