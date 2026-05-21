#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import request_benchmark


DEFAULT_TARGET_MODEL = "/var/models/google/gemma-4-31B-it"
DEFAULT_DFLASH_MODEL = "/var/models/z-lab/gemma-4-31B-it-DFlash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start an SGLang Gemma4 server, run JSONL benchmark requests, "
            "then shut the server down."
        )
    )
    parser.add_argument(
        "mode",
        choices=("baseline", "dflash"),
        help="baseline runs the target model only; dflash enables SGLang DFLASH speculation.",
    )
    request_benchmark.add_request_args(parser, include_connection=False)
    parser.add_argument("--python", default=None, help="Python binary with sglang installed.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--gpu",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA_VISIBLE_DEVICES value for the SGLang server. Default: env CUDA_VISIBLE_DEVICES or 0.",
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--dflash-model", default=DEFAULT_DFLASH_MODEL)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--readiness-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--keep-server-on-failure", action="store_true")
    return parser.parse_args()


def default_python() -> str:
    script_dir = Path(__file__).resolve().parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python or default_python(),
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.target_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
        "--dtype",
        args.dtype,
        "--log-level",
        args.log_level,
    ]

    if args.mode == "dflash":
        cmd.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.dflash_model,
            ]
        )

        optional_env_args = (
            ("SPECULATIVE_NUM_DRAFT_TOKENS", "--speculative-num-draft-tokens"),
            ("SPECULATIVE_DFLASH_BLOCK_SIZE", "--speculative-dflash-block-size"),
            (
                "SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE",
                "--speculative-dflash-draft-window-size",
            ),
        )
        for env_name, flag in optional_env_args:
            value = os.environ.get(env_name)
            if value:
                cmd.extend([flag, value])

    for extra in args.server_extra_arg:
        cmd.append(extra)

    return cmd


def wait_for_ready(base_url: str, proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with status {proc.returncode}")
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(f"SGLang server was not ready after {timeout_s:.0f}s: {last_error}")


def terminate_server(proc: subprocess.Popen[bytes], timeout_s: float = 60.0) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)


def main() -> None:
    args = parse_args()
    args.base_url = f"http://127.0.0.1:{args.port}/v1"
    args.model = args.target_model

    cmd = server_command(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_path = Path(args.server_log).expanduser() if args.server_log else None
    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    stdout = log_file if log_file else None
    stderr = subprocess.STDOUT if log_file else None

    print(f"Starting SGLang {args.mode} server on CUDA_VISIBLE_DEVICES={args.gpu}")
    print("Command:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)

    try:
        wait_for_ready(args.base_url, proc, args.readiness_timeout_s)
        print(f"SGLang server is ready at {args.base_url}", flush=True)
        request_benchmark.run_requests(args)
    except Exception:
        if args.keep_server_on_failure:
            print(f"Leaving server running with pid {proc.pid}", file=sys.stderr)
            raise
        raise
    finally:
        if not args.keep_server_on_failure:
            terminate_server(proc)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
