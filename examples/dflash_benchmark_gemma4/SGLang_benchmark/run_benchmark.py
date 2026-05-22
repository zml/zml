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
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--draft-attention-backend", default="fa4")
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=16)
    parser.add_argument("--readiness-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--keep-server-on-failure", action="store_true")
    return parser.parse_args()


def default_python() -> str:
    script_dir = Path(__file__).resolve().parent
    venv_python = Path("~/sglang_tests/.venv/bin/python").expanduser()
    if venv_python.exists():
        return str(venv_python)
    venv_python = script_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def server_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    sglang_tests = Path("~/sglang_tests").expanduser()
    path_prefixes = [Path("~/.cargo/bin").expanduser(), sglang_tests / "bin"]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(
        [str(path) for path in path_prefixes] + [existing_path]
    )

    protoc_shim = sglang_tests / "bin" / "protoc"
    if protoc_shim.exists():
        env.setdefault("PROTOC", str(protoc_shim))

    return env


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
        "--attention-backend",
        args.attention_backend,
        "--trust-remote-code",
    ]

    optional_args = (
        (args.context_length, "--context-length"),
        (args.mem_fraction_static, "--mem-fraction-static"),
        (args.cuda_graph_max_bs, "--cuda-graph-max-bs"),
        (args.dtype, "--dtype"),
        (args.log_level, "--log-level"),
    )
    for value, flag in optional_args:
        if value is not None:
            cmd.extend([flag, str(value)])

    if args.mode == "dflash":
        cmd.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.dflash_model,
                "--speculative-num-draft-tokens",
                str(args.speculative_num_draft_tokens),
                "--speculative-draft-attention-backend",
                args.draft_attention_backend,
            ]
        )

        optional_env_args = (
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
    env = server_env(args)

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
