import argparse
import os
import shutil
import subprocess
import sys


DEFAULT_MODEL_PATH = "/var/models/Qwen/Qwen3.5-4B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a vLLM server for a local Qwen model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model path to serve.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the vLLM server to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the vLLM server to.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to `vllm serve`.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("vllm") is None:
        print("error: vllm CLI not installed", file=sys.stderr)
        return 1
    if not os.path.isdir(args.model):
        print(f"error: model path not found: {args.model}", file=sys.stderr)
        return 1

    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.extra_args:
        command.extend(args.extra_args)

    print("running:", " ".join(command))
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
