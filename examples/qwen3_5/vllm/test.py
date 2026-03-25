import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


MODEL_PATH = "/var/models/Qwen/Qwen3.5-0.8B"
HOST = "127.0.0.1"
PROMPT = "Write a very long story about a cat"
MAX_TOKENS = 4000


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        sock.listen(1)
        return sock.getsockname()[1]


def wait_for_server(base_url: str, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2.0) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_error = err
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for vLLM server at {base_url}: {last_error}")


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    if shutil.which("vllm") is None:
        print("error: vllm CLI not installed", file=sys.stderr)
        return 1
    if not os.path.isdir(MODEL_PATH):
        print(f"error: model path not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    port = pick_free_port()
    base_url = f"http://{HOST}:{port}"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_PATH,
            "--host",
            HOST,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        models_payload = wait_for_server(base_url, timeout_s=120.0)
        print("server_ready:", json.dumps(models_payload))
        print(f"requesting_completion: max_tokens={MAX_TOKENS}")
        sys.stdout.flush()

        request_payload = {
            "model": MODEL_PATH,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
        }

        started_at = time.perf_counter()
        completion_payload = post_json(f"{base_url}/v1/completions", request_payload, timeout_s=1800.0)
        elapsed_s = time.perf_counter() - started_at

        usage = completion_payload.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            raise RuntimeError(f"missing completion_tokens in response: {completion_payload}")
        if completion_tokens <= 0:
            raise RuntimeError(f"expected positive completion_tokens, got {completion_tokens}")

        generated_text = completion_payload.get("choices", [{}])[0].get("text", "")
        finish_reason = completion_payload.get("choices", [{}])[0].get("finish_reason")
        tokens_per_second = completion_tokens / elapsed_s

        print(f"model: {MODEL_PATH}")
        print(f"prompt: {PROMPT}")
        print(f"completion_tokens: {completion_tokens}")
        print(f"elapsed_seconds: {elapsed_s:.3f}")
        print(f"tokens_per_second: {tokens_per_second:.3f}")
        print(f"finish_reason: {finish_reason}")
        print(f"generated_text_chars: {len(generated_text)}")
        print("generated_text_begin")
        print(generated_text)
        print("generated_text_end")

        if completion_tokens < MAX_TOKENS:
            print(
                f"warning: requested {MAX_TOKENS} completion tokens but got {completion_tokens}",
                file=sys.stderr,
            )

        return 0
    except Exception:
        if process.stdout is not None:
            try:
                remaining_logs = process.stdout.read()
            except Exception:
                remaining_logs = ""
            if remaining_logs:
                print("vllm_logs_begin", file=sys.stderr)
                print(remaining_logs, file=sys.stderr)
                print("vllm_logs_end", file=sys.stderr)
        raise
    finally:
        stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
