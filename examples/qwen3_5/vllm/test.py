import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


MODEL_PATH = "/var/models/Qwen/Qwen3.5-0.8B"
HOST = "127.0.0.1"
PROMPT = "<|im_start|>Write a very long story about a cat<|im_end|>"
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


def get_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def filter_metrics(metrics_text: str) -> list[str]:
    keep = ("vllm:", "vllm_", "token", "request", "time_to_first", "time_per_output", "prompt")
    lines: list[str] = []
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if any(key in lowered for key in keep):
            lines.append(line)
    return lines


def read_log_tail(path: str, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-max_lines:])


def extract_metric_value(metrics_text: str, metric_name: str) -> float | None:
    prefix = metric_name + "{"
    alt_prefix = metric_name + " "
    for line in metrics_text.splitlines():
        if line.startswith(prefix) or line.startswith(alt_prefix):
            try:
                return float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                return None
    return None


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
    log_file = tempfile.NamedTemporaryFile(mode="w+", prefix="vllm-serve-", suffix=".log", delete=False)
    log_path = log_file.name
    log_file.close()

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
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
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
        metrics_text = get_text(f"{base_url}/metrics", timeout_s=10.0)
        e2e_time_s = extract_metric_value(metrics_text, "vllm:e2e_request_latency_seconds_sum")
        if e2e_time_s is None or e2e_time_s <= 0:
            raise RuntimeError("missing or invalid vllm:e2e_request_latency_seconds_sum metric")
        decode_time_s = extract_metric_value(metrics_text, "vllm:request_decode_time_seconds_sum")
        if decode_time_s is None or decode_time_s <= 0:
            raise RuntimeError("missing or invalid vllm:request_decode_time_seconds_sum metric")
        python_timer_tokens_per_second = completion_tokens / elapsed_s
        end_to_end_tokens_per_second = completion_tokens / e2e_time_s
        decode_only_tokens_per_second = completion_tokens / decode_time_s

        print(f"python_timer_tokens_per_second: {python_timer_tokens_per_second:.3f}")
        print(f"end_to_end_tokens_per_second: {end_to_end_tokens_per_second:.3f}")
        print(f"decode_only_tokens_per_second: {decode_only_tokens_per_second:.3f}")

        if completion_tokens < MAX_TOKENS:
            print(
                f"warning: requested {MAX_TOKENS} completion tokens but got {completion_tokens}",
                file=sys.stderr,
            )

        print("generated_text_begin")
        print(generated_text)
        print("generated_text_end")

        return 0
    except Exception:
        log_tail = read_log_tail(log_path, max_lines=120)
        if log_tail:
            print("vllm_logs_begin", file=sys.stderr)
            print(log_tail, file=sys.stderr, end="")
            print("vllm_logs_end", file=sys.stderr)
        raise
    finally:
        stop_process(process)
        try:
            os.unlink(log_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
