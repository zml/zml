import argparse
import json
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_PATH = "/var/models/Qwen/Qwen3.5-4B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PROMPT = "<|im_start|>Write a very long story about a cat.<|im_end|>"
DEFAULT_MAX_TOKENS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a running vLLM server.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model id/path to request.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host where the vLLM server is listening.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port where the vLLM server is listening.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to /v1/completions.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Requested max_tokens.")
    parser.add_argument("--server-wait-seconds", type=float, default=30.0, help="How long to wait for /v1/models.")
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0, help="Completion request timeout.")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    wait_for_server(base_url, timeout_s=args.server_wait_seconds)

    request_payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
    }

    started_at = time.perf_counter()
    completion_payload = post_json(f"{base_url}/v1/completions", request_payload, timeout_s=args.request_timeout_seconds)
    elapsed_s = time.perf_counter() - started_at

    usage = completion_payload.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        raise RuntimeError(f"missing completion_tokens in response: {completion_payload}")
    if completion_tokens <= 0:
        raise RuntimeError(f"expected positive completion_tokens, got {completion_tokens}")

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
    generated_text = completion_payload.get("choices", [{}])[0].get("text", "")

    print(f"python_timer_tokens_per_second: {python_timer_tokens_per_second:.3f}")
    print(f"end_to_end_tokens_per_second: {end_to_end_tokens_per_second:.3f}")
    print(f"decode_only_tokens_per_second: {decode_only_tokens_per_second:.3f}")
    print("generated_text_begin")
    print(generated_text)
    print("generated_text_end")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
