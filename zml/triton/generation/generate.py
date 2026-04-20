import json
import signal
import sys
import traceback

import triton.backends as triton_backends
from triton.backends import Backend as BackendRegistration
from triton.runtime.driver import driver as runtime_driver

from fake_plugin.compiler import Backend as FakeCompilerBackend
from fake_plugin.driver import Driver as FakeDriver

# Import backend-specific compilers
from attention_compile import compile_attention
from moe_compile import compile_moe


def register_fake_backend() -> None:
    triton_backends.backends["mybackend_runtime"] = BackendRegistration(
        compiler=FakeCompilerBackend,
        driver=FakeDriver,
    )
    runtime_driver.set_active(FakeDriver())


def handle_sigint(signum, frame):
    sys.exit(0)


def compile_request(request: dict) -> str:
    backend = request["backend"]
    kernel = request["kernel"]
    config = request["config"]

    if backend == "attention":
        return compile_attention(kernel, config)
    elif backend == "moe":
        return compile_moe(kernel, config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    register_fake_backend()
    signal.signal(signal.SIGINT, handle_sigint)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")

            ttir = compile_request(request)
            response = {"ok": True, "result": ttir}
        except Exception:
            tb = traceback.format_exc()
            print("[triton_generation] compilation failed", file=sys.stderr, flush=True)
            print(tb, file=sys.stderr, flush=True)
            response = {"ok": False, "error": tb}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
