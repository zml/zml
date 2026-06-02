import argparse
import dataclasses
import importlib.util
import os
import sys
import time
from pathlib import Path

import ml_dtypes
import nki
import numpy as np
from python.runfiles import Runfiles


DEFAULT_SEQ_LEN = 128
ATOL = 1e-2
RTOL = 1e-2
MIN_CLOSE_FRACTION = 0.99
INPUT_CASES = ("baseline", "signed")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def heads_per_kv(self):
        return self.num_heads // self.num_kv_heads


MODEL_PRESETS = {
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the classic Neuron NKI attention correctness contract."
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_PRESETS.keys(), "custom"),
        default="llama-3.1-8b",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--dtype", choices=("bf16", "f16", "f32"), default="bf16")
    parser.add_argument("--run", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument("--mode", choices=("verify", "benchmark"), default="verify")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    return parser.parse_args()


def model_spec(args):
    custom_values = (args.num_heads, args.num_kv_heads, args.head_dim)
    if args.model != "custom":
        if any(value is not None for value in custom_values):
            raise ValueError("shape flags are only accepted with --model=custom")
        return MODEL_PRESETS[args.model]

    if any(value is None for value in custom_values):
        raise ValueError("--model=custom requires --num-heads, --num-kv-heads, and --head-dim")
    spec = ModelSpec(
        name="custom",
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    validate_spec(spec)
    return spec


def validate_spec(spec):
    if spec.num_heads <= 0 or spec.num_kv_heads <= 0 or spec.head_dim <= 0:
        raise ValueError("model dimensions must be positive")
    if spec.num_heads % spec.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")


def validate_seq_len(seq_len):
    if seq_len <= 0 or seq_len % 128 != 0:
        raise ValueError("--seq-len must be a positive multiple of 128")


def validate_execution_args(args):
    if args.mode == "verify":
        return
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")


def decode_positions(seq_len):
    return tuple(dict.fromkeys((0, 1, seq_len // 2, seq_len - 1)))


def configure_runtime_environment(mode):
    if mode == "benchmark":
        os.environ.pop("NEURON_RT_LOG_LEVEL", None)


def load_nki_attention_module():
    path = Path(__file__).with_name("attention.py")
    spec = importlib.util.spec_from_file_location("zml_nki_attention", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_neuron_tools_to_path():
    runfiles = Runfiles.Create()
    python_neuronx_cc = Path(
        runfiles.Rlocation("+neuron_packages+libpjrt_neuron/sandbox/bin/python-shims/neuronx-cc")
    )
    neuronx_cc = Path(
        runfiles.Rlocation("+neuron_packages+libpjrt_neuron/sandbox/bin/neuronx-cc")
    )
    os.environ["PATH"] = os.pathsep.join(
        (str(python_neuronx_cc.parent), str(neuronx_cc.parent), os.environ["PATH"])
    )


def numpy_dtype(name):
    if name == "bf16":
        return ml_dtypes.bfloat16
    if name == "f16":
        return np.float16
    if name == "f32":
        return np.float32
    raise ValueError(f"unsupported dtype: {name}")


def fill_input(shape, offset, input_case, dtype):
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), endpoint=False, dtype=np.float32)
    if input_case == "signed":
        data = offset + (values - 0.5) * 0.25
    else:
        data = offset + values * 0.03
    return data.reshape(shape).astype(dtype)


def qkv_inputs(spec, seq_len, input_case, decode, dtype):
    q_len = 1 if decode else seq_len
    q = fill_input((q_len, spec.num_heads, spec.head_dim), 0.01, input_case, dtype)
    k = fill_input((seq_len, spec.num_kv_heads, spec.head_dim), 0.02, input_case, dtype)
    v = fill_input((seq_len, spec.num_kv_heads, spec.head_dim), 0.03, input_case, dtype)
    return q, k, v


def reference_attention(q, k, v, token_index=None):
    q_len, num_heads, head_dim = q.shape
    seq_len, num_kv_heads, _ = k.shape
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)
    v_f32 = v.astype(np.float32)
    out = np.empty((q_len, num_heads, head_dim), dtype=np.float32)

    if token_index is None:
        valid = np.arange(seq_len)[None, :] <= np.arange(q_len)[:, None]
    else:
        if q_len != 1:
            raise ValueError("decode reference expects q_len=1")
        valid = (np.arange(seq_len) <= token_index)[None, :]

    for kv_head in range(num_kv_heads):
        keys = k_f32[:, kv_head, :]
        values = v_f32[:, kv_head, :]
        for q_idx in range(heads_per_kv):
            q_head = kv_head * heads_per_kv + q_idx
            scores = (q_f32[:, q_head, :] @ keys.T) * scale
            scores = np.where(valid, scores, -np.inf)
            scores -= np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= np.sum(probs, axis=1, keepdims=True)
            out[:, q_head, :] = probs @ values

    return out.astype(q.dtype)


def assert_attention_close(label, expected, actual):
    expected_f32 = expected.astype(np.float32)
    actual_f32 = actual.astype(np.float32)
    finite = np.isfinite(actual_f32)
    close = np.isclose(expected_f32, actual_f32, atol=ATOL, rtol=RTOL)
    close_fraction = float(close.mean())
    max_abs = float(np.nanmax(np.abs(expected_f32 - actual_f32)))
    print(
        f"{label} finite={int(finite.sum())}/{actual_f32.size} "
        f"close_fraction={close_fraction:.6f} max_abs={max_abs:.6g}"
    )
    if not finite.all():
        raise AssertionError(f"{label}: output contains NaN or Inf")
    if close_fraction < MIN_CLOSE_FRACTION:
        raise AssertionError(f"{label}: close_fraction={close_fraction:.6f}")


def timed_run(kernel, inputs, mode, warmups, iterations):
    if mode == "verify":
        start = time.perf_counter()
        output = kernel(*inputs)
        elapsed = time.perf_counter() - start
        return output, elapsed, 1

    output = kernel(*inputs)
    for _ in range(warmups):
        output = kernel(*inputs)

    total = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        output = kernel(*inputs)
        total += time.perf_counter() - start
    return output, total, iterations


def print_timing(label, elapsed, iterations, output):
    if iterations == 1:
        print(f"{label} elapsed={elapsed * 1000.0:.3f}ms output={output.shape}")
        return

    average = elapsed / iterations
    print(
        f"{label} total={elapsed * 1000.0:.3f}ms "
        f"avg={average * 1000.0:.3f}ms iterations={iterations} output={output.shape}"
    )


def run_prefill(prefill_kernel, spec, seq_len, input_case, dtype, mode, warmups, iterations):
    q, k, v = qkv_inputs(spec, seq_len, input_case, decode=False, dtype=dtype)
    output, elapsed, timed_iterations = timed_run(
        prefill_kernel,
        (q, k, v),
        mode,
        warmups,
        iterations,
    )
    label = f"prefill input_case={input_case}"
    if mode == "verify":
        expected = reference_attention(q, k, v)
        assert_attention_close(label, expected, output)
    print_timing(label, elapsed, timed_iterations, output)


def run_decode(decode_kernel, spec, seq_len, input_case, token_index, dtype, mode, warmups, iterations):
    q, k, v = qkv_inputs(spec, seq_len, input_case, decode=True, dtype=dtype)
    token = np.array([[token_index]], dtype=np.uint32)
    output, elapsed, timed_iterations = timed_run(
        decode_kernel,
        (q, k, v, token),
        mode,
        warmups,
        iterations,
    )
    label = f"decode input_case={input_case} token_index={token_index}"
    if mode == "verify":
        expected = reference_attention(q, k, v, token_index)
        assert_attention_close(label, expected, output)
    print_timing(label, elapsed, timed_iterations, output)


def main():
    args = parse_args()
    spec = model_spec(args)
    validate_spec(spec)
    validate_seq_len(args.seq_len)
    validate_execution_args(args)
    positions = decode_positions(args.seq_len)
    dtype = numpy_dtype(args.dtype)

    configure_runtime_environment(args.mode)
    add_neuron_tools_to_path()
    module = load_nki_attention_module()
    prefill_kernel = nki.jit(module.prefill)
    decode_kernel = nki.jit(module.decode)

    header = (
        "classic_attention_contract "
        f"model={spec.name} seq_len={args.seq_len} num_heads={spec.num_heads} "
        f"num_kv_heads={spec.num_kv_heads} heads_per_kv={spec.heads_per_kv} "
        f"head_dim={spec.head_dim} dtype={args.dtype} run={args.run} "
        f"mode={args.mode}"
    )
    if args.mode == "benchmark":
        header += f" warmups={args.warmups} iterations={args.iterations}"
    print(f"{header} decode_positions={positions}")
    for input_case in INPUT_CASES:
        if args.run in ("both", "prefill"):
            run_prefill(
                prefill_kernel,
                spec,
                args.seq_len,
                input_case,
                dtype,
                args.mode,
                args.warmups,
                args.iterations,
            )
        if args.run in ("both", "decode"):
            for token_index in positions:
                run_decode(
                    decode_kernel,
                    spec,
                    args.seq_len,
                    input_case,
                    token_index,
                    dtype,
                    args.mode,
                    args.warmups,
                    args.iterations,
                )


if __name__ == "__main__":
    main()
