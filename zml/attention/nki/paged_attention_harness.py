import argparse
import dataclasses
import importlib.util
import os
import sys
import time
from pathlib import Path

import ml_dtypes
import nki
import nki.language as nl
import numpy as np
from python.runfiles import Runfiles


RUN_CHOICES = (
    "both",
    "prefill",
    "decode",
    "mixed",
    "llmd-loop",
    "paged-update",
    "paged-update-batch2",
    "cache-reuse-update",
    "history-sensitivity",
)

PREFILL_QUERY_BLOCK = 128
ATOL = 1e-2
RTOL = 1e-2
MIN_CLOSE_FRACTION = 0.99


def largest_divisor_at_most(value, limit):
    divisor = min(value, limit)
    if divisor <= 0:
        raise ValueError("tile divisor limit must be positive")
    while value % divisor != 0:
        divisor -= 1
    return divisor


def paged_attention_tile_plan(
    num_tokens,
    batch_size,
    max_num_pages,
    page_size,
    heads_per_kv,
    query_lengths,
    live_seq_len,
    is_decode,
):
    if is_decode:
        q_rows_per_tile = 1
        q_partition_tile = heads_per_kv
    else:
        q_rows_per_tile = int(nl.tile_size.pmax) // heads_per_kv
        q_partition_tile = q_rows_per_tile * heads_per_kv
    k_tile_limit = int(nl.tile_size.gemm_moving_fmax)
    pages_per_k_tile = largest_divisor_at_most(max_num_pages, k_tile_limit // page_size)
    k_tile = pages_per_k_tile * page_size
    pages_per_v_tile = largest_divisor_at_most(pages_per_k_tile, int(nl.tile_size.pmax) // page_size)
    v_tile = pages_per_v_tile * page_size
    k_tiles = max_num_pages // pages_per_k_tile
    live_k_tiles = (live_seq_len + k_tile - 1) // k_tile
    k_tiles_per_segment = largest_divisor_at_most(k_tiles, min(k_tiles, 4)) if is_decode else 1
    k_segments = k_tiles // k_tiles_per_segment
    compiled_q_blocks = (num_tokens + q_rows_per_tile - 1) // q_rows_per_tile
    if query_lengths is None:
        live_q_per_seq = 1 if is_decode else num_tokens // batch_size
        live_q_blocks = batch_size * ((live_q_per_seq + q_rows_per_tile - 1) // q_rows_per_tile)
    else:
        live_q_blocks = sum((query_len + q_rows_per_tile - 1) // q_rows_per_tile for query_len in query_lengths)
    return (
        q_rows_per_tile,
        q_partition_tile,
        compiled_q_blocks,
        live_q_blocks,
        k_tile,
        v_tile,
        k_tiles,
        live_k_tiles,
        k_tiles_per_segment,
        k_segments,
    )


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
        description="Run the Neuron NKI paged-attention correctness contract."
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_PRESETS.keys(), "custom"),
        default="llama-3.1-8b",
    )
    parser.add_argument("--run", choices=RUN_CHOICES, default="both")
    parser.add_argument("--mode", choices=("verify", "benchmark"), default="verify")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-token-count",
        type=int,
        default=None,
        help=(
            "Compiled query-token capacity for prefill/mixed. Defaults to "
            "prompt-len * batch-size, matching LLMD-style padded capacity."
        ),
    )
    parser.add_argument("--active-lanes", type=int, default=None)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--query-lengths", type=str, default="128,0")
    parser.add_argument("--seq-lens-per-seq", type=str, default="129,0")
    parser.add_argument("--context-len", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--permute-pages", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--minimum-close-fraction", type=float, default=MIN_CLOSE_FRACTION)
    return parser.parse_args()


def resolve_model_spec(args):
    custom_values = (args.num_heads, args.num_kv_heads, args.head_dim)
    if args.model != "custom":
        if any(value is not None for value in custom_values):
            raise ValueError("shape flags are only accepted with --model=custom")
        return MODEL_PRESETS[args.model]

    if any(value is None for value in custom_values):
        raise ValueError("--model=custom requires --num-heads, --num-kv-heads, and --head-dim")
    return ModelSpec(
        name="custom",
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )


def install_model_spec(args, spec):
    args.num_heads = spec.num_heads
    args.num_kv_heads = spec.num_kv_heads
    args.head_dim = spec.head_dim


def load_paged_attention_module(module_name="zml_nki_paged_attention"):
    path = Path(__file__).with_name("paged_attention.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
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


def configure_runtime_environment(mode):
    if mode == "benchmark":
        os.environ.pop("NEURON_RT_LOG_LEVEL", None)


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_token_count is not None and args.max_token_count <= 0:
        raise ValueError("--max-token-count must be positive")
    if args.num_heads <= 0 or args.num_kv_heads <= 0 or args.head_dim <= 0:
        raise ValueError("model dimensions must be positive")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.active_lanes is not None and not (0 < args.active_lanes <= args.batch_size):
        raise ValueError("--active-lanes must be in 1..--batch-size")
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")
    if args.page_size not in (32, 64):
        raise ValueError("--page-size must be 32 or 64 for the NKI paged-attention bench path")
    if args.seq_len % args.page_size != 0:
        raise ValueError("--seq-len must be a multiple of --page-size")
    if args.context_len < 0:
        raise ValueError("--context-len must be non-negative")
    if args.mode == "benchmark":
        if args.warmups < 0:
            raise ValueError("--warmups must be non-negative")
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
    if args.decode_steps <= 0:
        raise ValueError("--decode-steps must be positive")


def fill_bf16(shape, offset):
    data = np.empty(int(np.prod(shape)), dtype=np.float32)
    idx = np.arange(data.size, dtype=np.float32)
    data[:] = (idx % 31) * 0.001 + offset
    return data.reshape(shape).astype(ml_dtypes.bfloat16)


def position_bf16(shape, token_pos, salt):
    idx = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    pos = float(token_pos + 1)
    values = (
        np.sin(idx * 0.013 + pos * 0.017 + salt) * 0.08
        + np.cos((idx % 67.0) * 0.031 + pos * 0.0011 + salt) * 0.02
    )
    return values.astype(ml_dtypes.bfloat16)


def synthetic_qkv_for_position(token_pos, args, heads_per_kv):
    q = position_bf16(
        (1, args.num_kv_heads, heads_per_kv, args.head_dim),
        token_pos,
        0.11,
    )
    k = position_bf16((1, args.num_kv_heads, args.head_dim), token_pos, 0.23)
    v = position_bf16((1, args.num_kv_heads, args.head_dim), token_pos, 0.37)
    return q, k, v


def parse_int_list(value, name):
    if value is None:
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            parsed = int(part)
            if parsed < 0:
                raise ValueError(f"{name} entries must be non-negative, got {parsed}")
            out.append(parsed)
    if not out:
        raise ValueError(f"{name} must contain at least one integer")
    return out


def prompt_length(args, query_lengths=None):
    prompt_len = args.prompt_len
    if prompt_len is None:
        prompt_len = max(query_lengths) if query_lengths is not None else args.seq_len
    if prompt_len <= 0 or prompt_len > args.seq_len:
        raise ValueError(f"--prompt-len must be in 1..{args.seq_len}")
    return prompt_len


def pages_per_sequence(args):
    return args.seq_len // args.page_size


def total_cache_pages(args, pages_per_seq):
    return args.batch_size * pages_per_seq


def round_up_to_prefill_block(num_tokens):
    return (
        (num_tokens + PREFILL_QUERY_BLOCK - 1) // PREFILL_QUERY_BLOCK
    ) * PREFILL_QUERY_BLOCK


def make_metadata(args, current_seq_len, num_tokens, query_lengths=None, seq_lens_per_seq=None):
    if current_seq_len <= 0 or current_seq_len > args.seq_len:
        raise ValueError(f"current_seq_len must be in 1..{args.seq_len}, got {current_seq_len}")
    if query_lengths is not None and len(query_lengths) != args.batch_size:
        raise ValueError(f"--query-lengths must have {args.batch_size} entries")
    if seq_lens_per_seq is not None and len(seq_lens_per_seq) != args.batch_size:
        raise ValueError(f"--seq-lens-per-seq must have {args.batch_size} entries")

    pages_per_seq = pages_per_sequence(args)
    total_pages = total_cache_pages(args, pages_per_seq)
    block_table = np.zeros((args.batch_size, pages_per_seq), dtype=np.int32)
    for batch_idx in range(args.batch_size):
        base = batch_idx * pages_per_seq
        for logical_page in range(pages_per_seq):
            block_table[batch_idx, logical_page] = base + logical_page
    if args.permute_pages:
        block_table = np.flip(block_table, axis=1).copy()

    if query_lengths is None:
        if num_tokens % args.batch_size != 0:
            raise ValueError("compiled query token count must be divisible by --batch-size")
        queries_per_seq = num_tokens // args.batch_size
        seq_lens = np.full((args.batch_size, 1), current_seq_len, dtype=np.int32)
        query_start_len = (
            np.arange(args.batch_size + 1, dtype=np.int32) * queries_per_seq
        ).reshape(args.batch_size + 1, 1)
        return block_table, seq_lens, query_start_len

    live_query_tokens = sum(query_lengths)
    if live_query_tokens > num_tokens:
        raise ValueError("sum(--query-lengths) must be <= compiled query token count")
    if seq_lens_per_seq is None:
        seq_lens_values = [args.context_len + query_len for query_len in query_lengths]
    else:
        seq_lens_values = seq_lens_per_seq
    for query_len, seq_len in zip(query_lengths, seq_lens_values):
        if query_len != 0 and not (query_len <= seq_len <= args.seq_len):
            raise ValueError("seq_lens entries must satisfy query_len <= seq_len <= --seq-len")

    seq_lens = np.asarray(seq_lens_values, dtype=np.int32).reshape(args.batch_size, 1)
    query_start_len = np.zeros((args.batch_size + 1, 1), dtype=np.int32)
    cursor = 0
    for idx, query_len in enumerate(query_lengths):
        query_start_len[idx, 0] = cursor
        cursor += query_len
    query_start_len[args.batch_size, 0] = cursor
    return block_table, seq_lens, query_start_len


def validate_wide_q_tile_capacity(query_start_len, num_tokens, q_per_head_tile):
    """Require enough compiled Q rows for wide AP Q load/store.

    The NKI prefill/mixed fast path loads and stores q_per_head_tile rows at a
    time for each GQA head group. That is the fast hardware shape; compact
    ragged schedules must therefore be padded so the final live tile remains in
    bounds even when a sequence has only a few live rows.
    """
    for batch_idx in range(query_start_len.shape[0] - 1):
        q_start = int(query_start_len[batch_idx, 0])
        q_end = int(query_start_len[batch_idx + 1, 0])
        query_len = q_end - q_start
        if query_len == 0:
            continue
        last_tile_start = q_start + ((query_len - 1) // q_per_head_tile) * q_per_head_tile
        required_rows = last_tile_start + q_per_head_tile
        if required_rows > num_tokens:
            raise ValueError(
                "wide Q-tile fast path needs padded compiled capacity: "
                f"batch={batch_idx} q_start={q_start} query_len={query_len} "
                f"q_per_head_tile={q_per_head_tile} requires {required_rows} rows, "
                f"but compiled num_tokens is {num_tokens}. Increase --max-token-count "
                "or use an explicitly safe row-granular debug kernel."
            )


def scatter_logical_kv_to_pages(k_logical, v_logical, block_table, page_size, total_pages):
    if k_logical.ndim == 3:
        k_logical = k_logical[None, :, :, :]
        v_logical = v_logical[None, :, :, :]
    _, _, num_kv_heads, head_dim = k_logical.shape
    k_cache = np.zeros((total_pages, page_size, num_kv_heads, head_dim), dtype=k_logical.dtype)
    v_cache = np.zeros_like(k_cache)
    k_private = k_cache.reshape((total_pages, num_kv_heads, head_dim, page_size))
    for batch_idx in range(block_table.shape[0]):
        for logical_page in range(block_table.shape[1]):
            physical_page = block_table[batch_idx, logical_page]
            start = logical_page * page_size
            stop = start + page_size
            k_private[physical_page, :, :, :] = np.transpose(
                k_logical[batch_idx, start:stop, :, :],
                (1, 2, 0),
            )
            v_cache[physical_page, :, :, :] = v_logical[batch_idx, start:stop, :, :]
    return k_cache, v_cache


def flattened_cache_slot(block_table, batch_idx, token_pos, page_size):
    logical_page = token_pos // page_size
    page_offset = token_pos % page_size
    return int(block_table[batch_idx, logical_page]) * page_size + page_offset


def private_k_flat_index(slot, num_kv_heads, head_dim, page_size, kv_head=0, hd_idx=0):
    physical_page = slot // page_size
    page_offset = slot % page_size
    return ((physical_page * num_kv_heads + kv_head) * head_dim + hd_idx) * page_size + page_offset


def read_private_k_scalar(k_cache, slot, num_kv_heads, head_dim, page_size, kv_head=0, hd_idx=0):
    k_private_flat = k_cache.reshape(-1)
    return float(k_private_flat[private_k_flat_index(slot, num_kv_heads, head_dim, page_size, kv_head, hd_idx)])


def write_logical_token(k_logical, v_logical, token_pos, k_value, v_value):
    if k_logical.ndim == 4:
        k_logical[0, token_pos : token_pos + 1, :, :] = k_value
        v_logical[0, token_pos : token_pos + 1, :, :] = v_value
    else:
        k_logical[token_pos : token_pos + 1, :, :] = k_value
        v_logical[token_pos : token_pos + 1, :, :] = v_value


def reference_attention(q, k_logical, v_logical, seq_len, used_queries, batch_idx=0):
    num_tokens, num_kv_heads, heads_per_kv, head_dim = q.shape
    if k_logical.ndim == 4:
        k_logical = k_logical[batch_idx]
        v_logical = v_logical[batch_idx]
    q_f32 = q.astype(np.float32)
    k_f32 = k_logical.astype(np.float32)
    v_f32 = v_logical.astype(np.float32)
    out = np.zeros((num_tokens, num_kv_heads, heads_per_kv, head_dim), dtype=np.float32)
    scale = 1.0 / np.sqrt(head_dim)
    first_query_pos = seq_len - used_queries

    for query_idx in range(used_queries):
        query_pos = first_query_pos + query_idx
        for kv_head in range(num_kv_heads):
            keys = k_f32[:seq_len, kv_head, :]
            values = v_f32[:seq_len, kv_head, :]
            for head_group in range(heads_per_kv):
                scores = (q_f32[query_idx, kv_head, head_group, :] @ keys.T) * scale
                if query_pos + 1 < seq_len:
                    scores[query_pos + 1 :] = -np.inf
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                out[query_idx, kv_head, head_group, :] = probs @ values
    return out.astype(ml_dtypes.bfloat16)


def assert_close(label, expected, actual, args, used_queries=None):
    if used_queries is not None:
        expected = expected[:used_queries]
        actual = actual[:used_queries]
    expected_f32 = expected.astype(np.float32)
    actual_f32 = actual.astype(np.float32)
    finite = np.isfinite(actual_f32)
    close = np.isclose(expected_f32, actual_f32, atol=args.atol, rtol=args.rtol)
    max_abs = float(np.nanmax(np.abs(expected_f32 - actual_f32)))
    close_fraction = float(close.mean())
    print(
        f"{label} finite={int(finite.sum())}/{actual_f32.size} "
        f"close_fraction={close_fraction:.6f} max_abs={max_abs:.6g}"
    )
    if not finite.all():
        raise AssertionError(f"{label}: output contains NaN or Inf")
    if close_fraction < args.minimum_close_fraction:
        raise AssertionError(f"{label}: close_fraction={close_fraction:.6f}")


def print_delta(label, previous, current):
    if previous is None:
        return None
    delta = current.astype(np.float32) - previous.astype(np.float32)
    max_abs = float(np.nanmax(np.abs(delta)))
    l2 = float(np.linalg.norm(delta.reshape(-1)))
    print(f"{label} delta_from_previous max_abs={max_abs:.6g} l2={l2:.6g}")
    return max_abs


def expected_fill_bf16_value(flat_index, offset):
    return float(ml_dtypes.bfloat16(float(flat_index % 31) * 0.001 + offset))


def assert_close_scalar(label, expected, actual, tolerance=1e-3):
    print(f"{label} expected={expected:.6g} actual={actual:.6g}")
    if abs(float(expected) - float(actual)) >= tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def launch_kernel(kernel, inputs):
    return kernel(*inputs)


def timed_kernel(kernel, inputs, args):
    if args.mode == "verify":
        start = time.perf_counter()
        output = launch_kernel(kernel, inputs)
        elapsed = time.perf_counter() - start
        return output, elapsed, 1

    start = time.perf_counter()
    output = launch_kernel(kernel, inputs)
    print(f"kernel_compile_and_first_run elapsed={(time.perf_counter() - start) * 1000.0:.3f}ms", flush=True)
    for _ in range(args.warmups):
        output = launch_kernel(kernel, inputs)

    total = 0.0
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = launch_kernel(kernel, inputs)
        total += time.perf_counter() - start
    return output, total, args.iterations


def print_timing(label, elapsed, iterations):
    if iterations == 1:
        print(f"{label} elapsed={elapsed * 1000.0:.3f}ms")
        return
    print(
        f"{label} total={elapsed * 1000.0:.3f}ms "
        f"avg={(elapsed / iterations) * 1000.0:.3f}ms iterations={iterations}"
    )


def run_attention_case(args, update_kernel, kernel, heads_per_kv, kind):
    query_lengths = parse_int_list(args.query_lengths, "--query-lengths") if kind == "mixed" else None
    seq_lens_per_seq = parse_int_list(args.seq_lens_per_seq, "--seq-lens-per-seq") if kind == "mixed" else None
    prompt_len = prompt_length(args, query_lengths)
    if args.context_len + prompt_len > args.seq_len:
        raise ValueError("--context-len + --prompt-len must be <= --seq-len")

    if kind == "decode":
        num_tokens = args.batch_size
    elif kind == "mixed":
        if query_lengths is None:
            raise ValueError("--run=mixed requires --query-lengths")
        num_tokens = args.max_token_count or (prompt_len * args.batch_size)
        num_tokens = max(num_tokens, round_up_to_prefill_block(sum(query_lengths)))
    else:
        num_tokens = args.max_token_count or (prompt_len * args.batch_size)

    is_decode = kind == "decode"
    if not is_decode and num_tokens % PREFILL_QUERY_BLOCK != 0:
        raise ValueError(
            "prefill/mixed compiled query tokens must be a multiple of "
            f"{PREFILL_QUERY_BLOCK}"
        )
    if kind == "prefill" and prompt_len > num_tokens // args.batch_size:
        raise ValueError("--prompt-len must be <= compiled tokens per batch row")

    metadata_query_lengths = query_lengths if kind == "mixed" else None
    metadata_seq_lens = seq_lens_per_seq
    if kind == "decode" and args.active_lanes is not None:
        metadata_query_lengths = [1 if idx < args.active_lanes else 0 for idx in range(args.batch_size)]
        metadata_seq_lens = [
            args.context_len + 1 if idx < args.active_lanes else 0
            for idx in range(args.batch_size)
        ]
    if metadata_query_lengths is None and not is_decode:
        metadata_query_lengths = [prompt_len] * args.batch_size
    current_seq_len = max(seq_lens_per_seq) if seq_lens_per_seq is not None else args.context_len + prompt_len
    block_table, seq_lens, query_start_len = make_metadata(
        args,
        current_seq_len,
        num_tokens,
        query_lengths=metadata_query_lengths,
        seq_lens_per_seq=metadata_seq_lens,
    )
    if not is_decode:
        validate_wide_q_tile_capacity(
            query_start_len,
            num_tokens,
            PREFILL_QUERY_BLOCK // heads_per_kv,
        )
    total_pages = total_cache_pages(args, block_table.shape[1])
    k_logical = fill_bf16((args.batch_size, args.seq_len, args.num_kv_heads, args.head_dim), 0.02)
    v_logical = fill_bf16((args.batch_size, args.seq_len, args.num_kv_heads, args.head_dim), 0.03)
    cache_k_logical = k_logical.copy()
    cache_v_logical = v_logical.copy()
    if is_decode:
        for batch_idx in range(args.batch_size):
            if metadata_query_lengths is None:
                used_queries = 1
                seq_len = current_seq_len
            else:
                q_start = int(query_start_len[batch_idx, 0])
                used_queries = int(query_start_len[batch_idx + 1, 0]) - q_start
                seq_len = int(seq_lens[batch_idx, 0])
            if used_queries == 0:
                continue
            token_start = seq_len - used_queries
            # Decode must rely on paged_kv_cache_update for these rows.
            # Poison them in the input cache so the harness catches update bugs.
            cache_k_logical[batch_idx, token_start:seq_len, :, :] = ml_dtypes.bfloat16(-0.625)
            cache_v_logical[batch_idx, token_start:seq_len, :, :] = ml_dtypes.bfloat16(0.875)
    k_cache, v_cache = scatter_logical_kv_to_pages(
        cache_k_logical, cache_v_logical, block_table, args.page_size, total_pages
    )
    q = fill_bf16((num_tokens, args.num_kv_heads, heads_per_kv, args.head_dim), 0.01)
    if metadata_query_lengths is not None and sum(metadata_query_lengths) < num_tokens:
        q[sum(metadata_query_lengths) :, :, :, :] = 0
    new_k = np.zeros((num_tokens, args.num_kv_heads, args.head_dim), dtype=ml_dtypes.bfloat16)
    new_v = np.zeros_like(new_k)
    slot_mapping = np.full((num_tokens,), np.iinfo(np.int32).max, dtype=np.int32)
    for batch_idx in range(args.batch_size):
        if metadata_query_lengths is None:
            used_queries = 1 if is_decode else num_tokens // args.batch_size
            q_start = batch_idx * used_queries
            seq_len = current_seq_len
        else:
            q_start = int(query_start_len[batch_idx, 0])
            used_queries = int(query_start_len[batch_idx + 1, 0]) - q_start
            seq_len = int(seq_lens[batch_idx, 0])
        if used_queries == 0:
            continue
        q_stop = q_start + used_queries
        token_start = seq_len - used_queries
        new_k[q_start:q_stop] = k_logical[batch_idx, token_start:seq_len]
        new_v[q_start:q_stop] = v_logical[batch_idx, token_start:seq_len]
        for query_idx, token_pos in enumerate(range(token_start, seq_len)):
            slot_mapping[q_start + query_idx] = flattened_cache_slot(
                block_table, batch_idx, token_pos, args.page_size
            )
    print(
        "paged_attention_contract "
        f"model={args.model} run={kind} mode={args.mode} batch_size={args.batch_size} "
        f"num_tokens={num_tokens} "
        f"context_len={args.context_len} prompt_len={prompt_len} seq_len={args.seq_len} "
        f"page_size={args.page_size} pages={total_pages} heads={args.num_heads} "
        f"kv_heads={args.num_kv_heads} head_dim={args.head_dim} permute_pages={args.permute_pages}",
        flush=True,
    )
    (
        q_rows_per_tile,
        q_partition_tile,
        compiled_q_blocks,
        live_q_blocks,
        k_tile,
        v_tile,
        k_tiles,
        live_k_tiles,
        k_tiles_per_segment,
        k_segments,
    ) = paged_attention_tile_plan(
        num_tokens,
        args.batch_size,
        block_table.shape[1],
        args.page_size,
        heads_per_kv,
        metadata_query_lengths,
        current_seq_len,
        is_decode,
    )
    print(
        "paged_attention_loop_product "
        f"q_rows_per_tile={q_rows_per_tile} q_partition_tile={q_partition_tile} "
        f"live_q_blocks={live_q_blocks} compiled_q_blocks={compiled_q_blocks} "
        f"k_tile={k_tile} v_tile={v_tile} "
        f"k_tiles_per_segment={k_tiles_per_segment} k_segments={k_segments} "
        f"k_tiles={k_tiles} live_k_tiles={live_k_tiles} "
        f"kv_work_product={live_q_blocks * args.num_kv_heads} "
        f"query_head_rows={live_q_blocks * args.num_heads}",
        flush=True,
    )
    kernel_label = "paged_attention_decode_2d" if is_decode else "paged_attention_2d"
    if is_decode:
        k_cache, v_cache = update_kernel(
            k_cache,
            v_cache,
            new_k,
            new_v,
            slot_mapping,
            query_start_len,
        )
    output, elapsed, timed_iterations = timed_kernel(
        kernel,
        (q, k_cache, v_cache, block_table, seq_lens, query_start_len),
        args,
    )
    print_timing(f"kernel={kernel_label} label={kind}", elapsed, timed_iterations)
    if args.mode != "verify":
        return

    expected = np.zeros_like(q)
    for batch_idx in range(args.batch_size):
        if metadata_query_lengths is None:
            used_queries = 1 if is_decode else num_tokens // args.batch_size
            q_start = batch_idx * used_queries
            seq_len = current_seq_len
        else:
            q_start = int(query_start_len[batch_idx, 0])
            used_queries = int(query_start_len[batch_idx + 1, 0]) - q_start
            seq_len = int(seq_lens[batch_idx, 0])
        if used_queries == 0:
            continue
        q_stop = q_start + used_queries
        expected[q_start:q_stop] = reference_attention(
            q[q_start:q_stop], k_logical, v_logical, seq_len, used_queries, batch_idx
        )
        assert_close(f"{kind} batch={batch_idx}", expected[q_start:q_stop], output[q_start:q_stop], args)

    live_queries = int(query_start_len[-1, 0]) if metadata_query_lengths is not None else None
    assert_close(f"{kind} all", expected, output, args, live_queries)


def run_update_contract(args, module):
    if args.run == "cache-reuse-update":
        return run_cache_reuse_update(args, module)

    compiled_tokens = PREFILL_QUERY_BLOCK
    live_tokens = 51
    if args.page_size < 2:
        raise ValueError("--run=paged-update requires --page-size >= 2")
    total_pages = max(pages_per_sequence(args), 4)
    elems_per_token = args.num_kv_heads * args.head_dim
    first_slot = 2 * args.page_size
    second_slot = first_slot + 1
    inactive_slot = 0
    preserved_slot = first_slot + live_tokens + 1
    if total_pages * args.page_size <= preserved_slot:
        raise ValueError("--run=paged-update requires enough cache slots")

    k_update = np.empty((compiled_tokens, args.num_kv_heads, args.head_dim), dtype=ml_dtypes.bfloat16)
    v_update = np.empty_like(k_update)
    for token_idx in range(compiled_tokens):
        k_update[token_idx, :, :] = fill_bf16((args.num_kv_heads, args.head_dim), float(token_idx + 1) / 100.0)
        v_update[token_idx, :, :] = fill_bf16((args.num_kv_heads, args.head_dim), float(token_idx + 1) / 50.0)

    k_cache = fill_bf16((total_pages, args.page_size, args.num_kv_heads, args.head_dim), 0.125)
    v_cache = fill_bf16((total_pages, args.page_size, args.num_kv_heads, args.head_dim), 0.25)
    slot_mapping = np.full((compiled_tokens,), np.iinfo(np.int32).max, dtype=np.int32)
    slot_mapping[:live_tokens] = np.arange(first_slot, first_slot + live_tokens, dtype=np.int32)
    batch_size = 2 if args.run == "paged-update-batch2" else 1
    query_start_len = np.full((batch_size + 1, 1), live_tokens, dtype=np.int32)
    query_start_len[0, 0] = 0

    print(
        "paged_update_contract "
        f"run={args.run} batch_size={batch_size} compiled_tokens={compiled_tokens} "
        f"live_tokens={live_tokens} seq_len={args.seq_len} page_size={args.page_size}"
    )
    kernel = nki.jit(module.paged_kv_cache_update)
    start = time.perf_counter()
    k_output, v_output = kernel(k_cache, v_cache, k_update, v_update, slot_mapping, query_start_len)
    print(f"paged_update elapsed={(time.perf_counter() - start) * 1000.0:.3f}ms")

    v_flat = v_output.reshape((total_pages * args.page_size, elems_per_token))
    assert_close_scalar(
        "paged_update first_slot_k",
        0.01,
        read_private_k_scalar(k_output, first_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar("paged_update first_slot_v", 0.02, float(v_flat[first_slot, 0]))
    assert_close_scalar(
        "paged_update second_slot_k",
        0.02,
        read_private_k_scalar(k_output, second_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar("paged_update second_slot_v", 0.04, float(v_flat[second_slot, 0]))
    assert_close_scalar(
        "paged_update inactive_slot_k",
        expected_fill_bf16_value(
            private_k_flat_index(inactive_slot, args.num_kv_heads, args.head_dim, args.page_size),
            0.125,
        ),
        read_private_k_scalar(k_output, inactive_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar(
        "paged_update inactive_slot_v",
        expected_fill_bf16_value(inactive_slot * elems_per_token, 0.25),
        float(v_flat[inactive_slot, 0]),
    )
    assert_close_scalar(
        "paged_update preserved_slot_k",
        expected_fill_bf16_value(
            private_k_flat_index(preserved_slot, args.num_kv_heads, args.head_dim, args.page_size),
            0.125,
        ),
        read_private_k_scalar(k_output, preserved_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar(
        "paged_update preserved_slot_v",
        expected_fill_bf16_value(preserved_slot * elems_per_token, 0.25),
        float(v_flat[preserved_slot, 0]),
    )


def run_cache_reuse_update(args, module):
    if args.seq_len < 2:
        raise ValueError("--run=cache-reuse-update requires --seq-len >= 2")
    total_pages = pages_per_sequence(args)
    elems_per_token = args.num_kv_heads * args.head_dim
    first_slot = args.seq_len - 2
    second_slot = args.seq_len - 1
    k_cache = np.zeros((total_pages, args.page_size, args.num_kv_heads, args.head_dim), dtype=ml_dtypes.bfloat16)
    v_cache = np.zeros_like(k_cache)
    query_start_len = np.array([[0], [1]], dtype=np.int32)
    kernel = nki.jit(module.paged_kv_cache_update)

    first_k = fill_bf16((1, args.num_kv_heads, args.head_dim), 0.42)
    first_v = fill_bf16((1, args.num_kv_heads, args.head_dim), 0.52)
    k_cache, v_cache = kernel(
        k_cache, v_cache, first_k, first_v, np.array([first_slot], dtype=np.int32), query_start_len
    )
    second_k = fill_bf16((1, args.num_kv_heads, args.head_dim), 0.73)
    second_v = fill_bf16((1, args.num_kv_heads, args.head_dim), 0.83)
    k_output, v_output = kernel(
        k_cache, v_cache, second_k, second_v, np.array([second_slot], dtype=np.int32), query_start_len
    )

    print(f"cache_reuse_update_contract seq_len={args.seq_len} page_size={args.page_size}")
    v_flat = v_output.reshape((total_pages * args.page_size, elems_per_token))
    assert_close_scalar(
        "cache_reuse first_slot_k",
        expected_fill_bf16_value(0, 0.42),
        read_private_k_scalar(k_output, first_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar("cache_reuse first_slot_v", expected_fill_bf16_value(0, 0.52), float(v_flat[first_slot, 0]))
    assert_close_scalar(
        "cache_reuse second_slot_k",
        expected_fill_bf16_value(0, 0.73),
        read_private_k_scalar(k_output, second_slot, args.num_kv_heads, args.head_dim, args.page_size),
    )
    assert_close_scalar("cache_reuse second_slot_v", expected_fill_bf16_value(0, 0.83), float(v_flat[second_slot, 0]))


def run_history_sensitivity(args, update_kernel, kernel, heads_per_kv):
    prompt_len = prompt_length(args)
    seq_len = args.context_len + prompt_len
    if seq_len < 2:
        raise ValueError("--run=history-sensitivity requires at least two tokens")
    block_table, _, _ = make_metadata(args, seq_len, args.batch_size)
    total_pages = total_cache_pages(args, block_table.shape[1])
    k_a = fill_bf16((args.batch_size, args.seq_len, args.num_kv_heads, args.head_dim), 0.02)
    v_a = fill_bf16((args.batch_size, args.seq_len, args.num_kv_heads, args.head_dim), 0.03)
    k_b = k_a.copy()
    v_b = v_a.copy()
    k_b[0, : seq_len - 1, :, :] = fill_bf16((seq_len - 1, args.num_kv_heads, args.head_dim), 0.77)
    v_b[0, : seq_len - 1, :, :] = fill_bf16((seq_len - 1, args.num_kv_heads, args.head_dim), 0.91)
    k_cache_a, v_cache_a = scatter_logical_kv_to_pages(k_a, v_a, block_table, args.page_size, total_pages)
    k_cache_b, v_cache_b = scatter_logical_kv_to_pages(k_b, v_b, block_table, args.page_size, total_pages)
    q = fill_bf16((1, args.num_kv_heads, heads_per_kv, args.head_dim), 0.41)
    seq_lens = np.array([[seq_len]], dtype=np.int32)
    query_start_len = np.array([[0], [1]], dtype=np.int32)
    new_k = k_a[0, seq_len - 1 : seq_len, :, :]
    new_v = v_a[0, seq_len - 1 : seq_len, :, :]
    slot_mapping = np.array(
        [flattened_cache_slot(block_table, 0, seq_len - 1, args.page_size)],
        dtype=np.int32,
    )
    k_cache_a, v_cache_a = update_kernel(k_cache_a, v_cache_a, new_k, new_v, slot_mapping, query_start_len)
    k_cache_b, v_cache_b = update_kernel(k_cache_b, v_cache_b, new_k, new_v, slot_mapping, query_start_len)
    out_a = kernel(q, k_cache_a, v_cache_a, block_table, seq_lens, query_start_len)
    out_b = kernel(q, k_cache_b, v_cache_b, block_table, seq_lens, query_start_len)
    delta = out_a.astype(np.float32) - out_b.astype(np.float32)
    max_abs = float(np.nanmax(np.abs(delta)))
    l2 = float(np.linalg.norm(delta.reshape(-1)))
    print(f"history_sensitivity max_abs_delta={max_abs:.6g} l2_delta={l2:.6g}")
    if max_abs == 0.0:
        raise AssertionError("decode output did not change when only historical cache contents changed")


def run_llmd_loop(args, update_kernel, prefill_kernel, decode_kernel, heads_per_kv):
    active_lanes = args.active_lanes or 1
    if active_lanes > args.batch_size:
        raise ValueError("--active-lanes must be <= --batch-size")
    prompt_len = prompt_length(args)
    if args.context_len + prompt_len + args.decode_steps > args.seq_len:
        raise ValueError("--context-len + --prompt-len + --decode-steps must be <= --seq-len")
    prefill_tokens = round_up_to_prefill_block(prompt_len * active_lanes)
    prefill_query_lengths = [
        prompt_len if batch_idx < active_lanes else 0
        for batch_idx in range(args.batch_size)
    ]
    prefill_seq_lens = [
        args.context_len + prompt_len if batch_idx < active_lanes else 0
        for batch_idx in range(args.batch_size)
    ]
    prefill_seq_len = args.context_len + prompt_len

    block_table, seq_lens_prefill, prefill_query_start_len = make_metadata(
        args,
        prefill_seq_len,
        prefill_tokens,
        query_lengths=prefill_query_lengths,
        seq_lens_per_seq=prefill_seq_lens,
    )
    total_pages = total_cache_pages(args, block_table.shape[1])
    k_logical = np.zeros(
        (args.batch_size, args.seq_len, args.num_kv_heads, args.head_dim),
        dtype=ml_dtypes.bfloat16,
    )
    v_logical = np.zeros_like(k_logical)
    q_prefill = np.zeros((prefill_tokens, args.num_kv_heads, heads_per_kv, args.head_dim), dtype=ml_dtypes.bfloat16)
    for batch_idx in range(active_lanes):
        for token_pos in range(args.context_len):
            _, k_tok, v_tok = synthetic_qkv_for_position(
                token_pos + batch_idx * args.seq_len,
                args,
                heads_per_kv,
            )
            k_logical[batch_idx, token_pos : token_pos + 1, :, :] = k_tok
            v_logical[batch_idx, token_pos : token_pos + 1, :, :] = v_tok
        q_start = int(prefill_query_start_len[batch_idx, 0])
        for query_idx in range(prompt_len):
            token_pos = args.context_len + query_idx
            q_tok, k_tok, v_tok = synthetic_qkv_for_position(
                token_pos + batch_idx * args.seq_len,
                args,
                heads_per_kv,
            )
            q_prefill[q_start + query_idx : q_start + query_idx + 1, :, :, :] = q_tok
            k_logical[batch_idx, token_pos : token_pos + 1, :, :] = k_tok
            v_logical[batch_idx, token_pos : token_pos + 1, :, :] = v_tok

    cache_k_logical = np.zeros_like(k_logical)
    cache_v_logical = np.zeros_like(v_logical)
    cache_k_logical[:, : args.context_len, :, :] = k_logical[:, : args.context_len, :, :]
    cache_v_logical[:, : args.context_len, :, :] = v_logical[:, : args.context_len, :, :]
    k_cache, v_cache = scatter_logical_kv_to_pages(
        cache_k_logical, cache_v_logical, block_table, args.page_size, total_pages
    )
    print(
        "llmd_loop_contract "
        f"model={args.model} batch_size={args.batch_size} active_lanes={active_lanes} "
        f"context_len={args.context_len} "
        f"prompt_len={prompt_len} prefill_tokens={prefill_tokens} "
        f"decode_steps={args.decode_steps} seq_len={args.seq_len} page_size={args.page_size}"
    )
    prefill_k = np.zeros((prefill_tokens, args.num_kv_heads, args.head_dim), dtype=ml_dtypes.bfloat16)
    prefill_v = np.zeros_like(prefill_k)
    prefill_slot_mapping = np.full((prefill_tokens,), np.iinfo(np.int32).max, dtype=np.int32)
    for batch_idx in range(active_lanes):
        q_start = int(prefill_query_start_len[batch_idx, 0])
        q_stop = int(prefill_query_start_len[batch_idx + 1, 0])
        prefill_k[q_start:q_stop] = k_logical[
            batch_idx,
            args.context_len : args.context_len + prompt_len,
            :,
            :,
        ]
        prefill_v[q_start:q_stop] = v_logical[
            batch_idx,
            args.context_len : args.context_len + prompt_len,
            :,
            :,
        ]
        for query_idx, token_pos in enumerate(range(args.context_len, prefill_seq_len)):
            prefill_slot_mapping[q_start + query_idx] = flattened_cache_slot(
                block_table, batch_idx, token_pos, args.page_size
            )
    k_cache, v_cache = update_kernel(
        k_cache,
        v_cache,
        prefill_k,
        prefill_v,
        prefill_slot_mapping,
        prefill_query_start_len,
    )
    prefill_out = prefill_kernel(
        q_prefill,
        k_cache,
        v_cache,
        block_table,
        seq_lens_prefill,
        prefill_query_start_len,
    )
    prefill_expected = np.zeros_like(q_prefill)
    for batch_idx in range(active_lanes):
        q_start = int(prefill_query_start_len[batch_idx, 0])
        q_stop = int(prefill_query_start_len[batch_idx + 1, 0])
        prefill_expected[q_start:q_stop] = reference_attention(
            q_prefill[q_start:q_stop],
            k_logical,
            v_logical,
            prefill_seq_len,
            prompt_len,
            batch_idx,
        )
        assert_close(
            f"llmd_prefill batch={batch_idx}",
            prefill_expected[q_start:q_stop],
            prefill_out[q_start:q_stop],
            args,
            prompt_len,
        )
    assert_close("llmd_prefill", prefill_expected, prefill_out, args, int(prefill_query_start_len[-1, 0]))

    previous_output = None
    for step in range(args.decode_steps):
        q_decode = np.zeros(
            (args.batch_size, args.num_kv_heads, heads_per_kv, args.head_dim),
            dtype=ml_dtypes.bfloat16,
        )
        k_decode = np.zeros(
            (args.batch_size, args.num_kv_heads, args.head_dim),
            dtype=ml_dtypes.bfloat16,
        )
        v_decode = np.zeros_like(k_decode)
        decode_slot_mapping = np.full((args.batch_size,), np.iinfo(np.int32).max, dtype=np.int32)
        decode_query_start_len = np.zeros((args.batch_size + 1, 1), dtype=np.int32)
        decode_seq_lens = np.zeros((args.batch_size, 1), dtype=np.int32)
        for batch_idx in range(args.batch_size):
            decode_query_start_len[batch_idx, 0] = min(batch_idx, active_lanes)
        decode_query_start_len[args.batch_size, 0] = active_lanes

        for batch_idx in range(active_lanes):
            seq_len = args.context_len + prompt_len + step + 1
            token_pos = seq_len - 1
            q_tok, k_tok, v_tok = synthetic_qkv_for_position(
                token_pos + batch_idx * args.seq_len,
                args,
                heads_per_kv,
            )
            q_decode[batch_idx : batch_idx + 1, :, :, :] = q_tok
            k_decode[batch_idx : batch_idx + 1, :, :] = k_tok
            v_decode[batch_idx : batch_idx + 1, :, :] = v_tok
            write_logical_token(
                k_logical[batch_idx],
                v_logical[batch_idx],
                token_pos,
                k_tok,
                v_tok,
            )
            decode_slot_mapping[batch_idx] = flattened_cache_slot(
                block_table,
                batch_idx,
                token_pos,
                args.page_size,
            )
            decode_seq_lens[batch_idx, 0] = seq_len

        k_cache, v_cache = update_kernel(
            k_cache,
            v_cache,
            k_decode,
            v_decode,
            decode_slot_mapping,
            decode_query_start_len,
        )
        output = decode_kernel(
            q_decode,
            k_cache,
            v_cache,
            block_table,
            decode_seq_lens,
            decode_query_start_len,
        )
        expected = np.zeros_like(q_decode)
        for batch_idx in range(active_lanes):
            seq_len = int(decode_seq_lens[batch_idx, 0])
            expected[batch_idx : batch_idx + 1] = reference_attention(
                q_decode[batch_idx : batch_idx + 1],
                k_logical,
                v_logical,
                seq_len,
                1,
                batch_idx,
            )
            assert_close(
                f"llmd_decode_step={step} batch={batch_idx}",
                expected[batch_idx : batch_idx + 1],
                output[batch_idx : batch_idx + 1],
                args,
                1,
            )
        assert_close(f"llmd_decode_step={step}", expected, output, args, active_lanes)
        max_delta = print_delta(
            f"llmd_decode_step={step}",
            previous_output,
            output[:active_lanes],
        )
        if max_delta == 0.0:
            raise AssertionError(f"llmd_decode_step={step}: decode output repeated exactly")
        previous_output = output[:active_lanes].copy()


def main():
    args = parse_args()
    spec = resolve_model_spec(args)
    install_model_spec(args, spec)
    validate_args(args)
    configure_runtime_environment(args.mode)
    heads_per_kv = spec.heads_per_kv

    add_neuron_tools_to_path()
    module = load_paged_attention_module()
    if args.run in ("paged-update", "paged-update-batch2", "cache-reuse-update"):
        run_update_contract(args, module)
        return

    prefill_kernel = nki.jit(module.paged_attention_2d)
    decode_kernel = nki.jit(module.paged_attention_decode_2d)
    update_kernel = nki.jit(module.paged_kv_cache_update)
    if args.run == "llmd-loop":
        run_llmd_loop(args, update_kernel, prefill_kernel, decode_kernel, heads_per_kv)
    elif args.run == "history-sensitivity":
        run_history_sensitivity(args, update_kernel, decode_kernel, heads_per_kv)
    elif args.run == "both":
        run_attention_case(args, update_kernel, prefill_kernel, heads_per_kv, "prefill")
        run_attention_case(args, update_kernel, decode_kernel, heads_per_kv, "decode")
    else:
        kernel = decode_kernel if args.run == "decode" else prefill_kernel
        run_attention_case(args, update_kernel, kernel, heads_per_kv, args.run)


if __name__ == "__main__":
    main()
