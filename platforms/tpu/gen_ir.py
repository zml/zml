import json
import math
import sys
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PJRT_DEVICE"] = "CPU"

import jax
import jax.numpy as jnp
from jax._src.interpreters import mlir
from jax._src.pallas import pallas_call
from jax.experimental import pallas as pallas_mod
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention


def _cpu_lowering_tpu_wrapper(ctx, *in_nodes, **params):
    # Reuse TPU lowering on CPU-only envs used by this IR generator.
    params.pop("backend", None)
    params.pop("which_linear", None)
    if "out_shapes" in params:
        params["out_avals"] = params.pop("out_shapes")
    from jax._src.pallas.mosaic.pallas_call_registration import (
        pallas_call_tpu_lowering_rule,
    )

    return pallas_call_tpu_lowering_rule(ctx, *in_nodes, **params)


mlir.register_lowering(
    pallas_call.pallas_call_p,
    _cpu_lowering_tpu_wrapper,
    platform="cpu",
)


def _extract_backend_config(op) -> str | None :
    backend = op.attributes.get("backend_config")
    if backend is not None:
        return backend.value if hasattr(backend, "value") else str(backend)
    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.operations:
                config = _extract_backend_config(sub_op)
                if config is not None:
                    return config
    return None


def flash_attention_on_tpu(kernel_params: dict) -> str | None:
    batch_size = kernel_params["batch_size"]
    num_heads = kernel_params["num_heads"]
    q_seq_len = kernel_params["q_seq_len"]
    kv_seq_len = kernel_params["kv_seq_len"]
    d_model = kernel_params["d_model"]

    q = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, d_model), jnp.float32)
    k = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.float32)
    v = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.float32)

    block = BlockSizes(
        block_q=min(128, q_seq_len),
        block_k_major=min(128, kv_seq_len),
        block_k=min(128, kv_seq_len),
        block_b=1,
    )
    sm_scale = kernel_params.get("sm_scale", 1.0 / math.sqrt(d_model))

    lowered = jax.jit(
        flash_attention,
        backend="cpu",
        static_argnames=["block_sizes", "causal", "sm_scale"],
    ).lower(q, k, v, block_sizes=block, causal=True, sm_scale=sm_scale)
    return _extract_backend_config(lowered.compiler_ir().operation)


def _fix_paged_attention_acc_specs(fn, **kwargs):
    grid_spec = kwargs.get("grid_spec")
    out_shape = kwargs.get("out_shape")
    out_specs = getattr(grid_spec, "out_specs", None)

    if (
        isinstance(out_specs, tuple)
        and isinstance(out_shape, list)
        and len(out_specs) == 3
        and len(out_shape) == 3
        and out_shape[1].shape[-1] == 1
    ):
        q_block_shape = out_specs[0].block_shape
        if q_block_shape is not None and q_block_shape[-1] not in (None, 1):
            fixed = pallas_mod.BlockSpec((*q_block_shape[:-1], 1), out_specs[0].index_map)
            grid_spec.out_specs = (out_specs[0], fixed, fixed)

    return _ORIG_PALLAS_CALL(fn, **kwargs)


def paged_attention_on_tpu(kernel_params: dict) -> str | None:
    batch_size = kernel_params["batch_size"]
    pages_per_sequence = kernel_params["pages_per_sequence"]

    q = jax.ShapeDtypeStruct((batch_size, kernel_params["num_q_heads"], kernel_params["head_dim"]), jnp.float32)
    k = jax.ShapeDtypeStruct(
        (kernel_params["num_kv_heads"], kernel_params["total_num_pages"], kernel_params["page_size"], kernel_params["head_dim"]),
        jnp.float32,
    )
    v = jax.ShapeDtypeStruct(
        (kernel_params["num_kv_heads"], kernel_params["total_num_pages"], kernel_params["page_size"], kernel_params["head_dim"]),
        jnp.float32,
    )
    lengths = jax.ShapeDtypeStruct((batch_size,), jnp.int32)
    page_indices_flat = jax.ShapeDtypeStruct((batch_size * pages_per_sequence,), jnp.int32)

    def _paged_attention_with_flat_indices(q, k, v, lengths, page_indices_flat, *, pages_per_compute_block):
        indices = page_indices_flat.reshape(batch_size, pages_per_sequence)
        return paged_attention(q, k, v, lengths, indices, pages_per_compute_block=pages_per_compute_block)

    comp_target = jax.jit(_paged_attention_with_flat_indices, backend="cpu", static_argnames=["pages_per_compute_block"])

    pallas_mod.pallas_call = _fix_paged_attention_acc_specs
    try:
        lowered = comp_target.lower(
            q,
            k,
            v,
            lengths,
            page_indices_flat,
            pages_per_compute_block=kernel_params["pages_per_compute_block"],
        )
    finally:
        pallas_mod.pallas_call = _ORIG_PALLAS_CALL

    return _extract_backend_config(lowered.compiler_ir().operation)


def main():
    try:
        request_raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
        if not request_raw.strip():
            raise ValueError("No backend config request provided on stdin")

        request = json.loads(request_raw)
        backend_kind = request["backend_config"]
        params = request["params"]

        if backend_kind not in {"paged", "flash"}:
            raise ValueError(f"Unsupported backend_config: {backend_kind}")
        if not isinstance(params, dict):
            raise ValueError("Invalid request: 'params' must be a JSON object")

        if backend_kind == "paged":
            tpu_code = paged_attention_on_tpu(params)
            if tpu_code is None:
                raise ValueError("Failed to extract TPU backend config from paged attention IR")
        else:
            tpu_code = flash_attention_on_tpu(params)
            if tpu_code is None:
                raise ValueError("Failed to extract TPU backend config from flash attention IR")

        print(tpu_code, end="")
    except BaseException as e:
        print(f"Error : {e}", file=sys.stderr)
        sys.exit(1)



_ORIG_PALLAS_CALL = pallas_mod.pallas_call

if __name__ == "__main__":
    main()
