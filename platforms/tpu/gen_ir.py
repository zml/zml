import contextlib
import json
import math
import os
import signal
import sys

# This IR generator only needs CPU lowering. Forcing CPU here avoids JAX
# probing local TPU runtime state and emitting a fallback warning.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax._src.interpreters import mlir
from jax._src.pallas import pallas_call
from jax.experimental import pallas as pallas_mod
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention

RAGGED_KERNEL_ENV = "ZML_TPU_RAGGED_KERNEL"
_TPU_INFERENCE_RAGGED_PATCHED = False


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


def _abstract_tpu_device_kind() -> str:
    return (
        os.environ.get("ZML_TPU_IR_DEVICE_KIND")
        or os.environ.get("TPU_DEVICE_KIND")
        or "TPU v5 lite"
    )


def _abstract_tpu_num_cores() -> int:
    raw = os.environ.get("ZML_TPU_IR_NUM_CORES", "1")
    try:
        cores = int(raw)
    except ValueError:
        cores = 1
    return max(1, cores)


def _tpu_version_from_kind(device_kind: str) -> int:
    kind = device_kind.strip()
    if "TPU" not in kind:
        return -1
    if kind == "TPU7x":
        return 7
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if not kind.startswith("TPU v"):
        return -1
    version = []
    for ch in kind[len("TPU v"):]:
        if not ch.isdigit():
            break
        version.append(ch)
    if not version:
        return -1
    return int("".join(version))


def _tpu_device_name_from_kind(device_kind: str, num_devices: int | None = None) -> str:
    parts = device_kind.split()
    name = " ".join(parts[:2]) if len(parts) >= 2 else device_kind
    if num_devices is not None:
        name += f"-{num_devices}"
    return name


def _tpu_inference_device_name(num_devices: int | None = None) -> str:
    kind = _abstract_tpu_device_kind()
    if "TPU" not in kind:
        return kind
    suffix = ""
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
        suffix = "e"
    elif kind.endswith("e"):
        kind = kind[:-1]
        suffix = "e"
    elif kind.endswith("p"):
        kind = kind[:-1]
        suffix = "p"
    elif kind == "TPU7x":
        kind = "TPU v7"
    if not kind.startswith("TPU v"):
        return kind
    kind += suffix
    if num_devices is not None:
        kind += f"-{num_devices}"
    return kind


def _ensure_vllm_logger_shim() -> None:
    if "vllm.logger" in sys.modules:
        return
    import logging
    import types

    vllm_mod = sys.modules.get("vllm")
    if vllm_mod is None:
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []
        sys.modules["vllm"] = vllm_mod

    logger_mod = types.ModuleType("vllm.logger")

    class _VllmLogger(logging.Logger):
        def info_once(self, msg, *args, **kwargs):
            self.info(msg, *args, **kwargs)

        def warning_once(self, msg, *args, **kwargs):
            self.warning(msg, *args, **kwargs)

        def warn_once(self, msg, *args, **kwargs):
            self.warning(msg, *args, **kwargs)

        def error_once(self, msg, *args, **kwargs):
            self.error(msg, *args, **kwargs)

        def debug_once(self, msg, *args, **kwargs):
            self.debug(msg, *args, **kwargs)

    def init_logger(name: str) -> _VllmLogger:
        logger = logging.getLogger(name)
        if not isinstance(logger, _VllmLogger):
            logger = _VllmLogger(name)
            logging.Logger.manager.loggerDict[name] = logger
        return logger

    logger_mod._VllmLogger = _VllmLogger
    logger_mod.init_logger = init_logger
    sys.modules["vllm.logger"] = logger_mod
    setattr(vllm_mod, "logger", logger_mod)


def _patch_tpu_inference_ragged_v3_device_probe() -> None:
    global _TPU_INFERENCE_RAGGED_PATCHED
    if _TPU_INFERENCE_RAGGED_PATCHED:
        return

    _ensure_vllm_logger_shim()

    try:
        from tpu_inference.kernels.ragged_paged_attention.v3 import kernel as v3_kernel
        from tpu_inference.kernels.ragged_paged_attention.v3 import util as v3_util
    except Exception:
        return
    try:
        from tpu_inference.kernels.ragged_paged_attention.v3 import tuned_block_sizes as v3_tuned
    except Exception:
        v3_tuned = None
    try:
        from tpu_inference.kernels.ragged_paged_attention.v3 import tuned_block_sizes_hd64 as v3_tuned_hd64
    except Exception:
        v3_tuned_hd64 = None
    try:
        from tpu_inference import utils as tpu_utils
    except Exception:
        tpu_utils = None

    tpu_version_fn = lambda: _tpu_version_from_kind(_abstract_tpu_device_kind())
    device_name_fn = lambda num_devices=None: _tpu_inference_device_name(num_devices)

    v3_util.get_tpu_version = tpu_version_fn
    v3_kernel.get_tpu_version = tpu_version_fn
    if v3_tuned is not None:
        v3_tuned.get_tpu_version = tpu_version_fn
        if hasattr(v3_tuned, "get_device_name"):
            v3_tuned.get_device_name = device_name_fn
    if v3_tuned_hd64 is not None:
        v3_tuned_hd64.get_tpu_version = tpu_version_fn
        if hasattr(v3_tuned_hd64, "get_device_name"):
            v3_tuned_hd64.get_device_name = device_name_fn
    if tpu_utils is not None:
        tpu_utils.get_device_name = device_name_fn

    _TPU_INFERENCE_RAGGED_PATCHED = True


def _effective_backend_kind(requested_backend_kind: str) -> str:
    if requested_backend_kind != "ragged_paged":
        return requested_backend_kind

    raw_selector = os.environ.get(RAGGED_KERNEL_ENV)
    selector = (raw_selector or "").strip().lower()

    if selector in ("", "mosaic", "jax"):
        return "ragged_paged"
    if selector in ("v3", "tpu_inference", "vllm", "v2"):
        return "ragged_paged_v3"

    raise ValueError(
        f"Invalid {RAGGED_KERNEL_ENV}={raw_selector!r}; expected one of: mosaic, v3"
    )


def _patch_ragged_tpu_tuning_device_probe() -> None:
    # The tuned block-size helper probes jax.devices()[0] directly, which is
    # CPU in this process by design. Override it with our abstract TPU kind.
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import tuned_block_sizes

    tuned_block_sizes.get_tpu_version = lambda: _tpu_version_from_kind(_abstract_tpu_device_kind())
    tuned_block_sizes.get_device_name = (
        lambda num_devices=None: _tpu_device_name_from_kind(_abstract_tpu_device_kind(), num_devices)
    )


@contextlib.contextmanager
def _tpu_abstract_mesh_context():
    # Force TPU chip metadata lookup to use an abstract TPU device while we
    # still lower on CPU-only JAX backends.
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1,),
        axis_names=("tpu_core",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind=_abstract_tpu_device_kind(),
            num_cores=_abstract_tpu_num_cores(),
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        yield


_patch_ragged_tpu_tuning_device_probe()


def _extract_backend_configs(op) -> list[str]:
    configs: list[str] = []
    backend = op.attributes.get("backend_config")
    if backend is not None:
        configs.append(backend.value if hasattr(backend, "value") else str(backend))
    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.operations:
                configs.extend(_extract_backend_configs(sub_op))
    return configs


def _extract_backend_configs_with_kernel_name(op) -> list[tuple[str | None, str]]:
    configs: list[tuple[str | None, str]] = []
    backend = op.attributes.get("backend_config")
    if backend is not None:
        kernel_attr = op.attributes.get("kernel_name")
        kernel_name: str | None = None
        if kernel_attr is not None:
            kernel_name = kernel_attr.value if hasattr(kernel_attr, "value") else str(kernel_attr)
        configs.append((kernel_name, backend.value if hasattr(backend, "value") else str(backend)))
    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.operations:
                configs.extend(_extract_backend_configs_with_kernel_name(sub_op))
    return configs


def _extract_backend_config(op) -> str | None:
    configs = _extract_backend_configs(op)
    return configs[0] if configs else None


def flash_attention_on_tpu(kernel_params: dict) -> str | None:
    batch_size = kernel_params["batch_size"]
    num_heads = kernel_params["num_heads"]
    q_seq_len = kernel_params["q_seq_len"]
    kv_seq_len = kernel_params["kv_seq_len"]
    d_model = kernel_params["d_model"]

    q = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, d_model), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.bfloat16)

    block = BlockSizes(
        block_q=min(128, q_seq_len),
        block_k_major=min(128, kv_seq_len),
        block_k=min(128, kv_seq_len),
        block_b=1,
    )
    sm_scale = kernel_params.get("sm_scale", 1.0 / math.sqrt(d_model))

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(
            flash_attention,
            backend="cpu",
            static_argnames=["block_sizes", "causal", "sm_scale"],
        ).lower(q, k, v, block_sizes=block, causal=True, sm_scale=sm_scale)
    configs_with_name = _extract_backend_configs_with_kernel_name(lowered.compiler_ir().operation)
    for kernel_name, cfg in configs_with_name:
        if kernel_name and "RPAm" in kernel_name:
            return cfg
    configs = [cfg for _, cfg in configs_with_name]
    for cfg in configs:
        if "RPAm" in cfg:
            return cfg
    return configs[-1] if configs else None


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
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((batch_size, kernel_params["num_q_heads"], kernel_params["head_dim"]), dtype)
    k = jax.ShapeDtypeStruct(
        (kernel_params["num_kv_heads"], kernel_params["total_num_pages"], kernel_params["page_size"], kernel_params["head_dim"]),
        jnp.bfloat16,
    )
    v = jax.ShapeDtypeStruct(
        (kernel_params["num_kv_heads"], kernel_params["total_num_pages"], kernel_params["page_size"], kernel_params["head_dim"]),
        jnp.bfloat16,
    )
    lengths = jax.ShapeDtypeStruct((batch_size,), jnp.int32)
    page_indices_flat = jax.ShapeDtypeStruct((batch_size * pages_per_sequence,), jnp.int32)

    def _paged_attention_with_flat_indices(q, k, v, lengths, page_indices_flat, *, pages_per_compute_block):
        indices = page_indices_flat.reshape(batch_size, pages_per_sequence)
        return paged_attention(q, k, v, lengths, indices, pages_per_compute_block=pages_per_compute_block)

    comp_target = jax.jit(_paged_attention_with_flat_indices, static_argnames=["pages_per_compute_block"])

    pallas_mod.pallas_call = _fix_paged_attention_acc_specs
    try:
        with _tpu_abstract_mesh_context():
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


def _parse_jnp_dtype(dtype_name: str):
    allowed_dtypes = {
        "jnp.bfloat16": jnp.bfloat16,
        "jnp.float32": jnp.float32,
    }
    try:
        return allowed_dtypes[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def ragged_paged_attention_on_tpu(kernel_params: dict) -> str | None:
    q_dtype = _parse_jnp_dtype(kernel_params["q_dtype"])
    kv_dtype = _parse_jnp_dtype(kernel_params["kv_dtype"])
    q = jax.ShapeDtypeStruct(
        (
            kernel_params["num_q_tokens"],
            kernel_params["num_q_heads"],
            kernel_params["head_dim"],
        ),
        q_dtype,
    )
    kv_pages = jax.ShapeDtypeStruct(
        (
            kernel_params["total_num_pages"],
            kernel_params["page_size"],
            kernel_params["num_kv_heads"] * 2,
            kernel_params["head_dim"],
        ),
        kv_dtype,
    )
    kv_lens = jax.ShapeDtypeStruct((kernel_params["max_num_seqs"],), jnp.int32)
    page_indices = jax.ShapeDtypeStruct(
        (kernel_params["max_num_seqs"], kernel_params["pages_per_seq"]),
        jnp.int32,
    )
    cu_q_lens = jax.ShapeDtypeStruct((kernel_params["max_num_seqs"] + 1,), jnp.int32)
    num_seqs = jax.ShapeDtypeStruct((1,), jnp.int32)

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(
            ragged_paged_attention,
            backend="cpu",
            static_argnames=[
                "sm_scale",
                "sliding_window",
                "soft_cap",
                "mask_value",
                "k_scale",
                "v_scale",
                "num_kv_pages_per_block",
                "num_queries_per_block",
                "vmem_limit_bytes",
            ],
        ).lower(
            q,
            kv_pages,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            sm_scale=kernel_params.get("sm_scale", 1.0),
            sliding_window=kernel_params.get("sliding_window"),
            soft_cap=kernel_params.get("soft_cap"),
            mask_value=kernel_params.get("mask_value"),
            k_scale=kernel_params.get("k_scale"),
            v_scale=kernel_params.get("v_scale"),
            num_kv_pages_per_block=kernel_params.get("num_kv_pages_per_block"),
            num_queries_per_block=kernel_params.get("num_queries_per_block"),
            vmem_limit_bytes=kernel_params.get("vmem_limit_bytes"),
        )
    return _extract_backend_config(lowered.compiler_ir().operation)


def ragged_paged_attention_v3_on_tpu(kernel_params: dict) -> str | None:
    _ensure_vllm_logger_shim()
    try:
        from tpu_inference.kernels.ragged_paged_attention.v3 import kernel as v3_kernel
    except Exception as exc:
        raise ValueError(f"Failed to import tpu-inference v3 ragged kernel: {exc}")

    _patch_tpu_inference_ragged_v3_device_probe()

    q_dtype = _parse_jnp_dtype(kernel_params["q_dtype"])
    kv_dtype = _parse_jnp_dtype(kernel_params["kv_dtype"])

    num_q_tokens = kernel_params["num_q_tokens"]
    num_q_heads = kernel_params["num_q_heads"]
    num_kv_heads = kernel_params["num_kv_heads"]
    head_dim = kernel_params["head_dim"]
    total_num_pages = kernel_params["total_num_pages"]
    page_size = kernel_params["page_size"]
    max_num_seqs = kernel_params["max_num_seqs"]
    pages_per_seq = kernel_params["pages_per_seq"]

    kv_cache_shape = v3_kernel.get_kv_cache_shape(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
    )

    q = jax.ShapeDtypeStruct((num_q_tokens, num_q_heads, head_dim), q_dtype)
    k = jax.ShapeDtypeStruct((num_q_tokens, num_kv_heads, head_dim), kv_dtype)
    v = jax.ShapeDtypeStruct((num_q_tokens, num_kv_heads, head_dim), kv_dtype)
    kv_cache = jax.ShapeDtypeStruct(kv_cache_shape, kv_dtype)
    kv_lens = jax.ShapeDtypeStruct((max_num_seqs,), jnp.int32)
    page_indices = jax.ShapeDtypeStruct((max_num_seqs * pages_per_seq,), jnp.int32)
    cu_q_lens = jax.ShapeDtypeStruct((max_num_seqs + 1,), jnp.int32)
    distribution = jax.ShapeDtypeStruct((3,), jnp.int32)

    def _block_sizes(name: str):
        value = kernel_params.get(name)
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return tuple(int(v) for v in value)
        raise ValueError(f"Invalid {name}: expected 4-element list or tuple")

    out_dtype = kernel_params.get("out_dtype")
    if isinstance(out_dtype, str):
        out_dtype = _parse_jnp_dtype(out_dtype)

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(
            v3_kernel.ragged_paged_attention,
            backend="cpu",
            static_argnames=[
                "use_causal_mask",
                "skip_kv_mask",
                "sm_scale",
                "sliding_window",
                "soft_cap",
                "out_dtype",
                "mask_value",
                "q_scale",
                "k_scale",
                "v_scale",
                "chunk_prefill_size",
                "d_block_sizes",
                "p_block_sizes",
                "m_block_sizes",
                "vmem_limit_bytes",
                "debug_mode",
                "disable_bounds_checks",
                "disable_semaphore_checks",
            ],
        ).lower(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            use_causal_mask=kernel_params.get("use_causal_mask", True),
            skip_kv_mask=kernel_params.get("skip_kv_mask", False),
            sm_scale=kernel_params.get("sm_scale", 1.0),
            sliding_window=kernel_params.get("sliding_window"),
            soft_cap=kernel_params.get("soft_cap"),
            out_dtype=out_dtype,
            mask_value=kernel_params.get("mask_value"),
            q_scale=kernel_params.get("q_scale"),
            k_scale=kernel_params.get("k_scale"),
            v_scale=kernel_params.get("v_scale"),
            chunk_prefill_size=kernel_params.get("chunk_prefill_size"),
            d_block_sizes=_block_sizes("d_block_sizes"),
            p_block_sizes=_block_sizes("p_block_sizes"),
            m_block_sizes=_block_sizes("m_block_sizes"),
            vmem_limit_bytes=kernel_params.get("vmem_limit_bytes"),
            debug_mode=kernel_params.get("debug_mode", False),
            disable_bounds_checks=kernel_params.get("disable_bounds_checks", True),
            disable_semaphore_checks=kernel_params.get("disable_semaphore_checks", True),
        )

    configs_with_names = _extract_backend_configs_with_kernel_name(lowered.compiler_ir().operation)
    if os.environ.get("ZML_TPU_IR_DEBUG"):
        print(f"debug: v3 custom call configs={len(configs_with_names)}", file=sys.stderr)
        for idx, (name, cfg) in enumerate(configs_with_names):
            snippet = cfg[:160].replace("\n", " ")
            print(
                f"debug: cfg[{idx}] kernel_name={name!r} len={len(cfg)} snippet={snippet!r}",
                file=sys.stderr,
            )

    if configs_with_names:
        for name, cfg in configs_with_names:
            if name and "RPAm" in name:
                return cfg
        # If kernel_name isn't present or doesn't mention RPAm, use the last
        # custom call (mixed is emitted last in v3).
        return configs_with_names[-1][1]

    configs = _extract_backend_configs(lowered.compiler_ir().operation)
    if not configs:
        return None
    return configs[-1]


def handle_sigint(signum, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_sigint)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")

            backend_kind = _effective_backend_kind(request["backend_config"])
            params = request["params"]

            if backend_kind not in {"paged", "flash", "ragged_paged", "ragged_paged_v3"}:
                raise ValueError(f"Unsupported backend_config: {backend_kind}")
            if not isinstance(params, dict):
                raise ValueError("Invalid request: 'params' must be a JSON object")

            tpu_code = None

            if backend_kind == "ragged_paged":
                tpu_code = ragged_paged_attention_on_tpu(params)
                if tpu_code is None:
                    raise ValueError("Failed to extract TPU backend config from ragged paged attention IR")
            if backend_kind == "ragged_paged_v3":
                tpu_code = ragged_paged_attention_v3_on_tpu(params)
                if tpu_code is None:
                    raise ValueError("Failed to extract TPU backend config from v3 ragged paged attention IR")
            if backend_kind == "paged":
                # tpu_code = paged_attention_on_tpu(params)
                # if tpu_code is None:
                #     raise ValueError("Failed to extract TPU backend config from paged attention IR")
                raise ValueError(f"Unsupported backend_config: {backend_kind} for now")
            if backend_kind == "flash":
                # tpu_code = flash_attention_on_tpu(params)
                # if tpu_code is None:
                #     raise ValueError("Failed to extract TPU backend config from flash attention IR")
                raise ValueError(f"Unsupported backend_config: {backend_kind} for now")

            if tpu_code is None:
                raise ValueError(f"Unsupported backend_config: {backend_kind}")
            response = {"ok": True, "result": tpu_code}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}

        print(json.dumps(response), flush=True)


_ORIG_PALLAS_CALL = pallas_mod.pallas_call

if __name__ == "__main__":
    main()
