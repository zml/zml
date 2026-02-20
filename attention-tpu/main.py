
import json

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

from jax._src.interpreters import mlir
from jax._src.pallas import pallas_call



# Monkey patch pallas_call to register a lowering rule for CPU
def cpu_lowering_tpu_wrapper(ctx, *in_nodes, **params):
    params.pop('backend', None)
    params.pop('which_linear', None)
    if 'out_shapes' in params:
        params['out_avals'] = params.pop('out_shapes')
    from jax._src.pallas.mosaic.pallas_call_registration import pallas_call_tpu_lowering_rule
    return pallas_call_tpu_lowering_rule(ctx, *in_nodes, **params)
mlir.register_lowering(pallas_call.pallas_call_p, cpu_lowering_tpu_wrapper, platform='cpu')


def extract_backend_config(op) -> str | None:
    if 'backend_config' in op.attributes:
        attr = op.attributes['backend_config']
        return attr.value if hasattr(attr, 'value') else str(attr)

    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.operations:
                result = extract_backend_config(sub_op)
                if result is not None:
                    return result
    return None


def flash_attention_on_tpu(kernel_params:str) -> str:

    _kernel_params: dict = json.loads(kernel_params)
    batch_size: int = _kernel_params["batch_size"]
    num_heads: int = _kernel_params["num_heads"]
    q_seq_len: int = _kernel_params["q_seq_len"]
    kv_seq_len: int = _kernel_params["kv_seq_len"]
    d_model: int = _kernel_params["d_model"]

    q = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, d_model), jnp.float32)
    k = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.float32)
    v = jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, d_model), jnp.float32)

    comp_target = jax.jit(flash_attention)

    mlir_ast: jax._src.stages.Lowered = comp_target.lower(q, k, v)
    # print(mlir_ast.as_text())
    
    backend_config = extract_backend_config(mlir_ast.compiler_ir().operation)
    return backend_config

def paged_attention_on_tpu(kernel_params:str) -> str | None:
    
    _kernel_params: dict = json.loads(kernel_params)
    batch_size: int = _kernel_params["batch_size"]
    num_q_heads: int = _kernel_params["num_q_heads"]
    num_kv_heads: int = _kernel_params["num_kv_heads"]
    head_dim: int = _kernel_params["head_dim"]
    pages_per_sequence: int = _kernel_params["pages_per_sequence"]
    total_num_pages: int = _kernel_params["total_num_pages"]
    page_size: int = _kernel_params["page_size"]
    pages_per_compute_block: int = _kernel_params["pages_per_compute_block"]
    
    q = jnp.zeros((batch_size, num_q_heads, head_dim), jnp.float32)
    k = jnp.zeros((num_kv_heads, total_num_pages, page_size, head_dim), jnp.float32)
    v = jnp.zeros((num_kv_heads, total_num_pages, page_size, head_dim), jnp.float32)
    lengths = jnp.zeros((batch_size,), jnp.int32)
    page_indices = jnp.zeros((batch_size, pages_per_sequence), jnp.int32)

    comp_target = jax.jit(paged_attention, static_argnames=["pages_per_compute_block"])
    mlir_ast: jax._src.stages.Lowered = comp_target.lower(
        q, k, v, lengths, page_indices, 
        pages_per_compute_block=pages_per_compute_block
    )
    # print(mlir_ast.as_text())
    
    backend_config = extract_backend_config(mlir_ast.compiler_ir().operation)
    return backend_config

def main():
    print("Hello from jax!")
    print("================= flash_attention_on_tpu =================")
    flash_attention_params_json: str = '{"batch_size": 1, "num_heads": 8, "q_seq_len": 128, "kv_seq_len": 128, "d_model": 64}'
    print("Flash Attention Backend Config:", flash_attention_on_tpu(flash_attention_params_json))
    print("================= paged_attention_on_tpu =================")
    paged_attention_params_json: str = '{"batch_size": 1, "num_q_heads": 8, "num_kv_heads": 1, "head_dim": 128, "pages_per_sequence": 128, "total_num_pages": 8, "page_size": 16, "pages_per_compute_block": 2}'
    print("Paged Attention Backend Config:", paged_attention_on_tpu(paged_attention_params_json))


if __name__ == "__main__":
    main()
