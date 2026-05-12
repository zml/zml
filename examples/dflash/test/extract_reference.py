#!/usr/bin/env python3
"""Extract DFlash reference activations to a safetensors fixture.

The fixture mirrors the Zig path in zml/examples/dflash/main.zig:
  1. run the target LLaMA prefill over a fixed block,
  2. concatenate the selected target hidden states,
  3. build the DFlash noise block from the target greedy token and mask token,
  4. run DFlash while saving every draft layer input/output.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
from collections import OrderedDict

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


DEFAULT_PROMPT = (
    "Paris is the city of lights, fun, and love where everyone can enjoy the "
    "amazing culture, food, and art."
)


def load_dflash_module(repo_root: pathlib.Path):
    candidates = []
    if source := os.environ.get("DFLASH_SOURCE"):
        source_path = pathlib.Path(source).expanduser()
        candidates.extend([
            source_path / "dflash" / "model.py",
            source_path / "model.py",
        ])
    candidates.extend([
        repo_root / "third_party" / "dflash" / "dflash" / "model.py",
        repo_root / "dflash" / "dflash" / "model.py",
        repo_root / "dflash" / "model.py",
    ])

    model_py = next((path for path in candidates if path.exists()), None)
    if model_py is None:
        searched = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"unable to find dflash/model.py; searched:\n  {searched}")

    spec = importlib.util.spec_from_file_location("dflash_reference_model", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_eos_token(model) -> int:
    eos = model.config.eos_token_id
    if isinstance(eos, (list, tuple)):
        return int(eos[0])
    return int(eos)


def fixed_block_tokens(tokenizer, target_model, prompt: str, block_size: int, device) -> torch.Tensor:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = torch.full((block_size,), first_eos_token(target_model), dtype=torch.uint32)
    token_ids[: min(block_size, len(encoded))] = torch.tensor(encoded[:block_size], dtype=torch.uint32)
    return token_ids.to(device=device, dtype=torch.long).unsqueeze(0)


def squeeze_batch(name: str, value: torch.Tensor, keep_batch: bool) -> torch.Tensor:
    value = value.detach().cpu()
    if not keep_batch and value.ndim > 0 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.is_floating_point():
        value = value.to(torch.float32)
    return value.contiguous()


def save_tensor(tensors: OrderedDict[str, torch.Tensor], key: str, value: torch.Tensor, keep_batch: bool) -> None:
    tensors[key] = squeeze_batch(key, value, keep_batch)


def save_heads(tensors: OrderedDict[str, torch.Tensor], key: str, value: torch.Tensor, keep_batch: bool) -> None:
    # Python attention tensors are often [batch, heads, seq, head_dim].
    # Zig keeps sequence/query first, so save [batch, seq, heads, head_dim].
    if value.ndim == 4:
        value = value.transpose(1, 2)
    save_tensor(tensors, key, value, keep_batch)


def with_tf32(allow_tf32: bool, fn):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def layer_debug_forward(module, layer, hidden, target_hidden, position_embeddings, position_ids):
    prefix_values = OrderedDict()

    residual = hidden
    input_norm = layer.input_layernorm(residual)
    prefix_values["input_layernorm.out"] = input_norm

    attn = layer.self_attn
    bsz, q_len = input_norm.shape[:-1]
    ctx_len = target_hidden.shape[1]
    head_dim = attn.head_dim

    q_proj = attn.q_proj(input_norm).view(bsz, q_len, -1, head_dim)
    k_ctx = attn.k_proj(target_hidden)
    k_noise = attn.k_proj(input_norm)
    v_ctx = attn.v_proj(target_hidden)
    v_noise = attn.v_proj(input_norm)
    v_ctx_replay_bf16 = torch.nn.functional.linear(
        target_hidden.to(torch.bfloat16),
        attn.v_proj.weight.to(torch.bfloat16),
        attn.v_proj.bias.to(torch.bfloat16) if attn.v_proj.bias is not None else None,
    )
    v_noise_replay_bf16 = torch.nn.functional.linear(
        input_norm.to(torch.bfloat16),
        attn.v_proj.weight.to(torch.bfloat16),
        attn.v_proj.bias.to(torch.bfloat16) if attn.v_proj.bias is not None else None,
    )
    v_ctx_replay_f32 = torch.nn.functional.linear(
        target_hidden.float(),
        attn.v_proj.weight.float(),
        attn.v_proj.bias.float() if attn.v_proj.bias is not None else None,
    )
    v_noise_replay_f32 = torch.nn.functional.linear(
        input_norm.float(),
        attn.v_proj.weight.float(),
        attn.v_proj.bias.float() if attn.v_proj.bias is not None else None,
    )
    v_ctx_replay_f32_tf32 = with_tf32(
        True,
        lambda: torch.nn.functional.linear(
            target_hidden.float(),
            attn.v_proj.weight.float(),
            attn.v_proj.bias.float() if attn.v_proj.bias is not None else None,
        ),
    )
    v_noise_replay_f32_tf32 = with_tf32(
        True,
        lambda: torch.nn.functional.linear(
            input_norm.float(),
            attn.v_proj.weight.float(),
            attn.v_proj.bias.float() if attn.v_proj.bias is not None else None,
        ),
    )
    k_proj = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)
    v_proj = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, head_dim)
    v_ctx = v_ctx.view(bsz, ctx_len, -1, head_dim)
    v_noise = v_noise.view(bsz, q_len, -1, head_dim)
    v_ctx_replay_bf16 = v_ctx_replay_bf16.view(bsz, ctx_len, -1, head_dim)
    v_noise_replay_bf16 = v_noise_replay_bf16.view(bsz, q_len, -1, head_dim)
    v_ctx_replay_f32 = v_ctx_replay_f32.view(bsz, ctx_len, -1, head_dim)
    v_noise_replay_f32 = v_noise_replay_f32.view(bsz, q_len, -1, head_dim)
    v_ctx_replay_f32_tf32 = v_ctx_replay_f32_tf32.view(bsz, ctx_len, -1, head_dim)
    v_noise_replay_f32_tf32 = v_noise_replay_f32_tf32.view(bsz, q_len, -1, head_dim)

    prefix_values["self_attn.q_proj.out"] = q_proj
    prefix_values["self_attn.k_proj.out"] = k_proj
    prefix_values["self_attn.v_proj.out"] = v_proj
    prefix_values["self_attn.v_proj.ctx"] = v_ctx
    prefix_values["self_attn.v_proj.noise"] = v_noise
    prefix_values["self_attn.v_proj.ctx_replay_bf16"] = v_ctx_replay_bf16
    prefix_values["self_attn.v_proj.noise_replay_bf16"] = v_noise_replay_bf16
    prefix_values["self_attn.v_proj.ctx_replay_f32"] = v_ctx_replay_f32
    prefix_values["self_attn.v_proj.noise_replay_f32"] = v_noise_replay_f32
    prefix_values["self_attn.v_proj.ctx_replay_f32_tf32"] = v_ctx_replay_f32_tf32
    prefix_values["self_attn.v_proj.noise_replay_f32_tf32"] = v_noise_replay_f32_tf32

    q_norm = attn.q_norm(q_proj)
    k_norm = attn.k_norm(k_proj)
    prefix_values["self_attn.q_norm.out"] = q_norm
    prefix_values["self_attn.k_norm.out"] = k_norm

    q = q_norm.transpose(1, 2)
    k = k_norm.transpose(1, 2)
    v = v_proj.transpose(1, 2)
    q_rope, k_rope = module.apply_rotary_pos_emb(q, k, *position_embeddings)
    prefix_values["self_attn.q_rope.out"] = q_rope
    prefix_values["self_attn.k_rope.out"] = k_rope
    prefix_values["self_attn.v.out"] = v

    num_kv_heads = layer.self_attn.config.num_key_value_heads
    num_query_heads = layer.self_attn.config.num_attention_heads
    num_key_value_groups = num_query_heads // num_kv_heads
    q_grouped = q_rope.transpose(1, 2).reshape(bsz, q_len, num_kv_heads, num_key_value_groups, head_dim)
    k_by_seq = k_rope.transpose(1, 2)
    v_by_seq = v.transpose(1, 2)
    k_scaled = k_by_seq * attn.scaling
    sdpa_logits_grouped = torch.einsum("bshgd,bthd->bshgt", q_grouped, k_scaled)
    sdpa_weights_grouped = torch.softmax(sdpa_logits_grouped, dim=-1, dtype=torch.float32).to(q_grouped.dtype)
    sdpa_grouped = torch.einsum("bshgt,bthd->bshgd", sdpa_weights_grouped, v_by_seq)
    sdpa_grouped_merged = sdpa_grouped.reshape(bsz, q_len, num_query_heads, head_dim)
    sdpa_weights_grouped_f32 = torch.softmax(sdpa_logits_grouped, dim=-1, dtype=torch.float32)
    sdpa_grouped_f32 = torch.einsum("bshgt,bthd->bshgd", sdpa_weights_grouped_f32, v_by_seq.float())
    sdpa_grouped_f32_merged = sdpa_grouped_f32.reshape(bsz, q_len, num_query_heads, head_dim)
    sdpa_grouped_replay_bf16 = torch.einsum(
        "bshgt,bthd->bshgd",
        sdpa_weights_grouped.to(torch.bfloat16),
        v_by_seq.to(torch.bfloat16),
    )
    sdpa_grouped_replay_f32 = torch.einsum(
        "bshgt,bthd->bshgd",
        sdpa_weights_grouped_f32.float(),
        v_by_seq.float(),
    )
    sdpa_grouped_replay_f32_tf32 = with_tf32(
        True,
        lambda: torch.einsum(
            "bshgt,bthd->bshgd",
            sdpa_weights_grouped_f32.float(),
            v_by_seq.float(),
        ),
    )

    prefix_values["self_attn.sdpa.q_grouped"] = q_grouped
    prefix_values["self_attn.sdpa.k_scaled"] = k_scaled
    prefix_values["self_attn.sdpa.v_grouped"] = v_by_seq
    prefix_values["self_attn.sdpa.logits_grouped"] = sdpa_logits_grouped
    prefix_values["self_attn.sdpa.weights_grouped"] = sdpa_weights_grouped
    prefix_values["self_attn.sdpa.grouped_out"] = sdpa_grouped
    prefix_values["self_attn.sdpa.grouped_merged_out"] = sdpa_grouped_merged
    prefix_values["self_attn.sdpa.weights_grouped_f32"] = sdpa_weights_grouped_f32
    prefix_values["self_attn.sdpa.grouped_out_f32"] = sdpa_grouped_f32
    prefix_values["self_attn.sdpa.grouped_merged_out_f32"] = sdpa_grouped_f32_merged
    prefix_values["self_attn.sdpa.grouped_out_replay_bf16"] = sdpa_grouped_replay_bf16
    prefix_values["self_attn.sdpa.grouped_out_replay_f32"] = sdpa_grouped_replay_f32
    prefix_values["self_attn.sdpa.grouped_out_replay_f32_tf32"] = sdpa_grouped_replay_f32_tf32

    attn_pre_o, _ = module.eager_attention_forward(
        attn,
        q_rope,
        k_rope,
        v,
        None,
        dropout=0.0,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
    )
    prefix_values["self_attn.sdpa.out"] = attn_pre_o
    attn_merged = attn_pre_o.reshape(bsz, q_len, -1)
    prefix_values["self_attn.sdpa_merged.out"] = attn_merged

    attn_out = attn.o_proj(attn_merged)
    prefix_values["self_attn.o_proj.out"] = attn_out
    post_attn = residual + attn_out
    prefix_values["post_attn_residual.out"] = post_attn

    post_attn_norm = layer.post_attention_layernorm(post_attn)
    prefix_values["post_attention_layernorm.out"] = post_attn_norm
    mlp_out = layer.mlp(post_attn_norm)
    prefix_values["mlp.out"] = mlp_out
    layer_out = post_attn + mlp_out
    prefix_values["out"] = layer_out
    return layer_out, prefix_values


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="/Users/tristan/models/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dflash-model", default="/Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat")
    parser.add_argument("--out", required=True, help="Output .safetensors path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--keep-batch", action="store_true", help="Keep the leading batch dimension in saved tensors")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    dflash_module = load_dflash_module(repo_root)
    DFlashDraftModel = dflash_module.DFlashDraftModel

    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    dflash = DFlashDraftModel.from_pretrained(
        args.dflash_model,
        torch_dtype=dtype,
    ).to(device)
    target.eval()
    dflash.eval()

    block_size = int(dflash.block_size)
    input_ids = fixed_block_tokens(tokenizer, target, args.prompt, block_size, device)
    target_position_ids = torch.arange(block_size, device=device, dtype=torch.long).unsqueeze(0)

    try:
        target_out = target(
            input_ids,
            position_ids=target_position_ids,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    except TypeError:
        target_out = target(
            input_ids,
            position_ids=target_position_ids,
            use_cache=False,
            output_hidden_states=True,
        )

    selected = [target_out.hidden_states[layer_id + 1] for layer_id in dflash.target_layer_ids]
    target_hidden = torch.cat(selected, dim=-1)
    target_token = torch.argmax(target_out.logits[:, -1:, :], dim=-1)
    mask_token_id = dflash.mask_token_id
    if mask_token_id is None:
        raise RuntimeError("DFlash config has no mask_token_id")
    noise_tokens = torch.full_like(input_ids, int(mask_token_id))
    noise_tokens[:, :1] = target_token
    noise_embedding = target.model.embed_tokens(noise_tokens)

    projected_target_hidden = dflash.hidden_norm(dflash.fc(target_hidden))
    position_ids = torch.arange(block_size * 2, device=device, dtype=torch.long).unsqueeze(0)
    position_embeddings = dflash.rotary_emb(noise_embedding, position_ids)

    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    tensors["input_ids"] = squeeze_batch("input_ids", input_ids.to(torch.uint32), args.keep_batch)
    tensors["target_token"] = squeeze_batch("target_token", target_token.to(torch.uint32), args.keep_batch)
    tensors["noise_tokens"] = squeeze_batch("noise_tokens", noise_tokens.to(torch.uint32), args.keep_batch)
    tensors["position_ids"] = squeeze_batch("position_ids", position_ids.to(torch.uint32), args.keep_batch)
    tensors["target_hidden"] = squeeze_batch("target_hidden", target_hidden, args.keep_batch)
    tensors["target_hidden_projected"] = squeeze_batch("target_hidden_projected", projected_target_hidden, args.keep_batch)
    tensors["noise_embedding"] = squeeze_batch("noise_embedding", noise_embedding, args.keep_batch)

    hidden = noise_embedding
    for i, layer in enumerate(dflash.layers):
        tensors[f"layers.{i}.in"] = squeeze_batch(f"layers.{i}.in", hidden, args.keep_batch)
        hidden, debug_values = layer_debug_forward(
            dflash_module,
            layer,
            hidden,
            projected_target_hidden,
            position_embeddings,
            position_ids,
        )
        for suffix, value in debug_values.items():
            key = f"layers.{i}.{suffix}"
            if value.ndim == 4 and value.shape[1] in (
                layer.self_attn.config.num_attention_heads,
                layer.self_attn.config.num_key_value_heads,
            ):
                save_heads(tensors, key, value, args.keep_batch)
            else:
                save_tensor(tensors, key, value, args.keep_batch)

    final = dflash.norm(hidden)
    tensors["final_out"] = squeeze_batch("final_out", final, args.keep_batch)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), out, metadata={
        "target_model": args.target_model,
        "dflash_model": args.dflash_model,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "block_size": str(block_size),
        "target_layer_ids": ",".join(str(i) for i in dflash.target_layer_ids),
    })
    print(f"wrote {out} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
