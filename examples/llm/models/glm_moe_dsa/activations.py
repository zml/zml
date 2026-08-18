#!/usr/bin/env python3
"""Generate small GLM-5.2 activation fixtures for the standalone ZML model.

The checkpoint remains untouched.  We construct only layers 0..3, which cover
the dense and sparse MLPs as well as full and shared DSA indexers, and override
the DSA selection width to eight tokens so a short prompt exercises both
prefill and cached decode.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MESSAGE = "Who are you? Give a concise answer and name two things you can help with."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local GLM-5.2 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Output .safetensors path")
    parser.add_argument("--layers", type=int, default=4, help="Leading layers to construct (default: 4)")
    parser.add_argument("--index-topk", type=int, default=8, help="Test-only DSA top-k override (default: 8)")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="User message rendered with the chat template")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    return parser.parse_args()


def first_tensor(values: Iterable[object]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
    return None


def tensor_outputs(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


class Recorder:
    def __init__(self) -> None:
        self.phase = "prefill"
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def save(self, name: str, value: torch.Tensor) -> None:
        self.activations[name] = value.detach().contiguous().clone()

    def hook(self, name: str):
        def record(_module, args, kwargs, output):
            input_tensor = kwargs.get("hidden_states")
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = None
            if input_tensor is None:
                input_tensor = first_tensor(args)
            if input_tensor is None:
                input_tensor = first_tensor(kwargs.values())
            prefix = f"{self.phase}.{name}"
            if input_tensor is not None:
                self.save(f"{prefix}.in", input_tensor)

            outputs = tensor_outputs(output)
            if len(outputs) == 1:
                self.save(f"{prefix}.out", outputs[0])
            else:
                for index, tensor in enumerate(outputs):
                    self.save(f"{prefix}.out.{index}", tensor)

        return record

    def register(self, module: torch.nn.Module, name: str) -> None:
        self.handles.append(module.register_forward_hook(self.hook(name), with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def configure_partial_model(model_path: Path, layer_count: int, index_topk: int):
    config = AutoConfig.from_pretrained(model_path)
    if layer_count < 1 or layer_count > config.num_hidden_layers:
        raise ValueError(f"--layers must be in [1, {config.num_hidden_layers}], got {layer_count}")
    if index_topk < 1:
        raise ValueError(f"--index-topk must be positive, got {index_topk}")

    config.num_hidden_layers = layer_count
    config.index_topk = index_topk
    config.indexer_types = list(config.indexer_types[:layer_count])
    config.mlp_layer_types = list(config.mlp_layer_types[:layer_count])
    config.layer_types = list(config.layer_types[:layer_count])
    config._attn_implementation = "eager"
    return config


def register_components(model: torch.nn.Module, recorder: Recorder) -> None:
    recorder.register(model.model.embed_tokens, "embed_tokens")
    recorder.register(model.model.norm, "norm")
    recorder.register(model.lm_head, "lm_head")

    for index, layer in enumerate(model.model.layers):
        base = f"layers.{index}"
        recorder.register(layer, base)
        recorder.register(layer.input_layernorm, f"{base}.input_layernorm")
        recorder.register(layer.post_attention_layernorm, f"{base}.post_attention_layernorm")

        attention = layer.self_attn
        recorder.register(attention, f"{base}.self_attn")
        for child in (
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
        ):
            recorder.register(getattr(attention, child), f"{base}.self_attn.{child}")
        if attention.indexer is not None:
            recorder.register(attention.indexer, f"{base}.self_attn.indexer")
            for child in ("wq_b", "wk", "k_norm", "weights_proj"):
                recorder.register(getattr(attention.indexer, child), f"{base}.self_attn.indexer.{child}")

        mlp = layer.mlp
        recorder.register(mlp, f"{base}.mlp")
        if hasattr(mlp, "experts"):
            recorder.register(mlp.gate, f"{base}.mlp.gate")
            recorder.register(mlp.experts, f"{base}.mlp.experts")
            recorder.register(mlp.shared_experts, f"{base}.mlp.shared_experts")
            for child in ("gate_proj", "up_proj", "down_proj"):
                recorder.register(
                    getattr(mlp.shared_experts, child),
                    f"{base}.mlp.shared_experts.{child}",
                )
        else:
            for child in ("gate_proj", "up_proj", "down_proj"):
                recorder.register(getattr(mlp, child), f"{base}.mlp.{child}")


def pad_cache(tensor: torch.Tensor, sequence_axis: int, max_sequence_length: int) -> torch.Tensor:
    missing = max_sequence_length - tensor.shape[sequence_axis]
    if missing < 0:
        raise ValueError(f"cache length {tensor.shape[sequence_axis]} exceeds {max_sequence_length}")
    if missing == 0:
        return tensor
    shape = list(tensor.shape)
    shape[sequence_axis] = missing
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=sequence_axis)


def save_cache(
    recorder: Recorder,
    name: str,
    cache,
    layer_count: int,
    max_sequence_length: int,
    index_head_dim: int,
) -> None:
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    indexer_keys: list[torch.Tensor] = []
    for index in range(layer_count):
        layer = cache.layers[index]
        keys.append(pad_cache(layer.keys, 2, max_sequence_length))
        values.append(pad_cache(layer.values, 2, max_sequence_length))
        if layer.is_indexer_initialized:
            index_key = layer.indexer_keys
        else:
            index_key = layer.keys.new_zeros((layer.keys.shape[0], 0, index_head_dim))
        indexer_keys.append(pad_cache(index_key, 1, max_sequence_length))

    stacked_keys = torch.stack(keys)
    stacked_values = torch.stack(values)
    stacked_indexer_keys = torch.stack(indexer_keys)
    recorder.save(f"{name}.cache.k", stacked_keys)
    recorder.save(f"{name}.cache.v", stacked_values)
    recorder.save(f"{name}.cache.indexer_k", stacked_indexer_keys)
    for index in range(layer_count):
        isolated_keys = torch.zeros_like(stacked_keys)
        isolated_values = torch.zeros_like(stacked_values)
        isolated_indexer_keys = torch.zeros_like(stacked_indexer_keys)
        isolated_keys[index] = stacked_keys[index]
        isolated_values[index] = stacked_values[index]
        isolated_indexer_keys[index] = stacked_indexer_keys[index]
        recorder.save(f"{name}.layers.{index}.self_attn.cache.k", isolated_keys)
        recorder.save(f"{name}.layers.{index}.self_attn.cache.v", isolated_values)
        recorder.save(
            f"{name}.layers.{index}.self_attn.cache.indexer_k",
            isolated_indexer_keys,
        )


def render_inputs(tokenizer, message: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": message}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] <= 8:
        raise ValueError(f"The rendered prompt must contain more than eight tokens, got {input_ids.shape[1]}")
    return input_ids


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch does not see a GPU; run this outside a device-restricted sandbox")

    device = torch.device(args.device)
    config = configure_partial_model(args.model, args.layers, args.index_topk)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = render_inputs(tokenizer, args.message, device)
    max_sequence_length = input_ids.shape[1] + 1

    print(
        f"Loading layers 0..{args.layers - 1} on {device} as bfloat16 "
        f"(prompt={input_ids.shape[1]} tokens, index_topk={args.index_topk})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    recorder = Recorder()
    register_components(model, recorder)
    recorder.save("input_ids", input_ids)

    with torch.inference_mode():
        recorder.phase = "prefill"
        prefill = model(input_ids=input_ids, use_cache=True)
        recorder.save("prefill.logits", prefill.logits)
        save_cache(
            recorder,
            "prefill",
            prefill.past_key_values,
            args.layers,
            max_sequence_length,
            config.index_head_dim,
        )

        next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        recorder.save("next_token", next_token)
        recorder.phase = "decode"
        decode = model(input_ids=next_token, past_key_values=prefill.past_key_values, use_cache=True)
        recorder.save("decode.logits", decode.logits)
        save_cache(
            recorder,
            "decode",
            decode.past_key_values,
            args.layers,
            max_sequence_length,
            config.index_head_dim,
        )

    recorder.close()
    torch.cuda.synchronize(device)
    cpu_activations = {name: tensor.cpu() for name, tensor in recorder.activations.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        cpu_activations,
        args.output,
        metadata={
            "model": str(args.model),
            "layers": str(args.layers),
            "index_topk": str(args.index_topk),
            "prompt_tokens": str(input_ids.shape[1]),
            "max_sequence_length": str(max_sequence_length),
            "dtype": "bfloat16",
            "message": args.message,
        },
    )
    print(f"Saved {len(cpu_activations)} tensors to {args.output}")


if __name__ == "__main__":
    main()
