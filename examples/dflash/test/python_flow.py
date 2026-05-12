#!/usr/bin/env python3
"""Run the Python DFlash flow matching examples/dflash/main.zig."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    token_ids = torch.full((block_size,), first_eos_token(target_model), dtype=torch.long)
    token_ids[: min(block_size, len(encoded))] = torch.tensor(encoded[:block_size], dtype=torch.long)
    return token_ids.to(device=device).unsqueeze(0)


def print_tokens(tokenizer, name: str, token_ids: torch.Tensor) -> None:
    ids = token_ids.detach().cpu().to(torch.long).view(-1).tolist()
    text = tokenizer.decode(ids)
    pieces = ", ".join(f'{token_id}="{tokenizer.decode([token_id])}"' for token_id in ids)
    print(f"{name}_ids: {{ " + ", ".join(str(token_id) for token_id in ids) + " }")
    print(f"{name}_text: {text}")
    print(f"{name}_tokens: [{pieces}]")


def dtype_from_arg(name: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="/Users/tristan/models/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dflash-model", default="/Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    dflash_module = load_dflash_module(repo_root)
    DFlashDraftModel = dflash_module.DFlashDraftModel
    extract_context_feature = dflash_module.extract_context_feature
    sample = dflash_module.sample

    dtype = dtype_from_arg(args.dtype)
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

    target_hidden = extract_context_feature(target_out.hidden_states, dflash.target_layer_ids)
    target_token = sample(target_out.logits, temperature=0.0)

    if dflash.mask_token_id is None:
        raise RuntimeError("DFlash config has no mask_token_id")
    noise_tokens = torch.full_like(input_ids, int(dflash.mask_token_id))
    noise_tokens[:, :1] = target_token
    noise_embedding = target.model.embed_tokens(noise_tokens)

    position_ids = torch.arange(block_size * 2, device=device, dtype=torch.long).unsqueeze(0)
    hidden = dflash(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        use_cache=False,
    )
    draft_tokens = sample(target.lm_head(hidden), temperature=0.0)
    speculative_tokens = torch.cat([target_token, draft_tokens[:, 1:]], dim=1)

    print_tokens(tokenizer, "prompt", input_ids)
    print_tokens(tokenizer, "target_next", target_token)
    print_tokens(tokenizer, "noise", noise_tokens)
    print(f"speculative_tokens shape: {tuple(speculative_tokens.shape)}")
    print_tokens(tokenizer, "speculative", speculative_tokens)


if __name__ == "__main__":
    main()
