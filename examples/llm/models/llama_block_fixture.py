"""Usage: python llama_block_fixture.py LOCAL_LLAMA_REPO OUTPUT_DIRECTORY

The fixture uses meta-llama/Llama-3.2-3B-Instruct, revision
0cb88a4f764b7a12671c53f0838cd831a0843b95. Its first safetensors shard (SHA-256
13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55) supplies
the complete first layer and the selected embedding rows. Use torch==2.8.0
(CPU), transformers==4.56.2 and safetensors. Only copy config.json and
model.safetensors from the generated directory to the accelerator machine.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import DynamicCache, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding


TOKENS = (128000, 791, 1401, 1207, 304, 279, 5885, 13)
PREFILL = 4
CAPACITY = len(TOKENS) + 2  # Include unwritten cache slots in every comparison.
LAYER = 'model.layers.0.'


def first_layer(checkpoint):
    index = checkpoint / 'model.safetensors.index.json'
    if index.exists():
        sources = json.loads(index.read_text())['weight_map']
    else:
        with safe_open(checkpoint / 'model.safetensors', framework='pt') as archive:
            sources = dict.fromkeys(archive.keys(), 'model.safetensors')
    weights = {}
    for name in sorted(sources):
        if name.startswith(LAYER):
            with safe_open(checkpoint / sources[name], framework='pt') as archive:
                weights[name] = archive.get_tensor(name)
    embedding = 'model.embed_tokens.weight'
    with safe_open(checkpoint / sources[embedding], framework='pt') as archive:
        rows = archive.get_slice(embedding)
        hidden = torch.cat([rows[token:token + 1] for token in TOKENS])
    if hidden.dtype != torch.bfloat16 or not all(w.dtype == hidden.dtype for w in weights.values()):
        raise ValueError('The fixture requires a BF16 Llama checkpoint')
    return weights, hidden


@torch.inference_mode()
def generate(checkpoint, destination):
    config = LlamaConfig.from_pretrained(checkpoint, local_files_only=True)
    config._attn_implementation = 'eager'
    weights, hidden = first_layer(checkpoint)
    # Initializing on meta avoids allocating or initializing dummy parameters.
    with torch.device('meta'):
        block = LlamaDecoderLayer(config, layer_idx=0)
    block.load_state_dict({k.removeprefix(LAYER): v for k, v in weights.items()}, strict=True, assign=True)
    block.eval()
    rotary = LlamaRotaryEmbedding(config)
    cache = DynamicCache()
    tensors = dict(weights)
    position = 0
    for step, length in enumerate([PREFILL] + [1] * (len(TOKENS) - PREFILL)):
        stop = position + length
        x = hidden[position:stop].unsqueeze(0)
        indices = torch.arange(position, stop)
        # The eager Llama attention expects an additive [batch, head, query, key] mask.
        mask = torch.zeros(length, stop, dtype=x.dtype)
        mask.masked_fill_(torch.arange(stop) > indices[:, None], float('-inf'))
        result = block(x, attention_mask=mask[None, None], position_ids=indices[None],
                       position_embeddings=rotary(x, indices[None]), cache_position=indices,
                       past_key_values=cache, use_cache=True)
        tensors[f'steps.{step}.input'] = x[0].clone()
        tensors[f'steps.{step}.output'] = result[0].contiguous()
        for name, values in zip(('k', 'v'), cache[0], strict=True):
            full = torch.zeros(1, CAPACITY, config.num_key_value_heads, config.head_dim, dtype=x.dtype)
            full[:, :stop] = values.transpose(1, 2)
            tensors[f'steps.{step}.{name}'] = full
        position = stop
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint / 'config.json', destination / 'config.json')
    path = destination / 'model.safetensors'
    save_file(tensors, path, metadata={'model': checkpoint.name, 'token_ids': json.dumps(TOKENS),
                                      'reference': 'transformers 4.56.2 LlamaDecoderLayer, torch 2.8.0 CPU'})
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    print(f'{digest.hexdigest()}  {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    generate(args.checkpoint, args.destination)
