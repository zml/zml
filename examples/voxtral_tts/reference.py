"""Optional numerical check against the public vLLM-Omni Voxtral implementation.

Requires torch, numpy, safetensors, mistral-common and einops in a separate
Python environment. This is a development test, not a runtime dependency.
Only pure model definitions from the reference checkout are loaded; vLLM/CUDA
serving infrastructure is not needed. See README.md for invocation.
"""

import argparse
import ast
import dataclasses
import enum
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union, get_args, get_origin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from safetensors import safe_open


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Reference definitions changed: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def compare(name, actual, expected, minimum_cosine):
    a, b = actual.float().cpu().flatten(), expected.float().cpu().flatten()
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
    cosine = F.cosine_similarity(a[None], b[None]).item()
    rmse = (a - b).square().mean().sqrt().item()
    print(f"{name}: cosine={cosine:.6f}, RMSE={rmse:.6f}, max_error={(a-b).abs().max().item():.6f}", flush=True)
    assert cosine >= minimum_cosine, name


def language_prefill(weights, ids, voice, device):
    """Independent eager implementation of the checkpoint's Mistral text backbone."""
    def weight(name):
        return weights.get_tensor(name).to(device)

    def norm(x, name):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return y.to(x.dtype) * weight(name)

    x = F.embedding(torch.tensor(ids, device=device), weight("mm_audio_embeddings.tok_embeddings.weight"))
    x[2:2 + voice.shape[0]] = voice.to(device)
    angles = torch.arange(len(ids), device=device, dtype=torch.float32)[:, None] * (
        1000000 ** (-torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128))[None]
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]

    def rope(x):
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2).to(x.dtype)

    for i in range(26):
        prefix = f"layers.{i}."
        n = norm(x, prefix + "attention_norm.weight")
        q, k, v = [F.linear(n, weight(prefix + "attention.w" + s + ".weight")).reshape(len(ids), h, 128)
                   for s, h in [("q", 32), ("k", 8), ("v", 8)]]
        q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.repeat_interleave(4, 1).transpose(0, 1)[None],
                                          v.repeat_interleave(4, 1).transpose(0, 1)[None], is_causal=True)
        x = x + F.linear(a[0].transpose(0, 1).reshape(len(ids), 4096), weight(prefix + "attention.wo.weight"))
        n = norm(x, prefix + "ffn_norm.weight")
        gate = F.silu(F.linear(n, weight(prefix + "feed_forward.w1.weight")))
        x = x + F.linear(gate * F.linear(n, weight(prefix + "feed_forward.w3.weight")), weight(prefix + "feed_forward.w2.weight"))
    return norm(x[-1:], "norm.weight")


def main():
    import copy
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="casual_female")
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--guidance", type=float, default=1.2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    params = json.loads((args.model / "params.json").read_text())
    device = torch.device(args.device)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    root = args.reference_root / "vllm_omni/model_executor/models/voxtral_tts"
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=math, Enum=enum.Enum,
              dataclass=dataclasses.dataclass, fields=dataclasses.fields,
              is_dataclass=dataclasses.is_dataclass, Union=Union, Any=Any,
              get_args=get_args, get_origin=get_origin, rms_norm=nn.RMSNorm,
              logger=logging.getLogger("reference"), deepcopy=copy.deepcopy,
              rearrange=rearrange, weight_norm=nn.utils.parametrizations.weight_norm,
              VllmConfig=Any, HAS_FLASH_ATTN=False, CODEC_NORM_EPS=1e-2)
    definitions(root / "voxtral_tts_audio_generation.py", [
        "AudioSpecialTokens", "AcousticTransformerArgs", "MultimodalAudioModelArgs",
        "_repeat_interleave", "repeat_kv", "from_nested_dict", "FeedForward",
        "BidirectionalAttention", "AcousticTransformerBlock", "TimeEmbedding",
        "FlowMatchingAudioTransformer",
    ], ns)
    definitions(root / "voxtral_tts_audio_tokenizer.py", [
        "AudioTokenizerArgs", "SemanticCodebook", "AcousticCodebook", "MistralAudioCodebook",
        "prepare_for_attention", "pad1d", "CausalConv1d", "CausalConvTranspose1d",
        "MultiVocabEmbeddings", "Attention", "TransformerBlock", "Transformer", "VoxtralTTSAudioTokenizer",
    ], ns)
    tok = MistralTokenizer.from_file(args.model / "tekken.json")
    expected_ids = tok.encode_speech_request(SpeechRequest(input=args.text, voice=args.voice)).tokens
    actual_ids = np.fromfile(args.dump_dir / "prompt.u32", dtype="<u4").tolist()
    assert actual_ids == expected_ids, (actual_ids, expected_ids)
    print(f"Prompt: {len(actual_ids)} token IDs exactly match mistral-common", flush=True)

    with torch.inference_mode(), safe_open(args.model / "consolidated.safetensors", framework="pt", device="cpu") as weights:
        hidden = torch.from_numpy(np.fromfile(args.dump_dir / "hidden.bf16", dtype="<u2").copy()).view(torch.bfloat16).reshape(1, 3072).to(device)
        voice = torch.load(args.model / "voice_embedding" / (args.voice + ".pt"), weights_only=True, map_location="cpu")
        expected_hidden = language_prefill(weights, actual_ids, voice, device)
        compare("Language-model hidden state", hidden, expected_hidden, minimum_cosine=0.995)
        codes = torch.from_numpy(np.fromfile(args.dump_dir / "codes.u32", dtype="<u4").astype(np.int64)).reshape(-1, 37).to(device)
        audio_args = copy.deepcopy(params["multimodal"]["audio_model_args"])
        audio_args["acoustic_transformer_args"]["n_decoding_steps"] = args.steps
        acoustic = ns["FlowMatchingAudioTransformer"](audio_args).to(device)
        loaded = acoustic.load_state_dict({k.removeprefix("acoustic_transformer."): weights.get_tensor(k).to(device)
                                          for k in weights.keys() if k.startswith("acoustic_transformer.")}, strict=False)
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        assert set(loaded.missing_keys) <= {"time_embedding.inv_freq"}, loaded.missing_keys
        noise = torch.from_numpy(np.fromfile(args.dump_dir / "noise.f32", dtype="<f4").copy()).reshape(1, 36).to(device, torch.bfloat16)
        llm = acoustic.llm_projection(torch.cat([hidden, torch.zeros_like(hidden)]))
        schedule = torch.linspace(0, 1, args.steps + 1, dtype=torch.float32, device=device).to(torch.bfloat16)
        alpha = torch.tensor(args.guidance, dtype=torch.bfloat16, device=device)
        x = noise
        for i in range(args.steps):
            t_proj = acoustic.time_projection(acoustic.time_embedding(schedule[i].reshape(1, 1)).to(torch.bfloat16)).repeat(2, 1)
            vel = acoustic._predict_velocity(torch.cat([x, x]), llm, t_proj)
            velocity = alpha * vel[:1] + (1 - alpha) * vel[1:]
            x = x + velocity * (schedule[i + 1] - schedule[i])
        quantized = (((x.clamp(-1, 1) + 1) / 2) * 20).round().long() + 2
        semantic = acoustic.semantic_codebook_output(hidden).float()[:, 1:8194].argmax(-1) + 1
        print("Semantic code:", codes[0, 0].item(), "reference:", semantic.item(), flush=True)
        assert codes[0, 0] == semantic[0]
        difference = (codes[0, 1:] - quantized[0]).abs()
        print(f"Acoustic codes: {(difference == 0).float().mean().item():.1%} exact, max bin delta={difference.max().item()}", flush=True)
        assert difference.max() <= 2
        del acoustic

        config = SimpleNamespace(audio_config={"codec_args": params["multimodal"]["audio_tokenizer_args"],
                                                "audio_model_args": audio_args},
                                 text_config=SimpleNamespace(hidden_size=3072))
        codec = ns["VoxtralTTSAudioTokenizer"](vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config))).to(device)
        loaded = codec.load_state_dict({k.removeprefix("audio_tokenizer."): weights.get_tensor(k).to(device)
                                       for k in weights.keys() if k.startswith("audio_tokenizer.")}, strict=False)
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        assert all(k.startswith(("input_proj.", "encoder_blocks.", "audio_token_embedding."))
                   for k in loaded.missing_keys), loaded.missing_keys
        if (args.dump_dir / "codec-0.bf16").exists():
            emb = codec.quantizer.decode((codes - 2).T[None], torch.bfloat16).transpose(1, 2)
            stages = [emb]
            for i, block in enumerate(codec.decoder_blocks):
                emb = block(emb.transpose(1, 2)).transpose(1, 2) if i % 2 == 0 else block(emb)
                if emb.dim() == 2:
                    emb = emb[None]
                stages.append(emb)
            stages.append(codec.output_proj(emb.transpose(1, 2)).transpose(1, 2))
            for i, expected_stage in enumerate(stages):
                actual_stage = torch.from_numpy(np.fromfile(args.dump_dir / f"codec-{i}.bf16", dtype="<u2").copy()).view(torch.bfloat16)
                compare(f"Codec stage {i}", actual_stage, expected_stage, minimum_cosine=-1)
        expected = codec.decode((codes - 2).T[None], dtype=torch.bfloat16).flatten()
        actual = torch.from_numpy(np.fromfile(args.dump_dir / "audio.f32", dtype="<f4").copy())
        compare("Waveform", actual, expected, minimum_cosine=0.95)


if __name__ == "__main__":
    main()
