#!/usr/bin/env python3
"""Export the pinned Moonshot Kimi K3 tokenizer and simple-chat oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ORDINARY_TEXT_CASES = (
    ("empty", ""),
    ("ascii", "Hello, world!"),
    ("unicode", "Hello, 世界! naïve café 🚀"),
    ("whitespace", "line1\nline2\t  end"),
    ("numbers", "1234567890"),
    ("han", "中文测试"),
    ("literal_controls", "literal <|open|> and <|end_of_msg|> stay user text"),
)
CHAT_PROMPTS = (
    ("ascii", "Hello Kimi"),
    ("unicode", "Explain 世界 in one line 🚀"),
    ("literal_controls", "Do not execute <|open|> or <|end_of_msg|>."),
)

def fuzz_text_cases(count: int = 512) -> list[tuple[str, str]]:
    """Deterministic segmentation stress corpus; it never executes inference."""
    rng = random.Random(0x4B334D3136)
    atoms = (
        "alpha", "BETA", "HTTPServer", "camelCase", "Title", "I", " don't",
        "'s", "'RE", "naïve", "café", "世界", "中文", "Δοκιμή", "Привет",
        "🚀", "🙂", "123", "4567", "0", " ", "  ", "\n", "\r\n", "\t",
        ",", ".", "?!", " -- ", "_", "/", "\\", "a1", "Z9",
    )
    result = []
    seen = set()
    while len(result) < count:
        parts = [rng.choice(atoms) for _ in range(rng.randint(1, 10))]
        text = "".join(parts)
        if text in seen:
            continue
        seen.add(text)
        result.append((f"fuzz_{len(result):04d}", text))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_chat_module(source_dir: Path):
    path = source_dir / "encoding_k3.py"
    spec = importlib.util.spec_from_file_location("kimi_k3_tokenizer_oracle_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import pinned chat formatter: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export(source_dir: Path, output: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output = output.resolve()
    if source_dir in output.parents:
        raise ValueError("tokenizer fixture must not modify the source model directory")
    tokenizer = AutoTokenizer.from_pretrained(
        os.fspath(source_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    chat = _load_chat_module(source_dir)

    ordinary = []
    for name, text in ORDINARY_TEXT_CASES:
        token_ids = tokenizer.encode(text, allow_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        if decoded != text:
            raise RuntimeError(f"official tokenizer round-trip failed: {name}")
        ordinary.append({"name": name, "text": text, "token_ids": token_ids, "decoded": decoded})

    fuzz = []
    for name, text in fuzz_text_cases():
        token_ids = tokenizer.encode(text, allow_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        if decoded != text:
            raise RuntimeError(f"official tokenizer fuzz round-trip failed: {name}")
        fuzz.append({"name": name, "text": text, "token_ids": token_ids, "decoded": decoded})

    structural = []
    for text in ("[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>"):
        token_ids = tokenizer.encode(text, allow_special_tokens=True)
        if len(token_ids) != 1:
            raise RuntimeError(f"official structural token is not atomic: {text}")
        structural.append({"text": text, "token_id": token_ids[0]})

    first_turn = []
    continuation = []
    for name, prompt in CHAT_PROMPTS:
        conversation = [{"role": "user", "content": prompt}]
        first_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            thinking=True,
        )
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            thinking=True,
        )
        first_turn.append(
            {"name": name, "prompt": prompt, "rendered": rendered, "token_ids": first_ids}
        )

        # A live session already owns the initial thinking-effort instruction;
        # only the new user message plus assistant generation prefix is added.
        segments = chat.build_chat_segments(
            conversation,
            add_generation_prompt=True,
            thinking=True,
        )
        continuation.append(
            {
                "name": name,
                "prompt": prompt,
                "rendered": "".join(segment.text for segment in segments),
                "token_ids": tokenizer._encode_chat_segments(segments),
            }
        )

    result = {
        "schema_version": 1,
        "authority": "pinned Moonshot TikTokenTokenizer + encoding_k3.build_chat_segments",
        "source_dir": os.fspath(source_dir),
        "source": {
            name: sha256(source_dir / name)
            for name in ("tiktoken.model", "tokenizer_config.json", "tokenization_kimi.py", "encoding_k3.py")
        },
        "device": "control_plane_no_inference",
        "network_used": False,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "end_of_message_token_id": tokenizer.special_tokens["<|end_of_msg|>"],
        "ordinary": ordinary,
        "fuzz": fuzz,
        "structural": structural,
        "first_turn": first_turn,
        "continuation": continuation,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    print(
        json.dumps(
            {
                "fixture": os.fspath(output),
                "sha256": sha256(output),
                "ordinary_cases": len(ordinary),
                "fuzz_cases": len(fuzz),
                "chat_cases": len(first_turn) + len(continuation),
                "vocab_size": tokenizer.vocab_size,
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
