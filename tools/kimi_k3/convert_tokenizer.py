#!/usr/bin/env python3
"""Convert the pinned Kimi K3 tiktoken vocabulary to tokenizer.json offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer
from transformers.integrations.tiktoken import TikTokenConverter


BASE_VOCAB_SIZE = 163_584
RESERVED_SPECIAL_TOKENS = 256
PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


class TokenizerConversionError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def special_token_map(config: dict[str, Any], base_vocab_size: int = BASE_VOCAB_SIZE) -> dict[str, int]:
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise TokenizerConversionError("tokenizer_config.json has no added_tokens_decoder")

    named: dict[int, str] = {}
    for raw_id, entry in decoder.items():
        try:
            token_id = int(raw_id)
        except ValueError as exc:
            raise TokenizerConversionError(f"invalid added-token ID: {raw_id!r}") from exc
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            raise TokenizerConversionError(f"invalid added-token entry for ID {token_id}")
        if not base_vocab_size <= token_id < base_vocab_size + RESERVED_SPECIAL_TOKENS:
            raise TokenizerConversionError(f"added-token ID outside K3 reserved range: {token_id}")
        named[token_id] = entry["content"]

    result: dict[str, int] = {}
    for token_id in range(base_vocab_size, base_vocab_size + RESERVED_SPECIAL_TOKENS):
        content = named.get(token_id, f"<|reserved_token_{token_id}|>")
        if content in result:
            raise TokenizerConversionError(f"duplicate special-token content: {content!r}")
        result[content] = token_id
    return result


def convert(source_dir: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    vocab_path = source_dir / "tiktoken.model"
    config_path = source_dir / "tokenizer_config.json"
    source_code_path = source_dir / "tokenization_kimi.py"
    chat_code_path = source_dir / "encoding_k3.py"
    for path in (vocab_path, config_path, source_code_path, chat_code_path):
        if not path.is_file():
            raise TokenizerConversionError(f"missing pinned tokenizer input: {path}")
    if source_dir in output.parents or source_dir in manifest_path.parents:
        raise TokenizerConversionError("converted artifacts must not modify the source model directory")

    ranks = load_tiktoken_bpe(os.fspath(vocab_path))
    if len(ranks) != BASE_VOCAB_SIZE or set(ranks.values()) != set(range(BASE_VOCAB_SIZE)):
        raise TokenizerConversionError(
            f"unexpected K3 base vocabulary: entries={len(ranks)} expected={BASE_VOCAB_SIZE}"
        )
    config = json.loads(config_path.read_text())
    specials = special_token_map(config)

    converted = TikTokenConverter(
        vocab_file=os.fspath(vocab_path),
        pattern=PATTERN,
        extra_special_tokens=specials,
    ).converted()
    if converted.get_vocab_size(with_added_tokens=True) != BASE_VOCAB_SIZE + RESERVED_SPECIAL_TOKENS:
        raise TokenizerConversionError("converted tokenizer vocabulary size mismatch")
    for content, token_id in specials.items():
        if converted.token_to_id(content) != token_id:
            raise TokenizerConversionError(f"converted special-token ID mismatch: {content!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    converted.save(os.fspath(temporary))
    os.replace(temporary, output)

    # Reload the artifact through the public tokenizer API before locking it.
    reloaded = Tokenizer.from_file(os.fspath(output))
    if reloaded.get_vocab_size(with_added_tokens=True) != BASE_VOCAB_SIZE + RESERVED_SPECIAL_TOKENS:
        raise TokenizerConversionError("reloaded tokenizer vocabulary size mismatch")

    manifest = {
        "schema_version": 1,
        "source_dir": os.fspath(source_dir),
        "source": {
            "tiktoken.model": sha256(vocab_path),
            "tokenizer_config.json": sha256(config_path),
            "tokenization_kimi.py": sha256(source_code_path),
            "encoding_k3.py": sha256(chat_code_path),
        },
        "base_vocab_size": BASE_VOCAB_SIZE,
        "reserved_special_tokens": RESERVED_SPECIAL_TOKENS,
        "vocab_size": BASE_VOCAB_SIZE + RESERVED_SPECIAL_TOKENS,
        "pattern_sha256": hashlib.sha256(PATTERN.encode()).hexdigest(),
        "special_tokens": specials,
        "output": os.fspath(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "network_used": False,
        "source_files_modified": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = convert(args.source, args.output, args.manifest)
    print(json.dumps({
        "converted": manifest["output"],
        "output_sha256": manifest["output_sha256"],
        "vocab_size": manifest["vocab_size"],
        "network_used": manifest["network_used"],
        "source_files_modified": manifest["source_files_modified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
