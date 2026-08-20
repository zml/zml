from __future__ import annotations

import copy

import pytest

import convert_tokenizer as conversion


def config() -> dict:
    return {
        "added_tokens_decoder": {
            "163584": {"content": "[BOS]"},
            "163585": {"content": "[EOS]"},
            "163586": {"content": "<|end_of_msg|>"},
            "163587": {"content": "<|open|>"},
            "163588": {"content": "<|close|>"},
            "163589": {"content": "<|sep|>"},
            "163838": {"content": "[UNK]"},
            "163839": {"content": "[PAD]"},
        }
    }


def test_special_token_map_preserves_named_ids_and_fills_reserved_range() -> None:
    tokens = conversion.special_token_map(config())
    assert len(tokens) == 256
    assert tokens["[BOS]"] == 163584
    assert tokens["<|end_of_msg|>"] == 163586
    assert tokens["<|reserved_token_163600|>"] == 163600
    assert tokens["[PAD]"] == 163839


def test_special_token_map_rejects_out_of_range_id() -> None:
    invalid = copy.deepcopy(config())
    invalid["added_tokens_decoder"]["42"] = {"content": "bad"}
    with pytest.raises(conversion.TokenizerConversionError, match="outside K3 reserved range"):
        conversion.special_token_map(invalid)


def test_special_token_map_rejects_duplicate_content() -> None:
    invalid = copy.deepcopy(config())
    invalid["added_tokens_decoder"]["163590"] = {"content": "[BOS]"}
    with pytest.raises(conversion.TokenizerConversionError, match="duplicate"):
        conversion.special_token_map(invalid)


def test_pattern_is_locked_to_official_k3_contract() -> None:
    assert "[\\p{Han}]+" in conversion.PATTERN
    assert "\\p{N}{1,3}" in conversion.PATTERN
    assert conversion.BASE_VOCAB_SIZE + conversion.RESERVED_SPECIAL_TOKENS == 163840
