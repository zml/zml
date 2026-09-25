#!/usr/bin/env python3
"""Prepare fixed WikiText-2 windows and compare Furiosa Llama logits.

Logit files come from //examples/llm:llama_logits. This is a sampled, short-context
comparison, not the standard full-corpus, long-context WikiText perplexity.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

import numpy as np

DATASET_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "00aa25585682d4957f9e86edc73f59be7419af99/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
MAGIC = b"ZMLLGTS1"


def prepare(model, output, seqlen, windows):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    if seqlen < 1 or windows < 1:
        raise ValueError("Positive sequence length and window count required")
    output.mkdir(parents=True, exist_ok=False)
    parquet = output / "wikitext-2-test.parquet"
    with urllib.request.urlopen(DATASET_URL) as response, parquet.open("xb") as destination:
        destination.write(response.read())
    text = "\n\n".join(pq.read_table(parquet)["text"].to_pylist())
    tokenizer_path = model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = tokenizer.token_to_id("<|begin_of_text|>")
    if bos is None:
        raise ValueError("Llama BOS token is missing")
    tokens = np.asarray([bos] + tokenizer.encode(text, add_special_tokens=False).ids, dtype="<u4")
    if len(tokens) < seqlen + 1 or windows > len(tokens) - seqlen:
        raise ValueError("Corpus is too small for the requested windows")
    starts = np.linspace(0, len(tokens) - seqlen - 1, windows, dtype=np.int64)
    samples = np.stack([tokens[start:start + seqlen + 1] for start in starts])
    data = samples.tobytes()
    (output / "tokens.u32").write_bytes(data)
    metadata = {
        "dataset_url": DATASET_URL,
        "dataset_license": "CC-BY-SA-3.0 / GFDL (see Salesforce/wikitext dataset card)",
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokens_sha256": hashlib.sha256(data).hexdigest(),
        "corpus_tokens": len(tokens), "seqlen": seqlen, "windows": windows,
        "window_starts": starts.tolist(), "scored_tokens": windows * seqlen,
        "protocol": "Uniformly spaced windows; positions reset at each window; score all next-token predictions. No chat template.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def read_header(stream, token_hash):
    header = stream.read(52)
    if len(header) != 52 or header[:8] != MAGIC or header[20:] != token_hash:
        raise ValueError("Invalid logit header or mismatched token-file hash")
    seqlen, vocab, windows = struct.unpack("<III", header[8:20])
    if not seqlen or not vocab or not windows:
        raise ValueError("Empty logit dimensions")
    expected = 52 + seqlen * vocab * windows * 2
    current = stream.tell()
    stream.seek(0, 2)
    if stream.tell() != expected:
        raise ValueError("Logit file has an unexpected length")
    stream.seek(current)
    return seqlen, vocab, windows


def read_logits(stream, seqlen, vocab):
    bits = np.frombuffer(stream.read(seqlen * vocab * 2), dtype="<u2")
    logits = (bits.astype(np.uint32) << 16).view(np.float32).reshape(seqlen, vocab)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    return logits.astype(np.float64)


def log_probabilities(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def compare(tokens_path, reference_path, candidate_path):
    token_bytes = tokens_path.read_bytes()
    token_hash = hashlib.sha256(token_bytes).digest()
    reference_nll = 0.0
    candidate_nll = 0.0
    kl = 0.0
    agreement = 0
    per_window = []
    with reference_path.open("rb") as reference, candidate_path.open("rb") as candidate:
        shape = read_header(reference, token_hash)
        if read_header(candidate, token_hash) != shape:
            raise ValueError("Reference and candidate shapes differ")
        seqlen, vocab, windows = shape
        tokens = np.frombuffer(token_bytes, dtype="<u4")
        if tokens.size != windows * (seqlen + 1) or (tokens >= vocab).any():
            raise ValueError("Token file does not match logit dimensions")
        targets = tokens.reshape(windows, seqlen + 1)[:, 1:]
        rows = np.arange(seqlen)
        for window in range(windows):
            ref = read_logits(reference, seqlen, vocab)
            test = read_logits(candidate, seqlen, vocab)
            agreement += int((ref.argmax(axis=1) == test.argmax(axis=1)).sum())
            p = log_probabilities(ref)
            q = log_probabilities(test)
            ref_nll = float(-p[rows, targets[window]].sum())
            test_nll = float(-q[rows, targets[window]].sum())
            reference_nll += ref_nll
            candidate_nll += test_nll
            kl += float((np.exp(p) * (p - q)).sum())
            per_window.append({"reference_mean_nll": ref_nll / seqlen, "candidate_mean_nll": test_nll / seqlen})
    count = seqlen * windows
    delta = (candidate_nll - reference_nll) / count
    return {
        "reference": str(reference_path), "candidate": str(candidate_path),
        "tokens_sha256": token_hash.hex(), "scored_tokens": count,
        "seqlen": seqlen, "windows": windows,
        "reference_perplexity": math.exp(reference_nll / count),
        "candidate_perplexity": math.exp(candidate_nll / count),
        "perplexity_relative_change": math.expm1(delta),
        "mean_nll_increase": delta, "mean_kl_reference_to_candidate": kl / count,
        "top1_agreement": agreement / count, "per_window": per_window,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seqlen", type=int, default=128)
    prep.add_argument("--windows", type=int, default=64)
    cmp = commands.add_parser("compare")
    cmp.add_argument("--tokens", type=Path, required=True)
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model, args.output, args.seqlen, args.windows)
    else:
        report = compare(args.tokens, args.reference, args.candidate)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "per_window"}, indent=2))
