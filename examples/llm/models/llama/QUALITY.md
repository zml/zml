# Comparing Llama checkpoints on Furiosa

`//examples/llm:llama_logits` runs teacher-forced Llama inference on RNGD with
vanilla attention and writes BF16 logits for every input position. It uses the
same embedding, transformer layers, weights and normalization as generation,
but exports the complete vocabulary distribution instead of sampling a token.
Each embedding, layer and head compilation produces one native TCL/EDF program.

`quality.py` prepares uniformly spaced windows from a pinned WikiText-2 raw test
revision, then scores next-token perplexity, KL divergence and top-token agreement
between two checkpoints. This is a sampled, short-context comparison, **not** a
standard full-corpus WikiText perplexity benchmark or a broad capability test.
Positions reset for every window; all positions are scored, including the first
prediction with little context. The corpus starts with BOS and uses no chat
template. Later windows can start mid-document.

Install NumPy, PyArrow and Hugging Face `tokenizers` in a Python environment, then:

```sh
python3 examples/llm/models/llama/quality.py prepare \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --output=/tmp/llama-quality --seqlen=128 --windows=64

bazel build //examples/llm:llama_logits \
  --@zml//platforms:furiosa=true --config=debug

# Set the SDK environment and XLA_FURIOSA_PJRT_LIBRARY as for generation.
unset XLA_FURIOSA_ALLOW_STAGED_FALLBACK
bazel-bin/examples/llm/llama_logits \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --tokens=/tmp/llama-quality/tokens.u32 --seqlen=128 \
  --output=/tmp/llama-quality/bf16.logits
bazel-bin/examples/llm/llama_logits \
  --model=/path/to/llama-fp8-weights \
  --tokens=/tmp/llama-quality/tokens.u32 --seqlen=128 \
  --output=/tmp/llama-quality/fp8.logits
python3 examples/llm/models/llama/quality.py compare \
  --tokens=/tmp/llama-quality/tokens.u32 \
  --reference=/tmp/llama-quality/bf16.logits \
  --candidate=/tmp/llama-quality/fp8.logits \
  --output=/tmp/llama-quality/comparison.json
```

The preparation directory records dataset, tokenizer and token-file hashes plus
window offsets. Each little-endian U32 window has `seqlen + 1` tokens: the first
`seqlen` are inputs and the last `seqlen` are targets. The exporter refuses an
existing output. Its 52-byte header contains `ZMLLGTS1`, three little-endian U32
dimensions (sequence length, vocabulary, windows), and the token-file SHA256.
BF16 logits follow in `[window, sequence, vocabulary]` order. The comparison
checks dimensions, hashes, file lengths and finite values, then accumulates
stable log probabilities in float64. At 64 windows of length 128 and vocabulary
128256, each file occupies about 2 GiB. These transfers are outside generation
throughput measurements.

## Measured weight-only FP8 comparison

On Llama 3.1 8B Instruct, 64 windows of 128 predictions (8192 total) gave:

| Metric | Result |
|---|---:|
| Original BF16 perplexity | 18.6501 |
| FP8 weight-storage perplexity | 18.9461 |
| Relative perplexity increase | 1.5875% |
| Mean NLL increase | 0.0157504 nats |
| Mean KL(BF16 \|\| FP8) | 0.00619413 nats |
| Top-token agreement | 95.7764% |

Both runs used RNGD, BF16 activations, all 32 layers and the same token file
(`0209d7d5b95968762fe307390c653013de5a7d00cbf7ee10c6eb3546c50bd225`).
This quantization changes predictions; BF16 remains the default.
Artifacts are in `~/.local/state/xla-rngd/internals/quality-wikitext/`.

The hardware evaluator was also checked with three windows: two identical
windows and a third with tokens 64 onward replaced. Repeated-window logits were
bit-identical, the third window's first 64 logits were bit-identical to the
original, and later logits changed. This checks cache reuse and causal masking.
The scoring tests cover manual next-token likelihood, normalization, changed
predictions, nonfinite logits, mismatched hashes and truncated files:

```sh
python3 -B -m unittest discover -s examples/llm/models/llama -p quality_test.py
```
