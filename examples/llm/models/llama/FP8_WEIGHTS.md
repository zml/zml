# Optional FP8 weight storage on RNGD

The default Llama checkpoint and execution remain BF16. For a separate experiment,
`quantize_fp8.py` converts the seven matrix projections in each transformer layer
to E4M3FN. It retains embeddings, the vocabulary head, normalization, activations,
and KV caches in BF16. This changes the model's weights; it is not lossless
packing or a promise of equivalent model quality.

For each output channel, the converter chooses the smallest normal F32 power of
two `scale` such that `max(abs(weight)) / scale <= 448`, then rounds
`weight / scale` to E4M3FN. Zero channels use scale one. The checkpoint contains
`weight_scale` vectors and a `zml_weight_storage` marker in `config.json`.
The Llama projection converts the weights to the activation dtype inside its
existing program, performs the dot, and applies the scale to its output. The
power-of-two scale does not introduce additional BF16 rounding outside the
dtype's range boundaries. There is no per-token host dequantization.

The output is a ZML-specific checkpoint: an unmodified Hugging Face loader will
not apply these scales. The converter requires PyTorch and NumPy, reads the
source without modifying it, refuses an existing destination, and publishes the
new directory only after all shards and metadata are written. An interrupted
conversion retains a `.partial` directory for diagnosis. The report records
per-matrix relative L2 and maximum absolute weight errors.

```sh
python3 examples/llm/models/llama/quantize_fp8.py \
  /var/models/meta-llama/Llama-3.1-8B-Instruct \
  /path/to/llama-fp8-weights

bazel run //examples/llm --@zml//platforms:furiosa=true --config=debug -- \
  --model=/path/to/llama-fp8-weights --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='Count from 1 to 100, separated by commas. Output only the numbers.'
```

Source the local RNGD SDK environment and set `XLA_FURIOSA_PJRT_LIBRARY` as for
the BF16 example. Keep `XLA_FURIOSA_ALLOW_STAGED_FALLBACK` unset. Compilation
continues to produce one TCL/EDF program per XLA compilation.

The converter's checks run with:

```sh
python3 -m unittest discover -s examples/llm/models/llama -p quantize_fp8_test.py
```

Actual Llama 3.1 8B conversion reduces loaded tensors from 14.96 GiB to 8.46 GiB.
The 224 converted matrices have relative L2 weight errors of 2.665–2.675%.
With four decode layers per native program and native preloading in the PJRT
plugin, two single-card runs generated the same 100-token counting sequence at
90.8 and 91.8 tok/s. Eight-layer BF16 blocks reached 67.7 and 68.1 tok/s.
The earlier single-layer implementation measured 77.3–77.4 tok/s for FP8 and
61.3 tok/s for BF16. Full-layer comparisons against the CPU using
the same quantized weights pass for 128-token prefill and one-token decode at
positions 0 and 127. Those checks verify execution correctness, not retained
language-model quality. A separate [sampled quality comparison](QUALITY.md)
scores 8192 next-token predictions: perplexity rises from 18.6501 to 18.9461
(+1.5875%), with 95.7764% top-token agreement. Broader quality evaluation remains
outstanding.

The grouped comparison also passes for FP8 layers 28–31 at cache position 127,
with nonzero cache history and exact checks on untouched entries:

```sh
bazel run //examples/llm:llama_tests --@zml//platforms:furiosa=true --config=debug -- \
  --model=/path/to/llama-fp8-weights --compare-cpu --transformer-only \
  --layers=4 --first-layer=28 --seqlen=1 --cache-seqlen=128 --token-offset=127
```

Decode uses eight-layer blocks for BF16 when the layer count is divisible by
eight, otherwise four-layer blocks when divisible by four. FP8 uses four-layer
blocks when divisible by four. Other counts, prefill, and other platforms retain
single-layer programs. Each block is one XLA compilation and one native program.
