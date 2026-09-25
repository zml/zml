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
Three single-card runs generated the same 100-token counting sequence at
77.3, 77.3 and 77.4 tok/s; the BF16 regression run reached 61.3 tok/s. Full-layer comparisons against the CPU using
the same quantized weights pass for 128-token prefill and one-token decode at
positions 0 and 127. Those checks verify execution correctness, not retained
language-model quality. A separate [sampled quality comparison](QUALITY.md)
scores 8192 next-token predictions: perplexity rises from 18.6501 to 18.9461
(+1.5875%), with 95.7764% top-token agreement. Broader quality evaluation remains
outstanding.
