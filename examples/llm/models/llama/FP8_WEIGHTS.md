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

bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --@zml//platforms:furiosa=true --@zml//platforms:cpu=false --config=debug //examples/llm -- \
  --model=/path/to/llama-fp8-weights --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='Count from 1 to 100, separated by commas. Output only the numbers.'
```

Prepare the SDK and repository override using the
[Furiosa guide](../../../../platforms/furiosa/README.md).
Leave `XLA_FURIOSA_PJRT_LIBRARY` unset to use the packaged plugin.

The converter's checks run with:

```sh
python3 -m unittest discover -s examples/llm/models/llama -p quantize_fp8_test.py
```

FP8 model execution and quality must be revalidated with the current typed backend.
Historical measurements used earlier execution strategies and are not current
support or throughput claims. See [QUALITY.md](QUALITY.md) for the reusable
comparison format and [WHOLE_FORWARD.md](WHOLE_FORWARD.md) for current architecture.
