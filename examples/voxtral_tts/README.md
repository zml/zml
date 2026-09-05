# Voxtral text to speech

Native Zig/ZML inference for `mistralai/Voxtral-4B-TTS-2603`. This is a
**separate binary** from `//examples/voxtral:voxtral`, which transcribes speech.
No Python, PyTorch, vLLM, or conversion script is needed at runtime.

## Run on Apple Silicon

From the ZML repository:

```sh
bazel run --config=release --@zml//platforms:metal=true \
  //examples/voxtral_tts:voxtral_tts -- \
  --model=/Users/dmehala/models/mistralai/Voxtral-4B-TTS-2603 \
  --text="Hello! This is Voxtral speaking through ZML." \
  --voice=casual_female \
  --output=/tmp/voxtral-speech.wav
```

Use `--config=debug` for runtime assertions. The accelerator flag is `metal`,
not `meta`. The model directory must contain `params.json`,
`consolidated.safetensors`, `tekken.json`, and `voice_embedding/<voice>.pt`.
Other Voxtral checkpoints are rejected.

The output is a mono, 24 kHz, signed PCM16 WAV. Existing output files are not
overwritten. First-run compilation can take a few minutes; weights stay in
BF16. A machine with enough memory for the roughly 8 GB checkpoint plus
compilation, caches, and activations is required.

## Options and scope

- `--voice`: a preset shipped in `voice_embedding`, default `casual_female`.
  Examples: `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male`,
  `fr_female`, `fr_male`, and the checkpoint's other language presets.
- `--max-frames`: generation limit, default 375 (30 seconds), maximum 1500.
  Each frame is 80 ms. Generation stops at the model's end-audio token; hitting
  the limit emits a truncation warning.
- `--steps`: acoustic Euler steps, default 7, range 1–50.
- `--guidance`: classifier-free guidance, default 1.2, range 0–5.
- `--seed`: acoustic noise seed, default 0. Backend floating-point differences
  can still change generated codes.
- `--dump-dir`: existing directory for development dumps; files must not exist.

This implementation uses greedy semantic-code selection, classifier-free
flow matching for 36 acoustic codebooks, and the checkpoint's waveform
decoder. It synthesizes one text prompt at a time and writes the WAV after
generation. Streaming playback, batching, and custom reference-audio voice
cloning are not implemented. Preset `.pt` tensors are read as stored ZIP
members with validated metadata; Python pickle code is never executed.

## Tests and reference comparison

```sh
bazel test --config=debug --@zml//platforms:metal=true \
  //examples/voxtral_tts:test --test_output=errors
```

For an optional independent numerical check, generate a short sample with
`--dump-dir=/absolute/path/to/empty-directory`. Then run `reference.py` in a
separate environment containing `torch`, `numpy`, `safetensors`,
`mistral-common`, and `einops`:

```sh
python examples/voxtral_tts/reference.py \
  --model=/absolute/path/to/Voxtral-4B-TTS-2603 \
  --reference-root=/absolute/path/to/vllm-omni \
  --dump-dir=/absolute/path/to/dumps \
  --text="Hello, this is a test."
```

Use the same text, voice, steps, and guidance as the native run. The helper
checks the exact prompt IDs, prefill hidden state, first acoustic frame, and
the decoded waveform. It loads only pure PyTorch model definitions from a
local reference checkout, without vLLM's serving/CUDA dependencies.

Validated on Apple M5 Max / Metal in release mode with
`"Hello, this is a test."`, `casual_female`, and seed 0:

- All eight regression tests pass, including nonconstant device convolution
  and quantizer inputs.
- All 226 prompt IDs match `mistral-common` exactly.
- Prefill hidden-state cosine similarity: 0.99994.
- First semantic code and all 36 acoustic codes match exactly.
- Waveform cosine similarity against the PyTorch reference: 0.99987.
- 39 frames / 3.12 seconds of audio, reaching end-audio normally. Autoregressive
  generation took approximately 1.54 seconds, excluding startup compilation,
  weight loading, and waveform decoding.

## Reference and model license

The architecture follows the Apache-2.0
[vLLM-Omni Voxtral TTS implementation](https://github.com/vllm-project/vllm-omni/tree/44d3ae100afa8411770b8b4d9442318bc5e897b3/vllm_omni/model_executor/models/voxtral_tts),
copyright contributors to the vLLM project. This example is a Zig/ZML port,
not a wrapper around that runtime. Reference checkout used during development:
`44d3ae100afa8411770b8b4d9442318bc5e897b3`.

The [model weights](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
have their own CC BY-NC 4.0 license; this implementation does not change it.
