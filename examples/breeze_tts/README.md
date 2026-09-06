# Breeze TTS 2 on Metal

Two native Zig/ZML binaries implement [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2):

- `text_to_speech`: synthesize text with an optional natural-language voice instruction.
- `voice_cloning`: synthesize new text using reference audio and its exact transcript.

Both explicitly initialize the Metal accelerator. Python is only used by the optional reference comparison script.

## Model

Download the complete checkpoint, including `audio_tokenizer/`:

```sh
hf download BreezeBlue/Breeze-TTS-2 --local-dir /path/to/breeze-tts-2
```

The implementation targets checkpoint revision `799624c0b4a1daa8db6d28bbd9850043c0270734`. It loads the original sharded BF16 model and F32 codec safetensors directly, without conversion. Architecture fields are checked before loading weights; incompatible architectures fail explicitly.

## Build and run

```sh
bazel build //examples/breeze_tts:text_to_speech //examples/breeze_tts:voice_cloning \
  --@zml//platforms:metal=true

bazel run //examples/breeze_tts:text_to_speech --@zml//platforms:metal=true -- \
  --model=/path/to/breeze-tts-2 \
  --text="Hello, this is Breeze speaking." \
  --instruction="A warm, clear voice with a relaxed delivery." \
  --output=/tmp/speech.wav

bazel run //examples/breeze_tts:voice_cloning --@zml//platforms:metal=true -- \
  --model=/path/to/breeze-tts-2 \
  --ref-audio=/path/to/reference.wav \
  --ref-text="The exact words spoken in the reference recording." \
  --text="Welcome back. It is good to hear from you." \
  --output=/tmp/cloned.wav
```

The executables are also available under `bazel-bin/examples/breeze_tts/` with their runfiles. Choose a new output filename: existing WAV files are not overwritten.

Reference input must be a nonempty **24 kHz WAV**, at most 30 seconds, with PCM16, PCM24, PCM32, or float32 samples. Both classic WAV and WAVE_FORMAT_EXTENSIBLE headers are supported. Multiple channels are averaged to mono. Convert other audio formats or sample rates first, for example:

```sh
ffmpeg -i reference.mp3 -ar 24000 -ac 1 -c:a pcm_s16le reference.wav
```

Output is mono 24 kHz PCM16 WAV. Synthesis stops at the backbone EOS token or `--max-frames` (750 by default, 60 seconds). Reaching the limit emits a truncation warning. Text, reference audio, and the generation budget must fit within 2048 backbone positions.

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--instruction` | `Speak clearly and naturally.` | Voice and delivery instruction |
| `--max-frames` | `750` | Maximum generated audio frames, 1–1500; 12.5 frames/second |
| `--temperature` | `0.9` | Sampling temperature for both decoders; 0 selects greedy decoding |
| `--top-k` | `50` | Number of sampling candidates, 1–2048 |
| `--repetition-penalty` | `1.1` | Penalty for repeated first-codebook tokens |
| `--seed` | `42` | Sampling seed |
| `--dump-dir` | unset | Existing directory for numerical validation artifacts |

This version performs single-request, offline synthesis with guidance scale 1. The waveform decoder processes up to 300 frames at once with 25 frames of left context, matching Qwen's offline decoder. It does not expose the reference implementation's streaming server or classifier-free guidance options. First-run graph compilation can take several minutes.

## Validation

```sh
bazel test //examples/breeze_tts:test --@zml//platforms:metal=true
```

Tests cover WAV parsing, malformed input, codec-token suppression, causal convolution padding, transposed convolution alignment, text attention windows, and architecture compatibility.

To compare against upstream, install the reference repository's Python requirements in a separate environment, generate a WAV with `--dump-dir`, and run:

```sh
mkdir -p /tmp/breeze-trace
bazel run //examples/breeze_tts:text_to_speech --@zml//platforms:metal=true -- \
  --model=/path/to/breeze-tts-2 --text="Hello, this is Breeze speaking." \
  --temperature=0 --max-frames=40 --output=/tmp/checked-speech.wav \
  --dump-dir=/tmp/breeze-trace

python examples/breeze_tts/reference.py \
  --model=/path/to/breeze-tts-2 --reference-repo=/path/to/breeze-tts \
  --dump-dir=/tmp/breeze-trace --text="Hello, this is Breeze speaking."
```

Use the same `--instruction`, `--ref-audio`, and `--ref-text` in both commands when applicable. The comparison checks text embeddings, backbone prefill and cached decode, depth-decoder logits, reference codec tokens, and waveform decoding. It uses the native run's generated tokens to isolate decoder correctness from sampling differences. Small floating-point differences can switch nearly tied residual codebook choices, so the reference encoder check compares continuous latents and verifies each selected centroid against its nearest-neighbor distance.

## Sources

The model equations follow [BreezeBlue/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) at `43e2ea1595297c4059477e2e4a300653761c759b`, Transformers 4.57.3's Mimi implementation, and Qwen3-TTS 0.1.1's 12 Hz audio tokenizer. Source code is Apache-2.0; the model weights retain the separate license published with the checkpoint.

Validated on an Apple M5 Max with Metal: both binaries generated complete WAVs and passed the CPU reference comparisons. The checked text-to-speech and cloning waveforms had relative L2 errors of 0.16% and 0.20%, respectively; the reference audio encoder latent error was 0.089%. These checks validate numerical agreement, not identical stochastic samples across backends.
