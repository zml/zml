# Voxtral Realtime

Streaming speech recognition with `mistralai/Voxtral-Mini-4B-Realtime-2602`.
This example does not implement Voxtral TTS.

The model directory must contain the original Mistral files:

- `params.json`
- `consolidated.safetensors` (not the HF-renamed `model.safetensors`)
- `tekken.json`

## Apple Silicon / Metal

The accelerator flag is `--@zml//platforms:metal=true` (not `meta`).

```sh
bazel run --config=release --@zml//platforms:metal=true //examples/voxtral:voxtral -- \
  --model=/path/to/Voxtral-Mini-4B-Realtime-2602 \
  --input=/path/to/audio.pcm
```

Input is **raw signed 16-bit little-endian PCM, mono, 16 kHz**. Omit `--input`
to read stdin; the program waits for audio after compiling and loading. For
example, if FFmpeg is installed:

```sh
ffmpeg -loglevel error -i audio.wav -ar 16000 -ac 1 -f s16le - | \
  bazel run --config=release --@zml//platforms:metal=true //examples/voxtral:voxtral -- \
    --model=/path/to/Voxtral-Mini-4B-Realtime-2602
```

Do not pass a WAV/MP3 file directly as `--input`; decode it to raw PCM first.
EOF pads the last audio chunk and flushes the transcription delay. A terminal
shows a waveform; redirected stdout contains only the final transcript.

`--transcription_delay_ms` defaults to 480 and accepts multiples of 80 from
80 to 2400. Use `--config=debug` instead of `release` for a checked build.
Compilation happens at startup and can take several minutes.

Metal uses an f32 DFT audio frontend because its PJRT backend does not support
FFT. Chunked/prefill attention uses masked SDPA; single-token decoder steps
use Metal flash attention. The encoder cache retains extra history for
per-query sliding windows across circular-buffer wraps.

## Tests

```sh
bazel test --config=debug --@zml//platforms:metal=true \
  //examples/voxtral:tokenizer_test //examples/voxtral:kernel_test //examples/voxtral:test
```

The first two targets check Tekken decoding and numerical kernels; `:test`
checks that the optional reference-tensor test binary builds. Running
`:voxtral_test` itself additionally requires a reference safetensors file.
