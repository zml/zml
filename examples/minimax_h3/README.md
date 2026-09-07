# MiniMax-H3 text-to-video

Prompt in, `video.rgb` + `audio.wav` out. Default 1344×768, 5 s, 24 fps, 30 Euler steps.

```
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3 \
  --prompt='A cinematic wide shot of waves at dusk.' \
  --out=out --seed=42
```

`--width` / `--height` must be multiples of 32 (area ≤ 768×1344). `--duration` is 5–15 seconds; frame count is snapped to a VAE-legal `17n+5`. `--steps` must be at least 2.

```
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1344x768 -r 24 -i out/video.rgb \
  -i out/audio.wav -pix_fmt yuv420p -c:v libx264 -c:a aac out/out.mp4
```

## Layout

| File | Role |
| --- | --- |
| `main.zig` | tokenize → encode → pack → denoise → unpatchify → visual VAE + audio VAE |
| `encoder.zig` | Qwen text tower (50 of 64 layers) |
| `pack.zig` | sequence layout (text/audio/video), σ schedules, noise, unpatchify |
| `config.zig` | canvas flags, geometry, pinned layer sizes (snapshot of repo JSON) |
| `dit.zig` | AdaLN DiT + Euler |
| `vae.zig` | tiled ViT decoder |
| `audio.zig` | audio VAE decoder |
| `ops.zig` | `Run`, load, Linear/RMS constructors |

## Equations

AdaLN block:

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLN(temb)
h ← h + gate_msa * Attn(RMS(h)*(1+scale_msa)+shift_msa)
h ← h + gate_mlp * FF(RMS(h)*(1+scale_mlp)+shift_mlp)
```

Schedule (`pack.Schedule`): `t ∈ [1→0]`, `σ = shift·t / (1+(shift-1)·t)`.
`--steps=N` includes terminal 0, so the DiT runs N−1 times. Video `shift=12`, audio `shift=3`.

Euler (η=0): `x0 = x + σ v`, then `x' = (σ'/σ)x + (1-σ'/σ)x0`.

MM-RoPE: concat t/h/w freqs, then duplicate (`ops.ropeCat3`).

Pack: text → audio → video. Tags: `tag_video=0`, `tag_text=1`, `tag_audio=2`. AdaLN row = `slot * modality_count + tag` (unique-sort over row times, 4 slots).
