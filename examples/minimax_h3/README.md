# MiniMax-H3 text-to-video

Prompt in, silent `video.rgb` out. 1344×768, 5 s, 24 fps, 30 Euler steps.

```
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3 \
  --prompt='A cinematic wide shot of waves at dusk.' \
  --out=out --seed=42
```

```
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1344x768 -r 24 -i out/video.rgb \
  -an -pix_fmt yuv420p -c:v libx264 out/out.mp4
```

## Layout

| File | Role |
| --- | --- |
| `main.zig` | tokenize → encode → pack → denoise → unpatchify → VAE → rgb |
| `encoder.zig` | Qwen text tower (50 of 64 layers) |
| `pack.zig` | sequence layout, σ schedule, patchify / unpatchify |
| `dit.zig` | AdaLN DiT + Euler |
| `vae.zig` | tiled ViT decoder |
| `ops.zig` | `Run`, `Checkpoint`, load / compile helpers |
| `config.zig` | 768P geometry and head-TP mesh |

## Equations

AdaLN block (`MiniMaxH3TransformerBlock` → `dit.BlockCore`):

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLN(temb)
h ← h + gate_msa * Attn(RMS(h)*(1+scale_msa)+shift_msa)
h ← h + gate_mlp * FF(RMS(h)*(1+scale_mlp)+shift_mlp)
```

Schedule (`pack.Schedule`): `t ∈ [1→0]`, `σ = shift·t / (1+(shift-1)·t)`.
`--steps=N` includes terminal 0, so the DiT runs N−1 times. `config.video_shift = 12`.

Euler (η=0): `x0 = x + σ v`, then `x' = (σ'/σ)x + (1-σ'/σ)x0`.

MM-RoPE: concat t/h/w freqs, then duplicate (`ops.ropeCat3`).

Pack tags: `tag_video=0`, `tag_text=1`. AdaLN row = `timestep * modality_count + tag`.

## Python ↔ Zig

Official names from `transformer_minimax_h3.py` and `autoencoder_kl_minimax_h3.py`.

| Python | Zig |
| --- | --- |
| `MiniMaxH3Transformer3DModel` | `dit.Dit` |
| `MiniMaxH3TransformerBlock` | `dit.DitBlock` (`BlockCore` + `AdaLn`) |
| `MiniMaxH3AdaLayerNormModulation` | `dit.AdaLn` |
| `MiniMaxH3AdaLayerNormOut` | `dit.FinalLayer` |
| `MiniMaxH3Attention` | `dit.Attention` |
| `MiniMaxH3TokenRefiner` / `MiniMaxH3TokenRefinerBlock` | `dit.TextPrep` / `TokenRefinerBlock` |
| `MiniMaxH3RotaryPosEmbed` | `ops.ropeCat3` |
| text encoder (Qwen) | `encoder.Encoder` |
| `AutoencoderKLMiniMaxH3` | `vae.Vae` |
| `MiniMaxH3VideoViTDecoder3d` | `vae.EmbedModel` + `VitBlock` + `FinishModel` |
| `MiniMaxH3VideoTransformerBlock` | `vae.VitBlock` |
| `MiniMaxH3VideoAttention` | `vae.VitAttn` |
