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
