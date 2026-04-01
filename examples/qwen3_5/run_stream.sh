VIDEO_WIDTH=512
VIDEO_HEIGHT=512
VIDEO_FPS=1

ffmpeg -hide_banner -loglevel error -i data/menu.mp4 -vf "fps=${VIDEO_FPS},scale=${VIDEO_WIDTH}:${VIDEO_HEIGHT}" -pix_fmt rgb24 -f rawvideo - \
| bazel run //examples/qwen3_5:qwen3_5 --@zml//platforms:cuda=true -- \
    --model="/var/models/Qwen/Qwen3.5-4B" \
    --prompt="Describe this video in one sentence." \
    --video-width=${VIDEO_WIDTH} \
    --video-height=${VIDEO_HEIGHT} \
    --video-fps=${VIDEO_FPS} 
