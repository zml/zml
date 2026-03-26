from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from safetensors.torch import save_file
from transformers import AutoProcessor

# Keep frame dimensions compatible with patch_size * spatial_merge_size.
IMAGE_DIM_DIV_FACTOR = 32

OVERRIDE_RESIZE_DIM = 128

def nearest_multiple(value: int, multiple: int) -> int:
    floor = (value // multiple) * multiple
    ceil = ((value + multiple - 1) // multiple) * multiple
    floor = max(multiple, floor)
    ceil = max(multiple, ceil)
    if value - floor < ceil - value:
        return floor
    return ceil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video input safetensors using the official HF processor pipeline")
    parser.add_argument("--model", type=Path, required=True, help="Local model path (contains processor/tokenizer config)")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out", type=Path, required=True, help="Output safetensors file path")
    parser.add_argument("--max-frames", type=int, default=512, help="Maximum number of frames to sample")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS used by the HF video processor sampler")
    return parser.parse_args()


def load_video_frames(video_path: Path, max_frames: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max_frames

    step = max(1, total_frames // max_frames)
    frames = []
    frame_idx = 0

    while True:
        print(f"Decoding video frames... {frame_idx}/{total_frames}", end="\r")
        ok, frame_bgr = cap.read()
        if not ok:
            if frame_idx < total_frames:
                cap.release()
                raise RuntimeError(
                    f"Video decode failed at frame {frame_idx}/{total_frames}. "
                    "OpenCV stopped returning frames before the reported frame count was reached."
                )
            break
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = nearest_multiple(w, IMAGE_DIM_DIV_FACTOR) if not OVERRIDE_RESIZE_DIM else OVERRIDE_RESIZE_DIM
            target_h = nearest_multiple(h, IMAGE_DIM_DIV_FACTOR) if not OVERRIDE_RESIZE_DIM else OVERRIDE_RESIZE_DIM
            if target_w != w or target_h != h:
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    print(f"Decoded {len(frames)}/{total_frames} frames from video.{' ' * 20}")
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames


def main() -> None:
    args = parse_args()

    frames = load_video_frames(args.video, args.max_frames)

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    processed = processor.video_processor(videos=[frames], fps=args.fps, return_tensors="pt")

    pixel_values_videos = processed["pixel_values_videos"] if "pixel_values_videos" in processed else processed["pixel_values"]
    video_grid_thw = processed["video_grid_thw"] if "video_grid_thw" in processed else processed["image_grid_thw"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pixel_values": pixel_values_videos.contiguous(),
            "video_grid_thw": video_grid_thw.contiguous(),
        },
        str(args.out),
    )

    print(
        f"Saved pixel_values shape={tuple(pixel_values_videos.shape)} dtype={pixel_values_videos.dtype}, "
        f"video_grid_thw={video_grid_thw.tolist()} to {args.out}"
    )


if __name__ == "__main__":
    main()
