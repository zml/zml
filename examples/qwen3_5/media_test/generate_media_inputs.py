from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata, load_video

IMAGE_DIM_DIV_FACTOR = 32


def nearest_multiple(value: int, multiple: int) -> int:
    floor = (value // multiple) * multiple
    ceil = ((value + multiple - 1) // multiple) * multiple
    floor = max(multiple, floor)
    ceil = max(multiple, ceil)
    if value - floor < ceil - value:
        return floor
    return ceil


def resize_dimensions(width: int, height: int, max_pixels: int, multiple: int) -> tuple[int, int]:
    if max_pixels <= 0:
        raise ValueError("--max-pixels must be positive")

    current_pixels = width * height
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))

    target_width = nearest_multiple(width, multiple)
    target_height = nearest_multiple(height, multiple)

    while target_width * target_height > max_pixels and (target_width > multiple or target_height > multiple):
        if target_width >= target_height and target_width > multiple:
            target_width -= multiple
        elif target_height > multiple:
            target_height -= multiple
        else:
            break

    return max(multiple, target_width), max(multiple, target_height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mixed media safetensors inputs for qwen3_5 media_test")
    parser.add_argument("--model", type=Path, required=True, help="Local model path (contains processor/tokenizer config)")
    parser.add_argument("--out", type=Path, required=True, help="Output safetensors path")
    parser.add_argument("--image", action="append", default=[], type=Path, help="Image input path (repeatable)")
    parser.add_argument("--video", action="append", default=[], type=Path, help="Video input path (repeatable)")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS used by HF video frame sampler")
    parser.add_argument("--max-pixels", type=int, default=262144, help="Maximum pixel count for each image and video frame after resizing")
    return parser.parse_args()


def load_video_with_fallback(processor: AutoProcessor, video_path: Path, fps: float):
    sample_indices_fn = partial(processor.video_processor.sample_frames, fps=fps)
    try:
        video, video_metadata = load_video(
            str(video_path),
            backend="pyav",
            sample_indices_fn=sample_indices_fn,
        )
        return video, video_metadata
    except IndexError:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        decoded_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not decoded_frames:
            raise RuntimeError(f"Failed to decode any frames from {video_path}")

        video_fps = float(stream.average_rate) if stream.average_rate is not None else 24.0
        video_metadata = VideoMetadata(
            total_num_frames=len(decoded_frames),
            fps=video_fps,
            duration=len(decoded_frames) / video_fps if video_fps else 0.0,
            video_backend="pyav",
            height=decoded_frames[0].shape[0],
            width=decoded_frames[0].shape[1],
        )
        indices = sample_indices_fn(metadata=video_metadata)
        video_metadata.frames_indices = indices
        video = np.stack([decoded_frames[int(idx)] for idx in indices], axis=0)
        return video, video_metadata


def process_image(processor: AutoProcessor, image_path: Path, max_pixels: int):
    image = Image.open(image_path).convert("RGB")
    target_width, target_height = resize_dimensions(image.width, image.height, max_pixels, IMAGE_DIM_DIV_FACTOR)
    image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)

    processed = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = processed["pixel_values"].contiguous()
    image_grid_thw = processed["image_grid_thw"][0].to(torch.int64).contiguous()
    return pixel_values, image_grid_thw


def process_video(processor: AutoProcessor, video_path: Path, fps: float, max_pixels: int):
    video, video_metadata = load_video_with_fallback(processor, video_path, fps)
    target_width, target_height = resize_dimensions(video_metadata.width, video_metadata.height, max_pixels, IMAGE_DIM_DIV_FACTOR)
    video = np.stack(
        [np.array(Image.fromarray(frame).resize((target_width, target_height), Image.Resampling.BICUBIC)) for frame in video],
        axis=0,
    )

    processed = processor.video_processor(
        videos=[video],
        video_metadata=[video_metadata],
        do_sample_frames=False,
        do_resize=False,
        return_metadata=True,
        return_tensors="pt",
    )

    pixel_values = processed["pixel_values_videos"] if "pixel_values_videos" in processed else processed["pixel_values"]
    video_grid_thw = processed["video_grid_thw"] if "video_grid_thw" in processed else processed["image_grid_thw"]
    video_metadata = processed["video_metadata"][0]
    timestamps = processor._calculate_timestamps(
        video_metadata.frames_indices,
        video_metadata.fps,
        processor.video_processor.temporal_patch_size,
    )

    return pixel_values.contiguous(), video_grid_thw[0].to(torch.int64).contiguous(), timestamps


def main() -> None:
    args = parse_args()

    if not args.image and not args.video:
        raise ValueError("At least one --image or --video must be provided")

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)

    tensors: dict[str, torch.Tensor] = {}
    media_index = 0

    for image_path in args.image:
        pixel_values, image_grid = process_image(processor, image_path, args.max_pixels)
        prefix = f"media.{media_index}"
        tensors[f"{prefix}.type"] = torch.tensor(0, dtype=torch.int64)
        tensors[f"{prefix}.grid_thw"] = image_grid
        tensors[f"{prefix}.pixel_values"] = pixel_values
        media_index += 1

    for video_path in args.video:
        pixel_values, video_grid, timestamps = process_video(processor, video_path, args.fps, args.max_pixels)
        prefix = f"media.{media_index}"
        tensors[f"{prefix}.type"] = torch.tensor(1, dtype=torch.int64)
        tensors[f"{prefix}.grid_thw"] = video_grid
        tensors[f"{prefix}.pixel_values"] = pixel_values
        tensors[f"{prefix}.timestamps"] = torch.tensor(timestamps, dtype=torch.float32)
        media_index += 1

    media_count = media_index
    tensors["media_count"] = torch.tensor(media_count, dtype=torch.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))

    print(
        f"Saved {args.out} with media_count={media_count}, "
        f"videos={len(args.video)}, images={len(args.image)}"
    )


if __name__ == "__main__":
    main()
