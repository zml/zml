from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import av
import numpy as np
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata, load_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video input safetensors using the official HF processor pipeline")
    parser.add_argument("--model", type=Path, required=True, help="Local model path (contains processor/tokenizer config)")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out", type=Path, required=True, help="Output safetensors file path")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS used by the HF video processor sampler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    sample_indices_fn = partial(processor.video_processor.sample_frames, fps=args.fps)
    try:
        video, video_metadata = load_video(
            str(args.video),
            backend="pyav",
            sample_indices_fn=sample_indices_fn,
        )
    except IndexError:
        container = av.open(str(args.video))
        stream = container.streams.video[0]
        decoded_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not decoded_frames:
            raise RuntimeError(f"Failed to decode any frames from {args.video}")

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

    video = np.stack( 
        [np.array(Image.fromarray(frame).resize((256, 256), Image.BICUBIC)) for frame in video], # Resize to limit input size
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

    pixel_values_videos = processed["pixel_values_videos"] if "pixel_values_videos" in processed else processed["pixel_values"]
    video_grid_thw = processed["video_grid_thw"] if "video_grid_thw" in processed else processed["image_grid_thw"]
    video_metadata = processed["video_metadata"][0]
    timestamps = processor._calculate_timestamps(
        video_metadata.frames_indices,
        video_metadata.fps,
        processor.video_processor.temporal_patch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pixel_values": pixel_values_videos.contiguous(),
            "video_grid_thw": video_grid_thw.contiguous(),
            "timestamps": pixel_values_videos.new_tensor(timestamps, dtype=pixel_values_videos.dtype).contiguous(),
        },
        str(args.out),
    )

    print(
        f"Saved pixel_values shape={tuple(pixel_values_videos.shape)} dtype={pixel_values_videos.dtype}, "
        f"video_grid_thw={video_grid_thw.tolist()} timestamps={timestamps} to {args.out}"
    )


if __name__ == "__main__":
    main()
