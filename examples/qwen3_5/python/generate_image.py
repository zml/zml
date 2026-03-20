from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vision input safetensors using the official HF processor pipeline"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/var/models/Qwen/Qwen3.5-0.8B"),
        help="Local model path (contains processor/tokenizer config)",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path (BMP/PNG/JPEG)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/tristan/zml/examples/qwen3_5/safetensors/vision_input.safetensors"),
        help="Output safetensors file",
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    image = Image.open(args.image).convert("RGB")

    # Use the exact vision preprocessor path (avoid processor text/chat template path).
    processed = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]

    # vision_tests.zig expects flattened patch rows.
    if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values[0]
    elif pixel_values.ndim != 2:
        raise ValueError(f"Unexpected pixel_values shape from processor: {tuple(pixel_values.shape)}")

    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(f"Unexpected image_grid_thw shape from processor: {tuple(image_grid_thw.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pixel_values": pixel_values.contiguous(),
            "image_grid_thw": image_grid_thw.contiguous(),
        },
        str(args.out),
    )
    print(
        f"Saved pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}, "
        f"image_grid_thw={image_grid_thw.tolist()} to {args.out}"
    )


if __name__ == "__main__":
    main()
