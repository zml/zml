from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


def preprocess_image_to_pixel_values(
    image_path: Path,
    resized_h: int = 256,
    resized_w: int = 256,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    if resized_h % (patch_size * temporal_patch_size) != 0 or resized_w % (patch_size * temporal_patch_size) != 0:
        raise ValueError(
            f"resized_h/resized_w must be multiples of patch_size*temporal_patch_size ({patch_size * temporal_patch_size})"
        )

    img = Image.open(image_path).convert("RGB").resize((resized_w, resized_h), Image.BICUBIC)

    # HWC uint8 -> CHW float32
    x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).to(torch.float32)

    # Match qwen3_vl preprocessing: (pixel / 255 - 0.5) / 0.5
    x = (x / 255.0 - 0.5) / 0.5

    # Add temporal axis and duplicate to temporal_patch_size=2
    x = x.unsqueeze(1).repeat(1, temporal_patch_size, 1, 1)  # (c, t, h, w)

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError("grid_h and grid_w must be divisible by spatial_merge_size")

    # Equivalent to qwen3_vl splitAxis+transpose path:
    # (c,t,h,w) -> (h_div,w_div,m1,m2,c,t,patch1,patch2) -> flatten
    x = x.view(
        3,
        temporal_patch_size,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    x = x.permute(2, 5, 3, 6, 0, 1, 4, 7).contiguous()
    pixel_values = x.view(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)

    # qwen3_5 vision_tests currently uses grid_thw={1,16,16}, so expected shape is (256, 1536)
    expected_n = 16 * 16
    expected_f = 3 * temporal_patch_size * patch_size * patch_size
    if pixel_values.shape != (expected_n, expected_f):
        raise ValueError(
            f"Unexpected pixel_values shape {tuple(pixel_values.shape)}, expected {(expected_n, expected_f)}"
        )
    return pixel_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3.5 vision pixel_values safetensor from an image")
    parser.add_argument("--image", type=Path, required=True, help="Input image path (BMP/PNG/JPEG; BMP is simplest)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/tristan/zml/examples/qwen3_5/safetensors/vision_input.safetensors"),
        help="Output safetensors file",
    )
    parser.add_argument("--height", type=int, default=256, help="Resized height (default 256 for qwen3_5 tests)")
    parser.add_argument("--width", type=int, default=256, help="Resized width (default 256 for qwen3_5 tests)")
    parser.add_argument("--patch_size", type=int, default=16, help="Vision patch size (qwen3_5 default: 16)")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="Vision temporal patch size (qwen3_5 default: 2)")
    args = parser.parse_args()

    pixel_values = preprocess_image_to_pixel_values(
        args.image,
        resized_h=args.height,
        resized_w=args.width,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"pixel_values": pixel_values}, str(args.out))
    print(f"Saved pixel_values with shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype} to {args.out}")


if __name__ == "__main__":
    main()
