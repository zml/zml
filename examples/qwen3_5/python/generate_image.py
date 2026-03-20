from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


def smart_resize(
    height: int,
    width: int,
    temporal_patch_size: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 32 * 32,
    max_pixels: int = 32 * 32 * 768,
) -> tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"height={height} width={width} must be >= factor={factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio must be <= 200")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(1 / temporal_patch_size) * temporal_patch_size  # image => one frame

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def preprocess_image_to_pixel_values_and_grid(
    image_path: Path,
    resized_h: int | None = None,
    resized_w: int | None = None,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = patch_size * spatial_merge_size

    with Image.open(image_path) as pil:
        pil = pil.convert("RGB")
        orig_w, orig_h = pil.size
        if resized_h is None or resized_w is None:
            resized_h, resized_w = smart_resize(
                orig_h,
                orig_w,
                temporal_patch_size=temporal_patch_size,
                factor=factor,
            )
        if resized_h % factor != 0 or resized_w % factor != 0:
            raise ValueError(f"resized_h/resized_w must be multiples of {factor}")
        img = pil.resize((resized_w, resized_h), Image.BICUBIC)

    # HWC uint8 -> TCHW float32 with T=1
    x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).to(torch.float32)
    x = x.unsqueeze(0)  # (t=1, c, h, w)

    # Match HF Qwen VL processing
    x = (x / 255.0 - 0.5) / 0.5

    # Ensure T is divisible by temporal_patch_size by repeating last frame (image case => repeat once)
    t = x.shape[0]
    if t % temporal_patch_size != 0:
        pad = temporal_patch_size - (t % temporal_patch_size)
        x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)], dim=0)

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    grid_t = x.shape[0] // temporal_patch_size

    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0 or grid_t != 1:
        raise ValueError("invalid grid dimensions for image preprocessing")

    # HF layout:
    # (b=1, t, c, h, w) ->
    # (b, grid_t, temporal_patch_size, c, h_div, m1, patch, w_div, m2, patch) ->
    # permute to (b, grid_t, h_div, w_div, m1, m2, c, temporal_patch_size, patch, patch) -> flatten
    x = x.unsqueeze(0)  # (b=1, t, c, h, w)
    x = x.view(
        1,
        grid_t,
        temporal_patch_size,
        3,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
    pixel_values = x.view(1, grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size).squeeze(0)

    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)
    expected_rows = int(image_grid_thw[0, 0].item() * image_grid_thw[0, 1].item() * image_grid_thw[0, 2].item())
    expected_cols = 3 * temporal_patch_size * patch_size * patch_size
    if pixel_values.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"pixel_values shape mismatch: got {tuple(pixel_values.shape)}, expected {(expected_rows, expected_cols)}"
        )
    return pixel_values, image_grid_thw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3.5 vision pixel_values safetensor from an image")
    parser.add_argument("--image", type=Path, required=True, help="Input image path (BMP/PNG/JPEG; BMP is simplest)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/tristan/zml/examples/qwen3_5/safetensors/vision_input.safetensors"),
        help="Output safetensors file",
    )
    parser.add_argument("--height", type=int, default=None, help="Optional fixed resized height")
    parser.add_argument("--width", type=int, default=None, help="Optional fixed resized width")
    parser.add_argument("--patch_size", type=int, default=16, help="Vision patch size (qwen3_5 default: 16)")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="Vision temporal patch size (qwen3_5 default: 2)")
    args = parser.parse_args()

    pixel_values, image_grid_thw = preprocess_image_to_pixel_values_and_grid(
        args.image,
        resized_h=args.height,
        resized_w=args.width,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, str(args.out))
    print(
        f"Saved pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}, "
        f"image_grid_thw={image_grid_thw.tolist()} to {args.out}"
    )


if __name__ == "__main__":
    main()
