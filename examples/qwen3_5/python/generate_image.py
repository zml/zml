from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor

# Each image dimension must be a multiple of this value (patch_size * spatial_merge_size) to be compatible with the patch setup in the model
IMAGE_DIM_DIV_FACTOR = 32 

def nearest_multiple(value: int, multiple: int) -> int:
    floor = (value // multiple) * multiple
    ceil = ((value + multiple - 1) // multiple) * multiple
    floor = max(multiple, floor)
    ceil = max(multiple, ceil)
    if value - floor < ceil - value:
        return floor
    return ceil

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vision input safetensors using the official HF processor pipeline")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local model path (contains processor/tokenizer config)",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path (BMP/PNG/JPEG)")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,        
        help="Output safetensors file path",
    ) 
    return parser.parse_args()

    
def main() -> None:
    # Get args
    args = parse_args()

    # Resize image
    image = Image.open(args.image).convert("RGB")
    target_width = nearest_multiple(image.width, IMAGE_DIM_DIV_FACTOR)
    target_height = nearest_multiple(image.height, IMAGE_DIM_DIV_FACTOR)
    image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)

    # Process image using the models' processor to get pixel values and grid info
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    processed = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]
 
    # Save the pixel_values and image_grid_thw
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
