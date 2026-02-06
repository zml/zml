
import numpy as np
import os
import sys
from PIL import Image


def convert_image_to_pil_png(image: np.ndarray, image_name: str) -> None:
    image_magik = (image / 2 + 0.5).clip(0, 1)
    print(f"image_magik dtype: {image_magik.dtype}")
    image_magik = image_magik.transpose(0, 2, 3, 1)
    image_magik = (image_magik * 255).round().astype("uint8")
    print(f"image 20 : {image_magik.flatten()[:20]}")
    first_image = image_magik[0]
    print(f"image_hand[0] 20 : {first_image.flatten()[:20]}")
    np.save(f"{image_name}.npy", first_image)
    first_image_pil = Image.fromarray(first_image)
    first_image_pil.save(f"{image_name}.png")


def verify_npy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False
    
    try:
        data = np.load(filename)
        print(f"Successfully loaded {filename}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Data sample: {data.flatten()[:10]}")
        convert_image_to_pil_png(data, filename)
        return True
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return False

if __name__ == "__main__":
    # Create a dummy file if it doesn't exist for testing logic
    # But main goal is to verify the ZIG output. 
    # Since I cannot run the ZIG code here easily (missing weights/input files),
    # I will just exit. The user can run this script after running the zig program.
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        verify_npy(filename)
    else:
        print("Usage: python verify_npy.py <filename>")
