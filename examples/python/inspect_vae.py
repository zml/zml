
import torch
from diffusers import Flux2KleinPipeline
import json

# Load Model
repo_id = "black-forest-labs/FLUX.2-klein-4B"
# Use float16 or bfloat16 to avoid large download if possible, but structure matters most
dtype = torch.bfloat16

print(f"Loading {repo_id} VAE...")
try:
    pipe = Flux2KleinPipeline.from_pretrained(repo_id, dtype=dtype)
    vae = pipe.vae
    print("VAE loaded.")

    print("\n--- VAE Keys Check ---")
    keys = list(vae.state_dict().keys())
    print(f"Total VAE state_dict keys: {len(keys)}")
    # Filter for up_blocks
    up_keys = [k for k in keys if "up_blocks" in k]
    
    # Sort to be readable
    up_keys.sort()

    print(f"Found {len(up_keys)} up_blocks keys.")
    for k in up_keys:
        # Print first few to save space, and check specifically for up_blocks.0.resnets.0
        if "up_blocks.0.resnets.0" in k:
            shape = vae.state_dict()[k].shape
            print(f"{k}: {shape}")

    print("\n--- Checking for up_blocks.0.resnets.0.conv2.weight ---")
    target = "decoder.up_blocks.0.resnets.0.conv2.weight"
    
    found = False
    list_of_keys = []
    for k in keys:
        list_of_keys.append({k: str(vae.state_dict()[k].dtype)})
        if "up_blocks.0.resnets.0.conv2.weight" in k:
            print(f"FOUND: {k} -> {vae.state_dict()[k].shape}")
            found = True
    
    with open("VAE.json", "w") as f:
        json.dump(list_of_keys, f, indent=4)

    if not found:
        print("NOT FOUND: up_blocks.0.resnets.0.conv2.weight")
    print("\n--- Checking for Batch Norm keys ---")
    bn_keys = [k for k in keys if "bn" in k or "running_mean" in k or "running_var" in k]
    for k in bn_keys:
        print(f"BN Key: {k} -> {vae.state_dict()[k].shape}")
    
    if not bn_keys:
        print("No 'bn' or 'running_mean' keys found.")

    print("\nKeys in up_blocks.0.resnets.0:")
    for k in keys:
             if "up_blocks.0.resnets.0" in k:
                 print(k)

except Exception as e:
    print(f"Error: {e}")
