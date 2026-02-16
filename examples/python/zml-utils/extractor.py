import torch
import transformers
import zml_utils
import sys
from diffusers import Flux2KleinPipeline
import safetensors.torch as safetensors_torch
import numpy as np

# dtype = torch.bfloat16
dtype = torch.float32
# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# prompt = "A flying surperman style cat"
prompt = "A photo of a cat"

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "black-forest-labs/FLUX.2-klein-4B"
pipeline = Flux2KleinPipeline.from_pretrained(
    model_path,
    dtype=dtype
)
pipeline.to(device)
pipeline = zml_utils.ActivationCollector(pipeline, max_layers=-1, stop_after_first_step=False, blacklist_regexes=[])
output, activations = pipeline(        
        prompt=prompt,
        height=32,
        width=32,
        guidance_scale=1.0,
        num_inference_steps=1,
        generator=torch.Generator(device=device).manual_seed(0))

if output:
    print(output)

filename = model_path.split("/")[-1]

match device:
    case "mps" | "cuda":
        activations = {k: v.contiguous() for k, v in activations.items()}
    case "cpu":
        activations = {k: v.clone(memory_format=torch.contiguous_format) for k, v in activations.items()}
    case _:
        raise ValueError(f"Unsupported device: {device}")
        
safetensors_torch.save_file(activations, f"{filename}.activations.safetensors")
print(f"Saved {len(activations)} activations to {filename}.activations.safetensors")
