import torch
import transformers
import sys
from diffusers import Flux2KleinPipeline

# dtype = torch.bfloat16
dtype = torch.float32
# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# device = "mps"

# prompt = "A flying surperman style cat"
prompt = "A photo of a cat"

model_path = "black-forest-labs/FLUX.2-klein-4B"
pipeline = Flux2KleinPipeline.from_pretrained(
    model_path,
    dtype=dtype
)
pipeline.to(device)

dim = 128

output = pipeline(
        prompt=prompt,
        height=dim,
        width=dim,
        guidance_scale=1.0,
        num_inference_steps=1,
        max_sequence_length=20,
        generator=torch.Generator(device=device).manual_seed(0))

output.images[0].save("flux-klein.png")
