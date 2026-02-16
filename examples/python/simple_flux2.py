import torch
import transformers
import sys
from diffusers import Flux2KleinPipeline

# uv pip install git+https://github.com/huggingface/diffusers.git
# uv pip install torch pils accelerate transformers
# export CUDA_VISIBLE_DEVICES=0

dtype = torch.bfloat16
torch.set_default_dtype(dtype)

# dtype = torch.float32
# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# device = "mps"
device = "cuda"

# prompt = "A flying surperman style cat"
prompt = "A photo of a cat"

model_path = "black-forest-labs/FLUX.2-klein-4B"
pipeline = Flux2KleinPipeline.from_pretrained(
    model_path,
    dtype=dtype
)
pipeline.to(device)

output = pipeline(
        prompt=prompt,
        width=1920,
        height=1080,
        num_inference_steps=4,
        max_sequence_length=512,
        generator=torch.Generator(device="cpu").manual_seed(0))

output.images[0].save("flux-klein.png")
