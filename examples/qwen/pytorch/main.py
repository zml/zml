import torch
from diffusers import QwenImage21Pipeline

size = 2048

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", 
    torch_dtype=torch.bfloat16
)

# Enable VAE tiling and slicing directly on the VAE module
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Enable CPU offloading on the main pipeline to handle memory shifts
pipe.enable_model_cpu_offload()

image = pipe(
    prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
    width=size,
    height=size,
    num_inference_steps=10,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("t2i_example.png")
