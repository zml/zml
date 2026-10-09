import torch
from safetensors.torch import save_file
from diffusers import QwenImage21Pipeline
import zml_utils

size = 2048
model_path = "Qwen/Qwen-Image-2.1"

pipe = QwenImage21Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
transformer = (
    getattr(pipe, pipe.transformer_name)
    if not hasattr(pipe, "transformer")
    else pipe.transformer
)
# print(transformer)
if transformer is None:
    exit(1)

# # 1. Print named submodules to map key paths directly to safetensors
# for name, module in transformer.named_children():
#     print(name, "->", type(module))

# 2. Inspect a single block's internals
# print(transformer.transformer_blocks[0])

# Enable VAE tiling and slicing directly on the VAE module
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Enable CPU offloading on the main pipeline to handle memory shifts
pipe.enable_model_cpu_offload()

pipe = zml_utils.ActivationCollector(
    pipe, skip=0, max_layers=500, stop_after_first_step=True
)
output, activations = pipe(
    prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
    width=size,
    height=size,
    num_inference_steps=1,
    generator=torch.Generator("cuda").manual_seed(42),
)

# `output` can be `None` if activations collection
# has stopped before the end of the inference
if output:
    print("Saving image output")
    output.images[0].save("t2i_example.png")

# Save activations to a file.
filename = "out.activations.safetensors"
save_file(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")
