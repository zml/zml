import torch
import diffusers

import zml_utils

model_path = "stabilityai/sd-turbo"

device = "cpu"
precision = torch.float32
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

pipe = diffusers.AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=precision)
pipe = zml_utils.ActivationCollector(pipe, blacklist_regexes=[r"text_encoder.*"])
output, activations = pipe(prompt=prompt, num_inference_steps=3, guidance_scale=0.0)

image = output.images[0]
image.save("output.png")

filename = model_path.split("/")[-1] + ".activations.pt"

print(f"Found {len(activations)} activations")
for k in list(activations.keys()):
    if (activations[k].dtype == torch.float32):
        activations[k] = activations[k].to(torch.float16).contiguous();

breakpoint()
print(f"Saving {len(activations)} activations to {filename}")

torch.save(activations, filename)
