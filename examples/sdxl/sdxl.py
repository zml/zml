import torch
import diffusers

import zml_utils

model_path = "stabilityai/sd-turbo"

device = "cpu"
precision = torch.float32
prompt = "A grand city in the year 2100, atmospheric, hyper realistic, 8k, epic composition"

pipe = diffusers.AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=precision)
pipe = zml_utils.ActivationCollector(pipe, blacklist_regexes=[r"text_encoder.*"])
output, activations = pipe(prompt=prompt, num_inference_steps=1, guidance_spcale=0.0, seed=1234)

image = output.images[0]
image.save("output.png")

filename = model_path.split("/")[-1] + ".activations.pt"

print(f"Found {len(activations)} activations")
breakpoint()
for k in list(activations.keys()):
    if k.startswith("text_encoder"):
        activations.pop(k)
print(f"Saving {len(activations)} activations to {filename}")

torch.save(activations, filename)
