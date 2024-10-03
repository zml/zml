import torch
import diffusers

import zml_utils

model_path = "stabilityai/sd-turbo"

device = "cpu"
precision = torch.float32
prompt = "A baby panda drinking tea"

pipe = diffusers.AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=precision)
pipe = zml_utils.ActivationCollector(pipe, blacklist_regexes=[r"text_encoder.*"])
output, activations = pipe(prompt=prompt, num_inference_steps=2, strength=0.5, guidance_spcale=0.0)

image = output.images[0]
image.save("output.png")

filename = model_path.split("/")[-1] + ".activations.pt"

print(f"Found {len(activations)} activations")
for k in list(activations.keys()):
    if (activations[k].dtype == torch.float32):
        activations[k] = activations[k].to(torch.float16);

breakpoint()
print(f"Saving {len(activations)} activations to {filename}")

torch.save(activations, filename)
