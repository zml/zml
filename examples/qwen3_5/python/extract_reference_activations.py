import torch
import transformers
import zml_utils
from safetensors.torch import load_file, save_file

model_path = "/var/models/Qwen/Qwen3.5-0.8B"
 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={},
    device="cuda",
)

model, tokenizer = pipeline.model, pipeline.tokenizer

prompt = "What is the capital of France?\n"
pipeline = zml_utils.ActivationCollector(
    pipeline,
    max_layers=1000,
    stop_after_first_step=True,
    include_layer_caches=True,
)
output, activations = pipeline(prompt)
print(output)

filename = "../safetensors/" + model_path.split("/")[-1] + ".activations-bf16-with-caches.safetensors"
for k in activations.keys():
    if k.endswith("linear_attn.out.1"): # Resize the torch linear conv-state cache which has an extra (useless) dimension
        activations[k] = activations[k][:,:,1:]
activations = {k: v.contiguous() for k, v in activations.items()}

save_file(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")
