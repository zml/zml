# /// script
# dependencies = [
#   "accelerate",
#   "torch",
#   "transformers",
#   "safetensors",
# ]
# ///
import sys
from pathlib import Path

import safetensors
import torch
import transformers

ROOT = Path(__file__).parents[2]
sys.path.append(str(ROOT / "tools"))
import zml_utils

model_path = "/var/models/openai/gpt-oss-20b"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    # device="cuda",
    device="cpu",
)
model, tokenizer = pipeline.model, pipeline.tokenizer

prompt = "What is the largest animal?"
# Wrap the pipeline, and extract activations.
# Activations files can be huge for big models,
# so let's stop collecting after 1000 layers.
pipeline = zml_utils.ActivationCollector(pipeline, max_layers=1000, stop_after_first_step=True)
output, activations = pipeline(prompt)

# `output` can be `None` if activations collection
# has stopped before the end of the inference
if output:
    print(output)

for i in range(4):
    print(activations[f"model.in.{i}"])

# Ask pytorch to dequantize to check if our dequantize works too.
activations["model.model.layers.22.mlp.experts.gate_up_proj"] = model.model.layers[22].mlp.experts.gate_up_proj.data.to(torch.bfloat16).cpu()
activations["model.model.layers.22.mlp.experts.down_proj"] = model.model.layers[22].mlp.experts.down_proj.data.to(torch.bfloat16).cpu()

# Save activations to a file.
filename = model_path.split("/")[-1] + ".activations.pt"
zml_utils.save_with_confirmation(filename, activations)
print(f"Saved {len(activations)} activations to {filename}")
