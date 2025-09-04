import torch
import transformers
import sys
import safetensors

sys.path.append("/Users/guw/Documents/zml/tools")
import zml_utils

model_path = "/Users/guw/models/openai/gpt-oss-20b"

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

# Save activations to a file.
filename = model_path.split("/")[-1] + ".activations.pt"
torch.save(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")

filename = model_path.split("/")[-1] + ".activations.safetensors"
safetensors.torch.save_file(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")
