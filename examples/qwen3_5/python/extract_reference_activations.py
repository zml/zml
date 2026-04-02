import torch
import transformers
from safetensors.torch import save_file

import zml_utils


model_path = "/var/models/Qwen/Qwen3.5-0.8B"
prompt = "What is the capital of France?\n"
filename = "../safetensors/" + model_path.split("/")[-1] + ".activations-bf16-with-caches.safetensors"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={},
    device="cuda",
)
model, tokenizer = pipeline.model, pipeline.tokenizer

chat = [{"role": "user", "content": prompt}]
model_inputs = tokenizer.apply_chat_template(
    chat,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
if hasattr(model_inputs, "to"):
    model_inputs = model_inputs.to(model.device)

if hasattr(model_inputs, "input_ids"):
    input_ids = model_inputs.input_ids
elif hasattr(model_inputs, "keys") and "input_ids" in model_inputs:
    input_ids = model_inputs["input_ids"]
else:
    input_ids = model_inputs

if not isinstance(input_ids, torch.Tensor):
    raise TypeError(f"Expected input_ids to be a Tensor, got {type(input_ids)}")

attention_mask = torch.ones_like(input_ids, device=model.device)

with torch.no_grad():
    model_output = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

cache_tensors = zml_utils._named_full_cache_tensors(model_output.past_key_values)
if not cache_tensors:
    raise RuntimeError("No cache tensors extracted from model_output.past_key_values")

activations = {
    "model.model.in.0": input_ids.detach().clone().cpu(),
    "model.model.out.0": model_output.last_hidden_state.detach().clone().cpu(),
}

for suffix, tensor in cache_tensors.items():
    activations[f"model.model.cache_out.{suffix}"] = tensor.contiguous()

save_file({key: value.contiguous() for key, value in activations.items()}, filename)
print("Saved keys:")
for key in sorted(activations):
    print(key, activations[key].shape, activations[key].dtype)
print(f"Saved {len(activations)} activations to {filename}")
