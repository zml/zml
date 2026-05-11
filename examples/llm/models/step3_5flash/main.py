import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file
import zml_utils
import os

MODEL_PATH = "stepfun-ai/Step-3.5-Flash"

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if isinstance(tokenizer.eos_token_id, list):
    config.pad_token_id = tokenizer.eos_token_id[0]
else:
    config.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain the significance of the number 42."}]

# replace model with zml_utils tracked model
model = zml_utils.ActivationCollector(model, max_layers=1000, stop_after_first_step=True)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# perform forward pass to collect activations
outputs, activations = model(inputs)

output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(output_text)
print(activations.keys())
