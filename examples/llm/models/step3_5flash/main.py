import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import zml_utils
import os

MODEL_PATH = "stepfun-ai/Step-3.5-Flash"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain the significance of the number 42."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 3. Generate
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
output_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(output_text)
