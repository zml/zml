import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import zml_utils
import os

def main():
    MODEL_PATH = "stepfun-ai/Step-3.5-Flash"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # Prepare Input
    prompt = "Q: Explain the significance of the number 42.\nA:"

    output = pipeline(prompt)

    # Guard against empty output (early stopping)
    if output:
        print(output)

if __name__ == "__main__":
    main()
