import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from accelerate import Accelerator
import zml_utils
import os

def main():
    # Accelerator for GPUs
    accelerator = Accelerator()
    device = accelerator.device

    MODEL_PATH = "stepfun-ai/Step-3.5-Flash"

    # Pipeline. used for zml_utils
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_PATH,
        model_kwargs={
            "torch_dtype": torch.float16,
            "device_map": "auto",
        },
    )

    # Setup model and tokenizer
    model, tokenizer = pipeline.model, pipeline.tokenizer

    # Prepare Input
    prompt = "Q: Explain the significance of the number 42.\nA:"

    # Wrap the pipeline, and extract activations.
    # Stop collecting after 1000 layers
    pipeline = zml_utils.ActivationCollector(pipeline, max_layers=1000, stop_after_first_step=True)
    output, activations = pipeline(prompt)

    # Guard against empty output (early stopping)
    if output:
        if accelerator.is_main_process:
            print(output)

    # Save activations to a file
    if accelerator.is_main_process:
        filename = model_path.split("/")[-1] + ".activations.safetensors"
        save_file(activations, filename)
        print(f"Saved {len(activations)} activations to {filename}")

if __name__ == "__main__":
    main()
