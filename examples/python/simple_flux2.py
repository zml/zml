import torch
import transformers
import sys
import os
# from diffusers import Flux2KleinPipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from flux2.pipeline_flux2_klein import Flux2KleinPipeline
import time



def run_pipeline():
    # uv pip install git+https://github.com/huggingface/diffusers.git
    # uv pip install torch pils accelerate transformers
    # export CUDA_VISIBLE_DEVICES=0

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)

    # dtype = torch.float32
    # device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # device = "mps"
    device = "cuda"

    # prompt = "A flying surperman style cat"
    prompt = "A photo of a cat on a bed"

    model_path = "black-forest-labs/FLUX.2-klein-4B"
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_path,
        dtype=dtype
    )
    pipeline.to(device)

    output = pipeline(
            prompt=prompt,
            # width=128,
            # height=128,
            width=1920,
            height=1080,
            num_inference_steps=4,
            max_sequence_length=512,
            generator=torch.Generator(device=device).manual_seed(0))
    output.images[0].save("flux-klein.png")


def main():
    print(">>> Starting pipeline...")
    # Time for the pipeline to run is measured in the main function to include all setup time.
    start_time = time.time()
    run_pipeline()
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f">>> Pipeline completed in {elapsed_time_ms:.2f} ms.")
    print(">>> Pipeline finished.")

if __name__ == "__main__":
    main()
