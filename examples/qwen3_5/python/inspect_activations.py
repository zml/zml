import argparse
from safetensors.torch import load_file, save_file


prefix_default = ""

p = argparse.ArgumentParser()
p.add_argument("--prefix", default=prefix_default, help="Prefix for filtering activations.")
p.parse_args()
prefix = p.parse_args().prefix


model_path = "/Users/tristan/models/qwen/Qwen3.5-0.8B"

filename = "../safetensors/" + model_path.split("/")[-1] + ".activations-bf16-with-caches.safetensors"
activations = load_file(filename)

print(f"Read {len(activations)} activations from {filename}")
for k in activations.keys():
    if not k.startswith(prefix):
        continue
    print(k, ":", activations[k].shape, str(activations[k].dtype).split(".")[-1])
    print(activations[k])
