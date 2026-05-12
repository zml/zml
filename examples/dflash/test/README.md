# DFlash Activation Comparison

This directory builds a local fixture-and-compare loop for the Zig DFlash model.

Generate the Python reference:

```sh
cd /Users/tristan/codebase/zml
uv venv --python /opt/homebrew/bin/python3.11 examples/dflash/test/.venv
uv pip install --python examples/dflash/test/.venv/bin/python \
  torch transformers==4.57.1 accelerate typing-extensions safetensors
examples/dflash/test/.venv/bin/python examples/dflash/test/extract_reference.py \
  --target-model /Users/tristan/models/meta-llama/Llama-3.1-8B-Instruct \
  --dflash-model /Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
  --out /tmp/dflash_activations.safetensors
```

Compare Zig against that reference:

```sh
bazel run //examples/dflash:dflash_compare_activations -- \
  --model /Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
  --activations /tmp/dflash_activations.safetensors
```

The safetensors fixture contains:

- `input_ids`, `target_token`, `noise_tokens`, `position_ids`
- `target_hidden`, the concatenated hidden states extracted from the target LLaMA layers
- `target_hidden_projected`, the DFlash `hidden_norm(fc(target_hidden))` value fed to every draft layer
- `noise_embedding`, the target embedding of the DFlash noise block
- `layers.N.in` and `layers.N.out`, for every DFlash decoder layer
- `final_out`, the final DFlash norm output
