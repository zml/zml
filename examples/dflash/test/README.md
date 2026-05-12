# DFlash Python Flow

Run the Python reference DFlash model with the same one-block flow as
`examples/dflash/main.zig`: target prefill, DFlash draft forward, target
`lm_head`, and argmax speculative tokens.

```sh
uv venv examples/dflash/test/.venv
uv pip install --python examples/dflash/test/.venv/bin/python torch transformers accelerate safetensors typing-extensions

DFLASH_SOURCE=/Users/tristan/codebase/third_party/dflash \
examples/dflash/test/.venv/bin/python examples/dflash/test/python_flow.py \
  --target-model /Users/tristan/models/meta-llama/Llama-3.1-8B-Instruct \
  --dflash-model /Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
```

On the GPU host:

```sh
DFLASH_SOURCE=/home/tristan/dflash \
examples/dflash/test/.venv/bin/python examples/dflash/test/python_flow.py \
  --target-model /var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --dflash-model /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/
```
