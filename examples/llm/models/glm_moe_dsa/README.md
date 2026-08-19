# GLM-5.2

This directory contains the GLM-MoE-DSA model, reusable prefill/decode graphs,
an `examples/llm` session, and an activation test harness. The session uses a
small built-in approximation of the checkpoint's chat template.

The default test configuration constructs literal checkpoint layers 0 through
3 and overrides `index_topk` to 8. That prefix covers dense MLPs, a routed MoE,
full DSA indexers, a shared DSA indexer, interleaved RoPE, and prefill/decode
caches while fitting comfortably on a development host.

Generate the PyTorch bf16 fixture with the Transformers checkout that defines
`GlmMoeDsaForCausalLM`:

```sh
HIP_VISIBLE_DEVICES=0 \
  ~/github/huggingface/transformers/.venv/bin/python \
  examples/llm/models/glm_moe_dsa/activations.py \
  --model=/var/models/zai-org/GLM-5.2 \
  --output=/tmp/glm_moe_dsa_4layers.safetensors
```

Build and run the ZML comparisons across all visible AMD devices:

```sh
bazel build //examples/llm:glm_moe_dsa_tests \
  --@zml//platforms:rocm=true \
  --@zml//platforms:cpu=false

bazel-bin/examples/llm/glm_moe_dsa_tests \
  --model=/var/models/zai-org/GLM-5.2 \
  --activations=/tmp/glm_moe_dsa_4layers.safetensors
```

The CPU backend runs the projection, normalization, dense/shared MLP, full and
shared DSA, RoPE, and K/V/indexer-cache checks. It skips the routed-MoE and full
four-layer checks because ZML has no fused MoE backend for CPU.

Run the complete checkpoint across the visible AMD devices with:

```sh
mkdir -p /tmp/$(whoami)/xla_cache

bazel build --config=release //examples/llm:llm \
  --@zml//platforms:rocm=true \
  --@zml//platforms:cpu=false

ZML_AUTOTUNE_CACHE_DIR=/tmp/$(whoami)/xla_cache \
  bazel-bin/examples/llm/llm \
  --model=/var/models/zai-org/GLM-5.2 \
  --prompt="Who are you?" \
  --seqlen=512 \
  --glm-index-topk=512 \
  --gpu-memory-fraction=0.97
```

The checkpoint's native `index_topk=2048` needs a roughly 16 GiB prefill
workspace per device in addition to the model weights. The override keeps all
78 checkpoint layers while reducing that temporary working set for bring-up.
