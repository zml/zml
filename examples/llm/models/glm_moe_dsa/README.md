# GLM-5.2 bring-up harness

This directory contains a standalone GLM-MoE-DSA implementation and activation
test harness. It is intentionally not registered with `examples/llm` inference
yet.

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
