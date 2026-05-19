# DFlash Gemma 4 Implementation Spec

This document specifies the first implementation pass for a Gemma 4 target model in
the DFlash example. It intentionally does not add Zig code yet. The implementation
should live under `examples/dflash_gemma4` and should mirror the existing
`examples/dflash` flow from branch `tristan/dflash`, replacing the Llama 3.1 8B
target model with Gemma 4.

## Reference Access

Use the local `zml` checkout at:

```text
/Users/tristan/codebase/zml
```

The original DFlash implementation is on the `tristan/dflash` branch:

```bash
cd /Users/tristan/codebase/zml
git checkout tristan/dflash
```

Read these files there:

```text
examples/dflash/main.zig
examples/dflash/dflash_model.zig
examples/dflash/common.zig
examples/dflash/llama/llama_inference.zig
examples/dflash/llama/llama_model.zig
examples/dflash/BUILD.bazel
examples/dflash/model.md
```

Return to this branch before implementing:

```bash
cd /Users/tristan/codebase/zml
git checkout tristan/dflash_gemma4
```

The Gemma 4 implementation reference is in the `llmd` checkout at:

```text
/Users/tristan/codebase/monorepo/llmd
```

Read these files there:

```text
models/gemma4_text.zig
models/gemma4.zig
models.zig
main.zig
chat_template.zig
openai_service.zig
tool_call_parsers.zig
reasoning_parsers.zig
```

The important Gemma 4 model symbols are:

```text
models/gemma4_text.zig: Config, Gemma4TextLM, Gemma4Text, Gemma4DecoderLayer,
Gemma4Attention, Gemma4Mlp, Gemma4RmsNorm, Gemma4LayerScalar

models/gemma4.zig: Gemma4LM

models.zig: Options, AttentionLayouts, AttentionParameters,
BufferizedAttentionParameters, Gemma4CacheKind, Gemma4CacheLayout,
Gemma4Cache, KvCache
```

## Goal

Build a new `examples/dflash_gemma4` example that runs the DFlash speculative
decoding loop with:

- a DFlash draft model loaded from `--model`;
- a Gemma 4 target model loaded from `--target-model`;
- target hidden states extracted from configured Gemma 4 layers;
- Gemma 4 tokenization and EOS behavior;
- Gemma 4 KV cache semantics, including separate full-attention and
  sliding-attention cache pools.

The CLI should keep the existing DFlash user shape where possible:

```text
--model=<path>
--target-model=<path>
--prompt=<text>
--max-seq-len=<n>
--temperature=<t>
```

The help text must say "Gemma 4 target model", not Llama.

## Directory Layout

Create the implementation in this directory:

```text
examples/dflash_gemma4/
```

Expected files:

```text
examples/dflash_gemma4/BUILD.bazel
examples/dflash_gemma4/main.zig
examples/dflash_gemma4/dflash_model.zig
examples/dflash_gemma4/common.zig
examples/dflash_gemma4/gemma4/gemma4_inference.zig
examples/dflash_gemma4/gemma4/gemma4_model.zig
examples/dflash_gemma4/model.md
```

Start by copying the structure of `examples/dflash` from branch
`tristan/dflash`, then replace the target-model-specific Llama pieces with
Gemma 4 pieces.

## Existing DFlash Flow To Preserve

The current Llama DFlash example has this high-level flow in
`examples/dflash/main.zig`:

1. Parse CLI args.
2. Initialize VFS, platform, model repos, configs, tensor registries, tensor
   stores, tokenizer, and shardings.
3. Tokenize and pad prompt to `prefill_seq_len`.
4. Initialize draft model, target model, target KV cache, draft KV cache,
   attention metadata/parameters, RNG, and sampling config.
5. Compile target prefill.
6. Compile DFlash prefill drafter.
7. Compile DFlash steady-state drafter.
8. Compile target verify.
9. Load draft and target weights.
10. Run target prefill, producing the first target token and a target hidden
    block for the drafter.
11. Run the DFlash speculative decoding loop:
    - feed target hidden plus noisy embeddings to the draft model;
    - sample draft tokens through the target LM head;
    - verify draft tokens with the target model;
    - commit accepted tokens and correction token;
    - carry forward target and draft KV caches;
    - stream decoded text and print acceptance statistics.

Keep this control flow unless Gemma 4 requires a targeted deviation.

## DFlash Draft Model

The DFlash draft model in `examples/dflash/dflash_model.zig` is mostly
target-architecture-independent. It consumes:

```text
target_hidden: {.s, .d}
noise_embedding: {.s, .d}
position_ids: {.s}
draft_kv_cache
cache_index
active_context_len
```

The DFlash config fields to preserve are:

```text
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
attention_bias
rms_norm_eps
rope_theta
rope_scaling
block_size
dflash_config.target_layer_ids
dflash_config.mask_token_id
```

The initial Gemma 4 implementation should keep `dflash_model.zig` close to the
Llama branch version. Only change it if Gemma 4 DFlash checkpoints require
different tensor names or hidden normalization dimensions.

Important invariant:

```text
DFlash fc.weight input dimension must match:
target_layer_ids.len * target_hidden_width
```

For Gemma 4 text target hidden states, `target_hidden_width` is expected to be
`Gemma4TextLM.config.hidden_size`.

## Gemma 4 Target Model

Implement `gemma4/gemma4_model.zig` as the target-specific wrapper analogous to
`llama/llama_model.zig`, but based on `llmd/models/gemma4_text.zig`.

Support Gemma 4 text first. The wrapper may later add the top-level multimodal
`Gemma4LM`, but DFlash decoding needs only the text language model path.

### Config

Port `gemma4_text.Config` fields needed for inference:

```text
head_dim
global_head_dim
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
num_global_key_value_heads
rms_norm_eps
rope_parameters
sliding_window
attention_k_eq_v
final_logit_softcapping
num_kv_shared_layers
hidden_size_per_layer_input
vocab_size_per_layer_input
use_double_wide_mlp
bos_token_id
eos_token_id
layer_types
```

Include `Config.fixup` behavior from `llmd`:

- default `num_global_key_value_heads` to `num_key_value_heads`;
- default `global_head_dim` to `head_dim`;
- default full-attention RoPE to proportional scaling with
  `partial_rotary_factor = 0.25` and `rope_theta = 1_000_000`;
- default sliding-attention RoPE to default scaling with `rope_theta = 10_000`;
- default layer pattern to full attention every sixth layer, using
  `(i + 1) % 6 == 0`, and sliding attention otherwise.

Preserve `Config.free` behavior for allocated `layer_types` and EOS token arrays.

### Layers

Port these Gemma 4 layer semantics:

- scaled token embedding: `embed_tokens.forward(tokens) * sqrt(hidden_size)`;
- RMSNorm with optional scale;
- attention Q/K/V norms, including V norm without scale;
- optional `v_proj`; if absent, allow K=V only for full-attention layers with
  `attention_k_eq_v = true`;
- GELU MLP, not Llama SiLU MLP;
- post-attention and post-MLP norms before residual addition;
- per-layer `layer_scalar`;
- final norm before logits.

### Logits

Gemma 4 ties the LM head to `embed_tokens.weight`. Implement a target wrapper
method equivalent to Llama's `logitsForward`, but apply Gemma 4 final logit
softcapping:

```text
if final_logit_softcapping is present:
    logits = tanh(logits / cap) * cap
```

Use this same logits path for:

- target prefill sample;
- target verify logits;
- DFlash draft logits after the draft model produces hidden states.

### EOS

Port `Gemma4TextLM.isEosToken` from `llmd`:

- treat token `50` as EOS for `<|tool_response>`;
- treat token `106` as EOS for the top-level eot token currently used by
  Gemma 4 configs;
- also honor `config.eos_token_id`, either a single int or an array.

## Gemma 4 KV Cache

The Llama DFlash target cache is a single dense tensor:

```text
{.layer, .kv, .k, .h, .hd}
```

Gemma 4 cannot use that unchanged because it has two attention layouts:

- full attention;
- sliding attention.

Port the `llmd/models.zig` cache concepts into this example or adapt them into a
simpler non-paged DFlash-specific form.

Required logical concepts:

```text
Gemma4CacheKind = enum { sliding, full }
Gemma4CacheLayout
Gemma4Cache
```

`Gemma4CacheLayout` must map each logical layer to:

- whether the layer is full or sliding;
- which physical cache layer it writes or reads;
- whether this logical layer updates cache;
- how many physical sliding cache layers exist;
- how many physical full cache layers exist.

Preserve `num_kv_shared_layers` behavior:

- layers before the shared suffix update cache;
- shared suffix layers reuse the latest physical cache layer of the same kind;
- reject shared layers if there is no prior anchor layer of that kind.

For a first DFlash implementation, prefer a contiguous sequence-length cache over
the full paged `llmd` cache if that matches the existing DFlash attention call
sites better. If using a contiguous cache, define two pools:

```text
sliding: {.layer, .kv, .k, .hkv, .hd}
full:    {.layer, .kv, .k, .hkv, .hd}
```

The sliding pool uses:

```text
num_key_value_heads
head_dim
```

The full pool uses:

```text
num_global_key_value_heads
global_head_dim
```

The cache update path must select the correct pool per layer and use
`physical_cache_layer[i]`, not the logical layer index.

## Gemma 4 Attention

Implement Gemma 4 attention from `llmd/models/gemma4_text.zig`, adapted to the
non-server DFlash execution model.

For each layer:

1. Determine `num_kv_heads` and `head_dim` from layer type:
   - full: `num_global_key_value_heads`, `global_head_dim`;
   - sliding: `num_key_value_heads`, `head_dim`.
2. Compute `num_head_groups = num_attention_heads / num_kv_heads`.
3. Project Q as `{.hkv, .hg, .hd}`.
4. Project K as `{.hkv, .hd}`.
5. Project V as `{.hkv, .hd}`, or reuse K for full attention when
   `attention_k_eq_v` is enabled and `v_proj` is absent.
6. Apply Q, K, and V norms.
7. Apply per-layer RoPE:
   - full attention uses full RoPE parameters;
   - sliding attention uses sliding RoPE parameters.
8. Update the correct KV cache pool only if `updates_cache[i]` is true.
9. Run attention:
   - full layers use unbounded causal attention;
   - sliding layers must restrict to `sliding_window`.
10. Merge `{.hkv, .hg, .hd}` back into `.d`.
11. Apply `o_proj`.

If the first implementation uses `zml.nn.sdpa`, add the sliding-window mask
explicitly. If it uses `zml.attention.attention` or `zml.attention.paged_attention`,
verify that the backend can represent Gemma 4's mixed full/sliding layouts.

## Target Hidden Extraction

The Llama reference collects selected layer outputs in `Llama.forward`:

```text
for each layer:
    hidden = layer.forward(...)
    if layer id is in target_layer_ids:
        append hidden
target_hidden = concatenate(selected, .d)
```

Implement the same behavior in the Gemma 4 text forward path. The selected
hidden states should be taken after the full Gemma 4 decoder layer output,
including `layer_scalar`, because that is what downstream layers see.

The target hidden tensor shape for compilation is:

```text
{ .s = hidden_len, .d = target_layer_ids.len * config.hidden_size }
```

Use the target embedding dtype for target hidden buffers, as the Llama reference
does.

## Target Prefill And Verify

Create `gemma4/gemma4_inference.zig`, analogous to
`llama/llama_inference.zig`.

It should provide:

```text
TargetLayers
TargetAttention or Gemma4TargetAttention
targetHiddenTensor
compileTargetPrefill
compileTargetVerify
targetPrefill
targetVerify
padTargetHidden
```

`targetPrefill` should:

- slice prompt tokens to the real prompt length;
- run Gemma 4 target forward from `token_index`;
- return padded target hidden;
- sample the last target token;
- return updated Gemma 4 KV cache and RNG.

`targetVerify` should:

- run Gemma 4 target forward over the proposed block;
- compute target logits;
- verify proposed draft tokens using the same acceptance logic as
  `llama_model.zig`;
- return padded target hidden, valid draft token count, correction token,
  updated Gemma 4 KV cache, and RNG.

The existing `verifyDraftTokens` algorithm can be reused nearly verbatim.
Make sure Gemma 4 final logit softcapping has already been applied before
verification probabilities are computed.

## Tokenization And Prompt Formatting

Do not reuse the Llama hardcoded chat wrapper:

```text
<|start_header_id|>user...
```

Gemma 4 should use the model repository's chat template, matching `llmd`:

```text
chat_template.jinja
tokenizer_config.json -> chat_template
chat_template.json -> chat_template
```

The implementation can either:

- port `llmd/chat_template.zig` into `examples/dflash_gemma4`, or
- if the `zml` branch already has equivalent chat-template support by the time
  code is written, reuse that shared implementation.

For an initial CLI with only `--prompt`, render one user message and request an
assistant continuation. The resulting token IDs should feed the same DFlash
prefill path.

Gemma 4 tokenizer loading should still read `tokenizer.json`, as `llmd/main.zig`
does.

## Main Loop Changes

Start from `examples/dflash/main.zig` and make these target substitutions:

```text
llama_inference -> gemma4_inference
llama           -> gemma4
llama.Config    -> gemma4.Config
llama.Model     -> gemma4.Model or Gemma4TextModel wrapper
llama.KvCache   -> gemma4.Gemma4Cache
llama.Buffers   -> gemma4.Buffers
llama.SamplingConfig -> Gemma 4 sampling config or shared SamplingConfig
```

Areas that need explicit changes:

- `Project.parsed_target_config`;
- tokenizer and chat-template loading;
- `tokenizePrompt`;
- `ModelsAndCaches.target_kv_cache`;
- target attention parameters;
- target cache buffer initialization and deinit;
- target cache replacement after prefill and verify;
- EOS checks;
- target hidden tensor construction.

Keep the DFlash draft cache and draft model types unchanged.

## Attention Backend Choice

The Llama DFlash branch uses `zml.attention.attention` metadata and parameters
for a uniform target attention layout.

Gemma 4 has mixed layouts. The implementation should choose one of two paths:

1. Simple path:
   - implement target attention with `zml.nn.sdpa`;
   - use explicit causal and sliding masks;
   - avoid backend-specific metadata for the first working version.

2. Backend path:
   - introduce a Gemma 4 attention bundle with full and sliding parameter sets;
   - mirror `llmd/models.zig` `AttentionParameters.gemma4`;
   - pass the correct parameter set per layer.

The simple path is likely easier for the first correctness pass. The backend path
may be needed later for performance.

## BUILD Target

Add `examples/dflash_gemma4/BUILD.bazel` with a runnable binary analogous to:

```text
//examples/dflash_gemma4:dflash_gemma4
```

The target should include the new Gemma 4 files and depend on `//zml`.

Avoid modifying unrelated examples.

## Smoke Run Notes

Document actual model paths after they are known. Use `examples/dflash/model.md`
on branch `tristan/dflash` as the template.

Expected command shape:

```bash
CUDA_VISIBLE_DEVICES=1 bazel run --@zml//platforms:cuda=true //examples/dflash_gemma4:dflash_gemma4 -- \
  --model=/path/to/gemma4-dflash-model \
  --target-model=/path/to/gemma4-target-model \
  --prompt="Give me a detailed account of the history of the Richelieu-Drouot part of Paris." \
  --max-seq-len=4096
```

## Verification Plan

After implementation, verify in this order:

1. `bazel build //examples/dflash_gemma4:dflash_gemma4`
2. A tiny prompt with `--max-seq-len` close to prompt length.
3. A deterministic greedy run with `--temperature=0`.
4. A nonzero temperature run to exercise rejection sampling.
5. Check that accepted tokens plus correction tokens stream without tokenizer
   decode errors.
6. Check that Gemma 4 EOS handling stops on configured EOS and the hardcoded
   Gemma 4 tool/eot tokens.
7. Compare target-only greedy output against the `llmd` Gemma 4 server or a
   small standalone Gemma 4 target path for the same prompt, if available.

## Open Questions

- What exact Gemma 4 DFlash checkpoint path and target checkpoint path should be
  documented in `model.md`?
- Does the Gemma 4 DFlash checkpoint use the same DFlash tensor names as
  `examples/dflash/dflash_model.zig`?
- Is a contiguous cache acceptable for Gemma 4 target verification, or should the
  implementation immediately port `llmd`'s paged cache and attention scheduler?
- Should this example support top-level multimodal `Gemma4LM`, or only
  text-only `Gemma4TextLM` for the first pass?
- Should chat-template rendering be copied locally, shared from a common module,
  or simplified temporarily for a known Gemma 4 tokenizer?
