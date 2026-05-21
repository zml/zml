# DFlash Tau Improvement Research

## Goal

Improve DFlash acceptance statistics in `examples/dflash_benchmark`, prioritizing higher tau on `math500.jsonl` without breaking correctness of speculative verification.

Primary metric for this investigation:

- `Accepted tau`: average accepted draft tokens per verification step.
- `Tau`: average committed tokens per verification step, equal to accepted tau + 1 in the dry-run report.

Current baseline from `//examples/dflash_benchmark:dry_run` on `9960x-5090x2`, `CUDA_VISIBLE_DEVICES=0`, `math500.jsonl`, `--samples=25`, `--max-new-tokens=1024`:

```text
DFlash TPOT:    4.59 ms (217.8 TPS)
Tau:            3.22 committed tokens/step
Accepted tau:   2.22 accepted draft tokens/step
Steps:          3570
EOS samples:    25/25
```

Histogram:

```text
0: 0.280
1: 0.233
2: 0.153
3: 0.097
4: 0.065
5: 0.053
6: 0.043
7: 0.025
8: 0.021
9: 0.029
10: 0.000
```

## Research Directions

1. Drafter input/context alignment
   - Check whether draft cache indices, active context length, and target hidden slices are aligned with the target verification step.
   - Look for off-by-one or stale hidden-state issues that would reduce later-position acceptance.

2. Sampling and verification semantics
   - Check greedy vs non-greedy behavior and whether draft token generation uses the same temperature/top-k assumptions as target verification.
   - Confirm that acceptance statistics measure accepted draft tokens rather than committed correction tokens.

3. DFlash model architecture/runtime choices
   - Inspect `dflash_model.zig` and inference path for normalization, target layer selection, logits projection, KV-cache updates, and mask token handling.
   - Identify low-risk toggles or alternative inputs that might improve acceptance.

4. Benchmark harness and measurement
   - Ensure dry-run and comparison benchmark use the same sample selection, max token behavior, and acceptance aggregation.
   - Add any needed instrumentation before changing behavior.

5. Candidate implementation tests
   - Each test below must be recorded before launch and updated after completion.

## Test Ledger

| ID | Branch/worktree | Hypothesis | Change | Dataset/settings | GPU | Status | Result |
|---|---|---|---|---|---|---|---|
| T0 | `tristan/dflash` | Baseline dry-run tau for comparison | none | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | DFlash TPOT 4.64 ms, 215.6 TPS, committed tau 3.21, accepted tau 2.21, steps 1833, EOS 15/15. Histogram: 0=.285, 1=.224, 2=.159, 3=.105, 4=.054, 5=.054, 6=.043, 7=.024, 8=.022, 9=.029, 10=.000. |
| T1 | TBD | Preserve config order for target hidden concatenation | Capture selected target hiddens in `target_layer_ids` order instead of execution order | `math500.jsonl`, samples=15, max-new=1024 | TBD | planned | |
| T2 | `/tmp/zml_tau_shift` | DFlash proposal/logit alignment is shifted | Verify drafter positions `0..block-1` against target positions instead of current `1..block` proposal slice | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Bad. DFlash TPOT 15.11 ms, 66.2 TPS, committed tau 1.00, accepted tau 0.00, steps 7213, EOS 14/15. Histogram all at 0. Current proposal/logit alignment is correct. |
| T3 | `/tmp/zml_tau_cache_pack` | Chunked prefill creates holes in physical DFlash KV cache for prompts >64 tokens | Pack retained prompt window at draft cache slot 0 while preserving absolute RoPE positions | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Slight tau win, TPS loss. DFlash TPOT 4.74 ms, 211.0 TPS, committed tau 3.27, accepted tau 2.27, steps 1885, EOS 15/15. Baseline T0 was TPOT 4.64 ms, 215.6 TPS, committed tau 3.21, accepted tau 2.21. |
| T4 | `/tmp/zml_tau_tail_window` | First DFlash step should use a full tail prefill window, not only prompt_len % 64 | Prefill prefix normally, then run final prefill over last min(prompt_len,64) prompt tokens and set draft base to tail start | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Strong positive. DFlash TPOT 4.46 ms, 224.1 TPS, committed tau 3.46, accepted tau 2.46, steps 2036, EOS 15/15. Histogram: 0=.258, 1=.213, 2=.159, 3=.110, 4=.065, 5=.054, 6=.038, 7=.032, 8=.026, 9=.046, 10=.000. |
| T5 | `/tmp/zml_tau_tail_pack` | Tail prefill and physical cache packing may need to be applied together | T4 full tail prefill plus T3 separation of physical cache index from absolute position base | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Not better than T4. DFlash TPOT 4.59 ms, 217.9 TPS, committed tau 3.29, accepted tau 2.29, steps 1845, EOS 15/15. Histogram: 0=.266, 1=.229, 2=.164, 3=.105, 4=.063, 5=.051, 6=.040, 7=.022, 8=.021, 9=.038, 10=.000. |
| T6 | `/home/tristan/zml` / `1d710762` | Confirm committed tail-window fix on a larger sample count | T4 fix on branch `tristan/dflash` | `math500.jsonl`, samples=25, max-new=1024 | 0 | done | Confirmed. DFlash TPOT 4.13 ms, 241.9 TPS, committed tau 3.50, accepted tau 2.50, steps 3440, EOS 23/25. Histogram: 0=.253, 1=.219, 2=.152, 3=.102, 4=.068, 5=.054, 6=.046, 7=.028, 8=.026, 9=.050, 10=.000. Old 25-sample baseline was TPOT 4.59 ms, 217.8 TPS, committed tau 3.22, accepted tau 2.22, steps 3570, EOS 25/25. |
| T7 | `/home/tristan/zml` / `1d710762` | Check whether tail-window fix also helps stories | T4 fix on branch `tristan/dflash` | `generic_jsonl`, `stories.jsonl`, samples=25, max-new=1024 | 0 | done | First attempt with `--dataset=stories` failed because `stories` is not a dataset enum. Rerun used `--dataset=generic_jsonl --dataset-path=stories.jsonl`. DFlash TPOT 5.96 ms, 167.7 TPS, committed tau 2.47, accepted tau 1.47, steps 7142, EOS 24/25. Histogram: 0=.352, 1=.268, 2=.162, 3=.105, 4=.055, 5=.029, 6=.017, 7=.007, 8=.003, 9=.002, 10=.000. |

## Research Findings

### Measurement and Harness

- Dry-run and baseline-comparison use the same DFlash execution path.
- Dry-run reports both:
  - `Tau`: committed tokens per verification step.
  - `Accepted tau`: accepted draft tokens per verification step.
- Existing comparison output prints committed tau only; raw accepted tau is inferable from histograms but not shown directly.
- Useful instrumentation before deeper behavior changes:
  - raw accepted tau;
  - effective accepted tau (`committed_lengths - 1`);
  - step count;
  - generated token count;
  - count of capped/final steps where raw accepted tokens exceed usable accepted tokens.

### Drafter Runtime Candidate Findings

- Target hidden capture may be order/index sensitive. Current code captures configured target layers in model execution order. If checkpoint training concatenated hiddens in config order or used HF hidden-state indexing conventions, the DFlash conditioning vector is permuted/off-by-one.
- Draft-logit/proposal alignment is a high-signal hypothesis. Current verification ignores drafter slot 0 and verifies slots `1..block_size` against target logits `0..block_size-1`. If the DFlash head was trained with next-token semantics, shifting proposals/logits may improve tau.
- Mask-token embedding source should be validated. Runtime fills future slots with `dflash_config.mask_token_id` and embeds those through the target LLaMA embedding table.
- Target-hidden normalization could mismatch training: runtime applies `fc` then `hidden_norm`, not per-layer normalization before concatenation.
- DFlash attention masking is bidirectional inside the proposal window. A causal/prefix mask variant is worth an ablation if lower-risk tests do not improve tau.
- High-priority cache/indexing hypothesis: chunked benchmark prefill keeps only the final 64-token target-hidden chunk. Current first DFlash step sets `draft_cache_base = prompt_len - last_prefill_chunk_len` and passes that as both physical DFlash KV cache index and absolute RoPE base. DFlash attention accepts every key `< active_context_len + block_size`, so long prompts may attend over zero-initialized draft KV positions before the retained final chunk. This could directly lower tau on prompts longer than 64 tokens.

### Sampling Semantics

- For default `--temperature=0`, acceptance is exactly draft proposal token equals target argmax token. Loosening comparison would change target greedy semantics and is not valid.
- Higher tau in greedy mode should come from better draft logits/context/model alignment, not verifier relaxation.
- Non-greedy tuning may be possible later by changing the drafter proposal distribution `q` and verifier probability calculation together, but it is out of scope for the current greedy math500 dry-run.

### Other Inference Implementations

- No separate TPU inference/speculative server loop was found in this repo.
- `examples/dflash`, `examples/dflash_gemma4`, and `examples/dflash_benchmark` share the same core DFlash loop shape.
- The benchmark differs from the older single-prompt `examples/dflash` by supporting chunked prefill and setting first `draft_cache_base = prompt_len - last_prefill_chunk_len`.
- Host overhead can be reduced by reusing scalar buffers/args/results like other session implementations, but that affects TPOT more than algorithmic tau.

### Tested Improvement

- T4 confirms the highest-value fix so far: when the prompt is longer than the 64-token prefill block and the final chunk would be short, re-run the final prefill over the full 64-token tail window. This gives DFlash a full recent target-hidden context for the first draft step.
- On `math500.jsonl`, 15 samples, max-new 1024, accepted tau improved from 2.21 to 2.46 and committed tau improved from 3.21 to 3.46.
- On `math500.jsonl`, 25 samples, max-new 1024, accepted tau improved from 2.22 to 2.50 and committed tau improved from 3.22 to 3.50. TPS improved from 217.8 to 241.9.
- Separately packing the retained window into physical draft-cache slot 0 was not additive in T5. It reduced accepted tau to 2.29, so the current local implementation should use the T4 tail-window policy without the T3/T5 physical-cache packing change.
- On `stories.jsonl`, 25 samples, the committed tail-window path reported accepted tau 1.47 and committed tau 2.47. The selected story prompts are only about 21-24 tokens, so this test does not exercise the long-prompt tail-window fix.

## Notes

- Main agent owns this file and records every proposed test before running it.
- Prefer `//examples/dflash_benchmark:dry_run` for fast DFlash-only iteration.
- Use `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` for two concurrent remote tests when possible.

## Test Process Notes

- Remote test host: `9960x-5090x2`.
- Remote repo for mainline runs: `/home/tristan/zml`.
- Remote experiment worktrees: `/tmp/zml_tau_*`, one hypothesis per worktree/branch.
- Use Bazel to launch Zig binaries; do not run copied binaries directly:

```bash
CUDA_VISIBLE_DEVICES=<gpu> bazel run --config=silent \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/dflash_benchmark:dry_run -- \
  --model=/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ \
  --target-model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --dataset=math500 \
  --dataset-path=math500.jsonl \
  --samples=15 \
  --max-new-tokens=1024
```

- Remote worktrees created under `/tmp/zml_tau_*` get separate Bazel output bases by default. That causes slow rebuilds and no GPU activity for several minutes.
- If `nvtop`/`nvidia-smi` shows no benchmark on GPU immediately after launch, first check whether Bazel is still building or waiting on an output-base lock. No GPU process appears until Bazel has finished analysis/build and started the benchmark binary.
- The warm main output base is:

```text
/home/tristan/.cache/bazel/_bazel_tristan/1dea1b3906af77bdf8a87df9b2f59d20
```

- To reuse the warm cache from a remote worktree, launch with:

```bash
bazel --output_base=/home/tristan/.cache/bazel/_bazel_tristan/1dea1b3906af77bdf8a87df9b2f59d20 run ...
```

- A shared Bazel output base serializes the Bazel build/launch phase because the server/output-base lock is exclusive. Once Bazel launches the benchmark binary, the binary runs independently, and the next queued `bazel run` can launch. This gives serialized build/launch but concurrent GPU execution.
- This is how the successful two-GPU concurrent run was achieved: start two `bazel --output_base=... run ...` commands in separate remote shells with different `CUDA_VISIBLE_DEVICES` values. The second command waits for the shared Bazel launch lock; after the first benchmark process is detached/running on its GPU, the second Bazel invocation proceeds and starts the second benchmark on the other GPU.
- Confirm actual GPU execution with:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv,noheader
```

- Map benchmark PIDs to test variants with:

```bash
pgrep -af "dry_run|examples/dflash_benchmark:dry_run|bazel --output_base"
```

- When collecting results from concurrent tests, include in the ledger:
  - worktree and branch;
  - GPU id;
  - exact dataset/sample/max-new settings;
  - whether the run used the warm shared output base;
  - benchmark PID if still running;
  - final DFlash TPOT/TPS, committed tau, accepted tau, step count, EOS count, and valid-draft-token histogram.
- For files in `examples/dflash_benchmark/data` that are not built-in dataset enum names, use `--dataset=generic_jsonl --dataset-path=<file>.jsonl`.

## Campaign 2: Post Tail-Window Tau Search

### Starting Point

The chunked-prefill tail-window issue is fixed on branch `tristan/dflash` at commit `1d710762`, with the research log updated at `6608224d`.

Current confirmed `math500.jsonl` dry-run baseline for this campaign:

```text
Samples:        25
DFlash TPOT:    4.13 ms (241.9 TPS)
Tau:            3.50 committed tokens/step
Accepted tau:   2.50 accepted draft tokens/step
Steps:          3440
EOS samples:    23/25
```

Current confirmed `stories.jsonl` dry-run baseline for this campaign:

```text
Samples:        25
DFlash TPOT:    5.96 ms (167.7 TPS)
Tau:            2.47 committed tokens/step
Accepted tau:   1.47 accepted draft tokens/step
Steps:          7142
EOS samples:    24/25
```

### Campaign 2 Plan

Directions to investigate:

1. Compare ZML DFlash loop against vLLM and other speculative decoding implementations for remaining semantic differences.
2. Revisit model-conditioning details: target layer order/indexing, target hidden normalization, mask-token embedding, RoPE position base, and attention mask shape.
3. Improve measurement to expose per-position and per-sample acceptance behavior before changing semantics.
4. Run bounded remote ablations on `9960x-5090x2`, two at a time on `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.

### Campaign 2 Test Ledger

| ID | Branch/worktree | Hypothesis | Change | Dataset/settings | GPU | Status | Result |
|---|---|---|---|---|---|---|---|
| C2-T0 | `/home/tristan/zml` / `6608224d` | Post-tail-window baseline | none | `math500.jsonl`, samples=25, max-new=1024 | 0 | done | DFlash TPOT 4.13 ms, 241.9 TPS, committed tau 3.50, accepted tau 2.50, steps 3440, EOS 23/25. |
| C2-T1 | TBD | Target hidden concatenation may need config order instead of execution order | Gather selected target layers in `dflash_config.target_layer_ids` order | `math500.jsonl`, samples=15, max-new=1024 | TBD | planned | |
| C2-T2 | `/tmp/zml_tau_c2_hf_layers` | Target layer IDs may use HF hidden-state convention (`1` means after layer 0) | Capture layer `i` when configured id equals `i + 1` | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Negative. DFlash TPOT 5.14 ms, 194.6 TPS, committed tau 2.94, accepted tau 1.94, steps 2082, EOS 15/15. Histogram: 0=.311, 1=.230, 2=.154, 3=.111, 4=.068, 5=.049, 6=.032, 7=.016, 8=.016, 9=.012, 10=.000. Config ids should be interpreted as current zero-based post-layer indices. |
| C2-T3 | `/tmp/zml_tau_c2_f32_logits` | Greedy argmax may be sensitive to low-precision lm-head logits | Compute target and draft lm-head logits in f32 for dry-run | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Negative. First attempt failed because target verify expected bf16 draft logits; rerun changed verify draft-logit input to f32. DFlash TPOT 4.47 ms, 223.7 TPS, committed tau 3.30, accepted tau 2.30, steps 1869, EOS 15/15. Histogram: 0=.256, 1=.235, 2=.159, 3=.109, 4=.066, 5=.057, 6=.035, 7=.026, 8=.027, 9=.028, 10=.000. Keep existing logits dtype path. |
| C2-T4 | `/tmp/zml_tau_c2_causal_proposal` | Proposal queries may need causal attention over proposal slots instead of seeing all masked future slots | Limit each proposal query to context plus proposal keys up to that query | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Negative. DFlash TPOT 4.84 ms, 206.8 TPS, committed tau 3.38, accepted tau 2.38, steps 1957, EOS 15/15. Histogram: 0=.248, 1=.229, 2=.162, 3=.105, 4=.064, 5=.059, 6=.049, 7=.027, 8=.024, 9=.032, 10=.000. Keep non-causal proposal attention. |
| C2-T5 | `/tmp/zml_tau_c2_zero_mask_embedding` | Target-model embedding for future mask tokens may be the wrong noise input | Keep slot 0 token embedding, zero future mask-token embeddings before DFlash | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Tentative positive. First attempt failed on i32/u32 compare; rerun fixed index dtype. DFlash TPOT 4.43 ms, 225.5 TPS, committed tau 3.54, accepted tau 2.54, steps 1954, EOS 13/15. Histogram: 0=.250, 1=.211, 2=.158, 3=.109, 4=.066, 5=.053, 6=.037, 7=.038, 8=.026, 9=.052, 10=.000. Needs 25-sample confirmation because EOS dropped. |
| C2-T6 | `/tmp/zml_tau_c2_zero_mask_embedding` | Confirm zero-mask embedding on larger math500 sample | Same as C2-T5 | `math500.jsonl`, samples=25, max-new=1024 | 1 | done | Negative/inconclusive. DFlash TPOT 4.38 ms, 228.4 TPS, committed tau 3.45, accepted tau 2.45, steps 3269, EOS 23/25. Histogram: 0=.255, 1=.222, 2=.148, 3=.108, 4=.074, 5=.053, 6=.045, 7=.027, 8=.024, 9=.045, 10=.000. This did not beat C2-T0 accepted tau 2.50, so do not merge zero-mask embedding. |
| C2-T7 | `/tmp/zml_tau_c2_window_mask` | DFlash attention may be attending stale/zero KV slots before the active tail window | Add a lower-bound `k >= cache_index` to the DFlash attention mask | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Negative. DFlash TPOT 11.30 ms, 88.5 TPS, committed tau 1.34, accepted tau 0.34, steps 4382, EOS 15/15. Histogram: 0=.717, 1=.232, 2=.045, 3=.005, 4=.001, 5=.000, 6=.000, 7=.000, 8=.000, 9=.000, 10=.000. The drafter relies on the existing bucketed cache layout and must not lower-bound attention by absolute `cache_index`. |
| C2-T8 | `/tmp/zml_tau_c2_norm_target_hidden` | Draft checkpoint may expect normalized selected target hidden streams | Store `self.norm.forward(hidden)` for captured target layers instead of raw post-layer residuals | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Negative. DFlash TPOT 7.46 ms, 134.0 TPS, committed tau 2.02, accepted tau 1.02, steps 3220, EOS 15/15. Histogram: 0=.470, 1=.271, 2=.125, 3=.076, 4=.035, 5=.016, 6=.006, 7=.002, 8=.000, 9=.000, 10=.000. Captured target hidden streams should stay raw post-layer residuals. |
| C2-T9 | `/tmp/zml_tau_c2_f32_fc` | Target-hidden conditioning projection may lose acceptance from bf16 `fc` matmul | Run DFlash `fc` projection with `linearForwardF32` before `hidden_norm` | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Negative. DFlash TPOT 4.79 ms, 208.9 TPS, committed tau 3.34, accepted tau 2.34, steps 1879, EOS 15/15. Histogram: 0=.251, 1=.233, 2=.163, 3=.107, 4=.068, 5=.055, 6=.043, 7=.021, 8=.026, 9=.033, 10=.000. Keep the existing `fc.forward` precision path. |
| C2-T10 | `/tmp/zml_tau_c2_f32_kproj` | DFlash K projection precision may matter for cross-attention over target hidden context | Use `linearForwardF32` for `k_proj` on target hidden and noise hidden before attention | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Negative. DFlash TPOT 4.46 ms, 224.4 TPS, committed tau 3.42, accepted tau 2.42, steps 2016, EOS 15/15. Histogram: 0=.263, 1=.219, 2=.155, 3=.104, 4=.061, 5=.057, 6=.045, 7=.032, 8=.020, 9=.044, 10=.000. This is closer than T9 but still below the campaign baseline accepted tau 2.50. |
| C2-T11 | `/tmp/zml_tau_c2_double_norm_logits` | Draft hidden may need target final RMSNorm before target lm head | Apply target LLaMA final norm to DFlash hidden before `target_model.logitsForward` | `math500.jsonl`, samples=15, max-new=1024 | 0 | done | Negative. DFlash TPOT 4.69 ms, 213.2 TPS, committed tau 3.40, accepted tau 2.40, steps 1958, EOS 15/15. Histogram: 0=.258, 1=.219, 2=.159, 3=.111, 4=.066, 5=.056, 6=.031, 7=.036, 8=.028, 9=.036, 10=.000. Keep the existing single DFlash final norm before target lm-head projection. |
| C2-T12 | `/tmp/zml_tau_c2_skip_draft_final_norm` | DFlash hidden may already be in target lm-head space before draft final norm | Return raw DFlash hidden from `dflash.Model.forward` instead of applying DFlash final norm | `math500.jsonl`, samples=15, max-new=1024 | 1 | done | Negative. DFlash TPOT 4.84 ms, 206.7 TPS, committed tau 3.18, accepted tau 2.18, steps 1982, EOS 15/15. Histogram: 0=.246, 1=.253, 2=.166, 3=.118, 4=.071, 5=.049, 6=.034, 7=.026, 8=.013, 9=.025, 10=.000. Keep the DFlash final norm in `dflash.Model.forward`. |
| C2-I1 | local `main-dry-run.zig` | Aggregate tau hides where acceptance is failing | Add per-position acceptance, boundary truncation, and EOS source/position reporting to dry-run | build-only | local | done | Build passed with `bazel build --config=silent //examples/dflash_benchmark:dry_run`. Use this instrumentation before more semantic ablations. |
| C2-I2 | local `main-dry-run.zig` | Per-position report included one impossible slot | Report only `block_size - 1` verifiable proposal positions; slot 0 is the anchor/current token | build-only | local | done | Build passed with `bazel build --config=silent //examples/dflash_benchmark:dry_run`. For `block_size=10`, positions p00-p08 are real proposals; p09 was an instrumentation artifact. |

### Campaign 2 Findings

- Pending.
- Online comparison notes:
  - vLLM's `DFlashProposer` asserts `method == "dflash"` and passes hidden states to the draft model. Source: <https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/dflash/>.
  - vLLM's DFlash Qwen3 attention says context K/V are pre-inserted into the KV cache and the forward pass handles only query tokens. Source: <https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/models/qwen3_dflash/>.
  - TensorRT-LLM documents DFlash as target-dependent speculative decoding using hidden states from configured target layers and a configured mask token. Source: <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md>.
  - The z-lab DFlash README documents `num_speculative_tokens: 15` for vLLM and SGLang DFlash examples, while this LLaMA DFlash checkpoint has `block_size = 10`. Source: <https://github.com/z-lab/dflash>.
- The actual LLaMA DFlash draft config on `9960x-5090x2` has `dflash_config.target_layer_ids = [1, 8, 15, 22, 29]`, `mask_token_id = 128002`, and `block_size = 10`. Because the ids are sorted, config-order gathering should be behavior-identical for this checkpoint; HF off-by-one remains worth testing.
- C2-T4 confirms the non-causal proposal-window attention is likely intentional for this checkpoint; causalizing proposal attention reduced accepted tau.
- C2-T2 confirms the LLaMA DFlash `target_layer_ids` are not HF `hidden_states` indices for this checkpoint. They should stay as current zero-based post-layer indices.
- C2-T3 suggests f32 lm-head logits are not a tau improvement for this path.
- C2-T5 initially suggested zeroing future mask-token embeddings might improve tau, but C2-T6 failed to confirm it on the 25-sample baseline set. Keep the target-model embedding path for future mask tokens.
- C2-I1 closes the immediate measurement gap: dry-run now prints per-position acceptance, boundary truncation, and EOS source/position. Use these fields before interpreting subtle changes as model-quality improvements.
- C2-T7 shows the DFlash cache attention mask should not be lower-bounded by absolute `cache_index`; doing so destroys tau. The current bucketed cache merge leaves the active logical context at the start of the bucketed slice, so `k < valid_k` is intentional.
- C2-T8 shows selected target hidden streams should remain raw post-layer residuals. Applying the final target RMSNorm to each selected layer strongly reduces accepted tau.
- C2-T9 and C2-T10 do not support changing conditioning projection precision. F32 `k_proj` was the closer run but still lost accepted tau versus the baseline.
- C2-T11 and C2-T12 rule out the simplest final-normalization alternatives. The current path applies the DFlash final norm once before projecting with the target lm head; adding the target final norm or skipping the DFlash final norm both reduce accepted tau.

### Improvement Track Report

#### Current Acceptance Shape

The instrumented C2-I1 math500 run on `fb1ff349`, 15 samples, reported committed tau 3.33 and accepted tau 2.33. There was no boundary truncation, and all 15 EOS cases came from the correction token rather than an accepted draft token.

The real per-position acceptance rates are positions p00-p08 because the LLaMA DFlash checkpoint has `block_size = 10`, where slot 0 is the anchor/current token and only 9 following positions are verifiable draft proposals:

```text
p00 0.739
p01 0.519
p02 0.350
p03 0.241
p04 0.180
p05 0.125
p06 0.086
p07 0.059
p08 0.035
```

This points to a positional/context-quality problem: the drafter is useful for the first few proposals, then collapses quickly. It is not a max-token boundary artifact.

#### Tested Tracks

1. Tail-window prompt context: positive. The benchmark originally seeded only the final short prefill chunk. Re-running the final prefill over a full 64-token tail improved 25-sample math500 accepted tau from 2.22 to 2.50 and TPS from 217.8 to 241.9.
2. Proposal/logit shift: negative. Verifying positions `0..block-1` instead of `1..block` produced accepted tau 0.00. The current proposal alignment is correct.
3. Physical cache packing with tail window: negative. It was not additive over the tail-window fix.
4. HF layer-id off-by-one: negative. The checkpoint remote code uses `hidden_states[layer_id + 1]`; the current ZML capture after layer `i == layer_id` matches that convention because HF hidden state 0 is embeddings.
5. f32 lm-head logits: negative.
6. Causal proposal attention: negative. DFlash requires non-causal attention inside the synthetic proposal block.
7. Zero future mask-token embeddings: not confirmed. A 15-sample run looked slightly positive but a 25-sample confirmation fell below baseline and had no EOS advantage.
8. Lower-bound attention mask `k >= cache_index`: strongly negative. This removed useful prior DFlash KV context and collapsed accepted tau to 0.34.
9. Normalize selected target hiddens before concatenation: negative. Raw post-layer residuals are correct.
10. f32 `fc` and f32 `k_proj`: negative.
11. Final norm alternatives: negative. Adding target final RMSNorm before target lm-head projection and skipping DFlash final norm both reduced tau.
12. Draft lm-head suspicion: ruled out by checkpoint inspection. The local checkpoint has no `lm_head`, no `embed_tokens`, and no vocab mapping tensors. The remote reference `dflash.py` also uses `target.lm_head(...)`, so the ZML target-head projection is intentional for this checkpoint.

#### Highest-Priority Untested Tracks

1. Full prompt DFlash KV precompute from target hidden states.
   - Clarification: the target verifier KV cache is already filled for the full prompt by chunked target prefill. The unresolved question is the DFlash drafter's own KV cache. DFlash does not directly consume target KV; it consumes target hidden states, projects them through the DFlash `k_proj`/`v_proj` path, and stores those projected context K/V in the DFlash cache.
   - Evidence: The remote reference `dflash.py` runs the first DFlash forward with target hidden for the full prompt and crops the DFlash cache to `start`, leaving prompt-derived DFlash K/V available for later steps. vLLM has an explicit `precompute_and_store_context_kv` path that inserts DFlash context K/V from target hidden states before query-token forward. ZML currently runs target prefill over the full prompt, but the first DFlash step receives only `last_prefill_chunk_len = min(prompt_len, 64)` target hidden rows and therefore only seeds the DFlash KV cache for the prompt tail.
   - Why it matters: C2-T4 showed that giving DFlash a better prompt tail improves tau; C2-T7 showed that older DFlash KV context is useful and should not be masked away. The missing full-prompt DFlash K/V prefix is therefore the largest remaining semantic gap if the implementation is intended to match the reference/vLLM path.
   - Test: implement `precomputeDraftContextKv` or a full-prompt first draft path that fills DFlash K/V for prompt positions `0..prompt_len-1`, potentially by running DFlash context precompute over multiple target-hidden chunks. Then start the first query block at `prompt_len`. Compare math500 samples=15 and 25. Expected signal: later-position acceptance p02-p08 should rise if missing prompt context is the limiter.

2. Canonical Llama 3.1 chat template.
   - Evidence: The checkpoint README's Transformers example uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. On the remote tokenizer this emits a default system message and double-newline header formatting. The Zig path manually builds a shorter template with no system message and single newlines after headers.
   - Why it matters: target and draft still agree with each other, but the drafter was trained/evaluated under the server/HF prompt distribution. Prompt-template drift can lower hidden-state distribution match and tau.
   - Test: add a `--chat-template=llama31_default|minimal` switch or replace the manual tokenizer wrapper with the canonical Llama 3.1 template. Run math500 and stories samples=25. Expected signal: p00/p01 and EOS-source behavior should improve if prompt OOD is hurting the drafter.

3. Prompt-length and first-step diagnostics.
   - Evidence: The current aggregate report hides whether losses are concentrated in first step, long prompts, or steady-state after rejection.
   - Test: add bins by prompt length and step index: first step vs later steps, prompt <=64, 65-128, 129-256, >256. Re-run current baseline before more model edits.
   - Expected signal: if long prompts underperform, full prompt DFlash context prefill becomes even higher priority. If first step is fine but later steps decay, cache crop/update semantics become higher priority.

4. Cache-crop equivalence with reference `dflash.py`.
   - Evidence: Reference code calls `past_key_values_draft.crop(start)` after draft forward and `past_key_values_target.crop(start)` after verification. ZML relies on token-index causal masks and overwrites instead of explicit crop.
   - Test: instrument or emulate a crop by carrying a logical DFlash length and masking `k < logical_len` after each acceptance. Do not lower-bound by `cache_index`; C2-T7 already showed that is wrong.
   - Expected signal: if stale in-window future proposal K/V ever leaks into attention, p02+ should improve. Probability is lower than full prompt prefill because current `valid_k` already excludes most stale future slots.

5. Reference-oracle first-step comparison.
   - Evidence: The checkpoint ships `dflash.py`, whose `spec_generate` is the model owner's reference implementation.
   - Test: for one or two selected math500 prompts, dump first-step block tokens from Python reference and from ZML with the same prompt template. Compare draft token IDs, target posterior IDs, and acceptance length.
   - Expected signal: if first-step draft tokens differ before verification, inspect context length, position_ids, and template. This is a diagnostic track, not a direct optimization.

#### Medium-Priority Untested Tracks

1. Effective speculation depth / shorter block.
   - The checkpoint predicts 9 proposals per step (`block_size=10` including anchor). Late proposal positions are weak. A smaller effective block could improve TPS even if accepted tau drops, because target verify and DFlash compute would shrink. This optimizes speedup, not accepted tau. Needs a `--effective-block-size` experiment.
2. Hybrid n-gram plus DFlash proposer.
   - For stories or repetitive prompts, prompt-lookup drafts can beat model drafts. A lossless hybrid could choose n-gram proposals when long exact prefix matches exist and DFlash otherwise. This is not pure DFlash tau but may maximize accepted tokens and speedup.
3. Tree verification / DDTree-style alternatives.
   - Use DFlash logits to build a small draft tree and verify multiple alternatives in one target pass. This attacks the sharp position decay by keeping likely alternatives when the top-1 draft is wrong. This is high complexity because it needs tree attention and branching KV management.
4. Domain fine-tuning.
   - The LLaMA DFlash checkpoint was trained on UltraChat-200K and ShareGPT with regenerated target responses. Stories and math500 may be distribution-shifted. Fine-tuning on math/story-style target generations could improve tau more than inference tweaks, but it is a training campaign rather than a runtime fix.
5. Non-greedy acceptance experiments.
   - Current tau work uses greedy decoding. For nonzero temperature, draft probability calibration and residual sampling matter. This is separate from the deterministic math500 path and should wait until greedy parity is better understood.

#### Low-Priority Or Ruled-Out Tracks

1. More dtype-only changes are unlikely after f32 logits, f32 `fc`, and f32 `k_proj` all failed.
2. More final-norm variants are unlikely after target-final-norm and skip-DFlash-norm both failed.
3. Causalizing DFlash proposal attention is wrong for this model family. Speculators docs and vLLM both describe non-causal proposal-block attention as required.
4. Reducing attention to only the active window is wrong. Prior prefix K/V matters.
5. Vocab mapping/draft head is not applicable to this specific checkpoint; the safetensors file has no `lm_head`, `embed_tokens`, `d2t`, or `t2d` tensors.

#### External References

- vLLM DFlash proposer uses query tokens as `1 + num_speculative_tokens`, passes hidden states to the draft model, and marks DFlash attention non-causal: <https://docs.vllm.ai/en/latest/api/vllm/v1/spec_decode/dflash/>
- vLLM Qwen3 DFlash precomputes and stores context K/V from target hidden states before the query-token forward: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_dflash.py>
- Speculators DFlash docs describe non-causal proposal-block attention and context hidden conditioning: <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/>
- Speculators training docs require matching target layer ids and define `block-size` as number of predicted tokens per block in that training framework: <https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/train_dflash_online/>
- The LLaMA DFlash checkpoint README recommends vLLM with `num_speculative_tokens=9` and Transformers inference through `tokenizer.apply_chat_template`: `/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/README.md` on `9960x-5090x2`.

## Campaign 3: Full Prompt DFlash Context KV Precompute

### Goal

Test the highest-priority remaining semantic gap: ZML already prefilled the target verifier KV cache for the whole prompt, but the DFlash drafter KV cache only had prompt-tail context K/V. The intended fix is to precompute and store DFlash context K/V for every target prefill chunk using the target hidden states from that chunk.

### Implementation

Files changed:

- `examples/dflash_benchmark/dflash_model.zig`
  - Added `Model.precomputeAndStoreContextKv`.
  - Added `DecoderLayer.precomputeAndStoreContextKv`.
  - Added `DFlashAttention.precomputeAndStoreContextKv`, which projects target hidden through DFlash `k_proj`/`v_proj`, applies K norm and RoPE, and writes the resulting context K/V into the DFlash KV cache at the absolute `cache_index`.
- `examples/dflash_benchmark/main-dry-run.zig`
  - Added a compiled `draft_context_precompute_exe`.
  - During target prefill chunks, optionally calls `runDraftContextPrecompute` after each target chunk and before the final DFlash draft loop.
  - `runDFlash` passes the DFlash KV cache into `runPrefillChunks`, so all prompt chunks seed DFlash context K/V.
  - `runBaseline` passes `null`, so target-only baseline behavior is unchanged.
- `examples/dflash_benchmark/main-baseline-comparison.zig`
  - Mirrored the same DFlash precompute path so the comparison binary does not retain the old tail-only DFlash behavior.

Design note: the first DFlash draft call is intentionally left otherwise unchanged. It still rewrites the active prompt tail and proposal rows. Because the DFlash attention mask is `k < valid_k` with no lower bound, the previously precomputed prefix rows remain visible. This is the smallest patch that tests the tau hypothesis without adding a query-only first draft path.

### Local Verification

```text
zig fmt examples/dflash_benchmark/dflash_model.zig examples/dflash_benchmark/main-dry-run.zig examples/dflash_benchmark/main-baseline-comparison.zig
git diff --check
bazel build --config=silent //examples/dflash_benchmark:dry_run
bazel build --config=silent //examples/dflash_benchmark:benchmark
```

All passed locally.

### Test Ledger

| ID | Branch/commit | Hypothesis | Change | Dataset/settings | GPU | Status | Result |
|---|---|---|---|---|---|---|---|
| C3-T1 | `13e972af` | Full prompt DFlash context K/V improves long-context conditioning and p02-p08 acceptance | Precompute DFlash context K/V for every target prefill chunk | `math500.jsonl`, samples=15, max-new=1024 | 5090 GPU 0 | done | Positive. DFlash TPOT 4.57 ms, 219.0 TPS, committed tau 3.51, accepted tau 2.51, steps 1754, EOS 15/15. Per-position: p00=.762, p01=.558, p02=.391, p03=.269, p04=.196, p05=.140, p06=.090, p07=.062, p08=.039. Compared with C2-I1 15-sample instrumentation, accepted tau improved 2.33 -> 2.51 and p00-p03 improved. |
| C3-T2 | `cc7ece96` | Confirm full-prompt DFlash context precompute on the 25-sample math500 baseline set | Same as C3-T1 | `math500.jsonl`, samples=25, max-new=1024 | 5090 GPU 0 | done | Positive. DFlash TPOT 4.47 ms, 223.7 TPS, committed tau 3.76, accepted tau 2.76, steps 3421, EOS 21/25. Per-position: p00=.763, p01=.558, p02=.420, p03=.311, p04=.239, p05=.178, p06=.126, p07=.096, p08=.069. Compared with C2-T0, accepted tau improved 2.50 -> 2.76 and committed tau improved 3.50 -> 3.76. |
| C3-T3 | `cc7ece96` | Same fix should also help story prompts or expose distribution-specific behavior | Same as C3-T1 | `stories.jsonl` via `generic_jsonl`, samples=25, max-new=1024 | 5090 GPU 1 | done | Neutral. DFlash TPOT 6.19 ms, 161.6 TPS, committed tau 2.48, accepted tau 1.48, steps 7261, EOS 25/25. Per-position: p00=.653, p01=.386, p02=.217, p03=.115, p04=.059, p05=.028, p06=.014. Essentially unchanged from the pre-fix stories baseline accepted tau 1.47; short story prompts do not benefit from full prompt context. |
| C3-T4 | temporary remote patch on `cc7ece96` | vLLM may use `dflash_config.target_layer_ids` as 1-based Llama aux hidden state layer IDs | Capture target hidden when `target_layer_id == i + 1` instead of `target_layer_id == i` | `math500.jsonl`, samples=15, max-new=1024 | 5090 GPU 0 | done | Negative. DFlash TPOT 4.75 ms, 210.6 TPS, committed tau 3.13, accepted tau 2.13, steps 1996, EOS 15/15. Per-position: p00=.736, p01=.500, p02=.335, p03=.220, p04=.152, p05=.091, p06=.051. Keep current ZML/HF-reference interpretation of layer ids. |

### Follow-Up Ideas From Review Agents

1. Canonical Llama 3.1 chat-template parity remains the next most actionable track. Current Zig prompt formatting is shorter than HF `apply_chat_template`, and prompt distribution drift can lower DFlash hidden-state conditioning quality.
2. Add prompt-length and first-step acceptance bins. This will show whether full-context precompute helps only the first step/long prompts or also steady-state drafting.
3. Add cache-crop diagnostics before changing crop semantics. Reference implementations crop DFlash caches logically; ZML currently relies on overwrite plus valid-key masks.
4. Add selected-sample manifests to compare Zig and Python/vLLM on identical rows and prompt token IDs.
5. Consider a top-k/tree diagnostic only after semantic parity work: log whether the target posterior token is often in DFlash top-2/top-4 when top-1 fails.
