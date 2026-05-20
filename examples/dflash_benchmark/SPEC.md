# ZML DFlash Dataset Benchmark Spec

## Goal

Build a ZML example binary that benchmarks the current `examples/dflash` speculative decoder against a target-model-only autoregressive baseline on a random subset of prompts from a command-line-selected dataset.

The benchmark should mirror the reporting style of `third_party/tpu-spec-decode`: per-sample progress for baseline and DFlash, then an aggregate comparison containing TPOT, TPS, speedup, tau/acceptance length, acceptance rates, acceptance histogram, and basic output-quality comparison.

## Example Shape

Create a new Bazel package:

- `examples/dflash_benchmark/BUILD.bazel`
- `examples/dflash_benchmark/main.zig`
- `examples/dflash_benchmark/datasets.zig`
- `examples/dflash_benchmark/stats.zig`
- `examples/dflash_benchmark/SPEC.md`

The implementation should reuse the existing DFlash model code rather than cloning it:

- depend on `//examples/dflash:dflash_model`
- reuse or extract the target LLaMA inference/model helpers from `examples/dflash/llama`
- extract reusable pieces from `examples/dflash/main.zig` only when needed, keeping the existing smoke-test binary intact

## Command Line

Initial CLI:

```text
bazel run //examples/dflash_benchmark -- \
  --model=<dflash model repo or hf://...> \
  --target-model=<target model repo or hf://...> \
  --dataset=<math500|sharegpt|alpaca|swe_bench_lite> \
  --split=<test|dev|train> \
  --samples=<n> \
  --seed=<u64> \
  --max-new-tokens=<n> \
  --max-prompt-tokens=<n> \
  --temperature=<f32> \
  --output-json=<optional path>
```

Defaults:

- `--split` should default to the natural split for each dataset (`test` for MATH-500 and SWE-bench Lite, `train` for Alpaca, only split for ShareGPT).
- `--samples` should be required.
- `--seed=0`, `--max-new-tokens=256`, `--temperature=0.0`.
- `--max-prompt-tokens` should default to the DFlash prefill length supported by the current model path. For the current smoke implementation this is 64, so overlong prompts must be skipped or truncated by an explicit flag.

## Bazel Dataset Wiring

The datasets visible in the local `../data` mirror are Hugging Face datasets. They should be plugged into Bazel as external repos so the benchmark does not rely on ad hoc relative paths.

Add a dataset repository mechanism, preferably by extending the existing `bazel/huggingface.bzl` repository rule pattern or adding `bazel/huggingface_dataset.bzl`, then declare repos in `MODULE.bazel`:

- `@dataset_math500`
  - source: `HuggingFaceH4/MATH-500`
  - include: `test.jsonl`, `README.md`, `eval.yaml`
- `@dataset_sharegpt`
  - source: ShareGPT dataset matching `data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json`
  - include: `ShareGPT_V3_unfiltered_cleaned_split.json`
- `@dataset_alpaca`
  - source: `tatsu-lab/alpaca`
  - include: parquet train shard(s)
- `@dataset_swe_bench_lite`
  - source: `princeton-nlp/SWE-bench_Lite`
  - include: `data/test-*.parquet`, `data/dev-*.parquet`, `README.md`

Each repo should expose a `filegroup(name = "files", ...)` plus stable file targets per split where useful:

- `@dataset_math500//:test_jsonl`
- `@dataset_sharegpt//:json`
- `@dataset_alpaca//:train_parquet`
- `@dataset_swe_bench_lite//:test_parquet`
- `@dataset_swe_bench_lite//:dev_parquet`

`examples/dflash_benchmark:benchmark` should list these targets in its `data` attribute so Bazel runfiles contain every supported source.

## Dataset Readers

`datasets.zig` should normalize all datasets into:

```zig
const Sample = struct {
    id: []const u8,
    prompt: []const u8,
    source_dataset: Dataset,
    source_split: []const u8,
};
```

Dataset-specific prompt extraction:

- `math500`: read JSONL, format `{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.`.
- `sharegpt`: read JSON array, use the first human/user turn as the prompt; skip malformed or empty conversations.
- `alpaca`: read parquet train shards, format `instruction` plus optional `Input:\n{input}` exactly like `tpu-spec-decode`.
- `swe_bench_lite`: read parquet split shard, format `Problem Statement:\n{problem_statement}\nPlease fix the issue described above.`.

File handling requirements:

- JSONL and JSON should be parsed directly in Zig using `std.json`.
- Parquet should be handled deliberately, not by string scanning. If native Zig parquet support is not already present, add a Bazel conversion step that converts parquet shards to normalized JSONL at repository setup or as a small tool target. The benchmark binary should consume the normalized JSONL runfile path.
- The loader should report counts for total rows, valid prompts, skipped malformed rows, skipped overlong prompts, and selected rows.

## Random Subset Selection

Selection should be deterministic:

- Load valid prompt metadata.
- Shuffle indices with `std.Random.DefaultPrng.init(seed)`.
- Select the first `--samples` valid prompts after filtering.
- If fewer than `--samples` prompts remain, fail with a clear error unless a later `--allow-fewer-samples` flag is added.

Filtering should happen before selection:

- empty prompt
- prompt token count exceeding supported prefill length or `--max-prompt-tokens`
- prompt that tokenizes to zero tokens

## Benchmark Methods

Run both methods for each selected sample:

1. `baseline`
   - Prefill the target model on the prompt.
   - Decode one token at a time with the target model and its KV cache.
   - Use the same tokenizer, sampling mode, `max_new_tokens`, EOS handling, and RNG seed family as DFlash.

2. `dflash`
   - Reuse the existing DFlash loop from `examples/dflash/main.zig`.
   - Return generated token ids and acceptance metadata instead of only streaming text.

The two methods should be run on the same prompt back to back. Initial implementation can run baseline first and then DFlash, with a warmup sample option added if compile or first-run effects dominate measurements.

## Stats And Output

Per sample, print:

```text
[3/16] dataset=math500 id=<id> prompt_tokens=57
  Baseline: 256 tokens, TPOT=12.34ms, TPS=81.0
  DFlash:   256 tokens, TPOT=5.67ms, TPS=176.4, tau=8.42
  Quality:  MATCH first 256 output tokens
```

Aggregate text report:

```text
============================================================
RESULTS
============================================================
Dataset:        math500/test
Samples:        16
Baseline TPOT:  12.34 ms (81.0 TPS)
DFlash TPOT:    5.67 ms (176.4 TPS)
Speedup:        2.18x
Tau:            8.42

Per-position acceptance rate:
  pos  1: 0.913 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  pos  2: 0.802 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Acceptance length histogram: [0.031, 0.094, ...]

Output quality: 14/16 samples match baseline exactly
```

Also keep the existing `examples/dflash/main.zig` YAML-like `decode_summary` fields available for single-run debugging, but the dataset benchmark should lead with the `tpu-spec-decode` comparison table/report.

Optional JSON output should include:

- config: model paths, dataset, split, sample count, seed, max tokens, temperature
- per-sample metrics for baseline and DFlash
- generated text for each method
- quality comparison metadata
- summary metrics: TPOT/TPS, speedup, tau, acceptance rates, histogram

## Implementation Plan

1. Create the Bazel package and wire a build-only placeholder.
2. Add Bazel-managed dataset repos and runfile targets for the four local-mirrored datasets.
3. Implement dataset normalization for JSONL/JSON datasets.
4. Add parquet normalization through a structured converter if no native Zig parquet reader exists.
5. Refactor DFlash smoke code into reusable project/model/session helpers.
6. Implement target-only baseline decode using the same target model and tokenizer.
7. Implement deterministic prompt filtering and random sampling.
8. Implement stats aggregation and `tpu-spec-decode`-style text/JSON output.
9. Add build tests for the new Bazel target and small parser tests using tiny fixture files.
10. Run `bazel build //examples/dflash_benchmark:benchmark` and at least one tiny dataset smoke run.

## Open Decisions

- Confirm whether parquet conversion should happen during repository setup, as a genrule/tool target, or inside the benchmark binary via a dependency.
- Confirm whether overlong prompts should be skipped by default or whether a `--truncate-prompts` flag is acceptable.
- Confirm whether output quality should require exact token match or only report exact match plus extracted-answer match for math datasets.
