# Dataset Conversion Helper

The DFlash benchmark reads JSONL files directly. Some Hugging Face datasets used
for benchmark prompts download as parquet shards, so they need a one-time
conversion before they can be passed through `--dataset-path`.

This folder keeps that conversion self-contained and separate from the benchmark
binary. It currently supports:

- `princeton-nlp/SWE-bench_Lite`
- `tatsu-lab/alpaca`

## Setup

From the repository root:

```bash
uv venv examples/dflash_benchmark/python_tools/.venv
uv pip install --python examples/dflash_benchmark/python_tools/.venv/bin/python -r examples/dflash_benchmark/python_tools/requirements.txt
```

## Convert Downloads

The commands below assume the datasets were downloaded to `~/data` with
`bazel run //tools/hf -- download ... --local-dir ...`.

```bash
examples/dflash_benchmark/python_tools/.venv/bin/python examples/dflash_benchmark/python_tools/convert_datasets.py swe_bench_lite
examples/dflash_benchmark/python_tools/.venv/bin/python examples/dflash_benchmark/python_tools/convert_datasets.py alpaca
```

Or convert both:

```bash
examples/dflash_benchmark/python_tools/.venv/bin/python examples/dflash_benchmark/python_tools/convert_datasets.py all
```

Expected outputs:

```text
~/data/SWE-bench_Lite/test.jsonl
~/data/alpaca/train.jsonl
```

Use `--data-root=<path>` if the datasets are not under `~/data`, and
`--overwrite` to regenerate existing JSONL files.
