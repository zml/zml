# Benchmarking and profiling

The harness has two modes. `benchmark` measures repeated, unprofiled application
invocations. `profile` prepares one case and submits one invocation inside the
named NVTX range `TSL@paged_attention.profile` so Nsight Compute can measure the
individual attention and merge kernels.

## Benchmark

On a CUDA host, list the cases and their zero-based indices:

```bash
bazel run //zml/kernels/paged_attention/cute:benchmark --@zml//platforms:cuda=true -- --list-cases
```

`--help` and `--list-cases` also work in a CPU-only build without a CUDA device.

Run the full suite, or select one case with `--case=0`:

```bash
bazel run //zml/kernels/paged_attention/cute:benchmark --@zml//platforms:cuda=true -- \
  --mode=benchmark --warmup=5 --iterations=101 \
  --output=profiling-data/paged_attention_benchmark
```

The output prefix produces two files:

- `.csv`: one summary row per case, including geometry, GPU count/compute
  capability, compilation time, sample count, and min/p10/median/p90/max/mean
  latency in microseconds. p10 and p90 use nearest-rank percentiles.
- `.json`: the same results and chronological samples, plus timing boundaries,
  dtype, page size, seed, warmup count, and cache assumptions.

The measured interval includes runtime dispatch, output allocation, attention,
merge, and completion waiting. Compilation, upload, warmup, and output destruction
are outside it. Inputs are reused without flushing caches. KV heads in case
descriptions are per device, and the harness shards heads over all visible GPUs.
The default mode is `benchmark`; the default output prefix is
`profiling-data/paged_attention_benchmark`. Reusing the prefix replaces both files.
Relative export prefixes are resolved against the workspace under `bazel run`.
Run this mode without Nsight instrumentation when comparing representative latency.

### Gemma 4 31B global decode

Cases 24, 25, and 26 use batches 16, 32, and 64, respectively, with four KV
heads, eight query heads per KV head (32 query heads), head dimension 512,
and an 8,192-token KV cache. These match the global-attention head geometry in
[Google's Gemma 4 31B configuration](https://huggingface.co/google/gemma-4-31B/blob/main/config.json).
The 8K context is a benchmark choice, not the model's maximum context. Each
sequence decodes one token using BF16, causal attention and 16-token pages.
These cases measure the attention operation with separate K/V buffers; they
do not include projections, RoPE, KV cache updates or other model layers.
Use `CUDA_VISIBLE_DEVICES=0` to measure the complete head geometry on one GPU;
the harness otherwise keeps the KV-head count constant per device.

Cases 27, 28, and 29 cover the model's sliding attention at the same batches
and 8K context: 16 KV heads, two query heads per KV head, head dimension 256,
and a 1,024-token sliding window. Only the window's KV data contributes to
attention; the allocated cache still represents the full 8K context.

```bash
CUDA_VISIBLE_DEVICES=0 bazel run //zml/kernels/paged_attention/cute:benchmark \
  --@zml//platforms:cuda=true -- --mode=benchmark --case=24 \
  --warmup=20 --iterations=201 --output=profiling-data/gemma4_b16
```

## Nsight Compute

Install a recent Nsight Compute CLI on the CUDA host. It must support
`--nvtx-push-pop-scope process`: XLA can launch CUDA kernels from worker threads,
outside the calling thread's NVTX stack. Select one visible GPU for a simple
kernel comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
ZML_NCU_OUTPUT="$PWD/profiling-data/paged_attention_case0" \
bazel run --config=ncu //zml/kernels/paged_attention/cute:benchmark -- \
  --mode=profile --case=0 --warmup=5
```

Compilation, input uploads, and synchronized warmup runs finish before the NVTX
range starts. The final invocation runs once and waits for completion inside the
range. Profile mode does not emit latency statistics; Nsight may replay each
captured kernel many times to collect counters.

The `//tools:ncu` wrapper writes:

- `paged_attention_case0.ncu-rep`: a native report that can be opened in the
  Nsight Compute UI, including on another computer.
- `paged_attention_case0.csv`: raw per-kernel metrics exported from that report
  with base units. This is separate from the benchmark summary CSV.

The full metric set includes GPU duration, launch/resource statistics, occupancy,
memory behavior, and applicable stall analysis. CUDA graph nodes are profiled
individually, and the kernel filter includes both attention and merge. Generated
SASS is available in the report; Zig source correlation depends on debug/line
information emitted by the compiler.

The wrapper leaves clocks unchanged and defaults to Nsight's cache flushing for
reproducible replay (`ZML_NCU_CACHE_CONTROL=all`). Set it to `none` to inspect
reused-cache behavior; multipass results can then depend on preceding replay work.
Record this choice when comparing reports. Nsight's serialized/replayed kernel
durations do not replace unprofiled application latency measurements.

`ZML_NCU_OUTPUT` accepts a file prefix or a `.ncu-rep` path. Without it, the wrapper
uses a timestamped prefix under `profiling-data/`. Existing reports are preserved;
choose a new prefix for another run. `ZML_NCU_BIN` can select a particular `ncu`
installation. The wrapper does not elevate privileges; GPU performance counter
access must already be enabled for the user.

See NVIDIA's [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
for NVTX filtering, kernel replay, metrics, and report import/export.
