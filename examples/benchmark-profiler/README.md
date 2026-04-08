# Benchmark Profiler

`//examples/benchmark-profiler` benchmarks and profiles two execution patterns with xprof, pprof, and Tracy:

- `saxpy`: very small compute (`y = a * x + y`) called **5 times** sequentially to expose host dispatch overhead between `exe.call` invocations.
- `matmul`: large matrix multiplication where device compute dominates, with each output reinjected as one input on the next `exe.call`.

Both benchmarks include a warmup call before profiling, then capture a profiling window via `platform.profiler(...).start()/stop()`.

## MATMUL pipeline behavior

When `--mode=matmul` (or `--mode=both`), MATMUL calls are chained:

```text
out_0 = matmul(lhs_0, rhs_0)
out_1 = matmul(lhs_1, rhs_1)
...
```

where one input is replaced by the previous output each step.

- If `K == N`, reinjection happens on the LHS: `lhs_{i+1} = out_i`, `rhs_{i+1} = rhs_i`.
- Otherwise, if `M == K`, reinjection happens on the RHS: `lhs_{i+1} = lhs_i`, `rhs_{i+1} = out_i`.
- If neither is true, MATMUL pipeline setup is rejected because shapes are incompatible.

`--matmulCalls=<n>` controls chained calls per pipeline and `--matmulPipelines=<m>` controls how many pipelines are run in the profiled window. Total MATMUL calls are:

```text
total_calls = n * m
```

## Build

```bash
bazel run //tools/tracy:*
bazel build //examples/benchmark-profiler:benchmark_profiler
```

## Run

### SAXPY dispatch profile

- bazel build //tools/pprof:pprof //examples/benchmark-profiler:benchmark_profiler
- bazel run //tools/pprof:pprof -- -top /tmp/pprof/benchmark-profiler-saxpy.prof
- bazel run //tools/pprof:pprof -- -http=: /tmp/pprof/benchmark-profiler-saxpy.prof

Pprof does not need a separate local install anymore. The Bazel pprof target downloads a pinned official Go toolchain, builds pinned upstream `google/pprof`, and reuses the cached binary under `${XDG_CACHE_HOME:-$HOME/.cache}/zml/pprof`. Override the cache location with `ZML_PPROF_CACHE_DIR` if needed.

- bazel build //tools/tracy:tracy-profiler //tools/tracy:tracy-capture //tools/tracy:tracy-capture-daemon

- bazel run //tools/tracy:tracy-profiler -- --help
- bazel run //tools/tracy:tracy-capture -- --help
- bazel run //tools/tracy:tracy-capture-daemon -- --help

Tracy does not need a separate local install anymore. The Bazel Tracy targets build from the same pinned upstream Tracy source archive that Bazel already fetched, then reuse the cached build under `${XDG_CACHE_HOME:-$HOME/.cache}/zml/tracy`. Override the cache location with `ZML_TRACY_CACHE_DIR` if needed.

```bash
bazel run //tools/tracy:tracy-profiler --
```

```bash
bazel run //examples/benchmark-profiler:benchmark_profiler -- \
  --mode=saxpy \
  --dtype=f32 \
  --saxpySize=4096

bazel run //examples/benchmark-profiler:benchmark_profiler -- \
    --mode=saxpy \
    --dtype=f32 \
    --saxpySize=4096

bazel run //tools/pprof:pprof -- -http=: \
    bazel-bin/examples/benchmark-profiler/benchmark_profiler \
    /tmp/pprof/benchmark-profiler-saxpy.prof

bazel run //examples/benchmark-profiler:benchmark_profiler \
  --@zml//platforms:tpu=true \
  --@zml//platforms:cpu=false -- \
  --mode=saxpy \
  --dtype=f32 \
  --saxpySize=4096 \
  --sessionId=dispatch \
  --xprofDir=/tmp/xprof
```

### MATMUL compute profile

```bash
bazel run //examples/benchmark-profiler:benchmark_profiler \
  -- \
  --mode=matmul \
  --dtype=f16 \
  --matmulM=8192 \
  --matmulK=8192 \
  --matmulN=8192 \
  --matmulCalls=5 \
  --matmulPipelines=4 \
  --sessionId=dispatch
```

```bash
export TPU_VISIBLE_DEVICES=0
bazel run //examples/benchmark-profiler:benchmark_profiler \
  --@zml//platforms:tpu=true \
  --@zml//platforms:cpu=false -- \
  --mode=matmul \
  --dtype=f16 \
  --matmulM=8192 \
  --matmulK=8192 \
  --matmulN=8192 \
  --matmulCalls=5 \
  --matmulPipelines=1 \
  --sessionId=dispatch \
  --xprofDir=/tmp/xprof
```

Example with non-square but pipeline-compatible shapes (RHS reinjection, because `M == K`):

```bash
bazel run //examples/benchmark-profiler:benchmark_profiler \
  --@zml//platforms:tpu=true \
  --@zml//platforms:cpu=false -- \
  --mode=matmul \
  --dtype=f16 \
  --matmulM=4096 \
  --matmulK=4096 \
  --matmulN=8192 \
  --matmulCalls=5 \
  --matmulPipelines=3 \
  --sessionId=compute-rhs \
  --xprofDir=/tmp/xprof
```

### Run both benchmarks

```bash
bazel run //examples/benchmark-profiler:benchmark_profiler -- \
  --mode=both \
  --sessionId=dispatch-vs-compute \
  --xprofDir=/tmp/xprof
```

When `--mode=both`, two sessions are created:

- `<sessionId>-saxpy`
- `<sessionId>-matmul`

## Open in xprof

```bash
bazel run //tools/xprof:xprof -- /tmp/xprof
```

Then open `http://localhost:6006`.

## Accelerator backend flags

Append backend flags to `bazel run` when needed. Example for CUDA:

```bash
bazel run //examples/benchmark-profiler:benchmark_profiler \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false -- \
  --mode=both --sessionId=gpu-run --xprofDir=/tmp/xprof
```

Other available switches follow repo conventions:

- `--@zml//platforms:rocm=true`
- `--@zml//platforms:tpu=true`
- `--@zml//platforms:neuron=true`

## CLI options

- `--mode=<saxpy|matmul|both>`: benchmark selection (default: `both`)
- `--dtype=<dtype>`: floating-point dtype for both benchmarks (default: `f32`)
- `--saxpySize=<n>`: SAXPY vector length (default: `4096`)
- `--saxpyAlpha=<a>`: SAXPY scalar coefficient (default: `1.5`)
- `--matmulM=<m>`: MATMUL left rows (default: `4096`)
- `--matmulK=<k>`: MATMUL shared axis (default: `4096`)
- `--matmulN=<n>`: MATMUL right cols (default: `4096`)
- `--matmulCalls=<n>`: chained MATMUL `exe.call` count per pipeline (default: `1`)
- `--matmulPipelines=<m>`: profiled MATMUL pipeline repeat count (default: `1`)
- `--tracy=<on|off>`: enable or disable Tracy zones and frame marks at runtime (default: `on`)
- `--xprofDir=<path>`: xprof repository directory (default: `/tmp/xprof`)
- `--pprofDir=<path>`: pprof output directory, empty disables pprof capture (default: `/tmp/pprof`)
- `--pprofFrequency=<hz>`: pprof sample frequency, capped by gperftools at `4000` (default: `4000`)
- `--pprofDurationMs=<ms>`: repeated sampling window used to collect pprof CPU samples (default: `250`)
- `--sessionId=<name>`: session id prefix (default: `benchmark-profiler`)

## Notes

- On platforms without a PJRT profiler extension, the benchmark still runs but no trace artifacts are emitted.
- For clear dispatch/compute contrast, keep SAXPY relatively small and MATMUL large.
- MATMUL reinjection requires output shape compatibility with one input: `K == N` (reinject into LHS) or `M == K` (reinject into RHS).
