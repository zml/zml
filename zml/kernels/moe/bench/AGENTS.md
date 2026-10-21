# MoE benchmarks

`//zml/kernels/moe/bench` measures the public `zml.moe.forwardMoe` API.
Read `cases.zig` or `--help` for presets and flags; use `--list` to inspect
workloads and skip reasons without graph compilation or input allocation.

## Running comparisons

```sh
CUDA_VISIBLE_DEVICES=0 bazel run --config=release --@zml//platforms:cuda=true \
  //zml/kernels/moe/bench -- \
  --backend=cute,triton --preset=deepseek-decode --scheme=mxfp4 \
  --tokens=1,17,512,4096 --experts=8 --global-experts=8 \
  --routing=uniform,skewed,balanced --seeds=1,2,3 \
  --check --json=/tmp/moe-before.jsonl
```

- Repeat with identical arguments, hardware, and environment after the change;
  save to a new path. Validate a narrow selection before expanding the sweep.
- `--experts` is per rank. Defaults select both unsharded and expert-parallel
  cases; set `--global-experts` explicitly to select one. For sharding,
  `global-experts / experts` must equal the expert mesh partition count.
- Save sweeps with `--dump-dir`, then use `--compare-dir` after the change.
  Keep binary outputs and JSON sidecars together. `--compare-exact` requires
  bitwise equality for saved outputs and warmed replay. Single-file
  `--dump`/`--compare` requires exactly one runnable workload.
- Compare medians with `mad_us` and individual samples. Wall time includes host
  dispatch, output disposal, and collectives. Inspect all `ranks`; legacy scalar
  routing statistics and throughput estimates describe rank 0 only.

## Invariants

- Keep benchmark data independent of correctness-test fixtures. Workload IDs
  describe effective parameters, excluding preset names.
- Routing, seeds, and operand scales reuse graph compilation; data mode changes
  the setup graph and belongs in the graph key. Preserve compatible setup and
  prepared-weight reuse.
- Keep compilation, generation, layout conversion, transfers, and validation
  outside timing. The timer disposes every output and rejects donated inputs.
- Compile a fresh reference per checked workload: reusing reference executables
  after large weight regenerations produced incorrect CUDA results.
- Preserve per-element reference checks and warmed-replay checks against the
  initial output. A successful first launch does not validate replay.
- Keep kernel tuning in implementation-owned executables using
  `//zml/kernels/moe/bench:lib` and `suite.run`'s compiler hook.

## CuTe tuning and profiling

`//zml/kernels/moe/cute:tune` requires a new `--output` directory. It measures
production defaults first, then requires candidate outputs and warmed replay to
match bit for bit; `--check` adds reference validation. Axes vary independently;
`--configuration` selects one candidate plus the baseline. Keep tuning as typed
`Tuning`/`LaunchPlan` arguments; production must not read tuning environment
variables. Explicit plans still require compute capability 10.x and supported
DeepSeek geometry.

Use `--config=nsys` or `--config=rocprofv3` with narrow workloads. For direct
Nsight captures, set `SKIP_PJRT_PROFILER=true` and `--cuda-graph-trace=node`.
Analyze SQLite exports with `//zml/kernels/moe/bench:profile_metrics` and the
matching JSONL. Only the last `samples_us.len` batch ranges are retained samples.
Preserve completeness checks and configuration-range selection. Report kernel
sums separately from elapsed GPU time; use means for additive tables. The
analyzer does not infer a dependency critical path.

## Relevant tests

- Benchmark logic: `bazel test //zml/kernels/moe/bench:test`
- Profile analysis: `bazel test //zml/kernels/moe/bench:profile_metrics_test`
- CuTe tuner: `bazel test //zml/kernels/moe/cute:tune_test`
