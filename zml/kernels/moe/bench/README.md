# MoE kernel benchmark

`//zml/kernels/moe/bench` sweeps configurations through the public
`zml.moe.forwardMoe` API. Start from a named, model-inspired case, vary dimensions
and options, and compare measurements before and after changing a kernel.
Benchmark cases and data generation are independent of the correctness tests.

```sh
CUDA_VISIBLE_DEVICES=0 bazel run --config=release --@zml//platforms:cuda=true \
  //zml/kernels/moe/bench -- \
  --backend=cute,triton --preset=deepseek-decode --scheme=mxfp4 \
  --tokens=1,17,512,4096 --experts=8 --global-experts=8 \
  --routing=uniform,skewed,balanced --seeds=1,2,3 \
  --check --json=/tmp/moe-baseline.jsonl
```

Repeat the same command and seeds after changing the kernel, with a new result
path. Use `--list` first to inspect runnable cases and skip reasons without
compilation or device allocations. Select fewer tokens or omit `--check` when
collecting a large performance sweep after validating a narrower selection.

## Cases and sweep axes

The complete preset catalog lives in `cases.zig`:

| Preset | Tokens | Top-K | Experts per rank | Width | Intermediate | Activation / placement |
|---|---:|---:|---:|---:|---:|---|
| `smoke` | 17 | 2 | 8 | 128 | 128 | SwiGLU / after down |
| `deepseek-decode` | 1 | 6 | 8 | 5120 | 2304 | SwiGLU, limit 10 / before down |
| `deepseek-prefill` | 512 | 6 | 8 | 5120 | 2304 | SwiGLU, limit 10 / before down |
| `laguna-decode` | 1 | 8 | 256 | 2048 | 512 | SwiGLU / after down |
| `gemma-prefill` | 17 | 2 | 8 | 5376 | 512 | GeGLU-tanh / after down |
| `many-experts` | 512 | 6 | 384 | 128 | 128 | SwiGLU / after down |

These are geometry seeds, with bounded expert counts for the wide matrices;
they do not reproduce an entire checkpoint. Defaults are BF16 activations, no
projection bias, and `quantize_input=false`. The default preset is `smoke`, or
`deepseek-decode` for CuTe/Fly. The default weight scheme is `none`, or `mxfp4`
for CuTe/Fly. Specify `--scheme` when comparing the same format across backends.
`--preset=a,b` selects named cases and `--preset=all` selects the entire catalog.

Every supplied axis **replaces** its preset value. Comma-separated values form a
Cartesian product; duplicate effective workloads are removed:

- Graph axes: `--backend`, `--scheme`, `--dtype`, `--activation`,
  `--quantize-input`, `--placement`, `--bias`, `--tokens`, `--topk`, `--width`,
  `--intermediate`, `--experts`, and `--swiglu-limit`.
- Runtime axes: `--routing` and `--seeds`. `--seed` selects one seed; `--seeds`
  overrides it with a list.
- `--input-scale=0,1,16` scales activations; `--routing-scale=1,2.5` scales
  normalized routing weights. Both reuse compilation. Use zero inputs/weights
  and larger activations to exercise zero output and SwiGLU clipping.
- `--swiglu-limit=none,7,10` applies to SwiGLU and SwiGLU-step. `none` removes
  clipping; other activation kinds do not use the limit.
- `--filter=<substring>` narrows the complete case IDs printed by `--list`.

Unspecified fields retain their preset values. Explicit unsupported combinations
are reported as skipped, including incompatible dtypes, bias, quantization,
activation parameters, and dimension alignment. For example, Triton width 129
is skipped even with unquantized weights. Measurement runs fail when no runnable cases remain; `--list` still succeeds
so a fully skipped selection can be inspected.

```sh
bazel run --config=release //zml/kernels/moe/bench -- \
  --platform=cpu --cpu-devices=1 --backend=stablehlo --preset=smoke \
  --dtype=bf16,f32 --bias=false,true --tokens=1,17 --routing=balanced \
  --check --json=/tmp/moe-cpu.jsonl
```

To add a case, add one named `Preset` entry with its `Case` configuration in
`cases.zig`. Selection, deduplication, graph reuse, and reporting require no
changes. Case IDs describe effective parameters and exclude the preset label,
so overriding two presets to the same workload measures it once. Results retain
the first selected preset's name as provenance.

## Execution and measurements

`suite.zig` selects and validates cases, groups identical graphs, compiles each
group, and measures its runtime variants. `cases.zig` owns the catalog and
selection; `workload.zig` owns operands, layouts, input generation, and routing;
`runner.zig` owns compilation, correctness checks, and measurements.

Routing distributions and seeds do not trigger recompilation. The data mode is
part of the graph key because it changes the setup graph. Compilation, weight
generation, layout conversion, data transfers, output inspection, and reference
checks are outside timing. Only one compatible setup is retained at a time.
Consecutive token counts reuse weight generation, prepared weights for the same
seed. Routing and operand-scale variants also reuse them; each checked workload
compiles a fresh reference as described below.

Defaults are three synchronized warmups, seven measurement samples, and at least
20 calls and 20 milliseconds per sample. `--warmup`, `--samples`, `--iters`, and
`--sample-ms` control these values. Samples report amortized wall time per call,
including host dispatch and synchronization. Results include the median,
median absolute deviation (`mad_us`), minimum/maximum, sample count, and
individual `samples_us` so an agent can distinguish
noise from a performance change. Expert-parallel timing includes the public
API's collectives. The `ranks` array reports actual route count, touched experts,
largest expert load and padded rows for every rank. The legacy scalar routing
statistics, TFLOP/s and touched-weight bandwidth still describe rank 0; they are
estimates, not hardware counters. A decode rank can receive no routes, so inspect
all ranks.

Execution is in process. Recoverable errors become failed case records; panics
or device faults can stop the run. Completed JSONL records remain readable.

## Routing, data, and devices

`--routing=uniform,skewed,balanced` is the default. Uniform and skewed routing
sample distinct experts per token; skew uses seeded Gumbel top-k sampling with
`--sigma=0.6`. Balanced routing cycles over eligible experts. `--active-experts=N`
restricts each rank's contiguous expert block to its first N entries; zero means
all experts. The restriction applies to every routing distribution.

`--data=random` is the default and generates reproducible random operands,
including FP4 values and scales. `--data=fixture` selects simple deterministic
patterns. Seeds control data and routing; selection order and filtering do not
change a case's inputs.

Use `--platform=cpu|cuda|rocm|metal|tpu|oneapi` to choose a platform and its normal
device-visibility environment variables to choose devices. Without `--backend`,
available accelerated backends are selected, falling back to StableHLO.
`--cpu-devices=N` creates N CPU devices (default 2).

`--experts` is the per-rank expert count. By default, each case runs both
unsharded and sharded over the available expert mesh. `--global-experts` selects
one mode: equal to `experts` means unsharded; a larger value must satisfy
`global-experts / experts == expert mesh partitions`. Weights are sharded along
the expert axis; inputs and global route IDs are replicated. Device count
mismatches are skipped before compilation.

```sh
bazel run --config=release //zml/kernels/moe/bench -- \
  --platform=cpu --cpu-devices=2 --backend=stablehlo --preset=smoke \
  --experts=2 --global-experts=4 --topk=3 --tokens=1,17 \
  --placement=before_down --swiglu-limit=10 --check
```

## Correctness and saved results

`--check` compares against the independent StableHLO reference using the same
global experts and inputs, including projection bias and expert-parallel
collectives. The reference evaluates one token at a time to bound temporary
memory. Each checked workload owns its reference executable: this prevents
incorrect CUDA reference results after large weight regenerations. Its
compilation time is recorded separately in `reference_compile_ns` and excluded
from all timing samples. Comparisons use absolute and relative tolerances and require every
element to pass. They also record exact matches, nonfinite counts, maximum
absolute error and its output index, relative L2 error, and the first failing
index with both values and its allowed error; a small aggregate error does not silently
waive individual tolerance failures. Runs without `--check` record that reference validation was not performed.
Every measured workload also compares the warmed executable with its initial
output outside the timed range, so command-buffer replay errors cannot silently
pass a first-launch check. `--compare-exact` makes this comparison bitwise too.

`--json=results.jsonl` writes one complete JSON object per line and flushes each
record. Run metadata records the source revision, dirty state, source hash, compiler
version/optimization, platform, device IDs, compute capabilities and SM counts,
measurement settings, and relevant environment settings (`CUDA_VISIBLE_DEVICES`, `XLA_FLAGS`, `ZML_AUTOTUNE_CACHE_DIR`). Case records include their full configuration,
preset, status and skip/failure reason, compilation time, timing samples, routing
statistics, output statistics, and comparison results. Keep the settings and
hardware fixed when comparing runs.

Binary output baselines require exactly one runnable case:

```sh
CUDA_VISIBLE_DEVICES=0 bazel run --config=release --@zml//platforms:cuda=true \
  //zml/kernels/moe/bench -- \
  --backend=cute --preset=deepseek-decode --global-experts=8 \
  --routing=skewed --seed=1 --dump=/tmp/moe-before.bin
```

The dump also writes `/tmp/moe-before.bin.json` with its workload identity.
Keep both files together. Repeat with `--compare=/tmp/moe-before.bin` after
changing the kernel; comparisons require matching baseline metadata. Comparison
reports exact matches, maximum absolute error, and tolerance failures.

Use `--dump-dir=/path/before` and `--compare-dir=/path/before` to validate an
entire sweep. Filenames hash the workload ID; each sidecar verifies the full
identity, dtype, shape, and byte count. `--compare-exact` requires every output
bit to match, while ordinary comparison uses the numerical tolerances. Default
operand scales retain the identities of older baselines.

For device-only timing, run a narrowly selected case with the existing
`--config=nsys` or `--config=rocprofv3` profiling configurations. Profiles contain
`moe_bench.measure/<case ID>` and nested `moe_bench.batch#iterations=N#` ranges.
The last `samples_us.len` batch ranges are the retained samples; earlier ranges
are warmup or duration calibration. Verify complete launches before comparing kernel durations. Report kernel sums separately from elapsed GPU
time: kernels and collective communication can overlap, and synchronized wall
time also includes host dispatch and gaps.

Workload axes describe the MoE operation. Implementation-specific executables
own kernel parameter sweeps. The main CLI depends on
`//zml/kernels/moe/bench:lib` (`@import("moe_bench")`): `suite.run` accepts an
implementation-owned compiler and reuses selection, setup, validation, timing,
and reporting. It owns no process-wide state; callers own platforms and meshes.
Stage benches can also use `cases`, `workload`, `runner.measure`, and `results`
individually. The timer disposes every tensor output, so stages returning both
quantized values and scales can use it directly. It rejects executables that
donate inputs, since the next call would otherwise reuse a freed input.

## CuTe tuning

`//zml/kernels/moe/cute:tune` is a Zig executable. It resolves typed `Tuning`
values into a `LaunchPlan` and passes each plan directly to
`cute.fusedExpertsWithPlan`. The same wrapper handles unsharded and
expert-parallel execution. Production CuTe uses its default launch policy and
reads no tuning environment variables. The former `ZML_MOE_CUTE_*` controls
and `tune.sh` have been removed.

```sh
CUDA_VISIBLE_DEVICES=0,1 bazel run --config=release --@zml//platforms:cuda=true \
  //zml/kernels/moe/cute:tune -- --output=/tmp/moe-tuning \
  --tiles=32,64,128 --quantize-lanes=1,4 \
  --experts=192 --global-experts=384 --topk=6 \
  --tokens=1,256,2048,4096,8192,16384 --routing=uniform,skewed --seeds=1,2
```

The output directory must be new. The tuner first measures the production
public API and saves a baseline, then requires every candidate output and
warmed replay to match it bit for bit. `--check` additionally checks the
independent reference. A failed baseline stops the sweep; failed candidates
remain recorded and the rest continue, with a nonzero final exit status.
Compilation and validation remain outside timed samples.

| Tuner option | Values |
|---|---|
| `--tiles` | Comma-separated `8,16,32,64,128`; wide tiles run paired and single-CTA controls; single-token decode retains its dedicated 8-row tile |
| `--quantize-lanes` | Comma-separated `1,4` |
| `--persistent-ctas` | Positive counts no greater than the smallest participating GPU's SM count; pairs require even counts |
| `--configuration` | One generated name, such as `tile-64-single` or `quantize-4`; the default baseline still runs first |

Each axis varies independently against the defaults. Use `none` to omit an axis;
`--persistent-ctas` defaults to `none`. Workload arguments are identical to the
main bench. The tuner owns backend, output, and baseline options; `--list`,
`--json`, `--dump*`, and `--compare*` are rejected. Each setting gets a separate
JSONL file containing its requested configuration, source provenance, workload
arguments, and the resolved `cute_launch` plan for every graph.

The backend supports compute capability 10.x and the DeepSeek expert geometry;
explicit plans do not bypass these requirements. Grids use device attributes,
and routing density uses the global expert count independently of mesh size.

For profiling, build the tuner and run it under Nsight with a narrow selection:

```sh
bazel build --config=release --@zml//platforms:cuda=true //zml/kernels/moe/cute:tune
CUDA_VISIBLE_DEVICES=0,1 SKIP_PJRT_PROFILER=true nsys profile \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output=/tmp/moe-tune bazel-bin/zml/kernels/moe/cute/tune \
  --output=/tmp/moe-tuning-profile --configuration=tile-64-single \
  --experts=192 --global-experts=384 --topk=6 --tokens=4096 --routing=uniform
nsys export --type=sqlite --output=/tmp/moe-tune.sqlite /tmp/moe-tune.nsys-rep
bazel run //zml/kernels/moe/bench:profile_metrics -- \
  /tmp/moe-tune.sqlite /tmp/moe-tuning-profile/tile-64-single.jsonl \
  --output=/tmp/moe-tuning-profile/tile-64-single-metrics.json
```

Configuration NVTX ranges let the analyzer select the matching setting from a
single capture even when several settings use identical workload IDs. To
isolate settings in separate processes, invoke the tuner again with a different
`--configuration` and a new output directory.

The analyzer rejects incomplete batches, missing ranks, changing graph nodes,
and incomplete ordinary launch counts. It reports kernel work, the union of GPU
activity, overlapping work, memory-only activity, and time without recorded GPU
activity. These account for the full measured batch, per rank and across ranks. Use the
reported means for additive tables; independently computed component medians
need not sum to the median batch time. Duplicate workloads or measurement
ranges are rejected.
Replay spans are available when each invocation is captured as one graph;
ordinary or mixed launches use counts instead. None of these metrics infer a
dependency critical path. Failed or skipped workloads remain in `unmeasured`.
