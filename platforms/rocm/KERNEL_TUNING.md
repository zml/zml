# Tuning custom ROCm kernels in ZML

Use this workflow when implementing or tuning a generator that specializes a
kernel for its execution device immediately before use. Keep generation logic
generic across supported devices: select instruction paths and configurations
from the workload and available capabilities, then compile and measure them on
the intended execution scope.

[Capabilities.zig](Capabilities.zig) is the API contract for field meanings,
units, sources, unknown values, initialization and merging. Keep hardware facts
there, tuning policy beside the generator, and instruction-specific legality in
backend lowering. This guide describes experiments, not product presets.

## 1. Define the workload and execution scope

Write down the case before changing the kernel:

- **Workload:** operation, shapes and their distribution, dtypes and storage
  encodings, strides/layouts, alignment guarantees, tails, sparsity/routing
  distribution, reuse, and any output initialization or aliasing requirements.
  Distinguish global tensor shapes from the shard processed by one launch.
- **Execution scope:** the visible device/partition, device set, sharding, local
  launch grid, graph/command-buffer use, and surrounding transfers or kernels.
  Tune for the devices that will execute the artifact, not the development host.
- **Numerical contract:** input, intermediate, accumulator and output precision;
  rounding, quantization/scales, overflow, NaNs/infinities and denormals where
  relevant; reduction ordering, determinism, and absolute/relative tolerances.
  Reduced precision, reassociation and split reductions need explicit acceptance.
- **Objective and baseline:** a correct existing path, representative cases, and
  the metric to improve (kernel latency, throughput, or end-to-end latency).
  Capture its configuration, correctness results, timings and spread first.

Read [platform population](../../zml/platform.zig) and the
[capability union](../capabilities.zig) when choosing the scope. Each
`Device.capabilities` describes one device; `Platform.capabilities` is merged
across all addressable devices. The current high-level kernel integration uses
that platform view. On one device these coincide; on several devices the merged
view may be narrower or unknown. A generator using that integration must respect
its common scope, even if a particular shard happens to run on a larger device.
Do not silently replace it with the first device's properties. Device-specific
specialization requires an execution path that actually confines the artifact
to that device.

For Fly, follow [the authoring guide](../../kernels/fly/README.md) and inspect
[`zml.kernel.fly`](../../zml/kernel.zig), the
[builder](../../kernels/fly/builder.zig) and
[`ops.fly`](../../zml/ops.zig). `Kernel.call` emits IR from `cfg` and shapes and
passes launch options to the ROCm plugin's `__gpu$xla.gpu.fly` custom call. The
plugin compiles and launches it. There is no automatic tuning loop or compiled
resource report returned by `Capabilities` or `Kernel.call`.

## 2. Form one hypothesis and choose a small experiment

Use a trace, resource report, access analysis or baseline measurement to name a
likely bottleneck. Keep the baseline and a few candidates that test that
hypothesis. Change one cause at a time unless layout/launch consistency requires
a coupled change; record that coupling. Decide what observation would reject the
hypothesis before running the experiment.

| Observation | Candidate change | Validation |
| --- | --- | --- |
| Adjacent lanes access distant addresses, or many narrow loads survive compilation | Change lane-to-element mapping; try a wider load only where alignment and valid extent are proved | Inspect addresses and emitted loads, test tails, compare traffic and elapsed time |
| Loads stall while independent compute exists | Add one prefetch stage or independent accumulator chain | Inspect waits and overlap, VGPR/LDS growth and spills; time the same case |
| Larger tiles reduce traffic but timing regresses | Reduce a tile dimension, unrolling or live staging state | Check compiled allocation and resident-wave evidence; compare traffic against concurrency |
| A reduction spends time on barriers/LDS | Keep more partial work within a wave, or change waves per group | Rebuild lane mapping and cross-wave combination; test masks and numerical error, then time |
| LDS accesses concentrate on the same banks | Try padding or a swizzle consistent with the consumer layout | Analyze addresses for the emitted instruction, include added LDS, and profile/time |
| A small grid leaves much of the device idle | Use smaller output tiles; consider a split reduction if the numerical contract permits it | Count useful groups, include merge/atomic/workspace costs, test reproducibility and end-to-end time |
| Repeated global loads dominate a reused tile | Stage it in LDS or change logical traversal order | Compare traffic saved with synchronization, LDS/register cost and actual reuse |
| Launches or data movement dominate the trace | Batch or fuse suitable work, or remove an unnecessary transfer | Measure the full operation, including new initialization and temporary buffers |

These observations suggest experiments; none proves an improvement. Low
occupancy alone is not a bottleneck diagnosis. A kernel with enough independent
work can benefit from register reuse even when fewer waves are resident.

## 3. Select instruction paths and prune illegal candidates

Use `compute.supports(...)` for the hardware gate and the actual lowering path
for compiler and instruction legality. For a matrix candidate, check the matrix
family and operand feature, then validate the atom's shape, required execution
width, per-lane fragments, accumulator type and numerical behavior. The builder
accepting an MLIR type is not proof that the installed plugin can lower it for
the selected target. Consult the pinned [FlyDSL sources and patches](../../third_party/flydsl/repo.bzl)
and [ROCm plugin version](rocm.bzl); preserve a supported fallback.

A concrete example: an FP32 matrix workload with an explicitly relaxed input
precision contract may justify testing an XF32 path when `tensorXf32` is true.
Check the backend's supported atom and compare accuracy and timings against the
baseline. Do not infer the feature from a target's age. Likewise, FNUZ and OCP
FP8 need their respective feature and encoding; a storage bitcast does not
convert between them. Packed arithmetic is a separate candidate from matrix
arithmetic, with its own layout and resource costs.

For direct-to-LDS candidates, test the feature matching the addressing form
(`bufferLoadToLds` or `globalLoadToLds`), and `wideLoadToLds` if required by the
instruction width. Lowering must establish alignment, lane mapping, completion
waits and synchronization before consumers access LDS. The flags do not promise
an asynchronous pipeline. `asyncEngineCount` is not a gate for this path.

Before compiling, reject candidates that exceed the effective per-block thread
or LDS ceilings, violate kernel shape/alignment constraints, leave work
uncovered, or require a feature or lowering that is unavailable. After compiling,
apply any tighter kernel limits and inspect actual resources. Do not interpret
a hardware ceiling as a recommendation to consume the entire resource.

When information is missing, make the consequence explicit:

- Unknown ROCm payload: retain a path whose assumptions are already established,
  or decline this custom kernel through the existing selector. Do not manufacture
  a target. `fly.isAvailable` only checks the ROCm backend tag;
  `fly.waveSize` currently falls back to 64 on an unknown payload, and
  `fly.mmaFlavor` returns null. Those helpers do not certify kernel support.
- Known target, missing device capacities: use the static limits and known
  restrictions per the API contract. Skip capacity-based heuristics that need
  missing values; for example, choose a coverage grid from the workload instead
  of inventing a multiprocessor count for a persistent grid.
- Unknown CU/WGP reporting mode: do not combine `multiprocessorCount` with
  per-physical-CU formulas until the units are established.
- Missing topology, cache policy details, throughput or compiled resources:
  label the corresponding model unavailable. Do not derive cache locality or
  workgroup placement from `xccCount`, or bandwidth from clock alone.

## 4. Tune execution geometry and resource use together

**Execution width, workgroup and layout.** Let `W` be the execution wave size and
`T` the threads in a workgroup. For the current Fly path, choose a positive `T`
and use the native width from the capability view. The integration requires
`T % W == 0` and sends `num_warps = T / W` to the plugin. The generator's thread
layout, shuffle masks, reductions, copy/MMA fragments and launch count must agree.
A memory vector width
is a different choice from `W`. A different compiled wave or CU/WGP mode requires
backend support and resource accounting for that mode; the capability table
supplies only its documented native geometry.

Start with the baseline's waves per group and a nearby legal alternative
motivated by the hypothesis. More waves can share a tile or reduce per-lane
work, but can also add synchronization and constrain scheduling. Fewer waves
can reduce those costs while increasing each lane's live state. Change the tile
partition and reduction scheme along with `T`; changing only the launch size can
invalidate the kernel's indexing.

**Tiles and registers.** Larger output tiles can reuse operands and amortize
addressing, but increase accumulators, reduce the number of groups in the grid,
and make tails more expensive. Larger reduction tiles, unrolling and more
pipeline stages extend live ranges. If compiled registers jump or scratch
traffic appears, try reducing one of those causes before forcing a register
budget. In Fly, a positive `waves_per_eu` sets `amdgpu-waves-per-eu` to `N, N`;
zero omits it. This is a compiler constraint, not a measurement of active waves.
Compare emitted allocations and timings when testing it.

**LDS and pipelines.** Staging can remove repeated global loads or enable
cooperation, but it costs LDS, waits and barriers. Another stage also keeps more
addresses, fragments and bookkeeping live. Check both per-workgroup capacity
and residency in the LDS domain, then inspect the generated schedule: allocating
two buffers does not establish overlap. Account for epilogue/reduction scratch
and padding, and prove a buffer is no longer in use before overwriting it.

The following formulas are estimates for pruning and interpreting experiments.
They are mathematical notation, not additional ZML APIs. Use checked sizes and
ceiling division for `ceil(x / g)` in `roundUp(x, g) = ceil(x / g) * g`;
allocation granules need not be powers of two.

For a dense matrix tile with output dimensions `BM, BN`, reduction chunk `BK`,
`S` simultaneously live staging buffers, and staged element widths `a, b` bytes:

```text
staging_bytes_est = S * (BM * BK * a + BK * BN * b)
L_est = staging_bytes_est + padding_and_other_live_LDS
L_alloc_est = roundUp(L_est, limits.sharedMemoryAllocationBytes)
LDS_groups_per_domain_est = floor(limits.sharedMemoryPerLdsDomain / L_alloc_est)
```

This assumes both operands are staged in every buffer. Quantized operands,
scales, expanded formats, unequal buffers and storage reuse require the actual
layout's byte count. Count simultaneously live allocations; confirm any reuse
in lowering. Require total static plus dynamic LDS to fit
`limits.sharedMemoryPerBlock`. Replace `L_est` with compiled static LDS plus
launch-time dynamic LDS when available, without double-counting metadata that
already includes it. If LDS use is zero, omit the LDS bound instead of dividing
by zero. The domain bound is not a count per physical CU on every target and
ignores other resident resources.

For an accumulator held entirely in registers with `c` bytes per element and
uniform distribution over `T` threads:

```text
accumulator_registers_per_lane_est = ceil(BM * BN * c / (4 * T))
R_alloc_est = roundUp(R, limits.vgprAllocationPerThread)
register_bytes_per_wave_est = 4 * W * R_alloc_est
waves_per_SIMD_est = min(limits.maxWavesPerSimd,
                        floor(limits.vgprFileBytesPerSimd / register_bytes_per_wave_est))
```

The first line estimates accumulator storage only. Real fragments may replicate
values or use a different distribution; addresses, masks, operands and pipeline
state add registers. In the remaining lines, `R` is the compiled allocation in
32-bit registers per lane charged to the modeled vector file. A plain VGPR count
can omit accumulators: shared VGPR/AGPR allocation and separate accumulator files
need appropriate accounting as documented in [Capabilities.zig](Capabilities.zig).
Do not add a separate file's bytes to general-purpose VGPR capacity. If `R` is
zero or the allocation interpretation is unknown, omit this estimate. It ignores
SGPR constraints, workgroup placement, register-index limits and other scheduler
restrictions. It cannot certify occupancy or launch legality.

For LDS access analysis, the bank-word model is:

```text
bank(byte_address) = floor(byte_address / limits.sharedMemoryBankBytes)
                     % limits.sharedMemoryBanks
```

Enumerate addresses for the lanes and words participating in each emitted LDS
instruction. Repeated banks with distinct addresses suggest a layout experiment;
same-address broadcasts and instruction-specific lane groups must be handled
separately. This formula does not predict conflict cycles for arbitrary vector
loads. Recheck the consumer's fragment layout and LDS size after a swizzle or
padding change.

**Available parallelism.** For a simple batched matrix kernel with one output
tile per group and no split reduction:

```text
groups = ceil(M / BM) * ceil(N / BN) * batch_count
```

Use local shard dimensions and adjust for routed/sparse work and masked groups.
Compare useful groups with the execution scope's scheduling capacity, keeping
CU, SIMD and LDS-domain units consistent. Only after establishing the reported
CU/WGP mode can you normalize `multiprocessorCount` to physical CUs and derive
an LDS-domain count. Combining measured/estimated groups per domain with that
count is still an idealized concurrency estimate; uneven work and placement
change utilization. A split reduction can create more groups but adds traffic,
workspace and combination costs and changes the numerical contract. Persistent
grids additionally need reliable device capacity; a tile that fits resources
can still leave too little parallel work.

For a traffic hypothesis, estimate `arithmetic_intensity = operations / bytes`
and report `effective_bandwidth = estimated_bytes / measured_seconds`. State
whether bytes mean algorithmic minimum, generated loads/stores, or measured
external-memory traffic. Cache reuse, staging and spills make these different.
The capability model has no compute-throughput ceiling or memory transfer-rate
convention for a complete roofline model. Use measured reference throughput for
the same device and access pattern if available; do not treat advertised clocks
and bus width as achieved bandwidth. Cache capacities can motivate a reuse
experiment but do not guarantee residency.

## 5. Compile and inspect the actual candidate

Compile for the intended execution target with the installed plugin/toolchain.
Save the generator configuration and emitted IR (`Kernel.emit` for Fly), then
inspect the backend dump, code object, compiler resource report or disassembly
available in that environment. The [Fly guide](../../kernels/fly/README.md#debugging)
describes the module dump path. An IR-only test cannot establish device support.

Check the instruction family and wave/mode, VGPR and accumulator allocation,
SGPR use, static/dynamic LDS, scratch/private-memory traffic, vectorized loads,
waits and barriers, and any compiler launch bounds. Replace estimates with
observations where possible. Determine whether a spill or allocation step
explains the timing change; scratch use alone does not identify its cause.

If an external tool can evaluate occupancy for the compiled kernel and actual
launch parameters, use it with the correct reporting units. ZML's capability
model exposes no such query. Otherwise retain explicit partial resource bounds
and use device measurements; do not report an exact occupancy percentage from
the formulas above. Keep a legal low-residency candidate when the hypothesis is
that its reuse or independent work compensates for fewer waves.

## 6. Validate correctness, then measure on the device

Validate every candidate against the numerical contract before accepting timing
results. Cover representative inputs and adversarial cases: tile boundaries,
partial waves, small/empty dimensions when supported, uneven routing, alignment,
large reduction lengths, scale extremes, and relevant exceptional values.
Check all written outputs, aliases, bounds, barriers and completion waits. Repeat
runs to expose races and stale temporary state. A relaxed matrix path or changed
reduction order needs error-distribution checks, not only one random comparison.

Test through the actual ZML call and execution mode. Fly custom calls pin
row-major layouts and pass raw pointers; dtype and argument order must match.
`zeroed_args` is ignored inside HIP graphs/command buffers, so arrange required
initialization explicitly in the executed computation. Include that cost in the
benchmark. Do not accept a standalone launch that passes while graph replay or
sharded execution fails.

Use existing tests near the generator. For Fly, `bazel test //kernels/fly:test`
checks the builder/emission layer; device integration tests live in
`//zml:test` and require the appropriate ROCm execution environment, for example
`bazel test //zml:test --@zml//platforms:rocm=true`. A skipped device test is not
correctness evidence for that device.

Warm up compilation, loading, allocations and the execution path before measuring
steady state. Use completed device work: device timing for kernel latency, or a
host timer with an explicit completion wait for end-to-end latency. Keep inputs,
cache/reuse conditions, graph mode, sharding and initialization identical between
baseline and candidate. Alternate/repeat trials, retain raw samples, and report
sample count, a central statistic and spread. Measure compilation/startup
separately when it matters to the objective.

Use [ZML's profiling workflow](../../docs/howtos/profiling.md#rocm-on-linux-with-rocprofv3)
for traces; the repository provides `--config=rocprofv3` for a Bazel run of the
benchmark that exercises your kernel. Collect only the counters needed to test
the hypothesis, if available. Record profiling separately from ordinary timing
because instrumentation can perturb execution. Record observed clocks, load or
throttling externally when available; the capability snapshot is not telemetry.
If no execution device is available, report compilation/inspection results and
leave performance and device correctness unverified.

## 7. Integrate the policy and preserve the evidence

Keep the winning decision near the generator: workload and numerical guards,
capability gates, a small configuration rule, and a supported fallback. Derive
launch dimensions, thread layouts and resource requirements from the same
configuration. Re-run validation through the real selector and compare the
representative workload suite, including small and tail-heavy cases. Preserve
other supported devices' paths; one device's result does not establish a
cross-device improvement. If a cached specialization depends on capabilities,
ensure its identity distinguishes those dependencies and the execution scope.

When a new hardware fact is needed, first establish its source and population
path in the capability model. Keep compiler/atom restrictions in lowering and
empirical thresholds in the generator. Avoid copying target tables into policy
or adding fields without a population path.

Record a reproducible result alongside the kernel or benchmark:

- Source revision, generator configuration, baseline and candidate identifiers,
  plugin/toolchain versions and compilation flags.
- Execution target and capability snapshot (including unknowns), visible device
  set/partition, sharding, graph mode and relevant environment settings.
- Exact build/run/profile commands, input shapes/dtypes/layouts/distributions,
  seeds, numerical contract and correctness results.
- Compiled resource observations and artifact locations; which values remain
  estimates, their assumptions, and unavailable measurements.
- Timing scope, warmup, repetitions, raw samples, statistic/spread, baseline and
  candidate results, and the policy decision with any regressions or limits.

Retain a change only when correctness and the chosen metric justify it. Report
an inconclusive or negative experiment as such; resource estimates and successful
compilation alone are not performance evidence.
