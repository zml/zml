# Tuning oneAPI kernels in ZML

Use this guide when implementing or tuning an Intel GPU kernel in ZML. Kernels
are specialized on the machine where they execute, immediately before use. Aim
for a small, measured set of choices driven by hardware capabilities and workload
properties, with generation logic shared across supported devices.

Read [Capabilities.zig](Capabilities.zig) for the API contract, units, property
sources, and unknown-value semantics. Keep hardware facts there. This document
covers how to use those facts to make and evaluate tuning decisions.

## Establish the workload and baseline

Before changing a kernel:

1. Read its generator, call sites, and backend lowering. Trace how configuration
   becomes instructions, launch dimensions, and compiler options. Check what the
   integration already provides for resource inspection, timing, and caching.
2. Identify the actual execution device and local workload after sharding:
   dimensions, dtypes, accumulation precision, strides, alignment including view
   offsets, tails, aliasing, and required synchronization. Preserve the numerical
   contract when changing reduction order or math operations.
3. Use that `Device.capabilities` when execution is bound to the device. For a
   configuration shared across devices, use `Platform.capabilities`; graph
   generation normally reaches it through `zml.Compiler.current().platform`.
   Keep an existing fallback when capabilities are unknown. Specializing from
   device zero requires execution to be bound to that same device.
4. Run the existing correctness check and collect baseline timings on the target
   device for representative workloads, including small inputs and tails. Record
   the device, toolchain, launch configuration, and available compiled resources.

Keep the baseline available for comparisons. If the target device is unavailable,
perform the feasible compilation and correctness checks and report performance
as unmeasured. Do not present an occupancy estimate as a measured speedup.

## Choose a hypothesis and a small experiment

Identify what limits the current kernel, then change the parameters that address
that limit. Use profiler or compiler evidence when available; otherwise treat the
bottleneck as a hypothesis to test.

| Evidence or hypothesis | Candidate change | What to compare |
| --- | --- | --- |
| Scattered memory accesses | Lane mapping, vectorization, block loads where supported | Traffic and execution time at the actual strides/alignment |
| Register spills or excessive live state | Smaller accumulator tile, less unrolling, fewer stages, another subgroup or register mode | Spills, thread capacity, and time |
| SLM limits residency | Smaller staged tile, fewer buffers, less padding/scratch | Reuse lost versus groups that can reside |
| Too few independent groups | Smaller output tile, split reduction, different work distribution | Device utilization and total time including extra passes |
| Contended atomics | Aggregate per subgroup or work-group, or use a separate reduction | Synchronization, scratch traffic, and total time |
| Matrix-shaped compute using vector arithmetic | Supported DPAS path and backend instruction shapes | Lowering, accumulator cost, tails, and time |
| Irregular work or many inactive lanes | Narrower supported subgroup, less work per lane | Useful work, live state, and time |

Start with a few candidates around the baseline. Change coupled parameters
jointly when necessary, such as subgroup width and matrix layout, or tile size
and register mode. Avoid sweeping every parameter independently. Retain a change
only when correctness holds and its benefit survives repeated measurements.

## Generate and prune candidates

### Select an implementation path

Gate native instruction paths with `caps.compute.supports(...)`. For example,
a BF16 matrix path needs `.dpas` and `.tensorBf16`, plus backend support for the
exact operand/accumulator types, subgroup width, instruction shape, and packing.
Use `.dpasw` and `.block2dLoadStore` separately for their respective paths.

Choose matrix tiles from the backend's supported instruction shapes, then apply
resource budgets. Check pitch, alignment, transforms, and tails for block-memory
operations. A feature flag alone does not validate a particular lowering. Keep
target-specific encodings in the backend and an existing implementation for
workloads that the specialized path cannot handle. SYCL's
[matrix-combination query](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/programming-intel-xmx-using-sycl-joint-matrix.html)
is one way a backend can obtain supported shapes.

### Couple subgroup, work-group, and register choices

Intersect `limits.subgroupSizes` with the algorithm's and backend's supported
widths. Request an explicit width when the layout or collectives depend on it.
Otherwise, use the compiler's actual choice in subsequent estimates.

Try subgroup 16 as an initial policy when supported. Consider 32 for regular
work with modest live state, or 8 when supported for narrow or irregular work.
These are experiment starting points, not product presets. Wider subgroups use
fewer hardware threads for the same work-group but can increase per-thread live
state. Do not use `physicalSimdWidth` as the preferred subgroup size.

For implementations built around full subgroups, require a positive work-group
size divisible by the selected subgroup size. Clamp it to
`limits.maxWorkgroupSize` and check backend-specific launch constraints. Translate
options such as `num_warps` using the backend's actual units; CUDA's fixed width
is not an Intel launch rule. Mask tails while preserving collective and barrier
participation.

Only consider register modes the backend can select. If the compiler chooses the
mode automatically, inspect the result before using a mode-dependent estimate.
Compare a smaller tile or less unrolling with a larger register mode when spills
appear. Include accumulators, unrolling, extra work per lane, and pipeline buffers
in live-state reasoning. Read the mode's thread capacity instead of assuming all
register-mode transitions halve it.

### Estimate residency to identify resource cliffs

For SPMD kernels with a known register mode and full subgroups, use:

```text
S = selected subgroup size
W = work-items per work-group
T = W / S
H = limits.vectorEnginesPerCore * mode.maxThreadsPerVectorEngine
threadBound = floor(H / T)

If per-core SLM is known and allocatedSlmBytes > 0:
    slmBound = floor(limits.sharedMemoryPerCore / allocatedSlmBytes)
    residencyBound = min(threadBound, slmBound)
Otherwise:
    residencyBound = threadBound, with SLM unknown or unused
```

Treat this as a resource upper bound. Allocation rounding, placement, barriers,
and compiler constraints may reduce residency. Explicit SIMD backends need their
own execution mapping. Intel's [occupancy guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html)
explains the SPMD mapping.

To propose work-group sizes, try a few desired resident-group counts such as
1, 2, and 4: `W = S * floor(H / desiredGroups)`. Clamp to the work-group ceiling,
round down to the implementation's required multiple, and remove zero or
duplicate candidates. Also consider smaller groups when the workload needs more
independent groups. Higher predicted residency is a hypothesis for better latency
hiding; select the winner by time, accounting for any reuse lost to smaller tiles.

### Account for the complete SLM footprint

For a matrix tile with `p` input stages, start with:

```text
inputSlm = p * (tileM * tileK * bytesA + tileK * tileN * bytesB)
totalSlm = inputSlm + padding + scratch + otherLiveSlm
```

Use checked or sufficiently wide arithmetic. Count every simultaneously live
allocation and require the total to fit `limits.sharedMemoryPerWorkgroup`.
When per-core SLM is known, also consider the performance budget
`floor(sharedMemoryPerCore / desiredResidentGroups)`. Subtract other allocations
before assigning the remainder to staging, and recheck actual usage after
compilation. When per-core SLM is unknown, compare a small and a larger footprint
locally without guessing residency from the per-work-group ceiling.

Use SLM for reuse or communication that repays its loads, stores, and barriers.
A one-pass stream usually needs a candidate without staging. Synchronize both
production and consumption before reusing a pipeline buffer. Derive padding from
the chosen access pattern and backend evidence; the API supplies no universal
bank geometry. Intel's [SLM guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/shared-local-memory.html)
provides background on the tradeoffs.

### Check memory mapping and available parallelism

Map consecutive lanes to consecutive elements where the layout permits. For
several scalar elements per lane, consider
`base + iteration * subgroupSize + lane`; backend block or matrix loads may need
a different distribution. Verify actual strides and alignment after slicing,
and preserve masked tails. Intel's [subgroup memory examples](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/sub-groups-and-simd-vectorization.html)
show why contiguous work within a lane can still produce scattered accesses.

Compare prefetch or additional pipeline stages against a baseline without them.
Keep them only if useful computation overlaps the access and the extra live state
pays off. Do not derive prefetch distance or cache policy from memory capacity.

Derive the ordinary grid from the workload and output tile. When `coreCount` is
known, check whether that grid exposes enough independent groups. Use it to
propose split reductions or persistent scheduling where appropriate, without
assuming simultaneous residency or safe cross-group synchronization. With an
unknown count, keep a workload-derived grid. Account for all scratch and extra
passes when comparing alternatives.

## Compile, validate, and measure

Use the integration's existing compilation and benchmark paths. For each survivor:

1. Compile for the execution device with the required numerical behavior. Discard
   unsupported specializations while retaining the baseline or fallback.
2. Check the compiled subgroup and work-group requirements, launch dimensions,
   and total SLM use. Collect register mode, spills, and private-memory usage
   where available. Recompute estimates if the compiler changed the configuration;
   leave unavailable metadata unknown. Inspect lowering when a particular matrix
   or memory instruction is central to the hypothesis.
3. Compare results against a reference before timing. Cover the changed path's
   tails, small dimensions, strides, alignments, and numerical tolerances. Run
   the relevant ZML integration check as well as any isolated kernel check.
4. Warm up, then measure completed device execution with representative stream
   dependencies and cache conditions. Reset outputs and scratch between runs as
   required, especially for atomic or in-place operations. Exclude compilation
   and allocation from steady-state timings and record those costs separately.
5. Repeat measurements, compare with the baseline, and include every pass needed
   by the new implementation. Reject apparent improvements within measurement
   noise. Stop expanding the search when likely gains cannot repay tuning cost.

Backends with a Level Zero kernel handle can inspect
[`zeKernelGetProperties`](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/kernel.html#zekernelgetproperties).
That inspection belongs in the backend. Successful compilation does not by itself
validate a launch configuration.

## Integrate the result and leave evidence

Keep the winning policy near the kernel generator, expressed through capabilities
and workload constraints. Put hardware facts and product exceptions in
`Capabilities.zig`, with sources and focused tests when behavior changes. Keep
backend instruction restrictions in the lowering that enforces them. Prefer a
rule justified by the workload and resources over a new switch on product names.

Use the existing cache integration to bind a selected configuration to its
execution scope, device/partition, workload specialization, generator revision,
and relevant compiler/runtime versions and options. Include dtype, accumulation
policy, dimensions or validated shape buckets, strides, alignment, and tails.
Equal `Compute` values are insufficient to share measured results across devices.
Check every bucket member satisfies the variant's assumptions.

In the task report or change description, include:

- The bottleneck hypothesis and the resulting generation rule.
- The device, backend/toolchain, and representative workloads used.
- Baseline and selected configurations, timings, and any regressions or tradeoffs.
- Correctness checks and exact commands needed to reproduce the comparison.
- Remaining unknowns, unavailable device measurements, and fallback behavior.
