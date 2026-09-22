# Tuning custom CUDA kernels in ZML

Kernels are specialized on the execution device immediately before use. Keep the
generation logic generic across supported devices: select paths from hardware
features, resources and the local workload, then measure the generated kernel.
Hardware facts belong in [Capabilities.zig](Capabilities.zig), tuning policy near
the generator, and instruction restrictions in backend lowering.

The capability file is the API contract for field meanings, units, sources,
unknown values and merge behavior. Use it alongside this workflow.

## 1. Establish the workload, execution scope and baseline

Before changing a configuration, record:

- **Workload:** operation, shapes after sharding, dtypes, strides/layouts, guaranteed
  alignment, reuse, reduction axes, masks and tail sizes. Distinguish static
  capacity from live lengths and record the distribution of runtime inputs.
- **Execution scope:** the devices/partitions that will execute this generated
  variant, per-device work, streams and concurrent work. A block (CTA/work-group)
  cooperates within a kernel; the grid distributes blocks across the device.
- **Numerical contract:** input, intermediate, accumulator and output precision;
  error metric/tolerances; allowed reassociation, contraction, approximations,
  denormal handling and nondeterminism. Include NaN/Inf behavior where relevant.
  A faster instruction path is eligible only within this contract.
- **Objective and baseline:** the current correct implementation and revision,
  representative and boundary cases, and whether the objective is isolated kernel
  latency, throughput, or end-to-end model performance. Preserve its launch count,
  scratch traffic and timing conditions for comparison.

Inspect [device and platform construction](../../zml/platform.zig) before choosing
a capability scope. `Device.capabilities` describes one device;
`zml.Compiler.current().platform.capabilities` is merged across all
`platform.devices`. That aggregate is not automatically restricted to the mesh or
subset executing this kernel. Use a device view only when execution is constrained
to that device; a shared variant must be valid across its execution scope. Use
local dimensions after partitioning when counting blocks and scratch bytes.

Match the actual [kernel integration](../../zml/kernel.zig) and
[custom-call options](../../zml/ops.zig):

| Interface | Where the configuration goes |
|---|---|
| `zml.kernel.triton.Kernel` | Generator-specific `cfg`, plus `grid`, `num_warps` and `num_stages` in `CallOpts`. |
| `zml.kernel.cute.Kernel` | Generator-specific `cfg`, plus `grid` and `block`; declared kernel storage determines dynamic shared memory. |
| `zml.kernel.cute.Program` | Generated host IR owns launches and descriptors; its `CallOpts` has no external grid/block. |
| `zml.kernel.cuda_tile.Kernel` | `cfg` and `grid`; warp/CTA choices are IR optimization hints, not `num_warps` or `num_stages` call options. |

These wrappers emit IR/custom calls for XLA to compile. They do not return a CUDA
function handle or compiled-resource/occupancy report to the generator. Inspect
the relevant [Triton](../../kernels/triton/README.md),
[CuTe](../../kernels/cute/README.md) or
[CUDA Tile](../../kernels/cuda_tile/README.md) integration before assuming a knob
or instruction path is available. Existing policies in
[paged attention](../../zml/kernels/paged_attention/triton/triton.zig) and
[CuTe attention](../../zml/kernels/flash_attention/cute/attention.zig) illustrate
where workload choices live; their constants and workarounds are kernel-specific.

## 2. Form a bottleneck hypothesis and choose a small experiment

Measure the baseline first. Write one falsifiable hypothesis, the evidence behind
it, a candidate change and the observation that would support or reject it. Keep
the baseline and a few nearby configurations; avoid a Cartesian sweep of every
knob. If knobs must change together for legality, explain that coupling.

| Observation | Candidate change | Validation beyond elapsed time |
|---|---|---|
| Few blocks for the local workload; much of the device is idle | Reduce work per block or split a long reduction | Count the new grid; include scratch, merge launch, tails and changed reduction order. |
| Excess sectors relative to useful global bytes | Map adjacent lanes to contiguous elements; try an aligned vector width | Inspect load/store instructions and transactions; test misalignment and masked tails. |
| Repeated global loads of data reused within a block | Stage a tile in shared memory or increase reuse per tile | Verify reduced traffic, added barriers, shared footprint and residency. |
| Spills or a residency drop near a register threshold | Reduce tile/unroll/stage count, or distribute values over more warps | Compare compiled registers, spill traffic, instructions and resident blocks. |
| Dependency stalls with little independent work | Try another independent accumulator or more items per thread | Check whether issue utilization improves without spilling or losing too much residency. |
| Loads serialize with compute despite reusable tiles | Try one additional pipeline stage on a supported async path | Verify actual overlap and correct wait/reuse ordering; check both register and shared costs. |
| Shared-memory access conflicts | Change layout, padding or swizzle supported by the backend | Inspect transactions/conflicts and account for extra storage and address arithmetic. |
| Barrier stalls or uneven work among warps | Reduce unnecessary synchronization or rebalance work | Validate participation, races, boundary paths and resulting utilization. |
| Atomic contention on a small hot set | Aggregate within a warp/block or use a separate reduction | Include temporary storage, synchronization, merge cost and numerical differences. |
| Launch/dispatch time dominates small inputs | Fuse work or keep tails in the main launch | Measure the full call sequence and verify aliasing and masking. |

Occupancy alone does not identify the bottleneck. A candidate may use more
registers and fewer resident warps yet expose more independent work. Conversely,
a higher occupancy number does not prove that warps are eligible to issue. Use
profiler evidence to distinguish dependency, memory, synchronization and issue
limits; treat an unprofiled explanation as a hypothesis.

## 3. Gate instruction paths and prune configurations

Read `compute`, `hardware.limits` and `hardware.properties` from the selected CUDA
capability view. Consult the [contract](Capabilities.zig) rather than copying
architecture tables into a generator.

1. **Check the numerical and workload requirements.** Tensor formats, sparse
   representations, vector alignment and reduction ordering must fit the input
   contract. Hardware support cannot grant permission to change precision.
2. **Check hardware features explicitly.** Use `compute.supports(...)` for the
   mechanism being selected: for example `.asyncCopy`, `.tma`, `.tensorBf16`,
   `.wgmma` or `.tcgen05Int8`. Do not infer a feature from a greater numeric compute
   value or from a family label. An operand-format flag does not select a specific
   MMA instruction family.
3. **Check the actual backend path.** The installed compiler, target options,
   instruction forms, layouts, alignment, participating threads and synchronization
   must all be supported. Keep these checks in lowering. A capability flag does
   not configure descriptors, memory opt-in, clusters or cache policies.
4. **Reject definite resource/launch violations.** Check each block/grid dimension,
   total threads, authored storage and the chosen instruction's participation
   rules. Use checked arithmetic. Treat register and residency estimates as
   pruning evidence with uncertainty; repeat the checks on compiled resources.

For example, the Triton wrapper currently sets `is_tma_allowed` only when
`global_scratch_memory_size > 0` and adds an implicit scratch result for XLA.
A true `.tma` feature alone does not arrange this ABI or supply the required
scratch size. Follow the current wrapper/lowering contract and inspect the result.

Missing information changes which decisions can be justified:

| Missing or limited information | Required response |
|---|---|
| CUDA capability view is null | Retain a proven fallback or decline that custom path. Do not substitute a nearby compute identity. |
| Optional count/capacity is null | Skip policies requiring it, or collect it separately in the experiment. Do not turn it into zero or a guessed product value. |
| Runtime launch support is false or unknown | Do not rely on that launch mechanism without establishing support in the backend/runtime. Static feature support is insufficient. |
| SM count or compiled residency is unavailable | Do not claim a device-sized persistent grid or exact scheduling-wave count; use a workload-based fallback and measure. |
| Driver shared-memory reserve is unknown | Mark the residency budget incomplete; a successful architectural capacity check does not prove residency. |
| Throughput/latency helper returns null | Leave that analytical term unknown. Do not borrow another target's rate. |
| Cluster size, tensor-memory capacity or instruction restrictions are needed | Obtain them at the backend/toolchain boundary; this capability model does not supply them. |

Known zero is distinct from null: for example, a zero persisting-L2 capacity rules
out a positive set-aside. Guard all ratios against zero. A merged view can contain
minima from different devices and unknown properties; it is not a measured GPU.

## 4. Tune coupled resources

Execution width has several meanings. `limits.warpSize` is hardware participation
width. Warps per block, logical tile shape, vector bytes and elements per thread
are different choices. More warps can distribute accumulators across threads, but
also consume warp slots and change layouts and synchronization. A larger logical
tile need not imply a larger block. Confirm the backend's physical mapping,
especially for compiler-managed CUDA Tile launches.

Larger tiles can improve reuse while increasing accumulator registers, staged
storage, tail work and time per block. More pipeline stages add storage and live
state while attempting to overlap transfers with compute. More items per thread
can expose independent instructions but reduce the number of blocks. Evaluate
these choices together with the amount of local work, rather than maximizing any
one of tile size, stages or occupancy.

The following formulas are **planning estimates**, not an occupancy implementation.
Let `L = hardware.limits`, `P = hardware.properties`, `ceilDiv(a,b)` round upward,
and `roundUp(x,q) = ceilDiv(x,q) * q`. Use wide, checked arithmetic and positive
denominators. They assume an ordinary grid of identical blocks, not clustered or
cooperative scheduling, and must be revisited when lowering changes the mapping.

### Threads and registers

```text
T = block_x * block_y * block_z
W = ceilDiv(T, L.warpSize)
B_warps = floor(L.maxWarpsPerSm / W)

R_warp = roundUp(r * L.warpSize, L.registerAllocationPerWarp)
R_block_est = W * R_warp
B_regs_est = floor(L.registersPerSm / R_block_est)
```

Here `r` is compiled 32-bit registers/thread when available; before compilation it
is only an estimate. Also check `T <= L.maxThreadsPerBlock`, per-axis dimensions,
`r <= L.registersPerThread`, and the per-block register ceiling. Partial warps still
consume warp resources. The register formula ignores subpartition placement and
other allocation constraints: allocation must fit within `L.smSubpartitions`, not
just the SM total. Use compiler/CUDA occupancy tools for the actual kernel when
accessible.

For an accumulator tile uniformly distributed across threads, a rough contribution
is `ceilDiv(accumulator_elements, T) * ceilDiv(accumulator_bytes, 4)` registers per
thread. This excludes operands, addresses, masks, loop/pipeline state, duplicated
fragments and compiler temporaries; it is not the register count. Compare nearby
tiles/warp counts around allocation thresholds, then inspect the compiled result.
A register cap can create spills; do not infer that a smaller count is faster.

### Shared memory and pipeline depth

For a tiled matrix operation staging distinct A and B buffers at every stage:

```text
S_stage_est = tile_M * tile_K * bytes_A + tile_K * tile_N * bytes_B
S_kernel_est = stages * S_stage_est + padding + barriers + other_live_scratch
S_block_est = roundUp(S_kernel_est + reserve, L.sharedMemoryAllocationBytes)
B_shared_est = floor(C / S_block_est)
```

`reserve` is a known `P.reservedSharedMemoryPerBlock`; `C` is a modeled partition
from `L.sharedMemoryCarveouts`. Use peak simultaneous storage if lifetimes permit
reuse. Padding, swizzles, alignment, descriptors and the actual lowered pipeline
can change this estimate. Once compiled, use the kernel's static plus requested
dynamic shared memory instead of `S_kernel_est`.

Check kernel storage against `L.sharedMemoryPerBlock`, or
`L.sharedMemoryPerBlockOptin` only when the function and launch are configured for
opt-in. Count the driver reserve as additional SM residency cost, not user scratch.
For a proposed `B` resident blocks,
`L.sharedMemoryCarveout(B * S_block_est)` finds a modeled fitting partition or
returns null. It neither requests that partition nor proves residency. The largest
partition is not the per-block allowance, and `unifiedL1SharedBytes` is not a shared
memory budget. More shared allocation can also change the cache/shared balance.

Try another stage only when there is independent compute to overlap with the next
transfer. Check prologue, steady state, drain, buffer reuse and short-loop/tail
paths. If another stage reduces residency or adds spills without overlap, remove
it. `P.asyncEngineCount` is not a kernel pipeline-stage limit.

### Residency and available work

```text
B_est = min(L.maxBlocksPerSm, B_warps, B_regs_est, B_shared_est)
resident_warps_est = B_est * W
occupancy_est = resident_warps_est / L.maxWarpsPerSm
slots_est = P.multiprocessorCount * B_est
waves_est = ceilDiv(grid_blocks, slots_est)
```

Omit a resource bound if its demand is zero; leave the result incomplete if an
input is unknown. Register allocation, active carveout, function limits and launch
mode can reduce the actual resident block count below this estimate. A zero bound
means the modeled configuration does not fit; do not clamp it to one to claim
legality. Occupancy is a fraction, not throughput.

`waves_est` assumes equal-duration blocks and no competing work. A small grid may
leave SMs idle; a partial final wave or irregular blocks can leave a long tail.
Reducing tiles or splitting reductions may expose more blocks, but adds launches,
partial results and merging. For a persistent kernel, a budget of
`SM_count * resident_blocks_per_SM` requires a known count and established
residency. It does not establish cooperative-launch or deadlock safety.

### Traffic, reuse and latency

For a concrete warp access, count distinct sectors touched by active lanes using
`L.globalMemorySectorBytes`. `useful_bytes / (sectors * sector_bytes)` is a
coalescing diagnostic if useful bytes count unique requested bytes. It is not DRAM
bandwidth: caches, repeated accesses and writes alter downstream traffic. Choose
vector width from proven alignment, contiguity and tail legality, then inspect
emitted instructions. Shared-bank address reasoning can use
`floor(byte_address / L.sharedMemoryBankBytes) % L.sharedMemoryBanks`; broadcasts,
wide instructions and swizzles require checking the actual access pattern.

`reusable_working_set_bytes / P.l2CacheBytes` can motivate a cache experiment when
the denominator is known and positive; it does not predict hit rate or available
cache under contention. Persistence additionally needs hardware support, usable
capacity/window limits and a backend/runtime path that actually sets the policy.
Do not assume the wrappers expose such a control.

For a measured workload, `arithmetic_intensity = operations / transferred_bytes`
is useful only with an explicit counting convention (e.g. FMA counts as two FLOPs)
and memory level. With independently measured effective rates:

```text
t_compute_est = operations / effective_operations_per_second
t_memory_est = transferred_bytes / effective_bytes_per_second
t_est = max(t_compute_est, t_memory_est) + uncovered_sync + launch + tail_cost
```

This assumes compute and memory can overlap and avoids counting hidden latency
twice; validate the assumption. These effective rates and costs are not capability
fields. `compute.throughput()` and `dependentFp32FmaLatencyCycles()` have narrower
contracts in [Capabilities.zig](Capabilities.zig). The latter can motivate exposing
independent FMA chains, but does not determine the warps needed to hide memory
latency. Clock and bus-width reports alone cannot supply a sustained-bandwidth or
whole-kernel performance model.

## 5. Compile and inspect the actual variant

Generate through the production wrapper with the candidate's exact local shapes,
configuration and compiler options. Enable
[`zml.Compiler.Options.xla_dump_to`](../../zml/compiler.zig) to preserve artifacts.
The backend guides describe the available dumps: CuTe includes `.cute.mlir`,
`.cute.json` and `.cubin`; CUDA Tile includes `.cuda_tile.mlir` and `.tilebc`.
For Triton, the [IR inspection workflow](../../kernels/triton/PORTING.md) shows how
to compare lowering stages. Available artifacts depend on the compilation path.

Record the selected target/instruction path, physical block mapping, registers
per thread, static and dynamic shared memory, spills/local-memory use and launch
restrictions where the compiler exposes them. Inspect PTX and machine code when
available: a source-level vector, async copy or tensor operation is not evidence
that the intended instruction was emitted. Local memory can also contain arrays
or stack data; confirm spill loads/stores rather than labeling all local storage
as spilling.

When a CUDA function handle is accessible in backend tooling, inspect
[function attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaFuncAttributes.html)
and use [CUDA occupancy APIs](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html)
with the actual block and dynamic shared-memory request. ZML's generator wrappers
do not expose these queries. Otherwise use compiler reports, dumps and profiling,
and record any resource/residency value that remains unavailable. Reconcile these
results with the estimates before promoting a candidate.

## 6. Validate correctness on the execution device

Run the affected kernel's tests against a trusted reference under the declared
numerical contract. Include representative inputs, odd dimensions, partial tiles,
short loops, extreme reduction lengths, permitted misalignment/strides, empty work
where supported, and supported dtype/sharding cases. Exercise every new dispatch
branch and both sides of thresholds. Check scratch bounds, initialization, output
aliasing and whether a split/merge path preserves the required normalization or
reduction semantics.

For async or cooperative changes, validate synchronization, participation and
buffer lifetime through prologue, tail and drain. Use device memory/race checking
where available. A successful compile and launch do not establish numerical or
synchronization correctness. Record errors/tolerances separately from timings;
discard invalid candidates before comparing performance.

## 7. Measure the device and the complete execution path

Warm up compilation, allocation and execution outside a steady-state timing
window. Synchronize completion or use device-side timing so asynchronous dispatch
is not mistaken for kernel latency. Time enough repetitions to resolve the change,
repeat samples, and retain the distribution rather than only the fastest run.
Interleave baseline and candidate runs when drift could affect the comparison.

State whether inputs are reused in cache or streamed, whether CUDA graphs are
used, what concurrent work is present, and what the timer includes. Measure all
extra launches, zeroing, scratch traffic and merge work in a split implementation.
Use unprofiled runs for final timings, and check the application path as well as
the isolated kernel when the objective requires it. Keep cold-start/compilation
cost separate if it matters to the use case.

If the difference is within run-to-run variation, retain the simpler established
policy or collect more evidence. If a candidate wins only for part of the workload,
identify a workload/resource condition that can select it and test the boundary.
Do not claim a gain on unmeasured devices or from an analytical score alone.

## 8. Integrate the policy and leave reproducible evidence

Put the winning selection rule next to the generator using existing configuration
fields and capability checks. Preserve a correct path for unsupported features,
unknown properties and untested workload regions. Keep measured thresholds and
compiler workarounds local to the affected kernel; do not turn them into hardware
facts in `Capabilities.zig`. Keep instruction legality in the backend. If generated
code is cached, verify that the existing cache identity distinguishes all changed
configuration, target and compiler inputs that affect that code.

Leave a compact experiment record with the change:

- Revision, generator/backend, exact build/test/benchmark commands, compiler and
  driver versions/options, device/partition identity, visible-device selection and
  the capability scope used. Record missing properties explicitly.
- Workload shapes after sharding, layouts, dtypes, live-length distribution and
  numerical contract; baseline and candidate configurations.
- Hypothesis, predicted resource changes, compiled-resource evidence and artifact
  locations; correctness cases and observed errors.
- Timer scope, warmups, iterations/samples, cache/concurrency/graph conditions,
  baseline/candidate distributions and units, plus regressions or uncertain cases.
- Final dispatch condition, fallback behavior, tested devices/workload range and
  the evidence needed to revisit the decision after a compiler or kernel change.
