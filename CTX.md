# Adaptive Vectored DmaMapped Loader Context

Snapshot: 2026-09-01. ZML HEAD is `60c56f10` (`faster dma bench 2`). The
optional-NUMA fallback is uncommitted in `CTX.md`, `examples/io/main.zig`, and
`zml/io.zig`. The user's benchmark recordings remain untracked and must be
preserved. The user moved the XLA checkout back to the intended revision; it
is clean at `92e7778d04` (`[XLA:GPU][oneAPI] Recognize
DmaMapped host memory as pinned`) on top of `b0990b33b1`
(`[XLA:GPU][oneAPI] Enable PJRT_Client_DmaMap for SYCL`). ZML selects the
user-built `pjrt-oneapi_linux-amd64-2026-07-21_22-43.tar`, whose configured
SHA-256 is
`91172bd90d59ab2f08d0c53e282843903fbed8ffb7ac2c7ccb215703f70af6a7`.
This artifact contains the range-aware SYCL `IsHostMemoryPinned` fix and has
been runtime-validated.

This file is the authoritative handoff for the current implementation and
measurements. `RESEARCH.md` is historical controller research; its adaptive
staging architecture is retired.

## Faster v5 DMA calibration (2026-09-01)

The cold calibration path was shortened without restoring the retired adaptive
controller or adding a persistent cache. The final algorithm keeps the complete
block grid and three 250 ms samples per measured local candidate. It reuses the
selected block's width-eight samples as the width baseline, then probes larger
widths in order and stops after two consecutive candidates leave the selected
smallest-within-5%-of-peak width unchanged. On the MI300X this measures widths
12 and 16, then stops instead of also measuring 24 and 32.

The selected tuple is no longer measured in isolation three times on every
otherwise-identical GPU. Those runs did not affect the tuple. The first
uncapped aggregate measurement now creates and warms the remaining device
cohorts concurrently, exercises the chosen tuple on every GPU, and supplies
the reported per-device rates. Capped fairness is Jain fairness over each
device's capped/uncapped retention, so it directly rejects a cap that penalizes
one shard without requiring isolated baselines.

Every global-cap candidate is measured in rotated order for three 250 ms
rounds. A cap is never emitted from that screen alone: a provisional
recommendation receives three additional cap/uncapped pairs, and the final
decision uses the median same-repeat throughput ratio. This caught a real
false-positive prototype run where a temporarily slow 214.6 GiB/s uncapped
median made cap 48 appear to reach 261.9 GiB/s. The paired confirmation path
subsequently measured cap 48 at 271.5 GiB/s and uncapped at 267.9 GiB/s; the
gain and latency reduction were insufficient, so no cap was emitted.

Five-round staged cap pruning was tested and removed. It saved only about
1--1.5 seconds while adding substantial state and candidate bookkeeping. The
final full three-round screen is shorter and simpler, while paired confirmation
protects the consequential action. A coarse block grid was also rejected: the
tensor-tail response need not be unimodal, and its sub-second potential saving
did not justify weakening the most important decision. Repeat counts are now
fixed internally at three; the repeat-count environment options were removed.

The block near-peak tolerance is now 8%, down from 15%. Three interleaved real
single-GPU loads of the 58.25 GiB Gemma workload measured these medians with
fixed 32 MiB reads, sixteen reads, and eight DMA events:

| DMA block | median loader GiB/s |
|---:|---:|
| 8 MiB | 23.84 |
| 16 MiB | 24.90 |
| 32 MiB | 25.43 |

Sixteen MiB stays approximately 2% from the loader peak without choosing the
largest transaction. More importantly, the replicated eight-GPU load committed
466.01 GiB in 7.829 seconds at 16 MiB, versus 10.694 seconds at 8 MiB. The old
15% synthetic preference could still call those points tied, so it was not
accurate enough for the multi-GPU workload.

The final v5 calibration matrix used automatic NUMA-local pools unless marked
explicit. Every run selected 16 MiB, eight events per device, and no global
cap:

| visible GPUs | NUMA placement | elapsed | sampling | windows | uncapped GiB/s |
|---|---|---:|---:|---:|---:|
| `0` | node 0 | 8.17 s | 5.29 s | 21 | 46.44 |
| `7` | node 1 | 8.05 s | 5.30 s | 21 | 44.91 |
| `0,1` | node 0 | 12.24 s | 9.09 s | 36 | 89.55 |
| `0,4` | nodes 0/1 | 12.96 s | 9.12 s | 36 | 86.15 |
| `0,1,2,3` | node 0 | 15.85 s | 10.61 s | 42 | 154.43 |
| `0,1,4,5` | nodes 0/0/1/1 | 13.70 s | 9.09 s | 36 | 174.56 |
| `0,1,4,5` | explicit 0/0/1/1 | 13.65 s | 9.09 s | 36 | 171.45 |
| all eight | 0/0/0/0/1/1/1/1 | 19.76 s | 9.11 s | 36 | 325.46 |

The 42-window four-GPU run entered paired confirmation and correctly rejected
the provisional cap. Compared with v4, the representative cross-node two-GPU,
split four-GPU, and all-eight wall times fell by 27.7%, 33.6%, and 25.9%.
Eight-GPU sampling fell from 19.16 to 9.11 seconds and windows from 76 to 36.
Setup rose from 7.48 to 10.65 seconds because the more accurate 16 MiB tuple
requires larger disjoint NUMA-local source rings; accuracy was kept ahead of
the more flattering 8 MiB calibration time.

The benchmark output format is now version 5. Per-device output reports
`measured_gib_s` from the concurrent uncapped cohort, while the global record
reports `uncapped_gib_s`. The old isolated/scaling fields and repeat-count
environment controls were removed. After all detailed records, a final
`dma_bench_total elapsed_ms=...` line repeats the total calibration time.

NUMA mapping is optional as a follow-up to v5. An empty
`DmaBenchmarkOpts.device_numa_nodes` now means that every device shares one
DmaMapped source pool; that pool uses the ordinary allocator and does not call
`mbind` or verify placement with `move_pages`. Explicit or successfully
discovered mappings retain one strictly bound and verified pool per node. The
CLI treats non-ROCm platforms, non-Linux hosts, and failed automatic discovery
as the single-pool case (while still propagating allocation failure), reports
`source_layout=single_pool numa_mapping=single`, and prints assumed node zero
for each device.

Short 10 ms-window ROCm smoke runs exercised both modes. A nonnumeric GPU UUID
visibility mask intentionally made topology discovery unavailable; the
benchmark completed with one MI300X and with two MI300X devices spanning the
physical NUMA nodes, using one 16--32 MiB growing source pool. A numeric `0,4`
mask still created and verified separate node-zero and node-one pools.

Validation completed after the final implementation:

- `zig fmt --check zml/io.zig examples/io/main.zig`
- `git diff --check`
- `bazel test //zml:test --test_output=errors`
- `bazel build -c opt --@zml//platforms:rocm=true \
  --@zml//platforms:cpu=false //examples/io:playground`
- `bazel build -c opt --@zml//platforms:oneapi=true \
  --@zml//platforms:cpu=false //examples/io:playground`

The sharded Gemma command cannot be used as another workload control because
its tensor dimensions are incompatible with the example's eight-way explicit
axis sharding. The prior four-B70 oneAPI cap-four measurements remain the
available positive cap case; oneAPI hardware was not present for this v5 run.
The paired-confirmation policy must be rechecked there when that host is
available. Logs for this work use `/tmp/zml-dma-final-v5-*` and
`/tmp/zml-load-{fast,8gpu}-block*.log`.

## v4 deterministic DMA calibration and automatic NUMA pools (2026-09-01)

The DMA benchmark now has one accuracy-first calibration path instead of the
separate adaptive and exhaustive controllers. Every block-size and device-width
candidate is measured in rotated order, and the smallest candidate within the
configured tolerance of the median peak is selected. Device-local candidates
use three 250 ms windows. Global-cap candidates use five 250 ms windows and
retain all per-device metrics from the median aggregate repeat. There are no
confidence intervals, rank-stability exits, finalist phases, dynamic budgets,
or `budget_exhausted` states.

The eight-event block screen is also the minimum width recommended to the real
loader. Smaller widths remain in the list only as global-cap candidates. This
is deliberate: the perfectly fed synthetic ring can make two events appear
within 5% of peak, but it does not include the loader's source and callback
scheduling gaps. The benchmark still verifies the selected tuple in isolation
on every otherwise-identical device, so global fairness uses measured
per-device rates instead of copying the first device's rate.

On ROCm/Linux, `examples/io` now discovers visible-device NUMA placement from
KFD topology and DRM sysfs. Numeric `ROCR_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, and `CUDA_VISIBLE_DEVICES` masks
are applied before mapping devices to nodes. One DmaMapped source pool is bound
and first-touched per detected node, disjoint rings are assigned to its local
devices, and sampled page placement is verified. `ZML_DMA_BENCH_NUMA_NODES`
remains the explicit override and is required for non-ROCm backends or
nonnumeric visibility masks.

The final default search consistently selected 8 MiB and eight events on the
MI300X host across these placements:

| visible GPUs | source placement | elapsed | uncapped GiB/s | recommendation |
|---|---|---:|---:|---|
| `0` | auto node 0 | 10.44 s | 42.55 isolated | 8 MiB / 8, no cap |
| `0,4` | auto nodes 0/1 | 17.92 s | 77.26 | 8 MiB / 8, no cap |
| `0,1,4,5` | auto nodes 0/0/1/1 | 20.64 s | 149.24 | 8 MiB / 8, no cap |
| `0,1,4,5` | forced node 0 | 20.42 s | 135.45 | 8 MiB / 8, no cap |
| all eight | auto nodes 0/0/0/0/1/1/1/1 | 26.65 s | 272.38 | 8 MiB / 8, no cap |

The local two-pool four-GPU result was 10.2% faster than forcing both sockets'
GPUs through node 0. The final eight-GPU run used 7.48 s of setup and 19.16 s
of sampling over 76 windows. A 100 ms local window was rejected as a default:
despite five repeats, ROCm pauses made it incorrectly select 32 MiB once.
Three 250 ms repeats restored the known 8 MiB optimum while keeping total time
near the earlier long accuracy-oriented runs. Global-cap selection also has a
small hysteresis margin around the two-times latency rule, preventing a noisy
almost-half result from becoming a machine-wide runtime constraint.

The core DMA benchmark section in `zml/io.zig` is approximately 900 lines
smaller than the previous implementation even after retaining the fair gate
and adding NUMA source pools.

## Fair global DMA benchmark follow-up (2026-08-31)

The `dma-bench` global-cap prototype now uses benchmark-local fair admission;
the loader and `LoadOpts` are unchanged. The old implementation put all device
workers behind one ordinary condition-variable gate. At a four-transfer cap,
arbitrary wakeups let one B70 receive 8.21 GiB/s while another received 24.63
GiB/s in the same repeat, so aggregate throughput hid temporary device
starvation. Isolated and nominally uncapped aggregate runs also crossed that
gate even though no admission limit was needed.

Capped runs now reserve grants per device under a weighted max-min scheduler.
It balances active slots relative to each device's calibrated width, rotates
ties, time-shares caps below the device count without starvation, and lends
slots when a device has no waiter. Isolated and uncapped measurements bypass
global admission entirely. Median aggregate results retain every device's
metrics from the same repeat rather than combining independently selected
medians.

Global candidates report minimum per-device retention relative to the
uncapped concurrent run and Jain fairness over each device's fraction of its
isolated rate. Selection requires aggregate throughput within 2% of the
measured peak, at least 95% retention on every device, and fairness of at least
0.98. Candidate caps include values below the device count and balanced
per-device multiples; a cap is still emitted only when it materially improves
throughput or halves latency while staying inside the 2% throughput band.

The exact four-B70 Qwen3.6-27B sharded command was repeated three times after
the change. Every run selected the same 4 MiB/eight-event per-device tuple and
recommended a fair global cap of four:

| run | uncapped GiB/s | cap-4 GiB/s | cap-4 minimum retention | cap-4 fairness |
|---:|---:|---:|---:|---:|
| 1 | 79.921 | 79.606 | 0.9831 | 0.9997 |
| 2 | 79.955 | 79.363 | 0.9783 | 0.9997 |
| 3 | 79.994 | 79.706 | 0.9852 | 0.9995 |

Cap four now gives each continuously backlogged B70 approximately one active
transfer and 19.4--20.6 GiB/s instead of allowing the old 8--32 GiB/s
per-device imbalance. Caps 8--24 remain near 80 GiB/s but only inflate event
latency: representative cap-4, cap-8, and uncapped latencies were 0.152,
0.303, and 1.225 ms. Removing the unnecessary gate from the uncapped path did
not raise its approximately 80 GiB/s result, so the large gap from the roughly
168--170 GiB/s sum of isolated rates is a real shared oneAPI/PJRT/host-path
ceiling rather than global-gate lock contention.

Validation passed:

```text
bazel test //zml:test --test_output=errors
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 bazel run --config=release \
  --@zml//platforms:oneapi=true //examples/io:playground -- \
  dma-bench /var/models/Qwen/Qwen3.6-27B/ sharded
```

## ROCm DMA benchmark NUMA source pools (2026-09-01)

The eight-MI300X host has two NUMA nodes, with devices 0--3 local to node 0
and devices 4--7 local to node 1. Memory placement has little effect on an
isolated device's DMA bandwidth on this machine, as the raw HIP controls below
also show, but it materially changes aggregate bandwidth when all eight GPUs
run concurrently. Binding the shared benchmark source to only one node favored
the four local GPUs by approximately 29--42% and limited aggregate throughput
to roughly 180--186 GiB/s. Page-interleaving one shared arena across both nodes
improved aggregate throughput to approximately 215--228 GiB/s, but every GPU
still fetched half of its pages remotely.

The benchmark prototype now supports the intended design: one DmaMapped source
pool per NUMA node, with each device using a disjoint source ring in its local
pool. The allocator applies `MPOL_BIND` with `mbind` before first touch, faults
the pages with `memset`, and samples up to 256 pages with `move_pages`; the
benchmark fails if any sampled page is not on the requested node. On this host,
the 8 MiB/eight-event device tuple allocates a 256 MiB pool on each node, for
512 MiB total pinned source memory. Both pools passed the 256-page placement
check.

Longer combined runs with the node-local pools reached **251--263 GiB/s**,
approximately 10--22% faster than the interleaved arena and typically about
15% faster. Fairness remained approximately 0.999. The two pools measured
155.591 and 163.165 GiB/s separately when uncapped, or 318.756 GiB/s summed,
so the simultaneous result still loses 17--21% to a real shared host/PJRT
ceiling. The important distinction is that NUMA-local pools raise that ceiling;
they do not eliminate it.

The longer aggregate cap sweep produced these representative results:

| total active transfers | aggregate GiB/s, two runs |
|---:|---:|
| 8 | 173.322, 158.554 |
| 16 | 237.236, 224.079 |
| 32 | 233.592, 230.830 |
| 48 | 252.164, 257.559 |
| 64 / uncapped | 262.763, 250.993 |

Caps 48 and 64 exchanged position between runs, so there is no stable
beneficial machine-wide admission cap below the natural sum of the per-device
widths. The accuracy-first recommendation remains an 8 MiB DMA chunk, eight
events per device, and no global cap. A total cap of 48 is an economical point
near the plateau, but it was not consistently within the configured 3% band
and should not be emitted as an optimum.

Short 20 ms screening and 50 ms confirmation windows incorrectly selected a
global cap of 32 in two runs because ROCm/PJRT pauses contaminated individual
windows. The longer 100 ms screening, 250 ms confirmation, 3 s target, and 8 s
maximum runs took about 13.5 seconds end to end: approximately 6.7 seconds of
PJRT setup and 6.8 seconds of sampling. The NUMA-pool result is reliable, but
the fast selector still needs more robust outlier handling before its global
cap decision is trusted from the default short run.

The prototype is enabled with an explicit device-index-to-node mapping in
PJRT device order:

```text
ZML_DMA_BENCH_NUMA_NODES=0,0,0,0,1,1,1,1 \
ZML_DMA_BENCH_BLOCK_MIB=8 \
ZML_DMA_BENCH_PARALLELISM=1,2,4,6,8 \
bazel run -c opt \
  --@zml//platforms:rocm=true \
  --@zml//platforms:cpu=false \
  //examples/io:playground -- \
  dma-bench /var/models/google/gemma-4-31B-it/ replicated
```

`ZML_DMA_BENCH_NUMA_NODES` currently applies only to the adaptive benchmark.
An empty mapping preserves the original shared source arena. Automatic
device-to-NUMA discovery and production loader pool selection are not yet
implemented. The eventual loader design should keep one registered pool per
node, route each GPU to its local pool, and stage replicated data once per node
before fanning out to the GPUs local to that node. This aggregate result does
not contradict the single-device raw HIP result below: remote DMA is not
slower in isolation, but eight concurrent devices need both sockets' memory
and interconnect resources to reach the higher global rate.

## ROCm direct-loader addendum (2026-08-31)

This is a new-host/new-backend investigation and must not be mixed with the
oneAPI/B70 medians below. The machine has eight MI300X devices, two NUMA nodes,
and roughly 2 TiB RAM. The requested command is:

```text
ROCR_VISIBLE_DEVICES=7 bazel run --config=release \
  --@zml//platforms:rocm=true //examples/io:playground -- \
  load /var/models/google/gemma-4-31B-it/
```

The Gemma load contains 1,188 tensors and 58.251 GiB of logical weights. Before
the fix, ROCm selected `vectored=false` because `zml/io.zig` allowed the direct
loader only for CUDA and oneAPI. It therefore allocated ordinary full-tensor
host buffers through `BufferedMemoryWriter`, then called
`BufferFromHostBuffer` with `ImmutableUntilTransferCompletes`. Of the 1,188
tensors, 1,187 totaling 55.626 GiB (95.49% of the bytes) are below XLA's 1 GiB
staging cutoff and were copied into PJRT's pinned staging allocations before
H2D. The one 2.625 GiB embedding tensor was above that explicit cutoff.

The active ZML worktree now has both required ROCm changes:

- `zml/mem.zig` routes `.rocm` through `DmaMapAllocator`, so the shared lazy
  64 MiB slabs are registered with `PJRT_Client_DmaMap` rather than being
  passthrough allocations. This edit predated the final loader-selection edit
  and remains uncommitted/user-owned.
- `zml/io.zig` includes `.rocm` in the `direct` predicate. This routes ROCm
  through `loadVectored`, so file reads target the registered pool blocks and
  the ordinary `BufferedMemoryWriter` path is not used.

The release ROCm playground target builds successfully. A full fixed-command
run reported `vectored=true`, mapped one 64 MiB slab, used 28 MiB pinned
high-water, submitted and committed all 58.25 GiB in 30,650 transfers averaging
1.95 MiB, and completed in 7.309 s in-loader / 7.317 s outside, or 7.96 GiB/s.
A profiled repeat used 30 MiB high-water and completed in 7.201 s in-loader /
7.21 s outside, or 8.08 GiB/s.

The pre/post `perf` comparison proves that this bypasses the extra PJRT
host-side linearization/staging copy; it does not mean that the necessary H2D
DMA disappears:

| sampled stack frame | pre-fix buffered | post-fix vectored |
|---|---:|---:|
| `LinearizeHostBufferInto` | 5,466 | 0 |
| `CommonPjRtClient::Linearize` | 5,409 | 0 |
| `TransposePlan::Execute` | 10,817 | 0 |
| `__memmove_avx512_unaligned_erms` | 5,415 | 9 |

The recordings are `/tmp/zml-rocm-load-release.perf.data` and
`/tmp/zml-rocm-load-vectored-release.perf.data`; the profiled post-fix log is
`/tmp/zml-rocm-load-vectored-release.log`. In the post-fix flat report,
`_copy_to_iter` is 27.40%, KFD event-wait locking is visible, a PJRT
`std::vector<...>::_M_fill_insert` is 2.75%, and `compiler_rt.memset` has fallen
to 0.63%. No staging/linearization frame is sampled.

### Why the default is still only approximately 8 GiB/s

The first fixed-command result is dominated by transfer granularity and
software scheduling, not the physical PCIe link. The default is a 2 MiB DMA
block, eight initially active PJRT events, and twelve retained local request
lifecycles. The two representative default runs made 30,650 PJRT submissions;
average submission-to-callback latency was 1.273--1.791 ms. Eight times roughly
2 MiB divided by that callback latency predicts only about 8.7--12.3 GiB/s,
matching the observed range. The local adaptive controller changes source
request size, but it does not enlarge the physical DMA block.

A standalone HIP control used the same GPU and the matching
`hipHostRegister` path. It was compiled as a tiny temporary program, not as an
XLA build, and its source was removed after the run. With 8 GiB transferred per
point on one stream it reached:

| HIP copy size | registered H2D GiB/s |
|---:|---:|
| 2 MiB | 39.03 (40.94 with NUMA-1 binding) |
| 8 MiB | 47.73 (47.71) |
| 16 MiB | 49.19 (49.41) |
| 32 MiB | 49.39 (50.14) |
| 64 MiB | 50.67 (50.74) |
| 128 MiB | 48.71 (50.89) |

GPU 7 is on NUMA node 1 at PCIe 32 GT/s x16. It was idle and had no VRAM
allocation after the tests. Thus neither host registration, ROCm's copy
engine, nor PCIe is an 8 GiB/s hardware ceiling.

A one-run diagnostic sweep fixed twelve request lifecycles and eight DMA
events, with the read request equal to the DMA block:

| request/block | submissions | wall GiB/s | average DMA latency |
|---:|---:|---:|---:|
| 2 MiB | 30,650 | 10.87 | 0.923 ms |
| 8 MiB | 8,390 | **18.04** | 1.258 ms |
| 16 MiB | 4,607 | 16.14 | 2.393 ms |
| 32 MiB | 2,752 | 17.08 | 4.094 ms |

Eight MiB is the best observed granularity on this Gemma tensor distribution,
although these are screening runs rather than balanced medians. Larger nominal
blocks have many partial tensor-tail transfers and raise source latency. At
8 MiB, raising the fixed DMA width or retained request width did not help:

| fixed reads | fixed DMA | wall GiB/s |
|---:|---:|---:|
| 12 | 4 | 13.85 |
| 12 | 16 | 14.19 |
| 12 | 32 | 14.07 |
| 24 | 8 | 14.32 |
| 24 | 16 | 12.13 |
| 48 | 16 | 12.06 |

The earlier 12/8/8 point was 18.04 GiB/s. This variability is real host noise,
not evidence that 16 events are faster: during the sweep another user's
`llmd` consumed roughly 3,245% CPU, an XLA ROCm PJRT test was active, NUMA node
0 had only about 3 GiB free, and several Bazel servers were busy. GPU 7 itself
was unused; the competing jobs occupied GPU 1. Repeat medians require a quiet
host before selecting a new default.

The remaining PJRT path is understood structurally. ZML creates one
`AsyncHostToDeviceTransferManager` and one final device allocation per tensor,
not one device allocation per chunk. Every chunk nevertheless crosses the C
API, takes the common transfer-manager mutex, creates a promise/future and
PJRT event, creates a `BufferSequencingEvent`, queues the memcpy on the one H2D
stream, allocates/records a ROCm event, schedules a host callback, crosses the
C API event callback again, and re-enters ZML's mutex-driven ready-queue pump.
More than eight outstanding events raises callback latency instead of opening
more link bandwidth.

The follow-up kept the physical DMA block at 2 MiB and eight PJRT events while
changing only the local source batch. The submission count therefore remained
exactly 30,650 at every point:

| read request | fixed reads | wall GiB/s | average read latency | average PJRT latency | pinned high-water |
|---:|---:|---:|---:|---:|---:|
| 2 MiB | 12 | 7.43 | 0.801 ms | 1.971 ms | 24 MiB |
| 8 MiB | 12 | 14.42 | 2.903 ms | 0.994 ms | 96 MiB |
| 32 MiB | 12 | 17.72 | 10.270 ms | 0.773 ms | 384 MiB |
| 64 MiB | 12 | **18.31** | 17.629 ms | 0.737 ms | 768 MiB |
| 128 MiB | 12 | 17.29 | 28.880 ms | 0.767 ms | 1.50 GiB |

This proves that the default is not slow solely because it makes 30,650 PJRT
calls: all five rows make the same number. Coupling one tiny source syscall and
one retained lifecycle to each tiny DMA block creates harmful interleaving and
does not keep a useful ready cohort in front of the H2D stream. Batching source
reads amortizes kernel/VFS scheduling and lets the newest-ready queue feed
PJRT continuously; callback latency itself drops by about 2.7x. The request
size and DMA block should therefore be tuned independently on ROCm.

At a 32 MiB source request and unchanged 2 MiB/eight-event DMA configuration,
the retained-width screen was:

| fixed reads | wall GiB/s | average PJRT latency | pinned high-water |
|---:|---:|---:|---:|
| 4 | 11.61 | 0.720 ms | 128 MiB |
| 8 | 16.76 | 0.718 ms | 256 MiB |
| 12 | 17.72 | 0.773 ms | 384 MiB |
| 16 | **18.77** | 0.752 ms | 512 MiB |
| 24 | 17.63 | 0.810 ms | 720 MiB |
| 32 | 17.35 | 0.812 ms | 832 MiB |

At 64 MiB, sixteen and 24 reads reached 17.96 and 16.62 GiB/s while consuming
1.00 and 1.50 GiB pinned. The 32 MiB/16-read tuple is therefore the better
2 MiB-DMA point.

The file-only control matters because two sequential `dd` processes reached
only 5.3--7.4 GiB/s and initially looked source-bound. That comparison was
invalid: it had two sequential readers while ZML used 16 positional readers.
A matched control split the 46 GiB shard across twelve disjoint range readers
and the 12 GiB shard across four. Two rounds read all 58.25 GiB in 1.68 and
1.65 seconds, or approximately 34.7 and **35.3 GiB/s**. The source stage and
the raw 2 MiB registered-H2D stage (39--41 GiB/s) are each roughly twice as
fast in isolation as the best 2 MiB-DMA loader point.

A load-window-only profile at 32 MiB reads, 2 MiB DMA blocks, sixteen reads,
and eight events completed at 18.25 GiB/s. Flat cycles were `_copy_to_iter`
49.84%, `filemap_get_read_batch` 5.00%, `filemap_read` 4.10%,
`kfd_wait_on_events` 1.18%, and no staging copy. The recording is
`/tmp/zml-rocm-r32-dma2-r16-d8-load-window.perf.data`. A whole-process profile
and stat run are also retained as `/tmp/zml-rocm-r32-dma2-r16-d8.perf.data`
and `/tmp/zml-rocm-r32-dma2-r16-d8.perf.stat`, but they include roughly eight
seconds of PJRT/BFC initialization around a 2.9-second logged weight-transfer
phase. In that whole-process profile, the 2.93% `std::vector::_M_fill_insert`
stack is `BFCAllocator::RegionManager::AddAllocationRegion` during device
allocation, not per-chunk transfer work.

The request/block controls were then separated correctly. With a fixed 32 MiB
source request, sixteen retained reads, and eight PJRT events:

| DMA block | PJRT submissions | wall GiB/s | average PJRT latency |
|---:|---:|---:|---:|
| 2 MiB | 30,650 | 19.88--19.97 profiled repeat | 0.704--0.723 ms |
| 4 MiB | 15,804 | 21.87 | 1.244 ms |
| 8 MiB | 8,390 | **23.79** | 1.893 ms |
| 16 MiB | 4,607 | 19.24 | 3.381 ms |

Eight MiB is the current combined optimum: it retains the useful large source
batch while removing almost three quarters of the C-API/event/callback
round-trips. Sixteen MiB has too many tensor-tail fragments and raises both
source and completion latency.

At 32 MiB reads and 8 MiB blocks, the last concurrency screen was:

| fixed reads | fixed DMA | wall GiB/s | average PJRT latency | pinned high-water |
|---:|---:|---:|---:|---:|
| 12 | 8 | 17.72 | 1.969 ms | 384 MiB |
| 16 | 4 | 22.64 | 1.151 ms | 512 MiB |
| 16 | 8 | **23.79** | 1.893 ms | 512 MiB |
| 16 | 12 | 22.96 | 3.247 ms | 512 MiB |
| 20 | 8 | 22.54 | 2.232 ms | 640 MiB |
| 24 | 8 | 23.77 | 2.213 ms | 768 MiB |

Interleaved repeats resolve the apparent tie. Including the screening run,
sixteen reads produced 23.79, 24.59, and 24.80 GiB/s: median **24.59 GiB/s**.
Twenty-four reads produced 23.77, 23.04, and 21.13 GiB/s: median
**23.04 GiB/s**, while requiring 768 rather than 512 MiB pinned. At the
selected sixteen/eight point, adjacent 16 and 64 MiB source requests reached
22.19 and 18.64 GiB/s in one-run screens; 32 MiB remains the clear candidate.

This approximately triples the original 8.0 GiB/s result and reaches about
70% of the matched 35.3 GiB/s file-only ceiling. It is still below the raw HIP
copy ceiling because the complete pipeline must do the page-cache copy and H2D
concurrently and still creates 8,390 PJRT/C-API/event/callback transactions.
Further material improvement requires lower-overhead/batched PJRT completion
or a different transfer API, not more event concurrency. A quiet-host repeat
is still appropriate before changing global defaults, especially because the
existing oneAPI results below prefer 2 MiB local requests/blocks.

The current explicit ROCm diagnostic command is:

```text
ROCR_VISIBLE_DEVICES=7 \
ZML_LOAD_DMA_BLOCK_MIB=8 \
ZML_LOAD_READ_REQUEST_MIB=32 \
ZML_LOAD_FIXED_READ_PARALLELISM=16 \
ZML_LOAD_FIXED_DMA_PARALLELISM=8 \
bazel-bin/examples/io/playground load /var/models/google/gemma-4-31B-it/
```

### Why the adaptive controller did not find this tuple

"Adaptive" currently applies to three credits/shape controls: retained read
parallelism, per-device DMA parallelism, and source request size. It does not
apply to `dma_block_size`. The `DmaBlockPool` is constructed once with that
fixed size, whose default remains 2 MiB. Consequently, no controller decision
can change the default run's 30,650 PJRT event/callback transactions into the
8,390 transactions produced by the winning 8 MiB block. This is the largest
ROCm-specific gain and it is outside the controller's search space.

The source-size search is also not an independent size sweep. A size candidate
doubles one step at a time, but `modeledReadConcurrency` simultaneously
replaces the retained read width using a per-request service-time model. In the
profiled default run the controller did the following:

```text
start                 reads=12 request=2 MiB DMA=8
read reduction probe  reads=6  request=2 MiB DMA=8  -> rollback
size probe            reads=4  request=4 MiB DMA=8  -> keep
read growth probe     reads=6  request=4 MiB DMA=8  -> keep
finish                reads=6  request=4 MiB DMA=8
```

Thus the 2-to-4 MiB probe did not compare 12/2 against 12/4; it compared 12/2
against 4/4. The model observes almost zero local TTFB and derives a narrow
width from the sum of individual read service rates. That is not the same as
the measured aggregate page-cache ceiling with sixteen disjoint readers. The
fixed 32 MiB-request screen makes the consequence concrete: four retained
reads reached 11.61 GiB/s, while sixteen reached 18.77 GiB/s with the same
2 MiB DMA block and eight DMA credits. Larger request candidates can therefore
be handicapped by an overly narrow modeled width before their batching benefit
is measured.

The controller is a cautious adjacent-probe controller, not a global tuple
optimizer. Read growth requires greater than 10% measured DMA starvation,
saturated reads, and no ready pressure. DMA growth additionally waits for the
source tuple to settle and for representative fed/saturated windows. Size
growth is sequential (2, 4, 8, 16, 32 MiB), probes require representative
bytes/requests, and completed or rejected probes impose cooldowns. The default
load lasts only about 7.2 seconds and then enters a finite-tail region where
new probes are suppressed. It therefore finished after 4 MiB rather than ever
testing the useful 8/16/32 MiB source sizes. Its DMA width remaining at eight
was reasonable: the fixed screens show that more simultaneous PJRT events
usually raise callback latency rather than bandwidth.

Two follow-up controls show that concurrency adaptation itself is alive once
the missing batching choices are supplied:

| externally supplied setting | controller trajectory | wall result |
|---|---|---:|
| DMA block 8 MiB | reads 12 -> 15, request stays 8 MiB, DMA stays 8 | 12.11 GiB/s |
| DMA block 8 MiB, initial request 32 MiB | reads 12 -> 18, request stays 32 MiB, DMA stays 8 | 18.41 GiB/s |

These two controls were run while the host was still noisy, so their absolute
rates must not be compared directly with the earlier quietest fixed median of
24.59 GiB/s. Their controller decisions are the relevant result: it can tune
credits around a viable starting granularity, but cannot select the physical
DMA transaction size, and its coupled source-size/width probe does not explore
the ROCm optimum reliably. The logs are
`/tmp/zml-block8-adaptive.log` and
`/tmp/zml-block8_req32-adaptive.log`.

### Proposed DMA preflight calibration

DMA block size and PJRT event width are backend/device properties and should
be calibrated before weight reads, then fixed for the load. Use the exact
`AsyncHostToDeviceTransferManager.transferData` path with prefilled DmaMapped
memory; do not use HIP or `BufferFromHostBuffer`. Model metadata/sharding is
already available, so synthetic submissions can reproduce the model's
full-block/tensor-tail distribution without reading weights. A bounded ring
may rewrite a destination only after its preceding event completes.

Screen 2/4/8/16/32 MiB blocks, then widths near the best block
(1/2/4/8/12/16/24/32), and select the smallest tuple within 3--5% of peak.
Short 50--100 ms windows contain hundreds of callbacks; interleave three
repeats for the top two tuples. Expected uncached cost is approximately 2--4
seconds; a cached tuple needs only a 200--500 ms neighbor check. Measure after
compilation and `warmupDeviceAllocators`, immediately before `loadBuffers`, so
device autotuning cannot contaminate it. Cache keys must include PJRT/plugin,
backend/device, visible topology, driver/runtime, and host identity.

Calibrate per-device width, then validate all devices simultaneously. Do not
add or search a global DMA cap unless aggregate throughput falls materially
(approximately 10%) below isolated scaling or callback latency explodes.
Independent PCIe links/copy engines should scale; the raw GPU-0/GPU-7 control
already reached 96.86 GiB/s aggregate from 48.43 GiB/s unique throughput.
Possible shared limits are PCIe switches/root complexes, host memory/IOMMU,
inter-socket fabric, and PJRT callback/submission CPU. The current approximately
35 GiB/s source ceiling is far below eight GPUs' aggregate raw DMA capacity,
and the global pinned-memory limit already bounds buffering. With no collapse,
the effective total limit is the sum of per-device limits and no global gate
is needed.

After preflight, keep DMA block/width fixed and adapt only source behavior
(read concurrency and, if retained, request size). Request-size probes must be
decoupled from modeled read-width changes; the current coupled probe handicaps
larger source batches. First implement an `examples/io` playground calibration
command and verify that it selects the measured ROCm 8 MiB/eight-event
neighborhood before automatic loader integration.

### NUMA diagnosis of the later 16 GiB/s result

A load-window-only `perf` recording of the fixed 32 MiB-request, 8 MiB-block,
sixteen-read/eight-DMA tuple completed at 16.27 GiB/s and attributed **74.17%**
of sampled cycles to `_copy_to_iter`. `filemap_get_read_batch` was another
7.44% and `filemap_read` 5.82%. The recording is
`/tmp/zml-rocm-r32-block8-r16-d8-load-window-3.perf.data`; its log is the same
path with `.perf.data` replaced by `.log`. This is direct evidence that the
slow repeat is in the page-cache-to-registered-buffer copy, not a resurrected
PJRT staging copy.

The topology check showed GPU 7 (`0000:dd:00.0`) on NUMA node 1, whose CPUs are
the odd-numbered logical CPUs. The unbound process could run read workers on
both sockets and allocate/first-touch the lazily created registered slabs on
either socket. Because pool blocks are subsequently recycled among arbitrary
workers, this permits remote CPU writes into the pinned destination. It also
permits GPU 7 to DMA from node-0 memory, although the raw HIP control below
subsequently proved that remote GPU DMA is not itself slower on this machine.
A combined CPU-and-memory locality control used:

```text
numactl --cpunodebind=1 --membind=1 env \
  ROCR_VISIBLE_DEVICES=7 \
  ZML_LOAD_DMA_BLOCK_MIB=8 \
  ZML_LOAD_READ_REQUEST_MIB=32 \
  ZML_LOAD_FIXED_READ_PARALLELISM=16 \
  ZML_LOAD_FIXED_DMA_PARALLELISM=8 \
  bazel-bin/examples/io/playground \
  load /var/models/google/gemma-4-31B-it/
```

Two unprofiled bound runs reached **23.56 and 26.40 GiB/s**, versus the two
immediately preceding unbound repeats at 17.00 and 16.27 GiB/s. A bound
load-window profile reached **26.89 GiB/s** and attributed 68.18% of cycles to
`_copy_to_iter`, 7.69% to `filemap_get_read_batch`, and 4.20% to
`filemap_read`. It is retained at
`/tmp/zml-rocm-numa1-r32-block8-r16-d8-load-window.perf.data` with the matching
`.log`. At this stage the combined binding test appeared to make NUMA locality
the main difference between the 16 GiB/s and approximately 27 GiB/s
observations, but it did not separately quantify CPU affinity, memory policy,
page-cache state, or concurrent host load. The later placement matrix below
shows that this initial causal claim was too strong.

This is not yet a ZML code fix. A production fix should make registered slab
allocation and the workers filling those slabs NUMA-local to each other. The
raw HIP control below shows that the slab does not also need to be local to the
destination GPU on this host. Per-node pools/read lanes may still use both
sockets' CPU and memory bandwidth for a multi-device load, but they are not
required merely to avoid remote GPU DMA. The external `numactl` prefix is a
valid single-visible-GPU workaround.
The host was also busy during these controls: two `llmd` processes consumed
roughly eighteen CPU cores in aggregate and another user was compiling ROCm
XLA kernels, so quiet-host medians may be higher.

The locality can be detected in-process without a direct HIP or ROCm-SMI
dependency. XLA's ROCm `DeviceDescription` already obtains the PCI BDF and
reads its sysfs NUMA node. `StreamExecutorGpuDevice` then publishes that value
as the PJRT device attribute named `numa_node`; ZML already exposes PJRT
description attributes through `Device.pjrt_desc.attributes`. The cached
ROCm plugin contains both `numa_node` and `pci_bus_id` support strings, and the
current XLA source has an explicit PJRT NUMA-attribute test for CUDA, ROCm, and
SYCL.

The clean single-device implementation is not merely to pin the `llmd` main
thread. The loader's concurrent `preadv` calls execute on the shared Zig IO
worker pool, and the 64 MiB registered slabs are lazily allocated and
first-touched by those workers. A dedicated load executor could discover the
PJRT `numa_node`, read and intersect that node's sysfs CPU list with the
process's allowed affinity, spawn its read workers with that CPU affinity, and
bind slab pages to the same node. Linux CPU affinity is per thread; memory
policy can be applied per thread with `set_mempolicy`, or more robustly to each
slab range with `mbind` before first touch. If the existing shared IO pool is
retained, any temporary affinity/policy must be restored after the read so
unrelated work is not accidentally pinned.

For multiple visible GPUs, blindly applying one process-wide policy remains
undesirable because it prevents using both sockets' CPU/memory resources. A
general high-throughput design can use per-NUMA-node slab pools and node-local
read admission, but a filled block may be transferred to a GPU on either node.
Missing/negative attributes and single-node hosts should silently retain the
current behavior, with an explicit override available for operators. This
belongs in `zml.io.load`/`zml.mem`, not only in `llmd`, so every loader caller
receives the same locality behavior.

NVIDIA exposes the same attribute. CUDA's StreamExecutor obtains its PCI BDF,
reads the sysfs NUMA node into `DeviceDescription`, and then uses the same
generic `StreamExecutorGpuDevice` code that publishes PJRT `numa_node` for
CUDA, ROCm, and SYCL. A ZML implementation based on the PJRT attribute is
therefore backend-neutral across NVIDIA, AMD, and oneAPI rather than a ROCm
special case.

The eight MI300X devices on this host form an exact 4+4 split:

| GPUs | PCI buses | NUMA node |
|---|---|---:|
| 0--3 | `1b`, `3d`, `4e`, `5f` | 0 |
| 4--7 | `9d`, `bd`, `cd`, `dd` | 1 |

The current loader cannot exploit that split correctly with one affinity
change. `DmaBlockPool.init` selects `platform.devices[0]` and creates one
global pool, while `VectoredReadScheduler` has one global job stream. A
request plan can scatter different blocks to different sharded destinations;
a mirrored/replicated block has a writer mask containing several devices and
may span both NUMA nodes. One `preadv` executing on one CPU cannot be local to
destination pages on both sockets.

A scalable eight-GPU design can create one registered slab pool and
read-worker/admission lane per node so the page-cache copy is always local and
both sockets contribute CPU/memory bandwidth. PJRT DMA credits remain tracked
separately per GPU as they are today, while global pinned-memory and source
concurrency budgets can be divided or adaptively shared between the read
lanes. A replicated block should be read once into either node-local lane and
DMAed directly to all destinations, including the remote socket; the raw
control below shows that duplicating it through a cross-node CPU copy is much
slower. Ordinary sharded blocks may be balanced between lanes for source-stage
throughput rather than assigned according to the destination GPU's node.

### Raw HIP cross-NUMA and replication control (2026-08-31)

A standalone C++/HIP benchmark, with no XLA build or PJRT involvement, bound
two 256 MiB anonymous arenas with `mbind`, first-touched them on their intended
nodes, and registered both with `hipHostRegister`. A `move_pages` query sampled
every 2 MiB and confirmed all 128 samples of each arena on the requested node.
GPU 0 was `0000:1b:00.0` on node 0 and GPU 7 was `0000:dd:00.0` on node 1.
Each point transferred 8 GiB and three complete runs were made. At the
loader-relevant 32 MiB size the medians were:

| path | median GiB/s |
|---|---:|
| node 0 -> GPU 0 (local) | 49.11 |
| node 0 -> GPU 7 (remote) | 50.03 |
| node 1 -> GPU 7 (local) | 50.36 |
| node 1 -> GPU 0 (remote) | 49.74 |
| one node-0 source -> GPU 0 local and GPU 7 remote concurrently | 48.43 unique / 96.86 aggregate |
| CPU copy node 0 -> node 1, then local GPU 7 DMA, copy CPU on node 0 | 5.15 |
| same, copy CPU on node 1 | 5.77 |
| feed GPU 0 locally while copying to node 1 and feeding GPU 7 locally | 5.18--5.76 unique |

The 128 MiB remote/local H2D points were likewise all approximately
49--51 GiB/s. The CPU-copy-plus-local-DMA path improved only to 7.9--9.5
GiB/s at 128 MiB, apparently because libc selects a more favorable large-copy
implementation, but remained far below direct remote DMA. Isolated registered
cross-node `memcpy` reached only 7.8--9.8 GiB/s with one thread. It could reach
roughly 30--41 GiB/s with four or eight threads and 54--74 GiB/s with sixteen,
but that consumes many CPU cores and extra memory bandwidth; even its best
case cannot improve on simply issuing the remote DMA. The raw output is
retained in `/tmp/zml_rocm_numa_bench_run{1,2,3}.txt`; the temporary source was
removed after the experiment.

This changes the interpretation of the earlier `numactl` result. The combined
binding control did not isolate host placement from GPU DMA locality. The raw
control now does: GPU DMA locality has no measurable penalty here. The weight
loader's possible NUMA-sensitive stage is the CPU `_copy_to_iter` that copies
page-cache folios into the anonymous registered slabs. Worker, page-cache, and
destination-slab placement determine whether that CPU reads or writes across
the socket link. The measured bound-loader improvement should not be attributed
to making the subsequent GPU DMA local.

Direct I/O would remove this exact bottleneck. An aligned `O_DIRECT` read into
the registered slab lets the storage stack DMA into those user pages and does
not execute the buffered-read page-cache-to-user `_copy_to_iter`. NUMA can
still matter for the storage DMA path, submission/completion CPUs, and the
placement of the destination pages relative to each NVMe controller, but it
is a different effect. This host's model filesystem is ext4 on an LVM volume;
its large RAID0 member spans four NVMe devices split evenly across the sockets
(`nvme0`/`nvme1` on node 1 and `nvme2`/`nvme3` on node 0), while the boot/LVM
NVMe is on node 0. A RAID read may therefore originate from both nodes no
matter where the target slab lives. Direct I/O can save the CPU copy and its
NUMA penalty, but a hot page-cache load may still be faster overall than the
physical disks. The direct-I/O throughput of this particular model has not
yet been measured in this ROCm investigation.

A subsequent two-run placement matrix used idle GPU 0 because GPU 7 had become
occupied by `llmd`. It held the same fixed loader tuple and hot page cache while
varying CPU binding and anonymous-slab memory binding independently:

| CPU node | slab node | load GiB/s, two runs |
|---:|---:|---:|
| 0 | 0 | 23.82, 23.71 |
| 1 | 1 | 24.22, 24.47 |
| 0 | 1 | 24.03, 23.84 |
| 1 | 0 | 21.79, 22.53 |
| unbound | unbound | 24.38, 25.25 |

This matrix does not support the idea that weights were fast simply because
most cached pages happened to reside on the GPU's node. Both same-node cases
were close, CPU 0 writing node-1 slabs was also close, and the current unbound
runs were fastest. Only the CPU-1/node-0 direction showed a repeatable roughly
8--10% penalty. The earlier 16-versus-27 GiB/s separation was therefore
confounded by time-varying host contention and possibly cache residency, not a
reproducible GPU-node locality effect. The kernel provides no cheap source-page
NUMA location through `preadv`; an `mmap` plus `move_pages` sample could inspect
the current cache retrospectively, but would not establish its placement
during the older recordings. Logs are
`/tmp/zml-rocm-cpu{0,1}-mem{0,1}-gpu0{,-repeat}.log` and
`/tmp/zml-rocm-unbound-gpu0-repeat{1,2}.log`.

No XLA build was performed. The current XLA checkout is clean at
`92fd6518987ecdfe0471d3cc49705e8144050b1b` (`GPU: use VMM only for command
buffer temp allocations`). The local ROCm archive currently hashes to
`32b36149625ff4dfc6713809a6751ed2210dbb14b446a84225f66dcee2770ce4`, while
the dirty `platforms/rocm/rocm.bzl` expects
`86a67903aab0e8960b1eda46572f17f357b7e4df804b3c4f3cc850333369ac78`.
Bazel therefore used its cached older PJRT artifact for these measurements.
That artifact tracks `DmaMap` ranges through `IsDmaMapped`, and the zero
linearization samples prove its bypass was effective. Reproducibility from a
cold Bazel repository still requires reconciling the configured archive hash.

## Active implementation: fully adaptive source tuple

Work started 2026-07-23 against `plan.md` from detached HEAD `9957ffdd`.
The pre-change loader had the fixed 32-worker `parallel_read` layer in
HF/S3/GCS, scalar read/DMA caps, a request size selected once per source, and a
precomputed fixed-size job array. The active work replaces those pieces with
one admitted VFS request per loader read, serial in-call retries and typed
timing telemetry, fixed/adaptive public controls, a dynamic round-robin range
scheduler, and joint source-concurrency/request-size adaptation.

The existing `perf.data` and `perf.data.old` recordings are user-owned and
must remain unmodified. XLA is explicitly out of scope.

Pre-change verification passed:

```text
./bazel.sh test --nocache_test_results //zml/io/vfs:test //zml:test \
  --test_output=errors
```

The VFS/data-plane conversion is now implemented. `parallel_read.zig` and its
worker/queue/chunk controls are gone. HF, S3, and GCS perform one whole Range
GET per positional call and retry serially while retaining the caller's source
credit. Shared range handling strictly validates covering `206` responses and
correctly discards the prefix of a `200` response that ignored Range. Typed
atomic telemetry records attempts, exact-size 2--128 MiB success timing,
transient/time-out/server/throttle failures, and retry delay. The focused VFS
tests pass after this conversion.

The example and benchmark controls now expose adaptive read initial/cap
(12/128), adaptive DMA initial/cap (8/32), fixed read/DMA overrides, adaptive
request initial/cap, and the existing fixed request-size override. Fixed
controls take precedence. Benchmark scripts preserve the read/DMA cap
variables, default to adaptive 128/32 caps, and forward explicit fixed values.

The loader now uses a mutex-protected round-robin range scheduler, one global
conservative source minimum, fixed/adaptive public configurations,
pinned-feasible read/lifecycle gates, typed failure feedback, exact-size local
and remote timing, modeled source width, request-size probes, and source-tuple
settling before DMA probes. The focused core suite, including scheduler size
transitions, fixed-dimension immutability, pinned-width/slack clipping, and
size-probe thresholds, passes:

```text
./bazel.sh test --nocache_test_results //zml:test --test_output=errors
```

## Current outcome and acceptance (2026-07-23)

`LoadOpts` now exposes:

- `read_parallelism = .adaptive.{ initial = 12, maximum = 128 }` or
  `.fixed`;
- `dma_parallelism = .adaptive.{ initial = 8, maximum = 32 }` or `.fixed`;
- `read_request_size = .adaptive.{ initial = source minimum, maximum =
  128 MiB }` or `.fixed`;
- the unchanged 2 MiB DMA block and 2 GiB pinned defaults.

Automatic request sizes are power-of-two MiB values and never smaller than
the largest source minimum or DMA block. A fixed request below the advertised
source minimum logs a warning and remains an explicit override. Fixed values
still obey the absolute read, DMA, request-size, DMA-block, and pinned-memory
bounds. The example and benchmark environment variables expose adaptive
initial values/caps plus fixed read, DMA, and request-size overrides; fixed
values win.

The range scheduler claims unscheduled tensor bytes round-robin under a mutex,
using the current request size and controller epoch. The read and retained
lifecycle gates resize dynamically. Pinned feasibility is
`max_pinned_bytes / request_size`; remote lifecycle slack is capped at eight
and clipped by that feasibility. The process creates workers through the
public read cap, but an index gate leaves only the selected width runnable.
This avoids a thundering herd at the 12-read local default while allowing
growth to 128 without reconstructing the worker group.

HF, S3, GCS, and generic HTTP each issue one complete Range GET per positional
call and retry serially while retaining admission. HF resolves and caches its
final download URI before range reads. S3/GCS/generic HTTP disallow redirects
for signed/range GETs; GCS token refresh is mutex-serialized and each request
owns a stable authorization copy. `206` requires a covering, case-insensitive
`Content-Range`; `200` responses that ignore Range are positioned and
scattered correctly. Only a clean first-attempt success enters the timing
model. All attempts publish typed failures and retry delay before sleeping.

Remote tensor readers forward the entire scatter list in one exact-fill
positional call, even above `IOV_MAX`; a short response fails without a hidden
resume request. Local readers retain `IOV_MAX` batching. The deterministic
loopback HTTP tests prove that a 17 MiB+257 scatter read is one GET, a
500-then-success retry has physical concurrency one and exact typed counters,
and three admitted callers produce physical high-water exactly three.

The controller owns one source `(read width, request size)` tuple plus one
per-device DMA width. High-latency sources bootstrap `12 -> 24 -> 32` only
before a response and never grow beyond 32 until an exact-size source timing
sample exists. Timing buckets are 2, 4, 8, 16, 32, 64, and 128 MiB.
Concurrency and request-size probes use only matching-epoch GPU-committed
logical bytes measured from probe activation. Request-size candidates need
eight full responses and `max(64 MiB, 4 * candidate size)` bytes; clear
results may finish after 50 ms and ambiguous results after 100 ms. A size
increase needs at least 3% goodput gain, so the lower-memory tuple wins inside
the 3% band. DMA probes wait until the source tuple settles and device queues
are fed. Throttles immediately reduce reads by 30% and impose a five-second
cooldown; transient failures use a rolling exact-size attempt cohort and
require two reliable windows above 10%.

Fresh validation after the final implementation:

```text
./bazel.sh test --nocache_test_results //zml:test //zml/io/vfs:test \
  --test_output=errors
./bazel.sh build //examples/io:playground //examples/mnist
./bazel.sh build --config=release --@zml//platforms:oneapi=true \
  //examples/io:playground
./bazel.sh build --config=release --@zml//platforms:cuda=true \
  //examples/io:playground
```

The oneAPI fixed 16 MiB / 32-read / eight-DMA control completed all 14.96 GiB
with 1,055 logical reads, peak width 32, and exactly 512 MiB pinned. Its
5.66 GiB/s wall result was taken during a transient slow-host interval and is
a correctness/data-plane result, not a replacement baseline.

A balanced one-B70 interleaved comparison after adding stable worker
activation was:

| mode | Five warm runs (GiB/s) | Median |
|---|---|---:|
| adaptive defaults | 27.13, 26.26, 24.91, 27.01, 27.49 | **27.01** |
| fixed 12 reads / 8 DMA / 2 MiB | 26.18, 26.18, 27.04, 27.02, 27.11 | **27.02** |

Both selected 12/8/2 MiB and used 24 MiB pinned. The adaptive median is within
0.1% of the static control and does not regress the 26.47 GiB/s historical
reference. Four-B70 correctness runs completed at 23.34 GiB/s sharded
(14.96 GiB physical, 54 MiB pinned) and 13.00 GiB/s replicated
(59.83 GiB physical, 54 MiB pinned), with peak DMA eight/device.

A current local profile is
`/tmp/zml-fully-adaptive-20260723.data`. It recorded 78,229 samples with none
lost while loading at 25.76 GiB/s and selecting 12/8/2 MiB with 24 MiB pinned.
Flat cycles remain dominated by the intended page-cache copy path:
`_copy_to_iter` 70.66%, `filemap_get_read_batch` 4.31%,
`entry_SYSCALL_64` 3.22%, `copy_page_to_iter` 2.10%, and
`folio_mark_accessed` 1.88%. No loader/controller symbol reached 0.5%.

Three final real-AWS adaptive runs were 947.76, 949.94, and 946.95 MiB/s:
median **947.76 MiB/s**, effectively identical to the recorded approximately
948 MiB/s reference. Every run performed exactly 1,055 physical GETs for
1,055 logical reads, transferred 14.96 GiB, and reported zero retries and
throttles. The first trace held at 32 until two 16 MiB timing successes
arrived, then performed scored read probes. Final read widths varied with the
finite tail, but selected request size remained 16 MiB and final DMA remained
eight/device.

The bundled S3Proxy fixture cannot produce valid current measurements: its
`206 Partial Content` responses omit `Content-Range`. The new strict reader
correctly rejects them with `InvalidContentRange`. Historical S3Proxy numbers
below remain useful, but rerunning those profiles requires a conforming or
patched fixture; strict production validation was deliberately not weakened.

## AWS controller follow-up (2026-07-24)

This pass was grounded in `../S3.md` and targeted controller policy only. The
data plane, S3 request validation, retry behavior, and XLA remained unchanged.

A fresh pre-follow-up adaptive trace completed at 947.94 MiB/s in-process
(946.78 MiB/s including the outer example timing), ending at 66 reads,
16 MiB requests, and eight DMA events with 1.11 GiB pinned. It exposed three
control defects:

- The no-response bootstrap correctly reached 32, but the first post-response
  read probe was installed before the 32-read tuple had a representative
  baseline. Its recorded baseline was only 61.90 MiB/s while the pipeline was
  still filling, so 32 -> 48 looked like a decisive gain.
- Probe accounting reset when lifecycle occupancy reached the candidate
  width. Requests already mostly complete at that point contributed their
  entire logical bytes over only the residual interval. Capacity activation
  also counted retained request lifecycles instead of active Range GETs.
- Read growth had strict priority over request-size growth and remembered no
  flat result. The finite 15 GiB transfer could spend every two-second slot on
  another width candidate and never evaluate 32 MiB.

### Static AWS screening grid

The screening grid used fixed DMA eight and one run per cell. Values below are
the loader's in-process logical goodput; every cell transferred exactly
14.96 GiB with zero retries and throttles.

| Request size | Read width | Logical MiB/s | Physical GETs | Pinned high-water |
|---:|---:|---:|---:|---:|
| 8 MiB | 64 | 954.27 | 1,981 | 536 MiB |
| 8 MiB | 96 | 955.10 | 1,981 | 792 MiB |
| 8 MiB | 128 | 928.44 | 1,981 | 1.03 GiB |
| 16 MiB | 16 | 774.97 | 1,055 | 288 MiB |
| 16 MiB | 24 | 950.26 | 1,055 | 416 MiB |
| 16 MiB | 32 | 951.42 | 1,055 | 560 MiB |
| 16 MiB | 64 | 955.33 | 1,055 | 1.03 GiB |
| 16 MiB | 96 | **956.55** | 1,055 | 1.54 GiB |
| 16 MiB | 128 | 953.19 | 1,055 | 2.00 GiB |
| 32 MiB | 32 | 949.10 | 641 | 1.06 GiB |
| 32 MiB | 48 | 950.53 | 641 | 1.57 GiB |
| 32 MiB | 64 | 951.38 | 641 | 2.00 GiB |
| 64 MiB | 16 | 853.58 | 417 | 1.11 GiB |
| 64 MiB | 24 | 931.96 | 417 | 1.55 GiB |
| 64 MiB | 32 | 942.63 | 417 | 1.91 GiB |

These are screening runs rather than repeated medians, and 128 MiB was not
screened, but the tested shape is clear. From 24 through 128 reads at 16 MiB,
useful goodput stays within 0.7% while average request latency rises almost
linearly with width: 0.390, 0.518, 1.029, 1.533, and 2.021 seconds at widths
24, 32, 64, 96, and 128. More requests divide one aggregate approximately
950 MiB/s path; they do not open more bandwidth. Relative to 16 MiB, a
32 MiB request reduces GET count by 39% but is no faster, 64 MiB regresses,
and the below-minimum 8 MiB control raises GET count by 88% without a gain.
The lowest tested tuple inside 3% of the best is 16 MiB / 24 reads; the
existing 32-read bootstrap is also inside 1%.

### Accepted controller changes

- Source-probe capacity and saturation now use active VFS read calls, with
  matching-epoch active/peak counters, rather than retained request
  lifecycles.
- Candidate logical bytes and successful reads are measured from tuple
  installation. Capacity must still be exercised before scoring, but reaching
  capacity no longer erases fill cost or admits almost-complete work into a
  short residual interval.
- Read probes require at least eight successful matching-epoch source reads
  and 64 MiB, in addition to the time and committed-byte floors.
- Slow high-latency sources establish the no-response bootstrap tuple before
  any post-response read probe. Fast/local startup behavior is unchanged.
- A scored width result gives request-size growth the next eligible source
  turn. Flat width results use a bounded five-second per-size retry delay
  instead of permanent suppression; a kept size starts a fresh width search
  for its own bucket.
- A width candidate that cannot exercise capacity also yields one eligible
  turn to request-size discovery without marking width settled. Read results
  at the maximum size preserve source-tuple settlement so DMA probing remains
  eligible.
- Relative starvation improvement no longer keeps a flat width increase.
  Starvation is materially removed only when it falls from above 10% to at
  most 10%.
- The scheduler counts total and full candidate requests per tensor. Size
  probes require eight full ranges and enough candidate jobs to exercise the
  proposed width, so aggregate bytes such as `7 * size + 1` cannot masquerade
  as eight full timing samples.
- The final tail eligibility check and request-size tuple switch now occur
  under one scheduler lock. Workers therefore cannot consume old-size work
  between the count and candidate installation.
- Candidate tail cost includes both the bytes needed for eight responses and
  one modeled candidate service time. Probe attribution and candidate timing
  begin at installation and survive capacity activation.

Focused tests cover active reads versus 48 retained lifecycles, exact
per-tensor full-range counts, atomic tail validation, candidate service-time
suppression, slow-source startup settling without changing slow local
behavior, install-time attribution, bounded read re-probing, maximum-size DMA
eligibility, and advancing to size after flat or unexercised read probes. The
core and VFS suites pass.

The first revised trace held at 32, rejected a 43-read candidate after
1.253 seconds of source-call evidence, tested 32 MiB once, rejected it, and
finished at 32 reads / 16 MiB / eight DMA events. Three revised in-process
runs were 946.30, 951.75, and 935.56 MiB/s: median **946.30 MiB/s**. The
outer example results were 944.95, 950.51, and 934.41 MiB/s: median
**944.95 MiB/s**, 0.3% below the prior 947.76 MiB/s median and well inside the
3% acceptance band. Final source width is now deterministic at 32; the runs
used 1,006, 1,009, and 1,013 GETs because one bounded 32 MiB probe replaced
some 16 MiB requests. Pinned high-water was 1.31, 1.19, and 1.09 GiB while
that joint size/width candidate was active.

The final code received one additional AWS confirmation run. It rejected
37- and 54-read candidates around one intervening 32 MiB candidate, restored
32 reads / 16 MiB / eight DMA events, and completed at **947.02 MiB/s**
in-process and **945.55 MiB/s** outside the loader. It used 1,006 GETs, no
retries or throttles, and 1.41 GiB pinned high-water. This is a single
confirmation rather than a new median, but it is consistent with the earlier
three-run result and shows no regression outside normal variance.

This work improves convergence correctness and prevents one probe dimension
from monopolizing the transfer, but it does **not** speed up this AWS path.
The tested static grid and revised measurements show no repeatably
demonstrated, material (>3%) controller-accessible gain above the existing
approximately 948 MiB/s result. Further remote speed work needs evidence
outside the current source tuple: regional/path placement, NIC/socket
saturation, connection-pool/DNS/TLS breakdown, or a different endpoint. Warm
local controls reached 29.16, 28.52, and 27.57 GiB/s at the unchanged
12 reads / 2 MiB / eight DMA tuple.

The complete historical investigation and earlier controller measurements
follow. Descriptions of `parallel_read`, scalar `LoadOpts`, precomputed jobs,
or once-selected automatic request sizes in those sections describe the
superseded implementation, not the current tree.

## Historical outcome before `plan.md` (superseded)

CUDA and oneAPI use a copy-free vectored path with independently controlled
source calls, retained request lifecycles, and physical DMA-event admission:

```text
adaptive active source calls
    -> bounded retained requests made from small client-DmaMapped blocks
    -> adaptive per-device PJRT event admission
    -> final destination shards/replicas
```

The ready queue contains the final DmaMapped transfer blocks; ZML introduces no
pageable staging allocation or userspace copy. The selected oneAPI PJRT
recognizes the imported ranges as pinned and transfers them without the former
hidden pageable stage. One `AdaptiveVectoredController` owns the source-call
limit and the shared per-device DMA limit, but enforces them independently.
The load-wide pinned pool remains shared across devices. Other targets retain
the ordinary buffered loader.

There are two read-side gates. The source gate limits calls currently inside
the positional backend read. The lifecycle gate retains a credit until every
DMA block from that request, including all replicas, has completed. Local
sources use no slack, preserving the successful lane-style read/DMA coupling.
VFS backends marked high-latency receive eight extra retained lifecycle slots,
so all 32 source calls can remain active while a small bounded cohort finishes
DMA. This costs at most another 128 MiB for 16 MiB S3/GCS requests or 256 MiB
for 32 MiB HF requests and avoids both unbounded eager reads and remote
underfill.

Request size is selected once per source from VFS hints; it is not tuned at
runtime:

```text
source                         automatic request size
local/default and file://              2 MiB
generic HTTP, S3, and GCS              16 MiB
HF                                      32 MiB
```

Each backend advertises `minimum_request_size` and whether it is high-latency.
Automatic sizing takes the greater of that source minimum and the configured
DMA block size, so a nominal request is never smaller than its block. An
explicit fixed override must also be at least one DMA block and logs a warning
when it is below the backend minimum. Tensor tails and dispatch boundaries may
still produce partial final blocks. S3/HF/GCS additionally expose physical-request, byte, retry,
throttle, and retry-delay counters. The controller backs source concurrency
off directly on retry/throttle evidence; it never interprets pool waits or read
latency as storage congestion.

The controller starts at twelve source calls and eight physical events per
device. This is a warm-start prior, not a fixed final tuple; both limits remain
adaptive. Before any source response, high-latency sources are sampled
every 10 ms and double `12 -> 24 -> 32`. Once any read returns, growth freezes until
at least one PJRT completion has arrived from every expected destination
device. Thereafter all performance changes are scored. Single-device startup
gets one read probe; multi-device startup may use two so independent device
queues can be fed. A startup read increase is retained when committed logical
goodput is within 3% and DMA remains unresolvedly starved; long loads can later
probe down to the lowest concurrency within 3%.

The controller sleeps on an interruptible event between samples. Load
completion wakes it immediately instead of adding a random 0-25 ms tail to
sub-second local loads.

The public `LoadOpts` fields are:

- `read_parallelism = 32` (hard adaptive cap)
- `dma_parallelism = 32` (hard adaptive cap per device)
- `read_request_size = .auto` (`.fixed = bytes` is the explicit override)
- `dma_block_size = 2 MiB`
- `max_pinned_bytes = 2 GiB` (lazy hard cap; ordinary local use is 24-40 MiB)
- `shardings`, `progress`, and `total_bytes`

There is deliberately no adaptive toggle or public controller threshold.
Previously removed fields remain removed: `initial_parallelism`, `adaptive_parallelism`,
`max_read_parallelism`, `read_chunk_size`, `max_staging_bytes`,
`max_pinned_buffers_per_device`, `pinned_buffer_size`, and
`transfer_quantum_size`.

## Implementation

### Exact vectored reads

`zml/safetensors.zig` adds `TensorReader.readPositionalAllV`.

- The combined tensor-relative range and backing-file offset use checked
  arithmetic.
- Empty in-bounds reads succeed; out-of-bounds and overflow fail.
- Partial reads resume at the exact slice and intra-slice offset.
- Local calls are split at the platform `IOV_MAX`.
- Unexpected EOF is reported only after all actual progress is consumed.
- Positional calls do not disturb the sequential reader position.

Generic HTTP range reads now fill every supplied slice. A `206` must have a
covering `Content-Range`; a server returning `200` and ignoring `Range` is
handled by discarding the requested prefix before scattering. S3, GCS, and HF
continue to use their existing internal scatter/chunk readers.

There are currently two source-concurrency layers for S3, GCS, and HF. The
loader controller admits logical positional reads; each backend then queues
physical range jobs on its own fixed 32-worker `parallel_read` pool. With the
default automatic sizes these layers are one-to-one: an S3/GCS 16 MiB logical
read is one 16 MiB physical job and an HF 32 MiB logical read is one 32 MiB
physical job (apart from tails). Consequently `parallel_read` does not split
ordinary controller reads today; it supplies the HTTP execution, retry,
throttle, and statistics machinery. A fixed logical request larger than the
backend chunk, or an automatic request enlarged by a larger DMA block, is
split into several simultaneous physical jobs. That hidden fan-out is not
admitted individually by the controller and would confound future adaptive
request-size/concurrency tuning.

The vectored loader opens each distinct backing file once and gives tensor
readers borrowed positional-only handles. This is safe because positional
reads do not mutate streaming position. It is important remotely: Llama 3.1
8B has 291 tensors but only four safetensor files, so per-tensor opens caused
291 object `HEAD` requests. Shared handles reduced the 1000 ms S3Proxy run from
49.03 s to 41.72 s.

### Reservation planner

`VectoredRequestPlan` is a pure planner over `DispatchSpans`.

- It accepts any tensor source range, including ranges crossing shard and DMA
  block boundaries.
- Scatter entries remain in global source/file order.
- Each DMA block contains bytes contiguous in one final destination layout and
  records the exact destination shard/replica mask and final offset.
- Replicated destinations reuse one host block; the block has one reference
  per submitted device transfer.
- Multiple requests from one tensor may complete out of order.

Final-transfer handling is per destination. A destination's final block waits
until the sum of all earlier submitted byte ranges reaches its final offset;
only then is it submitted with PJRT's `is_last_transfer=true`. Finals for one
tensor do not wait for unrelated tensors, which avoids pin-pool deadlocks.

### DMA pool and completion

`zml/mem.zig` adds one load-scoped pool shared by every device of the PJRT
client.

- Host memory is allocated in lazy slabs of up to 64 MiB, registered once via
  `PJRT_Client_DmaMap`, and divided into fixed independent blocks.
- The hard byte limit includes all mapped blocks, not only currently leased
  blocks.
- A request reserves all blocks atomically or waits without retaining a
  partial reservation.
- PJRT completion callbacks decrement reference-counted leases; only the last
  replica callback returns a block.
- Slab storage uses raw allocation so Zig does not poison/memset memory that a
  source read immediately overwrites.
- Pool close wakes blocked reservations. On failure the coordinator stops new
  work, drains active reads and events, aborts deferred transfers, marks
  unfinished PJRT buffers with an error, then destroys events/managers and
  finally unmaps slabs.

A 128 MiB pool made from two 64 MiB mappings completed sharded and replicated
loads across all four B70s. This validates that subranges of one registered slab
are accepted by all devices in this oneAPI client.

### Adaptive coordinator and metrics

The coordinator creates at most the public read cap in stable workers and
schedules tensor ranges round-robin, allowing several requests from one tensor.
A worker first takes a lifecycle credit, atomically reserves every DMA block
for the request, then takes a source credit only for the exact vectored read.
The source credit is released as soon as the backend read returns; the
lifecycle credit is released by the last block/replica PJRT callback. Reducing
either limit drains existing work naturally.

The jobs are constructed one range per tensor per round and then consumed by
all workers through one atomic index. This favors breadth across tensors, but
does not serialize a tensor: workers may execute different positional ranges
of the same tensor concurrently, particularly after smaller tensors finish or
when one large tensor must fill the available width. The borrowed
`TensorReader` and shared file are safe for this because these reads are
positional.

Each completed read creates physical entries in per-device queues. An entry
takes one device credit until its PJRT callback completes. One device cannot
block another. A replicated block appears once in every destination queue but
retains one shared host lease until every callback completes. A final entry is
held until every preceding byte for that transfer manager has been submitted,
then uses `is_last_transfer=true`.

The controller samples every 25 ms after PJRT warm-up and uses 50-100 ms
startup and 100-250 ms steady windows. Attributed probes ordinarily require
64 MiB/200 ms; an unambiguous >=10% gain or loss may finish at 100 ms when both
the epoch cohort and cumulative post-activation rates agree. Read probes score
logical bytes only when the complete request lifecycle retires. DMA probes
score physical bytes committed by matching PJRT events. Old/out-of-order
epochs cannot validate a candidate.

Source pressure covers all post-read states, both ready and submitted. It
requires a logical-goodput regression plus sustained growth, >75% occupancy
with >250 ms age, or 1.5x latency inflation, while DMA starvation is <=10%.
Pool exhaustion is only hard byte backpressure. Explicit VFS retry or throttle
feedback immediately reduces source concurrency by 30% and applies a five
second cooldown. Slow/bursty sources are protected from queue-based reduction
until DMA has remained fed for two seconds.

DMA admission remains per-device with one shared learned limit. Increases need
the candidate capacity exercised on every device that had queued demand and a
3% physical-goodput gain. Two reliable windows with >2x completion latency and
<95% best goodput back off by 30%. Resource reductions are considered only
after two seconds without starvation and only when estimated remaining work is
also above two seconds; any performance probe is suppressed below a 250 ms
tail.

Metrics include source operations/bytes/latency, complete request lifetime and
logical retirement, backend physical requests/retries/throttles, unique
post-read and ready bytes/age, per-device active/high-water events, DMA
submission/completion latency, pool wait/high-water, exact physical remaining
bytes, final settled limits, and wall time.

## Code map

- `zml/io.zig`: `LoadOpts`, planner, lazy source/tensor states, coordinator,
  PJRT final/error handling, buffered fallback, and planner/final tests.
- `zml/mem.zig`: `DmaBlockPool`, slab mapping, atomic bulk reservations, and
  lease/refcount tests.
- `zml/safetensors.zig`: exact positional scatter API, borrowed positional
  readers, and bounds/partial/EOF/IOV tests.
- `zml/io/vfs/http.zig`: complete scatter filling and strict range response
  validation.
- `zml/io/vfs/base.zig`, `zml/io/vfs.zig`: backend minimum request size,
  high-latency classification, and optional source-stat provider.
- `zml/io/vfs/parallel_read.zig`: scatter chunk validation plus atomic physical
  request/retry/throttle statistics.
- `pjrt/pjrt.zig`: helper for setting unfinished async buffers to an unknown
  PJRT error.
- `examples/io/main.zig`: automatic/fixed request-size selection and the four
  cap/block/pinned environment controls.
- `bench_file.sh`, `profile_file.sh`: local defaults; `PERF_DATA` prevents
  overwriting a caller's profile.
- `bench_s3.sh`: remote defaults and latency/bandwidth controls.

## Performance results

Model: Llama 3.1 8B, 14.96 GiB logical weights. Local numbers use warm page
cache. Medians are wall-clock goodput reported by `examples/io`.

### Linux file VFS and direct-I/O experiment (2026-07-22)

The Linux `File` VFS has long defaulted `direct_io=true`, but that does not
mean the loader benchmarks use direct I/O. It opens a second read-only file
description and sets `O_DIRECT`, then selects it only when the absolute file
offset, every destination address, and total length satisfy its configured
4 KiB alignment. Otherwise it deliberately uses the ordinary buffered file
description.

There are two independent reasons the existing model load is buffered:

- `bench_file.sh` passes a plain local path. The VFS routes it to the base
  `std.Io`, not the backend registered under the `file` URI scheme. A syscall
  trace of that path contained zero `F_SETFL(...O_DIRECT...)` calls.
- Using `file:///var/models/...` does create and verify the `O_DIRECT` file
  descriptions, but the safetensor data bases have non-zero 4 KiB residues
  (1328, 3936, 3472, and 568 bytes for the four files). All observed 2 MiB
  tensor payload reads therefore selected the buffered descriptions; none
  used the direct fds. Merely changing the benchmark URI does not make the
  payload direct.

A copy-free aligned-envelope control was added at
`~/github/uxlfoundation/sycl_playground/src/b70_direct_io_pipeline.cpp`. For
each arbitrary tensor range it rounds the file offset down and the end up to
4 KiB, reads that envelope with `O_DIRECT` into an aligned registered host
allocation, and transfers only the logical `buffer + prefix` subrange to the
B70. It adds no CPU staging copy and only 0.197% extra storage traffic with
2 MiB chunks. This is the shape a real loader fast path would need; the
current exact scatter API cannot manufacture safe prefix/suffix storage inside
the caller's fixed-size logical slices.

With twelve workers/queues, 2 MiB chunks, real tensor ranges, unique tensor
destinations, and the XLA-selected oneAPI runtime, three warm-page-cache
buffered runs reached 34.08, 34.69, and 34.62 GiB/s (median **34.62 GiB/s**).
The direct path reached 10.13, 10.29, and 10.28 GiB/s (median
**10.28 GiB/s**). Average direct read service was about 2.21 ms per 2 MiB,
versus about 0.60 ms buffered; direct DMA itself remained cheap at about
44--54 us.

The direct result is storage-limited rather than an admission mistake. At
2 MiB, 4/8/12/16 workers all plateaued at 10.28--10.29 GiB/s, while 24/32
workers regressed to about 9.45 GiB/s. With twelve workers, 1/4/8 MiB chunks
reached about 9.57/9.57/9.44 GiB/s, so 2 MiB is also the best tested direct
chunk. Four parallel `dd iflag=direct` shard reads reached about 10.2 GiB/s.

Dropping the page cache before every buffered run changes the conclusion for
a first load from storage. Five paired raw SYCL runs were:

| source path | Five logical GiB/s | Median |
|---|---|---:|
| cold buffered | 7.897, 7.916, 7.924, 7.903, 7.936 | **7.916** |
| `O_DIRECT` aligned envelope | 10.174, 10.179, 10.177, 10.175, 10.197 | **10.177** |

Thus direct I/O is 28.6% faster for a genuinely cold load on this Samsung
9100 Pro, but 70% slower than the warm page-cache pipeline. It should not
replace buffered loading unconditionally. A production implementation needs
an explicit cold/direct source policy plus alignment-aware padded DMA leases;
it cannot transparently infer whether the caller values retaining the model in
page cache. The playground experiment is retained, but the slower path was not
wired into ZML.

Five current ZML/PJRT loads with the cache dropped before each run took 1.744,
1.748, 1.749, 1.745, and 1.748 seconds: median **1.748 s / 8.56 GiB/s**.
The immediately following warm run took 565.7 ms / 26.44 GiB/s. Cold reads
averaged 3.47 ms while PJRT completion averaged only 0.09 ms, so the current
cold load is entirely source-bound. The raw direct pipeline's 10.18 GiB/s is
roughly 19% above current cold ZML goodput and is the relevant approximate
headroom for a production direct path on this drive; its 10.18 GiB/s storage
ceiling still precludes improving the warm 26--27 GiB/s result.

Linux 6.17 provides a cheap cache-selection signal through `cachestat(2)`
(`CONFIG_CACHESTAT_SYSCALL=y`, syscall 451 on this x86-64 host). Querying all
four complete shard ranges reports the exact aggregate cached-page count. It
took about 22.8 ms when all 14.96 GiB were cached and 3.2 us when none were;
full `mincore` took 0.39 s and is too expensive for admission. A first-stage
sample of sixteen evenly spaced 2 MiB windows per file (64 calls, 128 MiB
sampled) took only **0.194 ms** while correctly reporting the current 100%
resident state. A production `.auto` policy can use this sample, choose
buffered/direct immediately for clear hot/cold results, and pay for an exact
`cachestat` only in the ambiguous middle. The measured cold/warm curves put a
rough single-load break-even near 25--30% residency, but partially resident
benchmarks are still required before fixing the threshold. Policy also matters:
direct I/O wins one cold load but deliberately does not populate page cache,
so buffered I/O may win when another load is expected soon.

### Completion-aware source backpressure revision (2026-07-22)

The current 12/8 warm-start follow-up produced one-B70 runs of 26.76, 26.78,
26.84, 26.60, and 26.96 GiB/s: median **26.78 GiB/s**. Every run began at
12 reads/eight DMA events, then the scored startup probe retained 16/eight and
used 32 MiB pinned. This is 4.2% above the prior 25.70 GiB/s median and 1.2%
above the 26.47 GiB/s reference.

A control that disabled the startup probe and held the short load at 12/eight
gave 25.87, 25.82, 25.94, 26.19, and 26.04 GiB/s: median **25.94 GiB/s**, with
24 MiB pinned. Therefore 12/eight is retained as the cold-start tuple, not as
a fixed local limit. The controller remains free to keep 16/eight when its
scored result is better. Four-device and remote profiles have not yet been
repeated after this warm-start-only change.

The follow-up paired request/DMA-block sweep tested whether larger transfers
could amortize PJRT submission and callback overhead. With identical adaptive
caps, five interleaved rounds gave:

| Request / DMA block | Median | PJRT submissions | Pinned high-water |
|---|---:|---:|---:|
| 2 / 2 MiB | **26.52 GiB/s** | 7,723 | 32 MiB |
| 4 / 4 MiB | 26.28 GiB/s | 3,895 | 64 MiB |
| 8 / 8 MiB | 22.31 GiB/s | 1,981 | 128 MiB |

Matching outstanding read bytes was worse: 4/4 with eight reads medianed
23.65 GiB/s and 8/8 with four reads medianed 21.98 GiB/s. Matching active DMA
bytes made 4/4 with four events competitive in an initial five runs, but a
final directly interleaved shortlist gave **26.66 GiB/s for 2/2 with eight
events** versus **26.11 GiB/s for 4/4 with four events**. The latter halves
PJRT submissions and lowers average PJRT latency from roughly 0.31 to 0.28 ms,
but average source-read latency rises from roughly 0.74 to 1.88 ms while pinned
high-water doubles. The 2 MiB request/block therefore remains the fastest and
lowest-memory local choice; PJRT call amortization does not repay the loss on
the source-copy side.

The preceding local five-run results with automatic 2 MiB requests were:

| Placement | Five runs | Median | Reference |
|---|---|---:|---:|
| one B70 | 7.78 outlier, 25.12, 25.70, 25.71, 26.08 GiB/s | **25.70 GiB/s** | 26.47 GiB/s (-2.9%) |
| four B70 sharded | 23.37 outlier, 24.86, 26.04, 25.90, 26.13 GiB/s | **25.90 GiB/s** | 26.62 GiB/s (-2.7%) |
| four B70 replicated | 12.27, 12.54, 11.95, 12.48, 12.42 GiB/s | **12.42 GiB/s** | 11.84 GiB/s (+4.9%) |

Those one-device loads settled at 12 reads/eight events and used 24 MiB pinned. The
sharded load explores 12-16 reads/eight events and uses 40 MiB pinned.
Replication settles at 16/eight, transfers 59.83 GiB physically, and uses only
32 MiB pinned because replicas share each host block. Occasional 12-event
probes restore eight; average replicated PJRT completion latency remains
0.77-0.83 ms rather than the former 11-56 ms failure mode.

S3Proxy with automatic 16 MiB requests and eight retained-request slack slots:

| Profile | Runs | Median/reference |
|---|---|---:|
| 10 ms / 1000 MiB/s | 2.849, 2.861, 2.863 s | **2.861 s** / 2.689 s (+6.4%) |
| 250 ms / 1000 MiB/s | 11.211, 11.332, 11.308 s | **11.308 s** / 11.102 s (+1.9%) |
| 1000 ms / 100 MiB/s | 41.724 s | **41.724 s**, exact reference |

The 10 ms residual is source service time, not lifecycle starvation: 1,055
requests average about 83.3 ms each, whose 32-way lower bound is roughly
2.75 s. Widths 36/40/48 regress because read latency inflates. Four retained
slack requests also regress to a 3.00 s median; twelve adds variance without
improving the best case. Eight is the selected balance, peaking around 572 MiB
pinned on S3. All profiles reach 32 active source calls before the first
response and report no retries/throttles.

At identical concurrency and ample pinned budget, local 2 MiB requests median
25.97 GiB/s while 16 MiB requests median 20.31 GiB/s. For the 16 MiB case,
hard caps of 192/384/1024 MiB all map and use exactly 192 MiB; medians are
20.93/20.64/20.51 GiB/s. Extra available pinned memory therefore neither
increases read/DMA admission nor recreates long PJRT latency.

Fresh profiles are `/tmp/zml-lifecycle-one-20260722.data` and
`/tmp/zml-lifecycle-four-sharded-20260722.data`. `_copy_to_iter` accounts for
58.86% and 55.20% of sampled cycles. `controlSnapshot` is about 0.01%, the
controller loop rounds to 0.00%, and generic userspace memcpy/memmove is about
0.05%; pageable staging and userspace sharding copies remain absent. Remaining
visible CPU work is page-cache lookup/locking, scheduler/futex overhead, and
oneAPI/PJRT bookkeeping. `/tmp/zml-four-stuck-20260722.data` records one
transient four-device `queueFinish()` stall; immediate reruns and all final
four-device correctness/performance runs completed normally.

### Adaptive-controller iteration baseline

Before changing `30d18486`, five fresh runs were captured without overwriting
the workspace `perf.data` files. The host was in the previously observed slow
local-read state, so these numbers are a same-state regression control and do
not replace the historical 25-27 GiB/s warm-state results below:

| Placement | Five static runs | Median |
|---|---|---:|
| one B70 | 8.79, 9.36, 9.46, 9.28, 8.85 GiB/s | **9.28 GiB/s** |
| four B70 sharded | 8.86, 8.47, 8.98, 8.73, 8.27 GiB/s | **8.73 GiB/s** |
| four B70 replicated | 6.28, 6.14, 6.36, 6.54, 6.34 GiB/s | **6.34 GiB/s** |

During implementation the host returned to its faster state. Final balanced
adaptive runs were:

| Placement | Five adaptive runs | Median |
|---|---|---:|
| one B70 | 25.66, 25.67, 26.83, 26.83, 26.77 GiB/s | **26.77 GiB/s** |
| four B70 sharded | 26.80, 26.81, 26.80, 26.81, 26.78 GiB/s | **26.80 GiB/s** |
| four B70 replicated | 12.86, 12.79, 12.87, 12.86, 12.60 GiB/s | **12.86 GiB/s** |

The one-device runs converged to 12 reads/eight events per device; sharded
runs used 16/eight. Replicated runs explored 8-16 reads and occasionally
probed 12 events before restoring eight. All three medians exceed the
26.47/26.62/11.84 GiB/s static references below. The default bounded admission
high-water was only 64 MiB despite the 128 MiB hard pool.

### Pre-replacement baseline

Five balanced runs of the old adaptive/staging loader:

| Placement | Median | Individual wall times |
|---|---:|---|
| one B70 | 25.31 GiB/s | 590.969, 591.014, 591.071, 565.594, 592.490 ms |
| four B70 sharded | 24.72 GiB/s | 1.032 s outlier, 605.183, 604.719, 605.316, 580.092 ms |

The separately measured parallel page-cache ceiling on this machine is about
30-31 GiB/s.

### Static vectored configuration before adaptive admission

Configuration: 12 reads, 2 MiB request, 2 MiB block, 128 MiB pinned.

| Placement | Five runs | Median |
|---|---|---:|
| one B70 | 26.57, 25.52, 27.01, 26.47, 25.22 GiB/s | **26.47 GiB/s** |
| four B70 sharded | 26.62, 26.63, 27.16, 25.86, 22.43 GiB/s | **26.62 GiB/s** |
| four B70 replicated | 11.93, 11.87, 11.84, 10.98, 11.58 GiB/s | **11.84 GiB/s** |

The four-B70 sharded median is 7.7% above the old 24.72 GiB/s median and reaches
about 86% of the measured page-cache ceiling. Replicated goodput is logical;
the loader physically transfers 59.83 GiB to four devices.

### Sweeps

The required first block sweep held request/read width/pinned at
32 MiB/32/1 GiB. Five-run medians were:

| DMA block | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 32 MiB |
|---|---:|---:|---:|---:|---:|
| GiB/s | 13.64 | 14.51 | 14.62 | 14.81 | 14.98 |

That point is misleading in isolation: large blocks waste most of each lease
when the request is smaller, while 32 large concurrent requests consume the
whole 1 GiB pool. Repeating the block sweep at the selected 2 MiB request and
12-read point found the interaction:

| DMA block | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 32 MiB |
|---|---:|---:|---:|---:|---:|
| median GiB/s | 16.14 | **25.36** | 24.54 | 22.03 | 22.46 |

At 2 MiB request/block and 128 MiB pinned, read-width medians were:

| Reads | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| median GiB/s | 20.73 | 22.14 | **25.67** | 21.92 | 20.61 | 19.62 |

At 12 reads, 2 MiB block, and 128 MiB pinned, request medians were:

| Request | 2 MiB | 4 MiB | 8 MiB | 16 MiB | 32 MiB | 64 MiB |
|---|---:|---:|---:|---:|---:|---:|
| median GiB/s | **25.10** | 22.82 | 20.61 | 17.52 | 14.90 | 11.48 |

That sweep also changed effective memory admission: a 128 MiB pool can hold
only eight full 16 MiB requests. A follow-up requested by the user fixed
`read_parallelism=12`, `dma_block_size=2 MiB`, and
`max_pinned_bytes=192 MiB` for both 2 and 16 MiB requests. This is the minimum
budget that can admit twelve full 16 MiB requests, so only request size changed
and both cases reached `peak_reads=12`:

| Request | Five runs | Median | Pinned behavior |
|---|---|---:|---|
| 2 MiB | 9.44, 9.58, 9.35, 9.31, 9.53 GiB/s | **9.44 GiB/s** | 36-62 MiB high-water, 64 MiB mapped, no waits |
| 16 MiB | 8.20, 8.57, 8.33, 8.41, 8.06 GiB/s | **8.33 GiB/s** | 192 MiB high-water/mapped; both cases reached 12 reads |

The host was in a much slower local-read state during this follow-up (the
2 MiB control was 9.44 rather than the earlier 25-27 GiB/s), so these absolute
numbers must not be mixed with the warm-page-cache table. The controlled
comparison nevertheless exposes a real coupling after the read: one 16 MiB
completion submits roughly eight 2 MiB transfers, so twelve simultaneous read
completions can burst about 96 PJRT submissions. At 192 MiB, completion latency
for the same 2 MiB DMA blocks rose from roughly 0.09 ms in the 2 MiB case to
1.2-1.9 ms, and later reads waited for groups of eight blocks even though peak
read width was twelve.

Increasing the 16 MiB budget did not solve that coupling in the immediate-
submission implementation. A 384 MiB
double-bank trial fell to 6.67 GiB/s with 11.65 ms average DMA completion
latency; a 1 GiB five-run median fell to 5.33 GiB/s with 40-56 ms DMA latency
and the entire pool mapped/in use. More memory let read completions enqueue a
larger PJRT backlog. The adaptive event gate plus bounded read admission now
removes that coupling; final controlled results are recorded below.

Pinned-limit medians at 12 reads and 2 MiB request/block were:

| Limit | 128 MiB | 256 MiB | 512 MiB | 1 GiB |
|---|---:|---:|---:|---:|
| median GiB/s | **25.68** | 25.52 | 25.18 | 24.98 |

The lowest-memory candidate is also the fastest. The later final five-run
median improved to 26.47 GiB/s after shared source handles and normal run
variance.

With adaptive event admission and the internal read-admission bound, the
requested 16 MiB controlled rerun is qualitatively different:

| Hard pinned limit | Wall goodput | Peak read calls | Peak DMA/device | Avg DMA latency | Actual high-water |
|---|---:|---:|---:|---:|---:|
| 192 MiB | 19.69 GiB/s | 12 | 8 | 0.507 ms | 192 MiB |
| 384 MiB | 20.32 GiB/s | 16 | 8 | 0.532 ms | 256 MiB |
| 1 GiB | 20.33 GiB/s | 16 | 8 | 0.546 ms | 256 MiB |

Extra registered-memory allowance no longer raises admitted PJRT concurrency
or recreates the former 11-56 ms completion latency. The 384 MiB and 1 GiB
runs are effectively identical because both settle on the same 16-read/8-DMA
tuple and 256 MiB internal admission bound. The 192 MiB case cannot fully
exercise the temporary 16-read candidate, but remains close and bounded.

### S3Proxy

The selected remote configuration is 32 reads, 16 MiB requests, 2 MiB blocks,
and 512 MiB pinned. It was within 0.2% of 32 MiB/32 reads/1 GiB at 10 and 250 ms
while using half the registered memory. A 64 MiB/24-read/1 GiB candidate was
slower.

| S3Proxy profile | Final runs | Final median/time | Old controller |
|---|---|---:|---:|
| 10 ms / 1000 MiB/s | 2.691, 2.688, 2.689 s | **2.689 s / 5.56 GiB/s** | 2.912 s median |
| 250 ms / 1000 MiB/s | 11.105, 11.102, 11.099 s | **11.102 s / 1.35 GiB/s** | 11.462 s |
| 1000 ms / 100 MiB/s | one validation | **41.724 s / 367 MiB/s** | 43.175 s |

The full 10 ms discovery pass rose from 1.17 GiB/s at 2 MiB/12 reads to
5.40-5.41 GiB/s at 16-32 MiB/32 reads before source-handle sharing. This is why
the request-size knob remains explicit.

Final adaptive validation with the same fixed 16 MiB request, 2 MiB block,
512 MiB hard pool, and 32/32 public caps:

| S3Proxy profile | Adaptive runs | Median/time | Versus static reference |
|---|---|---:|---:|
| 10 ms / 1000 MiB/s | 3.070, 3.094, 3.069 s | **3.070 s / 4.87 GiB/s** | 14.2% slower |
| 250 ms / 1000 MiB/s | 11.223, 11.225, 11.191 s | **11.223 s / 1.33 GiB/s** | 1.1% slower |
| 1000 ms / 100 MiB/s | one validation | **41.862 s / 365.9 MiB/s** | 0.3% slower |

The no-progress bootstrap reaches 32 reads before the first response on the
250 and 1000 ms profiles, satisfying the 3% requirement there. At 10 ms there
is enough early progress to enter gradual read discovery, so serialized 200 ms
read probes consumed a material part of the original 3-second load.

### Probe-floor and startup follow-up

The follow-up implements two complementary changes rather than globally
weakening the 200 ms floor:

- Startup may publish one representative source-side growth step without a
  probe. The slow-source classification sends the 10 ms S3Proxy profile from
  eight to the 32-read cap; a fast local source takes only the ordinary gradual
  step when it is actually starved.
- Settled probes retain 64 MiB/200 ms unless both the epoch cohort and the
  cumulative post-activation cohort show the same decisive result. A gain of
  at least 10% or loss of at least 10% may finish at 100 ms. Unit tests cover
  early keep, early rollback, and an ambiguous 5% candidate that must wait.

The 100 ms exit does not fire in the current 10 ms S3Proxy run because source
startup reaches the cap without creating a steady probe. It removes latency
only when a real later candidate is clearly different. This is intentional:
the run-to-run issue left on 10 ms is no longer a serialized 200 ms decision.

Three final 10 ms runs were 3.044, 4.047, and 3.445 s (median **3.445 s**).
Steady committed goodput was consistently about 5.5--6.0 GiB/s, but initial
PJRT completion stalls varied by roughly 0.2--1.0 s and dominated wall-time
variance; the best run is essentially the prior 3.070 s adaptive median, while
the median is still 28% behind the 2.689 s static reference. These measurements
were taken while the local host was also in its previously documented degraded
state, so they do not replace the balanced reference table above.

One 250 ms / 1000 MiB/s regression run completed in 11.541 s / 1.30 GiB/s.
That is 2.8% behind the previous 11.223 s adaptive median and 4.0% behind the
11.102 s static reference; one run on the degraded host is not a replacement
for the required balanced three-run median, but it confirms correct high-latency
bootstrap (`8 -> 16 -> 32`) and completion.

Experiments rejected during this follow-up:

- Letting scored DMA probes interleave with startup raised the limit through
  12/16/20 events and produced 4.06 s remotely.
- Unscored broad DMA growth collapsed a local run to 6.07 GiB/s.
- A guarded remote-only 8-to-16 shortcut completed in 3.53 s and inflated the
  startup completion latency to about 39 ms.
- A transient jump to 32 events made the initial stall worse (about 0.69 s)
  and completed in 4.07 s. A large ready backlog is therefore not enough
  evidence to raise DMA concurrency.
- Holding the slow source at 16 reads completed in 5.27 s; its 16 MiB requests
  averaged about 84 ms and sustained only 3--4 GiB/s. This validates opening
  that source to the caller's 32-read cap.

Local queue traces also explain the value of recently filled buffers. Replacing
newest-ready `swapRemove` scheduling with strict FIFO reduced steady goodput
from about 12.4 to 10.1 GiB/s on the current host (1.26 s versus 1.58 s wall),
consistent with DMA benefiting from cache-hot pages. The newest-ready policy is
retained, but readiness diagnostics now exclude ordering-blocked final entries
and report an aged cohort rather than treating one oldest entry as pressure.
The accepted local run was 1.258 s / 11.88 GiB/s; it adaptively reduced reads
from eight to two while retaining eight DMA events. This host remains far below
the earlier 26 GiB/s balanced baseline, so the number is diagnostic only.

### Current-host 8--12 GiB/s diagnosis

Two effects were initially conflated: transient page-cache pressure explained
the worst 8 GiB/s run, but it does not explain the persistent fully warm
12 GiB/s ceiling.

During the first measurements this 62 GiB host had about 1.2 GiB free,
14.7 GiB available, 13.7 GiB total cached data, and all 8 GiB of swap
occupied. Read-only `mincore` accounting found only 10.501/14.958 GiB (70.2%)
of the model resident. After cache-displacing controls, residency fell to
8.256 GiB (55.2%) and the matching fixed-eight `20-53` ZML run fell to
8.04 GiB/s with source-call latency rising to 2.015 ms. Partial residency was
therefore a real confounder for that run.

The host memory state then changed: roughly 30 GiB of process/anonymous memory
was released, leaving 24--26 GiB free and 43--45 GiB available. All four model
shards became 100% resident (14.958/14.958 GiB), while memory PSI remained
zero. Swap still showed 8 GiB used because Linux does not eagerly swap cold
anonymous pages back in merely because RAM becomes free; occupied swap alone
is historical state, not evidence of current pressure.

In that verified fully resident state:

- File-only `pread` with twelve 2 MiB workers reached 61.6--62.3 GiB/s.
- Standalone registered-H2D and streaming SYCL controls were also substantially
  faster; their exact ceilings and queue comparisons are recorded below.
- ZML with reads capped at twelve and DMA fixed at eight still reached only
  11.89 GiB/s. Source calls averaged 0.240 ms, the ready queue remained
  non-empty, and 2 MiB PJRT completion latency averaged 1.275 ms. The source
  was no longer the bottleneck.
- Allowing a `8 -> 12` PJRT-event probe also remained slower: candidate
  completion latency rose to roughly 2.3--2.4 ms and candidate goodput fell to
  about 10 GiB/s, so the controller correctly restored eight.

The B70, PCIe path, Level Zero copy engine, DmaMap/host registration, and
page-cache source can all run substantially faster than the selected PJRT
path. Further controls reject the earlier hypotheses that one SYCL H2D queue,
64 MiB DmaMapped slabs, or serialized completion waits are responsible.

The actual mismatch is PJRT's staging decision. `PJRT_Client_DmaMap` imports
the ordinary allocation successfully, and the low-level SYCL copy path uses a
Level Zero extension to recognize it as asynchronous. However,
`PjRtStreamExecutorClient::ShouldStageHostToDeviceTransfers` first calls the
generic `StreamExecutor::IsHostMemoryPinned`; imported SYCL pointers remain
`usm::alloc::unknown`, so that generic method returns false. PJRT therefore
allocates another pinned buffer and executes `std::memcpy` before every 2 MiB
H2D transfer. More concurrent transfers cause those pageable-to-pinned copies
to contend for memory bandwidth, explaining why cap eight is much worse than
cap two.

The underlying oneAPI path has two distinct allocation registries. DPC++
implements `prepare_for_device_copy` as `urUSMImportExp`; the Level Zero adapter
then calls `zexDriverImportExternalPointer`, which records the range in NEO's
`HostPointerManager`. Copy command construction consults that manager and can
use the imported allocation. In contrast, `sycl::get_pointer_type` calls
`urUSMGetMemAllocInfo`, which maps `zeMemGetAllocProperties`; NEO implements
that query through `svmAllocsManager`, where an externally imported ordinary
allocation is absent, so it returns `ZE_MEMORY_TYPE_UNKNOWN`. The SYCL copy
optimization extension only promises to inform the implementation and permit
faster explicit copies; it does not promise that an ordinary allocation will
be reclassified as `usm::alloc::host`.

The original DmaMap patch handled the second half of the transfer only: its
Level Zero `zexDriverGetHostPointerBaseAddress` check let the low-level SYCL
copy remain asynchronous despite `usm::alloc::unknown`. It did not override
the earlier PJRT staging predicate, so `ShouldStageHostToDeviceTransfers`
branched to `std::memcpy` before that low-level check was ever applied to the
original source block.

For the current toolchain, querying the start and last byte and requiring the
same imported base is the sound driver-authoritative range test. The zex API
exposes membership/base but no size query, and NEO's own
`HostPointerManager::createHostPointerMultiAllocation` checks coverage the same
way: lookup `ptr`, lookup `ptr + size - 1`, and require the same
`HostPointerData`. Imported ranges are contiguous and overlapping registrations
are rejected, so matching bases prove that the complete half-open range is in
one import. XLA also rejects zero size and address overflow first.

Newer DPC++ source contains the cleaner experimental
`sycl_ext_oneapi_register_host_memory`: its contract explicitly treats the
range as a USM host allocation and makes pointer queries reflect registration.
The newer Level Zero v2 adapter implements it through external-memory
`zeMemAllocHost`. The installed oneAPI 2026.0 toolchain used by XLA does not
ship `register_host_memory.hpp` (a compile probe failed at that include), so it
cannot replace `prepare_for_device_copy` yet. Revisit it after the toolchain is
upgraded; then the custom zex query should no longer be necessary.

The committed XLA fix overrides `SyclExecutor::IsHostMemoryPinned`, matching
CUDA/ROCm's platform override. It queries both ends of the requested range via
`zexDriverGetHostPointerBaseAddress`, rejects overflow and zero-length ranges,
and requires both addresses to have the same imported base. This recognizes
subranges of ZML's 64 MiB slabs while rejecting transfers extending beyond the
registered allocation. The same range-aware helper now validates DmaMap and
the low-level asynchronous copy.

### `perf`

The decisive controlled recordings are:

- `/tmp/zml-pjrt-parallel-cap2.data`: selected staging behavior at DMA cap 2.
- `/tmp/zml-pjrt-parallel-cap8.data`: selected staging behavior at DMA cap 8.
- `/tmp/zml-no-stage-cap8.data`: same artifact with PJRT staging disabled for
  the benchmark's known-DmaMapped inputs.

| Flat symbol | staged cap 2 | staged cap 8 | no-stage cap 8 |
|---|---:|---:|---:|
| `_copy_to_iter` | 49.33% | 40.65% | 74.59% |
| `__memmove_avx512_unaligned_erms` | 7.61% | 21.28% | <0.30% |
| `entry_SYSCALL_64` | 12.12% | 10.24% | 3.85% |
| `filemap_get_read_batch` | 4.68% | 3.95% | 2.73% |
| `copy_page_to_iter` | 1.35% | 0.63% | 2.09% |

At cap two the userspace stage consumes about 1.6 billion sampled cycles; at
cap eight it consumes about 4.6 billion. The latter is not more logical data,
but memory-bandwidth contention among concurrent staging copies. With staging
disabled, `memmove` disappears below the report threshold and the profile
returns to the expected kernel page-cache copy into the final DmaMapped blocks.
The controller/queue remain below the material flat-report threshold.

Older profiles `/tmp/zml-adaptive-1b70.data` and
`/tmp/zml-adaptive-4b70-sharded.data` are retained, as are the workspace
recordings. Their adjacent page-cache costs and four-device BFC allocation
costs remain useful historical evidence, but their earlier interpretation as
proof that PJRT staging was absent was incorrect.

The XLA checkout contains an experiment in
`xla/pjrt/se/local_device_state.cc`: SYCL skips the separate callback-stream
path and falls through to the existing `DoHostCallback` plus XLA worker-thread
handoff. This preserves same-stream ordering and the contract that PJRT-facing
callbacks execute on XLA's worker, while replacing the worker's whole-queue
`BlockHostUntilDone`/`queue.wait()` with SYCL's asynchronous `host_task`.

Runtime validation rejects this approach. Five warm runs fell from 26.77 to
11.21 GiB/s on one B70 (-58.1%) and from 26.80 to 10.61 GiB/s on four sharded
B70s (-60.4%). Average 2 MiB DMA completion latency rose to approximately
1.45-1.72 ms in the representative runs, with bad 12/16-event probes reaching
2.6-3.0 ms. A host task is an ordered command in the transfer queue; adding one
after each of 7,723-7,924 small copies serializes expensive host-task/event
dispatch with DMA instead of merely observing completion.

Fresh profiles are `/tmp/zml-hosttask-1b70.data` and
`/tmp/zml-hosttask-4b70-sharded.data`. The former lost 20 of roughly 52k
samples; the latter lost none. `_copy_to_iter` drops to 37.72%/37.46% only
because the new overhead is large: `__memmove_avx512_unaligned_erms` accounts
for 24.19%/21.45%, `urEventWait -> sched_yield` for about 7.3%/3.9%, and SYCL
scheduler command enqueue for 0.92%/1.22%. The former callback worker's
`queueFinish` sample disappears, but polling moves to the host-task execution
thread and becomes larger.

Matching whole-process stat files are `/tmp/zml-hosttask-1b70.stat` and
`/tmp/zml-hosttask-4b70-sharded.stat`. Versus the previous profiles, context
switches rise from 31,906 to 57,482 on one device and from 48,848 to 91,209 on
four. Instructions rise from 5.426 G to 7.344 G and 12.487 G to 14.093 G.
Total cycles fall because the serialized transfer queue throttles concurrent
page-cache copying; this is not an efficiency win. Do not use one ordered
`host_task` per 2 MiB PJRT transfer as the replacement for the queue wait.

The first exact-event experiment kept XLA's single callback worker but replaced
its `queue.wait()` with `sycl::event::wait_and_throw`. The user built it as the
`21-21` artifact after one Bazel dependency-name correction. Controlled warm
runs reject it:

| XLA completion | DMA cap | wall goodput |
|---|---:|---:|
| original `queue.wait()` (`20-53`) | 2 | 21.17 GiB/s |
| single-worker exact event (`21-21`) | 2 | 17.98 GiB/s |
| original `queue.wait()` (`20-53`) | 8 | 11.22 GiB/s |
| single-worker exact event (`21-21`) | 8 | 11.43 GiB/s |

The single-worker exact-event changes were fully reverted. A fresh profile is
`/tmp/zml-exact-event-1b70.data`; it does not justify retaining the patch.

The user then built the parallel exact-event-wait experiment as the `21-57`
artifact. It was also neutral: cap two reached 21.17 GiB/s with 0.174 ms
average completion latency, while cap eight reached 11.01 GiB/s with 1.305 ms
latency. The XLA completion patch was fully reverted. The apparent preference
for two events was caused by concurrent host staging copies, not by the device
or controller: once staging is bypassed, caps two and eight are equivalent.

ZML's pool hypothesis was tested directly. Replacing each lazy 64 MiB slab
with independent 2 MiB allocations and DmaMap registrations produced
19.61 GiB/s at cap two and 10.21 GiB/s at cap eight; completion latency at cap
eight remained 1.308 ms. It adds registration/startup cost without changing
the failure, so the 64 MiB slab design is retained.

The missing pinned-memory predicate was validated without another XLA build by
temporarily setting PJRT's `should_stage_host_to_device_transfers=false` for
the IO benchmark, where every source range is known to be DmaMapped. This is a
diagnostic only and was reverted from ZML source. Results:

| path | DMA cap | wall goodput | average PJRT latency |
|---|---:|---:|---:|
| normal staged artifact | 2 | 21.17 GiB/s | 0.174 ms |
| normal staged artifact | 8 | 11.01 GiB/s | 1.305 ms |
| staging bypassed | 2 | 26.86 GiB/s | 0.088 ms |
| staging bypassed | 8 | 26.90 GiB/s | 0.105 ms |

The no-stage runs kept the ready queue empty, had only 7--10 pool waits, and
made source reads the limiting stage. Under `perf`, cap eight still reached
25.71 GiB/s with 0.131 ms PJRT latency and no reportable `memmove`. This both
restores the prior 26 GiB/s target and proves the XLA override affects
the exact predicate responsible for the regression.

The selected `22-43` artifact validates the committed fix at 26.83--26.90
GiB/s on one B70, without a reportable staging `memmove`. A SYCL executor test
covers unregistered memory, full and interior registered ranges, zero size, a
range extending beyond the import, and unregistration. PJRT completion
delivery, CUDA/ROCm, and ZML's 64 MiB pool are unchanged.

### Residual raw-SYCL and CUDA gap

The fixed artifact is not at the raw B70 limit. On the same host, same oneAPI
runtime, same four warm safetensor files, and one B70:

- playground `src/b70_h2d_queue_depth.cpp` holds registered 2 MiB H2D near
  49--50 GiB/s from depths 2 through 12 (depth one is about 44 GiB/s);
- `src/b70_shared_queue_pipeline.cpp` reaches about 20, 24, 26--27, and 34.5
  GiB/s with 2, 4, 8, and 12 workers; one shared in-order queue and one queue
  per worker are equivalent within noise, rejecting an XLA H2D stream pool;
- ZML/PJRT reaches approximately 26.9 GiB/s with its normal decoupled path.

The extended raw control is
`~/github/uxlfoundation/sycl_playground/src/b70_unique_destination_pipeline.cpp`.
It reports per-operation timings and can scan actual safetensor tensor ranges.
The following changes do not reduce its approximately 34.5 GiB/s ceiling:

- writing all 14.96 GiB to unique destination offsets rather than reusing a
  small destination;
- using the real 291 tensor ranges, including their non-page-aligned source
  offsets (34.58--34.64 GiB/s after warm-up);
- allocating one raw SYCL device buffer per tensor rather than one 14.96 GiB
  allocation (34.44--34.49 GiB/s).

At twelve workers the raw tensor-range control averages about 0.600--0.612 ms
per 2 MiB read and 0.069--0.072 ms per DMA wait. Normal ZML averages about
0.763--0.777 ms per read and 0.241--0.261 ms from PJRT submission to callback.
The PJRT number includes callback delivery rather than only device copy time,
but its extra worker activity also competes with the page-copy stage. A
same-run `perf` comparison is retained as `/tmp/b70-raw-unique.data` and
`/tmp/zml-pinned-fixed12.data`. Raw spends about 19.6 G sampled cycles in
`_copy_to_iter`; ZML spends about 26.6 G for essentially the same bytes, 36%
more. ZML also has one busy `py_xla_callback` thread in `queueFinish` and
several XLA allocation/submission workers; raw has only its twelve lanes.

Controls reject several simpler explanations:

- sorting ZML jobs by absolute file offset remains at 26.83 GiB/s;
- a local scalar `pread` special case slightly lowers reported read latency but
  does not change wall time;
- independent 2 MiB allocations/DmaMap registrations are worse than 64 MiB
  slabs subdivided into 2 MiB blocks: 24.37--24.41 versus approximately
  26.9 GiB/s. The many registrations increase startup, pool waits, and PJRT
  latency, so the slab design remains;
- normal read caps of 8, 10, and 12 give 24.68, 25.73, and 26.88 GiB/s, with
  average read latency of 0.601, 0.683, and 0.777 ms. Twelve is still the best
  continuously active width;
- pinning the twelve ZML reader tasks to CPUs 0--11 collapses to 7.53 GiB/s,
  and restricting the whole process to CPUs 0--15 stalls during PJRT startup.
  Both affinity diagnostics were reverted.

A temporary lane-coupled diagnostic made each reader wait for all PJRT
destinations of its request before reading again. With a fixed initial width,
twelve and sixteen lanes reach 29.53 and 29.51 GiB/s; twelve lanes with only
two PJRT events reach 28.10 GiB/s. At twelve/eight it reduces average read
latency to 0.563 ms and pinned high-water to 24 MiB. This recovers about
2.6 GiB/s and proves that continuously decoupled local reads create harmful
host pressure even while the ready queue looks empty. It was not retained:
hard coupling would idle high-latency remote reads and bypass the intended
independent controller dimensions. A production follow-up should represent
completion-aware request pacing as controller state or a local fast path,
while keeping remote reads independently refillable.

Even that coupled PJRT path remains about 14% below raw B70 SYCL. The correct
XLA revision completes each SYCL PJRT transfer through a per-device callback
worker that calls whole-stream `BlockHostUntilDone`; generic transfer-manager
locking, event/future callbacks, and worker scheduling remain additional to the
raw exact-event wait. Earlier exact-event completion controls show that merely
shortening the callback latency does not recover wall goodput, so this needs a
lower-overhead PJRT transfer/completion design rather than another event-wait
substitution.

The reported RTX 5090 result is approximately 39 GiB/s. Against the measured
raw B70 file-to-device ceiling of approximately 34.5 GiB/s, only about 13% is
an irreducible observed platform difference; the larger 39 versus 26.9 gap
includes the oneAPI PJRT integration and local pacing costs above. If the CUDA
result comes from another host, its CPU, page-cache, memory, and PCIe topology
also make it non-apples-to-apples.

The host has one NUMA node. Simple NUMA selection therefore does not explain
the residual cost, and affinity restriction is empirically harmful here.

Current whole-process `perf stat` (including PJRT/device initialization):

| Counter | adaptive one B70 | adaptive four-B70 sharded |
|---|---:|---:|
| task-clock | 7.330 s | 9.212 s |
| cycles | 37.995 G | 47.577 G |
| instructions | 5.426 G | 12.487 G |
| context switches | 31,906 | 48,848 |
| CPU migrations | 1,041 | 2,745 |
| page faults | 278,886 | 987,208 |

The following recordings and comparison are the earlier static-vectored
investigation retained for historical context.

Final recordings:

- `/tmp/zml-vectored-final-1b70.data`
- `/tmp/zml-vectored-final-4b70.data`
- old four-B70 baseline: `/tmp/zml-vectored-baseline-4b70.data`

Flat cycle profiles:

| Symbol | old four B70 | final one B70 | final four B70 |
|---|---:|---:|---:|
| `_copy_to_iter` | 67.06% | 70.55% | 64.28% |
| `filemap_get_read_batch` | 3.81% | 6.15% | 5.55% |
| `copy_page_to_iter` | 1.79% | 2.76% | 2.55% |
| `filemap_read` | 1.53% | 2.60% | 2.39% |
| `std::vector<...>::resize` | 1.72% | <0.5% | 1.64% |
| `xe_hw_fence_signaled` | <0.5% | <0.5% | 0.57% |

No pageable staging copy, userspace sharding copy, or slab poison/memset appears
in the final profiles. The remaining dominant CPU cost is Linux copying warm
page-cache pages into DmaMapped anonymous memory. DmaMap pins/registers those
pages for device access; it does not let ordinary file-backed page-cache pages
become the PJRT transfer source without this kernel copy.

Selected `perf stat` comparison:

| Counter | old one B70 | final one B70 | old four B70 | final four B70 |
|---|---:|---:|---:|---:|
| task-clock | 6.284 s | 7.602 s | 8.089 s | 8.321 s |
| cycles | 32.906 G | 39.506 G | 42.105 G | 43.298 G |
| instructions | 5.158 G | 5.549 G | 10.420 G | 10.911 G |
| context switches | 42,162 | 30,638 | 39,594 | 41,270 |
| CPU migrations | 3,558 | 1,369 | 2,680 | 1,946 |
| page faults | 275,781 | 278,678 | 985,254 | 988,597 |

The one-device path spends more aggregate CPU on many 2 MiB reads/transfers,
but finishes faster. In the sharded case—the original CPU concern—task-clock
is only 2.9% higher while median wall goodput is 7.7% higher, and the staging
and sharding copies are gone.

One initial post-change profile and the fifth pass of an early request-width
sweep were invalidated when another user's 32-way XLA/oneAPI compiler job
saturated the host. `/tmp/zml-vectored-new-1b70.data` is retained only as a
contaminated artifact and was not used for final conclusions.

## Rejected hypotheses and retained observations

- DmaMap itself did not make CPU writes materially slower. It changes page
  registration/device visibility, not the basic CPU cache hierarchy.
- Anonymous DmaMapped memory means ordinary anonymous virtual memory allocated
  by ZML and then registered with PJRT; it is not file-backed and not a hidden
  staging allocation.
- Huge pages reduce translation/registration pressure but do not remove the
  page-cache-to-anonymous copy or make CPU writes bypass cache coherency.
- A file-backed mmap used directly for DMA reached about 21.4 GiB/s; mmap plus
  an explicit copy reached about 23.7 GiB/s. Neither beat the final preadv into
  DmaMapped blocks.
- The useful local 12-read knee is a pipeline balance, not twelve dedicated DMA
  engines. Reads and transfers overlap, while the one global byte budget owns
  blocks through read, ready, and PJRT completion states.
- Per-device pools are unnecessary for this client: one DmaMapped slab was used
  successfully by all four B70 devices. A future PJRT client spanning distinct
  driver handles would need separate runtime validation.

## Validation and commands

Passed after the final edits; the 2026-07-21 controller follow-up repeated the
ZML test and CUDA build, then ran one-device oneAPI local and S3Proxy loads:

```text
./bazel.sh test --nocache_test_results //zml:test //zml/io/vfs:test
./bazel.sh build --config=release --@zml//platforms:oneapi=true \
  //examples/io:playground
./bazel.sh build --config=release --@zml//platforms:cuda=true \
  //examples/io:playground
```

The CUDA command compiled ZML against the existing PJRT artifact; it did not
build XLA.

Four-device oneAPI sharded and replicated loads completed repeatedly. CUDA
release compilation passes, but this host has no RTX GPU, so the requested RTX
5090 runtime sweep remains for the CUDA machine.

Useful runs:

```text
# Local/default; all values may still be overridden.
./bench_file.sh

# Remote request size is selected automatically from the S3 VFS hint.
LATENCY_MS=10 SPEED_MIB=1000 ./bench_s3.sh

# Preserve workspace perf.data by choosing a different output.
PERF_DATA=/tmp/zml-loader.data ./profile_file.sh

# Four B70s.
ONEAPI_DEVICE_SELECTOR='level_zero:*' ./bench_file.sh

# Four B70 replicas.
ONEAPI_DEVICE_SELECTOR='level_zero:*' ZML_LOAD_SHARDING=replicated ./bench_file.sh
```

## Workspace boundary

The user's benchmark selector behavior remains; the scripts carry read/DMA
caps, block/pinned controls, optional `ZML_LOAD_SHARDING`, and `PERF_DATA`.
They deliberately leave request size unset so VFS automatic sizing is tested;
`ZML_LOAD_READ_REQUEST_MIB` remains an inherited explicit override.
Workspace `perf.data` and `perf.data.old` were not overwritten or removed.
There is no `CTX.md.orig`. The user built and validated the current XLA SYCL
pinned-range fix. Do not build XLA unless that instruction changes again.

The adjacent production monorepo has an intentional uncommitted change in
`llmd/main.zig`: its five VFS registrations now use `registerBackend` so the
file/HTTP/HF/S3/GCS source hints and counters survive into the loader. In
particular, this fixes production HF loads incorrectly resolving to the 2 MiB
local/default request size instead of the advertised 32 MiB minimum. Its
untracked `log.txt` belongs to the user. A oneAPI `//llmd:llmd` build was
started but interrupted during dependency compilation and was not reported as
validation.

The SYCL playground is not part of the ZML Git checkout. The direct-I/O
experiment added `src/b70_direct_io_pipeline.cpp` and its CMake target there;
it compiles and produced the direct/cold-buffered results above.
