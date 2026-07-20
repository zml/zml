# Adaptive Loader Context

Snapshot: 2026-07-20, branch `brabier/adaptive-concurrency`, base commit
`181a8c0a` (`oneapi dma`). The working tree contains user changes to the three
benchmark scripts and local oneAPI artifact selection, plus the loader changes
described below; inspect `git diff` before assuming they are committed.
`bench_file.sh`, `bench_s3.sh`, and `profile_file.sh` currently build oneAPI and
select `ONEAPI_DEVICE_SELECTOR=level_zero:0`. The checked-out XLA tree is clean
at `9dac91281a` (`[XLA:GPU][oneAPI] Enable PJRT_Client_DmaMap for SYCL`). Do not
build XLA in this task; the user owns that build.

This is a durable implementation and performance handoff, not a chronological
lab notebook. Current source is authoritative if it disagrees with this file.
`RESEARCH.md` contains the controller theory and citations.

## Goal and Current State

Minimize finite model-load wall time from an unknown source to accelerator
memory. Among configurations within 3% of the best committed H2D goodput, use
the fewest pinned DMA buffers and pageable staging blocks. Sources may range
from warm local files to high-latency object storage, and many complete loads
last only seconds, so adaptation must start quickly.

The controller optimizes physical bytes successfully committed by PJRT per
second. Storage, ordered, and submitted bytes are stage diagnostics. The final
load record separately reports logical model goodput. These rates differ for
replicated placement: one validation read 14.96 GiB once and committed
59.83 GiB to four devices.

Current production defaults are:

```text
parallelism                         32  maximum DMA lanes
initial_parallelism                  8  initial read/DMA lanes
adaptive_parallelism              true
max_read_parallelism              null  resolves to max(32, parallelism)
read_chunk_size                  32 MiB
max_staging_bytes                 1 GiB
max_pinned_buffers_per_device        33
pinned_buffer_size                  auto  2 MiB for oneAPI, else 32 MiB
transfer_quantum_size            256 MiB
```

Pools allocate lazily. With the defaults, each device initially admits at most
eight DMA lanes and nine pinned buffers: 18 MiB on oneAPI, or 288 MiB at
32 MiB on other platforms. Longer or higher-latency
loads may explore up to 32 lanes and 33 buffers. An explicit
`pinned_buffer_size` always overrides the automatic policy.
`adaptive_parallelism=false` is retained as a fixed-path testing escape hatch,
not the recommended production setting.

The 32/8/33 defaults were revalidated on 2026-07-19. Keeping the initial width
at eight minimizes startup allocation; fast local loads now reliably probe to
the measured 12-lane knee, while slow sources expand read-ahead to 32 without
also widening DMA. Raising the initial width to 12 was neutral on low-latency
S3Proxy and used four more pinned buffers, so it was not retained.

Linux CUDA and oneAPI pinned mappings automatically request 2 MiB alignment
and `MADV_HUGEPAGE` before `PJRT_Client_DmaMap`. This is best effort: failure
falls back to ordinary pages. There is deliberately no public page-mode option.
The oneAPI path now uses ordinary anonymous host memory registered through
PJRT DmaMap; it no longer relies on PJRT host-pinned allocations or the
plugin's pageable staging copy.

`Platform.warmupDeviceAllocators()` creates and immediately deletes a one-byte
default-memory buffer on every addressable device. This idempotently forces
lazy device allocator/BFC initialization before weight loading. Both
`examples/io` and `examples/llm` call it immediately before loading buffers;
the LLM example does so after compilation. Warmup is outside the IO example's
load timer. It moves first-use work but does not reduce total startup time by
itself.

## Implementation Map

- `zml/io.zig`
  - `LoadOpts`, `load`, adaptive/fixed dispatch;
  - `LoadMetrics` and probe epoch attribution;
  - `MemoryWriter`, direct/sharded writers, and H2D completion;
  - `AdaptiveLoadController` pure decisions;
  - `AdaptiveLoadRuntime` sampling, probe activation, and pool limits;
  - pipeline, tensor, and DMA-lane state machines;
  - inline controller and data-plane tests.
- `zml/safetensors.zig`
  - exact concurrent `TensorReader.readPositionalAll` and tests.
- `zml/mem.zig`
  - lazy, bounded, trimmable `DynamicBufferPool`;
  - CUDA/oneAPI `DmaMapAllocator` transparent-huge-page advice.
- `zml/platform.zig`
  - `Platform.warmupDeviceAllocators()` and its idempotence test.
- `stdx/Io.zig`
  - runtime-adjustable `LimitedGroup`, cancellation-safe admission, and
    synchronous admission for work already running on a suitable task.
- `examples/io/main.zig`
  - benchmark environment mapping, platform warmup, loader logging.
- `bench_file.sh`, `bench_s3.sh`, `profile_file.sh`
  - local/S3/perf entry points; verify their platform selection first.

Relevant history: `a8ffb40c` introduced adaptive concurrency and research,
`c1ac887e` added controller logging, and `4f1b655f` introduced the positional
two-stage adapter before later fixes.

## Data Plane

The adaptive CUDA/oneAPI path is a bounded two-stage pipeline with a direct
bypass:

```text
source positional reads ──┬──> pinned writer window ──┐
                          └──> pageable read-ahead ────┼──> ordered writer
                                                       └──> async PJRT H2D
                                                            └──> committed bytes
```

One controller owns four dynamic limits: outstanding read workers, admitted
DMA lanes, pinned buffers per device, and active pageable blocks. A single
owner is important because separate read and DMA controllers would react to
the same intermediate queue and fight each other.

`pinned_buffer_size` is physical allocation size. `transfer_quantum_size` is a
logical scheduling/fairness boundary and allocates no memory. With 32/256 MiB,
a lane can submit up to eight full pinned buffers before yielding. Do not make
the quantum unbounded without validation: it limits how long a large tensor
owns a lane, flushes partial shard buffers, and advances controller accounting.

Adaptive eligibility requires CUDA or oneAPI, more than one read chunk of
logical data, and useful read or DMA concurrency. Other cases use the fixed
buffered loader. The DMA lane ceiling is:

```text
min(tensor_count, parallelism,
    max(1, max_pinned_buffers_per_device - 1))
```

The one spare pinned buffer permits global flip-flop progress while a previous
H2D is in flight, but it does not give every lane an independent second buffer.
Guaranteed double buffering for `N` simultaneously active one-device shard
writers requires `2*N` buffers per device. The smaller shared pool is an
intentional resource/performance tradeoff, not a strict per-lane double-buffer
invariant. Pageable capacity is
`floor(max_staging_bytes / read_chunk_size)`; a tail read still leases a full
block. Setting `max_staging_bytes=0` disables staging but leaves direct-path
adaptation active.

Pinned registration is not device-specific for the current CUDA/oneAPI DmaMap
path. `PJRT_Client_DmaMap` is client-scoped. ZML constructs one `DmaAllocator`
per device, but its CUDA/oneAPI implementation retains only the common
`Platform`, so these allocators are operationally identical. XLA registers via
addressable device zero; oneAPI then imports the pointer on the Level Zero
driver, whose host-pointer manager creates graphics allocations for all devices
owned by that driver. XLA's multi-GPU test explicitly transfers a single
DmaMapped pointer through two devices.

This is driver-scoped, not universally device/runtime independent. A different
device owned by the same Intel Level Zero driver handle is covered by the
driver-wide import. A device exposed by another driver handle is not: current
PJRT registers only through `addressable_devices_[0]` and would need to import
and release once per distinct driver/context. The DPC++
`SYCL_EXT_ONEAPI_COPY_OPTIMIZE` macro only proves toolchain API availability;
the installed Level Zero adapter/driver must also expose Intel's
`zexDriverImportExternalPointer`, `zexDriverReleaseImportedPointer`, and, for
XLA's post-import verification, `zexDriverGetHostPointerBaseAddress`. An older,
non-Intel, disabled, or resource-exhausted runtime can therefore reject DmaMap.
ZML's current CUDA/oneAPI `DmaMapAllocator` has no pageable fallback: a failed
registration makes that pool-block allocation fail and ultimately fails the
load rather than silently using the old staged path.

Upstream Intel NEO history gives several distinct support floors. The internal
host-pointer manager appeared in `20.49.18626` (December 2020), became
default-enabled in `21.20.19883` (May 2021), and became externally callable via
the three `zexDriver*` extension-function names in `22.16.22992` (April 2022).
Direct API unit tests followed in `22.21`, a functional TBX black-box test in
`22.23`, and user documentation in `22.30`. DPC++ added
`prepare_for_device_copy` on top of these functions in commit `65cc0cf` on
2023-07-19. The names and signatures have remained present since introduction,
but both layers still label the interface experimental.

Use `22.16.22992` only as the API-existence floor. Prefer at least
`25.44.36015.5` for arbitrary multi-root configurations: that release fixed an
undersized per-root allocation vector when visible root-device indices were
nonzero or sparse. Contiguous roots 0..N-1, including the tested four-B70
driver, were not affected. `26.18.38308.1` first advertised the formal
`ZEX_driver_import_host_pointer` version 1.0 through extension enumeration;
older supporting drivers expose only the callable function names. Enumeration
is therefore neither necessary on old drivers nor sufficient when the driver's
`EnableHostPointerImport` debug setting disables the manager. Dynamic lookup,
an attempted import, and a successful post-import query are authoritative.

The current host has `libze-intel-gpu1 26.22.38646.6`; its library contains the
extension name and all three functions, includes the multi-root fix, and has
successfully handled both one- and four-B70 DmaMapped loads. Confidence for this
exact stack is high. Cross-version/device confidence is medium-high for current
Intel NEO, but not equivalent to a standardized Level Zero guarantee: hardware
CI evidence in the driver repository is limited, the 2025 topology fix was
substantive, and the private API only gained an advertised version in 2026.

The XLA post-import Level Zero query is important rather than redundant. The
DPC++ Level Zero adapter discovers import/release dynamically, but its current
`urUSMImportExp` returns success when the extension is absent and discards the
native `zexDriverImportExternalPointer` result. Verification prevents such a
no-op or failed import from being recorded as DmaMapped. Retain the query unless
the implementation moves to an API with reliable capability and error
reporting. Also preserve non-overlapping ownership and wait for every DMA before
unmap: the driver has no independent-owner reference count for duplicate
subrange imports, and the SYCL copy-optimize contract declares overlapping
registrations undefined. The newer experimental `register_host_memory` API has
an explicit device aspect and error model, but its current Level Zero support
is narrower and it does not universally guarantee anonymous/huge-page mappings,
so it is not yet an unqualified portable replacement.

Consequently, separate per-device pools are a ZML resource-accounting and
fairness policy, not a correctness requirement. A single client-wide pool could
lease any buffer to any device as long as the lease lasts until every DMA using
it completes. Merging pools is not mechanically free: the controller's current
limit is per device, a shared pool needs anti-starvation/fairness rules, and
active shard writers may retain partially filled buffers. It is nevertheless a
valid optimization experiment, especially because current pinned-capacity
probes expand every device pool together.

Tensor sources, writers, and buffers are created lazily. A direct read aliases
a real pinned writer window and is reserved by a DMA lane. Extra read
parallelism uses the bounded pageable pool. Pageable reads receive monotonically
increasing per-tensor tickets, may complete out of order, and are consumed from
a ring in exact offset order. For every tensor:

```text
0 <= next_commit <= next_read <= total
```

`MemoryWriter` therefore receives file-order bytes even when reads finish out
of order. Shard dispatch sorts placement ranges, uses the zero-copy pinned alias
where possible, and copies mirror/replicated ranges to their writers.

When pageable staging is zero, a direct read executes on its persistent DMA
lane task while still taking `read_group` admission. This avoids a read-task
handoff and a resume-task handoff for every small pinned chunk. If staging is
enabled, reads retain the detached path so pageable work can run ahead of DMA.
Direct reads retain their admitted DMA lane until completion. Do not detach a
tensor that owns a direct pinned window into the shared ready queue: queued
owners can hold every pool block while new lanes block creating writers,
causing circular wait. Pageable tensors may detach only after parking their
writer safely. Once a tensor owns a transfer quantum, it schedules its next
direct read before yielding; yielding at that boundary previously caused a
no-progress ready-queue convoy.

Pinned shard writers alternate buffers/completion contexts and wait before
reuse. Parking commits without replenishing the public window. Acquiring a new
buffer merely to park can deadlock when lanes are retiring.

Concurrency structures:

- `read_group`: positional read admission/execution, either inline on a direct
  DMA lane or detached for staged reads, with a dynamic limit;
- `dma_group`: lane execution, dynamic limit;
- `staging_group`: pageable acquisition/prefetch setup;
- `resume_group`: staged and capacity waits outside DMA admission;
- FIFO ready queue of heap-stable tensor states;
- direct-event and direct-capacity wait structures.

Critical invariants:

- each tensor index is claimed once and its bytes reach `MemoryWriter` in order;
- every pageable block and direct reservation is released exactly once;
- `ready_bytes` represents completed, not-yet-consumed pageable bytes;
- queued nodes remain at stable addresses and detached writers are parked;
- limit reductions affect future admission and never cancel useful work;
- pinned/pageable allocation never exceeds public caps;
- probe bytes count only after physical capacity activation and epoch match;
- completion occurs only after all tensors finish or the first error wins.

## Controller and Metrics

Important byte positions are:

- `storage_bytes`: completed source reads;
- `direct_read_bytes` / `staged_read_bytes`: routing split;
- `ready_bytes`: completed pageable data awaiting ordered consumption;
- `ordered_bytes`: bytes delivered in tensor order;
- `logical_submitted_bytes`: logical bytes crossing writer fences;
- `submitted_bytes`: physical bytes submitted to PJRT;
- `committed_bytes`: physical bytes whose completion callback succeeded.

Only committed-byte goodput is the control objective. Diagnostics include
byte-weighted read/DMA latency, pageable copy time and ready age, pool/admission
waits, lane/writer high-water marks, H2D queued bytes, and the union of DMA
starvation intervals. Concurrent wait ratios may exceed 100%; they are summed
task time, not occupancy. Starvation intervals are unioned so multiple waiting
tensors cannot count the same GPU-idle interval repeatedly.

Do not interpret read latency or read-admission wait as congestion. Useful
source concurrency raises both. Read pressure requires data accumulating ahead
of DMA: sustained ready occupancy/age/growth. DMA latency is considered only
for windows that committed at least 32 MiB, preventing tiny startup transfers
from establishing a false baseline.

The runtime samples every 25 ms. Startup windows are 50-100 ms with a 32 MiB
pipeline-progress target. Steady windows are 100-250 ms with a 64 MiB committed
target. With no progress for 100 ms and source demand, startup may double reads
before committed bytes exist. Tiny startup reads do not establish a committed
baseline or prevent stalled-source bootstrap. Probes are suppressed when
estimated work remaining is below 250 ms; the former 500 ms cutoff prevented a
useful 8 -> 12 probe on sub-second warm-file loads.

Each probe changes one dimension, records the complete baseline tuple, and
waits to publish its epoch until the requested capacity is physically active.
Examples: a read increase must reach its active-read high-water; a DMA increase
requires distinct lanes to submit; pool increases require actual allocation;
reductions wait for old work to drain. Activation or live progress times out
after 5 s, which is intentionally long enough for a 1 s-latency source.

After activation, reads, slots, and writers carry the probe epoch. Evaluation
requires at least 64 MiB of matching committed bytes and 200 ms of active time.
Epoch bytes prove that candidate work reached PJRT; increase probes compare the
better of epoch goodput and cumulative aggregate goodput since physical
activation against a smoothed stable baseline. This avoids deciding from two
bursty 50 ms windows. An ordinary increase needs 3% improvement with no
relevant pressure. Read/staging expansion on a source already classified as
slow is retained while filling the bounded read-ahead budget; those probes are
started by source starvation, and later reductions can recover memory once DMA
has remained fed.

A resource reduction is accepted if it remains within 3% of the better of its
baseline and global peak, has no relevant pressure, and starvation is at most
10%. Resource probing requires two continuous seconds without meaningful DMA
starvation; a failed reduction cools down for five seconds. Failed performance
probes cool down for two seconds. Rollback restores the entire knob tuple and
trims newly free blocks when possible.

Ready growth is not congestion while DMA is starved, and a source classified
as slow is allowed to complete in bursts within the hard staging cap. Likewise,
a slow staged source must keep DMA fed for two seconds before a transient drain
burst can trigger more DMA lanes. Direct fast-source DMA startup uses a gradual
ladder: 8 -> 12 -> 16 instead of 8 -> 16.

Decision order, condensed:

1. Hold while capacity activation is pending.
2. Roll back or reduce DMA on hard H2D pressure.
3. Roll back or reduce reads/staging on sustained ready pressure from a fast
   source that is continuously feeding DMA.
4. Score a sufficiently long, attributed probe.
5. Bootstrap reads/staging on zero-progress slow sources.
6. When DMA-starved, expand reads/staging, then DMA when source fanout is ample.
7. After DMA has remained fed for two seconds, periodically probe fewer reads,
   pinned buffers, staging blocks, then DMA lanes.
8. Otherwise probe more DMA, then pinned capacity, or remain steady.

The implementation is a byte-attributed probe/rollback controller inspired by
BBR startup and Gradient/Vegas queue awareness. It is not Gradient2. Preserve
the single control owner, committed-byte objective, hard pool bounds, and
capacity/epoch protocol if replacing the policy.

## Cancellation

The first error atomically closes the scheduling fence, wakes waiters, and
signals completion. Every async scheduling path participates in the fence.
Teardown waits for admitted schedulers, cancels read/staging groups, drains
resume/DMA work, waits or cleans events, returns buffers, and destroys stable
states. Queued read jobs use uncancelable admission so they can still clean up
after group cancellation. A blocking transport read may still need to return;
cancellation does not promise transport-level interruption.

Do not simplify teardown ordering without failure-injection tests. Many events
and pool pointers refer to storage owned by the top-level `load` frame.

## Logging and Diagnosis

Adaptive logs use `zml/io/load`. The high-value records are `configured`,
`controller started/stopped`, `source bootstrap`, the four window/concurrency
records, `limits updated`, `probe capacity active/timeout`, `waiting for
progress`, cancellation, and final `completed`.

```sh
rg -n "configured:|completed: adaptive|controller stopped:" log.txt
rg -n "limits updated:|probe capacity|source bootstrap" log.txt
rg -n "window control:|window throughput:|window pressure:" log.txt
rg -n "pipeline concurrency:|waiting for progress:" log.txt
rg -n "pipeline cancellation requested|error" log.txt
```

Interpretation traps:

- `active_dma_streams` on tensor-start logs includes parked writers; use
  `pipeline concurrency dma_lanes` for admitted lanes.
- Lane snapshots may be zero between short executions; correlate submissions,
  committed goodput, and utilization.
- Read/staging wait can exceed 100% because it sums concurrent waits.
- `ready=0` with high starvation means data is consumed immediately but arrives
  too slowly.
- High read goodput with low committed goodput indicates a downstream bottleneck.
- Repeated capacity timeouts mean the candidate was never exercised; inspect
  pools and scheduling before changing timeouts.

Zig formatting permits at most 32 arguments, so keep the window report split.
Avoid per-lane hot-loop logging: one convoy generated a 767 MB, 7.6-million-line
log without adding useful state.

## Benchmark Controls

`examples/io/main.zig` accepts:

```text
ZML_LOAD_ADAPTIVE
ZML_LOAD_PARALLELISM
ZML_LOAD_INITIAL_PARALLELISM
ZML_LOAD_MAX_READ_PARALLELISM
ZML_LOAD_MAX_PINNED_BUFFERS_PER_DEVICE
ZML_LOAD_PINNED_BUFFER_MIB
ZML_LOAD_TRANSFER_QUANTUM_MIB
ZML_LOAD_READ_CHUNK_MIB
ZML_LOAD_MAX_STAGING_MIB
```

When `ZML_LOAD_PINNED_BUFFER_MIB` is absent, the IO example leaves
`LoadOpts.pinned_buffer_size=null`: oneAPI selects 2 MiB and other targets
select 32 MiB. This decision deliberately does not inspect tensor URIs. A
`file:` or path source may reside on slow network-attached storage, while a
remote-looking source may be cached locally, so the URI is not a reliable
performance signal. Supplying the environment variable is an explicit
override.

The playground always uses normal `Platform.auto` behavior. Temporary XLA GPU
allocator/preallocation environment controls used during investigation were
removed. Historical allocator comparisons below must not be read as supported
playground options.

## Performance Evidence

Unless noted otherwise, the accelerator workload is Llama 3.1 8B: 291 tensors,
14.96 GiB logical bytes, release IO playground, and warm files under
`~/s3proxy/data/lfm`. Use medians and alternate A/B order; first-use allocation,
cache state, and oneAPI variance are large enough to mislead single runs.

### Source concurrency

A Mac CPU-only fixed-loader sweep measured source/host concurrency only; it did
not exercise the CUDA/oneAPI adaptive pipeline:

```text
source/profile                  1      2      4      8     16     32     64
warm file MiB/s              1730   3125   5507   9306   9991   8983      -
proxy 1ms/10000 MiB/s         990   1697   2445   2914   3257   3211      -
proxy 250ms/1000 MiB/s          -    105    213    419    782   1170   1130
```

This established that 16 workers can be useful locally and 32 for latency-bound
storage. It motivated `max_read_parallelism=max(32, parallelism)`. The ceiling
does not preallocate 32 blocks or start 32 reads.

### CUDA warm local and remote

An early fixed legacy sweep with 256 MiB pinned blocks found its warm-local
concurrency knee at four:

```text
workers          1       2       4       8      16
elapsed (s)   2.123   1.490   1.157   1.254   1.751
MiB/s          7215   10276   13200   12212    8749
```

After the first controller fixes, adaptive converged from two to four, remained
entirely direct, and had a 1.168 s median, 0.95% behind fixed four. Forced width
eight completed 20/20 after the teardown fix and had about a 1.25 s median.

For S3Proxy at 10 ms/1000 MiB/s, fixed widths 1/2/4/8 took
50.788/26.102/14.577/9.053 s. The corrected adaptive path took 5.066-6.100 s;
a known-remote start at eight with read cap 32 and 1 GiB staging took 3.387 s.
At 1000 ms latency, final adaptive runs completed in 89.057 s at 1000 MiB/s
and 84.241 s at 100 MiB/s, reaching 32 reads/eight DMA lanes. These runs drove
the 5 s probe deadline and executable-staging-capacity fixes.

The 2026-07-19 default/convergence sweep used the current 32 MiB pinned buffers,
2 MiB THP policy, allocator warmup, preallocated BFC, CUDA device 1, and the
same 291-tensor/14.96 GiB model. Local width comparisons alternated order; S3
results came from `bench_s3.sh` with the stated proxy controls. Final results:

```text
source/profile                 runs    elapsed             final reads/DMA/pinned/staging
warm file                         5    0.428 s median       12 / 12 / 13 / 0
S3 10ms / 1000 MiB/s              3    2.912 s median       32 /  8 /  9 / 32
S3 250ms / 1000 MiB/s             1   11.462 s              32 /  8 /  9 / 32
S3 1000ms / 100 MiB/s             1   43.175 s              32 /  8 /  9 / 32
```

The warm-file range was 0.428-0.454 s and the 10 ms S3 range was
2.888-2.913 s. A seven-round balanced local cap sweep measured 0.503/0.453/
0.453 s medians at 8/12/16 lanes, so 12 is the local knee. Retaining
`initial_parallelism=8` plus the 250 ms finite-tail cutoff achieved that knee
in every final local run without paying for 13 pinned buffers at startup.

For 10 ms S3, initial widths 8 and 12 were effectively tied after convergence
(2.987 and 2.989 s medians in the direct A/B), so eight remains the lower-cost
universal start. Compared with the pre-change runs in this iteration, the final
controller improved the warm-file median from 0.503 to 0.428 s, the 10 ms S3
median from 3.664 to 2.912 s, and the 250 ms run from 12.292 to 11.462 s. A
same-iteration 1000 ms/100 MiB/s run with repeated read rollback took 47.284 s;
the final no-churn run took 43.175 s. Older 84-89 s measurements above predate
several data-plane and controller changes and are context, not an isolated A/B.

The final traces explain the universal defaults: local files stayed fully
direct and used the gradual 8 -> 12 DMA step; every S3 profile expanded only
read/staging capacity to 32 while retaining eight DMA lanes and nine pinned
buffers. Pools remained lazy, so the 32/33 hard caps did not impose their full
memory footprint on these runs. `parallelism=32`, `initial_parallelism=8`, and
`max_pinned_buffers_per_device=33` therefore remain unchanged.

### CUDA huge pages, dTLB, and CPU attribution

This section preserves enough evidence and procedure to restart the CPU/TLB
investigation, but it is closed unless a new regression or platform warrants
it.

Host: AMD Ryzen Threadripper PRO 9985WX, Linux 6.17.0-35, 4 KiB base pages,
2 MiB PMD THP, THP and defrag in `madvise` mode. While buffers were live,
`/proc/$PID/smaps` showed five 256 MiB anonymous DMA VMAs with exactly
256 MiB `AnonHugePages` each and `VmFlags` containing `hg`: 1.25 GiB was fully
THP-backed. `KernelPageSize`/`MMUPageSize` still reported 4 KiB on this kernel,
so use `AnonHugePages` plus `hg`, not those fields alone. A successful
`madvise` is only a hint; recheck `smaps` on a new host.

Five alternating adaptive A/B pairs, all converging to four reads/four DMA
lanes/five pinned buffers/zero staging, measured:

```text
metric                         base pages          2 MiB THP       change
loader elapsed                    1.054 s            0.879 s       -16.6%
logical goodput             14525 MiB/s       17429 MiB/s       +20.0%
minor faults                     314290             314281          0.0%
instructions                     7.207 G            4.811 G       -33.2%
cycles                          11.208 G           10.382 G        -7.4%
L1 DTLB misses                  11.603 M            7.788 M       -32.9%
L1+L2 misses/page walks          5.668 M            2.152 M       -62.0%
```

Every pair favored THP. Unchanged minor faults show that roughly 314K faults
were first-touch activity, not a measure of later page-walk cost. THP was kept
as an unconditional best-effort CUDA policy. Twelve later balanced pairs with
32 MiB pinned buffers still improved median load time from 0.584 to 0.506 s
(13.3%).

On this AMD CPU, generic `dTLB-loads` is actually
`ls_l1_d_tlb_miss.all`, and `dTLB-load-misses` is
`ls_l1_d_tlb_miss.all_l2_miss`. Perf's printed percentage is therefore the
fraction of L1 misses that also missed L2, not DTLB misses per data load. Use
the native `ls_l1_d_tlb_miss.*` events and split user/kernel and page sizes.

THP-mode native-event profiling found about 2.40 M L2 misses/walks: 1.31 M
user and 1.09 M kernel. User walks were overwhelmingly 4 KiB (~1.30 M) rather
than 2 MiB (~16K). Kernel samples centered on page clearing and
`_copy_to_iter`. Non-precise PMU sampling placed about 72% of user 4 KiB walk
samples in/around page faults and
`BFCAllocator::RegionManager::AddAllocationRegion`/`BFCAllocator::Extend`.

`smaps` supplied the matching explanation: besides the fully THP-backed DMA
VMAs, a ~1.034 GiB anonymous VMA had ~903 MiB resident, zero `AnonHugePages`,
and `THPeligible: 0`. It is very likely BFC's host-side allocation-region index.
XLA uses a 256-byte minimum allocation and stores one 8-byte handle per slot;
indexing the RTX 5090's ~32.6 GiB address space therefore needs about 1.019 GiB.
This metadata accounts for most faults and remaining user 4 KiB walks; it is
not a ZML pinned DMA buffer.

After `kernel.perf_event_paranoid` was set to 0, AMD IBS through `perf mem`
provided precise data-address/page-size evidence. The system-wide capture was
restricted to CPUs 0-15 and the workload pinned to the same CPUs. That affinity
reduced goodput, so elapsed time was discarded. Weighted samples were 91.8%
kernel `_copy_to_iter` and 6.5% `filemap_get_read_batch`; TLB outcomes were
90.61% L1 hits on 2 MiB pages, 3.52% L1 hits on 4 KiB pages, and 2.29% L2 hits
on 2 MiB pages. The steady transfer was dominated by the ext4/page-cache copy
into registered buffers, not page walks.

Independent ceilings confirmed it:

- four warm shard files: ~13.0 GiB/s sequential `dd`, ~29.3 GiB/s with one
  process per shard;
- CUDA pinned-host-to-device microbenchmark: ~57.0 GB/s with one through eight
  streams, including `MADV_HUGEPAGE`-backed registered memory;
- loader read stage: commonly 18-22 GiB/s and DMA-starved.

Thus more CUDA streams or 1 GiB pages do not address the current warm-file
bottleneck. A meaningful next prototype would mmap shard ranges, register
suitable file-backed mappings, and DMA directly from page-cache pages, tested
separately for warm and cold behavior. File-order/scatter-read scheduling may
reduce syscall/readahead overhead, but cannot remove `_copy_to_iter` in the
current `pread` design.

To resume the investigation, build release CUDA, verify live backing in
`/proc/$PID/smaps`, collect native events without multiplexing where possible,
then use IBS `perf mem` for data-address attribution:

```sh
./bench_file.sh

perf stat -e task-clock,page-faults,instructions,cycles,\
ls_l1_d_tlb_miss.all,ls_l1_d_tlb_miss.all_l2_miss -- \
env CUDA_VISIBLE_DEVICES=1 \
bazel-bin/examples/io/playground load ~/s3proxy/data/lfm/

# IBS requires system-wide access on this host. Pin record and workload to the
# same small CPU set, then filter perf mem report to the playground command.
perf mem record -a -C 0-15 -o perf.data -- \
taskset -c 0-15 env CUDA_VISIBLE_DEVICES=1 \
bazel-bin/examples/io/playground load ~/s3proxy/data/lfm/
perf mem report -i perf.data
```

Do not compare affinity-restricted IBS elapsed time with unrestricted runs.

#### 1 GiB pages

Not benchmarked. The CPU/kernel support 1 GiB HugeTLB, but the active THP sizes
stop at 2 MiB and the 1 GiB HugeTLB pool has zero pages. Guaranteed 1 GiB pages
would require explicitly reserved, unswappable HugeTLB memory, preferably at
boot. An arena also weakens lazy allocation/trimming: five 256 MiB buffers need
two 1 GiB pages, reserving 2 GiB for 1.25 GiB of live capacity.

The already-backed DMA footprint needs only 640 2 MiB translations, while the
dominant remaining user walks originate in BFC metadata and kernel work touches
other mappings. Expected gain is small. If revisited, preserve independently
schedulable slices and compare three identical-concurrency arms: separate THP
buffers, a 2 GiB anonymous THP arena registered once, and a 2 GiB 1 GiB-HugeTLB
arena registered once. That separates arena/registration effects from page
size. Measure load and registration time, pinned bytes, and native page-size
events; first prove PJRT/CUDA accepts whole-arena registration.

### CUDA pinned-buffer and BFC tuning

Separating the 256 MiB logical quantum from physical pinned-buffer size was the
largest retained tuning win. A temporary `cuda_async` playground control gave:

```text
pinned block       median elapsed       logical goodput
24 MiB                 0.531 s              28.2 GiB/s
32 MiB                 0.506 s              29.6 GiB/s
40-64 MiB          0.507-0.509 s          29.4-29.5 GiB/s
96 MiB                 0.587 s              25.5 GiB/s
256 MiB                0.625 s              23.9 GiB/s
512 MiB                0.822 s              18.2 GiB/s
```

Nine 32 MiB buffers require only 288 MiB instead of 2.25 GiB for nine 256 MiB
buffers. Read chunks of 32-64 MiB were at the knee. The temporary allocator
switch was removed because production keeps preallocated BFC for later runtime
reuse.

With preallocated BFC and 32 MiB buffers, concurrency medians were:

```text
DMA lanes / buffers     elapsed       logical goodput
4 / 5                    0.831 s          18.0 GiB/s
6 / 7                    0.728 s          20.6 GiB/s
8 / 9                    0.657 s          22.8 GiB/s
10 / 11                  0.657 s          22.8 GiB/s
12 / 13                  0.633 s          23.6 GiB/s
14-16 / 15-17        0.634-0.635 s         23.6 GiB/s
20 / 21                  0.661 s          22.6 GiB/s
```

The historical tuned 12-lane/13-buffer configuration had a ten-run 0.633 s
median at 23.62 GiB/s. Production instead starts at eight with a 32-lane cap
and 33-buffer cap so unknown/long sources can adapt. Direct DMA startup now uses
8 -> 12 -> 16; the 2026-07-19 warm-file sweep confirmed 12 as the local knee.

Historical allocator medians at 12 lanes/13 buffers/32 MiB were 0.633 s for
preallocated BFC, 0.508 s for growing BFC, 0.458 s for CUDA async, and 0.508 s
for platform. Much of preallocated BFC's ~125 ms gap was first non-zero
allocation: PJRT constructs the allocator at client creation but does not call
`BFCAllocator::Extend` until needed. `Platform.warmupDeviceAllocators()` now
moves that arena reservation and metadata initialization ahead of weight load
without changing XLA or exposing allocator selection.

### B70 DmaMap, chunk locality, and current oneAPI default

Current host: AMD Threadripper PRO 9955WX (16 cores/32 threads, two 32 MiB L3
domains), four Intel Arc Pro B70 devices, ext4 warm page cache, one NUMA node.
The workload is the same 291-tensor Llama 3.1 8B model, 14.96 GiB logical.
XLA `9dac91281a` implements PJRT DmaMap through oneAPI
`prepare_for_device_copy`; Level Zero is queried because an imported pointer
remains `usm::alloc::unknown` even after registration. ZML allocates anonymous,
2 MiB-aligned, `MADV_HUGEPAGE` host buffers and registers them with DmaMap.

The old 32 MiB local default was reproducibly CPU/page-cache-copy bound:

```text
profile                                load time median       logical goodput
one B70, explicit 32 MiB (5 balanced)       0.742 s                20.15 GiB/s
one B70, explicit 2 MiB (5 balanced)        0.566 s                26.43 GiB/s
four B70, explicit 32 MiB (1 run)           0.855 s                17.50 GiB/s
four B70, automatic 2 MiB (3 runs)          0.581 s                25.77 GiB/s
```

Four of the five balanced one-device 2 MiB runs were 0.566 s / 26.42-26.44
GiB/s; one outlier was 0.637 s / 23.48 GiB/s. All local runs were fully direct
(`staged=0`). The four-device result is sharded, so physical committed bytes
equal the 14.96 GiB logical total; adding GPUs cannot beat the shared
host/page-cache copy ceiling. Compared with the explicit 32 MiB arm, the
retained setting is about 31% faster on one B70 and 47% faster in the measured
four-B70 sharded load.

The size effect is not H2D bandwidth. Focused SYCL tests measured about
53.4 GiB/s registered anonymous H2D, about 30.0-30.7 GiB/s parallel `pread`
into registered buffers, and 33.5-33.9 GiB/s for the coupled 2 MiB
`pread -> H2D -> reuse` loop. Distinct device destinations did not reduce that
rate. A production size sweep put the knee at 2-3 MiB; throughput declined
through roughly 25.3 GiB/s at 4 MiB, 23.3 at 6 MiB, 22.4 at 8 MiB, and 20.2 at
32 MiB.

A 2026-07-20 fixed 12-lane check compared the shared 13-buffer setting with 24
buffers, which is enough to guarantee two buffers per active one-device writer.
For 32 MiB buffers, five balanced runs had loader medians of 0.798 s with 13
and 0.787 s with 24; outer timer medians were 0.811 and 0.805 s. The roughly
1% difference is inside the 3% resource-selection tolerance, and 24 buffers did
not recover the large gap to 2 MiB. For 2 MiB buffers, three runs per arm had
outer medians of 0.603 s for both pool sizes. Thus 13 buffers do not guarantee
per-lane double buffering, but pool starvation is not the primary explanation
for the chunk-size effect on this host. The earlier broad observation that
increasing the pool beyond 13 hurt was not reproduced by this controlled check.

Do not treat aggregate L3 capacity as the established cause of this size
effect. Normal `pread`/`copy_to_user` stores target write-back, cache-coherent
memory, so dirty cache-line ownership can affect a coherent device read even
though the CPU never consumes the destination again. However, transparent
huge pages primarily reduce CPU/IOMMU translation pressure, not data-cache
footprint, and both the 2 and 32 MiB buffers can be backed by 2 MiB pages. No
cache-flush or hardware-counter experiment has yet attributed the speedup to
L3 residency.

The focused coupled benchmark is not production-faithful enough to explain the
size effect by itself. Each of its workers currently does
`pread -> H2D wait -> reuse` on one buffer. Given the separately measured
roughly 30.5 GiB/s read and 53.4 GiB/s H2D rates, that serialization predicts
about 19.4 GiB/s and can explain why its 32 MiB result is only about
22.5 GiB/s. Production does not serialize the stages this way:
`DirectShardWriter` asynchronously submits one pooled buffer, uses its two
event-context slots as a flip-flop, and lets the next direct `pread` fill a new
buffer while the prior H2D remains in flight. An `AdaptivePipelineLane` is a
logical scheduler admission that drives an active tensor, not a serialized
hardware DMA channel. Therefore the focused benchmark's phase-overlap argument
must not be used as the explanation for ZML's 2 MiB improvement.

The production size effect remains unexplained. Cache coherency, burst shape,
pool contention, driver submission/completion behavior, and CPU-copy behavior
are candidates. The next useful focused test must reproduce
`DirectShardWriter` double buffering (or record a production per-stage
timeline); a cache flush/non-temporal-copy arm can then separate cache ownership
from scheduling and overlap.

`perf` on the old 32 MiB path attributed about 82% of cycles to kernel
`_copy_to_iter`. With 2 MiB buffers it remained the dominant useful work
(roughly 69-70%), but task-clock and system CPU fell with the smaller working
set. Executing direct reads inline on their DMA lane then removed scheduling
overhead without changing the staged path. In a controlled fixed 12-lane,
13-buffer, 2 MiB comparison, task-clock fell from 6.499 to 5.808 s, cycles from
32.70 to 30.01 G, instructions from 5.626 to 5.226 G, system time from 5.884
to 5.174 s, context switches from 86,961 to 38,613, and CPU migrations from
7,350 to 2,878. Wall goodput did not move reliably because `_copy_to_iter`
remains the bottleneck, but the CPU reduction is retained.

Experiments rejected for production:

- DmaMap itself did not slow CPU writes; registered and unregistered `pread`
  were equivalent.
- DMA directly from registered file-backed `mmap` reached only about
  21.4 GiB/s after warmup, consistent with fragmented 4 KiB page-cache/IOMMU
  mappings.
- `mmap` plus userspace `memcpy` into anonymous registered buffers reached only
  about 23.7 GiB/s, slower than the kernel copy.
- One 26 MiB registered host arena split into thirteen 2 MiB slices preserved
  33.9 GiB/s microbenchmark throughput but reduced setup by only about 4.5 ms
  (17.7 to 13.2 ms). It does not yet justify replacing lazy/trimmable pools.

Small pinned buffers regress fast staged sources because they multiply PJRT
submissions. At S3Proxy 10 ms/1000 MiB/s, forced 2 MiB took about 2.945 s /
5.08 GiB/s versus about 2.745 s / 5.45 GiB/s with 32 MiB in the initial A/B.
Three 32 MiB remote runs converged to 32 reads/eight DMA/nine pinned/32 staged
and took 2.747-2.851 s. Despite that evidence, automatic sizing must not branch
on the URI: a path can be network attached and a remote URI can be locally
cached. Revisit the oneAPI pinned-buffer default for remote and slow sources
using a source-independent signal or a policy selected before writers and
DmaMappings become live; until then callers can explicitly request 32 MiB.

The reusable experiment is
`~/github/uxlfoundation/sycl_playground/src/b70_load_bandwidth.cpp`; modes cover
`pread`, H2D, coupled pipeline, file-backed mmap H2D, and mmap-plus-copy.

### Earlier four-GPU oneAPI baselines (superseded)

The following measurements predate the current XLA DmaMap path, THP-backed
anonymous oneAPI buffers, and 2 MiB local tuning. Keep them only as historical
regression context; do not use their 32/64 MiB conclusion as the current local
default.

Hardware: four Intel Battlemage G31 / Arc Pro B70 devices selected with
`ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`. The original coupled 256 MiB
physical/logical setting had 5.154 s adaptive median. Fixed width two with
32 MiB physical blocks had 3.623 s median; fixed width four took 3.649 s once.

Holding the logical quantum at 256 MiB while varying physical buffers gave:

```text
buffer       warm local             S3 10ms/1000       S3 250ms/1000
32 MiB       3.762 s median          6.285 s median       21.187 s
48 MiB       3.820 s                 7.797 s                   -
64 MiB       ~3.708 s combined       6.871 s              32.731 s
128 MiB      3.711 s                 7.171 s                   -
256 MiB      5.154 s                 8.401 s              24.522 s
```

Variance was material. A known-remote initial width of eight with 32 MiB
buffers reached 32 reads/eight DMA lanes in 18.858 s at 250 ms latency, versus
21.187 s from the then-default start, but raised peak RSS from ~6.7 to ~8.7 GiB.
At that time the cross-source default was 32 MiB and 64 MiB was retained only
as a measured oneAPI-local override. The 8/32 concurrency defaults favored
rapid general convergence over the smallest short-local width.

Replicated four-device validation completed in 9.758 s with 14.96 GiB logical
and 59.83 GiB physical committed bytes, fully direct. An intermittent oneAPI
process-exit pause of ~35-40 s occurred after loader and playground completion;
targeted reruns did not reproduce it. Investigate PJRT/platform teardown rather
than loader completion if it returns.

### Pool allocation CPU optimization

A four-GPU oneAPI cycle profile originally showed `_copy_to_iter` 29.20%,
oneAPI host-callback `memmove` 19.55%, driver allocation `clear_page_erms`
11.45%, and `compiler_rt.memset` 8.17%. The memset came from Zig
`Allocator.alignedAlloc` poisoning each newly allocated pool block even though
the next read overwrote it completely.

`DynamicBufferPool` now uses `Allocator.rawAlloc` for deliberately
uninitialized blocks and the matching existing `rawFree`. The residual pool
memset fell to 0.11%. Controlled fixed-width A/B medians were:

```text
block       allocation path       CPU task       cycles       elapsed
256 MiB     poison/aligned          6.893 s       36.20 G       4.134 s
256 MiB     raw/uninitialized       6.572 s       34.62 G       3.847 s
64 MiB      poison/aligned          5.960 s       31.18 G       3.522 s
64 MiB      raw/uninitialized       5.780 s       30.27 G       3.418 s
```

The retained change reduced median CPU time 4.7%/3.0% and elapsed 6.9%/3.0%.
Remaining large CPU costs are useful file copying, plugin-owned oneAPI transfer
copying, and driver allocation clearing; no additional ZML copy hotspot was
measured. At 64 MiB, width four spent ~19% more CPU than width two but finished
~7.8% faster, so lower CPU was not the wall-time optimum.

## Regressions Worth Remembering

- **Read collapse:** treating request latency and admission wait as congestion
  reduced 32 useful reads to one and a load to 37 MiB/s. Only sustained ready
  accumulation now causes read backoff. Tiny DMA samples cannot seed latency.
- **Unexercisable recovery:** expanding reads without enough pageable blocks
  caused endless capacity timeouts after all sources became staged. Candidate
  staging now makes candidate read concurrency physically executable.
- **Bad attribution:** old high-water and old buffered work could validate new
  lower limits. Capacity activation, drained reductions, epoch-tagged bytes,
  64 MiB/200 ms scoring, and 5 s deadlines are all required.
- **Starvation overcount:** multiple waiters counted the same interval and
  produced >100% starvation. Intervals are now unioned.
- **Ready-queue convoy:** yielding before scheduling the next direct read cycled
  tensors without progress. A tensor now owns its full logical quantum.
- **Pinned deadlocks:** replenishing merely to park, detaching a direct owner,
  and replacing lanes after the scheduling fence closed caused three separate
  hangs. Park without replenish, retain direct lanes, and stop `ensureLanes`
  after closure. The last fix passed 20/20 forced-width-eight repetitions.
- **Remote timing:** 500 ms probe deadlines cannot work for 1 s requests. Both
  capacity and live-progress deadlines are now 5 s.
- **Bursty-source oscillation:** S3 completions arrive in waves. Single-window
  baseline/candidate comparisons, ready growth during DMA starvation, and
  drain-only `dma_saturated` samples caused repeated 16 <-> 32 read and 8 <->
  10 DMA probes. Cumulative post-activation goodput, a stable baseline, sticky
  slow-source classification, and two seconds of continuously fed DMA now
  distinguish a stable bottleneck from a burst.
- **Short local tail:** suppressing probes below 500 ms left a repeatable local
  12-lane knee unused. A 250 ms cutoff plus the gradual 8 -> 12 step reached the
  knee without increasing the initial allocation.

## Open Risks and Useful Next Work

1. The 2026-07-19 convergence sweep covered one CUDA host, warm local files,
   and S3Proxy profiles. Re-run on real S3/HF, cold storage, oneAPI, and hosts
   with materially different CPU/storage/device balance before tightening the
   universal caps.
2. The 250 ms finite-tail cutoff was validated on a 14.96 GiB model. Smaller
   eligible loads may spend a larger fraction of runtime on a late probe; add a
   small-model matrix if that workload matters.
3. Broad positional prefetch may hurt file locality/readahead on some storage.
   Benchmark before adding source-fanout heuristics.
4. `slow_reads < 1.5 GiB/s` is a fixed bootstrap hint, not a learned source
   model, and slow-source classification is sticky for a load. There is no
   transport throttle/retry feedback.
5. Slot metadata is `O(open_sources * max_read_parallelism)` even though buffer
   bytes are bounded.
6. `pinned_buffer_size` is static after load startup. The automatic policy
   chooses 2 MiB for oneAPI and 32 MiB otherwise, without inspecting the URI.
   Revisit the oneAPI size for remote and slow sources: local paths may be
   network attached, so any better policy needs a source-independent signal or
   an explicit caller choice before writers and DmaMappings become live.
7. Success teardown is stress-tested, but read/PJRT failure injection and true
   transport interruption need more coverage.
8. Recheck oneAPI DmaMap registration, the `usm::alloc::unknown` Level Zero
   verification path, and the post-completion exit pause when its PJRT/runtime
   package changes. The old plugin-owned pageable `memmove` is absent from the
   current DmaMapped direct profile.
9. A substantial warm-local CUDA gain now requires avoiding the page-cache
   copy, likely via a carefully validated mmap/register prototype. Huge pages,
   extra H2D streams, and more dTLB tuning are not current high-value work.
10. B70 steady windows reach about 29-30 GiB/s while end-to-end medians are
    about 26 GiB/s. Remaining opportunity is startup/tail and tensor/PJRT
    object overhead; direct file-backed DMA and mmap-plus-copy were both slower.

## Validation and Workflow

Focused validation:

```sh
./bazel.sh test //zml:test --test_output=errors
./bazel.sh build //zml
./bazel.sh build --config=release --@zml//platforms:cuda=true \
  //examples/io:playground //examples/llm
./bazel.sh build --config=release --@zml//platforms:oneapi=true \
  //examples/io:playground
git diff --check
```

At the current snapshot, `//zml:test` passes and the release oneAPI IO
playground builds. XLA was not rebuilt, per user instruction; the benchmark
uses the user's existing local plugin artifact. Earlier in the branch, all 216
then-existing core tests and release CUDA IO/LLM builds passed. The earlier
2026-07-19 sweep ran `./bench_file.sh` and
`./bench_s3.sh` at its default 1000 ms/100 MiB/s profile, plus 10 ms/1000 MiB/s
and 250 ms/1000 MiB/s overrides. That older snapshot did not re-benchmark
oneAPI; the current B70 evidence above supersedes it. Tests cover positional
reads,
sharding/mirrors, smaller physical buffers than logical quantum, pool/group
limits, probe decisions and attribution, slow/bursty-source convergence,
pressure/starvation, cooldowns, hard caps, scheduling fences, and stable queue
ownership.

For performance changes, compare identical model/source/cache state:

1. fixed escape-hatch baseline;
2. adaptive default;
3. a static sweep of the dimension being changed;
4. warm and cold local storage plus relevant HF/S3/network sources;
5. total logical goodput, committed H2D goodput, direct/staged bytes, peak/final
   knobs, allocated pools, starvation, CPU time, and RSS.

Acceptance remains completion within 3% of the best static setting, then the
fewest resources inside that band. Fast local loads should remain direct with
zero pageable allocation when possible; high-latency sources should ramp reads
within budget without repeated unexercisable probes, leaks, or hangs.

When diagnosing a new run:

1. Confirm revision, dirty files, platform, cache state, and whether the log
   matches the binary.
2. Read `configured`, final `completed`, and all `limits updated` lines.
3. Build the knob timeline and find the first lasting throughput change.
4. Correlate it with the preceding window records.
5. Classify the failure: bad controller action, unexercised capacity, missing
   epoch bytes, queue with no ordered progress, or H2D without completion.
6. Verify metric semantics before changing thresholds.
7. Change one dimension, add a deterministic regression test, run focused
   validation, then compare repeated same-workload wall time and knob timeline.
8. Update this file only with durable architecture, invariants, conclusions,
   and reproducible evidence—not every intermediate experiment.
