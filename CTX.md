# Adaptive Loader Agent Context

Snapshot date: 2026-07-18

Snapshot base commit: detached `HEAD` at `9b71c5a8` (`oneapi`). That commit
contains the four-GPU scripts, pinned-buffer/quantum separation, and initial
oneAPI sweep. The working tree adds the uninitialized pool-allocation
optimization described below; `profile_file.sh` and its `perf.data` output are
currently untracked. Inspect `git diff` before assuming later work is committed.

This file is an implementation handoff for agents. It is deliberately dense.
Treat current source as authoritative when it conflicts with this snapshot.
Read `RESEARCH.md` for the theory and citations; read this file for the actual
design, invariants, observed failures, and debugging procedure.

## Objective

Minimize finite-batch model load wall time from an unknown source to accelerator
memory while using the fewest scarce pinned DMA chunks and pageable staging
blocks within 3% of the best observed committed goodput.

The source is not known in advance. It may be:

- a fast local SSD capable of multi-GiB/s reads;
- a network-attached filesystem with local file semantics but remote latency;
- S3 or Hugging Face with request latency and useful request parallelism;
- a source whose behavior changes over the load.

Expected complete loads are short: seconds to roughly one minute. Startup must
react in tens of milliseconds. Slow additive convergence is unacceptable.

The implemented controller objective is physical PJRT bytes committed to device
memory per second. Raw read goodput, operation counts, tensor completion counts,
and host-side ordered bytes are stage diagnostics, not the objective. The final
completion record separately reports logical model bytes per second. With fixed
sharding these objectives are proportional; replicated multi-device analysis
must not compare their absolute byte rates directly.

Current validation state at this snapshot:

- deterministic `//zml:test` passes with 209 tests (one skipped on this host);
- release `//examples/io:playground` builds for oneAPI and both benchmark
  scripts run on four Intel Battlemage G31 / Arc Pro B70 devices selected by
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`;
- the 14.96 GiB Llama 3.1 8B workload has been swept across fixed/adaptive
  widths, 32/64/128/256 MiB DMA quanta, and independently sized pinned blocks
  from warm local files and S3Proxy;
- a 256 MiB logical quantum with a 64 MiB pinned block is the best balanced
  local four-GPU setting observed; after the allocation fix its adaptive median
  was 3.561 s versus a 3.418 s fixed-width-4 median, 4.2% behind and narrowly
  outside the target. A 32 MiB block performs better on the tested S3 profiles;
  no single fixed block size is within 3% on both sources;
- a four-GPU replicated load completed in 9.758 s with 14.96 GiB logical bytes
  and 59.83 GiB physical PJRT bytes committed, validating the physical/logical
  metric distinction end to end;
- a four-GPU sharded `perf` investigation removed the full-buffer undefined
  poison write from dynamic pool allocation. In controlled fixed-width runs it
  reduced median CPU task time by 4.7% with 256 MiB pinned blocks and 3.0% with
  64 MiB blocks;
- the earlier single-CUDA validation and its 20/20 teardown stress result remain
  recorded below, but must not be treated as the current hardware result.

## Design Summary

The adaptive CUDA/oneAPI loader is a bounded two-stage chunk pipeline with a
direct bypass:

```text
TensorStore / safetensors
        |
        | positional reads, read_chunk_size quanta
        |
        +----------------------+-----------------------+
        |                                              |
        v                                              v
pinned writer window                           pageable block pool
direct read                                    concurrent read-ahead
        |                                              |
        |                                              v
        |                                     in-order per-tensor head
        |                                              |
        +----------------------+-----------------------+
                               |
                               v
                      MemoryWriter / shard dispatch
                               |
                               v
                   PJRT async H2D submissions
                               |
                               v
                       committed GPU bytes
```

There is one controller for four related knobs:

- `read_workers`: maximum outstanding positional reads;
- `dma_workers`: maximum admitted tensor DMA lanes;
- `dma_chunks`: per-device pinned block limit;
- `staging_chunks`: active pageable block limit.

`dma_buffer_size` is a static per-device pinned block size, not a fifth
controller knob. `dma_chunk_size` remains the logical lane scheduling quantum.
They default to the same value for compatibility but can now be tuned
independently. This matters on multi-device sharding: multiplying a 256 MiB
logical quantum by every device made lazy pinned allocation a material fraction
of a short local load.

One controller is intentional. Independent read and DMA controllers would fight
through the intermediate queue. The single controller changes one dimension at
a time and uses stage metrics to choose the next dimension.

The implementation is currently a byte-attributed probe/rollback controller.
It is inspired by BBR startup and Gradient/Vegas queue awareness, but it is not
a literal Gradient2 implementation. Do not describe it as Gradient2.

## Source Map

- `RESEARCH.md`: algorithm research and original recommendation.
- `zml/io.zig`:
  - `LoadMetrics`: atomic counters and epoch attribution;
  - `MemoryWriter`, `DirectMemoryWriter`, `DirectShardWriter`: pinned/H2D path;
  - `AdaptiveLoadController`: pure control decisions;
  - `AdaptiveLoadRuntime`: sampling windows, probe activation, applying limits;
  - `AdaptivePipelineContext`: queues, groups, pools, cancellation, ownership;
  - `AdaptiveTensorLoad`: per-tensor read ordering and DMA quantum state;
  - `AdaptivePipelineLane`: admitted DMA lane state machine;
  - `LoadOpts` and `load`: public configuration and adaptive/fixed dispatch;
  - inline tests after `load`.
- `zml/safetensors.zig`:
  - `TensorReader.readPositionalAll` implements exact tensor-relative reads;
  - positional/concurrent read tests.
- `zml/mem.zig`:
  - `DynamicBufferPool` lazily allocates, limits, returns, and trims fixed blocks;
  - `getWithWait` separates admission wait from allocation/mutex time.
- `stdx/Io.zig`:
  - `LimitedGroup` has a runtime-adjustable limit;
  - `concurrentUncancelableAdmission` preserves queued cleanup tasks after group
    cancellation.
- `bench_file.sh` and `bench_s3.sh`: release-mode source benchmark entry points.
  Both use the repository `bazel.sh` wrapper. S3 accepts `LATENCY_MS` and
  `SPEED_MIB`; loader settings are inherited through the playground variables
  documented in the benchmark section below.
- `profile_file.sh`: the user's release oneAPI sharded-load entry point under
  `perf record`; loader environment overrides are inherited just like the
  benchmark scripts.
- `examples/io/main.zig`: enables `zml/io/load` debug logging and maps benchmark
  environment variables to `LoadOpts` without changing the default invocation.

Relevant history:

- `a8ffb40c`: first adaptive concurrency implementation and `RESEARCH.md`;
- `c1ac887e`: controller logging;
- `4f1b655f`: positional two-stage read/DMA adapter plus subsequent fixes.

## Public Configuration

`LoadOpts` semantics:

- `parallelism`: hard maximum active tensor DMA streams.
- `initial_parallelism = 2`: starting read and DMA limit, clamped to caps.
- `adaptive_parallelism = true`: enable adaptive path when eligible. Set false
  to use the fixed legacy loader.
- `max_read_parallelism = null`: hard outstanding-read cap. Default is
  `max(32, parallelism)`. This is only a ceiling; active reads still start at
  `initial_parallelism` and pageable blocks remain lazy.
- `read_chunk_size = 32 MiB`: positional read and pageable block size.
- `max_staging_bytes = 1 GiB`: hard pageable budget. Blocks are lazy.
- `max_staging_bytes = 0`: no pageable staging; direct/DMA adaptation remains.
- `dma_chunks`: hard pinned blocks per device.
- `dma_buffer_size = null`: per-device pinned block size; null preserves the
  old behavior and uses `dma_chunk_size`.
- `dma_chunk_size`: logical bytes a DMA lane processes before yielding,
  commonly 256 MiB. A shard writer may submit smaller buffers within this
  quantum when `dma_buffer_size < dma_chunk_size`.

Adaptive eligibility:

- target must be CUDA or oneAPI;
- adaptive option must be enabled;
- total logical bytes must exceed one read chunk;
- either DMA or read maximum must exceed one.

Other targets and ineligible loads retain the fixed buffered loader.

DMA worker cap is:

```text
min(tensor_count, parallelism, max(1, dma_chunks - 1))
```

The `dma_chunks - 1` constraint preserves one extra pinned chunk for flip-flop
overlap. Initial pinned limit is normally `initial_dma_workers + 1`.

Pageable block cap is floor division:

```text
max_staging_chunks = max_staging_bytes / read_chunk_size
```

Each tail read still leases a full block. Allocated pageable bytes never exceed
`max_staging_chunks * read_chunk_size`.

## Positional Read Source

`TensorReader.readPositionalAll(destination, tensor_offset)`:

- validates the range against tensor byte size;
- adds the tensor's absolute file offset with overflow checking;
- calls `File.readPositionalAll` without changing sequential reader position;
- requires an exact byte count or returns `UnexpectedEndOfFile`;
- permits concurrent calls against the same reader/file handle.

This is required for concurrent chunks inside a large tensor. The ordinary
sequential `TensorReader.interface` remains available to the fixed loader.

## Data Plane Ownership

### Tensor admission

`next_tensor` atomically assigns each model tensor exactly once. Tensor readers,
`MemoryWriter` instances, pageable slot arrays, and file handles are created
lazily. The pipeline does not create a writer or pinned buffer per model tensor
up front.

Pageable prefetch may open up to `min(staging_limit, max_read_parallelism)`
tensor sources. Each open `AdaptiveTensorLoad` allocates slot metadata sized to
`max_read_parallelism`, but slot buffers come from the bounded global pool.

### Direct path

A direct read targets `MemoryWriter.directWritable()`, which aliases a real
pinned shard buffer. It avoids pageable allocation and pageable-to-pinned copy.

Direct read reservations are globally capped by the current DMA lane limit, not
the read worker limit. This is deliberate: direct reads occupy pinned windows;
additional source parallelism must use pageable staging.

A direct read is permitted only for the next tensor offset. If no direct
reservation is available, the tensor tries to create staged read-ahead and then
parks. Direct capacity waiters receive priority when reservations are released.

While an asynchronous direct read is pending, its DMA lane remains admitted and
owns the tensor/writer. A resume task waits for the read event and reschedules
that same lane. Do not detach a pinned-buffer-owning direct tensor into a shared
ready queue: other lanes can block creating writers until every pool block is
held by queued states, producing a circular wait.

### Pageable path

`DynamicBufferPool` owns fixed `read_chunk_size` blocks. It allocates blocks only
on demand, applies a dynamic active limit, and can trim only currently free
blocks. A limit reduction does not cancel or reclaim in-flight blocks.

Per-tensor staging acquisition uses monotonically increasing tickets so block
admission follows planned tensor order. After a block is acquired, the actual
positional read runs through the global `read_group`, so reads remain concurrent
and may complete out of order.

`PageableReadSlot` carries offset, length, consumed bytes, epoch, completion
event, error, buffer, scheduling time, and ready time. Completed blocks increase
global `ready_bytes`; consumption or cleanup decrements it exactly once.

### Reordering and commit order

For each tensor:

```text
0 <= next_commit <= next_read <= total
```

Slots form a ring in planned offset order. Reads may finish in any order, but
`processQuantum` consumes only the head slot and verifies
`slot.offset + slot.consumed == next_commit`. A direct read similarly verifies
its offset. Any violation is `InvalidReadOrder`.

`MemoryWriter` always receives logical tensor bytes in file order. Its shard
dispatch logic sorts global placement ranges, sends the primary range through
the zero-copy pinned alias where possible, and copies mirror/replicated ranges
to their shard writers.

### DMA quantum and fairness

A DMA lane drains at most one `dma_chunk_size` quantum from one tensor before it
may requeue that tensor. If no other tensor is ready and the lane is not being
retired, it can immediately continue the same tensor without park/unpark churn.

Critical invariant: once a tensor owns a lane for a quantum, do not requeue it
at the direct boundary merely because another tensor is queued. It must schedule
its next direct read first. Violating this caused a ready-queue convoy with no
useful progress and hundreds of megabytes of logs.

Before detaching a staged tensor, the lane parks its writer and waits as needed
so no detached state exposes or mutates an unsafe pinned window. Direct-read
waiters retain their lane as described above. Heap-allocated tensor states
remain stable while linked into ready/wait queues.

### Pinned flip-flop

Each shard writer submits PJRT H2D asynchronously and alternates completion
contexts/buffers. It waits for the previous flip-flop event before reusing the
corresponding resource. This is why the useful minimum is normally one more
pinned chunk than admitted DMA lanes.

`parkAndWait` commits the public writer window without replenishing it, parks
shard writers, records logical submission, and waits for pending transfers.
`unpark` obtains/publishes a valid next pinned window. Acquiring a replacement
buffer merely to park can deadlock when every lane is retiring or queued.

## Concurrency Groups and Queues

- `read_group`: actual positional read concurrency; dynamic limit.
- `dma_group`: scheduled lane executions; dynamic limit.
- `staging_group`: pageable acquisition and prefetch setup tasks.
- `resume_group`: waits for staged read/direct-capacity events without DMA lane
  admission; direct-read event tasks deliberately retain their admitted lane.
- ready queue: FIFO of stable `AdaptiveTensorLoad` pointers ready for DMA work.
- direct-wait list: direct states with an owning lane waiting for a read event.
- direct-capacity condition: states waiting to reserve a direct pinned read.

Pageable-ready tensor states are popped before lanes claim new direct tensors.
Within a tensor, offset order is stronger than path priority.

## Controller Metrics

Metric byte positions are distinct:

- `storage_bytes`: positional read completed into host memory.
- `direct_read_bytes`: subset read into pinned memory.
- `staged_read_bytes`: subset read into pageable memory.
- `ready_bytes`: completed pageable bytes not yet consumed.
- `ordered_bytes`: bytes committed in tensor order to `MemoryWriter`.
- `logical_submitted_bytes`: logical bytes crossing writer submission fences.
- `submitted_bytes`: bytes submitted to PJRT H2D.
- `committed_bytes`: bytes whose PJRT completion callback succeeded.

Only `committed_bytes / elapsed` is controller objective goodput. These are
physical successful PJRT transfer bytes, not final logical model bytes.

Other metrics:

- read operation count and byte-weighted read latency;
- DMA submission count and byte-weighted completion latency;
- pageable copy bytes/time and byte-weighted ready age;
- read-admission, staging-pool, pinned-pool, and DMA-completion waits;
- active read high-water, active writers, admitted DMA lanes;
- H2D queued bytes = submitted minus committed;
- global DMA starvation intervals.

Read latency and read admission wait are diagnostics/demand signals. They are
not read congestion signals. Useful source concurrency naturally raises
per-request latency, and admission wait proves demand for the configured limit.
Treating either as pressure previously collapsed 32 useful reads to one.

Read-side pressure is only evidence that completed pageable data is accumulating
faster than DMA can drain it: ready occupancy above 75%, ready age above 250 ms,
or growth above 20% for two consecutive committed windows. A single 32 MiB
block arrival is not pressure: at a small staging limit it has a large growth
ratio even when DMA consumes it in the next window.

DMA completion latency is used only when a window committed at least 32 MiB.
An undersized sample must neither cause pressure nor establish the baseline.
The baseline is updated from reliable, lightly loaded windows.

Starvation intervals from multiple tensors overlap. `LoadMetrics` unions them
using `dma_starvation_covered_until_ns`; otherwise N waiters count the same GPU
idle interval N times. The reported global starvation ratio is capped at 100%
and divided by elapsed wall time, not by lane count.

Wait ratios for read/staging/pinned activity may exceed 100% because they sum
concurrent task wait time. They are aggregate diagnostic ratios, not occupancy.

## Sampling Windows

The runtime wakes every 25 ms.

Startup windows:

- minimum 50 ms;
- maximum 100 ms;
- 32 MiB progress floor;
- progress may be read, ordered, submitted, or committed bytes.

Steady windows:

- minimum 100 ms;
- maximum 250 ms;
- 64 MiB committed floor when not probing;
- probes use pipeline progress so dead-time can be observed.

If a startup window has no progress for 100 ms and source demand exists, the
controller can bootstrap read-ahead without waiting for committed bytes. This is
how a high-latency remote source reaches useful request concurrency quickly.

A live probe with zero pipeline progress for 5 s rolls back. A candidate whose
physical capacity is never exercised also rolls back after 5 s. Both deadlines
include source service time; a 500 ms deadline made a 1 s-latency S3 source
mathematically unable to activate or produce its first attributed byte.

Probes are skipped when estimated remaining work is under 500 ms. Remaining
time estimates include unread/ordered work, host bytes buffered before logical
submission, and queued H2D bytes.

`slow_reads` currently means estimated single-read service bandwidth below
1.5 GiB/s. This is a bootstrap/demand hint, not a backoff signal.

## Probe Attribution Protocol

Control changes have pipeline dead time. Never evaluate a candidate immediately
after changing a limit.

Each probe has:

- dimension: read, DMA, pinned, or staging;
- kind: increase or reduce-resource;
- baseline knobs and candidate knobs;
- epoch;
- baseline committed goodput.

Applying knobs first creates `pending_probe_activation`. The probe epoch is not
published until the requested physical capacity is demonstrably active:

- read increase: active-read high-water reached candidate read workers;
- read reduction: outstanding reads and staging blocks drained to candidate;
- DMA: candidate number of distinct lanes each submitted DMA in the probe
  capacity epoch;
- pinned increase: each device pool allocated beyond baseline pinned chunks;
- pinned reduction: pool limit and in-flight blocks reached candidate;
- staging increase: candidate blocks allocated and increased read capacity was
  exercised when applicable;
- staging reduction: staging in-flight reached candidate.

If activation does not happen within 5 s, rollback reason is
`capacity_not_exercised`.

Once active, reads/slots/writers carry the epoch. PJRT completion callbacks add
bytes to `probe_committed_bytes` only when their epoch matches the published
probe epoch. Evaluation waits for at least 64 MiB committed from that epoch and
at least 200 ms of active probe time. Increase probes compare the better of the
epoch-attributed average and the current committed window after that sustained
interval; staged candidate blocks can sit behind old-epoch blocks even after
aggregate throughput has improved. Resource-reduction probes retain the stricter
epoch-only comparison.

This prevents old buffered reads or old H2D submissions from being credited to
a new limit. In particular, a read-reduction probe must not use the previous
high-water mark; old 32-worker output cannot validate a 16-worker candidate.

Increase probe acceptance:

```text
candidate_goodput >= baseline_goodput * 1.03
and no relevant queue/H2D pressure
```

Resource reduction acceptance:

```text
candidate_goodput >= max(baseline_goodput, global_peak_goodput) * 0.97
and no relevant pressure
and DMA starvation <= 10%
```

The second rule implements "fewest resources within 3% of best".

Rollback restores the entire baseline knob tuple, resets growth state, trims
newly added resources where possible, and blocks immediate pressure backoff for
250 ms.

## Exact Decision Order

`AdaptiveLoadController.observe` is intentionally ordered. Earlier actions win.

1. If probe capacity is pending, hold all knobs.
2. Update reliable DMA latency baseline and derive H2D/ready pressure.
3. H2D hard pressure above 20%:
   - rollback an active probe; otherwise
   - reduce DMA workers by about 30%, bounded at one.
4. Ready-queue pressure above 10%:
   - rollback a relevant probe; otherwise
   - reduce read workers by 15% or 30%, but not below direct DMA width; and
   - trim staging toward one or two ready blocks per DMA worker.
5. If a probe has 64 MiB attributed bytes and has been active for 200 ms, keep
   or rollback using the rules above.
6. If probing is disabled near the tail, hold.
7. With no committed goodput and a slow/stalled saturated source, bootstrap:
   double reads and add staging, bounded by hard caps.
8. When DMA starvation exceeds 10%, read demand exists, source capacity is
   exercised, and the 500 ms cadence allows it: double read concurrency and
   add staging. Once read fanout is already at least 4x DMA width, balance the
   DMA stage before doubling staging again.
9. In steady state with starvation at or below 10%, periodically probe halving
   excess read workers.
10. Every two seconds in steady state, probe reducing pinned chunks, then
    staging blocks, then DMA workers, one dimension at a time.
11. When DMA is saturated, or direct reads show demand without H2D pressure,
    probe more DMA workers. Width 1/2 and startup steps double; later steady
    steps are roughly sqrt(C).
12. If DMA is at cap and pinned wait indicates benefit, probe one more pinned
    chunk.
13. Otherwise enter/hold steady state.

Read-ahead candidates double reads with at least +1. On first transition from
direct-only, required staging is `read_workers - dma_workers`. Once staging is
active, candidate staging grows to the candidate read count (bounded by the
staging cap), because all surviving sources may already be on the staged path.
Assuming DMA workers always supply direct reads caused unexecutable recovery
probes.

## Resource Reduction Details

Read reduction is attempted every 500 ms when there is no starvation and reads
exceed DMA workers. It halves reads but not below DMA workers. When staged mode
is active, it retains enough candidate staging blocks to actually exercise the
candidate reads. Measurement begins only after old reads/blocks drain.

Every two seconds, resource reduction prefers:

1. one fewer pinned chunk above `dma_workers + 1`;
2. one fewer staging block above the direct-capacity minimum;
3. fewer DMA workers by approximately sqrt(C).

Pool limit changes affect future acquisition. Existing operations drain. Free
blocks are trimmed lazily and periodically. Pinned effective limit is never set
below current in-flight blocks or admitted DMA lanes.

## Cancellation and Error Propagation

The first error wins and is stored as an error integer. `fail`:

- atomically records the original error;
- closes the scheduling fence so no new jobs are admitted;
- wakes direct-capacity waiters;
- signals the top-level done event.

Every asynchronous scheduling path calls `beginScheduling`/`endScheduling`.
The high bit closes scheduling; the remaining bits count admitted schedulers.
Top-level teardown waits for `scheduling_idle` before destroying shared state.

On failure, the load path cancels read/staging groups, drains resume/DMA groups,
waits or cleans all direct and pageable events, returns every pageable/pinned
buffer, destroys queued/active tensor states, and returns the first error.

Read jobs use uncancelable admission so a queued job can still run its cleanup
after group cancellation. Underlying blocking file/network reads may still need
to return before complete teardown; cancellation does not promise transport-
level interruption.

Do not simplify teardown ordering without failure-injection tests. Most objects
contain events and pointers into pools owned by the top-level `load` frame.

## Important Invariants

- Each tensor index is claimed once.
- Each tensor's bytes enter `MemoryWriter` in strict tensor byte order.
- A pageable block is returned exactly once after full consumption or error.
- `ready_bytes` is incremented only after a successful read and decremented for
  every consumed/released ready byte.
- Direct reservation count is released exactly once, including scheduling
  failure and tensor teardown.
- Queue nodes are heap-stable until removed.
- A detached writer is parked; no lane destroys a state with pending events.
- Knob reductions affect new admissions; they do not cancel useful in-flight
  operations.
- Probe bytes are evaluated only after capacity activation and epoch match.
- `dma_chunks` and staging bytes never exceed public hard caps.
- Completion is signaled only after all tensors complete or first failure.
- Fixed loader behavior remains available through `adaptive_parallelism=false`.

## Debug Logging

All adaptive logs use scope `zml/io/load`:

```zig
const load_log = std.log.scoped(.@"zml/io/load");
```

Zig formatting supports at most 32 format arguments. Keep the window report
split across `window control`, `window throughput`, `window pressure`, and
`pipeline concurrency`. Do not recombine them into one format call.

High-value records:

- `configured`: public caps, derived caps, initial values, chunk sizes.
- `controller started/stopped`: lifecycle and final counters.
- `source bootstrap`: zero-progress multiplicative read startup.
- `window control`: knob decision, saturation, lane utilization, pool state.
- `window throughput`: stage rates, copy rate, read/DMA latency.
- `window pressure`: waits, starvation, queues, probe attribution, tail estimate.
- `pipeline concurrency`: actual reads, direct reservations/waiters, detached and
  prefetched sources, ready states, actual/probe lanes, claimed tensors.
- `limits updated`: every applied controller action and rollback reason.
- `probe capacity active/timeout`: attribution boundary diagnostics.
- `waiting for progress`: 500 ms idle snapshot.
- pool limit/trim records: requested versus effective resource state.
- source/tensor open/ready/start/complete: ownership and progress.
- `pipeline cancellation requested`: first error and concurrency snapshot.
- `completed`: authoritative end-to-end elapsed time and logical goodput.

Interpretation traps:

- `tensor started ... active_dma_streams` currently prints active writer/tensor
  count, including parked writers. Use `pipeline concurrency dma_lanes` for
  admitted lanes.
- Lane snapshots can be zero between very short executions. Cross-check DMA
  submissions, committed goodput, and utilization.
- Read/staging wait above 100% is possible due summed concurrent waits.
- `ready=0` plus high starvation means source data is consumed immediately but
  arrives too slowly; it does not mean the source is idle.
- High read goodput with low committed goodput means downstream buffering or H2D
  pressure; optimize committed goodput.
- Repeated probe starts with capacity timeouts mean the candidate cannot be
  physically exercised. Inspect staging/pinned counts before tuning timeouts.

Avoid per-lane/per-requeue hot-loop debug logs. A previous convoy produced a
767 MB, 7.6 million-line log. State-transition/window logs are sufficient.

Useful log extraction:

```sh
rg -n "configured:|completed: adaptive|controller stopped:" log.txt
rg -n "limits updated:|probe capacity|source bootstrap" log.txt
rg -n "window control:|window throughput:|window pressure:" log.txt
rg -n "pipeline concurrency:|waiting for progress:" log.txt
rg -n "pipeline cancellation requested|error" log.txt
```

## Observed Regressions and Lessons

### Catastrophic read collapse

Representative Llama 3.1 8B HF load:

- 14.96 GiB logical bytes;
- startup reached 32 active reads;
- an early window reached about 2395 MiB/s committed goodput;
- controller reduced reads `32 -> 22 -> 15 -> 10 -> 7 -> 4 -> 2 -> 1`;
- controller later reduced staging `30 -> 1`;
- final load took 410.934 s at 37.27 MiB/s;
- 4.46 GiB used direct reads and 10.50 GiB used staging.

Root causes:

1. Read latency inflation was treated as congestion. The baseline came from an
   undersized early transfer; useful request parallelism naturally increased
   individual latency while aggregate goodput improved.
2. Read admission wait was treated as pressure even though it indicates demand.
3. DMA latency was baselined from an 8 KiB startup transfer (about 177 us), so
   normal larger transfers (about 680 us) falsely triggered `dma 2 -> 1`.
4. Recovery `reads 1 -> 2` retained one staging block by assuming one direct DMA
   read. At that point all relevant sources were staged, so two reads could not
   become active and every probe timed out.
5. Multiple waiting tensors counted the same DMA-starved interval, producing
   starvation ratios over 100% and corrupting control decisions.
6. Read-reduction probes could be marked active by the old high-water mark,
   crediting bytes produced by old concurrency to the lower candidate.

Fixes now present:

- only ready-queue formation causes read backoff;
- small DMA windows cannot seed or use latency baseline;
- staged expansion reserves executable block capacity;
- read reductions wait for old work to drain before epoch publication;
- starvation intervals are unioned and capped.

### Ready-queue convoy and huge logs

A tensor previously yielded at the direct boundary when another DMA-ready state
existed, before scheduling its own next direct read. Many states then cycled
through the ready queue without progressing. Hot lane logs amplified this into
hundreds of megabytes.

Fix: a tensor owns one DMA quantum and schedules its direct read before yielding.
Hot lane resume/yield/activate logs were removed.

### Probe pipeline dead time

Evaluating a probe immediately after changing a limit attributes old pipeline
work to the new configuration. The capacity-pending state, delayed epoch
publication, epoch-tagged reads/writers, 64 MiB floor, and activation timeout are
all required. Removing any one reintroduces false keeps/rollbacks.

### Tiny tensors distort latency

Byte-weighted latency helps but does not make an 8 KiB startup sample comparable
to a normal 32 MiB/256 MiB transfer. Reliability floors are still required.

### Pinned ownership and teardown deadlocks

Three independent lifecycle failures were reproduced on the 14.96 GiB CUDA
load:

1. Parking a writer replenished its just-flushed pinned buffer before giving up
   the lane. Several retiring lanes could consume all pool blocks merely to
   park, preventing the useful writer from acquiring one. Park now commits with
   `replenish=false`; normal writes/unpark still replenish.
2. A direct-read tensor was detached while retaining a pinned writer, and a
   generic lane claimed another tensor and blocked creating its writer. Enough
   queued owners produced a circular wait. Direct reads now retain and resume
   their admitted lane until the read event fires.
3. A lane decided to replenish immediately before the final tensor closed the
   scheduling fence. `ensureLanes` repeatedly activated a replacement that saw
   the closed fence and retired, so admission never reached the requested target
   and teardown spun forever. `ensureLanes` now exits when scheduling is closed.

The third bug explains runs that logged all 291 tensors, all 14.96 GiB committed,
and `controller stopped`, but never logged final `completed`. A live GDB trace
showed `releaseAdmission -> ensureLanes -> schedule` looping. The fix passed
20/20 forced-width-8 repetitions.

### False local probe rollback

The first 2-to-4 local DMA probe was sometimes evaluated on its transition
window. A burst drained by width 2 became the baseline, while lane turnover made
the candidate's first window look slower. Three of twenty runs rolled back to 2
and took about 1.5 s instead of 1.15 s. Probe evaluation now requires 200 ms as
well as 64 MiB; 20/20 subsequent runs ended at width 4.

### Remote ready growth and probe deadlines

One completed 32 MiB staging block was formerly treated as immediate pressure.
At a two-block limit that is 50% growth, so the controller backed reads down to
one and repeatedly reprobed. Growth must now persist for two committed windows,
and direct reads cannot be reduced below DMA width.

Capacity activation and active-probe progress also used 500 ms deadlines. A
source with 1000 ms request latency could never satisfy either deadline, causing
an endless `dma 2 -> 4 -> 2` cycle. Both are now 5 s. Keep the two deadlines
consistent when making future source-aware timeout changes.

### MacBook source-concurrency sweep, 2026-07-18

Harness and interpretation:

- model is 4.36 GiB across 148 tensors under `~/s3proxy/data/lfm`;
- platform auto-selection loads `libpjrt_cpu.dylib` and logs `target=cpu`;
- adaptive eligibility is deliberately CUDA/oneAPI only, so every run logs
  `adaptive=false` and uses the fixed loader;
- therefore these measurements locate source/host concurrency knees only. They
  do not validate positional read chunks, direct versus staged routing, pinned
  chunks, DMA streams, H2D goodput, or adaptive convergence;
- on the fixed CPU path `parallelism` is the actual concurrent tensor/read
  count. `read_chunk_size`, `dma_chunk_size`, staging, and the adaptive read cap
  are not meaningfully exercised.

Warm-file logical goodput in MiB/s:

```text
workers       1       2       4       8      16      32
goodput    1730    3125    5507    9306    9991    8983
```

The 8/16/32 values are medians of three warm repetitions. Sixteen workers are
about 7.4% faster than eight, while 32 regress about 10.1% from the 16-worker
knee. Local reads benefit from concurrency beyond the desired small DMA count,
but maximizing reads is not safe.

Near-ideal proxy, `LATENCY_MS=1 SPEED_MIB=10000`, one run per setting:

```text
workers       1       2       4       8      16      32
goodput     990    1697    2445    2914    3257    3211
```

Sixteen reads are best; 32 are within 1.4% and therefore inside the controller's
3% low-resource equivalence band.

Latency-dominated proxy, `LATENCY_MS=250 SPEED_MIB=1000`, one run per setting:

```text
workers       2       4       8      16      32      40      64
goodput     105     213     419     782    1170    1154    1130
```

The knee is 32 reads. The former null default produced a 16-read cap when DMA
`parallelism=8`, leaving about 33% of the best goodput unavailable even if the
controller correctly detected starvation. The default ceiling is now
`max(32, parallelism)`. This does not allocate 32 blocks or start 32 requests on
fast sources; it only lets the existing startup probes reach 32 when justified.

Benchmark loader overrides accepted by `examples/io/main.zig`:

```text
ZML_LOAD_ADAPTIVE
ZML_LOAD_PARALLELISM
ZML_LOAD_INITIAL_PARALLELISM
ZML_LOAD_MAX_READ_PARALLELISM
ZML_LOAD_DMA_CHUNKS
ZML_LOAD_DMA_BUFFER_MIB
ZML_LOAD_DMA_CHUNK_MIB
ZML_LOAD_READ_CHUNK_MIB
ZML_LOAD_MAX_STAGING_MIB
```

Example:

```sh
LATENCY_MS=250 SPEED_MIB=1000 \
ZML_LOAD_PARALLELISM=8 ZML_LOAD_MAX_READ_PARALLELISM=32 \
ZML_LOAD_DMA_BUFFER_MIB=32 \
./bench_s3.sh
```

Do not use the CPU sweep to tune DMA chunk or buffer size. Run
buffer/quantum/pinned/DMA sweeps on CUDA/oneAPI and compare committed H2D
goodput plus total wall time.

### CUDA Llama 3.1 8B sweep, 2026-07-18

Harness:

- CUDA device 1, release `//examples/io:playground`;
- 291 tensors, 14.96 GiB logical bytes, sharded model placement;
- warm local files under `~/s3proxy/data/lfm`;
- S3 cases use the checked-in `bench_s3.sh` and the same backing files;
- elapsed/goodput below are the adaptive loader's final logical values.

Warm local fixed legacy sweep:

```text
workers          1       2       4       8      16
elapsed (s)   2.123   1.490   1.157   1.254   1.751
MiB/s          7215   10276   13200   12212    8749
```

Four workers are the knee. The final adaptive default converges from 2 to
4 reads/4 DMA lanes, uses five pinned chunks and zero pageable blocks, and has
a 1.168 s median over the final ten repetitions (0.95% slower than the fixed-4
median). An earlier 20-run set after the sustained-probe fix had a 1.158 s
median. The final handoff run completed in 1.150 s at 13321.54 MiB/s. All runs
remained direct (`14.96 GiB direct`, `0 staged`).

Forced adaptive width 8 completed 20/20 runs after the scheduling fix, with a
roughly 1.25 s median. Before the fix it intermittently reached all 291 tensor
completion records and then spun forever during teardown.

S3Proxy at `LATENCY_MS=10 SPEED_MIB=1000`:

```text
configuration                         elapsed (s)   MiB/s
fixed legacy 1                           50.788       302
fixed legacy 2                           26.102       587
fixed legacy 4                           14.577      1051
fixed legacy 8                            9.053      1692
adaptive before ready/probe fixes        24.217       632
adaptive final, two runs              5.066-6.100  2511-3023
tuned start 8/read cap 32/staging 1 GiB   3.387      4522
```

The final default is materially faster than the best fixed legacy width, but it
still varies in how quickly it accepts the last read/staging expansion. For a
known high-latency source, `ZML_LOAD_INITIAL_PARALLELISM=4` or `8` is a useful
deployment override; changing the universal default to 4 was rejected because
it made the warm local controller accept 8 lanes and regress to about 1.25 s.

Latency deadline validation:

```text
LATENCY_MS  SPEED_MIB  adaptive result
1000        1000       89.057 s, 171.99 MiB/s, final 32 reads/8 DMA
1000         100       84.241 s, 181.82 MiB/s, final 32 reads/8 DMA
```

The legacy fixed-8 loader did not finish the `1000/1000` profile within 120 s,
likely because its sequential stream path uses a less favorable S3 request
granularity. Before raising both probe deadlines and bounding read backoff, the
adaptive `1000/100` run timed out after 180 s at 113/291 tensors and 6.41 GiB.
The final run completed all bytes, though one 4-to-6 DMA capacity attempt still
timed out before later probes reached the cap. Long-latency probe scoring and
repeated high-read staging retries remain optimization opportunities.

### Four-GPU oneAPI Llama 3.1 8B sweep, 2026-07-18

Harness:

- four Intel Battlemage G31 / Arc Pro B70 devices through oneAPI;
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3` and release
  `//examples/io:playground`;
- 291 tensors and 14.96 GiB logical bytes, normally sharded across the four
  devices;
- warm local files under `~/s3proxy/data/lfm`; S3Proxy uses the same files;
- elapsed values are the loader's `completed` time, not Bazel startup or the
  occasional process-exit delay described below.

The original 256 MiB-coupled default is a poor local fit. Three adaptive runs
were 5.400, 4.648, and 5.154 s (5.154 s median). They remained fully direct and
ended at four read/DMA lanes, five pinned blocks per device, and zero staging.
The first useful progress could take roughly 2.0-2.5 s while large per-device
pinned buffers and transfer managers were created.

Warm local fixed-loader sweep with the original 256 MiB blocks:

```text
workers          1       2       4       8      16
elapsed (s)   5.194   4.036   4.201   4.711   6.995
```

With 32 MiB blocks, fixed width 2 ran in 3.560, 3.642, and 3.623 s
(3.623 s median). Width 4 took 3.649 s once. Wider concurrency is not the
answer on this host; pinned allocation size and lane startup dominate first.

Changing `dma_chunk_size` alone conflates physical allocation with scheduling:

```text
coupled buffer/quantum    local adaptive       S3 10ms/1000MiB/s
32 MiB                    3.642 s median        9.877 s
64 MiB                    3.737 s median       10.216 s
128 MiB                   3.910 s               not run
256 MiB                   5.154 s median        8.401 s
```

Small coupled quanta make local pinned allocation cheaper but cause more lane
turnover/submission pressure and slower S3 convergence. This motivated the
separate `dma_buffer_size` setting. Keeping a 256 MiB scheduling quantum while
varying only the pinned block produced:

```text
pinned block   warm local                    S3 10ms/1000MiB/s  S3 250ms/1000MiB/s
32 MiB         3.762 s median (3 runs)       6.285 s median     21.187 s
48 MiB         3.820 s                       7.797 s             not run
64 MiB         3.548 s initial median        6.871 s             32.731 s
128 MiB        3.711 s                       7.171 s             not run
256 MiB        5.154 s original median       8.401 s             24.522 s
```

Later three-run local monitoring at 64 MiB had a 3.853 s median; the combined
six-run median is about 3.708 s, 2.3% slower than the fixed-width-2/32 MiB
median and still inside the 3% band. The initial three-run 3.548 s result shows
that oneAPI run-to-run variance is material. The 32 MiB S3 result spans
5.985-8.299 s; its 6.285 s median is still substantially better than the
original one-run 8.401 s result, but needs a larger repeated sample before a
universal default change.

At `250 ms/1000 MiB/s`, a known-remote override of initial width 8 with a
32 MiB buffer reached 32 reads/8 DMA lanes in 18.858 s, versus 21.187 s from
the default initial width and 24.522 s with the original 256 MiB buffer. Its
peak RSS was about 8.7 GiB, so the time gain is not free. The default-width
32 MiB run peaked around 6.7 GiB.

There is no defensible single static block size from this matrix: 64 MiB is the
balanced local choice, while 32 MiB is the tested remote choice, and 48 MiB
missed both observed 3% bands. Keep `dma_buffer_size` opt-in until buffer size
can be chosen from stronger cross-source evidence or adapted safely. Useful
commands are:

```sh
ZML_LOAD_DMA_BUFFER_MIB=64 ./bench_file.sh
LATENCY_MS=10 SPEED_MIB=1000 ZML_LOAD_DMA_BUFFER_MIB=32 ./bench_s3.sh
```

Four-device replicated validation with a 64 MiB block and 256 MiB quantum
completed in 9.758 s. The loader read 14.96 GiB once and reported 59.83 GiB
submitted/committed, exactly the expected physical-versus-logical distinction.
It finished fully direct with no pageable staging.

Problems/opportunities exposed by this sweep:

1. Pinned block size and logical DMA fairness quantum were incorrectly coupled.
   The new optional setting separates them and queue-capacity pressure is now
   normalized by physical buffer bytes.
2. Bursty S3 ready data still causes sawtooth staging probes and aggressive
   read/staging backoff. An experiment that retained staging for the reduced
   read width and counted draining blocks in the occupancy denominator improved
   one 64 MiB run but regressed a 32 MiB run; it was not retained. The next
   policy experiment needs repeated same-source scoring, not another threshold.
3. Startup-width 8 improves known-remote time but increases memory and hurts the
   unknown-source objective. Source classification or a cheaper read-only
   bootstrap remains preferable to changing the universal initial width.
4. Three runs intermittently logged all tensors, `controller stopped`, final
   `completed`, and `Loaded weights`, then took roughly 35-40 additional seconds
   to exit. Three targeted monitored repetitions did not reproduce it, and no
   live stack was captured. Because the loader's final records precede the
   pause, investigate oneAPI/PJRT/platform teardown separately from pipeline
   completion.
5. `dma_utilization` can exceed 100% in startup/transition windows because work
   time is completion-attributed across sampling boundaries. Treat it as a
   saturation hint, not a literal occupancy integral, until the metric is made
   window-resident.

### Four-GPU sharded CPU profile, 2026-07-18

The user's initial `profile_file.sh` capture used the warm local sharded
workload, release oneAPI build, four devices, default 256 MiB pinned blocks,
10 kHz cycle sampling, and call graphs. It contained 79K samples over 5.310 s
with no lost samples. The dominant flat symbols were:

```text
symbol                                      self cycles
kernel _copy_to_iter                            29.20%
libc __memmove_avx512_unaligned_erms            19.55%
kernel clear_page_erms                          11.45%
playground compiler_rt.memset                    8.17%
kernel entry_SYSCALL_64                          4.67%
kernel ___pte_offset_map                         2.83%
kernel filemap_get_read_batch                    2.37%
```

The call graphs distinguish these costs:

- `_copy_to_iter` is the positional ext4/page-cache read copying model bytes
  into the destination host buffer. It is useful storage work.
- the 19.55% `memmove` runs under oneAPI's
  `SyclStream::DoHostCallbackWithStatus` host task. No significant sample path
  enters `DirectMemoryWriter.commitWindow`; the large copy is owned by the
  PJRT/oneAPI transfer implementation, not ZML's sharding scatter loop.
- most `clear_page_erms` samples are under XE/TTM/Level Zero backing allocation.
  This is first-use driver page clearing for GPU-visible allocations.
- 7.96 percentage points of `compiler_rt.memset` came directly from
  `DynamicBufferPool.getWithWait`. Zig's `Allocator.alignedAlloc` writes the
  undefined-memory poison pattern across every new block even though these pool
  blocks are immediately filled by a read.

`DynamicBufferPool` now uses `Allocator.rawAlloc` for its deliberately
uninitialized blocks. This matches its existing `rawFree` lifecycle and does
not remove a data copy: readers still initialize every consumed byte before it
is submitted. A fresh profile reduced `compiler_rt.memset` from 8.17% to 0.11%;
the residual samples are unrelated small program allocations.

A controlled three-run A/B held the fixed loader at width 4 so adaptive lane
decisions could not confound the result:

```text
pinned block  allocator path    median CPU task  median cycles  median elapsed
256 MiB       poison/aligned          6.893 s        36.20 G          4.134 s
256 MiB       raw/uninitialized       6.572 s        34.62 G          3.847 s
64 MiB        poison/aligned          5.960 s        31.18 G          3.522 s
64 MiB        raw/uninitialized       5.780 s        30.27 G          3.418 s
```

The retained change therefore reduces median CPU task time by 4.7% at 256 MiB
and 3.0% at 64 MiB. Median elapsed time improved 6.9% and 3.0%, respectively.
Do not use the larger difference from uncontrolled adaptive before/after runs;
controller decisions and oneAPI variance changed between those samples.

Static lane measurements after the fix show that remaining concurrency CPU is
a tradeoff, not pure waste:

```text
block/width   median CPU task   median elapsed
256 MiB / 2        5.322 s          4.123 s
256 MiB / 3        6.132 s          4.162 s
256 MiB / 4        6.572 s          3.847 s
64 MiB / 2         4.859 s          3.686 s
64 MiB / 4         5.780 s          3.418 s
```

At 64 MiB, width 4 spends about 19% more CPU than width 2 but is about 7.8%
faster, so forcing two lanes would violate the current wall-time-first 3%
objective on this sample. Three lanes is dominated. Three adaptive 64 MiB runs
used 4.751-4.993 CPU-seconds and took 3.543-3.887 s; two finished at width 2.
That keeps CPU low but its 3.561 s median is 4.2% behind the fixed-width-4
median, narrowly outside the target amid known run variance.

Block size remains the larger safe operational lever for warm local sharded
loads: at fixed width 4, 64 MiB versus 256 MiB reduces median CPU task time by
12.0% and elapsed time by 11.2%. Use:

```sh
ZML_LOAD_DMA_BUFFER_MIB=64 ./profile_file.sh
```

Do not make 64 MiB universal from this result. The prior S3 sweep found that
32 MiB worked better on tested remote profiles and that 64 MiB regressed the
250 ms profile. Remaining large CPU costs are required file copying, oneAPI's
PJRT host callback copy, and XE/TTM allocation clearing. Further reduction
requires a PJRT/oneAPI transfer path that accepts the loader's pinned buffer
without the plugin copy, longer-lived/reused GPU-visible allocations, or safe
source-aware block sizing; no additional local ZML copy hotspot was measured.

## Research Versus Current Implementation

`RESEARCH.md` recommends a startup-plus-Gradient2/Vegas style controller with
EMAs, a no-load delay baseline, queue guardrails, and one global controller.

Implemented now:

- multiplicative startup/read expansion;
- committed-byte objective;
- byte-weighted latency;
- bounded queues and hard memory caps;
- independently configurable per-device pinned block size and logical DMA lane
  quantum;
- queue/latency guardrails on H2D;
- one-dimensional probes with 3% hysteresis;
- resource minimization within 3% of peak;
- periodic reprobes/reductions;
- finite-tail probe suppression;
- capacity activation and epoch attribution.

Not implemented literally:

- Gradient2 fast/slow latency EMAs and gradient update equation;
- an explicit bandwidth-delay-product model;
- source throttle/retry/HTTP status metrics;
- a learned read-latency baseline used for control (intentionally removed after
  the collapse; request latency alone is not safe pressure);
- dynamic `read_chunk_size`;
- source-type or locality-aware scheduling;
- a full static-sweep benchmark harness.

Do not add independent read and DMA controllers. If the policy is replaced with
a gradient law, preserve the shared knob owner, bounded pools, activation/epoch
protocol, and committed-byte objective.

## Known Risks and Follow-up Work

1. Broad pageable prefetch can open many tensors and interleave positional reads
   across files. This is useful for HF latency but may reduce sequential readahead
   and locality on some disks/network filesystems. Benchmark before adding a
   source-fanout/locality heuristic.
2. `slow_reads < 1.5 GiB/s` is a fixed heuristic. It may need normalization by
   configured read size or an observed direct-source baseline.
3. Global starvation is recorded when a waiter wakes. An interval crossing a
   sampling boundary is attributed at completion and capped. This is robust for
   control but not a perfect time-series integral.
4. The controller has no explicit transport throttling signal. Transport errors
   cancel the load rather than producing an AIMD backoff sample.
5. Resource reduction ordering is heuristic. Verify that trimming staging before
   DMA does not lose a better low-pinned configuration on new hardware.
6. `active_transfers` counts open writers, not active DMA lanes. Rename the
   misleading log label if touching logging.
7. Slot metadata is `O(open_sources * max_read_parallelism)`. Buffers remain
   bounded, but extreme caps could create metadata/file-handle pressure.
8. Startup zero-progress bootstrap does not have a rich source classification;
   fast local paths rely on making progress before the 100 ms trigger.
9. Multi-device pool totals and logical versus physical replicated bytes require
   care. Controller objective uses physical PJRT committed bytes while final
   logical goodput uses model logical bytes; compare like with like in analysis.
10. S3 committed output is bursty. Increase probes now accept the better of
    epoch and current-window goodput after 200 ms, but repeated 16-to-32 staging
    retries still produce run-to-run convergence variance. A cumulative
    post-activation committed-byte objective or an EWMA baseline is the next
    policy experiment; preserve epoch capacity proof.
11. CUDA and four-device oneAPI sharded loads are covered, and one oneAPI
    replicated load completed correctly. Replicated performance has not had a
    static width/buffer sweep and should not be inferred from the sharded knee.
12. Success-path lifecycle hangs were stress-tested. Read/PJRT failure injection
    and transport-level cancellation remain required before treating teardown as
    fully proven.
13. `dma_buffer_size` is static for a load. The observed local and S3 knees differ
    (64 versus 32 MiB), and the 48 MiB compromise missed the 3% band. Do not make
    an automatic default from this small matrix without cross-source repeats.
14. oneAPI process exit occasionally pauses for roughly 35-40 s after the loader
    and the playground both log completion. The loader counters are complete and
    three targeted reruns exited normally, so the responsible PJRT/platform
    teardown frame is still unknown.
15. On the four-GPU sharded profile, roughly one fifth of sampled cycles were in
    a libc `memmove` below oneAPI's SYCL host callback. The retained pool change
    removes ZML's allocation poison pass but cannot remove this plugin-owned
    transfer copy. Recheck this profile when the oneAPI PJRT package changes.

## Validation

Fast focused validation:

```sh
./bazel.sh test //zml:test
./bazel.sh build //zml
git diff --check
```

The inline suite covers:

- positional bounds, truncation, stream-position preservation, concurrency;
- direct writer sharding, mirrors, staged/direct transitions, park/unpark;
- mutable limited groups and dynamic pools;
- startup/read/DMA/pinned/staging probe decisions;
- hard caps, attribution, capacity activation, timeout rollback;
- pressure, latency reliability, starvation deduplication;
- resource reduction and global 3% band;
- scheduling fence and stable ready queue pointers.
- direct writers whose pinned block is smaller than their logical DMA quantum,
  including replicated multi-device placement.

Performance validation must compare the same model/source/cache state:

1. `adaptive_parallelism=false` fixed baseline.
2. Adaptive default.
3. Static sweep of read workers, DMA workers, pinned chunks, and staging blocks.
4. Repeat for warm local cache, cold local disk, HF, S3, and any network-mounted
   filesystem relevant to deployment.
5. Record total elapsed/logical goodput, committed goodput, direct/staged bytes,
   peak/final knobs, allocated pinned/pageable blocks, and starvation.

As of the snapshot date, 209 tests pass (one platform-specific test skips) and
the four-GPU oneAPI release playground builds with the raw pool-allocation
change. The earlier CUDA regression set still stands. On four-device oneAPI,
the post-fix adaptive 256 MiB quantum/64 MiB buffer median is 4.2% behind the
fixed-width-4 median in a three-run `perf stat` sample; both remain fully direct
with zero staging. A 32 MiB buffer improves both tested S3 profiles but
convergence remains noisy; the 10 ms profile spans 5.985-8.299 s over three
runs. One replicated load completed with the expected 4x physical committed
bytes. The intermittent post-completion oneAPI exit delay remains unresolved.

Acceptance target:

- completion time within 3% of best static sweep;
- among configurations in that band, choose the fewest pinned and pageable
  blocks;
- fast local source stays direct and allocates zero pageable blocks when direct
  reads keep DMA fed;
- high-latency source ramps reads quickly, stages within budget, and suppresses
  DMA starvation;
- no repeated unexercisable probe loop;
- no leak or hang on injected read/PJRT error.

## Agent Workflow for the Next Change

1. Confirm branch, commit, dirty files, and whether a new `log.txt` matches this
   code revision.
2. Read `configured`, final `completed`, and every `limits updated` line first.
3. Build a knob timeline. Locate the first irreversible throughput change.
4. Correlate that action with the four window records immediately before it.
5. Distinguish controller error from data-plane scheduling error:
   - bad action with valid metrics: controller policy;
   - configured capacity never active: scheduling/pool capacity;
   - capacity active but no epoch bytes: attribution/data-plane dead time;
   - queue cycling with no storage/ordered progress: lane state machine;
   - completed reads but no committed bytes: writer/H2D path.
6. Do not tune thresholds until checking metric semantics and attribution.
7. Make one control-dimension change at a time.
8. Add a deterministic controller/state regression test for the exact failure.
9. Run focused tests and package build.
10. Request a same-workload rerun; compare total time and knob timeline, not one
    peak window.
11. Update this file when changing architecture, invariants, metric semantics,
    or known failure modes.
