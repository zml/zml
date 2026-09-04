# Direct DMA loader context

This is a compact, agent-oriented handoff. It records current behavior, durable
decisions, useful measurements, rejected approaches, and open work. It is not a
runbook. Re-check code, Git refs, available accelerators, and plugin artifacts
on each machine before relying on an old result.

Last consolidated: 2026-09-03 at the start of the third pass (caller-controlled
concurrency), on commit `2f9cac2b`. `PLAN.md` is the sequential implementation
checklist; this file is the canonical description of the code after each
completed task. The "Current design" section describes the second-pass code
until third-pass tasks rewrite it; "Third-pass design" records the target.
`origin/master` was `e1e983c8` during the 2026-09-02 audit; never assume that
ref is still current.

## Current design

### Scope and API

- `zml.io.Loader` selects the direct path for CUDA, ROCm, and oneAPI and the
  buffered path otherwise. A loader owns its store, sharding/profile options,
  worker pool, and every handle it created. Every submission returns a
  `Handle`: `load(Model, model, buffers)` submits all single-source tensors of
  a model; `loadExecute(bindings)` submits the sources of one or more
  `Binding{tensor, output, exe}` as ONE planned submission so adjacent
  sources of different bindings coalesce. Any number of handles may be open;
  `Handle.await` (idempotent, cached) waits for the submission's reads and
  DMA, then for bindings runs each executable in binding order on the
  awaiting task with `.wait = true`, writes `output.*`, frees the inputs and
  commits the submission's logical bytes to `bytesLoaded()`. `awaitAll`
  awaits in publish order; `deinit` awaits open handles without running
  executables and frees every handle. A zero-byte tensor is `error.EmptyTensor`.
- The loader has no memory policy: `zml.io.Window{budget_bytes, max_handles}`
  is the caller-side policy, awaiting the oldest pending `loadExecute` handle
  before submitting the next one whose inputs (sized by
  `Loader.executeInputBytesPerDevice(exe)`) would exceed the budget or the
  handle cap. A window of one reproduces the former synchronous behaviour.
- `VFS.loadProfile(path)` is prepared once for a model load and passed as a
  borrowed `LoadProfile`. It contains a backend name, minimum read chunk,
  `high_latency`, and optional aggregate retry/throttle feedback. It assumes
  the load is the backend's only material user; feedback is not load-tagged.
- Profile minima are local/file 8 MiB, HTTP/S3/GCS 16 MiB, and HF 32 MiB.
  Effective source request size is the greater of the profile minimum and
  calibrated DMA block size, capped at the supported 32 MiB maximum.
- Adaptive and fixed source-width configurations remain supported. The current
  `Loader` API differs from the adjacent monorepo's older checkout: store,
  shardings, progress, and profile now belong to loader initialization.
- `DirectMemoryWriter`, `DirectShardWriter`, and `DynamicBufferPool` are no
  longer public loader mechanisms. There is no executor inside the loader:
  executables run on whichever task awaits the handle, so the caller's number
  of un-awaited `loadExecute` handles bounds device memory and its await
  order is the execution order.
- Model traversal, tensor-store lookup, resolved sharding selection, and output
  flattening happen once in the shared `Loader.load` front end. Executable
  source lookup, validation, input-shell allocation (`BoundExecutable`),
  execution, and output ownership similarly live once above the backend
  split; buffered and direct loaders implement only `submit(specs)` and
  `awaitBatch`. The buffered memory writer is private.
- `zml/io.zig` is the public/store front end and buffered backend. Direct
  planning, scheduling, adaptive control, transfer lifecycle, and their tests
  are in `zml/io/direct_loader.zig`; platform-owned DMA settings, retained
  arenas, and calibration are in `zml/io/dma_calibration.zig`; pure
  sharding-to-byte-span expansion is in `zml/io/dispatch_spans.zig`. Shared
  loader limits and option types are small leaf modules, and `zml.io` continues
  to re-export the existing public names.
- `Loader` owns IO, platform, store, and front-end options directly rather than
  recovering duplicated state from its active backend. The buffered backend
  retains none of the unused store/options; the direct backend retains only
  the load profile and progress pointer needed after construction.
- Both backends count a submission's logical bytes only when its await
  succeeds. Direct diagnostics are batch-owned (publish, seal, first-claim,
  first-read and completion offsets from loader creation, planned
  jobs/runs/items/transfers, plan count with total and longest per-file
  planning time, VFS delta since publish) and logged once per batch at await;
  loader-wide totals (reads, DMA submissions, pool high-water/mapped, width)
  are logged once at `destroy`.
- The compatibility target observed in `~/github/zml/monorepo` is behavioral:
  repeated `loadExecute`, whole-model `load`, multi-source `TensorStore`
  bindings, and cumulative loaded-byte accounting. That checkout is migrated
  to the handle API in task 5.

### Source and VFS data plane

- `safetensors.readFilePositionalAllV` is the one exact positional scatter
  implementation for tensor readers and direct-loader pinned reads. It handles
  short-read resumption and local `IOV_MAX` batching; `TensorReader` adds tensor
  range validation and borrowed readers over a shared open file. Each distinct
  safetensor object is opened once per model-wide load.
- HTTP, S3, GCS, and HF share one range read loop,
  `range_read.performRangeRead`: one whole Range request per admitted
  positional call, retries serial inside that caller's source credit (the
  retired backend-local `parallel_read` pools must not return). A backend
  contributes a `RequestSpec`: its name and the object for log lines, what a
  503 means, and a `prepare` hook called for every attempt that returns the
  URI, the authorization value and its extra headers; the loop appends
  `Range`. S3 recomputes the SigV4 timestamp and signature in the hook per
  attempt, GCS copies (refreshing when expired) its bearer per attempt, HTTP
  and HF return a static request. The backends keep URL construction,
  credential assembly and their `backend()` profile; retry settings are one
  `RetryConfig` built from each backend's `InitOpts`, and the stats provider
  is `AtomicReadStats.provider()`.
- Registration is `VFS.registerBackend` only (`VFS.register` is gone);
  `loadProfile` resolves a bare path through the registered `file` backend and
  falls back to `LoadProfile.local`.
- Shared Range handling reads `Content-Range` before taking the body reader
  (which releases the head bytes), validates that it covers the request,
  handles a server returning `200` and ignoring Range by discarding the
  prefix, then scatters into caller buffers; there is no response timing (the
  one-byte first-body probe and `ResponseTiming` were dead and are gone).
  Retry classification is
  one function (`range_read.classifyStatus`): 408 is a timeout, 429 a
  throttle, other 5xx a server failure, and 503 is a throttle on S3
  (`SlowDown`) and GCS but a server failure on generic HTTP and HF. A retried
  status whose response names a delay (`Retry-After` delta-seconds, or the
  `RateLimit` header's `t=` reset on HF; the HTTP-date form is not parsed)
  sleeps that long instead of the jittered delay. The backends expose
  aggregate request, retry, timeout, server-failure, throttle, byte and delay
  counters.
- One source job performs one exact absolute scatter read into pinned blocks.
  Extra physical calls occur only for short reads/retries or `IOV_MAX` limits.
  Diagnostics distinguish planned jobs from physical calls.

### Coalesced source planner

The old planner made jobs tensor-local. `source_request_size` was only a cap,
so small adjacent tensors each caused a source operation. `DirectLoader.submit`
now sorts the selected ranges once by file URI, absolute source offset and
size (`sortedItemOrder`) and plans one file at a time (`preparePlan`),
publishing each file's plan as soon as it exists (`publishFiles`), so the
workers read the first file while the later ones are planned. Each plan:

1. Covers one file group of the sorted order.
2. Forms the union of touching or overlapping requested ranges. It never reads
   across an unrequested gap (or a file boundary, which is now a plan
   boundary). Duplicate/overlapping bindings are read once but retain a
   transfer piece for every output.
3. Partitions each merged run into the minimum number of jobs permitted by the
   request/block/`IOV_MAX` limit.
4. Keeps that minimum count while preferring tensor-safe boundaries. For
   `N = ceil(run_length / max_job_length)`, each cut chooses the latest safe
   tensor boundary that leaves the remainder fitting in `N-1` jobs; otherwise
   it uses the latest feasible hard cut. A boundary is safe only if the next
   touching interval begins at the current union end, which preserves overlap
   and duplicate coverage.

Coalescing is plan-local: one plan per source file within a submission, and
a submission's plans are claimed in file order. Planning uses per-device
physical-byte charges and same-file predecessors to compute a deterministic
fair, predecessor-safe job order per plan (`fairOrder`), then discards the
temporary queues and charges; for one device the fair order is the identity
(every job charges that device and predecessors precede their tails in
planning order; asserted by a test), so the planner keeps the planning order
and skips the queues. While those spans are available, the planner also emits
the final item, block index/offset, writer mask, destination offset, and length
records. The published plan owns source jobs physically arranged in final fair
order and their final transfer records; runtime tensor state does not own
another dispatch plan. Planning-only predecessors, order indirection, and
remaining-work suffix arrays are discarded. Physical source bytes are distinct from logical tensor
bytes so duplication and replication do not distort diagnostics or fairness.
Planning is `O(tensors log tensors)` per submission and took 0.32 s for
DeepSeek-V4-Flash as one plan on MI300 (task 3 tree, 46 shards); as per-file
plans only the first file's planning delays the first claim, and the rest
overlaps the reads (Llama: 4 plans, 1-2 ms in total).

### Pinned blocks, scattering, and ownership

- A worker initializes the destinations referenced by its planned records,
  derives block references, NUMA affinities, and destination queue counts,
  atomically leases all pinned blocks for the source job, reads into them, and
  publishes the records to existing per-tensor PJRT transfer managers.
  Compatible adjacent records were already merged during planning; workers do
  not traverse sharding spans or rebuild/split a transfer list.
- The same pinned block may feed several tensor transfers. One atomic lease is
  counted across every consuming PJRT event; its final completion both releases
  the block and completes the parent request. It is released only after all child
  transfers finish or are abandoned. Source/allocation/enqueue/PJRT failures
  close scheduling, retire the batch's unclaimed jobs, fail unfinished
  buffers, and release each reference exactly once; the batch still reaches
  `done` through its normal completion units.
- `enqueueBlocks` reserves all affected destination queue capacity before
  mutation, publishes a complete source job under one metadata lock, and pumps
  once. The prior per-piece enqueue caused roughly 69k--79k locks and pumps.
  Pre-reservation makes allocation failure atomic.
- Workers retain scratch for leases, affinities, reference counts, iovecs,
  queue counts, and the pool's affinity-matching search; request, block and
  event contexts come preallocated with the plan (below). Positional-read
  rewrite scratch is stack bounded. A source job performs no allocator call
  in steady state, neither for pool matching nor for its contexts; each
  caller has separate matching scratch, so blocked acquisitions cannot
  overwrite one another.
- Source coalescing deliberately preserves per-tensor device buffers. Roughly
  one DMA submission per tensor is therefore the natural floor. Going much
  lower requires packed device allocations or a device-side scatter/copy stage
  and changes buffer ownership/model layout.

### Scheduling and concurrency

- Planning charges a coalesced job's physical bytes to every destination device
  and simulates the fairness policy once per plan (one source file of a
  submission); the plan's jobs are stored in that immutable order and a
  submission's plans are claimed in file order. Same-file predecessor order and per-target
  submitted-prefix accounting preserve ordering while unrelated reads and DMA
  complete out of order. There are no live device queues, debt counters,
  claimed bitmap, runtime order indirection, or suffix-metadata arrays.
- `FairVectoredReadScheduler` is a strict FIFO of published batches under one
  mutex and condition: `publish(batch, plan)` appends a plan to an open batch
  behind every earlier plan and batch (the batch joins the queue with its
  first plan that has jobs; capacity is reserved first so the plan, its
  completion units and the queue entry appear together), `seal(batch)` ends
  the submission, `claim` walks the head batch's plans in order and hands out
  the next job by advancing that plan's plain cursor, `waitForWork` sleeps
  while nothing is unclaimed, `fail` retires the unclaimed units of every
  plan of every queued batch and clears the queue, `snapshot` reports the
  unclaimed total. A later batch's first job is claimed only after every job
  of the earlier ones; fairness is intra-plan, plans follow file order and
  the caller's submission order is the cross-batch policy. The queue holds
  only open batches (their submission still planning) and batches with
  unclaimed jobs: a sealed batch is popped with its last claim, at its seal
  when already exhausted, or by `fail`. An open head whose published plans
  are exhausted keeps the head; the unclaimed total is then 0, so workers
  sleep in `waitForWork` until the next plan's broadcast, at most one file's
  planning time. A batch can only be freed after `done`, which needs the
  sentinel (dropped only after the seal) and every job claimed or retired,
  so a queued batch is never freed and no worker rendezvous is needed.
  Persistent workers loop `waitForWork` -> lifecycle credit -> `claim` ->
  read -> enqueue; the claim takes the mutex the idle wait already takes.
- A `Batch` owns its plans (heap `Plan{jobs, transfers, requests, blocks,
  events, events_used, planning_ns, cursor}`, one per file in file order)
  and its items from creation. The planner preallocates every context a
  plan's jobs can need: one `RequestContext` per job (initialised idle:
  nothing pending, completed), one `BlockContext` per job block (exact
  `divCeil(len, block_size)`; `Job.blocks` is the job's slice) and one
  `EventContext` per planned DMA submission (the transfers' writer count,
  `planned_dma_submissions` on the batch line), handed out in submission
  order under `metadata_mutex`. A `Job` carries its request and block
  slots and a `Claim` its plan, so `registerRequest` and `registerBlock`
  cannot fail and take no lock; a load allocates only per file. The batch
  completes when `remaining` (one publish sentinel held until the seal,
  plus one unit per job added as each plan is published) reaches zero and
  sets `done`. A job's unit is released exactly once: at the request's
  final reference drop (last DMA callback or abandonment; `completeOne`
  releases the lifecycle credit first and calls `finishJobs` last) or by
  `scheduler.fail` for unclaimed jobs, whose request slots stay idle.
  Callback order rule: the PJRT ready callback and the submission failure
  path copy `pipeline`, `device_index` and `block` into locals, call
  `eventCompleted` (which may pump on the callback's thread), push the
  context onto the pipeline's `retired` stack, then `block.complete()`
  last, because the block's completion may free the batch.
  `DirectLoader.submit` creates the batch and its items, sorts once, plans
  and publishes one file at a time, seals and
  drops the sentinel; a planning failure before the first publish destroys
  the batch and returns the error, and one after it fails the pipeline with
  that error (a partial submission can never complete), seals, awaits and
  retires the batch inside `submit`, so the caller sees only the sticky
  error. `awaitBatch` waits `done` (after which no worker or callback
  touches the batch), marks unsubmitted targets when the pipeline failed,
  retires the contexts under `metadata_mutex` (so neither `abortReady` nor
  a pump draining `retired` can race the free: leftover events are
  destroyed, the batch's contexts unlinked, and the completion asserts run
  over the arrays), logs the batch and frees items and plans with their
  arrays. Nothing drains the gates or the controller per batch and there is
  no barrier: the controller sees submissions only as activity (below). A
  pipeline failure is sticky: every open handle's await returns it and later
  submissions are rejected with it. There is no loader-wide context list or
  reap. The buffered backend mirrors this with `BufferedBatch{pending, done}`
  over the `LimitedGroup` read tasks.
- Source concurrency is adaptive for every direct-loader profile. The old
  conversion of local/default `.adaptive` to `.fixed = 12` was removed.
  `high_latency` only permits blind pre-response bootstrap to 24 then 32.
- Default source configuration remains adaptive initial 12, maximum 128,
  clipped by pinned-memory feasibility. Twelve is an empirical bootstrap, not
  a value derived from tensor count, request size, storage queue depth, or
  bandwidth-delay product.
- The source ladder is `1,2,4,8,12,16,24,32,48,64,96,128`. Ninety-six is only
  a ladder rung and was useful as a fixed S3Proxy control; it is not a default
  or model-derived number.
- Completed jobs contribute their actual byte count to adaptive evidence,
  including partial tails. The controller (`SourceReadWidthController`) is
  climb-and-hold with two states. Climbing: each window (at least 100 ms of
  busy time, max(8, width) completions, the width exercised) scores the
  current rung into that rung's mean; a rung that beats the best rate seen
  by 3% moves the width one rung up (or holds at the pinned clip); the first
  rung that does not ends the climb, and the controller holds at the lowest
  measured rung at or below the best within 3% of it. Before holding it
  re-measures that hold rung once when its retention is within 0.02 of 0.97,
  and when the hold rung is the start rung it probes the rung below once and
  holds at the better answer. Holding: evidence changes nothing. A plain
  load therefore spends three or four windows away from its final width. No
  tail rule: a window that cannot complete before the load ends leaves the
  width in place. Metadata can cheaply clip feasibility, but cannot predict
  a source's latency/bandwidth saturation point, so no job-size-derived
  initial-width heuristic was added. Confirmed on B70 (holds 12 or 16 at
  equal load time, 0.66 s against 0.63 s fixed 12); MI300 and CUDA
  confirmation pending.
- Measurement mechanics are separate from width policy. Runtime state is one
  value—inactive (holding), measuring (climbing) or blind (pre-response
  bootstrap)—rather than several coupled booleans. Every decision opens a
  generation: `applyDecision` puts both gates at the width and fences the
  generation's window at the next admission (`prepareProbe`), so every read
  is attributed to the width in effect when it was admitted and nothing is
  ever drained. The measurement layer rejects stale or insufficient evidence
  before invoking the controller. Probe counters are ordinary fields
  protected by one mutex; only the source-call configuration generation
  remains atomic for lock-free worker admission.
- Two gates separate clean read-measurement generations from complete request
  lifecycles. All workers compete for lifecycle capacity, configured
  as active reads plus one shared spare and clipped by pinned feasibility; the
  read gate alone limits source calls. A request returns lifecycle credit only
  after all its DMA children finish.
- Nothing closes the read gate. A changed width sets the new limit and the
  reads admitted under the previous generation return at their own pace,
  excluded from the new window by the fence. Source backpressure has two
  classes, read from the profile's stats side channel every control tick
  (`ReadStatsCursor.takeBackpressure` returns `{throttle, transient}`; the
  local file backend has no side channel and never sees either). Throttle (a
  throttle or timeout moved): `backoff` lowers the width one rung, clips the
  ladder there and holds. Transient (retries, connection failures or other
  5xx moved without a throttle): `stepDownTransient` lowers one rung with
  the ceiling and state unchanged; a climbing controller restarts its climb
  at that rung (it becomes the best rung and its mean is forgotten, so the
  next window there is a fresh climb sample and the width can climb back
  above the step), a holding one keeps holding. Both are limited to once per
  generation: a further sample in the generation a step opened is ignored
  unless a read admitted under that generation has begun, so delayed
  old-width feedback cannot ratchet through several rungs. A single early
  500 therefore costs one rung and one window instead of pinning the width.
  `gate_closed_ticks` in the loader summary counts control ticks that found
  the read gate at 0 with jobs unclaimed and is 0 by construction.
- A window opens at its generation's first completed read, which is not
  counted; from then on completions arrive at the source's steady rate, so
  the window's bytes over its busy time is the true throughput even on a
  high-latency source (a clock started at the first admission charged the
  whole round trip and reported 450 MiB/s for a width that delivers 600).
  The window clock counts busy time. The 10/25 ms control tick still runs
  while workers sleep in retries (it samples backpressure); a tick that finds
  nothing unclaimed, no pending source job and no read permit held, after
  the window opened, charges the interval since the previous tick to idle,
  and the window's elapsed time excludes it. Windows therefore span
  submissions: many short submissions jointly complete one window, an idle
  gap neither scores nor resets it, and the controller never learns that
  batches exist. `create` fences the first window before the workers start
  (born busy). Every scored window logs `source width window` (generation,
  width, rate, busy time, completions, exercised width, samples, next width,
  state).
- Pinned pre-growth happens in calibration, before any load: after
  selecting the block, `benchTransfer` grows every NUMA pool so it can lease
  `(32 + 1)` requests of `max(block, 16 MiB)` (`preallocated_source_width`,
  `preallocated_request_size`), clipped to the mapped ceiling with room for
  the feed reserves; 528 MiB for blocks up to 16 MiB. The rungs the
  controller climbs through never map a slab inside a load (146-230 ms of
  hipHostMalloc on MI300X, which at first sat inside the measured load when
  the growth ran at loader creation). `DirectLoader.create` only grows the
  remainder for larger requests (a 32 MiB HF profile with 8 MiB blocks adds
  528 MiB in about 90 ms on B70). The ready line logs `retained`, `pregrown`
  and `pregrowth_ms`.
- Worker tasks are spawned on demand (`WorkerPool`): the decision that opens
  the gates spawns as many workers as the lifecycle limit admits, and a
  raised width spawns more, up to the configured maximum. On one MI300X, 128
  persistent workers cost about 7% at a held width of 12 and made every rung
  measure slower (width 16: 21 GiB/s with 128 tasks, 36 GiB/s with 16); 13
  workers serve width 12.
- DMA width is fixed at eight per device by default after calibration work
  showed adaptive DMA width added substantial complexity and little load
  value. There is no global DMA parallelism cap.
- DMA event lifetime (`retire_events_early`, enabled): a ready callback
  hands its `EventContext` to the pipeline's intrusive `retired` stack
  under `metadata_mutex`, after its own `eventCompleted` (and any pump it
  ran) and before `block.complete()`; `pump` destroys the stack at the top
  of every iteration under the lock, so an event is destroyed by a later
  pump or by `retireBatch`, never inside its own callback. Live PJRT events
  are bounded by devices x 8 plus one pump batch instead of a submission's
  transfer count. Checked against the oneAPI plugin under sustained load
  with the playground's `ZML_LOAD_EVENT_RETIRE_CHECK` (PLAN.md task 9); the
  constant turns it off.

### DMA memory and calibration

- `DmaBlockPool` is a load-scoped view over platform-owned arenas. It supports
  blocking atomic multi-block acquisition, callback reference leases, a hard
  mapped-byte ceiling, demand growth, NUMA reserves, and augmenting-path
  affinity assignment. Matching is correctness logic: greedily assigning a
  replicated block can consume the only block usable by a later strict-local
  request. Construction is exclusively arena-provider backed, including tests;
  there is no alternate owned-slab mode or externally refreshed-arena path.
  Acquired block handles carry their NUMA node, so final callback release goes
  directly to the correct free list without scanning retained arenas.
- Calibration resources/settings belong to `Platform`. Conservative defaults
  work without calibration; `benchTransfer` atomically replaces them and its
  arenas are retained as the loader's initial pool. Platform state prevents
  calibration, loading, inspection, and teardown from borrowing the workspace
  concurrently.
- DMA configuration does not copy platform identity. It owns NUMA nodes in
  platform order plus block size, per-device width, and mapped budget; the
  stable `Platform` pointer is authoritative for device IDs and kind. Both
  default and calibrated settings reject empty, oversized, or heterogeneous
  platforms before allocating workspace.
- Current detector screens DMA blocks 2/4/8/16/32 MiB at width eight. Default
  screens require at least 2 ms and 32 completions. Borderline results use
  three alternating pairs at 25 ms/256 transfers. The 8% near-peak rule favors
  a smaller block. It tunes one representative device, warms every device
  allocator concurrently through `Platform.warmupDeviceAllocators(io)` (the
  only warm-up implementation), applies the uniform selected tuple, and grows
  the retained all-device working set. There is no decision-dead aggregate
  timing phase.
- Calibration code is specialized for that representative lane: a window
  returns one metric directly, screen candidates own three inline samples, and
  the report contains one measured recommendation. There are no lane slices,
  one-element result allocations, nullable candidate widths, or synthesized
  recommendations for devices that were not measured.
- Calibration reports one `dma_bench` summary line: platform, device kind,
  selected block and width, measured GiB/s, elapsed/calibration/allocator
  warm-up ms, retained mapped bytes and NUMA pool count. There is no timing
  decomposition, no per-window `dma_bench_sample` line, and no transfer
  latency accumulator; per-arena mapping cost is logged where each arena is
  mapped, and the loader's own `live loader ready` line reports pre-growth.
- Retained arenas are initial capacity, not the full permissible live set.
  Detection starts with one largest-candidate calibration ring, reuses it,
  grows after selection to the all-device working set, and permits bounded slab
  growth up to the mapped-memory ceiling. One `allocate(node, bytes)` is the
  only arena growth path - calibration ring, post-selection reserves,
  pre-grown working set and load-time demand - and it holds the mapped-ceiling
  check; `ensureLoadBlockReserves` and `ensureSourceWorkingSet` only compute
  per-pool block targets and grow independent nodes concurrently. Workspace
  validation, arena reserves, worker scratch, and adaptive pinned feasibility
  use the exact maximum
  coalesced-job bound `ceil(max_job_len / block_size)`; device or writer count
  does not inflate the blocks required by one source job.

## Latest DeepSeek result: why coalescing mattered

Fixture: DeepSeek-V4-Flash, 148.65 GiB, 69,187 selected tensors, 46 contiguous
checkpoint shards, warm one-GPU CUDA loads.

The tensor-local master planner made about 69,445 source operations, averaging
only about 2.19 MiB despite a 16 MiB request size. It paid tens of thousands of
avoidable positional-I/O submissions, claims, block acquisitions, and
completion transitions.

| implementation | median load | median epoch | source calls | DMA submissions | logical throughput |
|---|---:|---:|---:|---:|---:|
| master tensor-local | 6.946 s | not recorded | ~69,445 | 69,193 | 21.40 GiB/s |
| first coalesced, fixed-grid cuts | 4.041 s | 3.445 s | 9,524 | 78,665 | 36.78 GiB/s |
| tensor-aware cuts + batch enqueue + scratch | **3.958 s** | **3.376 s** | **9,524** | **69,572** | **37.55 GiB/s** |

Final load samples were 3.693, 3.958, and 4.005 s; selected source widths were
24, 32, and 24. Relative to master, median throughput improved about 75.5% and
source calls fell 86.3%. The older 16.818 s/8.84 GiB/s pre-coalescing branch
measurement demonstrates the original pathology but is not a clean master
comparison.

The first coalescer used a rigid 16 MiB grid. It achieved the minimum source
job count but cut through almost every boundary tensor, raising pieces/DMA to
78,665. Tensor-aware cuts kept 9,524 jobs and reduced pieces to 69,572. Only
385 pieces exceed tensor count because some tensors must span request limits;
current DMA is just 379 submissions above master's 69,193.

Pinned high-water across final runs was 528 MiB--1.02 GiB; mapped capacity was
576 MiB--1.06 GiB. It remains pool/gate bounded, although blocks can live
longer because multiple tensor DMA events share them.

## Durable performance evidence

Results are machine-, plugin-, topology-, model-, cache-, and host-load-
specific. Compare medians only on the same setup. Old `/tmp` paths are omitted
because they do not travel between machines.

### Source request size is backend-dependent

- One B70 oneAPI, warm 14.96 GiB Llama, fixed source width 12: 8/16/32 MiB
  reads produced about 27.05/24.21/21.33 GiB/s and pinned high-water
  96/192/384 MiB. Larger requests were worse on this local path.
- Same B70 at 32 MiB: widths 12/16/24/32/48/64 produced about
  21.33/20.69/18.90/17.33/15.05/13.16 GiB/s. Twelve was the local knee, not a
  universal concurrency law.
- One MI300X with 16 MiB DMA blocks: 16 MiB source reads took 0.442--0.443 s;
  32 MiB reads took 0.608--0.652 s because three 64 MiB slabs grew during the
  load. Subtracting 182--186 ms mapping made them equivalent. On four MI300X,
  both fit retained memory and measured 0.486 versus 0.488 s.
- S3Proxy at 10 ms and an artificial 1000 MiB/s *per-request* cap favored
  16 MiB at very high width: 16 MiB width 96 reached about 11.5 GiB/s, while
  32 MiB width 64 reached about 9.3 GiB/s. This fixture rewards concurrency
  unrealistically and must not determine production defaults.
- Real AWS plateaued near 950 MiB/s: with 16 MiB requests, widths 24--128 were
  within about 0.7% while latency and pinned memory rose almost linearly.
  Sixteen MiB/24 was the smallest screened tuple within 3% of peak. Thirty-two
  MiB reduced GET count 39% but did not improve throughput; 64 MiB regressed.

Conclusion: source request size, DMA block size, source width, and live pinned
workspace are distinct controls. Keep VFS-specific minima and measure changes;
do not restore a universal 32 MiB policy.

### DMA block and width evidence

- MI300X real one-device loads with 8/16/32 MiB DMA blocks measured
  23.84/24.90/25.43 GiB/s. Replicated eight-GPU Gemma was much more sensitive:
  16 MiB took 7.829 s versus 10.694 s at 8 MiB. Hence the block grid remains;
  a two-point or overly generous near-peak rule can choose a costly block.
- B70 local loads favored an 8 MiB request/block neighborhood. Reusing a
  MI300X 16 MiB preference on B70 cost about 10.5% in measured goodput.
- DMA width eight repeatedly won or tied on MI300X. Larger event widths mostly
  raised callback latency and memory. The four-B70 fair synthetic benchmark is
  the important global-cap exception: total cap four preserved roughly
  79.4--79.7 versus 79.9--80.0 GiB/s while reducing event latency from about
  1.225 ms to 0.152 ms.
- Shortening calibration screens from 10 ms/128 completions to 2 ms/32 cut an
  eight-MI300X calibration from 4.834 s to 0.956 s while retaining 16 MiB,
  width eight, and no cap. Borderline confirmation remains necessary because
  noisy short screens sometimes selected the wrong block size.

### Local copy bottleneck and NUMA

- On buffered local loads, profiles consistently place most CPU cycles in
  Linux `_copy_to_iter`: copying page-cache folios into anonymous DmaMapped
  memory. This is expected; DmaMap makes anonymous pages GPU-visible but does
  not make file-backed cache pages the transfer source.
- A MI300X slow run attributed 74.17% of cycles to `_copy_to_iter`. External
  CPU+memory binding once improved about 16--17 GiB/s to 23.6--26.9 GiB/s, but
  a later placement matrix did not reproduce a simple local-node rule. Only
  CPU-node-1 writing node-0 slabs showed a repeatable 8--10% loss; unbound was
  fastest in that matrix. The earlier causal NUMA claim was confounded by host
  load/cache state.
- Raw HIP at 32 MiB showed local and cross-NUMA H2D both near 49--50 GiB/s.
  Feeding two GPUs from one node-local source reached 48.43 GiB/s unique and
  96.86 GiB/s aggregate. Copying to the other socket first managed only
  5--10 GiB/s single-threaded. Do not duplicate replicated data through a CPU
  cross-node copy merely to make subsequent DMA local.
- Direct I/O avoids `_copy_to_iter` but trades warm-cache behavior for storage
  throughput and imposes alignment constraints. It was not established as a
  universal win; unaligned safetensor ranges may still select buffered I/O.

### ROCm pinned allocation and required XLA behavior

- `hipHostRegister` scales poorly with visible GPU count because KFD/IOMMU maps
  the whole range to every GPU, page by page. With eight MI300X GPUs it took
  about 1.18 s for 256 MiB, 6.3 s for 1 GiB, and 11.4 s for 2 GiB. Huge-page
  advice helped about 9%; pre-touching, registration flags, and splitting into
  many ranges did not remove the cost.
- `hipHostMalloc` was much faster: roughly 0.2/0.6--0.7/1.15 s for
  256 MiB/1 GiB/2 GiB. ZML therefore allocates ROCm arenas through standard
  PJRT `pinned_host` buffers, holds an external reference, obtains the opaque
  writable pointer, and keeps the PJRT buffer as sole owner. Never DmaMap or
  DmaUnmap this pointer; unregistering allocation-owned memory breaks the later
  `hipHostFree`.
- Select a representative device's `pinned_host` memory space for each NUMA
  node. CUDA and oneAPI retain DmaMap-owned arenas for now.
- XLA main already contains GPU pinned-range detection (`c3d1d50c0f`, with
  ROCm follow-up `e21e1f19a5`; audit main was `47149e4cbc`). No custom
  `PJRT_HostMemoryAllocator_Extension` is needed. Experimental extension
  commits were discarded. A plugin older than pinned-range detection may
  allocate quickly but silently stage every transfer, so verify the actual
  artifact rather than inferring correctness from allocation time.
- On one MI300X, the correct pinned-host path allocated a 256 MiB arena in
  99 ms, calibrated in 265 ms, selected 16 MiB, and measured 46.6 GiB/s. A
  stale plugin allocated similarly quickly but measured only 6.5 GiB/s.

### oneAPI pinned-range prerequisite

Older oneAPI PJRT staging logic did not recognize SYCL
`prepare_for_device_copy` imports as pinned because `sycl::get_pointer_type`
still returned unknown. The validated XLA fix overrides
`SyclExecutor::IsHostMemoryPinned` and checks both ends of the range with
`zexDriverGetHostPointerBaseAddress`, requiring the same imported base. This
recognizes slab subranges and rejects overrun. Revisit standard
`sycl_ext_oneapi_register_host_memory` after the toolchain supports it.

With staging present, concurrent host `memmove` dominated and more DMA credits
made performance worse. Bypassing staging for known DmaMapped inputs changed
one-B70 cap-2/cap-8 from 21.17/11.01 GiB/s to 26.86/26.90 GiB/s and removed
the userspace copy. The selected fixed artifact measured 26.83--26.90 GiB/s.

## Failure and rejected-approach ledger

### Correctness failures that define invariants

- **Multi-device sharded deadlock (2026-09-01):** destination-debt scheduling
  could claim a tensor tail before its predecessor. Tails filled all lifecycle
  credits while `transferReady` correctly waited for missing prefixes, so no
  DMA could start. Planning now emits a predecessor-safe order; atomic claims
  preserve that order. Pre-growing memory and ignoring the DMA global cap did
  not help and were reverted.
- **Coalescing boundary fan-out (2026-09-03):** rigid request-grid cuts reduced
  reads but increased DMA from 69,193 to 78,665. Tensor-aware feasible cuts
  restored 69,572 without increasing source jobs.
- **Pinned ownership:** replicas and multi-piece blocks require reference
  counts across every child event. Any failure path must abandon exactly the
  unpublished/unconsumed references and still drain the epoch. Never release a
  block merely because the source job returned.
- **Range correctness:** a `206` must cover the requested interval. A `200`
  that ignores Range may be handled by positioning/discarding; malformed
  partial responses must not silently load wrong bytes.

### Performance/control experiments not in the current design

- **Universal fixed 32 MiB reads:** simpler, but caused a measured 21% B70
  local regression and 4x pinned high-water versus 8 MiB. Superseded by one
  VFS-prepared profile per model load.
- **Joint adaptive request size + source width:** the controller coupled size
  probes to modeled width, handicapped larger sizes, drained/refilled between
  generations, and often exhausted finite loads before useful candidates. DMA
  block/width are now preflight-calibrated/fixed; source width remains adaptive;
  request size comes from the profile rather than runtime tuple search.
- **Source-order mid-read block yielding:** correct prototype and repeatably
  reduced leased high-water about 12--34% for multi-block requests, but speed
  was neutral within noise and normal local requests often equal one DMA block.
  It cannot shrink retained calibration arenas. Remote use would require a
  dynamic destination-buffer contract and suffix-only retries so yielded bytes
  are never overwritten. Removed/default-off rather than current policy.
- **Large remote streaming transaction:** a 128 MiB HTTP response rotating
  through two smaller DMA blocks is conceptually valid, but fixed-iovec
  positional reads lease all blocks up front. Implementing it needs the same
  dynamic buffer and resumable suffix contract; it is not present.
- **Adaptive DMA width:** added policy/state complexity but usually changed
  latency/memory more than throughput. Replaced by calibrated block size and
  fixed per-device width eight. The later global-cap experiment was also
  removed from the implementation; its measurements above remain historical.
- **Per-transfer SYCL `host_task`:** serialized host-task dispatch with DMA and
  cut B70 goodput about 58--60%. Reverted.
- **Exact-event callback experiments:** single-worker and parallel exact-event
  waits were neutral or worse; staging, not whole-queue wait, explained the
  large cap-dependent regression. Reverted.
- **Independent 2 MiB mapped allocations:** slower than 64 MiB slabs split into
  blocks and added registration/pool overhead. Retain slabs.
- **File sorting and scalar `pread`:** did not materially change local wall
  time. The dominant cost remained page-cache copying and pipeline pressure.
- **File-backed mmap/direct source and mmap+copy:** about 21.4 and 23.7 GiB/s
  in the B70 experiment, below `preadv` into DmaMapped blocks.
- **Hard CPU affinity:** pinning twelve B70 reader tasks to CPUs 0--11 collapsed
  throughput to 7.53 GiB/s; restricting the process also harmed startup. Any
  future NUMA policy must use measured per-node lanes, not a blind affinity.
- **Hard lane coupling:** making each reader await all DMA children improved a
  B70 diagnostic from about 26.9 to 29.5 GiB/s and reduced host pressure, but
  would idle high-latency sources. If revisited, represent completion-aware
  local pacing as policy rather than coupling the generic data plane.
- **Greedy NUMA assignment:** unsafe with strict-local plus replicated block
  mixtures. Keep augmenting-path matching.
- **Removing huge pages/pre-touching/changing HIP flags:** did not fix ROCm
  registration; the work is KFD/IOMMU mapping.
- **Custom PJRT host allocator extension:** wrong abstraction and unnecessary;
  standard `pinned_host` buffers solve ROCm ownership/locality.

## Historical benchmark anchors

These are retained to recognize regressions, not as universal targets.

| platform/workload | relevant result |
|---|---|
| one B70, local Llama 14.96 GiB | adaptive and fixed 12/8/2 MiB both ~27.0 GiB/s before later 8 MiB profile work |
| four B70, final post-deadlock policy | sharded 0.640 s / 23.36 GiB/s; replicated 1.147 s / ~52.2 GiB/s physical |
| real AWS, Llama 14.96 GiB | adaptive median ~946--948 MiB/s, zero retries/throttles; static path plateau ~950 MiB/s |
| eight MI300X, replicated Gemma 58.25 GiB logical | 16 MiB block 7.829 s versus 8 MiB 10.694 s with the older plugin baseline |
| four MI300X, local sharded Llama | 0.468--0.500 s, ~30--32 GiB/s depending on profile/calibration run |
| CUDA DeepSeek-V4-Flash 148.65 GiB | current coalesced median 3.958 s / 37.55 GiB/s versus master 6.946 s / 21.40 GiB/s |

Warm/cold state matters. Calibration time is normally excluded from loader
epoch throughput but included in process wall time. Host contention invalidated
several historical runs; isolated screening values should not replace repeated
same-host medians.

## Final validation state

- DeepSeek CUDA loads completed repeatedly. Focused loader tests separately
  exercised tensor shape/content correctness.
- VFS tests passed, including exact scatter, retries, and concurrency.
- Source-planner/final-record, three-phase adaptive-controller, immutable
  scheduler fairness/order, concurrent atomic claims, short-read,
  replication/sharding, NUMA-affinity, pool-scratch allocation failure and
  reuse, failure cleanup, and overlapping-batch tests passed. Public loader
  tests cover out-of-order handle awaits, a two-binding submission, `deinit`
  with open handles, the `Window`, a sticky read failure, and cumulative byte
  accounting.
- Zig formatting passes for `zml/io.zig`, every `zml/io/*.zig` module,
  `zml/mem.zig`, and `zml/safetensors.zig` with the repository's Bazel-managed
  toolchain. Buildifier check mode passes for `zml/BUILD.bazel`.
- `bazel build //examples/io:playground`, `bazel test //vfs:test`, and
  `bazel test //stdx:test` passed.
- `bazel test //zml:test --@zml//platforms:cuda=true
  --@zml//platforms:cpu=true` passes the loader-inclusive suite. The default
  configuration is still blocked at compilation by the
  existing missing `platforms/cuda/flashinfer_cutlass_moe` module mapping in
  `zml/moe/cutlass_flashinfer.zig`; it does not reach loader tests.
- Optimized playground builds had passed for CUDA, ROCm, and oneAPI during the
  preceding audits. Runtime coverage depends on hardware available per machine;
  oneAPI and remote-service behavior should be rechecked after relevant changes.
- ROCm-enabled aggregate tests were previously blocked before loader coverage
  by an existing CUDA flashinfer import in `zml/moe/cutlass_flashinfer.zig`.

## Third-pass design (target, 2026-09-03)

Decided after a seven-subsystem audit, four independent designs, three judges
and a synthesis (all agree on the core). The user's goals: shortest load with
the simplest code; the caller always controls how many `loadExecute`
submissions are in flight and in which order; keep up-front DMA calibration,
the per-VFS profile with its side channel, adaptive read width, reads and DMA
decoupled, low pinned memory.

### Why the epoch model must go

- `FairVectoredReadScheduler` holds one plan and `publish` rejects a second,
  so `loadExecute` must be `appendItems` + `await`. Laguna therefore pays 78
  full loader drains (39 sparse layers x 2 packs), each waiting for scheduler
  exhaustion, gate emptiness, a controller barrier that resolves on a 25 ms
  tick, an O(W^2) worker rendezvous and manager teardown, then runs the pack
  executable with the source pipeline idle, then restarts the width probe
  with the read gate closed. Epochs shorter than 100 ms can never be scored,
  so the controller never settles on that workload.
- Nothing in the planner, fair order, `transferReady`, gates or pool depends
  on one live plan; only `finishEpoch`'s free-the-plan rendezvous does.

### Target

- Public API (`zml/io.zig`): `Loader.load(Model, model, buffers) !Handle`;
  `Loader.loadExecute(bindings: []const Binding{tensor, output, exe}) !Handle`
  submitting all bindings' sources as ONE planned submission; `Handle.await()`
  waits for the submission's final DMA callbacks, then runs each executable
  on the awaiting task with `.wait = true` and frees its inputs;
  `Loader.awaitAll()`, `bytesLoaded()` (per-handle commit on success),
  `executeInputBytesPerDevice(exe)`; `zml.io.Window{budget_bytes,
  max_handles}` awaits the oldest handle before submitting the next. A window
  of one reproduces today's serialization; the loader has no memory policy
  and no executor thread.
- Direct backend: a strict FIFO of immutable batches; claim under the
  scheduler mutex (already taken every worker iteration; ~9.5k claims per
  DeepSeek load); `Batch.remaining` (sentinel + jobs) decremented at the
  `RequestContext` 1->0 transition and by `scheduler.fail` for unclaimed
  jobs; `done` event; per-plan preallocated request/block/event arrays
  (task 9) retired by the awaiting task under `metadata_mutex`, DMA events
  destroyed by the pump once their callback ran. Callback order rule:
  locals first, `eventCompleted`, the retired push, `block.complete()`
  last; `finishJobs` is the last access to batch memory.
- Unchanged: planner and tensor-aware cuts, fair predecessor-safe order per
  plan, two gates and lifecycle credit, pool with NUMA matching, calibration,
  per-tensor PJRT managers, VFS data plane, blind bootstrap for
  `high_latency`, fixed-width benchmark control.
- Later, separately measured: per-file incremental publish (done in task 6:
  the planner already grouped by file and reset predecessors per file, so
  per-file plans keep the same jobs and transfers), climb-and-hold width
  controller without gate drains and with a busy-time window clock, per-plan
  preallocated contexts and event retirement (done in task 9), VFS
  consolidation, calibration reporting cleanup, NUMA experiment (measurement
  only; deletion would be a follow-up).
- Rejected: an in-loader executor (`Execute` orders behind definition events
  so it would work, but caller-task execution keeps today's proven PJRT
  lifetime order and needs no thread); evaluating the controller only at
  read completion (loses backpressure sampling while workers sleep in
  retries); dropping `high_latency`; deleting NUMA before measuring.

### PJRT lifetime facts (header 0.113 and pjrt_client.h)

- Buffers from an async host-to-device manager may be passed to `Execute`
  immediately; execution orders behind the definition event.
- `PJRT_Buffer_Destroy` while an execution references the buffer is safe;
  device memory is freed when async operations complete. zml passes an empty
  `non_donatable_input_indices`, so inputs may be donated: never reuse them.
- Event destroy from another thread after the callback returned is current
  practice (`retireBatch`, `Buffer.await`, and since task 9 the pump for
  every callback-retired event); never inside the callback. Checked against
  the oneAPI plugin under sustained load (playground
  `ZML_LOAD_EVENT_RETIRE_CHECK`: 16k events per run destroyed right after
  their callback, 0 errors). Manager destroy before its transfers complete
  is undocumented: destroy only after completion, which per-batch retirement
  guarantees; events are destroyed before their manager either way.
- No blanket thread-safety statement; concurrent `Execute` on one executable
  is undocumented, so executions stay on one awaiting task.

### Baseline 2026-09-03 (commit 2f9cac2b, Llama-3.1-8B 14.96 GiB, one GPU, warm)

| host | load | GiB/s | DMA block | request | width | pinned high-water | calibration |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9985wx-5090x4 (CUDA, quiet host) | 0.606 / 0.609 s | 24.6 | 2 MiB | 8 MiB | 24 | 264 MiB | 751 ms |
| local B70 (oneAPI) | 0.794 / 0.758 s | 18.8-19.7 | 8 MiB | 8 MiB | 12 | 200 MiB | 809-891 ms |
| mi300 (ROCm) | 1.152 / 1.102 s | 13.0-13.6 | 16 MiB | 16 MiB | 24 | 784 MiB | 274-1109 ms |

- DMA calibration peaks were healthy on all three (CUDA 50.7 GiB/s at 2 MiB,
  oneAPI 48.8 at 8 MiB, ROCm 43.3 at 16 MiB), so ROCm's 13 GiB/s is
  source-side; the user reports a likely host problem, to be investigated
  later. ROCm allocator warm-up took 2.36 s.
- Page-cache read ceiling (`dd`, 512 MiB chunks, P parallel readers):
  CUDA host 14.1/24.7/31.6/55.7/62.1/60.7 GiB/s at P=1/2/4/8/16/32; local B70
  19.5/27.2/27.4/26.6/30.8/29.8; mi300 4.3/9.5/16.8/26.0/44.0/54.6. The
  loader reaches 41% of the CUDA ceiling and 66% of the B70 ceiling.
- The CUDA host is shared: with another user's job running (load average 16,
  GPU 0 busy) the same configuration measured 0.7 to 1.3 s. Check host state
  before every measurement; a width/block sweep taken under that load was
  discarded.
- NUMA topology: only mi300 is multi-node (two nodes, GPUs split 4/4); the
  CUDA and B70 hosts are single-node, so NUMA matching never engaged there.
- Laguna-XS-2.1 (local, 63 GB, 14 shards): 39 sparse layers x 256 experts;
  per expert down [2048,512], gate [512,2048], up [512,2048] at 2 MiB each,
  adjacent in file order; a layer's 768 expert tensors form one contiguous
  1.5 GiB run. A per-layer submission holding both packs coalesces into
  ~16 MiB jobs; separate down and gate_up submissions would read every third
  2 MiB tensor.
- DeepSeek-V4-Flash on mi300 (one MI300X, warm, 148.65 GiB, 69,187 tensors,
  46 shards): 8.264 / 8.204 s wall, epoch 7.438 / 7.631 s, 18.0 GiB/s,
  9,524 jobs, 69,572 transfers, planning 0.32 s, width 24, pinned high-water
  784 MiB. Same host loads Llama at only 13 GiB/s, so the per-byte cost is
  lower on the many-tensor model.
- HF remote (`hf://Qwen/Qwen3.5-9B`, 17.98 GiB, local B70 host, no token):
  22.35 / 21.02 s, 824 / 876 MiB/s, profile `hf`, 32 MiB requests, 577
  requests, zero retries or throttles, selected width 24 / 32, pinned
  high-water 1.53 GiB (width x 32 MiB requests; a remote load pins far more
  than a local one). Found and fixed on the way: `std.http.Client` returns
  HEAD responses before its redirect handling, so `HF.resolveDownloadUrl`
  failed on the Hub's 307; it now follows redirects itself (absolute or
  relative, credentials only to huggingface.co). The playground expects the
  `hf://owner/model` form.
- Pack instrument (playground, local B70, Llama, `ZML_LOAD_PACKS=64
  ZML_LOAD_PACK_WIDTH=16`, today's synchronous `loadExecute`): 14 packs of
  16 sources (13.0 GiB) load at 13.5-14.6 GiB/s in 0.89-0.97 s while the
  remaining 1.96 GiB bulk loads at 17.4-18.3 GiB/s; each pack epoch is under
  110 ms, never scored, and runs at the bootstrap width 12. Total wall
  1.005-1.086 s versus 0.77-0.79 s for the same bytes as one bulk load.
- CUDA host, quiet, tree after task 1 (2026-09-04): plain Llama 0.599 /
  0.603 / 0.607 / 0.612 s (one 1.235 s outlier on the first run after a
  sync), width 24, pinned high-water 264 MiB; pack instrument width 16:
  pack phase 0.644-0.650 s at 20.0-20.2 GiB/s, bulk remainder 1.96 GiB at
  20 GiB/s, total 0.750-0.757 s, content checks ok. DeepSeek-V4-Flash does
  NOT fit one 32 GB RTX 5090 (`ResourceExhausted` after ~28 GiB, the epoch
  failed cleanly and reported `successful=false`); DeepSeek measurements use
  the MI300 host (192 GB) from now on, and the CTX "CUDA DeepSeek" anchors
  predate this host configuration.
- MI300 host, quiet, tree after task 1 (2026-09-04): plain Llama 0.918 /
  0.919 / 0.984 s (15.2-16.3 GiB/s, better than the day before), width 24,
  pinned high-water 784 MiB; pack instrument width 16: pack phase
  0.805-0.859 s at 15.1-16.2 GiB/s, bulk remainder at 17-18 GiB/s, total
  0.957-1.024 s, content checks ok.
- Remote regression after task 2 (commit b0584f43, quiet hosts): CUDA plain
  0.611-0.620 s, packs width 16 pack phase 0.650-0.661 s; MI300 plain
  0.902-0.954 s, pack phase 0.821-0.906 s. Neutral within spread.
- Remote regression after task 3 (commit f55b8ea7, quiet hosts): CUDA plain
  0.586-0.604 s (24.8-25.5 GiB/s); packs width 16 window 1 total
  0.628-0.644 s (pack phase 0.546-0.561 s at 23.2-23.8 GiB/s), window 2
  total 0.593-0.612 s (pack phase 0.514-0.530 s at 24.5-25.3 GiB/s), i.e.
  packs plus bulk equal the plain load. MI300 plain 0.901-0.965 s (width 24,
  pinned high-water 784 MiB); window 1 total 0.745-0.808 s (pack phase
  0.615-0.681 s), window 2 total 0.677-0.689 s (pack phase 0.549-0.560 s at
  23.2-23.7 GiB/s), i.e. the pack workload beats the plain load by 25%.
  Four-GPU sharded packs at window 2 on MI300 complete in 1.04-1.09 s with
  correct contents (the 2026-09-01 deadlock family passes). DeepSeek on MI300
  7.255 / 7.603 s wall (19.6-20.5 GiB/s) versus 8.20-8.26 s on day one.
- MI300 width evidence (task 3 tree, plain Llama, one GPU): fixed width 12
  0.418 / 0.429 / 0.442 s (33.8-35.8 GiB/s, matching the CTX anchor for
  16 MiB reads); fixed 24 0.616-0.671 s with the first 64 MiB pinned slab
  growth costing 146 ms; adaptive 0.901-0.965 s selecting 24. The adaptive
  ramp (gate drains, probing 16/24/32, slab growth) costs more than the whole
  load on this host and settles on a worse width. This, not the host, is most
  of the "ROCm problem". A CUDA fixed-width sweep was discarded (host busy).
- After task 8 plus its follow-up (climb-and-hold controller without gate
  drains, on-demand worker pool, pinned working set mapped in calibration;
  commit 4a046807, quiet hosts, Llama one GPU):
  local B70 plain 0.636-0.644 s (23.4 GiB/s; day one 0.76-0.79); packs
  width 16 window 2 total 0.716-0.722 s.
  CUDA plain 0.502-0.548 s (27-30 GiB/s; day one 0.61), fixed 12
  0.488-0.500 s, fixed 16 0.456-0.477 s; packs window 1 total 0.506-0.551 s,
  window 2 total 0.477-0.511 s (pack phase 0.42-0.44 s at 29-31 GiB/s).
  MI300 plain 0.486-0.508 s (30 GiB/s; day one 1.10-1.15), fixed 12
  0.409-0.416 s; packs window 1 total 0.602-0.671 s, window 2 total
  0.537-0.594 s (pack phase 0.46-0.51 s at 25-28 GiB/s); DeepSeek
  7.091 / 7.195 s wall (day one 8.20-8.26). Every run: gate never closed,
  13 workers at width 12 growing with the width, pinned high-water
  136-272 MiB inside a 528 MiB working set mapped before the load.
- Worker-count evidence (MI300, task 8 tree, adaptive): 16 worker tasks
  0.424 s epoch, 24 tasks 0.476 s, 32 tasks 0.572 s, 128 tasks 0.613-0.675 s;
  with 128 tasks even width 16 measured 21 GiB/s versus 36 GiB/s with 16
  tasks. Hence the on-demand pool. The first measurement window is biased
  low by startup (lazy PJRT managers, file opens), so the climb usually
  visits one rung above the start before holding.
- Laguna-XS-2.1 through the migrated llmd on MI300 (device 0, warm): 40
  submissions (one bulk plus one two-pack submission per sparse layer),
  7,984 reads, 32,431 DMA pieces, 62.29 GiB in 12.19 s with a window of one
  and 12.21 s with a two-layer budget, 355-368 ms to first token, 64-72
  tokens/s. The comparison is VOID: calibration inside the server took
  4-7 s and chose 2 MiB blocks because the host had degraded (see next
  bullet), so the load was DMA-bound at 5.6 GiB/s regardless of the window.
- MI300 host degradation (2026-09-04, about 03:00 session time): the same
  tree that loaded Llama in 0.49-0.60 s two hours earlier took 3.05 s;
  calibration measured 6.9-7.9 GiB/s for every block size on both GPU 0 and
  GPU 1 (43 GiB/s at 16 MiB before) and took 3.5-5 s. This is the staged
  transfer signature (CTX: a stale plugin measured 6.5 GiB/s), not a loader
  change (CUDA and the local host were unaffected). Host state at the time:
  MemFree 21 GB of 2 TiB, page cache 1.84 TiB, Mlocked 27 MB. Left for the
  ROCm host investigation; redo the Laguna measurement afterwards.
- After task 6 (per-file incremental publish, identity fair order for one
  device; local B70, Llama, one GPU, 3 runs each): plain 0.635-0.651 s
  (23.0-23.6 GiB/s), the batch line `plans=4, planning_elapsed=0.001-0.002s,
  published=+0.001s, sealed=+0.002-0.003s, first_claim=+0.001s,
  first_read=+0.001s`, 1918 jobs / 2187 transfers unchanged; packs width 16
  window 2 total 0.701-0.726 s (pack phase 0.616-0.640 s), reads 1977,
  bulk batch 4 plans; two B70 sharded packs window 2: 0.891 s (task 3:
  0.900 s), pack checks ok. Neutral here by construction (Llama plans in
  1-2 ms); DeepSeek on MI300 is the measurement that can move, pending a
  trusted host (see the degradation bullet above).
- Fixtures: S3Proxy jar and a `lfm` bucket linking the Llama shards exist
  locally; no AWS credentials on this machine.

## Open work

The third pass is in progress; `PLAN.md` holds the checklist. Both earlier
simplification passes are complete. They removed remaining
planning/runtime genericity, consolidated epoch completion, specialized
representative-device calibration, narrowed the DMA pool, shared loader-front-
end preparation, and split the former monolithic IO module by responsibility.
Longer-term work still includes calibration caching, cross-platform 24/32 MiB
measurement, completion-aware local pacing, and any explicit
packed-device-buffer redesign needed to reduce DMA submission count below
roughly one per tensor.

- One-off, unexplained (2026-09-04, B70 `level_zero:1`, S3Proxy at 20 ms and
  200 MiB/s per request): an adaptive climb 32/48/64/96 -> 128 aborted in the
  oneAPI plugin with `host_to_device_transfer_manager.cc:342 Check failed:
  definition_events_[buffer_index]`, reached from `SetEventAsError` through
  the pump's `onReady` callback (a transfer error, then the pump's next piece
  into the same manager). Four further runs (ceiling 64, fixed 128 with 2 GiB
  pinned, the same adaptive command, 800 MiB/s) completed. If it recurs,
  make the pump stop submitting into a manager whose event errored and
  surface the plugin error instead of the CHECK.

## Suggested upstream decomposition

1. Exact safetensor positional scatter plus VFS Range/retry conversion and
   removal of backend `parallel_read`.
2. Plugin pin containing already-upstream pinned-range detection; ROCm arenas
   through standard PJRT `pinned_host` buffers.
3. Platform-owned DMA arenas, NUMA allocation, and `DmaBlockPool`, independent
   of calibration policy.
4. Model-wide coalescing planner/scheduler/pipeline and `load`/`loadInto`
   migration with conservative fixed DMA settings.
5. DMA block calibration and fixed per-device width eight.
6. Adaptive source width and aggregate VFS feedback.
7. `LoadProfile` and model-wrapper plumbing.

## Cross-machine handoff rules

- Treat commit IDs, branch divergence, plugin archives, available GPUs, model
  paths, warm cache, and background load as ephemeral. Inspect them anew.
- Do not rely on `/tmp` benchmark/perf artifacts being present elsewhere.
  Preserve user-owned recordings when they do exist; do not overwrite generic
  `perf.data` or `perf.data.old`.
- Verify that a ROCm/oneAPI plugin actually contains pinned-range recognition;
  successful compilation or fast allocation does not prove staging is absent.
- Do not build or modify an adjacent XLA checkout unless the current task
  explicitly requires it. The required pinned-range behavior was upstream at
  the last audit; local experimental allocator/completion patches were rejected.
- The adjacent production monorepo previously needed VFS registrations to use
  `registerBackend` so file/HTTP/HF/S3/GCS profiles and counters reached the
  loader. Re-check that integration rather than assuming a historical
  uncommitted edit still exists.
