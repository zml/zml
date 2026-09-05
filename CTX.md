# Direct DMA loader context

This is a compact, agent-oriented handoff. It records current behavior, durable
decisions, useful measurements, rejected approaches, and open work. It is not a
runbook. Re-check code, Git refs, available accelerators, and plugin artifacts
on each machine before relying on an old result.

Last consolidated: 2026-09-04 (fifth pass) at the end of the third pass (caller-controlled
concurrency), on commit `67464f3c`. `PLAN.md` is the sequential implementation
checklist; this file is the canonical description of the code after each
completed task. "Current design" describes the code as it exists; "Third-pass
design" records the target, the evidence gathered, and the day's baselines.
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
physical-byte charges to compute a deterministic fair job order per plan
(`fairOrder`), then discards the temporary queues and charges; for one
device the fair order is the identity (every job charges that device;
asserted by a test), so the planner keeps the planning order and skips the
queues. No job depends on another: every DMA piece is submitted as soon as
its block is read (fifth pass), so the request carrying a tensor's tail may
be planned, claimed and read before the tensor's earlier requests. While
those spans are available, the planner also emits the final item, block
index/offset, writer mask, destination offset, and length records. The
published plan owns source jobs physically arranged in final fair order and
their final transfer records; runtime tensor state does not own another
dispatch plan. Order indirection and remaining-work suffix arrays are
discarded. Physical source bytes are distinct from logical tensor
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
  submission's plans are claimed in file order. Reads and DMA complete in
  any order: PJRT makes a buffer ready once every transfer submitted to it
  has completed and one of them carried the last-transfer flag, so the pump
  flags the submission that completes the target's placement bytes
  (`Target.total` against `submitted_bytes`; the pieces partition the
  placement) and no piece ever waits for another (fifth pass). A batch that
  completes without an error must have closed every target, or `awaitBatch`
  fails the loader with `error.IncompleteTransfer` rather than leave a
  buffer that never becomes ready. Per-device ready queues are
  `std.Deque`s served in arrival order, so the oldest submission's pieces
  and the oldest requests' blocks complete first. There are no live device
  queues, debt counters, claimed bitmap, runtime order indirection, or
  suffix-metadata arrays.
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
  by 3% moves the width one rung up (or holds at the pinned clip). One rung
  that does not is tolerated and the climb carries on to the next
  (`stall_tolerance`, sixth pass); two in a row end it, and the controller
  holds at the lowest measured rung at or below the best within 3% of it.
  When that hold rung is the start rung it probes the rung below once and
  adopts it only if it beats the best rate by the same 3% -- never on
  retention, because a rung stepped down to inherits the wider rung's queued
  transfers and reads high. Holding: evidence changes nothing. A plain load
  therefore spends four or five windows away from its final width. No
  tail rule: a window that cannot complete before the load ends leaves the
  width in place. Metadata can cheaply clip feasibility, but cannot predict
  a source's latency/bandwidth saturation point, so no job-size-derived
  initial-width heuristic was added, and a wider start rung was measured and
  rejected (sixth pass). Confirmed on B70 (holds 12 or 16 at equal load
  time, 0.67 s against 0.65 s fixed 12); MI300 and CUDA confirmation
  pending.
- The adaptive climb stops at the widest rung whose lifecycle credits the
  pool already holds mapped (`retained - dma_stage`, reported as
  `width_ceiling`): 32 on gb300-2, 64 on the B70, 32 on the HF profile.
  Above it a scored window maps a new pinned slab, which the pre-growth
  above is meant to avoid; one such window on a GB300 read 20.8 GiB/s at
  width 48 with 136 completions against 400 in a normal window. A width the
  caller fixed is clipped only by feasibility.
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
  lifecycles. All workers compete for lifecycle capacity
  (`RequestGateLimits`): `min(feasible, max(retained, width + dma_stage))`,
  where `retained` is the pre-grown pinned capacity in requests and
  `dma_stage` the calibrated per-device DMA depth in requests, so the DMA
  stage keeps its depth whatever the read width (fourth pass); the read gate
  alone limits source calls. A request returns lifecycle credit only after
  all its DMA children finish.
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
  the gates spawns `min(lifecycle, width + 1)` workers (a worker hands its
  request to the DMA stage and claims the next, so credits beyond the read
  width need no workers), and a raised width spawns more, up to the
  configured maximum. Workers are never retired, but one whose index is
  beyond what the current width needs parks between jobs (fifth pass):
  after a rung steps down, the workers spawned for the wider rung would
  otherwise queue at the credit gate for the rest of the load and inflate
  the reported credit wait without moving a byte. On one MI300X, 128
  persistent workers cost about 7% at a held width of 12 and made every rung
  measure slower (width 16: 21 GiB/s with 128 tasks, 36 GiB/s with 16); 13
  workers serve width 12.
- DMA depth is fixed at eight blocks per device by default after calibration
  work showed adaptive DMA width added substantial complexity and little
  load value. The pump enforces it as a byte budget
  (`max_in_flight_per_device x block_size`) with a cap of 64 pieces in
  flight per device, so a block of small tensors keeps the calibrated bytes
  moving (fourth pass). There is no global DMA parallelism cap.
- DMA event lifetime (`retire_events_early`, enabled): a ready callback
  hands its `EventContext` to the pipeline's intrusive `retired` stack
  under `metadata_mutex`, after its own `eventCompleted` (and any pump it
  ran) and before `block.complete()`; `pump` destroys the stack at the top
  of every iteration under the lock, so an event is destroyed by a later
  pump or by `retireBatch`, never inside its own callback. Live PJRT events
  are bounded by devices x 64 plus one pump batch instead of a submission's
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
  DMA could start. Planning then emitted a predecessor-safe order that
  atomic claims preserved. Pre-growing memory and ignoring the DMA global
  cap did not help and were reverted. Fifth pass (2026-09-04): the prefix
  wait itself is gone. PJRT only requires the flagged call to be the last
  call into the buffer, not the tensor's last bytes, so the flag is set by
  submission count, no piece waits for another, the deadlock class cannot
  arise and the predecessor order was deleted.
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

- 2026-09-04, commit `67464f3c`: `bazel test --@zml//platforms:cpu=true
  --@zml//platforms:cuda=true //zml:test //vfs:test //stdx:test` passes
  (zml 244 passed / 3 unrelated skips; vfs 20; stdx 18); `zig fmt --check`
  passes for `zml/io.zig`, every `zml/io/*.zig`, `zml/mem.zig`,
  `zml/platform.zig`, `zml/safetensors.zig`, `vfs/*.zig` and the two example
  mains; no BUILD file changed; `//examples/llm:all`, `//examples/mnist:all`
  and `//examples/io:playground` build on the CPU platform and the playground
  builds in release for oneAPI, CUDA and ROCm (the remote runs build it).
- Loader-related test blocks 76 to 93: batch completion and retirement,
  FIFO order and open-batch head, concurrent claims across batches, handle
  awaits out of order, two-binding submissions, `Window` budgeting, sticky
  failure across handles, `deinit` with open handles, controller replays of
  the recorded B70 and flat curves, busy-time clock, gate-never-closed
  invariant, two-class backpressure, per-attempt request hook, throttle
  classification, Retry-After, preallocated context retirement.
- Playground `load` at the final tree: local B70 0.636-0.655 s plain and
  0.704-0.712 s for the 14-pack workload at window 2 (0.739-0.764 s at
  window 1); CUDA 0.513-0.524 s plain, 0.457-0.512 s packs at window 2;
  content checks pass on every host and configuration, including two-device
  sharded and four-MI300X sharded pack runs. MI300 numbers before its
  degradation are in the results table.
- Event retirement from the pump thread was accepted by all three plugins:
  oneAPI (16,384 events, two devices, 53.7 GiB/s), CUDA (2,048 events, 30.8
  GiB/s), ROCm (2,048 events, on the degraded host at 5.7 GiB/s), each with
  every fired event destroyed and zero errors.
- Remote fixtures: HF `hf://Qwen/Qwen3.5-9B` 19.6-19.7 s at 934-938 MiB/s
  with 577 requests and no retries; local S3Proxy runs complete with zero
  retries and throttles; a single 503 injection is covered by unit tests
  only (the proxy has no fault injection); real AWS not run (no credentials).
- The migrated monorepo (`loader-third-pass`, `d426dde4`) builds for oneAPI
  and ROCm; llmd serves Llama-3.1-8B on the local B70 (load 0.640 s, TTFT
  112 ms) and Laguna-XS-2.1 on MI300 (62.29 GiB, 40 submissions, TTFT
  355-368 ms, 64-72 tokens/s); its window comparison is void, see Open work.
- The default bazel configuration is still blocked at compilation by the
  pre-existing missing `platforms/cuda/flashinfer_cutlass_moe` mapping in
  `zml/moe/cutlass_flashinfer.zig`; it does not reach loader tests.

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
- `TransferRawDataToSubBuffer`'s `is_last_transfer` only closes the buffer to
  further calls: the buffer becomes ready when its in-flight transfer count
  reaches zero and the flag was seen, in any completion order
  (`xla/pjrt/host_to_device_transfer_manager.cc`, added by openxla/xla
  `8dfe2c4ff1` on 2025-04-28 and used by the GPU stream-executor client
  since `1b19ae012a` on 2025-10-22; the older
  `GpuAsyncHostToDeviceTransferManager` sequenced the definition event
  behind the flagged transfer on the one host-to-device stream, which also
  only needs the flagged call to be the last one). zml's XLA pin
  `41370d1124` (2026-07-02) contains both commits, and the shipped
  libpjrt_cuda (manual-2026-07-31), libpjrt_rocm (manual-2026-07-20) and
  libpjrt_oneapi (manual-2026-08-17) binaries carry
  `CommonAsyncHostToDeviceTransferManager` symbols and none of the old
  manager (`strings` check, 2026-09-04).
- The transfer's done event never carries an error: the manager's
  `on_done` has no status and the C API wrapper resolves the event's
  promise with an OK status (`pjrt_c_api_wrapper_impl.cc`, the
  `on_done_with_d2h_transfer` lambda), so `PJRT_Event_OnReady` always
  passes a null error. A copy that fails asynchronously errors the
  buffer's definition event, which surfaces when the buffer is first used
  (execute, host copy); the loader's `recordError` branch in the ready
  callback is dead on the shipped plugins.

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
- NUMA topology: mi300 (two nodes, GPUs split 4/4) and gb300-2 (two memory
  nodes, GPUs split 2/2) are both multi-node; the CUDA and B70 hosts are
  single-node, so NUMA matching never engaged there. See "Seventh pass".
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

### Third-pass results (2026-09-04, commit 67464f3c, warm, one GPU)

| workload | day one (2f9cac2b) | final | notes |
|---|---:|---:|---|
| local B70, Llama plain | 0.758-0.794 s | 0.636-0.655 s | held width 12, 13 workers, pinned high-water 136 MiB |
| local B70, 14 packs window 2 | 1.005-1.086 s (synchronous) | 0.704-0.712 s | pack phase 0.62 s at 21 GiB/s |
| CUDA 5090, Llama plain | 0.606-0.609 s | 0.513-0.524 s | 28.5-29.1 GiB/s |
| CUDA 5090, 14 packs window 2 | 0.750-0.757 s | 0.457-0.512 s | pack phase 0.40-0.44 s at 29-32 GiB/s |
| MI300X, Llama plain | 1.102-1.152 s | 0.486-0.508 s | measured before the host degraded; fixed 12 is 0.41 s |
| MI300X, 14 packs window 2 | 0.957-1.024 s | 0.537-0.594 s | before the degradation |
| MI300X, DeepSeek-V4-Flash | 8.20-8.26 s | 7.09-7.22 s | 9,524 jobs, 69,572 transfers, before the degradation |
| HF Qwen3.5-9B (local host) | 21.0-22.3 s | 19.6-19.7 s | 934-938 MiB/s, per-connection cap near 19 MiB/s |

Every final run: gate never closed (`gate_closed_ticks=0`), workers spawned
on demand, 528 MiB pinned working set mapped in calibration, no steady-state
allocation. Event retirement from the pump was accepted by the oneAPI plugin
(16,384 events, two devices, 53.7 GiB/s) and the CUDA plugin (2,048 events,
30.8 GiB/s); see the ROCm line in "Final validation state".

Size: production lines in `direct_loader.zig` 3,152 to 3,542 and `io.zig`
791 to 1,135 (batch FIFO, plans, handles, window, controller), calibration
1,816 to 1,667, VFS backends 4,612 to 3,370 with `range_read.zig` 174 to
347; tests 76 to 93 blocks (+790 lines). What shrank is the machinery: three
epoch flags, the single plan slot with its atomic cursor and worker
rendezvous, the controller epoch barrier, the three-phase controller with
its confirmation sub-machine and tail budget, the five-state measurement
union with two drain states, per-epoch reclamation, four copies of the
range/retry loop, per-sample calibration reporting, and 128 persistent
workers.

## Fourth pass: DMA-stage decoupling (2026-09-04, gb300-2)

Trigger: on `gb300-2` (144-core Grace, 34 NUMA nodes, four GB300, DMA
calibrates at ~180 GiB/s with 16 MiB blocks) the user's DeepSeek-V4-Flash
load printed a width-16 window that stayed open 3.8 s, then held 8.
Reproduced there with `CUDA_VISIBLE_DEVICES=0 bazel run --config=release
--@zml//platforms:cuda=true //examples/io:playground -- load
/var/models/deepseek-ai/DeepSeek-V4-Flash/` (user `benjamin`, checkout
`/home/benjamin/github/zml/zml`; GPU 0 is ours there).

Evidence (gb300-2, warm page cache, HEAD `8a5de654`):

- Per 16 MiB request a worker spent 9.2 ms in the claim stage and 3.0 ms
  reading; pinned-block acquisition, the read gate and the enqueue were
  under 10 us. The claim stage is the wait for a lifecycle credit, which a
  request holds until its last DMA callback. With `width + 1` credits the
  DMA stage held one request at a time.
- DeepSeek-V4-Flash has 69187 tensors: half are 256 KiB scales, the other
  half ~4 MiB weights (9524 requests, 69572 DMA pieces, 2.2 MiB mean). The
  fixed depth of 8 submissions per device therefore kept ~18 MiB in flight
  where calibration had measured 128 MiB.
- The pump is not the ceiling: 69572 submissions at 5.8 us each, 0.4 s of
  a 3.7 s load. Worker start is not a factor either (spawned tasks claimed
  within 1 ms).
- Environment sweep at fixed width 16: depth 8 -> 24.3 GiB/s (6.13 s);
  depth 32 -> 36.1 (4.12 s); depth 64 with 16 extra credits -> 43.8
  (3.39 s). The read gate held only 2 to 9 of 16 permits at any tick, so
  the window rule `exercised >= width` waited 0.3 to 4.7 s for a moment
  with 16 concurrent reads; dropping that rule alone scored the transition
  and held 12 (6.0 s), so the rule stays.
- Two rejected intermediate designs, both measured: credits equal to the
  whole retained capacity with workers spawned to the credits and the read
  permit taken before the pinned blocks (3.53-3.66 s, but 41 to 74 idle
  tasks per load, block waits counted as reads in flight, and a warm-up
  rule that fired on every load because the parked workers held every
  credit); and an unconditional warm-up window.

Change (commit after `8a5de654`):

- Lifecycle credits: `min(feasible, max(retained, width + dma_stage))`.
  `retained` is the pre-grown pinned capacity (`DmaBlockPool
  .retainedRequestWidth`: the 33-request source working set plus the
  calibrated DMA depth per device, now materialized at calibration; the
  smallest node under strict NUMA affinity). `dma_stage` is the calibrated
  in-flight bytes in requests (8 blocks per device), so the DMA stage keeps
  its depth at widths above the retained capacity. Workers stay at
  `width + 1`: a worker hands its request to the DMA stage and claims the
  next, so credits need no workers.
- The per-device DMA budget is `max_in_flight_per_device x block_size`
  bytes (8 blocks' worth, the depth calibration measured) with a cap of 64
  submissions in flight per device against tiny tensors; the documented
  bound on live PJRT events is now devices x 64 plus one pump batch.
- Calibration pre-growth targets `(32 + 1) requests + reserve` per pool and
  falls back to the source set alone when the reserves do not fit the
  mapped ceiling (logged).
- The width controller discards the load's first scoreable window when the
  lifecycle credits ran out during it: that window opened on an empty DMA
  stage and measured the fill burst (47 to 50 GiB/s where the steady rate
  was 42). Read-bound loads keep their first window. (Fifth pass: the
  trigger is now the workers' measured credit waiting against their read
  time within the window, since gate occupancy also touches the limit on a
  read-bound network load whenever a burst of reads completes together.)
- The pump reuses the ready index found by its budget pass instead of
  rescanning the chosen queue.
- The loader summary reports `credit_wait_ms_per_read`,
  `block_wait_ms_per_read` and `read_ms_per_read`: waits above the read
  time mean the load is DMA-completion bound and the read width is not the
  limiter.

Results (same day, same host state; HEAD `8a5de654` against this pass):

| host, workload | HEAD | this pass |
|---|---:|---:|
| gb300-2 GPU 0, DeepSeek 148.65 GiB (3 runs each) | 5.83-5.96 s (25 GiB/s) | 3.30-3.57 s (42-45 GiB/s) |
| gb300-2 GPU 0, fixed width 12 / 24 / 32 (intermediate tree) | - | 3.75 / 3.20-3.81 / 3.65 s |
| local B70, Llama plain, interleaved (2 each) | 678, 680 ms | 677, 676 ms |
| local B70, HF Qwen3.5-9B (1 run each) | 20.8 s | 21.9 s |
| MI300 (degraded, DMA 7.9 GiB/s), DeepSeek (1 run each, intermediate tree) | 27.9 s | 28.6 s |

On gb300-2 the load is DMA-completion bound at ~42 GiB/s at every read
width (fixed 12, 24 and 32 all land between 3.2 and 3.8 s); the remaining
ceiling is the per-piece cost of the PJRT/CUDA copy path for 4 MiB and
256 KiB pieces, not anything the loader schedules. Pinned memory rises by
the DMA reserve: 528 -> 656 MiB on one GB300, 592 MiB on the B70, 544 MiB
with 2 MiB blocks; an 8-GPU host with 16 MiB blocks pre-grows 1 GiB more
at platform init. The CUDA host could not be measured: both visible RTX
5090s held 29.8 GB of another user's `llmd` and the runs died in device
OOM. gb300-2 became busy after the final runs (GPU 0 held 255 GB of
another user's `llmd`, load average 28); a last run there died the same
way, so the `block_wait_ms_per_read` figure for DeepSeek is unrecorded.

Known limits recorded by the review of this pass: `active_events` remains
a redundant scalar beside the per-device arrays; the 64-piece cap is a
bound on event overhead, not a measured optimum, and for a block made of
tensors under `block / 8` it holds fewer bytes in flight than the budget
(DeepSeek's blocks mix 4 MiB and 256 KiB pieces, so there it matches);
on an 8-GPU host with 16 MiB blocks the pre-grown set plus reserves
(33 + 64 blocks) fills 1552 MiB of the 2 GiB mapped ceiling, and a 32 MiB
HF request there clips the pre-grown width to 31 with no headroom left
for growth. The review's claim that `waitForWork` can spin on an open
exhausted head was refuted: submissions seal before returning, so an open
head is always the last queued batch.

## Fifth pass: last transfer by bytes (2026-09-04)

Trigger: a Hugging Face load printed `source width warm-up window
discarded: generation=3, width=32, rate=1.01GiB/s`, and the B70 HF summary
showed `credit_wait_ms_per_read` of 194 to 408 against `read_ms_per_read` of
1200 to 1400 with DMA at 48 GiB/s. The first suspect was the tail rule: the
loader flagged the piece with the highest destination offset as PJRT's last
transfer and held it in the ready queue until every other byte of the
tensor had been submitted (`transferReady`), so the request carrying a tail
kept its lifecycle credit and one pinned block until the tensor's earlier
requests were read. Reading XLA (PJRT facts above) showed the positional
rule was never required: the contract is "last call", not "last bytes".

Removing it did not remove the credit waits, and the new per-request timers
showed why: on HF a request spends under 1 ms in the DMA stage
(`dma_stage_ms_per_read`) and 0.02 ms initializing its tensor state. A
per-tick trace of the gates found the read gate full for the whole load
(`reads=32/32`, then `48/48`, `64/64`) and credit waiters only for a few
ticks after each rung rise. The hundreds of milliseconds came from the
holding phase after a downward step: the workers spawned for the wider rung
(49 or 65) stayed alive, the credit limit fell back with the width (35 or
50), and the surplus queued at the credit gate for the rest of the load, so
every later claim carried a wait of up to a read time while the read gate
stayed full. Idle workers, not lost throughput, but a diagnostic that lied
and a warm-up proxy (`inUse >= limit`) that fired whenever a burst of reads
completed together against two spare credits.

Change (commits `d082a614`, `a9802595`, `41e73088`, `ab874827`,
`b6f23018`, `e9393f73`, `0af5c4d4`, `16098680`):

- The pump, the only submitter, flags the submission that completes the
  target's placement bytes (`Target.total` against `submitted_bytes`,
  `nextIsLast`/`noteSubmitted`/`fullySubmitted`); `transferReady` and the
  destination-prefix wait are gone, and no piece waits for another. A
  first version counted planned pieces per (item, writer) in the planner;
  the review replaced it with the byte total the target already carries
  (no new arrays, slices or validation, and a shape-derived oracle rather
  than the planner's own tally).
- The planner's predecessor order (`PlanningJob.predecessor`, the
  `fairOrder` constraint and the per-tensor order test) is deleted: no job
  depends on another. Per-device ready queues are `std.Deque`s.
- `awaitBatch` fails the loader with `error.IncompleteTransfer` when a
  batch completed without an error but a target did not receive its last
  transfer (`Batch.fullySubmitted`): a planner defect now fails the load
  instead of leaving a buffer that never becomes ready.
- `WorkerPool` parks workers the current width does not need (`wanted`,
  `admit`); `stopWorkers` wakes them and the pool refuses to spawn once it
  is stopping. Credits are unchanged.
- Failure path: PJRT's shared transfer manager drops a buffer's definition
  event once its last transfer was issued (accepted or not) or one of its
  transfers failed, and a later `SetBufferError` trips a CHECK and aborts
  the process (the oneAPI abort in "Open work" is the same class, on the
  pump's side). `Target.closed` (set by the pump before the flagged call)
  now keeps `awaitBatch` from marking such buffers; the outputs of a
  failed submission are undefined either way. An asynchronous transfer
  failure is invisible to the loader (PJRT fact below), so a buffer that
  failed that way and is then marked after a second, unrelated failure
  still aborts; a first version carried a per-target error flag set from
  the ready callback, which can never fire. `DirectLoader.submit` rejects
  empty sources itself; the DMA-stage timer ignores a request whose
  enqueue failed.
- The warm-up rule discards the first scoreable window when the requests
  completing in it spent at least as long in the DMA stage as reading
  (`dma_stage_ns` against `read_ns`, deltas since the generation opened),
  instead of when the lifecycle gate touched its limit. An intermediate
  version compared credit waiting with read time; the review showed it
  cannot fire at a narrow start rung on a GB300 (13 workers wait 2 ms per
  3 ms read while each request sits 7 ms in the stage), and on that tree
  two of five DeepSeek loads scored the inflated window and held 12.
- The summary reports `dma_stage_ms_per_read` (enqueue to last DMA
  callback) and `tensor_init_ms_per_read` (PJRT buffer and manager
  creation) beside the credit, block and read timers.
- Playground: `ZML_LOAD_CHECK=n` reads every n-th loaded tensor and the
  largest eligible one back to host and compares the bytes with the source
  (`load check: ok tensors_checked=...`). Each source file is opened once;
  a replicated buffer is compared on every replica (`Buffer.Shard.toHost`);
  a partitioned one is assembled with `toSliceAlloc`, whose element-stride
  placement of sub-byte shards is wrong, so sub-byte tensors partitioned
  over several devices are skipped (pre-existing `toSliceAlloc` defect,
  open). The `Loaded weights` summary excludes the check; a failing check
  stores its error and returns it after the buffers' block, since that
  block's `errdefer` and `defer` both release them.

Results (2026-09-04; baseline `c9fe01d4` measured the same day on the
same host; "count" is the intermediate tree `a9802595`, "final" is
`16098680`, `b6f23018` where noted):

| host, workload | baseline | this pass |
|---|---:|---:|
| gb300-2 GPU 0, DeepSeek 148.65 GiB, plain (bulk phase) | 2.96 s (climbed to 32), 3.40 s (held 8) | final: 3.17, 3.28, 3.31, 3.33 s (all held 12, first window discarded); `b6f23018`: 2.56 s (32), 3.33 s (12), 3.51 s (16, busy host) |
| gb300-2 GPU 0, DeepSeek read-back (`ZML_LOAD_CHECK=64`) | - | ok, 1082 tensors, `b6f23018` and final |
| local B70, Llama sharded, interleaved | 658, 674, 682 ms | count: 676, 681, 674 ms; final: 668 to 676 ms; full read-back ok 291/291 |
| local B70, HF Qwen3.5-9B (network, one run each unless noted) | 22.7, 22.0 s (climbed to 48) | count: 26.3, 24.0 s (held 32 after a probe at 24); final: 22.5, 23.9 s; no warm-up discard; sampled read-back ok |
| CUDA 9985wx (RTX 5090), Llama sharded, one GPU, interleaved | 475, 485, 457 ms | count: 443, 449, 463 ms; full read-back ok on one and two GPUs; final unmeasured (both GPUs held by another user's server) |
| MI300 (degraded: DMA 2.9 to 8.0 GiB/s, load 100 to 150) | void | void; read-back ok: two-GPU Llama 291/291 and DeepSeek 1082 sampled (count) |

Per-request timers on the final tree: gb300-2 DeepSeek credit wait 0.8 to
0.9 ms, block wait 0.001, read 3.1 to 3.3, DMA stage 6.3 to 6.6, tensor
init 0.06; B70 HF credit wait 4.5 ms (was 120 to 410), read 1200 to 1260,
DMA stage 1.0; B70 Llama credit wait 0, read 4.2, DMA stage 0.4.

Known limits recorded by the review of this pass:

- On gb300-2 the fastest loads are the ones whose controller reached 24 or
  32 (2.56 to 2.96 s at 55 to 58 GiB/s in the windows) and the slowest the
  ones that held 8 to 12 (3.2 to 3.5 s at 42 to 45). Which happens is
  decided by whether one window at 16 beats one window at 12 by 3%, and
  those windows differ by less than the noise (12: 42 to 45; 16: 40 to 48).
  The fourth-pass "hold at 8" is the same effect. A DMA-bound window
  (residency at or above read time) is the wrong evidence to end a climb
  on: a re-measure before stopping, or continuing while the stage stays
  the limiter, is the next controller change to measure.
- `Buffer.toSliceAlloc` places sub-byte shards by element stride (1 byte
  per element for `u4`, `f4e2m1`), so a 4-bit tensor partitioned over two
  or more devices is assembled at twice its offset: index out of bounds in
  safe builds. Pre-existing; the loader plans on the packed shape and is
  unaffected; the playground check skips such tensors.
- The pump can still submit into a manager whose event just errored
  (`host_to_device_transfer_manager.cc:342` CHECK, the oneAPI abort in
  "Open work"): the manager nulls the definition event under its own mutex
  before our callback runs, so no loader-side flag can close that window.
- `active_events` and `ready_entries` are derivable from the per-device
  arrays and the deque lengths; kept as assert witnesses.
- The load check compares every replica, but `checkPacks` keeps its own
  read-back loop (three packs, `source.reader`); a shared helper is
  possible once both need the same open strategy.
- Credit waiting is undercounted by the waits of workers that lose the
  claim race after `waitForWork` (pre-existing).

## Sixth pass: width detection on a flat plateau (2026-09-04)

Trigger: the fifth pass left "the climb on gb300-2 is the biggest lever
left" open, on the reading that loads reaching 24 or 32 finish in 2.6 to
3.0 s while loads holding 8 to 16 take 3.2 to 3.5. A fixed-width sweep
measured the curve instead of inferring it, and it is flatter than that.

DeepSeek-V4-Flash 148.65 GiB, gb300-2 GPU 0, `ZML_LOAD_FIXED_READ_PARALLELISM`,
three interleaved rounds per rung:

| fixed width | GiB/s per run | mean | load |
|---|---|---:|---:|
| 8 | 35.3, 39.1, 36.4 | 36.9 | 4.03 s |
| 12 | 45.2, 43.8, 45.3 | 44.8 | 3.32 s |
| 16 | 50.3, 42.1, 47.9 | 46.7 | 3.20 s |
| 24 | 44.8, 49.2, 50.5 | 48.2 | 3.09 s |
| 32 | 49.1, 50.5, 46.8 | 48.8 | 3.05 s |
| 48 | 40.8, 50.1, 51.0 | 47.3 | 3.18 s |

Everything from 12 to 48 is one plateau 8% wide; only 8 is a real cliff, 20%
below it. The whole prize a width controller can win on this host is 0.27 s
of a 3.3 s load, and the cost of the one bad rung is 0.7 s.

Why the controller cannot win the 8%: a throwaway patch kept the runtime
measuring while holding, so a fixed-width load logs every window. Inside one
load at width 12 the windows read 42.6 39.7 42.2 34.0 32.3 27.6 40.1 45.6
48.2 ... 53.1 54.7 54.7 GiB/s; at width 24 either 47 47 44 44 then 55 to 66,
or 58 57 57 59 60 59 55 then 43 to 50. The within-load spread is 13 to 15%
and drifts with the file: DeepSeek interleaves 256 KiB scales with 4 MiB
weights, so consecutive 5 GiB stretches carry different DMA piece counts and
sustain different rates. One 120 ms window at rung A and the next at rung B
measure different parts of the file, and that, not the width, decides which
looks faster. Six baseline loads agree: the first scored window at 12 read
37.8, 53.1, 39.3, 52.8, 40.7 and 40.5 GiB/s against a sustained 44.8, and
half of those are above the best rate any rung sustains. The load is 26
windows long and the controller decides in the first four.

So this pass stops chasing the peak and bounds the downside.

Change (commit COMMIT_PLACEHOLDER):

- The climb tolerates one rung that fails the 3% test and stops on two in a
  row (`stall_tolerance`). The rung above a stall is compared with the same
  best rate, so a genuinely declining curve still stops, one window later;
  the B70 replay reaches the same rung in four windows instead of three.
- The downward probe below the start rung is adopted only when it beats the
  best rate. In the baseline set a probe at 8 read 41.07 GiB/s right after a
  window at 16, against 36.9 sustained, and the 0.97 retention rule held the
  whole load at 8 (3.70 s).
- `isBorderline`/`borderline_used` are deleted. With the probe gated on
  improvement, a hold rung can only be borderline if the climb's 3%
  improvement and the 0.95-to-0.99 retention band overlap, which they do
  over a 0.1%-wide slice of retention. The mechanism could no longer fire
  outside its own unit test.
- The adaptive climb stops at the growth-free width (`width_ceiling`, see
  "Scheduling and concurrency"). The stall tolerance would otherwise push
  the climb into rungs that map pinned slabs mid-load: runs before this clip
  ended at 48, 64 and 96.

Results, gb300-2 GPU 0, DeepSeek 148.65 GiB, seven interleaved pairs of the
baseline (`fb8ccf55`) and this tree on one host state (bulk phase):

| pair | baseline | this pass | selected width |
|---|---:|---:|---|
| 1 | 4.150 s | 3.184 s | 8 -> 12 |
| 2 | 3.347 | 3.250 | 12 -> 24 |
| 3 | 3.703 | 3.408 | 12 -> 12 |
| 4 | 3.198 | 3.204 | 12 -> 24 |
| 5 | 3.773 | 3.393 | 8 -> 24 |
| 6 | 3.452 | 2.861 | 24 -> 24 |
| 7 | 3.728 | 3.249 | 8 -> 32 |
| mean | 3.622 | 3.221 | |

Six of seven pairs improve, mean -0.40 s (-11%), paired t = -3.2 over seven
differences. The worst load falls from 4.15 s to 3.41 s and no run settles
at 8; the baseline settled there in three of seven. Other hosts:

| host, workload | baseline | this pass |
|---|---:|---:|
| local B70, Llama sharded, `ZML_LOAD_CHECK=1` | 668 to 676 ms | 666, 680, 681 ms; holds 12 or 16; read-back ok 291/291 |
| local B70, HF Qwen3.5-9B (network) | 22.5, 23.9 s | 22.1, 20.6 s; `width_ceiling=32`, holds 32 |
| gb300-2, DeepSeek `ZML_LOAD_CHECK=64` | ok | ok, 1082 tensors, 3.243 s at width 24 |

Rejected in this pass, each measured:

- A wider start rung. Four interleaved rounds of adaptive loads at
  `ZML_LOAD_READ_INITIAL_PARALLELISM` 12, 24 and 32 averaged 3.37, 3.43 and
  3.23 s. The first window at any rung improves on nothing, so a higher
  start only climbs higher: those runs ended at 48, 64 and 96.
- Climbing while credits are not the limiter (`credit_wait_ms_per_read`
  below `read_ms_per_read`). It selects 32 on gb300-2, the optimum, but the
  B70 shows zero credit wait through width 24 and 2.0 to 2.3 ms at 32, so
  the rule climbs past the B70's peak (23.0 GiB/s at 12 to 16) into 21.9 at
  32 and above.
- Longer windows. The within-load drift is not noise that averages out in a
  few hundred milliseconds; reaching a 5% standard error needs about a
  second of busy time, a third of a DeepSeek load.

Known limits:

- The controller still settles at 12 in some gb300-2 loads and at 24 or 32
  in the rest, so the mean stays about 5% above the fixed-32 oracle
  (3.22 s against 3.05). Closing that needs several samples per rung, which
  on this workload means measuring for the whole load and revising the width
  instead of holding after four windows. That is a different controller, not
  a tuning of this one.
- The B70 sweep at 8 MiB requests (19.5, 18.4, 23.0, 23.0, 22.5, 21.9,
  21.9 GiB/s at widths 4 to 48) is reproducible to 0.1% between runs, while
  gb300-2 spreads 10 to 15% at every rung. The confidence a rate comparison
  deserves is a property of the host, and the controller does not measure it.

## Seventh pass: does NUMA placement matter? (2026-09-05)

Trigger: the NUMA placement experiment (task 12) had never been run on a
healthy multi-node host. Both bench hosts turn out to be two-node for GPUs,
which corrects an earlier note in this file:

- `mi300`: 8 MI300X, four on node 0 (`1b,3d,4e,5f`), four on node 1
  (`9d,bd,cd,dd`); 2 x 1 TiB.
- `gb300-2`: 4 GB300, two on node 0 (`0008,0009`), two on node 1
  (`0018,0019`); 2 x 490 GB. The other 32 NUMA nodes carry no CPU and no
  host memory (they are the device-coherent HBM nodes), so only 0 and 1 can
  ever back a DmaMapped arena.

`ZML_DMA_BENCH_NUMA_OFF=1` (new `Options.disable_numa_pools`) forces the
single shared unbound pool that a single-node host already gets;
`ZML_DMA_BENCH_NUMA_NODES=...` (already existed for `dma-bench`, now also
honoured by `load`) forces every device onto one node. Four arms: `local`
(default, one mbind-ed pool per device node), `off`, `node0`, `node1`.

### The MI300 host was not degraded, its plugin was stale

The released ROCm artifact (`manual-2026-07-20T15-30-00Z`) contains zero
occurrences of `IsHostMemoryPinned`; XLA's GPU pinned-range detection is
absent, so every DmaMapped transfer is staged. That is the whole of the
"MI300 degradation" recorded on 2026-09-04: 4.5 to 7.6 GiB/s at any block
size, any device count, any tree. Pointing `platforms/rocm/rocm.bzl` at a
locally built `libpjrt_c_api_gpu_plugin.so` (openxla/xla, 2026-09-03, 3
occurrences) restored 44 to 47 GiB/s immediately. Check the symbol before
blaming the host.

### Placement is worth 1.6x on GB300 and nothing on MI300X

Synthetic H2D only (`dma-bench`, 16 MiB blocks, one visible device, pinned
arena forced onto each node in turn). This measures the representative
device's link alone, so it is a clean locality probe:

| device (home node) | pinned on node 0 | pinned on node 1 |
| --- | --- | --- |
| GB300 0 (node 0) | **175.8** | 109.9 |
| GB300 1 (node 0) | **176.7** | 109.5 |
| GB300 2 (node 1) | 110.8 | **179.5** |
| GB300 3 (node 1) | 111.2 | **183.8** |
| MI300X 0 (node 0) | 46.0 | 46.7 |
| MI300X 3 (node 0) | 46.4 | 43.6 |
| MI300X 4 (node 1) | 46.0 | 47.2 |
| MI300X 7 (node 1) | 46.0 | 44.2 |

Remote costs 38% of H2D bandwidth on GB300, symmetrically in both
directions, and is free on MI300X. This is an interconnect fact, not a host
state: each GB300's NVLink-C2C lands on its own Grace socket, so a remote
arena crosses the inter-socket link, while an MI300X sits behind a PCIe root
complex whose ~50 GiB/s is below the cross-socket cost either way. It
confirms and explains the older raw-HIP observation (local and cross-NUMA
H2D both near 49--50 GiB/s on MI300X).

Device count does not change it: 1, 2 (same node), 2 (split) and 4 GB300
measure 176.2 / 176.5 / 176.3 / 179.3 local against 109.7 / 109.8 / 110.1 /
111.2 unbound.

### The loader does not benefit, because it is read-bound

DeepSeek-V4-Flash replicated across all four GB300 (148.65 GiB to each
device, so 594 GiB of DMA), 16 MiB blocks, five interleaved repetitions.
Load seconds, mean of five:

| arm | fixed width 32 | adaptive |
| --- | --- | --- |
| local | 5.48 | 5.41 |
| off | 5.40 | 5.38 |
| node0 | 5.89 | 5.70 |
| node1 | 5.58 | 5.24 |

`local` and `off` are indistinguishable, and the within-arm spread (`off`
ranged 5.03 to 5.96) is larger than any between-arm gap. The reason is
arithmetic: 148.65 GiB per device in 5.4 s is 27.5 GiB/s per link, a quarter
of even the remote 110 GiB/s ceiling. The DMA link is not the constraint;
the page-cache read path is. Only `node0`, which forces all four links and
every reader copy onto one node's memory controller, is repeatably worse,
and by about 5%.

So the 1.6x is real headroom, not realised throughput. It would begin to
matter only if the source could feed a GB300 faster than 110 GiB/s per
device, which no file source here does.

### Llama on gb300-2 inverts the recommendation

Llama-3.1-8B replicated on four GB300 is the large-tensor case and drives
almost twice DeepSeek's per-link DMA rate (46 GiB/s against 26.5). On an idle
host it separates the arms cleanly, seven interleaved repetitions, spread
under 2%:

| arm | mean load | against best |
| --- | --- | --- |
| off | 307.1 ms | -- |
| node1 | 306.4 ms | -- |
| local | 321.7 ms | +4.8% |
| node0 | 413.1 ms | +34% |

Node-local placement is a small **loss**, and `node0` against `node1` is the
decisive pair: both put all four links on one node, so pure GPU locality
would make them equal. They differ by 34%. The file's page cache sits on
node 0, so binding the pinned blocks there makes one memory controller serve
the page-cache reads and all four DMA engines at once. `local` is mildly bad
for the same reason -- it puts half the blocks on the busy node.

The loader's constraint is therefore host memory bandwidth on the node
holding the page cache, not the GPU link. The right placement rule is "away
from the page cache", which is the opposite of what strict affinity does, and
which no static `numa_node` attribute can express: it depends on where the
file was read, not where the device is.

### DeepSeek replicated is bound by submission count

Replicated DeepSeek on gb300-2 scales badly with device count, and it is not
the read path (`reads` stays at 9,524 and `read_ms_per_read` is flat):

| devices | GiB/s | read ms/read | dma_stage ms/read | DMA submissions |
| --- | --- | --- | --- | --- |
| 1 | 47.6 / 48.2 | 5.3 / 2.7 | 6.2 / 9.9 | 69,572 |
| 2 | 42.6 / 41.5 | 5.7 / 2.1 | 8.4 / 15.6 | 139,144 |
| 4 | 27.5 / 25.6 | 4.9 / 3.9 | 27.9 / 32.3 | 278,288 |

At four devices each link carries 27.5 GiB/s against a measured 176: 16% of
capacity, so bandwidth is not the limit either. The cost is the 278,288
separate H2D submissions (69,572 tensor pieces x 4 devices; reads coalesce
7.26:1 but DMA submissions do not coalesce at all), about 51,500 per second
on transfers averaging 2.24 MiB, which take ~13 us each at link speed.
Submission overhead is on par with transfer time.

Deepening the DMA stage recovers part of it. Six interleaved pairs, 16 MiB
blocks, `max_in_flight_per_device` 8 against 32:

| depth | mean | pinned high-water |
| --- | --- | --- |
| 8 | 5.536 s | 0.9 to 1.0 GiB |
| 32 | **5.024 s** (-9.2%, 6/6 wins) | 2.36 to 2.50 GiB |

A real 9% for 2.5x the pinned working set, which trades directly against the
"keep pinned host memory low" goal; 32 MiB blocks gain nothing at any depth.

What the depth buys is queue length, not DMA parallelism. In the same runs
`lifecycle_credits` went 49 -> 140 while `dma_stage_ms_per_read` went 17-26
ms -> 56-76 ms and `read_ms_per_read` did not move: in-flight x2.9, latency
x2.4, throughput x1.17. That is a deeper queue smoothing bursts ahead of a
serial submission path (~50-60k submissions/s), and the per-device gate was
already the 64-piece cap at depth 8 (128 MiB / 2.24 MiB). The single sweep
put depth 16 at ~80% of the gain for 2x pinned instead of 4x. Run-to-run
spread inside one arm (4.62 to 5.38 s at depth 32) is larger than one rung's
effect, so an on-the-fly controller would chase noise, and growing the stage
mid-load maps pinned slabs inside a scored window. If the depth is ever
derived, derive it once, on the client, from the safetensors header.

The count cannot be coalesced away: a submission is `transferData` on one
tensor's own transfer manager, so a host range holding several tensors still
needs one call per destination buffer. DeepSeek-V4-Flash has 69,187 tensors
and the plan has 69,572 pieces (block straddling adds 0.6%); submissions are
tensors x devices, and under one buffer per tensor that is the floor. The
distribution is bimodal, not "2.2 MiB average": 34,759 tensors (50.2%) are
under 1 MiB and carry 5.6% of the bytes, the median is 0.25 MiB. Half the
submissions are per-call overhead moving nothing.

### The submission ceiling is the loader's pump, not the driver

`dma-bench` at small blocks (one device, 200 ms windows, three repetitions)
gives the per-device engine ceiling; depth 8 and 32 are identical, so it is
a rate, not a latency:

| block | GiB/s | submissions/s | us per submission |
| --- | --- | --- | --- |
| 1 MiB | 58-65 | 59,000-66,000 | ~16 |
| 2 MiB | 104-107 | 53,000-55,000 | ~19 |
| 4 MiB | 142-153 | 36,000-39,000 | ~26 |
| 16 MiB | 183-184 | 11,700 | ~85 (link-bound) |

Note that the calibration measures **one device at a time** (`tuneDevice`
creates a cohort per device and `runBenchmarkWindow` runs one cohort), so
its `measured_gib_s` is never an aggregate. Four concurrent single-device
processes at 1 MiB reach 161,000 submissions/s and 640 GiB/s aggregate at
16 MiB (150-170 per device): neither the driver nor the engines are shared.

The new `dma-conc` subcommand of `//examples/io` drives every device at once
from one process (`depth` synchronous slots per device) and can serialise
the submit call behind one mutex, which is what the loader's single pump
thread does. gb300-2, 500 ms windows, three repetitions, `numactl -m 0`
for the last row:

| devices | block | submit | GiB/s | submissions/s |
| --- | --- | --- | --- | --- |
| 1 | 1 MiB | parallel | 46-55 | 47,000-57,000 |
| 4 | 1 MiB | parallel | 170-180 | **174,000-184,000** |
| 4 | 1 MiB | serialised | 88-125 | 90,000-128,000 |
| 4 | 1 MiB | parallel, node 0 | 199 | **203,000** |
| 4 | 16 MiB | parallel | 316-385 | 20,000-25,000 |

The loader on DeepSeek replicated submits 51,000-55,000/s across four
devices: below one device's engine alone, 3.5x below what one process
reaches with per-device submitters, and half of what a mutex-serialised
submitter reaches. So the ceiling is `VectoredLoadPipeline.pump` -- one
thread, one `metadata_mutex`, round-robin over devices, and roughly half of
its per-submission cost outside the PJRT call. The DMA stage depth was a 9%
patch on that; the lever is per-device pumps (the queues, active bytes and
piece counts are already per device, the mutex and the round-robin are not).
That lever is applied below: -13.5%, not the -40% this paragraph hoped for,
because the next ceiling is the engine's rate for the loader's own traffic.

Two side results from `dma-conc`: pinned memory it allocates without
placement landed on node 1 (`membind=1` reproduces 112 GiB/s on device 0,
`membind=0` gives 185), the same 1.6x link penalty as the seventh-pass
sweep; and one process can push 385 GiB/s of H2D into four GB300 with
unplaced memory, so the host is nowhere near limiting a file source.

Packing small tensors into one device buffer would cut the count itself,
but that is a `Buffer` model change rather than a loader one.

### Per-device pumps: -13.5%, and where the rest of the gap is

`VectoredLoadPipeline` now has one `DevicePump` per device -- its own mutex,
ready queue, in-flight bytes and pieces, retired-event stack -- instead of
one `metadata_mutex`, one `pumping` flag and a round-robin over devices.
`enqueueBlocks` and `retireBatch` take every pump's mutex in device order (a
request is still queued all-or-nothing); a completion, a pump and
`abortReady` hold one. `Plan.events_used` is an atomic. Five interleaved
pairs, DeepSeek replicated on four GB300, depth 8, 16 MiB blocks:

| arm | mean | runs | dma_stage ms/read | credit wait ms/read |
| --- | --- | --- | --- | --- |
| one pump | 5.41 s | 5.34 5.51 5.47 5.52 5.21 | 14-30 | 1.4-12.3 |
| per-device pumps | **4.68 s** (-13.5%, 5/5) | 4.62 4.66 4.52 5.05 4.53 | 9-10.5 | 1.2-1.4 |

Pinned high-water is unchanged or lower (896 MiB). Llama replicated on the
same host is at parity (320-324 ms against 321-325, four pairs, read-back
check `ZML_LOAD_CHECK=64` clean). Eight MI300X, Llama replicated, eight
pairs on a host at load 30: both arms bimodal between 0.88 and 1.49 s, means
1.19 against 1.28 -- not distinguishable from the host noise, and not a
clean parity either; worth one idle-host repeat.

Two variants were measured and rejected, both recorded in the `DevicePump`
doc comment. A dedicated task per device woken by completions was slower
with a wake per completion (5.0-5.4 s, a futex round trip per piece) and no
better with hysteresis (wake at half the budget: 4.4-5.1 s). Concurrent
submitters per device (every completion and enqueue submits, no exclusive
pump) measured the same (4.56-4.58 s) and is **incorrect**: a tensor's
pieces for a device are flagged last by `Target.nextIsLast` in submission
order, and two threads submitting the same tensor left targets unclosed --
`IncompleteTransfer`, 0 bytes loaded, in two of four Llama runs (DeepSeek's
single-piece tensors never hit it). One pump per device is a correctness
requirement.

Where the remaining gap to the read bound (3.1 s on one device) is, from
the loader's new summary fields (`dma_submit_us_per_piece`,
`dma_piece_latency_ms`, `pump_stops_empty/full`) and from `dma-conc` modes
added to reproduce the loader's conditions one at a time:

- The load is not CPU-bound: mid-load the process uses ~1.3 cores, all of
  it four reader threads in page-cache copies. The DMA path shows no CPU:
  `submitTransfer` *blocks* 16-28 us in the driver with four devices (5-12
  on one), and `dma-conc` measures the same 28 us per call once four devices
  each hold ~32 pieces in flight (5 us at depth 8). `dma-conc` only beats
  it with 8-32 blocked submitters per device.
- With per-device pumps each device runs ~15.5k pieces/s, 33 GiB/s, with a
  submit-to-callback latency of 2.0-2.3 ms at ~31 pieces in flight; the
  pumps stop mostly for lack of room. `dma-conc` with the loader's traffic
  (half 256 KiB, half 4 MiB, a fresh buffer per transfer flagged last,
  four devices) does ~32k pieces/s and 65-70 GiB/s per device: the loader
  was at half the engine's rate for its own traffic.
- Reproduced one condition at a time in `dma-conc` (4 devices, depth 32):
  source footprint 32 MiB to 1.5 GiB, no effect; a source shared by all
  four engines, no effect; callback-driven resubmission, no effect;
  **source misalignment -13%** (the files have no 64-byte-aligned tensor:
  16-byte at best, header base 400 mod 4096, so every DMA source is
  cache-line misaligned; 280 -> 244 GiB/s, 185 -> 162 on the plain link);
  **concurrent CPU writes into the rings at the loader's read rate -10%**
  (35 GB/s; -22% at 100 GB/s); **unplaced memory -23%** (194 against 252
  on node 0 and 272 on node 1, and the driver call rises to 60 us).
  Stacked, those put `dma-conc` at ~42-47 GiB/s per device against the
  loader's 33: the rest, ~1.3x, is unattributed and not in the pump.
- Neither process gets huge pages: THP is `madvise` and `AnonHugePages` is
  0 for the loader's arenas (`dma_map`) and for `dma-conc`'s advised source.
- NUMA placement matters on the loader again now that the pump is not the
  ceiling: `off` 4.24/4.42 s, `node1` 4.45/4.19, `local` (the current
  default when devices report `numa_node`) 4.55/5.14, `node0` 5.42/5.42.
  The seventh-pass recommendation stands and is now measurable: do not
  derive strict affinity from the attribute; steer away from the page-cache
  node if anything.

`//examples/io dma-conc` (`ZML_DMA_CONC_*`): drives every device at once
from one process, `depth` synchronous slots per device; `SERIAL_SUBMIT`
(one mutex round the submit), `CALLBACK` (resubmit from the ready callback),
`BUFFERS=reuse|fresh|prebuilt`, `SHARED_SOURCE`, `SMALL_KIB` (alternate
piece size), `SOURCE_MIB` (ring footprint), `MISALIGN` (bytes), `WRITERS`
(CPU threads copying into the rings). It reports GiB/s, submissions/s and
the mean submit call time.

Measurement note: the `load` path passed only `block_sizes` to
`benchmarkIfSupported`, so `ZML_DMA_BENCH_BLOCK_PARALLELISM` never reached
it and a first depth sweep measured nothing (`dma_budget_per_device` stayed
128 MiB at every depth). It is wired now. Check that line before trusting a
depth result.

### What the policy costs

Strict affinity takes the smallest node's capacity rather than the sum
(`DmaBlockPool.retainedRequestWidth`), and maps one pre-grown arena per
node. Measured on the same runs:

| | gb300-2 (4 dev) | mi300 (8 dev) |
| --- | --- | --- |
| `feasible_width` local / off | 79 / 128 | 64 / 128 |
| retained pinned local / off | 1.53 GiB / 1.02 GiB | 2.00 GiB / 1.52 GiB |
| `width_ceiling` local / off | 16 / 32 | **1** / 32 |

The pinned working set grows 1.3 to 1.5x and the feasible width roughly
halves, both against the project goal of keeping pinned host memory low.

### Defect: the growth-free ceiling mixes per-node and per-machine scopes

Calibration grows each NUMA arena correctly. `calibrated_node_reserves`
accumulates `max_in_flight_per_device` **per pool**, walking devices and
charging each one to its own node, and `ensureSourceWorkingSet` then grows
that node to `(preallocated_source_width + 1)` requests plus that node's own
reserve. Measured: gb300-2 has 2 devices per node, so 33 + 16 = 49 blocks =
784 MiB per node; mi300 has 4 per node, so 33 + 32 = 65, clipped to 64 by
the 2 GiB `max_mapped_bytes` ceiling.

The width ceiling then subtracts a quantity of a different scope
(`direct_loader.zig`, sixth pass `a7589b08`):

    retained_credits   = pool.retainedRequestWidth(.., strict_affinity)  // smallest NODE
    dma_stage_requests = dmaStageRequests(per_device, platform.devices.len, ..)  // ALL devices
    growth_free        = retained_credits -| dma_stage_requests

Under strict affinity `retained_credits` counts one node while
`dma_stage_requests` counts every device on the machine, so each node's pool
is charged for the other node's DMA stage as well:

| host | retained/node | stage charged | stage on that node | ceiling | correct |
| --- | --- | --- | --- | --- | --- |
| gb300-2 (4 dev) | 49 | 32 | 16 | 16 | **32** |
| mi300 (8 dev) | 64 | 64 | 32 | **1** | **32** |

Both corrected values equal the non-strict ceiling, which is the tell. The
bug cannot fire without strict affinity: with one shared pool serving every
device, the all-device stage count is the correct one, and `off`, `node0`
and `node1` all read 32.

The consequence on eight MI300X is that every adaptive load runs at **width
1**: five of five Llama-3.1-8B replicated runs took 4.5 to 5.9 s against 1.3
to 1.6 s with `ZML_DMA_BENCH_NUMA_OFF=1` (means 5.34 s against 1.42 s,
**3.7x**). Nothing to do with memory locality, and not a memory shortage
either: the node holds 64 requests and needs 32 for its own stage.

The `@max(1, ...)` floor in `SourceReadWidthController.init` is a second,
smaller defect: it turns "no growth-free headroom" into the worst possible
operating point instead of declining to clip. It only fires here because of
the miscount, but a genuinely tight pool deserves "accept some mid-load
growth", not width 1.

### The fix and its verification

`DmaBlockPool.growthFreeRequestWidth(blocks_per_request, strict_affinity)`
now does the subtraction node-wise, using each node's own `reserve`: the
minimum of `(capacity -| reserve) / blocks_per_request` over nodes when
strict, the sum otherwise. `SourceReadWidthController.init` takes it directly,
so `dmaStageRequests` is gone from the ceiling path and the scopes cannot
diverge again (it still feeds `RequestGateLimits`, where an all-device count
is correct). The `@max(1, ..)` floor became `@max(configured.initial(), ..)`:
the ceiling bounds the climb and must never force a start below the rung the
caller asked for.

`Options.max_mapped_bytes` went from 2 GiB to 16 GiB. It is a safety guard on
total pinned host memory, not a target -- the pool only grows to the
pre-grown working set plus each node's stage -- and at 2 GiB it was silently
clipping that working set on eight MI300X (65 blocks per node trimmed to 64).

Verified on both hosts, same fixtures, five interleaved repetitions:

- `width_ceiling` is **32 in all four arms on both hosts**, where strict
  affinity previously read 1 (mi300) and 16 (gb300-2).
- mi300, Llama-3.1-8B replicated on eight MI300X: `local` **5.34 s -> 1.26 s**
  mean, now the fastest arm rather than 3.7x the slowest (`off` 1.47,
  `node0` 1.40, `node1` 1.55; the host carried a load average of 37, so read
  these as parity). Selected width is 16 to 24 instead of 1, and the pinned
  high-water mark is *lower* than the aggregate pool's: 1.25 to 1.38 GiB
  against 1.52 GiB.
- gb300-2, DeepSeek replicated on four GB300: unchanged, `local` 5.63 against
  `off` 5.60, `node0` still worst at 5.99. The read-bound conclusion above
  stands.
- mi300 retained rose from 2.00 GiB (clipped) to 2.03 GiB, the full 65 blocks
  per node, and `feasible_width` from 64 to 959 now that the guard is not
  binding.

Regression tests: `DmaBlockPool growth-free width subtracts each node's own
DMA stage` (the eight-MI300X arithmetic: 64 retained, 64 machine-wide stage,
32 on that node, answer 32), `DmaBlockPool growth-free width saturates when a
reserve covers the node`, and `source read controller starts at the
configured rung without headroom`.

### Recommendation

The 38% link penalty is real, but no fixture reaches the link, and the two
loader-level measurements point the other way: strict affinity is neutral on
DeepSeek and a 4.8% loss on Llama, while costing 1.3 to 1.5x pinned memory.
Node-local pinned blocks are the wrong default today.

- Do not derive strict affinity from the presence of `numa_node` attributes.
  It buys nothing measured, costs pinned memory, and its worst case (`node0`,
  everything on the page-cache node) is 34%.
- If placement is ever steered, steer it away from the node holding the
  source's page cache, not toward the device. That is a property of the file,
  not of the topology.
- Keep the mechanism: it is what makes the locality measurable, and it will
  matter if a source ever outruns 110 GiB/s per device.
- On MI300X the aggregate pool is strictly better: same throughput, 25% less
  pinned memory, twice the feasible width.

### Defect: the growth-free ceiling mixes per-node and per-machine scopes

Calibration grows each NUMA arena correctly. `calibrated_node_reserves`
accumulates `max_in_flight_per_device` **per pool**, walking devices and
charging each one to its own node, and `ensureSourceWorkingSet` then grows
that node to `(preallocated_source_width + 1)` requests plus that node's own
reserve. Measured: gb300-2 has 2 devices per node, so 33 + 16 = 49 blocks =
784 MiB per node; mi300 has 4 per node, so 33 + 32 = 65, clipped to 64 by
the 2 GiB `max_mapped_bytes` ceiling.

The width ceiling then subtracts a quantity of a different scope
(`direct_loader.zig`, sixth pass `a7589b08`):

    retained_credits   = pool.retainedRequestWidth(.., strict_affinity)  // smallest NODE
    dma_stage_requests = dmaStageRequests(per_device, platform.devices.len, ..)  // ALL devices
    growth_free        = retained_credits -| dma_stage_requests

Under strict affinity `retained_credits` counts one node while
`dma_stage_requests` counts every device on the machine, so each node's pool
is charged for the other node's DMA stage as well:

| host | retained/node | stage charged | stage on that node | ceiling | correct |
| --- | --- | --- | --- | --- | --- |
| gb300-2 (4 dev) | 49 | 32 | 16 | 16 | **32** |
| mi300 (8 dev) | 64 | 64 | 32 | **1** | **32** |

Both corrected values equal the non-strict ceiling, which is the tell. The
bug cannot fire without strict affinity: with one shared pool serving every
device, the all-device stage count is the correct one, and `off`, `node0`
and `node1` all read 32.

The consequence on eight MI300X is that every adaptive load runs at **width
1**: five of five Llama-3.1-8B replicated runs took 4.5 to 5.9 s against 1.3
to 1.6 s with `ZML_DMA_BENCH_NUMA_OFF=1` (means 5.34 s against 1.42 s,
**3.7x**). Nothing to do with memory locality, and not a memory shortage
either: the node holds 64 requests and needs 32 for its own stage.

The `@max(1, ...)` floor in `SourceReadWidthController.init` is a second,
smaller defect: it turns "no growth-free headroom" into the worst possible
operating point instead of declining to clip. It only fires here because of
the miscount, but a genuinely tight pool deserves "accept some mid-load
growth", not width 1.

### The fix and its verification

`DmaBlockPool.growthFreeRequestWidth(blocks_per_request, strict_affinity)`
now does the subtraction node-wise, using each node's own `reserve`: the
minimum of `(capacity -| reserve) / blocks_per_request` over nodes when
strict, the sum otherwise. `SourceReadWidthController.init` takes it directly,
so `dmaStageRequests` is gone from the ceiling path and the scopes cannot
diverge again (it still feeds `RequestGateLimits`, where an all-device count
is correct). The `@max(1, ..)` floor became `@max(configured.initial(), ..)`:
the ceiling bounds the climb and must never force a start below the rung the
caller asked for.

`Options.max_mapped_bytes` went from 2 GiB to 16 GiB. It is a safety guard on
total pinned host memory, not a target -- the pool only grows to the
pre-grown working set plus each node's stage -- and at 2 GiB it was silently
clipping that working set on eight MI300X (65 blocks per node trimmed to 64).

Verified on both hosts, same fixtures, five interleaved repetitions:

- `width_ceiling` is **32 in all four arms on both hosts**, where strict
  affinity previously read 1 (mi300) and 16 (gb300-2).
- mi300, Llama-3.1-8B replicated on eight MI300X: `local` **5.34 s -> 1.26 s**
  mean, now the fastest arm rather than 3.7x the slowest (`off` 1.47,
  `node0` 1.40, `node1` 1.55; the host carried a load average of 37, so read
  these as parity). Selected width is 16 to 24 instead of 1, and the pinned
  high-water mark is *lower* than the aggregate pool's: 1.25 to 1.38 GiB
  against 1.52 GiB.
- gb300-2, DeepSeek replicated on four GB300: unchanged, `local` 5.63 against
  `off` 5.60, `node0` still worst at 5.99. The read-bound conclusion above
  stands.
- mi300 retained rose from 2.00 GiB (clipped) to 2.03 GiB, the full 65 blocks
  per node, and `feasible_width` from 64 to 959 now that the guard is not
  binding.

Regression tests: `DmaBlockPool growth-free width subtracts each node's own
DMA stage` (the eight-MI300X arithmetic: 64 retained, 64 machine-wide stage,
32 on that node, answer 32), `DmaBlockPool growth-free width saturates when a
reserve covers the node`, and `source read controller starts at the
configured rung without headroom`.

### Recommendation

Keep NUMA-local pools for CUDA/GB300 - the 38% link penalty is real and will
matter as soon as a source outruns 110 GiB/s per device - but stop paying for
them where they buy nothing:

- Derive strict affinity from a measured local/remote ratio rather than from
  the mere presence of `numa_node` attributes. One `dma-bench` pass per node
  already measures it (176 against 110, or 46 against 46).
- The width-ceiling scope mismatch is fixed; strict affinity is now at
  parity on eight MI300X instead of a 3.7x regression.
- On MI300X the aggregate pool is strictly better: same throughput, 25% less
  pinned memory, twice the feasible width.

## Open work

Third-pass items left open; `PLAN.md` holds the checklist.

- The NUMA placement experiment (task 12) is done; see "Seventh pass". The
  2026-09-04 "MI300 host degradation" was a stale ROCm plugin, so the Laguna
  window measurement (task 5) can be retaken once the plugin fix lands
  properly (`platforms/rocm/rocm.bzl` currently carries a machine-local
  `file://` override that must not be committed). The
  MI300 checkouts are on the `loader-third-pass` branches (zml `67464f3c`
  or later, monorepo `d426dde4`); the previous heads were zml `db961721` and
  monorepo `9efec789`.
- The first measurement window of a load is now discarded as a warm-up
  (fourth pass); the bias it removes is the DMA-stage fill burst, which is
  the opposite sign of the startup bias seen on hosts with slower DMA.
- Fourth-pass follow-ups: the CUDA host regression run on the final tree
  (both RTX 5090s were held by another user's server for the whole
  afternoon of 2026-09-04; the intermediate tree measured 443 to 463 ms
  against 457 to 485 ms there); the MI300 comparison once the host is
  healthy. The gb300-2 hold at 8 is explained by the fifth pass: on a
  DMA-bound host the rungs measure within noise of each other and the
  single-sample 3% climb rule stops at random; see "Fifth pass".
- Fifth-pass follow-ups: the controller rule for DMA-bound windows is the
  sixth pass; the `toSliceAlloc` sub-byte shard placement and the pump-side
  race with an errored manager are still open.
- Sixth-pass follow-up: the remaining 5% on gb300-2 needs a controller that
  keeps sampling for the whole load (see the sixth-pass limits). The B70,
  CUDA and MI300 trees have not been re-measured against the width ceiling;
  the B70 was (`width_ceiling=64`, no behaviour change).
- The 8% smallest-near-peak block rule is fragile on a busy host (it chose
  2 MiB on MI300 while degraded). Calibration caching per host/plugin, or
  re-screening when the measured rate is implausibly low, remains open.
- One oneAPI plugin abort on the failure path (`Check failed:
  definition_events_[buffer_index]` after a transfer error at width 128
  against the throttled proxy) was seen once and not reproduced.
- Backpressure is process-global and load-untagged (CTX assumption); real
  AWS runs need credentials this machine lacks. They removed remaining
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
