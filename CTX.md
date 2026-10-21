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

2026-10-14: merged with zml/monorepo master; see "Twentieth pass" for the
merge policy, the API changes it forced and the B70 re-validation.

The tenth-pass review below updates the module/API map and ownership fixes.
Earlier snapshots and measurements retain their historical names. The current
reader-facing design is in `docs/learn/loader.md`.

## Workspace-owned DMA allocation follow-up (2026-09-06)

Platform-specific host allocation now has one owner in
`io/host_memory.zig`: `Workspace.Backend` selects DmaMap-backed pages for
CUDA and oneAPI, PJRT-owned `pinned_host` buffers for ROCm, and pageable
pages for CPU. ROCm cannot reach DmaMap through a generic allocator.
`NumaPlacement`, `mem/dma.zig`, and its former `Allocator`, `MapAllocator`,
and `BufferAllocator` adapters are removed. Page-backed arenas automatically
interleave over every discoverable memory-bearing NUMA node; a single node,
unreadable topology, or refused automatic placement falls back to the kernel.
Each backend retains its concrete arena type directly. Page-backed backends
reuse one private, purpose-built `HugePageAllocator` value for NUMA fallback
state, huge-page advice, and optional DmaMap registration. It does not expose a
generic allocator interface. ROCm instead retains PJRT buffers and balances
whole arenas by allocated bytes across device-associated host nodes.
There is no separate arena-ownership union or public NUMA placement knob.
`Workspace` is single-owner and grows sequentially. Each backend allocation is
retained immediately, with ROCm node totals and workspace mapped-byte
accounting updated only after allocation retention succeeds. `growToBlocks`
allocates all missing blocks in one arena.

Loader initialization owns a local workspace during calibration and pregrowth,
then transfers it by value into `BlockPool`. Pool initialization consumes the
workspace only on success; failure leaves it owned by the caller. The loader
uses `initCalibratedBlockPool` to contain this lifetime: an ordinary `errdefer`
cleans up failed initialization, and success returns the owning pool together
with calibration, without a workspace-moved flag. Request sizing is derived
from the calibration by the caller, while pregrowth bytes and duration are
logged immediately rather than carried in the return value. The loader
retains only the pool, whose teardown also frees the workspace. Pool budgets
and mapped-byte totals come directly from its owned workspace. Growth reserves
free-list metadata before allocating an arena, so metadata failure cannot leave
retained memory unattached to the pool.

The playground's historical concurrent-DMA and early-event-retirement probes,
and the temporary `io/dma_diagnostics.zig` module, are removed.

## Per-submission progress follow-up (2026-09-05)

Progress is the final optional argument to `load`, `loadBuffer`, `loadExecute`,
and `Window.submit`, rather than a Loader option. Direct transfer items retain
only their submission's progress parent until completion. After reconsidering
the estimate policy, callers own totals: `store.view().count()` estimates a
whole checkpoint loaded once. Both backends complete one item per loaded
source; transformed tensors skipped by bulk loading contribute zero.
`loadExecute` counts input sources, not outputs, and completes source progress
before execution in `Handle.await`. Keep the parent alive through all handles.
Initialization no longer retains the options struct; calibration still happens
only in init.

## Per-submission store follow-up (2026-09-05)

`Loader.init` no longer takes or retains a TensorStore. `load` and `loadBuffer`
accept it before shardings; `loadExecute` accepts it before bindings, as does
`Window.submit` after the loader argument. Source metadata remains borrowed
until the submission completes. DMA calibration needs no dummy store.

## Per-load sharding follow-up (2026-09-05)

Shardings are passed to `Loader.load` and `Loader.loadBuffer` as their final
argument, rather than retained in `Loader.Options`. An empty slice selects
replicated placement. Source placements are resolved before submission;
`loadExecute` takes its input and output shardings from the executable without
an independent loader-wide output placement check. Calibration remains in init.

## Loader-owned initialization follow-up (2026-09-05)

The workspace is no longer public API. `mem.zig` exposes the NUMA placement
policy; workspace, arenas, and block pool now live in `io/host_memory.zig`.
The user's in-progress workspace validation changes were preserved when moving
that implementation.

`Loader.backendFor(target)` selects direct versus buffered loading explicitly;
`Workspace.isSupported` and `io.dma.isSupported` are removed. CPU uses the direct
transfer-manager path with ordinary pages, without implying DMA capability.

Every direct `Loader.init` owns a workspace, calibrates into it, then prepares
the load's block pool. It frees all of this in `deinit`; workspace borrowing,
external reuse, and the ownership flag are gone. CPU returns default sizing;
buffered backends skip calibration. `Loader.Options.dma` configures measurement,
`.max_host_bytes` bounds host memory, and `Loader.calibration()` reports the
selected sizing. Page-backed arena placement is automatic; the former `.numa`
option and `ZML_DMA_BENCH_NUMA` diagnostic override are removed.
`.dma_workspace`, `.dma_calibration`, and the public `io.dma.benchmark` entry
point are also removed.

LLM calibration no longer runs in a caller-managed future alongside model
compilation: it happens during loader initialization after compilation. The
IO diagnostic's dma-bench command initializes a loader with an empty store and
reports its selected calibration. Repeated loader instances calibrate and own
separate arena sets; arenas remain reusable across submissions of one loader.
Earlier sections below describe historical public-workspace APIs.

Verified on macOS arm64 with
`bazel test //zml:test //examples/io //examples/llm --test_output=errors`:
core tests passed (259 passed, three skipped) and both examples built.
Coverage includes backend selection independent of DMA, CPU/default versus
buffered/absent calibration, and workspace cleanup on initialization failure.
Zig formatting, Buildifier, and `git diff --check` passed.

## Current design

### Scope and API

- `zml.io.Loader` selects the direct path for CUDA, ROCm, oneAPI and CPU
  (`mem.DmaWorkspace.isSupported`; CPU arenas are plain pages, see "Ninth
  pass") and the buffered path for TPU, neuron and metal. A loader owns its
  worker pool and every submission it published, kept in a FIFO until
  retired. `load(Model, model, buffers, store, progress)` submits all
  single-source tensors of a model (no `shardings` argument since the
  2026-10-14 master merge: placement comes from each tensor's shape); `loadExecute(store, bindings,
  progress)` submits the sources of one or more `Binding{tensor, output,
  exe}` as ONE planned submission so adjacent sources of different bindings
  coalesce; both return `!void` (fourteenth pass; before it every submission
  returned a `Handle`). Retiring a submission waits for its reads and DMA,
  then for bindings runs each executable in binding order on the retiring
  task with `.wait = true`, writes `output.*`, frees the inputs and commits
  the submission's logical bytes to `bytesLoaded()`. `awaitAll` retires
  everything in publish order and returns the first error, which is sticky:
  later submissions are refused. `deinit` awaits what is pending without
  running executables. A zero-byte tensor is `error.EmptyTensor`.
- Memory policy lives in `loadExecute` (fourteenth pass): before publishing
  it retires the oldest pending submissions, running their executables on the
  calling task, until its own cost (inputs, output and the executable's
  compiled temporaries per device) fits the room the devices report
  (`bytes_limit - bytes_in_use` minus the loader's submitted-but-unallocated
  bytes minus a 64 MiB reserve, `zml/io/execute_admission.zig`); one
  submission is always admitted. Where a device reports no limit (CPU) or the
  backend does not count its allocations (buffered), every pending submission
  is retired first, the pre-rework synchronous order. `load` is never gated,
  and a transformed tensor counts as delivered once a `loadExecute` naming it
  was submitted, so a bulk submitted after the packs queues behind them.
  There is no caller-side window and no memory knob.
- `VFS.loadProfile(path)` is prepared once for a model load and passed as a
  borrowed `LoadProfile`. It contains a backend name, minimum read chunk,
  `high_latency`, and optional aggregate retry/throttle statistics. Those
  statistics are observability, not control (sixteenth pass): nothing in the
  loader reacts to them, and they are backend-wide rather than load-tagged.
- Profile minima are local/file 8 MiB, HTTP/S3/GCS 16 MiB, and HF 32 MiB.
  Effective source request size is the greater of the profile minimum and
  calibrated DMA block size, capped at the supported 32 MiB maximum.
- The source width is fixed per load (fifteenth pass): `Options.read_parallelism`,
  null for the profile's default of 16 (local) or 32 (high-latency). The current
  `Loader` API differs from the adjacent monorepo's older checkout: store,
  shardings, progress, and profile now belong to loader initialization.
- `DirectMemoryWriter`, `DirectShardWriter`, and `DynamicBufferPool` are no
  longer public loader mechanisms. There is no executor inside the loader:
  executables run on the caller's task, inside `loadExecute` when admission
  retires older submissions and inside `awaitAll`; publish order is the
  execution order.
- Model traversal, tensor-store lookup, resolved sharding selection, and output
  flattening happen once in the shared `Loader.load` front end. Executable
  source lookup, validation, input-shell allocation (`BoundExecutable`),
  execution, and output ownership similarly live once above the backend
  split; buffered and direct loaders implement only `submit(specs)` and
  `awaitBatch`. The buffered memory writer is private.
- `zml/io.zig` is the public facade. The shared loader front end is in
  `zml/io/loader.zig`, while `zml/io/backend.zig` owns backend selection,
  submission dispatch, and the `LoadSpec` contract. Direct planning,
  scheduling, transfer lifecycle, and their tests are in
  `zml/io/direct_loader.zig`; platform-owned DMA settings, retained arenas,
  and calibration are in `zml/io/dma_calibration.zig`; pure sharding-to-byte-span
  expansion is in `zml/io/DispatchSpans.zig`.
- `Loader` owns IO, platform, store, and front-end options directly rather than
  recovering duplicated state from its active backend. The buffered backend
  retains none of the unused store/options; the direct backend retains only
  the load profile and progress pointer needed after construction.
- The front end counts a submission's logical bytes when it is retired with
  execution (`bytesLoaded`; the backends no longer count, fifteenth pass).
  Direct diagnostics are batch-owned (publish, seal and completion offsets
  from loader creation, planned jobs/items/transfers, plan count with total
  planning time, VFS delta since publish) and logged once per batch at await;
  loader-wide totals (reads, physical bytes, DMA submissions, pool
  high-water/mapped, width) are logged once at `destroy`.
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
  `range_read.performRangeRead` over the governed loop of
  `vfs/request.zig`: one whole Range request per admitted positional call,
  retries and holds serial inside that caller's source credit (the retired
  backend-local `parallel_read` pools must not return). Every other request
  a backend makes (the HF tree and redirect chain, the S3 and GCS listings
  and HEADs, the GCS token exchanges, the HTTP HEAD chain) goes through the
  same loop, so one governor covers a backend's whole quota. A backend
  contributes a `RequestSpec`: its name and the object for log lines, what a
  503 means, and the rate-limit scope; a range read adds a `prepare` hook
  called for every attempt that returns the URI, the authorization value and
  its extra headers; the loop appends `Range`. S3 recomputes the SigV4 timestamp and signature in the hook per
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
  one function (`request.classifyStatus`): 408 is a timeout, 429 a
  throttle, other 5xx a server failure, and 503 is a throttle on S3
  (`SlowDown`) and GCS but a server failure on generic HTTP and HF. A
  throttle arms the backend's hold instead of charging the request's retry
  budget (sixteenth pass); any other retried status whose response names a
  delay (`Retry-After` delta-seconds, or the `RateLimit` header's `t=` reset
  on HF; the HTTP-date form is not parsed) sleeps that long instead of the
  jittered delay. The backends expose aggregate request, retry, timeout,
  server-failure, throttle, byte, delay, hold and hold-wait counters.
- Rate limiting is a hold, in `vfs/request.zig`. One `Governor` per backend
  instance (keyed by URI authority, so a per-authority scope is a change to
  one function) holds a deadline: a throttle arms it at
  `max(server delay, full-jitter backoff)` clamped to `[retry_initial_delay,
  max_hold]`, extends it when a later answer names something longer, and
  never shortens it. Every request of that backend waits the deadline out,
  each with its own jitter, and re-checks, since it may have been extended.
  A clean window as long as the last hold ends the episode and resets the
  backoff; an episode with no such window for longer than `throttle_budget`
  fails the backend's requests with `error.RateLimited` until one comes.
  Timeouts never hold. Two budgets, not one: a throttle is not charged to
  `max_retries`, so a limited server cannot exhaust a reader, and everything
  else keeps the per-request backoff. Defaults: five retries, 500 ms to 30 s
  of backoff, a 2 minute hold cap, a 5 minute episode budget, all
  `InitOpts` fields of the four backends. The hold is a deadline rather
  than a permit, so it composes with the loader's fixed width instead of
  competing with it, and a request holding a source credit simply waits.
- One source job performs one exact absolute scatter read into pinned blocks.
  Extra physical calls occur only for short reads/retries or `IOV_MAX` limits.
  The loader counts read operations, not physical calls: a remote source's
  physical request count comes from the VFS `batch source` line, and a local
  profile has none.

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
a submission's plans are claimed in file order. A plan's jobs keep their
planning order, which is file order (sixteenth pass: the byte-fair order
across destination devices was measured against it and made no difference,
see the ledger below). No job depends on another: every DMA piece is submitted as soon as
its block is read (fifth pass), so the request carrying a tensor's tail may
be planned, claimed and read before the tensor's earlier requests. While
those spans are available, the planner also emits the final item, block
index/offset, writer mask, destination offset, and length records. The
published plan owns its source jobs in planning order with their final
transfer records; runtime tensor state does not own another
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

- A plan's jobs are immutable and claimed in planning (file) order, and a
  submission's plans in file order. Reads and DMA complete in
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
- The scheduler is a strict FIFO of published plans under one mutex and
  condition (sixteenth pass; it queued batches before): `publish(batch,
  plan)` stamps the plan with its batch, appends it to the batch and adds
  its completion units, and queues it behind every earlier plan when it has
  jobs (capacity is reserved first so the plan, its units and the queue
  entry appear together); `seal(batch)` only stamps the diagnostics; `claim`
  advances the head plan's plain cursor and pops the plan with its last job;
  `waitForWork` sleeps while nothing is unclaimed; `fail` retires the
  unclaimed units of every queued plan and clears the queue; `remainingJobs`
  reports the unclaimed total. A submission publishes and seals on one task,
  so its plans are contiguous in the queue in file order and a later
  submission's first job is claimed only after every job of the earlier
  ones; fairness is intra-plan and the caller's submission order is the
  cross-submission policy. While a submission is still planning, its
  published plans are simply claimed out and the unclaimed total drops to 0,
  so workers sleep in `waitForWork` until the next plan's broadcast, at most
  one file's planning time. A plan is queued only with jobs left and its
  units are added in the same critical section, so a queued plan's batch
  still holds a unit, can never be freed under the scheduler, and needs no
  worker rendezvous. Persistent workers loop `waitForWork` -> lifecycle
  credit -> `claim` -> read -> enqueue; the claim takes the mutex the idle
  wait already takes.
- A `Batch` owns its plans (heap `Plan{batch, jobs, transfers, requests,
  blocks, events, events_used, planning_ns, cursor}`, one per file in file
  order)
  and its items from creation. The planner preallocates every context a
  plan's jobs can need: one `RequestContext` per job (initialised idle:
  nothing pending), one `BlockContext` per job block (exact
  `divCeil(len, block_size)`; `Job.blocks` is the job's slice) and one
  `EventContext` per planned DMA submission (the transfers' writer count,
  `planned_dma_submissions` on the batch line), handed out in submission
  order under `metadata_mutex`. A `Claim` names its plan and job index and
  slices the plan's request, block and transfer arrays from it, so
  `registerBlock` cannot fail and takes no lock; a load allocates only per
  file. The batch
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
  arrays. Nothing drains the gates per batch and there is no barrier. A
  pipeline failure is sticky: every open handle's await returns it and later
  submissions are rejected with it. There is no loader-wide context list or
  reap. The buffered backend mirrors this with `BufferedBatch{pending, done}`
  over the `LimitedGroup` read tasks.
- The source width is fixed for the load (fifteenth pass): the profile's
  default (`limits.defaultReadParallelism`: 16 for local files, 32 for a
  high-latency source, from the sweeps recorded in that pass) or the
  caller's `Options.read_parallelism`, clipped to one less than the requests
  the pre-grown pinned set holds. The adaptive controller, its measurement windows,
  generations and fences, the warm-up rule, the blind bootstrap, worker
  parking and `source_concurrency.zig` are gone; the evidence is in the
  fifteenth pass.
- Two gates bound the pipeline, both constant for the load (sixteenth pass).
  All workers compete for lifecycle capacity: the pre-grown pinned capacity
  in requests (`pool.capacity / blocks_per_request`), which is the source
  working set of `width + 1` requests plus the calibrated per-device DMA
  depth, so the DMA stage keeps its depth whatever the read width (fourth
  pass) and no credit can want a block the pre-growth did not map. The read
  gate alone limits source calls, at the width. A request returns its
  lifecycle credit only after all its DMA children finish.
- The width never changes during a load (sixteenth pass): rate limiting is
  handled one layer down, by the VFS governor, which holds every request of
  the throttled backend. The loader keeps the statistics side channel for
  observability only: the `batch source` line reports the submission's
  requests, bytes, retries, throttles, holds and hold wait.
- Pinned pre-growth happens at loader creation, before any load: the DMA
  reserve (calibrated depth x devices, at least one maximal request) plus
  `width + 1` source requests, as two arenas, the reserve first (ROCm
  balances bytes per allocation across its host nodes). The width itself is
  what the mapped ceiling fits: `Sizing.init` computes it once from the
  usable blocks plus the remaining budget and refuses below one request with
  `DmaMappedBudgetExceeded` (264 MiB for width 16 with 8 MiB requests on the
  CPU platform, against the 528 MiB the former 32-wide set took). The pool
  is then sized once and never maps again, so nothing maps a slab inside a
  load (146-230 ms of hipHostMalloc on MI300X when it did) and
  `pinned_mapped == high_water` on every recorded run. A dedicated
  pregrowth line logs `retained`, `pregrown` and `pregrowth_ms`.
- `width + 1` worker tasks are spawned at creation and
  never retired or parked: a worker hands its request to the DMA stage and
  claims the next, so credits beyond the read width need no workers of
  their own, and after a throttle step the surplus workers wait at the read
  gate (only remote loads step down, where a waiting worker costs nothing
  measurable; 128 persistent workers had cost 7% on one MI300X, which is why
  the count follows the width).
- DMA depth is fixed at eight blocks per device by default after calibration
  work showed adaptive DMA width added substantial complexity and little
  load value. The pump enforces it as a byte budget
  (`max_in_flight_per_device x block_size`) with a cap of 64 pieces in
  flight per device, so a block of small tensors keeps the calibrated bytes
  moving (fourth pass). There is no global DMA parallelism cap.
- DMA event lifetime: a ready callback hands its `EventContext` to its
  device pump's intrusive `retired` stack under that pump's mutex, after
  its own `eventCompleted` (and any pump it ran) and before
  `block.complete()`; `pump` destroys the stack at the top of every
  iteration under the lock, so an event is destroyed by a later pump or by
  `retireBatch`, never inside its own callback. Live PJRT events are
  bounded by devices x 64 plus one pump batch instead of a submission's
  transfer count. Checked against the oneAPI plugin under sustained load
  (PLAN.md task 9); the `retire_events_early` switch that could turn it off
  went in the fifteenth pass.

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
  validation, arena reserves, worker scratch, and pinned feasibility use the
  exact maximum
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
  throughput and imposes alignment constraints. The eleventh pass made it a
  loader option with per-file residency detection; see "Eleventh pass".

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

### What `off` really is, and an interleaved pool (2026-09-05)

`ZML_DMA_BENCH_NUMA_OFF=1` is not "unplaced". The CUDA driver applies a
`prefer` policy for the calling thread's node when it registers the arena,
and the main thread ran on node 1 in every run, so `off` was `node1` by
luck. A probe cannot fix that: it observes an arbitrary state, and on a
cold cache the loader's own readers create the state it would observe.

The knowledge-free alternative is one pool interleaved page by page over a
node mask (`DmaWorkspace.Options.interleave_numa_nodes`,
`ZML_DMA_BENCH_NUMA_INTERLEAVE=0,1`, `MPOL_INTERLEAVE` in `NumaAllocator`;
`numa_maps` shows each arena split half per node). DeepSeek replicated on
four GB300, per-device pumps, depth 8, 16 MiB blocks, three interleaved
reps:

| arm | mean | runs |
| --- | --- | --- |
| node1 | 4.20 s | 4.31 4.25 4.04 |
| off (driver prefer, node 1) | 4.26 s | 4.57 4.19 4.02 |
| interleave 0,1 | 4.35 s | 4.37 4.48 4.21 |
| local (per-node pools, the default) | 4.92 s | 5.24 4.62 4.90 |
| node0 | 5.50 s | 5.77 5.47 5.27 |

Interleave is within 2-3% of the best single node without knowing which
node holds the page cache, and cannot land on the worst. It also puts half
the pinned pages on node 0 exactly like `local` does and is 12% faster, so
part of `local`'s loss is the split into per-node pools itself (half the
retained capacity per node, strict affinity on every lease), not placement.

Does the mask need more than one device per node to help? The bottleneck
is the page-cache-to-pinned copy plus the DMA read on the node holding the
file, so it should not. Measured on the same host with one device (GPU 0,
node 0) and with two devices on node 0, same fixture, three reps each;
`off` here is the default (per-node pools), which for devices on one node
is the same placement as `node0`, so those six runs are pooled:

| devices | node0 (= local default), 6 runs | node1 | interleave 0,1 |
| --- | --- | --- | --- |
| 1 (node 0) | 3.38 s (3.71 3.32 3.35 3.35 3.31 3.26) | **2.98 s** (2.81 3.14 2.98) | 3.22 s (3.07 3.36 3.22) |
| 2 (both node 0) | 3.82 s (3.81 3.73 3.86 3.77 3.99 3.78) | 3.94 s (4.01 3.90 3.92) | 3.86 s (3.70 3.76 4.13) |

One device: placing the pinned pool on the *other* node is 12% faster than
local, and the readers show why (`read_ms_per_read` 2.3-2.6 ms against
3.0-4.9): the page-cache read and the pinned write then use two memory
systems. Interleave takes a third of that (-5%) without knowing the
page-cache node. Two devices on one node invert `node1`: both links cross
sockets and `dma_stage_ms_per_read` goes 10-11 -> 16 ms, so the remote
link starts to cost and `node1` is the worst arm, while interleave is at
parity with local. Across the three configurations interleave was never
the worst and always within 3-8% of the best; every single-node choice was
the worst in at least one.

On a host with a single memory node, `MPOL_INTERLEAVE` over a one-node
mask degenerates to that node (the kernel round-robins over the mask; one
entry means every page lands there, a soft bind). Nothing changes, and
nothing can: the copy is intra-node by construction and the channels
inside a node are already hardware-interleaved. The knob only has a
meaning with two or more memory nodes, and then it should cover every node
that has memory, not only the nodes the devices sit on.

Recommendation, refined: drop the per-node pools and strict affinity, keep
one pool with a single policy knob -- interleave over the host's memory
nodes by default, or one explicit node when the client knows better -- and
never derive placement from where the first thread happened to run.

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

## Eighth pass: one pool, one placement knob (2026-09-05)

The seventh pass showed that placement is a host-memory question (the node
holding the page cache saturates), that per-node pools cost on their own
(`local` split the retained capacity and forced strict affinity on every
lease), and that interleaving over the memory nodes was never the worst
placement on 1, 2 or 4 devices while every single node was. The per-node
structure is gone:

- `DmaWorkspace` owns one arena list. Removed: `resolveNumaNodes`,
  `KnownPoolTopology`, `device_pool_indices`, `numaPoolCount`,
  `hasStrictAffinity`, the concurrent `growToBlockTargets` (one arena grows
  at a time now, `growToBlocks`), `arenaForDevice` (`arenaAtLeast`). The
  PJRT `numa_node` attribute (`Device.numaNode`) no longer decides `dmaMap`
  placement; its one remaining use is balancing ROCm's PJRT-pinned arenas
  (below).
- `DmaBlockPool` is one free list. Removed: `Affinity`, `AcquireScratch`,
  the augmenting-path matcher (`planAssignments`/`assignJob`), per-node
  growth selection, `NodeStats`, `minimumStrictAffinityRequestWidth`, the
  `strict_affinity` parameter of every width query. `Block` is a `[]u8`;
  `acquireMany(io, output)` allocates nothing once its arenas are attached.
  The reserve is one number: the DMA stage of every device.
- Placement is `DmaWorkspace.Options.numa: NumaPlacement`:
  `.memory_nodes` (default) interleaves over the bits of
  `/sys/devices/system/node/has_memory`; `.nodes = mask` binds (one bit) or
  interleaves (several); `.none` applies no policy. A single memory node, or
  an unreadable list, applies no policy -- interleaving over one node is that
  node. An automatic placement the kernel refuses (a cpuset that excludes a
  node, `EINVAL`) logs a warning once and continues unplaced; an explicit
  one fails the arena. The example exposes it as `ZML_DMA_BENCH_NUMA`
  (unset, `off`, `1`, `0,1`); `ZML_DMA_BENCH_NUMA_NODES`, `_NUMA_OFF` and
  `_NUMA_INTERLEAVE` are gone.
- The placement allocator is built per `allocate` call around
  `DmaMapAllocator` because the workspace is a value that moves; the arena
  log line now reads `placement=interleave nodes=0x3`.
- Growth stays concurrent: `growToBlocks` maps the missing blocks as up to
  four arenas registered at once (`arena_mutex` guards the list). Serial
  growth was measured first: 1.02 GiB took 317 ms in one arena against 280
  ms for 1.53 GiB in two per-node arenas; four parts take 94 ms. Pinned
  registration scales with threads on GB300.
- ROCm arenas are PJRT-pinned (`hipHostMalloc` through a device's host
  memory space), where `mbind` has no say, and there the per-node pools
  were doing real work: each pool allocated through a device on its node,
  so the pinned set was split 50/50. With one pool everything came from
  device 0's space and eight MI300X lost the fast mode (loader elapsed
  1.27-1.45 s in every placement arm, DMA stage 112-127 ms per read,
  against 0.82-1.0 s and 60-74 ms before). Rotating arenas over the
  devices' spaces gave a 61/39 split and 0.98-1.33 s: monotone in the
  imbalance. Registering `mmap`ed arenas through `dmaMap` on ROCm instead
  would put `mbind` back in charge, but costs ~4.5 s/GiB there (6.5-7.2 s
  of pre-growth for 1.52 GiB against 1.1-2.3 s PJRT-pinned), so it is
  rejected. The workspace now keeps one `HostNode` per node the devices
  report (`Device.numaNode`, the only remaining use of the attribute) and
  allocates each arena through a device on the node with the fewest arena
  bytes so far. Devices with unknown nodes are ignored when any topology is
  available; if every node is unknown, allocation falls back to device 0.

### Verification

gb300-2, four GB300, 16 MiB blocks, before = the committed per-node pools,
after = one interleaved pool with concurrent growth. `Loaded weights` in
`//examples/io` now includes the loader's arena pre-growth (the calibration
used to retain the pre-grown set before the timer started), so the
loader's own `elapsed` is reported next to it:

| fixture | arm | wall | loader elapsed | pre-growth | pinned mapped |
| --- | --- | --- | --- | --- | --- |
| DeepSeek replicated (3 pairs) | before | 4.86 4.88 4.87 s | 4.45 4.46 4.46 s | 280 ms | 1.53 GiB |
| | after | **4.51 4.56 4.56 s** (-7%) | **4.28 4.33 4.34 s** (-3%) | 94 ms | 1.02 GiB |
| Llama replicated (4 pairs) | before | 602-609 ms | 319-321 ms | 280 ms | 1.53 GiB |
| | after | **413-423 ms** (-31%) | 314-325 ms | 94 ms | 1.02 GiB |

The first after-run, with serial growth, had the same loader elapsed
(DeepSeek 4.14-4.33 s over five pairs, -6%; Llama 313-317 ms) but a 317 ms
pre-growth, which is where the concurrent `growToBlocks` came from. Llama
placement arms on the single pool (two reps, loader elapsed): `off` 304-305
ms, `node1` 299-304, interleave 315-317, `node0` 410-412 -- interleave 4%
off the best and 23% from the worst, as on DeepSeek.

mi300, eight MI300X, Llama replicated, six interleaved pairs, host load
2-36 (both arms saw it), before = HEAD's per-node pools, after = one pool
with node-balanced PJRT-pinned arenas (four through device 0 on node 0,
five through device 4 on node 1):

| arm | wall | loader elapsed | pre-growth | pinned mapped |
| --- | --- | --- | --- | --- |
| before | 3.07 2.93 2.59 2.40 2.35 2.48 s | 1.18 1.09 0.97 0.96 0.85 0.93 s | 1.4-1.9 s | 2.03 GiB |
| after | 2.55 4.19 2.05 2.02 2.08 1.94 s | **1.05 0.95 0.91 0.90 0.92 0.84 s** | 1.1-1.5 s (one 3.2 s spike) | 1.52 GiB |

Parity to slightly better on the loader (five of six pairs), less pinned
memory, and a shorter pre-growth: PJRT-pinned allocation there costs
0.7-0.9 s/GiB, so the third of the working set the per-node pools mapped
twice was expensive. The two rejected single-pool variants on this host
are in the bullet above (all arenas through device 0: 1.27-1.45 s; device
rotation: 0.98-1.33 s).

`zml/mem.zig` went from 2102 to 1391 lines and the loader lost ~150; the
unit tests that exercised matching (replica balancing, strict-first
planning, invalid affinities) went with the matcher, the width and growth
tests were rewritten for one pool, and `parseNodeList` has its own.

## Ninth pass: the CPU platform takes the direct pipeline (2026-09-05)

The buffered backend got two passes on 2026-09-04 (host staging pool,
`579a00de`; load-profile driven reads, `28961ff5`). A design review of what
to do with it next produced the evidence below, and the decision: CPU joins
the direct pipeline with plain pages; TPU, neuron and metal keep the buffered
backend, byte for byte, until their transfer path can be measured.

### Two independent axes, conflated by one predicate

- How bytes reach the device: the PJRT async transfer manager
  (`createBuffersForAsyncHostToDevice` then `transferData` into a byte range
  of a pre-allocated device buffer, from any host address, in any order) or
  `bufferFromHostBuffer` with a whole contiguous host tensor. Only the second
  forces whole-tensor host staging. This axis has nothing to do with
  pinning: the CPU plugin (`libpjrt_cpu.so`) exports the transfer manager
  and copies from ordinary pages; its `DmaMap` is the base-class
  Unimplemented.
- What the host arena is: our pages registered with the plugin (CUDA,
  oneAPI: `DmaMapAllocator`), the plugin's pinned host memory borrowed
  through a PJRT buffer (ROCm: `PjrtPinnedHostAllocation`; TPU:
  `UninitializedBufferAllocator` in `DmaAllocator`), or plain pages (CPU,
  neuron, metal). This only changes transfer speed.
- `DmaWorkspace.isSupported` decided both. It is now an exhaustive switch:
  CUDA, ROCm, oneAPI and CPU take the direct pipeline; TPU, neuron and metal
  the buffered one. A new target forces a decision.
- libtpu `0.0.42.dev20260613` (the wheel pinned in `platforms/tpu/tpu.bzl`)
  carries `xla::TpuClient` overrides of both `CreateBuffersForAsyncHostToDevice`
  and `DmaMap`, and the same `CommonAsyncHostToDeviceTransferManager` with
  `TransferRawDataToSubBuffer` the fifth pass validated against. A symbol is
  not proof the override is more than a stub: TPU moves to the direct path
  after a runtime check on a TPU host, not before.

### What the change is

- `DmaArenaAllocation.pageable`: our pages, never registered. The arena
  kind is decided once, `DmaWorkspace.arenaKind(target)`: `dma_map` for
  CUDA and oneAPI, `pjrt_host` for ROCm, `pageable` for CPU (NUMA-placed and
  huge-page advised like the mapped arenas, through
  `DmaMapAllocator.initPageable`, whose null platform skips
  `dmaMap`/`dmaUnmap`) and for the `platform == null` test path; null is
  the buffered backend, and `isSupported` is that null check.
- Found by the review of this change and fixed: `numa_mask` was read and
  written outside `arena_mutex` while `growToBlocks` maps up to four arenas
  concurrently, so a refused automatic placement could be undone by a
  worker that had read the mask earlier. The mask's only transition is
  nonzero to zero (`leaveUnplaced`), taken under the mutex. The arena log
  line is now `DMA arena kind={dma_map,pageable,pjrt_host} placement=...
  map_ms=...`. Huge-page alignment and advice apply only to allocations of
  at least one huge page, so tiny test arenas are plain.
- The five end-to-end `loader ...` tests in `zml/io/loader.zig` run on both
  backends on the CPU platform (`LoaderTestFixture.backends`): the direct
  one through `Loader.init`, the buffered one built directly, since nothing
  else in the tree constructs it any more.
- Nothing in `direct_loader.zig` changed. The pipeline, the width
  controller and the pumps run on CPU as they are, and `transferData` is
  the plugin's copy. `dma.benchmark` returns `Calibration.default` on CPU
  without measuring (2026-09-05): the first CPU pass measured a memcpy
  (2 MiB at 102 GiB/s in 607 ms for four CPU devices) and the load took the
  same 1.55 to 1.60 s at 2, 8 and 16 MiB blocks, so there was nothing to
  select. The loader pre-grows its arenas itself.

### Why: the buffered read model, not its budget, was the problem

- Interleaved on the B70 (oneAPI `level_zero:1`, `hf://Qwen/Qwen3.5-4B`,
  same minutes): direct 10.9 s and 16.3 s (278 coalesced 32 MiB reads for
  738 tensors, width 32, ~1.5 GiB pinned) against 48.3 s and 47.6 s for the
  buffered build of `28961ff5`. The same buffered binary had done 22.4 s the
  day before: `hf://` varies 2x day to day, so only interleaved pairs count.
- The checkpoint is 738 tensors, 604 of them under 32 MiB. Tensor-shaped
  reads cost ~900 requests; the planner's coalescing costs 278. The direct
  loader also blind-grows to width 32 before the first response.
- Staging bounds from the safetensors headers, K = 12: Llama-3.1-8B
  11.7 GiB (K x largest, the buffered budget) against 3.05 GiB (sum of the
  top K); Qwen3.5-4B 14.2 against 1.67; Qwen3.6-27B 28.4 against 6.4. The
  direct pipeline needs ~300 MiB locally and 1.06 GiB on `hf://` for any of
  them, because it never holds a whole tensor on the host.
- Buffered backend on the CPU platform, fixed read width 1/2/4/8/12/24/48,
  page-cached Llama: 2.93/2.33/1.84/1.41/1.38/1.43/1.48 s. The plateau is at
  8, so a transfer concurrency cap is not a lever there.
- A sorted biggest-first order with a sum-of-top-K budget was considered and
  rejected: it forfeits coalescing (the 604 small tensors become 604 round
  trips at the tail), the trivial-reuse property it buys needs buffers that
  shrink, which PJRT-allocated pinned memory cannot do, and the checkpoints
  have only 5 to 23 distinct tensor sizes so the best-fit pool already had
  the reuse. Uniform blocks give the same invariant by construction.
- `examples/io/main.zig` hands `init.arena.allocator()` to `Loader.init`.
  `ArenaAllocator.free` frees only the most recent allocation, so the
  buffered backend's staging drops under that arena are leaks until the
  arena dies. The direct pipeline's workspace allocates through the page
  allocator and is unaffected; any staging design must own its allocator.

### Results (B70 host, four CPU devices, warm page cache, adaptive default)

Correction (2026-09-05 evening): `examples/io load <path>` defaults to
`sharded`, so every row below was a sharded load and the "replicated"
label was wrong. True replicated Llama-3.1-8B (4 x 14.96 GiB of device
buffers) does not fit this 62 GiB host: the buffered backend is OOM-killed
at 62.3 GiB RSS and the direct one completes at 60.4 GiB with the page
cache evicted, so that fixture is void here. The interleaved table in the
next section is the one to use.

| fixture (all sharded) | buffered (before) | direct (after) |
|---|---|---|
| Llama-3.1-8B | 1.36, 1.36, 1.50 s | 1.55, 1.55, 1.58 s |
| Qwen3.5-4B | 0.92, 0.92, 1.23 s | 0.88, 0.91, 1.10 s |
| Qwen3.5-9B | 2.91, 2.99, 3.87 s | 2.01, 2.04, 2.27 s |
| `hf://Qwen/Qwen3.5-4B` | (48 s on oneAPI, same hour) | 10.4 s, 278 reads, width 32 |

Host staging high water 292 to 298 MiB locally, 1.06 GiB on HF. Read-back
(`ZML_LOAD_CHECK=1`): 291/291 on Llama, 259/775 on Qwen3.5-9B. Width held
at 12 to 24.

### Calibration skipped on CPU, and what the CPU pump is doing (2026-09-05)

`dma.benchmark` returns the defaults on CPU (commit `6f75cd37`). The ready
line reads `dma_block_size=4 MiB, pregrown=392 MiB, pregrowth_ms=20` where
the benchmark had taken 607 ms, and the loads are unchanged. Two binaries
from the same tree (the buffered one with `arenaKind(.cpu)` temporarily
null), interleaved after a warm-up run:

| fixture | direct | buffered |
|---|---|---|
| Qwen3.5-4B replicated (34.7 GiB of device buffers) | 2.03, 2.01 s | 2.02, 1.98 s |
| Qwen3.5-4B sharded | 0.93, 0.92 s | 0.92, 0.91 s |
| Llama-3.1-8B sharded | 1.55, 1.57 s | 1.38, 1.35 s |

The CPU pump: `dma_submit_us_per_piece` equals `dma_piece_latency_ms`
(1.26 ms per ~3.1 MiB piece on Llama sharded), so the plugin's
`transferData` copies on the submitting thread, and the pump is saturated:
1234 pieces per device x 1.26 ms = 1.55 s, the whole load; true replicated
Llama is 4101 pieces x 1.26 ms = 5.2 s (measured 5.0 and 5.3 s). The
readers wait for it (`dma_stage_ms_per_read` 37 against `read_ms_per_read`
2.3, credit wait 4.8). The copy is not memcpy-bound: it first-touches the
plugin's fresh device buffers in 4 KiB pages (`/usr/bin/time -v`: 3.94M
minor faults for 14.96 GiB and 10 s of system time; 15.7M faults for
replicated), 2.4 GiB/s per pump thread, THP `madvise` on this host. That is
also why the calibration measured 102 GiB/s into its reused ring and had
nothing to select. The buffered backend faults the same pages from twelve
workers and lands on the same time on Qwen in both modes, so more copying
threads per device are worth at most the 14% seen on Llama sharded, and
that gap is not attributed. One pump per device stays (eighth pass) and
CPU is not a performance target. A host with THP `always` would fault at
2 MiB; untested, it needs root.

### If a platform ever needs `bufferFromHostBuffer` under the same pipeline

Keep everything through the read (planner, scheduler, workers, gate,
controller, credits) and swap the sink: a tensor is opened by the first job
that touches it with a contiguous span from the arena (a ring: tensors open
in plan order and complete roughly in order, so the tail is the oldest live
span); jobs scatter into `(span, offset)`; the per-tensor byte counter that
already flags the last transfer submits `Buffer.from` with `wait = false`
into the device pump, which awaits ready events and releases the span. Its
cost against the block sink is an arena floor of the largest tensor plus the
read working set and no overlap inside a tensor, which is why the transfer
manager is preferred wherever it exists.

## Tenth pass: loader design review (2026-09-05)

The review started from `093e065d` and used the earlier passes as constraints.
The core architecture holds: source coalescing, separate read and request
lifecycle credits, per-device pumps, reference-counted blocks, and caller-side
execution/concurrency each solve a distinct problem. Changing these algorithms
would discard measured decisions. The main simplification is to put their
boundaries and ownership in the code's structure.

### Design changes

- `mem.dma` owns `Workspace`, `BlockPool`, `Allocator`, `MapAllocator`,
  `BufferAllocator`, and `NumaPlacement` in `zml/mem/dma.zig`. Generic
  bufferization and `FixedBufferPool` remain in `mem.zig`. The workspace owns
  arenas; a pool is a load-scoped free-list/lease view. Duplicate arena lists,
  overlap scans inherited from external arena attachment, latest-arena state,
  and an unused retained-byte counter are removed.
- `io.zig` is the public facade. `io/loader.zig` starts with `Loader`,
  `Handle`, and `Window`, followed by shared preparation, handle/executable
  internals, and integration tests. Checkpoint lookup lives in
  `io/TensorStore.zig`; whole-tensor staging and its unit tests live in
  `io/buffered_loader.zig`. Both backends consume `backend.LoadSpec`; neither
  backend defines the other one's input contract. Sharding is resolved once
  by shared preparation.
- `direct_loader.Loader` now precedes its implementation. `Planner` owns
  coalescing and transfer planning; `Scheduler` owns FIFO publication and
  claims. Immutable `Job` and `Transfer` descriptors belong
  to `Batch.Plan`. Runtime names are `Pipeline`, `ReadRequest`, `Metrics`,
  `TensorTransfer`, and `SourceSlot`, without obsolete Fair/Vectored/Loader
  prefixes. The ineffective `cleaned` flag is removed: `destroy` frees self.
- `source_concurrency.zig` is a std-only policy module exposing `Parallelism`
  and `Controller`, with the existing curve/evidence tests. Runtime
  measurement/gates stay with the direct pipeline. `DispatchSpans.zig` is the
  dispatch container type, with nested `Span`.
- Calibration reads from public API into measurement, block selection, and
  private machinery. Workspace names replace stale plural source pools;
  private Benchmark prefixes and generic candidate `value` names are gone.
- Public API migrations: `Loader.Opts` -> `Loader.Options`, `.dma_pool` ->
  `.dma_workspace`, `mem.Dma*` -> `mem.dma.*`,
  `dma.default_benchmark_block_sizes` -> `dma.default_block_sizes`,
  `.minimum_transfers_per_device` -> `.minimum_transfers`, and
  `.confirmation_minimum_transfers_per_device` -> `.confirmation_minimum_transfers`.
  `io.max_load_*` limits become `io.limits.max_*`.
  Repository examples are migrated. `io.Loader`, `Handle`, `Window`,
  `TensorStore`, and `Parallelism` keep their public locations.
- Documented serialized front-end use and borrowed lifetimes. `Handle.isDone`
  means source reads/transfers finished; an await may still run executables.
  Backend byte counters remain there so the returned-by-value front-end
  Loader does not acquire a hidden stable-address requirement.

### Ownership fixes found by the review

- The PJRT-buffer allocator rounded `header + length` instead of reserving
  padding before the data. At 64-byte alignment and length 100 it allocated
  128 bytes but placed data at offset 64. It now reserves worst-case padding,
  aligns from the actual PJRT base, and keeps the header immediately before
  data so free can recover it. Tests cover alignments 1 through 4096,
  unaligned bases, multiple lengths, and size overflow.
- A canceled calibration sleep returned while transfer workers still borrowed
  the window's stack. Every exit now sets stop, releases the start gate, and
  joins the group. A CPU-backed cancellation regression covers teardown.
- Calibration reserved manager-list capacity after retrieving an owned PJRT
  buffer; append failure leaked it. Capacity is reserved before creating PJRT
  objects, making publication infallible.
- Recalibration consulted only the newest arena. A smaller arena added during
  loading could cause another calibration-ring allocation despite an older
  sufficient arena. `findArena` searches retained arenas before growing;
  its reuse test fills the budget and verifies the older arena remains usable.

### Remaining issues, separate from the structural refactor

- The prior passes' plugin-sensitive failure behavior remains: a manager can
  already have lost its definition event when another piece/finalization or
  `setBufferErrorUnknown` reaches it. The oneAPI abort and the `toSliceAlloc`
  sub-byte placement issue remain historical findings, not reproduced here.
- `pjrt.Event.await` returns immediately for an already-ready event without
  inspecting its error. Calibration uses this shared wrapper, so synchronous
  readiness can hide an error. This is a PJRT-wide follow-up; the wrapper was
  not changed as part of moving loader responsibilities.
- Failed ROCm arena allocation/publication leaves the host-node byte reservation
  inflated. A subsequent use of the retained workspace may choose a different
  node than its actual retained bytes warrant. Successful allocation/placement
  behavior is unchanged; failure-accounting repair is a separate follow-up.
- No new accelerator throughput measurements were made for this refactor.
  Existing performance evidence remains historical, and TPU/neuron/metal
  transfer-manager support still requires the runtime checks described above.

### Validation

On macOS arm64, using the repository's devenv toolchain and Xcode SDK:

- `bazel build //zml //examples/io //examples/llm` passed.
- `bazel test //zml:test //stdx:test //zml/tokenizer:test //vfs:test //examples/io //examples/llm --test_output=errors`
  passed all four test targets and built both examples: 303 cases passed,
  three unrelated attention cases skipped, zero failed. The five front-end
  integration tests exercised both CPU direct and buffered backends. All
  original direct-loader tests were retained.
- `.devenv/profile/bin/zig test zml/io/source_concurrency.zig` passed all 13
  policy tests independently.
- Zig formatting, Buildifier checking of `zml/BUILD.bazel`, and
  `git diff --check` passed.
- `bazel build //...` and `bazel test //...` both stopped in analysis because
  `//bin/zml-smi/platforms/nvml:nvml` has only Linux select branches, with no
  macOS configuration. That target was not changed.
- Before the refactor, core tests failed to compile on macOS because generic
  assertion diagnostics traversed Workspace/Handle/callback pointers and
  instantiated a CUDA-only module. Loader tests now compare pointer identity
  directly and check fallible creation as a void outcome; assertion semantics
  are retained and no CUDA module is needed for their error formatting.

## Eleventh pass: direct I/O as a loader option (2026-09-05)

The question was whether direct I/O can be a loader option that works
reliably with the current design. It can; three things stood in the way,
none of them the pipeline.

### What the storage does without the loader

A python probe (`preadv` into page-aligned anonymous memory, 16 MiB chunks,
`posix_fadvise(DONTNEED)` for the cold arms; `mincore` and `cachestat` both
refuse files this user cannot write, so residency was read from `Cached` in
`/proc/meminfo`):

| host | cold buffered | direct | warm buffered |
|---|---|---|---|
| gb300-2 (ext4 on a 4-NVMe raid0, 64 KiB pages) | 52.8 GiB/s | 53.2 (8 to 64 threads alike) | 120.9 |
| mi300 (ext4 on LVM over a 4-NVMe raid0 plus the boot NVMe) | 4.7 to 4.9 | 25.1 | 70.1 (32 threads), 43 (12) |

So direct I/O is the disk's ceiling on both hosts; cold buffered reaches it
on gb300-2 and gets a fifth of it on mi300 (readahead-bound through md and
LVM); warm buffered is 2.3 to 2.8 times the disk. Also found here: gb300-2
held only 16 GB of page cache at the start of this session, hours after
149 GB loads (the container has no memory limit; within a session the cache
retains the model, so whoever or whatever dropped it is unknown), which
means an earlier DeepSeek number there is warm or cold depending on what
ran before it (the two differ by about 10%: 4.1 against 4.5 s); and Llama
on mi300 lives on the volume's slow extent (its cold reads run at 3.3 GiB/s
buffered or direct, against 25 for DeepSeek on the same filesystem), which
makes it the wrong cold fixture there.

### Three defects, one of them old

1. **Bare paths never reached the file backend.** `VFS.lookupDir` sent an
   absolute path (and a relative one that is not a URI) to the inner `Io`,
   while `VFS.loadProfile` attributed the same path to the registered
   `file` backend. Every loader read of `/var/models/...` bypassed
   `vfs.File`, so its `direct_io: bool = true` had never applied to a load.
   Bare paths now go to the `file` backend when one is registered
   (`bareBackend`; a test opens a temp file by absolute path and checks the
   handle's backend).
2. **The old `vfs.File` failed the open when the filesystem refused
   `O_DIRECT`** (`fcntl` error -> `error.Unexpected`), and served streaming
   reads from the direct descriptor's own file position. It also depended
   on the reader lining up by luck: a direct read needs the offset, every
   buffer and the total aligned, and safetensors packs tensors at 16-byte
   boundaries (DeepSeek: 1566 of 1576 tensors in a file at exactly 16, the
   data section itself at 3096 mod 4096), so a read that starts at a
   tensor never qualifies.
3. **Zig's threaded `Io` treats `EINVAL`/`EFAULT` from `preadv` as a
   programmer error** (panic in debug, `error.Unexpected` in release).
   Those are exactly what a filesystem that accepts the flag but not the
   read, or a mapping the block layer cannot pin, answer to a direct read.

### The change

- `vfs.File.Config.direct_io` is a policy: `off`, `on` (every aligned read
  of a qualifying file), `auto` (default: `on` for a file that is mostly out
  of the page cache when opened). Residency comes from `cachestat`, or when
  the kernel refuses it (before 6.5, or a file this process may not write:
  both bench hosts) from 32 `RWF_NOWAIT` reads spread over the file, which
  answer `EAGAIN` for an uncached page and need no permission. Unknown
  counts as cached. A file is cold at or below `direct_io_cold_fraction`
  (0.5).
- The backend opens the direct descriptor itself and never fails the open
  over it; streaming reads use the buffered descriptor; a direct positional
  read is issued with raw `preadv` and its errno decoded: `EINVAL`, `EFAULT`
  and `EOPNOTSUPP` mark the file refused (one warning) and the read is
  answered by the buffered descriptor; a short read at the end of the file
  is returned as such.
- The backend publishes `ReadHints.direct_io_alignment` and a
  `DirectIoProbe` (`Backend.direct_io`); `VFS.loadProfile` forwards both
  (`LoadProfile.direct_io_alignment`, `LoadProfile.direct_io`, the probe
  translated to VFS handles).
- `Loader.Options.direct_io: bool = true` (`Config.source_alignment`): the
  planner widens each job's read to aligned bounds (`alignBackward` of the
  first tensor byte, `alignForward` of the last), transfers address the
  widened read, `maximumJobLen` leaves two alignment units of room so a
  widened job still fits `maximumCoalescedJobBlocks`, and
  `Batch.Plan.Job.minimum_len` tells the read how much must exist:
  `safetensors.readFilePositionalAllV` accepts the end of the file only past
  it. Neighbouring widened reads share up to one alignment unit; their
  transfers do not overlap. A file is widened only when the probe says its
  reads go direct (see the warm results below for why).
- The example exposes `ZML_DIRECT_IO=0|1` (loader) and
  `ZML_VFS_DIRECT_IO=off|on|auto` (backend); the `zml/vfs/file` scope logs
  at debug: the residency decision, the descriptor, the first unaligned
  read of a direct file.

### Results

`examples/io load <model> replicated`, 16 MiB blocks, adaptive width; arms
are `exact` (today's reads), `direct` (loader on, backend `on`), `auto`
(loader on, backend `auto`), `widened` (loader on, backend `off`: aligned
reads served buffered). Cold arms evict the model first. Loader `elapsed`,
then the read and DMA-stage milliseconds per 16 MiB request.

gb300-2, four GB300, three reps (two for DeepSeek):

| model, state | exact | direct | auto | widened |
|---|---|---|---|---|
| Llama-8B cold | 0.359 / 0.358 / 0.358 s (rd 6.4, dma 1.1) | 0.300 / 0.295 / 0.294 (rd 2.2, dma 15) | 0.300 / 0.299 / 0.295 | 0.357 / 0.358 / 0.358 |
| Llama-8B warm | 0.320 / 0.319 / 0.324 (rd 2.4, dma 17) | 0.299 / 0.297 / 0.298 (rd 2.2, dma 15) | 0.334 / 0.344 / 0.327 (dma 19 to 20) | 0.339 / 0.332 / 0.340 |
| DeepSeek cold | 4.48 / 4.59 (rd 5.0, dma 12) | 3.76 / 3.76 (rd 3.4, dma 22) | 3.74 / 3.46 | -- |
| DeepSeek warm | 3.83 (rd 2.7, dma 23) | 3.74 / 3.63 (rd 3.1, dma 21) | 4.29 / 4.36 (dma 24 to 26) | 4.06 / 3.95 |

Direct reads make the cold loads 12 to 16% shorter and leave the page
cache untouched (`Cached` stays at 174 GB where an `exact` load adds the
model). They also beat the warm buffered load by 5 to 6%: the loader is
not read-bound at 33 to 37 GiB/s, and the page-cache copy (`_copy_to_iter`,
1.3 cores of reader sys time) competes with the DMA engines for the same
host memory, which the DMA-stage time shows (17 -> 15 ms warm, and the
per-piece pump measurements of the sixth pass). `widened` costs 3 to 6% on
a warm load: with an exact read the first tensor of every job lands at
block offset 0, with a widened one at its file offset modulo 4 KiB, and a
DMA source that is not block-aligned is slower (the -13% of the pump
measurements). That is why `auto` widens only the files it will read
directly.

mi300, eight MI300X, host load 9 to 30 (other users; DeepSeek replicated
on ROCm is DMA-bound at 115 to 140 ms per request, a separate condition):

| model, state | exact | direct | auto | widened |
|---|---|---|---|---|
| Llama-8B warm | 1.12 / 1.10 s (rd 9.3, dma 72 to 74) | 4.57 / 4.53 (rd 58) | 1.11 / 1.07 | 0.96 / 0.94 |
| Llama-8B cold (slow extent) | 4.73 / 4.75 / 4.72 (rd 62) | 4.52 / 4.52 / 4.54 (rd 59) | 4.52 / 4.54 / 4.55 | -- |
| DeepSeek cold | 35.1 (rd 44, width 12) | 12.5 (rd 5.6, dma 117) | 13.0 (rd 5.7) | -- |

Direct I/O into ROCm's PJRT-pinned arenas (`hipHostMalloc`) works: no
refusal, verified loads (`ZML_LOAD_CHECK=16` on Llama, 256 on DeepSeek, on
both hosts). DeepSeek cold is 2.8 times faster and then DMA-bound. Forcing
direct on a warm Llama costs 4x here because its files sit on the volume's
slow extent, and `auto` chose buffered for it (probe: 100% cached) and
direct for the cold DeepSeek (0%).

With the probe (a file is widened only when its reads go direct), the
same arms again; each row's cache state was checked from `Cached`, the
label notwithstanding (`exact` r1 of each warm block was in fact the run
that warmed the cache, after the previous block's eviction):

| host, model, state | exact | auto | direct |
|---|---|---|---|
| gb300-2 Llama warm | 0.318 / 0.321 s | 0.328 / 0.333 / 0.314 | 0.300 / 0.298 / 0.296 |
| gb300-2 DeepSeek warm | 4.10 | 4.69 / 3.95 | 3.79 / 3.77 |
| gb300-2 DeepSeek cold | 4.58 | 3.73 | -- |
| mi300 Llama warm | 1.10 | 0.97 / 1.01 | -- |
| mi300 DeepSeek cold | 34.9 | 12.8 | -- |

`auto` on a warm Llama is now at parity with `exact` on both hosts (it was
4 to 7% behind on gb300-2 without the probe); DeepSeek warm on gb300-2 is
too noisy in this session (3.9 to 4.7 s for the same arm) to resolve a few
percent either way. Cold loads keep the direct gain.

### Recommendation and defaults

Loader `direct_io = true` and backend `auto` are the defaults. On a host
whose disk is as fast as its cold buffered path (gb300-2) a user who knows
the model is on that disk gains another 5% on warm loads with backend
`on`; on a host with a slow extent or a slow disk `on` is a 4x loss, so it
stays opt-in. Residency detection is a measurement, not a guess:
`cachestat` where permitted, `RWF_NOWAIT` sampling otherwise, buffered when
unknown. Nothing in the pipeline changed: widening is a planner concern,
the read helper learned that padding may end early, and the file backend
owns the descriptor and every failure mode.

## Twelfth pass: direct I/O as the VFS's own local case (2026-09-06)

A review of the eleventh pass asked three questions: whether the pair of
descriptors behind every `vfs.File` handle earns its keep when the loader
uses one or the other; why `DirectIoProbe` needs opaque data and function
pointers; and whether `vfs.File` has a purpose without the direct logic.
The answers led to a smaller design with the same read path.

### What the eleventh-pass shape was paying for

- **The pair.** From the loader's side a file was already one or the
  other: the policy and the residency measurement decided at open, the
  planner asked once per file and widened the whole file, and every loader
  read of a direct file was aligned by construction. The buffered twin
  served the readers whose reads never align, the safetensors header parser
  (streaming), `TensorReader` and the buffered loader (exact offsets), plus
  the refusal fallback. The pair was the cheapest way to give those readers
  a working handle, because `O_DIRECT` is a property of the open file
  description (a `dup` shares it, hence the second `open`), and a single
  descriptor would have had to bounce every unaligned read.
- **The probe.** `Backend` is a type-erased registry entry (an `std.Io`,
  itself userdata plus vtable, and side channels); the VFS stores
  heterogeneous backends by scheme and the loader takes a `LoadProfile`
  that must serve `hf`, `s3` and a null probe alike, so nothing on that
  path could name `vfs.File`; `ReadStatsProvider` set the precedent. The
  answer is per file and exists only after the open (residency at open,
  refusal at the first read), and `std.Io.File` carries only a handle and a
  `nonblocking` bit, so there was no return channel. The loader holds VFS
  handles and the backend its own, hence the two probe layers.
- **The file backend.** Created 2025-12-11 (#362) and given direct I/O a
  week later (#367): the two were never apart. Its read hints equal
  `LoadProfile.local`, every operation forwards to the inner `Io`, and its
  handle table mirrored the VFS's own entry for every open file (a second
  index translation under a second mutex on every read). Until the
  eleventh pass bare paths bypassed it, so it only ever served `file://`.
  Under `auto` it also opened a direct descriptor and measured residency at
  every read-only open of a qualifying file, including the registry
  parser's, which only ever read the header and closed.
- **Rejected on the way:** moving the mechanism into the loader with a
  `local_io: ?std.Io` in the profile so the loader could open local files
  outside the VFS. The VFS hides exactly one thing, the descriptor, and the
  mechanism is four descriptor operations (`fcntl` for the flag,
  `cachestat` or `preadv2(RWF_NOWAIT)` for residency, a raw `preadv` whose
  errno is decoded, `fcntl` again on refusal). It has to live where the
  descriptor is visible, and bypassing the VFS was the price of moving it,
  not a need. Once the VFS keeps it, the loader never needs a descriptor.

### The change

- `vfs/file.zig` is deleted. A bare path or a `file://` URI is the VFS's
  own local case: `lookupDir` resolves the `file` scheme and bare paths to
  the inner `Io` (`schemeRoot`, `localRoot`; registering a `file` backend
  is asserted against). `loadProfile` returns the local profile for them:
  8 MiB requests, `direct_io_alignment = 4096` on Linux, `vfs = self`.
  `LoadProfile.local` (the one no-VFS profile, and the loader's default)
  keeps a null alignment, so a loader without a VFS reads exact ranges as
  before.
- One descriptor per local handle, with a state on the VFS's existing
  handle entry: `undecided`, `buffered`, `direct`. `VFS.useDirectIo(file,
  policy)` is the planner's one call per file: it applies the policy,
  checks read-only (recorded at open), regular and at least one alignment
  unit long, samples residency on that descriptor under `auto`, sets the
  flag with `fcntl`, and returns whether the file is direct. The first
  decision under `on` or `auto` stands for the handle's life: the flag
  belongs to the open file description, which the loader's `SourceSlot`
  keeps for its life, and a caller answered buffered plans exact reads
  that a direct descriptor would reject. The transitions that touch the
  flag (`enterDirect`, `leaveDirect`) run under the VFS mutex so the flag
  and the state change together; reads only load the state.
  `vfs/direct_io.zig` holds the descriptor operations.
- The planner finds the VFS through the `Io` it opened the file with
  (`VFS.fromIo`, vtable identity), not through a pointer in the profile:
  a profile prepared by one VFS and a loader reading through another `Io`
  would otherwise index the wrong handle table. A loader without a VFS
  never widens.
- Reads of a direct handle go through the raw `preadv` in the VFS. A
  rejected read (`EINVAL`, `EFAULT`, `EOPNOTSUPP`) takes the file back to
  buffered: the flag comes off before the inner `Io` sees the descriptor
  again, one log line names the file, offset, buffer and total, and the
  read is answered buffered. An aligned rejection is the filesystem
  refusing direct I/O it accepted the flag for (a warning); a misaligned
  one is the reader's, its plan or a continuation after a short read,
  which a FUSE or NFS store may answer mid-file (an error). Demotion
  rather than failure, because that continuation is legitimate and a
  single descriptor can only finish it buffered. A streaming read, or a
  file copy reading the handle, demotes it the same way first, so the
  inner `Io` never sees the flag (it treats that errno as a programmer
  error). There is no per-read alignment routing any more: a handle is in
  one mode at a time.
- Residency is sampled only (32 `RWF_NOWAIT` reads spread over the file).
  The eleventh pass preferred `cachestat`, but a whole-file `cachestat`
  walks every cached folio and costs about 7 ms per cached GiB on a 4 KiB
  page kernel, unbounded in file size and exposed on the planner thread,
  and neither bench host ever exercised it (both refuse it for files the
  user cannot write). The sample resolves a 0.5 threshold and costs the
  same for any size.
- The policy is the loader's. `Loader.Options.direct_io: VFS.DirectIo`
  (`off`, `on`, `auto`; default `auto`) replaces the loader's `bool` and
  the backend's `Config.direct_io`; `backend.Config.direct_io` replaces
  `source_alignment`, which the direct loader now derives from the policy
  and the profile; `publishFiles` takes the policy instead of a probe.
  `ZML_DIRECT_IO=off|on|auto` replaces `ZML_DIRECT_IO=0|1` plus
  `ZML_VFS_DIRECT_IO` (an old value is an error that names the accepted
  ones); both examples stop registering a file backend. Removed:
  `DirectIoProbe`, `Backend.direct_io`, `ReadHints.direct_io_alignment`,
  `LoadProfile.direct_io`. `registerBackend("file", ...)` returns
  `error.ReservedScheme` in every build mode.
- `safetensors.readFilePositionalAllV` returns after a short read once
  `minimum` is in: only padding no caller addresses is left, and the call
  that would fetch it starts at an unaligned offset, which a direct file
  rejects (the pair used to absorb it on the buffered twin). For an
  unwidened read `minimum` is the whole read, so nothing changes for remote
  backends or `TensorReader`.
- Local `realPath` results are bare again (the eleventh pass routed bare
  paths through the `file` backend, which prefixed them with `file://`, so
  `Tensor.file_uri` carried the scheme for one commit). Nothing in the tree
  keys on the prefix.
- Downstream: llmd registers `zml.io.VFS.File`
  (`monorepo/llmd/main.zig:236`) and drops those two lines when it picks
  this up.
- Known limits, recorded rather than fixed: a long-lived loader that found
  a file cached at its first submit reads it buffered at every later
  submit, even after the cache was dropped, because the decision is per
  description and `SourceSlot` keeps one per file (a fresh open per submit
  would let `auto` re-measure); the alignment is a 4 KiB constant rather
  than `statx(STATX_DIOALIGN)`, so a filesystem with a larger logical
  block gets one rejected read per file and then buffered reads; the VFS's
  streaming-write operation still forwards the VFS handle untranslated
  and its seeded stdio entries map index 1 to stdin, a pre-existing pair
  of defects that only cancel out for stdout and stderr.

Eight review angles (line scan, removed behaviour, cross-file trace,
reuse, simplification, efficiency, altitude, conventions) ran over the
diff before the second round of changes above. They found the flag race
that the mutex now closes, the unchecked `LoadProfile.vfs` that `fromIo`
replaces, the `cachestat` cost, the assert on `registerBackend`, the test
assertions that encoded ext4's rejection of unaligned direct reads (tmpfs
serves them), and the cleanups folded in: `innerFile()` at every
forwarding site, the `refused` state that no reader distinguished from
`buffered`, one demotion helper instead of two, `loadProfile` classifying
paths through `lookupDir` and deriving the local profile from
`LoadProfile.local`, `schemeRoot` using the map lookup.

### What `auto` does to the page cache

A direct read never fills the page cache. Under `auto` a cold file is read
directly and is therefore still cold at the next load: a fixed point.
Whether that costs anything is a property of the host, not of the file:

| host, model | cold buffered, then the next load warm | direct, every load |
|---|---|---|
| gb300-2 Llama | 0.359 s, then 0.320 | 0.297 |
| gb300-2 DeepSeek | 4.5, then 3.8 | 3.7 |
| mi300 Llama | 4.73, then 1.10 | 4.52 |

(eleventh-pass numbers). On gb300-2 direct beats the warm buffered path,
so never warming the cache costs nothing. On mi300 `auto` trades a 4% gain
on the first Llama load for losing the 1.10 s warm load on every load after
it. No residency measurement can see that, because the question is whether
this host's disk beats its warm buffered path for this loader, which the
user knows and the file does not. The default stays `auto` (a first load
never slower than cold buffered, the cache left as found); a user who
reloads the same model on a host whose warm buffered reads beat its disk
should set `off`. The option's doc comment says so.

### Verification

- `bazel test //vfs:test //zml:test` and `bazel build //examples/io
  //examples/llm` pass. The VFS test opens a temp file on the local ext4
  under every policy: `on` goes direct (the unaligned read of the test is
  rejected by the kernel, answered buffered and logged once), `auto` reads
  the just-written and therefore cached file buffered, `off` never asks,
  a streaming read demotes a direct file deterministically.
- gb300-2 (2026-09-06, tree before the review round, in a detached
  worktree `~/github/zml/zml-directio` at the host's `b59c41b7` plus the
  57 changed files; kernel 6.8 64 KiB pages; four GB300, host idle):
  `ZML_LOAD_CHECK=16` under `on` warm and cold, and `ZML_LOAD_CHECK=256`
  on cold DeepSeek under `auto` twice, all `load check: ok`; no refused
  read, no `O_DIRECT refused`, no panic in 15 runs. `bulk phase` elapsed:

  | arm | this tree | eleventh pass |
  |---|---|---|
  | Llama warm `on` (3) | 0.301 / 0.301 / 0.301 s | 0.297 |
  | Llama warm `off` (3) | 0.337 / 0.334 / 0.343 | 0.320 |
  | Llama warm `auto` (3) | 0.341 / 0.334 / 0.336 | 0.328 / 0.333 / 0.314 |
  | Llama cold `auto` | 0.303 | 0.30 |
  | Llama cold `off` | 0.360 | 0.36 |
  | DeepSeek cold `auto` (2) | 3.99 / 3.89 | 3.73 |

  `auto` decided buffered for the warm files (`100% cached`, all four
  shards) and direct for the cold ones (`0% cached` on Llama, whose files
  the user owns so `cachestat` answered; `3% cached` on DeepSeek, which
  is the one header page out of 32 samples). Direct and cold arms are at
  parity. The two warm buffered arms are 5% above yesterday's numbers on
  the same host although their code path lost a handle table and a mutex;
  the cold buffered arm is at parity, so the difference is in the warm
  page-cache copy, whose speed depends on which node holds the cache
  (seventh pass). The DeepSeek pair calibrated to 8 MiB and 16 MiB blocks
  respectively (175 vs 181 GiB/s), which is calibration noise, not the
  read path.
- gb300-2 closing A/B (2026-09-06, the final tree after the review round
  in the same worktree, against the eleventh-pass tree in a second
  detached worktree `~/github/zml/zml-eleventh`; both built from
  `b59c41b7` plus their overlays; Llama replicated on four GB300, warm,
  four interleaved rounds of baseline exact, final `off`, baseline auto,
  final `auto`). `bulk phase` medians:

  | arm | eleventh pass | final tree |
  |---|---|---|
  | warm exact / `off` | 338.8 ms (334.9 to 341.6) | 335.8 (334.3 to 339.0) |
  | warm `auto` (buffered) | 339.6 (332.4 to 357.7) | 336.8 (334.2 to 341.3) |
  | cold `auto` (direct) | 295.4 | 301.6 |
  | warm `on` + check | -- | 302.1 |

  Per-read metrics are identical within noise (read 2.30 to 2.36 ms,
  DMA stage 18.3 to 18.6 ms). The warm buffered path is not slower: the
  final tree is 3 ms ahead on both pairs, inside the round-to-round
  spread, so the 5% against yesterday's numbers is the host's day, not
  the change. The baseline's slowest warm run (357.7 ms) had calibrated
  to 8 MiB blocks. Both `ZML_LOAD_CHECK=16` runs under `on` (warm and
  cold) passed; no refused or misaligned read in 20 runs. The cold `auto`
  run showed why the residency sample now sits mid-window: the sample at
  offset 0 was the header page the registry parser had just read, so a
  cold file reported 3% cached (the decision was unaffected).

## Thirteenth pass: loader-owned `loadExecute` concurrency (exploration, 2026-09-10)

Not a change: an exploration the user asked for, with its evidence. The
question: `loadExecute` needs device memory for its inputs (and for the
execution itself) beyond the model, which is why the caller sizes a
`Window`. The executable can report what it needs, so the loader could admit
submissions itself, execute them as soon as their inputs land, and expose
less concurrency control. Tree: `fa232bb2 direct io` (the twelfth pass,
committed by the user), read-only apart from a temporary log line in the
playground's `planPacks` that printed `PJRT_Executable_GetCompiledMemoryStats`
and `PJRT_Device_MemoryStats` (removed again).

### What the loader does today

- `loadExecute(bindings)` plans one submission; `Handle.await` runs the
  executables in binding order on the awaiting task with `.wait = true`,
  writes the outputs and frees the inputs (`zml/io/loader.zig`
  `HandleState.await`, `BoundExecutable.execute`). `zml.io.Window{budget_bytes,
  max_handles}` is the only admission control: `submit` awaits the oldest
  handle until the next submission fits, sized by
  `Loader.executeInputBytesPerDevice(exe)`, which sums the inputs'
  per-device placements and knows nothing about temporaries or the output.
- llmd Laguna submits one `loadExecute` per sparse layer (both packs) through
  a window whose budget is `--expert_pack_budget`, default 0, i.e. a window
  of one: submit layer k, await it (reads, then execute), submit k+1. The
  read pipeline drains at every layer.
- Device memory for a submission is not taken at submission. Every tensor
  allocates its PJRT buffers at the first read job that touches it
  (`direct_loader.zig` `Item.ensureState` from `ReadRequest.run`,
  `TensorTransfer.initResolved` -> `createBuffersForAsyncHostToDevice`), and
  jobs are claimed in strict FIFO order across submissions. Submitting many
  `loadExecute` handles therefore costs host metadata only; the device
  footprint of not-yet-executed inputs grows with the reads, and the reads
  run at most `width` jobs ahead of the batch whose await has not returned.
  What the `Window` bounds is the number of landed-but-unexecuted
  submissions the caller lets accumulate.

### What the consumer looks like (monorepo `master` a64fd7a9, 2026-09-10)

Read in a detached worktree of `~/github/zml/monorepo` at `master`; the
working copy there is still the `loader-third-pass` branch (`d426dde4`).

- `master` builds against the pre-rework zml (`origin/master` `f8ddb3e5`,
  `zml/io.zig`): `Loader.loadExecute(arena, io, tensor, buffer, store,
  shardings, exe, opts)` is synchronous. It spawns one `loadSingle` per
  source into the loader's single `LimitedGroup` (`parallelism` 16 in
  llmd), awaits the whole group, which also holds any bulk `load` tasks in
  flight, runs the executable with `.wait = true`, writes the output and
  frees the inputs before returning. One fused tensor's inputs at a time on
  the device, by construction; the read pipeline stops for every one.
- There is exactly one `loadExecute` call site: `llmd/weights.zig:985` in
  `loadPacked`, which walks the model's tensors in declaration order and
  submits every tensor that has a `Packer` recipe, one at a time.
  `weights.loadInto` (`weights.zig:1078-1095`) runs `loadPacked` first,
  then the bulk `loader.load`; `models.zig:254` awaits once. Recipes:
  `fuse` (head-interleaved QKV, gate/up), `stack` (experts),
  `concatenate_rows`, `convert_rows`, `dequantize_blocks` and the NVFP4
  cutlass layouts. The pack executables are compiled lazily inside that
  loop (`weights.zig:990-1069`, deduplicated by recipe and shapes), so a
  handful of XLA compiles sit on the load's critical path with nothing in
  flight.
- Every model packs, not only the MoE ones: llama, gemma3_text, ministral3,
  muse_glimmer and dflash_drafter fuse QKV and gate/up (2 per layer);
  lfm2 fuses `in_proj_b/c/x`, QKV and `w13`; gemma4_text, qwen3_5 and
  laguna add expert stacks (4 to 5 per layer); deepseek4 adds fp8 block
  scale conversion, weight+scale concatenation and block dequantization
  (many per layer). On Llama-3.1-8B the packed tensors are about 8.5 of
  the 15 GiB (QKV 48 MiB and gate/up 224 MiB per layer), so the
  sequential path carries most of the bytes of a dense model.
  `laguna_dflash.zig:793` loads its target with a bare `loader.load`,
  bypassing the packer (inconsistent with the other dflash wrappers).
- Device memory is budgeted before the load, not by it: `Model.init`
  compiles every executable, then `CacheSpec.initWithAllAvailableMemory`
  (`attention.zig:838-866`) sizes the KV cache as
  `bytes_limit x cache_memory_fraction (0.95) - fixed_memory`, where
  `fixed_memory` is the model's bytes per device plus the attention
  working set (`llama.zig:349`), and allocates it (`llama.zig:633`) before
  `Loader.init` (`main.zig:429`). `bytes_limit` is `gpu_memory_fraction`
  (0.9) of the device (`mem.zig:6`; CPU has no stats and falls back to
  16 GiB). Nothing reserves the packing transient (inputs plus `temp`):
  it lives in the 5% the cache leaves plus whatever the attention estimate
  overshoots, about 8 GB on a 192 GB MI300X. That is what made the
  sequential `loadExecute` the safe choice, and it is also exactly the
  headroom a loader would read back from `bytes_limit - bytes_in_use`
  minus its own unlanded weights.
- Nothing else runs on the devices during the load: no compile/load
  overlap (`models.zig:441-491` joins before the load), the tokenizer load
  is the only concurrent task and it is joined first, servers start after.
  No `loadExecute` output feeds another; the only ordering is packed
  before bulk and `loader.await` before `finalizeLoadedBuffers`.
- Consequence for a migration: the `loader-third-pass` branch predates
  `weights.zig` (it windowed Laguna's expert packs only, with
  `--expert_pack_budget`); the handle API has to be adopted in
  `loadPacked`, which can submit per tensor or per layer, and `master`
  has no `Window` to delete.

### What PJRT and XLA allow

Read in the local openxla checkout `~/github/openxla/xla` at `b014a9c1`
(2026-07-27, 25 days after zml's pin `41370d1124`); the shipped CUDA plugin
is manual-2026-07-31.

- `PJRT_Executable_GetCompiledMemoryStats` is implemented by the CPU client
  (`xla/pjrt/cpu/cpu_client.cc:1218`) and the stream-executor client for a
  single-program executable (`xla/pjrt/se/stream_executor_executable.cc:174`;
  MPMD returns Unimplemented). `compiled_memory_stats.cc:34-105` classifies
  the buffer assignment of one partition: entry parameters -> `argument`,
  live-out -> `output`, preallocated temporaries -> `temp`, plus host-memory
  variants. zml already binds it (`pjrt.zig` `Executable.getCompiledMemoryStats`,
  reached through `LoadedExecutable.executable`), unused so far.
- `PJRT_Device_MemoryStats` (`bytes_in_use`, `bytes_limit`,
  `largest_free_block`, `pool_bytes`) is the BFC allocator on the
  stream-executor GPU clients (`xla/pjrt/gpu/se_gpu_pjrt_client.cc:1714`); the
  CPU client has no `GetAllocatorStats`, so `Device.memoryStats()` returns
  zeroes there (`platform.zig:213`) after the plugin logs
  `Unimplemented: GetAllocatorStats is not supported` on every call.
- An async host-to-device manager allocates its device buffers synchronously
  when created (`host_to_device_transfer_manager.cc:137` ->
  `pjrt_stream_executor_client.cc:505-530`, `retry_on_oom = true`).
- Execute on inputs whose transfers have not landed: `ExecutePrepare`
  (`common_pjrt_client.cc:1912-1945`) collects the inputs' definition events
  as `extra_deps` and allocates the outputs as delayed memory
  (`AllocateRawBufferForExecute`, `pjrt_stream_executor_client.cc:540-550`),
  materialized only when the launch runs. The raw launch
  (`pjrt_stream_executor_client.cc:1866-1880`) is scheduled on the device's
  `async_dispatch_thread` when there is one and the call returns at once;
  otherwise it runs inline, and `BufferSequencingEvent::WaitForEventOnStream`
  (`buffer_sequencing_event.cc:58-61`) blocks the calling thread with
  `BlockUntilReady` until every input event has been recorded, that is,
  until the pump has submitted each input's last piece. The dispatch thread
  exists when `use_async_dispatch` is set (`se_gpu_pjrt_client.cc:1795-1800`,
  default off, env `PJRT_GPU_ENABLE_ASYNC_DISPATCH=1`) or the C API option
  `use_tfrt_gpu_client` is true (`plugin/xla_gpu/xla_gpu_pjrt_client.cc:27-31`,
  `pjrt_c_api_gpu_internal.cc:191-229`). zml passes
  `use_tfrt_gpu_client = gpu_async_dispatch` (default true) for CUDA only
  (`zml/platform.zig:792,836`); ROCm and oneAPI get the blocking form. The
  CPU client always defers the launch (`cpu_client.cc:1945`,
  `ExecuteWhenReady` on the input events). So "execute at submission and
  let PJRT order it" is host-free on CUDA and CPU today, and would block
  the submitting task inside `Execute` on ROCm/oneAPI unless the same
  option is passed there (untested).
- Unchanged from the third pass: `PJRT_Buffer_Destroy` while an execution
  references the buffer is safe; concurrent `Execute` on one executable is
  undocumented, so executions must stay on one task.

### Measurements (2026-09-10)

Executable memory stats of the playground pack executable (`stackPack`:
`Tensor.stack` of `width` rank-2 sources, replicated output), one call per
distinct source shape, `ZML_LOAD_PACKS=2 ZML_LOAD_PACK_WIDTH=16`:

| platform | executable | argument | output | temp |
|---|---|---:|---:|---:|
| B70 host, CPU x4 (Qwen3.5-4B) | 16 x {9216,2560} bf16 | 720 MiB | 720 MiB | 2.72 GiB |
| B70 host, CPU x4 | 16 x {2560,9216} bf16 | 720 MiB | 720 MiB | 2.72 GiB |
| gb300-2 GPU 1 (Llama-3.1-8B) | 16 x {14336,4096} bf16 | 1.75 GiB | 1.75 GiB | 0 |
| gb300-2 GPU 1 | 16 x {4096,4096} bf16 | 512 MiB | 512 MiB | 0 |

`alias`, `generated_code` and every host figure were 0. The numbers are per
partition, i.e. per device. On the GPU the stack is a copy into the output;
on XLA CPU it needs 3.8x the output in temporaries, so a CPU pack of 720 MiB
on four replicated devices holds 4 x (0.7 + 0.7 + 2.72) GiB while it
executes. `executeInputBytesPerDevice` cannot see that term; the
executable can. Device stats: CUDA `bytes_limit` 248.96 GiB,
`bytes_in_use` 0 before the load (`largest_free_block` and `pool` 0 until
the BFC pool grows); CPU nothing.

CPU anatomy of one pack (same run, Qwen3.5-4B, cold files read direct,
window 1): batch 0 (pack 0, 720 MiB) done at +0.201 s; batch 1 published at
+1.476 s, so the caller spent 1.27 s executing pack 0 on four CPU devices
between the two submissions; pack phase 2.935 s for 1.41 GiB (0.48 GiB/s)
against the bulk 7.27 GiB in 0.777 s (9.36 GiB/s). On CPU the execution is
six times the read, the opposite of the GPU case below.

Window sweep, gb300-2, GPU 1 alone (`CUDA_VISIBLE_DEVICES=1`; GPU 0 held
19 GB of another user's job), Llama-3.1-8B-Instruct warm (`auto` chose
buffered), `ZML_LOAD_PACKS=64 ZML_LOAD_PACK_WIDTH=16`: 14 packs of 16
sources (13.00 GiB) then the 1.96 GiB bulk remainder; three interleaved
rounds of `ZML_LOAD_PACK_WINDOW` 1, 2, 4, 8 in the `zml-directio`
worktree (tree `fa232bb2` plus the temporary log line):

| window | pack phase (3 rounds) | GiB/s | `Loaded weights` wall |
|---:|---|---:|---|
| 1 | 304 / 294 / 292 ms | 42.7 to 44.6 | 571 / 553 / 551 ms |
| 2 | 246 / 245 / 245 | 53.0 | 507 / 507 / 505 |
| 4 | 236 / 236 / 236 | 55.0 | 498 / 499 / 499 |
| 8 | 236 / 237 / 236 | 55.0 | 498 / 500 / 497 |

The bulk remainder read at 54 to 57 GiB/s in every arm. A window of one
costs 20% of the pack phase: about 4 ms per submission of drain, plan,
execute and ramp for packs that read in 17 ms each. Window 2 recovers
85% of it; window 4 is at bulk speed and window 8 adds nothing. This is
the 5090 result of the third pass (0.63 -> 0.60 s) with a clearer knee.

### The design space

The cost the loader must bound, per device: the inputs of every submission
that has started reading and not executed (held from first read until the
execution frees them), plus `temp + output` of the one executing. Everything
in it is known before submission: inputs and output from shapes and
shardings, `temp` from `GetCompiledMemoryStats`. What the loader can learn
about the room: on GPUs `bytes_limit - bytes_in_use` minus the bytes of
admitted tensors that have not been allocated yet (a per-device counter the
backend would keep, one atomic add in `initResolved`, because `bytes_in_use`
already contains the lazily allocated part) minus a reserve; on CPU
nothing, so a fallback depth.

- A. The window moves into the loader, execution stays on the caller's
  task. `loadExecute` admits by awaiting the oldest handles (running their
  executables, as `Window.submit` does now) until the new submission fits
  a budget the loader derives: headroom where the device reports it, else
  a depth. Deletes `zml.io.Window`, `executeInputBytesPerDevice`, llmd's
  `--expert_pack_budget` and `max_expert_pack_handles`, the playground's
  `ZML_LOAD_PACK_WINDOW`. No task, no new PJRT usage. Execution is not
  "as soon as landed" but at the next admission or await; the sweep says
  that costs nothing once two to four submissions overlap. Transient
  memory: up to the budget.
- B. An executor task in the loader awaits `loadExecute` batches in FIFO
  order, executes each as it lands, frees its inputs and signals the
  handle; `loadExecute` blocks on a condition while admitted-but-unexecuted
  bytes would exceed the budget; `Handle.await` waits for the executor's
  flag. Same throughput as A at the same depth, the minimum transient
  (one executing submission plus the read-ahead), and the caller never
  runs executables, so it can submit everything and do other work.
  Costs one task, cross-task state (`delivered`, failure, `deinit`
  stopping it) and the "execution happens elsewhere" semantics the third
  pass avoided. Its natural extension, E: one `loadExecute` for every
  layer with per-binding readiness counters, executing each binding as
  its own inputs land; more coalescing and no planning gaps, at the price
  of a per-binding counter on the DMA completion path.
- C. Device-ordered execution: at submission force `ensureState` for the
  bindings' inputs, enqueue the execute at once, drop the host references
  to the inputs and await the output's ready event in `Handle.await`.
  No loader task at all on CUDA and CPU (PJRT's dispatch thread or the
  CPU deferral does the waiting; outputs and temporaries are allocated at
  launch). Needs eager input allocation (the inputs occupy memory for the
  whole FIFO delay), a per-platform gate (blocking on ROCm/oneAPI unless
  `use_tfrt_gpu_client` is passed there too), completion callbacks to
  release the budget, and a new error path (a failed read reaches the
  caller as an errored output). Buys only the latency of freeing inputs
  over B; not worth its risk now, recorded because the facts above were
  not in the tree before.
- D. Change nothing in the loader; give llmd a default window of two to
  four. The cheapest way to collect the 20%, and it keeps every knob.

### Recommendation (the user's decision)

Confirmed by the rethink below after two corrections from the user: the
cache belongs after the weights (`master` allocates it before), and
machines where the cache would be small must be supported. A with the
derived byte budget (headroom from device stats minus unlanded weights,
cost = inputs + `temp` + output, sequential when the room is one pack,
depth 1 without stats), plus the ordering and compile items of the
rethink; B and C are not needed.

### Rethink: the cache belongs after the weights (2026-09-10)

The user's correction to the consumer survey: the KV cache should be
allocated after the weights are loaded, not before as `master` does today
(`Model.init` sizes `cache_spec`, compiles, then `State.init` ->
`KvCache.initBuffers` -> `Buffer.uninitialized`, `llama.zig:1118-1147`,
`attention.zig:79-87`, all before `Loader.init` at `main.zig:429`). That
ordering is llmd's to fix; the loader design below assumes the intended
order.

- Room during the load is then the device minus the landed weights, at
  least the eventual cache size: tens of GB on every GPU here, against
  pack transients of 0.2 to 1.75 GiB. Depth is bounded by the pipeline,
  not by memory: reads run at most one read width ahead of the submission
  being awaited, and a GPU pack executes in about a millisecond as it
  lands. Memory binds only on CPU (no allocator stats, host RAM is the
  device, one Qwen pack replicated on four devices costs 4 x (0.7 + 0.7 +
  2.72) GiB = 16.5 GB on the 62 GB B70), where depth 1 stays right.
- On this host a plain depth would do, but the user's follow-up stands:
  machines where the cache would be small must be supported properly, and
  there the room during the load is the future cache plus the 5% margin,
  possibly a few GB against packs of 1 to 3 GiB. So `A` keeps the derived
  byte budget of the design space: headroom = `bytes_limit - bytes_in_use`
  minus the loader's own unlanded weights (a per-device allocated-bytes
  counter, one atomic add in `initResolved`) minus a reserve; a submission
  costs `inputs + temp + output` per device (`temp` from
  `GetCompiledMemoryStats`, the term the fp8 `dequantize_blocks` recipes
  may carry); the loader awaits the oldest handles until the next fits and
  always admits one, which degrades to `master`'s sequential order when
  the room is one pack. Depth 1 where the device reports no memory (CPU).
  The transient is released by `awaitAll`, before the app sizes the cache.
  The budget does not depend on the allocation order: with the cache
  allocated first (as `master` does) the measured room is the residual
  margin, about 8 GB on a 192 GB MI300X and 12 GB on a GB300, which still
  overlaps dense packs four deep and fp8 packs one or two deep, and admits
  one submission when the room is smaller than that, i.e. the sequential
  order with the risk `master` already has. Allocating the cache after the
  load only widens the room. Runtime temporaries are one contiguous XLA
  allocation each, so the holes the load leaves matter near the margin;
  at the depths a tight device allows they are the reused hole of the
  previous pack.
  `B` and `C` buy nothing here: early freeing shortens the hold by
  milliseconds, and the caller has nothing else to do during the load
  (`master` compiles everything before it); `D` keeps a knob whose value
  the loader can measure.
- Fragmentation is the one way overlap could still hurt a cache sized
  from what is free after the load. Measured on gb300-2 GPU 1 (Llama, 14
  packs of 16, `PJRT_Device_MemoryStats` right after the bulk phase,
  before any output is freed, two rounds each; BFC pool = `bytes_limit`
  = 248.96 GiB):

  | window | pack phase | bytes_in_use | largest_free_block | num_allocs |
  |---:|---|---:|---:|---:|
  | 1 | 292.8 / 293.7 ms | 15.04 GiB | 232.12 / 232.12 GiB | 347 |
  | 8 | 240.4 / 235.7 ms | 14.99 / 15.05 GiB | 231.90 / 231.65 GiB | 347 |

  Free memory is 233.9 GiB in both arms; the sequential order leaves
  1.8 GiB of it outside the largest block, the overlapped order 0.2 to
  0.5 GiB more. Against the 5% that `cache_memory_fraction` keeps back
  (12 GiB on this device) that is noise; the KV buffers are per-layer
  allocations that fit the tail region either way.
- What the freedom allows and where it stops: a dense model could submit
  every packed tensor in ONE `loadExecute` (maximum coalescing, the
  executes run at the end while the bulk queued behind keeps the pipeline
  busy), because holding all pack inputs at once is 8.5 GiB on Llama-8B;
  a MoE model cannot, since its pack inputs are most of the model. Per-layer
  submissions with a depth of two to four work for both, so that is the
  general shape.
- The remaining work is ordering and compile, on the llmd side plus one
  loader change: `master` loads packed tensors before the bulk, and the
  handle API's `load` refuses transformed tensors that were not yet
  awaited (`prepareModelLoad`, `TransformedTensorNotDelivered`), so the
  bulk cannot queue behind the packs without a drain; letting `load` skip
  transformed tensors whose `loadExecute` was submitted (the `delivered`
  entry exists) removes the last drain. `master` also compiles the pack
  executables lazily inside the load loop with nothing in flight;
  compiling them in `Model.init` with the rest (they are deduplicated by
  recipe and shape) takes them off the critical path.

- Decided with the user (2026-09-10): the per-submission `Handle` goes.
  It existed so the caller could sequence `loadExecute`; with admission
  inside the loader nothing needs it. Every in-repo `load` user awaits its
  handle on the next line (`examples/llm` models, `examples/mnist`,
  `zml/testing.zig`), the playground's `Window` and `bulk.await()` were
  the control itself, and monorepo `master` already has one
  `loader.await`. Surface: `load`, `loadExecute`, `awaitAll`,
  `bytesLoaded`, `deinit`; submissions kept internally in FIFO order.
  Executables still run on the caller's task, inside `loadExecute` when
  admission retires older submissions and inside `awaitAll`, so calling
  `awaitAll` right after the last submission gives the progressive
  behaviour. `load` after `loadExecute` treats a transformed tensor as
  delivered once its `loadExecute` is submitted. `deinit` keeps the
  await-without-execute path for errors; `awaitAll` returns the first
  error (the sticky error made per-submission attribution moot). Dropped
  on purpose: selective or early awaits (no user) and the playground's
  separate pack/bulk timing (the batch diagnostics carry it).

### Open questions

- Laguna's `gate_up_proj` executable stacks twice and concatenates; the
  playground only stacks. Its `temp` on ROCm and CUDA should be read before
  the budget relies on the GPU `temp = 0` above.
- Whether the ROCm and oneAPI plugins honor `use_tfrt_gpu_client` the way
  the CUDA one does (only C needs it).
- Whether the CPU platform should use packs at all: execution is six times
  the read and needs 3.8x the output in temporaries there.
- `bytes_limit` on CUDA was 248.96 GiB of a 284 GB device; how it tracks
  `memory_fraction` and what `bytes_in_use` includes on ROCm were not
  checked.

## Fourteenth pass: loader-owned `loadExecute` admission, `awaitAll` only (2026-09-10)

Implements the thirteenth-pass decision on top of `2f170c75 simplification`
(uncommitted). Plan: `~/.claude/plans/abstract-crunching-donut.md`.

### The change

- `Loader.load`, `loadBuffer` and `loadExecute` return `!void`;
  `Loader.awaitAll` is the only wait. `Handle`, `Window`,
  `executeInputBytesPerDevice` and `Submission.isDone` are gone, with the
  `zml.io` exports. Submissions live in a `std.Deque(PendingSubmission)` in
  publish order; `retireOldest` awaits the reads, runs the executables when
  asked, frees the inputs and commits the bytes only when it ran. The first
  retire error is kept in `Loader.failure`: `load`/`loadExecute` refuse
  afterwards and `awaitAll` keeps returning it. `deinit` retires without
  executing.
- Admission in `loadExecute` (`zml/io/execute_admission.zig`, pure
  arithmetic with its own tests): cost per device = input placements +
  output placement + the executable's `temp_size_in_bytes` from
  `PJRT_Executable_GetCompiledMemoryStats` (queried per distinct `*const
  Exe` within the call, 0 when the plugin cannot answer, no cross-call
  cache because an `Exe` address can be reused); room per device =
  `bytes_limit - bytes_in_use - (submitted - allocated) - 64 MiB`, where
  `submitted` is the front end's per-device placement bytes of every
  published submission and `allocated` a cumulative per-device counter the
  direct backend increments in `Item.initTransfer` once per tensor
  (`Backend.allocatedBytesPerDevice`; the buffered backend does not count).
  The loop retires the FIFO head (a bulk `load` too, it just frees nothing)
  until `pending_execution + inputs + execution <= room` on every device;
  one submission is always admitted, with a warning once when it exceeds the
  room alone. `memory_supported` is decided once at init (direct backend and
  every device reporting `bytes_limit`); without it every pending submission
  is retired first, the pre-rework order. Placement bytes use
  `shape.packedShape()` like the backend, so `submitted` and `allocated`
  cancel exactly once everything landed (tested).
- `load` is never gated. A transformed tensor counts as delivered once a
  `loadExecute` naming it was published (`delivered` is now a set of ids
  filled after a successful publish), so the bulk queues behind the packs
  without a drain; a missing `loadExecute` still fails with
  `TransformedTensorNotDelivered`, now logged at `warn` because the test
  runner counts logged errors as failures.
- Callers: `examples/llm` models, `examples/mnist`, `zml/testing.zig`
  (`load` + `awaitAll`); the playground submits every pack, then the bulk,
  then one `awaitAll`, keeps `ZML_LOAD_PACKS/_WIDTH/_PAIRS/_CHECK/_MAX_ELEMENTS`,
  drops `ZML_LOAD_PACK_WINDOW` and the separate pack/bulk timings, and gains
  `ZML_GPU_MEMORY_FRACTION` (the platform's BFC `memory_fraction`, a platform
  setting used to shrink the pool and watch admission). Docs:
  `docs/learn/loader.md`, the README `Mnist.load` snippet (it called a
  `zml.io.load` that no longer existed), the loader header.
- Logs: `execute admission: retired=...` at debug when a retire happened,
  `loader admission: submissions=, execute_submissions=,
  execute_admission_retires=, memory_supported=, min_room_seen=, reserve=`
  at deinit.

### Verification

- `bazel test //zml:test //vfs:test` pass (275 tests, 3 skipped);
  `bazel build //examples/io //examples/llm //examples/mnist` pass; `zig fmt`
  clean.
- B70 CPU (`Qwen3.5-4B` sharded on four CPU devices, two packs of 16,
  `ZML_LOAD_CHECK=64`): `memory_supported=false`, one retire before the
  second pack (`inputs=720 MiB execution=3.43 GiB`, the 2.72 GiB of XLA CPU
  temporaries counted), `pack check: ok`, `load check: ok`. Warm A/B, two
  rounds each: with packs 3.58 / 3.73 s, without 1.03 / 1.00 s. The bulk now
  publishes right behind the second pack and its reads overlap that pack's
  1.2 s execution (its 7.27 GiB took 2.1 s instead of 1.0 s alone, CPU
  contention with the executable), which still beats the serial order
  (3.99 s this morning, cold direct).
- gb300-2 GPU 1 (`CUDA_VISIBLE_DEVICES=1`, GPU 0 held another user's job,
  load average 6 to 20; Llama-3.1-8B-Instruct warm, 14 packs of 16 =
  13.00 GiB then the 1.96 GiB bulk, `zml-directio` worktree carrying this
  tree): `memory_supported=true`, `bytes_limit` 248.96 GiB, `min_room_seen`
  237.64 GiB, `execute_admission_retires=0`, `pack check: ok` every run,
  `load check: ok` (`ZML_LOAD_CHECK=16`). Loader `elapsed` (from creation
  to summary, the metric that excludes calibration):

  | tree | loader elapsed | wall `Loaded weights` |
  |---|---|---|
  | this morning, window 1 (3) | 0.346 / 0.329 / 0.327 s | 571 / 553 / 551 ms |
  | this morning, window 8 (3) | 0.274 / 0.274 / 0.271 | 498 / 500 / 497 |
  | this tree, packs + bulk (6) | 0.267 / 0.276 / 0.327 / 0.261 / 0.265 / 0.265 | 369 / 586 / 638 / 364 / 576 / 575 |
  | this tree, bulk only (3) | 0.259 / 0.260 / 0.264 | 361 / 361 / 365 |

  Packs plus bulk now load at bulk-only speed: no drain between the two
  phases and the executes overlap the reads. The wall clock is bimodal in
  the packed runs only because DMA calibration (inside `Loaded weights`,
  before the loader's clock) took 96 ms and chose 16 MiB blocks in the fast
  runs and 310 ms with 8 MiB blocks in the slow ones (5 of 9 packed runs, 0
  of 3 bulk-only runs; calibration starts right after the pack executables
  compiled). Not the loader; a calibration-robustness item, see Open work.
- gb300-2, `ZML_GPU_MEMORY_FRACTION=0.08` (`bytes_limit` about 20 GiB for
  15 GiB of weights): room 11.4 GiB at the first retire, 4 retires (one
  before submission 11, three before 13), no OOM, `pack check: ok`, wall
  578 ms (8 MiB calibration). The measured path degrades to partial overlap
  as designed.
- Left on gb300-2: the `zml-directio` worktree now carries this tree (every
  file changed since `b59c41b7`), logs in `~/zml-directio-logs/adm_*.log`
  and `adm2_*.log`, scripts `~/zml-admission-run.sh`.

### Follow-ups

- llmd on monorepo `master`: migrate `loadPacked` (`llmd/weights.zig:985`)
  to submit every packed tensor, then the bulk, then one `awaitAll`;
  compile the pack executables in `Model.init`; allocate the KV cache after
  the weights when wanted (the budget measures either way); absorb the zml
  API drift since `f8ddb3e5`.
- DMA calibration right after an XLA compile on gb300-2 sometimes measures
  low (310 ms, 8 MiB blocks) and costs 200 ms of wall time; the loader is
  unaffected. Worth a look with the calibration diagnostics.
- `docs/howtos/howto_torch2zml.md:246` still shows the old `zml.io.load`.
- On CPU the bulk's reads overlap the last pack's execution and slow down
  under CPU contention; the total still improves. If CPU pack loads ever
  matter, the executable's 3.8x temporaries are the first thing to fix.

## Fifteenth pass: fixed source width and the review's simplifications (2026-09-10)

Trigger: a review of the loader code, `direct_loader.zig` in particular,
asked what could be simplified. The adaptive width machinery was the
largest block whose recorded evidence did not justify it; the user asked
to remove it with the best fixed values the recordings support and to apply
the rest of the review. Uncommitted on top of the fourteenth pass.

### The width evidence, and the values chosen

Every width measurement in this file, by source class and host:

| source, host | widths | result |
|---|---|---|
| local, B70 oneAPI, 32 MiB requests, Llama | 12 / 16 / 24 / 32 / 48 / 64 | 21.3 / 20.7 / 18.9 / 17.3 / 15.1 / 13.2 GiB/s: knee at 12, 24 costs 11% |
| local, B70, 8 MiB requests | 4 / 8 / 12 / 16 / 24 / 32 / 48 | 19.5 / 18.4 / 23.0 / 23.0 / 22.5 / 21.9 / 21.9 GiB/s: 12 and 16 tie |
| local, one MI300X, Llama | fixed 12 / fixed 24 / adaptive (24) | 0.41-0.44 / 0.62-0.67 / 0.90-0.97 s; 16 tasks 0.424, 24 0.476, 32 0.572, 128 0.61-0.68 s |
| local, RTX 5090 host, Llama | adaptive / fixed 12 / fixed 16 | 0.50-0.55 / 0.49-0.50 / 0.46-0.48 s |
| local, gb300-2, DeepSeek (sixth pass) | 8 / 12 / 16 / 24 / 32 / 48 | 36.9 / 44.8 / 46.7 / 48.2 / 48.8 / 47.3 GiB/s: one 8% plateau from 12, 16 within 5% of the best |
| local, gb300-2, DeepSeek, adaptive start rung | 12 / 24 / 32 | 3.37 / 3.43 / 3.23 s (four rounds); those loads ended at 48, 64 and 96 |
| remote, hf:// Qwen3.5-4B, buffered | 12 / 32 / 64 in flight | 37.3 / 20.8 / 20.8 s |
| remote, hf:// Qwen3.5-4B, direct | fixed 32 (blind bootstrap) | 10.9 / 16.3 s against 48 s buffered the same hour |
| remote, hf:// Qwen3.5-9B, direct adaptive | held 24 / 32 | 20.6-23.9 s, `width_ceiling=32` |
| remote, real AWS S3, 16 MiB requests | 24 to 128 | within 0.7%; latency and pinned memory rise with the width |

What the controller added on top of a good fixed width: locally nothing
outside noise (the sixth pass measured the gb300-2 plateau at 8% wide
against 13 to 15% of within-load spread, and the controller settled at 12
in some loads and at 24 or 32 in the rest, 5% above the fixed-32 oracle);
remotely it climbed to the 32 its blind bootstrap started at. Its cost was
about 900 lines carrying the subtlest invariants in the tree (generations,
admission fences, the busy clock, the warm-up rule, parking).

Chosen: **16 for local sources**, never more than 5% from the recorded
optimum of any host (B70 12 to 16, MI300X 12, RTX 5090 16, GB300 24 to 32;
12 would give up 5% on the RTX 5090 host and 8% on gb300-2), and **32 for
high-latency sources** (hf:// reads 32 and 64 alike, AWS is flat from 24,
and 32 x 32 MiB requests pin 1 GiB). `limits.zig` carries the constants
with this summary.

### The change

- `Loader.Options.read_parallelism: ?usize = null` (the profile's default
  when null), validated against `limits.max_read_parallelism`; the
  `Parallelism` union and `zml.io.Parallelism` are gone; the backend
  `Config` carries the resolved `usize`. The playground's
  `ZML_LOAD_READ_PARALLELISM` overrides it (`ZML_LOAD_FIXED_READ_PARALLELISM`
  and `ZML_LOAD_READ_INITIAL_PARALLELISM` are gone).
- `direct_loader.zig`: `SourceRuntime`, `SourceProbe`, `BusyWindowClock`,
  `WorkerPool` (parking), the generation and admission fields of
  `ReadRequest`, `gate_closed_ticks`, `RequestGate.waitEmpty`/`currentLimit`/
  `drained`, `shouldBootstrapSource`, `preallocated_source_width` and
  `source_concurrency.zig` are deleted (5051 -> 4167 lines with the tests;
  the working copy is 688 insertions against 2362 deletions). What remains
  of width control is `ThrottleWatch` (halve on a throttle or timeout, the
  settle rule, a 25 ms tick, only with a stats side channel) over
  `ReadStatsCursor.takeThrottle`. Workers are spawned once at creation; the
  pre-growth is `width + 1` requests; `host_memory.growthFreeRequestWidth`
  (the climb ceiling) is gone.
- The review's mechanical items: batch items are one `[]Item` allocation
  (`Item.deinit(allocator, api)`; `Loader.destroyBatch` releases the device
  state, `Batch.destroy` the memory); `retire_events_early` and its dead
  branch are gone; `RequestGateLimits.Config.at(width)` replaces four
  copies of the same call; `workers_started`/`controller_started` are gone
  (`Io.Group.await` on a group that never spawned returns at once);
  `Scheduler.remainingJobs` replaces the one-field `Snapshot`;
  `ReadRequest.run` asserts the planner's invariants instead of
  re-validating them; `Planner.Config{device_count, block_size,
  request_size, alignment}` threads through `publishFiles`, `preparePlan`
  and `maximumJobLen`, and `Planner.TensorPlan` carries the item into
  `appendTransfers`; `Planner.Job` and `finalJob` are gone, `Plan.Job`
  holds index ranges and `Scheduler.Claim{batch, plan, index}` slices the
  request, blocks and transfers from them; `Plan.source_slot` replaces the
  per-job copy; `TensorTransfer.init(direct, item)` and `deinit(allocator,
  api)` drop the per-tensor allocator and platform copies; `first_claim_at`,
  `first_read_ns`, `longest_planning_ns`, `source_runs` and
  `pending_source_jobs`/`source_finished` are gone; `checkOpen` is private.
- `bytes_loaded` moved to the front end (`Loader.bytes_loaded`, added in
  `retire`); `Backend.bytesLoaded`, `Submission.commitBytes` and both
  backends' counters are gone; the direct summary logs `read_bytes`
  (physical) instead.
- The buffered backend (TPU, neuron, metal, untestable here) takes the same
  width for its reads and permits (16 local, 32 remote, against the former
  12 and 128) but keeps its host staging at `min(width, 12)` tensors
  (`staging_tensors`), the former start width, so the staging bound of
  12 x largest tensor is unchanged; 32 reads is the plateau the hf://
  measurements put at 32 and 64 alike.
- `loader.zig`: `admit` is one loop over `measureFit` (`unmeasured`, `fits`,
  `exceeds`), and an executable submission is admitted as soon as nothing
  executable is pending, measured or not: a pending bulk load frees
  nothing, so retiring it first only delayed the pack's reads (the CPU peak
  is the same either way). `execute_admission.zig` keeps `room`,
  `roomPerDevice` and `admits`; `Decision`, `decide`, `addPending` and
  `subPending` are gone. The scratch words are a field;
  `BoundExecutable.deinit` is no longer idempotent (nothing calls it twice).
  The read-failure test now fails through a `loadExecute` of the missing
  tensor retired by the next `loadExecute`, deterministic on both backends.
- Docs: `docs/learn/loader.md` (width paragraph, implementation map), the
  README and mnist callers (`.read_parallelism = 1`).

### Verification

- `bazel test //zml:test //vfs:test` pass; `bazel build //examples/io
  //examples/llm //examples/mnist` pass; `zig fmt --check` clean. New unit
  tests: the throttle watch (halves once, ignores retries, waits for the
  in-flight reads to settle, floors at one) and the cursor's throttle-only
  delta.
- B70 CPU (`Qwen3.5-4B` sharded, two packs of 16, `ZML_LOAD_CHECK=64`):
  `source_width=16, lifecycle_credits=33, workers=17`, pregrown 264 MiB
  (528 MiB before), one serial retire, `pack check: ok`, `load check: ok`,
  `Loaded weights` 3.64 s against 3.58 / 3.73 s in the fourteenth pass.
- gb300-2 GPU 1 (`CUDA_VISIBLE_DEVICES=1`; GPU 0 held another user's job at
  98%, load average 10 to 18; Llama-3.1-8B-Instruct replicated, warm, the
  `zml-directio` worktree carrying this tree, script `~/zml-fifteenth-run.sh`,
  logs `~/zml-directio-logs/fw_*.log`): `source_width=16, lifecycle_credits=25,
  workers=17`, pinned 400 MiB mapped and high-water (656 MiB with the former
  32-wide set, as the width-32 run below shows), no throttle watch (local
  profile). Loader `elapsed`:

  | run | loader elapsed | wall `Loaded weights` |
  |---|---|---|
  | packs 64 x 16 + bulk (3) | 0.265 / 0.325 / 0.286 s | 387 / 422 / 384 ms |
  | bulk only (3) | 0.281 / 0.278 / 0.268 | 379 / 375 / 1279 |
  | bulk only, `ZML_LOAD_READ_PARALLELISM=12` | 0.277 | 374 |
  | bulk only, `ZML_LOAD_READ_PARALLELISM=32` | 0.276 | 380 |
  | fourteenth pass, packs + bulk (6) | 0.261 to 0.327 | 364 to 638 |
  | fourteenth pass, bulk only (3) | 0.259 / 0.260 / 0.264 | 361 / 361 / 365 |

  Parity with the adaptive tree on a busier host, and the width is flat
  (12, 16 and 32 within 2%), as the sixth-pass sweep said. `pack check: ok`
  in every packed run, `load check: ok` (`ZML_LOAD_CHECK=16`, 5 of 67
  tensors). The 1.279 s wall of the third bulk run is DMA calibration on
  the loaded host choosing 4 MiB blocks (8 MiB requests, 1918 reads); the
  loader's own elapsed stayed at 0.268 s, the calibration-robustness item
  of the fourteenth pass again.

## Sixteenth pass: simplification review and its checklist (2026-09-11)

Question (user, 2026-09-10): given how the monorepo uses the loader and the
recordings in this file, what else can be simplified in `zml/io` and
`vfs/direct_io`? Method: ten scoped finders (direct backend in four slices,
front end, host memory and calibration, buffered seam, VFS direct I/O, a
CTX.md evidence audit, a caller-surface audit against llmd at monorepo
`master` `a64fd7a9` and the zml examples), 54 raw candidates merged to 37, a
completeness critic that added 7 more, and three adversarial verifiers per
candidate (correctness and callers, recorded evidence here, net simplicity;
the last 19 verdicts on Opus 5). 23 candidates survived (14 with all three
verdicts clean, 9 in an amended form); 21 were rejected. `PLAN.md` holds the
survivors as a sequential checklist with file and line references; this
section records the decisions.

llmd at monorepo `master` still targets the previous single-file loader
(`origin/master` `f8ddb3e5`): one `Loader` per process, `loadExecute` per
packed tensor then one bulk `load` then one `await`, several stores and
sharding sets per loader (dflash), `bytes_loaded` for the bandwidth log. It
never calls `vfs.loadProfile`, so on this branch it would take the local
profile: width 16 on hf://, no direct I/O, no throttle watch. The migration
must pass `vfs.loadProfile(model)`.

### Survivors (details in PLAN.md)

- Mechanism changes, each with a device run: lifecycle credits are a
  constant (after pre-growth `retained >= width + dma_stage`, so
  `RequestGateLimits` always yields `read = width`, `lifecycle = retained`;
  both recorded ready lines confirm it: 33 = 17 + 16 on CPU, 25 = 17 + 8 on
  gb300-2); the pinned pool sized once at creation with no slab growth
  (every fifteenth-pass run has `pinned_mapped == high_water`; the two
  arena allocations stay for ROCm's per-node balance); the scheduler as a
  FIFO of plans (ordering is unchanged because `submit` publishes and seals
  on one task, CTX 1136-1139).
- Surface trims with no device run: dead TensorStore accessors, the
  `loadBuffer` alias, `backend.Config` folded into one `Options`, the
  `max_host_bytes` knob (the 16 GiB guard stays as a constant), dead pool
  fields, the host-memory backend union merge, single-pass DispatchSpans,
  the admission module's boundary types, two log-only metrics and the
  `call_count` parameter, three derivable pipeline fields, the bool on
  `allocatedBytesPerDevice`, one no-VFS profile at 8 MiB, one worker group.
- Needs a measurement first: the ReadyTransfer/EventContext merge (gb300-2
  DeepSeek pair plus a multi-device read-back), dropping `fairOrder` (an
  A/B on the B70, where the pump is not the ceiling), a `block_size`
  override that skips the DMA screen (the per-target table is refuted:
  CUDA hosts chose 2 MiB on the RTX 5090 host and 16 MiB on gb300-2).

### Decisions

- The throttle watch stays. The review found no recorded run in which the
  halving fired (real AWS and hf runs report zero throttles; the S3Proxy
  has no fault injection) and recommended deleting it; the user's
  requirement is the opposite: rate limits must be handled and traffic
  reduced when they occur. The watch's shortcomings stand recorded: it
  never recovers, it ignores the server-named delay for all but the one
  request, the five-retry budget can fail a load under a sustained limit,
  timeouts count as throttles, the counters are backend-global. The
  replacement, rate-limit handling inside the VFS (a hold on the
  server-named delay, reduction, recovery, a retry budget that does not
  burn during a hold), was designed the same day: three independent
  designs, two adversarial critics each (Opus 5). Placement in the VFS
  survived unanimously (the quota belongs to the backend and its
  credentials, shared by the tokenizer and config reads, a second loader
  and the buffered backend, none of which the watch covers); an AIMD permit
  ladder and additive recovery did not (unmeasured policy of the class the
  fifteenth pass deleted; the ladder collapses to one permit on a burst
  because permits are released per attempt). The surviving shape is a
  per-backend hold in `range_read.zig` on the server-named delay with a
  floor (`Retry-After: 0` parses to zero and would hot-loop once throttles
  stop charging `max_retries`), a jittered wake (`std.Io.Condition` has no
  timed wait), a wall-clock throttle budget on the limiter instead of the
  per-request retry count, timeouts never holding, the loader's width as an
  immutable ceiling, and the stats side channel kept for observability.
  Decided with the user the same day: the VFS owns it, one governor per
  backend instance keyed by URI authority so a per-authority scope is a
  local change later, and the loop governs every HTTP request a backend
  makes (range GETs, HEADs, listings, the HF tree and redirect hops, the
  GCS token POSTs). Defaults taken: holds capped at 2 minutes, a 5 minute
  throttle budget per episode, the hold floor at the initial retry delay,
  jittered wakes, timeouts never holding, one 401/403 retried on the HF
  data GET after re-resolving the URL, `error.Canceled` propagated through
  the backend wrappers. `PLAN.md` group D (tasks 23 to 28) is the
  implementation checklist. Until it lands the loader keeps the watch on
  the read gate alone.
- Rejected, with the decisive reason: batch diagnostics (every field is
  consumed by a table here); `Options.auto` (used by `zml/testing.zig`);
  eager file opens on the submit task (hf and s3 opens are HEAD round
  trips); `validateExecutableSharding` (guards `device.id` indexing);
  single-binding `loadExecute` (Laguna's per-layer pack coalescing);
  removing either gate (the read gate is the watch's actuator; the
  fourth-pass measurement); direct I/O demotion (the streaming fallback
  and a recorded defect); a per-target DMA block table (above); the
  buffered backend's chunked split (needs a TPU host); the placement,
  ledger and `Item` reshapes (net zero lines); folding `stop` into `fail`;
  inlining `initBuffered`; a `Batch` union for `Submission`.

### Implementation log

Landed in order, each as its own commit, validated with `zig fmt --check`,
`bazel test //zml:test //vfs:test` and the three example builds unless the
entry says otherwise. `PLAN.md` loses a task as it lands.

- Task 1 (C01), `buffered_loader.Loader.read_parallelism`: the field and its
  initialiser are gone; `create` still derives `group`, `staging_slots`,
  `tensor_workers` and `permits` from the parameter, so the TPU, neuron and
  metal values are identical. Nothing read the field.
- Task 2 (C02), dead `Scratch.execution`: the admission scratch is four
  `u64` slices in one allocation (room, allocated, inputs, placed); the
  unread fifth is gone. `admit` still fills `inputs` and `submit` still
  fills `placed`.
- Task 3 (C30), `execute_submissions`: the counter is derivable from the
  submissions that carried executables and nothing consumed it; the
  `loader admission` debug line keeps `submissions`,
  `execute_admission_retires`, `memory_supported`, `min_room_seen` and the
  reserve.
- Task 4 (C19), `Loader.loadBuffer`: the one-tensor alias of `load` is gone
  (its four callers were the loader's own tests); `docs/learn/loader.md` no
  longer names it. The dated CTX entries above stay as history.
- Task 5 (C03), dead TensorStore accessors: `getReaderById`, `View.parent`,
  `View.getShapeOpts` and `getPtrFromId` are gone (35 lines). No caller in
  zml, the examples, the tests or llmd at monorepo `master`, which uses
  `withPrefix`, `withLayer`, `createTensor`, `maybeCreate*`, `hasKey`,
  `count`, `prefix`, `getShape`, `getSourcesById` and `getReader`.
- Task 6 (C18), log-only metrics and `call_count`: `Metrics.source_calls`
  and `Metrics.transfer_pieces` are gone with the
  `physical_source_calls=` and `tensor_transfer_pieces=` fields of the two
  summary lines, and with them the `call_count` parameter of
  `safetensors.readFilePositionalAllV`. `read_operations`, `read_bytes`,
  `dma_submissions` and every timer and pump counter stay. Physical request
  counts for a remote source come from the VFS `batch source` line; a local
  profile has none.
- Task 7 (C06), derivable pipeline state: `ReadRequest.completed`,
  `EventContext.pipeline` and `DevicePump.ready_entries` are gone. The
  retirement checks assert `pending == 0` instead of the flag, the pump's
  `deinit` asserts `queue.len == 0`, and an event reaches the platform
  through `block.pipeline`, lazily: `destroyEvent` touches it only when
  there is an event or an error to destroy, which is what the fixture with
  an undefined platform relies on (the first attempt read the api eagerly
  and faulted that test).
- Task 8 (C23), `allocatedBytesPerDevice` without the bool: the seam reads
  the direct payload and returns `void`; `probeMemory` refuses a buffered
  backend before any `memoryStats` call, so the CPU and TPU stats
  behaviour is unchanged, and `readRoom` calls it plainly.
- Task 9 (C21), one no-VFS load profile: `LoadProfile.default` is gone and
  `.local` (8 MiB, not high latency, no alignment, no stats) is the
  `Loader.Options` default, so a caller without a VFS profile now asks for
  8 MiB requests instead of 16 MiB. Evidence for the size: B70 local
  8/16/32 MiB measured 27.05 / 24.21 / 21.33 GiB/s with twice the pinned
  high-water at 16 MiB. The buffered backend reads `read_chunk_size` only
  when the profile is high latency, so TPU, neuron and metal loads are
  byte-identical. `llama_tests` and `lfm2_tests` dropped their explicit
  `.load_profile = .local` (built to check).
- Task 10 (C14), one options type: `backend.Config` is gone. The five
  fields, their defaults, `auto` and the long doc comments live in
  `backend.Options`, which gained `readWidth()` (the option or
  `limits.defaultReadParallelism` of the profile); `Loader.Options` is an
  alias and `Loader.init` hands `opts` straight to `Backend.init`. The
  buffered backend still takes `(read_parallelism, profile)` and receives
  `opts.readWidth(), opts.load_profile`, so TPU, neuron and metal see the
  same values. `docs/learn/loader.md` lost the stale
  `Loader.backendFor(target)`.
- Task 11 (C24), the host-budget knob: `Options.max_host_bytes`,
  `Workspace.Options` and `minimum_mapped_bytes` are gone. The guard stays
  as `Workspace.mapped_bytes_ceiling`, a fixed 16 GiB that `init(allocator,
  io, platform)` applies; `initForTesting` still takes its own ceiling. The
  playground loses `ZML_DMA_BENCH_MAX_MAPPED_MIB` (never used in a
  recording) and the front-end test that covered the post-sizing errdefer
  path now does it with an invalid alignment (`.local` with
  `direct_io_alignment = 3` and `direct_io = .on`, `InvalidLoadProfile`).
- Task 12 (C10), dead pool surface: `BlockPool.newly_mapped_bytes`,
  `unused_tail_bytes` and `pub const Error = anyerror` are gone
  (`acquireMany` returns `!void`), and `init` no longer re-sums the arenas
  to compare against `workspace.mapped_bytes`: `attachArena` still refuses a
  zero-length arena and the block-size and ceiling checks stay. `findArena`
  and `initForTesting` stay (two production callers in
  `dma_calibration.zig`).
- Task 13 (C09), host-memory backend union: `dma_map` and `pageable` were
  the same `Pages` payload distinguished by whether the allocator holds a
  platform, so they merged into one `pages` variant;
  `HugePageAllocator.init` takes `?*const Platform` and `initPageable` is
  gone. The arena log keeps the exact `kind=dma_map` and `kind=pageable`
  strings, now chosen from `allocator.platform`. `place` clears its own mask
  after a refused `mbind` instead of calling a one-line helper, and the
  `.tpu, .neuron, .metal => error.DmaBenchmarkUnsupported` arm stays a
  recoverable error.
- Task 14 (C16), DispatchSpans in one pass: the span count pre-pass,
  `placementSpanCount`, `appendPlacementSpan` and the count assert are gone;
  the recursion appends with `try` into an empty list.
  `error.NonContiguousShardPlacement` became two `std.debug.assert`s with
  the reason recorded in place: `Placement.init` divides a sharded axis with
  `@divExact` (`zml/Sharding.zig:1827`), so the shards tile the axis, and
  `Planner.appendTransfers` already relies on the spans tiling
  `[0, byteSize)`. The gaps-and-overlaps test went with the error; mirrored
  masks, `deduplicateByRange` and the axis recursion stay.
- Task 15 (C28 narrow), admission without the boundary types:
  `DeviceStats`, `roomPerDevice` and `Cost` are gone. `room` takes five
  positional numbers, `admits` takes the four slices, and `readRoom` reads
  `device.memoryStats()` straight into `scratch.room` (one loop, no stats
  scratch), still refusing a device without a limit and still tracking
  `min_room_seen` from the devices that have one. The module and the `Fit`
  enum stay: the `.unmeasured` path is real on CPU.
- Group A validation (2026-09-11, CPU playground, Qwen3.5-4B sharded over
  four CPU devices, `ZML_LOAD_PACKS=2 ZML_LOAD_PACK_WIDTH=16
  ZML_LOAD_CHECK=64`): `source_width=16, lifecycle_credits=33, workers=17`,
  `pregrown=264 MiB`, `execute_admission_retires=1`, `pinned_high_water=252
  MiB` of `pinned_mapped=264 MiB`, `pack check: ok`, `load check: ok`,
  `Loaded weights [8.68GiB, 3.545s, 2.45GiB/s]` (fifteenth pass: 3.64 s;
  the second shard was uncached on this run and read direct). Every group A
  task also passed `bazel test //zml:test //vfs:test` and the three example
  builds.
- Task 16 (C07), the lifecycle credits are a constant: after pre-growth the
  retained capacity always satisfied `min(feasible, max(read + dma_stage,
  retained)) == retained`, so the gates are now `read_gate = width`,
  `request_gate = pool.capacity / blocks_per_request` and `workers = width +
  1`, with the width clipped to `credits - 1` and `credits < 2` refused at
  sizing as `DmaMappedBudgetExceeded`. `RequestGateLimits` (with `Config` and
  `at`), the write-only `Loader.limits`, `Sizing.feasible_width`,
  `Sizing.dma_stage_requests`, `dmaStageRequests`,
  `BlockPool.potentialRequestWidth`, the lifecycle-gate test and the DMA
  stage test are gone; the fourth-pass measurement that justifies credits
  beyond the width (24.3 against 43.8 GiB/s on a GB300) is now the doc
  comment of the gate field. The throttle watch keeps only the read gate
  (narrowing the credits would not reduce source traffic) and the rider C26
  landed with it: the watch runs in `worker_group` and `throttle_group` is
  gone. The ready line prints `lifecycle_credits` and no longer prints
  `feasible_width`.
- Task 16 gb300-2 verification (2026-09-11, GPU 1 with
  `CUDA_VISIBLE_DEVICES=1`; GPUs 0 to 2 held other jobs, load average 7 to
  15; Llama-3.1-8B-Instruct replicated, 14 packs of 16 then the bulk, the
  `zml-directio` worktree overlaid with every file changed since
  `b59c41b7`): `source_width=16, lifecycle_credits=25, workers=17` and
  `pinned_high_water == pinned_mapped == 400 MiB`, all three identical to
  the fifteenth pass; `memory_supported=true`, `bytes_limit` 248.96 GiB,
  `min_room_seen=237.64 GiB`, `execute_admission_retires=0`; loader
  `elapsed` 0.291 / 0.271 / 0.272 s against 0.26 to 0.33 s recorded, walls
  389 / 368 / 368 ms; `pack check: ok` on every run and `load check: ok`
  with `ZML_LOAD_CHECK=16`. Scripts: `~/zml-groupb-run.sh`, logs
  `~/zml-directio-logs/b16_*.log`.
- Task 17 (C04 amended), fixed-size pinned pool: the pool is sized once in
  `Sizing.init` and never grows. The width is fitted there (`(usable +
  remaining budget - reserve) / blocks_per_request - 1`, clipped to the
  option) and zero is `DmaMappedBudgetExceeded`; the two `growToBlocks`
  calls keep the recorded two-arena shape, reserve first, for ROCm's
  per-node byte balance. A request size that is not a multiple of the block
  size is refused at init (`InvalidDmaLoadConfig`), which is what makes the
  credits an exact request count; every shipped profile satisfies it.
  `BlockPool` lost `reserve`, `slab_blocks`, `default_slab_size`,
  `canEverAcquire`, `remainingBlockBudget`, `reservedGrowthBlocks`, `grow`,
  `allocateSlab` and the split `attachArena`; `acquireMany` refuses beyond
  the capacity and otherwise waits. `ensureLoadBlockReserve` and
  `ensureSourceWorkingSet` are gone with their reserve-drop fallback. Tests:
  the metadata-retry and free-list-capacity tests went with growth, the
  reblock test is reblock-only plus an over-capacity refusal, the reserve
  test counts retained requests over arena tails; the
  "allocates nothing once its arenas are attached" test is kept (the plan
  proposed deleting it; it is the direct assertion that a load maps
  nothing), with its arena mapped before the pool is built.
  Verification. CPU playground: the same two arenas (128 MiB + 136 MiB),
  `source_width=16, lifecycle_credits=33, workers=17`, `pregrown=264 MiB`,
  both checks ok, 3.675 s. gb300-2 (same conditions as task 16):
  `pregrown=144 MiB` into `retained=400 MiB`, width 16, credits 25, workers
  17, loader `elapsed` 0.265 / 0.265 / 0.268 s, `pinned_high_water ==
  pinned_mapped == 400 MiB`, `pack check: ok` and `load check: ok`. Logs
  `~/zml-directio-logs/b17_*.log`.
  Risk carried forward: the ROCm 41/59 per-node split under sequential
  growth (commit `a2a0a9b8`) post-dates the last MI300X runs. The two-arena
  pre-growth is unchanged, but one 8x MI300X Llama load should confirm it
  when the host is free (check the stale ROCm plugin first).
- Task 18 (C25), the scheduler is a FIFO of plans: the queue holds `*Plan`
  and a plan leaves it with its last claim, so `Batch.plan_cursor`,
  `queued`, `sealed`, `claimJob`, `exhausted`, `retireUnclaimed` and
  `appendPlanAssumeCapacity` are gone with the seal-time pop and the
  open-exhausted-head rule; `seal` only stamps the diagnostics. `publish`
  sets `plan.batch` and adds the units in the same critical section, and
  `Claim.batch()` and `ReadRequest` read the batch through the plan. The
  ownership rule re-derives one level down: a plan is queued only with jobs
  left, so its batch still holds a unit and cannot be freed under the
  scheduler, and a batch's queued plans are contiguous, so the `fail` that
  may complete it is the last access to it. Ordering is unchanged because a
  submission publishes every plan and seals on one task.
  Verification. CPU playground: same width, credits, workers and
  pre-growth, both checks ok, 3.243 s. gb300-2 (same conditions): loader
  `elapsed` 0.251 / 0.259 / 0.249 s against 0.265 to 0.268 s before the
  task, `pinned_high_water == pinned_mapped == 400 MiB`, `pack check: ok`,
  `load check: ok`. Logs `~/zml-directio-logs/b18_*.log`.
- Task 20 (C27), the byte-fair job order is gone (about 230 lines). It was
  measured first, as the plan required, with two binaries from this tree
  (the arm taking the planning order unconditionally logs
  `job_order=planning`), interleaved warm runs after a warm-up, on the two
  hosts where the DMA pump is not the ceiling:

  | fixture | fair order | planning order |
  |---|---|---|
  | four B70, Llama-3.1-8B sharded (4) | 0.661 / 0.658 / 0.661 / 0.660 s | 0.662 / 0.659 / 0.661 / 0.663 s |
  | four CPU devices, Qwen3.5-4B sharded (4) | 1.293 / 1.222 / 1.221 / 1.195 s | 1.291 / 1.224 / 1.230 / 1.228 s |
  | four B70, Qwen3.5-4B sharded (3) | 1.238 / 1.224 / 1.229 s | 1.243 / 1.225 / 1.243 s |

  Flat everywhere (under 0.5%, inside the run-to-run spread), so `fairOrder`,
  the per-device charge queues, the per-job physical row, the charging loop
  and parameter of `appendTransfers`, `TensorPlan.device_indices` and the
  five fair-order tests with their two helpers are deleted; the jobs keep
  their planning order, which is file order. The device-id bound check that
  the charge loop carried is kept where the plans are built, since every
  later use of a device id indexes a per-device array. `docs/learn/loader.md`
  no longer claims the planner orders across devices. CPU playground after
  the deletion: 3.521 / 3.521 / 3.537 s packed (no read-back), `pack check:
  ok`, unchanged width, credits, workers and pre-growth.
- Task 23 (group D), the governed request loop: `vfs/request.zig` holds the
  `Governor` (one per backend instance, a `Hold` keyed by
  `Governor.holdFor`, which ignores the key today so a per-authority map is
  a local change), `admit`, `reportThrottle`, `refresh`, the `perform` loop
  and the shared `exchange`, plus `classifyStatus`, `serverRetryDelay`,
  `fullJitterDelay`, `RequestSpec`, `Attempt` and `authorityOf`.
  `range_read.zig` keeps only the Range specifics (`RangeSpec` with its
  per-attempt `prepare` hook, `Content-Range`, the scatter). A throttle arms
  a backend-wide hold instead of charging a retry; every other retryable
  failure keeps `max_retries` and the per-request backoff; the hold is
  floored at `retry_initial_delay` (`Retry-After: 0` parses to zero),
  capped at `max_hold`, waited out with a per-waiter jitter, and an episode
  longer than `throttle_budget` fails with `error.RateLimited`.
  `AtomicReadStats` gained `holds` and `hold_wait_ns`. The four backends
  hold a `Governor` instead of a `RetryConfig` and their `InitOpts` gained
  `max_hold = 2 min` and `throttle_budget = 5 min`. Sixteen unit tests in
  `request.zig` drive the governor and the loop without a server (a named
  zero still holds, extension but never shortening, the doubling base, the
  clean window, the budget, cancellation, and that a timeout or a server
  failure never holds); the four HTTP acceptance tests still pass.
- Task 24 (group D), every backend request through the loop. The eleven
  sites now run inside `request.perform`: HF's tree GET (client-followed
  redirects, as before), each hop of its HEAD redirect chain and its data
  GET; S3's listing, HEAD and GET; GCS's metadata-server and ADC token
  requests, its listing, HEAD and GET; HTTP's HEAD chain and GET. Redirect
  handling became explicit (`follow`, `surface`, `forbid`) after the first
  attempt broke the HF tree fetch, which had relied on the std client's
  three-hop default. The HF data GET retries one 401 or 403 after
  re-resolving its download URL through the governed HEAD chain and storing
  the fresh one on the handle (a signed CDN URL can expire during a long
  hold); `RequestSpec.retry_once` carries that. The 404 of an S3 or GCS HEAD
  and GCS's 401/403 are accepted statuses mapped in `consume`, so they stay
  `FileNotFound` and `PermissionDenied`. The four backends' read, open, stat
  and read-dir wrappers pass `error.Canceled` through instead of flattening
  it to `Unexpected`. Smoke-tested against the real services: `ls` and `cat`
  on `hf://Qwen/Qwen3.5-4B` (tree, redirect chain, range GET), `ls`, `stat`
  and a listing on a public `s3://` bucket, and `cat https://iprs.fly.dev`.
  Note for the TPU, neuron and metal sign-off: `buffered_loader.zig` is
  untouched, but its reads go through this VFS, so those platforms gain the
  hold under throttling; on the happy path the loop adds one inert deadline
  check per attempt.
- Task 25 (group D), the mock server and the acceptance tests. `MockServer`
  gained throttle injection (`first_gets`, `first_heads`, `always`, the
  status, `Retry-After` and `RateLimit` resets, and a rate window of N GETs
  per M ms), the two timestamps that show how long the hold actually lasted,
  a `resetPeak` for the monotone peak, and a small path table with paths
  that answer 404 on purpose (task 27 needs both). Seven tests, each against
  a real socket: eight concurrent readers throttled at once share one hold
  and honour a one-second `Retry-After` without charging a retry; a 4-per-100
  ms window with twelve readers completes with holds and no retries at all;
  a `RateLimit; t=0` still holds for the initial delay, with two rounds of
  GETs instead of a spin; a throttled HEAD at open waits and retries (two
  HEADs, one hold); a 503 holds an object-store profile and only charges
  retries on a plain HTTP one; a server that never lets up fails with
  `error.RateLimited` inside its 300 ms budget; a reader cancelled while the
  backend is held returns `Canceled`. The four existing tests keep their
  assertions. Three consecutive runs of the suite pass (42 tests, 2.1 s).
- Task 26 (group D), the loader drops the throttle watch. `ThrottleWatch`,
  `ReadStatsCursor`, `RequestGate.setLimit`, the `throttle` field, the watch
  construction and spawn and their three tests are gone; `Loader.width` is
  written once at create and never again. The statistics side channel stays
  for observability: the `batch source` line gained `source_holds` and
  `hold_wait_ms` beside the requests, bytes, retries and throttles. The
  loader header, the `read_parallelism` doc, `docs/learn/loader.md` (the
  implementation map gained a `vfs/request.zig` row) and the CTX design
  bullets now say that a rate-limited source is handled one layer down.
- Task 27 (group D), one end-to-end test through the loader. `MockServer`
  moved to `vfs/mock_server.zig` and the module is exported as
  `VFS.mock_server`, so the loader's own test section can serve a fixture
  over a real socket. The test registers four tensor ranges with gaps (so
  the planner keeps four jobs) at one `http://127.0.0.1:port/object` URL,
  registers the `http` backend in a `VFS`, and loads them with the direct
  backend on CPU at `read_parallelism = 4` against a 2-GETs-per-100 ms
  window. It asserts the four buffers byte for byte, `throttles > 0`,
  `holds > 0`, `hold_wait_ns > 0`, `retries == 0`, exactly the tensor bytes
  transferred, and that holding cost time and not memory: pinned
  `high_water <= 4`, every lifecycle credit returned and the whole pool free
  after `awaitAll`. The simpler route than the plan's `fromRepo`: the
  loader's registry takes `file_uri` directly, so no entrypoint resolution
  is involved.
- Task 28 (group D), the record. The "Current design" bullets now describe
  the governed loop, the hold and the two budgets (CTX 240), the statistics
  side channel as observability rather than control (154), the loader
  without a throttle watch (177) and the immutable width (408).
  `docs/learn/loader.md` carries the `vfs/request.zig` row and the same
  statement about the width. Group D validation, all on 2026-09-11: `zig fmt
  --check` clean, `bazel test //zml:test //vfs:test` (237 and 42 tests) and
  the three example builds pass; `ls` and `cat` over hf://, s3:// and
  https:// against the real services; and one `hf://Qwen/Qwen3.5-4B sharded`
  load on the CPU playground at parity with the recorded profile:
  `source_width=32, request_size=32 MiB`, 278 reads and 282 physical
  requests for 8.68 GiB, `pinned_high_water=1.06 GiB` of 1.16 GiB mapped,
  `source_retries=0, source_throttles=0, source_holds=0, hold_wait_ms=0`,
  11.6 s (the fourteenth-pass direct arm measured 10.9 and 16.3 s; hf://
  varies about twofold day to day).
  Open decisions left for the user: the TPU, neuron and metal sign-off (the
  buffered loader is untouched but its reads gain the hold), whether to
  revisit `max_hold` once the pipeline's cancellation points allow
  `worker_group.cancel` at teardown, and when to flip the governor key to
  per-authority (one generic `HTTP` instance serves every host of its
  scheme and one `S3` spans buckets, so per-instance over-reaches there).

## Seventeenth pass: validating the DMA calibration rework (2026-09-13)

`61641ff5..f759af76` reworked the calibration and its neighbours: one shared
`AsyncHostToDeviceTransferManager` whose buffers are sliced per candidate
instead of one manager per slot per candidate, `warmupDeviceAllocators` moved
from the measurement into `Loader.init`, `std.Io.Duration`/`Timestamp` in
place of raw nanoseconds, finite error sets with a single log per failure,
`BlockPool.init` taking the workspace by value, and `validateOptions`
replaced by assertions. This pass only asks whether calibration still
measures and selects what it did.

Method: `99cff7bf` (the last commit before the four calibration commits
`2fc4d9fc`, `4f18b622`, `88a2e262`, `f759af76`) and `f759af76` built side
by side on `gb300-2` (`~/numa-bench/zml-base` and `~/numa-bench/zml`) and run
interleaved, one binary after the other, on an otherwise idle host. Both
bench trees add `.{ .scope = .@"zml/io", .level = .debug }` to the example's
`std_options`, because the new code logs the calibration line at `debug` on a
scope the example does not enable (see the defects below).

Selection and measured rate, `io dma-bench`, five runs per tree:

| tree | block chosen | measured GiB/s |
| --- | --- | --- |
| gb300-2 `99cff7bf` | 16 MiB x4, 8 MiB x1 | 175.3-185.3 |
| gb300-2 `f759af76` | 16 MiB x5 (10/10 across the session) | 178.0-184.3 |
| mi300 `f759af76` | 16 MiB x5 | 43.6-45.3 (calibration 0.39-1.25 s) |

Both bands sit inside the recorded per-device H2D history (GB300 176-184,
MI300X 44-47 with the pinned-range-aware plugin), so sharing one manager
across the eight in-flight transfers does not serialize them. The one 8 MiB
pick came from the *old* tree; the smallest-near-peak rule remains the noisy
part, and nothing in this pass made it worse.

Interleaved loads on gb300-2, warm, loader `elapsed`:

| round | Llama repl. `99cff7bf` | Llama `f759af76` | DeepSeek repl. `99cff7bf` | DeepSeek `f759af76` |
| --- | --- | --- | --- | --- |
| 1 | 0.293 | 0.293 | 6.684 | 6.537 |
| 2 | 0.294 | 0.291 | 6.717 | 6.501 |
| 3 | 0.295 | 0.292 | 6.648 | 6.002 |
| 4 | 0.291 | 0.288 | - | - |

Llama-3.1-8B-Instruct is 14.96 GiB over 291 tensors, DeepSeek-V4-Flash-NVFP4
156.75 GiB over 135235; both replicated on four GB300 and both warm. Every
round of both trees selected 16 MiB. The new tree is at parity or a little
ahead on both, which is what "still works" should look like.

mi300 (8 x MI300X, replicated, `f759af76` only): Llama cold 4.53 s twice
(14.96 GiB at 3.30 GiB/s -- `cat` of the same files takes 6.2 s, this
checkpoint is on the volume's slow extent), warm 0.980/1.031/1.075 s, inside
the 0.84-1.10 s history. DeepSeek-V4-Flash replicated 18.16 s, storage-bound.

Correctness: `ZML_LOAD_CHECK` passes on both hosts with the new tree --
gb300-2 DeepSeek-NVFP4 `ok tensors_checked=500 of 135235`, mi300 Llama
`ok tensors_checked=16 of 291`. `bazel test //zml:test //vfs:test` passes.

Three defects, none of them in what calibration measures:

- The `dma_bench version=13 ... measured_gib_s=...` line moved from
  `log.info` to `log.debug` on scope `zml/io`. `examples/io` enables
  `zml/io/load` and `zml/vfs` only, so the one line this file tells you to
  check first is invisible in the shipped example. Log it at `info`, or add
  the scope to `examples/io/main.zig`.
- `validateOptions` became `std.debug.assert`, and the example feeds
  environment values straight into `Options`. Measured on gb300-2 with
  `--config=release` (which is `release_safe`, so the assertions stand):
  `ZML_DMA_BENCH_BLOCK_MIB=64`, `ZML_DMA_BENCH_BLOCK_PARALLELISM=64` and
  `ZML_DMA_BENCH_BLOCK_PARALLELISM=0` each panic with "reached unreachable
  code" where they used to return `InvalidDmaBenchmarkOptions` or
  `NoFeasibleDmaBenchmarkTuple`. `docs/learn/loader.md` states the assertion
  contract, so the gap is in the caller: `examples/io` should range-check
  what it reads. (Under `release_fast` the same inputs would instead spin
  forever in `runWindow` with zero workers, or index `candidates[0]` of an
  empty slice.)
- `docs/learn/loader.md` lists `UnsupportedPlatform` among the loader's
  errors, but nothing returns it: `host_memory.Workspace.init` logs a message
  containing the name and then hits `unreachable`, and `host_memory.Backend.init`
  `@panic`s for tpu/neuron/metal. Either the table row or the code should move.

## Eighteenth pass: two-GB300 llmd loads, DMA depth, read width, placement (2026-09-19)

Fixture: monorepo `llmd` loading DeepSeek-V4.1-Flash (278.12 GiB logical:
160 packed expert tensors through `loadExecute`, then the bulk load), two
GB300 (`CUDA_VISIBLE_DEVICES=0,1`, both on host node 0) under
`numactl --cpunodebind=0 --membind=0`, warm page cache,
`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=200`. The model bodies were stubbed for
the day (pack programs are real, `Compiled all models` is 60 ms), so every
number below is the loader's `Loaded weights` time. The host carried a
stray 100%-CPU process all day (load average ~10), so the same arm measured
about 1 s slower than the day before; compare only same-day interleaved
rounds. Sweeps fix the DMA block at 8 MiB, which is also what calibration
picked every time (`dma_bench ... block_bytes=8388608 parallelism=8
measured_gib_s=179`).

### Where the 16 s went (llmd side, 2026-09-18)

Instrumenting `loadInto`: 7.2 s compiling four `llmd_pack_experts`
programs one after another with nothing transferring, 8.3 s of pack
submissions, 0.7 s draining. Two causes, both in `llmd/weights.zig`:

- `packStack` concatenates 384 replicated per-expert parameters and pins
  only the result to `.expert = .experts`. Each concatenate operand has size
  1 on the sharded axis, so the partitioner shards every operand on a row
  axis and reshards the result with one all-to-all per expert: 2310
  all-to-all ops per module, 6935 lines of HLO, 184,800
  `ncclDevKernel_SendRecv` launches (5.7 s of GPU time, 71% of all kernel
  time). A `sharding_constraint` pinning the stacked member replicated makes
  the partitioned program `dynamic-slice(concatenate, partition_id)` with
  zero collectives: compile 7.2 -> 2.8 s, pack GPU time 5.6 s -> 15 ms.
- The four distinct programs compiled serially on the load path; XLA uses
  ~1.2 cores per compile. Compiling them with `io.async` at discovery and
  awaiting at first use overlaps the rest.

Three runs each, one binary after the other: 16.23/16.29/16.48 s ->
7.01/7.04/7.10 s (17.0 -> 39.4 GiB/s).

### PJRT drives every H2D copy of a device through one stream

nsys (`CUPTI_ACTIVITY_KIND_MEMCPY`): all 96,473 H2D copies of GPU 0 are on
`streamId` 25 and all of GPU 1's on 43. `LocalDeviceState` creates one
`host_to_device_stream_` and `CopyRawHostToDeviceAndReturnEvent` always
takes it (`xla/pjrt/se/se_raw_buffer.cc`). So `block_parallelism` is a queue
depth, never parallel copy engines: the GB300 reports
`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 3`, unreachable through PJRT. The
CUDA plugin's device attributes are `compute_capability, coords,
core_count, device_vendor, memory_bandwidth, numa_node, pci_bus_id,
slice_index, vendor`; `memory_bandwidth` is HBM. Nothing describes the host
link or the copy engines. The only hardware hint for the depth is the rate
the block screen already measures (and more streams would not use the other
engines anyway; next section).

Queue state in the trace (interleaved pool, before the fixes below): 6.5%
of copies start with an empty queue; after an underrun the next issue
arrives p50 0.11 / p90 0.51 / p99 0.97 ms later, while the budget
(8 x 8 MiB at 179 GiB/s) is 0.35 ms of work. That looks like starvation,
but the sweep says the reads are what is late, not the issue path:

| pool placement | depth 8 | depth 16 | depth 32 | depth 32, width 32 |
| --- | --- | --- | --- | --- |
| interleave 0,1 (default) | 8.13 8.10 8.09 | 8.03 8.00 8.01 | 7.95 8.18 7.98 | 7.87 7.86 7.93 |
| node 0 (policy, below) | 5.96 5.97 | - | 5.77 5.88 | 5.84 5.89 |

Depth is worth nothing on an interleaved pool and about 2% on a node-local
one. A synthetic width screen would not have found even that: calibration
already reaches 179 GiB/s at depth 8 while the load ran the same stream at
60-68 GB/s (91 GB/s after placement) and 58-71% busy, because the load's
issue latency is the read pipeline's, which no ring benchmark sees. Keep
the fixed 8. If a depth that scales with the link is ever wanted, derive
it from the measured rate instead of screening it (Little's law, ~1 ms of
queued bytes: `clamp(rate * 1 ms / block, 8, 32)` is 8 on MI300X and 23 on
GB300); it costs no calibration time and no new state.

### More host-to-device streams do not help: one stream already saturates the link

Tried directly, since the plugin is buildable from `~/github/openxla/xla`
(`bazel build --config=cuda_remote_arm64 --config=baseline_arm64
//zml/cuda:archive`, ~40 min remote, uncommitted patch in that tree):
`LocalDeviceState` gains `extra_host_to_device_streams_` and
`GetHostToDeviceStream()` round-robins raw copies over them, count from
`XLA_PJRT_GPU_HOST_TO_DEVICE_STREAMS` (default 1, old behaviour); the
execute-path users of `host_to_device_stream()` are unchanged. Wired into
zml by pointing `_PLUGINS["arm64"]` at `file://.../archive.tar.zst` with
its sha256 and `strip_prefix = "lib"` (the fork's tar puts the plugin under
`lib/`, the mirror archive does not); not committed.

With 3 streams (the `ASYNC_ENGINE_COUNT`), each device's copies split
evenly over three `streamId`s, yet CUPTI shows an overlap factor of 1.00:
no two copies of a device ever run at the same time, and the rate over the
union of busy time is the same 91.9 GB/s. The microbenchmark says why:
one GPU pulling 8 MiB blocks from node 0 reaches 195.0 GB/s on one stream
and 194.5 on three at depth 24; two GPUs 327 (one stream each) against
367 GB/s (three each). Pinned H2D on GB300 is serialized per context and a
single stream already saturates the C2C link. Loads, two rounds each,
node-local pool: streams 1 / 3 at depth 8: 5.96 5.98 / 6.05 6.09; at
depth 24: 6.10 5.86 / 5.86 5.90. Noise. Not worth carrying a plugin
patch for; the loader's single-stream model is the right one.

### Read width: the plateau is flat, the requests were the problem

Width 24 and 32 at node-local placement: 5.98/6.08 and 6.00/5.93 against
5.96/5.97 at 16; nothing. What the "width" evidence was hiding: llmd
submitted each packed tensor as its own `loadExecute`. The two per-layer
*scale* packs are 384 sources of 360 KiB each (`experts.N.{w1,w2,w3}.scale`
are contiguous in the shard, 368,640 bytes apart), so they were 30,760 of
the load's 78,081 reads (39%) for 5.7% of the bytes, and each scale pack
was a solo phase of 384 latency-bound requests: 22 ms + 9 ms per layer with
the H2D stream idle, 1.49 s in total (the periodic `>2 ms` gaps in the
trace, followed by 608 copies of median 0.37 MB). Submitting a layer's four
packs as one `loadExecute` (bindings grouped up to 8 GiB of outputs,
`llmd/weights.zig`) lets the planner coalesce across bindings:
`coalescing_ratio` 1.00/1.99 -> 2.67, reads 78,081 -> 35,688 all of 8 MiB,
H2D busy 61% -> 71%, 8.05-8.13 -> 7.45-7.53 s. Nothing to change in the
loader: multi-binding submissions are what `loadExecute` was built for; the
consumer has to use them.

### Read-path microbenchmark: registration and iovecs are free, the interleave is not

`preadv` of 8 MiB requests from the warm page cache (shards 11-26, 4 GiB per
thread) into a 256 MiB-per-thread rotating destination, 16 threads on node
0, optionally with one DMA thread per GPU copying 8 MiB blocks out of the
same pool at depth 8 (`cuMemcpyHtoDAsync` on one stream, like PJRT). The
page cache of this checkpoint is 90% on node 1 (`/proc/self/numa_maps`
after touching one page per 16 MiB; under 1 s for 298 GiB).

| destination | placement | DMA | read GB/s (per thread) | DMA GB/s |
| --- | --- | --- | --- | --- |
| anonymous | first touch (node 0) | - | 105.5 (7.9) | |
| anonymous | interleave 0,1 | - | 80.7 (7.5) | |
| `cuMemHostRegister` | node 0 | - | 107.2 (8.1) | |
| registered | interleave | - | 79.5 (7.5) | |
| registered, 8 or 64 iovecs | interleave | - | 79.3 / 80.4 | |
| registered, 1 thread | node 0 / node 1 / interleave | GPU 0 | 10.3 / 6.0 / 7.8 | **193 / 54 / 169** |
| registered | interleave | GPU 0+1 | **40.9 (3.2)** | 208 |
| registered | node 0 | GPU 0+1 | **77.2 (5.6)** | 211 |
| registered, 24 / 32 threads | node 0 | GPU 0+1 | 101 / 100 | 206 / 188 |
| registered, DMA depth 32 or 16 MiB requests | node 0 | GPU 0+1 | 77 / 80 | 215 / 207 |

Pinning costs nothing, the iovec count costs nothing, `MADV_HUGEPAGE` does
nothing on this 64K-page kernel (only the 512 MiB THP size is enabled;
`AnonHugePages: 0 kB`). What costs is the interleave once the GPUs pull
from the pool: with both GPUs on node 0 DMA-ing from a pool that is half on
node 1, the readers fall to 40.9 GB/s, which is exactly the loader's read
rate at width 16, and remote H2D runs at 54 GB/s against 193 local (the
seventh pass measured 110 vs 176 with an idle CPU side; with the readers'
copies on the same C2C link it is worse). Node-local keeps the readers at
77 GB/s and reaches ~100 at 24-32 threads while the two links still take
211 GB/s.

### The change: the pool follows the task's memory policy

`host_memory.Workspace.init` now masks the interleave nodes with the
task's `get_mempolicy` node set (`taskMemoryPolicyMask`): under
`numactl --membind=0` the mask is node 0 alone, so the pool is left to the
kernel there (`placement=unplaced`); without a policy nothing changes
(interleave over every memory node). An arena `mbind`-interleaved across
sockets overrides `set_mempolicy`, which is what silently defeated the
user's `--membind`. Same binary otherwise, three runs:
7.45/7.53/7.50 s -> **5.99/5.84/5.97 s** (37.1 -> 47.6 GiB/s), and the 256
MiB arena registers in 3.6 ms instead of 77.

This does not contradict the seventh pass. There, readers were unbound and
the four GPUs sat on both sockets, so every single-node choice was wrong
for half of the traffic and interleave was the knowledge-free middle.
Here the user has already told the kernel where everything lives, and the
loader must not place pinned pages outside that. The rule is: interleave
by default, never outside the task's policy.

After the change the H2D stream copies at 92 GB/s while busy but is busy
only 58% (5.56 s window, 0.70 s of >2 ms gaps, 0.43 s of 0.1-2 ms gaps).
The load is read-bound at 53 GB/s of unique bytes, against the 77 GB/s the
same 16 `preadv` threads reach under the same DMA load in the
microbenchmark: about 30% is still lost inside the read pipeline (gates,
block leases, planning, the pump), unattributed. The other structural
cost is unchanged: every packed expert is a replicated parameter of the
pack program, so each GPU receives the full 298.6 GB (2x amplification).

## Nineteenth pass: where the last 30% goes (2026-09-19)

Same fixture as the eighteenth pass (llmd, DeepSeek-V4.1-Flash, two GB300
on host node 0, `numactl --cpunodebind=0 --membind=0`, warm page cache,
dummy model bodies, real pack programs), on the committed tree
(9f4dc2dc, pool follows the task's memory policy). Baseline over the day:
5.85-6.05 s with 8 MiB blocks; the block screen sometimes picks 16 MiB
(5.75-5.98 s, 219,474 pieces instead of 254,834), so arms are compared at
the same block size only.

Tools: the loader's debug metrics (`loader waits`, `loader DMA`), a
temporary 1 ms sampler thread inside `direct_loader.zig` logging both
gates and every pump's queue/in-flight state, nsys CUDA traces analysed
from the sqlite export, the XLA fork (`~/github/openxla/xla`, patch left
uncommitted) with five environment switches and per-stage timers, and two
microbenchmarks in the session scratchpad (`dmabench.c`: H2D completion
paths on the bare driver; `bench.c`: page-cache readers beside two GPUs'
DMA). Reproduction of the plugin wiring is in the eighteenth pass.

### The reads are not the problem; the readers are starved

Per read (8 MiB, page cache): 1.57 ms in the loader = 5.3 GB/s per thread,
the same as a bare `preadv` thread under the same DMA load. Only one read
of 35,688 exceeded 4 ms; the whole 475 GiB directory is page-cache
resident (mincore). The readers average 11 of 16 busy because they wait
0.65-0.70 ms per read for a lifecycle credit; the 33 credits are held by
blocks in the DMA stage, 2.1 ms per block for ~0.13 ms of hardware time.
The 1 ms sampler shows the pipeline oscillating between two states: every
credit taken, 0-3 reads running, 250-700 pieces queued per device; and
16 reads running with 20 pieces queued. The "30% inside the read
pipeline" of the eighteenth pass is the DMA stage holding the credits.

### Stall 1: allocation sync points behind the pack executes (~1 s)

The GPU client builds every `LocalDeviceState` with
`kComputeSynchronized`: a buffer allocated at time t records the compute
stream's next sync point, and `WaitForAllocation` makes its first copy
wait until the compute stream has passed it (`tracked_device_buffer.cc`,
`AllocatedRawSEDeviceMemory`). The loader allocates tensors continuously
(workers, lazily) while `retire` enqueues the pack executables, so every
tensor allocated during an execute's ~10 ms gets its copies queued behind
it, and since the H2D stream is in-order, the whole stream stops. In the
trace, 49 of 50 H2D idle gaps > 3 ms end within 0.3 ms of a graph
execution's end; the dummy graphs run 0.6-22.6 ms (p50 9.5 ms), 1.36 s of
GPU time per device; the stalls total 0.6-0.7 s per load (62-66 stalls of
5-22 ms, one after each submission's retirement), and while stalled the
credits sit at 33 and the readers at 0. Two proofs:

| arm (executes on unless noted)            | load        | stalls  | readers idle |
|-------------------------------------------|-------------|---------|--------------|
| default                                   | 5.87-6.01 s | 690-709 ms | 935-977 ms |
| no executes (`retire` skips them)         | 4.92-4.95 s | 0       | 125-127 ms   |
| plugin `XLA_PJRT_GPU_ALLOCATION_MODEL=async` (`kAsynchronous`) | 4.89-4.94 s | 0 | 123-133 ms |

With `kAsynchronous` the executables still run and the load is the same
as with no executes at all.

The bare switch is not safe on the GPU client. The asynchronous model's
contract (`local_device_state.h`) is that memory stays alive until the
device work using it has completed, which TPU's allocator guarantees and
the GPU BFC allocator does not: `RunGpuAsync` tears down an execution's
temporaries and dead outputs right after enqueue, and `ExecuteHelper`
holds inputs and results until the definition event only under
`kSynchronous`. Under the compute-synchronized model the next allocation's
sync point makes that safe; without it, a tensor the loader allocates
during a pack execute can be handed the pack program's temporary and be
written by DMA while the kernel still runs. The fork commit 145c8a1693
(`[PJRT:GPU] Opt-in asynchronous allocation model`,
`XLA_PJRT_GPU_ALLOCATION_MODEL=async`, default unchanged) therefore also
defers those releases to the definition event: temporaries and dead
outputs travel in `se_to_be_released`, and the non-compute-synchronized
branch of `ExecuteHelper` keeps inputs, results and that memory until the
event fires. Verified by hashing every loaded shard on the host (2,748
shards, 308 GB, Wyhash per shard): identical hash under both models over
two runs each. Timing with the stock completion path, host load average
~49 that evening: async 5.08-5.14 s, default 5.85-5.96 s (three pairs);
with polled completions the async model reached 4.89-4.94 s.

Tradeoffs of the asynchronous model: memory returns to the allocator when
the host observes completion rather than at enqueue, so peak use grows by
the in-flight executions' temporaries, inputs and dead outputs, and an
allocator query right after an execute (the loader's admission) sees less
room; a `Delete` during an execution is no longer an immediate free;
`StallStreamOnError` frees instead of stalling (error path only); the
transfer manager's `CanBufferBeAccessedNow` DCHECKs run in debug builds.
A finer fix in XLA would attach the sync point only to chunks freed after
the last sync point. On the ZML side the same effect needs allocations to
stop during executes, which the lazy per-worker allocation cannot
promise.

### Stall 2: tiny pieces drain at ~60k pieces/s per device (0.3-1.2 s)

Expert scales are 256-512 KiB pieces: 38% of the copies, 5.9% of the
bytes, 7.4% of copy-engine time. In scale-heavy stretches the pump holds
64 pieces = 21 MiB in flight against the 64 MiB budget, the queue grows to
700 pieces (every credit's blocks), and the pieces complete at ~60k/s per
device (~16 us each: PJRT's per-device callback thread runs the event
bookkeeping, 8 us, then ZML's callback and pump, 5-7 us, serially) =
20 GB/s, readers idle. With the allocation stalls present these stretches
cost 0.9-1.2 s per load (they compound: queues back up), without them
0.3-0.4 s. Raising `max_dma_pieces_per_device` 64 -> 512 changed nothing
(piece latency 0.9 -> 2.5 ms, rate unchanged): the cap is not the bound,
the serial completion path is. Fewer pieces (a scale blob per layer in
the pack recipe) is the lever; the loader cannot merge pieces that land
in different device buffers.

### The completion path, measured stage by stage

Per piece, stock plugin: ZML submit -> callback = 1.1-1.2 ms; hardware
(API call -> copy end) p50 133 us. Stage timers in the fork, poll+inline
arm: sched_delay 0.1 us, memcpy call 3 us, event record 6 us, event wait
720-800 us (stream queue plus the stalls above), callback-thread delay
170-190 us, XLA callback 8 us, ZML callback 5-7 us. Stock path extras:
`cuLaunchHostFunc` blocks 445 us on average (p50 82, p90 1,489): the
driver serializes host-function launches at ~21 us each with ~21 callers
blocked (unbounded work-queue threads, 360 of them, all named after their
spawner); each transfer bounces through that queue (sched_delay 50-75 us)
and the event pool creates events at 430 us each under the lock convoy.
On the bare driver (`dmabench`) the same host-function pattern does 62-68k
pieces/s per GPU at 2 MiB, polling 74k/s; so the serialization is the
convoy, not the mechanism.

Fork switches and what they bought (8 MiB blocks, executes on):

| switch                                         | piece latency | load        |
|------------------------------------------------|---------------|-------------|
| stock                                          | 1.14-1.18 ms  | 5.87-6.05 s |
| `XLA_PJRT_GPU_POLL_CALLBACKS=1` (event polling thread, no host functions) | 1.00-1.05 ms | 5.85-5.92 s |
| + `XLA_PJRT_GPU_INLINE_H2D=1` (copy issued on the caller's thread) | 0.88-0.93 ms | 5.86-5.95 s |
| + `XLA_PJRT_GPU_H2D_CALLBACK_THREAD=1` (own worker for H2D callbacks) | 0.88-0.99 ms | 5.84-5.92 s |
| `XLA_PJRT_GPU_HOST_TO_DEVICE_STREAMS=3` (eighteenth pass)   | -           | noise       |
| `ALLOCATION_MODEL=async` alone (stock completion path)      | -           | 5.23 s      |
| `ALLOCATION_MODEL=async` + poll + inline                    | 0.37-0.46 ms | 4.89-4.94 s |

None of the completion-path changes moves the load while the allocation
stalls dominate; once they are gone the polling+inline path is worth
0.3 s (5.23 -> 4.9). Not worth carrying alone.

### Copy engine and memory contention

Copy time fits 2 us + bytes / 100 GB/s per copy during a load (p50 17 us
for 1.38 MiB), against 155-193 GB/s idle: the readers' writes into the
node-0 pool and their remote page-cache reads share node 0's memory with
both GPUs' DMA. `bench.c`: 16 readers at 78 GB/s beside two GPUs at
230 GB/s total (2 MiB pieces, depth 27). When a copy is queued behind
another the inter-copy gap is 3.6 us p50; the stream idles 29% of the
steady state in 26-70 us gaps because nothing is queued, and 53-64% of the
window overall. Stream time per device: 3.26 s for 279.4 GiB.

### The window and its edges

- 0.85 s before the first submission: `loadPacked` awaits the first pack
  program's compile (`TMP first loadExecute at 863-891 ms`). Submitting
  the plain tensors first (a `TransformedTensorNotDelivered` skip plus the
  reorder in `loadInto`) fills only 0.15 s of it: their 9.17 GiB read in
  0.2 s; the batch's `done` is logged when the front-end thread gets to
  await it. A compilation cache for the pack programs is the fix.
- Active window: 4.70 s default, 3.78-3.82 s with the stalls gone =
  78.6 GB/s = the 16-reader ceiling beside two GPUs' DMA (bench 77-78).
  Width 24/32 under the async model: 4.89-4.93 s, no gain, per-read time
  2.0-2.3 ms.
- No execute tail: the sampler shows reads and DMA active until 10-50 ms
  before `Loaded weights`.

Accounting for the 5.9 s: 0.85 s compile wait + 4.7 s window, of which
~1 s is allocation stalls and their knock-on, the rest the read ceiling.
With the async allocation model: 0.85 + 3.8 + 0.25 = 4.9 s.

### What to do

1. Plugin: fork commit 145c8a1693 (`XLA_PJRT_GPU_ALLOCATION_MODEL=async`,
   with the deferred releases the model requires); measured 5.9 -> 5.1 s
   stock, 4.9 s with polled completions, hashes identical. Decide whether
   llmd sets it (memory returns later) or the plugin defaults to it.
2. llmd: cache compiled pack programs, or otherwise fill the 0.85 s.
3. llmd/loader: a scale blob per layer in the pack recipe to cut the
   256-512 KiB pieces (0.3 s without stalls, more with).
4. Keep the completion path as is until 1 lands; then polling+inline
   copies are worth 0.3 s and are a small plugin patch.

Not worth doing (measured): more H2D streams, a larger piece cap, a
dedicated H2D callback thread, wider reads.

## Twentieth pass: merge with master (2026-10-14)

Both checkouts are jj merges of `brabier/adaptive-concurrency` with
`master`, resolved in the working-copy commit:

| repo | branch | master | base | conflicted files |
| --- | --- | --- | --- | --- |
| zml | `553690e8` | `fe25f16b` (#935) | `550f2bbe` | 21 (+30 files changed on both sides) |
| monorepo | `6011a514` | `c40c5b92` (zig 0.17, #646) | `3c6f9944` | 8 |

Master moved ~100 commits in zml (zig 0.17 #923, sharding in the tensor's
shape #868, loader memory kinds #815, Exe eval/Runner #896/#897, Buffer
`HostAccessible` #899 and `PjrtBufferMetadata` #935, furiosa #871, device NUMA
node #891, `Platform.io_impl` #841, hf snapshots #872) and ~90 in llmd.

### Merge policy (what wins where)

- Master's API wins outside the loader and the VFS: sharding/mesh model,
  `Tensor`, `Exe`, `Buffer`, `Platform`, zig 0.17 idioms.
- The branch's loader (`zml/io/*.zig`) and VFS architecture win. Master's
  loader-facing features are ported into them, not dropped.
- Merged public loader API: `Loader.init(allocator, io, platform, opts)`,
  `load(Model, &model, &buffers, store, progress)`, `loadExecute(store,
  bindings, progress)`, `awaitAll()`, `bytesLoaded()`. The `shardings`
  argument of `load` is gone: placement comes from each tensor's shape
  (`Sharding.Placement.init(shape)`; a null mesh means
  `platform.replicated_mesh`), and `loadExecute` takes input/output placement
  from `exe.input_shapes` / `exe.output_shapes`.
- `TensorStore` is master's API (`fromRegistry(allocator, registry, meshes)`,
  `createTensor(subkey, tags, mesh, partitioning)`, `createReplicatedTensor`,
  `createHostPinnedTensor`, `Binding{tensors, transformed, memory}`) living
  in the branch's `zml/io/TensorStore.zig`. Branch-only additions kept.
- zig 0.17 removed `errdefer |err|`: the one log line that relied on it in
  `direct_loader` sizing (BlockPool.init failure) is gone; the error still
  propagates.

### Notable resolutions, zml

- `zml/io.zig` stays the branch's thin facade. Master's monolithic loader,
  `MemoryWriter`, `ProgressWriter`, `DirectShardWriter` and
  `mem.DmaAllocator`/`UninitializedBufferAllocator`/`DmaMapAllocator` are not
  re-exported (no zml caller; monorepo callers were rewritten).
- Master #815 host-pinned bindings are delivered by the branch loader into
  `device.memory(.host_pinned)` (option "a": staged through the pinned block
  pool and a PJRT transfer manager on that memory), excluded from HBM
  admission and from submitted/allocated device bytes. Accepted on CPU
  (master panicked); tpu/neuron/metal/furiosa return
  `error.UnsupportedMemoryKind`. `loadExecute` inputs still land in
  `.default`; the executable decides the output memory.
- Master #894 (transfer-manager init error handling) and #896 (donated and
  non-donated inputs freed once) are ported with their tests.
- `vfs/file.zig` stays deleted (branch). Master's write support through the
  VFS for local paths and the EINTR-safe `fcntl` helpers are ported into the
  branch's local case; master's "copy a local file through the VFS" test
  passes.
- `platform.zig`: master's `numaNode() ?u32` replaces the branch's `?usize`
  duplicate; `pjrt.Device.attributes()` is master's.
- The branch's strictness is kept: a model tensor with no store binding is
  `error.TensorNotFound` (master's loader skipped it silently).
- Root `BUILD.bazel`: `//examples/io:playground` dropped (the branch renamed
  the target to `//examples/io`). The root `bench_*.sh`/`profile_file.sh`
  scripts still say `playground` (already stale on the branch).

### Notable resolutions, monorepo

- The five `*dflash*.zig` files master deleted stay deleted.
- `llmd/main.zig`: master's file with `bff6c633` re-applied (early
  `Loader.init` in `io.concurrent` with `vfs.loadProfile`, `registerBackend`,
  no `VFS.File`), inside master's scoped loader block.
- `llmd/weights.zig`: master's recipes win; `6011a514` re-applied (pack
  programs compiled with `io.async` at discovery and awaited at first use;
  packs submitted as grouped `loadExecute`s of up to 8 GiB of outputs). The
  branch's per-member `replicated()` pin is dropped because master #566
  already pins the stacked tensor replicated for the same all-to-all reason.
- `aacae994` ("load only deepseek": stubbed deepseek41 bodies and
  `if (true) return;` in main) is NOT in the merge. It was a measurement hack
  for the eighteenth/nineteenth passes; re-create it as a separate commit if
  those load-only measurements are repeated.
- Master's engram path (`MemoryWriter(.host_pinned)` + zero padding on
  `loader.group`) and #634's `streamConcatenatedRows` used old loader
  internals. Stopgap: `weights.streamRowsToHostPinned`, one sequential CPU
  reader per table straight into PJRT `pinned_host` buffers (row-sharded,
  zero-padded, replicas memcpy'd), with `weights.markDelivered` poking
  `loader.delivered` so `load` skips the table. Its bytes are not in
  `bytesLoaded()`. Device-memory `concatenate_rows` (qwen4 without
  `offload_ngram`) falls back to the `llmd_concatenate_rows` pack program
  (replicated inputs): a regression against master #634 until the loader
  supports multi-source destinations.
- Superseded the same day by the twenty-first pass: the engram tables and
  same-dtype row concatenation are loader-native composite destinations,
  and the stopgap helpers are deleted.

### Verification (2026-10-14, B70 host, CPU build unless noted)

- zml: `bazel test //... --test_output=errors`: 26/26 targets
  (`//zml:test` 300 passed, 12 GPU-only skips; `//vfs:test` 43; `//stdx:test`
  28). `bazel build //zml:zml //examples/llm //examples/io
  --//platforms:cuda=true --//platforms:rocm=true --//platforms:oneapi=true`
  succeeds (comptime GPU paths type-check). `bazel build --nobuild //...`
  analyses 231 targets.
- monorepo: `bazel build --config=localzml //llmd/...` (16 non-manual
  targets), `bazel test --config=localzml //llmd/...` 6/6, and
  `//llmd:llmd --@zml//platforms:cuda=true` builds.

### Review of the resolution (four lenses, each finding verified)

Lenses: master changes preserved, branch changes preserved, loader-port
semantics, monorepo. 9 findings, 6 distinct confirmed, all low severity,
all fixed; 3 rejected (the VMM allocator does report `bytes_limit`, so
admission stays on; `Device.attributes()` panicking on a null state is
master's contract; the examples/io BFC-preallocation default change is
deliberate):

- `Buffer.Shard.toHost` (branch-added) leaked the PJRT event that master
  #854 taught every other host copy to destroy.
- `direct_loader` sizing: the zig 0.17 rewrite of `errdefer |err|` had
  narrowed its log to the arena growth; a `BlockPool.init` failure is
  logged again.
- `loadExecute` compared the bound tensor's shape with the executable's
  output by dims and dtype only. `validateOutputPlacement` now compares
  device ids and per-device slices (not mesh pointers, so a null-mesh
  replicated tensor matches an output on `replicated_mesh`); a mutation
  check confirmed the new test fails without it.
- The host-pinned loader test passed vacuously on CPU (every memory kind
  resolves to one space); it now skips that comparison only where the two
  kinds are the same PJRT memory.
- monorepo: the stopgap's streamed bytes were missing from
  `bytesLoaded()` (`addStreamedBytes`), and it now refuses targets whose
  pinned buffers are not dense row-major.

### Runtime check on the local 4x B70 (oneAPI), 2026-10-14

| arm | branch tip `553690e8` | merged |
| --- | --- | --- |
| examples/io `load` Llama-3.1-8B sharded, batch elapsed | 0.676-0.678 s | 0.684-0.687 s |
| same, replicated | 1.255-1.256 s | 1.263-1.279 s (1.251-1.254 at 8 MiB) |
| `ZML_LOAD_CHECK=1` read-back | - | ok 291/291 |
| DMA calibration pick | 8 MiB, 49.3 GiB/s | 2 MiB, 50.4 GiB/s |

Interleaved, same host, warm cache. Plain loads are within ~1.2% of the
branch and of the 0.636-0.676 s anchor; replicated is ~9% above the old
1.147 s anchor in both trees (host drift). "Loaded weights" in examples/io
includes `Loader.init` and its ~0.73 s calibration, so compare batch
elapsed. The branch tip needed its scratch copy patched to load
`libpjrt_oneapi.so` (its `oneapi.zig` asks for `libzml_oneapi.so`, which
its packaging does not ship, and silently falls back to CPU); the merged
tree loads the plugin.

- Host-pinned delivery runs on Level Zero: with `--//platforms:oneapi=true`
  `Platform.auto` picks oneAPI in tests, and every shard of the
  host-pinned test lives in `pinned_host` memory, distinct from `device`,
  on all four devices; the admission ledger counts only the device tensor.
- A genuine merge bug, found only at run time: master's VFS routes
  `file_write_streaming` through its handle table, whose index 0 was the
  CWD, so `stdout()` (fd 1) resolved to the stdin entry and every write to
  stdout went to fd 0 (`examples/llm` failed with `WriteFailed` whenever
  stdin was not a tty). Fixed in `vfs/vfs.zig` (std streams at indices
  0-2) with a test.
- The direct_loader planner/scheduler/dispatch tests used `Platform.auto`
  with a CPU option and broke on accelerator builds (they picked 4 B70s);
  they now use `Platform.init(.cpu)`. Pre-existing on the branch.
- `bazel test //zml:test --//platforms:oneapi=true` aborts in master's moe
  "Triton compatibility matrix" test (Level Zero cannot create
  `per_token_group_quant_fp8`); with it skipped, 301 pass, 11 skip.
- llmd (master's oneAPI plugin `202610011939`) segfaults while compiling
  Llama-3.1-8B: oneDNN `gemmstone::Generator<Core 8>::gemmAccessC` under
  `xla::gpu::SyclGemmWorkspacePass` in `PJRT_Client_Compile`, on 1 or 4
  GPUs, before any load. Not loader-related; the 2026-09 llmd run on this
  host used the August plugin. `examples/llm` compiles and generates.
- `examples/llm` Llama-3.1-8B on the merged tree loads in 21-22.5 s
  against 7.5 s on the branch tip: both plan ~1.56M DMA pieces
  (column-sharded weights), but per-piece latency is 3.4-3.7 ms against
  1.2 ms. See the attribution below.
- Leftover from the branch: `direct_loader.zig:164` logs
  `TMP loader created_at raw_ns` at debug level.

### Attribution: master's oneAPI plugin serializes piece completions

`examples/llm` Llama-3.1-8B on four B70 (column-sharded weights, ~1.56M
DMA pieces): code tree x oneAPI pins, three interleaved rounds each,
Loaded weights / batch elapsed / `dma_piece_latency_ms`:

| arm | runs |
| --- | --- |
| merged code + master pins | 20.6-21.9 s / 20.6-21.8 s / 3.33-3.53 |
| merged code + branch pins | 7.4-7.6 s / 7.4-7.5 s / 1.18-1.21 |
| branch code + branch pins | 7.4-7.6 s / 7.3-7.5 s / 1.18-1.20 |
| branch code + master pins | 21.0-22.3 s / 20.9-22.2 s / 3.38-3.60 |

Swapping single libraries into the merged binary's runfiles: the old
plugin alone restores 7.5-7.6 s; the old SYCL/UR runtime, the old Level
Zero driver + IGC, each alone with the new plugin, stay at ~21-22 s. So
the plugin binary (`mirror.zml.ai/plugins/202610011939.137.1.0d59bb389ee2`
against `pjrt-artifacts manual-2026-08-17`) is necessary and sufficient;
the merge's zml code is not involved. Platform creation sends identical
oneAPI options in both trees (BFC, preallocate, fraction 0.90), and
`io_impl` only reaches `Compiler.zig`.

Mechanism: both plugins issue each piece as an async H2D memcpy plus a
SYCL event. With the old plugin the four per-device `py_xla_callback`
threads each run ~85% CPU completing pieces in parallel; with the new
one a single SYCL `ThreadPool` worker runs at ~100% in
`DispatchHostTask -> Scheduler::NotifyHostTaskCompletion ->
cleanupCommands -> pthread_rwlock_trywrlock` while the callback threads
idle: every completion goes through a SYCL `host_task` and the scheduler's
global lock, ~13 us per piece, ~75k pieces/s for all four devices
together. `SYCL_QUEUE_THREAD_POOL_SIZE=4/8` (24.3/23.5 s) and a DMA depth
of 16 (23.1 s) do not help. The 2 MiB calibration pick is a symptom
(forcing 8 MiB stays at 22.5 s). Big-piece loads (examples/io) are
unaffected because they issue few pieces.

Levers: (1) the plugin should complete events on per-device callback
threads again instead of SYCL host tasks (raise with whoever builds the
mirror plugins); pinning the August plugin with the new runtime worked by
library swap (7.6 s) but was not tried as a bazel build. (2) Loader side,
independent of the plugin: stop cutting column-sharded tensors into one
piece per row by packing each device's strided shard contiguously in the
pinned block (the destination-ordered placement of the composite-tensor
design) so a block is one transfer per device. Scripts and logs:
session scratchpad `swap/`.

## Twenty-first pass: composite destinations, engram resident in host memory (2026-10-14)

Question from the user: does llmd (deepseek41) load the engram tables into
host memory and keep them there, and if not, make it so. Answer before this
pass: monorepo master did it on CUDA with three model-local mechanisms
built on master's old loader internals (`MemoryWriter(.host_pinned)` plus
`splatByteAll(0)` padding on `loader.group`, then
`swapEngramHostPinnedBuffers`). The merge replaced the first with a slow
stopgap. This pass makes the loader deliver them natively and checks the
"keep it there" part on the B70s.

### Engram facts (derived from the code, no checkpoint on this host)

- Engram layers 1 and 14 of DeepSeek-V4.1. Each row is 256 bytes of fp8
  values (`weight [row, channel]`) plus 8 bytes of ue8m0 scales
  (`scale [row, group]`). Rows per layer are the sum of 24 primes >= 16M:
  384,006,168 and 384,016,682 (padded to 384,016,688), about 94.4 GiB per
  layer, **~189 GiB in total**. Per device with `.row = .model`: 94.4 GiB
  at TP=2, 23.6 GiB at TP=8. The 278 GiB DeepSeek fixture of the
  eighteenth/nineteenth passes did not include engram.
- Rows are padded to a multiple of 8 (#555) and sharded on rows (#502)
  over `.tp_mesh`, memory `.host_pinned`.
- "Keep it there" (master #490, #645, unchanged by the merge): after
  loading, each table shard is wrapped by `Buffer.createView()` into a
  device-kind PJRT view of the pinned pointer; the owning pinned buffers
  stay in `Model.engram_host_buffers`. The program sees device memory and
  the GPU gathers through the mapping (zero copy). Declaring the tables as
  `pinned_host` to XLA instead would make `toMemory` copy the whole table
  to HBM every step. `engramHostPinnedBytes` is passed as
  `extra_device_memory_bytes` so XLA does not believe HBM is full; KV-cache
  sizing subtracts the engram bytes. qwen4 does the same for its offloaded
  n-gram tables.
- `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` must cover the engram bytes plus any
  other pinned use; nothing in zml checks it.

### The change: composite destinations (first slice of the concat design)

A `TensorStore` binding may now carry a layout: several sources placed into
ONE destination, with zero fill for uncovered bytes, in any memory kind the
loader supports, loaded by plain `Loader.load` (no program, no
`loadExecute`).

- `TensorStore.Binding.layout: ?*const Layout`; `Layout{ parts, fill = 0 }`;
  `Part{ src_offset, src_pitch, dst_offset, dst_pitch, width, rows }` in the
  destination's logical bytes. This slice accepts contiguous parts only
  (`rows == 1`, no pitch) and fill 0; anything else is
  `error.UnsupportedLayout`. The strided fields exist so interleaves,
  broadcasts and swizzles can be added without API churn.
- `View.createConcatenated(subkeys, tags, mesh, partitioning,
  .{ .rows | .pad_rows_to_multiple, .memory })` concatenates along dim 0;
  one subkey with `.pad_rows_to_multiple = 8` is the engram case.
  `View.createComposite(subkeys, shape, layout, memory)` is the general
  form. `TensorStore.setMemory(tensor, kind)` replaces callers poking
  `id_to_sources` (qwen4); `createHostPinnedTensor` uses it.
- Front end: one `LoadSpec` per destination with N `LoadPart`s
  (`parts`, `zero_fill`). Validation before anything is submitted: one
  part per source, same dtype, the source's dims past the first equal the
  destination's innermost dims, offsets and widths in whole source rows,
  parts sorted by destination offset, disjoint, inside source and
  destination, non-empty (`InvalidLayout`); without fill the parts must
  cover every byte (`SourceSizeMismatch`).
- Direct backend: `Item` is split into `Destination` (shape, placement,
  memory, output, LazyOnce transfer state) and `Part` (what the planner
  sorts by file offset, as before). Source byte -> part -> logical offset
  -> DispatchSpans -> (device mask, device offset). Pieces merge across
  parts of the same destination when block, writer mask, block offset and
  destination offset are all contiguous. Parts of one destination may sit
  in several files and plans; `Target.total` stays the placement bytes and
  the last-transfer flag stays "by bytes", so pieces from any part, plan or
  fill close the buffer whatever their order.
- Zero fill: sizing pre-grows one extra block; after `BlockPool.init`,
  `detachBlock` takes it out of the pool for good, zeroes it, and hands it
  to the pipeline. The plan holding a destination's first part gets one
  final fill job carrying every gap (cut by spans and block size); a worker
  claims it like a read job (credit, batch unit) but takes no read gate,
  no pool lease (`Lease.pool` is now optional) and opens no file.
- Buffered backend: staging zeroes only the gaps and reads each part at
  its offset (`ChunkedRead.source_offset`).
- Accounting: device bytes once per destination per device, host kinds on
  neither the submitted nor the allocated side; `bytesLoaded()` counts each
  destination's shape bytes once, padding included; progress is one node
  per destination counting bytes read.
- Found on the way: `enqueueBlocks` now checks the pipeline's sticky error
  while holding every pump lock. Before, a failure recorded between a
  worker's last `errorValue()` check and the locks could let `abortReady`
  drain the queues first, and whatever was queued afterwards never
  completed (the batch hung). Pre-existing on the read path.
- Tests (CPU, and on the 4 B70s with `--//platforms:oneapi=true`):
  `zml/io/testing.zig` (new) builds an N-device platform (the accelerator
  from `testing.env()` when it has >= N devices, else CPU) and reads shards
  back in mesh order. A planner replay on a 2x2 mesh (3 parts, 2 files, a
  gap, a tail; row/column/replicated; 64 and 16 byte blocks: every device
  byte written exactly once). An end-to-end loader test (concatenated +
  padded, single-source padded = engram, out-of-file-order parts, setMemory;
  2 and 4 devices; row, column, replicated; default and host_pinned; exact
  per-device allocated bytes, 0 for host memory). Error cases, a buffered
  composite with a nonzero `src_offset`. Mutation checks: disabling the
  dtype rule, sorting parts by source offset, or dropping `source_offset`
  each make `//zml:test` fail.
- llmd: deepseek41 binds both engram tables with `createConcatenated`
  (weight padded to 8 rows, scale sized to the weight's rows), producing the
  same shapes, tags, mesh, partitioning and memory as master (a test
  asserts it), and they load in the same `loader.load` as the rest of the
  model, so `swapEngramHostPinnedBuffers` finds them where it always did.
  Same-dtype `Packer.concatenateRows` (qwen4 n-grams) is a composite
  binding in any memory kind: device-memory concatenation streams straight
  into the final placement again (master #634), host-pinned via
  `setMemory`. `streamRowsToHostPinned`, `streamConcatenatedRows`,
  `markDelivered` and `addStreamedBytes` are deleted; dtype-changing
  concatenation stays the `llmd_concatenate_rows` program.

### On-device results (4x B70, oneAPI, 2026-10-14)

- Every composite and host-pinned test passes on Level Zero (309 passed,
  11 skipped, with master's moe Triton test skipped). All 78 host-pinned
  shards report `pinned_host` memory against `device` for default.
- "Keep it there" works on Level Zero, not only CUDA. Test "a padded
  host-pinned table stays in host memory and serves device gathers through
  device views" (kept in `zml/io/loader.zig`): a 100,003 x 264 byte table,
  padded to 8, row-sharded over 4 devices, host_pinned, wrapped with
  `Buffer.createView` like llmd, then a per-device gather of 256 rows under
  `manualComputation` (first row, last rows incl. padding). Results are
  byte-exact; owner shards `pinned_host`, views `device`, same pointer.
  Device `bytes_in_use`: 256 B before and after load + views, 68,864 B
  after the gather (indices + output only); the device-memory control grows
  by one shard. A temporary 1,000,003-row run (66 MB shards): same, and the
  per-device peak stayed at 277 KB, so XLA made no transient copy.
- Throughput of host-pinned delivery (staging block -> PJRT transfer into
  `pinned_host`, "option a"), examples/io with the new
  `ZML_LOAD_MEMORY=host_pinned` knob, warm, interleaved, batch elapsed:

| model, placement | default | host_pinned |
| --- | --- | --- |
| Llama-3.1-8B (14.96 GiB) sharded | 0.690-0.695 s (21.7 GiB/s) | 2.235-2.331 s (6.5 GiB/s) |
| LFM2.5-1.2B replicated | 0.187-0.200 s | 1.686-1.726 s (4 copies) |
| LFM2.5-1.2B sharded | 0.102-0.108 s | 0.437-0.452 s |

  ~60% of the host-pinned time is the PJRT pinned_host allocation inside
  `TensorTransfer.init` on the worker's read path: 12.0-12.7 ms per tensor
  against 0.013 ms for device memory (~1,918 inits x 12 ms / 17 workers =
  ~1.35 s of 2.3 s); DMA piece latency 1.56-1.87 ms against 0.78-1.07.
  Microbenchmarks: a first 8 GiB `pinned_host` allocation takes 0.61-0.78 s
  (10-13 GiB/s, serial; reallocation after a free ~1 ms, the plugin
  caches); memcpy into pinned memory 8-9 GiB/s on one thread, 29.5-31 GiB/s
  on 16. Under memory pressure (6 GB free, 4 GB swapped) the same load took
  14.4 s (76 ms per init, 22 ms per piece). Replicated host-pinned Llama was
  not run: one pinned copy per device is 59.8 GiB on this 62 GiB host.
- Plain default-memory loads unchanged: Llama sharded 0.690-0.695 s,
  replicated 1.273-1.294 s; `ZML_LOAD_CHECK=1` ok 291/291 for sharded
  default, sharded host_pinned and replicated default.

### What to do next for host-resident tables

Items 1 and 2 are done in the twenty-second pass below.

1. Take the pinned destination allocation off the read path (allocate a
   host-resident destination ahead of its reads, in parallel, or at
   submit): it is most of option (a)'s cost and helps option (b) too.
2. Option (b): read straight into the pinned destination (iovecs over the
   destination's host pointer, no block pool, pump or host-to-host copy),
   with ONE allocation shared by replicas through views (as
   `Buffer.HostAccessible` does on cpu/cuda/rocm; oneAPI sharing is
   unverified). Decisive for replicated host tensors (4x pinned RAM, 8.9x
   time today); for row-sharded engram the estimate is 1.6-2.3x, and a cold
   189 GiB read from NVMe may already be the bound.
3. A zml-owned device view (`createView` whose delete callback releases a
   ref-counted owner) would let llmd drop `engram_host_buffers` /
   `ngram_host_buffers`; a compiler-side "host-mapped" marker could compute
   `extra_device_memory_bytes` instead of each model doing it.
4. Not verified on CUDA hardware in this pass: the DeepSeek-V4.1 engram
   load through the loader (pinned_host on CUDA, `createView` swap on
   loader-produced buffers) and its load time against master's writer.

## Twenty-second pass: host-resident delivery without DMA (2026-10-18)

Task: speed up loader delivery into `.host_pinned` destinations, items 1
and 2 of the twenty-first pass's "What to do next". zml only (working copy,
not committed); llmd needs no change (same `LoadSpec`/`TensorStore` API).

### Where the time went (4x B70, oneAPI, Llama-3.1-8B sharded, warm)

- Baseline today, interleaved, batch elapsed: default 0.671-0.675 s,
  host_pinned 2.020-2.031 s (`tensor_init_ms_per_read` 10.2-10.3).
- First-time pinned allocation (a temporary microbenchmark binary, since
  deleted): `createUninitializedBuffer` and
  `createBuffersForAsyncHostToDevice` in `pinned_host` alike, 13.6-15.3
  GiB/s whatever the thread count (1, 4 and 16 threads, 1-64 MiB buffers,
  round-robin over the four devices' host spaces); 1 MiB buffers 7.4 GiB/s.
  After a free the plugin reuses the memory (8 GiB in ~1-10 ms). strace:
  nearly all of it is `DRM_IOCTL_XE_VM_BIND` on one device fd (4, 8, 16 ms
  per call, sizes doubling), the kernel populating and binding pages. It is
  serialized below the loader; nothing in zml can parallelize it.
- The first CPU write into a fresh allocation faults its CPU mapping in:
  memset 9.4 GiB/s on one thread against 21.5 GiB/s on reused memory, 30.5
  GiB/s on 16 threads. Allocating 6 GiB takes 0.577 s alone and 0.724-0.726
  s while 16 threads first-touch another 6 GiB (the touch takes 0.19-0.20 s
  either way): allocation and writes into pinned memory add up more than
  they overlap.
- Step 1 alone (allocation ahead, the DMA path kept): 2.03 -> 1.934-1.953
  s. The allocation task took 1.54 s and the host-to-host DMA became the
  stall (DMA stage 13.3 ms per request, credit waits 5.4 ms).
- Option (b) split in two, for attribution (synchronous allocation before
  publishing, a temporary switch): allocating every destination 1.151 s,
  then the reads into them 0.562-0.571 s (26 GiB/s, first touch included,
  faster than DMA into device memory). Overlapped: 1.66 s.

### The change

- Allocation ahead (`Loader.startAllocations`): a batch with host-resident
  destinations gets one task on the worker group that calls `ensureState`
  for `Batch.host_destinations`, each destination once, in the order of its
  first part in the sorted part order, which is claim order
  (`Planner.hostDestinationsInOrder`). It holds one completion unit, taken
  before the publish sentinel is dropped, and stops at a pipeline failure;
  a worker that reaches a destination first creates it itself (`LazyOnce`).
  Metric `ahead_init_ms`, debug line `batch allocations: ... done=+s`.
- Delivery classes (`Delivery`: `dma`, `host_direct`, `host_bounce`).
  `Destination.hostResident()` is a memory kind that does not count against
  the devices (host_pinned, host_unpinned). `Planner.JobCut` cuts a file's
  DMA parts and host parts separately (no job mixes them), merges the jobs
  back into file order (stable sort) and adds one fill job per class. Host
  transfers are not cut by blocks: `block_index` 0, `block_offset` from the
  read's start. A host job is `host_direct` when its read has no alignment
  padding and its transfers cover the read in order, each byte once
  (`coversReadInOrder`). Otherwise it is `host_bounce`: reads widened for
  direct I/O, and runs holding overlapping or duplicate parts. One overlap
  sends its whole job through blocks.
- Host jobs take no block contexts and no events (`Plan.create` takes the
  DMA submission count); a host job's request ends with its worker's
  scheduling reference. `host_direct` (`HostScatter`): iovecs over the
  destination ranges (memory-contiguous ones merged), read under the read
  gate in as many `readFilePositionalAllV` calls as
  `max_positional_iovecs` needs (a column-sharded tensor is one iovec per
  row per device), then copies to unshared replicas. `host_bounce`: leases
  the blocks, reads like a DMA job, `copyFromBlocks` to each writer,
  releases. Host fill: memset per writer.
- Host state (`TensorTransfer.initHost`): `createUninitializedBuffer` in
  each device's host space, awaited, host pointer read. Replicas (devices
  whose `Placement.slices` start at the same coordinates) share ONE
  allocation where `sharesHostMemory` (cpu, cuda, rocm, oneapi):
  `SharedHostAllocation` owns a buffer that is not a shard, every device's
  shard is a `createViewOfDeviceBuffer` in that device's host space whose
  `on_delete_callback` drops a reference, and the last drop destroys the
  owner (on `smp_allocator`, outliving the loader). Shards can therefore be
  freed in any order. oneAPI checked: the callback fires on view destroy,
  views read back on every device, and kernels gather through device-kind
  views of them (new test). Loader field `share_host_allocations` (tests
  clear it to cover the copies). Donating a shared shard to an execution
  that writes it shows on every device (the `HostAccessible.undonatable`
  rule).
- `Batch.fullySubmitted` covers host destinations: `host_written` (bytes
  times the devices a write lands on) against `host_total`.
- Planner and lifecycle invariants updated in `direct_loader.zig`
  comments: batch completion units (plus the allocation unit), plan
  contexts (DMA jobs' blocks only), `Job.block_start`/`block_len` and
  `Transfer` for host jobs, at most two fill jobs per plan.

### Results (4x B70, oneAPI, warm, interleaved, batch elapsed)

| model, placement | default | host_pinned before | host_pinned now |
| --- | --- | --- | --- |
| Llama-3.1-8B sharded | 0.669-0.672 s | 2.020-2.031 s | 1.626-1.644 s |
| Llama-3.1-8B replicated | 1.256-1.268 s | not runnable (4 x 15 GiB pinned, 62 GiB host) | 1.849-1.874 s |
| LFM2.5-1.2B sharded | 0.103-0.106 s | 0.437-0.452 s (21st pass) | 0.359-0.363 s |
| LFM2.5-1.2B replicated | 0.184-0.192 s | 1.686-1.726 s (21st pass) | 0.361-0.372 s |

- Allocation-bound. Llama: allocations done at +1.41 s, batch at
  +1.64-1.66 s; LFM: allocations end 5-10 ms before the batch. The target
  "sharded host_pinned near the 21.7 GiB/s ceiling" is out of reach on this
  host: allocating 15 GiB of pinned memory alone takes 1.15 s, and ~1.41 s
  beside the reads. The reads themselves run at 26 GiB/s into pinned
  memory, above DMA into device memory.
- Replicated host_pinned is 1.13x a sharded host_pinned load (target met):
  one allocation of 15 GiB instead of four. Its reads take 7.5 ms per read
  against 4.8 sharded; a guess, unverified: the 16 workers fault into a
  quarter as many buffers.
- Forced direct I/O (`ZML_DIRECT_IO=on`: every host job `host_bounce`,
  disk-bound): default 1.600-1.605 s, host_pinned 2.298-2.320 s
  (allocations done at +1.56 s).
- `ZML_LOAD_CHECK=1` ok 291/291 for sharded host_pinned, replicated
  host_pinned (every replica, 59.83 GiB compared) and direct-I/O host_pinned.
- Default memory unchanged (21st pass: 0.690-0.695 and 1.273-1.294 s).

### Tests

- `//zml:test` CPU: 311 passed, 12 skipped. oneAPI on the four B70: 312
  passed, 11 skipped (master's moe Triton test skipped locally, as in the
  twenty-first pass; the skip is not in the tree).
- Planner replay of composite destinations (2x2 mesh) also with host_pinned
  destinations at alignment 0 and 8: deliveries, no blocks or events for
  host jobs, every device byte written exactly once.
- "host-resident and DMA parts never share a job, and overlapping host
  parts read through blocks": class split, file order, bounce for a
  duplicated source, events and blocks for DMA only, allocation order.
- "host-resident destinations are written in place, through blocks when
  reads are widened or overlap, and replicas share one allocation": four
  devices (the B70s or CPU); rows, columns (8,192 transfers in one job,
  several vectored reads), replicated, a duplicated source, and a device
  tensor in one file; shared and unshared replicas; alignment 0 and 64;
  pool blocks and credits back to zero; host memory spaces; one pointer per
  replicated output when shared; every `SharedHostAllocation` released
  whatever the shard free order. Each run writes its own file: with one
  file, an unshared run found the previous run's bytes in recycled memory.
- "a replicated host-pinned table is one allocation that every device
  reads through device views" (new) and "a padded host-pinned table stays
  in host memory and serves device gathers through device views" (kept,
  now `expectHostTableGathers(.rows)`): both pass on oneAPI.
- "loader failures clean up ..." runs every failure with host_pinned
  destinations too.
- Mutations caught: dropping the replica copies, a bounce copy from offset
  0, writing every replica of a shared region, forcing overlapping jobs
  direct.

### Left open

- The allocation floor: a faster pinned allocation (plugin or driver), or
  allocating host destinations while something other than host writes
  runs, e.g. llmd submitting engram before the device weights, or a
  "prepare" API. Not tried.
- Not verified on CUDA: shared-allocation outputs with `on_delete`
  callbacks, pinned allocation speed and contention.
- One overlap in a host run sends the whole job through blocks.
- `host_bounce` under direct I/O copies twice; reading O_DIRECT straight
  into the destination needs destination and file offsets congruent modulo
  the alignment.
- CPU default memory could take `host_direct` too (it is host memory); not
  done, CPU default keeps the transfer managers.

## Twenty-third pass: monorepo re-merge with newer master (2026-10-18)

- The user redid the monorepo merge as `rwotnntx` (`jj new master
  brabier/adaptive-concurrency`), with master at `3a8b1dae` (2026-10-15,
  "llmd/crash/Sentry: missing upgrade to zig 0.17"). Conflicts: `main.zig`,
  `weights.zig`, `models/deepseek41.zig` and the five `*dflash*.zig` files.
- The twentieth/twenty-first pass resolution survives as `kqxnstry`
  (`052a7d2d`, no description), a merge of `6011a514` and the old master
  `c40c5b92` (zig 0.17, #646).
- Method: each file is a 3-way merge with the old master as base, `kqxnstry`
  as ours and the new master as theirs. Master changed 14 files since then.
  Only `deepseek41.zig` and `qwen3_5.zig` overlap with the resolution, and
  `git merge-file` merged both cleanly: compile progress nodes in
  `compileModel`, and the #701 prefill priority in qwen3_5. The dflash files
  are gone on master and stay deleted.
- Checks: `jj diff --from master --to @` touches exactly the 8 files that
  `kqxnstry` changed against the old master. `jj diff --from kqxnstry --to @`
  matches master's 14-file delta (114+/189-).
- Monorepo master moved its zml pin from `c1492917` to `e385d9c2`. Neither
  commit is in the local zml repo, which has not been fetched since
  2026-10-14; the zml merge is on master `cddc9cf6` (#928).
  `--config=localzml` builds against the local merge without needing either
  commit.
- Verified on the B70 host: `bazel build --config=localzml //llmd/...` (22
  targets), `bazel test --config=localzml //llmd/...` 6/6, and the same
  build with `--@zml//platforms:oneapi=true` and with
  `--@zml//platforms:cuda=true`. llmd was not run.
- Uncommitted: `@` (`rwotnntx`) holds the resolution, with no description.

## Concatenating tensors in the loader: design (2026-10-14, exploration)

The user's framing: packing is "just moving bytes to the right place".
Inventory of `llmd/weights.zig` (monorepo master) by class:

- **A, pure placement** (each source byte has one destination): expert
  stacks with members concatenated (`packStack`, all `.down` weights and
  scales), `fuse .copy` with groups=1, concat rows of one dtype (now
  native), engram padding (now native), NVFP4 weight stacks (the f4 dtype is
  a relabel), broadcast fills (A*: one source, many destinations).
- **B, data-independent permutation**: MXFP4 gate/up row interleave (row r
  of member m -> row 2r+m; ~2/3 of DeepSeek's expert bytes), `fuse` with
  groups > 1 (strided), the 128x4 block-scale swizzle (expert scales on
  cute_mxfp4 / NVFP4, `swizzle_block_fp8_scale` #548/#577 with its 128x
  "spread" broadcast).
- **C, compute**: `dequantize_block_fp8` (wo_a #593), `dequantize_blocks`,
  `convert_rows`, dtype-changing concat, NVFP4 globals (tiny reductions),
  CDNA3 `normalize_block_fp8` (byte-local, could run on the host block).

On the DeepSeek-V4.1 fixture every pack program is an expert stack (A or
B). What loader-native stacking removes, from the eighteenth/nineteenth
pass numbers: the 0.85 s wait for the first pack compile and 2.8 s of
compile CPU; ~0.95 s of H2D stalls behind pack executes (no-execute arm
4.92-4.95 s against 5.87-6.01 s); the 2x replication (each GPU receives
~279 GiB today, ~144 GiB when each expert's bytes go only to its owner);
most of the ~97k 256-512 KiB scale pieces. Estimate: 5.9 s -> ~4.0-4.1 s
without any plugin patch, possibly ~3.2 s if halving DMA traffic lifts the
read ceiling (unmeasured). Transient device memory per group drops from
replicated inputs + output to the output alone.

Phases (Phase 1's contiguous slice is the twenty-first pass):

1. Contiguous parts + zero fill + memory kinds (done). Next in this phase:
   `createStacked` for expert stacks (members concatenated), so the `.down`
   stacks and concatenated gate/up leave the pack programs; gate/up that
   must be interleaved can load concatenated and be interleaved by a local,
   collective-free program compiled at init and run after `awaitAll`.
   `normalizeBlockFp8` / `maybeSwizzleBlockFp8Scales` in llmd read the
   `.fuse`/`.stack` recipe to compose their programs, and still poke
   `id_to_sources` to mark bindings transformed; they must learn about
   layouts first.
2. Strided parts (`rows`, pitches, `src_pitch = 0` broadcast), expanded
   lazily per job, never per row up front (an interleave is ~1M chunks per
   tensor), and destination-ordered reads: a job's `preadv` scatters each
   chunk to its destination-ordered position in the pinned blocks (free on
   buffered reads; the 18th pass measured 8 vs 64 iovecs at parity,
   1,024-iovec cost unmeasured), so one DMA piece covers a destination run.
   Under O_DIRECT with sub-4 KiB chunks, a bounce copy in the worker. The
   same machinery fixes column-sharded tensors cut into one piece per row
   (Llama-3.1-8B on four B70: ~1.56M pieces, see the attribution above).
   Atomic units (an expert's w1+w3, a swizzle tile group) must stay inside
   one job; an expert split across two safetensors files needs a join or a
   post-pass.
3. Host-side unit transforms in the pinned block: the 128x4 swizzle and
   the CDNA3 FNUZ map; delete `packStackNvfp4CutlassBlockScale`,
   `packSwizzleBlockFp8Scale`, `packNormalizedBlockFp8`.
4. Class C hygiene: inputs sharded like the output (`bf16Linear` already
   does), compile at init, submit after the bulk load so no execute
   overlaps DMA.

Full exploration with file:line inventory: session scratchpad
`explore_concat.md` (not durable; the points above are the substance).

## Concatenation API: Source expressions (2026-10-18, proposal)

The user asked to compare the loader's concatenation API with the API they
landed on in monorepo `brabier/weights-add-explicit-tags-and-paritins`, to
list everything llmd needs, and to improve the zml API using the branch as
inspiration. The branch is one commit (`rsqorxts` / `898fb0b2`,
2026-10-09, "llmd/weights: use explicit tags and partitioning", on master
`13d57911`, +1099/-844 in 16 files). It is not in master, so master and the
merge still use `FusionSpec`. Nothing below is implemented yet.

### What exists

- zml (twenty-first pass): `View.createConcatenated(subkeys, tags, mesh,
  partitioning, .{ .rows | .pad_rows_to_multiple, .memory })` (dimension 0
  only); `View.createComposite(subkeys, shape, Layout{ parts, fill },
  memory)` with byte `Part`s the caller computes; `setMemory` and
  `createHostPinnedTensor`; `maybeCreateBinding` for program inputs
  (`loadExecute`). The loader takes one contiguous part per source, of the
  tensor's dtype, landing as whole rows.
- llmd on master and in the merge: `Packer.fusedLinear(store,
  FusionSpec{ members{ name, count, width }, groups, source, partitioning:
  BlockPartitioning }, out, k)` records `Recipe.fuse` for the `packFused`
  program. `splitFused` registers alias names in the checkpoint registry to
  split pre-fused tensors. `experts` gives `Recipe.stack` and `packStack`.
  `concatenateRows` is a composite for one dtype, a program otherwise.
  Counts: 28 `fusedLinear` sites in 11 model files, 10 `experts` sites,
  4 `splitFused` sites.
- The branch:
  - `linear.View{ store, definition, start, rows }` is a checkpoint Linear
    (weight, scale, bias, global scale) restricted to a range of output
    rows. `split(.{ .{ .name, .rows }, ... })` checks every component's row
    grid (block scales) and registers nothing; its tests assert no aliases
    and no bindings until assembly.
  - `concatenateLinears(views, out, k, partitioning)` and
    `interleaveLinears(.finest | .explicit{ rows_per_group }, out, k,
    partitioning)` return `FusedLinear{ linear, layout }`.
    `layout.extractOutput(y, i)` undoes the interleave on the output and
    keeps `y`'s tags and partitioning. That frees the group count from the
    head structure (master's `memberHeads` assumed groups = KV heads).
  - Partitioning is keyed by tags everywhere (`.{ .dout = .model, .d =
    .replicated }`), replacing `.out`, `.contracted`, `.none` and
    `BlockPartitioning`.
  - `splitTensor` binds contiguous slices of a non-Linear tensor (conv1d)
    without a program.
  - Still: every assembly records a `Recipe` and runs a pack program.
    Slices are made by editing `safetensors.Tensor.offset` and `.shape` in
    llmd, then bound by writing `store.id_to_sources` directly
    (`bindSources`). The branch predates the composite work, so rebasing it
    onto the merge conflicts in `weights.zig` (`streamConcatenatedRows`,
    `concatenateRows`).

### What llmd needs

Byte-only cases (each is modelled below under "Evidence"):

- Fused projections, groups > 1: qkv by KV head (llama, qwen3_5, gemma3/4,
  laguna, ministral3, muse_glimmer, lfm2, qwen4, dflash), gate/up
  (finest), the qwen3_5/qwen4 GDN `in_proj_qkvz` (three slices of the fused
  `in_proj_qkv` plus `in_proj_z`), and the lfm2 `in_proj` rows
  `[B; C; x]` to `b0 c0 x0 b1 ...`.
- Fused projections, groups = 1: deepseek41 `wq_a`+`wkv`, and qwen4
  `down_inject` and `key_value_proj`.
- Splits of pre-fused checkpoint tensors: `in_proj_qkv` and `conv1d`
  (qwen3_5, qwen4), and lfm2 `in_proj`.
- Broadcasts: the per-tensor FP8 scales of fused members become one scale
  per channel; NVFP4 global scales are repeated over each member's rows.
- Expert stacks: members concatenated (stablehlo, triton/fly without mxfp4,
  flashinfer, metal, tpu); rows interleaved (mxfp4 on cute/triton/fly);
  scalars `[E, P]`. NVFP4 weights are unpacked (u8 to 2 x f4e2m1, a
  relabel).
- Engram padding to 8 rows and n-gram shard concatenation, in pinned host
  memory.
- 128x4 block-scale swizzles (cute_mxfp4, flashinfer NVFP4, Blackwell FP8
  with its 128x row spread): pad, split, transpose and broadcast.

Compute, which stays a program: `dequantize_block_fp8` (wo_a),
`dequantize_blocks`, `convert_rows` and dtype-changing concatenation,
NVFP4 globals (max and reciprocal), and the CDNA3 fn -> fnuz map.

### Proposal: `TensorStore.Source`, Tensor's shape ops over checkpoint bytes

```zig
/// A tensor whose bytes the loader takes from checkpoint tensors by moving
/// them: a checkpoint tensor, or slices, reshapes, transposes, broadcasts,
/// paddings, concatenations and stacks of them. Same op names, tags and
/// argument conventions as zml.Tensor. Built during model init.
pub const Source = struct {
    store: *TensorStore,
    _shape: Shape, // tags, dims, dtype; no sharding until bind
    pieces: []const Piece, // see "Representation and lowering"

    pub fn shape(self: Source) Shape;
    pub fn dim(self: Source, axis_: anytype) i64;
    // Relabel only, never move a byte:
    pub fn withTags(self: Source, tags: anytype) Source;
    pub fn rename(self: Source, renames: anytype) Source;
    pub fn splitAxis(self: Source, axis_: anytype, split_shape: anytype) Source;
    pub fn merge(self: Source, merges: anytype) Source;
    pub fn reshape(self: Source, output_shape: anytype) Source;
    pub fn bitCast(self: Source, dt: DataType) Source;
    // Move bytes:
    pub fn slice(self: Source, axis_: anytype, s: Tensor.Slice) Source; // step 1, no dyn_start
    pub fn split(self: Source, axis_: anytype, sizes: []const i64) []Source;
    pub fn chunkExact(self: Source, axis_: anytype, comptime n: comptime_int) [n]Source;
    pub fn transpose(self: Source, axes_: anytype) Source;
    pub fn pad(self: Source, value: anytype, paddings: anytype) Source; // value 0 = the fill
    pub fn broad(self: Source, other: Shape) Source; // repeats: source stride 0
    pub fn concatenate(sources: []const Source, axis_: anytype) Source;
    pub fn stack(sources: []const Source, axis_: anytype, tag: anytype) Source;

    /// The model tensor the loader fills from this source: placed on `mesh`
    /// by `partitioning` (keyed by this source's tags; unspecified axes
    /// replicated; `.replicated` for no mesh), in `opts.memory`.
    pub fn bind(self: Source, mesh: @EnumLiteral(), partitioning: anytype, opts: BindOptions) Tensor;
};

pub const BindOptions = struct { memory: Memory.Kind = .default };

// On View. `createTensor(k, tags, mesh, p)` stays, as `source(k, tags).bind(mesh, p, .{})`.
pub fn source(self: View, subkey: []const u8, comptime tags: anytype) Source; // panics when missing
pub fn maybeSource(self: View, subkey: []const u8, comptime tags: anytype) ?Source;
```

- What the branch does by convention, zml now guarantees. A Source is a
  value: it never touches the registry or the bindings; only `bind` records
  a binding.
- Tags flow through ops the way they do for Tensor, so the partitioning is
  given once, at `bind`, keyed by the final tags. Memory is a bind option;
  `createHostPinnedTensor` and `setMemory` stay as shorthands.
- The ops mirror Tensor's names and conventions, so the existing pack
  functions port almost verbatim. The examples below are generic over the
  type to show that; requiring generic code is not the goal (see "Why
  mirror Tensor's ops" below). With `T = zml.Tensor` the code builds a
  program; with `T = Source` it is a load-time byte move:

```zig
fn interleave(comptime T: type, members: []const T, groups: i64) T {
    var parts: [MAX_MEMBERS]T = undefined;
    for (members, parts[0..members.len]) |member, *part| part.* = member.splitAxis(.qout, .{ .qg = groups, .qs = -1 });
    return T.concatenate(parts[0..members.len], .qs).merge(.{ .qout = .{ .qg, .qs } });
}

fn stackExperts(comptime T: type, members: []const []const T, layout: zml.moe.ProjectionLayout) T {
    var stacked: [MAX_MEMBERS]T = undefined;
    for (members, stacked[0..members.len]) |experts, *s| s.* = T.stack(experts, 0, .expert);
    return switch (layout) {
        .concatenated => T.concatenate(stacked[0..members.len], .qout),
        .interleaved => T.stack(stacked[0..members.len], 2, .proj)
            .withTags(.{ .expert, .row, .proj, .qk })
            .merge(.{ .qout = .{ .row, .proj } }),
    };
}
```

- Call sites (llmd):

```zig
// deepseek41 engram. Today:
//   store.createConcatenated(&.{"weight"}, .{ .row, .channel }, .tp_mesh,
//       .{ .row = .model, .channel = .replicated }, .{ .pad_rows_to_multiple = 8, .memory = .host_pinned })
const table = store.source("weight", .{ .row, .channel });
const weight = table.pad(0, .{ .row = .{ .high = paddingTo(table.dim(.row), 8) } })
    .bind(.tp_mesh, .{ .row = .model }, .{ .memory = .host_pinned });

// qwen4 n-grams. Today concatenateRows computes byte Parts for createComposite.
const ngrams = Source.concatenate(shards, .voc).bind(.tp_mesh, .{ .voc = .model }, .{ .memory = memory });

// The branch's linear.View on Sources: zml does the slicing and bindSources goes away.
pub fn weight(self: View) Source {
    return self.store.source(self.definition.weight, .{ .qout, .qk })
        .slice(.qout, .{ .start = self.start, .len = self.rows });
}
// interleaveLinears records no recipe, so it no longer needs the Packer:
const weight = interleave(Source, weights, groups).rename(.{ .qout = out, .qk = k })
    .bind(.tp_mesh, partitioning, .{});
```

- The loader contract becomes bytes only. Sources are valid by
  construction (in bounds, disjoint), so the loader keeps its bounds and
  disjointness checks and drops the same-dtype and whole-row rules, which
  forbid `bitCast` (fp4 unpack). `Layout` and `Part` stop being public.
  `createComposite`, `createConcatenated` and `ConcatenateOptions` go away;
  only llmd used them.

### Why mirror Tensor's ops

The user asked what writing the pack code once for both types buys. Less
than the first wording suggested. The benefit with measurements behind it
is the loader path itself. On DeepSeek (design above) it removes the
0.85 s compile wait, 2.8 s of compile CPU and ~0.95 s of H2D stalls
behind pack executes, and each GPU receives ~144 GiB instead of ~279 GiB;
estimate 5.9 s -> ~4.0 s. Code generic over the type helps in three
places:

- Porting. `packFused`, `packStack` and the swizzles already encode the
  layouts the kernels expect and run in production. With the same ops
  they move to `Source` by changing the type. A layout bug produces wrong
  outputs, not an error.
- Choosing where it runs. The same description can run in the loader (on
  the host) or as a program (on the device). That serves as a fallback
  while the loader cannot run a lowered shape yet (phases 2 and 3), and
  possibly for the 4-byte-chunk swizzles, where a permute on the device
  may beat a host copy (unmeasured).
- Tests. The Tensor evaluation (XLA) is an independent reference for the
  loader's lowering.

Costs: `Source` must keep Tensor's semantics (names, tag rules, `-1`
inference, `Slice`); `comptime T` code is harder to read; Tensor-only ops
(`convert`, `withPartitioning`) break it; and once phases 2 and 3 land,
production code only instantiates `T = Source`. Proposed: mirror the
vocabulary, write llmd's layout code against `Source`, and keep generic
code for zml's tests and, if a measurement favors the device, the
swizzles.

### Interleave per shard, not at the finest granularity

`extractOutput` works for any group count. When a fused tensor is
partitioned on `.dout` over D devices, every device holds the same rows of
each member for any group count that is a multiple of D. With groups = D
(1 when replicated), each member lands as one contiguous run per device.
In the model at Llama-3.1-8B sizes, per layer:

- gate/up at the finest granularity: 28,672 chunks of 8 KiB. At D = 4:
  8 chunks of 28 MiB.
- qkv by KV head: 24 chunks of 1 MiB. At D = 4: 12 chunks of 2 MiB.

This is valid whenever the sharding is. A block-FP8 member needs R_i/D to
be whole scale blocks, which row sharding already requires. Heads stay
whole when D divides the head counts, which tensor parallelism already
requires. With groups = D, the weights, block scales and biases of every
fused projection in llmd (the lfm2 `in_proj` included) lower to D
contiguous parts per member. The loader takes those today once one source
may feed several parts. The broadcast scalars need phase 2 below or keep
their small program. `.explicit` stays
for layouts a kernel imposes (mxfp4 experts); `.finest` has no remaining
use.

### Representation and lowering

- A Source is a shape plus pieces. A piece reads one checkpoint tensor
  through an affine lattice of up to 8 dims. Each dim has a count, a source
  byte stride (0 repeats), and the visible axis and index step it moves
  along. Destination offsets are implicit: row-major of the final shape.
- Cost of each op on the pieces:
  - Reshapes (merge, aligned `splitAxis`, `bitCast`, `withTags`) only
    change the shape or the steps.
  - `concatenate`, `stack` and `pad` shift origins; `transpose` permutes.
  - `slice` clips, adding at most 3 pieces per nesting level.
  - `splitAxis` cuts a piece only when a lattice dim straddles a row
    boundary.
  - No op ever enumerates rows.
- At `bind`, destination strides come from the final shape and dims are
  sorted and coalesced. The loader contract becomes `Part{ source: u32,
  src_offset, dst_offset, dims: [<= 8]{ count, src_stride, dst_stride } }`,
  where the innermost contiguous run is the chunk.
- Execution follows the lowered shape:
  - 1-D parts work as today.
  - A few large chunks (>= 1 MiB, <= 64 of them) expand into 1-D parts.
  - Many chunks of >= 4 KiB go to destination-ordered jobs that scatter
    each read with iovecs (phase 2 of the design above).
  - Chunks under 4 KiB and stride-0 dims use a contiguous read, then a host
    strided copy or replicate in the pinned block (swizzles, broadcasts).

### Evidence: a model of the algebra

`scratchpad/apidesign/lattice.py` (session scratchpad, not durable) is a
pure-Python model of the representation above. Each case is written once
against a facade and evaluated two ways: as lattice pieces lowered to
bytes, and as reference index arithmetic. All 15 cases are byte-exact at
small sizes, and the per-index fallback never triggered. The pieces and
chunks below are at full sizes (Llama-3.1-8B per layer; DeepSeek-like
experts: E = 256, 2048 rows, 3584 B per fp4 row):

| case | pieces | chunks | smallest chunk |
| --- | ---: | ---: | ---: |
| qkv, groups = KV heads | 3 | 24 | 1 MiB |
| qkv, groups = 4 shards | 3 | 12 | 2 MiB |
| gate/up, finest (1 row) | 2 | 28,672 | 8 KiB |
| gate/up, groups = 4 shards | 2 | 8 | 28 MiB |
| per-tensor FP8 scales to per channel | 3 | 6,144 | 4 B, stride 0 |
| experts, members concatenated | 512 | 512 | 7 MiB |
| experts, rows interleaved (mxfp4) | 512 | 1,048,576 | 3.5 KiB |
| expert scalars `[E, P]` | 512 | 512 | 4 B |
| lfm2 `[B; C; x]`, 1 row per group | 1 | 6,144 | 4 KiB |
| engram, pad to 8 rows | 1 | 1 | 91.6 GiB |
| n-gram shards (8 x 1,000,003 x 1024 bf16) | 8 | 8 | 1.9 GiB |
| GDN: fused qkv split, interleaved with z | 4 | 64 | 512 KiB |
| NVFP4 block scales, 128x4 swizzle | 512 | 117,440,512 | 4 B |
| FP8 scale, x128 spread | 1 | 7,168 | 56 B, stride 0 |
| fp4 unpack (u8 to 2 x f4) | 256 | 256 | 7 MiB |

The pieces stay proportional to the sources in every case. The chunk
counts give the execution class: contiguous, iovec scatter, or host copy.
Swizzles and stride-0 broadcasts must never become one read or DMA piece
per chunk.

### Phases

1. zml `Source` (algebra, lowering, `bind`). In the loader: parts name
   their source, checks become bytes only, and small chunk counts expand.
   - Tests: each generic pack function runs both as a Tensor program on CPU
     and as a bound Source through `Loader.load`, and the bytes match per
     shard on 1, 2 and 4 devices (row, column and replicated; default and
     host_pinned), with the `zml/io/testing.zig` helpers.
   - llmd: rebase the branch's `linear.View` onto Sources, interleave per
     shard, and stack concatenated experts; move engram, n-grams and splits
     to Sources. The `fuse`, `stack` (concatenated) and same-dtype
     `concatenate_rows` recipes and their programs go away.
2. Strided parts in the loader: mxfp4 row-interleaved experts (about 1M
   chunks of 3.5 KiB per DeepSeek layer) and broadcasts.
3. Swizzles become generic Source code on the host copy path, and their
   programs are deleted.
4. Programs take Sources as inputs (dequantize, convert, normalize,
   globals), with inputs placed like their outputs. Today `normalizeBlockFp8`
   and `maybeSwizzleBlockFp8Scales` compose with recipes and set
   `transformed` on bindings directly.

### Open points

- Naming: `Source` and `bind`.
- Intermediate pieces live in the store's arena. A DeepSeek-sized model
  builds about 150k pieces at init (roughly 20 to 50 MB), so use a scratch
  arena reset after init.
- The Linear-level verbs (split, concatenate, interleave over every
  quantization component) stay in llmd for now. Once they depend only on
  Source, they could move to `zml.nn` next to `Linear`.
- llmd ignores `ExpertsLayout.gate_up_order` (`.up_gate` for flashinfer
  without quantization). This is harmless while `Backend.auto` picks
  flashinfer only for NVFP4; `Source.concatenate` makes the order explicit.

## Twenty-fourth pass: Source phase 1 implemented (2026-10-18)

The user approved the proposal above ("Ok do it"). Phase 1 is in place in
both repositories; nothing is described, committed as final or pushed.

### jj state

- zml: `@` (`omuqxqpu`, undescribed) holds the Source work and these notes.
  Under it, `tnxxsqww` is a duplicate of the user's `c865a2cb` "zml: Add
  Shape.Partitionning" (bookmark `brabier/shape-partitioning@origin`), which
  the monorepo branch needs (`Sharding.Partitioning.Logical`,
  `Shape.partitioning()`, `eqlWithTagsAndPartitioning`). Two adaptations:
  its `io.zig` hunk now lives in `io/TensorStore.zig`
  (`maybeCreateHostPinnedTensor` takes a runtime partitioning), and
  `std.meta.fields` became `std.meta.fieldNames` (Zig 0.17). Below that is
  the side session's `mppyywvr`. `brabier/adaptive-concurrency` was not
  moved.
- monorepo: `@` (`rwqvkpqy`, undescribed) holds the llmd Source work. Under
  it, `vqvklyks` is a duplicate of the user's branch commit `rsqorxts`
  ("llmd/weights: use explicit tags and partitioning"), rebased onto the
  merge `rwotnntx` with its conflicts resolved: five hunks in
  `weights.zig` (the merge's async pack compiles and grouped `loadExecute`,
  plus the branch's `output_shape` arguments and `withLayout`), and qwen4's
  `GatedResidual`, written with `concatenateLinears`. The original branch
  and bookmark are untouched. `vqvklyks` alone does not build against the
  new zml, because it still calls `createConcatenated` and
  `maybeCreateComposite`; build it together with `rwqvkpqy`, or squash the
  two.

### zml

- `zml/io/Source.zig` (`TensorStore.Source`) implements Tensor's shape ops:
  `withTags`, `rename`, `renameTag`, `splitAxis`, `merge`, `reshape`,
  `flatten`, `insertAxes`, `appendAxes`, `squeeze`, `bitCast`, `slice`,
  `split`, `chunkExact`, `transpose`, `pad` (zeros only), `broad`,
  `concatenate` and `stack`. It also has `bind(mesh, partitioning,
  .{ .memory })` and `bindAs(shape, .{ .memory })` (the tags and sharding of
  a shape built elsewhere), and the aliases `Pad` and `Slice`. Pieces are
  affine lattices kept in flat row-major order, so reshapes never touch
  them. An op that addresses axes rewrites a piece in coordinates, cutting
  it at row boundaries only where a dimension carries, then flattens it
  again. `bind` lowers pieces to byte `Part`s. A whole checkpoint tensor
  stays a plain binding. Checkpoint tensors are deduplicated by registry
  entry (name pointer and offset), since every `View.source` returns its own
  copy.
- `View.source(subkey, tags)` and `maybeSource` create Sources.
  `createConcatenated`, `createComposite`, `ConcatenateOptions` and
  `Layout.fill` are gone. `Layout` is now `{ shape, parts }`, and `Part` is
  `{ source index, src_offset, dst_offset, width, dims[]{ count,
  src_stride, dst_stride } }`.
- Loader:
  - Checks are bytes only: the same-dtype and whole-row rules are gone.
    The layout must fill the tensor's bytes, and parts must reference
    sources the binding has, stay in bounds and not overlap.
  - Several parts may read one source.
  - A strided part becomes one load part per run, accepted for at most 64
    runs or for runs of at least 1 MiB; anything else is
    `UnsupportedLayout`.
  - A transformed binding with a layout feeds the assembled tensor to its
    executable as the one input; `loadExecute` checks it against
    `layout.shape`.
- `Tensor.Slice.absolute` ignored `.len` when `.end` was left at its
  default, so `.{ .start, .len }` took the rest of the axis. The first
  real-model run caught it (an LFM2 `in_proj` member got the rows of the
  members after it). `.len` is now checked first, with a test in `tensor.zig`
  and a `SliceByLength` Source case.
- Tests:
  - Every Source op is evaluated against the same function run as a Tensor
    program on the test platform, byte for byte: 22 cases, including
    interleaves, expert stacks, transposes, slices of interleaves, pads,
    broadcasts, bitcasts both ways and a 128x4 swizzle at small scale.
  - Exact lowering checks, and a size check: gate/up at Llama scale stays
    one piece per member.
  - Loader composites on 2 and 4 devices: concatenated, padded, row
    sharded, column sharded and replicated, in default and host pinned
    memory; per-device qkv interleave; expert stacks concatenated and row
    interleaved; split across two checkpoint tensors; hand-built invalid
    layouts; one source feeding several parts; a strided part expanded;
    a composite as an executable's input.
- Results: `bazel test //zml:test //vfs:test` passes on CPU. On the four
  B70s (`--//platforms:oneapi=true`) 320 pass and 11 skip, with master's
  moe "Triton compatibility matrix" test skipped by a temporary local patch
  (pre-existing Level Zero abort, see the twentieth pass), since reverted.
  `docs/learn/loader.md` describes Sources.

### llmd

- `linear.View` components (`weight`, `scale`, `bias`, `globalScale`) are
  Sources: a split slices the rows at each component's granularity.
  `linear.sameTensor` replaces the name-and-offset comparison.
- `Interleaving.finest` became `.per_shard`: one group per device of the
  output axis (`outputShards`), one when replicated. Every `.finest` and
  `.explicit` call site moved to `.per_shard` (qkv, gate/up, GDN qkvz,
  lfm2 `in_proj`, muse_glimmer qkvg, gemma4 qkv); `.explicit` stays for
  layouts a kernel imposes. The locals that only computed `rows_per_group`
  are gone.
- `bindFused`: copied components (weights, block scales, biases) are
  composites bound with `bindAs`, with no recipe or program. Broadcast
  scalars (per-tensor FP8 scales across members, NVFP4 globals) keep
  `Recipe.fuse` and `packFused`, because the loader does not repeat bytes
  yet.
- `bindExpert(..., .assembled | .program)`: concatenated or
  single-projection stacks (values, global and input scales, biases) are
  Source stacks bound with `bindAs`, each expert's bytes going only to its
  devices. Row-interleaved stacks (mxfp4) and swizzled scales
  (`.program`, used by the `stack_nvfp4_cutlass_block_scale` recipe) and the
  flashinfer NVFP4 path keep their programs.
- Also on Sources: same-dtype `concatenateRows`, `splitTensor`, and the
  deepseek41 engram tables (`pad` plus `bind`, host pinned).
- `program()` gives a composite binding one input of `layout.shape`.
  `normalizeBlockFp8` (CDNA3) and `maybeSwizzleBlockFp8Scales` (Blackwell)
  therefore compose with composites through their `.copy` source:
  `normalizeLinearBlockFp8` and `composeBlockFp8Normalization` were split
  out so a test can call them. `Recipe.Fuse`'s `.copy` kind is now reached
  only by the branch's FP8 transform test.
- The test fixture registers `tp_mesh` and `experts` itself, so
  `weights_test` no longer imports `parallelism.zig`, which would pull in
  every model.
- Tests in `weights_test`:
  - views split without registering anything (first-member dims checked
    too, which the slice bug slipped past);
  - `extractOutput` for 1 and 2 groups;
  - per-shard qkv loaded on 1, 2 and 4 CPU devices;
  - experts stacked on 1, 2 and 4 devices;
  - a fused block-FP8 weight normalized through the assembled input (FN to
    FNUZ bytes and doubled scales checked);
  - shared and independent FP8 scales; block-scale boundaries;
    `splitTensor`; n-gram rows.

  Mutation check: forcing one group fails two tests.
- Results: `bazel test --config=localzml //llmd/...` passes 7/7.
  `//llmd:llmd` builds with `--@zml//platforms:cuda=true`, `rocm=true` and
  `oneapi=true`.

### End to end on the four B70s (TP=4)

llmd release build with the August oneAPI plugin (master's plugin segfaults
compiling llmd): a private runfiles copy in the session scratchpad
(`e2e/`), with `libzml_oneapi.so` pointing at the cached
`libpjrt_oneapi.so` from manual-2026-08-17. Each run sends one greedy chat
request ("What is the capital of France? ..."):

| model | load | answer |
| --- | --- | --- |
| LFM2.5-1.2B-Thinking | 2.18 GiB, 3.09 s | coherent reasoning, Paris |
| Llama-3.1-8B-Instruct | 14.96 GiB, 10.0 s | "The capital of France is Paris." |
| Qwen3.5-4B (GDN) | 7.83 GiB, 8.0 s | coherent reasoning, Paris |
| gemma-4-26B-A4B-it (MoE) | 47.00 GiB, 11.4 s | "The capital of France is Paris." |
| Laguna-XS-2.1 (256 experts), `--io-impl=threaded` | 62.29 GiB, 9.4 s | coherent reasoning, Paris |

- These runs cover the loader-assembled `in_proj` split and interleave,
  per-shard qkv and gate/up, GDN qkv slices interleaved with z, and expert
  stacks with each expert on its own device.
- LFM2.5-8B-A1B: llmd reports `InvalidModelType` (unsupported model type,
  not related).
- Laguna with the default `--io-impl=zio` segfaults compiling its forward
  pass. The debug build shows a stack overflow on a zio coroutine
  (`laguna.compileModel` -> `FnExe.compile` -> `Compiler.emitMlir`, a
  149 KiB frame probe); zio stacks are capped at 8 MiB
  (`RuntimeOptions.stack_pool.maximum_size`). It compiles and serves with
  threaded Io. This does not involve loading, and whether it predates the
  merge or the branch is not established.
- Llama-3.1-8B takes 10 s here. Column-sharded `o_proj`/`down_proj` give
  one DMA piece per row, which took 7.5 s with this plugin in the twentieth
  pass. Fixed in the twenty-fifth pass (0.76 s).

### Left open

- Phase 2: strided parts in the loader (mxfp4 row-interleaved experts,
  broadcast scalars), then retire `Recipe.fuse` and the interleaved
  `Recipe.stack`. Done in the twenty-sixth pass. Phase 3: swizzles as Source code. Phase 4: programs that
  take Sources as inputs, placed like their outputs (composites are fed
  replicated today, as single checkpoint inputs were).
- The zio coroutine stack overflow compiling Laguna.
- Not run: CUDA and ROCm hardware (the Blackwell swizzle and CDNA3
  normalization composition were tested only through the CPU FP8 test),
  and DeepSeek-V4.1.

## Twenty-fifth pass: column-sharded loads staged by destination (2026-10-18)

Request from the user: keep threaded Io on Intel; 10 s to load Llama is
far too slow; one device should load at 20-30 GiB/s and TP=4 not far
below.

### Cause

Measured with `examples/io load <model> tp`, a new mode that shards like
llmd's tensor parallelism (`o_proj`/`down_proj` on their input axis, the
rest on their first axis), August oneAPI plugin, four B70, page cache warm:

| layout | planned DMA submissions | loader elapsed | `dma_piece_latency_ms` |
| --- | --- | --- | --- |
| rows (`sharded`) | 3,034 | 0.673 s | 0.43 |
| tp, file-order blocks | 1,051,178 | 5.144 s | 1.22 (pumps stopped full 1.05M times) |

`DispatchSpans` gives a column-sharded tensor one span per row and device,
and the planner turned every span into a submission: 2-7 KiB pieces, at
most 64 in flight per device, ~1.2 ms each. Threaded Io did not change
llmd (9.9 s against 10.0 s with zio).

### Change (zml `direct_loader.zig`)

- The planner collects a DMA job's pieces first (read-relative, not cut by
  blocks), then places them (`JobCut.appendDmaTransfers`). In file order
  as before, unless staging saves at least `Config.minimum_staging_gain`
  submissions (default 64, one device's in-flight pieces) and the pieces
  tile the job's bytes once (overlapping parts never stage). A staged job's
  read lands in its blocks grouped by (destination, writer mask) in order
  of first appearance, each group sorted by destination offset, so a
  device takes one piece per block of its group.
- `Plan.segments` (`{offset, len}` u32 pairs, file order, merged where
  their landings touch) say where each run of the read lands;
  `Job.segment_start/segment_len/data_offset`, `Plan.direct_reads`,
  `staged_jobs` in the `batch planning` line.
- `readStaged`: a buffered read scatters straight into the blocks,
  `max_positional_iovecs` ranges per `preadv`, under one hold of the read
  gate. A direct read (iovecs must be aligned) lands in a per-worker bounce
  buffer, page aligned, allocated on first use at a job's block size, and
  is copied into place.
- Tests: the composite planner test also runs staged (gain 1, exact and
  widened reads) and replays each job's blocks from its segments; a new
  load test puts row-sharded and two column-sharded tensors (12,000
  one-byte pieces, several vectored reads) through a real load, exact and
  widened, and checks every shard. Mutating the buffered scatter or the
  bounce copy fails it.
- `docs/learn/loader.md` describes staging.

### Results

`examples/io`, Llama-3.1-8B, four B70, same plugin:

| run | submissions | staged jobs | loader elapsed |
| --- | --- | --- | --- |
| tp, warm | 5,468 | 599 | 0.714-0.745 s (~20-21 GiB/s) |
| rows, warm | 3,034 | 0 | 0.673 s |
| tp, cold (O_DIRECT, bounce copy) | 4,881 | | 1.631 s |
| rows, cold | 3,077 | | 1.604 s |

Cold runs evict the files with `posix_fadvise(DONTNEED)`; both layouts are
bound by the NVMe (Samsung 9100 PRO, ~9.2 GiB/s here).
`ZML_LOAD_CHECK=1` read back all 291 tensors (14.96 GiB) correctly, warm
and cold.

llmd end to end, `--io-impl=threaded`, August plugin, one greedy question,
`Loaded weights`:

| model | before (24th pass) | now | answer |
| --- | --- | --- | --- |
| Llama-3.1-8B, TP=4, warm | 10.0 s | 0.759-0.769 s (19.5-19.7 GiB/s) | Paris |
| Llama-3.1-8B, one B70 (`ZE_AFFINITY_MASK=0`), warm | | 0.658 s (22.7 GiB/s) | Paris |
| Llama-3.1-8B, TP=4, cold | | 1.834 s (8.2 GiB/s) | Paris |
| LFM2.5-1.2B-Thinking | 3.09 s | 0.134 s | reasoning, Paris |
| Qwen3.5-4B, warm | 8.0 s | 0.437-0.448 s (17.5-17.9 GiB/s) | reasoning, Paris |
| gemma-4-26B-A4B (47 GiB) | 11.4 s | 4.99-5.01 s (9.4 GiB/s) | reasoning, Paris |
| Laguna-XS-2.1 (62 GiB) | 9.4 s | 6.34 s (9.8 GiB/s) | reasoning, Paris |

- gemma-4 and Laguna do not fit beside llmd in this host's 62 GB of page
  cache: gemma's first file was 9% cached and read direct, the reads took
  13 ms per 8 MiB with no credit or DMA waits. They are disk-bound.
- One Qwen3.5 run at 0.96 s had its first file 66% cached; the warm runs
  are the two above (batch elapsed 0.38-0.39 s, 467 staged jobs).
- Planning stays 73-112 ms per batch (it was 0.102 s for the unstaged tp
  plan): `DispatchSpans` still enumerates one span per row and device.
- `bazel test //zml:test` passes on CPU; on the B70s 321 pass, 11 skip
  (the moe Triton test skipped by a temporary local patch, reverted).
  `bazel test --config=localzml //llmd/...` 7/7.

### Left open

- Planning cost: a strided span representation (phase 2's lattices) would
  let the planner cut a column-sharded tensor without enumerating its rows.
  Fixed in the twenty-sixth pass (1-4 ms).
- llmd TP=4 at 0.76 s against 0.66 s on one device: staged submissions
  cost 20-39 us each to submit (8 us for row pieces); not investigated.
- Not run on CUDA or ROCm hardware.

## Twenty-sixth pass: strided parts in the loader, placement without rows (2026-10-18)

Request from the user: "Ok do phase 2", after asking whether strided
support would cut the planning cost. Uncommitted in both working copies, on
top of the twenty-fourth and twenty-fifth passes.

### zml

- `zml/io/ShardGrid.zig` replaces `DispatchSpans.zig`. `Sharding.Placement`
  splits every axis evenly, so the devices hold the boxes of a grid, each
  packed row-major; devices with one box are one writer mask. Box order and
  logical order agree on a box's elements, so a contiguous logical range
  lands as one contiguous range of every box it touches: `rankIn(box,
  offset)` gives it by arithmetic. Aligned runs of `run` bytes (a shard of
  the innermost split axis) each land in one box (`boxOf`, `boxOffset`).
  Nothing lists rows: a column-sharded tensor was one span per row and
  device. Tests compare every range walk with the devices' slices, element
  by element (replicated, packed u2, mirrored, folded, 2D, 3D with two split
  axes, a scalar).
- Planner (`direct_loader.zig`): a read job's pieces are logical ranges of
  their destinations. File order lists runs (`appendRunTransfers`); a
  staged job lands each piece's bytes box by box (`JobCut.stage`), its
  `Item`s recording per box where they start in the blocks (`landings`);
  the worker walks the runs as it reads (`landJob`, `walkItem`, a scatter
  sink for buffered reads, a copy sink from the bounce buffer for direct
  ones). `Plan.segments` is gone. The staging decision estimates file
  order's submissions and lists runs only when the estimate is within
  merges and block crossings of deciding; a job already in file order (each
  box one run, groups in run order) is not staged, as before.
- Strided parts: the front end (`loader.zig`) still expands a strided part
  of at most 64 runs, or of runs of at least 1 MiB. A composite with any
  other strided part goes to the backend whole, as `LoadPart`s with `dims`
  (`backend.Lattice`, a cursor walks runs without a division each).
  `checkParts` requires strided parts to land in run order, in bounds, and
  proves two interleaved parts disjoint exactly when their footprints (the
  destination side, touching dimensions merged) repeat with one stride;
  other interleaving pairs are `UnsupportedLayout`, strided parts into host
  memory too.
- Gather jobs (`JobCut.appendGatherJobs`): a destination with strided parts
  is planned whole, with the file of its first part, cut into logical
  ranges of at most a job. Each range is one job: for every part with runs
  in it, an `Extent` reads the source range those runs come from, from
  whichever file holds it (`Planner.Reads` decides each file's alignment
  once, kept in `SourceSlot.alignment`), and the range lands box by box,
  one transfer per box and block, bytes no part covers zeroed. A part whose
  source runs are in order reads only its runs of the range, scattered
  straight into the blocks when the read is exact and tiled; a broadcast or
  a transpose reads its whole source range in every job through the bounce
  buffer. One that does not fit a read is refused at submit, before
  publication (`createDestinations`), not during planning.
- Buffered backend: a strided part reads its source range once and copies
  its runs into the staging tensor.
- `Source.stack(..., .last, ...)` did not compile (`rank()` is `u4`); no
  caller had used it. Fixed, with a Source case (`InterleaveExpertVectors`).
- Tests: a load through `Loader` on 1, 2 and 4 devices of a row interleave
  of two members from two files, scalar broadcasts and a padded transpose,
  sharded by rows, by columns and replicated, jobs of 128 bytes, exact and
  widened; the composite planner test replays staged jobs through
  `landJob` over an in-memory reader; a transpose larger than a read is
  refused at submit and the loader stays usable; the layout error test now
  loads the transpose it used to refuse and checks overlapping and
  unprovable strided pairs. Mutating the copy sink (source + 1) or the
  scatter sink (landing ^ 16) fails the gather, staged and column-sharded
  tests.

### llmd (`weights.zig`)

- `bindFused` `.broadcast`: each member's scalar is broadcast over its rows
  of every group, then concatenated and merged like the copied components
  (`Packer.Fusion`). Per-tensor FP8 scales across members and NVFP4 global
  scales no longer take a program.
- `bindExpert` `.assembled` takes interleaved stacks too: `Source.stack`
  of the projections at axis 2 (`.last` for per-expert vectors), then
  `reshape`. The `.program` path (swizzled block scales) records no recipe
  of its own; its caller records `stack_nvfp4_cutlass_block_scale`.
- Removed: `Recipe.fuse`, `Recipe.stack`, `Recipe.Fuse`, `packFused`, the
  `.fuse`/`.stack` sources of the FP8 normalization and swizzle recipes
  (they read the assembled tensor, `.copy`, the only case left), their
  compile arms and `bindSources`. `packStack` stays for the NVFP4 programs.
- Tests: interleaved expert weights and biases loaded on 1, 2 and 4 CPU
  devices (80 rows per member, so the loader gathers them); independent
  FP8 per-tensor scales of q (512 rows) and kv (256) repeated per shard on 1,
  2 and 4 devices; the split-views test now expects a composite and no
  recipe; the FP8 transform test feeds the assembled scales.

### Results

`examples/io load Llama-3.1-8B tp`, four B70, August plugin, warm:

| | before (25th pass) | now |
| --- | --- | --- |
| planning (4 plans) | 0.085 s | 0.002-0.004 s |
| published / sealed | +0.014 s / +0.086 s | +0.001 s / +0.002-0.005 s |
| loader elapsed | 0.714-0.745 s | 0.688-0.692 s |
| submissions, staged jobs | 5,468, 599 | 5,468, 599 |
| sharded (rows) | 0.673 s | 0.668 s |
| tp cold (O_DIRECT) | 1.631 s | 1.630 s |

`ZML_LOAD_CHECK=1` read back all 291 tensors correctly, warm and cold.

llmd, `--io-impl=threaded`, with the loader's debug scopes (a temporary
build, reverted): planning 1-2 ms per batch (73-112 ms in the 25th pass).

| model | load | jobs | answer |
| --- | --- | --- | --- |
| Llama-3.1-8B, TP=4 | 0.73-0.97 s (see below) | 474-884 staged, 0 gather | Paris |
| LFM2.5-1.2B-Thinking | 0.113 s (0.134 s) | 143 staged | reasoning, Paris |
| Qwen3.5-4B | 0.44 s warm (0.62 s first) | 260-467 staged | reasoning, Paris |
| gemma-4-26B-A4B | 5.02 s (5.0 s, disk-bound) | 376 staged | Paris |
| Laguna-XS-2.1 | 6.66 s (6.34 s, disk-bound) | 241 staged | reasoning, Paris |

- Llama's llmd loads varied more this time for both builds: the 25th-pass
  binary, which measured 0.76 s then, took 1.05 and 1.15 s alternating with
  the new one at 0.84 and 0.95 s; the new one gave 0.73 s right after
  `cat`-warming the files and 0.94-0.97 s otherwise, read-bound (4.3-5.7 ms
  per 8-16 MiB read, DMA pumps never full). The calibration also picked
  8 MiB or 16 MiB requests from run to run.
- No model here takes gather jobs: none has MXFP4 experts or per-tensor FP8
  scales across members. The gather path is exercised by the zml and llmd
  tests above, on CPU and on the B70s.
- Tests: `bazel test //zml:test //vfs:test` passes on CPU (322 pass, 12
  skip). On the B70s 323 pass, 11 skip (master's moe Triton test skipped by
  the same temporary patch, reverted). `bazel test --config=localzml
  //llmd/...` 7/7 (`weights_test` 16 tests). `//llmd:llmd` builds with
  `--@zml//platforms:cuda=true` and `rocm=true`.

### Left open

- No measurement with a model that gathers (DeepSeek MXFP4 experts, FP8
  per-tensor checkpoints): neither the checkpoints nor the hardware are
  here. For MXFP4 block scales (rows of 224 bytes) a buffered gather
  scatters one iovec per row, about 37k per 8 MiB job; whether a bounce
  copy beats that is unmeasured.
- Two strided parts interleaving with different strides are refused
  (`UnsupportedLayout`) rather than proven disjoint; none of llmd's layouts
  does that.
- Phase 3 (swizzles as Source code): a permutation's source is read whole
  in every job, so it must fit one read; per-expert swizzles do, a
  model-wide one would not.
- A destination with strided parts is planned whole in the plan of its
  first part's file, so its jobs are claimed with that file's.

## Twenty-seventh pass: swizzles as Sources, DeepSeek on GB300 (2026-10-20)

Phase 3 of `TensorStore.Source`: the block-scale swizzles are written with
Source ops and the loader places them from the checkpoint; their pack
programs are gone. Measured on gb300-1 (2 of 4 GB300, 953 GB RAM, four
NVMe in RAID 0), DeepSeek-V4.1-Flash, warm page cache, llmd with the
loader's debug scopes (a temporary local edit there, not committed).

### zml

- `Lattice.copyRuns`: copies a range of runs with nested loops over the two
  innermost dimensions, a run of 1, 2, 4, 8 or 16 bytes as one unaligned
  load and store, others by `memcpy`; head and tail rows go run by run.
  The direct loader's bounce copies use it a stretch at a time
  (`copyItem`): a stretch is a destination range inside one run of the
  grid and one block, so it lands linearly; the runs a stretch's ends cut,
  and items with fewer than 16 runs per stretch, keep the per-run walk
  (`walkRuns`). The buffered loader copies strided parts with it too.
- Disjointness of strided parts (`checkParts`, which now takes an
  allocator): parts are grouped into clusters whose spans meet. A cluster
  of more than 8 parts with at most 2^21 runs lists and sorts its runs (a
  column-split swizzle: 384 parts each spanning the tensor made the
  pairwise check quadratic, 20 s of DeepSeek's front end). Otherwise pair
  by pair (`overlaps`): exact by bisection when one part has at most 1024
  runs; else a bounded search over index differences when the footprints
  repeat with the same strides, merged or, failing that, as they are
  (counts and widths may differ); else `UnsupportedLayout`.
- `Source.ofBinding` / `TensorStore.sourceOf`: the source a binding loads
  (a checkpoint tensor relabeled, or the layout's parts as pieces), in a
  tensor's shape; `Source.rebind` binds a source in place of a tensor's
  binding, in its memory. llmd reworks a `Linear`'s bound scales with it.
- `Source.transpose` treats a permutation that only moves axes of one
  element as a reshape. The FP8 swizzle's `(qout, device, k)` transpose
  with one column device otherwise cut its lattice into 192 pieces.
- Tests: `copyRuns` against the cursor on random lattices; footprint and
  part overlap verdicts against byte enumeration (same strides, different
  widths and counts, few and many runs); clusters of 9-16 interleaved
  parts. The layout error test's unprovable pair now has over 1024 runs per
  part (the 128-run pair is now proven). Shifting the fast copy's
  destination by one fails three zml and two llmd tests.

### llmd (`weights.zig`)

- `swizzleBlockScales(T, stacked)` and `swizzleBlockFp8Scale(T, scales,
  spread, shape)`, generic over `Tensor` and `TensorStore.Source`: the
  former programs' ops, unchanged.
- `bindExpert` stacking `.swizzled` (cute MXFP4 and FlashInfer NVFP4 block
  scales) and `.fp4` (FlashInfer NVFP4 weights: the stacked bytes bitcast
  to `f4e2m1`). `bindExpertNvfp4CutlassWeight` and
  `bindExpertNvfp4CutlassBlockScale` are gone.
- `maybeSwizzleBlockFp8Scales` → `swizzleBlockFp8Scales` per projection:
  `store.sourceOf(scales)`, swizzled, `rebind`; fused scales included.
- Removed: `Recipe.stack_nvfp4_cutlass_weight`,
  `stack_nvfp4_cutlass_block_scale`, `swizzle_block_fp8_scale`, `Stack`,
  `StackNvfp4CutlassWeight`, `packStack` and the three pack programs.
  NVFP4 global scales and dequantizations stay programs (they compute).
- Tests (CPU, through the loader, against the kernels' index formula):
  interleaved expert scales padded to 128x4 on 1, 2 and 4 devices; FP4
  relabel; block-FP8 scales plain and fused (DeepSeek's `wq_a`+`wkv`),
  split by rows and by columns, on 1, 2 and 4 devices.

### Results

DeepSeek-V4.1-Flash, `CUDA_VISIBLE_DEVICES=0,1`, `--token-batch-size=256
--batch-size=1`, cute MXFP4 experts, 467 GiB:

| | baseline (26th pass) | swizzles as Sources |
| --- | --- | --- |
| loaded weights | 79.6 s (first), 66.5 s warm, 7.0 GiB/s | 15.4 s, 14.9 s, 30-31 GiB/s |
| pack programs | 10 programs, 330 tensors, 3 execute batches (17 GiB) | 1 program (`wo_a` to BF16), 40 tensors |
| main batch | 450 GiB in 65 s | 466 GiB in 14.9 s |
| worker wait in destination creation | 11.3 ms per read | 0.16-0.2 ms per read |
| gather jobs | 10,840 | 12,090-23,890 |

- The baseline's workers spent most of their time creating destinations:
  every per-expert scale was its own program input, about 50K device
  buffers of 368 KB, each a `createBuffersForAsyncHostToDevice` per device.
- First swizzle build: 39.5 s, of which 20 s were the front end's quadratic
  overlap check (above), before the main batch was published.
- Answers: "Paris" and "391" (17 x 23), with reasoning.
- The calibration picked 16 MiB requests in one run and 8 MiB in the next.
- Width sweep, DeepSeek's layer forward stubbed out so compilation takes
  2.3 s instead of 67 s (a temporary edit on gb300-1, as are the width
  environment variable and copy counters below): width 16 13.3 s (35.1
  GiB/s), 24 12.0 s, 32 11.7 s, 48 11.2 s (41.5 GiB/s); the full model at
  width 32 12.45 s, answers unchanged. Sixteen `dd` with O_DIRECT read
  57 GB/s from the RAID.
- Bounce copies (`landExtent` through `copyItem`): 94.8K extents, 119.8
  GiB, 22-27 s of worker time summed (about 1.5 s per worker at width 16).
  `direct_io=auto` reads a file buffered when it is mostly cached, so how
  much goes through the bounce buffer follows the page cache of this
  shared host: 16.5 GiB in one run.
- `bazel test //zml:test` 326 pass, 12 skip; `//llmd:weights_test` 19
  tests pass.

### Left open

- Model compilation takes 67 s before the load starts; llmd compiles in
  `init` and loads after, so overlapping the two would hide most of the
  load.
- The default width (16) leaves DeepSeek 15% slower than 48 on this host;
  the GB300 sweeps behind the default were flat from 12 to 48 on lighter
  models.
- Destination creation still costs 22-27 ms for some tensors (40 above
  20 ms in one run).
- Not run: NVFP4 (FlashInfer) checkpoints, Blackwell FP8 models other than
  DeepSeek, TP=4.

## Twenty-eighth pass: programs over Sources, expert layouts on the device (2026-10-20)

Phase 4 of `TensorStore.Source`: a program's inputs are Sources, each placed
by its own shape. The user, skeptical of swizzling on the host, asked to
compare the loader's small-run copies with programs doing the same layouts
from an intermediate buffer.

### zml

- `TensorStore.Binding.transformed` is gone. A program's binding has
  `inputs: ?[]const Layout`, one per program input, each placed by its
  shape; `Layout.whole` spots an input that is one whole checkpoint tensor.
- `Source.place(mesh, partitioning)` / `placeAs(shape)` make a
  `Source.Input`; `Source.bindProgram(inputs, output)` binds the tensor a
  program computes (checkpoint tensors deduplicated across inputs).
  `bind`, `bindAs` and `bindProgram` share one lowering. A bound layout's
  shape is now placed like its tensor.
- `TensorStore.unbind(tensor)` returns and removes a binding (`rebind` uses
  it). `sourceOf` is null for a program's tensor.
- `View.maybeCreateBinding(keys, shape)` is now `bindProgram` over whole
  checkpoint tensors, replicated.
- `loadExecute` lands each input from its layout (whole tensors as before,
  composites gathered and zero-filled); a non-program binding is one input,
  the tensor as `load` fills it. Inputs are checked against the executable
  for count, dims and placement (`validateSamePlacement`, formerly
  `validateOutputPlacement`).
- Test: a program over a concatenated composite and a whole tensor, its
  result checked, and executables of another input count or shape refused.

### llmd (`weights.zig`)

- `normalizeBlockFp8`: `sourceOf` + `bindProgram` placed like the tensor
  (each device converts its slice; the old composite input was
  replicated), then `unbind`. `normalized` maps a replaced id to its
  conversion for copies of a Linear.
- `bf16Linear` (`wo_a`): weight and scale grid as Sources placed like the
  result, the old bindings unbound (it used to rewrite the binding's
  tensor shapes in place).
- NVFP4 globals: two stacked inputs (`stackExpertScalars`) instead of one
  input per expert and projection (768 for 384 experts).
- `Packer.expert_layout`: `.device` (default) binds each expert stack with
  its projections concatenated (whole per-expert tensors, large runs) and
  records `Recipe.lay_out_experts`, whose program runs
  `layOutExperts(Tensor, ...)`: the row interleave (split, transpose,
  merge) and `swizzleBlockScales`. `.loader` keeps the phase 3 Sources.
- Tests: interleaved and swizzled expert stacks on 1, 2 and 4 devices in
  both layouts; NVFP4 globals on 1 and 2 devices. Mutations (no interleave
  in the program, `min` for `max` in the weight global) fail them.
  `bazel test //zml:test`, `//llmd:weights_test` (20) pass.

### Results

DeepSeek-V4.1-Flash on gb300-1, GPUs 0 and 1, forward stubbed (compile
2.3 s), width 16, runs alternated. Temporary instrumentation: bounce-copy
counters, program execute time, `TMP_DIRECT_IO`, GPU memory sampled every
250 ms (the larger GPU per run, KV cache included).

| direct_io | layout | load | bounce copies | worker copy time | executes | peak GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| auto | loader | 14.43 s, 13.71 s | 76.8, 77.1 GiB | 17.7, 17.5 s | 40: 3.1, 2.4 s | 243.6, 246.5 GiB |
| auto | device | 13.36 s, 12.86 s | 1.2 GiB | 0.7 s | 160: 4.2, 4.3 s | 249.4, 252.2 GiB |
| on | loader | 12.98 s, 13.11 s | 188.6 GiB | 36 s | 40: 2.5, 2.7 s | 246.5, 246.7 GiB |
| on | device | 12.88 s, 12.69 s | 3.1 GiB | 1.0 s | 160: 3.5 s | 252.5, 251.4 GiB |

- Device layout: 41 submissions (40 pack groups of 4.65 GiB, 2.3 GiB in and
  out per device), 19-22 admission retires, minimum room 45 GiB. 4
  programs, compiled before their groups are submitted: 1.8-2.4 s, which
  the reads of earlier groups only partly hide. The 120 layout executes
  cost about 1.1 s in all (9 ms each).
- The 40 `wo_a` dequantizations take 60-77 ms each (2.4-3.1 s) while the
  bulk load runs. Not attributed.
- Full model, device layout: compiled in 69.4 s, loaded in 13.2 s
  (35.4 GiB/s; the loader layout took 14.9-15.4 s in the twenty-seventh
  pass), answers "Paris" and "391". Another user's job took all four
  GPUs from 20:17:51, possibly during this load; the eight runs above
  (20:00-20:05) had the GPUs to themselves.
- Host copies are not this host's bottleneck: with 217 threads the loader
  layout still loads at 35 GiB/s while copying 189 GiB. A host with fewer
  cores would show more of the difference.

### Left open

- Compile the pack programs during model compilation, so the device
  layout's first groups do not wait 2 s for them.
- Not run: NVFP4 (FlashInfer) checkpoints, TP=4, MI300 (FP8 normalization
  now converts per device slice).

## Twenty-ninth pass: profile of DeepSeek's load (2026-10-20)

Where DeepSeek-V4.1-Flash's load time goes, compilation overlap aside. Same
setup as the twenty-eighth pass (gb300-1, GPUs 0 and 1, forward stubbed,
device layout, width 16), the committed code plus temporary
instrumentation: a 100 ms sampler (md127 bytes, every thread's CPU ticks and
state), log lines stamped with the monotonic clock, `zml/vfs` debug logs,
and one run under `sudo nsys profile --run-as=benjamin` (CUDA and OS runtime
trace, CPU sampling with DWARF backtraces). Scripts in gb300-1
`~/phase5/` (`setup5.sh` applies the code, `run5.sh` runs, `nsysan.py`,
`window.py`, `stacks.py`, `apis.py`, `timeline.py` read the report). The
GPUs were taken by another job after these two runs, so nothing below is
validated by a fix.

Baseline: 467 GiB in 11.9 s (39 GiB/s); under nsys 14.8 s.

### Findings

1. **A 2.4 s stall when the engram tables are allocated** (4.6 s under
   nsys). DeepSeek binds `layers.{1,14}.engram.embed.{weight,scale}`
   (2 x 91.6 GiB FP8 plus 2 x 2.9 GiB) to `.host_pinned`, row-split over the
   two devices: 4 host destinations, 8 shards. The main batch's allocation
   task creates them (`createUninitializedBuffer` on the pinned host
   space), and the plugin registers each shard with `cuMemHostRegister`:
   0.56, 0.57, 0.62 and 2.5 s for the 46 GiB shards, one at a time. Each
   call holds the driver lock. Meanwhile `cuMemSetAccess` (up to 2.5 s),
   `cuEventRecord` (3.5 ms each) and the uploads wait, so the pumps stop,
   the 33 credits stay held and every worker waits: no H2D and no disk
   reads for the whole window, about 2 cores busy. The batch log shows it
   too: groups 21-39 complete within 5 ms of `batch allocations ... done`.
   Nearly all of the run's credit waits fall in this window
   (`credit_wait_ms_per_read=0.696` x 60,667 reads = 42 s of worker time,
   17 workers x 2.4 s = 41 s). The main batch is published only after the
   pack groups (+1.9 s, waiting on the pack-program compiles), so the stall
   lands mid-load.
   CPU-only measurements on gb300-1 (46 GiB anonymous, 64 KiB pages):
   `mlock` of unfaulted memory 0.48 s (the driver's 0.56 s is mostly this
   fault-in under its lock); faulting first takes 0.17 s on 16 threads
   (0.8 s on one), after which `mlock` takes 0.2 s. `MADV_HUGEPAGE` got no
   huge pages: THP is 512 MiB on this 64 KiB-page kernel, mTHP sizes are
   disabled (the loader's own arena aligns to 2 MiB, so it gets none
   either). The fault-in is not zeroing: gb300-1 boots with
   `init_on_free=1`, so pages are cleared when freed and allocations take
   them already clear; one thread clears 56 GiB/s, slower than the 90-100
   GiB/s of `mlock`, `MADV_POPULATE_WRITE` and `MAP_POPULATE`, which all
   take ~0.5 s per 46 GiB whatever the thread count (only user faults on
   16 threads reach 0.18 s). The cost is per page (~750K pages of 64 KiB
   per shard); 2 MiB mTHP is disabled and the hugetlb pools are empty (both
   need root). Clearing would come back on an allocation that has to
   reclaim (the freed pages are cleared on that thread), possibly the one
   46 GiB `mlock` run that took 9 s; not verified. A read-only mapping of a
   cached checkpoint file is per-page work too: 0.16 s per 46 GiB on 16
   threads.
2. **Otherwise the load is read-bound.** Workers spend ~70% reading,
   22% in credit waits (the stall) and 7% creating destinations
   (`tensor_init_ms_per_read=0.229`). Outside the stall: pack groups at
   ~52 GiB/s, the rest ~50 GiB/s; the copy engines are ~40% busy and the
   pumps mostly stop empty. A read averages 7.9 MiB in 2.2 ms (3.5 GiB/s per
   worker). Buffered reads of cached data into private buffers measured
   7.6-7.9 GiB/s on one thread, 85 GiB/s on 16, 96 GiB/s on 32; the RAID
   reads 57 GB/s with O_DIRECT.
3. **O_DIRECT re-reads cached data.** At load time 311 of 467 GiB were in
   the page cache, but 205 GiB came from disk. `direct_io=auto` reads a file
   direct when it is at most 50% cached: 16 files at 38-50% (99 GiB) were
   read direct, re-reading their cached 44 GiB from disk. A per-range
   choice would have read 156 GiB.
4. **Destination creation uses the VMM allocator** (llmd configures
   `.vmm`): each device buffer is `cuMemCreate` + `cuMemAddressReserve` +
   `cuMemMap` + `cuMemSetAccess` (4,748 `cuMemSetAccess` in the load, p50
   142 us, p90 706 us under nsys), on the workers' path: ~14 s of worker
   time, ~0.8 s of wall time.
5. **The engram host jobs form the tail.** The last ~2.5 s have no H2D:
   11-13 workers copy the engram files (buffered, 66-78% cached) into the
   pinned tables, 10-28 GiB/s from disk. nsys sampled no kernel stacks, so
   the split between page-cache copies and readahead is unmeasured.
6. **NCCL init holds one CPU for ~35 s.** `preloadLibraries` →
   `ncclCommInitAll` → `ncclInitKernelsForDevice` → `cudaFuncGetAttributes`
   → `cuLibraryLoadData`, in libcuda the whole time. NCCL 2.30.4 ships
   cubins for sm_75/80/90/100/110/120 and PTX for sm_120, none built for
   sm_103 (GB300); `~/.nv/ComputeCache` holds 96 KiB. Hidden by the full
   model's 67 s compile; with the stub llmd waits on it after the load
   (9.7 s under nsys, ~16 s in the baseline), so any short compile (cached
   executables) puts it on the critical path. It also competes for the
   driver lock while it runs.
7. **Device 1's uploads are slower per copy.** CUPTI reports them as
   peer-to-peer from pinned memory (the arena is registered through one
   client call); for copies of 8 MiB and more, median 67 GB/s against
   84 GB/s on device 0 (p90 112 against 208). Not a limit today (copy
   engines ~40% busy).
8. Smaller: the front end publishes the first group 0.17 s after the load
   starts; tokenizer, platform and loader creation take ~2 s before
   compilation.

### Suggested next steps (need GPUs)

- Engram stall: let the loader allocate host destinations itself (the
  arena's allocator: placed pages, `DmaMap`, a PJRT view freed by its
  delete callback), fault them in on several threads (or let the workers'
  copies fault them) and `DmaMap` afterwards, so the driver holds its lock
  ~0.2 s per 46 GiB instead of ~0.56 s, ideally after the device DMA.
  First check that a view of `DmaMap`ped memory in the pinned host space
  works for the executables' device-kind views.
- Width sweep again once the stall is gone; the 16-48 sweep of the
  twenty-seventh pass ran with the stall in it.
- Per-range direct or buffered reads (residency of each job's range, e.g.
  `cachestat`), with a second descriptor per file for the direct reads.
- NCCL: time `preload` alone, with a larger `CUDA_CACHE_MAXSIZE`, and with
  an NCCL build carrying sm_103 (or sm_100f) code.

## Thirtieth pass: end-to-end loader review (2026-10-21)

A read-only review of the whole load path, from llmd (`main.zig`,
`models.zig`, `weights.zig`, the model files) through `zml/io` and `vfs/`,
against the brief "code earns its place, fewer abstractions, logic local".
The full report is `LOADER_REVIEW.md`:
- "Read this first" ranks what matters.
- Part 1 covers bugs and missing parts, tests and docs.
- Part 2 covers the simplification plan (ten moves), performance, and what
  earns its place.
- Appendix A lists every finding.

Method:
- Twelve reviewers (eight by area, four cross-cutting) and two completeness
  rounds produced 185 findings.
- Each finding was attacked by one to three verifiers. 184 survived, 123 of
  them with corrections; 1 was refuted (Appendix B).
- Nothing was built or run.

Headlines, each checked against the code:
- The Source op harness (`Source.zig:1253`) runs zero cases:
  `std.meta.declarations` lists only `pub` declarations, and every case is
  private.
- `//llmd:test` no longer compiles: `deepseek41.zig:3214` reads the
  `Binding.transformed` field that the twenty-eighth pass removed.
- llmd's quantized bindings have three correctness bugs:
  - compressed-tensors NVFP4 expert globals ignore `.divide` on the
    FlashInfer path;
  - dense NVFP4 drops `input_scale`;
  - FP8 per-tensor gate/up experts get `[E,2]` scales.
- llmd never passes `io_impl` to `Platform.auto`, so under zio compiles skip
  `blockInPlace` and run on 8 MiB coroutines. This is the likely cause of
  the Laguna stack overflow.
- Remote reads have no stall timeout, and HF re-resolve frees URLs that are
  still in use.
- The `auto` O_DIRECT probe warms the pages it samples, so the next load
  reads that file buffered.
- Error paths:
  - pack-execution failures panic in `Exe.call`;
  - `deinit` with pending submissions reads and transfers them all;
  - progress stops after about 125 destinations;
  - `Workspace.init` hits `unreachable` on mixed GPU kinds.

The byte core (Source lowering, ShardGrid, planner, FIFO claims, pumps,
credits, the drain) held on every traced path. The simplifications with the
best value per risk:
- the credits as the only pinned-memory bound (`BlockPool`'s wait cannot
  fire), about 250-300 lines;
- validation once and no overlap prover, about 300 lines;
- one host-resident path;
- `Pipeline` folded into `Loader`.

The review's move 5 proposed a fixed DMA block size per target in place
of `dma_calibration.zig` (about 850-900 lines with the growth API). The
user declined it on 2026-10-21: calibration stays, so do not propose
removing it again. Only the review's "If declined, do the minimum" items
remain:
- delete `findArena`, which is always null in production;
- tests pin one block size;
- CPU calibration honors a one-element list.

## Thirty-first pass: fixes from the loader review (2026-10-21)

The bugs the thirtieth pass headlined are fixed, along with steps 1 and 2 of
its simplification move 2. Each commit was reviewed adversarially (12
reviewers; 31 findings, none high), and the follow-ups are their own
commits. GPUs: none of gb300; four B70 (oneAPI, threaded io) for the
loader smoke test.

zml commits, in order (on top of `qztxwlmv`):
- `msskpwqx`: the Source op harness runs its 23 cases. They were private,
  and `std.meta.declarations` lists only public ones. `WidenBits` is fixed,
  and every case round-trips through sourceOf/rebind.
- `tnwrwmmu`: progress counts one item per destination, by an unread-bytes
  countdown, on the caller's node (#19). Backends take
  `std.Progress.Node` by value.
- `uxvqmwtu`: mixed device kinds load (#9).
- `kuwqpzmn`: `Exe.call` returns `pjrt.ApiError`. A failed pack is the
  loader's sticky error, not a panic (#17).
- `uyrpwyxp`: `Backend.abort`. The first failure aborts what is pending,
  and so does deinit (#18).
- `zmvvsmnw`: the `auto` probe draws a random shift per probe, so it no
  longer reads its own planted pages (#7).
- `knuqkqxr`: a request attempt is cancelled and retried after 60 s with
  no bytes (`RetryConfig.stall_timeout`, #1). A send failure closes its
  connection.
- `tqpmrsnq`: each HF read copies the download URL. Logs name the file,
  not the signed URL. The tree cache is locked (#2, #14).
- `xmrzxtsl` (move 2, step 1): `BlockPool` never waits. The credits are
  the one bound on pinned blocks.
- `tmtwwsts` (move 2, step 2): a credit is a fixed slot of
  `maximum_blocks_per_job` blocks. The following go:
  - `BlockPool` and the per-block `Lease`;
  - `BlockContext` and `abandonSubmissions`;
  - the plan's block contexts and `block_start`.

  A DMA piece carries its request and data. The diff is 275 lines
  shorter. Benchmarked against `xmrzxtsl` on GB300 in the thirty-third
  pass: no load-time change detected at about ±2.5%. The summary's pinned
  high water became `credits_high_water` (a count) in `zqzrpnor`, so it
  does not compare.
- `zqzrpnor` (review follow-ups):
  - `Loader.cancel()`, which deinit calls and llmd uses;
  - the device bound is `Platform.MAX_NUM_DEVICES` (32 without TPU);
  - the buffered backend counts one item per tensor;
  - doc and comment fixes.
- `sxymmslo` (review follow-ups):
  - the stall clock pauses while an attempt runs its own governed request
    (a HF re-resolve or a GCS token refresh);
  - an outer cancel keeps a finished value;
  - a `receiveHead` failure closes its connection;
  - the probe no longer divides by zero;
  - HF leak and order fixes, and HF open failures are logged.
- `lmnpvsso`: `pjrtx` compiles on the calling thread. `io.async` had put
  XLA's compile on a zio coroutine.

monorepo commits (on top of `oxpzxnvt`):
- `vqkkvptr`: llmd passes `--io-impl` to `Platform.auto` (#4). Under zio,
  compiles then use `blockInPlace`; with `lmnpvsso`, neither MLIR emission
  nor XLA's compile runs on an 8 MiB coroutine. This is the likely fix for
  the Laguna zio crash. It is not verified: that needs a Laguna run under
  zio, which the B70 rule (threaded io) excludes here.
- `pmnrknqz`: `//llmd:test` compiles again (`binding.transformed`).
- `ulnnrozk`: llmd tries `Exe.call`.
- `pxzvsqqp`:
  - compressed-tensors NVFP4 FlashInfer globals honor `.divide` (#3);
  - dense NVFP4 keeps `input_scale` (#10);
  - FP8 per-tensor gate/up experts get per-row scales `[E, 2n, 1]`,
    per channel (#11);
  - NVFP4A16 FlashInfer experts are an error, not a panic.
- `kvklzrov`:
  - one `Operation` for both NVFP4 global recipes;
  - tests through `Packer.experts` (FlashInfer) and fused NVFP4;
  - `loadInto` cancels the loader on error before freeing its programs.

Verification:
- `bazel test //zml:test //vfs:test` and `bazel build //examples/...` pass.
- `bazel test --config=localzml //llmd/...` passes 7/7, and
  `bazel build --config=localzml //llmd/...` passes.
- Four B70: `examples/io load Llama-3.1-8B-Instruct tp` with
  `ZML_LOAD_CHECK=1` reads back 291/291 in 1.45-1.62 s, cold (O_DIRECT).
  Host-pinned `sharded` also reads back 291/291.

Left from the fix review:
- The asynchronous half of `Exe.call`'s failure path has no test.
- The buffered backend's abort lets admitted tasks finish their tensor.
- Under `std.Io.Threaded` with a finite `async_limit`, the per-attempt
  watch task halves the buffered backend's remote read parallelism.
- No test covers the range-read progress plumbing.
- HF has no test: it needs an injectable Hub endpoint.

The review's other Part 1 items are open, for example:
- S3 credentials and key encoding (#5, #12);
- s3/https missing optional files (#6);
- `source_slots` keeping borrowed URIs (#8);
- https redirects (#13);
- GCS key files (#16);
- safetensors header validation (#20);
- the `loadExecute` plain branch (#21).

## Thirty-second pass: API review of the loader (2026-10-21)

A read-only review of the loader's APIs, from llmd's and the examples' call
sites down to the backends' seams, against the same brief plus "focus on the
APIs". The report is `LOADER_API_REVIEW.md`:
- "Read this first" lists the bugs found through the APIs.
- Part 1 is the recommended API, Part 2 its sixteen moves (R1-R16), and
  Part 3 the decisions it needs from the user.
- Appendix A lists every finding.

Method:
- Twelve reviewers plus four gap reviewers reported 123 findings.
- 36 duplicates were folded and 3 restatements dropped.
- Each of the 78 remaining findings was attacked by one or two verifiers:
  77 survived, almost all with corrections, and 1 was refuted.
- Three designers (caller-first, seams-first, incremental) and a judge
  produced the design. Nothing was built or run.

Bugs, each checked against the code:
- examples/llm deinitializes the stale init-time copy of its Loader
  (`main.zig:86-92` vs `:145`): `Future.cancel` after `await` returns the
  stored result. The live loader leaks its deque and keeps the pinned
  memory for the whole session. AGENTS.md's future idiom produces exactly
  this for a by-value handle.
- Direct `submit` still returns `Closed` instead of the cause (#29).
- The front end's `failure` is not set when `backend.submit` fails after a
  partial publish, so a later `awaitAll` returns success. Latent.
- DeepSeek's DSpark projections and `main_proj` miss the sm_10x swizzle
  opt-in (`deepseek41.zig:1001` vs `:3779-3923`, `:4014`); their scales are
  e8m0 block-32.
- qwen4 frees `ngram_host_buffers` with the arena after `arena.deinit()`
  (`qwen4/spec.zig:432`, `:485-487`).
- `resolveModelRepo`'s directories are never closed.
- #30, #35 and #36 are still open.

Recommended API:
- one `load(allocator, model, store, progress)` that returns the
  `Bufferized` tree and borrows nothing after it returns;
- programs recorded in the binding (`Source.bindProgram(inputs, output,
  func, args)`), so `loadExecute`, `Loader.Binding`, the `delivered` rule and
  llmd's recipes and Packer go;
- `Loader.init` returns at once and runs calibration on its own task;
- the store owns the registry;
- `VFS.Remotes` bundles the remote backends under one error contract;
- block-FP8 for the platform is decided at bind time;
- placement is a Source property set last.

Decisions it reverses or needs:
- reversals: CTX:3082 (load/awaitAll split), CTX:62-69 (caller-owned
  totals), CTX:3829-3838 (asserts on options), CTX:4973-4974 (memory as a
  bind option), CTX:5024-5026 (via Move 3);
- the user's call: dropping the block-FP8 scale-overflow guard;
- needs a GPU first: `host_mapped` (a device view of loader-owned pinned
  memory) is unverified on CUDA.

## Thirty-third pass: benchmark of move 2 step 2 (2026-10-21, gb300-1)

Question: does step 2 (`tmtwwsts`, a credit is a fixed slot of blocks)
change load time against step 1 (`xmrzxtsl`, a pool leased under the
credits)? The two are adjacent commits; monorepo at `pxzvsqqp`, which
builds against both.

Method:
- gb300-1 (four GB300, two Grace nodes, 953 GB RAM; shared). Both trees
  were applied as patches to gb300-1's checkouts and built once. Their
  `llmd` and `examples/io` binaries were saved, and every run went through
  a `--run_under` wrapper (`~/phase6/swap.sh`) that execs the saved binary
  of the run's tree with the fixed tree's runfiles. The reason:
  `zig build-lib //llmd:llmd` misses the remote cache on every source
  change, which costs 7 to 12 min each time the tree switches. Every log
  records its binary's sha; all 68 runs match their tree.
- Rounds alternate AB, BA, AB, ... A run waits until its GPUs are free
  and records the other users' GPU processes and the load average.
- Workloads:
  - llmd DeepSeek-V4.1-Flash (467 GiB, forward stubbed, the twenty-ninth
    pass's setup) on GPUs 2,3: 24 rounds.
  - `examples/io` Qwen3.8-27B (51.75 GiB) replicated and sharded on GPUs
    2,3: 5 rounds each.
- GPUs 2,3 are on node 1. Another user's llmd and then sglang held GPUs
  0,1 from round 4 on, so most rounds share the host.
- Scripts and logs are in gb300-1 `~/phase6/`: `bench2.sh`, `runds2.sh`,
  `runio2.sh`, `extract.py` (writes `runs.jsonl`), `parse.py`. The local
  `analyze.py` pairs the rounds.

Correctness: `ZML_LOAD_CHECK=1`, Qwen3.8-27B replicated on all four GB300:
1199/1199 tensors for both trees (4.13 s and 4.24 s, first and second
load).

### DeepSeek llmd, 24 rounds

A pair is comparable when both runs calibrated the same request size and
neither stalled in host allocation (`ahead_init_ms` < 4 s; the reclaim of
node 1's page cache for the 189 GiB pinned engram tables took up to 20 s).
15 of the 24 pairs are comparable.

| B vs A over the comparable pairs | value |
| --- | --- |
| median, Hodges-Lehmann | +0.86%, +0.93% |
| mean, 95% bootstrap CI | -0.08%, [-2.8%, +2.2%] |
| B slower | 9/15 (sign test p=0.61, Wilcoxon p=0.39) |
| without the 5 pairs whose GPU neighbours changed mid-run | +1.95%, [+0.4%, +3.5%], 7/10 |
| the same, with `ahead_init_ms` as a covariate | +0.8%, [-1.2%, +3.1%] |
| all 24 pairs, load minus allocation stall | median -0.47%, [-1.8%, +1.3%] |

- Load times: A median 12.62 s (n=15 clean runs at 16 MiB), B median
  12.65 s (n=20), Mann-Whitney p=0.97.
- The pairs' two largest differences, rounds 18 and 23 (-12.0%, -11.1%),
  are sglang starting during A's run.
- The +1.95% of the steady pairs follows B's higher host-allocation time
  in rounds 3, 8 and 14 (+0.45 to +0.54 s). That is the same code in both
  trees. B's unstalled `ahead_init_ms` is slightly higher overall: median
  2477 against 2415 ms, 13 of 19 pairs. Not attributed.
- One counter moves consistently: B stops its pumps on the in-flight
  budget more often, `pump_stops_full` +~1000 of ~103k stops, higher in
  12/15 pairs (p=0.035). `pump_stops_empty` is lower by about the same.
  No load time goes with it.
- Credit wait (median 0.37 against 0.46 ms per read) and `dma_stage`
  (1.06 against 1.30 ms) are higher in B in only 9/15 pairs, uncorrelated
  with the load difference.
- `dma_submit_us_per_piece` takes two modes in both trees, ~6 or ~13 µs.
  B's three 23-29 µs values came in rounds 1, 2 and 5 only.

Code (the review's mechanism lens): here a request is one block
(`request_size` = block size, `maximumCoalescedJobBlocks` = 1). Both trees
therefore run 33 one-block credits, return a block in the same callback as
its credit, and reuse it last-in first-out. Step 2 removes two mutex trips
per request (`acquireMany`, the callback's `pool.release`) and
`Plan.blocks`. That predicts under 0.1%.

Verdict: no change detected at about ±2.5%. A 1-2% difference either way
is not excluded, and this host's resolution is about 3.8% per comparison.
A ±1% answer needs a quiet host, one request size, and a load time that
excludes host allocation (a temporary timer from the first read to the
last DMA callback).

### Qwen3.8-27B through `examples/io`, 5 rounds each

At 16 MiB, load time follows destination creation almost exactly
(r=0.99-1.00, 3.2 to 8.9 ms per read), and step 2 does not touch it. These
runs say little about step 2. Replicated: 2.03-2.05 s for A, 1.99-2.15 s
for B plus one 2.57 s. Sharded: 1.75-1.77 s for A, 1.73-1.87 s for B plus
one 2.38 s. Both outliers are slow destination creations.

### Found on the way

- **The calibrated request size costs 25-30% on Qwen.**
  - The calibration picked 8 MiB instead of 16 MiB in 4 of 10 replicated
    and 2 of 10 sharded Qwen runs, and in 7 of 48 DeepSeek runs. Nothing
    in the step 2 diff touches it: A 9/34, B 4/34, Fisher p=0.22.
  - At 8 MiB, Qwen takes 2.54 s instead of 2.04 s replicated and 2.30 s
    instead of 1.77 s sharded. It has twice the reads, and each read pays
    the workers' per-request cost. DeepSeek is unaffected (12.59 against
    12.62 s).
  - The calibration measures DMA only, so its choice between two rates
    within noise decides a cost it does not see.
- **Request sizes measured (2026-10-21).** Step 2's binary. The block
  sizes are forced with `ZML_DMA_BENCH_BLOCK_MIB`, so request = block.
  Three interleaved repeats:

  | | 8 MiB | 16 MiB | 32 MiB |
  | --- | --- | --- | --- |
  | GB300 x2, Qwen3.8-27B replicated, direct | 3.09 / 2.84 / 2.85 s | 2.51 / 2.59 / 2.07 | 2.47 / 2.76 / 2.77 |
  | the same, buffered, warm | 3.64 (cold) / 3.07 / 2.95 | 2.67 / 2.78 / 2.29 | 2.64 / 2.63 / 2.56 |
  | B70 x4, Llama-3.1-8B tp, direct | 2.66 / 1.72 / 1.70 | 3.26 / 2.42 / 2.34 | - |
  | the same, buffered, warm | 2.17 / 1.35 / 1.03 | 2.96 / 1.65 / 1.97 | - |
  | B70 x4, Llama-3.1-8B replicated, direct | 2.09 / 2.00 / 1.99 | 3.08 / 2.62 / 2.11 | - |
  | the same, buffered, warm | 1.98 / 1.68 / 1.88 | 3.39 / 2.07 / 2.02 | - |

  - GB300 loses 10-25% at 8 MiB, direct or buffered; 16 and 32 MiB tie.
    Buffered 8 MiB reads take the same total worker time as 16 MiB (6.0
    against 6.1 s), so the cost is per request, not in the disk. These
    Qwen loads spend ~33 of their ~40 s of worker time creating
    destinations, at every size.
  - The B70 is the opposite: 8 MiB wins by 10-30%. No fixed size suits
    both hosts.
- **The DMA screen** (`examples/io dma-bench`, repeated):
  - GB300 x2: at the default 8% tolerance it picks 16 MiB 7 times in 8
    and 8 MiB once; at 2% it picks 32 MiB 8 times in 8. So 32 MiB is the
    DMA peak, 16 MiB lies 2-8% below it, and 8 MiB sits on the 8% line,
    which is the coin flip.
  - B70 x4: 2 MiB 6/6 at 8%, 8 MiB 6/6 at 2%. Its reads are 8 MiB
    because request = max(the local profile's 8 MiB chunk, block), not
    because of the block.
  - So only GB300's choice is unstable, at the tolerance edge, and there
    the block also sets the read size.
- `examples/io load ... sharded` fails on DeepSeek-V4.1-Flash with
  `IncompatibleSharding`: some leading dimensions do not split four ways.
- The first DeepSeek load after the host's memory filled took 30.9 s, 20 s
  of it allocating the pinned engram tables. Node 1 had 16 GB free and
  ~260 GB of page cache.
- gb300-1's checkouts are back to their clean state (`utulmkkq`,
  `nomouprz`). The saved binaries stay in `~/phase6/bin`. To use them
  again, apply `zmlA.patch` or `zmlB.patch` plus `setup.sh` first: the
  runfiles come from the tree.

## Open work

- 2026-10-21 benchmark of move 2 step 2 (thirty-third pass): no load-time
  change detected at about ±2.5% on two GB300. The calibration's 8 MiB
  choice costs Qwen3.8-27B 25-30%. 8 MiB sits on the screen's 8%
  tolerance line there, so the choice flips (~1 load in 5). The B70s
  want 8 MiB.
- 2026-10-21 fixes from the loader review (thirty-first pass): headline
  bugs fixed and move 2 done in two commits. The Laguna-under-zio run is
  still to do. The other review items are listed in that pass.
- 2026-10-21 end-to-end loader review (thirtieth pass): see
  `LOADER_REVIEW.md`.
- 2026-10-20 profile of DeepSeek's load (twenty-ninth pass, measurements
  only): a 2.4 s mid-load stall while the plugin pins the 189 GiB engram
  tables; otherwise read-bound at ~50 GiB/s; O_DIRECT re-read 44 GiB of
  cached data; NCCL init holds a CPU for ~35 s. Next steps listed there.
- 2026-10-20 programs over Sources (twenty-eighth pass, zml and llmd):
  `Source.bindProgram`; llmd's programs take placed Sources; expert stacks
  laid out by programs by default, faster than the loader's host copies on
  GB300. Next: compile pack programs alongside the model.
- 2026-10-20 swizzles as Sources (twenty-seventh pass, zml and llmd):
  DeepSeek-V4.1-Flash loads in 15 s on two GB300 (was 66.5 s) and answers
  correctly.
- 2026-10-18 strided parts in the loader and placement without rows
  (twenty-sixth pass, committed with the 24th and 25th as zml omuqxqpu and
  monorepo rwqvkpqy): Llama TP=4 plans in 1-4 ms
  (was 85 ms); row interleaves, broadcasts and small transposes load as
  gather jobs; llmd's `Recipe.fuse` and interleaved `Recipe.stack` are gone.
  Not measured on a model that gathers.
- 2026-10-18 column-sharded loads are staged by destination (twenty-fifth
  pass, uncommitted, zml only): Llama-3.1-8B TP=4 on the B70s loads in
  0.76 s (was 10 s). `examples/io load` gained a `tp` mode.
- 2026-10-18 `TensorStore.Source` phase 1 is implemented (twenty-fourth
  pass), uncommitted in both working copies, on top of duplicates of the
  user's `zml: Add Shape.Partitionning` and `llmd/weights: use explicit
  tags and partitioning`. Next: phase 2 (strided parts), the zio stack
  overflow compiling Laguna, CUDA/ROCm hardware runs.
- 2026-10-14 merge and composite destinations (twentieth and twenty-first
  passes): the merge commit is not described or split yet (resolution,
  review fixes and the composite/engram feature share the working-copy
  commit in both repos). Not run on CUDA hardware: the DeepSeek-V4.1
  engram load through the loader and the `createView` swap on
  loader-produced `pinned_host` buffers. Host-resident destinations
  (twenty-second pass, uncommitted): allocation-bound on oneAPI; see its
  "Left open". Concat phases 1 (expert stacks) to 4 above. oneAPI:
  master's plugin `202610011939` serializes piece completions (3x on
  tiny-piece loads) and segfaults compiling llmd's Llama; master's moe
  Triton test aborts on Level Zero. `Buffer.toSlice`/`toSliceAlloc`
  misread a buffer placed on a sub-mesh (they walk the platform's full
  physical mesh). `--test_filter` is ignored by the rules_zig test runner.
  (The note that llmd's `swapEngramHostPinnedBuffers` errdefer deinits
  owners it leaves in the list was refuted by the thirtieth pass:
  `Buffer.deinit` empties them, so nothing is freed twice.) `direct_loader.zig` still logs
  `TMP loader created_at` at debug level.

Third-pass items left open; `PLAN.md` holds the checklist.

- Nineteenth pass: the eighteenth pass's "30% in the read pipeline" is
  attributed: ~1 s of H2D stalls behind the pack executes (PJRT's
  `kComputeSynchronized` allocation sync points; `kAsynchronous` measured
  5.9 -> 4.9 s in the fork, needs a free-semantics review before adoption),
  0.85 s waiting for the first pack program's compile, and the tiny scale
  pieces draining serially. The 2x replication of packed experts is the
  llmd side's next structural cost. `dma_bench` and `live loader ready`
  are still `debug` on scopes llmd does not enable. A depth derived from
  the measured rate is written up but not applied (worth ~2% today).

- Thirteenth pass (exploration): decided and implemented as the fourteenth
  pass (loader-owned admission, `awaitAll` only). Left for the llmd side, on
  monorepo `master`: migrate `loadPacked`, compile the pack executables in
  `Model.init`, allocate the KV cache after the weights.

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
- Sixth-pass follow-up (a controller that keeps sampling for the whole
  load): moot since the fifteenth pass, the width is fixed per profile.
- The 8% smallest-near-peak block rule is fragile on a busy host (it chose
  2 MiB on MI300 while degraded). Calibration caching per host/plugin, or
  re-screening when the measured rate is implausibly low, remains open.
- One oneAPI plugin abort on the failure path (`Check failed:
  definition_events_[buffer_index]` after a transfer error at width 128
  against the throttled proxy) was seen once and not reproduced.
- Backpressure is process-global and load-untagged (CTX assumption); real
  AWS runs need credentials this machine lacks.
- CPU: the direct path is 14% slower than the buffered one on Llama-3.1-8B
  sharded (parity on Qwen3.5-4B in both modes); the pump is saturated by
  4 KiB first-touch faults of the plugin's device buffers. Not attributed;
  THP `always` is the untested lever.
- Ninth pass: check `TpuClient::CreateBuffersForAsyncHostToDevice` at
  runtime on a TPU host and move TPU to the direct path with pinned arenas
  (`DmaMap` on our pages, or PJRT `pinned_host` buffers as ROCm does; the
  workspace has both branches). Neuron and metal plugins are unchecked. The
  buffered backend stays as it is until then. They removed remaining
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
