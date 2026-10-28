# Direct DMA loader context

Agent-oriented log of the ZML direct DMA weight loader: why the code is the
way it is. "Current design" maps the contracts to the code's doc comments
and `docs/learn/loader.md` and was kept current through the later refactor
passes. Then come durable evidence, the failure and rejected-approach
ledger, the passes in chronological order, and "Open work". Results are
machine-, plugin- and host-state-specific: re-check code, Git refs,
accelerators and plugin artifacts on each machine before relying on one.
`PLAN.md` was the sequential checklist (its task numbers appear below).
2026-10-14: merged with zml/monorepo master, see "Twentieth pass".

## API follow-ups after the tenth pass (2026-09-05/06)

These steps moved loader state into each submission and into loader
initialization; later passes reshaped the API again. What still explains
current choices:

- Host memory has one owner, `io/host_memory.zig`: `Workspace.Backend`
  uses DmaMap-backed pages for CUDA and oneAPI, PJRT-owned `pinned_host`
  buffers for ROCm (which cannot reach DmaMap through a generic allocator;
  see "ROCm pinned allocation"), pageable pages for CPU. Page-backed arenas
  interleave over every memory-bearing NUMA node automatically (kernel
  fallback on one node, unreadable topology or refused placement; no public
  knob) through one private `HugePageAllocator`, not a generic allocator.
  ROCm balances whole arenas by bytes across device-associated host nodes.
  Growth is sequential (`growToBlocks`: one arena) and reserves free-list
  metadata before allocating, so a failure cannot strand memory.
- Every direct loader owns its workspace: it calibrates into it, pregrows,
  then moves it into the `BlockPool` (consumed only on success, so a plain
  `errdefer` cleans up). Workspace borrowing and the public
  `io.dma.benchmark` went. Each loader calibrates and owns its arenas,
  reused across its submissions; CPU returns default sizing, buffered
  backends skip calibration. Calibration thus runs in init after model
  compilation, not in a caller-managed future beside it. (The option and
  reporting names of the time, `.max_host_bytes`, `Loader.calibration()`,
  `initCalibratedBlockPool`, are gone; the 16 GiB ceiling is a constant
  since the sixteenth pass.)
- Progress, the store and shardings became per-submission arguments, so
  init needs no store, and callers own progress totals
  (`store.view().count()` for a checkpoint loaded once). An empty sharding
  slice means replicated. (Handles went in the fourteenth pass and
  `loadExecute` in the thirty-fourth.)
- Backend selection became explicit per target (now in backend.zig; the
  `Loader.backendFor` of the time is gone); CPU uses the direct path with
  ordinary pages, implying no DMA capability.

Verified on macOS arm64: `bazel test //zml:test //examples/io
//examples/llm --test_output=errors` (259 passed, three skipped), covering
backend selection, CPU/default versus buffered/absent calibration and
workspace cleanup on init failure.

## Current design

### Scope and API

Since R16 (forty-eighth pass) the contracts live in the pub doc comments,
and docs/learn/loader.md is the guide. This section points to them:

- API, lifetimes and errors: `Loader` (`init`, `start`, `wait`, `cancel`,
  `deinit`, `Options`, `Error`), `TensorStore` (`View`, `Binding`,
  `Program`), `Source`, `Quantization.Target`, and the registry
  (`TensorRegistry.fromPath`, `addRepo`, `Tensor.read`, `readRange`).
- From bindings to specs, program groups, admission and accounting: the
  loader.zig `//!` header. Where admission cannot measure the room (a
  device that reports no limit, such as CPU, or the buffered backend), one
  program group is in flight at a time.
- Backend selection and the `LoadSpec` contract: backend.zig. The direct
  backend's sizing, planner, credits and pumps: the direct_loader.zig
  header. The buffered backend: the buffered_loader.zig header. The module
  map: `zml/io.zig`.
- Diagnostics: each direct batch logs its planning and completion lines at
  debug when awaited; the backend logs its totals (reads, bytes, credits
  high water, pinned mapped, waits, DMA) at debug from `destroy`; the
  loader logs its admission at debug from `deinit`.

### Source and VFS data plane

- `safetensors.readFilePositionalAllV` is the one exact positional scatter
  for tensor readers and direct-loader pinned reads (short-read resumption,
  local `IOV_MAX` batching); `Tensor.readRange` adds range validation over
  a shared open file. The direct backend opens each distinct safetensor
  object once and keeps it open until the loader's `deinit`.
- HTTP, S3, GCS and HF share one range read loop,
  `range_read.performRangeRead`, over the governed loop of
  `vfs/request.zig`: one whole Range request per admitted positional call,
  retries and holds serial inside that caller's source credit (the retired
  backend-local `parallel_read` pools must not return). Every other request
  a backend makes (HF tree and redirect chain, S3/GCS listings and HEADs,
  GCS token exchanges, the HTTP HEAD chain) uses the same loop, so one
  governor covers a backend's whole quota. A backend contributes a
  `RequestSpec` (name and object for logs, what a 503 means, the rate-limit
  scope); a range read adds a per-attempt `prepare` hook returning URI,
  authorization and extra headers, and the loop appends `Range`. S3
  recomputes the SigV4 timestamp and signature per attempt, GCS copies
  (refreshing when expired) its bearer, HTTP and HF are static. Backends
  keep URL construction, credentials and `backend()` read hints; retry
  settings are one `request.Retry` (each backend's `retry` option); each
  backend counts reads in an `AtomicReadStats` and logs totals at debug
  from `deinit` (forty-seventh pass).
- Registration, `Remotes`, load profiles, the remote error contract and the
  VFS's lifetime: the vfs/vfs.zig header and the doc comments of
  `registerBackend`, `Remotes`, `io` and `loadProfile`.
- Range handling reads `Content-Range` before taking the body reader (which
  releases the head bytes), validates that it covers the request, handles a
  `200` that ignores Range by discarding the prefix, then scatters into
  caller buffers; there is no response timing (`ResponseTiming` and the
  first-body probe were dead). `request.classifyStatus`: 408 timeout, 429
  throttle, other 5xx server failure; 503 is a throttle on S3 (`SlowDown`)
  and GCS but a server failure on HTTP and HF. A throttle arms the
  backend's hold instead of charging the retry budget (sixteenth pass); any
  other retried status naming a delay (`Retry-After` delta-seconds, or HF's
  `RateLimit` `t=`; HTTP-date is not parsed) sleeps that long instead of
  the jittered delay. Backends count requests, retries, timeouts, server
  failures, throttles, bytes, delay, holds and hold waits.
- Rate limiting is a hold, in `vfs/request.zig`. One `Governor` per backend
  instance (keyed by URI authority, so a per-authority scope is a one-
  function change) holds a deadline: a throttle arms it at `max(server
  delay, full-jitter backoff)` clamped to `[initial_delay, max_hold]`,
  extends it when a later answer names something longer, never shortens
  it. Every request of that backend waits it out with its own jitter and
  re-checks. A clean window as long as the last hold ends the episode and
  resets the backoff; an episode without one for longer than
  `throttle_budget` fails requests with `error.RateLimited` until one
  comes. Timeouts never hold. Two budgets: a throttle is not charged to
  `max_retries`, so a limited server cannot exhaust a reader; everything
  else keeps per-request backoff. Defaults (`request.Retry`): five retries,
  500 ms to 30 s backoff, 2 minute hold cap, 5 minute episode budget. A
  deadline, not a permit, composes with the loader's fixed width; a request
  holding a source credit simply waits.
- One source job is one exact absolute scatter read into pinned blocks;
  extra physical calls occur only for short reads/retries or `IOV_MAX`.
  The loader counts read operations, not physical calls: a remote source's
  request count is in its backend's totals; a local file has none.

### Coalesced source planner

The direct_loader.zig `//!` header is the current summary of the direct
backend. The sections below keep the detail and reasons from the passes
that built it; a name here can lag a later rename (the passes record them).

The old planner made jobs tensor-local: `source_request_size` was only a
cap, so small adjacent tensors each cost a source operation.
`DirectLoader.submit` sorts the selected ranges once by file URI, offset
and size (`sortedItemOrder`) and plans one file at a time (`preparePlan`),
publishing each plan as soon as it exists (`publishFiles`), so workers read
the first file while later ones are planned. Each plan:

1. Covers one file group of the sorted order.
2. Forms the union of touching or overlapping requested ranges, never
   reading across an unrequested gap or a file boundary. Duplicate or
   overlapping bindings are read once but keep a transfer piece per output.
3. Partitions each merged run into the minimum number of jobs the
   request/block/`IOV_MAX` limit permits.
4. Keeps that count while preferring tensor-safe boundaries: for
   `N = ceil(run_length / max_job_length)`, each cut takes the latest safe
   tensor boundary that leaves the remainder fitting in `N-1` jobs, else
   the latest feasible hard cut. A boundary is safe only if the next
   touching interval begins at the current union end, which preserves
   overlap and duplicate coverage.

Jobs keep planning order, which is file order (sixteenth pass: a byte-fair
order across destination devices made no difference). No job depends on
another: every DMA piece is submitted as soon as its block is read (fifth
pass), so a tensor's tail may be read before its earlier requests. The
planner also emits the final item, block index/offset, writer mask,
destination offset and length records; the published plan owns its jobs
with those records, and runtime tensor state owns no other dispatch plan
(no order indirection or remaining-work suffix arrays). Physical source
bytes are distinct from logical tensor bytes so duplication and replication
do not distort diagnostics. Planning is `O(tensors log tensors)`: 0.32 s
for DeepSeek-V4-Flash as one plan on MI300 (46 shards); with per-file plans
only the first file's planning delays the first claim (Llama: 4 plans,
1-2 ms in total).

### Pinned blocks, scattering, and ownership

- A worker initializes the destinations its records reference, derives
  block references and destination queue counts, atomically leases all
  pinned blocks for the job, reads into them, and publishes the records to
  existing per-tensor PJRT transfer managers. Compatible adjacent records
  were merged during planning; workers neither traverse sharding spans nor
  rebuild a transfer list. (Per-node NUMA affinity matching was removed in
  the eighth pass.)
- One pinned block may feed several tensor transfers: one atomic lease is
  counted across every consuming PJRT event, and its final completion
  releases the block and completes the parent request, only after all
  child transfers finish or are abandoned. Source/allocation/enqueue/PJRT
  failures close scheduling, retire the batch's unclaimed jobs, fail
  unfinished buffers and release each reference exactly once; the batch
  still reaches `done` through its normal completion units.
- `enqueueBlocks` reserves all affected destination queue capacity first
  (allocation failure is atomic), publishes a whole job under one metadata
  lock, and pumps once; per-piece enqueue had cost ~69k-79k locks and
  pumps.
- Workers retain scratch for leases, reference counts, iovecs and queue
  counts; request, block and event contexts come preallocated with the
  plan; positional-read scratch is stack bounded. A source job makes no
  allocator call in steady state.
- Coalescing preserves per-tensor device buffers, so about one DMA
  submission per tensor is the floor. Going lower needs packed device
  allocations or a device-side scatter/copy stage and changes buffer
  ownership/model layout.

### Scheduling and concurrency

- Reads and DMA complete in any order: PJRT makes a buffer ready once every
  transfer submitted to it completed and one carried the last-transfer
  flag, so the pump flags the submission that completes the target's
  placement bytes (`Target.total` against `submitted_bytes`; the pieces
  partition the placement) and no piece waits for another (fifth pass). A
  batch completing without error must have closed every target, or
  `awaitBatch` fails the loader with `error.IncompleteTransfer` rather than
  leave a buffer that never becomes ready. Per-device ready queues are
  `std.Deque`s served in arrival order, so the oldest submission's pieces
  complete first. No live device queues, debt counters, claimed bitmap,
  runtime order indirection or suffix-metadata arrays.
- The scheduler is a strict FIFO of published plans (a `std.Deque` under
  one mutex and condition; sixteenth pass, it queued batches before).
  `publish(batch, plan)` stamps the plan with its batch, appends it, adds
  its completion units and queues it when it has jobs (capacity reserved
  first, so plan, units and queue entry appear together); `claim` advances
  the front plan's cursor and pops it with its last job; `waitForWork`
  sleeps while nothing is unclaimed; `fail` retires the unclaimed units of
  every queued plan and clears the queue. A submission publishes and seals
  (`sealed_at`) on one task, so its plans are contiguous in file order and
  a later submission's first job is claimed only after every earlier one:
  fairness is intra-plan, and the caller's submission order is the
  cross-submission policy. While a submission is still planning, idle
  workers sleep at most one file's planning time. A plan is queued only
  with jobs left and its units are added in the same critical section, so
  a queued plan's batch still holds a unit, cannot be freed under the
  scheduler, and needs no worker rendezvous. Persistent workers loop
  `waitForWork` -> lifecycle credit -> `claim` -> read -> enqueue.
- A `Batch` owns its items and plans (heap `Plan{batch, jobs, transfers,
  requests, blocks, events, events_used, planning_ns, cursor}`, one per
  file). The planner preallocates every context a plan can need: one
  `RequestContext` per job (initialised idle), one `BlockContext` per job
  block (exact `divCeil(len, block_size)`; `Job.blocks` is the job's slice)
  and one `EventContext` per planned DMA submission
  (`planned_dma_submissions` on the batch line), handed out in submission
  order under `metadata_mutex`. A `Claim` names its plan and job index and
  slices the plan's arrays, so `registerBlock` cannot fail and takes no
  lock; a load allocates only per file.
- The batch completes when `remaining` (a publish sentinel held until the
  seal, plus one unit per job) reaches zero and sets `done`. A job's unit is
  released exactly once: at the request's final reference drop (last DMA
  callback or abandonment; `completeOne` releases the lifecycle credit
  first and calls `finishJobs` last) or by `scheduler.fail` for unclaimed
  jobs. Callback order rule: the PJRT ready callback and the submission
  failure path copy `pipeline`, `device_index` and `block` into locals,
  call `eventCompleted` (which may pump on the callback's thread), push the
  context onto the pipeline's `retired` stack, then call `block.complete()`
  last, because the block's completion may free the batch.
- A planning failure before the first publish destroys the batch and
  returns the error; after it, `submit` fails the pipeline (a partial
  submission can never complete), seals, awaits and retires the batch, so
  the caller sees only the sticky error. `awaitBatch` waits `done` (after
  which nothing touches the batch), marks unsubmitted targets when the
  pipeline failed, retires the contexts under `metadata_mutex` (so neither
  `abortReady` nor a pump draining `retired` can race the free), logs the
  batch and frees it. No per-batch gate drain, barrier, loader-wide
  context list or reap. A pipeline failure is sticky: every open handle's
  await returns it and later submissions are rejected with it. The
  buffered backend mirrors this with `BufferedBatch{pending, done}` over
  its `LimitedGroup` read tasks.
- The source width is fixed for the load (fifteenth pass) and never
  changes during it (sixteenth pass): the profile's default
  (`limits.defaultReadParallelism`: 16 for local files, 32 for a
  high-latency source; sweeps in the fifteenth pass) or
  `Options.read_parallelism`, clipped to one less than the requests the
  pre-grown pinned set holds. The adaptive controller (fourth to sixth
  passes) is gone. Rate limiting is handled one layer down by the VFS
  governor; the loader reads no VFS statistics (forty-seventh pass).
- Two gates, both constant for the load (sixteenth pass). Lifecycle
  capacity is the pre-grown pinned capacity in requests
  (`pool.capacity / blocks_per_request`): `width + 1` source requests plus
  the calibrated per-device DMA depth, so the DMA stage keeps its depth at
  any read width (fourth pass) and no credit can want an unmapped block. A
  request returns its credit only after all its DMA children finish. The
  read gate alone limits source calls, at the width.
- Pinned pre-growth happens at loader creation: the DMA reserve
  (calibrated depth x devices, at least one maximal request) plus
  `width + 1` source requests, as two arenas, reserve first (ROCm balances
  bytes per allocation across host nodes). `create` computes the width
  once from the blocks the mapped ceiling leaves beside the reserve
  (264 MiB for width 16 with 8 MiB requests on CPU, against 528 MiB for the
  former 32-wide set) and returns `OutOfMemory`, logged at err, when it fits
  no request (direct_loader.zig:232-237). Nothing maps inside a load
  (146-230 ms of hipHostMalloc on MI300X when it did) and `pinned_mapped ==
  high_water` on every recorded run. A pregrowth line logs `retained`,
  `pregrown` and `pregrowth_ms`.
- `width + 1` workers are spawned at creation and never retired or parked:
  a worker hands its request to the DMA stage and claims the next, so
  credits beyond the width need no workers (128 persistent workers had
  cost 7% on one MI300X).
- DMA depth is fixed at eight blocks per device by default (see "Adaptive
  DMA width" in the ledger), enforced by the pump as a byte budget
  (`max_in_flight_per_device x block_size`) with at most 64 pieces in
  flight per device, so a block of small tensors keeps the calibrated bytes
  moving (fourth pass). No global DMA cap.
- DMA event lifetime: a ready callback pushes its `EventContext` onto its
  device pump's intrusive `retired` stack under the pump's mutex, after its
  own `eventCompleted` and before `block.complete()`; `pump` destroys the
  stack at the top of every iteration, so an event is destroyed by a later
  pump or by `retireBatch`, never inside its own callback. Live PJRT events
  are bounded by devices x 64 plus one pump batch. Checked on the oneAPI
  plugin under sustained load (PLAN.md task 9).

### DMA memory and calibration

- Host memory (arenas per platform, NUMA interleave, the 16 GiB mapped
  ceiling, `OutOfMemory`): the host_memory.zig header. The direct backend
  owns its workspace and maps it once, before the first read.
- Calibration (the screen, CPU's defaults, `measure` for dma-bench): the
  dma_calibration.zig header. Screen defaults (`dma_calibration.Options`):
  2 ms and 32 transfers per window, 25 ms and 256 transfers per borderline
  confirmation. One `dma_bench` debug line: platform, device kind, block,
  width, GiB/s, duration, retained mapped bytes.
- `Platform.warmupDeviceAllocators(io)` is the only allocator warm-up;
  `init`'s task and `measure` run it first. Calibration measures one
  representative device; its block and depth apply to every device.

## Latest DeepSeek result: why coalescing mattered

Fixture: DeepSeek-V4-Flash, 148.65 GiB, 69,187 selected tensors, 46
contiguous shards, warm one-GPU CUDA loads (an earlier CUDA host
configuration; the model does not fit the 32 GB RTX 5090). The
tensor-local master planner made ~69,445 source operations averaging
~2.19 MiB despite a 16 MiB request size.

| implementation | median load | median epoch | source calls | DMA submissions | logical throughput |
|---|---:|---:|---:|---:|---:|
| master tensor-local | 6.946 s | not recorded | ~69,445 | 69,193 | 21.40 GiB/s |
| first coalesced, fixed-grid cuts | 4.041 s | 3.445 s | 9,524 | 78,665 | 36.78 GiB/s |
| tensor-aware cuts + batch enqueue + scratch | **3.958 s** | **3.376 s** | **9,524** | **69,572** | **37.55 GiB/s** |

Final samples 3.693, 3.958, 4.005 s (widths 24, 32, 24): throughput +75.5%
and source calls -86.3% against master. (The older 16.818 s / 8.84 GiB/s
pre-coalescing number shows the pathology but is no clean master
comparison.) The rigid 16 MiB grid cut through almost every boundary
tensor; tensor-aware cuts keep only 385 pieces above the tensor count
(tensors that must span request limits). Pinned high-water 528 MiB-1.02
GiB, mapped 576 MiB-1.06 GiB: pool/gate bounded, though shared blocks live
longer.

## Durable performance evidence

Machine-, plugin-, topology-, model-, cache- and host-load-specific;
compare medians only on the same setup.

### Source request size is backend-dependent

- One B70 oneAPI, warm 14.96 GiB Llama, width 12: 8/16/32 MiB reads gave
  ~27.05/24.21/21.33 GiB/s with pinned 96/192/384 MiB. At 32 MiB, widths
  12/16/24/32/48/64 gave ~21.33/20.69/18.90/17.33/15.05/13.16 GiB/s:
  twelve was that host's knee, not a law.
- One MI300X, 16 MiB DMA blocks: 16 MiB reads 0.442-0.443 s; 32 MiB reads
  0.608-0.652 s only because three 64 MiB slabs grew mid-load (182-186 ms
  of mapping). On four MI300X, both fitting retained memory: 0.486 versus
  0.488 s.
- S3Proxy at 10 ms with an artificial 1000 MiB/s *per-request* cap favored
  16 MiB at very high width (width 96 ~11.5 GiB/s; 32 MiB width 64 ~9.3).
  It rewards concurrency unrealistically; never base defaults on it.
- Real AWS plateaued near 950 MiB/s (adaptive median ~946-948, zero
  retries/throttles): 16 MiB requests at widths 24-128 within ~0.7% while
  latency and pinned memory rose linearly; 16 MiB/24 was the smallest
  tuple within 3% of peak. 32 MiB cut GETs 39% for no gain; 64 MiB
  regressed.

Request size, DMA block size, source width and live pinned workspace are
distinct controls. Keep VFS-specific minima; do not restore a universal
32 MiB policy.

### DMA block and width evidence

- MI300X one-device loads with 8/16/32 MiB blocks: 23.84/24.90/25.43
  GiB/s. Replicated eight-GPU Gemma (58.25 GiB logical, older plugin): 16
  MiB 7.829 s versus 10.694 s at 8 MiB. Hence the block grid; a two-point
  or generous near-peak rule can pick a costly block.
- B70 favored an 8 MiB request/block neighborhood; the MI300X 16 MiB
  choice cost ~10.5% there.
- DMA width eight repeatedly won or tied on MI300X; wider mostly raised
  callback latency and memory. Exception: on the four-B70 fair synthetic
  benchmark a global cap of four kept ~79.4-79.7 versus 79.9-80.0 GiB/s
  while cutting event latency from ~1.225 to 0.152 ms.
- Calibration screens of 2 ms/32 completions instead of 10 ms/128 cut an
  eight-MI300X calibration from 4.834 to 0.956 s with the same choice.
  Borderline confirmation stays: noisy short screens sometimes picked the
  wrong block.

### Local copy bottleneck and NUMA

- Buffered local loads spend most CPU in Linux `_copy_to_iter`, copying
  page-cache folios into anonymous DmaMapped memory (74.17% on one slow
  MI300X run); DmaMap makes anonymous pages GPU-visible, not cache pages.
- External CPU+memory binding once lifted MI300X from ~16-17 to
  23.6-26.9 GiB/s, but a later placement matrix found no simple local-node
  rule: only CPU node 1 writing node-0 slabs lost a repeatable 8-10%, and
  unbound was fastest; the earlier claim was confounded by host load/cache
  state (see "Seventh pass").
- Raw HIP at 32 MiB: local and cross-NUMA H2D both ~49-50 GiB/s; one
  node-local source fed two GPUs at 48.43 GiB/s unique, 96.86 aggregate. A
  single-threaded copy to the other socket managed 5-10 GiB/s: never
  duplicate replicated data through a CPU cross-node copy.
- Direct I/O avoids `_copy_to_iter` at the cost of warm-cache behavior and
  alignment constraints; see "Eleventh pass".

### ROCm pinned allocation and required XLA behavior

- `hipHostRegister` scales with visible GPU count (KFD/IOMMU maps the range
  to every GPU page by page): eight MI300X took ~1.18 s for 256 MiB, 6.3 s
  for 1 GiB, 11.4 s for 2 GiB. Huge-page advice helped ~9%; pre-touching,
  HIP flags and splitting did not.
- `hipHostMalloc` took ~0.2/0.6-0.7/1.15 s. ZML allocates ROCm arenas as
  standard PJRT `pinned_host` buffers (a representative device's memory
  space per NUMA node), holds an external reference, takes the opaque
  writable pointer and keeps the PJRT buffer as sole owner. Never DmaMap or
  DmaUnmap that pointer: it breaks the later `hipHostFree`.
- XLA main detects GPU pinned ranges (`c3d1d50c0f`, ROCm follow-up
  `e21e1f19a5`; audit main `47149e4cbc`), so no custom
  `PJRT_HostMemoryAllocator_Extension` is needed. An older plugin
  allocates as fast but silently stages every transfer: on one MI300X the
  correct path allocated 256 MiB in 99 ms, calibrated in 265 ms (16 MiB)
  and measured 46.6 GiB/s; a stale plugin measured 6.5 GiB/s. Verify the
  artifact, not allocation time.

### oneAPI pinned-range prerequisite

Older oneAPI PJRT staging did not treat SYCL `prepare_for_device_copy`
imports as pinned (`sycl::get_pointer_type` returned unknown). The
validated XLA fix overrides `SyclExecutor::IsHostMemoryPinned` and checks
both ends of the range with `zexDriverGetHostPointerBaseAddress` against
the same imported base (slab subranges pass, overruns fail). Revisit
`sycl_ext_oneapi_register_host_memory` once the toolchain has it. With
staging, host `memmove` dominated and more DMA credits made it worse;
bypassing it moved one-B70 cap-2/cap-8 from 21.17/11.01 to 26.86/26.90
GiB/s (selected artifact 26.83-26.90).

## Failure and rejected-approach ledger

### Correctness failures that define invariants

- **Multi-device sharded deadlock (2026-09-01):** destination-debt
  scheduling could claim a tensor tail before its predecessor; tails filled
  every lifecycle credit while `transferReady` waited for missing prefixes.
  First fixed by a predecessor-safe plan order (pre-growing memory or
  ignoring the DMA cap did not help); the fifth pass removed the class
  (PJRT needs the flagged call to be last, not the last bytes).
- **Coalescing boundary fan-out (2026-09-03):** rigid grid cuts raised DMA
  from 69,193 to 78,665; tensor-aware feasible cuts restored 69,572 with
  the same 9,524 jobs.
- **Pinned ownership:** replicas and multi-piece blocks need reference
  counts across every child event. A failure path abandons exactly the
  unpublished/unconsumed references and still completes the batch. Never
  release a block merely because its source job returned.
- **Range correctness:** a `206` must cover the requested interval; a `200`
  ignoring Range may be handled by discarding the prefix; malformed partial
  responses must never silently load wrong bytes.

### Performance/control experiments not in the current design

- **Universal fixed 32 MiB reads:** a measured 21% B70 local regression and
  4x pinned high-water versus 8 MiB. Replaced by one VFS profile per load.
- **Joint adaptive request size + source width:** coupled size probes to
  modeled width, handicapped larger sizes, drained between generations,
  and often finished a load before useful candidates. Request size comes
  from the profile; source width stayed adaptive until the fifteenth pass.
- **Source-order mid-read block yielding:** cut leased high-water 12-34%
  for multi-block requests, but speed was neutral, local requests often
  equal one block, and it cannot shrink retained arenas. Remote use needs a
  dynamic destination-buffer contract and suffix-only retries. Removed.
- **Large remote streaming transaction** (a 128 MiB response rotating
  through two DMA blocks): fixed-iovec positional reads lease all blocks up
  front; needs the same dynamic-buffer/resumable-suffix contract.
- **Adaptive DMA width:** moved latency and memory more than throughput.
  Replaced by a calibrated block and fixed per-device width eight; the
  global-cap experiment was removed too.
- **Per-transfer SYCL `host_task`:** serialized dispatch with DMA, B70
  goodput -58-60%.
- **Exact-event callback waits:** neutral or worse; staging explained the
  cap-dependent regression.
- **Independent 2 MiB mapped allocations:** slower than 64 MiB slabs split
  into blocks.
- **File sorting, scalar `pread`:** no material local change.
- **File-backed mmap source, mmap+copy:** ~21.4 and 23.7 GiB/s on B70,
  below `preadv` into DmaMapped blocks.
- **Hard CPU affinity:** twelve B70 readers pinned to CPUs 0-11 collapsed
  to 7.53 GiB/s and harmed startup; use measured per-node lanes, not blind
  affinity.
- **Hard lane coupling:** readers awaiting their DMA children lifted a B70
  diagnostic from ~26.9 to 29.5 GiB/s but would idle high-latency sources;
  if revisited, make completion-aware local pacing a policy.
- **Greedy NUMA assignment:** unsafe with strict-local plus replicated
  block mixtures; augmenting-path matching replaced it until the eighth
  pass removed per-node pools.
- **Huge-page removal, pre-touching, HIP flags:** did not fix ROCm
  registration; the cost is KFD/IOMMU mapping.
- **Custom PJRT host allocator extension:** wrong abstraction and
  unnecessary; standard `pinned_host` buffers solve ROCm ownership and
  locality.

## Historical benchmark anchors

For recognizing regressions, not targets. DeepSeek on CUDA and eight-MI300X
Gemma are in the sections above.

| platform/workload | relevant result |
|---|---|
| one B70, local Llama 14.96 GiB | adaptive and fixed 12/8/2 MiB both ~27.0 GiB/s (before the 8 MiB profile work) |
| four B70, post-deadlock policy | sharded 0.640 s / 23.36 GiB/s; replicated 1.147 s / ~52.2 GiB/s physical |
| four MI300X, local sharded Llama | 0.468-0.500 s, ~30-32 GiB/s by profile/calibration run |

Calibration is excluded from loader epoch throughput but included in
process wall time. Host contention voided several historical runs.

## Final validation state

Historical: end of the third pass (2026-09-04, `67464f3c`).
`bazel test --@zml//platforms:cpu=true --@zml//platforms:cuda=true
//zml:test //vfs:test //stdx:test` passed (zml 244 with 3 unrelated
skips, vfs 20, stdx 18); examples built; loader test blocks 76 -> 93.
Content checks passed on every host, including two-device and four-MI300X sharded packs; B70 packs
at window 1 took 0.739-0.764 s. Pump-thread event retirement was accepted
by all three plugins, every event destroyed, zero errors: oneAPI 16,384
events on two devices at 53.7 GiB/s, CUDA 2,048 at 30.8, ROCm 2,048 (stale
plugin). S3Proxy runs had zero retries; a 503 injection was unit-tested
only; no real AWS. The migrated monorepo (`d426dde4`) served Llama-3.1-8B
on the B70 (load 0.640 s, TTFT 112 ms). The default bazel configuration
was blocked by the missing `platforms/cuda/flashinfer_cutlass_moe` mapping
in `zml/moe/cutlass_flashinfer.zig`.

## Third-pass design (target, 2026-09-03)

User goals: shortest load with the simplest code; the caller controls how
many `loadExecute` submissions are in flight and their order; keep
up-front DMA calibration, the per-VFS profile, decoupled reads and DMA, low
pinned memory.

### Why the epoch model must go

`FairVectoredReadScheduler` held one plan, so `loadExecute` was
`appendItems` + `await`. Laguna paid 78 full drains (39 sparse layers x 2
packs), each waiting for scheduler exhaustion, gate emptiness, a 25 ms
controller barrier, an O(W^2) worker rendezvous and manager teardown, then
ran the pack executable with the pipeline idle; epochs under 100 ms were
never scored, so the width never settled. Only `finishEpoch`'s
free-the-plan rendezvous depended on one live plan.

### Target

- `loadExecute` submits all bindings' sources as ONE planned submission;
  `Handle.await()` waits for its final DMA callbacks, then runs each
  executable on the awaiting task with `.wait = true` and frees its inputs.
  `zml.io.Window{budget_bytes, max_handles}` awaited the oldest handle
  before the next submission (admission moved into the loader in the
  fourteenth pass).
- Direct backend: a strict FIFO of immutable batches (~9.5k claims per
  DeepSeek load under the scheduler mutex), `Batch.remaining` + `done`,
  per-plan preallocated contexts (task 9): now "Scheduling and
  concurrency".
- Rejected: an in-loader executor (`Execute` orders behind definition
  events so it would work, but caller-task execution keeps the proven PJRT
  lifetime order and needs no thread); scoring the controller only at read
  completion (loses backpressure sampling while workers sleep in retries);
  dropping `high_latency`; deleting NUMA before measuring.

### PJRT lifetime facts (header 0.113 and pjrt_client.h)

- Buffers from an async host-to-device manager may go to `Execute`
  immediately; execution orders behind the definition event.
- `PJRT_Buffer_Destroy` while an execution references the buffer is safe;
  device memory is freed when async work completes. zml passes an empty
  `non_donatable_input_indices`, so inputs may be donated: never reuse
  them.
- Destroying an event from another thread after its callback returned is
  current practice (`retireBatch`, `Buffer.await`, the pump since task 9);
  never inside the callback. Checked on oneAPI under sustained load
  (`ZML_LOAD_EVENT_RETIRE_CHECK`: 16k events per run, 0 errors).
  Destroying a manager before its transfers complete is undocumented:
  destroy only after completion (per-batch retirement guarantees it),
  events before their manager.
- No blanket thread-safety statement; concurrent `Execute` on one
  executable is undocumented, so executions stay on one awaiting task.
- `TransferRawDataToSubBuffer`'s `is_last_transfer` only closes the buffer
  to further calls: it becomes ready when its in-flight count reaches zero
  and the flag was seen, in any completion order
  (`xla/pjrt/host_to_device_transfer_manager.cc`, openxla/xla `8dfe2c4ff1`
  2025-04-28, used by the GPU stream-executor client since `1b19ae012a`
  2025-10-22; the older `GpuAsyncHostToDeviceTransferManager` sequenced the
  definition event behind the flagged transfer on its one H2D stream, which
  also only needs the flagged call to be last). zml's XLA pin `41370d1124`
  (2026-07-02) has both; the shipped libpjrt_cuda (manual-2026-07-31),
  libpjrt_rocm (manual-2026-07-20) and libpjrt_oneapi (manual-2026-08-17)
  carry `CommonAsyncHostToDeviceTransferManager` and none of the old
  manager (`strings`, 2026-09-04).
- The transfer's done event never carries an error: `on_done` has no
  status and the C API wrapper resolves it OK
  (`pjrt_c_api_wrapper_impl.cc`, `on_done_with_d2h_transfer`), so
  `PJRT_Event_OnReady` always passes a null error. An asynchronous copy
  failure errors the buffer's definition event and surfaces at first use;
  the ready callback's `recordError` branch is dead on shipped plugins.

### Baseline 2026-09-03 (commit 2f9cac2b, Llama-3.1-8B 14.96 GiB, one GPU, warm)

| host | load | GiB/s | DMA block | request | width | pinned high-water | calibration |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9985wx-5090x4 (CUDA, quiet host) | 0.606 / 0.609 s | 24.6 | 2 MiB | 8 MiB | 24 | 264 MiB | 751 ms |
| local B70 (oneAPI) | 0.794 / 0.758 s | 18.8-19.7 | 8 MiB | 8 MiB | 12 | 200 MiB | 809-891 ms |
| mi300 (ROCm) | 1.152 / 1.102 s | 13.0-13.6 | 16 MiB | 16 MiB | 24 | 784 MiB | 274-1109 ms |

- DMA calibration peaks were healthy (CUDA 50.7 GiB/s at 2 MiB, oneAPI
  48.8 at 8, ROCm 43.3 at 16), so ROCm's 13 GiB/s was source-side. ROCm
  allocator warm-up took 2.36 s.
- Page-cache read ceiling (`dd`, 512 MiB chunks, P = 1/2/4/8/16/32
  readers): CUDA host 14.1/24.7/31.6/55.7/62.1/60.7 GiB/s; B70
  19.5/27.2/27.4/26.6/30.8/29.8; mi300 4.3/9.5/16.8/26.0/44.0/54.6. The
  loader reached 41% of the CUDA ceiling, 66% of the B70's.
- The CUDA host is shared: under another user's job the same run took
  0.7-1.3 s. Check host state before every measurement.
- NUMA: mi300 (GPUs 4/4) and gb300-2 (GPUs 2/2) are two-node; the CUDA and
  B70 hosts single-node. See "Seventh pass".
- Laguna-XS-2.1 (63 GB, 14 shards): 39 sparse layers x 256 experts; per
  expert down [2048,512], gate and up [512,2048], 2 MiB each, adjacent in
  file order, so a layer's 768 expert tensors are one 1.5 GiB run. A
  per-layer submission holding both packs coalesces into ~16 MiB jobs;
  separate down and gate_up submissions would read every third tensor.
- DeepSeek-V4-Flash does NOT fit one 32 GB RTX 5090 (`ResourceExhausted`
  after ~28 GiB, failed cleanly), so it was measured on MI300 (192 GB)
  from then on: day one 8.20-8.26 s wall, epoch 7.44-7.63 s, width 24,
  pinned 784 MiB.
- HF (`hf://Qwen/Qwen3.5-9B`, 17.98 GiB, B70 host, no token): 21.0-22.4 s,
  824-876 MiB/s, 32 MiB requests, 577 requests, zero retries, width 24/32,
  pinned 1.53 GiB (width x 32 MiB: remote loads pin far more). Fixed on the
  way: `std.http.Client` returns HEAD responses before its redirect
  handling, so `HF.resolveDownloadUrl` failed on the Hub's 307; it now
  follows redirects itself (absolute or relative, credentials only to
  huggingface.co).
- Pack instrument (playground, `ZML_LOAD_PACKS=64 ZML_LOAD_PACK_WIDTH=16`,
  synchronous `loadExecute`): 14 packs of 16 sources (13.0 GiB) at
  13.5-14.6 GiB/s, each pack epoch under 110 ms and never scored; total
  1.005-1.086 s against 0.77-0.79 s for the same bytes in one bulk load.
- Fixtures: an S3Proxy jar and an `lfm` bucket linking the Llama shards on
  the B70 machine; no AWS credentials there.

Trajectory through the pass (quiet hosts, Llama, one GPU unless noted):

- Task 3 (`f55b8ea7`, handles, `Window`): packs at window 2 equalled the
  plain load on CUDA (0.593-0.612 against 0.586-0.604 s) and beat it by 25%
  on MI300 (0.677-0.689 against 0.901-0.965). Four-GPU sharded packs at
  window 2 on MI300 completed in 1.04-1.09 s with correct contents (the
  2026-09-01 deadlock family passes).
- MI300 width (task 3 tree): fixed 12 0.418-0.442 s (33.8-35.8 GiB/s);
  fixed 24 0.616-0.671 s, the first 64 MiB slab growth costing 146 ms;
  adaptive 0.901-0.965 s, selecting 24. The adaptive ramp (gate drains,
  probes, slab growth) cost more than the load and settled worse: most of
  the "ROCm problem".
- Task 8 (`4a046807`: climb-and-hold controller without gate drains,
  on-demand workers, working set mapped in calibration): CUDA adaptive
  0.502-0.548 s against fixed 12 0.488-0.500 and fixed 16 0.456-0.477;
  MI300 adaptive 0.486-0.508 against fixed 12 0.409-0.416.
- Worker count (MI300, task 8 tree): 16 workers 0.424 s epoch, 24 0.476,
  32 0.572, 128 0.613-0.675; with 128 even width 16 ran 21 GiB/s against
  36 with 16. Hence workers follow the width.
- Task 6 (per-file publish): neutral on B70 Llama by construction (4 plans,
  1-2 ms planning, jobs and transfers unchanged).
- Laguna through the migrated llmd on MI300 (40 submissions, 7,984 reads,
  32,431 DMA pieces, 62.29 GiB in 12.2 s at window 1 or 2, TTFT 355-368 ms,
  64-72 tokens/s): VOID, calibration chose 2 MiB blocks and the load was
  DMA-bound at 5.6 GiB/s.
- The "MI300 host degradation" from 2026-09-04 ~03:00 (calibration
  6.9-7.9 GiB/s at every block, Llama 3.05 s) was a stale ROCm plugin
  staging every transfer (seventh pass); MI300 numbers marked degraded in
  this and the next two passes are void.

### Third-pass results (2026-09-04, commit 67464f3c, warm, one GPU)

| workload | day one (2f9cac2b) | final | notes |
|---|---:|---:|---|
| local B70, Llama plain | 0.758-0.794 s | 0.636-0.655 s | held width 12, 13 workers, pinned high-water 136 MiB |
| local B70, 14 packs window 2 | 1.005-1.086 s (synchronous) | 0.704-0.712 s | pack phase 0.62 s at 21 GiB/s |
| CUDA 5090, Llama plain | 0.606-0.609 s | 0.513-0.524 s | 28.5-29.1 GiB/s |
| CUDA 5090, 14 packs window 2 | 0.750-0.757 s | 0.457-0.512 s | pack phase 0.40-0.44 s at 29-32 GiB/s |
| MI300X, Llama plain | 1.102-1.152 s | 0.486-0.508 s | fixed 12 is 0.41 s |
| MI300X, 14 packs window 2 | 0.957-1.024 s | 0.537-0.594 s | |
| MI300X, DeepSeek-V4-Flash | 8.20-8.26 s | 7.09-7.22 s | 9,524 jobs, 69,572 transfers |
| HF Qwen3.5-9B (local host) | 21.0-22.3 s | 19.6-19.7 s | 934-938 MiB/s, per-connection cap near 19 MiB/s |

Every final run: gate never closed, 528 MiB pinned working set mapped in
calibration, no steady-state allocation. Production lines grew
(`direct_loader.zig` 3,152 -> 3,542, `io.zig` 791 -> 1,135) while VFS
backends shrank 4,612 -> 3,370 and calibration 1,816 -> 1,667; what went
was machinery: epoch flags, the single plan slot and worker rendezvous,
the controller epoch barrier and three-phase controller, per-epoch
reclamation, four copies of the range/retry loop, 128 persistent workers.

## Fourth pass: DMA-stage decoupling (2026-09-04, gb300-2)

Trigger: on `gb300-2` (144-core Grace, 34 NUMA nodes, four GB300, DMA
calibrating ~180 GiB/s at 16 MiB blocks) a DeepSeek-V4-Flash load held a
width-16 window open 3.8 s, then held 8. Reproduce with
`CUDA_VISIBLE_DEVICES=0 bazel run --config=release
--@zml//platforms:cuda=true //examples/io:playground -- load
/var/models/deepseek-ai/DeepSeek-V4-Flash/`.

Evidence (warm, HEAD `8a5de654`):

- Per 16 MiB request a worker waited 9.2 ms for a lifecycle credit (held
  until the request's last DMA callback) and read for 3.0 ms; blocks, read
  gate and enqueue took under 10 us. With `width + 1` credits the DMA stage
  held one request at a time.
- Half of DeepSeek's 69,187 tensors are 256 KiB scales, half ~4 MiB
  weights (2.2 MiB mean piece), so a depth of 8 submissions per device kept
  ~18 MiB in flight where calibration had measured 128 MiB.
- The pump is not the ceiling (69,572 submissions at 5.8 us, 0.4 s of a
  3.7 s load), nor worker start (claims within 1 ms).
- At fixed width 16: DMA depth 8 -> 24.3 GiB/s (6.13 s), 32 -> 36.1
  (4.12 s), 64 with 16 extra credits -> 43.8 (3.39 s).
- Rejected, measured: credits equal to the whole retained capacity with
  workers spawned to the credits and the read permit taken before the
  pinned blocks (3.53-3.66 s, but 41-74 idle tasks per load and block
  waits counted as reads in flight).

Change: lifecycle credits `min(feasible, max(retained, width +
dma_stage))`, `retained` being the pre-grown capacity (33-request source
set plus calibrated DMA depth per device, materialized at calibration);
workers stay at `width + 1`. The per-device DMA budget became bytes
(`max_in_flight_per_device x block_size`) with a 64-submission cap per
device. Summary timers `credit_wait_ms_per_read`, `block_wait_ms_per_read`
and `read_ms_per_read`: waits above the read time mean a DMA-completion
bound load. The controller's first window, opened on an empty DMA stage,
measured the fill burst (47-50 GiB/s against a steady 42) and was
discarded from then on (rule gone with the controller).

| host, workload | HEAD `8a5de654` | this pass |
|---|---:|---:|
| gb300-2 GPU 0, DeepSeek 148.65 GiB (3 runs each) | 5.83-5.96 s (25 GiB/s) | 3.30-3.57 s (42-45 GiB/s) |
| gb300-2 GPU 0, fixed width 12 / 24 / 32 | - | 3.75 / 3.20-3.81 / 3.65 s |
| local B70, Llama plain (2 each) | 678, 680 ms | 677, 676 ms |
| local B70, HF Qwen3.5-9B (1 each) | 20.8 s | 21.9 s |

On gb300-2 the load is DMA-completion bound at ~42 GiB/s at every read
width; the ceiling is the PJRT/CUDA per-piece copy cost for 4 MiB and
256 KiB pieces. Pinned memory rose by the DMA reserve: 528 -> 656 MiB on
one GB300, 592 MiB on the B70, 544 MiB with 2 MiB blocks; an 8-GPU host
with 16 MiB blocks pre-grows 1 GiB more.

Known limits: the 64-piece cap bounds event overhead and is no measured
optimum; for blocks of tensors under `block / 8` it holds fewer bytes than
the budget (DeepSeek's mix matches). On an 8-GPU host with 16 MiB blocks
the set plus reserves (33 + 64 blocks) filled 1552 MiB of the then 2 GiB
mapped ceiling (16 GiB since the seventh pass). Refuted: `waitForWork`
spinning on an open exhausted head (submissions seal before returning).

## Fifth pass: last transfer by bytes (2026-09-04)

Trigger: the B70 HF summary showed `credit_wait_ms_per_read` 194-408
against `read_ms_per_read` 1200-1400 with DMA at 48 GiB/s. First suspect,
the tail rule: the loader flagged the piece with the highest destination
offset as PJRT's last transfer and held it until every other byte of the
tensor was submitted (`transferReady`), so a request carrying a tail kept
its credit and a pinned block until the tensor's earlier requests were
read. XLA (PJRT facts in "Third-pass design") showed the contract is "last
call", not "last bytes".

Removing it did not remove the waits: new timers showed an HF request
spends under 1 ms in the DMA stage while the read gate was full all load.
The waits were surplus workers from a wider controller rung queueing at
the credit gate after a downward step: idle workers, not lost throughput,
but a lying diagnostic.

Change (commits `d082a614` to `16098680`):

- The pump, the only submitter, flags the submission that completes the
  target's placement bytes (`Target.total` against `submitted_bytes`); a
  first version counting planned pieces per (item, writer) was replaced by
  the byte total the target already carries (a shape-derived oracle, not
  the planner's own tally). `transferReady`, the prefix wait and the
  planner's predecessor order (`PlanningJob.predecessor`, `fairOrder`) are
  gone.
- `awaitBatch` fails with `error.IncompleteTransfer` when a batch completed
  without error but a target missed its last transfer.
- Failure path: PJRT's shared transfer manager drops a buffer's definition
  event once its last transfer was issued (accepted or not) or one of its
  transfers failed, and a later `SetBufferError` trips a CHECK and aborts
  the process. `Target.closed` (set by the pump before the flagged call)
  keeps `awaitBatch` from marking such buffers; a failed submission's
  outputs are undefined either way. An asynchronous transfer failure is
  invisible to the loader (the done event carries no error), so a buffer
  failed that way and then marked after an unrelated failure still aborts;
  a per-target flag set from the ready callback could never fire.
  `DirectLoader.submit` rejects empty sources.
- An intermediate warm-up rule (credit wait against read time) could not
  fire at a narrow rung on GB300 (13 workers wait 2 ms per 3 ms read while
  each request sits 7 ms in the stage); two of five DeepSeek loads then
  scored an inflated window and held 12.
- The summary adds `dma_stage_ms_per_read` (enqueue to last DMA callback)
  and `tensor_init_ms_per_read` (PJRT buffer and manager creation).
- Playground `ZML_LOAD_CHECK=n` reads every n-th loaded tensor and the
  largest eligible one back and compares bytes with the source (`load
  check: ok tensors_checked=...`), every replica of replicated buffers;
  partitioned sub-byte tensors are skipped (defect below).

Results (baseline `c9fe01d4`, final `16098680`, same day and host):
gb300-2 DeepSeek bulk phase 2.96 s (climbed to 32) and 3.40 s (held 8)
before, 3.17-3.33 s after (four runs, all held 12; one `b6f23018` run
reached 32 in 2.56 s); B70 Llama sharded 658-682 ms before, 668-676 after;
B70 HF Qwen3.5-9B 22.0-22.7 s before, 22.5-23.9 after; CUDA 5090 Llama
457-485 before, 443-463 on the intermediate `a9802595`. Read-back ok
everywhere: DeepSeek 1082 sampled tensors, B70 and CUDA Llama full
(291/291), MI300 (stale plugin, timing void) two-GPU Llama and DeepSeek.
Final per-request timers: gb300-2 DeepSeek credit wait 0.8-0.9 ms, read
3.1-3.3, DMA stage 6.3-6.6, tensor init 0.06; B70 HF credit wait 4.5 ms
(was 120-410), read 1200-1260, DMA stage 1.0; B70 Llama credit wait 0,
read 4.2, DMA stage 0.4.

Known limits:

- gb300-2 loads whose controller reached 24-32 took 2.56-2.96 s, those
  holding 8-12 3.2-3.5 s, decided by one window at 16 beating one at 12 by
  3% while they differ by less than the noise (12: 42-45 GiB/s; 16:
  40-48). Taken up by the sixth pass.
- `Buffer.toSliceAlloc` places sub-byte shards by element stride (1 byte
  per element for `u4`, `f4e2m1`), so a 4-bit tensor partitioned over two
  or more devices is assembled at twice its offset (out of bounds in safe
  builds). Pre-existing; the loader plans on the packed shape. Still open
  (see "Open work").
- The pump can still submit into a manager whose event just errored
  (`host_to_device_transfer_manager.cc:342` CHECK, the oneAPI abort in
  "Open work"): the manager nulls the definition event under its own mutex
  before our callback runs, so no loader-side flag can close that window.
  Still open.
- `active_events` and `ready_entries` are derivable from the per-device
  arrays and deque lengths; kept as assert witnesses. `checkPacks` keeps
  its own read-back loop. Credit waiting is undercounted by workers that
  lose the claim race after `waitForWork`.

## Sixth pass: width detection on a flat plateau (2026-09-04)

The fifth pass read the gb300-2 spread as "the climb is the biggest lever
left". A fixed-width sweep measured the curve instead (DeepSeek-V4-Flash,
gb300-2 GPU 0, `ZML_LOAD_FIXED_READ_PARALLELISM`, three interleaved rounds
per rung):

| fixed width | GiB/s per run | mean | load |
|---|---|---:|---:|
| 8 | 35.3, 39.1, 36.4 | 36.9 | 4.03 s |
| 12 | 45.2, 43.8, 45.3 | 44.8 | 3.32 s |
| 16 | 50.3, 42.1, 47.9 | 46.7 | 3.20 s |
| 24 | 44.8, 49.2, 50.5 | 48.2 | 3.09 s |
| 32 | 49.1, 50.5, 46.8 | 48.8 | 3.05 s |
| 48 | 40.8, 50.1, 51.0 | 47.3 | 3.18 s |

12 to 48 is one plateau 8% wide; only 8 is a cliff, 20% below. The whole
prize a width controller can win here is 0.27 s of a 3.3 s load; the one
bad rung costs 0.7 s.

Why a controller cannot win the 8%: measuring every window of a fixed-width
load showed 13-15% within-load spread that drifts with the file (width 12:
27.6 to 54.7 GiB/s, rising through the load; width 24: either 44-47 then
55-66, or 55-60 then 43-50). DeepSeek interleaves 256 KiB scales with
4 MiB weights, so consecutive 5 GiB stretches carry different piece counts
and sustain different rates; one 120 ms window at rung A and the next at
rung B measure different parts of the file. Six baseline loads' first
window at 12 read 37.8-53.1 GiB/s against a sustained 44.8. The load is 26
windows long and the controller decided in the first four.

Change: the controller (deleted in the fifteenth pass) was made to bound
the downside: tolerate one failed rung, adopt a downward probe only when it
beat the best rate (a probe at 8 had read 41.07 GiB/s right after a window
at 16, against 36.9 sustained, and held a whole load at 8, 3.70 s), and
stop at the growth-free width (without it, runs ended at 48, 64 and 96,
mapping slabs mid-load). Seven interleaved pairs against `fb8ccf55`: mean
3.622 -> 3.221 s (-11%, paired t = -3.2), six of seven better, worst 4.15
-> 3.41 s; the baseline settled at 8 in three of seven, this tree never. B70
Llama (666-681 ms, read-back ok) and B70 HF (20.6-22.1 s, held 32) were
unchanged.

Rejected, each measured:

- A wider start rung: adaptive starts at 12, 24, 32 averaged 3.37, 3.43,
  3.23 s and only climbed higher (ending at 48, 64, 96).
- Climbing while credits are not the limiter (`credit_wait_ms_per_read`
  below `read_ms_per_read`): picks 32 on gb300-2, the optimum, but the B70
  shows zero credit wait through width 24, so it climbs past the B70's
  peak (23.0 GiB/s at 12-16) into 21.9 at 32 and above.
- Longer windows: a 5% standard error needs about a second of busy time, a
  third of a DeepSeek load.

The adaptive mean stayed ~5% above the fixed-32 oracle (3.22 against
3.05 s). The B70 sweep at 8 MiB requests (19.5, 18.4, 23.0, 23.0, 22.5,
21.9, 21.9 GiB/s at widths 4 to 48) reproduces to 0.1% between runs while
gb300-2 spreads 10-15% at every rung: how much a rate comparison can be
trusted is a property of the host. Both points fed the fifteenth pass's
fixed width.

## Seventh pass: does NUMA placement matter? (2026-09-05)

Task 12 had never run on a healthy multi-node host. Both bench hosts are
two-node for GPUs: `mi300` has 8 MI300X, four per node (node 0
`1b,3d,4e,5f`, node 1 `9d,bd,cd,dd`), 2 x 1 TiB; `gb300-2` has 4 GB300, two
per node (node 0 `0008,0009`, node 1 `0018,0019`), 2 x 490 GB, plus 32 NUMA
nodes of device-coherent HBM with no CPU or host memory, so only 0 and 1 can
back an arena. Arms: `local` (the default then: one `mbind`-ed pool per
device node, strict affinity), `off` (`ZML_DMA_BENCH_NUMA_OFF=1`, one
unbound pool), `node0`/`node1` (`ZML_DMA_BENCH_NUMA_NODES`, every device on
one node). Pools and knobs went in the eighth pass.

### The MI300 host was not degraded, its plugin was stale

The released ROCm artifact (`manual-2026-07-20T15-30-00Z`) contains no
`IsHostMemoryPinned`: no XLA pinned-range detection, so every DmaMapped
transfer was staged. That was the whole 2026-09-04 "MI300 degradation" (4.5
to 7.6 GiB/s at any block size, device count or tree). A locally built
`libpjrt_c_api_gpu_plugin.so` (openxla/xla 2026-09-03) behind
`platforms/rocm/rocm.bzl` restored 44-47 GiB/s. Check the symbol before
blaming the host.

### Placement is worth 1.6x on GB300 and nothing on MI300X

`dma-bench`, 16 MiB blocks, one visible device, arena forced onto each node
in turn (a clean probe of one link): every GB300 reads 175.8-183.8 GiB/s
from its own node and 109.5-111.2 from the other; MI300X 0, 3, 4 and 7 read
43.6-47.2 from either. Remote costs 38% on GB300, symmetrically, and nothing
on MI300X. An interconnect fact: each GB300's NVLink-C2C lands on its own
Grace socket, so a remote arena crosses the inter-socket link, while an
MI300X's PCIe root complex (~50 GiB/s) is below the cross-socket cost (as
the older raw-HIP 49-50 GiB/s local and cross-NUMA showed). Device count
does not change it: 1, 2 (same node), 2 (split), 4 GB300 measure
176.2-179.3 local, 109.7-111.2 unbound.

### DeepSeek does not benefit: the loader is read-bound

DeepSeek-V4-Flash replicated on four GB300 (148.65 GiB per device), 16 MiB
blocks, mean of five interleaved reps, fixed width 32 / adaptive: `local`
5.48 / 5.41 s, `off` 5.40 / 5.38, `node0` 5.89 / 5.70, `node1` 5.58 / 5.24.
`local` equals `off` within noise (`off` alone spread 5.03-5.96). 27.5 GiB/s
per link is a quarter of even the remote 110: the page-cache read path is
the constraint, not the link. Only `node0` (all links and reader copies on
one memory controller) is repeatably ~5% worse. The 1.6x is headroom that
matters only for a source faster than 110 GiB/s per device.

### Llama: the page-cache node is the contended resource

Llama-3.1-8B replicated on four GB300 (46 GiB/s per link against DeepSeek's
26.5), idle host, seven interleaved reps, spread under 2%: `off` 307.1 ms,
`node1` 306.4, `local` 321.7 (+4.8%), `node0` 413.1 (+34%). `node0` and
`node1` both put all four links on one node, so GPU locality alone would
make them equal. The file's page cache sits on node 0: pinned blocks there
make one memory controller serve the page-cache reads and all four DMA
engines, and `local` puts half the blocks there. The constraint is host
memory bandwidth on the page-cache node; the right rule is "away from the
page cache", the opposite of strict affinity, and no static `numa_node`
attribute can express it.

### DeepSeek replicated is bound by submission count

Two runs each on 1, 2, 4 devices: 47.6-48.2, 41.5-42.6, 25.6-27.5 GiB/s
per device; DMA stage 6.2-9.9, 8.4-15.6, 27.9-32.3 ms/read; reads flat
(9,524 of them, 2.1-5.7 ms/read): not the read path. At four devices a link
carries 16% of its 176 GiB/s. The cost is 278,288 H2D submissions (69,572
pieces x 4; reads coalesce 7.26:1, submissions not at all), ~51,500/s of
2.24 MiB average that take ~13 us at link speed: per-call overhead on par
with transfer.

`max_in_flight_per_device` 32 against 8 (six interleaved pairs): **5.024**
against 5.536 s (-9.2%, 6/6) for a pinned high-water of 2.36-2.50 against
0.9-1.0 GiB; depth 16 gave ~80% of it for 2x pinned; 32 MiB blocks gain
nothing at any depth. The depth buys queue length, not parallelism
(`lifecycle_credits` 49 -> 140, DMA stage 17-26 -> 56-76 ms/read, reads
flat: in-flight x2.9, latency x2.4, throughput x1.17) ahead of a serial
submission path (~50-60k/s); depth 8 already hit the 64-piece cap (128 MiB
/ 2.24 MiB). In-arm spread (4.62-5.38 s at depth 32) exceeds one rung, so
an online controller would chase noise, and growing the stage mid-load maps
pinned slabs inside a scored window: if the depth is ever derived, derive it
once, on the client, from the safetensors header.

The count cannot be coalesced: a submission is `transferData` on one
tensor's transfer manager, so tensors x devices is the floor under one
buffer per tensor (69,187 tensors; 69,572 pieces with block straddling).
Sizes are bimodal: 34,759 tensors (50.2%) under 1 MiB carry 5.6% of the
bytes (median 0.25 MiB), so half the submissions move almost nothing.
Packing small tensors into one device buffer is a `Buffer` model change,
not a loader one.

### The submission ceiling was the loader's pump, not the driver

`dma-bench`, one device, 200 ms windows, three reps; depth 8 and 32 are
identical, so this is a rate, not a latency:

| block | GiB/s | submissions/s | us per submission |
| --- | --- | --- | --- |
| 1 MiB | 58-65 | 59,000-66,000 | ~16 |
| 2 MiB | 104-107 | 53,000-55,000 | ~19 |
| 4 MiB | 142-153 | 36,000-39,000 | ~26 |
| 16 MiB | 183-184 | 11,700 | ~85 (link-bound) |

Calibration measures one device at a time (`tuneDevice` makes one cohort
per device, `runBenchmarkWindow` runs one), so `measured_gib_s` is never an
aggregate. Four single-device processes reach 161,000 submissions/s at
1 MiB and 640 GiB/s at 16 MiB: neither driver nor engines are shared.

`dma-conc` (`//examples/io`; `ZML_DMA_CONC_*`: `SERIAL_SUBMIT`, `CALLBACK`,
`BUFFERS=reuse|fresh|prebuilt`, `SHARED_SOURCE`, `SMALL_KIB`, `SOURCE_MIB`,
`MISALIGN`, `WRITERS`) drives every device from one process with `depth`
synchronous slots per device, optionally serialising submits behind one
mutex as the loader's single pump did. gb300-2, 500 ms windows, three reps:

| devices | block | submit | GiB/s | submissions/s |
| --- | --- | --- | --- | --- |
| 1 | 1 MiB | parallel | 46-55 | 47,000-57,000 |
| 4 | 1 MiB | parallel | 170-180 | **174,000-184,000** |
| 4 | 1 MiB | serialised | 88-125 | 90,000-128,000 |
| 4 | 1 MiB | parallel, `numactl -m 0` | 199 | **203,000** |
| 4 | 16 MiB | parallel | 316-385 | 20,000-25,000 |

The loader submitted 51,000-55,000/s across four devices: below one
device's engine, 3.5x below per-device submitters, half a mutex-serialised
one. The ceiling was `VectoredLoadPipeline.pump` (one thread, one
`metadata_mutex`, round-robin over devices, half its per-submission cost
outside the PJRT call); the depth was a 9% patch on it. Side results:
unplaced `dma-conc` memory landed on node 1 (`membind=1` 112 GiB/s on
device 0, `membind=0` 185: the same 1.6x), and one process pushes 385 GiB/s
into four GB300, so the host does not limit a file source.

Measurement note: `load` passed only `block_sizes` to
`benchmarkIfSupported`, so `ZML_DMA_BENCH_BLOCK_PARALLELISM` never reached
it and a first depth sweep measured nothing (`dma_budget_per_device` stayed
128 MiB). Check that line before trusting a depth result.

### Per-device pumps: -13.5%, and a correctness requirement

One `DevicePump` per device (own mutex, ready queue, in-flight bytes and
pieces, retired-event stack) replaced the single `metadata_mutex`,
`pumping` flag and round-robin; `enqueueBlocks` and `retireBatch` take every
pump's mutex in device order (a request still queues all-or-nothing).
DeepSeek replicated, four GB300, depth 8, five interleaved pairs: 5.41 s
(5.21-5.52) -> **4.68 s** (4.52-5.05; -13.5%, 5/5); DMA stage 14-30 ->
9-10.5 ms/read, credit wait 1.4-12.3 -> 1.2-1.4; pinned high-water
unchanged or lower (896 MiB). Llama on the same host at parity (320-324
against 321-325 ms, `ZML_LOAD_CHECK=64` clean). Eight MI300X at host load
30: both arms bimodal 0.88-1.49 s (means 1.19 against 1.28), unresolved;
worth one idle-host repeat.

Rejected (in the `DevicePump` doc comment): a task per device woken by
completions (5.0-5.4 s, a futex round trip per piece; 4.4-5.1 s waking at
half the budget); concurrent submitters per device, 4.56-4.58 s and
**incorrect**: `Target.nextIsLast` flags a tensor's last piece per device in
submission order, and two threads submitting one tensor left targets
unclosed (`IncompleteTransfer`, 0 bytes loaded, two of four Llama runs;
DeepSeek's single-piece tensors never hit it).

### Where the rest of the gap to the read bound is

Against the 3.1 s one-device read bound, from new summary fields
(`dma_submit_us_per_piece`, `dma_piece_latency_ms`,
`pump_stops_empty/full`) and `dma-conc` reproductions of one condition at a
time:

- Not CPU-bound: ~1.3 cores, all four readers' page-cache copies.
  `submitTransfer` *blocks* 16-28 us in the driver with four devices (5-12
  on one); `dma-conc` sees the same 28 us at ~32 pieces in flight per
  device (5 us at depth 8).
- Each device runs ~15.5k pieces/s, 33 GiB/s, 2.0-2.3 ms submit-to-callback
  at ~31 in flight; pumps stop mostly for lack of room. `dma-conc` with the
  loader's traffic (half 256 KiB, half 4 MiB, a fresh buffer per transfer
  flagged last) does ~32k pieces/s, 65-70 GiB/s per device: the loader ran
  at half the engine's rate for its own traffic.
- In `dma-conc` (4 devices, depth 32), no effect from source footprint
  (32 MiB-1.5 GiB), a source shared by all engines, or callback
  resubmission. Effects: **misalignment -13%** (no tensor in the files is
  64-byte aligned, 16 at best, header base 400 mod 4096; 280 -> 244 GiB/s,
  185 -> 162 on one link); **CPU writes into the rings at the loader's read
  rate -10%** (35 GB/s; -22% at 100 GB/s); **unplaced memory -23%** (194
  against 252 on node 0 and 272 on node 1; the driver call rises to 60 us).
  Stacked: ~42-47 GiB/s per device against the loader's 33; the remaining
  ~1.3x is unattributed and not in the pump.
- No huge pages either side (THP `madvise`, `AnonHugePages` 0).
- With the pump no longer the ceiling, placement shows: `off` 4.24/4.42 s,
  `node1` 4.45/4.19, `local` 4.55/5.14, `node0` 5.42/5.42.

### What `off` really is, and an interleaved pool (2026-09-05)

`off` was not unplaced: the CUDA driver applies a `prefer` policy for the
registering thread's node, and the main thread ran on node 1 in every run,
so `off` was `node1` by luck. A probe cannot fix that: it observes an
arbitrary state, and on a cold cache the loader's own readers create it.

The knowledge-free alternative is one pool `MPOL_INTERLEAVE`d page by page
over the nodes (`numa_maps`: half per node). gb300-2, DeepSeek replicated,
per-device pumps, depth 8, mean of three (six for the pooled `node0` =
`local` cells, where all devices sit on node 0):

| devices | node0 | node1 | interleave 0,1 | local | off |
| --- | --- | --- | --- | --- | --- |
| 4 (two per node) | 5.50 s | **4.20** | 4.35 | 4.92 | 4.26 (= node1) |
| 1 (GPU 0, node 0) | 3.38 (= local) | **2.98** | 3.22 | | |
| 2 (both node 0) | **3.82** (= local) | 3.94 | 3.86 | | |

- Four devices: interleave is within 2-3% of the best node without knowing
  where the page cache is. It puts half the pages on node 0 like `local`
  and is 12% faster, so part of `local`'s loss is the per-node split itself
  (half the retained capacity per node, strict affinity on every lease).
- One device: the *other* node is 12% faster than local
  (`read_ms_per_read` 2.3-2.6 against 3.0-4.9 ms: page-cache read and pinned
  write on two memory systems); interleave gets a third of that blind.
- Two devices on node 0: `node1` turns worst (both links cross sockets, DMA
  stage 10-11 -> 16 ms/read); interleave is at parity with local.

Interleave was never the worst and always within 3-8% of the best; every
single node was the worst at least once. On a one-memory-node host
interleave over that node is a soft bind: nothing changes, and nothing can.
With several memory nodes the mask should cover every node with memory,
not only the devices' nodes.

### What strict affinity costs

Strict affinity takes the smallest node's capacity rather than the sum
(`DmaBlockPool.retainedRequestWidth`) and maps a pre-grown arena per node.
`local` against `off`: gb300-2 `feasible_width` 79/128, retained pinned
1.53/1.02 GiB, `width_ceiling` 16/32; mi300 64/128, 2.00/1.52 GiB, **1**/32.
Pinned memory 1.3-1.5x, feasible width roughly halved.

### Defect: the growth-free ceiling mixes per-node and per-machine scopes

(Strict affinity went in the eighth pass, the adaptive controller in the
fifteenth; the invariant stands: a ceiling subtracts quantities of one
scope.) Each node's arena was grown correctly to `(preallocated_source_width
+ 1)` requests plus that node's own DMA stage (gb300-2: 33 + 16 = 49 blocks,
784 MiB per node; mi300: 33 + 32 = 65, clipped to 64 by the 2 GiB
`max_mapped_bytes`). The width ceiling (`direct_loader.zig`, sixth pass
`a7589b08`) then subtracted `dmaStageRequests` over *all* devices from the
*smallest node's* `retainedRequestWidth`, charging each node for the other
node's stage:

| host | retained/node | stage charged | stage on that node | ceiling | correct |
| --- | --- | --- | --- | --- | --- |
| gb300-2 (4 dev) | 49 | 32 | 16 | 16 | **32** |
| mi300 (8 dev) | 64 | 64 | 32 | **1** | **32** |

The correct values equal the non-strict ceiling; with one shared pool the
all-device count is right. On eight MI300X every adaptive load ran at
**width 1**: Llama-3.1-8B replicated 4.5-5.9 s against 1.3-1.6 s with
`off` (means 5.34 against 1.42 s, **3.7x**), with no memory shortage. A
second defect: the `@max(1, ...)` floor in `SourceReadWidthController.init`
turned "no growth-free headroom" into the worst operating point instead of
accepting some mid-load growth.

### The fix and its verification

`DmaBlockPool.growthFreeRequestWidth` subtracted node-wise with each node's
own reserve (minimum over nodes when strict, sum otherwise) and fed the
controller directly; `dmaStageRequests` stayed only in `RequestGateLimits`,
where an all-device count is correct. The floor became
`@max(configured.initial(), ..)`: a ceiling bounds the climb, never the
start. `max_mapped_bytes` went from 2 to 16 GiB: a guard on total pinned
memory, not a target, and at 2 GiB it clipped eight MI300X's working set.

Five interleaved reps per host: `width_ceiling` 32 in every arm on both.
mi300 `local` **5.34 -> 1.26 s**, then the fastest arm (`off` 1.47, `node0`
1.40, `node1` 1.55, at load average 37: parity), width 16-24, pinned
high-water 1.25-1.38 GiB against the aggregate pool's 1.52, retained 2.03
GiB (the full 65 blocks per node), `feasible_width` 64 -> 959. gb300-2
DeepSeek unchanged (`local` 5.63, `off` 5.60, `node0` 5.99).

### Recommendation

The 38% link penalty is real but no fixture reaches the link; at loader
level strict affinity is neutral on DeepSeek and a 4.8% loss on Llama while
costing 1.3-1.5x pinned memory. Node-local pinned blocks are the wrong
default:

- Do not derive strict affinity from the presence of `numa_node`: it buys
  nothing measured, costs pinned memory, and its worst case (everything on
  the page-cache node) is 34%.
- If placement is ever steered, steer it away from the node holding the
  source's page cache: a property of the file, not of the topology.
- Keep the means to measure locality; it matters once a source outruns
  110 GiB/s per device.
- On MI300X the aggregate pool is strictly better: same throughput, 25%
  less pinned memory, twice the feasible width.
- Refined by the interleave runs: one pool, one policy knob (interleave
  over the host's memory nodes by default, one explicit node when the
  client knows better), never placement from where the first thread ran.
  Done in the eighth pass.

An earlier recommendation (keep node-local pools on GB300, derive strict
affinity from a measured local/remote ratio) was superseded by the
interleave measurements and the eighth pass.

## Eighth pass: one pool, one placement knob (2026-09-05)

Following the seventh pass, the per-node structure went:

- `DmaWorkspace` holds one arena list and `DmaBlockPool` one free list.
  Per-node topology, strict affinity, the augmenting-path matcher
  (`planAssignments`/`assignJob`) and every width query's
  `strict_affinity` are gone; `acquireMany(io, output)` allocates nothing
  once arenas are attached; the reserve is one number, the DMA stage of
  every device. `Device.numaNode` no longer decides `dmaMap` placement.
- Placement became one knob, `NumaPlacement` (`ZML_DMA_BENCH_NUMA`:
  unset, `off`, `1`, `0,1`): by default interleave over
  `/sys/devices/system/node/has_memory`, no policy with one memory node or
  an unreadable list; a refused automatic placement (cpuset, `EINVAL`)
  warns once and continues unplaced, an explicit one fails the arena. (The
  knob went in the loader-owned initialization follow-up; the interleave is
  masked by the task's memory policy since the eighteenth pass.)
- Growth was concurrent: `growToBlocks` mapped up to four arenas at once.
  Serial growth took 317 ms for 1.02 GiB in one arena against 280 ms for
  1.53 GiB in two per-node arenas; four parts took 94 ms (pinned
  registration scales with threads on GB300). (Sequential again since the
  workspace-owned DMA allocation follow-up.)
- ROCm arenas are PJRT-pinned (`hipHostMalloc` through a device's host
  memory space), outside `mbind`'s reach; the per-node pools had split them
  50/50 by allocating through a device on each node. One pool through
  device 0 lost eight MI300X's fast mode (loader 1.27-1.45 s in every arm,
  DMA stage 112-127 ms/read, against 0.82-1.0 s and 60-74 ms); rotating
  over devices gave a 61/39 split and 0.98-1.33 s, monotone in the
  imbalance. `dmaMap` of `mmap`ed arenas on ROCm would restore `mbind` but
  costs ~4.5 s/GiB (6.5-7.2 s of pre-growth for 1.52 GiB against 1.1-2.3 s):
  rejected. The workspace keeps a `HostNode` per device-reported node (the
  attribute's one remaining use) and allocates each arena through a device
  on the node with the fewest arena bytes (unknown nodes ignored; device 0
  when none is known).

### Verification

gb300-2, four GB300, 16 MiB blocks, per-node pools -> one interleaved pool
with concurrent growth. `Loaded weights` in `//examples/io` now includes
pre-growth (calibration used to retain the pre-grown set before the timer
started), so the loader's `elapsed` is shown too:

| fixture | wall | loader elapsed | pre-growth | pinned |
| --- | --- | --- | --- | --- |
| DeepSeek repl. (3 pairs) | 4.86-4.88 -> **4.51-4.56 s** (-7%) | 4.45-4.46 -> **4.28-4.34 s** (-3%) | 280 -> 94 ms | 1.53 -> 1.02 GiB |
| Llama repl. (4 pairs) | 602-609 -> **413-423 ms** (-31%) | 319-321 -> 314-325 ms | 280 -> 94 ms | 1.53 -> 1.02 GiB |

Serial growth had the same loader elapsed but 317 ms of pre-growth. Llama
arms on the single pool (loader elapsed): `off` 304-305 ms, `node1`
299-304, interleave 315-317, `node0` 410-412: interleave 4% from the best,
23% from the worst, as on DeepSeek.

mi300, eight MI300X, Llama replicated, six interleaved pairs, host load
2-36; after = node-balanced PJRT-pinned arenas (four through device 0, five
through device 4): loader elapsed 0.85-1.18 s (mean 1.00) -> **0.84-1.05 s
(0.93)**, better in five of six pairs; wall 2.35-3.07 -> 1.94-4.19 s;
pre-growth 1.4-1.9 -> 1.1-1.5 s (one 3.2 s spike); pinned 2.03 -> 1.52 GiB.
PJRT-pinned allocation costs 0.7-0.9 s/GiB there, so the third of the
working set the per-node pools mapped twice was expensive. `zml/mem.zig`
went from 2102 to 1391 lines, the loader lost ~150.

## Ninth pass: the CPU platform takes the direct pipeline (2026-09-05)

After two passes on the buffered backend (2026-09-04: host staging pool
`579a00de`, load-profile driven reads `28961ff5`), a design review decided:
CPU joins the direct pipeline with plain pages; TPU, neuron and metal keep
the buffered backend, byte for byte, until their transfer path is measured.

### Two independent axes, conflated by one predicate

- How bytes reach the device: the PJRT async transfer manager
  (`createBuffersForAsyncHostToDevice`, then `transferData` into a byte
  range of a pre-allocated device buffer, from any host address, in any
  order), or `bufferFromHostBuffer` with a whole contiguous tensor. Only the
  second forces whole-tensor staging, and it is unrelated to pinning:
  `libpjrt_cpu.so` exports the transfer manager and copies from ordinary
  pages (its `DmaMap` is the base-class Unimplemented).
- What the host arena is (registered pages on CUDA and oneAPI, PJRT-pinned
  buffers on ROCm and TPU, plain pages elsewhere) only changes speed.
- libtpu `0.0.42.dev20260613` (`platforms/tpu/tpu.bzl`) carries
  `xla::TpuClient` overrides of `CreateBuffersForAsyncHostToDevice` and
  `DmaMap` and the `CommonAsyncHostToDeviceTransferManager` with
  `TransferRawDataToSubBuffer` the fifth pass validated. A symbol is not
  proof the override is more than a stub: TPU moves only after a runtime
  check on a TPU host.

### What the change is

- `DmaWorkspace.isSupported` became an exhaustive switch over targets, so a
  new one forces a decision, and a `pageable` arena kind (our pages, never
  registered, NUMA-placed and huge-page advised) serves CPU. (Now
  `Loader.backendFor` and `host_memory.Workspace.Backend`, per the
  follow-ups at the top.) The five end-to-end `loader ...` tests run on
  both backends on CPU.
- Fixed from the review: `numa_mask` was read and written outside
  `arena_mutex` while `growToBlocks` mapped arenas concurrently, so a
  refused automatic placement could be undone by a worker that had read
  the mask earlier; its one transition (nonzero to zero) is taken under the
  mutex.
- `direct_loader.zig` is unchanged; `transferData` is the plugin's copy.
  Calibration returns the defaults on CPU without measuring (`6f75cd37`):
  it had measured a memcpy (2 MiB at 102 GiB/s, 607 ms for four CPU
  devices) and the load took 1.55-1.60 s at 2, 8 and 16 MiB alike.

### Why: the buffered read model, not its budget, was the problem

- Interleaved on the B70 (oneAPI `level_zero:1`, `hf://Qwen/Qwen3.5-4B`,
  same minutes): direct 10.9 and 16.3 s (278 coalesced 32 MiB reads for 738
  tensors, width 32, ~1.5 GiB pinned) against 48.3 and 47.6 s for buffered
  `28961ff5`, which had done 22.4 s the day before: `hf://` varies 2x day
  to day, so only interleaved pairs count.
- 604 of the 738 tensors are under 32 MiB: tensor-shaped reads cost ~900
  requests, coalescing 278. The direct loader also blind-grows to width 32
  before the first response.
- Staging bounds, K = 12, K x largest (the buffered budget) against the sum
  of the top K: Llama-3.1-8B 11.7 against 3.05 GiB, Qwen3.5-4B 14.2 against
  1.67, Qwen3.6-27B 28.4 against 6.4. The direct pipeline needs ~300 MiB
  locally and 1.06 GiB on `hf://` for any of them: it never holds a whole
  tensor.
- Buffered on CPU, fixed read width 1/2/4/8/12/24/48, page-cached Llama:
  2.93/2.33/1.84/1.41/1.38/1.43/1.48 s. The plateau is at 8: a transfer
  concurrency cap is not a lever.
- Rejected: biggest-first order with a sum-of-top-K budget. It forfeits
  coalescing (604 small tensors become 604 tail round trips), its reuse
  needs buffers that shrink, which PJRT-allocated pinned memory cannot do,
  and the checkpoints have only 5-23 distinct tensor sizes, so best-fit
  already reused. Uniform blocks give the invariant by construction.
- `examples/io/main.zig` hands `init.arena.allocator()` to `Loader.init`,
  whose `free` only frees the latest allocation, so buffered staging drops
  leaked until the arena died. Any staging design must own its allocator.

### Results (B70 host, four CPU devices, warm page cache, adaptive default)

All runs were sharded (`examples/io load <path>` defaults to `sharded`;
they were first mislabelled replicated). Buffered against direct, three
runs each: Llama-3.1-8B 1.36-1.50 against 1.55-1.58 s, Qwen3.5-4B 0.92-1.23
against 0.88-1.10, Qwen3.5-9B 2.91-3.87 against 2.01-2.27;
`hf://Qwen/Qwen3.5-4B` direct 10.4 s (278 reads, width 32) against 48 s
buffered on oneAPI the same hour. Staging high water 292-298 MiB locally,
1.06 GiB on HF; read-back (`ZML_LOAD_CHECK=1`) 291/291 on Llama, 259/775 on
Qwen3.5-9B; width 12-24. True replicated Llama-3.1-8B (4 x 14.96 GiB of
device buffers) does not fit this 62 GiB host: buffered is OOM-killed at
62.3 GiB RSS, direct completes at 60.4 GiB with the page cache evicted.
Prefer the interleaved table of the next section.

### Calibration skipped on CPU, and what the CPU pump is doing (2026-09-05)

The ready line reads `dma_block_size=4 MiB, pregrown=392 MiB,
pregrowth_ms=20` where the benchmark took 607 ms. Same tree (`6f75cd37`),
buffered built by forcing the CPU arena kind to null, interleaved after a
warm-up:

| fixture | direct | buffered |
|---|---|---|
| Qwen3.5-4B replicated (34.7 GiB of device buffers) | 2.03, 2.01 s | 2.02, 1.98 s |
| Qwen3.5-4B sharded | 0.93, 0.92 s | 0.92, 0.91 s |
| Llama-3.1-8B sharded | 1.55, 1.57 s | 1.38, 1.35 s |

`dma_submit_us_per_piece` equals `dma_piece_latency_ms` (1.26 ms per
~3.1 MiB piece, Llama sharded): the plugin copies on the submitting thread
and the pump is saturated (1234 pieces per device x 1.26 ms = 1.55 s, the
whole load; replicated 4101 x 1.26 = 5.2 s, measured 5.0-5.3); readers
wait on it (DMA stage 37 against read 2.3 ms/read). The copy first-touches
the plugin's fresh device buffers in 4 KiB pages (3.94M minor faults for
14.96 GiB, 10 s system time): 2.4 GiB/s per pump thread under THP
`madvise`, and why calibration's reused ring measured 102 GiB/s. Buffered
faults the same pages from twelve workers and ties on Qwen, so more
copying threads per device are worth at most Llama sharded's unattributed
14%. One pump per device stays (seventh pass); CPU is not a performance
target. THP `always` is untested (needs root).

### If a platform ever needs `bufferFromHostBuffer` under the same pipeline

Keep everything through the read and swap the sink: the first job touching
a tensor opens a contiguous span from the arena (a ring: tensors open in
plan order and complete roughly in order); jobs scatter into
`(span, offset)`; the per-tensor byte counter that flags the last transfer
submits `Buffer.from` with `wait = false` into the device pump, which
awaits ready events and releases the span. The cost against the block sink
is an arena floor of the largest tensor plus the read working set and no
overlap inside a tensor, which is why the transfer manager is preferred
wherever it exists.

## Tenth pass: loader design review (2026-09-05)

The review started from `093e065d`. The core architecture holds: source
coalescing, separate read and request lifecycle credits, per-device pumps,
reference-counted blocks and caller-side execution each solve a distinct,
measured problem. The change was structural: boundaries and ownership.

### Design changes

- `zml/mem/dma.zig` took the workspace (owns arenas), the block pool (a
  load-scoped lease view), the allocator adapters and `NumaPlacement`; all
  moved to `io/host_memory.zig` in the follow-ups at the top.
- `io.zig` is the facade; `io/TensorStore.zig` does checkpoint lookup,
  `io/buffered_loader.zig` whole-tensor staging; both backends consume
  `backend.LoadSpec`. In `direct_loader`, `Planner` owns coalescing and
  `Scheduler` FIFO publication and claims, and runtime types lost their
  Fair/Vectored/Loader prefixes (`Pipeline`, `ReadRequest`, `SourceSlot`).
- API renames: `Loader.Opts` -> `Loader.Options`, `.dma_pool` ->
  `.dma_workspace`, `mem.Dma*` -> `mem.dma.*`, `io.max_load_*` ->
  `io.limits.max_*`, and `_per_device` dropped from the minimum-transfer
  options.
- Documented: serialized front-end use, borrowed lifetimes, and that
  `Handle.isDone` means reads and transfers finished while an await may
  still run executables.

### Ownership fixes found by the review

- The PJRT-buffer allocator rounded `header + length` instead of reserving
  padding before the data (alignment 64, length 100: 128 bytes allocated,
  data at offset 64). It now reserves worst-case padding, aligns from the
  actual PJRT base and keeps the header just before the data (tests:
  alignments 1-4096, unaligned bases, size overflow).
- A canceled calibration sleep returned while transfer workers still
  borrowed the window's stack; every exit now stops, releases the start
  gate and joins.
- Calibration leaked an owned PJRT buffer when the manager-list append
  failed; capacity is reserved before PJRT objects are created.
- Recalibration consulted only the newest arena; `findArena` searches every
  retained arena before growing.

### Remaining issues, separate from the structural refactor

- A manager may already have lost its definition event when another piece,
  finalization or `setBufferErrorUnknown` reaches it (plugin-sensitive).
  The oneAPI abort and the `toSliceAlloc` sub-byte placement issue are
  historical, not reproduced.
- `pjrt.Event.await` returns at once for an already-ready event without
  inspecting its error, so calibration can miss a synchronous error
  (PJRT-wide follow-up).
- A failed ROCm arena allocation left the host-node byte count inflated
  (fixed in the workspace-owned DMA allocation follow-up).
- No accelerator throughput was measured for this refactor.

### Validation

macOS arm64: `bazel build //zml //examples/io //examples/llm` and
`bazel test //zml:test //stdx:test //zml/tokenizer:test //vfs:test
//examples/io //examples/llm` passed (303 cases, three unrelated attention
cases skipped). `bazel build //...` stops in analysis at
`//bin/zml-smi/platforms/nvml:nvml` (Linux-only selects). Core tests had not
compiled on macOS: generic assertion diagnostics traversed
Workspace/Handle/callback pointers into a CUDA-only module; loader tests now
compare pointer identity directly.

## Eleventh pass: direct I/O as a loader option (2026-09-05)

Direct I/O can be a reliable loader option; three things stood in the way,
none of them the pipeline. (The twelfth pass replaced this pass's
`vfs.File` mechanics; the measurements, defects and defaults stand.)

### What the storage does without the loader

Python probe: `preadv` into page-aligned anonymous memory, 16 MiB chunks,
`posix_fadvise(DONTNEED)` for cold; residency from `Cached` in
`/proc/meminfo` (`mincore` and `cachestat` refuse files this user cannot
write). GiB/s:

| host | cold buffered | direct | warm buffered |
|---|---|---|---|
| gb300-2 (ext4, 4-NVMe raid0, 64 KiB pages) | 52.8 | 53.2 (8 to 64 threads) | 120.9 |
| mi300 (ext4, LVM over a 4-NVMe raid0 + boot NVMe) | 4.7 to 4.9 | 25.1 | 70.1 (32 threads), 43 (12) |

Direct I/O is the disk's ceiling on both; cold buffered reaches it on
gb300-2 and a fifth of it on mi300 (readahead-bound through md and LVM);
warm buffered is 2.3-2.8x the disk. Fixture hazards: gb300-2 held only
16 GB of page cache hours after 149 GB loads (no container memory limit;
cause unknown), so an earlier DeepSeek number there may be warm or cold
(4.1 against 4.5 s); Llama on mi300 sits on the volume's slow extent (cold
3.3 GiB/s buffered or direct, against 25 for DeepSeek on the same
filesystem), the wrong cold fixture there.

### Three defects, one of them old

1. **Bare paths never reached the file backend.** `VFS.lookupDir` sent
   absolute (and non-URI relative) paths to the inner `Io` while
   `loadProfile` attributed them to the `file` backend, so every loader
   read of `/var/models/...` bypassed `vfs.File` and its
   `direct_io = true` had never applied to a load.
2. **The old `vfs.File` failed the open when the filesystem refused
   `O_DIRECT`** (`error.Unexpected`), served streaming reads from the direct
   descriptor's position, and aligned only by luck: a direct read needs
   offset, every buffer and total aligned, and safetensors packs tensors at
   16-byte boundaries (DeepSeek: 1566 of 1576 tensors in a file at exactly
   16, the data section at 3096 mod 4096), so a read starting at a tensor
   never qualifies.
3. **Zig's threaded `Io` treats `EINVAL`/`EFAULT` from `preadv` as a
   programmer error** (panic in debug, `error.Unexpected` in release), and
   those are what a filesystem that accepts the flag but not the read, or a
   mapping the block layer cannot pin, answer.

### The change

- Backend policy `off`/`on`/`auto` (default `auto`: direct for a file at
  most half in the page cache at open). Residency from `cachestat`, or where
  refused (kernel < 6.5, or a file this process may not write: both bench
  hosts) from 32 `RWF_NOWAIT` reads spread over the file (`EAGAIN` for an
  uncached page, no permission needed); unknown counts as cached. Direct
  reads use raw `preadv`; `EINVAL`, `EFAULT`, `EOPNOTSUPP` mark the file
  refused (one warning) and the buffered descriptor answers.
- Loader `direct_io: bool = true`: the planner widens each job's read to
  aligned bounds (`alignBackward` of the first tensor byte, `alignForward`
  of the last), transfers address the widened read, `maximumJobLen` leaves
  two alignment units so a widened job still fits
  `maximumCoalescedJobBlocks`, and `Batch.Plan.Job.minimum_len` tells
  `safetensors.readFilePositionalAllV` how much must exist (end of file is
  accepted only past it). Neighbouring widened reads share up to one
  alignment unit; their transfers never overlap. A file is widened only
  when its reads go direct (see the warm results).

### Results

`examples/io load <model> replicated`, 16 MiB blocks, adaptive width. Arms:
`exact` (unwidened), `direct` (loader on, backend `on`), `auto` (loader on,
backend `auto`), `widened` (loader on, backend `off`: aligned reads served
buffered). Cold arms evict the model first. Loader `elapsed`; in
parentheses read and DMA-stage ms per 16 MiB request.

gb300-2, four GB300, three reps (two for DeepSeek):

| model, state | exact | direct | auto | widened |
|---|---|---|---|---|
| Llama cold | 0.358-0.359 s (rd 6.4, dma 1.1) | 0.294-0.300 (rd 2.2, dma 15) | 0.295-0.300 | 0.357-0.358 |
| Llama warm | 0.319-0.324 (rd 2.4, dma 17) | 0.297-0.299 (rd 2.2, dma 15) | 0.327-0.344 (dma 19-20) | 0.332-0.340 |
| DeepSeek cold | 4.48 / 4.59 (rd 5.0, dma 12) | 3.76 / 3.76 (rd 3.4, dma 22) | 3.74 / 3.46 | -- |
| DeepSeek warm | 3.83 (rd 2.7, dma 23) | 3.74 / 3.63 (rd 3.1, dma 21) | 4.29 / 4.36 (dma 24-26) | 4.06 / 3.95 |

Direct makes cold loads 12-16% shorter and leaves the page cache untouched
(`Cached` stays at 174 GB where `exact` adds the model). It also beats the
warm buffered load by 5-6%: the loader is not read-bound at 33-37 GiB/s,
and the page-cache copy (`_copy_to_iter`, 1.3 cores of reader sys time)
competes with the DMA engines for host memory (DMA stage 17 -> 15 ms warm;
seventh-pass pump measurements). `widened` costs 3-6% warm: an exact read
lands a job's first tensor at block offset 0, a widened one at its file
offset mod 4 KiB, and a non-block-aligned DMA source is slower (the seventh
pass's -13%). Hence `auto` widens only files it reads directly.

mi300, eight MI300X, host load 9-30 (other users; DeepSeek replicated on
ROCm is DMA-bound at 115-140 ms per request, a separate condition):

| model, state | exact | direct | auto | widened |
|---|---|---|---|---|
| Llama warm | 1.10-1.12 s (rd 9.3, dma 72-74) | 4.53-4.57 (rd 58) | 1.07-1.11 | 0.94-0.96 |
| Llama cold (slow extent) | 4.72-4.75 (rd 62) | 4.52-4.54 (rd 59) | 4.52-4.55 | -- |
| DeepSeek cold | 35.1 (rd 44, width 12) | 12.5 (rd 5.6, dma 117) | 13.0 (rd 5.7) | -- |

Direct I/O into ROCm's PJRT-pinned arenas (`hipHostMalloc`) works: no
refusal, verified loads (`ZML_LOAD_CHECK=16` Llama, 256 DeepSeek, both
hosts). DeepSeek cold is 2.8x faster, then DMA-bound. Forcing direct on a
warm Llama costs 4x here (slow extent); `auto` chose buffered for it (probe:
100% cached) and direct for cold DeepSeek (0%).

Rerun with the probe gating the widening (cache state checked per row from
`Cached`), warm, `exact` / `auto` / `direct`: gb300-2 Llama 0.318-0.321 /
0.314-0.333 / 0.296-0.300 s, DeepSeek 4.10 / 3.95-4.69 / 3.77-3.79; mi300
Llama 1.10 / 0.97-1.01. `auto` on warm Llama is at parity with `exact` on
both hosts (4-7% behind on gb300-2 without the probe); DeepSeek warm was
too noisy to resolve a few percent. Cold loads keep the direct gain
(gb300-2 DeepSeek 4.58 -> 3.73 s, mi300 34.9 -> 12.8).

### Recommendation and defaults

Loader on and backend `auto` are the defaults. Where the disk is as fast as
cold buffered (gb300-2), a user who knows the model lives there gains
another 5% warm with `on`; on a slow extent or disk `on` is a 4x loss, so it
stays opt-in. Residency is measured, not guessed; unknown means buffered.
Nothing in the pipeline changed: widening is a planner concern, the read
helper learned that padding may end early, and the descriptor and its
failure modes stay below the loader.

## Twelfth pass: direct I/O as the VFS's own local case (2026-09-06)

A review of the eleventh pass asked whether the pair of descriptors behind
every `vfs.File` handle earns its keep, why `DirectIoProbe` needs opaque
data and function pointers, and whether `vfs.File` has a purpose without
the direct logic. The answers gave a smaller design with the same read
path.

### What the eleventh-pass shape was paying for

- **The pair.** To the loader a file was already one or the other (decided
  at open, widened whole, every loader read aligned). The buffered twin
  served readers that never align (the header parser, `TensorReader`, the
  buffered loader) and the refusal fallback; `O_DIRECT` belongs to the open
  file description (a `dup` shares it), so one descriptor would have had to
  bounce every unaligned read.
- **The probe.** `Backend` is type-erased and `LoadProfile` serves `hf`,
  `s3` and none alike, so nothing could name `vfs.File`, and `std.Io.File`
  (a handle and a `nonblocking` bit) had no return channel for an answer
  that exists only after the open.
- **The file backend** (#362, direct I/O in #367) only forwarded to the
  inner `Io`, and its handle table mirrored the VFS's (a second translation
  under a second mutex per read). Under `auto` it measured residency at
  every read-only open, the registry parser's header-only ones included.
- **Rejected:** moving the mechanism into the loader (a
  `local_io: ?std.Io` in the profile). It is four descriptor operations
  (`fcntl` for the flag, `cachestat` or `preadv2(RWF_NOWAIT)`, a raw
  `preadv` with decoded errno, `fcntl` on refusal) and must live where the
  descriptor is visible; the VFS hides exactly that.

### The change

- `vfs/file.zig` is deleted. Bare paths and `file://` are the VFS's own
  local case (inner `Io`, local profile: 8 MiB requests,
  `direct_io_alignment = 4096` on Linux); the no-VFS `LoadProfile.local`
  keeps a null alignment and reads exact ranges.
  `registerBackend("file", ...)` returns `error.ReservedScheme`; llmd's
  registration (`monorepo/llmd/main.zig:236`) must go.
- One descriptor per local handle, with a state on the VFS handle entry
  (`undecided`, `buffered`, `direct`). `VFS.useDirectIo(file, policy)` is
  the planner's one call per file (policy, read-only, regular, at least one
  alignment unit, residency under `auto`, then the `fcntl` flag). The first
  decision stands for the handle's life: the flag belongs to the
  description, which `SourceSlot` keeps, and a caller told "buffered" plans
  exact reads a direct descriptor would reject. The flag and the state
  change together under the VFS mutex (a race the review found).
  Descriptor operations live in `vfs/direct_io.zig`.
- The planner finds the VFS through the `Io` that opened the file
  (`VFS.fromIo`, vtable identity), not a profile pointer that could name
  another VFS. A loader without a VFS never widens.
- A rejected direct read (`EINVAL`, `EFAULT`, `EOPNOTSUPP`) demotes the
  file to buffered: the flag comes off before the inner `Io` sees the
  descriptor, one log line names file, offset, buffer and total, and the
  read is answered buffered. Aligned rejection: the filesystem refuses
  direct I/O it accepted the flag for (warning); misaligned: the reader's
  plan or a continuation after a short read, which FUSE or NFS may return
  mid-file (error). Demotion rather than failure, because that
  continuation is legitimate and one descriptor can only finish it
  buffered. Streaming reads and file copies demote first, so the inner `Io`
  (which panics on that errno) never sees the flag.
- Residency is sampled only (32 `RWF_NOWAIT` reads): a whole-file
  `cachestat` walks every cached folio, ~7 ms per cached GiB on a 4 KiB
  page kernel, unbounded on the planner thread, and neither bench host ever
  exercised it. The sample resolves a 0.5 threshold at constant cost.
- The policy is the loader's: `Loader.Options.direct_io: VFS.DirectIo`
  (`off`, `on`, `auto`; default `auto`), `ZML_DIRECT_IO=off|on|auto`
  (replacing `0|1` and `ZML_VFS_DIRECT_IO`). `DirectIoProbe` and the
  profile and hint fields that carried it are gone.
- `safetensors.readFilePositionalAllV` returns after a short read once
  `minimum` is in: only unaddressed padding remains, and the follow-up read
  would start unaligned. Unwidened reads have `minimum` = the whole read,
  so remote backends and `TensorReader` are unchanged.
- Known limits: a long-lived loader that found a file cached at its first
  submit reads it buffered at every later submit, even after the cache is
  dropped (one description per file in `SourceSlot`; a fresh open per
  submit would let `auto` re-measure); the alignment is a 4 KiB constant,
  not `statx(STATX_DIOALIGN)`, so a filesystem with a larger logical block
  gets one rejected read per file and then buffered reads. Pre-existing:
  the VFS's streaming write forwarded handles untranslated and its seeded
  stdio entries mapped index 1 to stdin, cancelling out for stdout and
  stderr (fixed in the twentieth pass).

### What `auto` does to the page cache

A direct read never fills the page cache, so under `auto` a cold file stays
cold at the next load: a fixed point. Its cost is a property of the host
(eleventh-pass numbers):

| host, model | cold buffered, then the next load warm | direct, every load |
|---|---|---|
| gb300-2 Llama | 0.359 s, then 0.320 | 0.297 |
| gb300-2 DeepSeek | 4.5, then 3.8 | 3.7 |
| mi300 Llama | 4.73, then 1.10 | 4.52 |

On gb300-2 direct beats warm buffered, so never warming costs nothing. On
mi300 `auto` trades a 4% gain on the first Llama load for losing the
1.10 s warm load on every later one. No residency measurement sees that: it
depends on whether the host's disk beats its warm buffered path, which the
user knows and the file does not. The default stays `auto` (a first load
never slower than cold buffered, the cache left as found); a user who
reloads one model on a host whose warm buffered reads beat its disk should
set `off`, as the option's doc comment says.

### Verification

- `bazel test //vfs:test //zml:test` and the example builds pass. The VFS
  test runs every policy on a temp file on ext4: `on` goes direct (its
  unaligned read is rejected, answered buffered, logged once), `auto` reads
  the just-written, cached file buffered, `off` never asks, a streaming read
  demotes. Test assertions that encoded ext4's rejection were dropped:
  tmpfs serves unaligned direct reads.
- gb300-2 (kernel 6.8, 64 KiB pages, four GB300, idle; trees on
  `b59c41b7`), `bulk phase` elapsed. Pre-review tree: `ZML_LOAD_CHECK=16`
  under `on` warm and cold and `ZML_LOAD_CHECK=256` on cold DeepSeek under
  `auto` all ok, no refused read or panic in 15 runs; direct and cold arms
  at parity with the eleventh pass (Llama warm `on` 0.301 s against 0.297,
  cold `auto` 0.303 against 0.30), warm `off`/`auto` 0.334-0.343, 5% above
  it. `auto` chose buffered for warm files (100% cached), direct for cold.
- Closing A/B, final tree against the eleventh-pass tree, Llama replicated,
  four interleaved rounds, medians:

  | arm | eleventh pass | final tree |
  |---|---|---|
  | warm exact / `off` | 338.8 ms (334.9-341.6) | 335.8 (334.3-339.0) |
  | warm `auto` (buffered) | 339.6 (332.4-357.7) | 336.8 (334.2-341.3) |
  | cold `auto` (direct) | 295.4 | 301.6 |
  | warm `on` + check | -- | 302.1 |

  Per-read metrics identical (read 2.30-2.36 ms, DMA stage 18.3-18.6 ms);
  the final tree is 3 ms ahead on both warm pairs, inside the spread, so the
  5% was the host's day, not the change. Both `ZML_LOAD_CHECK=16` runs under
  `on` passed; no refused or misaligned read in 20 runs. The residency
  sample now sits mid-window: at offset 0 it hit the header page the
  registry parser had just read, so a cold file reported 3% cached
  (decision unaffected).

## Thirteenth pass: loader-owned `loadExecute` concurrency (exploration, 2026-09-10)

An exploration; the fourteenth pass implemented its decision. Question:
`loadExecute` needs device memory for its inputs and its execution
beyond the model, which is why the caller sized a `Window`; the
executable can report what it needs, so could the loader admit
submissions itself? Tree `fa232bb2 direct io` (twelfth pass) plus a
temporary log of `PJRT_Executable_GetCompiledMemoryStats` and
`PJRT_Device_MemoryStats`.

### The loader and its consumer at the time

- `Handle.await` ran a submission's executables on the awaiting task
  (`.wait = true`), wrote the outputs and freed the inputs.
  `zml.io.Window{budget_bytes, max_handles}` was the only admission
  control, sized by `executeInputBytesPerDevice(exe)`: inputs only, blind
  to temporaries and the output. llmd Laguna (`loader-third-pass`)
  submitted one `loadExecute` per sparse layer through a window of one
  (`--expert_pack_budget` 0): the read pipeline drained at every layer.
- Device memory is not taken at submission: a tensor allocates its PJRT
  buffers at the first read job touching it (`Item.ensureState`,
  `TensorTransfer.initResolved` -> `createBuffersForAsyncHostToDevice`),
  and jobs are claimed FIFO across submissions, at most `width` jobs
  ahead of the batch being awaited. The `Window` bounded
  landed-but-unexecuted submissions.
- llmd on monorepo `master` (`a64fd7a9`) used the pre-rework zml
  (`f8ddb3e5`), whose `loadExecute` was synchronous (read every source,
  execute, free): one fused tensor's inputs on the device at a time, the
  read pipeline stopped for each. Its one call site, `loadPacked`
  (`llmd/weights.zig:985`), submitted every tensor with a `Packer` recipe
  (`fuse`, `stack`, `dequantize_blocks`, ...) one at a time before the
  bulk `load` and one await, compiling the pack executables lazily in
  that loop: XLA compiles on the critical path with nothing in flight.
  `laguna_dflash.zig:793` bypassed the packer, unlike the other dflash
  wrappers.
- Every model packs: dense ones QKV and gate/up (2 per layer), MoE ones
  add expert stacks (4 to 5 per layer), deepseek4 fp8 conversions and
  dequantization. On Llama-3.1-8B packed tensors are 8.5 of the 15 GiB,
  so the sequential path carried most of a dense model's bytes.
- Device memory was budgeted before the load: after compiling,
  `CacheSpec.initWithAllAvailableMemory` (`attention.zig:838-866`) sized
  the KV cache as `bytes_limit x cache_memory_fraction (0.95) -
  fixed_memory` (model bytes plus attention working set) and allocated it
  before `Loader.init`; `bytes_limit` is 0.9 of the device (CPU: 16 GiB). Nothing reserved the
  packing transient (inputs plus `temp`): it lived in the 5% margin plus
  the attention estimate's overshoot, about 8 GB on a 192 GB MI300X,
  which made the sequential `loadExecute` safe.
- Nothing else ran on the devices during the load, no `loadExecute`
  output feeds another, and the only ordering is packed before bulk.

### What PJRT and XLA allow

Read in openxla `b014a9c1` (2026-07-27; zml pin `41370d1124`; shipped
CUDA plugin manual-2026-07-31).

- `PJRT_Executable_GetCompiledMemoryStats` is implemented by the CPU
  client (`xla/pjrt/cpu/cpu_client.cc:1218`) and the stream-executor
  client for a single-program executable
  (`xla/pjrt/se/stream_executor_executable.cc:174`; MPMD: Unimplemented).
  `compiled_memory_stats.cc:34-105` classifies one partition's buffer
  assignment: parameters `argument`, live-out `output`, preallocated
  temporaries `temp`, plus host variants. zml binds it
  (`Executable.getCompiledMemoryStats`).
- `PJRT_Device_MemoryStats` (`bytes_in_use`, `bytes_limit`,
  `largest_free_block`, `pool_bytes`) is the BFC allocator on GPU clients
  (`xla/pjrt/gpu/se_gpu_pjrt_client.cc:1714`); the CPU client has no
  `GetAllocatorStats`, so `Device.memoryStats()` returns zeroes after the
  plugin logs `Unimplemented` on every call.
- An async host-to-device manager allocates its device buffers
  synchronously at creation (`host_to_device_transfer_manager.cc:137` ->
  `pjrt_stream_executor_client.cc:505-530`, `retry_on_oom = true`).
- Execute on inputs not yet landed: `ExecutePrepare`
  (`common_pjrt_client.cc:1912-1945`) takes the inputs' definition events
  as `extra_deps` and allocates outputs as delayed memory
  (`pjrt_stream_executor_client.cc:540-550`). The launch (`:1866-1880`)
  goes to the device's `async_dispatch_thread` if there is one;
  otherwise `BufferSequencingEvent::WaitForEventOnStream`
  (`buffer_sequencing_event.cc:58-61`) blocks the caller until every
  input's last piece is submitted. The thread exists with
  `use_async_dispatch` (`se_gpu_pjrt_client.cc:1795-1800`, default off,
  `PJRT_GPU_ENABLE_ASYNC_DISPATCH=1`) or the C API option
  `use_tfrt_gpu_client` (`plugin/xla_gpu/xla_gpu_pjrt_client.cc:27-31`,
  `pjrt_c_api_gpu_internal.cc:191-229`), which zml passes as
  `gpu_async_dispatch` (default true) for CUDA only
  (`zml/platform.zig:792,836`). The CPU client always defers
  (`cpu_client.cc:1945`, `ExecuteWhenReady`). So "execute at submission"
  is host-free on CUDA and CPU and would block on ROCm/oneAPI unless that
  option is passed there (untested).
- Unchanged from the third pass: destroying a buffer an execution
  references is safe; concurrent `Execute` on one executable is
  undocumented, so executions stay on one task.

### Measurements (2026-09-10)

Compiled memory stats of the playground pack (`Tensor.stack` of 16
rank-2 sources, replicated output), per partition, i.e. per device:

| platform | executable | argument | output | temp |
|---|---|---:|---:|---:|
| B70 host, CPU x4 (Qwen3.5-4B) | 16 x {9216,2560} bf16, and its transpose | 720 MiB | 720 MiB | 2.72 GiB |
| gb300-2 GPU 1 (Llama-3.1-8B) | 16 x {14336,4096} bf16 | 1.75 GiB | 1.75 GiB | 0 |
| gb300-2 GPU 1 | 16 x {4096,4096} bf16 | 512 MiB | 512 MiB | 0 |

Every other figure was 0. On the GPU the stack is a copy; XLA CPU needs
3.8x the output in temporaries, so a 720 MiB CPU pack on four replicated
devices holds 4 x (0.7 + 0.7 + 2.72) GiB while it executes, which
`executeInputBytesPerDevice` cannot see. CUDA `bytes_limit` 248.96 GiB,
`bytes_in_use` 0 before the load; CPU none. CPU anatomy (cold, direct,
window 1): the caller spent 1.27 s executing one 720 MiB pack between two
submissions; pack phase 0.48 GiB/s (1.41 GiB in 2.935 s) against the bulk
at 9.36 GiB/s: on CPU execution is six times the read.

Window sweep, gb300-2 GPU 1 alone (`CUDA_VISIBLE_DEVICES=1`),
Llama-3.1-8B-Instruct warm (buffered), `ZML_LOAD_PACKS=64
ZML_LOAD_PACK_WIDTH=16`: 14 packs of 16 sources (13.00 GiB) then the
1.96 GiB bulk, three interleaved rounds of `ZML_LOAD_PACK_WINDOW`:

| window | pack phase | GiB/s | `Loaded weights` wall |
|---:|---|---:|---|
| 1 | 292-304 ms | 42.7-44.6 | 551-571 ms |
| 2 | 245-246 | 53.0 | 505-507 |
| 4 | 236 | 55.0 | 498-499 |
| 8 | 236-237 | 55.0 | 497-500 |

The bulk read at 54 to 57 GiB/s in every arm. A window of one costs 20%
of the pack phase (about 4 ms of drain, plan, execute and ramp per
submission for packs that read in 17 ms); window 2 recovers 85% of it, 4
is at bulk speed, 8 adds nothing (the third pass's 5090 result with a
clearer knee).

### Options and decision

Cost to bound per device: the inputs of every submission that started
reading and has not executed, plus `temp + output` of the one executing,
all known before submission. Room: on GPUs `bytes_limit - bytes_in_use`
minus admitted bytes not yet allocated minus a reserve; on CPU nothing.

- A (taken): admission moves into the loader, execution stays on the
  caller's task: `loadExecute` retires the oldest submissions (running
  their executables) until the new one fits. No task, no new PJRT usage;
  execution waits for the next admission or await, which the sweep says
  costs nothing at depth 2 to 4.
- B: a loader executor task executing each batch as it lands (extension
  E: one call for all layers, per-binding readiness counters). Not taken:
  a task and cross-task state to free inputs milliseconds earlier, for a
  caller with nothing else to do during the load.
- C: enqueue the execute at submission and let PJRT order it on the input
  events. Not taken: eager input allocation, a per-platform gate
  (ROCm/oneAPI block), completion callbacks and a new error path for only
  B's gain.
- D: a default window of 2 to 4 in llmd. Not taken: keeps a knob whose
  value the loader can measure.

The user's two corrections: the KV cache belongs after the weights
(`master` allocates it before, `llama.zig:1118-1147`; llmd's to fix), and
devices where the cache would be small must be supported.

- With the cache after the load the room is tens of GB on every GPU here
  against pack transients of 0.2 to 1.75 GiB, and depth is bounded by the
  pipeline (a GPU pack executes in about a millisecond). Memory binds only
  on CPU: one Qwen pack replicated on four devices costs 16.5 GB of the
  62 GB B70.
- Where the cache would be small the room is a few GB against 1 to 3 GiB
  packs, hence a derived byte budget: room = `bytes_limit - bytes_in_use`
  minus the loader's own unlanded weights (a per-device allocated-bytes
  counter) minus a reserve; cost = `inputs + temp + output` (`temp`
  matters for the fp8 `dequantize_blocks` recipes); always admit one, so
  a one-pack room degrades to `master`'s sequential order; depth 1
  without stats (CPU). The budget does not depend on the allocation
  order: with the cache first the room is the residual margin (about 8 GB
  on a MI300X, 12 GB on a GB300), enough for dense packs four deep and
  fp8 packs one or two deep.
- Fragmentation, the one way overlap could hurt a cache sized from what
  is free after the load (gb300-2 GPU 1, Llama, 14 packs of 16, stats
  right after the bulk phase, two rounds, BFC pool 248.96 GiB):

  | window | pack phase | bytes_in_use | largest_free_block | num_allocs |
  |---:|---|---:|---:|---:|
  | 1 | 292.8 / 293.7 ms | 15.04 GiB | 232.12 GiB | 347 |
  | 8 | 240.4 / 235.7 ms | 14.99 / 15.05 GiB | 231.90 / 231.65 GiB | 347 |

  Free memory is 233.9 GiB in both; the sequential order leaves 1.8 GiB
  outside the largest block, the overlapped one 0.2 to 0.5 GiB more:
  noise against the 12 GiB `cache_memory_fraction` keeps back.
- Per-layer submissions at depth 2 to 4 suit both dense and MoE models (a
  dense model could submit every pack at once, 8.5 GiB of inputs on
  Llama-8B; a MoE model's pack inputs are most of the model).
- `load` must count a transformed tensor as delivered once its
  `loadExecute` is submitted (it refused with
  `TransformedTensorNotDelivered`, forcing a drain before the bulk), and
  pack executables should compile in `Model.init` with the rest.
- Decided with the user: the per-submission `Handle` goes. It existed so
  the caller could sequence `loadExecute`; every in-repo `load` user
  awaited it on the next line, and monorepo `master` has one
  `loader.await`. Surface: `load`, `loadExecute`, `awaitAll`,
  `bytesLoaded`, `deinit`. Executables run on the caller's task, inside
  `loadExecute` when admission retires older submissions and inside
  `awaitAll`; `deinit` retires without executing; `awaitAll` returns the
  first error. Dropped on purpose: selective or early awaits (no user),
  separate pack/bulk timing (batch diagnostics carry it).

### Open questions

- Laguna's `gate_up_proj` executable stacks twice and concatenates; read
  its `temp` on ROCm and CUDA before relying on GPU `temp = 0`.
- Whether ROCm and oneAPI honor `use_tfrt_gpu_client` (only C needs it).
- Whether CPU should pack at all (execution 6x the read, 3.8x the output
  in temporaries).
- CUDA `bytes_limit` was 248.96 GiB of a 284 GB device; how it tracks
  `memory_fraction`, and what `bytes_in_use` includes on ROCm, unchecked.

## Fourteenth pass: loader-owned `loadExecute` admission, `awaitAll` only (2026-09-10)

Implements the thirteenth-pass decision on `2f170c75 simplification`.

- `load` and `loadExecute` return `!void`; `awaitAll` is the only wait;
  `Handle`, `Window` and `executeInputBytesPerDevice` are gone.
  Submissions live in a FIFO; `retireOldest` awaits the reads, runs the
  executables when asked, frees the inputs and commits the bytes only
  when it ran. The first retire error is sticky in `Loader.failure`.
- Admission in `loadExecute` (`execute_admission.zig`; folded into
  loader.zig in the thirty-eighth pass): cost per device = input and
  output placements + the executable's `temp_size_in_bytes` (queried per
  distinct `*const Exe` within the call, 0 when the plugin cannot answer;
  no cross-call cache because an `Exe` address can be reused). Room =
  `bytes_limit - bytes_in_use - (submitted - allocated) - 64 MiB`, with
  `submitted` the placement bytes of every published submission and
  `allocated` a per-device counter the direct backend bumps once per
  tensor in `Item.initTransfer`. It retires the FIFO head (a bulk `load`
  too) until `pending + inputs + execution <= room` on every device,
  always admitting one (warning once when it alone exceeds the room).
  Without `memory_supported` (direct backend and every device reporting
  `bytes_limit`, decided at init) every pending submission is retired
  first. Placement bytes use `shape.packedShape()` like the backend, so
  `submitted` and `allocated` cancel exactly once everything landed.
- `load` is never gated: a transformed tensor counts as delivered once a
  `loadExecute` naming it was published, so the bulk queues behind the
  packs without a drain; a missing one still fails with
  `TransformedTensorNotDelivered`, logged at `warn` (the test runner
  counts logged errors as failures).
- The playground submits every pack, then the bulk, then one `awaitAll`,
  and gains `ZML_GPU_MEMORY_FRACTION` (the BFC `memory_fraction`, to
  shrink the pool and watch admission). The deinit line logs
  `submissions`, `execute_admission_retires`, `memory_supported`,
  `min_room_seen`, `reserve`.

Verification (275 tests pass):

- B70 CPU (Qwen3.5-4B on four CPU devices, two packs of 16):
  `memory_supported=false`, one retire before the second pack
  (`execution=3.43 GiB`, the CPU temporaries counted), checks ok. Warm,
  with packs 3.58 / 3.73 s, without 1.03 / 1.00 s; the bulk now reads
  during the last pack's 1.2 s execution (2.1 s instead of 1.0, CPU
  contention), still beating the serial 3.99 s (cold, same morning).
- gb300-2 GPU 1 (GPU 0 busy, load average 6 to 20; Llama warm, 14 packs
  of 16 then the bulk): `min_room_seen` 237.64 GiB, no admission retires,
  checks ok. Loader `elapsed` (excludes calibration):

  | tree | loader elapsed | wall `Loaded weights` |
  |---|---|---|
  | before, window 1 (3) | 0.327-0.346 s | 551-571 ms |
  | before, window 8 (3) | 0.271-0.274 | 497-500 |
  | this tree, packs + bulk (6) | 0.261-0.276, one 0.327 | 364-369 or 575-638 |
  | this tree, bulk only (3) | 0.259-0.264 | 361-365 |

  Packs plus bulk load at bulk-only speed: no drain, and the executes
  overlap the reads. The bimodal wall is DMA calibration, before the
  loader's clock: 96 ms choosing 16 MiB blocks in fast runs, 310 ms
  choosing 8 MiB in slow ones (5 of 9 packed runs, 0 of 3 bulk-only; it
  starts right after the pack executables compile).
- gb300-2, `ZML_GPU_MEMORY_FRACTION=0.08` (`bytes_limit` about 20 GiB for
  15 GiB of weights): room 11.4 GiB at the first retire, 4 retires, no
  OOM, check ok, wall 578 ms: partial overlap as designed.

Follow-ups: llmd `master` to submit every packed tensor, then the bulk,
then one `awaitAll`, compile pack executables in `Model.init`, and
allocate the KV cache after the weights when wanted (the budget measures
either way); post-compile calibration on gb300-2 sometimes measures low
(310 ms, 8 MiB) and costs 200 ms of wall time; the torch2zml howto showed
the old `zml.io.load` (updated in the thirty-eighth pass); if CPU pack
loads ever matter, the 3.8x temporaries are the first thing to fix.

## Fifteenth pass: fixed source width and the review's simplifications (2026-09-10)

A code review asked what could be simplified in `direct_loader.zig`; the
adaptive width machinery was the largest block whose recorded evidence
did not justify it. The user asked to replace it with the best fixed
values the recordings support and to apply the rest of the review.

### The width evidence, and the values chosen

| source, host | widths | result |
|---|---|---|
| local, B70 oneAPI, 32 MiB requests, Llama | 12 / 16 / 24 / 32 / 48 / 64 | 21.3 / 20.7 / 18.9 / 17.3 / 15.1 / 13.2 GiB/s: knee at 12, 24 costs 11% |
| local, B70, 8 MiB requests | 4 / 8 / 12 / 16 / 24 / 32 / 48 | 19.5 / 18.4 / 23.0 / 23.0 / 22.5 / 21.9 / 21.9 GiB/s: 12 and 16 tie |
| local, one MI300X, Llama | fixed 12 / fixed 24 / adaptive (24) | 0.41-0.44 / 0.62-0.67 / 0.90-0.97 s; 16 tasks 0.424, 24 0.476, 32 0.572, 128 0.61-0.68 s |
| local, RTX 5090 host, Llama | adaptive / fixed 12 / fixed 16 | 0.50-0.55 / 0.49-0.50 / 0.46-0.48 s |
| local, gb300-2, DeepSeek (sixth pass) | 8 / 12 / 16 / 24 / 32 / 48 | 36.9 / 44.8 / 46.7 / 48.2 / 48.8 / 47.3 GiB/s: one 8% plateau from 12 |
| local, gb300-2, DeepSeek, adaptive start rung | 12 / 24 / 32 | 3.37 / 3.43 / 3.23 s, ending at 48, 64 and 96 |
| remote, hf:// Qwen3.5-4B, buffered | 12 / 32 / 64 in flight | 37.3 / 20.8 / 20.8 s |
| remote, hf:// Qwen3.5-4B, direct | fixed 32 (blind bootstrap) | 10.9 / 16.3 s against 48 s buffered the same hour |
| remote, hf:// Qwen3.5-9B, direct adaptive | held 24 / 32 | 20.6-23.9 s, `width_ceiling=32` |
| remote, real AWS S3, 16 MiB requests | 24 to 128 | within 0.7%; latency and pinned memory rise with the width |

What the controller added over a good fixed width: locally nothing
outside noise (gb300-2's plateau is 8% wide against 13 to 15% of
within-load spread; the controller settled at 12 in some loads and 24 or
32 in others, 5% above the fixed-32 oracle); remotely it climbed to the
32 its blind bootstrap started at. Its cost: about 900 lines carrying the
subtlest invariants in the tree (generations, admission fences, the busy
clock, the warm-up rule, parking).

Chosen: **16 for local sources**, never more than 5% from any host's
recorded optimum (B70 12 to 16, MI300X 12, RTX 5090 16, GB300 24 to 32;
12 would give up 5% on the RTX 5090 host and 8% on gb300-2), and **32 for
high-latency sources** (hf:// reads 32 and 64 alike, AWS is flat from 24,
and 32 x 32 MiB requests pin 1 GiB). The constants lived in `limits.zig`
(`backend.Options.readWidth` since the thirty-eighth pass).

### The change

- `Loader.Options.read_parallelism: ?usize = null` (null: the profile's
  default) replaces the `Parallelism` union; the playground's
  `ZML_LOAD_READ_PARALLELISM` overrides it.
- The adaptive controller is deleted (`SourceRuntime`, `SourceProbe`,
  `BusyWindowClock`, the parking `WorkerPool`, request generations and
  admission fences, `source_concurrency.zig`; `direct_loader.zig` 5051 ->
  4167 lines). Width control left: `ThrottleWatch` (halve on a throttle
  or timeout, only with a stats side channel), removed in the sixteenth
  pass. Workers spawn once at creation; pre-growth is `width + 1`
  requests.
- Review items: batch items in one `[]Item` allocation; `Plan.Job` holds
  index ranges that `Scheduler.Claim{batch, plan, index}` slices;
  `Planner.Config{device_count, block_size, request_size, alignment}`;
  `retire_events_early` and write-only diagnostics gone; `bytes_loaded`
  counted by the front end in `retire` (the direct summary logs the
  physical `read_bytes`).
- The buffered backend (TPU, neuron, metal; untestable here) takes the
  same widths (16/32, formerly 12/128) but keeps host staging at
  `min(width, 12)` tensors, so its staging bound is unchanged.
- An executable submission is admitted as soon as nothing executable is
  pending, measured or not: a pending bulk load frees nothing, so
  retiring it first only delayed the pack's reads (the CPU peak is the
  same).

Verification:

- B70 CPU (fourteenth-pass fixture): `source_width=16,
  lifecycle_credits=33, workers=17`, pre-grown 264 MiB (528 MiB before),
  checks ok, 3.64 s against 3.58 / 3.73 s.
- gb300-2 GPU 1 (GPU 0 at 98% for another job, load average 10 to 18;
  Llama warm): `source_width=16, lifecycle_credits=25, workers=17`, pinned
  400 MiB mapped and high-water (656 MiB with the former 32-wide set).

  | run | loader elapsed | wall |
  |---|---|---|
  | packs 64 x 16 + bulk (3) | 0.265-0.325 s | 384-422 ms |
  | bulk only (3) | 0.268-0.281 | 375-379 (one 1279) |
  | bulk only, width 12 / 32 | 0.277 / 0.276 | 374 / 380 |

  Parity with the adaptive tree (0.259-0.327 s) on a busier host; widths
  12, 16 and 32 within 2%, as the sixth pass said. The 1279 ms wall was
  calibration choosing 4 MiB blocks on the loaded host (loader 0.268 s).

## Sixteenth pass: simplification review and its checklist (2026-09-11)

Question (user, 2026-09-10): given how the monorepo uses the loader and
the recordings here, what else can be simplified in `zml/io` and
`vfs/direct_io`? A review with adversarial verification kept 23 of 44
candidates (some amended); `PLAN.md` held them as a checklist.

llmd at monorepo `master` still targeted the pre-rework loader
(`f8ddb3e5`): one `Loader` per process, `loadExecute` per packed tensor,
one bulk `load`, one `await`, several stores and sharding sets per loader
(dflash), `bytes_loaded` for the bandwidth log. It never called
`vfs.loadProfile`, so on this branch it would take the local profile
(width 16 on hf://, no direct I/O): the migration must pass
`vfs.loadProfile(model)`.

### Survivors and rejections

- Mechanism changes, each with a device run: tasks 16 to 18 below.
  Surface trims without one: tasks 1 to 15.
- Needing a measurement first: the ReadyTransfer/EventContext merge (a
  gb300-2 DeepSeek pair plus a multi-device read-back), dropping
  `fairOrder` (done, task 20), a `block_size` override that skips the DMA
  screen (a per-target table is refuted: CUDA hosts chose 2 MiB on the
  RTX 5090 host and 16 MiB on gb300-2).
- Rejected, with the decisive reason: batch diagnostics (every field
  feeds a table here); `Options.auto` (used by `zml/testing.zig`); eager
  file opens on the submit task (hf and s3 opens are HEAD round trips);
  `validateExecutableSharding` (guards `device.id` indexing);
  single-binding `loadExecute` (Laguna's per-layer pack coalescing);
  removing either gate (the read gate was the throttle watch's actuator;
  the fourth-pass measurement); direct I/O demotion (the streaming
  fallback and a recorded defect); a per-target DMA block table (above);
  the buffered backend's chunked split (needs a TPU host); the placement,
  ledger and `Item` reshapes (net zero lines); folding `stop` into
  `fail`; inlining `initBuffered`; a `Batch` union for `Submission`.

### Rate limiting moves into the VFS

The review recommended deleting the throttle watch: no recorded run had
fired it (real AWS and hf runs report zero throttles; the S3Proxy has no
fault injection). The user required the opposite: rate limits must be
handled and traffic reduced when they occur. The watch's shortcomings:
it never recovered, ignored the server-named delay for all but one
request, its five-retry budget could fail a load under a sustained
limit, timeouts counted as throttles, its counters were backend-global.

Of three independent designs under adversarial critique, placement in
the VFS survived unanimously: the quota belongs to the backend and its
credentials, shared by the tokenizer and config reads, a second loader
and the buffered backend, none of which the watch covered. An AIMD
permit ladder and additive recovery were rejected: unmeasured policy of
the class the fifteenth pass deleted, and the ladder collapses to one
permit on a burst because permits are released per attempt. Decided with
the user: a hold on the server-named delay with a floor (`Retry-After: 0`
parses to zero and would hot-loop once throttles stop charging
`max_retries`), jittered wakes (`std.Io.Condition` has no timed wait), a
wall-clock throttle budget instead of the per-request retry count,
timeouts never holding, the loader's width an immutable ceiling, the
stats side channel kept for observability only. One governor per backend
instance, keyed by URI authority so a per-authority scope is a local
change later, governing every HTTP request a backend makes. Defaults:
holds capped at 2 minutes, a 5 minute throttle budget per episode, the
floor at the initial retry delay, one 401/403 retried on the HF data GET
after re-resolving the URL (a signed CDN URL can expire during a long
hold), `error.Canceled` propagated through the backend wrappers.
Mechanics: "Current design".

### Implementation log

Each task its own commit, with `zig fmt --check`, `bazel test //zml:test
//vfs:test` and the three example builds.

- Tasks 1 to 15, surface trims: the buffered backend's unread
  `read_parallelism`, the `execute_submissions` counter,
  `Loader.loadBuffer` (only the loader's tests called it), four dead
  TensorStore accessors (no caller in zml, the examples or llmd
  `master`), `Metrics.source_calls`/`transfer_pieces` (a remote source's
  physical counts come from the VFS `batch source` line), derivable
  pipeline fields, `max_host_bytes` (the guard stays as a fixed 16 GiB
  `Workspace.mapped_bytes_ceiling`), dead `BlockPool` fields, the
  admission boundary types (the `.unmeasured` path stays: it is real on
  CPU). Non-obvious points: an event reaches the platform through
  `block.pipeline` only when there is something to destroy (an eager read
  faulted a fixture with an undefined platform); `.local` (8 MiB
  requests) became the `Loader.Options` default, so a caller without a
  VFS profile asks 8 MiB instead of 16 (B70 evidence in "Source request
  size is backend-dependent"; the buffered
  backend reads `read_chunk_size` only for high latency, so TPU, neuron
  and metal are byte-identical); `backend.Config` folded into
  `backend.Options` (with `readWidth()`); the host-memory `dma_map` and
  `pageable` variants merged (same `Pages` payload); DispatchSpans built
  in one pass, `NonContiguousShardPlacement` turned into asserts because
  `Placement.init` divides a sharded axis with `@divExact`
  (`zml/Sharding.zig:1827`) and `appendTransfers` already relies on
  spans tiling `[0, byteSize)`. CPU playground after them: width 16,
  credits 33, workers 17, pre-grown 264 MiB, high-water 252 MiB, checks
  ok, 3.545 s.
- Task 16, lifecycle credits are a constant: after pre-growth `retained
  >= width + dma_stage` always held, so `min(feasible, max(read +
  dma_stage, retained)) == retained` (33 = 17 + 16 on CPU, 25 = 17 + 8 on
  gb300-2). Gates are `read = width` and `request = pool.capacity /
  blocks_per_request`, workers `width + 1`, the width clipped to
  `credits - 1`, `credits < 2` refused as `DmaMappedBudgetExceeded`;
  `RequestGateLimits` is gone. The fourth-pass measurement for credits
  beyond the width (24.3 against 43.8 GiB/s on a GB300) is the gate
  field's doc comment. The throttle watch narrowed only the read gate
  (narrowing credits would not reduce source traffic). gb300-2 GPU 1:
  width, credits, workers and pinned 400 MiB unchanged, loader
  0.271-0.291 s, checks ok.
- Task 17, the pinned pool is sized once in `Sizing.init` and never grows
  (every fifteenth-pass run had `pinned_mapped == high_water`). The width
  is fitted there (`(usable + remaining budget - reserve) /
  blocks_per_request - 1`, clipped to the option); the two arenas stay,
  reserve first, for ROCm's per-node byte balance. A request size that
  is not a multiple of the block size is refused
  (`InvalidDmaLoadConfig`), which makes the credits an exact request
  count. The "allocates nothing once its arenas are attached" test stays
  (the plan wanted it gone; it is the direct assertion that a load maps
  nothing). CPU: two arenas (128 + 136 MiB), 3.675 s; gb300-2: 144 MiB
  pre-grown into 400 MiB retained, loader 0.265-0.268 s. Risk carried:
  the ROCm 41/59 per-node split under sequential growth (`a2a0a9b8`)
  post-dated the last MI300X runs (the seventeenth pass's 8x MI300X loads
  were inside the recorded history).
- Task 18, the scheduler is a FIFO of `*Plan`; a plan leaves it with its
  last claim. Ownership: a plan is queued only with jobs left, so its
  batch still holds a unit; a batch's queued plans are contiguous, so the
  `fail` that may complete it is the last access. Ordering is unchanged
  because a submission publishes every plan and seals on one task. CPU
  3.243 s; gb300-2 loader 0.249-0.259 s (0.265-0.268 before).
- Task 20, the byte-fair job order is gone (about 230 lines), measured
  first with interleaved warm runs on hosts where the DMA pump is not the
  ceiling:

  | fixture | fair order | planning order |
  |---|---|---|
  | four B70, Llama-3.1-8B sharded (4) | 0.658-0.661 s | 0.659-0.663 s |
  | four CPU devices, Qwen3.5-4B sharded (4) | 1.195-1.293 | 1.224-1.291 |
  | four B70, Qwen3.5-4B sharded (3) | 1.224-1.238 | 1.225-1.243 |

  Flat (under 0.5%, inside the spread); jobs keep the planning order,
  which is file order. The device-id bound check the charge loop carried
  stays where plans are built.
- Tasks 23 to 28, the governed request loop: `vfs/request.zig`
  (`Governor`, `perform`, `classifyStatus`, `RequestSpec`), with
  `range_read.zig` keeping only the Range specifics. Every backend request
  goes through it (HF tree, HEAD hops and data GET; S3 and GCS listing,
  HEAD, GET, GCS token requests; HTTP HEAD and GET). Redirect handling
  became explicit (`follow`, `surface`, `forbid`) after the first attempt
  broke the HF tree fetch, which relied on the std client's three-hop
  default. The loader dropped `ThrottleWatch` (`Loader.width` is written
  once); the `batch source` line gained `source_holds` and
  `hold_wait_ms`. Tests: sixteen governor unit tests without a server;
  seven acceptance tests on a real socket against `MockServer`
  (`vfs/mock_server.zig`): concurrent readers throttled at once share one
  hold and honour `Retry-After` without a retry; a 4-per-100 ms window
  with twelve readers completes with holds and no retries; `RateLimit;
  t=0` still holds for the initial delay; a throttled HEAD at open waits
  and retries; a 503 holds an object-store profile but only charges
  retries on plain HTTP; a server that never lets up fails with
  `error.RateLimited` inside its budget; a reader cancelled during a hold
  returns `Canceled`. One end-to-end loader test (2 GETs per 100 ms)
  asserts the bytes, holds > 0, retries == 0, and that holding cost time
  and not memory (pinned high-water <= 4, every credit returned).
  Smoke-tested against real hf://, s3:// and https://. One
  `hf://Qwen/Qwen3.5-4B` sharded CPU load at parity: width 32, 32 MiB
  requests, high-water 1.06 of 1.16 GiB mapped, no retries, throttles or
  holds, 11.6 s (the fourteenth-pass direct arm: 10.9 and 16.3 s; hf://
  varies twofold day to day). `buffered_loader.zig` is untouched, but its
  reads gain the hold.

Open decisions left for the user: the TPU, neuron and metal sign-off;
revisiting `max_hold` once the pipeline's cancellation points allow
`worker_group.cancel` at teardown; when to key the governor per authority
(one generic `HTTP` instance serves every host of its scheme and one `S3`
spans buckets, so per-instance over-reaches there).

## Seventeenth pass: validating the DMA calibration rework (2026-09-13)

`61641ff5..f759af76` reworked calibration: one shared
`AsyncHostToDeviceTransferManager` sliced per candidate instead of one
per slot per candidate, `warmupDeviceAllocators` moved into
`Loader.init`, `BlockPool.init` taking the workspace by value,
`validateOptions` replaced by assertions. Question: does calibration
still measure and select what it did? `99cff7bf` (before the four
calibration commits) and `f759af76`, built side by side on an idle
gb300-2, run interleaved.

| `io dma-bench`, five runs | block chosen | measured GiB/s |
| --- | --- | --- |
| gb300-2 `99cff7bf` | 16 MiB x4, 8 MiB x1 | 175.3-185.3 |
| gb300-2 `f759af76` | 16 MiB x5 (10/10 across the session) | 178.0-184.3 |
| mi300 `f759af76` | 16 MiB x5 | 43.6-45.3 (calibration 0.39-1.25 s) |

Both inside the recorded per-device H2D history (GB300 176-184, MI300X
44-47), so sharing one manager across eight in-flight transfers does not
serialize them. The one 8 MiB pick came from the old tree; the
smallest-near-peak rule stays the noisy part.

| warm, replicated on four GB300, loader `elapsed` | `99cff7bf` | `f759af76` |
| --- | --- | --- |
| Llama-3.1-8B-Instruct (14.96 GiB, 291 tensors), 4 rounds | 0.291-0.295 s | 0.288-0.293 s |
| DeepSeek-V4-Flash-NVFP4 (156.75 GiB, 135,235 tensors), 3 rounds | 6.648-6.717 | 6.002-6.537 |

Every round chose 16 MiB; parity or a little ahead. mi300 (8x MI300X,
replicated, `f759af76`): Llama cold 4.53 s (3.30 GiB/s; `cat` of the
files takes 6.2 s, the checkpoint is on the volume's slow extent), warm
0.98-1.075 s, inside the 0.84-1.10 s history; DeepSeek-V4-Flash 18.16 s,
storage-bound. `ZML_LOAD_CHECK` ok on both hosts.

Three defects, none in what calibration measures:

- The `dma_bench` line moved from `info` to `debug` on scope `zml/io`,
  which `examples/io` does not enable: the line this file says to check
  first is invisible in the shipped example (still `debug`; see Open
  work).
- With `validateOptions` turned into asserts, the example's environment
  values panic under `release_safe` (`ZML_DMA_BENCH_BLOCK_MIB=64`,
  `ZML_DMA_BENCH_BLOCK_PARALLELISM=64` or `0`) where they returned
  `InvalidDmaBenchmarkOptions` or `NoFeasibleDmaBenchmarkTuple`; under
  `release_fast` they would spin in `runWindow` with zero workers or
  index an empty `candidates`.
- `docs/learn/loader.md` listed `UnsupportedPlatform`, which nothing
  returned (`Workspace.init` hit `unreachable`, `host_memory.Backend.init`
  panicked for tpu/neuron/metal).

The last two were fixed in the thirty-eighth pass (`validate` returns a
logged `InvalidOptions`; `measure` returns `UnsupportedPlatform`).

## Eighteenth pass: two-GB300 llmd loads, DMA depth, read width, placement (2026-09-19)

Fixture: monorepo llmd loading DeepSeek-V4.1-Flash (278.12 GiB logical:
160 packed expert tensors through `loadExecute`, then the bulk), two
GB300 (`CUDA_VISIBLE_DEVICES=0,1`, both on host node 0) under `numactl
--cpunodebind=0 --membind=0`, warm page cache,
`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=200`, model bodies stubbed (pack
programs real), so every number is the `Loaded weights` time. A stray
100%-CPU process made the same arm ~1 s slower than the day before:
compare same-day interleaved rounds only. DMA block 8 MiB, which
calibration also picked every time (179 GiB/s at depth 8).

### Where the 16 s went (llmd side, 2026-09-18)

`loadInto` spent 7.2 s compiling four `llmd_pack_experts` programs one
after another with nothing transferring, 8.3 s submitting packs, 0.7 s
draining. Both causes in `llmd/weights.zig`:

- `packStack` concatenated 384 replicated per-expert parameters and
  pinned only the result to `.expert = .experts`. With operands of size 1
  on the sharded axis, the partitioner sharded each on a row axis and
  resharded with one all-to-all per expert: 2310 all-to-alls per module,
  184,800 `ncclDevKernel_SendRecv` launches (5.7 s, 71% of kernel time).
  A `sharding_constraint` pinning the stacked member replicated gives
  `dynamic-slice(concatenate, partition_id)` with no collectives: compile
  7.2 -> 2.8 s, pack GPU time 5.6 s -> 15 ms.
- Compiling the programs with `io.async` at discovery and awaiting at
  first use overlaps them (XLA uses ~1.2 cores per compile).

16.23-16.48 s -> 7.01-7.10 s (17.0 -> 39.4 GiB/s).

### PJRT drives every H2D copy of a device through one stream

nsys: all 96,473 H2D copies of GPU 0 are on one stream, all of GPU 1's
on another. `LocalDeviceState` creates one `host_to_device_stream_` and
`CopyRawHostToDeviceAndReturnEvent` always takes it
(`xla/pjrt/se/se_raw_buffer.cc`). So `block_parallelism` is a queue
depth, never parallel copy engines (the GB300's
`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 3` is unreachable through
PJRT). The CUDA plugin's device attributes describe neither the host
link nor the copy engines (`memory_bandwidth` is HBM); the only hint for
the depth is the rate the block screen measures.

Before the fixes below, 6.5% of copies started on an empty queue, and
after an underrun the next issue came p50 0.11 / p99 0.97 ms later
against 0.35 ms of budgeted work. That looked like a starved issue path,
but the sweep (load seconds) says the reads are what is late:

| pool placement | depth 8 | depth 16 | depth 32 | depth 32, width 32 |
| --- | --- | --- | --- | --- |
| interleave 0,1 (default) | 8.09-8.13 | 8.00-8.03 | 7.95-8.18 | 7.86-7.93 |
| node 0 (policy, below) | 5.96-5.97 | - | 5.77-5.88 | 5.84-5.89 |

Depth is worth nothing on an interleaved pool and about 2% on a
node-local one. A synthetic screen would not find even that: calibration
reaches 179 GiB/s at depth 8 while the load ran the same stream at
60-68 GB/s (91 after placement), 58-71% busy, because the load's issue
latency is the read pipeline's. Keep the fixed 8. A depth that scales
with the link should be derived from the measured rate, not screened
(Little's law, ~1 ms of queued bytes: `clamp(rate * 1 ms / block, 8,
32)`, 8 on MI300X and 23 on GB300; no calibration time, no new state).

### More host-to-device streams do not help: one stream already saturates the link

Tried in the XLA fork (`~/github/openxla/xla`, uncommitted; `bazel build
--config=cuda_remote_arm64 --config=baseline_arm64 //zml/cuda:archive`,
~40 min remote): `GetHostToDeviceStream()` round-robins raw copies over
`XLA_PJRT_GPU_HOST_TO_DEVICE_STREAMS` streams (default 1; execute-path
users unchanged). Wired into zml by pointing `_PLUGINS["arm64"]` at the
local `archive.tar.zst` with its sha256 and `strip_prefix = "lib"` (the
fork's tar puts the plugin under `lib/`).

With 3 streams the copies split evenly, yet CUPTI shows an overlap factor
of 1.00 (no two copies of a device ever run together) and the same
91.9 GB/s over busy time. Microbenchmark: one GPU pulling 8 MiB blocks
from node 0 reaches 195.0 GB/s on one stream and 194.5 on three; two
GPUs 327 (one stream each) against 367 GB/s (three each). Pinned H2D on
GB300 is serialized per context and one stream saturates the C2C link.
Loads, node-local, streams 1 / 3: depth 8 5.96-5.98 / 6.05-6.09 s, depth
24 5.86-6.10 / 5.86-5.90 s. Noise; not worth a plugin patch.

### Read width: the plateau is flat, the requests were the problem

Width 24 and 32 at node-local placement: 5.93-6.08 s against 5.96-5.97 s
at 16. What the width evidence hid: llmd submitted each packed tensor as
its own `loadExecute`. The two per-layer scale packs are 384 sources of
360 KiB (`experts.N.{w1,w2,w3}.scale`, contiguous in the shard): 39% of
the load's 78,081 reads for 5.7% of the bytes, each scale pack a solo
phase of 384 latency-bound requests (22 + 9 ms per layer with the H2D
stream idle, 1.49 s in total). Submitting a layer's four packs as one
`loadExecute` (bindings grouped up to 8 GiB of outputs) lets the planner
coalesce across bindings: `coalescing_ratio` 1.00/1.99 -> 2.67, reads
78,081 -> 35,688 all of 8 MiB, H2D busy 61% -> 71%, 8.05-8.13 ->
7.45-7.53 s. Nothing to change in the loader: multi-binding submissions
are what `loadExecute` is for.

### Read-path microbenchmark: registration and iovecs are free, the interleave is not

`preadv` of 8 MiB requests from the warm page cache (90% of it on node 1)
into a 256 MiB-per-thread rotating destination, 16 threads on node 0,
optionally one DMA thread per GPU copying 8 MiB blocks out of the same
pool at depth 8 on one stream, like PJRT. Without DMA, anonymous and
`cuMemHostRegister`ed destinations read alike (105.5 / 107.2 GB/s on
node 0, 80.7 / 79.5 interleaved), as do 8 or 64 iovecs (79.3 / 80.4);
`MADV_HUGEPAGE` is inert on this 64K-page kernel. With DMA (registered):

| placement | readers | DMA | read GB/s (per thread) | DMA GB/s |
| --- | --- | --- | --- | --- |
| node 0 / node 1 / interleave | 1 thread | GPU 0 | 10.3 / 6.0 / 7.8 | **193 / 54 / 169** |
| interleave | 16 | GPU 0+1 | **40.9 (3.2)** | 208 |
| node 0 | 16 | GPU 0+1 | **77.2 (5.6)** | 211 |
| node 0 | 24 / 32 | GPU 0+1 | 101 / 100 | 206 / 188 |
| node 0, DMA depth 32 or 16 MiB requests | 16 | GPU 0+1 | 77 / 80 | 215 / 207 |

The interleave costs once the GPUs pull from the pool: with both GPUs on
node 0 DMA-ing from a pool half on node 1, the readers fall to
40.9 GB/s, exactly the loader's read rate at width 16, and remote H2D
runs at 54 GB/s against 193 local (the seventh pass measured 110 vs 176
with an idle CPU side). Node-local keeps the readers at 77 GB/s while
the two links still take 211 GB/s.

### The change: the pool follows the task's memory policy

`host_memory.Workspace.init` masks the interleave nodes with the task's
`get_mempolicy` node set (`taskMemoryPolicyMask`): under `numactl
--membind=0` that is node 0 alone, so the pool is left to the kernel
(`placement=unplaced`); without a policy nothing changes. An arena
`mbind`-interleaved across sockets overrides `set_mempolicy`, which
silently defeated the user's `--membind`. 7.45-7.53 s -> **5.84-5.99 s**
(37.1 -> 47.6 GiB/s), and the 256 MiB arena registers in 3.6 ms instead
of 77.

Not a contradiction of the seventh pass: there the readers were unbound
and the GPUs sat on both sockets, so interleave was the knowledge-free
middle; here the user told the kernel where everything lives. Rule:
interleave by default, never outside the task's policy.

After the change the H2D stream copies at 92 GB/s while busy but is busy
only 58%. The load is read-bound at 53 GB/s of unique bytes against the
77 GB/s the same 16 readers reach in the microbenchmark: about 30% lost
inside the read pipeline (attributed in the nineteenth pass). The other
structural cost: every packed expert is a replicated parameter of the
pack program, so each GPU receives the full 298.6 GB (2x amplification).

## Nineteenth pass: where the last 30% goes (2026-09-19)

Same fixture as the eighteenth pass, on `9f4dc2dc` (pool follows the
task's memory policy). Baseline 5.85-6.05 s with 8 MiB blocks; the block
screen sometimes picks 16 MiB (5.75-5.98 s, 219,474 pieces instead of
254,834), so arms are compared at the same block size. Tools: the
loader's debug metrics, a temporary 1 ms sampler of the gates and pump
queues, nsys traces, the XLA fork with per-stage timers and environment
switches (uncommitted), and two microbenchmarks (`dmabench.c`: H2D
completion paths on the bare driver; `bench.c`: page-cache readers
beside two GPUs' DMA).

### The reads are not the problem; the readers are starved

Per 8 MiB read: 1.57 ms = 5.3 GB/s per thread, the same as a bare
`preadv` thread under the same DMA load; the 475 GiB directory is
page-cache resident. Readers average 11 of 16 busy because each waits
0.65-0.70 ms for a lifecycle credit; the 33 credits are held by blocks
in the DMA stage, 2.1 ms per block for ~0.13 ms of hardware time. The
pipeline alternates between every credit taken, 0-3 reads running and
250-700 pieces queued per device, and 16 reads running with 20 pieces
queued. The eighteenth pass's "30%" is the DMA stage holding the credits.

### Stall 1: allocation sync points behind the pack executes (~1 s)

The GPU client builds every `LocalDeviceState` with
`kComputeSynchronized`: a buffer allocated at time t records the compute
stream's next sync point, and `WaitForAllocation` makes its first copy
wait until the compute stream passes it (`tracked_device_buffer.cc`,
`AllocatedRawSEDeviceMemory`). The loader allocates tensors lazily on
its workers while `retire` enqueues pack executables, so every tensor
allocated during an execute's ~10 ms queues its copies behind it, and
the in-order H2D stream stops. In the trace 49 of 50 H2D gaps > 3 ms end
within 0.3 ms of a graph's end; stalls total 0.6-0.7 s per load (62-66
of 5-22 ms, one after each retirement), with credits at 33 and readers
at 0 meanwhile.

| arm (executes on unless noted) | load | stalls | readers idle |
|---|---|---|---|
| default | 5.87-6.01 s | 690-709 ms | 935-977 ms |
| no executes (`retire` skips them) | 4.92-4.95 s | 0 | 125-127 ms |
| plugin `XLA_PJRT_GPU_ALLOCATION_MODEL=async` (`kAsynchronous`) | 4.89-4.94 s | 0 | 123-133 ms |

The bare switch is not safe on the GPU client. The asynchronous model's
contract (`local_device_state.h`) is that memory stays alive until the
device work using it completed, which TPU's allocator guarantees and the
GPU BFC allocator does not: `RunGpuAsync` frees an execution's
temporaries and dead outputs right after enqueue, and `ExecuteHelper`
holds inputs and results until the definition event only under
`kSynchronous`. Without the sync point a tensor the loader allocates
during a pack execute can get the program's temporary and be written by
DMA while the kernel runs. Fork commit 145c8a1693 (`[PJRT:GPU] Opt-in
asynchronous allocation model`, default unchanged) therefore also defers
those releases to the definition event (temporaries and dead outputs
in `se_to_be_released`; the non-compute-synchronized `ExecuteHelper`
branch keeps inputs, results and that memory until the event fires).
Verified by hashing every loaded shard on the host (2,748 shards,
308 GB): identical under both models. Stock completion path, host load
average ~49: async 5.08-5.14 s, default 5.85-5.96 s; async with polled
completions 4.89-4.94 s.

Tradeoffs: memory returns to the allocator when the host observes
completion rather than at enqueue, so peak use grows by the in-flight
executions' temporaries, inputs and dead outputs, and an allocator query
right after an execute (the loader's admission) sees less room; a
`Delete` during an execution is no longer an immediate free;
`StallStreamOnError` frees instead of stalling (error path only); the
transfer manager's `CanBufferBeAccessedNow` DCHECKs run in debug builds.
A finer XLA fix would attach the sync point only to chunks freed after
the last sync point. On the ZML side the same effect needs no
allocations during executes, which lazy per-worker allocation cannot
promise.

### Stall 2: tiny pieces drain at ~60k pieces/s per device (0.3-1.2 s)

Expert scales are 256-512 KiB pieces: 38% of the copies, 5.9% of the
bytes. In scale-heavy stretches the pump holds 64 pieces (21 MiB) in
flight against the 64 MiB budget, the queue grows to 700 pieces, and
pieces complete at ~60k/s per device (~16 us each: PJRT's per-device
callback thread runs its event bookkeeping, 8 us, then ZML's callback
and pump, 5-7 us, serially) = 20 GB/s with the readers idle: 0.9-1.2 s
per load with the allocation stalls, 0.3-0.4 s without. Raising
`max_dma_pieces_per_device` 64 -> 512 changed nothing (piece latency
0.9 -> 2.5 ms, same rate): the serial completion path is the bound.
Fewer pieces (a scale blob per layer in the pack recipe) is the lever;
the loader cannot merge pieces that land in different device buffers.

### The completion path, measured stage by stage

Per piece, stock plugin: ZML submit -> callback 1.1-1.2 ms; hardware
p50 133 us. Fork stage timers (poll+inline arm): event wait 720-800 us
(stream queue plus the stalls), callback-thread delay 170-190 us,
everything else under 10 us. The stock path adds `cuLaunchHostFunc`
blocking 445 us on average (p90 1,489): the driver serializes
host-function launches at ~21 us each with ~21 callers blocked (360
unbounded work-queue threads), each transfer bounces through that queue,
and the event pool creates events at 430 us each under the lock convoy.
On the bare driver (`dmabench`) the same host-function pattern does
62-68k pieces/s per GPU at 2 MiB, polling 74k/s: the convoy serializes,
not the mechanism.

| fork switch (8 MiB blocks, executes on) | piece latency | load |
|---|---|---|
| stock | 1.14-1.18 ms | 5.87-6.05 s |
| `XLA_PJRT_GPU_POLL_CALLBACKS=1` (event polling, no host functions) | 1.00-1.05 ms | 5.85-5.92 s |
| + `XLA_PJRT_GPU_INLINE_H2D=1` (copy issued on the caller's thread) | 0.88-0.93 ms | 5.86-5.95 s |
| + `XLA_PJRT_GPU_H2D_CALLBACK_THREAD=1` (own H2D callback worker) | 0.88-0.99 ms | 5.84-5.92 s |
| `ALLOCATION_MODEL=async` alone | - | 5.23 s |
| `ALLOCATION_MODEL=async` + poll + inline | 0.37-0.46 ms | 4.89-4.94 s |

No completion-path change moves the load while the allocation stalls
dominate; without them polling+inline is worth 0.3 s. Not worth carrying
alone.

### Contention, the window and its edges

- During a load a copy takes 2 us + bytes / 100 GB/s (p50 17 us for
  1.38 MiB) against 155-193 GB/s idle: the readers' writes into the
  node-0 pool and their remote page-cache reads share node 0's memory
  with both GPUs' DMA (`bench.c`: 16 readers at 78 GB/s beside two GPUs
  at 230 GB/s). The stream idles 29% of the steady state in 26-70 us gaps
  with nothing queued. Stream time per device: 3.26 s for 279.4 GiB.
- 0.85 s before the first submission: `loadPacked` awaits the first pack
  program's compile. Submitting the plain tensors first fills only
  0.15 s of it (their 9.17 GiB read in 0.2 s); a compilation cache for
  the pack programs is the fix.
- Active window 4.70 s by default, 3.78-3.82 s without the stalls =
  78.6 GB/s = the 16-reader ceiling beside two GPUs' DMA (bench 77-78).
  Width 24/32 under the async model: 4.89-4.93 s, no gain.
- No execute tail: reads and DMA stay active until 10-50 ms before
  `Loaded weights`.

Accounting for the 5.9 s: 0.85 s compile wait + 4.7 s window, of which
~1 s is allocation stalls and their knock-on, the rest the read ceiling.
With the async allocation model: 0.85 + 3.8 + 0.25 = 4.9 s.

### What to do

1. Plugin: fork commit 145c8a1693 (`XLA_PJRT_GPU_ALLOCATION_MODEL=async`
   with the deferred releases): 5.9 -> 5.1 s stock, 4.9 s with polled
   completions, hashes identical. Decide whether llmd sets it (memory
   returns later) or the plugin defaults to it.
2. llmd: cache compiled pack programs, or otherwise fill the 0.85 s.
3. llmd/loader: a scale blob per layer in the pack recipe to cut the
   256-512 KiB pieces (0.3 s without stalls, more with).
4. Keep the completion path until 1 lands; then polling+inline copies
   are worth 0.3 s as a small plugin patch.

Not worth doing (measured): more H2D streams, a larger piece cap, a
dedicated H2D callback thread, wider reads.

## Twentieth pass: merge with master (2026-10-14)

Both checkouts are jj merges of `brabier/adaptive-concurrency` with
`master`, resolved in the working-copy commit: zml `553690e8` with master
`fe25f16b` (#935), base `550f2bbe`, 21 conflicted files; monorepo
`6011a514` with `c40c5b92` (zig 0.17, #646), base `3c6f9944`, 8. Master
had moved ~100 commits in zml (zig 0.17 #923, sharding in the tensor's
shape #868, loader memory kinds #815, Exe eval/Runner #896/#897,
`HostAccessible` #899, `PjrtBufferMetadata` #935, furiosa #871, device
NUMA node #891, `Platform.io_impl` #841, hf snapshots #872), ~90 in llmd.

### Merge policy and resolutions

- Master's API wins outside the loader and VFS (sharding/mesh, `Tensor`,
  `Exe`, `Buffer`, `Platform`, zig 0.17 idioms). The branch's loader
  (`zml/io/*.zig`) and VFS win, with master's loader-facing features
  ported in. `zml/io.zig` stays a thin facade: master's monolithic loader,
  its writers (`MemoryWriter`, `ProgressWriter`, `DirectShardWriter`) and
  the `mem.DmaAllocator` family are not re-exported.
- Loader API: `Loader.init(allocator, io, platform, opts)`, `load(Model,
  &model, &buffers, store, progress)`, `loadExecute(store, bindings,
  progress)`, `awaitAll()`, `bytesLoaded()`. `load` lost `shardings`:
  placement comes from each tensor's shape (`Sharding.Placement.init`; a
  null mesh is `platform.replicated_mesh`); `loadExecute` uses
  `exe.input_shapes` / `exe.output_shapes`. `TensorStore` is master's API
  (`fromRegistry`, `createTensor(subkey, tags, mesh, partitioning)`,
  `createReplicatedTensor`, `createHostPinnedTensor`, `Binding{tensors,
  transformed, memory}`) in the branch's `zml/io/TensorStore.zig`.
- Master #815 host-pinned bindings: the branch loader delivers into
  `device.memory(.host_pinned)` ("option a": pinned block pool + a PJRT
  transfer manager on that memory), excluded from HBM admission and
  device byte counters. Works on CPU (master panicked); tpu, neuron,
  metal, furiosa return `error.UnsupportedMemoryKind`. `loadExecute`
  inputs still land in `.default`.
- Ported with tests: #894 (transfer-manager init errors), #896 (donated
  and non-donated inputs freed once), VFS writes for local paths and
  EINTR-safe `fcntl` (`vfs/file.zig` stays deleted). `numaNode() ?u32`
  and `pjrt.Device.attributes()` are master's.
- Branch strictness kept: a model tensor with no binding is
  `error.TensorNotFound` (master skipped it).
- `//examples/io:playground` is `//examples/io`; the root `bench_*.sh` /
  `profile_file.sh` still say `playground` (stale).
- monorepo: master's deleted `*dflash*.zig` stay deleted. `main.zig` is
  master's with `bff6c633` re-applied (early `Loader.init` in
  `io.concurrent`, `vfs.loadProfile`, `registerBackend`). `weights.zig` is
  master's recipes with `6011a514` (pack programs compiled with `io.async`
  at discovery; packs as grouped `loadExecute`s of up to 8 GiB of
  outputs); the branch's per-member `replicated()` pin is dropped since
  master #566 pins the stacked tensor replicated for the same all-to-all
  reason. `aacae994` ("load only deepseek" stubs, a measurement hack of
  the eighteenth/nineteenth passes) is not in the merge.
- Master's engram path and #634's `streamConcatenatedRows` used old
  loader internals. A stopgap (one CPU reader per table into
  `pinned_host`, device-memory `concatenate_rows` back on a program, a
  regression against #634) was replaced the same day by the twenty-first
  pass.

### Verification and review

- B70 host, CPU build: zml `bazel test //...` 26/26 (`//zml:test` 300
  passed, 12 GPU-only skips); cuda, rocm and oneapi builds type-check.
  monorepo `--config=localzml //llmd/...` builds, tests 6/6, cuda builds.
- Review: 6 low findings, all fixed. `Buffer.Shard.toHost` leaked the
  PJRT event master #854 destroys elsewhere. The zig 0.17 rewrite of
  `errdefer |err|` had narrowed a `direct_loader` sizing log to arena
  growth; a `BlockPool.init` failure is logged again. `loadExecute` compared output
  shapes by dims and dtype only; `validateOutputPlacement` now compares
  device ids and per-device slices (not mesh pointers, so a null-mesh
  replicated tensor matches `replicated_mesh`), mutation-checked. The
  host-pinned test passed vacuously on CPU (one memory space); it skips
  only where both kinds are the same PJRT memory. The stopgap missed
  `bytesLoaded()`. Rejected: the VMM allocator does report `bytes_limit`
  (admission stays on); `Device.attributes()` panicking on a null state is
  master's contract; the examples/io BFC-preallocation change is
  deliberate.

### Runtime check on the local 4x B70 (oneAPI), 2026-10-14

examples/io, warm, interleaved, batch elapsed ("Loaded weights" includes
`Loader.init` and its ~0.73 s calibration): Llama-3.1-8B sharded
0.676-0.678 s on the branch tip, 0.684-0.687 s merged; replicated
1.255-1.256 / 1.263-1.279 s (1.251-1.254 at 8 MiB). Calibration picked 8
MiB (49.3 GiB/s) on the tip, 2 MiB (50.4 GiB/s) merged. `ZML_LOAD_CHECK=1`
ok 291/291. Within ~1.2% of the 0.636-0.676 s anchor; replicated ~9%
above the old 1.147 s anchor in both trees (host drift). The tip's
`oneapi.zig` asks for `libzml_oneapi.so`, which its packaging does not
ship, and silently falls back to CPU; the merged tree loads
`libpjrt_oneapi.so`.

- Host-pinned delivery runs on Level Zero: every shard in `pinned_host`,
  distinct from `device`; the ledger counts only device tensors.
- Merge bug found only at run time: master's VFS routed
  `file_write_streaming` through its handle table, whose index 0 was the
  CWD, so `stdout()` (fd 1) wrote to fd 0 (`examples/llm` `WriteFailed`
  when stdin was not a tty). Fixed in `vfs/vfs.zig` (std streams at
  indices 0-2), with a test.
- direct_loader planner/scheduler/dispatch tests picked the B70s through
  `Platform.auto`; now `Platform.init(.cpu)` (pre-existing).
- `//zml:test --//platforms:oneapi=true` aborts in master's moe "Triton
  compatibility matrix" test (Level Zero cannot create
  `per_token_group_quant_fp8`); skipped: 301 pass, 11 skip.
- llmd with master's oneAPI plugin `202610011939` segfaults compiling
  Llama-3.1-8B (oneDNN `gemmstone::Generator<Core 8>::gemmAccessC` under
  `xla::gpu::SyclGemmWorkspacePass`), before any load; not loader-related
  (the 2026-09 llmd run here used the August plugin). `examples/llm`
  compiles and generates.
- Leftover `TMP loader created_at raw_ns` debug log (removed in the
  forty-ninth pass).

### Attribution: master's oneAPI plugin serializes piece completions

`examples/llm` Llama-3.1-8B on four B70 (column-sharded, ~1.56M DMA
pieces in both trees). Code x oneAPI pins, three interleaved rounds:

| arm | Loaded weights | `dma_piece_latency_ms` |
| --- | --- | --- |
| merged code + master pins | 20.6-21.9 s | 3.33-3.53 |
| merged code + branch pins | 7.4-7.6 s | 1.18-1.21 |
| branch code + branch pins | 7.4-7.6 s | 1.18-1.20 |
| branch code + master pins | 21.0-22.3 s | 3.38-3.60 |

Swapping single libraries into the merged runfiles: the old plugin alone
restores 7.5-7.6 s; the old SYCL/UR runtime or the old Level Zero + IGC
alone stay at ~21-22 s. The plugin binary
(`mirror.zml.ai/plugins/202610011939.137.1.0d59bb389ee2` against
`pjrt-artifacts manual-2026-08-17`) is necessary and sufficient; the
merge's code is not involved (same BFC options; `io_impl` only reaches
`Compiler.zig`).

Mechanism: both plugins issue each piece as an async H2D memcpy plus a
SYCL event. The old plugin completes pieces on the four per-device
`py_xla_callback` threads (~85% CPU each, in parallel). The new one runs
every completion through a SYCL `host_task`: one `ThreadPool` worker at
~100% in `DispatchHostTask -> NotifyHostTaskCompletion -> cleanupCommands
-> pthread_rwlock_trywrlock` on the scheduler's global lock, ~13 us per
piece, ~75k pieces/s for all four devices. `SYCL_QUEUE_THREAD_POOL_SIZE`
4/8 (24.3/23.5 s) and DMA depth 16 (23.1 s) do not help; the 2 MiB
calibration pick is a symptom (forced 8 MiB: 22.5 s). Big-piece loads are
unaffected.

Levers: (1) the plugin should complete events on per-device threads again
(raise with whoever builds the mirror plugins; the August plugin with the
new runtime worked by library swap, untried as a bazel build). (2) Pack
each device's strided shard contiguously in the pinned block so a block
is one transfer per device (done in the twenty-fifth pass).

## Twenty-first pass: composite destinations, engram resident in host memory (2026-10-14)

Question: does llmd (deepseek41) load the engram tables into host memory
and keep them there? Monorepo master did, on CUDA, with model-local code
on old loader internals (`MemoryWriter(.host_pinned)` plus
`splatByteAll(0)` padding, then `swapEngramHostPinnedBuffers`), which the
merge had replaced with a slow stopgap.

### Engram facts (from the code; no checkpoint on this host)

- Engram layers 1 and 14 of DeepSeek-V4.1. A row is 256 B of fp8
  (`weight [row, channel]`) plus 8 B of ue8m0 scales (`scale [row,
  group]`). Rows per layer are the sum of 24 primes >= 16M: 384,006,168
  and 384,016,682 (padded to 384,016,688), ~94.4 GiB per layer, **~189
  GiB in total**; per device with `.row = .model`, 94.4 GiB at TP=2, 23.6
  at TP=8. Rows padded to a multiple of 8 (#555), row-sharded (#502) over
  `.tp_mesh`, `.host_pinned`. The 278 GiB fixture of the
  eighteenth/nineteenth passes had no engram.
- "Keep it there" (master #490, #645): each loaded shard is wrapped by
  `Buffer.createView()` into a device-kind PJRT view of the pinned
  pointer; owners stay in `Model.engram_host_buffers`. The GPU gathers
  through the mapping (zero copy). Declaring the tables `pinned_host` to
  XLA would make `toMemory` copy the whole table to HBM every step.
  `engramHostPinnedBytes` is passed as `extra_device_memory_bytes` so XLA
  does not think HBM is full; KV-cache sizing subtracts it. qwen4 does the
  same for its offloaded n-grams.
- `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` must cover the engram bytes plus
  other pinned use; nothing in zml checks it.

### The change: composite destinations

A binding may place several sources into ONE destination, zero-filling
uncovered bytes, in any supported memory kind, through plain
`Loader.load`. The API (`Binding.layout`, `Layout{ parts, fill }`,
`Part{ src_offset, src_pitch, dst_offset, dst_pitch, width, rows }` with
only contiguous parts accepted, else `error.UnsupportedLayout`;
`View.createConcatenated` with `.pad_rows_to_multiple`,
`View.createComposite`) was replaced by `Source` in the twenty-fourth
pass, as were the same-dtype and whole-row validation rules.
`TensorStore.setMemory(tensor, kind)` replaced callers poking
`id_to_sources`. What remains:

- Front end: one `LoadSpec` per destination with N `LoadPart`s,
  validated before anything is submitted (sorted, disjoint, in bounds;
  full coverage without fill).
- Direct backend: `Item` split into `Destination` (shape, placement,
  memory, output, LazyOnce transfer state) and `Part` (what the planner
  sorts by file offset). Pieces merge across parts when block, writer
  mask, block offset and destination offset are contiguous. A
  destination's parts may span files and plans; `Target.total` stays the
  placement bytes and the last transfer is decided by bytes, so pieces
  from any part, plan or fill close the buffer in any order.
- Zero fill: sizing pre-grows one block that `detachBlock` takes out of
  the pool for good and zeroes. The plan holding a destination's first
  part gets one fill job with every gap, claimed like a read job (credit,
  batch unit) but with no read gate, pool lease (`Lease.pool` optional) or
  file. The buffered backend zeroes only the gaps.
- Accounting: device bytes once per destination per device; host kinds on
  neither the submitted nor the allocated side; `bytesLoaded()` counts
  shape bytes once, padding included.
- Defect found (pre-existing on the read path): `enqueueBlocks` now
  checks the pipeline's sticky error while holding every pump lock.
  Before, a failure recorded between a worker's last `errorValue()` check
  and the locks let `abortReady` drain the queues first, and anything
  queued afterwards never completed (the batch hung).
- Tests: `zml/io/testing.zig` (new) builds an N-device platform (the
  accelerator when it has >= N devices, else CPU) and reads shards back in
  mesh order; a 2x2 planner replay writes every device byte exactly once.
- llmd: deepseek41's engram tables load in the same `loader.load` as the
  model, same shapes, tags, mesh and memory as master (tested), so
  `swapEngramHostPinnedBuffers` finds them. Same-dtype
  `Packer.concatenateRows` (qwen4 n-grams) is a composite in any memory
  kind, streaming into the final placement as #634 did. The stopgap
  (`streamRowsToHostPinned`, `streamConcatenatedRows`, `markDelivered`,
  `addStreamedBytes`) is deleted; dtype-changing concatenation stays the
  `llmd_concatenate_rows` program.

### On-device results (4x B70, oneAPI, 2026-10-14)

- Tests pass on Level Zero (309 passed, 11 skipped, moe Triton test
  skipped).
- "Keep it there" works on Level Zero. Test "a padded host-pinned table
  stays in host memory and serves device gathers through device views":
  100,003 x 264 B table padded to 8, row-sharded over 4 devices,
  host_pinned, wrapped with `Buffer.createView`, then a 256-row gather per
  device under `manualComputation`. Byte-exact; owners `pinned_host`,
  views `device`, same pointer. Device `bytes_in_use` 256 B before and
  after load + views, 68,864 B after the gather (indices + output); a
  1,000,003-row run (66 MB shards) peaked at 277 KB per device, so XLA
  made no transient copy.
- Host-pinned delivery ("option a"), examples/io with the new
  `ZML_LOAD_MEMORY=host_pinned` knob, warm, batch elapsed: Llama-3.1-8B
  (14.96 GiB) sharded 0.690-0.695 s default (21.7 GiB/s) against
  2.235-2.331 s host_pinned (6.5 GiB/s); LFM2.5-1.2B replicated
  0.187-0.200 / 1.686-1.726 s, sharded 0.102-0.108 / 0.437-0.452 s.
- ~60% of the host-pinned time is the PJRT `pinned_host` allocation in
  `TensorTransfer.init` on the read path: 12.0-12.7 ms per tensor against
  0.013 ms for device memory (~1,918 inits x 12 ms / 17 workers = ~1.35 s
  of 2.3 s). A first 8 GiB `pinned_host` allocation takes 0.61-0.78 s
  (serial; ~1 ms after a free, the plugin caches); memcpy into pinned
  memory 8-9 GiB/s on one thread, 29.5-31 GiB/s on 16. Under memory
  pressure (6 GB free, 4 GB swapped): 14.4 s.
- Default-memory loads unchanged; `ZML_LOAD_CHECK=1` ok 291/291.

### What to do next for host-resident tables

1. Take the pinned allocation off the read path, and 2. option (b): read
   straight into the pinned destination with one allocation shared by
   replicas. Both done in the twenty-second pass.
3. A zml-owned device view released by its delete callback, so llmd drops
   `engram_host_buffers` / `ngram_host_buffers`: done as `.host_mapped`
   in the fifty-first pass (`extra_device_memory_bytes` stays). A
   compiler-side "host-mapped" marker was not done.
4. CUDA verification of the engram load through the loader: DeepSeek with
   engram ran on GB300 in the twenty-ninth pass; view lifetime was
   checked on CUDA in the fifty-first. No load-time comparison against
   master's writer is recorded.

## Twenty-second pass: host-resident delivery without DMA (2026-10-18)

Items 1 and 2 above; zml only, uncommitted; llmd unchanged.

### Where the time went (4x B70, oneAPI, Llama-3.1-8B sharded, warm)

- Baseline batch elapsed: default 0.671-0.675 s, host_pinned
  2.020-2.031 s (`tensor_init_ms_per_read` 10.2-10.3).
- First-time pinned allocation runs at 13.6-15.3 GiB/s whatever the
  thread count (1, 4, 16; 1-64 MiB buffers), 7.4 GiB/s at 1 MiB; reuse
  after a free ~1-10 ms for 8 GiB. strace: nearly all of it is
  `DRM_IOCTL_XE_VM_BIND` on one device fd (4-16 ms per call), the kernel
  populating and binding pages, serialized below the loader.
- The first CPU write into a fresh allocation faults its mapping in:
  memset 9.4 GiB/s on one thread against 21.5 on reused memory, 30.5 on 16
  threads. Allocating 6 GiB takes 0.577 s alone, 0.724-0.726 s while 16
  threads first-touch another 6 GiB: allocation and pinned writes add up
  more than they overlap.
- Allocation ahead with the DMA path kept: 2.03 -> 1.934-1.953 s; the
  host-to-host DMA became the stall (13.3 ms per request).
- Option (b), attributed: allocating every destination 1.151 s, then
  reads into them 0.562-0.571 s (26 GiB/s, faster than DMA into device
  memory); overlapped 1.66 s.

### The change

- Allocation ahead (`Loader.startAllocations`): one worker-group task
  per batch calls `ensureState` on `Batch.host_destinations` in claim
  order (`Planner.hostDestinationsInOrder`), holding one completion unit
  taken before the publish sentinel drops; a worker reaching a destination
  first creates it itself (`LazyOnce`). Metric `ahead_init_ms`.
- Delivery classes (`dma`, `host_direct`, `host_bounce`; plus fills).
  `Planner.JobCut` never mixes DMA and host-resident parts in a job,
  merges jobs back into file order and adds one fill job per class. A host
  job is `host_direct` when its read has no alignment padding and its
  transfers cover the read in order, each byte once
  (`coversReadInOrder`): `HostScatter` reads with iovecs over the
  destination ranges, in as many `readFilePositionalAllV` calls as
  `max_positional_iovecs` needs. Otherwise `host_bounce` (widened direct
  I/O reads, overlapping or duplicate parts): leased blocks, then
  `copyFromBlocks`. Host jobs take no block contexts or events; host fill
  is a memset.
- Host state (`TensorTransfer.initHost`): `createUninitializedBuffer` in
  each device's host space. Replicas share ONE allocation:
  `SharedHostAllocation` owns a non-shard buffer, every shard is a
  `createViewOfDeviceBuffer` whose `on_delete_callback` drops a reference,
  and the last drop destroys the owner (on `smp_allocator`, outliving the
  loader), so shards free in any order. Checked on oneAPI: the callback
  fires, views read back, kernels gather through device-kind views.
  Donating a shared shard to an execution that writes it shows on every
  device (the `HostAccessible.undonatable` rule). The unshared copy path
  (`share_host_allocations`, `sharesHostMemory`) went in the fiftieth
  pass.
- `Batch.fullySubmitted` counts `host_written` (bytes times the devices a
  write lands on) against `host_total`.

### Results (4x B70, oneAPI, warm, interleaved, batch elapsed)

| model, placement | default | host_pinned before | host_pinned now |
| --- | --- | --- | --- |
| Llama-3.1-8B sharded | 0.669-0.672 s | 2.020-2.031 s | 1.626-1.644 s |
| Llama-3.1-8B replicated | 1.256-1.268 s | not runnable (4 x 15 GiB pinned, 62 GiB host) | 1.849-1.874 s |
| LFM2.5-1.2B sharded | 0.103-0.106 s | 0.437-0.452 s | 0.359-0.363 s |
| LFM2.5-1.2B replicated | 0.184-0.192 s | 1.686-1.726 s | 0.361-0.372 s |

- Allocation-bound: Llama's allocations end at +1.41 s, the batch at
  +1.64-1.66 s. Sharded host_pinned near the 21.7 GiB/s ceiling is out of
  reach on this host: allocating 15 GiB pinned takes 1.15 s alone.
- Replicated is 1.13x sharded (one 15 GiB allocation instead of four).
- `ZML_DIRECT_IO=on` (every host job bounces, disk-bound): default
  1.600-1.605 s, host_pinned 2.298-2.320 s.
- `ZML_LOAD_CHECK=1` ok 291/291 sharded, replicated (every replica) and
  direct-I/O host_pinned. `//zml:test` CPU 311/12 skipped, oneAPI 312/11.
- Tests cover class split and file order, columns (8,192 transfers in one
  job), duplicated sources, alignment 0 and 64, every
  `SharedHostAllocation` released in any shard free order, and a
  replicated table read by every device through views. Each run writes
  its own file: with one file an unshared run found the previous run's
  bytes in recycled memory.

### Left open

- The allocation floor: a faster pinned allocation (plugin or driver), or
  allocating while something other than host writes runs (engram before
  the device weights, a "prepare" API). On GB300 it is the 2.4 s engram
  stall of the twenty-ninth pass.
- CUDA: pinned allocation speed and contention (shared allocations and
  their `on_delete` release were checked on GB300 in the fifty-first
  pass).
- One overlap in a host run sends the whole job through blocks.
- `host_bounce` under direct I/O copies twice; reading O_DIRECT straight
  into the destination needs destination and file offsets congruent
  modulo the alignment.
- CPU default memory could take `host_direct`; it keeps the transfer
  managers.

## Twenty-third pass: monorepo re-merge with newer master (2026-10-18)

The user redid the monorepo merge as `rwotnntx` (`jj new master
brabier/adaptive-concurrency`), master `3a8b1dae` (2026-10-15); the
previous resolution survives as `kqxnstry` (`052a7d2d`, on old master
`c40c5b92`). Each conflicted file was 3-way merged (old master base,
`kqxnstry` ours, new master theirs); of master's 14 changed files only
`deepseek41.zig` and `qwen3_5.zig` overlap, and both merged cleanly.
`jj diff --from master` touches exactly `kqxnstry`'s 8 files; `--from
kqxnstry` matches master's delta (114+/189-). Master's zml pin
(`e385d9c2`) is not in the local zml repo (zml merge on master
`cddc9cf6`, #928); `--config=localzml` does not need it. Builds (cpu,
oneapi, cuda) and tests 6/6 on the B70 host; llmd not run; `@` holds the
resolution, undescribed.

## Concatenating tensors in the loader: design (2026-10-14, exploration)

The API and phases were superseded by the Source proposal below; the
inventory, the DeepSeek estimate and the strided-part constraints stand.
Packing is "just moving bytes to the right place" (the user). Pack
recipes in `llmd/weights.zig` by class:

- **A, pure placement**: expert stacks with members concatenated
  (`packStack`, all `.down`), `fuse .copy` with groups=1, same-dtype row
  concat, engram padding, NVFP4 weight stacks (f4 is a relabel),
  broadcast fills (one source, many destinations).
- **B, data-independent permutation**: MXFP4 gate/up row interleave (row
  r of member m -> row 2r+m; ~2/3 of DeepSeek's expert bytes), `fuse`
  with groups > 1, the 128x4 block-scale swizzle
  (`swizzle_block_fp8_scale` #548/#577 with its 128x broadcast).
- **C, compute**: `dequantize_block_fp8` (wo_a #593),
  `dequantize_blocks`, `convert_rows`, dtype-changing concat, NVFP4
  globals, CDNA3 `normalize_block_fp8` (byte-local).

On the DeepSeek-V4.1 fixture every pack program is an expert stack (A or
B). Loader-native stacking removes, by eighteenth/nineteenth pass numbers,
the 0.85 s first-compile wait and 2.8 s of compile CPU, ~0.95 s of H2D stalls
behind pack executes (no-execute arm 4.92-4.95 s against 5.87-6.01 s),
the 2x replication (each GPU receives ~279 GiB, ~144 GiB if each expert
goes only to its owner) and most of the ~97k 256-512 KiB scale pieces.
Estimate 5.9 s -> ~4.0-4.1 s without a plugin patch, maybe ~3.2 s if
halving DMA traffic lifts the read ceiling (unmeasured).

Constraints kept by the later phases:

- Strided parts expand lazily per job, never per row up front (an
  interleave is ~1M chunks per tensor). A job's `preadv` scatters each
  chunk to its destination-ordered position in the pinned blocks (free on
  buffered reads: the eighteenth pass measured 8 vs 64 iovecs at parity;
  1,024 unmeasured), so one DMA piece covers a destination run; O_DIRECT
  with sub-4 KiB chunks needs a bounce copy. This also fixes
  column-sharded tensors cut into one piece per row (twenty-fifth pass).
- Atomic units (an expert's w1+w3, a swizzle tile group) stay inside one
  job; an expert split across two files needs a join or a post-pass.
- llmd's `normalizeBlockFp8` / `maybeSwizzleBlockFp8Scales` read the
  `.fuse`/`.stack` recipe and poke `id_to_sources`; they must learn about
  layouts first.
- Class C hygiene: inputs sharded like the output (`bf16Linear` already
  does), compiled at init, submitted after the bulk load so no execute
  overlaps DMA.

## Concatenation API: Source expressions (2026-10-18, proposal)

The user asked to compare the loader's API with monorepo
`brabier/weights-add-explicit-tags-and-paritins` (`rsqorxts` /
`898fb0b2`, 2026-10-09, on master `13d57911`, +1099/-844; not in master,
which still uses `FusionSpec`), and to improve zml's API from it.
Implemented from the twenty-fourth pass on.

### What exists

- zml: `createConcatenated`, `createComposite` (caller-computed byte
  parts), `setMemory`. One contiguous part per source, same dtype, whole
  rows.
- llmd: `Packer.fusedLinear(store, FusionSpec{ members, groups, source,
  partitioning: BlockPartitioning }, out, k)` records `Recipe.fuse` for
  `packFused`; `splitFused` registers alias names to split pre-fused
  tensors; `experts` records `Recipe.stack`. 28 `fusedLinear` sites in 11
  model files, 10 `experts`, 4 `splitFused`.
- The branch: `linear.View{ store, definition, start, rows }` (a
  checkpoint Linear restricted to a row range; `split` checks every
  component's row grid and registers nothing); `concatenateLinears` and
  `interleaveLinears(.finest | .explicit{ rows_per_group }, ...)` return
  `FusedLinear{ linear, layout }`, whose `extractOutput(y, i)` undoes the
  interleave on the output, freeing the group count from the head
  structure (master's `memberHeads` assumed groups = KV heads).
  Partitioning is keyed by tags (`.{ .dout = .model }`), replacing
  `BlockPartitioning`. `splitTensor` binds slices of non-Linear tensors.
  Still, every assembly runs a pack program, and slices are made by
  editing `safetensors.Tensor.offset`/`.shape` and writing
  `store.id_to_sources` (`bindSources`).

### What llmd needs

Byte-only: fused projections with groups > 1 (qkv by KV head in llama,
qwen3_5, gemma3/4, laguna, ministral3, muse_glimmer, lfm2, qwen4, dflash;
gate/up; qwen3_5/qwen4 GDN `in_proj_qkvz` from three slices of
`in_proj_qkv` plus `in_proj_z`, lfm2 `in_proj` `[B; C; x]` to `b0 c0 x0
b1 ...`) and groups = 1 (deepseek41 `wq_a`+`wkv`, qwen4 `down_inject`,
`key_value_proj`); splits of pre-fused `in_proj_qkv`, `conv1d`, lfm2
`in_proj`; broadcasts (per-tensor FP8 scales to per channel, NVFP4 global
scales over each member's rows); expert stacks, concatenated (stablehlo,
triton/fly without mxfp4, flashinfer, metal, tpu) or row-interleaved
(mxfp4 on cute/triton/fly), scalars `[E, P]`, NVFP4 u8 to 2 x f4e2m1;
engram padding and n-gram concatenation in pinned memory; 128x4
block-scale swizzles (cute_mxfp4, flashinfer NVFP4, Blackwell FP8 with
its 128x row spread). Compute stays a program: dequantize, `convert_rows`
and dtype-changing concat, NVFP4 globals, the CDNA3 fn -> fnuz map.

### Proposal: `TensorStore.Source`, Tensor's shape ops over checkpoint bytes

A `Source` is a value (`store`, a tagged `Shape` without sharding,
`pieces`) with zml.Tensor's op names and conventions. Relabels:
`withTags`, `rename`, `splitAxis`, `merge`, `reshape`, `bitCast`. Byte
moves: `slice` (step 1), `split`, `chunkExact`, `transpose`, `pad` (zero),
`broad` (source stride 0), `concatenate`, `stack`.

```zig
pub fn bind(self: Source, mesh: @EnumLiteral(), partitioning: anytype, opts: BindOptions) Tensor;
pub fn source(self: View, subkey: []const u8, comptime tags: anytype) Source; // maybeSource: optional
// createTensor(k, tags, mesh, p) == source(k, tags).bind(mesh, p, .{})

const table = store.source("weight", .{ .row, .channel }); // engram
const weight = table.pad(0, .{ .row = .{ .high = paddingTo(table.dim(.row), 8) } })
    .bind(.tp_mesh, .{ .row = .model }, .{ .memory = .host_pinned });

fn interleave(comptime T: type, members: []const T, groups: i64) T { // T = Tensor or Source
    var parts: [MAX_MEMBERS]T = undefined;
    for (members, parts[0..members.len]) |m, *p| p.* = m.splitAxis(.qout, .{ .qg = groups, .qs = -1 });
    return T.concatenate(parts[0..members.len], .qs).merge(.{ .qout = .{ .qg, .qs } });
}
```

- A Source never touches the registry or bindings; only `bind` records
  one. Tags flow through ops, so partitioning is given once at `bind`,
  keyed by the final tags; memory is a bind option.
- `linear.View.weight()` becomes `source(...).slice(.qout, ...)`, so
  `bindSources` goes away and `interleaveLinears` needs no Packer.
- The loader contract becomes bytes only: Sources are valid by
  construction; the loader keeps bounds and disjointness checks and drops
  the same-dtype and whole-row rules (they forbid `bitCast`, the fp4
  unpack). `Layout`/`Part` stop being public; `createComposite`,
  `createConcatenated` and `ConcatenateOptions` go.

### Why mirror Tensor's ops

Writing pack code once for both types buys less than first suggested;
the measured benefit is the loader path itself (DeepSeek estimate in the
design section above). Generic code helps porting (`packFused`,
`packStack` and the swizzles encode the kernels' layouts; a layout bug
gives wrong outputs, not an error), choosing where it runs (a device
fallback while the loader cannot run a lowered shape, maybe for
4-byte-chunk swizzles, unmeasured) and tests (XLA as an independent
reference). Costs: Tensor's semantics (tag rules, `-1`, `Slice`) must be
kept, `comptime T` reads worse, Tensor-only ops (`convert`,
`withPartitioning`) break it, and production ends up instantiating only
`Source`. Decision: mirror the vocabulary, write llmd's layout code
against `Source`, keep generic code for tests and, if the device measures
better, the swizzles.

### Interleave per shard, not at the finest granularity

With a fused tensor partitioned on `.dout` over D devices, every group
count that is a multiple of D gives every device the same rows of each
member; with groups = D (1 if replicated) each member lands as one
contiguous run per device. Llama-3.1-8B per layer: gate/up finest is
28,672 chunks of 8 KiB, at D = 4 8 chunks of 28 MiB; qkv by KV head 24
chunks of 1 MiB, at D = 4 12 of 2 MiB. Valid whenever the sharding is
(block-FP8 needs R_i/D in whole scale blocks, heads need D to divide the
head counts; both already required). Every fused projection in llmd then
lowers to D contiguous parts per member. `.explicit` stays for
kernel-imposed layouts (mxfp4 experts); `.finest` has no remaining use.

### Representation and lowering

- A piece reads one checkpoint tensor through an affine lattice of up to 8
  dims (count, source byte stride with 0 to repeat, visible axis and
  step); destination offsets are row-major of the final shape. (As
  implemented, pieces are kept in flat row-major order; see the
  twenty-fourth pass.) Reshapes change only shape or steps; `concatenate`,
  `stack`, `pad` shift origins; `transpose` permutes; `slice` adds at most
  3 pieces per nesting level; `splitAxis` cuts only where a lattice dim
  straddles a row boundary. No op enumerates rows.
- `bind` sorts and coalesces dims into `Part{ source, src_offset,
  dst_offset, dims: [<= 8]{ count, src_stride, dst_stride } }`; the
  innermost contiguous run is the chunk. Execution by shape: few large
  chunks (>= 1 MiB, <= 64) expand to 1-D parts; many chunks >= 4 KiB go to
  destination-ordered iovec scatter; chunks < 4 KiB and stride-0 dims use
  a contiguous read plus a host strided copy in the pinned block.

### Evidence: a model of the algebra

A pure-Python model (session scratchpad, not kept) evaluated each case
both as lowered lattice pieces and as reference index arithmetic: all 15
byte-exact at small sizes, per-index fallback never triggered. Full sizes
(Llama-3.1-8B per layer; experts E = 256, 2048 rows, 3584 B per fp4 row):

| case | pieces | chunks | smallest chunk |
| --- | ---: | ---: | ---: |
| qkv, groups = KV heads / 4 shards | 3 | 24 / 12 | 1 / 2 MiB |
| gate/up, finest / 4 shards | 2 | 28,672 / 8 | 8 KiB / 28 MiB |
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

Pieces stay proportional to the sources; chunk counts pick the execution
class. Swizzles and stride-0 broadcasts must never become one read or DMA
piece per chunk.

### Phases

1. zml `Source` (algebra, lowering, `bind`), byte-only loader checks.
   Tests run each generic pack function as a Tensor program on CPU and as
   a bound Source through `Loader.load`, matching bytes per shard on 1, 2
   and 4 devices. llmd: `linear.View` on Sources, per-shard interleave,
   concatenated expert stacks, engram, n-grams and splits; the `fuse`,
   concatenated `stack` and same-dtype `concatenate_rows` programs go.
2. Strided parts: mxfp4 row-interleaved experts (~1M chunks of 3.5 KiB
   per DeepSeek layer) and broadcasts.
3. Swizzles as Source code on the host copy path.
4. Programs (dequantize, convert, normalize, globals) take Sources as
   inputs, placed like their outputs.

### Open points

- Naming: `Source` and `bind`.
- ~150k pieces (~20-50 MB) at init for a DeepSeek-sized model: use a
  scratch arena reset after init.
- The Linear-level verbs stay in llmd; once they depend only on Source
  they could move to `zml.nn`.
- llmd ignores `ExpertsLayout.gate_up_order` (`.up_gate` for flashinfer
  without quantization), harmless while `Backend.auto` picks flashinfer
  only for NVFP4; `Source.concatenate` makes the order explicit.

## Twenty-fourth pass: Source phase 1 implemented (2026-10-18)

The user approved the Source proposal ("Ok do it"). Built on duplicates
of two user commits it needs: zml `c865a2cb` "zml: Add
Shape.Partitionning" (`Sharding.Partitioning.Logical`,
`Shape.partitioning()`; its `io.zig` hunk moved to `io/TensorStore.zig`)
and monorepo `rsqorxts` "llmd/weights: use explicit tags and
partitioning", rebased onto the merge. Passes 24-26 were committed
together as zml `omuqxqpu`, monorepo `rwqvkpqy`.

### zml

- `zml/io/Source.zig` (`TensorStore.Source`) mirrors Tensor's shape ops
  (renames, reshapes, `bitCast`, `slice`/`split`, `transpose`, `pad`
  with zeros, `broad`, `concatenate`, `stack`) and ends in
  `bind(mesh, partitioning, .{ .memory })` or `bindAs(shape, ...)`.
  Pieces are affine lattices in flat row-major order, so reshapes never
  touch them; an axis op rewrites a piece in coordinates, cutting it at
  row boundaries only where a dimension carries. `bind` lowers pieces to
  byte `Part`s; a whole checkpoint tensor stays a plain binding.
  Checkpoint tensors are deduplicated by registry entry, since every
  `View.source` returns its own copy.
- `createConcatenated`, `createComposite`, `ConcatenateOptions` and
  `Layout.fill` are gone. `Layout` is `{ shape, parts }`; `Part` is
  `{ source index, src_offset, dst_offset, width, dims[]{ count,
  src_stride, dst_stride } }`.
- Loader checks are bytes only (same-dtype and whole-row rules gone):
  the parts fill the tensor, stay in bounds, do not overlap, and may
  share a source. A strided part becomes one load part per run when it
  has at most 64 runs or runs of at least 1 MiB, else
  `UnsupportedLayout`.
- Bug: `Tensor.Slice.absolute` ignored `.len` when `.end` kept its
  default, so `.{ .start, .len }` took the rest of the axis; the first
  real-model run caught it (an LFM2 `in_proj` member got the rows of the
  members after it). `.len` is checked first, with tests.
- Tests: 22 Source cases checked byte for byte against the same function
  as a Tensor program (the thirtieth pass found this harness ran zero
  cases, all private; fixed in the thirty-first); lowering checks
  (gate/up at Llama scale stays one piece per member); loader composites
  on 2 and 4 devices, every sharding, default and host-pinned memory.
  CPU passes; four B70 320 pass, 11 skip (moe Triton test skipped by a
  temporary patch, see twentieth pass).

### llmd

- `linear.View` components are Sources. `Interleaving.finest` became
  `.per_shard`: one group per device of the output axis, one when
  replicated; every fused site (qkv, gate/up, GDN qkvz, lfm2 `in_proj`,
  muse_glimmer qkvg, gemma4 qkv) uses it, `.explicit` only for layouts a
  kernel imposes.
- `bindFused` binds copied components as composites with no program;
  broadcast scalars (per-tensor FP8 scales, NVFP4 globals) kept
  `Recipe.fuse` until the loader could repeat bytes (twenty-sixth pass).
  `bindExpert(..., .assembled | .program)`: concatenated stacks are
  Sources, each expert's bytes going only to its devices; row-interleaved
  mxfp4, swizzled scales and FlashInfer NVFP4 kept programs (twenty-sixth
  and twenty-seventh passes). `concatenateRows`, `splitTensor` and the
  deepseek41 engram tables (`pad` plus `bind`, host pinned) too.
- A composite feeds a program as one input of `layout.shape`, so
  `normalizeBlockFp8` (CDNA3) and `maybeSwizzleBlockFp8Scales` (Blackwell)
  compose with composites. Composites were fed replicated (placed per
  device in the twenty-eighth pass).
- `weights_test` covers splits (first-member dims too, which the slice
  bug slipped past), per-shard qkv and experts on 1, 2 and 4 CPU devices,
  block-FP8 normalization through the assembled input; forcing one group
  fails two tests. 7/7; builds for cuda, rocm, oneapi.

### End to end on the four B70s (TP=4)

llmd release, August oneAPI plugin (manual-2026-08-17; master's plugin
segfaults compiling llmd), one greedy "capital of France" request, all
answered Paris: LFM2.5-1.2B-Thinking 2.18 GiB in 3.09 s,
Llama-3.1-8B-Instruct 14.96 GiB in 10.0 s, Qwen3.5-4B (GDN) 7.83 GiB in
8.0 s, gemma-4-26B-A4B-it (MoE) 47.00 GiB in 11.4 s, Laguna-XS-2.1 (256
experts, `--io-impl=threaded`) 62.29 GiB in 9.4 s.

- Covered: `in_proj` split and interleave, per-shard qkv and gate/up, GDN
  qkv interleaved with z, one expert per device. LFM2.5-8B-A1B:
  `InvalidModelType` (unsupported, unrelated).
- Laguna under the default `--io-impl=zio` segfaulted compiling: a zio
  coroutine stack overflow in `Compiler.emitMlir` (149 KiB frame probe;
  stacks cap at 8 MiB, `RuntimeOptions.stack_pool.maximum_size`). Not a
  loading bug. Likely cause found in the thirtieth pass, fixed in the
  thirty-first, unverified under zio.
- Llama's 10 s: column-sharded `o_proj`/`down_proj` give one DMA piece
  per row (7.5 s in the twentieth pass). Fixed in the twenty-fifth pass.

### Left open

Phases 2-4 came in the twenty-sixth to twenty-eighth passes. Not run:
CUDA, ROCm (the swizzle and FP8 normalization composition only through
the CPU FP8 test), DeepSeek (GB300 in the twenty-seventh pass).

## Twenty-fifth pass: column-sharded loads staged by destination (2026-10-18)

Request from the user: keep threaded Io on Intel; 10 s for Llama is far
too slow; one device should load at 20-30 GiB/s and TP=4 not far below.

### Cause

`examples/io load <model> tp`, a new mode sharded like llmd's TP
(`o_proj`/`down_proj` on their input axis), August plugin, four B70,
warm, Llama-3.1-8B:

| layout | DMA submissions | loader elapsed | `dma_piece_latency_ms` |
| --- | --- | --- | --- |
| rows (`sharded`) | 3,034 | 0.673 s | 0.43 |
| tp, file order | 1,051,178 | 5.144 s | 1.22 (pumps full 1.05M times) |

`DispatchSpans` gave a column-sharded tensor one span per row and
device, each a submission: 2-7 KiB pieces, at most 64 in flight per
device, ~1.2 ms each. Threaded Io did not change llmd (9.9 against
10.0 s with zio).

### Change (zml `direct_loader.zig`)

- The planner collects a DMA job's pieces, then places them: in file
  order, unless staging saves at least `Config.minimum_staging_gain`
  submissions (64, one device's in-flight pieces) and the pieces tile the
  job's bytes once (overlapping parts never stage). A staged read lands
  grouped by (destination, writer mask), each group sorted by
  destination offset, so a device takes one piece per block.
- `readStaged`: a buffered read scatters straight into the blocks
  (`preadv`, under one hold of the read gate); a direct read lands in a
  per-worker page-aligned bounce buffer and is copied into place, since
  O_DIRECT iovecs must be aligned.
- Tests replay staged jobs and load 12,000 one-byte column pieces;
  mutating the scatter or the bounce copy fails them.

### Results

`examples/io`, Llama-3.1-8B, four B70: tp warm 0.714-0.745 s (~20-21
GiB/s; 5,468 submissions, 599 staged jobs) against rows 0.673 s; cold
(`posix_fadvise(DONTNEED)`, O_DIRECT) tp 1.631 s, rows 1.604 s, both
bound by the NVMe (Samsung 9100 PRO, ~9.2 GiB/s). `ZML_LOAD_CHECK=1`
read back all 291 tensors, warm and cold.

llmd, `--io-impl=threaded`, `Loaded weights`, all answers correct:

| model | 24th pass | now |
| --- | --- | --- |
| Llama-3.1-8B, TP=4, warm | 10.0 s | 0.759-0.769 s (19.5-19.7 GiB/s) |
| Llama-3.1-8B, one B70 (`ZE_AFFINITY_MASK=0`), warm | | 0.658 s (22.7 GiB/s) |
| Llama-3.1-8B, TP=4, cold | | 1.834 s (8.2 GiB/s) |
| LFM2.5-1.2B-Thinking | 3.09 s | 0.134 s |
| Qwen3.5-4B, warm | 8.0 s | 0.437-0.448 s (17.5-17.9 GiB/s) |
| gemma-4-26B-A4B (47 GiB) | 11.4 s | 4.99-5.01 s (9.4 GiB/s) |
| Laguna-XS-2.1 (62 GiB) | 9.4 s | 6.34 s (9.8 GiB/s) |

- gemma-4 and Laguna do not fit beside llmd in the host's 62 GB of page
  cache: disk-bound (13 ms per 8 MiB direct read, no credit or DMA
  waits).
- Planning stayed 73-112 ms per batch: `DispatchSpans` still enumerated
  one span per row and device. Fixed in the twenty-sixth pass (1-4 ms).

### Left open

- llmd TP=4 at 0.76 s against 0.66 s on one device: staged submissions
  cost 20-39 us each to submit (8 us for row pieces); not investigated.

## Twenty-sixth pass: strided parts in the loader, placement without rows (2026-10-18)

Request from the user: "Ok do phase 2", after asking whether strided
support would cut the planning cost.

### zml

- `zml/io/ShardGrid.zig` replaces `DispatchSpans.zig`. Placement splits
  every axis evenly, so devices hold the boxes of a grid, each packed
  row-major. A contiguous logical range lands as one contiguous range of
  every box it touches, found by arithmetic; nothing lists rows. Tests
  compare every range walk with the devices' slices element by element.
- The planner treats pieces as logical ranges of their destinations; a
  staged job lands each piece box by box and the worker walks the runs
  as it reads (a scatter sink for buffered reads, a copy sink from the
  bounce buffer for direct ones). The staging decision estimates file
  order's submissions instead of listing runs.
- Strided parts beyond the front end's expansion rule go to the backend
  whole (`backend.Lattice`). Gather jobs: a destination with strided
  parts is planned whole, in its first part's file, cut into ranges of at
  most a job; each part reads the source range its runs come from, and
  the range lands box by box, uncovered bytes zeroed. In-order source
  runs scatter straight into the blocks; a broadcast or transpose reads
  its whole source range in every job through the bounce buffer, and is
  refused at submit if that does not fit one read. Two interleaved parts
  must be provably disjoint (footprints repeating with one stride), else
  `UnsupportedLayout` (widened in the twenty-seventh pass; the overlap
  prover went in the forty-fifth). The buffered backend reads a strided
  part's source range once and copies its runs.
- `Source.stack(..., .last, ...)` did not compile (`rank()` is `u4`);
  fixed.
- Tests load row interleaves from two files, scalar broadcasts and a
  padded transpose on 1, 2 and 4 devices in every sharding; mutating
  either sink fails them.

### llmd (`weights.zig`)

- `bindFused` `.broadcast`: each member's scalar is broadcast over its
  rows, then concatenated, so per-tensor FP8 scales across members and
  NVFP4 globals take no program. `bindExpert` `.assembled` takes
  interleaved stacks (`Source.stack` at axis 2, then `reshape`).
- `Recipe.fuse`, `Recipe.stack`, `packFused` and `bindSources` are gone;
  the FP8 normalization and swizzle recipes read only the assembled
  tensor. Tests gather interleaved experts and per-shard FP8 scales on 1,
  2 and 4 CPU devices.

### Results

`examples/io load Llama-3.1-8B tp`, four B70, warm: planning (4 plans)
0.085 s -> 0.002-0.004 s, sealed +0.086 s -> +0.002-0.005 s, loader
elapsed 0.714-0.745 -> 0.688-0.692 s (same 5,468 submissions, 599 staged
jobs); rows 0.668 s, tp cold 1.630 s. `ZML_LOAD_CHECK=1` read back all
291 tensors. llmd planning 1-2 ms per batch (73-112 ms before); loads
(25th pass in parentheses), all answers correct: Llama-3.1-8B TP=4
0.73-0.97 s (0.76 s; 474-884 staged jobs, 0 gather), LFM2.5 0.113 s
(0.134 s), Qwen3.5-4B 0.44 s warm, 0.62 s first (0.44 s), gemma-4
5.02 s and Laguna 6.66 s (5.0 and 6.34 s, disk-bound).

- Llama varied for both builds this time (the 25th-pass binary took
  1.05-1.15 s alternating with the new one at 0.84-0.95 s; 0.73 s right
  after `cat`-warming), read-bound; calibration picked 8 or 16 MiB run to
  run.
- No model here gathers (no MXFP4 experts, no per-tensor FP8 scales
  across members); only the tests exercise gather jobs. `//zml:test` 322
  pass on CPU, 323 on the B70s; llmd 7/7.

### Left open

- For MXFP4 block scales (224-byte rows) a buffered gather scatters one
  iovec per row, ~37k per 8 MiB job; bounce copy versus that unmeasured
  (the twenty-eighth pass compared host against device layouts instead).
- Phase 3: a permutation's source is read whole in every job, so it must
  fit one read; per-expert swizzles do, a model-wide one would not.
- A strided destination's jobs are claimed with its first part's file.

## Twenty-seventh pass: swizzles as Sources, DeepSeek on GB300 (2026-10-20)

Phase 3: the block-scale swizzles are Source ops, placed by the loader
from the checkpoint; their pack programs are gone. gb300-1 (2 of 4 GB300,
953 GB RAM, four NVMe in RAID 0), DeepSeek-V4.1-Flash, warm cache.

### zml

- `Lattice.copyRuns`: nested loops over the two innermost dimensions,
  runs of 1-16 bytes as one unaligned load and store, others `memcpy`.
  The bounce copies use it per stretch (a destination range inside one
  grid run and one block); items with under 16 runs per stretch keep the
  per-run walk.
- Overlap check: a column-split swizzle (384 parts, each spanning the
  tensor) made the pairwise check quadratic, 20 s of DeepSeek's front
  end. Parts now form clusters; a cluster of more than 8 parts with at
  most 2^21 runs sorts its runs, else pairs are checked exactly (one part
  of at most 1024 runs) or by a bounded search when footprints repeat
  with the same strides. (Removed with the overlap prover in the
  forty-fifth pass.)
- `TensorStore.sourceOf` gives the source a binding loads, in a tensor's
  shape; `Source.rebind` replaces a tensor's binding with a source.
- `Source.transpose` treats a permutation that only moves size-1 axes as
  a reshape: the FP8 swizzle's transpose with one column device otherwise
  cut its lattice into 192 pieces.

### llmd (`weights.zig`)

- `swizzleBlockScales` and `swizzleBlockFp8Scale` are generic over
  `Tensor` and `Source`, the former programs' ops unchanged. Expert
  stacks gain `.swizzled` (cute MXFP4 and FlashInfer NVFP4 block scales)
  and `.fp4` (FlashInfer NVFP4 weights bitcast to `f4e2m1`); block-FP8
  scales are swizzled per projection through `sourceOf`/`rebind`.
- The `stack_nvfp4_cutlass_*` and `swizzle_block_fp8_scale` recipes,
  `packStack` and three pack programs are gone; NVFP4 global scales and
  dequantizations stay programs (they compute).
- CPU tests through the loader check against the kernels' index formula.

### Results

DeepSeek-V4.1-Flash, GPUs 0,1, `--token-batch-size=256 --batch-size=1`,
cute MXFP4 experts, 467 GiB:

| | baseline (26th pass) | swizzles as Sources |
| --- | --- | --- |
| loaded weights | 79.6 s first, 66.5 s warm (7.0 GiB/s) | 14.9-15.4 s (30-31 GiB/s) |
| pack programs | 10 programs, 330 tensors (17 GiB) | 1 (`wo_a` to BF16), 40 tensors |
| worker wait in destination creation | 11.3 ms per read | 0.16-0.2 ms per read |
| gather jobs | 10,840 | 12,090-23,890 |

- The baseline's workers mostly created destinations: every per-expert
  scale was its own program input, ~50K device buffers of 368 KB, each a
  `createBuffersForAsyncHostToDevice` per device.
- First swizzle build: 39.5 s, 20 s of it the quadratic overlap check.
  Answers "Paris" and "391". Calibration picked 16 MiB, then 8 MiB.
- Width sweep, forward stubbed (compile 2.3 s instead of 67 s): width 16
  13.3 s (35.1 GiB/s), 24 12.0 s, 32 11.7 s, 48 11.2 s (41.5 GiB/s);
  full model at width 32 12.45 s. Sixteen O_DIRECT `dd` read 57 GB/s
  from the RAID.
- Bounce copies: 119.8 GiB, 22-27 s of worker time summed (~1.5 s per
  worker at width 16). `direct_io=auto` reads a mostly cached file
  buffered, so the bounce volume follows the shared host's page cache.

### Left open

- Model compilation takes 67 s before the load; llmd compiles in `init`
  and loads after, so overlapping them would hide most of the load.
- Default width 16 leaves DeepSeek 15% slower than 48 here (the GB300
  sweeps behind the default were flat from 12 to 48 on lighter models).
- Destination creation still costs 22-27 ms for some tensors.
- Not run: NVFP4 (FlashInfer) checkpoints, other Blackwell FP8 models,
  TP=4.

## Twenty-eighth pass: programs over Sources, expert layouts on the device (2026-10-20)

Phase 4: a program's inputs are Sources, each placed by its own shape.
The user, skeptical of swizzling on the host, asked to compare the
loader's small-run copies with programs doing the same layouts from an
intermediate buffer.

### zml

- `Binding.transformed` is gone: a program binding has one `Layout` per
  input, placed by that input's shape (`Source.place`/`placeAs`,
  `Source.bindProgram(inputs, output)`). `TensorStore.unbind` removes a
  binding; `sourceOf` is null for a program's tensor.
- `loadExecute` lands each input from its layout and checks count, dims
  and placement against the executable.

### llmd (`weights.zig`)

- `normalizeBlockFp8` places its input like the tensor, so each device
  converts its slice (the old composite input was replicated).
  `bf16Linear` (`wo_a`) likewise. NVFP4 globals take two stacked inputs
  instead of one per expert and projection (768 for 384 experts).
- `Packer.expert_layout`: `.device` (default) binds each expert stack
  with its projections concatenated (whole per-expert tensors, large
  runs) and lays it out in a program (row interleave and
  `swizzleBlockScales`); `.loader` keeps the phase 3 Sources.

### Results

DeepSeek-V4.1-Flash on gb300-1, GPUs 0,1, forward stubbed, width 16,
runs alternated, temporary counters (`TMP_DIRECT_IO`, GPU memory sampled
every 250 ms, KV cache included):

| direct_io | layout | load | bounce copies | worker copy time | executes | peak GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| auto | loader | 13.71-14.43 s | 77 GiB | 17.5 s | 40: 2.4-3.1 s | 244-247 GiB |
| auto | device | 12.86-13.36 s | 1.2 GiB | 0.7 s | 160: 4.2 s | 249-252 GiB |
| on | loader | 12.98-13.11 s | 188.6 GiB | 36 s | 40: 2.5-2.7 s | 247 GiB |
| on | device | 12.69-12.88 s | 3.1 GiB | 1.0 s | 160: 3.5 s | 251-253 GiB |

- Device layout: 40 pack groups of 4.65 GiB, 19-22 admission retires,
  minimum room 45 GiB. Its 4 programs compile in 1.8-2.4 s before their
  groups are submitted, only partly hidden; the 120 layout executes cost
  ~1.1 s (9 ms each). The 40 `wo_a` dequantizations take 60-77 ms each
  during the bulk load (not attributed).
- Full model, device layout: compiled in 69.4 s, loaded in 13.2 s
  (35.4 GiB/s; loader layout 14.9-15.4 s in the twenty-seventh pass),
  answers correct; another job may have taken the GPUs during this run
  (the table's runs had them to themselves).
- Host copies are not this host's bottleneck: with 217 threads the
  loader layout still loads at 35 GiB/s while copying 189 GiB; a host
  with fewer cores would show more of the difference.

### Left open

- Compile the pack programs during model compilation (the thirty-fourth
  pass starts them in `Loader.start` and publishes the bulk first).
- Not run: NVFP4 (FlashInfer), TP=4, MI300 (FP8 normalization now
  converts per device slice).

## Twenty-ninth pass: profile of DeepSeek's load (2026-10-20)

Where DeepSeek-V4.1-Flash's load time goes, compilation aside. Setup of
the twenty-eighth pass (gb300-1, GPUs 0,1, forward stubbed, device
layout, width 16), committed code plus temporary instrumentation (a
100 ms sampler of md127 bytes and per-thread CPU, monotonic log stamps,
`zml/vfs` debug logs) and one `nsys` run (CUDA and OS runtime trace, CPU
sampling with DWARF backtraces). The GPUs were taken after two runs, so
no finding is validated by a fix. Baseline: 467 GiB in 11.9 s
(39 GiB/s); under nsys 14.8 s.

### Findings

1. **A 2.4 s stall when the engram tables are allocated** (4.6 s under
   nsys). The four `layers.{1,14}.engram.embed.{weight,scale}` tables
   (2 x 91.6 GiB FP8, 2 x 2.9 GiB) bind `.host_pinned`, row-split over
   two devices: 8 shards. The allocation task creates them in the pinned
   host space and the plugin registers each with `cuMemHostRegister`:
   0.56, 0.57, 0.62 and 2.5 s for the 46 GiB shards, one at a time, under
   the driver lock. `cuMemSetAccess` (up to 2.5 s), `cuEventRecord`
   (3.5 ms each) and the uploads wait, so the pumps stop, the 33
   credits stay held and every worker waits: no H2D or disk reads for
   the window. Nearly all credit waits fall here
   (`credit_wait_ms_per_read=0.696` x 60,667 reads = 42 s of worker time,
   ~17 workers x 2.4 s). The main batch was published after the pack
   groups (+1.9 s, waiting on pack compiles), so the stall landed
   mid-load (the thirty-fourth pass publishes the bulk first).
   CPU-only, 46 GiB anonymous, 64 KiB pages: `mlock` of unfaulted memory
   0.48 s (most of the driver's 0.56 s); faulting on 16 threads first
   takes 0.17 s (0.8 s on one), then `mlock` 0.2 s. No huge pages: THP is
   512 MiB on this kernel, mTHP disabled, hugetlb pools empty (both need
   root). The cost is per page (~750K per shard), not zeroing: the host
   boots with `init_on_free=1`, so pages are cleared when freed; `mlock`,
   `MADV_POPULATE_WRITE` and `MAP_POPULATE` all take ~0.5 s per 46 GiB
   at any thread count (one thread clears only 56 GiB/s). One 46 GiB
   `mlock` run took 9 s, possibly clearing during reclaim (not verified).
   A read-only mapping of a cached checkpoint file costs 0.16 s per
   46 GiB on 16 threads.
2. **Otherwise the load is read-bound.** Workers spend ~70% reading, 22%
   in credit waits (the stall), 7% creating destinations. Outside the
   stall ~50-52 GiB/s; copy engines ~40% busy, pumps mostly stop empty.
   A read averages 7.9 MiB in 2.2 ms (3.5 GiB/s per worker). Buffered
   reads of cached data: 7.6-7.9 GiB/s on one thread, 85 on 16, 96 on
   32; the RAID reads 57 GB/s with O_DIRECT.
3. **O_DIRECT re-reads cached data.** 311 of 467 GiB were cached, yet
   205 GiB came from disk: `direct_io=auto` reads a file direct when at
   most 50% cached, so 16 files at 38-50% re-read their cached 44 GiB. A
   per-range choice would have read 156 GiB.
4. **Destination creation uses the VMM allocator** (llmd sets `.vmm`):
   `cuMemCreate` + `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess`
   per device buffer (p50 142 us, p90 706 us under nsys), on the
   workers' path: ~14 s of worker time, ~0.8 s of wall time.
5. **The engram host jobs form the tail.** The last ~2.5 s have no H2D:
   11-13 workers copy the engram files (66-78% cached) into the pinned
   tables at 10-28 GiB/s.
6. **NCCL init holds one CPU for ~35 s** (`ncclCommInitAll` ->
   `cudaFuncGetAttributes` -> `cuLibraryLoadData`): NCCL 2.30.4 ships
   nothing for sm_103 (GB300), only cubins up to sm_120 and sm_120 PTX;
   `~/.nv/ComputeCache` holds 96 KiB. Hidden by the full model's 67 s
   compile; with a short compile (cached executables) it is on the
   critical path (~16 s after the load in the stubbed baseline), and it
   competes for the driver lock.
7. **Device 1's uploads are slower per copy**: CUPTI reports them as
   peer-to-peer from pinned memory (the arena is registered through one
   client); for copies of 8 MiB and more, median 67 against 84 GB/s on
   device 0. Not a limit today.
8. Smaller: the first group is published 0.17 s after the load starts;
   tokenizer, platform and loader creation take ~2 s before compilation.

### Suggested next steps (need GPUs)

- Engram stall: let the loader allocate host destinations itself
  (arena-style placed pages, `DmaMap`, a PJRT view freed by its delete
  callback), fault them in on several threads and `DmaMap` afterwards,
  so the driver holds its lock ~0.2 s per 46 GiB instead of ~0.56 s,
  ideally after the device DMA. Prerequisite: a device-kind view of such
  memory works (held on CUDA in the fifty-first pass).
- Width sweep again once the stall is gone.
- Per-range direct or buffered reads (each job's residency, e.g.
  `cachestat`), with a second descriptor per file for direct reads.
- NCCL: time `preload` alone, with a larger `CUDA_CACHE_MAXSIZE`, and
  with an NCCL build carrying sm_103 (or sm_100f) code.

## Thirtieth pass: end-to-end loader review (2026-10-21)

A read-only review of the whole load path (llmd through `zml/io` and
`vfs/`) against "code earns its place, fewer abstractions, logic local".
Report: `LOADER_REVIEW.md` (Part 1 bugs, Part 2 ten simplification
moves, Appendix A every finding). 185 findings from twelve reviewers;
adversarial verifiers kept 184 and refuted 1. Nothing was built or run.

Headlines, all fixed in the thirty-first pass:
- The Source op harness ran zero cases: `std.meta.declarations` lists
  only `pub` declarations, and every case was private.
- `//llmd:test` no longer compiled (`Binding.transformed`, removed in the
  twenty-eighth pass).
- llmd quantized bindings: compressed-tensors NVFP4 expert globals
  ignored `.divide` on FlashInfer; dense NVFP4 dropped `input_scale`; FP8
  per-tensor gate/up experts got `[E,2]` scales.
- llmd never passed `io_impl` to `Platform.auto`, so under zio compiles
  skipped `blockInPlace` and ran on 8 MiB coroutines: the likely cause of
  the Laguna stack overflow (fix unverified under zio).
- Remote reads had no stall timeout; HF re-resolve freed URLs in use;
  the `auto` O_DIRECT probe warmed the pages it sampled, so the next load
  read that file buffered.
- Error paths: pack failures panicked in `Exe.call`; `deinit` with
  pending submissions read and transferred them all; progress stopped
  after ~125 destinations; mixed GPU kinds hit `unreachable`.

The byte core (Source lowering, ShardGrid, planner, FIFO claims, pumps,
credits, drain) held on every traced path. Best simplifications per
risk: credits as the only pinned-memory bound, ~250-300 lines (move 2,
thirty-first pass); validation once and no overlap prover, ~300 lines
(Move 3, forty-fifth pass); one host-resident path (Move 1, fiftieth
pass); `Pipeline` folded into `Loader` (Move 4, forty-sixth and fiftieth
passes).

Move 5 proposed a fixed DMA block size per target in place of
`dma_calibration.zig` (~850-900 lines). The user declined it on
2026-10-21: calibration stays, so do not propose removing it again. Its
"If declined, do the minimum" items: delete `findArena`, always null
(done in the forty-sixth pass); tests pin one block size; CPU
calibration honors a one-element list.

## Thirty-first pass: fixes from the loader review (2026-10-21)

The thirtieth pass's headline bugs are fixed, with steps 1 and 2 of its
move 2; each commit reviewed adversarially (31 findings, none high).
zml (on `qztxwlmv`):
- The Source harness runs its 23 cases; `WidenBits` fixed.
- Progress counts one item per destination (#19); mixed device kinds
  load (#9); `Exe.call` returns `pjrt.ApiError`, a failed pack is the
  loader's sticky error (#17); `Backend.abort`: the first failure, or
  deinit, aborts what is pending (#18); `Loader.cancel()`.
- The `auto` probe draws a random shift per probe (#7). A request attempt
  is retried after 60 s with no bytes (`RetryConfig.stall_timeout`, #1),
  its clock paused while the attempt runs its own HF re-resolve or GCS
  token refresh. Each HF read copies its URL; logs name the file, not
  the signed URL (#2, #14).
- `pjrtx` compiles on the calling thread (`io.async` had put XLA's
  compile on a zio coroutine).
- Move 2: `xmrzxtsl` (step 1) makes `BlockPool` never wait, the credits
  being the one bound on pinned blocks; `tmtwwsts` (step 2) makes a
  credit a fixed slot of `maximum_blocks_per_job` blocks, dropping
  `BlockPool`, `Lease`, `BlockContext` and the plan's block contexts
  (275 lines shorter). No load-time change detected (thirty-third pass).
  The summary's pinned high water became `credits_high_water`, a count.

monorepo (on `oxpzxnvt`; `pxzvsqqp`, `kvklzrov` are later measured
trees):
- llmd passes `--io-impl` to `Platform.auto` (#4): with the `pjrtx`
  change, no compile runs on an 8 MiB coroutine, the likely fix for the
  Laguna zio crash; unverified (the B70 rule is threaded io).
- `//llmd:test` compiles; NVFP4 FlashInfer globals honor `.divide` (#3);
  dense NVFP4 keeps `input_scale` (#10); FP8 per-tensor gate/up experts
  get per-row scales `[E, 2n, 1]` (#11); NVFP4A16 FlashInfer experts are
  an error, not a panic.

Smaller: the device bound is `Platform.MAX_NUM_DEVICES` (32 without
TPU); a send or `receiveHead` failure closes its connection; the probe no
longer divides by zero; the HF tree cache is locked; llmd's `loadInto`
cancels the loader on error before freeing its programs.

Verification: zml, vfs, examples and llmd (7/7) tests pass. Four B70:
`examples/io load Llama-3.1-8B-Instruct tp`, `ZML_LOAD_CHECK=1`, 291/291
in 1.45-1.62 s cold; host-pinned `sharded` 291/291.

Left: no test for `Exe.call`'s asynchronous failure half, range-read
progress, or HF (needs an injectable Hub endpoint); the buffered
backend's abort lets admitted tasks finish their tensor; under
`std.Io.Threaded` with a finite `async_limit`, the per-attempt watch
task halves the buffered backend's remote read parallelism. Review items
still open: S3 credentials and key encoding (#5, #12), https redirects
(#13), GCS key files (#16); #6 came in the thirty-ninth pass, #8 and
part of #20 in the thirty-fifth, and #21 went with `loadExecute` in the
thirty-fourth.

## Thirty-second pass: API review of the loader (2026-10-21)

A read-only review of the loader's APIs, from llmd's and the examples'
call sites down to the backends' seams. Report: `LOADER_API_REVIEW.md`
(Part 1 the recommended API, Part 2 sixteen moves R1-R16, Part 3 the
user's decisions). 123 findings from sixteen reviewers, 78 after
dedupe, 77 survived verification; three designers and a judge produced
the design. Nothing was built or run.

Bugs (fixed in the thirty-fifth pass, R1, unless noted):
- examples/llm deinitialized the stale init-time copy of its Loader:
  `Future.cancel` after `await` returns the stored result, so the live
  loader leaked and kept its pinned memory. AGENTS.md's future idiom
  produces exactly this for a by-value handle (fixed by R5, thirty-eighth
  pass).
- Direct `submit` returned `Closed` instead of the cause (#29).
- The front end's `failure` was not set when `backend.submit` failed
  after a partial publish, so a later `awaitAll` succeeded (already fixed
  by the thirty-fourth pass).
- DeepSeek's DSpark projections and `main_proj` missed the sm_10x
  swizzle opt-in (R1 stop-gap, then R9).
- qwen4 freed `ngram_host_buffers` after `arena.deinit()`;
  `resolveModelRepo`'s directories were never closed; #30, #35, #36.

Recommended API (R moves, done in the thirty-fourth to fifty-first
passes): one `load` returning the `Bufferized` tree and borrowing nothing
after; programs recorded in the binding, so `loadExecute`,
`Loader.Binding` and llmd's recipes and Packer go; `Loader.init` returns
at once; the store owns the registry; `VFS.Remotes` under one error
contract; block-FP8 decided at bind time; placement a Source property set
last.

It reverses: the load/awaitAll split (thirteenth and fourteenth passes),
caller-owned progress totals (per-submission progress follow-up,
2026-09-05), asserts on options (seventeenth pass), memory as a bind
option and the loader's own bounds and disjointness checks (Source
proposal; the latter via LOADER_REVIEW Move 3). The user's call: dropping
the block-FP8 scale-overflow guard. Needs a GPU: `host_mapped`, a device
view of loader-owned pinned memory (verified on CUDA in the fifty-first
pass).

## Thirty-third pass: benchmark of move 2 step 2 (2026-10-21, gb300-1)

Question: does step 2 (`tmtwwsts`, a credit is a fixed slot of blocks)
change load time against step 1 (`xmrzxtsl`, a pool leased under the
credits)? Monorepo `pxzvsqqp` builds against both.

Method: gb300-1 (four GB300, two Grace nodes, 953 GB RAM; shared). Each
tree's binaries were built once and run against one fixed runfiles tree
(`zig build-lib //llmd:llmd` misses the remote cache on every source
change, 7-12 min per switch); 68 runs, shas checked. Rounds alternate
AB, BA; a run waits for free GPUs and records other users' processes.
llmd DeepSeek-V4.1-Flash (twenty-ninth pass setup) on GPUs 2,3 (node 1),
24 rounds; `examples/io` Qwen3.8-27B (51.75 GiB), replicated and
sharded, 5 rounds each. Another user's llmd, then sglang, held GPUs 0,1
from round 4. `ZML_LOAD_CHECK=1` on four GB300: 1199/1199 tensors for
both trees.

### DeepSeek llmd, 24 rounds

A pair is comparable when both runs calibrated the same request size and
neither stalled in host allocation (`ahead_init_ms` < 4 s; reclaiming
node 1's page cache for the 189 GiB pinned engram tables took up to
20 s): 15 of 24 pairs.

| B vs A over the comparable pairs | value |
| --- | --- |
| median, Hodges-Lehmann | +0.86%, +0.93% |
| mean, 95% bootstrap CI | -0.08%, [-2.8%, +2.2%] |
| B slower | 9/15 (sign test p=0.61, Wilcoxon p=0.39) |
| without the 5 pairs whose GPU neighbours changed mid-run | +1.95%, [+0.4%, +3.5%], 7/10 |
| the same, with `ahead_init_ms` as a covariate | +0.8%, [-1.2%, +3.1%] |
| all 24 pairs, load minus allocation stall | median -0.47%, [-1.8%, +1.3%] |

- Medians: A 12.62 s (n=15 clean runs at 16 MiB), B 12.65 s (n=20),
  Mann-Whitney p=0.97. The largest pair differences (-12.0%, -11.1%) are
  sglang starting during A's run.
- The steady pairs' +1.95% follows B's higher host-allocation time in
  three rounds (+0.45 to +0.54 s), code identical in both trees. Not
  attributed.
- B stops its pumps on the in-flight budget more often
  (`pump_stops_full` +~1000 of ~103k, 12/15 pairs, p=0.035), with no
  load time attached. Credit wait and `dma_stage` differ in only 9/15
  pairs, uncorrelated with load time. `dma_submit_us_per_piece` is
  bimodal in both trees, ~6 or ~13 µs.

Code: here a request is one block (`maximumCoalescedJobBlocks` = 1), so
both trees run 33 one-block credits, return a block in its credit's
callback and reuse it LIFO. Step 2 removes two mutex trips per request
and `Plan.blocks`: under 0.1% predicted.

Verdict: no change detected at about ±2.5%; 1-2% either way is not
excluded (this host resolves ~3.8% per comparison). A ±1% answer needs a
quiet host, one request size, and a load time excluding host allocation.

### Qwen3.8-27B through `examples/io`, 5 rounds each

At 16 MiB, load time follows destination creation (r=0.99-1.00, 3.2 to
8.9 ms per read), which step 2 does not touch. Replicated: A
2.03-2.05 s, B 1.99-2.15 s plus one 2.57 s; sharded: A 1.75-1.77 s, B
1.73-1.87 s plus one 2.38 s, both outliers slow destination creations.

### Found on the way

- **The calibrated request size costs 25-30% on Qwen.** Calibration
  picked 8 MiB instead of 16 MiB in 6 of 20 Qwen and 7 of 48 DeepSeek
  runs, independent of step 2 (Fisher p=0.22). At 8 MiB Qwen takes
  2.54 s instead of 2.04 s replicated, 2.30 s instead of 1.77 s sharded:
  twice the reads, each paying the workers' per-request cost. DeepSeek
  is unaffected (12.59 against 12.62 s). Calibration measures DMA only,
  so its choice between two rates within noise decides a cost it does
  not see.
- **Request sizes measured** (block forced with `ZML_DMA_BENCH_BLOCK_MIB`
  so request = block; ranges of three interleaved repeats, s):

  | | 8 MiB | 16 MiB | 32 MiB |
  | --- | --- | --- | --- |
  | GB300 x2, Qwen3.8-27B replicated, direct | 2.84-3.09 | 2.07-2.59 | 2.47-2.77 |
  | the same, buffered, warm | 2.95-3.07 (3.64 cold) | 2.29-2.78 | 2.56-2.64 |
  | B70 x4, Llama-3.1-8B tp, direct | 1.70-2.66 | 2.34-3.26 | - |
  | the same, buffered, warm | 1.03-2.17 | 1.65-2.96 | - |
  | B70 x4, Llama-3.1-8B replicated, direct | 1.99-2.09 | 2.11-3.08 | - |
  | the same, buffered, warm | 1.68-1.98 | 2.02-3.39 | - |

  GB300 loses 10-25% at 8 MiB, direct or buffered; 16 and 32 MiB tie.
  Buffered 8 MiB reads take the same total worker time as 16 MiB (6.0
  against 6.1 s), so the cost is per request, not in the disk; these
  loads spend ~33 of ~40 s of worker time creating destinations. The B70
  is the opposite: 8 MiB wins by 10-30%. No fixed size suits both hosts.
- **The DMA screen** (`examples/io dma-bench`): GB300 x2 picks 16 MiB 7
  times in 8 (8 MiB once) at the default 8% tolerance and 32 MiB 8 in 8
  at 2%: 32 MiB is the DMA peak, 16 MiB 2-8% below, 8 MiB on the 8% line
  (the coin flip). B70 x4: 2 MiB at 8%, 8 MiB at 2% (6/6 each); its reads
  are 8 MiB because request = max(the local profile's 8 MiB chunk,
  block). Only GB300's choice is unstable, and there the block also sets
  the read size.
- `examples/io load ... sharded` fails on DeepSeek-V4.1-Flash with
  `IncompatibleSharding`: some leading dimensions do not split four ways.
- The first DeepSeek load after the host's memory filled took 30.9 s,
  20 s of it allocating the pinned engram tables (node 1: 16 GB free,
  ~260 GB of page cache).

## Thirty-fourth pass: one start for the whole model (2026-10-22)

The user's decision on the API review's Part 3: submit everything at once,
with the loader orchestrating the programs (R2, plus R4 as start/wait
rather than a blocking `load`). The CPU map for the CDNA3 conversion was
set aside.

- A program is a plain-data value (comptime check) of a named type with
  `pub fn forward(self, output: Shape, inputs: []const Tensor) Tensor`,
  recorded by `Source.bindProgram(inputs, output, program)` or
  `View.maybeCreateBinding(keys, shape, program)` as a
  `TensorStore.Program` (value in the store's arena, per-type vtable;
  `Program.as(T)` reads it back). The type name names the module in
  traces, dumps and logs; names taken from the args type had given the
  two NVFP4 programs one name (review finding).
- `Loader.start(model, buffers, store, progress)` takes the whole model,
  progress by value. It compiles each distinct program on its own task
  (deduped by `Program.sameExecutable`: type, value, placed shapes),
  queues program tensors in model order in groups of at most
  `Loader.program_group_bytes` (8 GiB, was llmd's `group_bytes`), and
  publishes all loader-filled tensors (the bulk) at once.
- Compiles start with `io.concurrent`, else `io.async`: on
  `std.Io.Threaded`, `io.async` runs inline once busy_count reaches
  cpu_count-1, and the direct backend's width+1 workers count, so on such
  hosts every compile would have run inside `start` before the bulk was
  published. llmd's zio and its threaded mode (unlimited async) were not
  affected.
- `Loader.wait()` publishes each group once its programs compiled (output
  checked: `ProgramOutputMismatch`) and the devices have room, retires
  everything in publish order running each group's programs on the
  caller's task, and logs one info line (tensors computed, programs,
  groups, compile wait). Idempotent.
- One `Loader.Error`. Any error of `start` or `wait` fails the loader and
  cancels everything started, so nothing stays borrowed; "validation
  leaves the loader usable" is gone. `cancel` stays public for an error
  between `start` and `wait`. Binding rejections log at err. Gone:
  `load`, `loadExecute`, `Loader.Binding`, `awaitAll`, the `delivered`
  map, the per-call error sets, and the executable-versus-binding checks
  (input count and shape, platform, placement): the loader compiles from
  the binding's own shapes, so only the output is checked.
- Order: bulk first, then groups, admitted with the bulk's bytes counted
  (before, the bulk went last and ungated). The bulk no longer waits for
  any compile: DeepSeek's main batch had been published 1.9 s behind the
  pack compiles (twenty-ninth pass). Group reads come after the bulk's.
  Not measured: no GPU. Not taken: counting bulk retirements apart from
  group retirements in the admission metric (a bulk retire is a real
  head-of-line wait).
- llmd: each `bindProgram` + `recipes.put` pair is one call with a named
  program (`DequantizeBlockFp8`, `NormalizeBlockFp8`,
  `Nvfp4ActivationGlobal`, `Nvfp4WeightGlobal`, `LayOutExperts`,
  `ConcatenateRows`); the `Recipe` union, `Packer.recipes`, `loadPacked`
  and `compileRecipe` went. A model's `startLoad(loader, store,
  progress)` only starts its loads (target, resharded lm_head, drafter);
  `models.zig` owns the "Loading weights" node, waits once, then calls
  `finalizeLoadedBuffers`, which nothing had implemented: deepseek41's
  Engram and qwen4's n-gram view swaps and qwen4's `bakeWeights` moved
  there (the swaps went with `host_mapped`, fifty-first pass). qwen3_5's
  drafter now overlaps its base load. The drafters' progress nodes, never
  ended, went.
- examples/io: the packs are part of the one model (`StackPack`);
  `ZML_LOAD_PACK_PAIRS` became `ZML_LOAD_PACK_GROUP_BYTES`; `load_took`
  now includes the pack compiles.

Checks: `//zml:test` 322 (12 skip), examples build, llmd 7/7. On CPU,
`examples/io load` with 2 packs of 4 passes in one group and in one group
per pack (serial admission retired 2 before the second group).

## Thirty-fifth pass: seam bugs (R1) (2026-10-22)

The `LOADER_API_REVIEW.md` "Read this first" seam bugs, no API change.
Item 1 (examples/llm's stale loader copy) went with R5. Item 3 (success
after a terminal failure) was already fixed by the thirty-fourth pass:
every backend submit goes through the front end's `submit`, and `start`
and `wait` pass their errors to `fail`.

- Direct `submit` (LOADER_REVIEW #29): a planning failure before the
  first publish returns `pipeline.errorValue() orelse err`. An older batch
  that failed between `checkOpen` and that publish had stopped the
  scheduler, so the caller saw only `Closed`. No other `Closed` reaches a
  caller (the read gate closes only after a recorded error or in
  `destroy`; after a partial publish `awaitBatch` returns the stored
  error); loader.md calls it internal.
- `source_slots` (#8): `sourceSlot` dupes the path (key and `slot.uri`),
  freed in `destroy`; a file stays open until then. Under a VFS with
  direct IO (llmd's setup), planning opens it to choose the alignment.
  #8's cached open failure is moot: any error fails the loader.
- safetensors header (#20): the header length must fit in the file;
  `data_offsets` must be two non-negative integers with start <= end <=
  data bytes, values past i64 included (`std.json` gives those as
  `number_string`); else a logged `InvalidSafetensorsHeader`.
- Pairing (item 7): `MapRestrict`, so `Bufferized`, rejects a bounded
  array at compile time, and `prepareModel` asserts equal tensor and
  buffer counts (replaced by structural pairing, forty-ninth pass).
- #35: a failed `dmaUnmap` in `HugePageAllocator.free` warns and leaks the
  pages instead of reaching `catch unreachable`. #36:
  `warmupDeviceAllocators` cancels its group when a spawn fails, so
  earlier tasks no longer outlive it. Neither has a test.
- llmd: qwen4 frees `ngram_host_buffers` with `arena.child_allocator`,
  which allocated it, before `arena.deinit()`. `resolveModelRepo`'s
  directories are closed; `main`'s when `main` returns, since closing
  after `model.init` would need an is-open flag for the error paths.
  Stop-gap until R9: deepseek41's DSpark layers swizzle `wq_a`, `wq_b`,
  `wkv`, `wo_b`, the shared experts' `gate_up`/`down` and `main_proj`
  (`wo_a` is BF16), relying on the review's check that the `mtp.*`
  projections are block-32 F8_E4M3; an unquantized one would panic in
  `swizzleBlockFp8Scales` on sm_10x.

Open: the rest of #20 (a tensor's byte size against `end - start`,
`F8_E5M2`, duplicate names across shards; other missing or mistyped
fields still panic on `.?` or union access). Checks: `//zml:test` 325
(12 skip); both direct-backend tests fail with their fix reverted; llmd
7/7.

## Thirty-sixth pass: deinitBufferized everywhere (R3) (2026-10-22)

The owner of each `bufferize` result frees it with one
`zml.mem.deinitBufferized` call. Hand-written `unloadBuffers` walkers leak
device memory when their struct gains a field they forget; 107 of the 112
went (all 70 in llmd).

- `deinitBufferized` (zml/mem.zig) needed no fix: it frees each slice with
  the allocator passed, at `bufferize`'s alignment, children first,
  through optionals, tagged unions (void payloads included), arrays and
  nested slices, visiting each Buffer once. `Buffer.deinit` empties
  `_shards`, so a buffer never loaded, or already freed by
  `freeDeviceMemoryButKeepHostMetadataMemory`, is a no-op. It cannot
  guard against one Buffer copied into two fields or a second call on the
  same tree (slices are not reset); no owner does either.
- examples/llm keeps `LoadedModel.unloadBuffers`, now the one call;
  `Linear.unloadBuffers` in nn.zig went. llmd frees each `ModelBuffers`
  tree (base, and speculative, which carries the drafters) in init's
  errdefer and in deinit; deepseek41's Engram and qwen4's n-gram
  host-pinned owners are freed right after the tree.
- Each walker that listed fields by hand matched its struct, so no leak
  was fixed, and no two fields share a buffer.

Open: runtime-state helpers keep their names (`KvCache.deinitBuffer`,
`deinitBuffers` in attention.zig and multi_latent_attention.zig);
`speculative.Tensors.initBuffers` could be one `bufferize`. No model was
loaded and freed on a device. Checks: `//zml:test` 326, llmd 7/7.

## Thirty-seventh pass: the rest of R4 (2026-10-22)

The rest of R4, on the thirty-fourth pass's start/wait. The caller still
owns the buffer tree; a `load` that returns it was not taken.

- `wait()` logs "Loaded weights [bytes, time, rate]" in scope
  `zml/io/load`, timed from the first `start` since the last `wait`, with
  the program figures on the same line (the "Computed" line went); a
  second `wait` logs nothing. `bytesLoaded` is gone. `start` adds its own
  progress items (one per loader-filled tensor, one per composite, one
  per program input), which is what the backends complete; callers no
  longer estimate them.
- Invariant: after any error of `start`, `wait` or `cancel`, every buffer
  is empty or holds shards the caller frees, so `deinitBufferized` on the
  tree is always right. Backends and programs write a buffer whole or not
  at all; the gap was a buffer the caller left `undefined`, so `start`
  now empties every buffer before it checks a binding. A failed loader or
  a failed first allocation returns before that, so the docs ask for
  `bufferize`'s buffers. A probe showed every bulk buffer held its shards
  when a load failed, so the new test (a tensor in a missing file,
  `FileNotFound`) does free buffers from a failed load.
- `Loader.Error` is `InvalidBinding` and `Canceled`, plus allocation,
  PJRT, file open and seek, safetensors read and compiler errors. The
  binding errors (TensorNotFound, EmptyTensor, UnsupportedMemoryKind, the
  layout errors, IncompatibleSharding, ProgramOutputMismatch) became
  `InvalidBinding`, each logged at err with tensor, source and rule.
  Backends keep their own names (`backendError` maps them); `Overflow` is
  gone (plain sums). `Closed`, `Internal` and post-`init`
  `InvalidOptions` panic: a backend returns the error that closed it, and
  `init` checked the options. No caller in either repo switched on a
  loader error.
- examples/io prints two "Loaded weights" lines: the loader's (from the
  first `start`) and its own in scope `vfs`, which includes `Loader.init`
  and DMA calibration. Earlier passes' examples/io figures are the `vfs`
  line. In llmd the line now comes from the loader (scope was `models`)
  and no longer includes `finalizeLoadedBuffers`; "Loading weights" gets
  a total. examples/llm's four uncalled Model-level loaders went.
- Review fixes: a host-memory tensor with a sharding its dimensions
  cannot divide was an unlogged `InvalidBinding` (the front end checked
  sharding only for device memory; `addPlacementBytes` now checks every
  space). A source offset near u64 max gave `InvalidTensorRange` logged
  only at debug; `Part.fileRange` and `appendExtent` log it at err.

Open: pairing by construction (done in the forty-ninth pass). The direct
backend's job-end alignment overflow returned `InvalidTensorRange` with
no err log (logged since the forty-ninth pass). Tests cannot tell which
binding rule fired, and a binding error in a partly published direct
batch can log at err twice. The progress total is untested
(`std.Progress` keeps its counters private). Checks: `//zml:test` 327,
llmd 7/7; on CPU both "Loaded weights" lines show 320.25 KiB.

## Thirty-eighth pass: Loader.init returns at once (R5) (2026-10-22)

`Loader.init` blocked for the allocator warm-up, the DMA calibration and
the pinned pre-growth, so callers ran it in a future of their own: llmd
carried one plus a `loader_awaited` flag, and examples/llm deinitialized a
stale copy (R1 item 1). The Loader stays a value.

- `Loader.Options` is `checkpoint`, `read_parallelism`,
  `dma_block_sizes` and `direct_io`; `.auto` is gone, callers pass `.{}`.
  `init` resolves `checkpoint` through `VFS.fromIo(io)` and `loadProfile`
  (a plain io or no checkpoint gets the local profile). The loader
  borrowed the profile's name and stats from the VFS backend, so the VFS
  must outlive it (stats left the loader in R15). The other DMA knobs are
  fixed at their old defaults (eight transfers per device, the windows,
  8% tolerance), so calibration selects the same sizing; on CPU: request
  8 MiB, block 4 MiB, width 16, alignment 4096.
- `init` returns `error{ InvalidOptions, OutOfMemory }`. Every option is
  checked once, in `backendOptions`, each rejection logged: read width
  1..128; read chunk 1..32 MiB; a non-empty block list, each block
  between the direct I/O alignment and 32 MiB; the chunk a whole number
  of every block calibration may select, the CPU default included. That
  last check is conservative: it tests the 4 MiB CPU default on GPU
  targets too. A backend `InvalidOptions` after `init` panics.
- The warm-up and the backend run in `setUp`, started `io.concurrent
  catch io.async`, so inline on an io without concurrency. The first
  `start` awaits it, sets `backend` (now `?Backend`) and probes memory;
  its error fails the loader. The load's time starts after it. `deinit`
  cancels a set-up no `start` awaited; `cancel` and `wait` before any
  `start` do nothing. `Loader.Error` gains `UnsupportedPlatform` and
  `ConcurrencyUnavailable`.
- `dma_calibration.measure(allocator, io, platform, opts)` serves
  dma-bench with every knob in a workspace of its own. `validate` turns
  `calibrate`'s debug asserts into logged `InvalidOptions` (block
  parallelism 1..32, blocks 1..32 MiB, positive windows, tolerance and
  margin in [0, 1)); review restored the 32 MiB cap it had dropped. Review
  also found `measure` reached host_memory's panic on buffered targets,
  as the old bench did: TPU, neuron, metal and furiosa now return
  `UnsupportedPlatform`.
- `limits.zig`, `execute_admission.zig` and `zml.io.limits` are gone:
  `max_read_request_size` (32 MiB) is in backend.zig, the admission
  arithmetic in loader.zig; `Sizing` takes `@max(chunk, block)` and
  asserts divisibility.
- examples/llm frees the loader right after `loadBuffers`, so the pinned
  pools and workers no longer live through the chat. llmd holds
  `Loader.init`, its deinit, `model.init` and `loadBuffers` in one block;
  calibration still overlaps model init and compile. The tokenizer await
  moved before `Loader.init` (it should be done by then; not measured).

Open: dma-bench's `load` mode no longer reads
`ZML_DMA_BENCH_BLOCK_PARALLELISM`. The alignment floor applied with
`direct_io = .off` (no longer since R15). monorepo/AGENTS.md and PLAN.md
caught up in the forty-eighth pass. Not run on a GPU or a remote
checkpoint; no set-up cancelled mid-calibration. Checks: `//zml:test` 330
(also `--runs_per_test=5`), llmd 7/7; dma-bench prints 4194304/8, and
`ZML_DMA_BENCH_BLOCK_MIB=64` logs `InvalidOptions`.

## Thirty-ninth pass: VFS Remotes and one error contract (R6) (2026-10-22)

Four callers assembled the HTTP client and the remote backends by hand,
and examples/llm had drifted (no https, no proxies). Each backend failed a
missing object its own way: HTTP with `Unexpected`, S3 with
`FileNotFound` for a 404 but `Unexpected` for a 403, hf with `Unexpected`
for any tree failure, `Canceled` included. A missing optional file
aborted llmd on s3:// and https:// (LOADER_REVIEW #6).

- `VFS.Remotes.init(self, allocator, vfs, environ)` builds the client on
  the VFS's base io with `initDefaultProxies` and registers `https`,
  `hf`, `s3` and `gs` through their `.auto` constructors. It is
  initialized in place (the backends point at the client) and reserves
  map room first, so nothing is registered on error. It takes a mutable
  `*Environ.Map` because the `.auto` constructors do. examples/io,
  examples/llm, llmd main and zoo use it. `VFS.unregister` and
  `request.RateLimited` (dead) went; `VFSBase` is private.
- Contract for open, stat and access of a remote path: 404 and 410 are
  `FileNotFound`, 401 and 403 `PermissionDenied`, `Canceled` passes
  through; anything else is logged at err (backend, path, error) and
  surfaces as `Unexpected`. One mapping (`request.open_statuses`,
  `openFailure`) serves http, hf, s3 and gs. The VFS no longer logs 401,
  403 or 404 at open or stat; an unexpected status logs two err lines
  (`exchange`'s and `openFailure`'s).
- Fixed: `access` sent nothing on http, s3 and gs and always succeeded
  (it now sends `statFile`'s HEAD); hf `statFile` of a missing path
  returned `IsDir`.
- gs:// requests had no leading slash (pre-existing; review finding):
  `gcsUri` used `bucket/key` as the raw URI path, so every real request
  sent `HEAD bucket/key?alt=media`, and behind a proxy
  `https://storage.googleapis.combucket/key`. The new test hid it by
  passing `/bucket/` for GCS only. Not checked against real GCS.
- S3 answers 403 for a missing key when the principal lacks
  `s3:ListBucket`. It is `PermissionDenied`, and a probe of an optional
  file treats it as absent (llmd's two chat-template probes do, with a
  warning). Wrong credentials look the same, and a required open
  (`config.json`) still fails on it.
- `resolveModelEntrypoint` tries `model.safetensors` after a denied index
  and returns `PermissionDenied` only when neither opens; before, every
  failure was a silent `FileNotFound`. `resolveModelRepo` passes
  `PermissionDenied` and `Canceled` through instead of `InvalidPath`.

Open: a failed streaming read still turned into `EndOfStream` after an
err log, so a throttled `cp` ended early (LOADER_REVIEW #10;
`InputOutput` since the fiftieth pass). HF's denied paths are untested
(its URLs are fixed to huggingface.co), and so is `openFailure`'s
`Unexpected` branch (an err log fails a Zig test). The how-to named
`HUGGINGFACE_TOKEN` while `HF.auto` reads `HF_TOKEN` (fixed in the
forty-ninth pass). Tests use the mock server only. Checks: `//zml:test`
331, `//vfs:test` 50/50, llmd 7/7.

## Fortieth pass: the store owns its registry (R7) (2026-10-22)

The store borrowed a registry that could still grow, so every
`View.source` and create path copied the entry into the store's arena
(`dupeSource`), and Lowering copied it again. A checkpoint tensor had no
identity: Lowering deduped by name pointer and llmd compared names.
`hasKey` scanned every key, yet callers used it as an exact test.
`TensorReader` had two constructors, an ownership flag and two vtables.
About 290 lines parsed safetensors metadata that nothing read.

- `TensorStore.init(allocator, registry, opts)` takes the registry by
  value and locks its map (`lockPointers`): a later insert panics in safe
  builds and is undefined behavior in ReleaseFast. `deinit` frees the
  bindings, then unlocks and frees the registry. `opts` is
  `TensorStore.Options { meshes }`. Tests that built several stores over
  one registry give each a copy (`testing.storeOf`). Callers read the
  registry right before `TensorStore.init`, so nothing fallible sits
  between the read and the move; llmd main now reads it after the
  platform is created (not timed).
- An entry's address is its tensor's identity: bindings point into the
  registry, and Lowering dedupes by entry pointer.
- `View.get(subkey) ?*const safetensors.Tensor`, one hash lookup,
  replaces `getShape` and `hasKey`; `View.hasPrefix` keeps the scan.
  llama's `q_norm` prefix probe became exact `get`s.
- `safetensors.Tensor.read(io, buffer)` reads a whole tensor;
  `readRange(io, file, buffer, tensor_offset)` reads a positional range
  of a caller-opened file. `TensorReader` is gone, and `SeekError` with
  it. 744 lines of dead metadata code left safetensors.zig; a header's
  `__metadata__` is skipped. `fetchRegistry` leaked the registry when a
  parse failed; it frees it now.
- llmd: `sameTensor` is gone; `haveSharedScale` compares registry
  entries, which is exact because an `fp8_per_tensor` scale has one
  element that every view takes whole. `canDoubleScale` streams the
  scale's file 64 KiB at a time (removed by R9).
- Review: the torch2zml howto left registries `var` once their defers
  went, which Zig 0.17 rejects as never mutated; they are `const`.

Open: `Source.pieces` stayed public (private since R13);
`readFilePositionalAllV` and `advanceScatter` stay in safetensors.zig;
#20 unchanged. Checks: `//zml:test` 331, llmd 7/7 (`llmd:test` 262); on
CPU `examples/io` on a 2-shard index whose headers carry `__metadata__`
passes its load (5/5) and pack (1/1) checks.

## Forty-first pass: the drafter joins the model's store (R8) (2026-10-22)

`SpeculativeContext` built a second registry and store for the drafter,
so llmd threaded the target store's meshes into it, and each speculative
model made three starts: base, resharded lm_head, drafter.

- `TensorRegistry.addRepo(io, repo, prefix)` registers a checkpoint's
  tensors under a prefix; a drafter under `"drafter."` is reached through
  `View.withPrefix("drafter")`, which only scopes lookups. `fromRepo` is
  `init` plus `addRepo(.., "")`; `fetchRegistry` became `addCheckpoint`.
- The added checkpoint owns its prefix: a tensor already named under it
  is `error.DuplicateTensor` before anything is read. The first version
  merged with `put` (review finding): a prefixed name equal to an
  existing key silently replaced the first checkpoint's entry, and target
  keys under `drafter.` would have shown up in the drafter's view. Within
  one checkpoint a later shard's duplicate still wins, as before.
- llmd: `speculative.addDrafter` adds the drafter before
  `TensorStore.init` (`"native"`, DeepSeek's DSpark layers in the target
  checkpoint, adds nothing). `drafterStore(target_store)` roots the
  drafter at the target view's root, so a prefixed target view (gemma4's
  `model.language_model`) does not shift it. The five speculative models
  make two starts from one store, before the one `wait`.
  `SpeculativeContext` keeps only the parsed config. main resolved the
  drafter flag for every model type, as the move asked; R11 reverted it.

Open: one start over `Weights{ base, speculative }`: each `start` is its
own submission, so the drafter's reads are planned apart from the
target's; two starts needed the least churn (kept as is by the
forty-ninth pass's audit). The drafter repo is resolved twice (for its
tensors, then for config.json); passing one Dir would change every
`Model.init`. zoo's compile progress total now counts the drafter's
tensors. Not run with a real drafter. Checks: `//zml:test` 332, llmd 7/7.

## Forty-second pass: block-FP8 at bind time (R9) (2026-10-22)

The block-FP8 encoding and scale layout were post-passes that models opted
into. Only qwen3_5 and deepseek41 called `normalizeBlockFp8`, so on CDNA3
every other model bound FN weights against FNUZ activations. The sm_10x
swizzle was a hand list that missed DSpark until R1's stop-gap. The scale
guard (`canDoubleScale`) read every scale tensor at model init: 92,160
serial opens on Qwen3.5-MoE. On Part 3 #4 the user kept the guard, but
inside the conversion program, per block, with no read at init.

- `Quantization.Target{ fp8, swizzled_scales }` and
  `Target.of(platform)` are the one platform switch. `fp8` is FNUZ on
  CDNA3, FN on CUDA (sm_8x included), CDNA4 and RDNA4, null elsewhere
  (CPU included), where block-FP8 activations stay BF16;
  `swizzled_scales` is set on CUDA sm_10x. The store carries it
  (`TensorStore.Options.quantization`, default FN unswizzled;
  `View.quantization()`).
- llmd's `linear.bindBlockFp8(view, scheme, weight, scales)` converts or
  swizzles every block-FP8 Linear while binding it (create, fuse,
  experts). FN weights on an FNUZ target bind `NormalizeBlockFp8Weight`
  and `NormalizeBlockFp8Scale`; e8m0 scales of a matrix on a swizzling
  target bind `swizzleBlockFp8Scale` as a Source expression, rank-2 only,
  so stacked experts keep the layout their MoE backend reads.
- The guard, per block: where a block's scale can double, keep the FN
  bits; elsewhere re-encode to FNUZ, saturating at ±240. Both paths map
  -0 to 0 and both FN NaNs to the FNUZ NaN, since XLA turns -0 and 448
  into NaN when converting to FNUZ. A scale doubles where the result
  stays finite; one bit test, `canDouble`: e8m0 below 254, bf16 and f32
  exponent below max-1, so NaN and infinite scales do not double.
  `expandBlocks` maps a scale grid onto 2-D blocks (partial last block,
  leading expert axes); fp8_block32 is 32x32 tiles.
- An expert stack a program lays out that would need conversion still
  fails with `UnsupportedFp8NormalizationProgram`. The memory rule holds
  by order: the conversion is bound before any `setMemory`, and llmd puts
  no Linear in host memory. deepseek41's wo_a binds `DequantizeBlockFp8`
  once and is not swizzled.
- A CPU test runs both programs on plain and stacked weights with e8m0,
  bf16 and f32 grids (scales that cannot double, NaN, ordinary, a partial
  block): ordinary blocks decode to exactly the FN value times scale;
  blocks that cannot double give the value clamped at ±240 under the
  unchanged scale (XLA CPU runs f8e4m3fnuz). Four deliberate breakages
  each failed it.

Open: on MI300 every block-FP8 model now gets FNUZ weights, and the
per-block re-encode has not run on ROCm (still open after the
fifty-first pass). On GB300 every rank-2 e8m0 Linear built through create
or fuse is now swizzled, including DSpark and any e8m0 deepseek41
compressor, indexer, router or head projection the old list skipped (the
fifty-first pass measured no decode change). A hand-built target with
both FNUZ and swizzled scales would turn the swizzle's zero padding into
1; `Target.of` never returns both. On CUDA, `quantizeInput` asks for the
compute capability once per block-FP8 Linear at trace time. Checks:
`//zml:test` 332, llmd 7/7 (`weights_test` 24/24).

## Forty-third pass: a Source carries its placement (R10) (2026-10-22)

Seven entry points made a model tensor from a key, six said where a Source
lands, and five set its memory kind; `.replicated` dropped its
partitioning argument, which 25 calls passed. On Part 3 #6 the user
reversed "memory is a bind option" (the "Concatenation API: Source
expressions" proposal): the after-bind setter stays.

- A Source carries its placement, set last: `withPartitioning(mesh,
  partitioning)` or `as(shape)`. `bind()` and `bindProgram` are the only
  terminals and take only placed sources; an op on a placed source
  panics. `as(shape)` accepts any shape with the source's element count
  and dtype (looser than `placeAs`, which wanted equal dimensions).
  `TensorStore.bindNew(shape, binding)` mints the tensor.
- `withPartitioning(.replicated, p)` is a compile error unless `p` is
  `.{}`; it removed 24 dead arguments (the review had counted 25).
- `unplaced()`, not in the plan, drops a placement: llmd's block-FP8
  scale swizzle is handed placed scales and reworks them per device.
- `View.setMemory(tensor, kind)` is the one memory setter, called after
  binding. It panics on a tensor a program computes, whose output lands
  in default device memory; qwen4's n-gram tables are concatenations in
  the shards' own dtype, so no program is involved.
- `View.mesh`; model code no longer reaches `view.store`, and the fused
  Linear's store identity check went (one store since R8). Gone:
  `bindAs`, `place`, `placeAs`, `BindOptions`, `sourceOf`, `unbind`,
  `rebind`, `ofBinding`, `createHostPinnedTensor`.
- Review defect: routing `createTensor` through Source made an empty
  checkpoint tensor a composite without parts, so `start` logged "from no
  source has no parts" instead of the key, and the loader test's empty
  case passed by the wrong rule. `maybeCreateTensor` binds it plainly
  again; a doc-only fix was rejected because it would have changed that
  binding.

Open: a whole sub-byte tensor whose bits are not a multiple of 8 now
panicked at bind (`InvalidBinding` since R13; no known checkpoint has
one). A fused Linear over two stores panics instead of returning
`IncompatibleProjections`. `createTensor` builds pieces and a layout it
drops (arena cost not measured). An empty source bound by hand through
`Source.bind` still logs "from no source". Checks: `//zml:test` 333, llmd
7/7.

## Forty-fourth pass: no Packer (R11) (2026-10-22)

After R9 the Packer held only an allocator and the expert layout, and its
`deinit` was empty, yet about 70 init functions took it. The five MoE
models each repeated about 45 lines of expert setup, ministral3 kept a
hand-rolled FP8 Linear, and a checkpoint that did not match failed with
one of four bare error names and no key.

- `linear.fuse(views, out, contracting, partitioning)` replaces
  `interleaveLinears(.per_shard)` and `concatenateLinears`: each device
  the output axis splits over holds its rows of every member, and
  replicated rows are a plain concatenation. The three
  `concatenateLinears` callers were replicated, so they bind as before.
  Members share the scheme, the kinds of global and input scale, and
  having a bias; independent per-tensor scales become per channel.
- `weights.experts(allocator, view, spec)` binds a MoE layer's gate_up and
  down: fused (per expert or pre-stacked), else gate and up apart, else
  `CheckpointMismatch` at the gate_up key. `weights.moeBackend` picks the
  kernel from the scheme and dtype of one projection. qwen4 asks with
  `gate_up_proj`, which broke Qwen3.8-Flash-Next-FP8 (found in the
  fifty-first pass). ExpertLayout keeps both behaviours and the `.device`
  default; the choice is still the user's.
- ministral3 goes through `linear.create` (scaled_dot already ran as bf16
  there): `linear.detect` classifies the Mistral FP8 layout (scalar
  `weight_scale_inv` plus `activation_scale`) as `fp8_per_tensor`;
  `activation_scale` and the emulated activation rounding are gone.
- One `weights.Error = error{CheckpointMismatch}` via `mismatch(view, key,
  fmt, args)`, which logs the full key and reason at err (at debug in
  tests, since the test runner counts logged errors as failures). It
  replaces `MissingFusedComponent`, `InvalidProjectionLayout`,
  `IncompatibleProjections` and `MissingWeight`; conditions only a code
  bug reaches are asserts.
- Review defects: without gate_up, `experts` fell back to gate and up
  even when those were missing and hit an assert that named no key;
  `fusedSource` reported a bias or scale mismatch at the weight's key; a
  public `ExpertSpec.layout` was never read.
- The drafter checkpoint is added only for llama, laguna, qwen3_5, gemma4
  and muse_glimmer (`ModelType.hasDrafterCheckpoint`), undoing R8's
  regression. MoE layers now bind experts before the router, so tensor
  ids changed; the loader works in model order.

Open: the five speculative models still make two starts; one start over
`Weights{ base, speculative }` needs `Bufferized(Weights)` and changes
load planning. Not run on a GPU or a real checkpoint. Checks:
`//zml:test` 333, llmd 7/7 (`weights_test` 26/26).

## Forty-fifth pass: one binding representation (R13) (2026-10-22)

A binding re-encoded its Source as `Layout`/`Part` over a deduplicated
tensor list, and the front end lowered that again to `LoadPart`s. Checks
were spread over the front end, both backends and the planner: `zero_fill`
only existed so that a size mismatch surfaced in the backends, an overlap
prover checked what Source already guarantees, and the planner checked
device ids after earlier files were published (#28). Part 3 #7 approved
dropping the overlap check: only Source makes bindings. Also LOADER_REVIEW
Moves 3, 7 and 9.

- `TensorStore.Binding` is a union: `load: { source, memory }` or
  `compute: { program, inputs }`, each source a `Source.Bound` (placed
  shape and pieces, no store). `sameExecutable` moved to
  `Binding.Compute`. The load arm keeps the bound shape: a `pad` that
  adds only trailing rows keeps the pieces and grows the shape, so the
  size rule cannot be read from the pieces.
- `Source.lowerPiece` is the one bits-to-bytes rule, called as the front
  end builds each `LoadSpec`. A run that starts or ends inside a byte is
  `InvalidBinding` at start (a whole u4[4] loads, a u4[3] does not).
- The front end checks every tensor and program input once, before
  anything is published (`prepareSpec`, `checkPlacement`), each failure a
  logged `InvalidBinding` naming its target: no pieces or a piece of an
  empty tensor; bound bytes differing from the tensor's; a sub-byte run;
  placement in any memory space, computed tensors included; a mesh
  device id at or above the platform's count (#28); a memory kind the
  platform cannot take; strided parts into host memory; parts out of
  bounds or order (`backend.checkParts`).
- Backends trust that contract: `zero_fill`, the overlap prover and the
  direct backend's own checks go; the buffered backend always zeroes the
  gaps. The read-limit check moved to `Planner.checkGatherReads`; the
  copy in `appendExtent` is an assert, unchecked in ReleaseFast. Part
  dimensions live in a per-start arena, so both backends copy them at
  submit. Source's test harness fails on a byte two pieces write
  (`error.DoubleWrite`).
- Move 9 still made sense with program groups gated: only the backend
  that allocates knows what is still to land. The direct backend reports
  `unlandedBytesPerDevice`, which `readRoom` reads, and a group's
  temporaries are charged as their per-device maximum, not their sum
  (admits more than before). Fixed `[MAX_NUM_DEVICES]` arrays were not
  taken: per-device slices leave the buffered backend's device ids
  uncapped.
- Backend seam: one `create(allocator, io, platform, opts)`, `inline
  else` forwarders, `loadsDirect` the one platform list; the union stays
  in backend.zig (Part 4). `Platform.placedShape` replaces three helpers.
  `Source._pieces` is private by convention only; `Bound.whole()` says
  whether a binding reads one checkpoint tensor whole.
- Review defects: an empty tensor bound as a program input had no piece,
  so `start` logged "reads no checkpoint bytes" with no key and the
  output's id beside the input's shape; `Source.emptyOf` gives it one
  piece, and the log names `program input {i} {shape} of tensor {id}
  {shape}`.
  `checkOutput` rechecked the tensor's own placement; it now panics
  there and checks only the program's output.

Open: a source that reads nothing after `pad` or `slice` still logs no
key; a fake piece for an empty entry would break `pad`, `slice` and
`concatenate`. Dropped with decision 7: the hand-built cases Source
cannot make (overlapping parts, parts past their source). The
misplaced-program-output case runs only with 2 or 4 devices. Not run on a
GPU: the unlanded-bytes room (CPU reports no limit), host-pinned routing.
Checks: `//zml:test` 333, llmd 7/7; on CPU, `examples/io load` of a
synthetic 52-tensor checkpoint, `ZML_LOAD_CHECK=1`, 2 packs of 4: load
44/44 and pack 2/2 replicated, sharded and tp (the last two also
host_pinned); tp with 3 packs, one group each: 3 admission retires,
40/40 and 3/3.

## Forty-sixth pass: one root in the direct backend (R14) (2026-10-22)

The first part of LOADER_REVIEW Move 4. The direct backend had two roots:
`Pipeline.init` took 10 parameters and copied the Loader's allocator, io
and platform, with back-pointers to its read gate, credits, metrics and
scheduler. `Sizing` restated the width and blocks per job the Loader also
stored. A Job spread seven kinds over four flags, and the Workspace was
five calls in a fixed order whose ceiling arithmetic its owner did. No
behavior change was asked for.

- `Pipeline`, `Sizing` and `pregrow` are gone. The Loader holds the
  sticky error (`recordError`, `errorValue`; `StickyError`), the pumps
  and the DMA budget. `create` computes request size, reserve and width
  as locals and stores each once (`read_gate.limit`,
  `credits.blocks_per_slot`, `plan_config`). It copies the local
  `workspace` and `credits` into the Loader while their errdefers run on
  the locals, which is right because the copies share buffers.
- A Job is `{ file_offset, transfer_start, transfer_len, kind }`, the kind
  a tagged union (`dma_read`, `dma_staged`, `dma_gather`, `dma_fill`,
  `host_direct`, `host_bounce`, `host_fill`) with only its own fields; the
  block count is computed (`Job.blockLen`). Fill jobs, at offset zero,
  are appended after the sort and claimed last. A failed request logs the
  kind beside offset and length (review kept the length, needed to check
  a short file's end-of-file error against its size).
- The Workspace is `{ init, deinit, allocate, capacity, carve }`.
  `carve` maps the reserve as its own arena, then the rest, and cuts
  every arena into blocks; after it, `allocate` or a second `carve` fails
  a debug assertion. `findArena` was always null, since both callers
  calibrate on a fresh workspace, and went.

Open: the rest of Move 4 (fiftieth pass). Not run on a GPU: real DMA,
ROCm's two-arena split. Checks: `//zml:test` 334; on CPU the R13 runs
give the same checks and log request 8 MiB, block 4 MiB, width 16,
alignment 4096, 33 credits, two arenas of 128 and 140 MiB; only the
credits high-water mark varies.

## Forty-seventh pass: the loader meets the VFS only through io (R15) (2026-10-22)

With the delete-only stages of LOADER_REVIEW Move 10. The loader read the
VFS's read statistics through four types to feed one debug line. One
alignment per loader, from the checkpoint's profile, gated a decision the
VFS makes per file. The five retry knobs were declared four times, and
each governed request spelled its `RequestSpec` and context twice.

- Stats stay in the VFS: each remote backend logs its read totals once,
  at debug, from `deinit`, if it sent a request (in llmd, at shutdown).
  The per-batch "batch source" line, which double-counted overlapping
  batches, and the loader's stored `load_profile` are gone.
- Direct I/O per file: `VFS.direct_io_alignment` (0 where unsupported)
  and `VFS.isLocal(path)` (opens nothing) give the loader its alignment
  (`backend.readAlignment`). A remote file is planned exact and never
  opened by the planner. Visible: under a VFS a local file may now be read
  directly, with no checkpoint or beside a remote one; the loader-wide
  alignment is 4096 for every checkpoint (was 0 for a remote one), so
  only the gather ranges of a plan with remote files are 8 KiB shorter;
  the 4 KiB minimum block applies under a VFS whatever the checkpoint,
  and not with `direct_io = off`.
- One `request.Retry` (the five knobs plus `stall_timeout`, settable per
  backend; range asserts in `Governor.init`) and one `init(allocator, io,
  client, options)` for the four remote backends. An `Attempt` carries
  the spec. GCS's dead JWT and endpoint code went; S3 signs through one
  `Signature.sign` (was three copies).
- Review defects (both tests fail with the fix reverted): the
  strided-part check before publication used the loader-wide alignment,
  now 4096 under any VFS, so a remote strided part whose exact span fit
  the read limit but not once widened to 4 KiB was refused although read
  exact; it now uses each file's alignment. `init` let a registered
  backend with a 4 or 8 KiB read chunk and an equal block through, and
  set-up then panicked on a GPU; `init` refuses a request of at most two
  alignment units.
- A new test fails the first worker start
  (`std.Io.failingGroupConcurrent`), unwinding carve, credits and pumps.
  A FailingAllocator index was not used: the workers allocate as soon as
  they start, so a count read after `create` races them.

Open: Move 10's handle table (remote backends as stores; high risk, to be
staged with `http_acceptance_test` through a VFS), removing `retry_once`
(it changes HF's 401/403 handling), GCS's behavior stages and the VFSBase
forwarders; the forty-ninth pass's audit kept all four. `create`'s
`maximumJobLen` check is reached only by a caller that skips `init`.
Not run against a real remote; the totals line was never seen printed.
Checks: `//vfs:test` 49, `//zml:test` 336 (both `--runs_per_test=3`); on
CPU, `ZML_DIRECT_IO=on` logs "reading direct" and alignment 4096, `off`
alignment 0.

## Forty-eighth pass: docs (R16) (2026-10-22)

R16 of `LOADER_API_REVIEW.md` (K-docs-one-place-per-contract,
K-newcomer-recipe, M2-source-doc-contract, J-lifetime-graph); zml
`szunznok`, monorepo `ulnsvlvu`. Each loader contract was written up to
four times (loader.md, the doc comments, CTX "Current design", private
comments) and the copies disagreed; two-thirds of loader.md described
internals, and no tutorial loaded a checkpoint.

zml:
- The pub doc comments are the contract (`Loader`, `TensorStore`,
  `Source`, `Quantization.Target`, the VFS, the registry): each says what
  it borrows, for how long, and what fails and how. The loader borrows
  the platform and `io` (with the VFS behind it, which keeps the source
  files open until `deinit`) for its life; file paths are copied.
  Source.zig's module doc states the one-store rules: pieces live in the
  store's arena, the store must not move, sources of two stores panic in
  `concatenate`, `stack` and `bindProgram`, placement comes last,
  `bind`/`bindProgram` are the only terminals.
- docs/learn/loader.md is a guide (recipe, placement, composites,
  programs, a borrow table, actionable errors); its internals moved to the
  `//!` headers that own them (loader.zig, direct_loader.zig with the
  DeepSeek loader-versus-program numbers, host_memory.zig,
  dma_calibration.zig, buffered_loader.zig, vfs/direct_io.zig, log.zig,
  zml/io.zig). CTX "Current design" points to them. write_first_model
  gains a "Loading weights" step that writes a checkpoint with Python and
  loads it.
- `TensorRegistry.fromPath` takes a relative file path. It opened the
  full path inside the file's own directory (`ckpt/layer.safetensors`
  looked for `ckpt/ckpt/layer.safetensors`, `FileNotFound`; a bare name
  was `InvalidPath`). It opens the base name now, a bare name's directory
  is the cwd, and a denied local directory or entrypoint is
  `PermissionDenied`, as on a remote.

monorepo: AGENTS.md's `errdefer if (fut.cancel(io)) |value|
value.deinit(...)` gains a caveat: after the await, `cancel` returns the
future's stored copy, so the idiom is wrong for a by-value handle that
changes once the future completes (examples/llm's `Loader`).

Copies that disagreed, settled on what the code does:
- Unmeasured admission (no device limit, or the buffered backend) keeps
  one program group in flight; CTX and loader.md had said every pending
  submission was retired first.
- A denied local file is `AccessDenied` (EACCES through
  `std.Io.Threaded`); `PermissionDenied` is a remote's 401 or 403.
- Device memory that runs out is PJRT's `ResourceExhausted`; only host
  memory is `OutOfMemory`.
- A truncated checkpoint fails `fromPath` with `InvalidSafetensorsHeader`.
  The loader's `UnexpectedEndOfFile` means a file changed after its header
  was read, or a shape needs more bytes than its `data_offsets` give.
- The direct backend's one-read check of a broadcast or transpose runs
  when its submission is published: in `start` for the tensors the loader
  fills, in `wait` for a program's inputs.
- Also: `dma_block_sizes` is read by `init`'s task until the first
  `start` or `deinit`; `registerBackend` keeps its scheme string
  uncopied, and `io()` points into the VFS, which must not move; slices
  are static and contiguous, padding zeros only; llmd makes several
  starts from one store, not several stores per loader.

Review (refuting verifier): nineteen findings, ten distinct, all fixed
(`fromPath`'s relative paths in the code, not the doc).

Not done:
- The tutorial has no build target. Both versions ran on CPU from a
  temporary `//examples/simple_layer` (`{6,11,16}`), since deleted; the
  code can drift.
- "Current design"'s planner, pinned-block and scheduling sections keep
  sixteenth-pass names (`DirectLoader.submit`, `RequestContext`,
  `metadata_mutex`); they are only marked.
- In `fromPath`, a denied local file opened by name (given, or listed by
  an index) is still `AccessDenied`; the doc says so.
- loader.md is 205 lines, past the 200 allowed. mnist and the README
  still passed `.read_parallelism = 1` (fixed in the forty-ninth pass).

`//zml:test` (337, 12 skip), `//vfs:test`, `//examples/...` and llmd
7/7 pass; no docs build or lint target exists.

## Forty-ninth pass: follow-ups from the review audit (2026-10-22)

An audit of `LOADER_API_REVIEW.md` against the code after R1-R16 ranked
what was left: eight groups to do, the rest kept as is with a reason.
This pass does its "Do now" items 1, 2, 4, 6 and 7 (3, 5 and 8 are the
fiftieth pass). zml `zklqvywm`, monorepo `nykmyswz`.

zml:
- Tensors and buffers pair by structure. `mem.pairBufferized(Model,
  model, bufferized, report, context, callback)` walks both trees like
  `bufferize` and hands each (tensor, buffer) pair to the callback in
  `meta.visit` order. At the first difference (another union arm, an
  optional set on one side, slices of other lengths) it stops with
  `error.Mismatch`, and `report` holds the field path, as
  `.ffns[0].dense.weights: a slice of 1 in the model, of 2 in the
  buffers`. Tuple fields are renumbered as `MapRestrict` does; a
  pointer-to-one field is a compile error.
- `prepareModel` runs that walk first. A mismatch logs `buffers shaped
  unlike the model at model<path>` at err and is `InvalidBinding` before
  any buffer is emptied, so R1's guarantee holds. Before, two trees with
  as many tensors in another structure passed the count assert and paired
  a tensor with another's buffer. The count assert and both visit-order
  walks are gone.
- `TensorStore.getSourcesById` is `bindingOf` (B-keep-binding-seam), in
  zml, examples/io and llmd.
- `Loader.deinit` warns `Loader.deinit cancels loads started without a
  wait` when `loadsStarted()`. A failed or canceled loader has none left
  (the review found a `failure == null` test dead), so the warning names
  a caller that skipped both `wait` and `errdefer loader.cancel()`.
- `zml.testing.testLayer` is compiled at last (`countWithPrefix` gives
  way to `View.count`); a CPU test runs a one-weight `Scale` layer.
- examples/llm: `LoadedModel.loadBuffers` in models.zig bufferizes,
  starts, waits (errdefer `deinitBufferized`) and unloads for every
  model, keyed on `@TypeOf(m.inner)`; the four per-model copies are gone.
- Smaller: an alignment overflow logs the file, the read's end and the
  alignment before `InvalidTensorRange`; mnist and the README call
  `Loader.init(..., .{})`; the HF how-to names `HF_TOKEN`, which
  `vfs/hf.zig` reads; two "Current design" sentences were corrected.

llmd:
- qwen4's n-gram table is one `Source.concatenate` of its
  `shard_{d}.weight` sources along `.voc`, then `withPartitioning(.tp_mesh,
  .{ .voc = .model }).bind()`. `weights.concatenateRows` and the
  `ConcatenateRows` program are gone; the test checks the same bytes,
  placement and memory (device and host_pinned); one shard is now
  accepted (`concatenateRows` asserted two).
- Experts whose FP8 weights a program would have to convert log the first
  expert's key, the scheme and the target FP8 type before
  `UnsupportedFp8NormalizationProgram`.

Review (refuting verifier), four findings, all fixed. In the mismatch
test every mismatch was the first tensor met, so "no buffer written" held
for free; each tree now starts with a matching `head` tensor (with the
emptying moved into the callback it fails `expected 0, found 1`, with the
old `prepareModel` `expected error.InvalidBinding, found void`).

The audit kept as is, among others: one start over base and drafter,
the FNUZ conversion of experts laid out by a program, the GCS
service-account refusal, `retry_once` (removing it changes HF's 401/403)
and the VFS forwarders (handle-table move). Part 3's #4, MI300 accuracy
after R9, needs a GPU.

Not done:
- `testLayer` logs a mismatch and returns no error, so its test shows
  only that it compiles, runs and matches; making it fail is a behavior
  change for the user to decide.
- Untested logs: the `InvalidTensorRange` overflow (a read ending within
  one alignment unit of the u64 maximum) and the FP8 expert log (an FNUZ
  target).
- LOADER_*.md still say `getSourcesById`; `bazel/huggingface.bzl` reads
  `HUGGINGFACE_TOKEN`, which nothing loads; loader.md is 207 lines.
- The `deinit` warning also fires when the caller's own code fails
  between `start` and `wait` without the `errdefer loader.cancel()`
  loader.md asks for.

`//zml:test` (341, 12 skip), `//vfs:test`, `//examples/...` and llmd
7/7 (`weights_test` 26/26) pass.

## Fiftieth pass: VFS fixes and the rest of Moves 1 and 4 (2026-10-22)

The audit's items 3, 5 and 8. zml only (`lxltkwot`): llmd builds and
passes unchanged.

VFS (item 3):
- A streaming read in http, hf, s3 or gcs that fails after its retries
  logs at err and returns `InputOutput`; it ended with `EndOfStream`, so a
  reader saw a short file. `Canceled` passes through.
- `fileSeekBy` saturates at both ends in all four backends and takes
  `@abs` of the offset: `@intCast(-x)` overflowed on `minInt(i64)`, and
  http and hf did not saturate.
- A remote path is read-only: `writableDir` refuses create, delete,
  atomic create and `dirCreateFile` on it (from cwd or a directory opened
  on the remote), and `dirOpenFile` any mode but `read_only`, each with a
  logged `ReadOnlyFileSystem` before any request. No remote backend
  implemented these: `dirCreateFile` reached the local io as `bucket/key`,
  the others as the local path `s3:/bucket/key`, and an open for writing
  succeeded, its writes reaching the inner io with the backend's handle
  index as a local fd (index 1 is stdout).
- A local path relative to a directory opened through the VFS gets that
  directory's real descriptor, not the VFS's handle number.
- `stripScheme` cuts only a `scheme://` prefix; a path from cwd without
  `://` is local. `std.Uri.parse` had taken any `name:` as a scheme:
  `notes:old.txt` became `d.txt`, `a:b` sliced past its end, and from cwd
  such a name was `Unexpected`.
- `base.scopedLog` logs err at debug in test builds, as `zml/io/log.zig`
  does, so a failure-path test can pass. The four backends use it;
  vfs.zig keeps `std.log.scoped` (`refusal_log` for the refusals),
  because moving it hid a misaligned direct read (`refuseDirect`) in
  tests.

Direct backend, the rest of LOADER_REVIEW Move 4 (item 5), no behavior
change:
- The Scheduler's queue is a `std.Deque`; `submit` stamps `sealed_at`
  after `publishFiles`, before `finishJobs(1)` drops the sentinel;
  `logBatch` counts from `batch.plans`, only under debug logging; a
  gather job has one item per extent.
- The allocation task walks the host jobs' transfers in claim order and
  skips a destination already created. It logs `host_destinations=44,
  ensured=N`; N, what it created or waited for, was 26 to 37 of 44 on
  CPU, the workers creating the rest.
- No scatter term in the job length: `maximumCoalescedJobBlocks` is gone,
  `maximumJobLen` is the request size less two alignment units,
  `readLimit` returns `request_size`. `Staged.data_offset` stays for
  `Job.blockLen`.
- `recordError`'s doc says why failing other batches is safe from any
  thread: claims are FIFO and a queued plan holds a unit of its batch
  (`fail`); `retireBatch` holds every pump lock while each queued transfer
  holds its request (`abortReady`); `destroy` waits for the worker group.
  Batches do not retire in publish order (a test awaits newest first, and
  `submit` awaits a partly published batch at once), as the doc had said.

LOADER_REVIEW Move 1's host write half (item 8): the copy path
(`share_host_allocations`, `HostWriters`, `HostScatter.Copy`, `addCopy`)
and `sharesHostMemory`, true on every direct target, are gone; its
reasoning is `initHost`'s doc. `initHost` finds each box's devices from
`grid.writers`, which share one allocation; `hostLanding` writes a
transfer once, through its first writer's shard, and counts `len *
popCount(writer_mask)` bytes. `supportsMemory` was left for R12.

Review (refuting verifier): six findings, all fixed, among the points
above.

Not done:
- Rename, symlink and hardlink to a remote path still reach the inner io
  as a local path (`s3:/bucket/key`), VFSBase's other forwarders hand it
  a backend's handle index, and atomic create of a local path returns the
  inner io's handles: the handle-table move. An unregistered scheme is
  `Unexpected`, like every other VFS call.
- Without the scatter term a custom `dma_block_sizes` block under 32 KiB
  makes longer jobs; each read is still cut into `IOV_MAX` batches. The
  default blocks are 2 MiB or more.
- hf's streaming and seek fixes are untested (its URLs are fixed).

Checks: `//vfs:test` (52; new tests on the mock server for http, s3 and
gs, for local create and delete with `notes:old.txt`, and for remote
writes refused with no request and nothing on disk), `//zml:test` (340,
12 skip), `//examples/...`, llmd 7/7. On CPU, `examples/io load` with
`ZML_LOAD_CHECK=1` passes every load and pack check (replicated, sharded,
tp, host_pinned, `ZML_DIRECT_IO=on`/`off`) with the parent's sizing and
planning lines. `ZML_DMA_BENCH_BLOCK_MIB=64` fails `InvalidOptions`;
`examples/io cp` to `s3://bucket/key` fails `ReadOnlyFileSystem`.

## Fifty-first pass: host_mapped (R12) and the rework on GB300 (2026-10-23, gb300-1)

R12, the last move of `LOADER_API_REVIEW.md`, and the first GPU runs of
the reworked loader and llmd. zml `zoyytxpt`, monorepo `zllzwuwx`.

R12:
- `TensorStore.Memory` (`default`, `host_pinned`, `host_mapped`)
  replaces the PJRT memory kind on bindings, `LoadSpec` and `setMemory`;
  `host_unpinned` and `device` are no longer loader targets.
  `supportsMemory` takes the host kinds on the direct pipeline only, where
  every device has a pinned host space.
- A `host_mapped` destination is written like a `host_pinned` one,
  through `SharedHostAllocation`, but every shard is a view of the
  allocation in the device's default memory space. The loader keeps no
  owner: the allocation goes with the last view's deletion callback. The
  buffered backend never receives it.
- llmd: deepseek41's engram tables and qwen4's offloaded n-grams bind
  `.host_mapped`; the swaps and owner lists
  (`swapEngramHostPinnedBuffers`, deepseek41's `finalizeLoadedBuffers`,
  qwen4's `createView` swap) are gone. `extra_device_memory_bytes` stays:
  XLA still counts the views as device memory.
- The Part 3 precondition (a device view of loader-owned pinned memory
  whose deletion outlives its readers) holds on CUDA. The test frees the
  table while a gather queued behind eight 4096x4096 products still has
  to read it, then allocates pinned decoys of its size: on two and four
  GB300, in both placements, the gather was pending at the release and
  returned the right rows; the shared allocations were released after.
- Residency checks count device allocations (`num_allocs`) after
  `awaitSetup`. Bytes in use failed on four GPUs: the CUDA plugin reports
  its pool (`bytes_in_use` = `pool_bytes` = `bytes_reserved`), in which
  freed memory leaves room, and the set-up's DMA calibration allocates
  about 512 MiB on its device after `Loader.init` returns.

zml suite on CUDA (`bazel run --//platforms:cuda=true //zml:test`, four
GB300): 352 tests, 322 OK; every io test passes, the host-resident one
included (`rows` is `host_mapped`). Failing, outside the rework:
- tests that build CPU devices: the CPU PJRT plugin ships for x86 Linux
  and macOS only, not aarch64 Linux;
- `moe.tests`: the FlashInfer matrix fails `MoeComplianceFailed`; the
  Triton MXFP4 matrix panics in `validateInputs` ("Invalid MXFP4 tensor
  layout") and the CuTe MXFP4 one in MLIR verification of
  `make_non_exec_tiled_tma_load`. Each panic aborts the binary; the run
  reached 352 by skipping both by hand.

llmd, saved binaries over one frozen runfiles tree: `new` is the pass's
tree; `pre` is the commit before the API rework (zml `kskxyxur`,
monorepo `kvklzrov`); `base` is zml `omuqxqpu`, monorepo `rwqvkpqy`, 58
commits back (the twenty-fourth to twenty-sixth passes' commit).
- DeepSeek-V4.1-Flash, GPUs 2,3, engram tables `host_mapped`: "Paris"
  and "391" on every run. Load over five interleaved pairs: `pre`
  10.13-10.63 s (median 10.45), `new` 9.99-13.08 s (median 10.55);
  `new` slower in 4 pairs of 5, median +0.22 s (+2%), which five pairs
  cannot resolve (the thirty-third pass needed 15). Decode at batch 1:
  6.07-6.13 ms per step on `pre`, 6.14-6.21 on `new`; 512 tokens in
  3.12-3.28 s on every tree, `base` included, so R9's newly swizzled
  projections change nothing measurable. `base` loads in 1m27-1m30.
- Qwen3.8-Flash-Next (BF16), GPUs 0,1, `--cache-memory-fraction=0.6`:
  47.68 GiB of n-grams `host_mapped`; 329.58 GiB in 18.6 s; answers.
- Qwen3.8-27B-FP8 with the z-lab DFlash2 drafter (R8's store), GPUs 0,1:
  33.40 GiB in 1.1-1.4 s on every tree; a decode step costs 36.5-37.3 ms
  on `new` and `pre` alike; 512 tokens take 5.1-7.4 s, following the
  accepted drafts per step (2.77-3.22).
- At temperature 0 the text is fixed within a process but changes between
  processes of the same binary, on every tree: compare trees by the
  decode metrics (`/metrics` deltas), not text hashes.

Found: Qwen3.8-Flash-Next-FP8 loaded on no tree (fixed in the
fifty-second pass). CPU checks pass. Not run: MI300 (R9's FNUZ
accuracy), a remote checkpoint.

## Fifty-second pass: Qwen3.8-Flash-Next-FP8 loads (2026-10-24, gb300-1)

monorepo `oxtrzwyy`, on zml `zoyytxpt`.
- qwen4 picked its MoE kernel from `experts.gate_up_proj`; the FP8
  checkpoint stores `gate_proj` and `up_proj` per expert. It reads
  `down_proj` now, like the other models (`weights.experts` already
  combines split projections; CUDA picks Triton for `fp8_block128`).
- Loaded, it generated "!!!!": `NgramEmbedding` ignored
  `ngram_embedding.weight_scale`. The FP8 checkpoint stores the n-gram
  tables as e4m3 values times one BF16 per-tensor scale (its rows equal
  the BF16 checkpoint's times 0.000199), the checkpoint's only per-tensor
  scale; the rest is block-FP8 `weight_scale_inv`. `lookup` returns the
  gathered rows times the scale, in f32, then the scale's dtype.
- Tests: the expert-layout test reads the kernel from each layout's down
  projection; a new qwen4 test looks up an FP8 table's rows on two CPU
  devices (without the scale: 1.75 for 0.4375).
- gb300-1, GPUs 2,3: 169.41 GiB in 3.8-4.3 s; "Paris", "391" and a
  coherent long answer; 11.65 ms per decode step at batch 1, 13.55 with
  the n-grams `host_mapped` (23.84 GiB per device,
  `--cache-memory-fraction=0.3`). llmd 7/7 and its build pass.

## Open work

Status on 2026-10-24: R1-R16 and the audit's "Do now" items are in. The
fifty-first and fifty-second passes' GB300 runs closed the GPU and
real-model runs the 2026-10-22 passes left open (R5, R7, R8, R10, R11,
R14, host-pinned paths, qwen4's n-grams, R9's GB300 speed; R9 had
replaced R1's DSpark swizzle list) and the thirty-fourth pass's DeepSeek
measurement with group reads after the bulk (10.55 s against `pre`'s
10.45 s, within noise).

Commits of the 2026-10-22 passes, recorded only here (pass, move, zml /
monorepo): 35th R1 `zmxzvvly`/`szqpusxv`; 36th R3 `rktvmymx`/`yyqsnszp`;
37th R4 `ryvvruus`/`vrounrwt`; 38th R5 `vyswqxpw`/`sltmxrzn`; 39th R6
`zvrossur`/`wlwvxtwp`; 40th R7 `orqlttun`/`zmrvtnlx`; 41st R8
`unqksvss`/`vvpmvkl`; 42nd R9 `urqymvvl`/`sukonzkw`; 43rd R10
`nuulylpo`/`uxxqmoxo`; 44th R11 `twuoxmxk`/`moptmxsw`; 45th R13
`lnnnonkm`/`xmvssoyq`; 46th R14 `nxtvovxr`; 47th R15 `lluzuwnv` (zml
only).

Loader and llmd:
- MI300 / ROCm, not run since the rework: R9's block-FP8 accuracy on
  CDNA3 (every block-FP8 model, re-encoded blocks especially), the FNUZ
  conversion of experts laid out by a program, the FP8 expert log
  (forty-second, forty-ninth passes).
- GB300 MoE kernel tests and the aarch64 CPU plugin (fifty-first pass).
- No real remote (HF, S3, GCS, HTTPS) since R5, R6 and R15: the remote
  totals line was never seen printed, hf's streaming and seek fixes are
  untested (thirty-eighth, thirty-ninth, forty-seventh, fiftieth passes).
- The handle-table move (LOADER_REVIEW Move 10, remote backends as
  stores; high risk, to be staged with `http_acceptance_test` through a
  VFS): rename, symlink and hardlink to remote paths, VFSBase's
  forwarders, atomic create's handles (forty-seventh, fiftieth passes).
  Also from Move 10: GCS fetching its metadata token as a plain request.
- `testLayer` cannot fail; the user's call (forty-ninth pass).
- Docs: the tutorial has no build target; "Current design" keeps
  sixteenth-pass names in its direct backend sections; loader.md is 207
  lines; LOADER_*.md say `getSourcesById`; `bazel/huggingface.bzl` reads
  `HUGGINGFACE_TOKEN` (forty-eighth, forty-ninth passes).
- Known limits, documented: a denied local file opened by name in
  `fromPath` is `AccessDenied`; the `deinit` warning fires on a caller's
  own failure without `errdefer loader.cancel()`; a source that reads
  nothing after `pad` or `slice` logs no key, and a whole sub-byte tensor
  whose bits are not a multiple of 8 is `InvalidBinding` (forty-fifth
  pass); a `dma_block_sizes` block under 32 KiB makes longer jobs
  (fiftieth pass).
- Not checked on their own on a GPU (may be stale: see fifty-first
  pass): the unlanded-bytes room, temporaries charged as a maximum,
  host-pinned routing (forty-fifth); a model's free path (thirty-sixth);
  a progress bar watched to the end (thirty-seventh).
- ExpertLayout (`.loader` or `.device`) stays the user's call
  (forty-fourth pass).
- LOADER_REVIEW items open since the thirty-first pass: S3 credentials
  and key encoding (#5, #12), https redirects (#13), GCS key files (#16),
  and the rest of #20's header checks (a tensor's byte size against
  `end - start`, `F8_E5M2`, duplicate names across shards; other missing
  or mistyped fields still panic; thirty-fifth pass). #6 went with R6, #8
  in the thirty-fifth pass; #21 is moot (`loadExecute` went in the
  thirty-fourth).
- Laguna under zio: the zio coroutine stack overflow compiling Laguna
  (twenty-fourth pass) is likely fixed by the thirty-first pass's
  `vqkkvptr` and `lmnpvsso`; unverified, since the B70 rule (threaded io)
  excludes a zio run there.

DeepSeek and calibration (2026-10-20, 10-21):
- Twenty-ninth-pass profile: a 2.4 s mid-load stall while the plugin
  registers the 189 GiB engram tables (`cuMemHostRegister` under the
  driver lock; up to 20 s when node 1's page cache had to be reclaimed,
  thirty-third pass); otherwise read-bound at ~50 GiB/s; O_DIRECT re-read
  44 GiB of cached data; NCCL init holds a CPU for ~35 s (no sm_103
  code). Next steps are listed there. R12's `host_mapped` allocates
  through the same `SharedHostAllocation`; not re-profiled.
- The calibrated request size (thirty-third pass): on GB300, 8 MiB sits
  on the DMA screen's 8% tolerance line, so the choice flips (4 of 10
  replicated and 2 of 10 sharded Qwen runs, 7 of 48 DeepSeek runs), and
  8 MiB costs Qwen3.8-27B 25-30% (2.54 against 2.04 s replicated). The
  B70s want 8 MiB (10-30% faster); no fixed size suits both, and the
  calibration measures DMA only. The rule also chose 2 MiB on a degraded
  MI300. Calibration caching per host/plugin, or re-screening on an
  implausibly low rate, is open. The user declined a fixed block size
  (thirtieth pass): do not propose removing calibration.

Third-pass items left open; `PLAN.md` holds the checklist. Items whose
setup changed since are marked.
- Nineteenth pass (may be stale: measured before the twenty-eighth
  pass's device layouts and the thirty-fourth's bulk-first order): ~1 s
  of H2D stalls behind the pack executes (PJRT's `kComputeSynchronized`
  allocation sync points; `kAsynchronous` measured 5.9 -> 4.9 s in the
  fork, needs a free-semantics review), 0.85 s waiting for the first pack
  compile, tiny scale pieces draining serially, 2x replication of packed
  experts. `dma_bench` and `live loader ready` log at debug on scopes
  llmd does not enable. A depth derived from the measured rate is written
  up, not applied (worth ~2%).
- Thirteenth pass, llmd: allocate the KV cache after the weights (may be
  stale). `loadPacked` went in the thirty-fourth pass, and with it the
  pack executables' compile in `Model.init`.
- MI300: the 2026-09-04 "host degradation" was a stale ROCm plugin; the
  Laguna window measurement (task 5) can be retaken once the plugin fix
  lands (`platforms/rocm/rocm.bzl` carries a machine-local `file://`
  override that must not be committed). The MI300 checkouts were on the
  `loader-third-pass` branches (zml `67464f3c` or later, monorepo
  `d426dde4`; may be stale).
- `Buffer.toSliceAlloc` places sub-byte shards by element stride, so a
  4-bit tensor partitioned over several devices lands at twice its offset
  (out of bounds in safe builds; the load check skips them; fifth pass),
  and `toSlice`/`toSliceAlloc` misread a buffer placed on a sub-mesh (they
  walk the platform's full physical mesh; twentieth pass).
- oneAPI: master's plugin `202610011939` serializes piece completions (3x
  on tiny-piece loads) and segfaults compiling llmd's Llama; master's moe
  Triton test aborts on Level Zero. Host-resident destinations are
  allocation-bound there (twenty-second pass, "Left open"; its CUDA
  `on_delete` point is settled by the fifty-first pass).
- One oneAPI abort, unexplained (2026-09-04, B70 `level_zero:1`, S3Proxy
  at 20 ms and 200 MiB/s per request, width climbing to 128):
  `host_to_device_transfer_manager.cc:342 Check failed:
  definition_events_[buffer_index]`, from `SetEventAsError` through the
  pump's `onReady` (a transfer error, then the pump's next piece into the
  same manager); four further runs completed. The manager nulls the event
  under its own mutex before our callback runs, so no loader-side flag
  closes the window (fifth pass). If it recurs, stop submitting into a
  manager whose event errored and surface the plugin error.
- `--test_filter` is ignored by the rules_zig test runner.
- Backpressure is process-global and load-untagged; real AWS runs need
  credentials this machine lacks.
- CPU: the direct path is 14% slower than the buffered one on
  Llama-3.1-8B sharded (parity on Qwen3.5-4B); the pump is saturated by
  4 KiB first-touch faults of the plugin's device buffers. Not
  attributed; THP `always` is the untested lever.
- Ninth pass: check `TpuClient::CreateBuffersForAsyncHostToDevice` on a
  TPU host and move TPU to the direct path with pinned arenas (`DmaMap` on
  our pages, or PJRT `pinned_host` buffers as ROCm does). Neuron and metal
  are unchecked; the buffered backend stays until then.
- Longer term: calibration caching, cross-platform 24/32 MiB
  measurement, completion-aware local pacing, and a packed-device-buffer
  redesign if DMA submissions must drop below roughly one per tensor.

## Suggested upstream decomposition

May be stale: it names the adaptive source width (gone since the
fifteenth pass) and `load`/`loadInto` (gone since the thirty-fourth).

1. Exact safetensor positional scatter plus VFS Range/retry conversion and
   removal of backend `parallel_read`.
2. Plugin pin containing already-upstream pinned-range detection; ROCm
   arenas through standard PJRT `pinned_host` buffers.
3. Platform-owned DMA arenas, NUMA allocation, and `DmaBlockPool`,
   independent of calibration policy.
4. Model-wide coalescing planner/scheduler/pipeline and `load`/`loadInto`
   migration with conservative fixed DMA settings.
5. DMA block calibration and fixed per-device width eight.
6. Adaptive source width and aggregate VFS feedback.
7. `LoadProfile` and model-wrapper plumbing.

## Cross-machine handoff rules

- Treat commit IDs, branch divergence, plugin archives, available GPUs,
  model paths, warm cache, and background load as ephemeral. Inspect them
  anew.
- Do not rely on `/tmp` benchmark/perf artifacts being present elsewhere.
  Preserve user-owned recordings when they exist; do not overwrite
  generic `perf.data` or `perf.data.old`.
- Verify that a ROCm/oneAPI plugin actually contains pinned-range
  recognition; successful compilation or fast allocation does not prove
  staging is absent.
- Do not build or modify an adjacent XLA checkout unless the task
  explicitly requires it. The required pinned-range behavior was upstream
  at the last audit; local experimental allocator/completion patches were
  rejected.
- The adjacent production monorepo previously needed VFS registrations to
  use `registerBackend` so file/HTTP/HF/S3/GCS profiles and counters
  reached the loader. Re-check that integration rather than assuming a
  historical uncommitted edit still exists (may be stale: `VFS.Remotes`
  registers the remote schemes since R6, and the loader reads no VFS
  counters since R15).
