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

The contracts live in the pub doc comments, and docs/learn/loader.md is
the guide. This section points to them and keeps the reasons:

- API, lifetimes and errors: `Loader` (`init`, `start`, `wait`, `cancel`,
  `deinit`, `Options`), `TensorStore` (`View`, `Binding`, `Program`),
  `Source` and `Source.Placed`, `Quantization.Target`, and the registry
  (`TensorRegistry.fromPath`, `addRepo`, `Tensor.read`, `readRange`).
- From bindings to specs, program groups, admission and accounting: the
  loader.zig `//!` header. Where admission cannot measure the room (a
  device that reports no limit, such as CPU, or the buffered backend), one
  program group is in flight at a time.
- Backend selection (`backend.loadsDirect`) and the `LoadSpec` contract:
  backend.zig. The direct backend's sizing, planner, credits and pumps: the
  direct_loader.zig header, then the doc comments of `Planner`,
  `Scheduler`, `Batch`, `ReadRequest`, `Dma` and `Credits`. The buffered
  backend: the buffered_loader.zig header. The module map: `zml/io.zig`.
- Diagnostics: each direct batch logs its planning and completion lines at
  debug when awaited; the backend logs its totals (reads, bytes, credits
  high water, pinned mapped, waits, DMA) at debug from `destroy`; the
  loader logs its admission at debug from `deinit`.

### Source and VFS data plane

- The direct backend's pinned reads go through its private
  `readFilePositionalAllV`, the one exact positional scatter (short-read
  resumption, local `IOV_MAX` batching). `Tensor.readRange` validates the
  range and reads with std's `readPositionalAll` over a shared open file.
  The direct backend opens each distinct safetensor object once
  (`SourceSlot`) and keeps it open until the loader's `deinit`.
- HTTP, S3, GCS and HF share one range read loop,
  `range_read.performRangeRead`, over the governed loop of
  `vfs/request.zig`: one whole Range request per positional call, retries
  and holds serial inside the caller's read permit (the retired
  backend-local `parallel_read` pools must not return). Every other request
  a backend makes (HF tree and redirect chain, S3/GCS listings and HEADs,
  GCS token exchanges, the HTTP HEAD chain) uses the same loop, so one
  `request.Governor` per backend covers its whole quota. A request is a
  `RequestSpec` (`target` for logs, the rate-limit `key`, `retry_once`); a
  range read adds a per-attempt `prepare` hook (URI, authorization, extra
  headers; the loop appends `Range`): S3 re-signs SigV4 per attempt, GCS
  refreshes an expired bearer, HTTP is static, HF re-resolves an expired
  signed URL once (`retry_once` on 401 and 403). `exchange` retries as a
  timeout an exchange silent for `Retry.stall_timeout` and maps an open's
  404, 410, 401 and 403 (`ExchangeOptions.open`). Backends keep URL
  construction, credentials and read hints; `Governor.logTotals` logs
  their totals at debug from `deinit`.
- Registration, `Remotes`, read hints, the remote error contract and the
  VFS's lifetime: the vfs/vfs.zig header and the doc comments of
  `registerBackend`, `Remotes`, `io` and `readHints` (`ReadHints` is in
  vfs/base.zig). `Content-Range` validation, the `200` that ignores Range
  and the scatter: the vfs/range_read.zig header and `readResponse`.
- `request.classifyStatus`: 408 timeout, 429 throttle, other 5xx server
  failure; 503 is a throttle on S3 (`SlowDown`) and GCS but a server
  failure on HTTP and HF. Any other retried status naming a delay
  (`Retry-After` delta-seconds, or HF's `RateLimit` `t=`; HTTP-date is not
  parsed) sleeps that long, at most `Retry.max_delay`, instead of the
  jittered delay.
- Rate limiting is a hold (the vfs/request.zig header). One `Governor` per
  backend instance holds one deadline; each request carries its URI
  authority (`RequestSpec.key`), so a per-authority hold is a change to
  `Governor.holdFor` alone. A throttle arms the deadline at `max(server
  delay, full-jitter backoff)` clamped to `[initial_delay, max_hold]` and
  only ever extends it; every request of that backend waits it out with
  its own jitter. A clean window as long as the last hold ends the
  episode; an episode without one past `throttle_budget` fails requests
  with `error.RateLimited`. Timeouts never hold. Two budgets (sixteenth
  pass): a throttle is not charged to `max_retries`, so a limited server
  cannot exhaust a reader; everything else keeps per-request backoff.
  Defaults (`request.Retry`): five retries, 500 ms to 30 s backoff, 2
  minute hold cap, 5 minute episode budget. A deadline, not a permit,
  composes with the loader's fixed width: a read waiting out a hold keeps
  its read permit.
- A `dma_read` or `host_bounce` job is one positional read into its slot's
  blocks, exact or widened to the direct-I/O alignment. A `host_direct`
  job, and a landing job's (`dma_land`) scattered extent, make one read per
  `max_positional_iovecs` (`IOV_MAX`, 1024 on Linux) non-adjacent ranges
  (`Scatter.flush`); a landing job's other extents are one read each, into
  the worker's bounce buffer. A fill job reads nothing. Within one read,
  `readFilePositionalAllV` adds physical calls only for short reads,
  retries or `IOV_MAX`. The loader counts read operations (`readGated`
  calls), not physical calls: a remote source's request count is in its
  backend's totals; a local file has none.

### Coalesced source planner

Tensor-local jobs made every small tensor its own source operation:
~69,445 reads for DeepSeek-V4-Flash against 9,524 coalesced ("Latest
DeepSeek result"). `Loader.submit` sorts the parts once by file URI,
delivery class (DMA, host, gathered), offset and size
(`Planner.sortedPartOrder`) and plans one file at a time (`preparePlan`),
publishing each plan as soon as it exists (`publishFiles`), so workers
read the first file while later ones are planned. Each plan
(`JobCut.cutRuns`):

1. Covers one file group of the sorted order.
2. Forms the union of touching or overlapping requested ranges, never
   reading across an unrequested gap or a file boundary. Duplicate or
   overlapping bindings are read once but keep a transfer per output.
3. Partitions each merged run into the minimum number of jobs of at most
   `maximumJobLen`: a request, less two alignment units for a file read
   directly, whose reads are widened to the alignment.
4. Keeps that count while preferring part-safe boundaries: for
   `N = ceil(run_length / maximum_job_len)`, each cut takes the latest safe
   boundary that leaves the remainder fitting in `N-1` jobs, else the
   latest feasible hard cut. A boundary is safe only if the next touching
   range begins at the current union end, which preserves overlap and
   duplicate coverage.

Jobs are claimed in file order (by first read; a stable sort interleaves
the classes), the fill jobs last (sixteenth pass: a byte-fair order across
destination devices made no difference). No job depends on another: every
DMA piece is submitted as soon as its job is read (fifth pass), so a
tensor's tail may land before its head. The plan holds each job's
transfers (`Batch.Plan.Transfer`: destination, landing offset in the job's
blocks, writer mask, destination offset, length) and the landing jobs'
extents and items; nothing else dispatches at runtime (no order
indirection or suffix arrays). Staging by destination box and gathering
strided parts (`JobCut.placeDma`, `JobCut.appendGatherJobs`), with their
measurements: the direct_loader.zig header and those functions' doc
comments. Physical source bytes (`Plan.source_bytes`) are distinct from
logical bytes so duplication and replication do not distort diagnostics.
Planning is `O(parts log parts)`: 0.32 s for DeepSeek-V4-Flash as one plan
on MI300X (46 shards); with per-file plans only the first file's planning
delays the first claim (Llama: 4 plans, 1-2 ms in total).

### Pinned blocks, scattering, and ownership

- A worker takes a lifecycle credit, a fixed slot of blocks (`Credits`),
  claims a job, creates the device state of each destination it is first
  to touch (`Destination.ensureState`), reads into its slot and hands the
  transfers to `Dma.enqueue`. Pieces were merged during planning; workers
  neither walk sharding spans nor rebuild a transfer list. A host-resident
  destination (`Memory.isHost`) is written in place by the worker that
  reads it, not through DMA (`Batch.Plan.Job.Kind`), and created ahead of
  its reads by the batch's allocation task (`Loader.startAllocations`).
- One block may feed several transfers and devices: a `ReadRequest` holds
  its scheduling reference plus one per queued piece, and its last
  release (`ReadRequest.release`) returns the credit, then the job's
  completion unit, only after every piece completed or was abandoned.
- `Dma.enqueue` counts each device's pieces, takes every pump lock in
  device order, checks the sticky error, reserves each device's queue (an
  allocation failure queues nothing), takes the piece references, pushes
  the whole job, unlocks and pumps each device that got pieces: per-piece
  enqueue had cost ~69k-79k locks and pumps per DeepSeek load.
- Workers keep one iovec array and a page-aligned bounce buffer allocated
  on first use (`ReadRequest.Scratch`); request and event contexts come
  with the plan; per-device piece counts are a stack array. A source job
  makes no allocator call in steady state.
- Coalescing keeps per-tensor device buffers, so roughly tensors x
  destination devices DMA submissions is the floor. Going lower needs
  packed device allocations or a device-side scatter/copy stage and changes
  buffer ownership and model layout.

### Scheduling and concurrency

- Reads and DMA complete in any order: PJRT makes a buffer ready once every
  transfer submitted to it completed and one carried the last-transfer
  flag, so the pump flags the piece that completes the target's bytes
  (`TensorTransfer.Target.nextIsLast`) and no piece waits for another
  (fifth pass: a tail that waited for its prefixes could hold every credit
  while the prefixes stayed unclaimed). `awaitBatch` panics if a batch
  that completed without error left a target short, a bookkeeping bug.
  Per-device ready queues (`Dma.ReadyQueue`) are served in arrival order,
  so the oldest submission's pieces complete first.
- `Scheduler` is a strict FIFO of published plans (its doc comment). A
  submission publishes and seals (`sealed_at`) on one task, so its plans
  are contiguous in file order and a later submission's first job is
  claimed only after every earlier one: fairness is intra-plan, and the
  caller's submission order is the cross-submission policy. While a
  submission is still planning, idle workers sleep at most one file's
  planning time. A queued plan holds a completion unit of its batch, so
  the scheduler never sees a freed batch and needs no worker rendezvous.
- A `Batch`'s arena holds its plans (one per file, built by
  `JobCut.finish` with one `ReadRequest` per job and one `Dma.EventContext`
  per planned DMA submission, handed out in submission order through
  `events_used`), its destinations and its parts until it retires. A load
  allocates per file, not per job.
- The batch completes when `remaining` (a publish sentinel held until the
  seal, one unit per job, one for the allocation task while it runs)
  reaches zero and sets `done`. A job's unit is released exactly once: by
  `ReadRequest.release` at its last reference (last DMA callback, a host
  job's end, or abandonment) or by `Scheduler.fail` for an unclaimed job.
  Memory-order rule (the ready callback and `Dma.submitOne`'s failure
  path): copy the fields into locals, call `eventCompleted` (which may pump
  on that thread), hand the context to the pump (`retireEvent`, callback
  only), then call `request.release()` last, because the release may free
  the batch.
- `recordError` keeps the first error (`first_error`,
  `stdx.FirstError(anyerror)`), logs it once, retires the unclaimed jobs
  (`Scheduler.fail`) and releases the queued pieces (`Dma.abortReady`);
  requests in flight stop at their next read (`readGated`, after the
  permit) or enqueue. The error is sticky: every await returns it and
  later submissions are refused with it. A planning failure, before or
  after the first publish, fails the loader too (a partial submission can
  never complete): `submit` seals, awaits and returns the sticky error
  (fifty-fourth pass). `awaitBatch` waits `done` uncancelably, after which
  nothing touches the batch; `Dma.retireBatch` drains every pump's
  `retired` under every pump lock, so neither `abortReady` nor a pump can
  race the free; destroying the batch destroys its transfer managers,
  whose PJRT destructor errors every buffer whose last transfer never went
  out, so a buffer a failure left open reports an error, never success.
- The buffered backend's batch is a cursor over its specs, run by up to
  `read_parallelism` workers of its own (a `std.Io.Group`, started with
  `concurrent`; inside `submit` when none starts) and awaited
  uncancelably; read permits bound their reads and one staging budget
  (`min(width, 12)` x the largest staging request, live and kept buffers)
  their bytes, not their number. Details: the buffered_loader.zig header.
- The source width is fixed for the load: `Options.read_parallelism`, else
  16 locally or 32 when the checkpoint's read hints say `high_latency`
  (`Loader.Options` holds the sweeps behind both; fifteenth pass), resolved
  once by `backendOptions`. The direct backend clips it so `width + 1`
  requests fit under the mapped ceiling beside the DMA reserve and the
  zero block. There is no adaptive controller (fourth to sixth passes); a
  throttled source is held one layer down by the VFS governor, and the
  loader reads no VFS statistics (forty-seventh pass).
- Two bounds, both constant for the load. Lifecycle credits
  (`Credits.count()`): the carved pinned set after the zero block, in
  requests, at least `width + 1` source requests plus the DMA reserve, so
  the DMA stage keeps its depth at any read width (fourth pass; the
  `Loader.credits` doc comment has the GB300 numbers) and no credit can
  want an unmapped block. A request returns its credit only after all its
  pieces finish. Read permits (`read_permits`, a `std.Io.Semaphore` of
  `read_width`) alone limit source calls.
- Pinned pre-growth happens in `create`: calibration's ring stays mapped,
  then `Workspace.carve` maps the blocks missing below the DMA reserve
  (`dma_calibration.depth` blocks per device, at least one maximal
  request) as one arena and those missing below the total (`width + 1`
  requests, the reserve and the zero block) as another (ROCm balances
  bytes per allocation across host nodes), and cuts every arena into
  blocks. `create` computes the width once from the blocks the mapped
  ceiling leaves beside the reserve and returns `OutOfMemory`, logged at
  err, when it fits no request. Four CPU devices, 4 MiB blocks, 8 MiB
  requests and width 16 map 17 x 8 MiB + 33 x 4 MiB = 268 MiB. Nothing
  maps inside a load (146-230 ms of hipHostMalloc on MI300X when it did);
  `Workspace.allocate` asserts it. The "host workspace pregrown" line logs
  `retained`, `pregrown` and `pregrowth_ms`.
- `width + 1` workers are spawned at creation and never retired or parked:
  a worker hands its request to the DMA stage and claims the next, so
  credits beyond the width need no workers (128 persistent workers had
  cost 7% on one MI300X).
- DMA depth is the constant `dma_calibration.depth` (8) blocks per device
  (see "Adaptive DMA width" in the ledger), enforced by the pump as a byte
  budget (`dma_budget_bytes`, `DevicePump.hasRoom`) with at most 64 pieces
  in flight per device (`max_dma_pieces_per_device`), so a block of small
  tensors keeps the bytes moving (fourth pass). No global DMA cap.
- DMA event lifetime: a ready callback pushes its `EventContext` onto its
  device pump's intrusive `retired` stack under the pump's mutex
  (`retireEvent`), after its own `eventCompleted` and before
  `request.release()`; `pump` destroys the stack at the top of every
  iteration (`destroyRetired`), so an event is destroyed by a later pump
  or by `retireBatch`, never inside its own callback. Live PJRT events are
  bounded by devices x 64 plus one pump batch. Every shipped plugin
  accepts destruction outside the callback (PLAN.md task 9;
  `DevicePump.retired`).

### DMA memory and calibration

- Host memory (arenas per platform, NUMA interleave, the 16 GiB mapped
  ceiling `Workspace.mapped_bytes_ceiling`, `OutOfMemory`): the
  host_memory.zig header. The direct backend owns its workspace and maps
  it once, before the first read.
- Calibration (the screen, CPU's `cpu_block_size`, `measure` for
  dma-bench): the dma_calibration.zig header. `calibrate` and `measure`
  return the block. Screen defaults (`dma_calibration.Options`): 2 ms and
  32 transfers per window; borderline confirmations run 25 ms and 256
  transfers (private constants). One `dma_bench` debug line: platform,
  devices, device kind, block, parallelism, GiB/s, duration, retained
  mapped bytes.
- `Platform.warmupDeviceAllocators(io)` is the only allocator warm-up;
  `init`'s task (`setUp`) and `measure` run it first. Calibration measures
  one representative device (device 0); its block applies to every
  device, and the depth is the constant `dma_calibration.depth`.

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
  4x pinned high-water versus 8 MiB. Replaced by the checkpoint backend's
  read hints (`VFS.readHints`).
- **Joint adaptive request size + source width:** coupled size probes to
  modeled width, handicapped larger sizes, drained between generations,
  and often finished a load before useful candidates. Request size comes
  from the read hints; source width stayed adaptive until the fifteenth
  pass.
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

## Fifty-third pass: design review, do more with less (2026-10-28)

A read-only review of the API and the design's simplicity, on zml
93428bd0 and monorepo 0bf031d9 (after the master merge). Report:
`LOADER_DESIGN_REVIEW.md` (Part 2 nineteen moves M1-M19 in three waves
plus M20 for llmd, Part 3 decisions D1-D9). 21 reviewers (16 surfaces, 5
lenses) produced 159 findings in 105 groups; refuting verifiers kept 102
(65 confirmed, 37 partial) and refuted 3. Nothing was built or run.

- Verdict: the byte pipeline earns its place; the excess is at the seams:
  placement as a flag plus `Bound` (`Source.Placed` replaces both),
  `LoadPart`/`Part` restating `Lattice`, `backendError` over six error
  sets, spec copies forced by the front end freeing at publish, backends
  re-checking front-end guarantees, four copied remote shells (S3's and
  GCS's vtable blocks are identical), llmd's drafter decided in five
  places and its load split over two optional hooks.
- About -2,500 lines over the 19 moves (wave 1 -1,630, no measured path
  changes); the D4 remote core would remove about 1,030 more.
- Bugs met: a server-named retry delay is slept unbounded (`failure.delay
  orelse` jitter, vfs/request.zig:195); HTTP's `fetchSize` copies a
  redirect over the buffer its base URI points into (vfs/http.zig:350);
  `dirRealPathFile` keeps the scheme; buffered helper reads run inline
  under a busy Threaded io; S3 and GCS listings stop at 1000 keys
  (unverified lead).
- Spot-checked against the code: the placed-flag guards, the S3/GCS diff,
  `backendError`, the spec-copy rule, llmd's hooks, `LimitedGroup`'s
  growth over master, the two request-layer bugs.

## Fifty-fourth pass: errors by two rules, and the HTTP fixes (2026-10-28)

The user's calls on the fifty-third pass: reworking the VFS is not part of
this work (its M5, M6 and D4 are out; its two HTTP bugs are fixed). Errors
follow two rules: log at the place with the most context, once; keep an
explicit error only where the caller can act on it or the information
helps, and otherwise let an inferred `!` (or a stored `anyerror`) carry it,
since a gigantic union helps nobody. Uncommitted, on zml 93428bd0 and
monorepo 0bf031d9.

VFS:
- A server-named delay on a retry that is not a throttle is clamped to
  `Retry.max_delay` (it was slept unbounded: `Retry-After: 3600` on an HTTP
  503). Throttles and holds are unchanged.
- HTTP `fetchSize` resolved each `Location` over the buffer its base URI
  pointed into: three rotating buffers as in hf.zig, at most
  `request.max_redirects` (3) hops, `try` for `catch unreachable`. HTTP
  range GETs follow redirects (`RangeSpec.redirects`; S3, GCS and HF still
  forbid them), with redirect room for that many hops (LOADER_REVIEW #13).
  The mock server serves redirects; the new test fails against the old
  walker (`UnsupportedUriScheme`).

Loader:
- `Loader.Error` (about 57 names), `backend.SubmitError`/`AwaitError`, the
  per-backend sets, `StickyError` and `ShardGrid.InitError` are gone;
  `start`, `wait` and the backends infer, and the sticky errors are
  `anyerror`. Kept explicit: `init`'s `{InvalidOptions, OutOfMemory}`,
  `PrepareError`, `TransferInitError` (host memory or PJRT),
  `checkPlacement`, `checkOutput`, `backendOptions`, `lowerPiece`.
- `backendError` is gone. Bug found by the inventory: its
  `error.Internal => panic` arm also matched PJRT's `Internal`, so a device
  INTERNAL aborted the process instead of failing the load. The direct
  backend returns `InvalidBinding` where it logs a refusal (strided read
  too long, range past u64); placement and memory kind are asserted where
  the front end checked them; `maximumJobLen` asserts what `init` checked;
  an unsubmitted target at batch end panics, naming the tensor; an
  impossible overflow is plain arithmetic.
- A direct submit whose planning fails always fails the backend and returns
  the sticky error (before, a failure ahead of the first publish left it
  usable, which nothing used); two tests changed to say so.
- Logging once: `log.failure` logs `InvalidBinding` at debug (logged where
  its rule was checked); a gather extent's failed read or open is recorded
  with its own file (`FileReader.read`, `Reads.failed_source`); create no
  longer re-logs a workspace failure, which logs its own (now including a
  failed page allocation); the worker spawn failure is logged. A plain
  allocation failure and `Canceled` are returned unlogged, as the docs now
  say.
- llmd: `weights.unsupported` beside `mismatch`, each logging the full key
  once; `UnsupportedWeightDtype` and `UnsupportedFp8NormalizationProgram`
  fold into `UnsupportedQuantization`, `MissingQkvScale` into
  `CheckpointMismatch`; a refused MoE kernel or layout is logged with its
  key. qwen4's n-gram test passed `loader.start` an `undefined` buffer tree
  with an optional field; it bufferizes now.

Review: an error inventory (5 agents) and an adversarial review (5 lenses,
a refuting verifier per finding; 8 confirmed, 7 refuted), all confirmed
ones fixed or recorded here. Left: a compile failure is still logged twice
(the Compiler with its MLIR, then the loader with the program); safetensors
header errors (several names, panics on malformed fields) are untouched.
Checks: `//zml:test` 371 passed, 29 skipped; `//vfs:test` 54; examples
build; llmd 10/10 (`llmd:test` 257, `weights_test` 26). "one throttled GET
holds every reader" failed once (2 holds) beside the zml suite and passed
the 8 runs after: timing, on a path the change does not touch.

## Fifty-fifth pass: wave 1 of the design review, and the VFS against master (2026-10-29)

The user's calls on the fifty-third and fifty-fourth passes: the VFS is part
of this work after all (the governor came with the rework), but its diff
against master stays as small as possible: rate limits and direct I/O are
reworked, not everything, and there is no comptime remote core (D4). This
replaces the fifty-fourth pass's "reworking the VFS is not part of this
work". `RequestSpec.key` (the host) stays, since rate limits will be scoped
by it (D2). Sharding is always explicit: [87] is declined, `checkOutput` and
`withLayout(x, output)` stay (D6). Uncommitted, on zml 93428bd0 and monorepo
0bf031d9, after the fifty-fourth pass.

Loader, moves of `LOADER_DESIGN_REVIEW.md` Part 2 (M5 and M6 are replaced by
the VFS steps below; M11-M19 are waves 2 and 3):
- M1: `bufferize(allocator, &model)` and `deinitBufferized(allocator,
  &buffers)` without the type; `pairBufferized` returns `[]Pair`;
  `addRepo(io, repo, "drafter")`; `resolveModelRepo` takes directories only;
  `readFilePositionalAllV` is private to the direct backend. It also deletes
  `Shape.partitioning()`, `Logical.set` and `Partitioning.parse`, the user's
  own commits (D9, the user's call).
- M2: `Program` answers name, eql and compile through a private vtable;
  `fail(err)` is the one writer of failure; `retire` is folded into
  `retireOldest`; `backend.Options` copies `dma_block_sizes`
  (`DmaBlockSizes`, at most 16) and holds a resolved `read_parallelism`;
  `program_group_bytes` is a Loader option; `Platform.init` refuses more than
  `MAX_NUM_DEVICES` devices (`InvalidDeviceCount`).
- M3: `LoadPart{source, lattice}` and direct's `Part{source, source_slot,
  destination, lattice}`. Everything a spec names lives until its await, and
  the backends borrow parts and dims (the pending submission owns the specs'
  arena until release; no copies at submit). `checkParts` is one loop with
  one floor, keeping the strided rule that parts cover at most the
  destination; `ShardGrid.locate`; the buffered backend asserts `.default`.
- M4: `Source.Placed` (store, shape, pieces) is what `withPartitioning` and
  `as` return and what binds (`bind`, `unplaced`, `lower`, the one
  bits-to-bytes rule, and `whole` as a test probe), so a misuse is a compile
  error. `Source.Bound`, `emptyOf`, `keepEmpty`, `maybeCreateBinding` and
  `hasPrefix` are gone; one empty rule, a source that reads no bytes, whose
  log names no key (the cost Part 2 accepted); `View.prefix()` is
  `[]const u8`.
- M7, llmd weights (-249 lines): one `expertStack` loop with a `Projection`;
  one `layOutExperts` expression, evaluated by the loader or by the
  LayOutExperts program; block-FP8 experts are asserted concatenated and
  plain (the `UnsupportedFp8NormalizationProgram` refusal is an assert);
  `bindQuantized` returns `{ Tensor, Quantization }`; `linear.createBf16`;
  routers, gates and optional heads go through `linear.create` or
  `maybeCreate`, so a quantized lm_head or g_proj now loads quantized
  instead of as raw bytes; `splitTensor` is gone. The expert tests' program
  counts and bytes were measured before and after: unchanged. Fixed on the
  way: `DSparkDrafter.init` leaked its backbone when its head failed.
- M8: `speculative.Drafter.open` decides the drafter once, before the store
  (a model without drafters warns and gets null, a wrong drafter is
  `UnsupportedSpeculativeDrafter`, logged once); `Model.init` takes
  `?*const Drafter`; each model holds one `Bufferized(Weights)` and has one
  `load` that starts (base, then drafter), waits and bakes.
  `View.hostMappedBytes()` replaces `engramHostPinnedBytes` and qwen4's
  `offload_ngram_size`. zoo's unread `--gpu-memory-fraction` is gone, and
  its token budget follows `Drafter.open`.
- M9: mnist and the README teach the in-main recipe with a plain VFS
  (mnist's path is a local Bazel file, and `Remotes` would print credential
  warnings on every run). examples/io's load path runs in straight lines,
  its checks compare against the registry entries, and its own load timer
  is gone: only the loader's `Loaded weights` line is left, so earlier
  passes' examples/io figures (the `vfs` line, with init and calibration)
  are not comparable.
- M10: `Buffer.from`, `uninitialized`, `HostAccessible.init`, `toSlice` and
  `toSliceAlloc` walk the shape's mesh (`buffer.deviceMemory` asserts the
  device is the platform's). This fixes a buffered load onto a sub-mesh and
  the sub-mesh half of the toSlice open item. zml/io/testing.zig is folded
  into `zml.testing`, with a rank-N `expectShards(..., memory)`.

A review of the whole wave (five dimensions on a frozen copy, a refuting
skeptic per dimension; 12 confirmed, 2 refuted) was fixed in turn:
- `log.failure` logs `OutOfMemory` at debug. Every `OutOfMemory` that
  reaches it is a plain allocation or host memory that logged its own
  failure (checked caller by caller), so this replaces four guards and fixes
  both backends' `recordError`.
- `Buffer.deinitAll(&buffers)` without the type (it refuses a tree with no
  Buffer); `LoadedModel.unloadBuffers`, `ReshardedLmHead.Buffers` and
  `deinitOwnedParameterBuffers` are gone; llmd says `Model.load` and
  `buffers` everywhere.
- Leaks on an init failure: qwen4's `TextModel.init` (the sampler is built
  before the layers) and DeepSeek's `Model.init` (its executables' errdefers
  sat inside a labeled block; older than the wave).

VFS, by the re-scope study's plan. Against master the vfs/ diff went from
6,810 changed lines to 6,294, and the net addition from 2,884 to 2,526:
- V1: `dirRealPathFile` strips its scheme (it gave `hf://hf://a`); direct
  I/O is decided in one locked transition (`settle`), which fixes an
  overwrite race; `Handle.read_only` is gone and `direct_io.enable` refuses
  a descriptor not opened read-only.
- V2: `VFS.readHints(path)` replaces `LoadProfile` and `loadProfile`
  (`backend.Options.read_hints`; the refusals log `checkpoint=` instead of
  the scheme); one read-stats record (`AtomicReadStats`); `MockServer` owns
  its group (`start`, `stop` returning its first error, `deinit`, `path`).
- V3: each backend has one `request.Governor` (io, client, name, what a 503
  means, stats); `request.Context` and `RequestSpec.backend`/`.unavailable`
  are gone, and a spec is target, key and `retry_once`;
  `ExchangeOptions.open` maps 404, 410, 401 and 403 before `consume`. GCS's
  token and HF's tree cache have locks of their own, so neither holds the
  handle table across a governed request (the HF lock is older than this
  pass). A non-2xx from the GCS metadata server now logs one err line.
- V4: the inactivity watch runs in `exchange`, so a nested request (HF's
  re-resolve, GCS's token refresh) is watched by its own exchange;
  `Attempt.waiting`, `watchedToken` and range_read's pause are gone.
  `VFSBase.inner` is 16-aligned (the Governor's i96 durations raised the
  backends' alignment), so http, s3 and hf use master's `@fieldParentPtr`
  without `@alignCast`.
- A final review of the whole diff against master (a skeptic confirmed 17):
  a canceled read returns `Canceled`, unlogged (it surfaced as
  `ReadFailed`, so an unwatched exchange retried a caller's cancel and a
  stall logged twice); `openFailure` and `fatal` no longer log a logged
  failure again or a plain `OutOfMemory`; a refused range body logs once;
  registering `file` is an assert (`RegisterError` is gone); an oversized
  GCS token response fails once instead of retrying. Back to master's lines:
  vfs/BUILD.bazel, twelve forwarders, `lookupDir`, three `pub`s, S3's and
  GCS's `fileSeekBy` (the overflow fix is one line each), `Protocol`, hf's
  logger.
- "one throttled GET holds every reader" was a test race: a reader held
  before its first GET sent it after the hold, still among the mock's first
  8, and armed a second hold. The mock now parks every reader's first GET
  (`barrier_gets`); 35 uncached runs clean since.

Lines: zml +2,593/-3,372 over 56 files (vfs/ -358, the rest -421), monorepo
+864/-1,366 (-502). Each move and step was implemented, reviewed read-only
and verified by a fixer who first tried to refute each finding, with both
gates after each.

Checks: `//zml:test` 373 passed, 29 skipped (+1 `hostMappedBytes`, +1
sub-mesh buffer); `//vfs:test` 54; examples build; llmd 10/10 (`llmd:test`
256, 9 skipped; `weights_test` 25: the `splitTensor` test went with it). The
CUDA-only cute kernels compile with `--@zml//platforms:cuda=true`. Nothing
ran on a GPU or a real remote.

Left:
- On a GPU (Open work lists them): the direct backend borrowing dims; M10's
  walks on an accelerator with more devices than a test asks for; M7's
  block-FP8, MXFP4 and NVFP4 paths on GB300 and MI300; the routers and heads
  now detected, on real checkpoints; llmd with a drafter (M8) and
  `hostMappedBytes` against the old values.
- The user's calls: the study's optional X2 (master's dead code deleted in
  the remote backends, about 130 lines) and X3 (constructor shapes, about
  45; GCS's `InitOpts` rename alone is free to undo); D9; `VFSBase` stays
  private (thirty-ninth pass).
- A compile failure is still logged twice (fifty-fourth pass);
  `//examples/mnist:mnist_dp` aborts at `profiler.stop()` on CPU, before and
  after the wave; `Buffer.uninitialized` on a sub-mesh has no test;
  `ops/mla/state.zig` keeps a dead `deinitOwnedParameterBuffers` (older than
  the wave).

## Fifty-sixth pass: wave 1 on GB300 and MI300X, and F1 (2026-10-30, gb300-1 and mi300)

Wave 1 (fifty-fifth pass) is committed as `rework`: zml 8af7fd3c on
93428bd0, monorepo b791da11 on 0bf031d9. It ran on gb300-1 (aarch64, 4x
GB300, CUDA) and mi300 (x86_64, 8x MI300X, ROCm, plugin
202610231457.291.1.799ad5c39a34) against its parents. Trees: w0 = the
parents, built from the same checkout with `jj restore --from` and restored
after; w1 = `rework`; w1ng and w1fix = temporary mi300 builds of w1 with
F1's n-gram fix, then with the n-gram and page-cost fixes; w2 = `rework` +
M11-M16 + F1 (fifty-seventh pass). llmd builds with
`--config=release_with_symbols`, which is release_safe, so
`stdx.debug.assert` and M7's block-FP8 expert assert are live in every GPU
binary. Reports and saved binaries stay in `~/phase8` on each host. Every
llmd run answered "Paris" and "391" unless stated.

zml suite:
- gb300-1, w1, 4 GPUs: 402 tests, 365 passed, 34 skipped, 3 failed. The
  sub-mesh buffer test and Sharding's 'GSPMD executes a reduction across a
  folded logical axis' fail with FileNotFound on libzml_cpu.so (aarch64 has
  no CPU plugin); moe's 'FlashInfer compatibility matrix' fails
  MoeComplianceFailed. The Triton and CuTe MXFP4 matrices were skipped by a
  temporary edit (they abort the binary, fifty-first pass). io: 80 OK, 14
  SKIP (they need a CPU platform).
- mi300, 4 GPUs (`ROCR_VISIBLE_DEVICES=0,1,2,3`): w1 402 tests, 377 passed,
  24 skipped, 1 failed; w0 401, 376/24/1. All 94 io tests pass. The failure
  (moe's Fly matrix) and an abort on 8 GPUs are the same on both trees
  (pre-existing, below).

examples/io with `ZML_LOAD_CHECK=1` (M9; M3's borrowed dims):
- GB300, Qwen3.8-27B-FP8 sharded on 2 GPUs: 28.75 GiB in 1.446 s (dbg), 1606
  of 1606 tensors ok, 1,860 reads, credits 23 of 33.
- MI300, Llama-3.1-8B-Instruct on 4 GPUs: sharded 1.186 s, tp 1.221 s,
  host_mapped 14.109 s (allocating its 291 host destinations takes 14.10 s);
  291 of 291 ok each.

llmd on GB300, w0 against w1:
- DeepSeek-V4.1-Flash, GPUs 2,3, `--token-batch-size=256 --batch-size=1`, 3
  runs each: 467.10 GiB in 11.53 s against 10.72 s (medians; w0 waited
  2.5-3.4 s for compiles, w1 0-0.64 s); decode 5.67-5.95 against 5.74-5.76
  ms/step; the same KV plan (5148591 pages, 95.593 GiB). The spread is XLA's
  per-process autotuning: the before_optimizations HLO is identical across
  trees once PJRT pointers are masked.
- The same with the native DSpark drafter, 6 runs each: 474.50 GiB in
  10.37-11.49 s against 10.12-11.03 s; median tokens/step 2.47 and 2.43.
- Qwen3.8-27B-FP8 + z-lab DFlash2: ~980 ms loads, ~20 ms/step, 2.93-3.26
  tokens/step on both. Qwen3.8-27B + RedHatAI DSpark: 2.88-2.96 tokens/step,
  one resolve line. A qwen4 model given a drafter logs one warning and
  resolves nothing.
- Qwen3.8-Flash-Next BF16 at `--cache-memory-fraction=0.6`: 329.58 GiB in
  9.266 s (w0) and 8.237 s (w1). Flash-Next-FP8, 15 runs: 169.41 GiB in
  3.5-4.6 s, median decode 9.14 (w0) and 9.07 (w1) ms/step.
- Smokes at w1: Qwen3.8-27B, Qwen3.6-35B-A3B, gemma-4-26B-A4B-it,
  gemma-4-31B-it, Laguna-XS-2.1, MiMo-V2.6-Flash-MOPD (block-FP8 MoE, 158.11
  GiB in 3.207 s). Both trees log the same kinds of lines.

llmd on MI300:
- DeepSeek-V4.1-Flash, GPUs 0-3, 2 runs each: 466.95 GiB in 103.6 s (w0) and
  102.1 s (w1) (medians; slow on both, below); decode 12.53-12.74 ms/step;
  4987539 KV pages every run; MXFP4 experts through fly. With the native
  drafter (w1): 474.34 GiB in 1m26.9s, 24.16 ms/step at 2.43 tokens/step.
- Qwen3.8-27B-FP8 + DFlash2: 3.26-3.36 s loads, ~39 ms/step, 2.68-3.37
  tokens/step. Block-FP8 on CDNA3 (R9's FNUZ conversion) gives coherent
  text; three of four runs are byte-identical across trees.
- Drafters at w1: Llama-3.1-8B + z-lab DFlash-UltraChat 3.22 tokens/step,
  Laguna-XS-2.1 + poolside DFlash 1.98, gemma-4-31B-it + z-lab DFlash 3.75,
  Muse-Glimmer-30B + incoai DFlash2 2.96.
- MiMo-V2.6-Flash-MOPD (fp8 128x128, 256 experts), GPUs 4,5: 158.11 GiB in
  11.186 s, 15.03 ms/step, long answer correct. Smokes: Qwen3.6-27B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B-it, Qwen3-4B-Instruct-2507.

M7's paths: no MI300 run reached the block-FP8 expert assert (MiMo,
Flash-Next-FP8 with F1's n-gram fix, Qwen3.8-27B-FP8, DeepSeek). GB300 ran
block-FP8 experts through `bindQuantized` (Flash-Next-FP8, MiMo). MXFP4
experts ran through cute on GB300 and fly on MI300 (DeepSeek). NVFP4 never
ran: no NVFP4 checkpoint on either host has a model type llmd supports. No
FP8 checkpoint gb300-1's llmd runs used (Flash-Next-FP8, MiMo,
Qwen3.8-27B-FP8, DeepSeek) stores a quantized lm_head, router or gate;
mi300's checkpoints were not inspected.

`hostMappedBytes` (M8) equals the old `extra_device_memory_bytes` on w0 and
w1. Its only consumer is XLA's `device_memory_size` = bytes_limit + extra
(zml/Compiler.zig:906), read from XLA config dumps on GB300 and from a
temporary log line on MI300. Per device: GB300 DeepSeek 101379016992 B
(94.42 GiB of Engram tables), Flash-Next BF16 at 0.6 51200245760 B, FP8 at
0.3 25600122880 B; MI300 DeepSeek 50689508496 B for prefill and decode (0
for loader programs), Flash-Next BF16 at 0.75 51200245760 B.

What failed, and F1's fixes (1-3 older than the rework, w0 fails alike; 4
is wave 1's own, M10's new test, absent at w0):
1. Flash-Next BF16 does not load on 2x MI300X at the default fraction
   (0.95): ResourceExhausted on layers.12.mlp.experts.gate_up_proj
   (pa_allocated 184253677568); at 0.85 on layers.35. The plan offloads
   the n-grams (164.15 GiB available, 166.46 GiB fixed, 47.68 GiB
   offloaded). Cause:
   `qwen4/spec.zig:159` subtracted the PLE convolution cache's per-page term
   (from `ngramMemory`) from `full_bytes_per_page`, but that cache stays in
   device memory; only the n-gram tables bind host_mapped. The page count
   came out ~3.5x too high. Line 160 subtracted the n-gram bytes from the
   fixed memory a second time (harmless in every run). gb300-1 never failed:
   at 0.6 it overshot the fraction silently (below). Fix: `ngramBytes()`
   returns only the tables' bytes per device, the PLE term stays in
   `memoryRequirements`, and the offload test compares `available >
   fixed_memory - ngram_bytes` before reducing the fixed memory once.
2. Flash-Next-FP8 cannot serve on ROCm: the first request fails with
   'Unsupported data type for reduction operation: f8e4m3fn', then panics at
   llmd/models.zig:249. `NgramEmbedding.gather` all-reduced the e4m3 rows in
   their own dtype, which CUDA accepts. Fix: an 8-bit table's rows are
   summed across shards in bf16, exact since e4m3 converts to bf16
   losslessly and one shard owns each row.
3. An unsupported model_type failed with a bare 'error: InvalidModelType'
   naming neither the type nor the checkpoint (mistral_large4, glm_moe_dsa,
   beam, lfm3, deepseek_v4, glm5_next). Fix: `fromConfig` refuses with
   `weights.refuse("Unsupported model_type {s}")` or "config.json has no
   model_type string", logged once at err.
4. M10's sub-mesh test built its own 4-device CPU platform, so it failed on
   gb300-1 and ran on a CPU platform on mi300: M10's walk never ran on an
   accelerator. Fix: `testing.Devices.init(allocator, io, 4) catch return
   error.SkipZigTest`, the accelerator's devices when it has 4.

F1 is zml +6/-2 and monorepo 3 files +29/-24. Checked on the GPUs as w2,
against w1 on gb300-1 and against w1fix on mi300:
- Suites: gb300-1 403 tests, 366 passed, 35 skipped, 2 failed (Sharding's
  GSPMD test and FlashInfer); mi300 403, 378/24/1. The sub-mesh test passes
  on 4 GB300 and on 4 MI300X: M10's walk ran on an accelerator.
- MI300 Flash-Next BF16 at the default fraction loads and serves: 48.11 s
  against 47.78 s (medians), within this model's 47.66-49.73 s spread (the
  load waits on host_mapped allocation); 16.58 ms/step.
- MI300 Flash-Next-FP8 on 2 GPUs: 10.16 s against 10.15 s, 15.14 against
  15.13 ms/step. 3000-token greedy generations: 5 of 6 (w2) and 5 of 5
  (w1fix) coherent to the end; one w2 run looped on one sentence after
  ~1,377 words, judged not a defect (w1fix sums in bf16 too).
- GB300 Flash-Next-FP8's qwen4_forward has 96 bf16 and 1 f8e4m3fn
  all-reduces at w1, 97 bf16 and none in f8 at w2; median decode 8.76 (w2)
  against 9.25 (w1) ms/step, noisy. BF16 at 0.6: plan lines and
  `device_memory_size` (318512309334) unchanged.
- The refusal: 'error(llmd): Unsupported model_type mistral_large4' (GB300)
  or '... lfm3' (MI300), then Zig's 'error: InvalidModelType'.

Behaviour change: when qwen4 offloads its n-grams it now plans fewer KV
pages and stays within `--cache-memory-fraction`. Flash-Next BF16 at 0.6 on
2x GB300: 127,286 pages at w1, 35,959 at w2 (read from qwen4_forward's
entry_computation_layout), so 575,344 cached tokens per device instead of
2,036,576. Peak per GPU (nvidia-smi) fell from 239,531-240,409 to
159,739-160,963 MiB: w1 overshot the fraction by ~77.7 GiB per device, the
640 KiB per page of PLE cache it did not count (127,286 x 640 KiB). FP8 at
0.3: peak 115,257-116,117 -> 83,745-84,365 MiB (pages not dumped). FP8 at
default args offloads nothing and is unchanged. qwen4 logs no page count.

Pre-existing, w0 the same:
- mi300, 8 GPUs: the suite aborts at test 6, the paged-attention decltest
  ("can't shard {...,hkv=4/model,...} ... for Mesh(name=model,
  partitions=8)", paged_attention.zig:457 via triton.zig:277).
- mi300: moe's 'Fly compatibility matrix' cases 38, 39, 41, 42 (geglu,
  mxfp4, before_down) fail MoeComplianceFailed after 'geglu was requested
  but we only support the tanh approximation'.
- mi300 DeepSeek-V4.1-Flash loads in 84-117 s against ~10 s on GB300: ~68-70
  s allocating its 189 GiB of host_mapped engram tables (~2.8 GiB/s)
  overlaps 59-91 s waiting for 17 loader compiles that start only with the
  load, after the 3m51 model compile. With the native drafter it waited 0 ms
  (not investigated).
- mi300 Qwen3-4B-Instruct-2507 answers 17x23 = 401 at temperature 0 (411 and
  491 when sampling); not checked on CUDA.
- Flash-Next-FP8 does not fit one MI300X (ResourceExhausted in
  NormalizeBlockFp8Weight, with F1 too). On two, and for Qwen3.8-27B-FP8,
  the loader warns 'program group exceeds the device room alone ... admitted
  anyway'.
- GB300 Flash-Next-FP8 decode is bimodal per process (~9.0 or ~12.6 ms/step;
  1 of 8 w1 and 2 of 7 w0 processes). Its fixed memory went 87.13 -> 86.38
  GiB and decode 11.65 -> ~9.0 ms/step between the fifty-second pass and w0:
  unexplained.
- 'loader admission' prints min_room_seen=0B when no program group ran (the
  null prints as 0).
- Text at temperature 0 differs between processes; compare trees by
  /metrics.

Notes for later runs: the loader's debug scope is `zml/io/load`
(zml/io/log.zig:10) and `dma_bench` logs on `zml/io`; the GB300 reports'
"GB" peaks are nvidia-smi MiB / 1000. Saving a binary built with
`--config=remote` (which ends with `--remote_download_outputs=minimal`)
needs `--remote_download_toplevel` on gb300-1 and
`--remote_download_outputs=toplevel --build_runfile_links` on mi300. The
mi300 plugin's `RocmExecutor::IsHostMemoryPinned` is a local symbol,
invisible to `nm -D`; dma_bench gives 46.6-47.3 GiB/s on 4 devices against
4.5-7.6 with the stale plugin.

Left of the fifty-fifth pass's GPU list: NVFP4; a quantized head on a real
checkpoint (DeepSeek's `createBf16` `wo_a` is exercised only by DeepSeek
answering correctly); `expectShards`' memory argument against real host
spaces (no report names it); the buffered backend (no TPU, neuron or metal);
remote checkpoints; oneAPI; zoo; the `DSparkDrafter.init` leak fix's failure
path; a model's free path and a progress bar watched to the end (every llmd
was stopped by its PID).

## Fifty-seventh pass: waves 2 and 3 (M11-M19) and their GPU A/B (2026-10-30/31)

Waves 2 and 3 of `LOADER_DESIGN_REVIEW.md`, uncommitted on `rework`, in this
order: M11, M12, M13, M14, M15, M16, F1 (fifty-sixth pass), M17, W2F, M18,
M19. The monorepo changes only by F1. Each move was implemented, reviewed
read-only and fixed by a fixer who first tried to refute each finding, with
both gates after each. A review of wave 2 as a whole (direct, front and
completeness on a frozen copy after M17, a refuting verifier per finding: 7
findings, 6 confirmed) gave W2F; its sixth finding, stale docs, is this pass
and the "Current design" refresh. Line counts are snapshot diffs, against
Part 2's estimates.

Moves:
- M11, admission by fixed arrays ([0]; loader.zig): `DeviceBytes =
  [Platform.MAX_NUM_DEVICES]u64`. `admit(executables) !DeviceBytes` fills
  inputs, execution and temporaries only while `memory_supported`, so
  `compiledTempBytes` and `readRoom` no longer run unmeasured; unmeasured,
  it still retires until no program group is pending. New: `fits`,
  `pendingExecution` (summed from `pending`, saturating),
  `programGroupPending`, `readRoom() ?DeviceBytes`. Gone: `Scratch`,
  `pending_execution`, `pending_executes`, `Fit`, `measureFit`, the heap
  `execution` slice and its hand-off, init's only allocation. `Loader.init`
  returns `error{InvalidOptions}`; a backend set-up error comes from the
  first `start` (docs/learn/loader.md:179). The unmeasured `execute
  admission:` debug line prints only `retired=` and `measured=false`. -12
  lines against -35: `room: ?DeviceBytes` keeps "unmeasured" when a device
  stops reporting a limit mid-load, and `admission.admits` keeps slices so
  its pure tests do not depend on `MAX_NUM_DEVICES`. Reverses the
  forty-fifth pass's "Fixed `[MAX_NUM_DEVICES]` arrays were not taken"
  (`Platform.init` caps the count with `InvalidDeviceCount`, and only a
  measured loader fills them) and the fifty-fourth pass's explicit
  `{InvalidOptions, OutOfMemory}` for `init`.
- M12, one arena per direct batch ([38], [39], [29], [32], [33.1]):
  `Batch.arena` holds the destinations, the parts and every plan until
  `destroy`. `JobCut.finish(arena)` copies a plan's jobs, transfers, extents
  and items into it, with one `ReadRequest` per job and one `EventContext`
  per planned DMA submission (counted by job kind). Gone: `Plan.allocator`,
  `Plan.create`, `Plan.destroy`, the take-on-success rule and its test,
  eight errdefers, `JobCut.dma_submissions`, `Batch.freeing`,
  `Batch.planning_source`, the `sealed_at` assert, LazyOnce's context
  structs and trampolines (now `LazyOnce(T).ensure(io, initFn, args)`),
  `Destination.hostResident` (`memory.isHost()`). -93 against -100. Part 2's
  "publishFiles' errdefer logs the file" is impossible, since Zig 0.17
  rejects `errdefer |err|`: `publishFiles` hands the file out through
  `source: *[]const u8` and submit's `recordError` logs it once, when first.
  "No allocator call per job" holds by construction (finish makes a fixed
  set of arena calls per plan); a test counting them could not fail and
  went. JobCut keeps a scratch allocator for its growing lists.
- M13, planner representation ([40]-[42], [35], [33.2], [44]): `dma_staged`
  and `dma_gather` became one `dma_land: Land{len, extent_start, extent_len,
  zeroed}`; each `Extent` owns its items, and a staged job is a one-extent
  landing job. Kinds: dma_read, dma_land, dma_fill, host_direct,
  host_bounce, host_fill. A Transfer is `{destination, landing, writer_mask,
  destination_offset, len}`; the block is found once, at enqueue.
  `Destination.Delivery{dma, host, gathered}`: `sortedPartOrder` sorts by
  uri, class, offset and size, and `preparePlan` cuts three slices.
  `Planner.Reads{io, vfs, policy, alignment}` is built once in `create`
  (`Loader.reads`); `boundOf(uri)` gives a file's widest alignment without
  opening it. Gone: `Config.alignment`, `Loader.direct_io`, the per-submit
  Reads, `Reads.failed_source`, `readLimit`, `JobCut.read_limit`,
  `Plan.direct_reads`, `Staged.data_offset` (the fiftieth pass kept it),
  `class_order`. All 184 plans of `//zml:test` hashed identical before and
  after (dma_staged 18 + dma_gather 298 = dma_land 316). -56 against -75.
  Deviations: `preparePlan` takes `source` (the file of a planning-time open
  failure), since a Reads built once cannot hold it; `finish(arena)` has no
  `source_slot` (JobCut holds it); the host_bounce bound assert stays (the
  last block holds stale bytes past the read); run's bounds assert moved to
  enqueue, after the read; `staged_jobs` and `gather_jobs` stay in
  `logBatch`, derived per landing job; the class partition moved into
  `sortedPartOrder`, outside `planning_ns`.
- M14, one scatter, one block primitive ([48], [45], [47], [49]):
  `FileReader`, `BytesReader` and the `anytype` readers are gone.
  `readSlot(loader, slot, iovecs, file_offset, minimum_len)` opens the
  slot's file, reads through `readGated` and records `open {s}` or `read {s}
  at {d}, {d} bytes at least`; every worker read goes through it
  (`ReadRequest.readBlocks` too). `HostScatter` and `ScatterSink` fold into
  one concrete `Scatter`; `blockPiece` replaces `copyToBlocks`,
  `copyFromBlocks` and the split loops. `Dma.enqueueBlocks` became
  `Dma.enqueue(request, blocks)`: it counts pieces per device on the stack,
  then takes every pump lock, checks the sticky error, reserves the queues
  and takes the piece references (no rollback); `Scratch.queue_counts` and
  `host_iovecs` are gone. `retireBatch` drains every pump's retired stack;
  `EventContext.err` is gone and onReady frees its PJRT error. -110 against
  -105. Deviations: `blockPiece` clamps and `enqueue` asserts `data.len ==
  transfer.len`; one flush rule for both former scatters; the shared ensure
  loop skips consecutive transfers to one destination, so `tensor_init_ns`
  sums fewer calls. A failed read's line no longer names the job kind and
  gives the bytes needed (16769664) rather than the widened read (16773120).
- M15, gates and shutdown ([46], [43], [37], [34]): `RequestGate` became
  `read_permits: std.Io.Semaphore` of `read_width`; `readGated` waits
  uncancelably, then checks `first_error` once and returns the loader's
  first error. Gone: `Credits.close`, `Scheduler.stop`, `unclaimed_total`,
  `failed()`, `errorValue()`, `error.Closed` (log.zig's clause too;
  `publish` returns `Canceled` once failed), and five scattered failure
  checks (Part 2 said four). `recordError` sets `first_error`, logs, fails
  the scheduler and aborts ready DMA. `stopWorkers` asserts the queue is
  empty (`fail` now drains it), calls `Scheduler.fail` and awaits the
  workers. `stdx.FirstError(E)` (`set`, `get`) replaces seven hand-encoded
  first-error atomics: both loaders, ChunkedRead, dma_calibration,
  platform's allocator warm-up, mock_server and http_acceptance_test (new
  against master, so no VFS-diff cost). +40 against -100 (8 files
  +299/-259): direct_loader +29 (outside test blocks -76, the test helper
  `unclaimedJobs`'s +10 included; test blocks +105, the two new tests 81
  and 26 lines), `FirstError` and its test +27, the six other files -16.
  `first_error` is `FirstError(anyerror)`, not Part 2's
  `FirstError(StickyError)` (the error rules; the pjrt sites and ChunkedRead
  keep their sets); `Scheduler.stopping` stays. The sixteenth pass's
  rejection of folding `stop` into `fail` is now done, and the thirty-fifth
  pass's "the read gate closes only after a recorded error or in `destroy`"
  no longer applies: nothing closes.
- M16, calibration returns its block ([50], [51], [54]): `Result`,
  `Result.default` and `max_in_flight_per_device` are gone;
  `dma_calibration.depth` (8) is a constant and `cpu_block_size` (4 MiB) is
  returned on CPU without measuring; `calibrate` and `measure` return the
  block. The confirmation's 25 ms, 256 transfers and 0.02 margin are private
  constants; `validate` still checks parallelism 1..32, the block list,
  `minimum_duration` and the tolerance. `Report` and `measureTransfer` are
  inlined; `Session` owns the handles, `transfer`, `warmup` and one
  `FirstError(pjrt.ApiError)`. `Workspace.max_mapped_bytes` (eighth pass)
  became the constant `mapped_bytes_ceiling` (16 GiB). -57 against -50.
  Deviations: `cpu_block_size` and `Candidate.block` added; the calibrate
  failure line dropped `mapped_ceiling=`; host_memory's ceiling tests are
  rewritten against the constant, not deleted.
- M17, buffered backend ([24], [27], [22] with [25]; the fallback [23] not
  needed): `stdx.LimitedGroup` and its five tests are gone (-216). Master's
  own `LimitedGroup` goes with it (only master's zml/io.zig used it), so
  stdx's diff against master is 2 files +28/-59 instead of growth. A batch
  is `{specs, progress, next, workers: std.Io.Group}`: `submit` starts
  min(width, specs.len) workers with `concurrent`, never `async`, and runs
  the batch inside `submit` when none start; `awaitBatch` waits with cancel
  protection blocked. `loadOneStaged` went; each step calls `fail(err, step,
  spec, part)`. One `Staging{budget, live_bytes, free, retained_bytes}`
  replaces `StagingAdmission` and `StagingPool`: `acquire` takes the
  smallest kept buffer that fits without waiting, else waits for room, frees
  too-small kept buffers and allocates; `release` keeps the buffer. The
  budget is min(width, 12) x the largest `stagingBytes` (tensor plus widest
  strided span), kept buffers included. `readInto` lowers its claim width
  when fewer helpers start than permits taken. -149 against -310: `Staging`
  keeps best-fit reuse and eviction, which the serial 14.96 GiB load needs
  (ninth pass). Deviations: workers are W per pending batch, not "up to 2W"
  (2W with programs or two starts, 3W in llmd's two-start loads; 384 threads
  only at read_parallelism 128 with three starts); "except a request that
  runs alone" is dropped, since `acquire` asserts `size <= budget`; small
  tensors on kept large buffers run about 12 wide rather than the read width
  (kept: a 2x eviction rule would map per tensor in a serial load); a failed
  transfer logs `spec.parts[0]`. Fixed: the fifty-third pass's "buffered
  helper reads run inline under a busy Threaded io", the unheld budget and
  the latent hang on a canceled `LimitedGroup` admission (Part 2's bugs
  met). The thirty-first pass's watch task halving the buffered remote
  parallelism under a finite `async_limit` no longer applies, and the
  fifteenth pass's staging at "`min(width, 12)` tensors" is now the byte
  budget above.
- W2F, the wave-2 review's five confirmed code findings (+73/-65): the
  worker's catch logs only `source request: file={s}, job={t}`, since
  `readSlot` records every open and read and only an OutOfMemory (bounce
  buffer, queue reservation, logged at debug) reaches it; offset and minimum
  length, which the forty-sixth pass kept on the request's line, are on
  readSlot's line. A failed `onReady` awaits and destroys its event at once,
  and `retireBatch`'s event loop went (it drains the pumps and asserts every
  request completed). The buffered `loadOne` re-checks `first_error` after
  acquiring its staging: after `abort`, a worker parked in `Staging.acquire`
  still loaded its whole tensor, up to about 3W-12 tensors at W=32; a new
  test fails 'expected 0, found 1' without the check. `staging_tensors`'s
  doc lost its history and `Staging.widen`'s a wrong reason. `writeTestFile`
  replaces six copies of a tmp-file block (seven callers with M18's test).
  Refuted: that M12's arena, M15's reads in flight and M17's staging high
  water need a probe in the A/B. None is logged, but the last two hold by
  construction (the `read_permits` semaphore; `Staging.acquire`'s assert),
  both exercised by CPU tests, and M12's was already recorded.
- M18, completion self-checks PJRT already does ([28], [30], [56]'s
  setBufferError half): first the test 'a default-memory output a failed
  load left unfinished reports an error on every device' (`testing.Devices`
  with 2; a replicated u8 tensor of 3 requests whose reads fail from the
  third; each shard's ready event must carry an error). Gone: awaitBatch's
  `SetBufferError` pass with its double-failure CHECK abort,
  `Target.closed`, `HostWritten`, `host_written`, `host_total`,
  `TensorTransfer.fullySubmitted` (inlined), pjrt's `setBufferError` and
  `setBufferErrorUnknown`. Destroying a batch destroys its transfer
  managers, and XLA's `CommonAsyncHostToDeviceTransferManager` destructor
  errors every buffer whose last transfer never went out: 'Internal: Async
  transfer object was deleted before transfers completed.' A host
  destination keeps only the debug `unread_bytes == 0` check (Part 2's
  accepted cost); its zero fill (host_fill returns before `recordRead`) is
  covered only by CPU content tests. -16 against -70. Deviations: Part 2
  counts the fifth pass's oneAPI abort among the removed CHECKs, but that is
  the submit-path CHECK in `TransferRawDataToSubBuffer` (through
  `transferData`), unchanged; "the `Internal` failure for a batch that
  missed a last transfer" has been a panic since before wave 2, kept for DMA
  targets. Supersedes the fifth pass's `Target.closed` marking (its "a
  buffer failed asynchronously, then marked, still aborts" is gone with the
  marking), the tenth pass's `setBufferErrorUnknown` (its transferData half
  holds) and the twenty-second pass's `host_written` count.
- M19, one host-memory arena list ([52]): `HugePageAllocator`,
  `PinnedHostAllocation`, the `PjrtHost` and `Pages` payload types, the
  per-arm `allocations` lists, `arenaCount`/`arenaAt`, `Backend.allocate`,
  `discovered_nodes` and the running `Workspace.mapped_bytes` are gone.
  `Workspace.arenas: std.ArrayList(Arena{bytes, pjrt_buffer})` with
  `mappedBytes()`; `Backend` is `union{pjrt_host{platform, host_nodes},
  pages{platform, numa_mask}}` with `map` and `unmap`; `HostNode.node: ?u32`
  dedupes nodes. `allocate` reserves the arena's slot before mapping, so a
  mapped arena is always retained and counted by the ROCm balancer. Log
  lines are unchanged. -73 against -40. Deviations: `mappedBytes()` is new
  pub surface; `place`, `pageAlignment` and `adviseHugePages` are helpers
  inside `Backend`. This replaces the `HugePageAllocator` of "API follow-ups
  after the tenth pass" and of the thirty-fifth pass (#35's `free` is
  `Backend.unmap`) and the sixteenth pass's `Pages` payload.

Checks: `//zml:test` 376 passed, 29 skipped, against the fifty-fifth pass's
373: -1 M12 ('plan construction releases allocations and takes transfers
only on success'); +2 M15 ('exactly the width of reads are in flight, and a
read failing mid-load leaves nothing held', 'a worker waiting for a credit
when the loader fails gets one back and stops'); M17 swaps 'staging
admission bounds concurrent tensors by bytes and always admits one' and
'staging pool retains up to the peak the tasks held and prefers the sizes
that fit' for 'staging holds live and retained bytes within one budget' and
'buffered loader runs overlapping batches over their cursors, with or
without concurrency'; +1 W2F ('a worker waiting for its staging loads
nothing once the loader failed'); +1 M18. `//vfs:test` 54; examples build;
llmd 10/10 (`llmd:test` 256, 9 skipped; `weights_test` 25). Outside the
gate, `//stdx:test` gains `FirstError`'s test and loses `LimitedGroup`'s
five.

Lines (CTX.md and LOADER_DESIGN_REVIEW.md excluded): zml against `rework` 15
files +1,943/-2,457 (-514), against Part 2's about -885 (-775 and -110);
against 93428bd0, 60 files +4,688/-5,938 (-1,250). Monorepo against `rework`
3 files +29/-24 (+5, F1); against 0bf031d9, 31 files +914/-1,412 (-498). Per
step: M11 -12, M12 -93, M13 -56, M14 -110, M15 +40, M16 -57, F1 +4, M17
-149, W2F +8, M18 -16, M19 -73.

GPU A/B. Part 2 asked for a GB300 and B70 A/B against HEAD; it ran on
gb300-1 and mi300 against `rework`. Trees: w1 and w2 as in the fifty-sixth
pass (w2 has no M17); w3 = the final tree. Builds as there, but gb300-1's
examples/io is now release_safe (asserts checked); mi300's stays dbg, on
page-cached checkpoints. gb300-1's examples/io series pin the block
(`ZML_DMA_BENCH_BLOCK_MIB=16`) and direct I/O (`ZML_DIRECT_IO=on`); mi300's
calibrate on every load (16 MiB in every run). Trees are interleaved; log
builds enable `zml/io/load`, `zml/io` and `zml/mem` at debug. Every llmd
run answered "Paris" and "391".

gb300-1 (w1 against w2, then against w3):
- Suites: w3 405 tests, 368 passed, 35 skipped, 2 failed (w2 in the
  fifty-sixth pass; Sharding's GSPMD test and FlashInfer, pre-existing). 15
  io tests skip on aarch64 (no CPU platform), M15's width test among them;
  M18's test runs on the CUDA devices.
- Counters identical in every pair: DeepSeek-V4.1-Flash in examples/io
  sharded over 4 GPUs (with an instrument edit that replicates its
  indivisible tensors, and a BFC pool, `ZML_GPU_MEMORY_FRACTION=0.9`) 30,446
  reads, 413,349 transfers, 431,751 DMA submissions, credits 49 of 49;
  Qwen3.8-27B-FP8 sharded 1,860 reads and 4,919 submissions, tp 424 staged
  jobs and 5,261 submissions, host_pinned and host_mapped 1,860 host jobs.
- Loads within spread (medians, s): DeepSeek 10.12 -> 9.88 (w2, n=5) and
  10.73 -> 10.34 (w3, n=3); Qwen sharded 1.463 -> 1.404 (w2, n=5), not
  reproduced at w3 (1.293 -> 1.299, n=3); tp 1.282 -> 1.268 (w2, n=5) and
  1.275 -> 1.265 (w3, n=7); host_pinned 0.731 -> 0.724 (w2) and 0.586 both
  (w3); host_mapped 0.736 -> 0.737 (w2) and 0.590 -> 0.587 (w3), n=3 each.
  DeepSeek `planning_elapsed` 0.078 -> 0.073 s (w2; every w2 run below
  every w1 run) and 0.073 -> 0.068 s (w3).
- Admission (M11), llmd DeepSeek-V4.1-Flash TP=2 log builds: submissions 41,
  `execute_admission_retires` 39-40, every group measured (inputs 2.32 GiB,
  execution 2.34 GiB), max `pending_execution` 44.43-44.46 GiB and one
  oversize warning per run, on every tree.
- Calibration (M16): `io dma-bench` 10 of 10 `block_bytes=16777216
  parallelism=8`; llmd `dma_bench` 183.5-184.2 GiB/s at 16 MiB on both
  trees.
- M19's lines identical: examples/io Qwen on 2 GPUs `kind=dma_map
  placement=interleave nodes=0x3` arenas of 128/128/288 MiB, retained 544,
  pregrown 416 MiB; DeepSeek on 4 GPUs 128/384/288, retained 800, pregrown
  672; llmd Qwen on 1 GPU 256/160 MiB with a 416 MiB teardown in 26-27 ms;
  llmd DeepSeek TP=2 256/288 MiB, retained 544, pregrown 288, teardown 544
  MiB in 31-34 ms.
- Failure injection: a Qwen3.8-27B-FP8 copy whose last-planned file is cut
  to 160002048 B mid-load, 12 runs (w2) and 14 (w3) over sharded, tp,
  host_pinned and host_mapped: one err line, the error from `wait`, exit 1,
  start to exit 3.8-6.5 s on both trees; no abort, hang, PJRT or CHECK line;
  `destroy` ran (the summary and DMA counters print after the error). One
  read permit (`ZML_LOAD_READ_PARALLELISM=1`): one line, exit 1. chmod 000
  or rm after start fails in the planner's open ('plan batch 0, source=...,
  published_plans=65: AccessDenied') with 11 DMA submissions out, M18's
  case: one line, exit 1 after 2.0 s. llmd on the cut copy: one line, exit
  0.36-0.38 s after it.
- M18's test passes on 4 GB300 (`testing.Devices` with 2 takes all four). A
  probe build printed, per shard, 'ready with error internal: Async transfer
  object was deleted before transfers completed.'
- `ZML_LOAD_CHECK=1`: DeepSeek-V4.1-Flash at w2 96085 of 96085 tensors
  (758.49 GiB checked); Qwen3.8-27B-FP8 at w3, 8 runs over host_pinned and
  host_mapped (sharded, replicated, tp, `ZML_DIRECT_IO=on`) and default
  memory: 1606 of 1606.
- End to end at w3: DeepSeek-V4.1-Flash 5.75 ms/step with the same KV plan
  (5148591 pages); Qwen3.8-27B-FP8 + DFlash2 20.12 ms/step at 3.28
  tokens/step.

mi300 (w1 against w2, then against w3; GPUs 0-3 unless stated):
- Suites: w3 405 tests, 380 passed, 24 skipped, 1 failed (w2 in the
  fifty-sixth pass; Fly, pre-existing). Both M15 tests pass on x86.
- Counters identical: Llama-3.1-8B sharded 960 reads and 2,088 submissions,
  tp 1,842 reads, 3,464 submissions and 331 staged jobs, host_mapped 960
  host jobs; DeepSeek-V4-Flash on one GPU (BFC 0.95) 9,524 reads and 69,572
  submissions, about one per tensor of 69,187 (Part 2's DeepSeek reference;
  its "about 5,468" submissions and 1-4 ms planning are Llama-3.1-8B tp on
  four B70, twenty-fifth and twenty-sixth passes; DeepSeek here plans in
  155-190 ms over 46 plans, dbg); Qwen3.8-27B-FP8 1,860 and 8,131; llmd
  DeepSeek-V4.1-Flash 75,786 reads, 176,318 submissions, 160 staged jobs.
- Loads within spread: Llama sharded 1.165 -> 1.163 s (n=10, rank-sum
  p~0.55), tp 1.173 -> 1.182 s (p~0.41), host_mapped 13.18 -> 13.10 s
  (allocation-bound); DeepSeek-V4-Flash 9.021 -> 8.914 s; llmd DeepSeek
  85.06 -> 84.08 s (log builds; w3log 87.89 and 81.98 s).
- Admission: submissions 41, retires 40 and the `pending_execution` sequence
  identical on every tree; `extra_device_memory_bytes` 50689508496
  throughout.
- Calibration: 16 MiB in every io and llmd load; its time is bimodal on
  every tree (~345-375 ms or ~1120-1195 ms).
- M19's lines identical across 27 runs: `kind=pjrt_host device=0`
  256/256/288 MiB on GPUs 0-3, and 'device=0 256, device=2 256, device=0
  288' on GPUs 0,1,4,5 (the per-node balance); retained 800, pregrown 544,
  pinned_mapped 800 MiB; `retained_mapped_bytes=268435456`; teardown 800
  MiB. pregrowth 238.4 -> 245.7 ms in io (wave 1: 245-249).
- Failure injection: a Llama copy truncated mid-`lm_head` before any read,
  15 runs (5 modes x 3 trees): one line with file and offset,
  UnexpectedEndOfFile from `wait`, `destroy` 13-20 ms after the error with
  default memory and 1.9-2.0 s with host-resident memory, alike on every
  tree. An LD_PRELOAD shim returning EOF or EIO mid-file (w2): the same. A
  real device-allocation failure (DeepSeek-V4-Flash on one GPU under the
  default allocator): one 'initialize tensor ... ResourceExhausted' line,
  `destroy` 0.5 s later. No PJRT line names the errored buffers: nothing
  awaits them before they are freed.
- M18: the suite passes it; alone, 10 of 10 on 4 and on 2 MI300X; a probe
  printed the same 'Internal: Async transfer object was deleted before
  transfers completed.'; no CHECK abort or hang.
- `ZML_LOAD_CHECK=1` at w3: Llama host_pinned and host_mapped, sharded and
  replicated, direct I/O auto and on: 291 of 291 in all six.
  DeepSeek-V4-Flash at w2: 69187 of 69187.
- End to end at w3: Qwen3.8-27B-FP8 + DFlash2 39.25 and 38.61 ms/step at
  3.37-3.39 tokens/step; DeepSeek-V4.1-Flash 466.95 GiB in 2m3.4s (86.8 s
  waiting for compiles), 12.53 ms/step, the same KV plan.

Findings:
- DeepSeek examples/io on gb300-1 exits ~2 s sooner since wave 2, all in
  teardown: wall 17.33-17.88 s (w3) against 19.27-20.04 s (w1) at the same
  load time; loader summary minus 'Loaded weights' 0.485-0.502 s against
  1.107-1.354 s. Likely M12's one arena per batch; not traced.
- MI300 staged tensor-parallel loads hold more credits and spend longer per
  DMA submission since w2, at no cost. llmd Llama-3.1-8B over 4 GPUs (474
  staged jobs; log builds, w1 against w3), 5 runs each:
  `credits_high_water` 25, 23, 25, 25, 25 -> 28, 26, 28, 31, 35 of 49
  (p~0.008), `dma_submit_us_per_piece` 13.65 -> 19.03 (p~0.03), load 0.894
  -> 0.855 s; io tp (331 staged jobs; w1 against w2) dma_stage 2.89 -> 3.41
  ms/read. Absent from sharded loads, DeepSeek-V4-Flash and GB300's tp.
  The timed code matches w1's; suspected: M13's `dma_land` changes when
  staged jobs reach the queues. Credits are fixed slots in mapped memory, so
  the higher peak costs nothing. To confirm (a code change): time
  `transferData` and `onReady` apart in `submitTransfer` and count the
  callbacks run inline (`dma_piece_ns` near 0).
- llmd's DeepSeek planning counters vary between runs of one binary on
  gb300-1: 71,134 / 71,122 / 71,120 DMA submissions and 60,607 / 60,697 /
  60,712 reads, batch 0 identical. Not a tree difference. Guess, unchecked:
  per-file direct-I/O auto decisions that depend on the page cache
  (examples/io with `ZML_DIRECT_IO=on` and mi300's page-cached llmd repeat
  exactly).
- llmd still flips between 8 and 16 MiB on GB300 (thirty-third pass; llmd
  cannot pin the block): 8 MiB in the first DeepSeek log pair on both trees,
  once per tree in the Qwen log series, once in an end-to-end pair. Arenas
  and counters follow the block (Qwen on 1 GPU: the 256 MiB ring alone at 8
  MiB, 256 + 160 at 16; 3,535 reads and 4,465 submissions against 1,776 and
  2,705), so llmd load A/Bs are confounded; examples/io pins it.
- Not the loader: one 18.43 s end-to-end DeepSeek load at w3 beside another
  load at host load 37 (three later w3 loads took 9.64-10.06 s); one 19.24 s
  w2 load, the Engram registration stall (ahead_init 11.3 s; thirty-third
  pass); one compile-time transient to 281,191/283,143 MiB before any load.
- `min_room_seen` reaches 0 B in some DeepSeek runs on every tree: the
  pre-existing admitted-anyway path. A proposal to record the device and
  submission of the minimum was not taken.

Declined: the mi300 report proposed making M18's test print the plugin's
message (a `std.log.warn`, or the `output.await` the review deleted). The
test asserts that every shard's event is ready with an error, a passing test
prints nothing, and the message is recorded here.

Left:
- Not reached (Open work lists them): the DMA-callback failures, which need
  a plugin-level fault injector (M18's double failure, M14, W2F's failed
  onReady, M15, M16); a remote source failing mid-load; Furiosa, oneAPI
  (the B70 A/B) and the buffered backend (M17, W2F); M12's arena, M15's
  reads in flight and M17's staging high water, none logged; the buffered
  start counts and a planning failure beside a stuck one, untested.
- readSlot's open failure was not reached: when `io` is a VFS and direct
  I/O is auto or on, the planner opens each local file before its reads
  (`Reads.alignmentOf`), and an open descriptor survives chmod and rm. With
  `ZML_DIRECT_IO=off` or an `io` that is no VFS, the first read opens a
  local file, so a chmod or rm after `start` would reach readSlot; so would
  a remote file's open failure (neither run).
- W2F's buffered test relies on a 5 ms sleep to park its worker; it cannot
  fail falsely.
- Left as records: LOADER_REVIEW.md:452 and :758 build on `host_written`
  (if that CUDA registration-stall fix is ever taken, trigger on
  `Destination.unread_bytes` reaching 0); :718 proposes what M18 did
  (SetBufferError, `Target.closed`); PLAN.md:103-104; Part 2's texts for
  M12, M15, M17 and M18 (the deviations above).

With the records, two code doc comments were fixed (the `DevicePump`
comment no longer names `IncompleteTransfer`; `Options.read_parallelism`
says the direct backend clips so `width + 1` requests fit under the mapped
ceiling beside its DMA reserve), and "Current design"'s CPU pre-growth was
measured: Llama-3.1-8B sharded on four CPU devices (opt) logs
`retained=268.00MiB` in two pageable arenas of 128 and 140 MiB,
`source_width=16, lifecycle_credits=33, workers=17`, 14.96 GiB in 1.798 s.

## Open work

Status on 2026-10-31: wave 1 is committed as `rework` (zml 8af7fd3c,
monorepo b791da11); waves 2 and 3 (M11-M19), F1 and W2F are in the tree,
uncommitted on top. All of it ran on gb300-1 and mi300 (fifty-sixth and
fifty-seventh passes): no loader regression (one perf-neutral MI300
credits shift, below), and F1 fixed what wave 1's runs found. Left of
`LOADER_DESIGN_REVIEW.md`: M20 (outside the rework, a separate PR) and the
Part 3 decisions not taken. "Current design" describes the final tree.
Closed since 2026-10-29: the MI300 block-FP8 item (accuracy, M7's assert,
Flash-Next-FP8 with F1) and most of wave 1 on a GPU (borrowed dims, M10's
walks on 4 GB300 and 4 MI300X, the drafters, `hostMappedBytes`,
examples/io on CUDA and ROCm), fifty-sixth pass; the wave-2 GPU A/B and
read failures injected on CUDA and ROCm, fifty-seventh pass.

Status on 2026-10-29: wave 1 of `LOADER_DESIGN_REVIEW.md` (M1-M4,
M7-M10) and the VFS re-scope's steps are in the tree, uncommitted, none
run on a GPU or a real remote (fifty-fifth pass); waves 2 and 3 (M11-M19)
are open.

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
- GB300 MoE kernel tests and the aarch64 CPU plugin (fifty-first pass):
  the Triton and CuTe MXFP4 matrices abort the binary and are skipped by
  hand, the FlashInfer matrix fails MoeComplianceFailed, and without a CPU
  plugin Sharding's GSPMD folded-axis test fails and 15 io tests skip,
  M15's width test among them (fifty-seventh pass). M7's NVFP4 paths
  (FlashInfer NVFP4 experts through `layOutExperts`, the NVFP4 global
  programs) never ran: no NVFP4 checkpoint on gb300-1 or mi300 has a model
  type llmd supports (mistral_large4, beam, lfm3), and ROCm refuses nvfp4
  MoE (zml/kernels/moe/moe.zig:110). Block-FP8 experts through
  `bindQuantized` and cute MXFP4 ran (fifty-sixth pass).
- Wave 1 on a GPU, left (the fifty-sixth pass ran the rest):
  `expectShards`' memory argument against real host spaces (the MI300 io
  tests pass; no report names it); examples/io on oneAPI and from a remote
  checkpoint; the buffered backend on TPU, neuron or metal (M3's
  `.default` assert, M10).
- Real checkpoints for M7's detect changes: the routers, gates and heads
  now built by `linear.create` or `maybeCreate`, a quantized lm_head or
  g_proj (now loaded quantized; none of the FP8 checkpoints gb300-1's llmd
  runs used stores one, and mi300's were not inspected), DeepSeek's
  `createBf16` `wo_a` (exercised only by DeepSeek answering correctly).
  The `DSparkDrafter.init` leak fix's failure path. zoo was built, not run
  (it needs network and an accelerator).
- Waves 2 and 3, not reached by the GPU runs (fifty-seventh pass): every
  DMA-callback failure needs a plugin-level fault injector: M18's
  double-failure path (an asynchronous DMA failure, then an unrelated read
  failure), M14's onReady freeing its PJRT error, W2F's failed-onReady
  branch (no CPU coverage either), M15's `recordError` from a callback
  while workers wait on credits, M16's failing transfer during the
  screen. A remote source failing mid-load. Furiosa (M18, M19's
  pjrt_host arm). The buffered backend (M17, W2F) on TPU, neuron or
  metal: the staging high water and kept bytes against the min(width, 12)
  x largest-request budget on a dense checkpoint, a serial
  (read_parallelism 1) load of the 14.96 GiB Llama-3.1-8B, an hf://
  Qwen3.5-4B load at width 32 (small tensors on kept large buffers, helper
  splitting), and the peak thread count and first-batch retire time in a
  two-start load. The B70 (oneAPI) A/B against `rework` that Part 2 asked
  for: host-resident load time and read_operations after M14, and the
  time from a failure to `wait` with `tensor_init_ms_per_read`, since M15
  lets a job claimed before a failure create its destinations (at most
  width + 1 jobs). Not logged, so unmeasured: M12's arena
  (`Batch.arena.queryCapacity()` at retire) and peak host RSS on DeepSeek,
  and reads in flight against `source_width` (a CPU test only, skipped on
  aarch64). No test asserts the buffered workers' and helpers' start
  counts (M17) or drives a planning failure while another failure is
  stuck (M12).
- MI300 staged tensor-parallel loads since wave 2 (fifty-seventh pass):
  `credits_high_water` 23-25 -> 26-35 of 49 and `dma_submit_us_per_piece`
  13.65 -> 19.03 on llmd Llama-3.1-8B over 4 GPUs (log builds, w1 against
  w3), load equal or faster; cause unconfirmed (suspected: M13's
  `dma_land` timing). Confirm by timing `transferData` and `onReady` apart
  in `submitTransfer` and counting the callbacks run inline (a code
  change).
- After F1 (fifty-sixth pass): qwen4 logs no KV page count, so MI300's
  count under the new page cost is unmeasured and GB300's FP8 at 0.3 was
  not dumped (a page-count log line would close both); Flash-Next BF16 on
  2x GB300 at the default fraction was not run, and the fifty-first
  pass's need for 0.6 is unexplained (at the default fraction FP8 sees
  236.50 GiB available, above BF16's 166.46 GiB fixed); MI300's FP8
  greedy loop (1 of 6 runs on w2, 0 of 5 on w1fix) rests on few samples
  and was not compared with CUDA.
- Pre-existing, examples/io: its `sharded` mode splits axis 0 of every
  tensor of rank > 0 (examples/io/main.zig:276-279), so no DeepSeek-V4.x
  checkpoint binds over 2 or 4 GPUs: 89 tensors of V4.1-Flash (90 of
  V4-Flash), 86 of them f32[3] `hc_attn_scale` and `hc_ffn_scale`, fail
  with InvalidBinding (thirty-third pass, as `IncompatibleSharding`).
  Proposed, not applied: replicate a tensor whose dim(0) % partitions !=
  0, or an explicit mode.
- Pre-existing, device allocators on examples/io's DeepSeek loads: CUDA's
  default allocator fails with ResourceExhausted ('cuMemSetAccess for
  peer device: CUDA_ERROR_OUT_OF_MEMORY') after ~97 GiB of
  DeepSeek-V4.1-Flash sharded 4 ways (~96k buffers per device); ROCm's
  default VMM allocator runs out of PA budget at 122.6 GiB of
  DeepSeek-V4-Flash on one GPU. A BFC pool (`ZML_GPU_MEMORY_FRACTION` 0.9
  or 0.95) loads them in ~10 s and ~9 s. Causes guessed: per-buffer VMM
  and peer mapping; per-buffer rounding over ~34k small scale tensors.
  llmd's DeepSeek (stacked experts, fewer buffers) is unaffected.
- Pre-existing on mi300 (fifty-sixth pass): moe's Fly matrix cases 38,
  39, 41, 42 (geglu, MXFP4) fail MoeComplianceFailed
  (zml/kernels/moe/tests.zig:502); on 8 GPUs the suite aborts at the
  paged-attention decltest, hkv=4 over 8 partitions
  (paged_attention.zig:457; build its mesh over min(devices, hkv)
  devices, or skip when hkv % devices != 0); DeepSeek-V4.1-Flash loads in
  84-117 s against ~10 s on GB300 (its 189 GiB host_mapped engram
  allocation at ~2.8 GiB/s overlaps 59-91 s waiting for loader compiles
  that start only with the load; with the native drafter it waited 0 ms,
  not investigated); Qwen3-4B-Instruct-2507 answers 17x23 = 401 at
  temperature 0 (not checked on CUDA); Flash-Next-FP8 does not fit one
  MI300X (NormalizeBlockFp8Weight).
- Pre-existing, admission: 'program group exceeds the device room alone
  ... admitted anyway' on DeepSeek-V4.1-Flash (both hosts), Flash-Next-FP8
  and Qwen3.8-27B-FP8 (2 MI300X), and `min_room_seen` reaches 0 B in some
  DeepSeek runs. Proposal, not taken: record the device and submission of
  the minimum (loader.zig:631) and print them in 'loader admission:'.
  That line prints min_room_seen=0B when no program group ran (the null
  prints as 0, loader.zig:712).
- Pre-existing on gb300-1: Flash-Next-FP8 decode is bimodal per process
  (~9.0 or ~12.6 ms/step, per-process autotuning); its fixed memory went
  87.13 -> 86.38 GiB and decode 11.65 -> ~9.0 ms/step between the
  fifty-second pass and the rework's parent, unexplained. llmd's DeepSeek
  planning counters vary between runs of one binary (71,120-71,134 DMA
  submissions; guess, unchecked: direct-I/O auto decisions that depend on
  the page cache).
- No real remote (HF, S3, GCS, HTTPS) since R5, R6 and R15, nor since the
  fifty-fifth pass's VFS steps: the remote totals line was never seen
  printed; untested are hf's streaming and seek fixes, signed S3 through
  the governor, any GCS request with a token (the token lock, the
  metadata server), HF's tree fetch no longer stalling reads, and a real
  peer stalling mid-body (thirty-eighth, thirty-ninth, forty-seventh,
  fiftieth, fifty-fifth passes).
- The handle-table move (LOADER_REVIEW Move 10, remote backends as
  stores; high risk, to be staged with `http_acceptance_test` through a
  VFS): rename, symlink and hardlink to remote paths, VFSBase's
  forwarders, atomic create's handles (forty-seventh, fiftieth passes).
  Also from Move 10: GCS fetching its metadata token as a plain request
  (the re-scope's X7, not taken; a non-2xx from the metadata server logs
  one err line since the fifty-fifth pass).
- `testLayer` cannot fail; the user's call (forty-ninth pass).
- The user's calls from the fifty-fifth pass: the VFS re-scope's optional
  X2 (master's dead code in the remote backends, about 130 lines) and X3
  (master's constructor shapes, about 45); D9; `VFSBase` stays private.
- Left by the fifty-fourth and fifty-fifth passes: a compile failure is
  logged twice (the Compiler with its MLIR, then the loader);
  `//examples/mnist:mnist_dp` aborts at `profiler.stop()` on CPU, before
  and after the wave; no test shows examples/io's `checkLoaded` catches a
  corrupted buffer; llmd's ops/mla/state.zig keeps a dead
  `deinitOwnedParameterBuffers`.
- Docs: the tutorial has no build target; loader.md is 221 lines;
  LOADER_*.md say `getSourcesById`; `bazel/huggingface.bzl` reads
  `HUGGINGFACE_TOKEN` (forty-eighth, forty-ninth passes).
- Known limits, documented: a denied local file opened by name in
  `fromPath` is `AccessDenied`; the `deinit` warning fires on a caller's
  own failure without `errdefer loader.cancel()`; a source that reads no
  bytes (after `pad` or `slice`, or an empty checkpoint tensor) logs its
  target but no key, and so does a sub-byte refusal, and a whole sub-byte
  tensor whose bits are not a multiple of 8 is `InvalidBinding`
  (forty-fifth, fifty-fifth passes); a `dma_block_sizes` block under
  32 KiB makes longer jobs (fiftieth pass).
- A model's free path (thirty-sixth pass) and a progress bar watched to
  the end (thirty-seventh), both changed by M8: every GPU llmd run was
  stopped by its PID. The measured admission (the unlanded-bytes room,
  temporaries as a maximum; M11 rewrote it) and host-pinned routing ran
  on both GPU hosts (fifty-seventh pass).
- ExpertLayout: both arms (`.loader`, `.device`) stay (D7); choosing one
  stays the user's call (forty-fourth pass).
- LOADER_REVIEW items open since the thirty-first pass: S3 credentials
  and key encoding (#5, #12), GCS key files (#16), and the rest of #20's
  header checks (a tensor's byte size against `end - start`, `F8_E5M2`,
  duplicate names across shards; other missing or mistyped fields still
  panic; thirty-fifth pass). #6 went with R6, #8 in the thirty-fifth
  pass, #13 (https redirects) in the fifty-fourth; #21 is moot
  (`loadExecute` went in the thirty-fourth).
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
  through the same `SharedHostAllocation`; not re-profiled. The stall
  showed again in one wave-2 GB300 load: ahead_init 11.3 s (11,346 ms;
  fifty-seventh pass).
- The calibrated request size (thirty-third pass): on GB300, 8 MiB sits
  on the DMA screen's 8% tolerance line, so the choice flips (4 of 10
  replicated and 2 of 10 sharded Qwen runs, 7 of 48 DeepSeek runs), and
  8 MiB costs Qwen3.8-27B 25-30% (2.54 against 2.04 s replicated). The
  B70s want 8 MiB (10-30% faster); no fixed size suits both, and the
  calibration measures DMA only. The rule also chose 2 MiB on a degraded
  MI300. Calibration caching per host/plugin, or re-screening on an
  implausibly low rate, is open. The user declined a fixed block size
  (thirtieth pass): do not propose removing calibration. llmd still flips
  on GB300 at w1, w2 and w3, and the arenas and counters follow the block
  (fifty-seventh pass).

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
  (out of bounds in safe builds; the load check skips them; fifth pass).
  The sub-mesh half (twentieth pass) went in M10; `Buffer.uninitialized`
  on a sub-mesh has no test (fifty-fifth pass).
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
  manager whose event errored and surface the plugin error. M18 does not
  touch this submit-path CHECK (fifty-seventh pass).
- `--test_filter` is ignored by the rules_zig test runner, and so is a
  `--test-filter` in `zig_test`'s `zigopts` (handoff rules).
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
7. `LoadProfile` (now `VFS.readHints`) and model-wrapper plumbing.

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
  use `registerBackend` so file/HTTP/HF/S3/GCS read hints (then load
  profiles) and counters reached the loader. Re-check that integration
  rather than assuming a historical uncommitted edit still exists (may be
  stale: `VFS.Remotes` registers the remote schemes since R6, and the
  loader reads no VFS counters since R15).
- rules_zig ignores a `--test-filter` passed through `zig_test`'s
  `zigopts`: every test runs. To run one test on a GPU host, run the
  whole suite (on GB300 with the moe Triton and CuTe MXFP4 matrices
  skipped, since they abort the binary), or drive the test binary's
  `--listen=-` protocol directly.
