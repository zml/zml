# Direct DMA loader context

This is a compact, agent-oriented handoff. It records current behavior, durable
decisions, useful measurements, rejected approaches, and open work. It is not a
runbook. Re-check code, Git refs, available accelerators, and plugin artifacts
on each machine before relying on an old result.

Last consolidated: 2026-09-03 on `brabier/adaptive-concurrency`, after
`11d87cf3` (coalesced reads) and `95b1e730` (tensor-aware cuts, batched enqueue,
worker scratch). `origin/master` was `e1e983c8` during the 2026-09-02 audit;
never assume that ref is still current.

## Current design

### Scope and API

- The model-wide `zml.io.load`/`loadInto` direct path is used for CUDA, ROCm,
  and oneAPI. The buffered CPU/TPU/Metal loader is unchanged.
- `VFS.loadProfile(path)` is prepared once for a model load and passed as a
  borrowed `LoadProfile`. It contains a backend name, minimum read chunk,
  `high_latency`, and optional aggregate retry/throttle feedback. It assumes
  the load is the backend's only material user; feedback is not load-tagged.
- Profile minima are local/file 8 MiB, HTTP/S3/GCS 16 MiB, and HF 32 MiB.
  Effective source request size is the greater of the profile minimum and
  calibrated DMA block size, capped at the supported 32 MiB maximum.
- No public API changed for source-read coalescing. Existing fixed-width
  overrides remain supported.
- The older public `Loader`, `DirectMemoryWriter`, flip-flop
  `DirectShardWriter`, `DynamicBufferPool`, and fused-tensor `loadExecute` path
  remain for compatibility. No in-tree caller used the old public loader at
  the 2026-09-02 audit. Multi-source/fused tensor capability must move before
  removing it.

### Source and VFS data plane

- `TensorReader.readPositionalAllV` provides checked, exact positional scatter
  reads, short-read resumption, local `IOV_MAX` batching, and borrowed readers
  over a shared open file. Each distinct safetensor object is opened once per
  model-wide load.
- HTTP, S3, GCS, and HF issue one whole Range request per admitted positional
  call. Retries are serial inside that caller's source credit; the retired
  backend-local `parallel_read` pools must not return.
- Shared Range handling validates a covering `Content-Range`, handles a server
  returning `200` and ignoring Range by discarding the prefix, scatters into
  caller buffers, retries with jitter, and exposes aggregate request, retry,
  throttle, byte, and delay counters.
- One source job performs one exact absolute scatter read into pinned blocks.
  Extra physical calls occur only for short reads/retries or `IOV_MAX` limits.
  Diagnostics distinguish planned jobs from physical calls.

### Coalesced source planner

The old planner made jobs tensor-local. `source_request_size` was only a cap,
so small adjacent tensors each caused a source operation. The current
`prepareBatch` instead:

1. Sorts selected ranges by file URI and absolute source offset.
2. Forms the union of touching or overlapping requested ranges. It never reads
   across an unrequested gap or file boundary. Duplicate/overlapping bindings
   are read once but retain a transfer piece for every output.
3. Partitions each merged run into the minimum number of jobs permitted by the
   request/block/`IOV_MAX` limit.
4. Keeps that minimum count while preferring tensor-safe boundaries. For
   `N = ceil(run_length / max_job_length)`, each cut chooses the latest safe
   tensor boundary that leaves the remainder fitting in `N-1` jobs; otherwise
   it uses the latest feasible hard cut. A boundary is safe only if the next
   touching interval begins at the current union end, which preserves overlap
   and duplicate coverage.

Coalescing is intentionally batch-local to one `load()`/`appendItems()` call;
an earlier batch may already be executing. The immutable plan stores source
jobs and flattened transfer pieces once. Jobs carry source slot, absolute
offset, physical length, piece range, per-device physical-byte charges, and a
same-file predecessor. Physical source bytes are distinct from logical tensor
bytes so duplication and replication do not distort diagnostics or fairness.
Planning is `O(tensors log tensors)` and took about 0.40 s for DeepSeek-V4-Flash.

### Pinned blocks, scattering, and ownership

- A worker atomically leases all pinned blocks for one source job, reads into
  them, then maps block-relative slices to existing per-tensor PJRT transfer
  managers. Pieces include tensor, block index/offset, writer/replica mask,
  destination offset, and length. Compatible adjacent pieces are merged.
- The same pinned block may feed several tensor transfers. It is reference
  counted across every consuming PJRT event and released only after all child
  transfers finish or are abandoned. Source/allocation/enqueue/PJRT failures
  close scheduling, fail unfinished buffers, drain the epoch, and release each
  reference exactly once.
- `enqueueBlocks` reserves all affected destination queue capacity before
  mutation, publishes a complete source job under one metadata lock, and pumps
  once. The prior per-piece enqueue caused roughly 69k--79k locks and pumps.
  Pre-reservation makes allocation failure atomic.
- Workers retain scratch for leases, affinities, reference counts, iovecs,
  block contexts, queue counts, and transfer construction. Positional-read
  rewrite scratch is stack bounded. This removed per-job allocation pressure;
  its wall-time effect was within noise.
- Source coalescing deliberately preserves per-tensor device buffers. Roughly
  one DMA submission per tensor is therefore the natural floor. Going much
  lower requires packed device allocations or a device-side scatter/copy stage
  and changes buffer ownership/model layout.

### Scheduling and concurrency

- The scheduler charges a coalesced job's physical bytes to every destination
  device and retains cross-device fairness. Same-file predecessor claims and
  per-target submitted-prefix accounting preserve ordering while unrelated
  reads and DMA complete out of order.
- Source concurrency is adaptive for every direct-loader profile. The old
  conversion of local/default `.adaptive` to `.fixed = 12` was removed.
  `high_latency` only permits blind pre-response bootstrap to 24 then 32.
- Default source configuration remains adaptive initial 12, maximum 128,
  clipped by remaining jobs and pinned-memory feasibility. Twelve is an
  empirical bootstrap, not a value derived from tensor count, request size,
  storage queue depth, or bandwidth-delay product.
- The source ladder is `1,2,4,8,12,16,24,32,48,64,96,128`. Ninety-six is only
  a ladder rung and was useful as a fixed S3Proxy control; it is not a default
  or model-derived number.
- Completed jobs contribute their actual byte count to adaptive evidence,
  including partial tails. The controller uses clean 100 ms generations and
  selects the smallest width within 3% of peak; unfinished finite-tail probes
  roll back. Metadata can cheaply clip feasibility, but cannot predict a
  source's latency/bandwidth saturation point, so no job-size-derived initial
  width heuristic was added.
- Separate gates represent the stable worker set, clean read-measurement
  generations, and complete request lifecycles. Lifecycle capacity is active
  reads plus one shared spare, clipped by exact pinned feasibility. A request
  returns lifecycle credit only after all its DMA children finish.
- DMA width is fixed at eight per device by default after calibration work
  showed adaptive DMA width added substantial complexity and little load
  value. An optional calibrated global cap is applied only if the active device
  count can exceed it.

### DMA memory and calibration

- `DmaBlockPool` is a load-scoped view over platform-owned arenas. It supports
  blocking atomic multi-block acquisition, callback reference leases, a hard
  mapped-byte ceiling, demand growth, NUMA reserves, and augmenting-path
  affinity assignment. Matching is correctness logic: greedily assigning a
  replicated block can consume the only block usable by a later strict-local
  request.
- Calibration resources/settings belong to `Platform`. Conservative defaults
  work without calibration; `benchTransfer` atomically replaces them and its
  arenas are retained as the loader's initial pool. Platform state prevents
  calibration, loading, inspection, and teardown from borrowing the workspace
  concurrently.
- Current detector screens DMA blocks 2/4/8/16/32 MiB at width eight. Default
  screens require at least 2 ms and 32 completions. Borderline results use
  three alternating pairs at 25 ms/256 transfers. The 8% near-peak rule favors
  a smaller block. Global-cap selection requires repeated evidence of at least
  2% aggregate gain, at least 95% per-device retention, and adequate fairness;
  ambiguous results stay uncapped.
- Retained arenas are initial capacity, not the full permissible live set.
  Demand growth remains bounded by the mapped-memory ceiling. Allocation still
  occurs synchronously under the pool mutex; pre-growing the exactly known
  initial deficit is open work.

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
  noisy short screens sometimes selected the wrong block or provisional cap.

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
  DMA could start. Jobs now carry predecessors and become eligible only after
  predecessor claim. Pre-growing memory and ignoring the DMA global cap did
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
  latency/memory more than throughput. Replaced by calibrated block size,
  fixed per-device width eight, and conservative optional global cap.
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

## Validation state at consolidation

- DeepSeek CUDA loads completed repeatedly. Focused loader tests separately
  exercised tensor shape/content correctness.
- VFS tests passed, including exact scatter, retries, and concurrency.
- Source-planner, adaptive-controller, scheduler fairness/predecessor,
  short-read, replication/sharding, NUMA-affinity, failure cleanup, and epoch
  reuse tests passed.
- The aggregate CUDA `zml:test` reached 239/240, then hit an unrelated existing
  `attention.triton_attention` assertion: MLA `num_kv_splits (2)` exceeded
  sparse tile count `(1)`. Loader tests completed before it.
- Optimized playground builds had passed for CUDA, ROCm, and oneAPI during the
  preceding audits. Runtime coverage depends on hardware available per machine;
  oneAPI and remote-service behavior should be rechecked after relevant changes.
- ROCm-enabled aggregate tests were previously blocked before loader coverage
  by an existing CUDA flashinfer import in `zml/moe/cutlass_flashinfer.zig`.

## Open work, in priority order

1. **Reduce DeepSeek planning overhead.** About 0.40 s remains before the
   source epoch. Profile allocations and repeated dispatch-span walks in the
   already-flat representation.
2. **Measure 24/32 MiB after coalescing.** Metadata simulation predicts
   6,356 jobs/69,406 pieces at 24 MiB and 4,774 jobs/69,353 pieces at 32 MiB,
   versus 9,524/69,572 at 16 MiB. Do not change global defaults without CUDA,
   ROCm, oneAPI, local, and remote evidence because larger requests increase
   live pinned memory and have regressed B70.
3. **Pre-grow the initial load working set.** The scheduler knows initial read
   width, one-spare lifecycle capacity, maximum blocks/job, affinities, and DMA
   reserve. Allocate that deficit outside the timed load/pool lock; keep demand
   growth as the bounded fallback.
4. **Stage calibration arenas.** Device tuning is sequential and initially
   needs only `8 * 32 MiB` per participating NUMA node. Grow after block
   selection to `devices_on_node * 8 * selected_block` for the aggregate phase.
   This can avoid eager 2 GiB setup on 4+4 MI300X when 16 MiB wins.
5. **Cache/reuse calibration.** Key by backend, device kind, topology,
   PJRT/plugin, driver/runtime, host identity, and detector version. Resource
   warmup/allocation can dominate short process wall time even after sampling
   became fast.
6. **Define settled-source congestion behavior.** Retry/throttle evidence rolls
   back an unsettled probe, but settled width currently does not decrease. If
   telemetry is only evidence hygiene, document that; otherwise add ongoing
   backpressure. Aggregate provider attribution remains a limitation.
7. **Decide old-loader compatibility.** Remove dead parallel/adaptive helpers
   only after fused/multi-source tensor behavior has a model-wide equivalent.
   Known dead candidates include adjustable `stdx.Io.LimitedGroup` additions,
   unused `DynamicBufferPool` tuning methods, standalone productionless
   `DmaBlockPool` mode, and `VectoredLoadMetrics.resetReadPeak`.
8. **Split `zml/io.zig`.** Data plane, calibration, and block pool are large
   enough to warrant separate modules before upstream review.
9. **Lower DMA submissions only with explicit ownership redesign.** Packing
   tensors into shared device buffers or device-side scatter is invasive and
   independent of source coalescing; assess layout/API consequences first.
10. **Consider completion-aware local pacing.** The lane-coupled diagnostic
    showed a possible ~10% B70 gain, but any solution must preserve remote
    refill and multi-device fairness.

## Suggested upstream decomposition

1. Exact safetensor positional scatter plus VFS Range/retry conversion and
   removal of backend `parallel_read`.
2. Plugin pin containing already-upstream pinned-range detection; ROCm arenas
   through standard PJRT `pinned_host` buffers.
3. Platform-owned DMA arenas, NUMA allocation, and `DmaBlockPool`, independent
   of calibration policy.
4. Model-wide coalescing planner/scheduler/pipeline and `load`/`loadInto`
   migration with conservative fixed DMA settings.
5. DMA block calibration, width eight, and conservative global-cap detection.
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
