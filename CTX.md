# Direct DMA loader context

This is a compact, agent-oriented handoff. It records current behavior, durable
decisions, useful measurements, rejected approaches, and open work. It is not a
runbook. Re-check code, Git refs, available accelerators, and plugin artifacts
on each machine before relying on an old result.

Last consolidated: 2026-09-03 on `brabier/adaptive-concurrency` at `507cabd4`.
`PLAN.md` is the ordered implementation checklist; this file is the canonical
description of the code after each completed task. `origin/master` was
`e1e983c8` during the 2026-09-02 audit; never assume that ref is still current.

## Current design

### Scope and API

- `zml.io.Loader` selects the direct path for CUDA, ROCm, and oneAPI and the
  buffered path otherwise. A loader owns its store, sharding/profile options,
  worker pool, and sequential epochs.
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
  longer public loader mechanisms. `Loader.loadExecute` remains and is
  synchronous: it loads every source for a multi-source/fused binding, drains
  the epoch, executes the binding, and returns with the output ready.
- The compatibility target observed in `~/github/zml/monorepo` is behavioral:
  repeated `loadExecute`, followed by at most one `load`/`await`, multi-source
  `TensorStore` bindings, and cumulative loaded-byte accounting. That checkout
  intentionally continues to use its older `../zml` dependency for now.

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

Coalescing is batch-local to one `load()`/`appendItems()` call; the current
scheduler can append a later batch while an earlier one executes. The plan
stores source jobs and logical source pieces. Jobs carry source slot, absolute
offset, physical length, piece range, per-device physical-byte charges, and a
same-file predecessor. Workers currently re-walk tensor dispatch spans and
split logical pieces at DMA block boundaries to construct final transfer
records. Physical source bytes are distinct from logical tensor bytes so
duplication and replication do not distort diagnostics or fairness.
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
  rewrite scratch is stack bounded. `DmaBlockPool.acquireMany` still allocates
  matching arrays per source job, so steady-state allocation removal remains
  open work; the earlier worker-scratch change was neutral within noise.
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
- Two gates separate clean read-measurement generations from complete request
  lifecycles. All persistent workers compete for lifecycle capacity, configured
  as active reads plus one shared spare and clipped by pinned feasibility; the
  read gate alone limits source calls. A request returns lifecycle credit only
  after all its DMA children finish.
- Every changed width, including settled backoff, closes and drains the read
  gate, discards telemetry at the generation boundary, and reopens at the new
  width. Another settled backoff requires at least one new-generation source
  admission, so delayed old-width feedback cannot ratchet through several
  ladder rungs.
- DMA width is fixed at eight per device by default after calibration work
  showed adaptive DMA width added substantial complexity and little load
  value. There is no global DMA parallelism cap.

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
  a smaller block. It tunes one representative device, warms device allocators
  on all devices, applies the uniform selected tuple, and grows the retained
  all-device working set. There is no decision-dead aggregate timing phase.
- Retained arenas are initial capacity, not the full permissible live set.
  Detection starts with one largest-candidate calibration ring, reuses it,
  grows after selection to the all-device working set, and permits bounded slab
  growth up to the mapped-memory ceiling. Workspace validation, arena reserves,
  worker scratch, and adaptive pinned feasibility use the exact maximum
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

`PLAN.md` is the completion checklist. The current ordered design work is:

1. Enforce one immutable batch per epoch, precompute the deterministic fair
   predecessor-safe job order, and retain persistent workers across sequential
   `loadExecute` epochs.
2. Emit final block/DMA transfer records during planning instead of storing
   logical pieces and re-walking dispatch spans in workers.
3. Reduce the six-phase source controller, remove write-only metrics and
   singleton wrappers, and move block-pool matching arrays into reusable
   scratch.

After those deletions, reassess splitting `zml/io.zig`; splitting before them
would distribute obsolete concepts across more modules. Longer-term work still
includes calibration caching, cross-platform 24/32 MiB measurement,
completion-aware local pacing, and any explicit packed-device-buffer redesign
needed to reduce DMA submission count below roughly one per tensor.

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
