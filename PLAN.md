# Loader redesign plan (third pass: caller-controlled concurrency)

`CTX.md` is the authoritative description of the loader's current design and
measured behavior. This file is the execution checklist. After every task:

1. run the task's focused validation;
2. update the result and status here;
3. update `CTX.md` so it describes the code that now exists;
4. do not begin the next task until those updates are complete.

Passes one and two (simplification) are complete and folded into `CTX.md`.
This pass changes the loader's concurrency model. The compatibility target is
behavioral, based on the callers in `~/github/zml/monorepo`: repeated expert
pack loads, whole-model loads, multi-source bindings, cumulative byte
accounting. Signatures are not a constraint; the monorepo is migrated here.

## Goals (from the user)

- Shortest load time with the simplest code: simple state machines, few
  branches, clear control flow.
- The caller always controls concurrency: how many `loadExecute` submissions
  are in flight, and in which order, to avoid device OOM.
- Keep: DMA block detection up front (8 in flight per device, no global cap),
  one VFS profile per model with a feedback side channel, adaptive read
  concurrency, reads and DMA decoupled in size and count, low pinned memory.

## Target design (summary; details in CTX.md "Third-pass design")

- Every submission returns a `Handle`. `load(Model, ...)` submits every
  single-source tensor of a model; `loadExecute(bindings, outputs)` submits the
  sources of one or more executable bindings as ONE planned submission (so a
  layer's packs coalesce). `Handle.await()` waits for the submission's DMA,
  then (for bindings) runs each executable on the awaiting task with
  `.wait = true` and frees its inputs. `Loader.awaitAll()`, `bytesLoaded()`.
- Direct backend: a strict FIFO of immutable planned batches replaces the one
  epoch slot. Claims move under the scheduler mutex. A batch completes when
  its last claimed job's final DMA callback lands (per-batch counter + event).
  The epoch flag, worker rendezvous, controller epoch barrier and loader-wide
  reclamation disappear. Workers, gates, pool, pump, planner, calibration and
  VFS data plane are unchanged.
- Caller-side policy: `zml.io.Window` (byte budget + ticket cap) awaits the
  oldest handle before submitting the next; `executeInputBytesPerDevice(exe)`
  sizes it. A window of one reproduces today's serialized behavior.
- Later, separately measured: per-file incremental publish, climb-and-hold
  width controller without gate drains, per-plan preallocated contexts, VFS
  consolidation, NUMA experiment.

## Third-pass status

- [x] 0. Baseline capture and pack instrument in the playground.
- [x] 1. Zero-risk dead-code sweep, `LazyOnce`, shared helpers.
- [ ] 2. Per-batch completion behind the existing single slot.
- [ ] 3. FIFO scheduler, `Handle` API, multi-binding submissions, `Window`.
- [ ] 4. Controller continuity across submissions (minimal).
- [ ] 5. Monorepo migration and Laguna window measurement.
- [ ] 6. Per-file incremental publish; identity fair order for one device.
- [ ] 7. VFS: throttle classification, two-class backpressure, dead timing.
- [ ] 8. Climb-and-hold controller without gate drains (gated, revertible).
- [ ] 9. Per-plan preallocated contexts and event retirement (gated).
- [ ] 10. VFS range/retry consolidation (gated on tests).
- [ ] 11. Calibration reporting cleanup.
- [ ] 12. NUMA placement experiment (measurement only).
- [ ] 13. Final validation and documentation reconciliation.

## Measurement protocol

- Compare medians of at least 5 warm repeats on the same host, plugin, model
  and cache state; report the spread; claim a change only outside the spread.
- Check host state first (`uptime`, `nvidia-smi`/`rocm-smi`): the CUDA host is
  shared and was measured bimodal (0.55 s vs 1.3 s) under another user's load.
  The scratch script `bench_host.sh <cuda|rocm|oneapi> <runs> <label>` prints
  host state, syncs `zml/`, `examples/io/`, `vfs/`, runs the playground and
  greps the result lines. `./run_io_load_matrix.sh` remains the canonical
  three-host run.
- The ROCm host currently loads at 13 GiB/s (CTX anchor 24 GiB/s); the user
  reports a host problem. Treat ROCm numbers as a smoke test only until it is
  investigated.
- Additional fixtures suggested by the user: `hf:///Qwen/Qwen3.5-9B` (remote
  HF profile: 32 MiB requests, blind bootstrap, side channel) and
  `/var/models/deepseek-ai/DeepSeek-V4-Flash` on mi300 (many tensors for its
  size). Baselines are recorded in CTX.md.
- Regression oracle at every task: three-host matrix at packs=0 (Llama
  14.96 GiB, one GPU). Pack instrument (`ZML_LOAD_PACKS=64
  ZML_LOAD_PACK_WIDTH=64 ZML_LOAD_PACK_WINDOW=K ZML_LOAD_PACK_PAIRS=1`) from
  task 0 onward. DeepSeek-V4-Flash on the CUDA host for planner-touching tasks
  (planned jobs must stay 9,524 and transfers 69,572).
- Test commands (verified 2026-09-03, ~20 s warm):
  `bazel test --@zml//platforms:cpu=true --@zml//platforms:cuda=true //zml:test --test_output=errors`,
  `bazel test //vfs:test //stdx:test --test_output=errors`,
  `bazel build --config=release --@zml//platforms:oneapi=true //examples/io:playground`,
  `zig fmt --check` on touched files.

## Task details

### 0. Baseline capture and pack instrument

- `examples/io/main.zig` `load`: add env knobs `ZML_LOAD_PACKS` (N pack
  submissions, default 0), `ZML_LOAD_PACK_WIDTH` (sources per pack, default
  64), `ZML_LOAD_PACK_WINDOW` (K, default 1), `ZML_LOAD_PACK_PAIRS` (0/1,
  honoured from task 3). When N > 0: walk registry keys in file order, group
  W consecutive rank-2 tensors of identical shape and dtype into a binding via
  `store.view().maybeCreateBinding(keys, shape.insert(0, .{ .expert = W }))`,
  compile one stack executable per distinct source shape with
  `platform.compileFn` (replicated output), exclude packed keys from the bulk
  model. Run packs with today's `loadExecute` sequentially, then `load` and
  `await` for the remainder. Log per-phase wall time, pack count, bytes; keep
  the `Loaded weights` line unchanged. Content check: read a sample of packs
  back with `toSliceAlloc` and compare with `safetensors.Tensor.reader`.
- Record baselines in CTX.md: packs=0 on all three hosts (done 2026-09-03,
  see CTX.md), packs=64 on all three hosts, DeepSeek on CUDA when the host is
  quiet.
- Validation: playground builds for oneapi/cuda/rocm; packs=0 medians match
  the 2026-09-03 baseline; content check passes on every host.

Result (2026-09-04): `examples/io/main.zig` gained `planPacks`, `packShape`,
`stackPack`, `checkPacks` and the knobs `ZML_LOAD_PACKS`, `ZML_LOAD_PACK_WIDTH`,
`ZML_LOAD_PACK_WINDOW`, `ZML_LOAD_PACK_PAIRS`, `ZML_LOAD_PACK_CHECK`,
`ZML_LOAD_PACK_MAX_ELEMENTS` (oneAPI rejects stack kernels above 2^31
elements). Packs are formed per (shape, dtype) class in file order because
Llama never has more than two adjacent same-shape tensors; bindings are
replicated so `pickSharding` and the executable's output placement agree.
Local B70, Llama, 3 runs each: packs=0 unchanged (0.772-0.792 s wall);
width 64 gives 2 packs (2.5 GiB, pack phase 0.156-0.203 s at 12-16 GiB/s);
width 16 gives 14 packs (13 GiB, pack phase 0.888-0.966 s at 13.5-14.6 GiB/s
versus 17.4-18.3 GiB/s for the bulk, about 6.4 ms of drain plus execute per
pack, every pack epoch stuck at the bootstrap width 12). Content checks pass.
Width 16 is the sensitive pack instrument for the local host. The HF fixture
needed a VFS fix (HEAD redirects), committed separately. CUDA host busy;
DeepSeek and pack baselines there are deferred until it is quiet.

### 1. Zero-risk dead-code sweep

- Delete `rollbackTail`, `restartFitsTail` and the tests whose only subject
  they are (port live `advanceAfterScore` assertions); `Snapshot.has_unscheduled`
  (use `remaining_jobs != 0`); the `writer_mask == 0` branch and
  `error.InconsistentReplicaLayout` (debug asserts); `VectoredTensorTransfer.total`;
  the `appendItems` placement recompute (keep the logical-byte sum).
- Move `effectiveSourceRequestSize` to `limits.zig`; both users import it.
- `sourceSlot`: `StringHashMap` instead of the linear scan.
- `LazyOnce(T, Ctx, initFn)` replaces `LoaderSourceSlot` and
  `LoaderLoadItem.StateSlot` (four-state cmpxchg + Event + u16 error code).
- Delete `selectDmaBenchmarkCandidate` and its two tests; fold their
  assertions into one test of `confirmAndSelectDmaBenchmarkCandidate`; fold
  the duplicated `DmaPlatformState` constants in `platform.zig` into
  `dma_calibration.zig`.
- CTX.md: remove the "unfinished finite-tail probes roll back" claim.
- Validation: full test set, `zig fmt --check`; matrix packs=0 within spread.

Result (2026-09-04): deleted `rollbackTail`, `restartFitsTail` and their three
tests (the tail-fit boundary is now asserted through `blindGrow` in the
backoff test; the short-tail `observe` test was kept and renamed),
`Snapshot.has_unscheduled`, the `writer_mask == 0` branch and
`error.InconsistentReplicaLayout` (debug asserts), `VectoredTensorTransfer.total`,
the `appendItems` placement recompute, `selectDmaBenchmarkCandidate` and its
two tests (one test now drives `confirmAndSelectDmaBenchmarkCandidate` with
non-borderline data so no confirmation window is scheduled), and the
`DmaPlatformState` constants in `platform.zig` (the `dma_platform_*`
constants of `dma_calibration.zig` are public). `effectiveSourceRequestSize`
lives in `limits.zig`. `sourceSlot` uses a `StringHashMapUnmanaged`.
`LazyOnce(T, Ctx, initFn)` backs `LoaderSourceSlot.file` and
`LoaderLoadItem.state`. Tests 232 -> 228 passed, 3 skipped. Local B70, Llama,
3 runs: packs=0 0.774/0.810/0.778 s (19.32/18.46/19.24 GiB/s, width 12);
packs=64 width 16: pack phase 0.901/0.896/0.909 s (14.43/14.51/14.30 GiB/s),
pack check ok, total 1.017/1.012/1.024 s. Unchanged from the task-0 baseline.

### 2. Per-batch completion behind the single slot

- `Batch{allocator, jobs, transfers, items, remaining: atomic, done: Event,
  requests, blocks, events, diagnostics, freeing(debug)}` and
  `finishJobs(n)`: `if (remaining.fetchSub(n) == n) done.set()`.
  `RequestContext.batch`; `registerRequest`/`registerBlock`/`submitOne`
  append to the batch's lists.
- `completeOne` at 1->0: copy `pipeline`/`batch` to locals, `endRequest`,
  `request_gate.release`, `batch.finishJobs(1)` as the final statement.
- Callback order rule (fixes a use-after-free): in `onReady` load `pipeline`,
  `device_index`, `block` into locals, store `err`, record the error, call
  `eventCompleted(device_index)`, then `block.complete()` LAST. Document at
  the callback and at `finishJobs`; debug `freeing` assertion.
- `workerMain`: when `registerRequest` fails after a claim, `batch.finishJobs(1)`
  before `recordError`; `scheduler.fail` retires unclaimed units.
- `appendItems` builds the Batch (remaining = 1 + jobs, then drop the
  sentinel) into `current: ?*Batch`; `await` waits `batch.done` (keep
  `waitExhausted`/`waitEmpty` as debug asserts), retires the batch's lists
  and items under `metadata_mutex`.
- Tests: N jobs complete after N request completions; abandonment completes;
  `fail` retires unclaimed units; late-callback failure sets `done`; a
  200-batch sequential stress test.
- Validation: full test set; matrix packs=0 and packs=64 within spread.

### 3. FIFO scheduler, Handle API, multi-binding submissions, Window

- Scheduler: `queue: []*Batch`, `head`, `unclaimed_total`, `stopping`, one
  mutex + condition. `publish(batch)` appends (never rejects); `claim()`
  under the mutex pops sealed and exhausted heads; `waitForWork` waits while
  `!stopping and nothing claimable`; `fail()` retires unclaimed units of every
  queued batch; `snapshot()` returns `unclaimed_total`. Delete `plan`, the
  atomic cursor, `waiting_workers`, `worker_count`, `finishEpoch`,
  `waitExhausted`, `initForTest`, `TestJob`, `Job.transfers` default, the
  empty-slice guards, `epoch_active`, `epoch_items`, the epoch branch of
  `checkOpen`, `loadPrepared`/`loadBinding`/`await`, `reapCompleted`, the
  duplicate `deinit` loops, `resetEpoch`/`logEpoch` (per-batch log line in
  `awaitBatch`, loader summary in `destroy`), `metrics.outstanding_requests`
  (`shouldBootstrapSource` reads `request_gate.inUse()`).
- `DirectLoader.submit(specs) !*Batch`, `awaitBatch(*Batch)`; `destroy`
  awaits open batches then drains the gate.
- `io.zig`: `Handle{state}` with `await()`, `isDone()`, `logicalBytes()`;
  `Loader.load(M, model, buffers) !Handle`; `Loader.Binding{tensor, output, exe}`;
  `Loader.loadExecute(bindings: []const Binding) !Handle` validating every
  binding, allocating input shells, one submission over all sources;
  `Handle.await` runs the executables in binding order with `.wait = true`
  and frees inputs; `Loader.awaitAll()`; `Loader.executeInputBytesPerDevice(exe)`;
  `Loader.open` list; `Loader.deinit` awaits leftovers (reads only for
  bindings). `Window{budget_bytes, max_handles}` with `submit`, `drain`,
  `deinit`. Zero-byte tensors return `error.EmptyTensor`. Buffered backend:
  `BufferedBatch{pending, done}`; delete its epoch state.
- Tests: fair-order tests call `fairOrder` directly; concurrent claims and
  exhaustion tests run against the FIFO; new: batches claimed in publish
  order, a batch completes while a later batch has unclaimed jobs, `fail`
  marks every queued batch done. Public test: `loadExecute` A and B and
  `load` C back to back; await B, A, C; contents correct; bytes = 3x; an
  injected read failure fails every pending await with the same error;
  `deinit` with handles open; a two-binding submission yields two outputs
  from one batch.
- Playground: honour `ZML_LOAD_PACK_WINDOW` through `Window` and
  `ZML_LOAD_PACK_PAIRS` (two packs per submission).
- Validation: full test set; matrix packs=0 within spread; packs=64 K=1
  faster than task 2 (no rendezvous/barrier); K=2/4 hide executables
  (submission k+1 first read before submission k await returns); pairs reduce
  planned jobs per submission; multi-device sharded and replicated loads with
  packs interleaved complete without hang (local 2xB70; 4/8 MI300X when the
  host is trusted).

### 4. Controller continuity (minimal)

- Delete `epochBarrier` and its two fields. `finishIdleMeasurement`: apply
  `RequestGateLimits.init(controller.width(), feasible)` to both gates and set
  `reported_width` (today the read gate is left at the last probed rung).
  Busy transition: `metrics.prepareProbe(generation, next_read_admission)` and
  `measurement = .measuring` when not settled; never touch the read gate on an
  activity transition. `applyDecision` fall-through becomes an assert. Add a
  debug counter of gate-closed-while-pending intervals to the loader summary.
- Validation: controller tests unchanged; packs=64 K=1 shows zero gate-closed
  intervals at boundaries; matrix packs=0 neutral, selected widths unchanged.

### 5. Monorepo migration and Laguna window

- `llmd/main.zig`: `vfs.registerBackend(scheme, x.backend())` for the five
  schemes; `platform.benchTransfer` next to `warmupDeviceAllocators`;
  `const load_profile = try vfs.loadProfile(model)` threaded to `loadBuffers`.
- `llama.zig`, `gemma4_text.zig` and every other `Loader.init` user:
  `Loader.init(allocator, io, platform, store, .{shardings, load_profile,
  progress})`, `var h = try loader.load(...)`, `try h.await()`, `bytesLoaded()`.
- `laguna.zig`: one whole-model `load`; per layer one `Window.submit` with
  both packs; delete the per-struct `loadBuffers`/`LoadOpts` plumbing and the
  per-call arena; keep `preloadExpertPackers`. Option `expert_pack_budget`
  (0 = window of one; default two layers of inputs). Handle the new hard
  errors (`NotFound`, placement equality with `.expert = .experts`).
- Validation: llmd builds; Llama and Gemma4 serve; Laguna at budget 0 / 1 /
  2 layers: byte total equal to the pre-migration total, temperature-0 tokens
  identical, peak HBM steps by one layer's inputs, wall time decreasing; the
  loader line shows the VFS profile for an `hf://` model.

### 6. Per-file incremental publish; identity fair order for one device

- `DirectLoader.submit`: sort once; per file group `prepareBatch` then
  `publish` immediately; seal at the end (sentinel). Skip `fairOrder` for
  `device_count == 1` (assert identity in a test). Per-batch diagnostics gain
  `first_read_at` and per-file planning time.
- Validation: DeepSeek on CUDA: jobs 9,524 and transfers 69,572 unchanged,
  median at or below the anchor (target 3.6 to 3.8 s); multi-device sharded
  medians within spread. Fallback: whole-model planning for `device_count > 1`.

### 7. VFS throttle classification and two-class backpressure

- S3/GCS `503` -> `.throttle`. Delete `ResponseTiming`,
  `writeFirstAndReadScatter` and the one-byte probe; body path is discard
  then scatter. `ReadStatsCursor.takeBackpressure` returns `{throttle,
  transient}`; throttle -> `backoff()` (settle one rung down); transient ->
  one rung down without settling, at most once per generation. Optional:
  Retry-After parsing in the shared loop.
- Validation: `//vfs:test`; S3Proxy with 2% injected 503 (ceiling drops, load
  completes, floor > 1); a single early 500 no longer pins the width; real
  AWS if credentials are available.

### 8. Climb-and-hold controller without gate drains (gated)

- Controller `{index, max_index, best_index, rates[12], samples[12],
  state: climbing|holding, borderline_used, probed_down, generation,
  last_backoff_generation}`; climb while a rung beats best by 3%; hold at the
  lowest rung within 0.97 x best; one borderline re-measure; one optional
  downward probe below the start rung. Delete `Confirmation`, `restartAt`,
  `refine_down`, `beginRefineOrSettle`, `probeCost`/`probeFitsTail`,
  `ramp_scores`/`unchanged_candidates`. Runtime `{inactive, measuring, blind}`;
  `applyDecision` never sets the read gate to 0; busy-time window clock in
  the existing 10/25 ms poll; two-class backoff from task 7.
- Tests replay the recorded B70 curve (12/16/24/32 = 21.33/20.69/18.90/17.33
  GiB/s -> hold 12) and a flat curve (hold at the first rung failing to beat
  best by 3%).
- Validation: per-rung rate table on B70 reproduces the ordering; matrix and
  S3Proxy within spread; gate-closed counter zero. Revert the whole task if
  any anchor moves outside spread.

### 9. Per-plan preallocated contexts and event retirement (gated)

- `prepareBatch` returns `dma_submissions`; plans own `requests`, `blocks`,
  `events` arrays; the callback pushes retired events under the metadata lock
  and the pump destroys them; `awaitBatch` drains the rest. Plugin check first.
- Validation: tests; DeepSeek median and peak RSS; no PJRT error in debug.

### 10. VFS range/retry consolidation (gated)

- One `range_read.performRangeRead(io, client, stats, retry, spec, data,
  offset, size)` with a per-attempt request hook (S3 SigV4 stays per attempt);
  HTTP/S3/GCS/HF keep URL/auth/profile only. Delete `VFS.register` in favour
  of `registerBackend`. Validation: `//vfs:test` and the acceptance tests.

### 11. Calibration reporting cleanup

- Remove the timing decomposition, per-sample logging and latency
  accumulator (one summary line), the duplicate device-allocator warm-up, and
  the overlapping arena growth entry points. Validation: dma-bench on the
  three hosts selects the same block as today.

### 12. NUMA placement experiment (measurement only)

- Debug env override for a single unbound pool (and an interleaved variant).
  Arms on eight MI300X (when the host is trusted): auto, all node 0, all node
  1, single pool, single pool + interleave; replicated Gemma and sharded
  Llama, 7 repeats; record `numastat` and a `_copy_to_iter` profile. Decision
  rule in CTX.md: delete affinity only if the single-pool arm is within 3% of
  auto on both workloads with overlapping spreads.

### 13. Final validation and documentation reconciliation

- Full test set, `zig fmt --check`, buildifier, playground builds, matrix,
  DeepSeek, packs, monorepo serves. CTX.md "Current design" rewritten for the
  code that exists; measurement tables refreshed; open work updated.
