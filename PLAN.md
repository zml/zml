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
- [x] 2. Per-batch completion behind the existing single slot.
- [x] 3. FIFO scheduler, `Handle` API, multi-binding submissions, `Window`.
- [x] 4. Controller continuity across submissions (minimal).
- [x] 5. Monorepo migration and Laguna window measurement (migration done and
      served on two hosts; the window comparison is void until MI300 is healthy).
- [x] 6. Per-file incremental publish; identity fair order for one device.
- [x] 7. VFS: throttle classification, two-class backpressure, dead timing.
- [x] 8. Climb-and-hold controller without gate drains (gated, revertible).
- [x] 9. Per-plan preallocated contexts and event retirement (gated).
- [x] 10. VFS range/retry consolidation (gated on tests).
- [x] 11. Calibration reporting cleanup.
- [ ] 12. NUMA placement experiment (measurement only; blocked: MI300 degraded).
- [x] 13. Final validation and documentation reconciliation.
- [x] 14. Fourth pass (2026-09-04): decouple the DMA stage from the read
  width (lifecycle credits = max(pre-grown pinned capacity, width + the
  calibrated DMA depth), workers stay at width + 1, byte-based per-device
  DMA budget with a 64-piece cap, DMA reserves materialized at calibration
  with a ceiling fallback, warm-up window when credits ran out, stage
  timers in the summary). Result: gb300-2 DeepSeek 5.9 s -> 3.3-3.6 s;
  B70, HF and degraded MI300 at parity; CUDA host unmeasured (GPUs
  occupied). Details in CTX.md "Fourth pass".

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
  task 0 onward. DeepSeek-V4-Flash on the MI300 host for planner-touching
  tasks (it does not fit one 32 GB RTX 5090); planned jobs must stay 9,524
  and transfers 69,572 at 16 MiB requests.
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

Result (2026-09-04): `Batch{allocator, io, jobs, transfers, items, remaining,
done, requests, blocks, events, diagnostics, freeing(debug)}` owns one plan,
its items and every request/block/event context; `finishJobs(n)` releases
completion units and the last one sets `done`. Units are released at the
`RequestContext` 1->0 transition (`completeOne`, final statement after the
gate release), by the worker when `registerRequest` fails after a claim, and
by `scheduler.fail` for unclaimed jobs (a cursor swap under the scheduler
mutex partitions claims exactly). `claim` returns `Claim{batch, job}`. The
PJRT ready callback and the `submitOne` failure path (`submitTransfer`) copy
`pipeline`/`device_index`/`block` into locals, call `eventCompleted`, then
`block.complete()` last; `abortReady` holds `metadata_mutex`, under which
`retireBatch` destroys contexts, so it cannot race the free;
`abandonSubmissions` runs under the worker's scheduling sentinel. `await`
waits `batch.done`, debug-asserts the scheduler is exhausted, drains the
lifecycle gate (only permits of workers that claimed nothing), runs the
controller barrier, empties the slot (`finishEpoch` rendezvous, still needed
because `claim` reads the slot before it holds a unit), retires and destroys
the batch. Deleted: `reapCompleted`, the loader-wide context lists and their
duplicate `deinit` loops, `epoch_items`, `epoch_active` (`current: ?*Batch`),
`resetEpoch`; `logEpoch` reads the batch's diagnostics. Tests 228 -> 232
passed, 3 skipped: N-request completion, claimed-then-abandoned job, `fail`
in three cursor states, the late-callback failure now drives a real pool
lease through `abortReady` to `done`, and 200 sequential batches with four
concurrent claimers through `waitForWork`/`finishEpoch`/`retireBatch`. Local
B70, Llama, 3 runs: packs=0 0.783/0.777/0.782 s (19.11/19.24/19.14 GiB/s,
width 12); packs=64 width 16: pack phase 0.901/0.891/0.917 s
(14.42/14.59/14.17 GiB/s), pack check ok, total 1.016/1.008/1.037 s.
Unchanged from the task-1 numbers.

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

Result (2026-09-04): `FairVectoredReadScheduler` is a FIFO `{allocator,
queue, head, unclaimed_total, stopping, mutex, condition}`: `publish` appends
(a batch without jobs is never queued and completes when its sentinel drops),
`claim` advances the batch's plain `cursor` under the mutex and pops the batch
with its last job, so a queued batch can never be freed; `fail` retires the
unclaimed units of every queued batch and clears the queue; `snapshot` returns
`unclaimed_total`. `DirectLoader.submit(specs) !*Batch` and `awaitBatch`
replace `appendItems`/`loadPrepared`/`loadBinding`/`await`; each batch logs
one line with publish/first-claim/done offsets and `destroy` logs a loader
summary. Deleted: `plan`, the atomic cursor, `waiting_workers`,
`worker_count`, `finishEpoch`, `waitExhausted`, `initForTest`, `TestJob`,
the `Job.transfers` default and `PreparedBatch` guards, `current`, the epoch
branch of `checkOpen`, `logEpoch`/`DirectLoaderDiagnostics`,
`error.LoaderEpochActive`, the buffered `epoch_active`/`epoch_logical_bytes`,
`PreparedExecutableBinding`, `executeLoadedBinding`, `Loader.await`,
`Loader.checkOpen`. `epochBarrier` stays unused and `metrics.outstanding_requests`
stays, both for task 4. `io.zig`: `Loader.load -> Handle`, `Loader.Binding`,
`loadExecute(bindings) -> Handle` over the union of the bindings' sources,
`Handle{await, isDone, logicalBytes}` (executables run in binding order on
the awaiting task with `.wait = true`, inputs freed, bytes committed per
handle, idempotent), `awaitAll`, `executeInputBytesPerDevice`, `deinit`
awaiting open handles reads-only, `Window{budget_bytes, max_handles}` with
`submit`/`drain`/`deinit`, `BufferedBatch{pending, done}`, `error.EmptyTensor`.
The in-repo callers were migrated mechanically to `const h = try
loader.load(...); try h.await();`: `zml/testing.zig` (`testLayer`),
`examples/mnist/mnist.zig`, and the llama/lfm2/qwen3_5/qwen3_5_moe models
under `examples/llm/models`; `//examples/llm`, `//examples/llm:llama_tests`,
`//examples/mnist` and the playground build. Tests 232 -> 235 passed, 3
skipped: fair-order tests call
`fairOrder`; FIFO tests cover publish order, completion while a later batch is
still claimed, `fail` over every queued batch, wake/stop, concurrent claims
across two batches and 210 overlapping batches awaited newest first; public
tests cover out-of-order awaits with 3x bytes, a two-binding submission,
`deinit` with open handles, the window, and a missing-file read failure that
fails every pending await and later submissions with `error.FileNotFound`
(a past-EOF entry reads zero bytes without error on the buffered path).
Local B70, Llama, 3 runs each: plain 0.785/0.784/0.810 s (19.05/19.08/18.47
GiB/s, width 12/12/8, pinned high-water 200-264 MiB); packs=64 width 16
window 1: pack phase 0.712/0.695/0.804 s (task 2: 0.891-0.917 s), total
0.809/0.791/0.919 s; window 2: pack phase 0.676/0.664/0.674 s at 19.2-19.6
GiB/s, total 0.767/0.757/0.776 s, i.e. the plain load time; window 4:
0.780/0.782/0.782 s (16.6 GiB/s, width 16, pinned 264 MiB); window 2 with
pairs=1: 0.789/0.787/0.743 s, 8 batches, reads unchanged at 1977 because
Llama's same-shape packs are not file-adjacent (widths 16/24/8). Pack runs
pin 104 MiB at width 12. Two B70 (`level_zero:0,1`) with packs window 2:
sharded 0.900 s and replicated 0.939 s, both complete, pack checks ok.
Window-2 timeline: batch k+1's first claim precedes batch k's completion
(+0.090 s vs +0.095 s, +0.235 vs +0.242), so executables are hidden. Window
4 and pairs lose to the controller probing 16/24 once work is continuous
(the B70 knee is 12): task 4/8 territory. Deviations: `DirectLoader.destroy`
does not await batches itself (the front end's `deinit` does, reads only,
and `scheduler.deinit` asserts the queue is empty); `Window` records each
entry's input bytes next to its handle.

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

Result (2026-09-04): `SourceReadRuntime` lost `epochBarrier` and its two
fields, `metrics.outstanding_requests` with `beginRequest`/`endRequest` (the
pipeline teardown assert and `shouldBootstrapSource` read
`request_gate.inUse`, the lifecycle credit taken before the claim) and
`applyDecision`'s `force_probe`: every remaining caller passes a changed or a
settled decision, so the old fall-through is an assert. `finishIdleMeasurement`
scores or drops the interval as before, then puts both gates at
`controller.width()` and reports it; the new `resumeMeasurement` prepares a
probe at the current width behind the admission fence without touching the
gates. `create` seeds `reported_width` and calls `resumeMeasurement` before
the workers start (born busy): the old startup `applyDecision` raced the
workers' first admissions and closed the read gate for a full drain at the
start of every load, and the resulting position of the first included read
decided which 25 ms tick scored first. `gate_closed_ticks` (control ticks
with the read gate at 0 and jobs unclaimed) is in the loader summary via
`AdaptiveRequestGate.currentLimit`. Tests 235 -> 236 (`activity transitions
keep both gates at the controller width`). Local B70, Llama, 3 runs each with
the final binary: plain 0.777/0.828/0.819 s (width 12/8/8, 5-6 closed ticks,
all rung-change drains); packs=64 width 16 window 1: pack phase
0.742/0.751/0.740 s, total 0.836/0.846/0.831 s, widths 16/12/12,
gate_closed_ticks 2/1/1; window 2: pack phase 0.659/0.741/0.664 s, total
0.758/0.834/0.758 s, width 12, ticks 1/5/1 (three traced runs of the same
code: 0.661-0.663 s, one tick each); window 4: pack phase 0.745/0.751/0.749 s
(task 3: 0.78), total 0.841/0.862/0.841 s, widths 8/8/16, ticks 5/6/5. Not
anticipated: (1) the counter does not read 0 at window 1 because the only
closures left are scoring freezes, and a probe spans packs whenever the 25 ms
tick misses the 7-15 ms idle gap between them (traced: freeze at t=110 ms
with pack 1 already published); boundary closures themselves are gone. (2)
Window 2 is bimodal on the first score: at t=135 ms the queue holds a short
tail (6-35 jobs), `probeFitsTail` fails and the controller settles at 12
(0.66 s); one tick later pack 3's 224 jobs are queued, the whole ladder runs
with 4-5 drains and 16 is selected half the time (0.72-0.77 s). Born-busy
startup made the first mode 5 of 6 runs. (3) Window 4 ramps
12->16->24->32->12 like a plain load because four queued packs always leave
about 480 jobs, and the pack workload measured 16 at 22.98 GiB/s against
19.40/22.21 at 12, so 12 misses the 3% band by 0.35%; the controller scores
read throughput only and never sees the per-batch DMA tail or its own drains,
which is why it can select 16 while the pack phase is slower there. (4) At
window 1, `finishIdleMeasurement` scores a complete `.measuring` interval with
an infinite tail, so now that idle is seen between most packs the width moves
one rung per pack unclipped (traced 12->8->4 and 12->16->12->24) and the bulk
batch inherits it. (2)-(4) are task 8 territory (busy-time clock, no drains).

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

Result (2026-09-04, build and Llama verified; Laguna measurement pending):
`llmd/main.zig`: the five `vfs.register(scheme, x.io())` calls are
`vfs.registerBackend(scheme, x.backend())`; after the platform block
`platform.benchTransfer(allocator, io, .{})`, `platform.warmupDeviceAllocators()`
and `const load_profile = try vfs.loadProfile(args.model)`; new flag
`--expert-pack-budget=<bytes>` (0 = one layer's packs at a time) stored as
`models.Options.expert_pack_budget`; `zml/io/load` at debug in
`std_options` so the two loader lines print; the bench client gets `base_io`
(the VFS `io` resolves fd 1 through its handle table, so the TTFT line failed
with `WriteFailed`). `llmd/models.zig`: `Options.expert_pack_budget`,
`Model.init(..., load_profile, progress)` threaded to every model.
`llmd/dflash.zig`: `DFlashContext.initOptional(allocator, io, &vfs, ...)`
computes the drafter repo's own profile, carried in
`DflashOptions.load_profile`. Eleven loaders (`llama`, `llama_dflash` x2,
`gemma3_text`, `gemma4_text`, `gemma4_dflash` x2, `ministral3`, `qwen3_5`,
`lfm2`, `dflash_drafter`, `laguna_dflash` drafter): `Loader.init(allocator,
io, platform, store, .{ .shardings, .load_profile, .progress })`, `const
handle = try loader.load(T, model, &buffers); try handle.await();`,
`bytesLoaded()`; the `initTextOnly` wrappers (`gemma3`, `gemma4`, `mistral3`)
pass `load_profile` through. `laguna.zig`: `Tensors.loadBuffers(self,
allocator, io, platform, parallelism, store, load_profile,
expert_pack_budget, progress)` compiles the packers, submits ONE
`loader.load(Tensors, self, &buffers)` for every single-source tensor (fused
bindings skipped), then per sparse layer ONE `Window.submit(&loader, &.{down,
gate_up})` through `Window.init(allocator, budget orelse 1, 8)`, then
`window.drain()`, `bulk.await()`, `bytesLoaded()`; the first (dense) layer is a
different type and has no packs; `Laguna.loadBuffers`,
`LagunaDecoderLayer.loadBuffers`, `LagunaSparseMoe.loadBuffers`,
`LagunaExperts.loadBuffers` and the arena are gone, `preloadExpertPackers`,
`LagunaExperts.preload`, `ExpertPackExecutables`, `tensorsFromSources` and
the pack functions stay; `laguna_dflash.zig` calls it with
`options.expert_pack_budget`. Placement validation: the pack tensors carry
`.expert = .experts`; `pickSharding(shardings, shape, .explicit_axis_binding)`
returns the `experts` sharding (`tp_mesh` and `replicated` bind no `.experts`
axis) and the packers are compiled with the same `shardings` list, where
`Partitioning.selectSharding` (`.any_covering`) picks the same `experts`
sharding for the `.expert = .experts` output, so `validateSamePlacement`
compares one registered sharding with itself. Unrelated drift: the monorepo
was last built against zml of 2026-07-27 and `../zml` HEAD needed
`zml.Compiler(f)` -> `zml.Compiler.Typed(f)` (42 sites), `slice1d` -> `slice`
(126 sites), three `dynamicSlice` -> `slice(ax, .dynSingle(i))`,
`forwardMoe(x, ids, w, Linear, Linear, .{}, metadata, parameters)` (three
MoE models), `flashinfer_cutlass` in the MoE metadata switches, the
`stablehlo` paged-attention arm (mirrors `triton`) in `attention.zig` and
`parallelism.zig`. Build: `cd ~/github/zml/monorepo && bazel build
--config=release --@zml//platforms:oneapi=true --@zml//platforms:cpu=false
//llmd:llmd` (green; `zig fmt --check` clean). Local B70:
`ONEAPI_DEVICE_SELECTOR=level_zero:1 bazel run ... //llmd:llmd --
--model=/var/models/meta-llama/Llama-3.1-8B-Instruct --bench-prompt="Hello"`
printed `live loader ready: target=oneapi, profile=local, request_size=8.00MiB,
dma_block_size=8.00MiB, workers=13, ... retained=528.00MiB`, `Loaded weights
[14.96GiB, 640.175ms, 23.36GiB/s]`, `loader summary: batches=1,
successful=true, bytes_loaded=14.96GiB, elapsed=0.640s, reads=1918,
dma_submissions=2187, selected_source_width=12, gate_closed_ticks=0,
pinned_high_water=136.00MiB`, `TTFT: 112.44 ms`, `Average decode throughput:
27.71 tokens/s (23 completion tokens)`. Laguna-XS-2.1 (63 GB) does not fit a
32 GB B70; on the ROCm host run, from `~/github/zml/monorepo`,
`ROCR_VISIBLE_DEVICES=0 bazel run --config=release --@zml//platforms:rocm=true
--@zml//platforms:cpu=false //llmd:llmd --
--model=/var/models/poolside/Laguna-XS-2.1 --expert-pack-budget=0
--bench-prompt="Hello"` and the same with `--expert-pack-budget=3221225472`
(two layers of bf16 pack inputs: down 512 MiB + gate_up 1 GiB per layer), and
compare `Loaded weights`, `batch completed`, `loader summary` and the bench
lines (`grep -a`, the logo makes the log binary for grep).

### 6. Per-file incremental publish; identity fair order for one device

- `DirectLoader.submit`: sort once; per file group `prepareBatch` then
  `publish` immediately; seal at the end (sentinel). Skip `fairOrder` for
  `device_count == 1` (assert identity in a test). Per-batch diagnostics gain
  `first_read_at` and per-file planning time.
- Validation: DeepSeek on MI300: jobs 9,524 and transfers 69,572 unchanged,
  median at or below the 7.4-7.6 s epoch baseline (planning 0.32 s is the
  upper bound of the win); multi-device sharded medians within spread.
  Fallback: whole-model planning for `device_count > 1`.

Result (2026-09-04): `Batch` owns `plans: ArrayListUnmanaged(Plan)` with
`Plan{jobs, transfers, planning_ns, cursor}`, a `plan_cursor`, `sealed` and
`queued` (all scheduler-mutex state) and its items from creation; `remaining`
starts at the sentinel and grows by each plan's job count as
`scheduler.publish(batch, plan)` appends it (list and queue capacity reserved
first, so the plan, its units and the queue entry appear together).
`DirectLoader.submit` creates the batch and its items, then
`scheduler.publishFiles` sorts once by (file URI, offset, size, index)
(`sortedItemOrder`, `fileGroupEnd`), plans each file group with `preparePlan`
(the former `prepareBatch` without its file loop; `finalJob` builds the
claimable job) and publishes it immediately; `seal` ends the submission
before the sentinel drop. The queue holds open batches and batches with
unclaimed jobs: `claim` walks the head's plans in order, pops a sealed
exhausted batch with its last claim and returns null on an open exhausted
head (the unclaimed total is 0 then, so workers sleep in `waitForWork` until
the next plan's broadcast); `seal` pops a batch that is already exhausted;
`fail` retires the unclaimed units of every plan of every queued batch. A
planning failure before the first publish destroys the batch and returns
the error; after it, `failPublished` records the error on the pipeline,
seals, drops the sentinel, awaits and retires the batch inside `submit` and
returns the sticky error. `preparePlan` skips `fairOrder` and the device
queues for `device_count == 1`. Diagnostics: `plans`, total `planning_ns`
and per-plan `planning_ns`, `sealed_at`, `first_read_ns` (atomic, stamped by
the first admitted read); the batch line gained `sealed`, `first_read`,
`plans`, `longest_planning`. Tests 239 -> 243 passed, 3 skipped: fair order
is the identity for one device; a two-file submission is published as two
plans claimed in file order; the coalescing test plans per file through
`publishFiles` with the totals unchanged (4 jobs, 7 transfers, 24 bytes, 3
runs over two plans); an open head keeps the head with nothing to claim and
a waiting worker wakes at the next publish, a batch sealed while exhausted
leaves the queue at its seal; `fail` on an open batch with two published
plans retires the unclaimed units and the seal plus sentinel complete it;
the FIFO tests are unchanged through `publishTestBatch` (publish, seal,
sentinel). Local B70, Llama, 3 runs each: plain 0.644/0.635/0.651 s
(23.2/23.6/23.0 GiB/s, widths 12/12/16), the batch line `plans=4,
planning_elapsed=0.001-0.002s, longest_planning<=0.001s,
planned_source_jobs=1918, planned_transfers=2187, published=+0.001s,
sealed=+0.002-0.003s, first_claim=+0.001s, first_read=+0.001s`; packs=64
width 16 window 2: total 0.701/0.726/0.724 s (pack phase 0.616/0.640/0.639
s), reads 1977, bulk batch `plans=4` with 313 jobs / 317 transfers, pack
checks ok; two B70 (`level_zero:0,1`) sharded packs window 2: 0.891 s,
complete, pack check ok (task 3: 0.900 s). Llama plans in 1-2 ms, so the
change is neutral here by construction. DeepSeek on MI300 (46 shards) is
pending a trusted host (the CTX degradation bullet): look for `plans=46`,
`planned_source_jobs=9524`, `planned_transfers=69572`, `first_claim` and
`first_read` within one file's planning time of `published` (about 7 ms;
`longest_planning` bounds the wait), `sealed - published` close to the old
0.32 s planning time now overlapped with reads, and a median wall below
the 7.09-7.22 s post-task-8 anchor by at most that time. Deviation: the
fallback (whole-model planning for `device_count > 1`) was not implemented.
Per-file plans change the multi-device order only across files: the old
single plan could interleave a later file's jobs to balance device bytes,
while now a file's jobs are claimed before the next file's; sharded loads
charge every device in every job, so their order is unchanged (two-device
sharded within spread). A model whose files map to disjoint device sets
would now be read file by file; revisit if such a layout appears.

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

Result (2026-09-04): `range_read.classifyStatus(status, unavailable)` is the
one retry classification: 408 timeout, 429 throttle, other 5xx server
failure, and 503 is `unavailable`, which S3 (`SlowDown`) and GCS pass as
`.throttle` and generic HTTP and HF as `.server_failure` (S3 and GCS keep a
one-line `classifyStatus` with a unit test each). `range_read.serverRetryDelay`
reads `Retry-After` delta-seconds or the HF `RateLimit` `t=` reset for every
retried status and replaces the jittered delay when present; the HTTP-date
form was skipped (a date parser alone is the 40-line budget). Deleted:
`ResponseTiming`, `writeFirstAndReadScatter`, the one-byte first-body probe,
`elapsedNanoseconds`, `readResponse`'s `io` parameter, `readScatter`'s
destination offset, HTTP `retryForStatus`, HF `hfThrottleDelay`;
`AttemptResult.success` is void and the body path is discard then scatter.
`ReadStatsCursor.takeBackpressure` returns `{throttle, transient}`: throttle
when throttles or timeouts moved, transient when retries, transient retries
or server failures moved without a throttle; no side channel (local files)
is neither. `backoff` is unchanged (one rung down, ceiling clipped, holding);
`stepDownTransient` is one rung down with the ceiling and state unchanged,
and a climbing controller restarts its climb at that rung (it becomes the
best rung and its mean is forgotten, so the next window there is a fresh
climb sample and can lead back above the step); both share
`last_backoff_generation`, so each fires at most once per generation of
fresh admissions. The runtime logs `source width backoff` or `source width
transient step-down`. Tests: vfs 15 -> 18 (classification per backend,
server retry delay, S3 and GCS 503; the HTTP `retryForStatus` test moved to
`range_read`), zml 242 (239 passed, 3 skipped; +1 controller test, the
cursor test covers both classes and the throttle-wins rule). S3Proxy (local,
`LATENCY_MS=20 SPEED_MIB=200`, `level_zero:1`): the first adaptive run
climbed 32/48/64/96 at 3.07/5.24/6.56/8.69 GiB/s and aborted at the move to
128 inside the oneAPI plugin (`host_to_device_transfer_manager.cc:342 Check
failed: definition_events_[buffer_index]`, reached from `SetEventAsError`
through the pump's `onReady` callback: a transfer error, then the next piece
submitted into the same manager). It did not reproduce: ceiling 64 loaded in
2.881 s at 5.19 GiB/s (hold 64, pinned 1.02 GiB); fixed 128 in 1.54 s at
9.71 GiB/s (pinned 2.00 GiB); the same adaptive command again in 3.761 s at
3.98 GiB/s (32/48/64 at 3.54/4.26/4.17 GiB/s, hold 48); `SPEED_MIB=800` in
3.264 s at 4.58 GiB/s (32/48/24 at 7.22/5.04/6.51, hold 32). Every run had 0
retries, 0 throttles and `gate_closed_ticks` 0, and no backoff or step-down
fired, so the width path is the one before this task; the uncommitted task
11 calibration cleanup was compiled into all five runs. S3Proxy has no
fault-injection middleware (latency, throttled stream, eventual, read-only,
null only), so the 503 ceiling drop and the early-500 no-pin are covered by
the unit tests only; real AWS was not run (no credentials).

### 8. Climb-and-hold controller without gate drains (gated)

Evidence that raised its priority (2026-09-04, MI300, plain Llama): fixed
width 12 loads in 0.42 s, fixed 24 in 0.63 s, adaptive in 0.90 s selecting
24. Task 4 traces: the scoring freeze still closes the gate; a probe that
spans a submission boundary is scored at idle with an infinite tail and moves
the width one rung per pack; the climb probes 16/24/32 above the knee because
read throughput alone is within 3% across those rungs. The pinned pool grows
by hipHostMalloc slabs mid-load when the width exceeds the retained arena
(146 ms for the first slab); pre-growing to the widths the controller may
probe belongs to this task.

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

Result (2026-09-04): `SourceReadWidthController` is `{fixed_width, index,
start_index, max_index, best_index, rates (per-rung mean), samples, state:
climbing|holding, borderline_used, probed_down, generation,
last_backoff_generation}`. `observe` folds the window's rate into the rung's
mean; a climb sample (the best rung or the one above it) that beats the best
by 3% moves one rung up, or holds at the pinned clip; otherwise the hold rung
is the lowest measured rung at or below the best within 3% of it, re-measured
once when its retention is within 0.02 of 0.97, and probed one rung lower
once when it is the start rung; then hold. Every decision opens a generation.
`backoff(fresh_admissions)` lowers one rung, clips `max_index` there and
holds; a second sample in the generation a backoff opened is ignored unless a
read admitted under it has begun (`probe_peak_reads != 0` on the window
`applyDecision` fences for every generation, holding or not). Blind growth
moves the start rung to the reached width. Deleted: `Phase`, `Confirmation`,
`confirmation_used`, `restartAt`, `refine_down`, `beginRefineOrSettle`,
`probeCost`/`probeFitsTail`, `ramp_scores`/`unchanged_candidates`,
`recomputePeakAndSelection`, `settle`, `Decision.changed/settled`,
`Evidence.remaining_full_jobs`, `selectedWidth`. Runtime `Measurement =
{inactive, measuring, blind}`; `applyDecision` sets both gate limits, fences
the generation's window at the next admission and measures while climbing;
`start` (born busy) replaces `resumeMeasurement`; the blind-to-measured
transition is a new generation at the reached width. Deleted
`.transitioning`, `.scoring`, `activatePendingProbe`,
`backoff_admission_start`/`backoffReady`, `finishIdleMeasurement`,
`scheduler_idle`. `BusyWindowClock`: a control tick that finds nothing
unclaimed, pending or admitted after the window's first admission charges
the interval since the previous tick to idle, subtracted from the window's
elapsed time, so a window spans submissions and idle gaps never score or
reset it. Pre-growth: `DmaPlatformSettings.ensureSourceWorkingSet` at
`create` grows every NUMA pool to `(32 + 1) * maximum_blocks_per_job`
blocks, clipped to the largest width whose growth leaves the mapped ceiling
room for the non-materialized feed reserves (B70: 264 MiB retained, 8 MiB
beyond calibration's 256 MiB in 1.5 ms; 16 MiB blocks and requests: 528
MiB; 2 MiB blocks with 8 MiB requests: 132 MiB); the ready line logs
`retained`, `pregrown`, `pregrowth_ms` and every scored window logs
`source width window: generation, width, rate, busy_ms, completed,
exercised, samples, next_width, state`. Tests 236 -> 238. Local B70, Llama:
plain 0.672/0.658/0.657/0.657/0.666 s (task 4: 0.777-0.828), widths
12/12/16/16/12, windows 12 at 21.8-23.1 GiB/s, 16 at 22.3-22.6, then 8 at
20.3-21.8 (hold 12) or 24 at 21.6 (hold 16); fixed 12 0.631/0.628/0.627 s;
packs width 16 window 1 pack phase 0.668/0.684/0.692 s, total
0.764/0.778/0.787 s (task 4: 0.74-0.75 / 0.83-0.85), width 16 (16 at
24.8-25.3 against 12 at 21.8-23.2); window 2 pack phase 0.674/0.675/0.669
s, total 0.773/0.768/0.768 s, width 12; window 4 pack phase
0.688/0.651/0.666 s, total 0.787/0.747/0.757 s, widths 8/24/12 (the pack
rates at window 4 spread 17.8-20.0 GiB/s per rung between runs);
`gate_closed_ticks` 0 in all 17 runs. Not reproduced as written: the
recorded B70 curve holds 12 without the borderline re-measure of 16 the task
text expected, because the re-measure applies to the hold rung below the
best (16 above 12 at 0.970 can neither climb nor become the hold), so the
replay's third window is the downward probe of 8 (test: 19.90 GiB/s, hold
12). The B70 8 MiB curve is flat between 12 and 16 within the band (16 beat
12 by 3.5% in two of five plain runs), so the held width alternates 12/16
at equal load times. MI300/CUDA confirmation pending.

Follow-ups landed after the task 8 measurements (commits 4a046807 and
f76cd626): (a) the pinned working set is mapped in calibration, not at
loader creation, after MI300 showed 203-238 ms of hipHostMalloc inside the
load; (b) worker tasks are spawned on demand from the lifecycle limit
(MI300, adaptive: 16 tasks 0.424 s epoch, 24 tasks 0.476 s, 32 tasks
0.572 s, 128 tasks 0.613-0.675 s; even width 16 measured 21 GiB/s with 128
tasks against 36 GiB/s with 16); (c) windows open at the generation's
first completed read so a high-latency source is not charged its round trip
(HF windows 447-495 MiB/s before, 600 MiB/s after, for the same width).
Results on quiet hosts, Llama one GPU: local B70 0.636-0.672 s (day one
0.76-0.79); CUDA 0.467-0.548 s (day one 0.61); MI300 0.486-0.508 s (day one
1.10-1.15; fixed width 12 is 0.41-0.42 s, the remaining gap is one probe
window); MI300 DeepSeek 7.09-7.22 s (day one 8.2); HF Qwen3.5-9B 19.6-19.7 s
at 934-938 MiB/s (day one 21-22 s), where the climb to 48-64 is correct
because the CDN caps each connection near 19 MiB/s. Pinned high-water
136-272 MiB locally inside a 528 MiB working set; 1.5-2 GiB on HF.

### 9. Per-plan preallocated contexts and event retirement (gated)

- `prepareBatch` returns `dma_submissions`; plans own `requests`, `blocks`,
  `events` arrays; the callback pushes retired events under the metadata lock
  and the pump destroys them; `awaitBatch` drains the rest. Plugin check first.
- Validation: tests; DeepSeek median and peak RSS; no PJRT error in debug.

Result (2026-09-04): Part A: `preparePlan` returns a heap `Batch.Plan{jobs,
transfers, requests, blocks, events, events_used, source_bytes, source_runs,
planning_ns, cursor}` (`PreparedPlan` is gone; `Batch.plans` holds `*Plan`
so contexts can keep the plan's address): `requests` is one `RequestContext`
per job, initialised `idle` (nothing pending, completed); `blocks` is one
`BlockContext` per job block, sized exactly (`PlanningJob.block_start/
block_len` = `divCeil(len, block_size)`, so `Job.blocks` is the job's slice)
rather than `jobs x maximum_blocks_per_job`; `events` is one `EventContext`
per planned DMA submission (`sum popCount(writer_mask)`, reported as
`planned_dma_submissions` on the batch line; equals `dma_submissions`).
`Job` carries `request` and `blocks`, `Claim` carries the plan;
`registerRequest(claim)` and `registerBlock` cannot fail and take no lock,
`submitTransfer` takes the next event slot under `metadata_mutex`, and
`reserveBlockCapacity`, `abandonSourceJob`, the worker's register-failure
path and the scratch block-pointer array are gone (the zero-reference check
moved before the block lease). `retireBatch` walks the arrays with the same
asserts (request completed, registered leases complete; idle slots pass by
construction) and `Batch.destroy` frees the plans. Steady state: about 6k
`allocator.create` calls per Llama load (1918 requests, 1918 blocks, 2187
events) plus three per-batch list growths became zero; a load allocates
per file (the plan object and its five arrays). Part B is enabled
(`retire_events_early = true`): `EventContext{pjrt_event: ?, next_retired}`;
the ready callback, after `eventCompleted` (and any pump it ran) and before
`block.complete()`, pushes its context onto the pipeline's intrusive
`retired` stack under `metadata_mutex`; `pump` destroys the stack at the
top of every iteration under the lock, so an event is destroyed by a later
pump or by `retireBatch`, never inside its own callback; `retireBatch`
destroys what is left and unlinks the batch's contexts (`pjrt_event ==
null`) before the batch is freed; `deinit` asserts the stack is empty. Live
PJRT events are bounded by devices x 8 plus one pump batch. Plugin check
first: the playground's `ZML_LOAD_EVENT_RETIRE_CHECK=1` (`EventRetireCheck`:
per-device streams of 8 MiB pinned transfers, 8 in flight, every event
handed by its callback to a retire task that destroys it at once on another
thread, events destroyed before their manager) on the B70: 2048 events at
23.5 GiB/s, 16384 at 43.7 GiB/s (128 GiB), two devices 16384 at 53.7 GiB/s,
and 8192 on the final build at 38.8 GiB/s, each with fired = destroyed and
0 errors, no plugin objection. Tests 243 -> 244 passed, 3 skipped: removed
"claimed job abandoned before its request" (the path no longer exists),
added "retirement accepts the idle slots of jobs a failure retired" and
"retired events are destroyed by the next pump or unlinked by the batch
retirement"; the coalescing test checks the array sizes and job slices, the
lifecycle tests that claims take their plan slots; vfs/stdx pass. Local
B70, Llama, 3 runs each: before (baseline) plain 0.638/0.645/0.628 s, peak
RSS 1,870,544/1,869,448/1,869,940 kB; after Part A plain 0.650/0.636/0.634
s, RSS 1,870,740/1,870,408/1,869,556 kB; final (A+B) plain
0.637/0.637/0.637 s (23.5 GiB/s, widths 16/12/12, pinned high-water
136-200 MiB), RSS 1,871,820/1,870,976/1,870,056 kB, `planned_source_jobs=
1918, planned_transfers=2187, planned_dma_submissions=2187`,
`dma_submissions=2187`; packs=64 width 16 window 2: total 0.697/0.723/0.725
s (pack phase 0.612/0.637/0.639 s), reads 1977, 15 batches, pack checks ok;
two B70 sharded packs window 2: 0.860 s (task 6: 0.891 s), pack check ok;
Debug build: plain 0.644 s, packs 0.758 s, two devices 0.882 s, no
assertion or PJRT error. RSS is unchanged because Llama's contexts are
about 0.4 MiB either way against 1.78 GiB of pinned memory and plugin
state; the DeepSeek median and RSS on MI300 (9,524 jobs, 69,572 events)
are pending a trusted host, as for task 6. Deviations: blocks are sized
exactly per job, plans are heap objects, and the retired list is an
intrusive stack instead of an index list (no capacity bound to reason
about).

### 10. VFS range/retry consolidation (gated)

- One `range_read.performRangeRead(io, client, stats, retry, spec, data,
  offset, size)` with a per-attempt request hook (S3 SigV4 stays per attempt);
  HTTP/S3/GCS/HF keep URL/auth/profile only. Delete `VFS.register` in favour
  of `registerBackend`. Validation: `//vfs:test` and the acceptance tests.

Result (2026-09-04): `range_read.performRangeRead(io, client, stats, retry,
spec, data, offset, size)` is the one range read loop: one GET with `Range:
bytes=off-(off+size-1)` per attempt, `recordAttempt` before each,
`classifyStatus(status, spec.unavailable)`, `recordFailure`, `recordRetry`,
`recordRetryDelay`, `serverRetryDelay` else full-jitter backoff,
`error.RetriesExhausted` after `retry.max_retries`, non-retried statuses fail
at once, `readResponse` for 206/200, `recordSuccess(size)`. `spec` is a
`RequestSpec{backend, target, unavailable, context, prepare}` whose
`prepare(context, Attempt{ordinal, range})` returns a `PreparedRequest{uri,
authorization, extra_headers}` for every attempt; the loop appends `Range`
itself (a hook returning `&.{range}` would hand back a dangling temporary) and
owns the one `backend: stage for target failed` log line. S3's
`SignedRequest.prepare` recomputes the SigV4 timestamp and signature per
attempt; GCS's `BearerRequest.prepare` copies the bearer per attempt, so a
token that expired during a retry delay is refreshed (before, it was copied
once per read: the one semantic change); HTTP and HF pass a static
`PreparedRequest` through `prepareStatic`. `RetryConfig.fromOptions(InitOpts)`
replaces the three retry fields and `assertValidOptions` without changing any
public `InitOpts`; `AtomicReadStats.provider()` replaces the four
`readStatsSnapshot` trampolines; the per-backend `classifyStatus` wrappers
became `S3.unavailable`/`GCS.unavailable` declarations used by the spec and by
the task 7 tests. `VFS.register` is deleted (no callers here or in the
monorepo); `loadProfile` resolves a bare path through a registered `file`
backend and falls back to `LoadProfile.local`. Lines: http 540 -> 395, s3
995 -> 852, gcs 1167 -> 1029, hf 951 -> 803, range_read 228 -> 432, vfs
549 -> 545, base 802 -> 811, acceptance test 417 -> 514 (net -160). Tests:
vfs 18 -> 20 (`RetryConfig.fromOptions`; "the shared range loop prepares the
request once per attempt": an S3-style spec stamps an `x-attempt` header the
mock server checks on every GET, a zero-budget call returns
`RetriesExhausted` after one attempt and one hook call, a one-retry call
succeeds after two, counters 3 requests / 1 retry / 2 server failures),
zml 243 passed, 3 skipped. The acceptance tests caught one bug in the pass:
`Response.reader` releases the head bytes, so `Content-Range` is read before
the body reader is taken. Runs on `level_zero:0`: S3Proxy (`LATENCY_MS=20
SPEED_MIB=800`) `Loaded weights [14.96GiB, 2.007s, 7.45GiB/s]`, 960
source_requests, 0 retries, 0 throttles, climb 32/48/64/96 at
7.10/7.62/8.47/6.43 GiB/s, hold 64, `pinned_high_water=1.52GiB`,
`gate_closed_ticks=0`; HF `hf://Qwen/Qwen3.5-9B` `[17.98GiB, 19.641s,
937.40MiB/s]`, 577 requests, 0 retries, width 48; local Llama
0.649/0.627/0.642 s (23.0-23.9 GiB/s), logged as `profile=file`.

### 11. Calibration reporting cleanup

- Remove the timing decomposition, per-sample logging and latency
  accumulator (one summary line), the duplicate device-allocator warm-up, and
  the overlapping arena growth entry points. Validation: dma-bench on the
  three hosts selects the same block as today.

Result (2026-09-04): `dma_calibration.zig` 2010 -> 1785 lines (-334/+117).
Calibration now logs exactly one line per run: `dma_bench version=13
platform, devices, kind, block_bytes, parallelism, measured_gib_s,
elapsed_ms, calibration_ms, allocator_warmup_ms, retained_mapped_bytes,
numa_pools`. Deleted: `DmaBenchmarkSample` and its `dma_bench_sample` line,
`DmaBenchmarkPhase` (its `phase` argument threaded through `measure`,
`measureDmaBenchmarkCandidates` and `confirmAndSelectDmaBenchmarkCandidate`),
`total_latency_ns`/`averageLatencyNs` on the run metrics, the atomic latency
counter and the per-transfer timestamp that fed it, `DeviceDmaRecommendation
.average_latency_ns`/`.windows`, the report's `setup_ns`, `sampling_ns`,
`source_registration_ns`, `benchmark_setup_ns`, `benchmark_overhead_ns`,
`source_cleanup_ns` (hard-coded 0) and `windows`, `finishDmaBenchmarkReport`,
the session's `samples`/`setup_ns`/`sampling_ns`/`windows`, the pools'
`registration_ns`, the `dma_bench_numa` per-device lines (the summary keeps
`numa_pools`; every arena still logs its own node and mapping cost) and the
`dma_bench_device` line (folded into the summary). The report keeps
`elapsed_ns` (whole call), `calibration_ns` (warm-up to selection, measured
from timestamps) and `device_allocator_warmup_ns`. `DeviceDmaRecommendation`
keeps only what the selection uses: device index/id, block size, width,
measured rate. `platform.transferSettings()` and the playground's
`dma_settings` line use `TransferSettings` and are unchanged.
`warmupDmaBenchmarkDeviceAllocators` is deleted; its concurrent body moved
into `Platform.warmupDeviceAllocators(io)` (callers: calibration,
`examples/io`, `examples/llm`, the platform test). Arena growth is one
`DmaBenchmarkSourcePools.allocate(node, bytes)` holding the ceiling check and
the mapping timing, called by the calibration ring (the duplicate
`growPool` of the same ring in `benchmarkSyntheticDma` is gone), by the
concurrent `growToBlockTargets` that `ensureLoadBlockReserves` and
`ensureSourceWorkingSet` (behaviour unchanged) feed per-pool block targets
to, and directly by the pool's `ArenaProvider.allocateFn`, which no longer
repeats the budget check; `growPool`, `allocatePool` and `ensureBlockReserves`
are gone. `requiredDmaWorkspaceBytes` counts each node's devices in one pass
(first-seen pool order, 64-device stack arrays) instead of the nested
seen/rescan loops, returns the single-pool formula directly when the topology
is unknown, and now errors instead of panicking on a mixed config. Decision
logic untouched: 2/4/8/16/32 MiB at width 8, 2 ms/32 completions, three
alternating confirmation pairs at 25 ms/256, the 8% prefer-smaller rule,
retained arenas, `preallocated_source_width`/`preallocated_request_size`.
Three fixes found while editing: `session.measure` became a wrapper that only
passed `self.io`, so the two call sites now call
`runReusableDmaBenchmarkWindow` directly; `pjrtx` was imported only by the
deleted warm-up; and `tuneDmaBenchmarkDevice` held both an `errdefer` and a
later `defer` freeing `block_candidates`, a double free on any measurement
error (one `defer` at the allocation now). No test was added or removed.
Validation: `//zml:test` 242 tests, 239 pass / 3 skip / 0 fail (the 5
calibration tests and "platform device allocators can be warmed repeatedly"
among them); `//vfs:test`, `//stdx:test` pass; `zig fmt --check` clean;
oneAPI release playground builds. Local B70 dma-bench 3x: 8 MiB at width 8
every time, 49.44/48.80/49.18 GiB/s, calibration_ms 861/854/851,
allocator_warmup_ms 348/338/351, retained 528 MiB, numa_pools 1. Llama
sharded warm loads 0.676/0.654/0.644 s after the final edit (0.626/0.649/
0.638/0.631 s before it, host load average 1.1-2.5 throughout, another
agent's runs on the same box), band 0.63-0.68; `live loader ready` still
reports retained=528.00MiB, pregrown=0B.

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

Result (2026-09-04, commit 67464f3c): the full test set, format checks and
example builds pass; the playground was rerun on all three hosts and the
event-retirement check passed on all three plugins. CTX.md now carries the
third-pass results table, the size accounting, the validation state and the
open items (Laguna window comparison and NUMA experiment on a healthy MI300,
first-window bias, calibration fragility on a busy host, one unreproduced
oneAPI failure-path abort). Task 12 was not run: the MI300 host degraded
during the day (DMA 7 GiB/s with any tree) and the experiment needs it.
